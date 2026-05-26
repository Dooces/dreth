from __future__ import annotations

"""Persistent Nethra memory store.

Append-only JSONL persistence for runtime-visible familiarity/provenance records.
This module is deliberately storage and provenance only: it does not issue
authority, revoke authority, suppress skips, replace fit, force probes, increase
monitoring, increase repair priority, or support derivation.
"""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .memory_sleep import HIDDEN_TRUTH_LIKE_FIELDS


MEMORY_RECORD_TYPES: frozenset[str] = frozenset({
    "background_nethra",
    "context_role",
    "uncertainty",
    "authority_strength",
    "scaffold_proposal",
    "temporal_event",
})


@dataclass
class NethraMemoryRecord:
    record_id: str
    record_type: str
    run_id: str
    seed: int
    schedule: str
    n_vars: int
    cycle_start: int
    cycle_end: int
    vars: list[int] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    source_kind: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    authority_allowed: bool = False

    def __post_init__(self) -> None:
        if self.record_type not in MEMORY_RECORD_TYPES:
            raise ValueError(f"unknown nethra memory record_type: {self.record_type}")
        self.authority_allowed = False
        self.vars = [int(v) for v in (self.vars or [])]
        self.contexts = [str(c) for c in (self.contexts or [])]
        self.payload = _sanitize_payload(self.payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_kind"] = "record"
        d["authority_allowed"] = False
        d["payload"] = _sanitize_payload(d.get("payload") or {})
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NethraMemoryRecord":
        return cls(
            record_id=str(d.get("record_id", "")),
            record_type=str(d.get("record_type", "")),
            run_id=str(d.get("run_id", "")),
            seed=int(d.get("seed", 0) or 0),
            schedule=str(d.get("schedule", "")),
            n_vars=int(d.get("n_vars", 0) or 0),
            cycle_start=int(d.get("cycle_start", 0) or 0),
            cycle_end=int(d.get("cycle_end", 0) or 0),
            vars=[int(v) for v in (d.get("vars") or [])],
            contexts=[str(c) for c in (d.get("contexts") or [])],
            source_kind=str(d.get("source_kind", "")),
            payload=_sanitize_payload(d.get("payload") or {}),
            authority_allowed=False,
        )


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key in HIDDEN_TRUTH_LIKE_FIELDS or key.startswith("debug_"):
                continue
            out[key] = _sanitize_payload(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(v) for v in value]
    return value


def _vars_from_payload(payload: dict[str, Any]) -> list[int]:
    vals: list[int] = []
    raw = payload.get("vars")
    if isinstance(raw, list):
        vals.extend(int(v) for v in raw if _intlike(v))
    for key in ("var", "target_var"):
        if _intlike(payload.get(key)):
            vals.append(int(payload[key]))
    return sorted(set(vals))


def _contexts_from_payload(payload: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    raw = payload.get("context_keys")
    if isinstance(raw, list):
        contexts.extend(str(c) for c in raw if c)
    if payload.get("context_key"):
        contexts.append(str(payload["context_key"]))
    return list(dict.fromkeys(contexts))


def _intlike(value: Any) -> bool:
    try:
        int(value)
        return value is not None and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


class NethraMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append_records(self, records: Iterable[NethraMemoryRecord | dict[str, Any]]) -> int:
        rows: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, NethraMemoryRecord):
                rows.append(record.to_dict())
            elif isinstance(record, dict):
                rows.append(NethraMemoryRecord.from_dict(record).to_dict())
            else:
                raise TypeError(f"unsupported memory record type: {type(record)!r}")
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return len(rows)

    def append_run_summary(self, summary: dict[str, Any]) -> None:
        row = {
            "entry_kind": "run_summary",
            "authority_allowed": False,
            **_sanitize_payload(summary),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def append_sleep_result(self, result: dict[str, Any]) -> None:
        row = {
            "entry_kind": "sleep_result",
            "authority_allowed": False,
            **_sanitize_payload(result),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def load_records(
        self,
        record_type: str | None = None,
        since_run: str | None = None,
        limit: int | None = None,
    ) -> list[NethraMemoryRecord]:
        records: list[NethraMemoryRecord] = []
        for row in self._iter_rows():
            if row.get("entry_kind") != "record":
                continue
            if record_type is not None and row.get("record_type") != record_type:
                continue
            if since_run is not None and str(row.get("run_id", "")) < str(since_run):
                continue
            records.append(NethraMemoryRecord.from_dict(row))
            if limit is not None and len(records) >= max(0, int(limit)):
                break
        return records

    def load_scaffold_proposals(self, limit: int | None = None) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for record in self.load_records("scaffold_proposal", limit=limit):
            proposals.append(_sanitize_payload(record.payload))
        return proposals

    def count_backlog(self, record_type: str | None = None) -> int:
        return sum(1 for _ in self.load_records(record_type=record_type))

    def summarize(self) -> dict[str, Any]:
        by_type: Counter[str] = Counter()
        run_ids: set[str] = set()
        sleep_runs = 0
        authority_allowed = 0
        total = 0
        for row in self._iter_rows():
            if row.get("entry_kind") == "record":
                total += 1
                by_type[str(row.get("record_type", ""))] += 1
                if row.get("run_id"):
                    run_ids.add(str(row["run_id"]))
                if bool(row.get("authority_allowed")):
                    authority_allowed += 1
            elif row.get("entry_kind") == "sleep_result":
                sleep_runs += 1
        return {
            "path": str(self.path),
            "records": total,
            "records_by_type": dict(by_type),
            "runs": len(run_ids),
            "sleep_runs": sleep_runs,
            "authority_allowed_count": authority_allowed,
        }

    def compact(self, max_records: int | None = None) -> dict[str, Any]:
        records = self.load_records()
        if max_records is not None:
            records = records[-max(0, int(max_records)):]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        tmp.replace(self.path)
        return self.summarize()

    def to_sleep_rows(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for record in self.load_records():
            row = grouped.setdefault(
                record.run_id,
                {
                    "seed": record.seed,
                    "schedule": record.schedule,
                    "n_vars": record.n_vars,
                    "background_nethra_export": {"records": [], "edges": [], "role_shift_examples": []},
                    "context_role_index": {"nodes": [], "edges": [], "roles": [], "match_attribution": [], "records": []},
                    "authority_strength": {"records": [], "summary": {}, "controller": {}},
                    "background_nethra_records": 0,
                    "background_nethra_edges": 0,
                },
            )
            payload = _sanitize_payload(record.payload)
            if record.record_type in {"background_nethra", "uncertainty", "temporal_event"}:
                row["background_nethra_export"]["records"].append(payload)
            elif record.record_type == "context_role":
                row["context_role_index"]["records"].append(payload)
            elif record.record_type == "authority_strength":
                row["authority_strength"]["records"].append(payload)
        for row in grouped.values():
            row["background_nethra_records"] = len(row["background_nethra_export"]["records"])
        return list(grouped.values())

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def records_from_batch_record(row: dict[str, Any]) -> list[NethraMemoryRecord]:
    run_id = str(row.get("run_id") or _default_run_id(row))
    seed = int(row.get("seed", 0) or 0)
    schedule = str(row.get("schedule", ""))
    n_vars = int(row.get("n_vars", 0) or 0)
    cycle_end = int(row.get("cycles", row.get("recorded_cycles", 0)) or 0)
    base = {
        "run_id": run_id,
        "seed": seed,
        "schedule": schedule,
        "n_vars": n_vars,
        "cycle_start": 0,
        "cycle_end": cycle_end,
    }
    out: list[NethraMemoryRecord] = []

    for i, payload in enumerate(((row.get("background_nethra_export") or {}).get("records") or [])):
        if not isinstance(payload, dict):
            continue
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:background:{i}:{payload.get('nethra_id', '')}",
            record_type="background_nethra",
            source_kind=str(payload.get("kind", "")),
            vars=_vars_from_payload(payload),
            contexts=_contexts_from_payload(payload),
            payload=payload,
            **base,
        ))

    for i, payload in enumerate(((row.get("context_role_index") or {}).get("records") or [])):
        if not isinstance(payload, dict):
            continue
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:context_role:{i}:{payload.get('nethra_id', '')}",
            record_type="context_role",
            source_kind=str(payload.get("source", payload.get("kind", ""))),
            vars=_vars_from_payload(payload),
            contexts=_contexts_from_payload(payload),
            payload=payload,
            **base,
        ))

    for i, payload in enumerate(((row.get("authority_strength") or {}).get("records") or [])):
        if not isinstance(payload, dict):
            continue
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:authority_strength:{i}:{payload.get('nethra_id', '')}",
            record_type="authority_strength",
            source_kind=str(payload.get("strength", "authority_strength")),
            vars=_vars_from_payload(payload),
            contexts=_contexts_from_payload(payload),
            payload=payload,
            **base,
        ))

    if row.get("uncertainty_consolidation_mode") not in (None, "off"):
        payload = {
            "mode": row.get("uncertainty_consolidation_mode"),
            "cases_seen": row.get("uncertainty_cases_seen", 0),
            "clusters": row.get("uncertainty_clusters", 0),
            "compression_ratio": row.get("uncertainty_compression_ratio", 0.0),
            "giant_cluster_count": row.get("giant_cluster_count", 0),
            "max_cluster_size": row.get("max_cluster_size", 0),
        }
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:uncertainty:summary",
            record_type="uncertainty",
            source_kind="uncertainty_consolidation",
            vars=[],
            contexts=[],
            payload=payload,
            **base,
        ))

    for i, payload in enumerate(row.get("scaffold_memory_match_examples") or []):
        if not isinstance(payload, dict):
            continue
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:scaffold_match:{i}:{payload.get('matched_proposal_id', '')}",
            record_type="scaffold_proposal",
            source_kind="scaffold_memory_match",
            vars=[],
            contexts=[],
            payload=payload,
            **base,
        ))

    return out


def _default_run_id(row: dict[str, Any]) -> str:
    return (
        f"{row.get('schedule', '')}:n{row.get('n_vars', '')}:"
        f"c{row.get('cycles', '')}:seed{row.get('seed', '')}:"
        f"{row.get('policy', '')}"
    )


def records_by_type(records: Iterable[NethraMemoryRecord]) -> dict[str, int]:
    return dict(Counter(r.record_type for r in records))


def grouped_payloads(records: Iterable[NethraMemoryRecord]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.record_type].append(record.payload)
    return dict(grouped)
