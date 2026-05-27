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
    "nethra_handle",
})

USE_RIGHTS: frozenset[str] = frozenset({
    "record_only",
    "feature_only",
    "ranking_hint",
    "probe_hint",
    "soft_filter",
    "hard_filter",
    "block",
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
    nethra_id: str = ""
    touched_atoms: list[str] = field(default_factory=list)
    touched_structure_refs: list[str] = field(default_factory=list)
    member_nethras: list[str] = field(default_factory=list)
    context_scope: str = ""
    role_history: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    use_right: str = "record_only"
    salience: float = 0.0
    source: str = "runtime"
    created_cycle: int = 0
    last_used_cycle: int = 0
    last_success_cycle: int = 0
    success_count: int = 0
    failure_count: int = 0
    lift_history: list[dict[str, Any]] = field(default_factory=list)
    invalidators: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.record_type not in MEMORY_RECORD_TYPES:
            raise ValueError(f"unknown nethra memory record_type: {self.record_type}")
        self.authority_allowed = False
        self.vars = [int(v) for v in (self.vars or [])]
        self.contexts = [str(c) for c in (self.contexts or [])]
        self.payload = _sanitize_payload(self.payload)
        self.nethra_id = self.nethra_id or self.record_id
        self.touched_atoms = [str(a) for a in (self.touched_atoms or [])]
        self.touched_structure_refs = [str(r) for r in (self.touched_structure_refs or [])]
        self.member_nethras = [str(n) for n in (self.member_nethras or [])]
        self.context_scope = str(self.context_scope or (self.contexts[0] if self.contexts else ""))
        self.role_history = [
            _sanitize_payload(r) for r in (self.role_history or []) if isinstance(r, dict)
        ]
        self.evidence_refs = [str(e) for e in (self.evidence_refs or [])]
        if self.use_right not in USE_RIGHTS:
            self.use_right = "record_only"
        if self.use_right == "hard_filter" and self.source != "runtime_local_evidence":
            self.use_right = "record_only"
        self.salience = float(self.salience or 0.0)
        self.source = str(self.source or "runtime")
        self.created_cycle = int(self.created_cycle or self.cycle_start or 0)
        self.last_used_cycle = int(self.last_used_cycle or 0)
        self.last_success_cycle = int(self.last_success_cycle or 0)
        self.success_count = int(self.success_count or 0)
        self.failure_count = int(self.failure_count or 0)
        self.lift_history = [
            _sanitize_payload(r) for r in (self.lift_history or []) if isinstance(r, dict)
        ]
        self.invalidators = [str(i) for i in (self.invalidators or [])]

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
            nethra_id=str(d.get("nethra_id", "")),
            touched_atoms=[str(a) for a in (d.get("touched_atoms") or [])],
            touched_structure_refs=[
                str(r) for r in (d.get("touched_structure_refs") or [])
            ],
            member_nethras=[str(n) for n in (d.get("member_nethras") or [])],
            context_scope=str(d.get("context_scope", "")),
            role_history=[
                _sanitize_payload(r)
                for r in (d.get("role_history") or [])
                if isinstance(r, dict)
            ],
            evidence_refs=[str(e) for e in (d.get("evidence_refs") or [])],
            use_right=str(d.get("use_right", "record_only")),
            salience=float(d.get("salience", 0.0) or 0.0),
            source=str(d.get("source", "runtime")),
            created_cycle=int(d.get("created_cycle", 0) or 0),
            last_used_cycle=int(d.get("last_used_cycle", 0) or 0),
            last_success_cycle=int(d.get("last_success_cycle", 0) or 0),
            success_count=int(d.get("success_count", 0) or 0),
            failure_count=int(d.get("failure_count", 0) or 0),
            lift_history=[
                _sanitize_payload(r)
                for r in (d.get("lift_history") or [])
                if isinstance(r, dict)
            ],
            invalidators=[str(i) for i in (d.get("invalidators") or [])],
        )


@dataclass
class ExperienceEvent:
    run_id: str
    seed: int
    cycle: int
    context_key: str
    active_atoms: list[str]
    active_nethras: list[str]
    hook: str
    use_right: str
    candidates_before: list[Any] = field(default_factory=list)
    candidates_after: list[Any] = field(default_factory=list)
    probes_before: list[Any] = field(default_factory=list)
    probes_after: list[Any] = field(default_factory=list)
    selected_candidate: Any = None
    selected_probe: Any = None
    behavior_effect: int = 0
    authority_effect: int = 0
    full_audit_delta: int = 0
    intervention_delta: int = 0
    candidate_reduction_delta: int = 0
    quality_delta: float = 0.0
    success: bool = False
    failure_reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    hidden_truth_used: bool = False

    def __post_init__(self) -> None:
        self.hidden_truth_used = False
        if self.use_right not in USE_RIGHTS:
            self.use_right = "record_only"
        self.active_atoms = [str(a) for a in (self.active_atoms or [])]
        self.active_nethras = [str(n) for n in (self.active_nethras or [])]
        self.evidence_refs = [str(e) for e in (self.evidence_refs or [])]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_kind"] = "experience_event"
        d["hidden_truth_used"] = False
        return _sanitize_payload(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperienceEvent":
        return cls(
            run_id=str(d.get("run_id", "")),
            seed=int(d.get("seed", 0) or 0),
            cycle=int(d.get("cycle", 0) or 0),
            context_key=str(d.get("context_key", "")),
            active_atoms=[str(a) for a in (d.get("active_atoms") or [])],
            active_nethras=[str(n) for n in (d.get("active_nethras") or [])],
            hook=str(d.get("hook", "")),
            use_right=str(d.get("use_right", "record_only")),
            candidates_before=list(d.get("candidates_before") or []),
            candidates_after=list(d.get("candidates_after") or []),
            probes_before=list(d.get("probes_before") or []),
            probes_after=list(d.get("probes_after") or []),
            selected_candidate=d.get("selected_candidate"),
            selected_probe=d.get("selected_probe"),
            behavior_effect=int(d.get("behavior_effect", 0) or 0),
            authority_effect=int(d.get("authority_effect", 0) or 0),
            full_audit_delta=int(d.get("full_audit_delta", 0) or 0),
            intervention_delta=int(d.get("intervention_delta", 0) or 0),
            candidate_reduction_delta=int(d.get("candidate_reduction_delta", 0) or 0),
            quality_delta=float(d.get("quality_delta", 0.0) or 0.0),
            success=bool(d.get("success", False)),
            failure_reason=str(d.get("failure_reason", "")),
            evidence_refs=[str(e) for e in (d.get("evidence_refs") or [])],
            hidden_truth_used=False,
        )


@dataclass
class SleepProduct:
    proposal_id: str
    member_nethras: list[str]
    touched_atoms: list[str]
    touched_structure_refs: list[str]
    proposed_use_right: str
    proposed_context_scope: str
    salience_delta: float
    evidence_summary: str
    invalidators: list[str]
    reason: str
    authority_allowed: bool = False

    def __post_init__(self) -> None:
        self.authority_allowed = False
        if self.proposed_use_right == "hard_filter":
            self.proposed_use_right = "record_only"
            self.invalidators = list(dict.fromkeys(
                list(self.invalidators) + ["sleep_hard_filter_rejected"]
            ))
        if self.proposed_use_right not in USE_RIGHTS:
            self.proposed_use_right = "feature_only"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_kind"] = "sleep_product"
        d["record_type"] = "sleep_product"
        d["authority_allowed"] = False
        return _sanitize_payload(d)


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

    def append_experience_events(self, events: Iterable[ExperienceEvent | dict[str, Any]]) -> int:
        rows: list[dict[str, Any]] = []
        for event in events:
            if isinstance(event, ExperienceEvent):
                rows.append(event.to_dict())
            elif isinstance(event, dict):
                rows.append(ExperienceEvent.from_dict(event).to_dict())
            else:
                raise TypeError(f"unsupported experience event type: {type(event)!r}")
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return len(rows)

    def append_sleep_products(self, products: Iterable[SleepProduct | dict[str, Any]]) -> int:
        rows: list[dict[str, Any]] = []
        for product in products:
            if hasattr(product, "to_dict") and not isinstance(product, dict):
                rows.append(product.to_dict())
            elif isinstance(product, dict):
                rows.append(SleepProduct(
                    proposal_id=str(product.get("proposal_id", "")),
                    member_nethras=[str(n) for n in (product.get("member_nethras") or [])],
                    touched_atoms=[str(a) for a in (product.get("touched_atoms") or [])],
                    touched_structure_refs=[
                        str(r) for r in (product.get("touched_structure_refs") or [])
                    ],
                    proposed_use_right=str(product.get("proposed_use_right", "feature_only")),
                    proposed_context_scope=str(product.get("proposed_context_scope", "")),
                    salience_delta=float(product.get("salience_delta", 0.0) or 0.0),
                    evidence_summary=str(product.get("evidence_summary", "")),
                    invalidators=[str(i) for i in (product.get("invalidators") or [])],
                    reason=str(product.get("reason", "")),
                    authority_allowed=False,
                ).to_dict())
            else:
                raise TypeError(f"unsupported sleep product type: {type(product)!r}")
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

    def load_experience_events(self, limit: int | None = None) -> list[ExperienceEvent]:
        events: list[ExperienceEvent] = []
        for row in self._iter_rows():
            if row.get("entry_kind") != "experience_event":
                continue
            events.append(ExperienceEvent.from_dict(row))
            if limit is not None and len(events) >= max(0, int(limit)):
                break
        return events

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
            elif row.get("entry_kind") == "experience_event":
                by_type["experience_event"] += 1
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
        nid = str(payload.get("nethra_id", ""))
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:background:{i}:{payload.get('nethra_id', '')}",
            record_type="background_nethra",
            source_kind=str(payload.get("kind", "")),
            vars=_vars_from_payload(payload),
            contexts=_contexts_from_payload(payload),
            payload=payload,
            nethra_id=nid,
            touched_atoms=[f"x{v}" for v in _vars_from_payload(payload)],
            touched_structure_refs=[
                f"parents:{','.join(str(int(p)) for p in ps)}"
                for ps in (payload.get("parent_sets") or [])
                if isinstance(ps, list)
            ] + [str(sig) for sig in (payload.get("fit_signatures") or [])],
            member_nethras=[nid] if nid else [],
            context_scope=(_contexts_from_payload(payload) or [""])[0],
            role_history=[
                {"role": str(role), "cycle": int(payload.get("last_seen_cycle", 0) or 0)}
                for role in (payload.get("source_roles") or [])
            ],
            evidence_refs=[nid] if nid else [],
            use_right="ranking_hint" if payload.get("parent_sets") else "feature_only",
            salience=float(payload.get("salience_score", payload.get("cheap_recognition_score", 0.0)) or 0.0),
            source="runtime",
            created_cycle=int(payload.get("first_seen_cycle", 0) or 0),
            last_used_cycle=int(payload.get("last_seen_cycle", 0) or 0),
            **base,
        ))

    for i, payload in enumerate(((row.get("context_role_index") or {}).get("records") or [])):
        if not isinstance(payload, dict):
            continue
        nid = str(payload.get("nethra_id", ""))
        out.append(NethraMemoryRecord(
            record_id=f"{run_id}:context_role:{i}:{payload.get('nethra_id', '')}",
            record_type="context_role",
            source_kind=str(payload.get("source", payload.get("kind", ""))),
            vars=_vars_from_payload(payload),
            contexts=_contexts_from_payload(payload),
            payload=payload,
            nethra_id=nid,
            touched_atoms=[f"x{v}" for v in _vars_from_payload(payload)],
            touched_structure_refs=[
                f"parents:{','.join(str(int(p)) for p in (payload.get('learned_parents') or []))}",
                str(payload.get("signature", "")),
            ],
            member_nethras=[nid] if nid else [],
            context_scope=(_contexts_from_payload(payload) or [str(payload.get("context_key", ""))])[0],
            role_history=[{
                "role": str(payload.get("kind", "")),
                "cycle": int(payload.get("last_seen_cycle", payload.get("cycle", 0)) or 0),
            }],
            evidence_refs=[nid] if nid else [],
            use_right="ranking_hint" if payload.get("learned_parents") else "feature_only",
            salience=0.2,
            source="runtime",
            created_cycle=int(payload.get("first_seen_cycle", payload.get("cycle", 0)) or 0),
            last_used_cycle=int(payload.get("last_seen_cycle", payload.get("cycle", 0)) or 0),
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
