from __future__ import annotations

"""Runtime use-right enforcement for loaded persistent nethra handles.

Loaded memory is non-authoritative. This module may reorder existing candidate
or probe lists in assist mode, and it records attribution for both applied and
possible effects. It never writes certificates or mutates the ledger.
"""

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .nethra_memory_store import ExperienceEvent, HIDDEN_TRUTH_LIKE_FIELDS, NethraMemoryRecord, USE_RIGHTS
from dreth.learner.nethra_projection import ProjectionIndex


_MAX_EXPERIENCE_EVENTS = 200

BEHAVIOR_USE_RIGHTS: frozenset[str] = frozenset({
    "ranking_hint",
    "probe_hint",
    "soft_filter",
})


@dataclass(frozen=True)
class SalienceExplanation:
    score: float
    components: dict[str, float]


@dataclass
class RuntimeMemoryMetrics:
    loaded: int = 0
    used: int = 0
    sleep_products_loaded: int = 0
    sleep_products_used: int = 0
    behavior_effects: int = 0
    authority_effects: int = 0
    candidate_reorders: int = 0
    probe_reorders: int = 0
    soft_filter_fallbacks: int = 0
    hard_filter_rejected: int = 0
    block_events: int = 0
    lookups: int = 0
    matches: int = 0
    use_right_counts: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_nethras_loaded": self.loaded,
            "persistent_nethras_used": self.used,
            "sleep_products_loaded": self.sleep_products_loaded,
            "sleep_products_used": self.sleep_products_used,
            "nethra_memory_behavior_effects": self.behavior_effects,
            "nethra_memory_authority_effects": self.authority_effects,
            "nethra_memory_candidate_reorders": self.candidate_reorders,
            "nethra_memory_probe_reorders": self.probe_reorders,
            "nethra_memory_soft_filter_fallbacks": self.soft_filter_fallbacks,
            "nethra_memory_hard_filter_rejected": self.hard_filter_rejected,
            "nethra_memory_block_events": self.block_events,
            "nethra_memory_lookups": self.lookups,
            "nethra_memory_matches": self.matches,
            "nethra_memory_use_right_counts": dict(self.use_right_counts),
            "nethra_memory_examples": list(self.examples),
        }


class PersistentNethraIndex:
    def __init__(self, *, mode: str = "off", run_id: str = "", seed: int = 0) -> None:
        if mode not in {"off", "record", "assist"}:
            raise ValueError("persistent nethra mode must be off, record, or assist")
        self.mode = mode
        self.run_id = run_id
        self.seed = int(seed)
        self.records: list[NethraMemoryRecord] = []
        self._by_atom: dict[str, list[NethraMemoryRecord]] = defaultdict(list)
        self._by_id: dict[str, NethraMemoryRecord] = {}
        self._projection: ProjectionIndex = ProjectionIndex()
        self.metrics = RuntimeMemoryMetrics()
        self.events: deque[ExperienceEvent] = deque(maxlen=_MAX_EXPERIENCE_EVENTS)
        # Pending hint per target: set when rank_candidates fires a reorder, cleared
        # by record_fit_outcome so the fit result can confirm or discard the hint.
        self._pending_candidate_hints: dict[int, dict] = {}

    def load_path(self, path: str | Path) -> int:
        loaded = 0
        path = Path(path)
        if not path.exists():
            return 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or _has_hidden_truth(row):
                    continue
                record = self._record_from_row(row)
                if record is None:
                    continue
                self._add(record)
                loaded += 1
        self.metrics.loaded = len(self.records)
        return loaded

    def add_records(self, records: Iterable[NethraMemoryRecord]) -> None:
        for record in records:
            self._add(record)
        self.metrics.loaded = len(self.records)

    def query(self, active_atoms: Iterable[str], context_key: str, *, cycle: int = 0) -> list[tuple[NethraMemoryRecord, SalienceExplanation]]:
        atoms = {str(a) for a in active_atoms}
        self.metrics.lookups += 1
        seen: set[str] = set()
        out: list[tuple[NethraMemoryRecord, SalienceExplanation]] = []
        for atom in atoms:
            for record in self._by_atom.get(atom, ()):
                if record.record_id not in seen and record.use_right != "block":
                    seen.add(record.record_id)
                    out.append((record, SalienceExplanation(score=0.0, components={})))
        self.metrics.matches += len(out)
        return out

    def rank_candidates(
        self,
        *,
        var: int,
        context_key: str,
        candidates: Any,
        hook: str,
        cycle: int,
    ) -> Any:
        original = list(candidates or [])
        if self.mode == "off":
            return candidates
        active_atoms = [f"x{int(var)}", *(f"x{int(c)}" for c in original if _intlike(c))]
        scored_handles = self.query(active_atoms, context_key, cycle=cycle)
        usable = [
            (record, explanation)
            for record, explanation in scored_handles
            if record.use_right in BEHAVIOR_USE_RIGHTS or self.mode == "record"
        ]

        reordered = original
        behavior_effect = 0
        if self.mode == "assist":
            # Only use ranking records that match this specific context (or have no context).
            # Without context scoping, all records for any context match the query and
            # produce a preferred set covering all vars → no reordering.
            ranking_handles = [
                r for r, _ in usable
                if r.use_right == "ranking_hint"
                and (not r.context_scope or r.context_scope == context_key)
            ]
            if ranking_handles:
                preferred: set[int] = set()
                for r in ranking_handles:
                    for atom in r.touched_atoms:
                        if atom.startswith("x") and atom[1:].isdigit():
                            preferred.add(int(atom[1:]))
                # Remove the target var itself from preferred (it's not a candidate)
                preferred.discard(int(var))
                reordered = sorted(
                    original,
                    key=lambda c: (0 if (_intlike(c) and int(c) in preferred) else 1),
                )
                if reordered != original:
                    behavior_effect = 1
                    self.metrics.behavior_effects += 1
                    self.metrics.candidate_reorders += 1
                    if any(r.source == "sleep" for r in ranking_handles):
                        self.metrics.sleep_products_used += 1
                    # Save pending hint so record_fit_outcome can close the feedback loop:
                    # if a memory-promoted candidate is confirmed as the fitted source_edge,
                    # emit a success experience event for sleep to compound on.
                    self._pending_candidate_hints[int(var)] = {
                        "cycle": cycle,
                        "context_key": context_key,
                        "active_nethras": [r.nethra_id for r, _ in usable],
                        "evidence_refs": [r.record_id for r, _ in usable],
                        "promoted": frozenset(preferred),
                    }

        self._record_event(
            cycle=cycle,
            context_key=context_key,
            active_atoms=active_atoms,
            active_nethras=[r.nethra_id for r, _ in usable],
            hook=hook,
            use_right=_dominant_use_right(usable),
            candidates_before=original,
            candidates_after=reordered,
            behavior_effect=behavior_effect,
            evidence_refs=[r.record_id for r, _ in usable],
        )
        if behavior_effect:
            return type(candidates)(reordered) if isinstance(candidates, (list, tuple, set)) else reordered
        return candidates

    def rank_probes(
        self,
        *,
        var: int,
        context_key: str,
        probes: tuple[tuple[int, float], ...],
        cycle: int,
    ) -> tuple[tuple[int, float], ...]:
        original = tuple(probes or ())
        if self.mode == "off":
            return original
        active_atoms = [f"x{int(var)}", *(f"x{int(p[0])}" for p in original)]
        scored_all = self.query(active_atoms, context_key, cycle=cycle)
        usable = [
            (record, explanation)
            for record, explanation in scored_all
            if record.use_right == "probe_hint" or self.mode == "record"
        ]

        reordered = original
        behavior_effect = 0
        if self.mode == "assist":
            probe_handles = [
                r for r, _ in usable
                if r.use_right == "probe_hint"
                and (not r.context_scope or r.context_scope == context_key)
            ]
            if probe_handles:
                preferred: set[int] = set()
                for r in probe_handles:
                    for atom in r.touched_atoms:
                        if atom.startswith("x") and atom[1:].isdigit():
                            preferred.add(int(atom[1:]))
                reordered = tuple(sorted(
                    original,
                    key=lambda p: (0 if p[0] in preferred else 1),
                ))
                if reordered != original:
                    behavior_effect = 1
                    self.metrics.behavior_effects += 1
                    self.metrics.probe_reorders += 1
                    if any(r.source == "sleep" for r in probe_handles):
                        self.metrics.sleep_products_used += 1

        self._record_event(
            cycle=cycle,
            context_key=context_key,
            active_atoms=active_atoms,
            active_nethras=[r.nethra_id for r, _ in usable],
            hook="probe_hint",
            use_right="probe_hint" if usable else "record_only",
            probes_before=list(original),
            probes_after=list(reordered),
            evidence_refs=[r.record_id for r, _ in usable],
        )
        return reordered

    def runtime_metrics(self) -> dict[str, Any]:
        return self.metrics.to_dict()

    def export_experience_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def _add(self, record: NethraMemoryRecord) -> None:
        self.records.append(record)
        if record.source == "sleep":
            self.metrics.sleep_products_loaded += 1
        atoms = record.touched_atoms or [f"x{v}" for v in record.vars]
        if not atoms:
            atoms = [record.nethra_id]
        for atom in atoms:
            self._by_atom[str(atom)].append(record)
        if record.record_id:
            self._by_id[record.record_id] = record
        # Index mind nodes into the projection for fast context/hook filtering
        if isinstance(record.payload, dict) and str(record.payload.get("entry_kind", "")) == "nethra_mind_node":
            self._projection.index_node_from_row(record.payload)

    def _record_from_row(self, row: dict[str, Any]) -> NethraMemoryRecord | None:
        entry_kind = str(row.get("entry_kind", ""))
        if entry_kind == "nethra_mind_node":
            nethra_id = str(row.get("nethra_id", ""))
            if not nethra_id:
                return None
            use_rights_seen = [str(r) for r in (row.get("use_rights_seen") or [])]
            use_right = _effective_mind_use_right(use_rights_seen)
            invalidators = [str(i) for i in (row.get("invalidators") or [])]
            contexts = list(row.get("contexts") or [])
            context_scope = str(contexts[0]) if contexts else ""
            source_counts = row.get("source_counts") or {}
            source = "sleep" if int(source_counts.get("sleep", 0)) > 0 else "mind"
            return NethraMemoryRecord(
                record_id=nethra_id,
                record_type="nethra_handle",
                run_id="mind",
                seed=0,
                schedule="",
                n_vars=0,
                cycle_start=int(row.get("first_seen_cycle", 0) or 0),
                cycle_end=int(row.get("last_seen_cycle", 0) or 0),
                nethra_id=nethra_id,
                touched_atoms=[str(a) for a in (row.get("touched_atoms") or [])],
                touched_structure_refs=[str(r) for r in (row.get("touched_structure_refs") or [])],
                member_nethras=[str(n) for n in (row.get("member_nethras") or [])],
                context_scope=context_scope,
                evidence_refs=[str(r) for r in (row.get("sample_evidence_refs") or [])],
                use_right=use_right,
                salience=float(row.get("salience", 0.0) or 0.0),
                source=source,
                created_cycle=int(row.get("first_seen_cycle", 0) or 0),
                last_used_cycle=int(row.get("last_seen_cycle", 0) or 0),
                success_count=int(row.get("success_count", 0) or 0),
                failure_count=int(row.get("failure_count", 0) or 0),
                lift_history=[r for r in (row.get("lift_history") or []) if isinstance(r, dict)],
                invalidators=invalidators,
                payload=row,
            )
        if entry_kind == "nethra_mind_edge":
            return None
        if entry_kind == "record":
            try:
                return NethraMemoryRecord.from_dict(row)
            except ValueError:
                return None
        if entry_kind == "sleep_product" or row.get("record_type") == "sleep_product":
            use_right = str(row.get("proposed_use_right", "feature_only"))
            invalidators = [str(i) for i in (row.get("invalidators") or [])]
            if use_right == "hard_filter":
                use_right = "record_only"
                invalidators.append("sleep_hard_filter_rejected")
                self.metrics.hard_filter_rejected += 1
            return NethraMemoryRecord(
                record_id=str(row.get("proposal_id", "")),
                record_type="nethra_handle",
                run_id="sleep",
                seed=0,
                schedule="",
                n_vars=0,
                cycle_start=0,
                cycle_end=0,
                nethra_id=str(row.get("proposal_id", "")),
                touched_atoms=[str(a) for a in (row.get("touched_atoms") or [])],
                touched_structure_refs=[
                    str(r) for r in (row.get("touched_structure_refs") or [])
                ],
                member_nethras=[str(n) for n in (row.get("member_nethras") or [])],
                context_scope=str(row.get("proposed_context_scope", "")),
                evidence_refs=[str(n) for n in (row.get("member_nethras") or [])],
                use_right=use_right,
                salience=float(row.get("salience_delta", 0.0) or 0.0),
                source="sleep",
                invalidators=invalidators,
                payload=row,
            )
        if row.get("proposal_id"):
            use_right = str(row.get("suggested_runtime_use", "feature_only"))
            if use_right == "hard_filter":
                use_right = "record_only"
                self.metrics.hard_filter_rejected += 1
            if use_right not in USE_RIGHTS:
                use_right = "feature_only"
            return NethraMemoryRecord(
                record_id=str(row.get("proposal_id", "")),
                record_type="nethra_handle",
                run_id="sleep",
                seed=0,
                schedule="",
                n_vars=0,
                cycle_start=int(row.get("first_seen_cycle", 0) or 0),
                cycle_end=int(row.get("last_seen_cycle", 0) or 0),
                nethra_id=str(row.get("proposal_id", "")),
                touched_atoms=[f"x{int(v)}" for v in (row.get("vars") or []) if _intlike(v)],
                touched_structure_refs=[
                    *(str(sig) for sig in (row.get("common_signatures") or [])),
                    *(
                        "source_edges:" + ",".join(str(int(p)) for p in source_edges)
                        for source_edges in (row.get("common_source_edges") or [])
                    ),
                ],
                member_nethras=[str(n) for n in (row.get("source_record_ids") or [])],
                context_scope=(row.get("contexts") or [""])[0],
                evidence_refs=[str(n) for n in (row.get("source_record_ids") or [])],
                use_right=use_right,
                salience=float(row.get("action_relevance_score", 0.0) or 0.0),
                source="sleep",
                payload=row,
            )
        return None

    def record_fit_outcome(self, var: int, source_edges: tuple, cycle: int) -> None:
        """Close the feedback loop: if a memory-promoted candidate was installed as the
        fitted source_edge, emit a success experience event.  Sleep consolidation reads
        success=True to distinguish confirmed hints from mere reorders, which lets it
        compound ranking_hint products across generations rather than plateauing."""
        hint = self._pending_candidate_hints.pop(var, None)
        if hint is None or hint["cycle"] != cycle:
            return
        promoted: frozenset[int] = hint["promoted"]
        if not promoted:
            return
        fitted = {int(se) for se in source_edges if _intlike(se)}
        confirmed = promoted & fitted
        if confirmed:
            # At least one promoted candidate was installed: confirmed hint.
            # candidate_reduction_delta = +1: the hint correctly identified a
            # source_edge, reducing the effective search space.
            self._record_event(
                cycle=cycle,
                context_key=hint["context_key"],
                active_atoms=[f"x{se}" for se in sorted(confirmed)],
                active_nethras=hint["active_nethras"],
                hook="source_edge_candidates",
                use_right="ranking_hint",
                behavior_effect=1,
                candidate_reduction_delta=1,
                success=True,
                evidence_refs=hint["evidence_refs"],
            )
        else:
            # Promoted candidates were not among the fitted source_edges: bad hint.
            # candidate_reduction_delta = -1: the hint wasted a top-m slot on a
            # wrong candidate.  Emit a failure event so sleep can demote this pair.
            self._record_event(
                cycle=cycle,
                context_key=hint["context_key"],
                active_atoms=[f"x{p}" for p in sorted(promoted)],
                active_nethras=hint["active_nethras"],
                hook="source_edge_candidates",
                use_right="ranking_hint",
                behavior_effect=1,
                candidate_reduction_delta=-1,
                success=False,
                failure_reason="promoted_candidate_not_fitted",
                evidence_refs=hint["evidence_refs"],
            )

    def _record_event(self, **kwargs: Any) -> None:
        if self.mode == "off":
            return
        event = ExperienceEvent(
            run_id=self.run_id,
            seed=self.seed,
            cycle=int(kwargs.get("cycle", 0) or 0),
            context_key=str(kwargs.get("context_key", "")),
            active_atoms=list(kwargs.get("active_atoms") or []),
            active_nethras=list(kwargs.get("active_nethras") or []),
            hook=str(kwargs.get("hook", "")),
            use_right=str(kwargs.get("use_right", "record_only")),
            candidates_before=list(kwargs.get("candidates_before") or []),
            candidates_after=list(kwargs.get("candidates_after") or []),
            probes_before=list(kwargs.get("probes_before") or []),
            probes_after=list(kwargs.get("probes_after") or []),
            selected_candidate=kwargs.get("selected_candidate"),
            selected_probe=kwargs.get("selected_probe"),
            behavior_effect=int(kwargs.get("behavior_effect", 0) or 0),
            authority_effect=int(kwargs.get("authority_effect", 0) or 0),
            candidate_reduction_delta=int(kwargs.get("candidate_reduction_delta", 0) or 0),
            success=bool(kwargs.get("success", False)),
            failure_reason=str(kwargs.get("failure_reason", "")),
            evidence_refs=list(kwargs.get("evidence_refs") or []),
            hidden_truth_used=False,
        )
        self.events.append(event)

    def _example(self, item: dict[str, Any]) -> None:
        if len(self.metrics.examples) < 20:
            self.metrics.examples.append(item)


def _has_hidden_truth(row: dict[str, Any]) -> bool:
    stack = [row]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in HIDDEN_TRUTH_LIKE_FIELDS or str(key).startswith("debug_"):
                    return True
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return False


def _dominant_use_right(items: list[tuple[NethraMemoryRecord, SalienceExplanation]]) -> str:
    for right in ("soft_filter", "ranking_hint", "probe_hint", "feature_only", "record_only"):
        if any(record.use_right == right for record, _ in items):
            return right
    return "record_only"


def _intlike(value: Any) -> bool:
    try:
        int(value)
        return value is not None and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


def _effective_mind_use_right(use_rights_seen: list[str]) -> str:
    _rank = {"soft_filter": 4, "ranking_hint": 3, "probe_hint": 2, "feature_only": 1, "record_only": 0}
    best = "record_only"
    for right in use_rights_seen:
        if _rank.get(right, -1) > _rank.get(best, -1):
            best = right
    return best
