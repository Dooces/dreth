from __future__ import annotations

"""Runtime use-right enforcement for loaded persistent nethra handles.

Loaded memory is non-authoritative. This module may reorder existing candidate
or probe lists in assist mode, and it records attribution for both applied and
possible effects. It never writes certificates or mutates the ledger.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .memory_sleep import HIDDEN_TRUTH_LIKE_FIELDS
from .nethra_memory_store import ExperienceEvent, NethraMemoryRecord, USE_RIGHTS
from .nethra_projection import ProjectionIndex


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
class SalienceScorer:
    """Central salience scorer.

    Raw count is intentionally capped and outweighed by context match,
    specificity, success/lift, recency, and negative penalties.
    """

    current_cycle: int = 0

    def score(
        self,
        record: NethraMemoryRecord,
        *,
        active_atoms: Iterable[str] = (),
        context_key: str = "",
    ) -> SalienceExplanation:
        atoms = {str(a) for a in active_atoms}
        touched = {str(a) for a in record.touched_atoms}
        context_scope = str(record.context_scope or "")
        context_match = 1.5 if context_key and context_key == context_scope else 0.0
        if not context_match and context_key and context_scope:
            context_match = 0.4 if context_key.split("|", 1)[0] == context_scope.split("|", 1)[0] else -0.8
        atom_overlap = len(atoms & touched)
        specificity = 1.0 / max(1, len(touched))
        broad_penalty = -0.25 * max(0, len(touched) - 2)
        recency = 0.0
        last_cycle = max(record.last_used_cycle, record.last_success_cycle, record.created_cycle)
        if last_cycle:
            age = max(0, int(self.current_cycle or last_cycle) - int(last_cycle))
            recency = max(0.0, 1.0 - age / 500.0)
        success = min(1.5, 0.3 * record.success_count)
        failure = -0.7 * record.failure_count
        revocation = -0.5 * len(record.invalidators)
        prior_lift = 0.0
        audit_saved = 0.0
        candidate_lift = 0.0
        probe_lift = 0.0
        quality_penalty = 0.0
        sentinel_survival = 0.0
        for item in record.lift_history:
            if not isinstance(item, dict):
                continue
            prior_lift += float(item.get("prior_lift", 0.0) or 0.0)
            candidate_lift += float(item.get("candidate_reduction_lift", 0.0) or 0.0)
            probe_lift += float(item.get("probe_lift", 0.0) or 0.0)
            audit_saved += float(item.get("audit_saved_lift", 0.0) or 0.0)
            sentinel_survival += float(item.get("sentinel_survival", 0.0) or 0.0)
            q = float(item.get("quality_delta", 0.0) or 0.0)
            if q < 0:
                quality_penalty += q
        use_count = min(0.25, 0.03 * (record.success_count + record.failure_count))
        stale = -0.6 if recency == 0.0 and last_cycle else 0.0
        components = {
            "context_match": context_match,
            "active_atom_overlap": 0.5 * atom_overlap,
            "recency": recency,
            "prior_success": success,
            "prior_lift": min(1.0, prior_lift),
            "candidate_reduction_lift": min(1.0, candidate_lift),
            "probe_lift": min(1.0, probe_lift),
            "audit_saved": min(1.0, audit_saved),
            "sentinel_survival": min(1.0, sentinel_survival),
            "specificity": specificity,
            "raw_use_count_cap": use_count,
            "failure_count": failure,
            "revocation_count": revocation,
            "stale_evidence": stale,
            "context_mismatch": context_match if context_match < 0 else 0.0,
            "broad_atom_penalty": broad_penalty,
            "quality_regression": quality_penalty,
            "sentinel_failure": -0.7 if "sentinel_failure" in record.invalidators else 0.0,
            "base_salience": float(record.salience or 0.0),
        }
        return SalienceExplanation(
            score=sum(components.values()),
            components={k: round(v, 6) for k, v in components.items()},
        )


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
        self.events: list[ExperienceEvent] = []

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
        candidates: list[NethraMemoryRecord] = []
        for atom in atoms:
            for record in self._by_atom.get(atom, ()):
                if record.record_id not in seen:
                    seen.add(record.record_id)
                    candidates.append(record)
        scorer = SalienceScorer(current_cycle=cycle)
        scored = [
            (record, scorer.score(record, active_atoms=atoms, context_key=context_key))
            for record in candidates
            if record.use_right != "block"
        ]
        scored = [item for item in scored if item[1].score > -1.0]
        scored.sort(key=lambda item: (-item[1].score, item[0].record_id))
        self.metrics.matches += len(scored)
        return scored

    def rank_candidates(
        self,
        *,
        var: int,
        context_key: str,
        candidates: Any,
        hook: str,
        cycle: int,
    ) -> Any:
        original_type = type(candidates)
        original = list(candidates or [])
        if self.mode == "off":
            return candidates
        active_atoms = [f"x{int(var)}", *(f"x{int(c)}" for c in original if _intlike(c))]
        target_var_atom = f"x{int(var)}"
        if self._projection.size() > 0:
            proj_entries = self._projection.query(target_var_atom, context_key, hook, top_k=20)
            proj_ids = {e.nethra_id for e in proj_entries}
            projected_records = [r for r in self.records if r.record_id in proj_ids or r.nethra_id in proj_ids]
            if projected_records:
                scorer = SalienceScorer(current_cycle=cycle)
                atoms_set = set(active_atoms)
                scored_handles = [
                    (r, scorer.score(r, active_atoms=atoms_set, context_key=context_key))
                    for r in projected_records
                    if r.use_right != "block"
                ]
                scored_handles = [item for item in scored_handles if item[1].score > -1.0]
                scored_handles.sort(key=lambda item: (-item[1].score, item[0].record_id))
            else:
                scored_handles = self.query(active_atoms, context_key, cycle=cycle)
        else:
            scored_handles = self.query(active_atoms, context_key, cycle=cycle)
        usable = [
            (record, explanation)
            for record, explanation in scored_handles
            if record.use_right in BEHAVIOR_USE_RIGHTS or self.mode == "record"
        ]
        self._record_event(
            cycle=cycle,
            context_key=context_key,
            active_atoms=active_atoms,
            active_nethras=[r.nethra_id for r, _ in usable],
            hook=hook,
            use_right=_dominant_use_right(usable),
            candidates_before=original,
            candidates_after=original,
            behavior_effect=0,
            evidence_refs=[r.record_id for r, _ in usable],
        )
        if len(original) < 2 or self.mode != "assist" or not usable:
            return candidates

        candidate_scores: dict[Any, float] = {}
        reason_by_candidate: dict[Any, list[str]] = {}
        for candidate in original:
            catom = f"x{int(candidate)}" if _intlike(candidate) else str(candidate)
            total = 0.0
            reasons: list[str] = []
            for record, explanation in usable:
                if catom in set(record.touched_atoms):
                    total += explanation.score
                    reasons.extend(k for k, v in explanation.components.items() if v)
            candidate_scores[candidate] = total
            reason_by_candidate[candidate] = sorted(set(reasons))
        if not any(score > 0 for score in candidate_scores.values()):
            return candidates
        ranked = sorted(original, key=lambda c: (-candidate_scores.get(c, 0.0), original.index(c)))
        if ranked != original:
            self.metrics.behavior_effects += 1
            self.metrics.candidate_reorders += 1
            if any(record.use_right == "soft_filter" for record, _ in usable):
                self.metrics.soft_filter_fallbacks += 1
            self.metrics.used += len(usable)
            self.metrics.sleep_products_used += sum(1 for r, _ in usable if r.source == "sleep")
            for record, _ in usable:
                self.metrics.use_right_counts.update([record.use_right])
            self._record_event(
                cycle=cycle,
                context_key=context_key,
                active_atoms=active_atoms,
                active_nethras=[r.nethra_id for r, _ in usable],
                hook=hook,
                use_right=_dominant_use_right(usable),
                candidates_before=original,
                candidates_after=ranked,
                selected_candidate=ranked[0] if ranked else None,
                behavior_effect=1,
                candidate_reduction_delta=0,
                success=True,
                evidence_refs=[r.record_id for r, _ in usable],
            )
            self._example({
                "hook": hook,
                "context_key": context_key,
                "before": [_label(v) for v in original],
                "after": [_label(v) for v in ranked],
                "reason_by_candidate": {
                    _label(k): v for k, v in reason_by_candidate.items() if v
                },
            })
        return _restore_type(ranked, original_type)

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
        target_var_atom = f"x{int(var)}"
        if self._projection.size() > 0:
            proj_entries = self._projection.query(target_var_atom, context_key, "probe_hint", top_k=20)
            proj_ids = {e.nethra_id for e in proj_entries}
            projected_records = [r for r in self.records if r.record_id in proj_ids or r.nethra_id in proj_ids]
            if projected_records:
                scorer = SalienceScorer(current_cycle=cycle)
                atoms_set = set(active_atoms)
                scored_all = [
                    (r, scorer.score(r, active_atoms=atoms_set, context_key=context_key))
                    for r in projected_records
                    if r.use_right != "block"
                ]
                scored_all = [item for item in scored_all if item[1].score > -1.0]
                scored_all.sort(key=lambda item: (-item[1].score, item[0].record_id))
            else:
                scored_all = self.query(active_atoms, context_key, cycle=cycle)
        else:
            scored_all = self.query(active_atoms, context_key, cycle=cycle)
        usable = [
            (record, explanation)
            for record, explanation in scored_all
            if record.use_right == "probe_hint" or self.mode == "record"
        ]
        self._record_event(
            cycle=cycle,
            context_key=context_key,
            active_atoms=active_atoms,
            active_nethras=[r.nethra_id for r, _ in usable],
            hook="probe_hint",
            use_right="probe_hint" if usable else "record_only",
            probes_before=list(original),
            probes_after=list(original),
            evidence_refs=[r.record_id for r, _ in usable],
        )
        if len(original) < 2 or self.mode != "assist" or not usable:
            return original
        scores: dict[tuple[int, float], float] = {}
        for probe in original:
            atom = f"x{int(probe[0])}"
            scores[probe] = sum(
                explanation.score
                for record, explanation in usable
                if atom in set(record.touched_atoms)
            )
        if not any(score > 0 for score in scores.values()):
            return original
        ranked = tuple(sorted(original, key=lambda p: (-scores.get(p, 0.0), original.index(p))))
        if ranked != original:
            self.metrics.behavior_effects += 1
            self.metrics.probe_reorders += 1
            self.metrics.used += len(usable)
            self.metrics.sleep_products_used += sum(1 for r, _ in usable if r.source == "sleep")
            for record, _ in usable:
                self.metrics.use_right_counts.update([record.use_right])
            self._record_event(
                cycle=cycle,
                context_key=context_key,
                active_atoms=active_atoms,
                active_nethras=[r.nethra_id for r, _ in usable],
                hook="probe_hint",
                use_right="probe_hint",
                probes_before=list(original),
                probes_after=list(ranked),
                selected_probe=ranked[0] if ranked else None,
                behavior_effect=1,
                success=True,
                evidence_refs=[r.record_id for r, _ in usable],
            )
        return ranked

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
                source="mind",
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
                        "parents:" + ",".join(str(int(p)) for p in parents)
                        for parents in (row.get("common_parents") or [])
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


def _restore_type(values: list[Any], original_type: type) -> Any:
    if original_type is tuple:
        return tuple(values)
    if original_type is set:
        return set(values)
    if original_type is frozenset:
        return frozenset(values)
    return values


def _intlike(value: Any) -> bool:
    try:
        int(value)
        return value is not None and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


def _label(value: Any) -> str:
    if _intlike(value):
        return f"x{int(value)}"
    return str(value)


def _effective_mind_use_right(use_rights_seen: list[str]) -> str:
    _rank = {"soft_filter": 4, "ranking_hint": 3, "probe_hint": 2, "feature_only": 1, "record_only": 0}
    best = "record_only"
    for right in use_rights_seen:
        if _rank.get(right, -1) > _rank.get(best, -1):
            best = right
    return best
