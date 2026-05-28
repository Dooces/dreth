from __future__ import annotations

"""Canonical persistent nethra graph.

Folds repeated records/sleep products/events into stable NethraMindNode
structures without losing provenance. This module does not issue authority,
mutate the ledger, revoke certs, suppress skips, or convert sleep products to
hard filters.

Core invariant: authority_allowed and authority_effect_count are always zero.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dreth.nethra_memory_store import HIDDEN_TRUTH_LIKE_FIELDS, USE_RIGHTS
from .nethra_assimilator import NethraAssimilator, Disposition
from .nethra_projection import ProjectionIndex


_USE_RIGHT_RANK: dict[str, int] = {
    "soft_filter": 4,
    "ranking_hint": 3,
    "probe_hint": 2,
    "feature_only": 1,
    "record_only": 0,
}

# Active-mind caps
_MAX_NODES = 500
_MAX_EDGES = 2000
_MAX_SAMPLE_REFS = 4
_MAX_TEMPORAL_SPANS = 16
_MAX_LIFT_HISTORY = 8
_MAX_CONTEXTS = 16
_MAX_TOUCHED_ATOMS = 64
_MAX_TOUCHED_STRUCTURE_REFS = 32
_MAX_MEMBER_NETHRAS = 32
_MAX_SURFACE_TRANSITIONS = 32

# Ingestion policy: reject these entry_kinds and sources from the delta stream.
# They are outputs of compaction, not world-backed evidence.
_REJECT_ENTRY_KINDS: frozenset[str] = frozenset({
    "nethra_mind_node",
    "nethra_mind_edge",
    "nethra_mind_summary",
})
_REJECT_SOURCES: frozenset[str] = frozenset({
    "mind",
    "sleep_derived_mind",
    "compacted",
})


@dataclass
class NethraMindNode:
    nethra_id: str
    kind: str = ""
    touched_atoms: list[str] = field(default_factory=list)
    touched_structure_refs: list[str] = field(default_factory=list)
    member_nethras: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    roles_by_context: dict[str, list[str]] = field(default_factory=dict)
    use_rights_seen: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    invalidator_counts: dict[str, int] = field(default_factory=dict)
    evidence_count: int = 0
    behavior_effect_count: int = 0
    authority_effect_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    candidate_reorder_count: int = 0
    probe_reorder_count: int = 0
    sleep_product_count: int = 0
    first_seen_line: int = 0
    last_seen_line: int = 0
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
    first_seen_generation: int = 0
    last_seen_generation: int = 0
    salience: float = 0.0
    lift_history: list[dict[str, Any]] = field(default_factory=list)
    invalidators: list[str] = field(default_factory=list)
    sample_evidence_refs: list[str] = field(default_factory=list)
    temporal_spans: list[dict[str, Any]] = field(default_factory=list)
    role_surfaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    residual_buckets: dict[str, dict[str, Any]] = field(default_factory=dict)
    surface_transitions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_kind": "nethra_mind_node",
            "nethra_id": self.nethra_id,
            "kind": self.kind,
            "touched_atoms": self.touched_atoms,
            "touched_structure_refs": self.touched_structure_refs,
            "member_nethras": self.member_nethras,
            "contexts": self.contexts,
            "use_rights_seen": self.use_rights_seen,
            "use_right_summary": effective_use_right(self.use_rights_seen),
            "source_counts": self.source_counts,
            "invalidators": self.invalidators,
            "invalidator_counts": self.invalidator_counts,
            "evidence_count": self.evidence_count,
            "behavior_effect_count": self.behavior_effect_count,
            "authority_effect_count": 0,
            "authority_allowed": False,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "candidate_reorder_count": self.candidate_reorder_count,
            "probe_reorder_count": self.probe_reorder_count,
            "sleep_product_count": self.sleep_product_count,
            "first_seen_line": self.first_seen_line,
            "last_seen_line": self.last_seen_line,
            "first_seen_cycle": self.first_seen_cycle,
            "last_seen_cycle": self.last_seen_cycle,
            "first_seen_generation": self.first_seen_generation,
            "last_seen_generation": self.last_seen_generation,
            "salience": self.salience,
            "lift_history": self.lift_history[-_MAX_LIFT_HISTORY:] if self.lift_history else [],
            "sample_evidence_refs": self.sample_evidence_refs,
            "temporal_span_summary": {
                "first_cycle": self.first_seen_cycle,
                "last_cycle": self.last_seen_cycle,
                "first_generation": self.first_seen_generation,
                "last_generation": self.last_seen_generation,
                "span_count": len(self.temporal_spans),
            },
            "role_surfaces": dict(self.role_surfaces),
            "residual_buckets": dict(self.residual_buckets),
            "surface_transitions": self.surface_transitions[:_MAX_SURFACE_TRANSITIONS],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NethraMindNode":
        # Handle use_right_summary → use_rights_seen (new format) with fallback to old list
        use_rights_seen = [str(r) for r in (d.get("use_rights_seen") or [])]
        if not use_rights_seen:
            urs = str(d.get("use_right_summary", ""))
            if urs:
                use_rights_seen = [urs]

        # Handle invalidator_counts → invalidators (new format) with fallback to old list
        invalidator_counts: dict[str, int] = {
            str(k): int(v)
            for k, v in (d.get("invalidator_counts") or {}).items()
        }
        invalidators_from_list = [str(i) for i in (d.get("invalidators") or [])]
        if invalidator_counts:
            # Merge: counts take precedence, add any extras from old list
            all_invs = list(dict.fromkeys(list(invalidator_counts.keys()) + invalidators_from_list))
            invalidators = all_invs
        else:
            invalidators = invalidators_from_list
            invalidator_counts = {inv: 1 for inv in invalidators}

        return cls(
            nethra_id=str(d.get("nethra_id", "")),
            kind=str(d.get("kind", "")),
            touched_atoms=[str(a) for a in (d.get("touched_atoms") or [])],
            touched_structure_refs=[str(r) for r in (d.get("touched_structure_refs") or [])],
            member_nethras=[str(n) for n in (d.get("member_nethras") or [])],
            contexts=[str(c) for c in (d.get("contexts") or [])],
            roles_by_context={
                str(k): [str(r) for r in v]
                for k, v in (d.get("roles_by_context") or {}).items()
            },
            use_rights_seen=use_rights_seen,
            source_counts={str(k): int(v) for k, v in (d.get("source_counts") or {}).items()},
            invalidator_counts=invalidator_counts,
            evidence_count=int(d.get("evidence_count", 0) or 0),
            behavior_effect_count=int(d.get("behavior_effect_count", 0) or 0),
            authority_effect_count=0,
            success_count=int(d.get("success_count", 0) or 0),
            failure_count=int(d.get("failure_count", 0) or 0),
            candidate_reorder_count=int(d.get("candidate_reorder_count", 0) or 0),
            probe_reorder_count=int(d.get("probe_reorder_count", 0) or 0),
            sleep_product_count=int(d.get("sleep_product_count", 0) or 0),
            first_seen_line=int(d.get("first_seen_line", 0) or 0),
            last_seen_line=int(d.get("last_seen_line", 0) or 0),
            first_seen_cycle=int(d.get("first_seen_cycle", 0) or 0),
            last_seen_cycle=int(d.get("last_seen_cycle", 0) or 0),
            first_seen_generation=int(d.get("first_seen_generation", 0) or 0),
            last_seen_generation=int(d.get("last_seen_generation", 0) or 0),
            salience=float(d.get("salience", 0.0) or 0.0),
            lift_history=[r for r in (d.get("lift_history") or []) if isinstance(r, dict)],
            invalidators=invalidators,
            sample_evidence_refs=[str(r) for r in (d.get("sample_evidence_refs") or [])],
            temporal_spans=[],  # not persisted in compact form; use temporal_span_summary
            role_surfaces={
                str(k): dict(v) for k, v in (d.get("role_surfaces") or {}).items()
                if isinstance(v, dict)
            },
            residual_buckets={
                str(k): dict(v) for k, v in (d.get("residual_buckets") or {}).items()
                if isinstance(v, dict)
            },
            surface_transitions=[
                r for r in (d.get("surface_transitions") or []) if isinstance(r, dict)
            ],
        )


@dataclass
class NethraMindEdge:
    src: str
    dst: str
    relation: str
    context: str = ""
    count: int = 1
    first_seen: int = 0
    last_seen: int = 0
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_kind": "nethra_mind_edge",
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "context": self.context,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "evidence_refs": self.evidence_refs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NethraMindEdge":
        return cls(
            src=str(d.get("src", "")),
            dst=str(d.get("dst", "")),
            relation=str(d.get("relation", "")),
            context=str(d.get("context", "")),
            count=int(d.get("count", 1) or 1),
            first_seen=int(d.get("first_seen", 0) or 0),
            last_seen=int(d.get("last_seen", 0) or 0),
            evidence_refs=[str(r) for r in (d.get("evidence_refs") or [])],
        )


class NethraMindStore:
    def __init__(self) -> None:
        self._nodes: dict[str, NethraMindNode] = {}
        self._edges: dict[tuple[str, str, str], NethraMindEdge] = {}
        self._raw_row_count: int = 0
        self._skipped_hidden_truth: int = 0
        self._source_row_counts: Counter[str] = Counter()
        # Rejection counters
        self._rows_rejected_mind_derived: int = 0
        self._rows_rejected_compacted: int = 0
        # Fold counters
        self._exact_folds: int = 0
        self._structural_folds: int = 0
        self._assimilation_folds: int = 0
        # Pruning counters
        self._nodes_pruned: int = 0
        self._edges_pruned: int = 0
        # Snapshot before delta ingestion (set by snapshot_delta_start())
        self._nodes_before_delta: int = 0
        self._edges_before_delta: int = 0
        # Understanding components
        self._assimilator: NethraAssimilator = NethraAssimilator()
        self._projection: ProjectionIndex = ProjectionIndex()

    def snapshot_delta_start(self) -> None:
        """Record node/edge counts after loading previous mind, before ingesting delta."""
        self._nodes_before_delta = len(self._nodes)
        self._edges_before_delta = len(self._edges)

    def _check_and_reject(self, row: dict[str, Any]) -> bool:
        """Return True if this row must be rejected as mind-derived or compacted output."""
        entry_kind = str(row.get("entry_kind", ""))
        if entry_kind in _REJECT_ENTRY_KINDS:
            self._rows_rejected_mind_derived += 1
            return True
        source = str(row.get("source", ""))
        if source in _REJECT_SOURCES:
            if source in ("mind", "sleep_derived_mind"):
                self._rows_rejected_mind_derived += 1
            else:
                self._rows_rejected_compacted += 1
            return True
        return False

    def load(self, path: str | Path) -> int:
        """Load a previously compacted mind JSONL into the node/edge index.

        This is for loading a *previous* mind as the base for delta compaction.
        It does NOT reject mind-derived entry_kinds — those are the expected format here.
        It does NOT count towards rejection counters.
        """
        path = Path(path)
        if not path.exists():
            return 0
        loaded = 0
        pending_residuals: list[dict[str, Any]] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                entry_kind = str(row.get("entry_kind", ""))
                if entry_kind == "nethra_mind_node":
                    node = NethraMindNode.from_dict(row)
                    if node.nethra_id:
                        self._nodes[node.nethra_id] = node
                        self._assimilator.index_node(
                            node.nethra_id,
                            node.touched_atoms,
                            structure_refs=node.touched_structure_refs,
                            member_nethras=node.member_nethras,
                            contexts=node.contexts,
                            use_rights_seen=node.use_rights_seen,
                            roles_by_context=node.roles_by_context,
                        )
                        self._projection.index_node(node)
                        loaded += 1
                elif entry_kind == "nethra_mind_edge":
                    edge = NethraMindEdge.from_dict(row)
                    if edge.src and edge.dst and edge.relation:
                        key = (edge.src, edge.dst, edge.relation)
                        self._edges[key] = edge
                        self._assimilator.index_edge(edge.src, edge.dst)
                        loaded += 1
                elif entry_kind == "nethra_residual":
                    pending_residuals.append(row)
                    loaded += 1
        if pending_residuals:
            self._assimilator.residuals.load_rows(pending_residuals)
        return loaded

    def ingest_record(self, row: dict[str, Any], line_no: int, generation: int) -> str | None:
        if _has_hidden_truth(row):
            self._skipped_hidden_truth += 1
            return None
        if self._check_and_reject(row):
            return None

        self._raw_row_count += 1
        self._source_row_counts["record"] += 1

        nethra_id = str(row.get("nethra_id", "") or row.get("record_id", ""))
        explicit_id = bool(nethra_id)
        kind = str(row.get("record_type", "") or row.get("source_kind", ""))
        touched_atoms = [str(a) for a in (row.get("touched_atoms") or [])]
        touched_structure_refs = [str(r) for r in (row.get("touched_structure_refs") or [])]
        member_nethras = [str(n) for n in (row.get("member_nethras") or [])]
        context_scope = str(row.get("context_scope", ""))
        contexts = [str(c) for c in (row.get("contexts") or [])]
        if context_scope and context_scope not in contexts:
            contexts = [context_scope] + contexts
        use_right = str(row.get("use_right", "record_only"))
        if use_right not in USE_RIGHTS:
            use_right = "record_only"
        source = str(row.get("source", "runtime"))

        if not nethra_id:
            nethra_id = _structural_id(
                kind, touched_atoms, touched_structure_refs,
                member_nethras, context_scope, use_right,
            )

        # Track fold type before upsert
        is_fold = nethra_id in self._nodes
        if is_fold:
            if explicit_id:
                self._exact_folds += 1
            else:
                self._structural_folds += 1

        cycle_start = int(row.get("created_cycle", 0) or row.get("cycle_start", 0) or 0)
        cycle_end = int(row.get("last_used_cycle", 0) or row.get("cycle_end", 0) or 0)
        salience = float(row.get("salience", 0.0) or 0.0)

        role_history = [r for r in (row.get("role_history") or []) if isinstance(r, dict)]
        roles_by_context: dict[str, list[str]] = {}
        for rh in role_history:
            ctx = str(rh.get("context", context_scope or ""))
            role = str(rh.get("role", ""))
            if role:
                roles_by_context.setdefault(ctx, [])
                if role not in roles_by_context[ctx]:
                    roles_by_context[ctx].append(role)

        evidence_ref = str(row.get("record_id", nethra_id))

        # For genuinely new rows (no exact/structural fold), run assimilator
        if not is_fold:
            result = self._assimilator.explain(row, self._nodes)
            disp = result.disposition
            if disp == Disposition.NOISE:
                self._raw_row_count -= 1
                return None
            elif disp == Disposition.RESIDUAL:
                if result.has_candidates:
                    # Overlapping structures exist but none strong enough → hold pending
                    self._assimilator.residuals.add(row)
                    return None
                # No overlapping structures at all → genuinely new pattern, fall through to create node
            elif disp == Disposition.ASSIMILATED:
                self._assimilation_folds += 1
                nethra_id = result.explained_by  # type: ignore[assignment]
            elif disp == Disposition.CONTRADICTION:
                if result.explained_by and result.explained_by in self._nodes:
                    node = self._nodes[result.explained_by]
                    node.failure_count += max(1, int(row.get("failure_count", 0) or 0))
                    for inv in [str(i) for i in (row.get("invalidators") or [])]:
                        node.invalidator_counts[inv] = node.invalidator_counts.get(inv, 0) + 1
                        if inv not in set(node.invalidators):
                            node.invalidators.append(inv)
                return result.explained_by
            # SPLIT_CANDIDATE: proceed normally — new node develops context variant

        self.upsert_node(
            nethra_id,
            kind=kind,
            touched_atoms=touched_atoms,
            touched_structure_refs=touched_structure_refs,
            member_nethras=member_nethras,
            contexts=contexts,
            roles_by_context=roles_by_context,
            use_right=use_right,
            source=source,
            evidence_ref=evidence_ref,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            line_no=line_no,
            generation=generation,
            salience=salience,
            lift_history=[r for r in (row.get("lift_history") or []) if isinstance(r, dict)],
            invalidators=[str(i) for i in (row.get("invalidators") or [])],
            success_count=int(row.get("success_count", 0) or 0),
            failure_count=int(row.get("failure_count", 0) or 0),
        )

        for member_id in member_nethras:
            if member_id and member_id != nethra_id:
                self.upsert_edge(
                    src=nethra_id,
                    dst=member_id,
                    relation="member_of",
                    context=context_scope,
                    cycle=cycle_start,
                    evidence_ref=evidence_ref,
                )

        return nethra_id

    def ingest_experience_event(self, row: dict[str, Any], line_no: int, generation: int) -> str | None:
        if _has_hidden_truth(row) or bool(row.get("hidden_truth_used")):
            self._skipped_hidden_truth += 1
            return None
        source = str(row.get("source", ""))
        if source in _REJECT_SOURCES:
            if source in ("mind", "sleep_derived_mind"):
                self._rows_rejected_mind_derived += 1
            else:
                self._rows_rejected_compacted += 1
            return None

        self._raw_row_count += 1
        self._source_row_counts["experience_event"] += 1

        active_nethras = [str(n) for n in (row.get("active_nethras") or [])]
        behavior_effect = int(row.get("behavior_effect", 0) or 0)
        cycle = int(row.get("cycle", 0) or 0)
        hook = str(row.get("hook", ""))
        success = bool(row.get("success", False))

        is_candidate_reorder = hook in ("parent_candidates", "ranking_hint") and behavior_effect > 0
        is_probe_reorder = hook == "probe_hint" and behavior_effect > 0

        for nethra_id in active_nethras:
            if not nethra_id or nethra_id not in self._nodes:
                continue
            node = self._nodes[nethra_id]
            node.behavior_effect_count += behavior_effect
            node.authority_effect_count = 0
            if success and behavior_effect > 0:
                node.success_count += 1
            elif not success and behavior_effect > 0:
                node.failure_count += 1
            if is_candidate_reorder:
                node.candidate_reorder_count += 1
            if is_probe_reorder:
                node.probe_reorder_count += 1
            if cycle > node.last_seen_cycle:
                node.last_seen_cycle = cycle

        return None

    def ingest_sleep_product(self, row: dict[str, Any], line_no: int, generation: int) -> str | None:
        if _has_hidden_truth(row):
            self._skipped_hidden_truth += 1
            return None
        # Sleep products (entry_kind=sleep_product, source=sleep) ARE allowed.
        # Only reject if the entry_kind is a mind-compaction output kind.
        entry_kind = str(row.get("entry_kind", ""))
        if entry_kind in _REJECT_ENTRY_KINDS:
            self._rows_rejected_mind_derived += 1
            return None

        self._raw_row_count += 1
        self._source_row_counts["sleep_product"] += 1

        nethra_id = str(row.get("proposal_id", ""))
        explicit_id = bool(nethra_id)
        member_nethras = [str(n) for n in (row.get("member_nethras") or [])]
        touched_atoms = [str(a) for a in (row.get("touched_atoms") or [])]
        touched_structure_refs = [str(r) for r in (row.get("touched_structure_refs") or [])]
        proposed_use_right = str(row.get("proposed_use_right", "feature_only"))
        proposed_context_scope = str(row.get("proposed_context_scope", ""))
        salience_delta = float(row.get("salience_delta", 0.0) or 0.0)
        invalidators = [str(i) for i in (row.get("invalidators") or [])]

        if proposed_use_right == "hard_filter":
            proposed_use_right = "record_only"
            if "sleep_hard_filter_rejected" not in invalidators:
                invalidators.append("sleep_hard_filter_rejected")
        if proposed_use_right not in USE_RIGHTS:
            proposed_use_right = "feature_only"

        if not nethra_id:
            nethra_id = _structural_id(
                "sleep_product", touched_atoms, touched_structure_refs,
                member_nethras, proposed_context_scope, proposed_use_right,
            )

        # Track fold type before upsert
        is_fold = nethra_id in self._nodes
        if is_fold:
            if explicit_id:
                self._exact_folds += 1
            else:
                self._structural_folds += 1

        # For genuinely new sleep products, run assimilator
        if not is_fold:
            result = self._assimilator.explain(row, self._nodes)
            disp = result.disposition
            if disp == Disposition.NOISE:
                self._raw_row_count -= 1
                return None
            elif disp == Disposition.RESIDUAL:
                if result.has_candidates:
                    self._assimilator.residuals.add(row)
                    return None
                # No candidates → genuinely new pattern, fall through to create node
            elif disp == Disposition.ASSIMILATED:
                self._assimilation_folds += 1
                nethra_id = result.explained_by  # type: ignore[assignment]
            elif disp == Disposition.CONTRADICTION:
                if result.explained_by and result.explained_by in self._nodes:
                    node = self._nodes[result.explained_by]
                    node.failure_count += max(1, len(invalidators))
                    for inv in invalidators:
                        node.invalidator_counts[inv] = node.invalidator_counts.get(inv, 0) + 1
                        if inv not in set(node.invalidators):
                            node.invalidators.append(inv)
                return result.explained_by
            # SPLIT_CANDIDATE: proceed — new node develops context variant

        self.upsert_node(
            nethra_id,
            kind="sleep_product",
            touched_atoms=touched_atoms,
            touched_structure_refs=touched_structure_refs,
            member_nethras=member_nethras,
            contexts=[proposed_context_scope] if proposed_context_scope else [],
            use_right=proposed_use_right,
            source="sleep",
            evidence_ref=nethra_id,
            cycle_start=0,
            cycle_end=0,
            line_no=line_no,
            generation=generation,
            salience=salience_delta,
            invalidators=invalidators,
            is_sleep_product=True,
        )

        for member_id in member_nethras:
            if member_id and member_id != nethra_id:
                self.upsert_edge(
                    src=nethra_id,
                    dst=member_id,
                    relation="member_of",
                    context=proposed_context_scope,
                    cycle=0,
                    evidence_ref=nethra_id,
                )

        return nethra_id

    def upsert_node(
        self,
        nethra_id: str,
        *,
        kind: str = "",
        touched_atoms: list[str] | None = None,
        touched_structure_refs: list[str] | None = None,
        member_nethras: list[str] | None = None,
        contexts: list[str] | None = None,
        roles_by_context: dict[str, list[str]] | None = None,
        use_right: str = "",
        source: str = "",
        evidence_ref: str = "",
        cycle_start: int = 0,
        cycle_end: int = 0,
        line_no: int = 0,
        generation: int = 0,
        salience: float = 0.0,
        lift_history: list[dict[str, Any]] | None = None,
        invalidators: list[str] | None = None,
        success_count: int = 0,
        failure_count: int = 0,
        behavior_effect: int = 0,
        candidate_reorder: int = 0,
        probe_reorder: int = 0,
        is_sleep_product: bool = False,
        evidence_count_delta: int = 1,
    ) -> NethraMindNode:
        if nethra_id not in self._nodes:
            self._nodes[nethra_id] = NethraMindNode(nethra_id=nethra_id)
        node = self._nodes[nethra_id]

        if kind and not node.kind:
            node.kind = kind

        if touched_atoms:
            existing = set(node.touched_atoms)
            for a in touched_atoms:
                if a not in existing and len(node.touched_atoms) < _MAX_TOUCHED_ATOMS:
                    node.touched_atoms.append(a)
                    existing.add(a)

        if touched_structure_refs:
            existing = set(node.touched_structure_refs)
            for r in touched_structure_refs:
                if r not in existing and len(node.touched_structure_refs) < _MAX_TOUCHED_STRUCTURE_REFS:
                    node.touched_structure_refs.append(r)
                    existing.add(r)

        if member_nethras:
            existing = set(node.member_nethras)
            for n in member_nethras:
                if n not in existing and len(node.member_nethras) < _MAX_MEMBER_NETHRAS:
                    node.member_nethras.append(n)
                    existing.add(n)

        if contexts:
            existing = set(node.contexts)
            for c in contexts:
                if c and c not in existing and len(node.contexts) < _MAX_CONTEXTS:
                    node.contexts.append(c)
                    existing.add(c)

        if roles_by_context:
            ctx_count = len(node.roles_by_context)
            for ctx, roles in roles_by_context.items():
                if ctx not in node.roles_by_context:
                    if ctx_count >= _MAX_CONTEXTS:
                        continue
                    node.roles_by_context[ctx] = []
                    ctx_count += 1
                existing_roles = set(node.roles_by_context[ctx])
                for role in roles:
                    if role not in existing_roles:
                        node.roles_by_context[ctx].append(role)
                        existing_roles.add(role)

        if use_right and use_right not in node.use_rights_seen:
            node.use_rights_seen.append(use_right)

        if source:
            node.source_counts[source] = node.source_counts.get(source, 0) + 1

        node.evidence_count += evidence_count_delta
        node.success_count += success_count
        node.failure_count += failure_count
        node.behavior_effect_count += behavior_effect
        node.authority_effect_count = 0
        node.candidate_reorder_count += candidate_reorder
        node.probe_reorder_count += probe_reorder
        if is_sleep_product:
            node.sleep_product_count += 1

        # Temporal provenance
        cycle_min = cycle_start
        cycle_max = max(cycle_start, cycle_end)
        if line_no > 0:
            if node.first_seen_line == 0 or line_no < node.first_seen_line:
                node.first_seen_line = line_no
            if line_no > node.last_seen_line:
                node.last_seen_line = line_no
        if cycle_min > 0:
            if node.first_seen_cycle == 0 or cycle_min < node.first_seen_cycle:
                node.first_seen_cycle = cycle_min
        if cycle_max > node.last_seen_cycle:
            node.last_seen_cycle = cycle_max
        if generation > 0:
            if node.first_seen_generation == 0:
                node.first_seen_generation = generation
            if generation > node.last_seen_generation:
                node.last_seen_generation = generation

        if salience > node.salience:
            node.salience = salience

        if lift_history:
            node.lift_history.extend(lift_history[:4])
            if len(node.lift_history) > _MAX_LIFT_HISTORY:
                node.lift_history = node.lift_history[-_MAX_LIFT_HISTORY:]

        if invalidators:
            existing_inv = set(node.invalidators)
            for inv in invalidators:
                node.invalidator_counts[inv] = node.invalidator_counts.get(inv, 0) + 1
                if inv not in existing_inv:
                    node.invalidators.append(inv)
                    existing_inv.add(inv)

        if evidence_ref and len(node.sample_evidence_refs) < _MAX_SAMPLE_REFS:
            if evidence_ref not in node.sample_evidence_refs:
                node.sample_evidence_refs.append(evidence_ref)

        if (cycle_min > 0 or cycle_max > 0) and len(node.temporal_spans) < _MAX_TEMPORAL_SPANS:
            node.temporal_spans.append({
                "cycle_start": cycle_min,
                "cycle_end": cycle_max,
                "generation": generation,
                "line_no": line_no,
            })

        # Keep all sub-indexes in sync after node creation/update
        self._assimilator.index_node(
            nethra_id,
            node.touched_atoms,
            structure_refs=node.touched_structure_refs,
            member_nethras=node.member_nethras,
            contexts=node.contexts,
            use_rights_seen=node.use_rights_seen,
            roles_by_context=node.roles_by_context,
        )
        self._projection.index_node(node)

        return node

    def upsert_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        *,
        context: str = "",
        cycle: int = 0,
        evidence_ref: str = "",
    ) -> NethraMindEdge:
        key = (src, dst, relation)
        if key not in self._edges:
            self._edges[key] = NethraMindEdge(
                src=src,
                dst=dst,
                relation=relation,
                context=context,
                count=0,
                first_seen=cycle,
                last_seen=cycle,
            )
            self._assimilator.index_edge(src, dst)
        edge = self._edges[key]
        edge.count += 1
        if cycle > 0:
            if edge.first_seen == 0 or cycle < edge.first_seen:
                edge.first_seen = cycle
            if cycle > edge.last_seen:
                edge.last_seen = cycle
        if context and not edge.context:
            edge.context = context
        if evidence_ref and len(edge.evidence_refs) < 4:
            if evidence_ref not in edge.evidence_refs:
                edge.evidence_refs.append(evidence_ref)
        return edge

    def prune_to_cap(self) -> None:
        """Prune nodes and edges to _MAX_NODES/_MAX_EDGES when caps are exceeded.

        Safety nodes (failure_count > 0, non-empty invalidators, behavior_effect > 0)
        are always preserved. Remaining slots filled by highest-salience nodes.
        Invalidators and failure counts from pruned nodes are kept in summary counters
        but the nodes themselves are dropped.
        """
        if len(self._nodes) > _MAX_NODES:
            priority: dict[str, NethraMindNode] = {}
            normal: dict[str, NethraMindNode] = {}
            for nid, node in self._nodes.items():
                if node.failure_count > 0 or node.invalidators or node.behavior_effect_count > 0:
                    priority[nid] = node
                else:
                    normal[nid] = node

            # Safety nodes fill first; if over cap, sort by failure_count descending
            if len(priority) > _MAX_NODES:
                sorted_p = sorted(
                    priority.values(),
                    key=lambda n: -(n.failure_count * 10 + len(n.invalidators) + n.behavior_effect_count),
                )
                keep = {n.nethra_id: n for n in sorted_p[:_MAX_NODES]}
            else:
                keep = dict(priority)

            remaining = _MAX_NODES - len(keep)
            if remaining > 0:
                sorted_normal = sorted(
                    normal.values(),
                    key=lambda n: -(n.salience + n.evidence_count * 0.001 + n.sleep_product_count * 0.01),
                )
                for node in sorted_normal[:remaining]:
                    keep[node.nethra_id] = node

            pruned_ids = set(self._nodes.keys()) - set(keep.keys())
            for nid in pruned_ids:
                self._assimilator.remove_node(nid)
                self._projection.remove_node(nid)
            self._nodes_pruned += len(pruned_ids)
            self._nodes = keep

        if len(self._edges) > _MAX_EDGES:
            surviving_ids = set(self._nodes.keys())
            valid: dict[tuple[str, str, str], NethraMindEdge] = {
                k: e for k, e in self._edges.items()
                if e.src in surviving_ids and e.dst in surviving_ids
            }
            pruned_by_node_drop = len(self._edges) - len(valid)
            if len(valid) > _MAX_EDGES:
                sorted_edges = sorted(valid.values(), key=lambda e: -e.count)
                self._edges = {(e.src, e.dst, e.relation): e for e in sorted_edges[:_MAX_EDGES]}
                self._edges_pruned += pruned_by_node_drop + (len(valid) - _MAX_EDGES)
            else:
                self._edges_pruned += pruned_by_node_drop
                self._edges = valid

    def write_compact(
        self,
        path: str | Path,
        *,
        nodes_before: int | None = None,
        edges_before: int | None = None,
    ) -> dict[str, Any]:
        self.prune_to_cap()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        canonical_nodes = len(self._nodes)
        canonical_edges = len(self._edges)
        raw_rows = self._raw_row_count
        compression_ratio = raw_rows / max(1, canonical_nodes) if canonical_nodes > 0 else 1.0

        if nodes_before is None:
            nodes_before = self._nodes_before_delta
        if edges_before is None:
            edges_before = self._edges_before_delta

        sleep_products_folded = sum(n.sleep_product_count for n in self._nodes.values())
        top_folded = sorted(
            self._nodes.values(),
            key=lambda n: -(n.evidence_count + n.sleep_product_count),
        )[:5]

        assimilation_stats = self._assimilator.stats()
        residual_rows = self._assimilator.residuals.rows()

        with open(path, "w") as fh:
            for node in sorted(self._nodes.values(), key=lambda n: n.nethra_id):
                fh.write(json.dumps(node.to_dict(), sort_keys=True) + "\n")
            for edge in sorted(
                self._edges.values(), key=lambda e: (e.src, e.dst, e.relation)
            ):
                fh.write(json.dumps(edge.to_dict(), sort_keys=True) + "\n")
            for residual_row in residual_rows:
                fh.write(json.dumps({
                    "entry_kind": "nethra_residual",
                    **{k: v for k, v in residual_row.items() if k != "entry_kind"},
                }, sort_keys=True) + "\n")
            summary = {
                "entry_kind": "nethra_mind_summary",
                "authority_allowed": False,
                "raw_rows_read": raw_rows,
                "rows_rejected_mind_derived": self._rows_rejected_mind_derived,
                "rows_rejected_compacted": self._rows_rejected_compacted,
                "rows_ingested": raw_rows,
                "nodes_before": nodes_before,
                "nodes_after": canonical_nodes,
                "edges_before": edges_before,
                "edges_after": canonical_edges,
                "exact_folds": self._exact_folds,
                "structural_folds": self._structural_folds,
                "assimilation_folds": self._assimilation_folds,
                "sleep_products_folded": sleep_products_folded,
                "nodes_pruned": self._nodes_pruned,
                "edges_pruned": self._edges_pruned,
                "residuals_kept": len(residual_rows),
                "residuals_evicted": self._assimilator.residuals.total_evicted,
                "active_mind_bytes": 0,  # updated below after file is written
                "estimated_raw_bytes_folded": 0,
                "compression_ratio": round(compression_ratio, 4),
                "top_folded_structures": [
                    {
                        "nethra_id": n.nethra_id,
                        "evidence_count": n.evidence_count,
                        "sleep_product_count": n.sleep_product_count,
                    }
                    for n in top_folded
                ],
                "canonical_nodes": canonical_nodes,
                "canonical_edges": canonical_edges,
                "skipped_hidden_truth": self._skipped_hidden_truth,
                "source_row_counts": dict(self._source_row_counts),
                "assimilation_stats": assimilation_stats,
                "NOTICE": "compacted mind is not authority",
            }
            fh.write(json.dumps(summary, sort_keys=True) + "\n")

        active_mind_bytes = path.stat().st_size

        return {
            "raw_rows_read": raw_rows,
            "rows_rejected_mind_derived": self._rows_rejected_mind_derived,
            "rows_rejected_compacted": self._rows_rejected_compacted,
            "rows_ingested": raw_rows,
            "nodes_before": nodes_before,
            "canonical_nodes": canonical_nodes,
            "nodes_after": canonical_nodes,
            "canonical_edges": canonical_edges,
            "edges_after": canonical_edges,
            "exact_folds": self._exact_folds,
            "structural_folds": self._structural_folds,
            "assimilation_folds": self._assimilation_folds,
            "sleep_products_folded": sleep_products_folded,
            "nodes_pruned": self._nodes_pruned,
            "edges_pruned": self._edges_pruned,
            "residuals_kept": len(residual_rows),
            "residuals_evicted": self._assimilator.residuals.total_evicted,
            "active_mind_bytes": active_mind_bytes,
            "compression_ratio": round(compression_ratio, 4),
            "assimilation_stats": assimilation_stats,
        }

    def write_report(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = list(self._nodes.values())
        canonical_nodes = len(nodes)
        canonical_edges = len(self._edges)
        raw_rows = self._raw_row_count
        compression_ratio = raw_rows / max(1, canonical_nodes) if canonical_nodes > 0 else 1.0

        top_sources = sorted(self._source_row_counts.items(), key=lambda x: -x[1])[:10]
        top_by_evidence = sorted(nodes, key=lambda n: -n.evidence_count)[:10]
        behavior_nodes = [n for n in nodes if n.behavior_effect_count > 0]
        sleep_nodes = [n for n in nodes if n.sleep_product_count > 0]
        total_sleep_folded = sum(n.sleep_product_count for n in sleep_nodes)
        authority_nonzero = [n for n in nodes if n.authority_effect_count != 0]

        lines = [
            "Nethra Mind Compaction Report",
            "=" * 50,
            "",
            "A. Raw rows read:              {:>10}".format(raw_rows),
            "B. Canonical nodes:            {:>10}".format(canonical_nodes),
            "C. Canonical edges:            {:>10}".format(canonical_edges),
            "D. Compression ratio:          {:>10.2f}x".format(compression_ratio),
            "",
            "E. Top repeated source rows folded:",
        ]
        for src, count in top_sources:
            lines.append("   {:<30} {:>8}".format(src, count))

        lines.extend(["", "F. Top nodes by evidence_count:"])
        for n in top_by_evidence:
            lines.append("   {:<40} evidence={}".format(n.nethra_id, n.evidence_count))

        lines.extend([
            "",
            "G. Nodes with behavior_effect_count > 0: {:>6}".format(len(behavior_nodes)),
        ])
        for n in behavior_nodes[:10]:
            lines.append(
                "   {:<40} behavior_effects={}".format(n.nethra_id, n.behavior_effect_count)
            )

        lines.extend([
            "",
            "H. Sleep products folded: {:>6} nodes, {:>6} total products".format(
                len(sleep_nodes), total_sleep_folded
            ),
        ])

        assimilation_stats = self._assimilator.stats()
        lines.extend([
            "",
            "I. Folding summary:",
            "   {} rows folded into existing nodes".format(max(0, raw_rows - canonical_nodes)),
            "   exact_folds={}  structural_folds={}  assimilation_folds={}".format(
                self._exact_folds, self._structural_folds, self._assimilation_folds,
            ),
            "   assimilation disposition breakdown:",
            *[
                "     {:20} {:>6}".format(k + ":", v)
                for k, v in sorted(assimilation_stats.items())
            ],
            "   residuals_kept={}  residuals_evicted={}".format(
                len(self._assimilator.residuals), self._assimilator.residuals.total_evicted
            ),
        ])

        if authority_nonzero:
            lines.extend([
                "",
                "J. WARNING: {} nodes have non-zero authority_effect_count".format(
                    len(authority_nonzero)
                ),
                "   (authority_effect_count must be zero)",
            ])
        else:
            lines.extend(["", "J. authority_effect_count: all zero (correct)"])

        lines.extend([
            "",
            "K. WARNING: compacted mind is not authority.",
            "   Assist mode may use mind nodes as ranking/probe hints only.",
            "   authority_effects must remain zero in all runtime metrics.",
            "",
            "L. Ingestion policy rejections:",
            "   rows_rejected_mind_derived={}".format(self._rows_rejected_mind_derived),
            "   rows_rejected_compacted={}".format(self._rows_rejected_compacted),
            "",
            "M. Cap enforcement:",
            "   nodes_pruned={}  edges_pruned={}".format(self._nodes_pruned, self._edges_pruned),
            "   max_nodes={}  max_edges={}".format(_MAX_NODES, _MAX_EDGES),
            "",
        ])

        report = "\n".join(lines)
        with open(path, "w") as fh:
            fh.write(report)
        return report


def effective_use_right(use_rights_seen: list[str]) -> str:
    """Return the most permissive safe use_right from an observed set.

    hard_filter and block are never propagated from a mind node.
    """
    best = "record_only"
    for right in use_rights_seen:
        if _USE_RIGHT_RANK.get(right, -1) > _USE_RIGHT_RANK.get(best, -1):
            best = right
    return best


def _structural_id(
    kind: str,
    touched_atoms: list[str],
    touched_structure_refs: list[str],
    member_nethras: list[str],
    context_scope: str,
    use_right: str,
) -> str:
    parts = [
        "k:" + kind,
        "a:" + ",".join(sorted(set(touched_atoms))),
        "r:" + ",".join(sorted(set(touched_structure_refs))),
        "m:" + ",".join(sorted(set(member_nethras))),
        "c:" + context_scope,
        "u:" + use_right,
    ]
    key = "|".join(parts)
    return "struct:" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _has_hidden_truth(row: dict[str, Any]) -> bool:
    stack: list[Any] = [row]
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
