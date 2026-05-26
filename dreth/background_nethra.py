from __future__ import annotations

"""Passive background-familiarity index over recurring trass/unresolved/quarantined structure.

A nethra may be tareth in one context, trass in another, unresolved in another,
or background-familiar without being operationally active.

Background familiarity is structure recognition, not authority. This module
records passive familiarity without granting runtime authority.

Invariants enforced by this module:
  - Does not issue authority records or certificates
  - Does not revoke certificates
  - Does not suppress skips
  - Does not force probes
  - Does not increase monitoring or repair priority
  - Does not allow derivation support from background nethras
  - Does not read the debug manifest or hidden truth
  - record mode behavior equals off on all operational metrics
  - operational_authority_count is always 0

background_confidence means "familiar enough to recognize," not "safe enough to act."
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BackgroundKind = Literal[
    "trass_pattern",
    "unresolved_pattern",
    "quarantined_pattern",
    "recurring_low_salience_pattern",
    "tied_frontier_pattern",
    "dormant_alternative_pattern",
    "context_role_pattern",
    "temporal_cohort_pattern",
    "unknown",
]

BackgroundEdgeKind = Literal[
    "shares_var_with",
    "shares_parent_with",
    "shares_context_with",
    "role_variant_of",
    "trass_variant_of",
    "tareth_variant_of",
    "unresolved_variant_of",
    "temporally_near",
    "frequently_cooccurs_with",
]


@dataclass
class BackgroundNethra:
    nethra_id: str
    kind: BackgroundKind = "unknown"
    vars: tuple[int, ...] = ()
    context_keys: tuple[str, ...] = ()
    source_roles: tuple[str, ...] = ()
    fit_signatures: tuple[str, ...] = ()
    parent_sets: tuple[tuple[int, ...], ...] = ()
    operation_roles: tuple[str, ...] = ()
    recurring_signals: tuple[str, ...] = ()
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
    seen_count: int = 1
    stability_count: int = 0
    contexts_seen_count: int = 0
    cheap_recognition_score: float = 0.0
    salience_score: float = 0.0
    action_relevance_score: float = 0.0
    background_confidence: float = 0.0
    relation_edges: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackgroundNethraEdge:
    source_id: str
    target_id: str
    kind: BackgroundEdgeKind
    cycle: int = 0
    evidence_summary: str = ""


class BackgroundNethraIndex:
    """Passive familiarity index over recurring nethra-like structure.

    Records trass/unresolved/quarantined/dormant/tied-frontier patterns as
    background-familiar structure. Does not issue authority, revoke, suppress
    skips, force probes, increase monitoring, or allow derivation.
    """

    def __init__(self, *, mode: str = "off") -> None:
        if mode not in {"off", "record", "assist_feature"}:
            raise ValueError(
                "BackgroundNethraIndex mode must be off, record, or assist_feature"
            )
        self.mode = mode
        self.records: dict[str, BackgroundNethra] = {}
        self.edges: list[BackgroundNethraEdge] = []
        self._by_var: dict[int, set[str]] = defaultdict(set)
        self._by_context: dict[str, set[str]] = defaultdict(set)
        self._by_signature: dict[str, set[str]] = defaultdict(set)

        # Counters for new records by kind (incremented once per new record)
        self._kind_new_counts: Counter[str] = Counter()

        # Metrics — all zero; operational_authority_count must remain 0
        self.background_nethra_records: int = 0
        self.background_nethra_edges: int = 0
        self.background_contexts_seen: set[str] = set()
        self.background_role_shift_examples: list[dict[str, Any]] = []
        self.background_trass_patterns: int = 0
        self.background_unresolved_patterns: int = 0
        self.background_quarantined_patterns: int = 0
        self.background_giant_cluster_patterns: int = 0
        self.background_dormant_patterns: int = 0
        self.background_tied_frontier_patterns: int = 0
        self.background_records_used_as_features: int = 0
        self.background_feature_hits: int = 0
        self.background_feature_noops: int = 0
        # Must remain 0 always — records carry no authority
        self.operational_authority_count: int = 0

    # ── public guard ──────────────────────────────────────────────────────────

    def _enabled(self) -> bool:
        return self.mode != "off"

    # ── internal upsert ───────────────────────────────────────────────────────

    def _upsert(
        self,
        nethra_id: str,
        *,
        kind: BackgroundKind,
        vars: tuple[int, ...] = (),
        context_key: str = "",
        source_role: str = "",
        fit_signature: str = "",
        parents: tuple[int, ...] = (),
        operation_role: str = "",
        signals: tuple[str, ...] = (),
        cycle: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra:
        existing = self.records.get(nethra_id)
        if existing is None:
            rec = BackgroundNethra(
                nethra_id=nethra_id,
                kind=kind,
                vars=vars,
                context_keys=(context_key,) if context_key else (),
                source_roles=(source_role,) if source_role else (),
                fit_signatures=(fit_signature,) if fit_signature else (),
                parent_sets=(parents,) if parents else (),
                operation_roles=(operation_role,) if operation_role else (),
                recurring_signals=signals,
                first_seen_cycle=cycle,
                last_seen_cycle=cycle,
                seen_count=1,
                stability_count=0,
                contexts_seen_count=1 if context_key else 0,
                cheap_recognition_score=0.1,
                salience_score=0.0,
                action_relevance_score=0.0,
                background_confidence=0.1,
                payload=payload or {},
            )
            self.background_nethra_records += 1
            self._bump_kind_counter(kind)
        else:
            new_contexts = tuple(dict.fromkeys(
                list(existing.context_keys) + ([context_key] if context_key else [])
            ))
            new_roles = tuple(dict.fromkeys(
                list(existing.source_roles) + ([source_role] if source_role else [])
            ))
            new_sigs = tuple(dict.fromkeys(
                list(existing.fit_signatures) + ([fit_signature] if fit_signature else [])
            ))
            new_parent_sets: tuple[tuple[int, ...], ...] = existing.parent_sets
            if parents and parents not in set(existing.parent_sets):
                new_parent_sets = existing.parent_sets + (parents,)
            new_op_roles = tuple(dict.fromkeys(
                list(existing.operation_roles) + ([operation_role] if operation_role else [])
            ))
            new_signals = tuple(dict.fromkeys(
                list(existing.recurring_signals) + list(signals)
            ))
            seen_count = existing.seen_count + 1
            contexts_seen = len(new_contexts)
            stability = existing.stability_count + (1 if seen_count > 2 else 0)
            cheap_score = min(1.0, 0.1 * seen_count + 0.05 * contexts_seen)
            confidence = min(
                1.0,
                0.1 * seen_count + 0.1 * stability / max(1, seen_count),
            )
            merged_payload = dict(existing.payload)
            if payload:
                merged_payload.update(payload)
            rec = BackgroundNethra(
                nethra_id=nethra_id,
                kind=kind if kind != "unknown" else existing.kind,
                vars=tuple(sorted(set(existing.vars) | set(vars))),
                context_keys=new_contexts,
                source_roles=new_roles,
                fit_signatures=new_sigs,
                parent_sets=new_parent_sets,
                operation_roles=new_op_roles,
                recurring_signals=new_signals,
                first_seen_cycle=min(existing.first_seen_cycle, cycle),
                last_seen_cycle=max(existing.last_seen_cycle, cycle),
                seen_count=seen_count,
                stability_count=stability,
                contexts_seen_count=contexts_seen,
                cheap_recognition_score=cheap_score,
                salience_score=existing.salience_score,
                action_relevance_score=existing.action_relevance_score,
                background_confidence=confidence,
                relation_edges=existing.relation_edges,
                payload=merged_payload,
            )
        self.records[nethra_id] = rec
        self._index_record(rec, context_key=context_key, signature=fit_signature)
        if context_key:
            self.background_contexts_seen.add(context_key)
        return rec

    def _bump_kind_counter(self, kind: BackgroundKind) -> None:
        if kind == "trass_pattern":
            self.background_trass_patterns += 1
        elif kind == "unresolved_pattern":
            self.background_unresolved_patterns += 1
        elif kind == "quarantined_pattern":
            self.background_quarantined_patterns += 1
        elif kind == "dormant_alternative_pattern":
            self.background_dormant_patterns += 1
        elif kind == "tied_frontier_pattern":
            self.background_tied_frontier_patterns += 1

    def _index_record(
        self,
        rec: BackgroundNethra,
        *,
        context_key: str = "",
        signature: str = "",
    ) -> None:
        for var in rec.vars:
            self._by_var[int(var)].add(rec.nethra_id)
        if context_key:
            self._by_context[context_key].add(rec.nethra_id)
        if signature:
            self._by_signature[signature].add(rec.nethra_id)

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        kind: BackgroundEdgeKind,
        *,
        cycle: int = 0,
        evidence_summary: str = "",
    ) -> None:
        edge = BackgroundNethraEdge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            cycle=cycle,
            evidence_summary=evidence_summary,
        )
        self.edges.append(edge)
        self.background_nethra_edges += 1

    # ── public update methods ─────────────────────────────────────────────────

    def add_or_update_from_context_role(
        self,
        *,
        nethra_id: str,
        role: str,
        var: int,
        context_key: str = "",
        cycle: int = 0,
        operation_role: str = "",
        fit_signature: str = "",
        parents: tuple[int, ...] = (),
        signals: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Observe a context-role assignment from ContextRoleIndex.

        Only trass, unresolved, and best_available roles create background
        records. tareth is an active authority role and is not recorded here.
        """
        if not self._enabled():
            return None
        if role not in {"trass", "unresolved", "best_available"}:
            return None
        kind: BackgroundKind = (
            "trass_pattern" if role == "trass"
            else "unresolved_pattern" if role == "unresolved"
            else "context_role_pattern"
        )
        return self._upsert(
            nethra_id,
            kind=kind,
            vars=(int(var),),
            context_key=context_key,
            source_role=role,
            fit_signature=fit_signature,
            parents=parents,
            operation_role=operation_role,
            signals=signals,
            cycle=cycle,
            payload=payload,
        )

    def add_or_update_from_tied_frontier(
        self,
        *,
        nethra_id: str,
        var: int,
        context_key: str = "",
        cycle: int = 0,
        candidate_count: int = 0,
        stable_count: int = 0,
        parents: tuple[int, ...] = (),
        signals: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Observe a recurring tied-frontier candidate (non-operative, not forced)."""
        if not self._enabled():
            return None
        return self._upsert(
            nethra_id,
            kind="tied_frontier_pattern",
            vars=(int(var),),
            context_key=context_key,
            source_role="unresolved",
            parents=parents,
            signals=signals,
            cycle=cycle,
            payload={
                "candidate_count": candidate_count,
                "stable_count": stable_count,
                **(payload or {}),
            },
        )

    def add_or_update_from_dormant_alternative(
        self,
        *,
        nethra_id: str,
        var: int,
        context_key: str = "",
        cycle: int = 0,
        revival_count: int = 0,
        parents: tuple[int, ...] = (),
        signals: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Observe a dormant alternative (does not make it active)."""
        if not self._enabled():
            return None
        return self._upsert(
            nethra_id,
            kind="dormant_alternative_pattern",
            vars=(int(var),),
            context_key=context_key,
            source_role="dormant",
            parents=parents,
            signals=signals,
            cycle=cycle,
            payload={"revival_count": revival_count, **(payload or {})},
        )

    def add_or_update_from_uncertainty_cluster(
        self,
        *,
        nethra_id: str,
        vars: tuple[int, ...] = (),
        context_key: str = "",
        cycle: int = 0,
        is_giant: bool = False,
        signals: tuple[str, ...] = (),
        parents: tuple[int, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Record an uncertainty cluster as background structure.

        Giant clusters are recorded separately as low-specificity global
        background. They must not become action pressure.
        """
        if not self._enabled():
            return None
        kind: BackgroundKind = (
            "recurring_low_salience_pattern" if is_giant else "unresolved_pattern"
        )
        is_new = nethra_id not in self.records
        if is_giant and is_new:
            self.background_giant_cluster_patterns += 1
        return self._upsert(
            nethra_id,
            kind=kind,
            vars=tuple(int(v) for v in vars),
            context_key=context_key,
            source_role="uncertainty_cluster",
            parents=parents,
            signals=signals,
            cycle=cycle,
            payload={"is_giant": is_giant, **(payload or {})},
        )

    def add_or_update_from_authority_debt(
        self,
        *,
        nethra_id: str,
        var: int,
        context_key: str = "",
        cycle: int = 0,
        authority_state: str = "",
        parents: tuple[int, ...] = (),
        signals: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Observe a visible authority-state pattern (contested/quarantined/repair).

        Does not cause monitoring or repair action.
        """
        if not self._enabled():
            return None
        if authority_state not in {
            "contested_best_available",
            "quarantined_for_derivation",
            "repair_candidate",
        }:
            return None
        return self._upsert(
            nethra_id,
            kind="quarantined_pattern",
            vars=(int(var),),
            context_key=context_key,
            source_role=f"authority_state:{authority_state}",
            parents=parents,
            signals=signals,
            cycle=cycle,
            payload={"authority_state": authority_state, **(payload or {})},
        )

    def add_or_update_from_temporal_event_if_available(
        self,
        *,
        nethra_id: str,
        vars: tuple[int, ...] = (),
        context_key: str = "",
        cycle: int = 0,
        temporal_event: Any = None,
        payload: dict[str, Any] | None = None,
    ) -> BackgroundNethra | None:
        """Adapter hook for TemporalEventLedger cohorts.

        If temporal_event_ledger.py does not exist, this is a no-op. The hook
        is intentionally minimal so temporal evidence can be wired later without
        changing any invariants here.
        """
        if not self._enabled():
            return None
        if temporal_event is None:
            return None
        return self._upsert(
            nethra_id,
            kind="temporal_cohort_pattern",
            vars=tuple(int(v) for v in vars),
            context_key=context_key,
            source_role="temporal_event",
            cycle=cycle,
            payload={"temporal_event": str(temporal_event), **(payload or {})},
        )

    # ── query methods ─────────────────────────────────────────────────────────

    def query_by_var(self, var: int) -> tuple[BackgroundNethra, ...]:
        ids = sorted(self._by_var.get(int(var), ()))
        return tuple(self.records[nid] for nid in ids if nid in self.records)

    def query_by_context(self, context_key: str) -> tuple[BackgroundNethra, ...]:
        ids = sorted(self._by_context.get(str(context_key), ()))
        return tuple(self.records[nid] for nid in ids if nid in self.records)

    def query_by_signature(self, signature: str) -> tuple[BackgroundNethra, ...]:
        ids = sorted(self._by_signature.get(str(signature), ()))
        return tuple(self.records[nid] for nid in ids if nid in self.records)

    def query_neighbors(self, nethra_id: str) -> tuple[BackgroundNethra, ...]:
        neighbors: list[BackgroundNethra] = []
        seen: set[str] = set()
        for edge in self.edges:
            target_id = None
            if edge.source_id == nethra_id:
                target_id = edge.target_id
            elif edge.target_id == nethra_id:
                target_id = edge.source_id
            if target_id and target_id in self.records and target_id not in seen:
                seen.add(target_id)
                neighbors.append(self.records[target_id])
        return tuple(neighbors)

    # ── role shift tracking ───────────────────────────────────────────────────

    def record_role_shift(
        self,
        nethra_id: str,
        *,
        from_role: str,
        to_role: str,
        var: int,
        cycle: int,
    ) -> None:
        """Record a tareth/trass role shift for the same nethra across contexts.

        This is passive observation only — no authority effect.
        """
        if not self._enabled():
            return
        if len(self.background_role_shift_examples) < 200:
            self.background_role_shift_examples.append({
                "nethra_id": nethra_id,
                "from_role": from_role,
                "to_role": to_role,
                "var": int(var),
                "cycle": int(cycle),
            })

    # ── summary / export ──────────────────────────────────────────────────────

    def summarize(self) -> dict[str, Any]:
        vals = list(self.records.values())
        if vals:
            rec_mean_recog = sum(r.cheap_recognition_score for r in vals) / len(vals)
            rec_mean_action = sum(r.action_relevance_score for r in vals) / len(vals)
        else:
            rec_mean_recog = 0.0
            rec_mean_action = 0.0

        kinds = Counter(r.kind for r in vals)
        return {
            "background_nethra_mode": self.mode,
            "background_nethra_records": self.background_nethra_records,
            "background_nethra_by_kind": dict(kinds),
            "background_nethra_edges": self.background_nethra_edges,
            "background_contexts_seen": len(self.background_contexts_seen),
            "background_role_shift_examples": len(self.background_role_shift_examples),
            "background_trass_patterns": self.background_trass_patterns,
            "background_unresolved_patterns": self.background_unresolved_patterns,
            "background_quarantined_patterns": self.background_quarantined_patterns,
            "background_giant_cluster_patterns": self.background_giant_cluster_patterns,
            "background_dormant_patterns": self.background_dormant_patterns,
            "background_tied_frontier_patterns": self.background_tied_frontier_patterns,
            "background_recognition_score_mean": rec_mean_recog,
            "background_action_relevance_score_mean": rec_mean_action,
            "background_records_used_as_features": self.background_records_used_as_features,
            "background_feature_hits": self.background_feature_hits,
            "background_feature_noops": self.background_feature_noops,
            # Critical distinction: familiar background ≠ operational authority
            "familiar_background_count": self.background_nethra_records,
            "operational_authority_count": self.operational_authority_count,
        }

    def export_records(self, limit: int = 200) -> dict[str, Any]:
        limit = max(0, int(limit))
        recs = list(self.records.values())[:limit]
        return {
            "records": [asdict(r) for r in recs],
            "edges": [asdict(e) for e in self.edges[:limit]],
            "role_shift_examples": list(self.background_role_shift_examples[:50]),
        }
