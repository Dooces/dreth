from __future__ import annotations

"""Shadow-only frontier evaluator for diagnostic NethraGraph snapshots.

This module proposes bounded local candidate sets from graph neighbors and
scores those sets against post-run observable ledger artifacts. It is
diagnostic only: it must not mutate agent or ledger state, and it must not
affect skips, fit, sentinels, cert issuance, revocation, route certs, provider
choice, policy selection, or defaults.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set, Tuple


@dataclass(frozen=True)
class GraphFrontierProposal:
    target_var: int
    context_key: Optional[str]
    source_node_id: str
    candidate_node_ids: Tuple[str, ...]
    relation_types_used: Tuple[str, ...]
    frontier_size: int


@dataclass(frozen=True)
class GraphFrontierEvaluation:
    target_var: int
    frontier_size: int
    chosen_parent_hits: int
    chosen_parent_total: int
    revoked_neighbor_hits: int
    revoked_total: int
    dormant_neighbor_hits: int
    dormant_total: int


@dataclass(frozen=True)
class TemporalFrontierProposalRecord:
    cycle: int
    target_var: int
    visible_count: int
    proposal: GraphFrontierProposal


@dataclass(frozen=True)
class TemporalFrontierEvaluationRecord:
    cycle: int
    target_var: int
    frontier_size: int
    visible_count: int
    chosen_parent_hits: int
    chosen_parent_total: int
    revoked_hits: int
    revoked_total: int


def _parse_var_from_node_id(node_id: str) -> int:
    if node_id.startswith("var:"):
        try:
            return int(node_id.split(":", 2)[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _relation_allowed(relation_type: str, allowed: Optional[Set[str]]) -> bool:
    return allowed is None or relation_type in allowed


def propose_frontier(
    snapshot,
    source_node_id: str,
    relation_types: Optional[Iterable[str]] = None,
    max_depth: int = 2,
    max_candidates: int = 20,
) -> GraphFrontierProposal:
    """Return a bounded graph-neighbor frontier for one source node.

    The frontier is an undirected BFS over existing snapshot relations only.
    The source node itself is never emitted as a candidate.
    """
    nodes_by_id = {node.node_id: node for node in getattr(snapshot, "nodes", ())}
    source_node = nodes_by_id.get(source_node_id)
    target_var = getattr(source_node, "var", None)
    if target_var is None:
        target_var = _parse_var_from_node_id(source_node_id)

    if source_node is None or max_depth <= 0 or max_candidates <= 0:
        return GraphFrontierProposal(
            target_var=int(target_var if target_var is not None else -1),
            context_key=getattr(source_node, "context_key", None),
            source_node_id=source_node_id,
            candidate_node_ids=(),
            relation_types_used=(),
            frontier_size=0,
        )

    allowed_types = set(relation_types) if relation_types is not None else None
    adjacency: Dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relation in getattr(snapshot, "relations", ()):
        relation_type = relation.relation_type
        if not _relation_allowed(relation_type, allowed_types):
            continue
        source_id = relation.source.node_id
        target_id = relation.target.node_id
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        adjacency[source_id].append((target_id, relation_type))
        adjacency[target_id].append((source_id, relation_type))

    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda item: (item[0], item[1]))

    queue = deque([(source_node_id, 0)])
    visited = {source_node_id}
    candidates = []
    used_relation_types = set()

    while queue and len(candidates) < max_candidates:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor_id, relation_type in adjacency.get(node_id, ()):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            used_relation_types.add(relation_type)
            candidates.append(neighbor_id)
            if len(candidates) >= max_candidates:
                break
            queue.append((neighbor_id, depth + 1))

    return GraphFrontierProposal(
        target_var=int(target_var if target_var is not None else -1),
        context_key=getattr(source_node, "context_key", None),
        source_node_id=source_node_id,
        candidate_node_ids=tuple(candidates),
        relation_types_used=tuple(sorted(used_relation_types)),
        frontier_size=len(candidates),
    )


def _revoked_cert_node_ids(snapshot, target_var: int, nethra) -> Set[str]:
    revoked_nodes = set()
    for operation, cert in (getattr(nethra, "certificates", {}) or {}).items():
        if getattr(cert, "revoked_by", None):
            revoked_nodes.add(f"cert:{target_var}:{operation}")
    for candidate_var, cert in (getattr(nethra, "route_certs", {}) or {}).items():
        if getattr(cert, "revoked_by", None):
            revoked_nodes.add(f"route_cert:{target_var}:{candidate_var}")

    snapshot_node_ids = {node.node_id for node in getattr(snapshot, "nodes", ())}
    return revoked_nodes & snapshot_node_ids


def _revoked_neighbor_labels(snapshot, target_var: int, nethra) -> Set[str]:
    """Observable labels around revoked certs for a target var.

    Include the revoked cert node itself and any graph neighbors connected to
    that cert, excluding the source var node. This keeps the label sparse while
    allowing route-cert candidates to count when the graph exposes them.
    """
    source_node_id = f"var:{target_var}"
    labels = set()
    revoked_nodes = _revoked_cert_node_ids(snapshot, target_var, nethra)
    labels.update(revoked_nodes)
    for relation in getattr(snapshot, "relations", ()):
        source_id = relation.source.node_id
        target_id = relation.target.node_id
        if source_id in revoked_nodes and target_id != source_node_id:
            labels.add(target_id)
        if target_id in revoked_nodes and source_id != source_node_id:
            labels.add(source_id)
    return labels


def _dormant_node_ids(snapshot, target_var: int) -> Set[str]:
    return {
        node.node_id
        for node in getattr(snapshot, "nodes", ())
        if node.kind == "dormant_alternative" and node.var == target_var
    }


def _relation_connects(
    relation,
    node_a: str,
    node_b: str,
    relation_types: Optional[Set[str]] = None,
) -> bool:
    if relation_types is not None and relation.relation_type not in relation_types:
        return False
    source_id = relation.source.node_id
    target_id = relation.target.node_id
    return (
        (source_id == node_a and target_id == node_b)
        or (source_id == node_b and target_id == node_a)
    )


def _snapshot_without_relations(snapshot, should_remove):
    return type(snapshot)(
        nodes=getattr(snapshot, "nodes", ()),
        relations=tuple(
            relation
            for relation in getattr(snapshot, "relations", ())
            if not should_remove(relation)
        ),
        authority_records=getattr(snapshot, "authority_records", ()),
    )


def _revoked_labels_for_cert(
    snapshot,
    source_node_id: str,
    revoked_cert_node_id: str,
) -> Set[str]:
    labels = {revoked_cert_node_id}
    for relation in getattr(snapshot, "relations", ()):
        source_id = relation.source.node_id
        target_id = relation.target.node_id
        if source_id == revoked_cert_node_id and target_id != source_node_id:
            labels.add(target_id)
        if target_id == revoked_cert_node_id and source_id != source_node_id:
            labels.add(source_id)
    return labels


def evaluate_frontier_against_agent(snapshot, agent) -> Tuple[GraphFrontierEvaluation, ...]:
    """Evaluate graph frontiers against final observable agent ledger state.

    Labels are not hidden world truth. They are post-run utility checks: current
    parents, revoked cert neighborhoods, and dormant alternative nodes.
    """
    ledger = getattr(agent, "ledger", None)
    vars_by_id = getattr(ledger, "vars", {}) or {}
    evaluations = []

    for target_var in sorted(vars_by_id):
        source_node_id = f"var:{target_var}"
        if not any(node.node_id == source_node_id for node in getattr(snapshot, "nodes", ())):
            continue
        nethra = vars_by_id[target_var]
        proposal = propose_frontier(snapshot, source_node_id)
        candidates = set(proposal.candidate_node_ids)

        chosen_parent_labels = {
            f"var:{parent}"
            for parent in (getattr(nethra, "parents", ()) or ())
        }
        revoked_labels = _revoked_neighbor_labels(snapshot, target_var, nethra)
        dormant_labels = _dormant_node_ids(snapshot, target_var)

        if not chosen_parent_labels and not revoked_labels and not dormant_labels:
            continue

        evaluations.append(
            GraphFrontierEvaluation(
                target_var=target_var,
                frontier_size=proposal.frontier_size,
                chosen_parent_hits=len(chosen_parent_labels & candidates),
                chosen_parent_total=len(chosen_parent_labels),
                revoked_neighbor_hits=len(revoked_labels & candidates),
                revoked_total=len(revoked_labels),
                dormant_neighbor_hits=len(dormant_labels & candidates),
                dormant_total=len(dormant_labels),
            )
        )

    return tuple(evaluations)


def evaluate_frontier_leave_one_out(snapshot, agent) -> Tuple[GraphFrontierEvaluation, ...]:
    """Evaluate frontier recall after removing direct target-label support.

    For each label, the relation that directly defines target-to-label support
    is removed from a temporary snapshot before proposing the frontier. This
    tests whether the graph still exposes the label through other local paths.
    """
    ledger = getattr(agent, "ledger", None)
    vars_by_id = getattr(ledger, "vars", {}) or {}
    snapshot_node_ids = {node.node_id for node in getattr(snapshot, "nodes", ())}
    evaluations = []

    for target_var in sorted(vars_by_id):
        source_node_id = f"var:{target_var}"
        if source_node_id not in snapshot_node_ids:
            continue
        nethra = vars_by_id[target_var]

        chosen_parent_labels = {
            f"var:{parent}"
            for parent in (getattr(nethra, "parents", ()) or ())
        }
        revoked_cert_nodes = _revoked_cert_node_ids(snapshot, target_var, nethra)
        dormant_labels = _dormant_node_ids(snapshot, target_var)

        if not chosen_parent_labels and not revoked_cert_nodes and not dormant_labels:
            continue

        frontier_sizes = []
        chosen_parent_hits = 0
        for label in sorted(chosen_parent_labels):
            loo_snapshot = _snapshot_without_relations(
                snapshot,
                lambda relation, label=label: _relation_connects(
                    relation,
                    source_node_id,
                    label,
                ),
            )
            proposal = propose_frontier(loo_snapshot, source_node_id)
            frontier_sizes.append(proposal.frontier_size)
            if label in proposal.candidate_node_ids:
                chosen_parent_hits += 1

        revoked_hits = 0
        revoked_total = 0
        seen_revoked_labels = set()
        for revoked_cert_node in sorted(revoked_cert_nodes):
            labels = _revoked_labels_for_cert(
                snapshot,
                source_node_id,
                revoked_cert_node,
            )
            loo_snapshot = _snapshot_without_relations(
                snapshot,
                lambda relation, revoked_cert_node=revoked_cert_node: _relation_connects(
                    relation,
                    source_node_id,
                    revoked_cert_node,
                    {"coactive_with", "exception_to"},
                ),
            )
            proposal = propose_frontier(loo_snapshot, source_node_id)
            frontier_sizes.append(proposal.frontier_size)
            candidates = set(proposal.candidate_node_ids)
            for label in sorted(labels):
                if label in seen_revoked_labels:
                    continue
                seen_revoked_labels.add(label)
                revoked_total += 1
                if label in candidates:
                    revoked_hits += 1

        dormant_hits = 0
        for label in sorted(dormant_labels):
            loo_snapshot = _snapshot_without_relations(
                snapshot,
                lambda relation, label=label: _relation_connects(
                    relation,
                    source_node_id,
                    label,
                    {"substitutes_for"},
                ),
            )
            proposal = propose_frontier(loo_snapshot, source_node_id)
            frontier_sizes.append(proposal.frontier_size)
            if label in proposal.candidate_node_ids:
                dormant_hits += 1

        evaluations.append(
            GraphFrontierEvaluation(
                target_var=target_var,
                frontier_size=(
                    int(sum(frontier_sizes) / len(frontier_sizes))
                    if frontier_sizes
                    else 0
                ),
                chosen_parent_hits=chosen_parent_hits,
                chosen_parent_total=len(chosen_parent_labels),
                revoked_neighbor_hits=revoked_hits,
                revoked_total=revoked_total,
                dormant_neighbor_hits=dormant_hits,
                dormant_total=len(dormant_labels),
            )
        )

    return tuple(evaluations)


class TemporalGraphFrontierEvaluator:
    """Shadow-only warmup temporal frontier evaluator.

    The evaluator is proposal-prior only. It records a graph frontier before an
    audit and scores that stored proposal after the normal audit/fit result has
    already been installed. It never returns candidate choices to the agent.
    """

    def __init__(
        self,
        warmup_cycles: int,
        max_depth: int = 2,
        max_candidates: int = 20,
        snapshot_builder: Optional[Callable[[object], object]] = None,
    ) -> None:
        self.warmup_cycles = max(0, int(warmup_cycles))
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.proposals: list[TemporalFrontierProposalRecord] = []
        self.evaluations: list[TemporalFrontierEvaluationRecord] = []
        self._snapshot_builder = snapshot_builder

    def _build_snapshot(self, agent):
        if self._snapshot_builder is not None:
            return self._snapshot_builder(agent)
        from dreth.relative_authority_observer import build_snapshot_from_agent

        return build_snapshot_from_agent(agent)

    def before_audit(
        self,
        agent,
        target_var: int,
        cycle: int,
    ) -> Optional[TemporalFrontierProposalRecord]:
        if cycle < self.warmup_cycles:
            return None
        snapshot = self._build_snapshot(agent)
        source_node_id = f"var:{target_var}"
        proposal = propose_frontier(
            snapshot,
            source_node_id,
            max_depth=self.max_depth,
            max_candidates=self.max_candidates,
        )
        visible_count = _visible_count(agent)
        record = TemporalFrontierProposalRecord(
            cycle=cycle,
            target_var=target_var,
            visible_count=visible_count,
            proposal=proposal,
        )
        self.proposals.append(record)
        return record

    def after_audit(
        self,
        agent,
        token: Optional[TemporalFrontierProposalRecord],
        target_var: int,
        cycle: int,
        parents: Tuple[int, ...],
        func: str,
        sig_changed: bool,
    ) -> None:
        del func, sig_changed
        if token is None:
            return
        candidates = set(token.proposal.candidate_node_ids)
        chosen_parent_labels = {f"var:{parent}" for parent in parents}

        snapshot = self._build_snapshot(agent)
        nethra = getattr(getattr(agent, "ledger", None), "vars", {}).get(target_var)
        revoked_labels = (
            _revoked_neighbor_labels(snapshot, target_var, nethra)
            if nethra is not None
            else set()
        )

        self.evaluations.append(
            TemporalFrontierEvaluationRecord(
                cycle=cycle,
                target_var=target_var,
                frontier_size=token.proposal.frontier_size,
                visible_count=token.visible_count,
                chosen_parent_hits=len(chosen_parent_labels & candidates),
                chosen_parent_total=len(chosen_parent_labels),
                revoked_hits=len(revoked_labels & candidates),
                revoked_total=len(revoked_labels),
            )
        )

    def summary(self) -> Dict[str, float | int]:
        evals = len(self.evaluations)
        frontier_size_total = sum(ev.frontier_size for ev in self.evaluations)
        chosen_hits = sum(ev.chosen_parent_hits for ev in self.evaluations)
        chosen_total = sum(ev.chosen_parent_total for ev in self.evaluations)
        revoked_hits = sum(ev.revoked_hits for ev in self.evaluations)
        revoked_total = sum(ev.revoked_total for ev in self.evaluations)
        reduction_total = sum(
            1.0 - (ev.frontier_size / max(1, ev.visible_count))
            for ev in self.evaluations
        )
        misses = (chosen_total - chosen_hits) + (revoked_total - revoked_hits)
        return {
            "temporal_frontier_evals": evals,
            "temporal_frontier_avg_size": (
                frontier_size_total / evals if evals else 0.0
            ),
            "temporal_frontier_chosen_parent_hits": chosen_hits,
            "temporal_frontier_chosen_parent_total": chosen_total,
            "temporal_frontier_chosen_parent_recall": (
                chosen_hits / chosen_total if chosen_total else 0.0
            ),
            "temporal_frontier_revoked_hits": revoked_hits,
            "temporal_frontier_revoked_total": revoked_total,
            "temporal_frontier_revoked_recall": (
                revoked_hits / revoked_total if revoked_total else 0.0
            ),
            "temporal_frontier_candidate_reduction_vs_visible": (
                reduction_total / evals if evals else 0.0
            ),
            "temporal_frontier_misses": misses,
        }


def _visible_count(agent) -> int:
    visible = getattr(getattr(agent, "world", None), "visible_count", None)
    if visible is not None:
        try:
            return int(visible)
        except (TypeError, ValueError):
            pass
    vars_by_id = getattr(getattr(agent, "ledger", None), "vars", {}) or {}
    return len(vars_by_id)
