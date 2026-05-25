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
from typing import Dict, Iterable, Optional, Set, Tuple


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
