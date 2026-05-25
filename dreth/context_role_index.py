from __future__ import annotations

"""Context-indexed provenance over learned nethra graph structure.

The index is a view over learned structures, not a separate storage class for
trass objects. Nethras remain graph nodes; tareth/trass/unresolved/best_available
are context-role annotations on those nodes.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


NethraKind = Literal[
    "var_fit",
    "tied_frontier_candidate",
    "dormant_alternative",
    "composite",
    "route_handle",
    "trass_equivalence",
    "regime_handle",
    "unknown",
]

NethraSource = Literal[
    "audit",
    "tied_frontier",
    "dormant_alternative",
    "route_cert",
    "operation_role",
    "composite",
    "regime",
    "uncertainty_cluster",
]

EdgeKind = Literal[
    "component-of",
    "composed-with",
    "predicts",
    "substitutes-for",
    "conflicts-with",
    "coactive-with",
    "prior-trass-with",
    "prior-tareth-with",
    "shares-parent",
    "shares-context",
]

ContextRole = Literal["tareth", "trass", "unresolved", "best_available"]


@dataclass(frozen=True)
class NethraNode:
    nethra_id: str
    kind: NethraKind = "unknown"
    target_var: int | None = None
    components: tuple[int, ...] = ()
    learned_parents: tuple[int, ...] = ()
    learned_func: str = ""
    signature: str = ""
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
    observations: int = 1
    passive_evidence_count: int = 0
    active_probe_count: int = 0
    composition_links: tuple[str, ...] = ()
    source: NethraSource = "audit"


@dataclass(frozen=True)
class NethraEdge:
    source_id: str
    target_id: str
    kind: EdgeKind
    context_key: str = ""
    cycle: int = 0
    evidence_summary: str = ""


@dataclass(frozen=True)
class ContextRoleRecord:
    nethra_id: str
    context_key: str
    operation: str
    role: ContextRole
    cycle: int
    evidence_summary: str = ""
    witness_probes: tuple[Any, ...] = ()
    sentinel_passes: int = 0
    sentinel_failures: int = 0
    fit_margin: int | None = None
    tie_count: int = 0
    near_tie_count: int = 0
    strong_observations: int = 0
    revocations: int = 0
    skip_role: str = ""
    route_role: str = ""
    uncertainty_signals: tuple[str, ...] = ()
    validity_scope: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContextRoleMatch:
    nethra_id: str
    match_score: float
    match_reason: str
    shared_components: tuple[int, ...] = ()
    shared_context: tuple[str, ...] = ()
    shared_parents: tuple[int, ...] = ()
    shared_uncertainty_signals: tuple[str, ...] = ()
    prior_roles: tuple[str, ...] = ()
    current_candidate_role: str = "best_available"


class ContextRoleIndex:
    """Cheap retrieval index over nethra graph provenance and role history."""

    def __init__(self) -> None:
        self.nodes: dict[str, NethraNode] = {}
        self.edges: list[NethraEdge] = []
        self.roles: list[ContextRoleRecord] = []
        self._by_component: dict[int, set[str]] = defaultdict(set)
        self._by_var: dict[int, set[str]] = defaultdict(set)
        self._by_parent: dict[int, set[str]] = defaultdict(set)
        self._by_context: dict[str, list[ContextRoleRecord]] = defaultdict(list)
        self._roles_by_nethra: dict[str, list[ContextRoleRecord]] = defaultdict(list)
        self._last_role_key: dict[tuple[str, str, str], ContextRoleRecord] = {}
        self.index_queries = 0
        self.index_matches = 0
        self.matches_used_as_local_anchor = 0
        self.assist_feature_hits = 0

    def add_or_update_node(self, node: NethraNode) -> NethraNode:
        existing = self.nodes.get(node.nethra_id)
        if existing is None:
            merged = node
        else:
            first_seen = min(existing.first_seen_cycle, node.first_seen_cycle)
            merged = NethraNode(
                nethra_id=node.nethra_id,
                kind=node.kind if node.kind != "unknown" else existing.kind,
                target_var=node.target_var if node.target_var is not None else existing.target_var,
                components=tuple(sorted(set(existing.components) | set(node.components))),
                learned_parents=tuple(sorted(set(existing.learned_parents) | set(node.learned_parents))),
                learned_func=node.learned_func or existing.learned_func,
                signature=node.signature or existing.signature,
                first_seen_cycle=int(first_seen),
                last_seen_cycle=max(existing.last_seen_cycle, node.last_seen_cycle),
                observations=existing.observations + max(1, node.observations),
                passive_evidence_count=existing.passive_evidence_count + node.passive_evidence_count,
                active_probe_count=existing.active_probe_count + node.active_probe_count,
                composition_links=tuple(sorted(set(existing.composition_links) | set(node.composition_links))),
                source=node.source,
            )
        self.nodes[node.nethra_id] = merged
        self._index_node(merged)
        return merged

    def add_edge(self, edge: NethraEdge) -> None:
        self.edges.append(edge)

    def assign_context_role(self, role_record: ContextRoleRecord) -> None:
        key = (role_record.nethra_id, role_record.context_key, role_record.operation)
        prev = self._last_role_key.get(key)
        if prev is not None and prev.role == role_record.role and prev.cycle == role_record.cycle:
            return
        self.roles.append(role_record)
        self._roles_by_nethra[role_record.nethra_id].append(role_record)
        self._last_role_key[key] = role_record
        self._by_context[role_record.context_key].append(role_record)

    def query_by_nethra(self, nethra_id: str) -> tuple[NethraNode | None, tuple[ContextRoleRecord, ...]]:
        self.index_queries += 1
        return self.nodes.get(nethra_id), tuple(self._roles_by_nethra.get(nethra_id, ()))

    def query_by_context(self, context_key: str) -> tuple[ContextRoleRecord, ...]:
        self.index_queries += 1
        return tuple(self._by_context.get(context_key, ()))

    def query_by_component(self, component: int) -> tuple[NethraNode, ...]:
        self.index_queries += 1
        return tuple(self.nodes[nid] for nid in sorted(self._by_component.get(int(component), ())))

    def query_by_var(self, var: int) -> tuple[NethraNode, ...]:
        self.index_queries += 1
        return tuple(self.nodes[nid] for nid in sorted(self._by_var.get(int(var), ())))

    def query_by_parent(self, parent: int) -> tuple[NethraNode, ...]:
        self.index_queries += 1
        return tuple(self.nodes[nid] for nid in sorted(self._by_parent.get(int(parent), ())))

    def query_for_uncertainty_cluster(self, cluster: Any) -> tuple[ContextRoleMatch, ...]:
        """Return local graph/index matches for an uncertainty cluster.

        Generic signal overlap is insufficient. A match must have component or
        parent locality, and context mismatch blocks weak single-node matches.
        """
        self.index_queries += 1
        cluster_vars = {int(v) for v in getattr(cluster, "vars", ()) or ()}
        cluster_parents = {int(v) for v in getattr(cluster, "shared_parents", ()) or ()}
        cluster_neighbors = {int(v) for v in getattr(cluster, "shared_graph_neighbors", ()) or ()}
        cluster_components = cluster_vars | cluster_parents | cluster_neighbors
        cluster_signals = {str(s) for s in getattr(cluster, "shared_signals", ()) or ()}
        cluster_context = _cluster_context_tokens(cluster)

        candidate_ids: set[str] = set()
        for component in cluster_components:
            candidate_ids.update(self._by_component.get(int(component), ()))
        for parent in cluster_parents:
            candidate_ids.update(self._by_parent.get(int(parent), ()))

        matches: list[ContextRoleMatch] = []
        for nethra_id in sorted(candidate_ids)[:300]:
            node = self.nodes[nethra_id]
            node_components = set(node.components)
            node_parents = set(node.learned_parents)
            shared_components = tuple(sorted(cluster_components & node_components))
            shared_parents = tuple(sorted(cluster_parents & node_parents))
            if not shared_components and not shared_parents:
                continue
            prior_roles = self._roles_by_nethra.get(node.nethra_id, ())
            role_context = set()
            role_signals: set[str] = set()
            witnessed = False
            for role in prior_roles:
                role_context.update(_context_tokens(role.context_key))
                role_context.update(str(v) for v in role.validity_scope)
                role_signals.update(role.uncertainty_signals)
                if (
                    role.role in {"tareth", "trass"}
                    and (role.witness_probes or role.sentinel_passes > 0 or role.strong_observations > 0)
                ):
                    witnessed = True
            shared_context = tuple(sorted(cluster_context & role_context))
            shared_signals = tuple(sorted(cluster_signals & role_signals))
            context_mismatch = bool(role_context) and not shared_context and not shared_signals
            if context_mismatch and len(shared_components) < 2 and not shared_parents:
                continue
            score = 0.0
            score += min(0.45, 0.18 * len(shared_components))
            score += min(0.25, 0.15 * len(shared_parents))
            score += min(0.20, 0.10 * len(shared_context))
            score += min(0.15, 0.08 * len(shared_signals))
            if witnessed:
                score += 0.15
            if context_mismatch:
                score -= 0.25
            if score < 0.35:
                continue
            reasons = []
            if shared_components:
                reasons.append("shared_components")
            if shared_parents:
                reasons.append("shared_parents")
            if shared_context:
                reasons.append("shared_context")
            if shared_signals:
                reasons.append("shared_uncertainty_signals")
            if witnessed:
                reasons.append("scoped_witnessed_prior_role")
            matches.append(ContextRoleMatch(
                nethra_id=node.nethra_id,
                match_score=round(score, 6),
                match_reason=",".join(reasons) or "local_overlap",
                shared_components=shared_components,
                shared_context=shared_context,
                shared_parents=shared_parents,
                shared_uncertainty_signals=shared_signals,
                prior_roles=tuple(r.role for r in prior_roles[-8:]),
            ))
        matches.sort(key=lambda m: (-m.match_score, m.nethra_id))
        self.index_matches += len(matches)
        return tuple(matches)

    def mark_matches_used_as_local_anchor(self, count: int = 1) -> None:
        self.matches_used_as_local_anchor += max(0, int(count))
        self.assist_feature_hits += max(0, int(count))

    def summarize(self) -> dict[str, Any]:
        role_counts = Counter(r.role for r in self.roles)
        context_counts = Counter(r.context_key for r in self.roles)
        return {
            "context_role_index_nodes": len(self.nodes),
            "context_role_records": len(self.roles),
            "context_role_tareth": role_counts.get("tareth", 0),
            "context_role_trass": role_counts.get("trass", 0),
            "context_role_unresolved": role_counts.get("unresolved", 0),
            "context_role_best_available": role_counts.get("best_available", 0),
            "context_role_index_queries": self.index_queries,
            "context_role_index_matches": self.index_matches,
            "context_role_matches_used_as_local_anchor": self.matches_used_as_local_anchor,
            "context_role_assist_feature_hits": self.assist_feature_hits,
            "context_role_nodes_by_kind": dict(Counter(r.kind for r in self.nodes.values())),
            "context_role_nodes_by_source": dict(Counter(r.source for r in self.nodes.values())),
            "context_roles_by_context": dict(context_counts),
            "context_roles_by_role": dict(role_counts),
            "context_role_edges": len(self.edges),
            "context_role_edges_by_kind": dict(Counter(e.kind for e in self.edges)),
            # Compatibility aliases for existing report/smoke field names.
            "nethra_reservoir_records": len(self.nodes),
            "nethra_context_roles": len(self.roles),
            "nethra_role_tareth": role_counts.get("tareth", 0),
            "nethra_role_trass": role_counts.get("trass", 0),
            "nethra_role_unresolved": role_counts.get("unresolved", 0),
            "nethra_role_best_available": role_counts.get("best_available", 0),
            "reservoir_queries": self.index_queries,
            "reservoir_matches": self.index_matches,
            "reservoir_matches_used_as_local_anchor": self.matches_used_as_local_anchor,
            "reservoir_assist_feature_hits": self.assist_feature_hits,
            "reservoir_records_by_kind": dict(Counter(r.kind for r in self.nodes.values())),
            "reservoir_records_by_source": dict(Counter(r.source for r in self.nodes.values())),
            "reservoir_roles_by_context": dict(context_counts),
            "reservoir_roles_by_role": dict(role_counts),
        }

    def export_records(self, limit: int = 200) -> dict[str, Any]:
        limit = max(0, int(limit))
        return {
            "nodes": [asdict(r) for r in list(self.nodes.values())[:limit]],
            "edges": [asdict(e) for e in self.edges[:limit]],
            "roles": [asdict(r) for r in self.roles[:limit]],
            # Compatibility aliases for older report code.
            "records": [asdict(r) for r in list(self.nodes.values())[:limit]],
        }

    def _index_node(self, node: NethraNode) -> None:
        if node.target_var is not None:
            self._by_var[int(node.target_var)].add(node.nethra_id)
        for component in node.components:
            self._by_component[int(component)].add(node.nethra_id)
        for parent in node.learned_parents:
            self._by_parent[int(parent)].add(node.nethra_id)


def var_fit_id(var: int, parents: tuple[int, ...], func: str) -> str:
    return f"var_fit:x{int(var)}:{func}({','.join(str(int(p)) for p in parents)})"


def candidate_id(prefix: str, var: int, parents: tuple[int, ...], func: str) -> str:
    return f"{prefix}:x{int(var)}:{func}({','.join(str(int(p)) for p in parents)})"


def context_key(*, operation: str, var: int | None = None, visible: int | None = None, parents: tuple[int, ...] = ()) -> str:
    bits = [operation]
    if var is not None:
        bits.append(f"x{int(var)}")
    if visible is not None:
        bits.append(f"vis={int(visible)}")
    if parents:
        bits.append("parents=" + ",".join(str(int(p)) for p in parents))
    return "|".join(bits)


def _context_tokens(value: str) -> set[str]:
    return {part for part in str(value).replace(",", "|").split("|") if part}


def _cluster_context_tokens(cluster: Any) -> set[str]:
    tokens = {"uncertainty_cluster"}
    tokens.update(f"x{int(v)}" for v in getattr(cluster, "vars", ()) or ())
    tokens.update(f"p{int(v)}" for v in getattr(cluster, "shared_parents", ()) or ())
    tokens.update(str(sig) for sig in getattr(cluster, "shared_signals", ()) or ())
    tokens.add(str(getattr(cluster, "proposed_handle_kind", "unknown")))
    return tokens
