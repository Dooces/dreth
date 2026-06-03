from __future__ import annotations

"""Route index for runtime search-space narrowing.

This index is deliberately operational: rows describe how a scoped search point
may narrow candidates or probes. It does not confer authority.
"""

from dataclasses import dataclass, field
from typing import Any

from dreth.nethra_memory_store import USE_RIGHTS


_GLOBAL = "*"

_HOOK_USE_RIGHTS: dict[str, frozenset[str]] = {
    "source_edge_candidates": frozenset({"ranking_hint", "soft_filter"}),
    "ranking_hint": frozenset({"ranking_hint", "soft_filter"}),
    "probe_hint": frozenset({"probe_hint", "soft_filter"}),
}

_PRIMARY_HOOKS: frozenset[str] = frozenset({"source_edge_candidates", "ranking_hint", "probe_hint"})
_BLOCKED_PRIMARY_ROLE_STATES: frozenset[str] = frozenset({"trass", "unresolved"})


@dataclass(frozen=True)
class SearchRoute:
    route_id: str
    nethra_id: str = ""
    operation_hook: str = ""
    target_anchor: str = ""
    trigger_anchors: tuple[str, ...] = ()
    candidate_region: tuple[str, ...] = ()
    deferred_region: tuple[str, ...] = ()
    residual_bucket_key: str = ""
    probe_region: tuple[str, ...] = ()
    invalidators: tuple[str, ...] = ()
    role_state: str = ""
    use_right: str = "record_only"
    saved_search_count: int = 0
    wasted_search_count: int = 0
    miss_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    first_seen: int = 0
    last_seen: int = 0
    salience: float = 0.0
    evidence_refs: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if self.use_right not in USE_RIGHTS:
            object.__setattr__(self, "use_right", "record_only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_kind": "nethra_search_route",
            "route_id": self.route_id,
            "nethra_id": self.nethra_id,
            "operation_hook": self.operation_hook,
            "target_anchor": self.target_anchor,
            "trigger_anchors": list(self.trigger_anchors),
            "candidate_region": list(self.candidate_region),
            "deferred_region": list(self.deferred_region),
            "residual_bucket_key": self.residual_bucket_key,
            "probe_region": list(self.probe_region),
            "invalidators": list(self.invalidators),
            "role_state": self.role_state,
            "use_right": self.use_right,
            "saved_search_count": self.saved_search_count,
            "wasted_search_count": self.wasted_search_count,
            "miss_count": self.miss_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "salience": self.salience,
            "evidence_refs": list(self.evidence_refs),
            "source": self.source,
            "authority_allowed": False,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SearchRoute | None":
        if str(row.get("entry_kind", "")) != "nethra_search_route":
            return None
        route_id = str(row.get("route_id", ""))
        if not route_id:
            return None
        use_right = str(row.get("use_right", "record_only"))
        if use_right == "hard_filter" and str(row.get("source", "")) != "runtime_local_evidence":
            use_right = "record_only"
        return cls(
            route_id=route_id,
            nethra_id=str(row.get("nethra_id", "")),
            operation_hook=str(row.get("operation_hook", "")),
            target_anchor=str(row.get("target_anchor", "")),
            trigger_anchors=_tuple_str(row.get("trigger_anchors")),
            candidate_region=_tuple_str(row.get("candidate_region")),
            deferred_region=_tuple_str(row.get("deferred_region")),
            residual_bucket_key=str(row.get("residual_bucket_key", "")),
            probe_region=_tuple_str(row.get("probe_region")),
            invalidators=_tuple_str(row.get("invalidators")),
            role_state=str(row.get("role_state", "")),
            use_right=use_right,
            saved_search_count=int(row.get("saved_search_count", 0) or 0),
            wasted_search_count=int(row.get("wasted_search_count", 0) or 0),
            miss_count=int(row.get("miss_count", 0) or 0),
            success_count=int(row.get("success_count", 0) or 0),
            failure_count=int(row.get("failure_count", 0) or 0),
            first_seen=int(row.get("first_seen", 0) or 0),
            last_seen=int(row.get("last_seen", 0) or 0),
            salience=float(row.get("salience", 0.0) or 0.0),
            evidence_refs=_tuple_str(row.get("evidence_refs")),
            source=str(row.get("source", "")),
        )


class NethraRouteIndex:
    """Posting-list route index keyed by target, context prefix, and hook."""

    def __init__(self) -> None:
        self._routes: dict[str, SearchRoute] = {}
        self._by_target: dict[str, set[str]] = {}
        self._by_context: dict[str, set[str]] = {}
        self._by_hook: dict[str, set[str]] = {}
        self._by_use_right: dict[str, set[str]] = {}
        self._cache: dict[tuple[str, str, str, tuple[str, ...]], tuple[SearchRoute, ...]] = {}

    def index_route(self, route: SearchRoute) -> None:
        if route.route_id in self._routes:
            self.remove_route(route.route_id)
        self._routes[route.route_id] = route
        self._by_target.setdefault(route.target_anchor or _GLOBAL, set()).add(route.route_id)
        self._by_context.setdefault(_context_prefix_from_route(route), set()).add(route.route_id)
        self._by_hook.setdefault(route.operation_hook or _GLOBAL, set()).add(route.route_id)
        self._by_use_right.setdefault(route.use_right, set()).add(route.route_id)
        self._cache.clear()

    def index_row(self, row: dict[str, Any]) -> bool:
        route = SearchRoute.from_row(row)
        if route is None:
            return False
        self.index_route(route)
        return True

    def remove_route(self, route_id: str) -> None:
        route = self._routes.pop(route_id, None)
        if route is None:
            return
        for mapping in (self._by_target, self._by_context, self._by_hook, self._by_use_right):
            for bucket in mapping.values():
                bucket.discard(route_id)
        self._cache.clear()

    def query(
        self,
        target_anchor: str,
        context_key: str,
        operation_hook: str,
        *,
        active_invalidators: set[str] | None = None,
        top_k: int = 20,
    ) -> list[SearchRoute]:
        ctx_prefix = context_key.split("|")[0] if context_key else ""
        invalidators_key = tuple(sorted(active_invalidators or set()))
        cache_key = (target_anchor, ctx_prefix, operation_hook, invalidators_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached[:top_k])

        target_ids = self._ids_for(self._by_target, target_anchor)
        context_ids = self._ids_for(self._by_context, ctx_prefix)
        hook_ids = self._ids_for(self._by_hook, operation_hook)

        # Posting-list intersection when possible; fall back to narrower unions
        # for wildcard routes and partially specified legacy rows.
        candidate_ids = target_ids & context_ids & hook_ids
        if not candidate_ids:
            candidate_ids = (target_ids & hook_ids) | (context_ids & hook_ids)
        if not candidate_ids:
            candidate_ids = hook_ids or target_ids or context_ids or set(self._routes.keys())

        allowed = _HOOK_USE_RIGHTS.get(operation_hook)
        active_invalidators = active_invalidators or set()
        scored: list[tuple[float, SearchRoute]] = []
        for route_id in candidate_ids:
            route = self._routes.get(route_id)
            if route is None:
                continue
            if allowed is not None and route.use_right not in allowed:
                continue
            if operation_hook in _PRIMARY_HOOKS and route.role_state in _BLOCKED_PRIMARY_ROLE_STATES:
                continue
            if active_invalidators and set(route.invalidators) & active_invalidators:
                continue
            score = _score_route(route, target_anchor, ctx_prefix, operation_hook)
            scored.append((score, route))

        scored.sort(key=lambda item: (-item[0], item[1].route_id))
        result = tuple(route for _, route in scored)
        self._cache[cache_key] = result
        return list(result[:top_k])

    def size(self) -> int:
        return len(self._routes)

    def _ids_for(self, mapping: dict[str, set[str]], key: str) -> set[str]:
        return set(mapping.get(key, set())) | set(mapping.get(_GLOBAL, set()))


def _score_route(route: SearchRoute, target_anchor: str, ctx_prefix: str, hook: str) -> float:
    score = 0.0
    if route.target_anchor == target_anchor:
        score += 2.0
    elif route.target_anchor in ("", _GLOBAL):
        score += 0.2
    if ctx_prefix and _context_prefix_from_route(route) == ctx_prefix:
        score += 1.0
    if route.operation_hook == hook:
        score += 1.0
    score += min(2.0, route.saved_search_count * 0.1)
    score -= min(2.0, route.wasted_search_count * 0.2)
    score -= min(2.0, route.miss_count * 0.3)
    score += min(1.5, route.success_count * 0.2)
    score -= min(1.5, route.failure_count * 0.3)
    score += min(1.0, route.salience * 0.1)
    return score


def _context_prefix_from_route(route: SearchRoute) -> str:
    for anchor in route.trigger_anchors:
        if "|" in anchor:
            return anchor.split("|")[0]
    if "|" in route.residual_bucket_key:
        return route.residual_bucket_key.split("|")[0]
    return _GLOBAL


def _tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int, float)):
        return (str(value),)
    if not isinstance(value, list | tuple | set):
        return ()
    return tuple(str(v) for v in value)
