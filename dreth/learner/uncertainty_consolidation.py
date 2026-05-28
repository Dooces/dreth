from __future__ import annotations

"""Visible-evidence uncertainty consolidation.

This module factors repeated uncertainty signals into candidate higher handles.
It is intentionally conservative: clustering and assist proposals use only
agent-visible evidence surfaces, and assists are hints for existing attention
paths rather than authority records.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


ProposedHandleKind = Literal[
    "shared_ambiguity",
    "possible_missing_operator",
    "possible_latent_regime",
    "proxy_confounding_candidate",
    "dense_fanin_candidate",
    "delayed_effect_candidate",
    "weak_noise_floor_candidate",
    "unknown",
]

ProbeFamily = Literal[
    "separating_probe",
    "shared_sentinel_probe",
    "regime_probe",
    "source_edge_disambiguation_probe",
    "no_probe_yet",
]

AssistKind = Literal[
    "prioritize_attention",
    "preserve_alternatives",
    "request_separating_probe",
    "increase_monitoring",
    "repair_priority_bonus",
]


@dataclass(frozen=True)
class UncertaintyCase:
    var: int
    cycle: int
    action: str
    active_signals: tuple[str, ...] = ()
    learned_source_edges: tuple[int, ...] = ()
    near_tie_candidates: tuple[tuple[int, ...], ...] = ()
    tied_frontier_info: dict[str, Any] = field(default_factory=dict)
    novelty_state: str = "closed"
    recent_fit_history: tuple[dict[str, Any], ...] = ()
    sentinels: tuple[tuple[int, float], ...] = ()
    consequence_tier: str = "none"
    graph_neighbors: tuple[int, ...] = ()


@dataclass(frozen=True)
class UncertaintyCluster:
    cluster_id: str
    vars: tuple[int, ...]
    shared_signals: tuple[str, ...] = ()
    shared_source_edges: tuple[int, ...] = ()
    shared_near_tie_candidates: tuple[tuple[int, ...], ...] = ()
    shared_graph_neighbors: tuple[int, ...] = ()
    temporal_persistence: int = 0
    proposed_handle_kind: ProposedHandleKind = "unknown"
    evidence_summary: str = ""
    proposed_next_probe_family: ProbeFamily = "no_probe_yet"
    cluster_size: int = 0
    cluster_fraction_of_visible: float = 0.0
    shared_source_edge_count: int = 0
    shared_neighbor_count: int = 0
    shared_near_tie_count: int = 0
    shared_signal_specificity: float = 0.0
    temporal_cofailure_count: int = 0
    is_giant_cluster: bool = False


@dataclass(frozen=True)
class ConsolidationAssist:
    target_vars: tuple[int, ...]
    cluster_id: str
    assist_kind: AssistKind
    bounded_strength: float
    reason: str


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_signal(sig: str) -> str:
    return str(sig).split("=")[0]


def _recent_churn(history: Iterable[dict[str, Any]]) -> bool:
    entries = list(history)
    if len(entries) < 2:
        return False
    a = tuple(sorted(_as_int(v) for v in entries[-1].get("best_source_edges") or ()))
    b = tuple(sorted(_as_int(v) for v in entries[-2].get("best_source_edges") or ()))
    return a != b


def _case_action(active: set[str]) -> str:
    if "sentinel_failures" in active or "recent_revocations" in active:
        return "repair_attention"
    if "open_novelty" in active and ("near_tie_count" in active or "tie_count" in active):
        return "separate_ambiguity"
    if "open_novelty" in active:
        return "consolidate_novelty"
    if "low_margin" in active or "near_tie_count" in active or "tie_count" in active:
        return "preserve_ambiguity"
    if "repeated_fit_churn" in active:
        return "monitor_churn"
    return "observe"


def _active_from_row(item: dict[str, Any]) -> tuple[str, ...]:
    active: list[str] = []
    if bool(item.get("open_novelty")) or _as_int(item.get("open_novelty_observations")) > 0:
        active.append("open_novelty")
    if item.get("last_fit_margin") is not None and _as_int(item.get("last_fit_margin")) <= 1:
        active.append("low_margin")
    if _as_int(item.get("last_fit_near_tie_count")) > 1:
        active.append("near_tie_count")
    if _as_int(item.get("last_fit_tie_count")) > 1:
        active.append("tie_count")
    if _as_int(item.get("dormant_alternatives")) > 0 or bool(item.get("alternatives_existed")):
        active.append("alternatives")
    if _recent_churn(item.get("recent_fit_history") or ()):
        active.append("repeated_fit_churn")
    if _as_int(item.get("consecutive_sentinel_failures")) > 0:
        active.append("sentinel_failures")
    if _as_int(item.get("recent_revocations")) > 0 or _as_int(item.get("revoked_certs")) > 0:
        active.append("recent_revocations")
    if _as_int(item.get("recent_deferred")) > 0:
        active.append("deferred")
    return tuple(sorted(set(active)))


def _consequence_from_row(item: dict[str, Any]) -> str:
    if _as_int(item.get("route_certs")) > 0:
        return "route"
    role = str(item.get("skip_role") or "none")
    if role in {"tareth", "trass", "noise_floor", "untested"}:
        return f"skip_{role}"
    return "none"


def _frontier_info_from_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "active": bool(item.get("frontier_active")),
        "candidate_count": _as_int(item.get("frontier_candidate_count")),
        "stable_count": _as_int(item.get("frontier_stable_count")),
        "distinct_contexts": _as_int(item.get("frontier_distinct_contexts")),
    }


def _latest_fit_by_var(agent: Any) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    for fd in getattr(agent, "fit_diagnostics", ()) or ():
        out.setdefault(int(fd.var), []).append(fd)
    return out


def extract_uncertainty_cases_from_agent(agent: Any, cycle: int) -> list[UncertaintyCase]:
    """Extract visible uncertainty cases from a live agent.

    Reads ledger state, fit diagnostics, novelty records, and current believed
    graph neighbors. It does not inspect world internals or debug manifests.
    """
    fit_by_var = _latest_fit_by_var(agent)
    open_novelty = {
        int(nv.affected_var)
        for nv in getattr(getattr(agent, "ledger", None), "novelty", ()) or ()
        if getattr(nv, "status", "") == "open"
    }
    cases: list[UncertaintyCase] = []
    visible_count = int(getattr(getattr(agent, "world", None), "visible_count", 0) or 0)
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return cases

    for var in range(visible_count):
        n = ledger.vars[var]
        var_fits = fit_by_var.get(var, [])
        last_fit = var_fits[-1] if var_fits else None
        recent_history = tuple(
            {
                "cycle": int(fd.cycle),
                "best_source_edges": tuple(int(p) for p in fd.best_source_edges),
                "best_func": fd.best_func,
                "margin": int(fd.margin),
                "failure_class": fd.failure_class,
                "tie_count": len(fd.tie_set),
                "near_tie_count": len(fd.near_tie_candidates),
            }
            for fd in var_fits[-5:]
        )
        rowish = {
            "open_novelty": var in open_novelty,
            "last_fit_margin": getattr(last_fit, "margin", None),
            "last_fit_near_tie_count": len(getattr(last_fit, "near_tie_candidates", ()) or ()),
            "last_fit_tie_count": len(getattr(last_fit, "tie_set", ()) or ()),
            "dormant_alternatives": len(getattr(n, "dormant_alternatives", ()) or ()),
            "alternatives_existed": bool(
                getattr(n, "dormant_alternatives", ())
                or (getattr(n, "tied_frontier", None) is not None)
            ),
            "recent_fit_history": recent_history,
            "consecutive_sentinel_failures": getattr(n, "consecutive_sentinel_failures", 0),
            "recent_revocations": sum(
                1 for cert in list(getattr(n, "certificates", {}).values())
                + list(getattr(n, "route_certs", {}).values())
                if getattr(cert, "revoked_by", None)
            ),
            "route_certs": len(getattr(n, "route_certs", {}) or {}),
            "skip_role": n.role_for("skip"),
        }
        active = _active_from_row(rowish)
        if not active:
            continue
        frontier = getattr(n, "tied_frontier", None)
        near_tie_candidates: tuple[tuple[int, ...], ...] = ()
        if last_fit is not None:
            near_tie_candidates = tuple(
                tuple(int(p) for p in source_edges)
                for source_edges, _, _ in getattr(last_fit, "near_tie_candidates", ()) or ()
            )
        graph_neighbors = set(int(p) for p in getattr(n, "source_edges", ()) or ())
        try:
            graph_neighbors.update(int(v) for v in ledger.variable_dependents(var))
        except Exception:
            pass
        cases.append(UncertaintyCase(
            var=var,
            cycle=int(cycle),
            action=_case_action(set(active)),
            active_signals=active,
            learned_source_edges=tuple(sorted(int(p) for p in getattr(n, "source_edges", ()) or ())),
            near_tie_candidates=tuple(sorted(set(near_tie_candidates))),
            tied_frontier_info={
                "active": frontier is not None,
                "candidate_count": len(frontier.candidates) if frontier else 0,
                "stable_count": frontier.stable_count if frontier else 0,
                "distinct_contexts": frontier.distinct_contexts_seen if frontier else 0,
                "has_separating_probes": bool(frontier and frontier.separating_probes),
            },
            novelty_state="open" if var in open_novelty else "closed",
            recent_fit_history=recent_history,
            sentinels=tuple(getattr(n, "sentinels", ()) or ()),
            consequence_tier=rowish["skip_role"] if rowish["skip_role"] else "none",
            graph_neighbors=tuple(sorted(graph_neighbors)),
        ))
    return cases


def extract_uncertainty_cases_from_rows(rows: Iterable[dict[str, Any]]) -> list[UncertaintyCase]:
    """Extract uncertainty cases from batch JSONL rows using visible fields."""
    cases: list[UncertaintyCase] = []
    for row in rows:
        evaluation = row.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            continue
        behavior = evaluation.get("blind_challenge_behavior") or {}
        if not isinstance(behavior, dict):
            continue
        cycle = _as_int(behavior.get("cycles_observed") or row.get("cycles"))
        for item in behavior.get("per_var") or ():
            if not isinstance(item, dict):
                continue
            active = _active_from_row(item)
            if not active:
                continue
            learned = tuple(sorted(_as_int(v) for v in item.get("learned_source_edges") or ()))
            recent_history = tuple(
                dict(h) for h in (item.get("recent_fit_history") or ())
                if isinstance(h, dict)
            )
            near_from_history = tuple(
                tuple(sorted(_as_int(v) for v in h.get("best_source_edges") or ()))
                for h in recent_history
                if _as_int(h.get("near_tie_count")) > 1
            )
            cases.append(UncertaintyCase(
                var=_as_int(item.get("var")),
                cycle=cycle,
                action=_case_action(set(active)),
                active_signals=active,
                learned_source_edges=learned,
                near_tie_candidates=tuple(sorted(set(near_from_history))),
                tied_frontier_info=_frontier_info_from_row(item),
                novelty_state="open" if "open_novelty" in active else "closed",
                recent_fit_history=recent_history,
                sentinels=(),
                consequence_tier=_consequence_from_row(item),
                graph_neighbors=learned,
            ))
    return cases


def _case_overlap_score(a: UncertaintyCase, b: UncertaintyCase) -> int:
    signals_a = {_normalize_signal(s) for s in a.active_signals}
    signals_b = {_normalize_signal(s) for s in b.active_signals}
    score = 0
    if signals_a & signals_b:
        score += 1
    if signals_a == signals_b and signals_a:
        score += 1
    if (
        "sentinel_failures" in signals_a
        and "sentinel_failures" in signals_b
        and abs(a.cycle - b.cycle) <= 5
    ):
        score += 1
    if set(a.learned_source_edges) & set(b.learned_source_edges):
        score += 2
    near_overlap = {
        cand for cand in (set(a.near_tie_candidates) & set(b.near_tie_candidates))
        if cand
    }
    if near_overlap:
        score += 2
    if set(a.graph_neighbors) & set(b.graph_neighbors):
        score += 1
    if _recent_churn(a.recent_fit_history) and _recent_churn(b.recent_fit_history):
        score += 1
    if a.consequence_tier == b.consequence_tier and a.consequence_tier != "none":
        score += 1
    if bool(a.tied_frontier_info.get("active")) == bool(b.tied_frontier_info.get("active")):
        if a.tied_frontier_info.get("active") or b.tied_frontier_info.get("active"):
            score += 1
    if a.novelty_state == b.novelty_state == "open":
        score += 1
    return score


def _shared_tuple_sets(values: Iterable[Iterable[Any]]) -> tuple[Any, ...]:
    sets = [set(v) for v in values]
    if not sets:
        return ()
    shared = set.intersection(*sets)
    return tuple(sorted(shared))


def _temporal_cofailure_count(cases: list[UncertaintyCase], *, recent_cycles: int = 5) -> int:
    failed = [
        c for c in cases
        if "sentinel_failures" in {_normalize_signal(sig) for sig in c.active_signals}
    ]
    if len(failed) < 2:
        return 0
    cycles = [c.cycle for c in failed]
    if max(cycles) - min(cycles) <= recent_cycles:
        return len(failed)
    return 0


def _signal_specificity(
    *,
    shared_source_edge_count: int,
    shared_neighbor_count: int,
    shared_near_tie_count: int,
    temporal_cofailure_count: int,
) -> float:
    anchors = (
        int(shared_source_edge_count > 0)
        + int(shared_neighbor_count > 0)
        + int(shared_near_tie_count > 0)
        + int(temporal_cofailure_count > 0)
    )
    return anchors / 4.0


def cluster_has_specific_local_anchor(cluster: UncertaintyCluster) -> bool:
    """Return whether a cluster has a visible local anchor for runtime assist."""
    return (
        cluster.shared_source_edge_count > 0
        or cluster.shared_near_tie_count > 0
        or cluster.shared_neighbor_count > 0
        or cluster.temporal_cofailure_count > 0
    )


def _handle_kind(cases: list[UncertaintyCase]) -> ProposedHandleKind:
    signal_counts = Counter(
        _normalize_signal(sig)
        for c in cases
        for sig in c.active_signals
    )
    if signal_counts["sentinel_failures"] >= 2:
        return "possible_latent_regime"
    if signal_counts["open_novelty"] >= 2 and (
        signal_counts["near_tie_count"] or signal_counts["tie_count"]
    ):
        return "possible_missing_operator"
    if any(len(c.learned_source_edges) >= 2 for c in cases) and _shared_tuple_sets(c.learned_source_edges for c in cases):
        return "dense_fanin_candidate"
    if signal_counts["repeated_fit_churn"] >= 2:
        return "proxy_confounding_candidate"
    if signal_counts["low_margin"] >= 2 and not signal_counts["open_novelty"]:
        return "weak_noise_floor_candidate"
    if any(c.tied_frontier_info.get("active") for c in cases):
        return "shared_ambiguity"
    if max((c.cycle for c in cases), default=0) - min((c.cycle for c in cases), default=0) > 50:
        return "delayed_effect_candidate"
    return "unknown"


def _probe_family(cases: list[UncertaintyCase], kind: ProposedHandleKind) -> ProbeFamily:
    signals = {
        _normalize_signal(sig)
        for c in cases
        for sig in c.active_signals
    }
    if "sentinel_failures" in signals:
        return "shared_sentinel_probe"
    if kind == "possible_latent_regime":
        return "regime_probe"
    if "near_tie_count" in signals or "tie_count" in signals:
        return "separating_probe"
    if _shared_tuple_sets(c.learned_source_edges for c in cases):
        return "source_edge_disambiguation_probe"
    return "no_probe_yet"


def cluster_uncertainty_cases(
    cases: list[UncertaintyCase],
    *,
    visible_count: int | None = None,
) -> list[UncertaintyCluster]:
    if not cases:
        return []
    visible = int(visible_count or 0)
    if visible <= 0:
        visible = max((c.var for c in cases), default=-1) + 1
    visible = max(1, visible)
    source_edge = list(range(len(cases)))

    def find(i: int) -> int:
        while source_edge[i] != i:
            source_edge[i] = source_edge[source_edge[i]]
            i = source_edge[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            source_edge[rb] = ra

    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            if _case_overlap_score(cases[i], cases[j]) >= 3:
                union(i, j)

    groups: dict[int, list[UncertaintyCase]] = {}
    for idx, case in enumerate(cases):
        groups.setdefault(find(idx), []).append(case)

    clusters: list[UncertaintyCluster] = []
    for ordinal, group in enumerate(groups.values(), start=1):
        vars_ = tuple(sorted({c.var for c in group}))
        signal_sets = [tuple(_normalize_signal(sig) for sig in c.active_signals) for c in group]
        shared_signals = _shared_tuple_sets(signal_sets)
        shared_source_edges = _shared_tuple_sets(c.learned_source_edges for c in group)
        shared_near = tuple(
            cand for cand in _shared_tuple_sets(c.near_tie_candidates for c in group)
            if cand
        )
        shared_neighbors = _shared_tuple_sets(c.graph_neighbors for c in group)
        temporal = max(c.cycle for c in group) - min(c.cycle for c in group)
        temporal_cofailure = _temporal_cofailure_count(group)
        cluster_size = len(vars_)
        cluster_fraction = cluster_size / visible
        shared_source_edge_count = len(shared_source_edges)
        shared_neighbor_count = len(shared_neighbors)
        shared_near_tie_count = len(shared_near)
        specificity = _signal_specificity(
            shared_source_edge_count=shared_source_edge_count,
            shared_neighbor_count=shared_neighbor_count,
            shared_near_tie_count=shared_near_tie_count,
            temporal_cofailure_count=temporal_cofailure,
        )
        kind = _handle_kind(group)
        probe = _probe_family(group, kind)
        evidence_bits: list[str] = []
        if shared_signals:
            evidence_bits.append("signals=" + ",".join(shared_signals))
        if shared_source_edges:
            evidence_bits.append("source_edges=" + ",".join(f"x{p}" for p in shared_source_edges))
        if shared_near:
            evidence_bits.append(f"shared_near_tie={len(shared_near)}")
        if shared_neighbors:
            evidence_bits.append("neighbors=" + ",".join(f"x{p}" for p in shared_neighbors[:5]))
        if temporal:
            evidence_bits.append(f"persistence={temporal}")
        if temporal_cofailure:
            evidence_bits.append(f"temporal_cofailure={temporal_cofailure}")
        clusters.append(UncertaintyCluster(
            cluster_id=f"uc{ordinal}",
            vars=vars_,
            shared_signals=tuple(str(s) for s in shared_signals),
            shared_source_edges=tuple(int(p) for p in shared_source_edges),
            shared_near_tie_candidates=tuple(shared_near),
            shared_graph_neighbors=tuple(int(p) for p in shared_neighbors),
            temporal_persistence=int(temporal),
            proposed_handle_kind=kind,
            evidence_summary="; ".join(evidence_bits) if evidence_bits else "single visible uncertainty case",
            proposed_next_probe_family=probe,
            cluster_size=cluster_size,
            cluster_fraction_of_visible=cluster_fraction,
            shared_source_edge_count=shared_source_edge_count,
            shared_neighbor_count=shared_neighbor_count,
            shared_near_tie_count=shared_near_tie_count,
            shared_signal_specificity=specificity,
            temporal_cofailure_count=temporal_cofailure,
            is_giant_cluster=cluster_fraction > 0.40,
        ))
    clusters.sort(key=lambda c: (-len(c.vars), c.cluster_id))
    return clusters


def propose_consolidation_assists(clusters: list[UncertaintyCluster]) -> list[ConsolidationAssist]:
    assists: list[ConsolidationAssist] = []
    for cluster in clusters:
        if len(cluster.vars) < 2:
            continue
        strength = min(0.5, 0.15 + 0.05 * len(cluster.vars))
        signals = set(cluster.shared_signals)
        reason = cluster.evidence_summary
        assists.append(ConsolidationAssist(
            target_vars=cluster.vars,
            cluster_id=cluster.cluster_id,
            assist_kind="prioritize_attention",
            bounded_strength=strength,
            reason=reason,
        ))
        if signals & {"near_tie_count", "tie_count", "alternatives"} or cluster.shared_near_tie_candidates:
            assists.append(ConsolidationAssist(
                target_vars=cluster.vars,
                cluster_id=cluster.cluster_id,
                assist_kind="preserve_alternatives",
                bounded_strength=min(0.35, strength),
                reason="clustered ambiguity has visible alternatives",
            ))
        if (
            cluster.proposed_next_probe_family in {"separating_probe", "source_edge_disambiguation_probe"}
            or "alternatives" in signals
            or bool(cluster.shared_near_tie_candidates)
        ):
            assists.append(ConsolidationAssist(
                target_vars=cluster.vars,
                cluster_id=cluster.cluster_id,
                assist_kind="request_separating_probe",
                bounded_strength=min(0.35, strength),
                reason=f"next probe family is {cluster.proposed_next_probe_family}",
            ))
        if signals & {"sentinel_failures", "repeated_fit_churn", "open_novelty"}:
            assists.append(ConsolidationAssist(
                target_vars=cluster.vars,
                cluster_id=cluster.cluster_id,
                assist_kind="increase_monitoring",
                bounded_strength=min(0.3, strength),
                reason="clustered uncertainty indicates local monitoring should be denser",
            ))
        if signals & {"sentinel_failures", "recent_revocations", "deferred"}:
            assists.append(ConsolidationAssist(
                target_vars=cluster.vars,
                cluster_id=cluster.cluster_id,
                assist_kind="repair_priority_bonus",
                bounded_strength=min(0.4, strength),
                reason="clustered repair signals should be visible to the repair agenda",
            ))
    return assists


def summarize_clusters(clusters: list[UncertaintyCluster]) -> dict[str, Any]:
    case_count = sum(len(c.vars) for c in clusters)
    cluster_count = len(clusters)
    sizes = [len(c.vars) for c in clusters]
    specificity_values = [c.shared_signal_specificity for c in clusters]
    return {
        "uncertainty_clusters": cluster_count,
        "uncertainty_cases_seen": case_count,
        "uncertainty_compression_ratio": (
            case_count / cluster_count if cluster_count else 0.0
        ),
        "max_cluster_size": max(sizes) if sizes else 0,
        "avg_cluster_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "cluster_specificity_mean": (
            sum(specificity_values) / len(specificity_values)
            if specificity_values else 0.0
        ),
        "giant_cluster_count": sum(1 for c in clusters if c.is_giant_cluster),
        "cluster_size_distribution": dict(Counter(sizes)),
        "handle_kinds": dict(Counter(c.proposed_handle_kind for c in clusters)),
        "probe_families": dict(Counter(c.proposed_next_probe_family for c in clusters)),
    }
