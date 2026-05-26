from __future__ import annotations

"""Visible-evidence authority-strength metadata.

Authority strength is a runtime classification over evidence the agent already
has in hand. It is not a certificate, does not revoke certificates, and does
not replace the current best fit.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AuthorityStrength = Literal[
    "strong",
    "usable",
    "weak",
    "contested",
    "insufficient",
]


@dataclass(frozen=True)
class AuthorityStrengthRecord:
    var: int
    nethra_id: str
    context_key: str
    cycle: int
    strength: AuthorityStrength
    reason: str
    active_evidence: tuple[str, ...] = ()
    contradictory_evidence: tuple[str, ...] = ()
    required_future_evidence: tuple[str, ...] = ()
    uncertainty_signals: tuple[str, ...] = ()
    prior_role: str = ""
    best_available: bool = False


@dataclass(frozen=True)
class AuthorityStrengthSummary:
    counts_by_strength: dict[str, int] = field(default_factory=dict)
    counts_by_reason: dict[str, int] = field(default_factory=dict)
    weak_best_available: int = 0
    contested_best_available: int = 0
    monitoring_increases: int = 0
    alternatives_preserved: int = 0
    future_evidence_requirements: int = 0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _latest_fit_by_var(agent: Any) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = defaultdict(list)
    for fd in getattr(agent, "fit_diagnostics", ()) or ():
        out[int(getattr(fd, "var", -1))].append(fd)
    return dict(out)


def _recent_churn(fits: list[Any]) -> bool:
    if len(fits) < 2:
        return False
    recent = fits[-4:]
    signatures = {
        (
            tuple(int(p) for p in getattr(fd, "best_parents", ()) or ()),
            str(getattr(fd, "best_func", "")),
        )
        for fd in recent
    }
    return len(signatures) > 1


def _recent_role_for_nethra(agent: Any, nethra_id: str) -> str:
    index = getattr(agent, "_context_role_index", None)
    if index is None:
        return ""
    roles = getattr(index, "_roles_by_nethra", {}).get(nethra_id, ())
    if not roles:
        return ""
    return str(roles[-1].role)


def _cluster_signals_by_var(agent: Any) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for cluster in getattr(agent, "_uncertainty_latest_clusters", ()) or ():
        signals = {str(sig) for sig in getattr(cluster, "shared_signals", ()) or ()}
        if getattr(cluster, "is_giant_cluster", False):
            signals.add("uncertainty_giant_cluster")
        if getattr(cluster, "cluster_size", 0):
            signals.add("uncertainty_cluster")
        for var in getattr(cluster, "vars", ()) or ():
            out[int(var)].update(signals)
    return out


def _open_novelty_vars(agent: Any) -> set[int]:
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return set()
    return {
        int(getattr(item, "affected_var", -1))
        for item in getattr(ledger, "novelty", ()) or ()
        if getattr(item, "status", "") == "open"
    }


def _var_nethra_id(var: int, parents: tuple[int, ...], func: str) -> str:
    return f"var_fit:x{int(var)}:{func}({','.join(str(int(p)) for p in parents)})"


def _context_key(var: int, visible: int, parents: tuple[int, ...]) -> str:
    bits = [f"authority_strength", f"x{int(var)}", f"vis={int(visible)}"]
    if parents:
        bits.append("parents=" + ",".join(str(int(p)) for p in parents))
    return "|".join(bits)


def _revocation_count(n: Any) -> int:
    certs = list(getattr(n, "certificates", {}).values())
    certs += list(getattr(n, "route_certs", {}).values())
    return sum(1 for cert in certs if getattr(cert, "revoked_by", None))


def _sentinel_passes(n: Any) -> int:
    return sum(
        _as_int(getattr(cert, "sentinel_passes", 0))
        for cert in getattr(n, "certificates", {}).values()
    )


def _classify(
    *,
    strong_observations: int,
    sentinel_count: int,
    sentinel_passes: int,
    fit_margin: int | None,
    near_tie_count: int,
    tie_count: int,
    open_novelty: bool,
    churn: bool,
    recent_revocations: int,
    sentinel_failures: int,
    passive_stress: int,
    dormant_alternatives: int,
    cluster_signals: set[str],
    best_available: bool,
) -> tuple[AuthorityStrength, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    active: list[str] = []
    contradictory: list[str] = []
    future: list[str] = []

    if strong_observations > 0:
        active.append(f"strong_observations={strong_observations}")
    if sentinel_count > 0:
        active.append(f"sentinel_coverage={sentinel_count}")
    if sentinel_passes > 0:
        active.append(f"sentinel_passes={sentinel_passes}")
    if fit_margin is not None and fit_margin >= 3:
        active.append(f"fit_margin={fit_margin}")

    if fit_margin is None:
        contradictory.append("fit_margin_missing")
    elif fit_margin <= 1:
        contradictory.append(f"low_fit_margin={fit_margin}")
    if near_tie_count > 1:
        contradictory.append(f"near_ties={near_tie_count}")
    if tie_count > 1:
        contradictory.append(f"tied_frontier={tie_count}")
    if open_novelty:
        contradictory.append("open_novelty")
    if churn:
        contradictory.append("repeated_fit_churn")
    if recent_revocations > 0:
        contradictory.append(f"recent_revocations={recent_revocations}")
    if sentinel_failures > 0:
        contradictory.append(f"sentinel_failures={sentinel_failures}")
    if passive_stress > 0:
        contradictory.append(f"passive_stress={passive_stress}")
    if dormant_alternatives > 0:
        contradictory.append(f"dormant_alternatives={dormant_alternatives}")
    if cluster_signals:
        contradictory.extend(sorted(cluster_signals))

    if sentinel_count == 0:
        future.append("sentinel_coverage")
    if strong_observations < 2:
        future.append("more_stable_observations")
    if fit_margin is None or fit_margin <= 1:
        future.append("larger_fit_margin")
    if near_tie_count > 1 or tie_count > 1 or dormant_alternatives > 0:
        future.append("separating_evidence")
    if open_novelty or churn or recent_revocations or sentinel_failures:
        future.append("stability_after_uncertainty")

    contested_markers = {
        "open_novelty",
        "repeated_fit_churn",
        "uncertainty_cluster",
        "uncertainty_giant_cluster",
    }
    has_contested_marker = (
        any(item.split("=", 1)[0] in contested_markers for item in contradictory)
        or near_tie_count > 1
        or tie_count > 1
        or dormant_alternatives > 0
        or recent_revocations > 0
        or sentinel_failures > 0
    )
    if has_contested_marker:
        return (
            "contested",
            "active_visible_conflict",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    if strong_observations <= 0 and sentinel_count <= 0 and fit_margin is None:
        return (
            "insufficient",
            "too_little_visible_evidence",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    low_observation = strong_observations < 2
    low_margin = fit_margin is not None and fit_margin <= 1
    limited_sentinel = sentinel_count <= 0
    if best_available and (low_observation or low_margin or limited_sentinel):
        return (
            "weak",
            "best_available_but_limited_evidence",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    if (
        strong_observations >= 3
        and sentinel_count > 0
        and (fit_margin is None or fit_margin >= 2)
        and not contradictory
    ):
        return (
            "strong",
            "stable_visible_evidence",
            tuple(active),
            tuple(),
            tuple(),
        )

    return (
        "usable",
        "best_available_mostly_stable",
        tuple(active),
        tuple(dict.fromkeys(contradictory)),
        tuple(dict.fromkeys(future)),
    )


def compute_authority_strength_records(agent: Any, cycle: int) -> list[AuthorityStrengthRecord]:
    """Classify current handles using visible runtime evidence only."""
    ledger = getattr(agent, "ledger", None)
    world = getattr(agent, "world", None)
    if ledger is None or world is None:
        return []
    visible = _as_int(getattr(world, "visible_count", 0))
    fit_by_var = _latest_fit_by_var(agent)
    novelty_vars = _open_novelty_vars(agent)
    cluster_signals = _cluster_signals_by_var(agent)
    records: list[AuthorityStrengthRecord] = []

    for var in range(visible):
        n = ledger.vars[var]
        parents = tuple(int(p) for p in getattr(n, "parents", ()) or ())
        func = str(getattr(n, "func", ""))
        fits = fit_by_var.get(var, [])
        last_fit = fits[-1] if fits else None
        fit_margin = (
            int(getattr(last_fit, "margin"))
            if last_fit is not None and getattr(last_fit, "margin", None) is not None
            else None
        )
        near_tie_count = len(getattr(last_fit, "near_tie_candidates", ()) or ()) if last_fit else 0
        tie_count = len(getattr(last_fit, "tie_set", ()) or ()) if last_fit else 0
        frontier = getattr(n, "tied_frontier", None)
        if frontier is not None:
            tie_count = max(tie_count, len(getattr(frontier, "candidates", ()) or ()))
        nid = _var_nethra_id(var, parents, func)
        strength, reason, active, contradictory, future = _classify(
            strong_observations=_as_int(getattr(n, "strong_observations", 0)),
            sentinel_count=len(getattr(n, "sentinels", ()) or ()),
            sentinel_passes=_sentinel_passes(n),
            fit_margin=fit_margin,
            near_tie_count=near_tie_count,
            tie_count=tie_count,
            open_novelty=var in novelty_vars,
            churn=_recent_churn(fits),
            recent_revocations=_revocation_count(n),
            sentinel_failures=_as_int(getattr(n, "consecutive_sentinel_failures", 0)),
            passive_stress=_as_int(getattr(getattr(n, "envelope", None), "out_of_band_count", 0)),
            dormant_alternatives=len(getattr(n, "dormant_alternatives", ()) or ()),
            cluster_signals=cluster_signals.get(var, set()),
            best_available=True,
        )
        records.append(AuthorityStrengthRecord(
            var=var,
            nethra_id=nid,
            context_key=_context_key(var, visible, parents),
            cycle=int(cycle),
            strength=strength,
            reason=reason,
            active_evidence=active,
            contradictory_evidence=contradictory,
            required_future_evidence=future,
            uncertainty_signals=tuple(sorted(cluster_signals.get(var, set()))),
            prior_role=_recent_role_for_nethra(agent, nid),
            best_available=True,
        ))
    return records


def summarize_authority_strength_records(
    records: list[AuthorityStrengthRecord],
    *,
    monitoring_increases: int = 0,
    alternatives_preserved: int = 0,
) -> AuthorityStrengthSummary:
    strength_counts = Counter(record.strength for record in records)
    reason_counts = Counter(record.reason for record in records)
    return AuthorityStrengthSummary(
        counts_by_strength=dict(strength_counts),
        counts_by_reason=dict(reason_counts),
        weak_best_available=sum(
            1 for record in records
            if record.best_available and record.strength == "weak"
        ),
        contested_best_available=sum(
            1 for record in records
            if record.best_available and record.strength == "contested"
        ),
        monitoring_increases=max(0, int(monitoring_increases)),
        alternatives_preserved=max(0, int(alternatives_preserved)),
        future_evidence_requirements=sum(
            1 for record in records if record.required_future_evidence
        ),
    )


def records_to_dicts(records: list[AuthorityStrengthRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def summary_to_dict(summary: AuthorityStrengthSummary) -> dict[str, Any]:
    return asdict(summary)
