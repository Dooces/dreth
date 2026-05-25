from __future__ import annotations

"""Shadow-only authority throttle evaluator for blind_challenge offline analysis.

Estimates whether downgrading or withholding authority when visible evidence
was weak or contradictory would have reduced authority/evidence mismatches, and
how much useful authority would have been lost.

INVARIANTS (enforced by design, not runtime):
  - Diagnostic only. No effect on skip behavior, cert issuance, revocation, fit,
    sentinel behavior, route certs, provider choice, or defaults.
  - Classification uses agent-visible evidence fields only. truth_parents,
    truth_func, truth_delayed_parents, and all other hidden-world fields are
    never read by this module.
  - Never imported by agent.py, ChainedAgent, or any runtime path.
"""

from dataclasses import dataclass
from typing import Any


CLASSIFICATIONS = (
    "evidence_supported_surrogate",
    "weakly_supported_surrogate",
    "contradicted_authority",
    "insufficient_evidence",
    "unknown",
)

THROTTLE_MODES = ("conservative", "strict")


@dataclass
class AuthorityThrottleDecision:
    var: int
    reason: str
    would_throttle: bool
    evidence_class: str
    strong_observations: int
    sentinel_count: int
    fit_history_count: int
    last_fit_margin: int
    recent_revocations: int
    recent_detected_drift: int
    consecutive_sentinel_failures: int
    open_novelty: bool
    alternatives_existed: bool
    tie_count: int
    near_tie_count: int


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _has_contradiction(item: dict[str, Any]) -> bool:
    return (
        _as_int(item.get("recent_revocations")) >= 2
        or _as_int(item.get("recent_detected_drift")) >= 2
        or _as_int(item.get("consecutive_sentinel_failures")) > 0
        or _as_int(item.get("open_novelty_observations")) > 0
        or (
            item.get("passive_stress_recent") is not None
            and _as_int(item.get("passive_stress_recent")) > 0
        )
    )


def _has_low_evidence(item: dict[str, Any]) -> bool:
    return (
        ("strong_observations" in item and _as_int(item.get("strong_observations")) <= 0)
        or ("sentinel_count" in item and _as_int(item.get("sentinel_count")) <= 0)
        or ("fit_history_count" in item and _as_int(item.get("fit_history_count")) <= 0)
        or (
            item.get("last_fit_margin") is not None
            and _as_int(item.get("last_fit_margin")) <= 0
        )
    )


def _has_stable_support(item: dict[str, Any]) -> bool:
    return (
        bool(item.get("repeatedly_stable_under_probes"))
        and _as_int(item.get("strong_observations")) >= 2
        and _as_int(item.get("sentinel_count")) > 0
        and _as_int(item.get("fit_history_count")) > 0
        and _as_int(item.get("last_fit_margin")) > 0
        and not _has_contradiction(item)
    )


def _no_better_alternative_visible(item: dict[str, Any]) -> bool:
    return (
        not bool(item.get("alternatives_existed"))
        and _as_int(item.get("last_fit_tie_count")) <= 1
        and _as_int(item.get("last_fit_near_tie_count")) <= 1
    )


def classify_visible_authority_evidence(item: dict[str, Any]) -> str:
    """Classify authority support from agent-visible evidence fields only.

    Uses the same classification scheme as the blind_challenge authority
    evidence summarizer. Never reads truth_parents, truth_func, or any
    other hidden-world fields.
    """
    if _has_contradiction(item):
        return "contradicted_authority"
    if _has_low_evidence(item):
        return "insufficient_evidence"
    if _has_stable_support(item) and _no_better_alternative_visible(item):
        return "evidence_supported_surrogate"
    if (
        _as_int(item.get("strong_observations")) > 0
        or _as_int(item.get("sentinel_count")) > 0
        or _as_int(item.get("last_fit_margin")) > 0
    ):
        return "weakly_supported_surrogate"
    return "unknown"


def would_throttle_authority(
    item: dict[str, Any], mode: str = "conservative"
) -> AuthorityThrottleDecision:
    """Estimate whether authority would have been throttled given visible evidence.

    Conservative mode:
      - Throttle if contradicted_authority.
      - Throttle if insufficient_evidence and the var is authoritative (authority
        is strong).
      - Do not throttle evidence_supported_surrogate.
      - Do not throttle weakly_supported_surrogate.

    Strict mode:
      - Throttle contradicted_authority.
      - Throttle insufficient_evidence regardless of authority strength.
      - Throttle weakly_supported_surrogate when alternatives_existed is True,
        or tie_count > 1, or near_tie_count > 1.
    """
    if mode not in THROTTLE_MODES:
        raise ValueError(
            f"Unknown throttle mode {mode!r}; expected one of {THROTTLE_MODES}"
        )

    evidence_class = classify_visible_authority_evidence(item)
    authority_strong = bool(item.get("authoritative"))
    alternatives_existed = bool(item.get("alternatives_existed"))
    tie_count = _as_int(item.get("last_fit_tie_count"))
    near_tie_count = _as_int(item.get("last_fit_near_tie_count"))

    would_throttle = False
    reason = "no_throttle"

    if evidence_class == "contradicted_authority":
        would_throttle = True
        reason = "contradicted_authority"
    elif evidence_class == "insufficient_evidence":
        if mode == "strict" or authority_strong:
            would_throttle = True
            reason = "insufficient_evidence"
    elif evidence_class == "weakly_supported_surrogate":
        if mode == "strict" and (
            alternatives_existed or tie_count > 1 or near_tie_count > 1
        ):
            would_throttle = True
            reason = "weakly_supported_strict"

    return AuthorityThrottleDecision(
        var=_as_int(item.get("var", 0)),
        reason=reason,
        would_throttle=would_throttle,
        evidence_class=evidence_class,
        strong_observations=_as_int(item.get("strong_observations")),
        sentinel_count=_as_int(item.get("sentinel_count")),
        fit_history_count=_as_int(item.get("fit_history_count")),
        last_fit_margin=_as_int(item.get("last_fit_margin")),
        recent_revocations=_as_int(item.get("recent_revocations")),
        recent_detected_drift=_as_int(item.get("recent_detected_drift")),
        consecutive_sentinel_failures=_as_int(item.get("consecutive_sentinel_failures")),
        open_novelty=_as_int(item.get("open_novelty_observations")) > 0,
        alternatives_existed=alternatives_existed,
        tie_count=tie_count,
        near_tie_count=near_tie_count,
    )
