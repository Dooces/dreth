from __future__ import annotations

"""Shadow-only authority throttle evaluator for blind_challenge offline analysis.

Estimates whether downgrading or withholding authority when visible evidence
was weak or contradictory would have reduced authority/evidence mismatches, and
how much useful authority would have been lost.

INVARIANTS (enforced by design, not runtime):
  - Diagnostic only. No effect on skip behavior, cert issuance, revocation, fit,
    sentinel behavior, route certs, provider choice, or defaults.
  - Classification uses agent-visible evidence fields only. truth_source_edges,
    truth_func, truth_delayed_source_edges, and all other hidden-world fields are
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

TRIGGER_NAMES = (
    "recent_revocations_trigger",
    "recent_detected_drift_trigger",
    "consecutive_sentinel_failure_trigger",
    "open_novelty_trigger",
    "passive_stress_trigger",
    "low_strong_observations_trigger",
    "low_sentinel_count_trigger",
    "low_fit_history_trigger",
    "low_margin_trigger",
    "alternatives_or_ties_trigger",
)


@dataclass
class EvidenceTriggers:
    """Boolean flags for each individual observable contradiction or low-evidence signal."""

    recent_revocations_trigger: bool
    recent_detected_drift_trigger: bool
    consecutive_sentinel_failure_trigger: bool
    open_novelty_trigger: bool
    passive_stress_trigger: bool
    low_strong_observations_trigger: bool
    low_sentinel_count_trigger: bool
    low_fit_history_trigger: bool
    low_margin_trigger: bool
    alternatives_or_ties_trigger: bool

    def active_names(self) -> list[str]:
        """Return names of all triggers that are True."""
        return [n for n in TRIGGER_NAMES if getattr(self, n)]

    def count_active(self) -> int:
        return sum(getattr(self, n) for n in TRIGGER_NAMES)


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
    # Per-signal trigger flags
    recent_revocations_trigger: bool
    recent_detected_drift_trigger: bool
    consecutive_sentinel_failure_trigger: bool
    open_novelty_trigger: bool
    passive_stress_trigger: bool
    low_strong_observations_trigger: bool
    low_sentinel_count_trigger: bool
    low_fit_history_trigger: bool
    low_margin_trigger: bool
    alternatives_or_ties_trigger: bool


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
    evidence summarizer. Never reads truth_source_edges, truth_func, or any
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


def extract_evidence_triggers(item: dict[str, Any]) -> EvidenceTriggers:
    """Extract all observable contradiction and low-evidence signal flags from an item.

    Each flag reflects exactly one observable evidence condition. Never reads
    truth_source_edges, truth_func, or any hidden-world field.
    """
    return EvidenceTriggers(
        recent_revocations_trigger=_as_int(item.get("recent_revocations")) >= 2,
        recent_detected_drift_trigger=_as_int(item.get("recent_detected_drift")) >= 2,
        consecutive_sentinel_failure_trigger=_as_int(item.get("consecutive_sentinel_failures")) > 0,
        open_novelty_trigger=_as_int(item.get("open_novelty_observations")) > 0,
        passive_stress_trigger=(
            item.get("passive_stress_recent") is not None
            and _as_int(item.get("passive_stress_recent")) > 0
        ),
        low_strong_observations_trigger=(
            "strong_observations" in item
            and _as_int(item.get("strong_observations")) <= 0
        ),
        low_sentinel_count_trigger=(
            "sentinel_count" in item
            and _as_int(item.get("sentinel_count")) <= 0
        ),
        low_fit_history_trigger=(
            "fit_history_count" in item
            and _as_int(item.get("fit_history_count")) <= 0
        ),
        low_margin_trigger=(
            item.get("last_fit_margin") is not None
            and _as_int(item.get("last_fit_margin")) <= 0
        ),
        alternatives_or_ties_trigger=(
            bool(item.get("alternatives_existed"))
            or _as_int(item.get("last_fit_tie_count")) > 1
            or _as_int(item.get("last_fit_near_tie_count")) > 1
        ),
    )


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

    trig = extract_evidence_triggers(item)

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
        recent_revocations_trigger=trig.recent_revocations_trigger,
        recent_detected_drift_trigger=trig.recent_detected_drift_trigger,
        consecutive_sentinel_failure_trigger=trig.consecutive_sentinel_failure_trigger,
        open_novelty_trigger=trig.open_novelty_trigger,
        passive_stress_trigger=trig.passive_stress_trigger,
        low_strong_observations_trigger=trig.low_strong_observations_trigger,
        low_sentinel_count_trigger=trig.low_sentinel_count_trigger,
        low_fit_history_trigger=trig.low_fit_history_trigger,
        low_margin_trigger=trig.low_margin_trigger,
        alternatives_or_ties_trigger=trig.alternatives_or_ties_trigger,
    )
