from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.shadow_authority_throttle import (
    classify_visible_authority_evidence,
    would_throttle_authority,
)


def _item(**overrides):
    base = {
        "var": 0,
        "authoritative": True,
        "status": "certified",
        "skip_role": "tareth",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# classify_visible_authority_evidence
# ---------------------------------------------------------------------------


def test_classification_ignores_truth_fields() -> None:
    item = _item(
        truth_parents=[1, 2, 3],
        truth_func="HIDDEN_FUNC",
        truth_delayed_parents=[4],
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
        repeatedly_stable_under_probes=True,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    item_no_truth = {
        k: v
        for k, v in item.items()
        if k not in ("truth_parents", "truth_func", "truth_delayed_parents")
    }
    assert (
        classify_visible_authority_evidence(item)
        == classify_visible_authority_evidence(item_no_truth)
    )


# ---------------------------------------------------------------------------
# contradicted_authority
# ---------------------------------------------------------------------------


def test_contradicted_authority_throttles_conservative() -> None:
    item = _item(
        strong_observations=5,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        recent_revocations=2,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    assert classify_visible_authority_evidence(item) == "contradicted_authority"
    decision = would_throttle_authority(item, mode="conservative")
    assert decision.would_throttle
    assert decision.reason == "contradicted_authority"
    assert decision.evidence_class == "contradicted_authority"


def test_contradicted_authority_throttles_strict() -> None:
    item = _item(
        strong_observations=5,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        recent_revocations=2,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    decision = would_throttle_authority(item, mode="strict")
    assert decision.would_throttle
    assert decision.reason == "contradicted_authority"


# ---------------------------------------------------------------------------
# insufficient_evidence
# ---------------------------------------------------------------------------


def test_insufficient_evidence_throttles_conservative_authority_strong() -> None:
    item = _item(
        authoritative=True,
        strong_observations=0,
        sentinel_count=2,
        fit_history_count=2,
        last_fit_margin=3,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    assert classify_visible_authority_evidence(item) == "insufficient_evidence"
    decision = would_throttle_authority(item, mode="conservative")
    assert decision.would_throttle
    assert decision.reason == "insufficient_evidence"


def test_insufficient_evidence_no_throttle_conservative_authority_not_strong() -> None:
    item = _item(
        authoritative=False,
        strong_observations=0,
        sentinel_count=2,
        fit_history_count=2,
        last_fit_margin=3,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    assert classify_visible_authority_evidence(item) == "insufficient_evidence"
    decision = would_throttle_authority(item, mode="conservative")
    assert not decision.would_throttle
    assert decision.evidence_class == "insufficient_evidence"


def test_insufficient_evidence_throttles_strict_regardless_of_authority() -> None:
    item = _item(
        authoritative=False,
        strong_observations=0,
        sentinel_count=2,
        fit_history_count=2,
        last_fit_margin=3,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    assert classify_visible_authority_evidence(item) == "insufficient_evidence"
    decision = would_throttle_authority(item, mode="strict")
    assert decision.would_throttle
    assert decision.reason == "insufficient_evidence"


# ---------------------------------------------------------------------------
# evidence_supported_surrogate — never throttle
# ---------------------------------------------------------------------------


def test_evidence_supported_surrogate_does_not_throttle_conservative() -> None:
    item = _item(
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
        repeatedly_stable_under_probes=True,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    assert classify_visible_authority_evidence(item) == "evidence_supported_surrogate"
    decision = would_throttle_authority(item, mode="conservative")
    assert not decision.would_throttle
    assert decision.reason == "no_throttle"


def test_evidence_supported_surrogate_does_not_throttle_strict() -> None:
    item = _item(
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
        repeatedly_stable_under_probes=True,
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
    )
    decision = would_throttle_authority(item, mode="strict")
    assert not decision.would_throttle


# ---------------------------------------------------------------------------
# weakly_supported_surrogate
# ---------------------------------------------------------------------------


def _weak_item(**overrides):
    # Base weakly-supported item: some positive evidence, no stable support.
    # fit_history_count intentionally absent to avoid triggering _has_low_evidence.
    base = {
        "var": 1,
        "authoritative": True,
        "strong_observations": 2,
        "sentinel_count": 2,
        "last_fit_margin": 5,
        "last_fit_tie_count": 1,
        "last_fit_near_tie_count": 1,
        "alternatives_existed": False,
        "repeatedly_stable_under_probes": False,
        "recent_revocations": 0,
        "recent_detected_drift": 0,
        "consecutive_sentinel_failures": 0,
        "open_novelty_observations": 0,
    }
    base.update(overrides)
    return base


def test_weakly_supported_does_not_throttle_conservative() -> None:
    item = _weak_item()
    assert classify_visible_authority_evidence(item) == "weakly_supported_surrogate"
    decision = would_throttle_authority(item, mode="conservative")
    assert not decision.would_throttle
    assert decision.evidence_class == "weakly_supported_surrogate"


def test_weakly_supported_does_not_throttle_strict_without_trigger() -> None:
    # Strict mode: weakly_supported only throttles when alternatives/ties present.
    item = _weak_item(alternatives_existed=False, last_fit_tie_count=1, last_fit_near_tie_count=1)
    assert classify_visible_authority_evidence(item) == "weakly_supported_surrogate"
    decision = would_throttle_authority(item, mode="strict")
    assert not decision.would_throttle


def test_weakly_supported_throttles_strict_with_alternatives() -> None:
    item = _weak_item(alternatives_existed=True)
    assert classify_visible_authority_evidence(item) == "weakly_supported_surrogate"
    decision = would_throttle_authority(item, mode="strict")
    assert decision.would_throttle
    assert decision.reason == "weakly_supported_strict"


def test_weakly_supported_throttles_strict_with_tie_count() -> None:
    item = _weak_item(last_fit_tie_count=2)
    assert classify_visible_authority_evidence(item) == "weakly_supported_surrogate"
    decision = would_throttle_authority(item, mode="strict")
    assert decision.would_throttle
    assert decision.reason == "weakly_supported_strict"


def test_weakly_supported_throttles_strict_with_near_tie_count() -> None:
    item = _weak_item(last_fit_near_tie_count=2)
    assert classify_visible_authority_evidence(item) == "weakly_supported_surrogate"
    decision = would_throttle_authority(item, mode="strict")
    assert decision.would_throttle
    assert decision.reason == "weakly_supported_strict"


# ---------------------------------------------------------------------------
# AuthorityThrottleDecision fields
# ---------------------------------------------------------------------------


def test_decision_fields_populated() -> None:
    item = _item(
        var=7,
        strong_observations=3,
        sentinel_count=2,
        fit_history_count=2,
        last_fit_margin=4,
        recent_revocations=2,
        recent_detected_drift=0,
        consecutive_sentinel_failures=1,
        open_novelty_observations=0,
        alternatives_existed=False,
        last_fit_tie_count=1,
        last_fit_near_tie_count=0,
    )
    decision = would_throttle_authority(item)
    assert decision.var == 7
    assert decision.strong_observations == 3
    assert decision.sentinel_count == 2
    assert decision.fit_history_count == 2
    assert decision.last_fit_margin == 4
    assert decision.recent_revocations == 2
    assert decision.consecutive_sentinel_failures == 1
    assert decision.open_novelty is False
    assert decision.alternatives_existed is False
    assert decision.tie_count == 1
    assert decision.near_tie_count == 0


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown throttle mode"):
        would_throttle_authority(_item(), mode="aggressive")
