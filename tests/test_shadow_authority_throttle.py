from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.shadow_authority_throttle import (
    TRIGGER_NAMES,
    EvidenceTriggers,
    classify_visible_authority_evidence,
    extract_evidence_triggers,
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
        truth_source_edges=[1, 2, 3],
        truth_func="HIDDEN_FUNC",
        truth_delayed_source_edges=[4],
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
        if k not in ("truth_source_edges", "truth_func", "truth_delayed_source_edges")
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


# ---------------------------------------------------------------------------
# EvidenceTriggers / extract_evidence_triggers
# ---------------------------------------------------------------------------


def test_extract_evidence_triggers_passive_stress_only() -> None:
    # rev=0, drift=0, but passive_stress_recent > 0: only passive_stress fires.
    item = _item(
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
        passive_stress_recent=3,
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
    )
    trig = extract_evidence_triggers(item)

    assert trig.passive_stress_trigger is True
    assert trig.recent_revocations_trigger is False
    assert trig.recent_detected_drift_trigger is False
    assert trig.consecutive_sentinel_failure_trigger is False
    assert trig.open_novelty_trigger is False
    assert trig.low_strong_observations_trigger is False
    assert trig.low_sentinel_count_trigger is False
    assert trig.low_fit_history_trigger is False
    assert trig.low_margin_trigger is False
    assert trig.alternatives_or_ties_trigger is False

    assert trig.active_names() == ["passive_stress_trigger"]
    assert trig.count_active() == 1

    # Item is contradicted_authority due to passive stress alone.
    assert classify_visible_authority_evidence(item) == "contradicted_authority"


def test_extract_evidence_triggers_all_clear() -> None:
    item = _item(
        recent_revocations=0,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
        repeatedly_stable_under_probes=True,
    )
    trig = extract_evidence_triggers(item)

    assert trig.active_names() == []
    assert trig.count_active() == 0


def test_extract_evidence_triggers_multiple_contradiction_signals() -> None:
    item = _item(
        recent_revocations=2,
        recent_detected_drift=2,
        consecutive_sentinel_failures=1,
        open_novelty_observations=1,
        passive_stress_recent=1,
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
    )
    trig = extract_evidence_triggers(item)

    assert trig.recent_revocations_trigger is True
    assert trig.recent_detected_drift_trigger is True
    assert trig.consecutive_sentinel_failure_trigger is True
    assert trig.open_novelty_trigger is True
    assert trig.passive_stress_trigger is True
    assert trig.count_active() == 5


def test_extract_evidence_triggers_low_evidence_fields() -> None:
    item = _item(
        recent_revocations=0,
        recent_detected_drift=0,
        strong_observations=0,
        sentinel_count=0,
        fit_history_count=0,
        last_fit_margin=0,
    )
    trig = extract_evidence_triggers(item)

    assert trig.low_strong_observations_trigger is True
    assert trig.low_sentinel_count_trigger is True
    assert trig.low_fit_history_trigger is True
    assert trig.low_margin_trigger is True


def test_extract_evidence_triggers_alternatives_or_ties() -> None:
    item_alt = _item(alternatives_existed=True)
    item_tie = _item(last_fit_tie_count=2)
    item_near = _item(last_fit_near_tie_count=2)

    assert extract_evidence_triggers(item_alt).alternatives_or_ties_trigger is True
    assert extract_evidence_triggers(item_tie).alternatives_or_ties_trigger is True
    assert extract_evidence_triggers(item_near).alternatives_or_ties_trigger is True

    item_none = _item(
        alternatives_existed=False, last_fit_tie_count=1, last_fit_near_tie_count=1
    )
    assert extract_evidence_triggers(item_none).alternatives_or_ties_trigger is False


def test_extract_evidence_triggers_ignores_truth_fields() -> None:
    item = _item(
        truth_source_edges=[1, 2],
        truth_func="HIDDEN",
        truth_delayed_source_edges=[3],
        recent_revocations=0,
        recent_detected_drift=0,
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
    )
    item_no_truth = {
        k: v
        for k, v in item.items()
        if k not in ("truth_source_edges", "truth_func", "truth_delayed_source_edges")
    }
    assert extract_evidence_triggers(item).active_names() == extract_evidence_triggers(item_no_truth).active_names()


def test_decision_trigger_fields_populated() -> None:
    # Item with recent_revocations=2 only — only rev trigger fires.
    item = _item(
        var=3,
        recent_revocations=2,
        recent_detected_drift=0,
        consecutive_sentinel_failures=0,
        open_novelty_observations=0,
        strong_observations=5,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
    )
    decision = would_throttle_authority(item)

    assert decision.recent_revocations_trigger is True
    assert decision.recent_detected_drift_trigger is False
    assert decision.consecutive_sentinel_failure_trigger is False
    assert decision.open_novelty_trigger is False
    assert decision.passive_stress_trigger is False
    assert decision.low_strong_observations_trigger is False
    assert decision.low_sentinel_count_trigger is False
    assert decision.low_fit_history_trigger is False
    assert decision.low_margin_trigger is False
    assert decision.alternatives_or_ties_trigger is False


def test_evidence_triggers_type_exported() -> None:
    assert EvidenceTriggers is not None
    assert TRIGGER_NAMES is not None
    assert len(TRIGGER_NAMES) == 10
