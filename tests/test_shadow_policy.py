"""
Tests for dreth.shadow_policy.ShadowPolicySelector.

Uses synthetic DiagnosticFeatures; no RunResult or agent machinery needed.

Test coverage:
  - predict: high regime_fail_rate → sensitivity/none
  - predict: low regime_fail_rate → history_rescue/history_rescue
  - predict: high no_sentinel_rate → sensitivity/none
  - predict: high passive_stress_rate (with zero regime fails) → sensitivity/none
  - predict: high unique_fail_rate at large n_vars → sensitivity/none
  - predict: low everything → history_rescue
  - observe: correct flag set correctly
  - observe: false_switch_to_history_rescue_under_regime_switch
  - observe: missed_history_rescue_under_false_trass
  - observe: schedule=None does not set either error-mode flag
  - summary: accuracy calculation
  - summary: error-mode counts
  - summary: empty returns empty dict
  - log_summary: outputs accuracy and both error-mode fields
  - log_summary: empty case
  - unknown actual_best_policy raises ValueError
"""

import io
import pytest

from dreth.shadow_policy import (
    DiagnosticFeatures,
    ShadowPolicySelector,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _feats(
    *,
    regime_sentinel_fails: int = 0,
    regime_no_sentinel: int = 0,
    passive_stress_count: int = 0,
    unique_fails: int = 0,
    n_vars: int = 50,
    cycles: int = 5000,
    revocations: int = 0,
    full_audits: int = 20,
    parent_rank_mean: float = 0.0,
    probe_no_effect: int = 0,
    probe_improved: int = 0,
    active_composites: int = 0,
    composite_components: int = 0,
) -> DiagnosticFeatures:
    return DiagnosticFeatures(
        revocations=revocations,
        full_audits=full_audits,
        unique_fails=unique_fails,
        regime_sentinel_fails=regime_sentinel_fails,
        regime_no_sentinel=regime_no_sentinel,
        parent_rank_mean=parent_rank_mean,
        probe_no_effect=probe_no_effect,
        probe_improved=probe_improved,
        passive_stress_count=passive_stress_count,
        active_composites=active_composites,
        composite_components=composite_components,
        cycles=cycles,
        n_vars=n_vars,
    )


# ---------------------------------------------------------------------------
# DiagnosticFeatures properties
# ---------------------------------------------------------------------------

def test_regime_fail_rate():
    f = _feats(regime_sentinel_fails=500, cycles=5000)
    assert f.regime_fail_rate == pytest.approx(0.1)


def test_regime_fail_rate_zero():
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    assert f.regime_fail_rate == pytest.approx(0.0)


def test_unique_fail_rate():
    f = _feats(unique_fails=100, cycles=5000, n_vars=50)
    assert f.unique_fail_rate == pytest.approx(100 / (5000 * 50))


def test_no_sentinel_rate():
    f = _feats(regime_no_sentinel=2500, cycles=5000)
    assert f.no_sentinel_rate == pytest.approx(0.5)


def test_passive_stress_rate():
    f = _feats(passive_stress_count=1000, cycles=5000, n_vars=50)
    assert f.passive_stress_rate == pytest.approx(1000 / (5000 * 50))


def test_probe_no_effect_rate_zero_total():
    f = _feats(probe_no_effect=0, probe_improved=0)
    assert f.probe_no_effect_rate == pytest.approx(0.0)


def test_probe_no_effect_rate():
    f = _feats(probe_no_effect=3, probe_improved=1)
    assert f.probe_no_effect_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# predict: high regime fail → sensitivity/none
# ---------------------------------------------------------------------------

def test_predict_high_regime_fail_returns_sensitivity():
    sel = ShadowPolicySelector()
    # 500 fails / 5000 cycles = 0.1 >> threshold
    f = _feats(regime_sentinel_fails=500, cycles=5000)
    assert sel.predict(f) == "sensitivity/none"


def test_predict_regime_fail_just_above_threshold():
    sel = ShadowPolicySelector()
    # 0.01 * 5000 = 50 fails → just at boundary; 51 → above
    f = _feats(regime_sentinel_fails=51, cycles=5000)
    assert sel.predict(f) == "sensitivity/none"


def test_predict_regime_fail_just_below_threshold():
    sel = ShadowPolicySelector()
    # 49/5000 = 0.0098 < 0.01
    f = _feats(regime_sentinel_fails=49, cycles=5000)
    assert sel.predict(f) == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# predict: high no_sentinel_rate → sensitivity/none
# ---------------------------------------------------------------------------

def test_predict_high_no_sentinel_rate():
    sel = ShadowPolicySelector()
    # no regime fails but sentinel missing >50% of cycles
    f = _feats(regime_sentinel_fails=0, regime_no_sentinel=3000, cycles=5000)
    assert sel.predict(f) == "sensitivity/none"


def test_predict_low_no_sentinel_rate_does_not_override():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, regime_no_sentinel=100, cycles=5000)
    assert sel.predict(f) == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# predict: high passive_stress_rate → sensitivity/none
# ---------------------------------------------------------------------------

def test_predict_high_passive_stress_zero_regime_fails():
    sel = ShadowPolicySelector()
    # 0.02 * 5000 * 50 = 5000 stress events → stress rate = 0.02
    f = _feats(
        regime_sentinel_fails=0,
        passive_stress_count=5001,
        cycles=5000,
        n_vars=50,
    )
    assert sel.predict(f) == "sensitivity/none"


def test_predict_moderate_passive_stress_does_not_trigger():
    sel = ShadowPolicySelector()
    # Below the 0.02 threshold: 4999 / (5000 * 50) = 0.01999
    f = _feats(
        regime_sentinel_fails=0,
        passive_stress_count=4999,
        cycles=5000,
        n_vars=50,
    )
    assert sel.predict(f) == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# predict: high unique fail rate at large n_vars → sensitivity/none
# ---------------------------------------------------------------------------

def test_predict_high_unique_fail_large_n_vars():
    sel = ShadowPolicySelector()
    # unique_fail_rate = 30 / (5000 * 100) = 6e-5 < threshold, need higher
    # 5e-4 * 5000 * 100 = 250 unique fails → exactly at threshold
    # use 251 to go above
    f = _feats(unique_fails=251, cycles=5000, n_vars=100, regime_sentinel_fails=0)
    assert sel.predict(f) == "sensitivity/none"


def test_predict_high_unique_fail_small_n_vars_does_not_trigger():
    sel = ShadowPolicySelector()
    # Same rate but n_vars < 100 → rule doesn't apply
    f = _feats(unique_fails=251, cycles=5000, n_vars=75, regime_sentinel_fails=0)
    assert sel.predict(f) == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# predict: clean false_trass-like → history_rescue
# ---------------------------------------------------------------------------

def test_predict_false_trass_like_returns_history_rescue():
    sel = ShadowPolicySelector()
    f = _feats(
        regime_sentinel_fails=0,
        regime_no_sentinel=0,
        passive_stress_count=0,
        unique_fails=0,
        n_vars=50,
        cycles=5000,
    )
    assert sel.predict(f) == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# observe: correct flag
# ---------------------------------------------------------------------------

def test_observe_correct_when_prediction_matches():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    p = sel.observe(f, actual_best_policy="history_rescue/history_rescue")
    assert p.correct is True
    assert p.predicted_policy == "history_rescue/history_rescue"


def test_observe_incorrect_when_prediction_differs():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    # Prediction will be history_rescue but actual is sensitivity/none
    p = sel.observe(f, actual_best_policy="sensitivity/none")
    assert p.correct is False


# ---------------------------------------------------------------------------
# observe: false_switch error mode
# ---------------------------------------------------------------------------

def test_false_switch_to_history_rescue_under_regime_switch():
    sel = ShadowPolicySelector()
    # Prediction: history_rescue (low fails), schedule: regime_switch, actual: sensitivity/none
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    p = sel.observe(
        f,
        actual_best_policy="sensitivity/none",
        schedule="regime_switch",
    )
    assert p.false_switch_to_history_rescue_under_regime_switch is True
    assert p.missed_history_rescue_under_false_trass is False


def test_no_false_switch_when_prediction_correct_under_regime_switch():
    sel = ShadowPolicySelector()
    # High fails → predicts sensitivity/none; actual is sensitivity/none → correct, no false switch
    f = _feats(regime_sentinel_fails=500, cycles=5000)
    p = sel.observe(
        f,
        actual_best_policy="sensitivity/none",
        schedule="regime_switch",
    )
    assert p.false_switch_to_history_rescue_under_regime_switch is False
    assert p.correct is True


# ---------------------------------------------------------------------------
# observe: missed_history_rescue error mode
# ---------------------------------------------------------------------------

def test_missed_history_rescue_under_false_trass():
    sel = ShadowPolicySelector()
    # Prediction: sensitivity/none (e.g. from high passive stress), actual: history_rescue
    f = _feats(
        regime_sentinel_fails=0,
        passive_stress_count=5001,
        cycles=5000,
        n_vars=50,
    )
    p = sel.observe(
        f,
        actual_best_policy="history_rescue/history_rescue",
        schedule="false_trass",
    )
    assert p.missed_history_rescue_under_false_trass is True
    assert p.false_switch_to_history_rescue_under_regime_switch is False


def test_no_missed_rescue_when_prediction_correct():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    p = sel.observe(
        f,
        actual_best_policy="history_rescue/history_rescue",
        schedule="false_trass",
    )
    assert p.missed_history_rescue_under_false_trass is False
    assert p.correct is True


# ---------------------------------------------------------------------------
# observe: schedule=None does not set error modes
# ---------------------------------------------------------------------------

def test_no_error_modes_without_schedule():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    p = sel.observe(f, actual_best_policy="sensitivity/none", schedule=None)
    assert p.false_switch_to_history_rescue_under_regime_switch is False
    assert p.missed_history_rescue_under_false_trass is False


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_empty():
    sel = ShadowPolicySelector()
    assert sel.summary() == {}


def test_summary_accuracy_all_correct():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    for _ in range(4):
        sel.observe(f, actual_best_policy="history_rescue/history_rescue")
    s = sel.summary()
    assert s["n_predictions"] == 4
    assert s["accuracy"] == pytest.approx(1.0)


def test_summary_accuracy_mixed():
    sel = ShadowPolicySelector()
    # 2 correct (history_rescue predicted, history_rescue actual)
    f_good = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f_good, actual_best_policy="history_rescue/history_rescue")
    sel.observe(f_good, actual_best_policy="history_rescue/history_rescue")
    # 2 incorrect (history_rescue predicted, sensitivity actual)
    sel.observe(f_good, actual_best_policy="sensitivity/none")
    sel.observe(f_good, actual_best_policy="sensitivity/none")
    s = sel.summary()
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["n_predictions"] == 4


def test_summary_false_switch_count():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f, actual_best_policy="sensitivity/none", schedule="regime_switch")
    sel.observe(f, actual_best_policy="sensitivity/none", schedule="regime_switch")
    sel.observe(f, actual_best_policy="history_rescue/history_rescue", schedule="false_trass")
    s = sel.summary()
    assert s["false_switch_to_history_rescue_under_regime_switch"] == 2
    assert s["missed_history_rescue_under_false_trass"] == 0


def test_summary_missed_rescue_count():
    sel = ShadowPolicySelector()
    # High passive stress → predicts sensitivity/none; actual is history_rescue
    f = _feats(regime_sentinel_fails=0, passive_stress_count=5001, cycles=5000, n_vars=50)
    sel.observe(f, actual_best_policy="history_rescue/history_rescue", schedule="false_trass")
    sel.observe(f, actual_best_policy="history_rescue/history_rescue", schedule="false_trass")
    s = sel.summary()
    assert s["missed_history_rescue_under_false_trass"] == 2
    assert s["false_switch_to_history_rescue_under_regime_switch"] == 0


def test_summary_predicted_policy_counts():
    sel = ShadowPolicySelector()
    f_rescue = _feats(regime_sentinel_fails=0, cycles=5000)
    f_sens = _feats(regime_sentinel_fails=500, cycles=5000)
    sel.observe(f_rescue, actual_best_policy="history_rescue/history_rescue")
    sel.observe(f_rescue, actual_best_policy="history_rescue/history_rescue")
    sel.observe(f_sens, actual_best_policy="sensitivity/none")
    s = sel.summary()
    assert s["predicted_policy"]["history_rescue/history_rescue"] == 2
    assert s["predicted_policy"]["sensitivity/none"] == 1


# ---------------------------------------------------------------------------
# log_summary output
# ---------------------------------------------------------------------------

def test_log_summary_empty():
    sel = ShadowPolicySelector()
    buf = io.StringIO()
    sel.log_summary(file=buf)
    assert "no predictions" in buf.getvalue()


def test_log_summary_contains_required_fields():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f, actual_best_policy="sensitivity/none", schedule="regime_switch")
    sel.observe(f, actual_best_policy="history_rescue/history_rescue", schedule="false_trass")
    buf = io.StringIO()
    sel.log_summary(file=buf)
    out = buf.getvalue()
    assert "accuracy" in out
    assert "false_switch_to_history_rescue_under_regime_switch" in out
    assert "missed_history_rescue_under_false_trass" in out
    assert "predicted_policy" in out
    assert "actual_best_policy" in out


def test_log_summary_accuracy_value():
    sel = ShadowPolicySelector()
    # Prediction is sensitivity/none (high fails), actual is sensitivity/none → all correct
    f = _feats(regime_sentinel_fails=500, cycles=5000)
    for _ in range(3):
        sel.observe(f, actual_best_policy="sensitivity/none")
    buf = io.StringIO()
    sel.log_summary(file=buf)
    assert "1.000" in buf.getvalue()


# ---------------------------------------------------------------------------
# observe: history_history_wins_missed
# ---------------------------------------------------------------------------

def test_observe_history_history_wins_missed_when_actual_is_hh():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    # Selector predicts history_rescue; actual is history/history → miss
    p = sel.observe(f, actual_best_policy="history/history")
    assert p.history_history_wins_missed is True
    assert p.predicted_policy == "history_rescue/history_rescue"


def test_observe_history_history_wins_missed_false_when_actual_not_hh():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    p = sel.observe(f, actual_best_policy="history_rescue/history_rescue")
    assert p.history_history_wins_missed is False


def test_observe_history_history_wins_missed_false_when_sensitivity_actual():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=500, cycles=5000)
    # Predicts sensitivity/none; actual is sensitivity/none
    p = sel.observe(f, actual_best_policy="sensitivity/none")
    assert p.history_history_wins_missed is False


def test_summary_history_history_wins_missed_count():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f, actual_best_policy="history/history")
    sel.observe(f, actual_best_policy="history/history")
    sel.observe(f, actual_best_policy="history_rescue/history_rescue")
    s = sel.summary()
    assert s["history_history_wins_missed"] == 2


def test_summary_history_history_wins_missed_zero_when_none():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f, actual_best_policy="history_rescue/history_rescue")
    s = sel.summary()
    assert s["history_history_wins_missed"] == 0


def test_log_summary_contains_history_history_wins_missed():
    sel = ShadowPolicySelector()
    f = _feats(regime_sentinel_fails=0, cycles=5000)
    sel.observe(f, actual_best_policy="history/history")
    buf = io.StringIO()
    sel.log_summary(file=buf)
    assert "history_history_wins_missed" in buf.getvalue()


# ---------------------------------------------------------------------------
# observe: invalid actual_best_policy
# ---------------------------------------------------------------------------

def test_observe_invalid_policy_raises():
    sel = ShadowPolicySelector()
    f = _feats()
    with pytest.raises(ValueError, match="not in known policies"):
        sel.observe(f, actual_best_policy="made_up/policy")
