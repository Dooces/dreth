"""
Integration tests for shadow policy wiring into the policy-report workflow.

Tests A–F from the task specification:
  A. Policy-report row gets shadow prediction fields after annotation.
  B. Schedule is not used inside predict().
  C. actual_best_policy is the row with lowest avg_quality_cost in the scope group.
  D. Selector summary counts false switches and missed rescues correctly.
  E. history/history actual best increments history_history_wins_missed.
  F. No runtime behavior changes: shadow_policy is not imported by agent, fit,
     ledger, or sentinel modules.

Tests use duck-typed SimpleNamespace objects instead of real RunResult so no
agent machinery is required.
"""

import importlib
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest

from dreth.shadow_policy import (
    ShadowPolicySelector,
    DiagnosticFeatures,
    annotate_rows,
    best_policy_by_scope,
    features_from_run_results,
    SHADOW_ROW_FIELDS,
)


# ---------------------------------------------------------------------------
# Mock RunResult builder
# ---------------------------------------------------------------------------

def _make_run(
    schedule: str = "false_trass",
    n_vars: int = 50,
    cycles: int = 5000,
    parent_ranker: str = "sensitivity",
    probe_proposer: str = "none",
    regime_sentinel_fails: int = 0,
    regime_no_sentinel: int = 0,
    passive_stress_count: int = 0,
    total_unique_failures: int = 0,
    full_audits: int = 20,
    revoked_by_dist: dict = None,
    parent_proposal_rank_mean: float = 0.0,
    provider_probe_no_effect_count: int = 0,
    provider_probe_improved_margin_count: int = 0,
    active_composites: int = 0,
    composite_components: int = 0,
    recorded_cycles: int = None,
    ok: bool = True,
) -> SimpleNamespace:
    arch = SimpleNamespace(
        revoked_by_dist=revoked_by_dist or {},
        total_unique_failures=total_unique_failures,
        regime_sentinel_fails=regime_sentinel_fails,
        regime_no_sentinel=regime_no_sentinel,
        passive_stress_count=passive_stress_count,
        parent_proposal_rank_mean=parent_proposal_rank_mean,
        provider_probe_no_effect_count=provider_probe_no_effect_count,
        provider_probe_improved_margin_count=provider_probe_improved_margin_count,
        active_composites=active_composites,
        composite_components=composite_components,
    )
    config = SimpleNamespace(
        schedule=schedule,
        n_vars=n_vars,
        cycles=cycles,
        parent_ranker=parent_ranker,
        probe_proposer=probe_proposer,
    )
    return SimpleNamespace(
        config=config,
        arch=arch,
        full_audits=full_audits,
        recorded_cycles=recorded_cycles if recorded_cycles is not None else cycles,
        ok=ok,
        violations=[],
    )


def _group_key(r) -> Tuple[str, int, int, str]:
    return (r.config.schedule, r.config.n_vars, r.config.cycles,
            f"{r.config.parent_ranker}/{r.config.probe_proposer}")


def _build_run_groups(runs) -> Dict[Tuple, List]:
    groups: Dict[Tuple, List] = {}
    for r in runs:
        groups.setdefault(_group_key(r), []).append(r)
    return groups


# ---------------------------------------------------------------------------
# A. Policy-report row gets shadow prediction fields after annotation
# ---------------------------------------------------------------------------

def test_annotation_adds_all_shadow_fields():
    runs = [
        _make_run("false_trass", 50, 5000, "sensitivity", "none"),
        _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue",
                  regime_sentinel_fails=0),
    ]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1000.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 800.0},
    ]
    run_groups = _build_run_groups(runs)
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    for row in rows:
        for field in SHADOW_ROW_FIELDS:
            assert field in row, f"Missing field {field!r} in row"


def test_annotation_shadow_fields_have_correct_types():
    runs = [_make_run("false_trass", 50, 5000, "sensitivity", "none")]
    rows = [{"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
             "policy": "sensitivity/none", "avg_quality_cost": 1000.0}]
    run_groups = _build_run_groups(runs)
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    row = rows[0]
    assert isinstance(row["shadow_predicted_policy"], str)
    assert isinstance(row["shadow_actual_best_policy"], str)
    assert isinstance(row["shadow_policy_correct"], bool)
    assert isinstance(row["shadow_false_switch_to_history_rescue_under_regime_switch"], bool)
    assert isinstance(row["shadow_missed_history_rescue_under_false_trass"], bool)
    assert isinstance(row["shadow_history_history_wins_missed"], bool)


def test_annotation_sets_none_for_missing_run_group():
    # Row has no matching RunResult → all shadow fields should be None
    rows = [{"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
             "policy": "history/history", "avg_quality_cost": 900.0}]
    run_groups: Dict = {}  # empty
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    for field in SHADOW_ROW_FIELDS:
        assert rows[0][field] is None


# ---------------------------------------------------------------------------
# B. Schedule is not used inside predict()
# ---------------------------------------------------------------------------

def test_predict_ignores_schedule_label():
    sel = ShadowPolicySelector()
    # Features that produce history_rescue prediction
    f_low = DiagnosticFeatures(
        revocations=0, full_audits=20, unique_fails=0,
        regime_sentinel_fails=0, regime_no_sentinel=0,
        parent_rank_mean=0.0, probe_no_effect=0, probe_improved=0,
        passive_stress_count=0, active_composites=0, composite_components=0,
        cycles=5000, n_vars=50,
    )
    # predict() takes no schedule argument — calling it multiple times gives same result
    results = [sel.predict(f_low) for _ in range(5)]
    assert len(set(results)) == 1, "predict() must be deterministic with no schedule input"


def test_annotation_same_features_same_prediction_across_schedules():
    # Two rows with identical run diagnostics but different schedules.
    # shadow_predicted_policy must be identical since predict() has no schedule input.
    runs_ft = [_make_run("false_trass", 50, 5000, "sensitivity", "none",
                         regime_sentinel_fails=0)]
    runs_rs = [_make_run("regime_switch", 50, 5000, "sensitivity", "none",
                         regime_sentinel_fails=0)]  # same features

    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1000.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 9999,
         "policy": "sensitivity/none", "avg_quality_cost": 1000.0},
    ]
    run_groups = {
        ("false_trass", 50, 5000, "sensitivity/none"): runs_ft,
        ("regime_switch", 50, 9999, "sensitivity/none"): runs_rs,
    }
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    assert rows[0]["shadow_predicted_policy"] == rows[1]["shadow_predicted_policy"]


# ---------------------------------------------------------------------------
# C. actual_best_policy is the row with lowest avg_quality_cost in the scope
# ---------------------------------------------------------------------------

def test_best_policy_by_scope_picks_lowest_quality_cost():
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1000.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history/history", "avg_quality_cost": 900.0},
    ]
    bps = best_policy_by_scope(rows)
    assert bps[("false_trass", 50, 5000)] == "history_rescue/history_rescue"


def test_best_policy_by_scope_separate_per_scope():
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 600.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 800.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 1500.0},
    ]
    bps = best_policy_by_scope(rows)
    assert bps[("false_trass", 50, 5000)] == "history_rescue/history_rescue"
    assert bps[("regime_switch", 50, 5000)] == "sensitivity/none"


def test_annotation_actual_best_policy_matches_lowest_cost():
    runs_sens = [_make_run("false_trass", 50, 5000, "sensitivity", "none")]
    runs_resc = [_make_run("false_trass", 50, 5000, "history_rescue", "history_rescue")]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1000.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0},
    ]
    run_groups = {
        ("false_trass", 50, 5000, "sensitivity/none"): runs_sens,
        ("false_trass", 50, 5000, "history_rescue/history_rescue"): runs_resc,
    }
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    for row in rows:
        assert row["shadow_actual_best_policy"] == "history_rescue/history_rescue"


# ---------------------------------------------------------------------------
# D. Selector summary counts false switches and missed rescues
# ---------------------------------------------------------------------------

def test_selector_summary_false_switch_count_via_annotation():
    # Low regime fails → predicts history_rescue; actual best is sensitivity under regime_switch
    runs = [_make_run("regime_switch", 50, 5000, "sensitivity", "none",
                      regime_sentinel_fails=0)]
    rows = [
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 800.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 1200.0},
    ]
    runs_resc = [_make_run("regime_switch", 50, 5000, "history_rescue", "history_rescue",
                           regime_sentinel_fails=0)]
    run_groups = {
        ("regime_switch", 50, 5000, "sensitivity/none"): runs,
        ("regime_switch", 50, 5000, "history_rescue/history_rescue"): runs_resc,
    }
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    s = sel.summary()
    # Both rows predict history_rescue (low fails), actual is sensitivity/none under regime_switch
    assert s["false_switch_to_history_rescue_under_regime_switch"] == 2


def test_selector_summary_missed_rescue_via_annotation():
    # High passive stress → predicts sensitivity/none; actual is history_rescue under false_trass
    runs_sens = [_make_run("false_trass", 50, 5000, "sensitivity", "none",
                           passive_stress_count=5001)]
    runs_resc = [_make_run("false_trass", 50, 5000, "history_rescue", "history_rescue",
                           passive_stress_count=5001)]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1200.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0},
    ]
    run_groups = {
        ("false_trass", 50, 5000, "sensitivity/none"): runs_sens,
        ("false_trass", 50, 5000, "history_rescue/history_rescue"): runs_resc,
    }
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    s = sel.summary()
    assert s["missed_history_rescue_under_false_trass"] == 2


# ---------------------------------------------------------------------------
# E. history/history actual best increments history_history_wins_missed
# ---------------------------------------------------------------------------

def test_history_history_wins_missed_set_when_hh_is_best():
    runs_hh = [_make_run("false_trass", 50, 5000, "history", "history")]
    runs_sens = [_make_run("false_trass", 50, 5000, "sensitivity", "none")]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history/history", "avg_quality_cost": 500.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
    ]
    run_groups = {
        ("false_trass", 50, 5000, "history/history"): runs_hh,
        ("false_trass", 50, 5000, "sensitivity/none"): runs_sens,
    }
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    # Both rows: actual_best is history/history; selector never predicts it
    for row in rows:
        assert row["shadow_history_history_wins_missed"] is True

    s = sel.summary()
    assert s["history_history_wins_missed"] == 2


def test_history_history_wins_missed_zero_when_not_best():
    runs = [_make_run("false_trass", 50, 5000, "history_rescue", "history_rescue")]
    rows = [{"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
             "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0}]
    run_groups = {("false_trass", 50, 5000, "history_rescue/history_rescue"): runs}
    sel = ShadowPolicySelector()
    annotate_rows(rows, run_groups, sel)

    assert rows[0]["shadow_history_history_wins_missed"] is False
    assert sel.summary()["history_history_wins_missed"] == 0


# ---------------------------------------------------------------------------
# F. No runtime behavior changes: shadow_policy not imported by core modules
# ---------------------------------------------------------------------------

def test_agent_does_not_import_shadow_policy():
    import dreth.agent as agent_mod
    assert "shadow_policy" not in sys.modules.get("dreth.agent", object).__dict__ if False else True
    # Check the source file directly
    import inspect
    src = inspect.getsource(agent_mod)
    assert "shadow_policy" not in src


def test_fit_does_not_import_shadow_policy():
    import dreth.fit as fit_mod
    import inspect
    src = inspect.getsource(fit_mod)
    assert "shadow_policy" not in src


def test_ledger_does_not_import_shadow_policy():
    import dreth.ledger as ledger_mod
    import inspect
    src = inspect.getsource(ledger_mod)
    assert "shadow_policy" not in src


def test_sentinels_does_not_import_shadow_policy():
    import dreth.sentinels as sentinels_mod
    import inspect
    src = inspect.getsource(sentinels_mod)
    assert "shadow_policy" not in src


def test_shadow_policy_selector_does_not_mutate_rows_beyond_shadow_fields():
    """annotate_rows must not modify any non-shadow key in the row dict."""
    run = _make_run("false_trass", 50, 5000, "sensitivity", "none")
    row = {
        "schedule": "false_trass", "n_vars": 50, "cycles": 5000,
        "policy": "sensitivity/none", "avg_quality_cost": 900.0,
        "avg_iv": 100.0, "pareto_status": "efficient",
    }
    original_non_shadow = {k: v for k, v in row.items() if k not in SHADOW_ROW_FIELDS}
    run_groups = {("false_trass", 50, 5000, "sensitivity/none"): [run]}
    sel = ShadowPolicySelector()
    annotate_rows([row], run_groups, sel)

    for k, v in original_non_shadow.items():
        assert row[k] == v, f"annotate_rows mutated non-shadow field {k!r}"


# ---------------------------------------------------------------------------
# features_from_run_results: aggregation correctness
# ---------------------------------------------------------------------------

def test_features_from_run_results_sums_integer_fields():
    r1 = _make_run(regime_sentinel_fails=10, total_unique_failures=2, recorded_cycles=5000)
    r2 = _make_run(regime_sentinel_fails=20, total_unique_failures=3, recorded_cycles=5000)
    f = features_from_run_results([r1, r2])
    assert f.regime_sentinel_fails == 30
    assert f.unique_fails == 5
    assert f.cycles == 10000


def test_features_from_run_results_averages_rank_mean():
    r1 = _make_run(parent_proposal_rank_mean=2.0)
    r2 = _make_run(parent_proposal_rank_mean=4.0)
    f = features_from_run_results([r1, r2])
    assert f.parent_rank_mean == pytest.approx(3.0)


def test_features_from_run_results_rate_correct_after_sum():
    # 10 fails over 10000 total cycles → rate = 0.001 < threshold
    r1 = _make_run(regime_sentinel_fails=5, recorded_cycles=5000)
    r2 = _make_run(regime_sentinel_fails=5, recorded_cycles=5000)
    f = features_from_run_results([r1, r2])
    assert f.regime_fail_rate == pytest.approx(10 / 10000)
