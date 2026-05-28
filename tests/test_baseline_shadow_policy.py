"""
Tests for baseline_annotate_rows in dreth.shadow_policy.

The baseline-only shadow selector differs from the full shadow selector:
  - Predicts once per (schedule, n_vars, cycles) scope, not once per policy row.
  - Features come exclusively from the sensitivity/none RunResult group.
  - actual_best_policy still comes from the lowest avg_quality_cost in the scope.
  - schedule is never passed into predict().

Test coverage:
  - baseline_predicts_once_per_scope_not_per_policy_row
  - baseline_features_from_sensitivity_none_runs_only
  - baseline_actual_best_policy_from_lowest_quality_cost
  - baseline_schedule_not_in_predict
  - baseline_annotation_adds_all_fields
  - baseline_all_rows_in_scope_get_same_prediction
  - baseline_none_when_no_sensitivity_none_runs
  - baseline_does_not_mutate_non_baseline_fields
  - baseline_two_scopes_produce_two_predictions
"""

from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest

from dreth.shadow_policy import (
    ShadowPolicySelector,
    BASELINE_SHADOW_ROW_FIELDS,
    baseline_annotate_rows,
)


# ---------------------------------------------------------------------------
# Mock RunResult builder (mirrors test_shadow_policy_integration.py)
# ---------------------------------------------------------------------------

def _make_run(
    schedule: str = "false_trass",
    n_vars: int = 50,
    cycles: int = 5000,
    source_edge_ranker: str = "sensitivity",
    probe_proposer: str = "none",
    regime_sentinel_fails: int = 0,
    regime_no_sentinel: int = 0,
    passive_stress_count: int = 0,
    total_unique_failures: int = 0,
    full_audits: int = 20,
    revoked_by_dist: dict = None,
    source_edge_proposal_rank_mean: float = 0.0,
    provider_probe_no_effect_count: int = 0,
    provider_probe_improved_margin_count: int = 0,
    active_composites: int = 0,
    composite_components: int = 0,
    recorded_cycles: int = None,
) -> SimpleNamespace:
    arch = SimpleNamespace(
        revoked_by_dist=revoked_by_dist or {},
        total_unique_failures=total_unique_failures,
        regime_sentinel_fails=regime_sentinel_fails,
        regime_no_sentinel=regime_no_sentinel,
        passive_stress_count=passive_stress_count,
        source_edge_proposal_rank_mean=source_edge_proposal_rank_mean,
        provider_probe_no_effect_count=provider_probe_no_effect_count,
        provider_probe_improved_margin_count=provider_probe_improved_margin_count,
        active_composites=active_composites,
        composite_components=composite_components,
    )
    config = SimpleNamespace(
        schedule=schedule,
        n_vars=n_vars,
        cycles=cycles,
        source_edge_ranker=source_edge_ranker,
        probe_proposer=probe_proposer,
    )
    return SimpleNamespace(
        config=config,
        arch=arch,
        full_audits=full_audits,
        recorded_cycles=recorded_cycles if recorded_cycles is not None else cycles,
        ok=True,
        violations=[],
    )


def _build_run_groups(runs) -> Dict[Tuple, List]:
    groups: Dict[Tuple, List] = {}
    for r in runs:
        key = (
            r.config.schedule, r.config.n_vars, r.config.cycles,
            f"{r.config.source_edge_ranker}/{r.config.probe_proposer}",
        )
        groups.setdefault(key, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# baseline-only selector predicts once per scope, not once per policy row
# ---------------------------------------------------------------------------

def test_baseline_predicts_once_per_scope_not_per_policy_row():
    """A scope with 3 policy rows must produce exactly 1 prediction from the selector."""
    runs = [
        _make_run("false_trass", 50, 5000, "sensitivity", "none"),
        _make_run("false_trass", 50, 5000, "history", "history"),
        _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue"),
    ]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history/history", "avg_quality_cost": 700.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 800.0},
    ]
    run_groups = _build_run_groups(runs)
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    assert sel.summary()["n_predictions"] == 1


def test_baseline_two_scopes_produce_two_predictions():
    """Two distinct scopes each produce exactly one prediction."""
    runs = [
        _make_run("false_trass", 50, 5000, "sensitivity", "none"),
        _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue"),
        _make_run("regime_switch", 50, 5000, "sensitivity", "none", regime_sentinel_fails=500),
        _make_run("regime_switch", 50, 5000, "history_rescue", "history_rescue"),
    ]
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 600.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 800.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 1500.0},
    ]
    run_groups = _build_run_groups(runs)
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    assert sel.summary()["n_predictions"] == 2


# ---------------------------------------------------------------------------
# features come only from sensitivity/none rows
# ---------------------------------------------------------------------------

def test_baseline_features_from_sensitivity_none_runs_only():
    """Prediction must use sensitivity/none run features, not features from other runs.

    sensitivity/none run has high regime_sentinel_fails → predicts sensitivity/none.
    history_rescue run has zero fails → would predict history_rescue if used instead.
    """
    run_sens = _make_run(
        "false_trass", 50, 5000, "sensitivity", "none",
        regime_sentinel_fails=500,  # fail_rate=0.1 >> threshold → predicts sensitivity/none
    )
    run_rescue = _make_run(
        "false_trass", 50, 5000, "history_rescue", "history_rescue",
        regime_sentinel_fails=0,    # would predict history_rescue/history_rescue if used
    )
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0},
    ]
    run_groups = _build_run_groups([run_sens, run_rescue])
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    for row in rows:
        assert row["baseline_shadow_predicted_policy"] == "sensitivity/none"


# ---------------------------------------------------------------------------
# actual_best_policy still comes from lowest avg_quality_cost in the scope
# ---------------------------------------------------------------------------

def test_baseline_actual_best_policy_from_lowest_quality_cost():
    """actual_best_policy must reflect the row with the lowest avg_quality_cost."""
    run_sens = _make_run("false_trass", 50, 5000, "sensitivity", "none")
    run_rescue = _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue")
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 1200.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 600.0},
    ]
    run_groups = _build_run_groups([run_sens, run_rescue])
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    for row in rows:
        assert row["baseline_shadow_actual_best_policy"] == "history_rescue/history_rescue"


def test_baseline_actual_best_switches_when_sensitivity_is_cheapest():
    """When sensitivity/none has the lowest quality_cost, actual_best reflects that."""
    run_sens = _make_run("regime_switch", 50, 5000, "sensitivity", "none",
                         regime_sentinel_fails=500)
    run_rescue = _make_run("regime_switch", 50, 5000, "history_rescue", "history_rescue")
    rows = [
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 700.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 1500.0},
    ]
    run_groups = _build_run_groups([run_sens, run_rescue])
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    for row in rows:
        assert row["baseline_shadow_actual_best_policy"] == "sensitivity/none"


# ---------------------------------------------------------------------------
# schedule is not passed into predict()
# ---------------------------------------------------------------------------

def test_baseline_schedule_not_in_predict():
    """Identical sensitivity/none features across different schedules give the same prediction."""
    run_ft = _make_run("false_trass", 50, 5000, "sensitivity", "none", regime_sentinel_fails=0)
    run_rs = _make_run("regime_switch", 50, 5000, "sensitivity", "none", regime_sentinel_fails=0)

    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "regime_switch", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
    ]
    run_groups = {
        ("false_trass", 50, 5000, "sensitivity/none"): [run_ft],
        ("regime_switch", 50, 5000, "sensitivity/none"): [run_rs],
    }
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    assert (
        rows[0]["baseline_shadow_predicted_policy"]
        == rows[1]["baseline_shadow_predicted_policy"]
    )


# ---------------------------------------------------------------------------
# All BASELINE_SHADOW_ROW_FIELDS present after annotation
# ---------------------------------------------------------------------------

def test_baseline_annotation_adds_all_fields():
    """All BASELINE_SHADOW_ROW_FIELDS must appear on each row after annotation."""
    run = _make_run("false_trass", 50, 5000, "sensitivity", "none")
    rows = [{"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
             "policy": "sensitivity/none", "avg_quality_cost": 900.0}]
    run_groups = _build_run_groups([run])
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    for field in BASELINE_SHADOW_ROW_FIELDS:
        assert field in rows[0], f"Missing field {field!r}"


# ---------------------------------------------------------------------------
# All rows in a scope share the same prediction
# ---------------------------------------------------------------------------

def test_baseline_all_rows_in_scope_get_same_prediction():
    """Every policy row in the same scope must carry identical baseline_shadow fields."""
    run_sens = _make_run("false_trass", 50, 5000, "sensitivity", "none",
                         regime_sentinel_fails=500)
    run_hh = _make_run("false_trass", 50, 5000, "history", "history")
    run_rescue = _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue")
    rows = [
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "sensitivity/none", "avg_quality_cost": 900.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history/history", "avg_quality_cost": 700.0},
        {"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
         "policy": "history_rescue/history_rescue", "avg_quality_cost": 800.0},
    ]
    run_groups = _build_run_groups([run_sens, run_hh, run_rescue])
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    # All 3 rows must carry an identical baseline prediction
    predictions = {row["baseline_shadow_predicted_policy"] for row in rows}
    assert len(predictions) == 1

    actual_bests = {row["baseline_shadow_actual_best_policy"] for row in rows}
    assert len(actual_bests) == 1


# ---------------------------------------------------------------------------
# Missing sensitivity/none runs → None fields
# ---------------------------------------------------------------------------

def test_baseline_none_when_no_sensitivity_none_runs():
    """If sensitivity/none is absent from a scope, all baseline fields must be None."""
    run_rescue = _make_run("false_trass", 50, 5000, "history_rescue", "history_rescue")
    rows = [{"schedule": "false_trass", "n_vars": 50, "cycles": 5000,
             "policy": "history_rescue/history_rescue", "avg_quality_cost": 700.0}]
    run_groups = _build_run_groups([run_rescue])  # no sensitivity/none key
    sel = ShadowPolicySelector()
    baseline_annotate_rows(rows, run_groups, sel)

    for field in BASELINE_SHADOW_ROW_FIELDS:
        assert rows[0][field] is None

    # No predictions should be recorded
    assert sel.summary() == {}


# ---------------------------------------------------------------------------
# Does not mutate non-baseline fields
# ---------------------------------------------------------------------------

def test_baseline_does_not_mutate_non_baseline_fields():
    """baseline_annotate_rows must not modify any key that is not in BASELINE_SHADOW_ROW_FIELDS."""
    run = _make_run("false_trass", 50, 5000, "sensitivity", "none")
    row = {
        "schedule": "false_trass", "n_vars": 50, "cycles": 5000,
        "policy": "sensitivity/none", "avg_quality_cost": 900.0,
        "avg_iv": 100.0, "pareto_status": "efficient",
    }
    original = {k: v for k, v in row.items() if k not in BASELINE_SHADOW_ROW_FIELDS}
    run_groups = _build_run_groups([run])
    sel = ShadowPolicySelector()
    baseline_annotate_rows([row], run_groups, sel)

    for k, v in original.items():
        assert row[k] == v, f"baseline_annotate_rows mutated non-baseline field {k!r}"
