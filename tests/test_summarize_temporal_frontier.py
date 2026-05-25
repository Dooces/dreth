from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from summarize_temporal_frontier import (  # noqa: E402
    compute_frontier_summaries,
    compute_scale_curve,
    compute_warmup_curve,
    load_jsonl,
    print_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "temporal_frontier.jsonl"
    _write_jsonl(
        path,
        [
            {
                "record_type": "policy_report",
                "schedule": "false_trass",
                "n_vars": 999,
                "cycles": 999,
                "parent_ranker": "ignored",
                "probe_proposer": "ignored",
                "temporal_frontier_evals": 999,
                "temporal_frontier_chosen_parent_hits": 999,
                "temporal_frontier_chosen_parent_total": 999,
            },
            {
                "schedule": "false_trass",
                "n_vars": 50,
                "cycles": 3000,
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "warmup_cycles": 1000,
                "max_candidates": 20,
                "max_depth": 2,
                "temporal_frontier_evals": 2,
                "temporal_frontier_avg_visible_count": 50.0,
                "temporal_frontier_avg_size": 10.0,
                "temporal_frontier_chosen_parent_hits": 1,
                "temporal_frontier_chosen_parent_total": 2,
                "temporal_frontier_revoked_hits": 0,
                "temporal_frontier_revoked_total": 0,
                "temporal_frontier_misses": 1,
            },
            {
                "schedule": "false_trass",
                "n_vars": 50,
                "cycles": 3000,
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "warmup_cycles": 1000,
                "max_candidates": 20,
                "max_depth": 2,
                "temporal_frontier_evals": 3,
                "temporal_frontier_avg_visible_count": 50.0,
                "temporal_frontier_avg_size": 5.0,
                "temporal_frontier_chosen_parent_hits": 2,
                "temporal_frontier_chosen_parent_total": 3,
                "temporal_frontier_revoked_hits": 0,
                "temporal_frontier_revoked_total": 0,
                "temporal_frontier_misses": 1,
            },
            {
                "schedule": "false_trass",
                "n_vars": 50,
                "cycles": 7500,
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "warmup_cycles": 2500,
                "max_candidates": 10,
                "max_depth": 1,
                "temporal_frontier_evals": 1,
                "temporal_frontier_avg_visible_count": 50.0,
                "temporal_frontier_avg_size": 5.0,
                "temporal_frontier_chosen_parent_hits": 0,
                "temporal_frontier_chosen_parent_total": 0,
                "temporal_frontier_revoked_hits": 0,
                "temporal_frontier_revoked_total": 0,
                "temporal_frontier_misses": 0,
            },
            {
                "schedule": "regime_switch",
                "n_vars": 100,
                "cycles": 3000,
                "parent_ranker": "history_rescue",
                "probe_proposer": "history_rescue",
                "warmup_cycles": 1000,
                "max_candidates": 20,
                "max_depth": 2,
                "temporal_frontier_evals": 4,
                "temporal_frontier_avg_visible_count": 100.0,
                "temporal_frontier_avg_size": 25.0,
                "temporal_frontier_chosen_parent_hits": 6,
                "temporal_frontier_chosen_parent_total": 8,
                "temporal_frontier_revoked_hits": 1,
                "temporal_frontier_revoked_total": 2,
                "temporal_frontier_misses": 3,
            },
        ],
    )
    return load_jsonl(str(path))


def _summary(rows, **match):
    summaries = compute_frontier_summaries(rows)
    for summary in summaries:
        key = summary.key
        if all(getattr(key, name) == value for name, value in match.items()):
            return summary
    raise AssertionError(f"no summary matching {match}")


def test_policy_report_rows_ignored(rows):
    assert len(rows) == 4
    summaries = compute_frontier_summaries(rows)

    assert all(summary.key.n_vars != 999 for summary in summaries)


def test_raw_denominators_are_aggregated(rows):
    summary = _summary(
        rows,
        schedule="false_trass",
        n_vars=50,
        cycles=3000,
        policy="sensitivity/none",
        warmup_cycles=1000,
        max_candidates=20,
        max_depth=2,
    )

    assert summary.runs == 2
    assert summary.evals == 5
    assert summary.chosen_parent_hits == 3
    assert summary.chosen_parent_total == 5
    assert summary.chosen_parent_recall == pytest.approx(3 / 5)


def test_frontier_fraction_and_recall_lift_computed_from_denominators(rows):
    summary = _summary(
        rows,
        schedule="false_trass",
        n_vars=50,
        cycles=3000,
        policy="sensitivity/none",
        warmup_cycles=1000,
        max_candidates=20,
        max_depth=2,
    )

    assert summary.avg_frontier_size == pytest.approx(7.0)
    assert summary.avg_visible == pytest.approx(50.0)
    assert summary.frontier_fraction == pytest.approx(0.14)
    assert summary.random_recall_baseline == pytest.approx(0.14)
    assert summary.recall_lift == pytest.approx((3 / 5) / 0.14)
    assert summary.candidate_reduction_vs_visible == pytest.approx(0.86)


def test_zero_denominator_prints_na(rows, capsys):
    print_report(rows)
    output = capsys.readouterr().out

    assert "  50    7500" in output
    assert "N/A" in output


def test_grouping_includes_n_cycles_policy_warmup_depth_and_cap(rows):
    keys = {summary.key for summary in compute_frontier_summaries(rows)}

    assert len(keys) == 3
    assert any(
        key.n_vars == 50
        and key.cycles == 7500
        and key.policy == "sensitivity/none"
        and key.warmup_cycles == 2500
        and key.max_candidates == 10
        and key.max_depth == 1
        for key in keys
    )


def test_warmup_and_scale_curves(rows):
    summaries = compute_frontier_summaries(rows)
    warmup = compute_warmup_curve(summaries)
    scale = compute_scale_curve(summaries)

    assert {(cycles, warmup_cycles) for cycles, warmup_cycles, _ in warmup} == {
        (3000, 1000),
        (7500, 2500),
    }
    scale_by_n = {row.label: row for row in scale}
    assert scale_by_n[50].frontier_fraction == pytest.approx(40 / 300)
    assert scale_by_n[100].frontier_fraction == pytest.approx(0.25)


def test_warning_present(rows, capsys):
    print_report(rows)
    output = capsys.readouterr().out

    assert "D. Interpretation warning:" in output
    assert "Diagnostic only" in output
    assert "proposal-prior quality" in output
    assert "not safe exclusive filtering" in output
