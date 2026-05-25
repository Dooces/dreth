"""
Tests for scripts/summarize_policy_report.py using a small synthetic TSV.

Covers:
  - winner selection per group
  - grouped means
  - dominated counts
  - recommendation block
  - IV-vs-structure tradeoff
"""

import io
import textwrap
import sys
import os
import pytest

# Make the scripts directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from summarize_policy_report import (
    load_tsv,
    compute_winner_table,
    compute_winner_counts,
    compute_grouped_means,
    compute_dominated_counts,
    compute_recommendations,
    compute_iv_tradeoff,
    print_recommendations,
)


SYNTHETIC_TSV = textwrap.dedent("""\
    schedule\tn_vars\tcycles\tpolicy\truns\tavg_quality_cost\tavg_iv\tavg_full_audits\tavg_revocations\tavg_unique_fails\tavg_regime_fail\tavg_no_sentinel\tavg_skip_pct\tavg_elapsed\tinvariants_ok\tdelta_quality_cost_vs_sensitivity\tdelta_iv_vs_sensitivity\tdelta_audits_vs_sensitivity\tdelta_revocations_vs_sensitivity\tdelta_unique_fails_vs_sensitivity\tpareto_status
    sched_a\t10\t100\tpolicyX\t3\t500.0\t100.0\t5.0\t1.0\t2.0\t0.0\t0.0\t99.9\t1.0\tTrue\t-200.0\t-50.0\t0.0\t0.0\t0.0\tefficient
    sched_a\t10\t100\tpolicyY\t3\t800.0\t200.0\t6.0\t2.0\t3.0\t0.0\t0.0\t99.8\t1.2\tTrue\t100.0\t50.0\t1.0\t1.0\t1.0\tdominated
    sched_a\t10\t100\tsensitivity/none\t3\t700.0\t150.0\t5.5\t1.5\t2.5\t0.0\t0.0\t99.85\t1.1\tTrue\t0.0\t0.0\t0.0\t0.0\t0.0\tefficient
    sched_a\t10\t200\tpolicyX\t3\t600.0\t120.0\t5.5\t1.0\t2.0\t0.0\t0.0\t99.9\t1.5\tTrue\t-300.0\t-80.0\t0.0\t0.0\t0.0\tefficient
    sched_a\t10\t200\tpolicyY\t3\t1200.0\t300.0\t7.0\t3.0\t4.0\t0.0\t0.0\t99.7\t1.8\tTrue\t300.0\t100.0\t1.5\t2.0\t2.0\tdominated
    sched_a\t10\t200\tsensitivity/none\t3\t900.0\t200.0\t5.5\t1.0\t2.0\t0.0\t0.0\t99.85\t1.6\tTrue\t0.0\t0.0\t0.0\t0.0\t0.0\tdominated
    sched_b\t10\t100\tpolicyX\t3\t2000.0\t500.0\t20.0\t5.0\t30.0\t100.0\t200.0\t98.0\t5.0\tTrue\t500.0\t-100.0\t5.0\t2.0\t10.0\tefficient
    sched_b\t10\t100\tsensitivity/none\t3\t1500.0\t600.0\t15.0\t3.0\t20.0\t50.0\t0.0\t99.0\t4.0\tTrue\t0.0\t0.0\t0.0\t0.0\t0.0\tefficient
""")


@pytest.fixture
def rows(tmp_path):
    tsv_file = tmp_path / "policy_report.tsv"
    tsv_file.write_text(SYNTHETIC_TSV)
    return load_tsv(str(tsv_file))


# ---------------------------------------------------------------------------
# load_tsv
# ---------------------------------------------------------------------------

def test_load_tsv_row_count(rows):
    assert len(rows) == 8


def test_load_tsv_types(rows):
    r = rows[0]
    assert isinstance(r.n_vars, int)
    assert isinstance(r.cycles, int)
    assert isinstance(r.avg_quality_cost, float)
    assert isinstance(r.invariants_ok, bool)


def test_load_tsv_invariants_ok_true(rows):
    assert all(r.invariants_ok for r in rows)


# ---------------------------------------------------------------------------
# 1. Winner table
# ---------------------------------------------------------------------------

def test_winner_table_groups(rows):
    winners = compute_winner_table(rows)
    # 3 groups: (sched_a,10,100), (sched_a,10,200), (sched_b,10,100)
    assert len(winners) == 3


def test_winner_table_sched_a_100(rows):
    winners = compute_winner_table(rows)
    w = next(w for w in winners if w.schedule == "sched_a" and w.cycles == 100)
    assert w.best_policy == "policyX"
    assert w.best_quality_cost == pytest.approx(500.0)
    assert w.runner_up == "sensitivity/none"
    assert w.margin == pytest.approx(200.0)


def test_winner_table_sched_a_200(rows):
    winners = compute_winner_table(rows)
    w = next(w for w in winners if w.schedule == "sched_a" and w.cycles == 200)
    assert w.best_policy == "policyX"
    assert w.best_quality_cost == pytest.approx(600.0)
    assert w.runner_up == "sensitivity/none"
    assert w.margin == pytest.approx(300.0)


def test_winner_table_sched_b(rows):
    winners = compute_winner_table(rows)
    w = next(w for w in winners if w.schedule == "sched_b")
    assert w.best_policy == "sensitivity/none"
    assert w.runner_up == "policyX"
    assert w.margin == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 2. Winner counts
# ---------------------------------------------------------------------------

def test_winner_counts(rows):
    winners = compute_winner_table(rows)
    counts = compute_winner_counts(winners)
    # policyX wins sched_a/100 and sched_a/200; sensitivity/none wins sched_b/100
    assert counts["policyX"] == 2
    assert counts["sensitivity/none"] == 1
    assert "policyY" not in counts


# ---------------------------------------------------------------------------
# 3. Grouped means
# ---------------------------------------------------------------------------

def test_grouped_means_keys(rows):
    means = compute_grouped_means(rows)
    keys = {(m["schedule"], m["policy"]) for m in means}
    assert ("sched_a", "policyX") in keys
    assert ("sched_b", "sensitivity/none") in keys


def test_grouped_means_policyX_sched_a(rows):
    means = compute_grouped_means(rows)
    row = next(m for m in means if m["schedule"] == "sched_a" and m["policy"] == "policyX")
    # policyX in sched_a has two entries: 500 and 600 → mean 550
    assert row["avg_quality_cost"] == pytest.approx(550.0)
    # avg_iv: (100 + 120) / 2 = 110
    assert row["avg_iv"] == pytest.approx(110.0)


def test_grouped_means_policyY_dominated_rows(rows):
    means = compute_grouped_means(rows)
    row = next(m for m in means if m["schedule"] == "sched_a" and m["policy"] == "policyY")
    # policyY: avg_quality_cost: (800 + 1200) / 2 = 1000
    assert row["avg_quality_cost"] == pytest.approx(1000.0)


def test_grouped_means_single_entry_sched_b(rows):
    means = compute_grouped_means(rows)
    row = next(m for m in means if m["schedule"] == "sched_b" and m["policy"] == "policyX")
    assert row["avg_quality_cost"] == pytest.approx(2000.0)
    assert row["avg_regime_fail"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 4. Dominated counts
# ---------------------------------------------------------------------------

def test_dominated_counts(rows):
    dominated = compute_dominated_counts(rows)
    d = {(r["schedule"], r["policy"]): r["dominated_count"] for r in dominated}
    # policyY dominated twice in sched_a (cycles 100 and 200)
    assert d[("sched_a", "policyY")] == 2
    # sensitivity/none dominated once in sched_a cycles=200
    assert d[("sched_a", "sensitivity/none")] == 1
    # policyX never dominated
    assert ("sched_a", "policyX") not in d
    assert ("sched_b", "sensitivity/none") not in d


def test_dominated_counts_empty_when_none():
    # All efficient rows → no dominated entries
    rows_efficient = [
        r for r in []
    ]
    assert compute_dominated_counts(rows_efficient) == []


# ---------------------------------------------------------------------------
# 5. IV-vs-structure tradeoff
# ---------------------------------------------------------------------------

def test_iv_tradeoff_excludes_sensitivity(rows):
    tradeoff = compute_iv_tradeoff(rows)
    for row in tradeoff:
        assert row["policy"] != "sensitivity/none"


def test_iv_tradeoff_policyX_sched_a(rows):
    tradeoff = compute_iv_tradeoff(rows)
    row = next(r for r in tradeoff if r["policy"] == "policyX" and r["schedule"] == "sched_a")
    # delta_quality_cost: (-200 + -300) / 2 = -250
    assert row["mean_delta_quality_cost"] == pytest.approx(-250.0)
    # delta_iv: (-50 + -80) / 2 = -65
    assert row["mean_delta_iv"] == pytest.approx(-65.0)


def test_iv_tradeoff_policyX_sched_b(rows):
    tradeoff = compute_iv_tradeoff(rows)
    row = next(r for r in tradeoff if r["policy"] == "policyX" and r["schedule"] == "sched_b")
    assert row["mean_delta_quality_cost"] == pytest.approx(500.0)
    assert row["mean_delta_audits"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 6. Recommendation block
# ---------------------------------------------------------------------------

def test_recommendations_sched_a(rows):
    recs = compute_recommendations(rows)
    # sched_a means: policyX=550, policyY=1000, sensitivity/none=800 → policyX wins
    assert recs["sched_a"] == "policyX"


def test_recommendations_sched_b(rows):
    recs = compute_recommendations(rows)
    # sched_b means: policyX=2000, sensitivity/none=1500 → sensitivity/none wins
    assert recs["sched_b"] == "sensitivity/none"


def test_recommendations_print_includes_warning(rows, capsys):
    recs = compute_recommendations(rows)
    print_recommendations(recs)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "diagnostic only" in out
    assert "no runtime policy switching" in out


def test_recommendations_print_lists_all_schedules(rows, capsys):
    recs = compute_recommendations(rows)
    print_recommendations(recs)
    out = capsys.readouterr().out
    assert "sched_a" in out
    assert "sched_b" in out
