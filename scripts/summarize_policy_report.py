"""
Offline summarizer for policy_report.tsv produced by --policy-report.

Usage:
    python scripts/summarize_policy_report.py --tsv reports/policy_report.tsv

WARNING: diagnostic only; no runtime policy switching.
"""

import argparse
import csv
import sys
from collections import defaultdict
from typing import NamedTuple


METRIC_COLS = [
    "avg_quality_cost",
    "avg_iv",
    "avg_full_audits",
    "avg_revocations",
    "avg_unique_fails",
    "avg_regime_fail",
    "avg_no_sentinel",
    "avg_elapsed",
]

DELTA_COLS = [
    "delta_quality_cost_vs_sensitivity",
    "delta_iv_vs_sensitivity",
    "delta_audits_vs_sensitivity",
    "delta_revocations_vs_sensitivity",
    "delta_unique_fails_vs_sensitivity",
]


class Row(NamedTuple):
    schedule: str
    n_vars: int
    cycles: int
    policy: str
    runs: int
    avg_quality_cost: float
    avg_iv: float
    avg_full_audits: float
    avg_revocations: float
    avg_unique_fails: float
    avg_regime_fail: float
    avg_no_sentinel: float
    avg_skip_pct: float
    avg_elapsed: float
    invariants_ok: bool
    delta_quality_cost_vs_sensitivity: float
    delta_iv_vs_sensitivity: float
    delta_audits_vs_sensitivity: float
    delta_revocations_vs_sensitivity: float
    delta_unique_fails_vs_sensitivity: float
    pareto_status: str
    # Shadow selector fields — present only in TSVs produced with shadow annotation.
    # Defaults keep load_tsv backward-compatible with older reports.
    shadow_predicted_policy: str = ""
    shadow_actual_best_policy: str = ""
    shadow_policy_correct: str = ""
    shadow_false_switch_to_history_rescue_under_regime_switch: str = ""
    shadow_missed_history_rescue_under_false_trass: str = ""
    shadow_history_history_wins_missed: str = ""
    # Baseline-only shadow selector fields (sensitivity/none diagnostics → one prediction/scope).
    baseline_shadow_predicted_policy: str = ""
    baseline_shadow_actual_best_policy: str = ""
    baseline_shadow_correct: str = ""
    baseline_shadow_false_switch_to_history_rescue_under_regime_switch: str = ""
    baseline_shadow_missed_history_rescue_under_false_trass: str = ""
    baseline_shadow_history_history_wins_missed: str = ""


def load_tsv(path: str) -> list[Row]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(Row(
                schedule=r["schedule"],
                n_vars=int(r["n_vars"]),
                cycles=int(r["cycles"]),
                policy=r["policy"],
                runs=int(r["runs"]),
                avg_quality_cost=float(r["avg_quality_cost"]),
                avg_iv=float(r["avg_iv"]),
                avg_full_audits=float(r["avg_full_audits"]),
                avg_revocations=float(r["avg_revocations"]),
                avg_unique_fails=float(r["avg_unique_fails"]),
                avg_regime_fail=float(r["avg_regime_fail"]),
                avg_no_sentinel=float(r["avg_no_sentinel"]),
                avg_skip_pct=float(r["avg_skip_pct"]),
                avg_elapsed=float(r["avg_elapsed"]),
                invariants_ok=r["invariants_ok"].strip().lower() == "true",
                delta_quality_cost_vs_sensitivity=float(r["delta_quality_cost_vs_sensitivity"]),
                delta_iv_vs_sensitivity=float(r["delta_iv_vs_sensitivity"]),
                delta_audits_vs_sensitivity=float(r["delta_audits_vs_sensitivity"]),
                delta_revocations_vs_sensitivity=float(r["delta_revocations_vs_sensitivity"]),
                delta_unique_fails_vs_sensitivity=float(r["delta_unique_fails_vs_sensitivity"]),
                pareto_status=r["pareto_status"].strip(),
                shadow_predicted_policy=r.get("shadow_predicted_policy", "") or "",
                shadow_actual_best_policy=r.get("shadow_actual_best_policy", "") or "",
                shadow_policy_correct=r.get("shadow_policy_correct", "") or "",
                shadow_false_switch_to_history_rescue_under_regime_switch=(
                    r.get("shadow_false_switch_to_history_rescue_under_regime_switch", "") or ""
                ),
                shadow_missed_history_rescue_under_false_trass=(
                    r.get("shadow_missed_history_rescue_under_false_trass", "") or ""
                ),
                shadow_history_history_wins_missed=(
                    r.get("shadow_history_history_wins_missed", "") or ""
                ),
                baseline_shadow_predicted_policy=(
                    r.get("baseline_shadow_predicted_policy", "") or ""
                ),
                baseline_shadow_actual_best_policy=(
                    r.get("baseline_shadow_actual_best_policy", "") or ""
                ),
                baseline_shadow_correct=(
                    r.get("baseline_shadow_correct", "") or ""
                ),
                baseline_shadow_false_switch_to_history_rescue_under_regime_switch=(
                    r.get("baseline_shadow_false_switch_to_history_rescue_under_regime_switch", "") or ""
                ),
                baseline_shadow_missed_history_rescue_under_false_trass=(
                    r.get("baseline_shadow_missed_history_rescue_under_false_trass", "") or ""
                ),
                baseline_shadow_history_history_wins_missed=(
                    r.get("baseline_shadow_history_history_wins_missed", "") or ""
                ),
            ))
    return rows


# ---------------------------------------------------------------------------
# 1. Winner table
# ---------------------------------------------------------------------------

class WinnerRow(NamedTuple):
    schedule: str
    n_vars: int
    cycles: int
    best_policy: str
    best_quality_cost: float
    runner_up: str
    margin: float


def compute_winner_table(rows: list[Row]) -> list[WinnerRow]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.schedule, r.n_vars, r.cycles)].append(r)

    result = []
    for (schedule, n_vars, cycles), group in sorted(groups.items()):
        sorted_group = sorted(group, key=lambda r: r.avg_quality_cost)
        best = sorted_group[0]
        runner_up = sorted_group[1] if len(sorted_group) > 1 else None
        margin = (runner_up.avg_quality_cost - best.avg_quality_cost) if runner_up else float("nan")
        result.append(WinnerRow(
            schedule=schedule,
            n_vars=n_vars,
            cycles=cycles,
            best_policy=best.policy,
            best_quality_cost=best.avg_quality_cost,
            runner_up=runner_up.policy if runner_up else "",
            margin=margin,
        ))
    return result


# ---------------------------------------------------------------------------
# 2. Winner counts
# ---------------------------------------------------------------------------

def compute_winner_counts(winner_table: list[WinnerRow]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for w in winner_table:
        counts[w.best_policy] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# 3. Mean metrics by schedule/policy
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def compute_grouped_means(rows: list[Row]) -> list[dict]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.schedule, r.policy)].append(r)

    result = []
    for (schedule, policy), group in sorted(groups.items()):
        result.append({
            "schedule": schedule,
            "policy": policy,
            "avg_quality_cost": _mean([r.avg_quality_cost for r in group]),
            "avg_iv": _mean([r.avg_iv for r in group]),
            "avg_full_audits": _mean([r.avg_full_audits for r in group]),
            "avg_revocations": _mean([r.avg_revocations for r in group]),
            "avg_unique_fails": _mean([r.avg_unique_fails for r in group]),
            "avg_regime_fail": _mean([r.avg_regime_fail for r in group]),
            "avg_no_sentinel": _mean([r.avg_no_sentinel for r in group]),
            "avg_elapsed": _mean([r.avg_elapsed for r in group]),
        })
    return result


# ---------------------------------------------------------------------------
# 4. Dominated summary
# ---------------------------------------------------------------------------

def compute_dominated_counts(rows: list[Row]) -> list[dict]:
    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        if r.pareto_status == "dominated":
            counts[(r.schedule, r.policy)] += 1

    result = []
    for (schedule, policy), count in sorted(counts.items()):
        result.append({"schedule": schedule, "policy": policy, "dominated_count": count})
    return result


# ---------------------------------------------------------------------------
# 5. IV-vs-structure tradeoff relative to sensitivity/none
# ---------------------------------------------------------------------------

def compute_iv_tradeoff(rows: list[Row]) -> list[dict]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        if r.policy == "sensitivity/none":
            continue
        groups[(r.policy, r.schedule)].append(r)

    result = []
    for (policy, schedule), group in sorted(groups.items()):
        result.append({
            "policy": policy,
            "schedule": schedule,
            "mean_delta_iv": _mean([r.delta_iv_vs_sensitivity for r in group]),
            "mean_delta_audits": _mean([r.delta_audits_vs_sensitivity for r in group]),
            "mean_delta_revocations": _mean([r.delta_revocations_vs_sensitivity for r in group]),
            "mean_delta_unique_fails": _mean([r.delta_unique_fails_vs_sensitivity for r in group]),
            "mean_delta_quality_cost": _mean([r.delta_quality_cost_vs_sensitivity for r in group]),
        })
    return result


# ---------------------------------------------------------------------------
# 6. Recommendation block
# ---------------------------------------------------------------------------

def compute_recommendations(rows: list[Row]) -> dict[str, str]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r.schedule][r.policy].append(r.avg_quality_cost)

    recs = {}
    for schedule, policy_costs in sorted(groups.items()):
        best_policy = min(policy_costs, key=lambda p: _mean(policy_costs[p]))
        recs[schedule] = best_policy
    return recs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, width: int = 14) -> str:
    if v != v:  # nan
        return "nan".rjust(width)
    if abs(v) >= 1e6:
        return f"{v:>{width}.3e}"
    return f"{v:>{width}.2f}"


def print_winner_table(winners: list[WinnerRow]) -> None:
    print("\n=== 1. Winner Table (best policy by quality_cost per schedule/n_vars/cycles) ===")
    header = f"{'schedule':<18} {'n_vars':>6} {'cycles':>7}  {'best_policy':<28} {'best_quality_cost':>18}  {'runner_up':<28} {'margin':>14}"
    print(header)
    print("-" * len(header))
    for w in winners:
        print(
            f"{w.schedule:<18} {w.n_vars:>6} {w.cycles:>7}  {w.best_policy:<28}"
            f" {_fmt(w.best_quality_cost)}  {w.runner_up:<28} {_fmt(w.margin)}"
        )


def print_winner_counts(counts: dict[str, int]) -> None:
    print("\n=== 2. Winner Counts (wins by quality_cost) ===")
    print(f"{'policy':<30} {'wins':>6}")
    print("-" * 38)
    for policy, wins in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{policy:<30} {wins:>6}")


def print_grouped_means(means: list[dict]) -> None:
    print("\n=== 3. Mean Metrics by schedule/policy ===")
    cols = ["avg_quality_cost", "avg_iv", "avg_full_audits", "avg_revocations",
            "avg_unique_fails", "avg_regime_fail", "avg_no_sentinel", "avg_elapsed"]
    header = f"{'schedule':<18} {'policy':<28}" + "".join(f" {c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in means:
        vals = "".join(_fmt(row[c], 16) for c in cols)
        print(f"{row['schedule']:<18} {row['policy']:<28}{vals}")


def print_dominated_summary(dominated: list[dict]) -> None:
    print("\n=== 4. Dominated Row Counts by schedule/policy ===")
    if not dominated:
        print("  (no dominated rows)")
        return
    print(f"{'schedule':<18} {'policy':<30} {'dominated_count':>16}")
    print("-" * 66)
    for row in dominated:
        print(f"{row['schedule']:<18} {row['policy']:<30} {row['dominated_count']:>16}")


def print_iv_tradeoff(tradeoff: list[dict]) -> None:
    print("\n=== 5. IV-vs-Structure Tradeoff (relative to sensitivity/none) ===")
    cols = ["mean_delta_iv", "mean_delta_audits", "mean_delta_revocations",
            "mean_delta_unique_fails", "mean_delta_quality_cost"]
    header = f"{'policy':<28} {'schedule':<18}" + "".join(f" {c:>22}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in tradeoff:
        vals = "".join(_fmt(row[c], 22) for c in cols)
        print(f"{row['policy']:<28} {row['schedule']:<18}{vals}")


def print_recommendations(recs: dict[str, str]) -> None:
    print("\n=== 6. Default Policy Recommendations (lowest mean quality_cost per schedule) ===")
    print("  WARNING: diagnostic only; no runtime policy switching.")
    print()
    print(f"{'schedule':<20} {'recommended_policy'}")
    print("-" * 50)
    for schedule, policy in sorted(recs.items()):
        print(f"  {schedule:<18} {policy}")


# ---------------------------------------------------------------------------
# 7. Shadow selector summary (from per-row shadow fields in TSV)
# ---------------------------------------------------------------------------

def compute_shadow_selector_summary(rows: list[Row]) -> dict:
    """Compute shadow selector accuracy from per-row shadow fields.

    Returns an empty dict if the TSV was produced without shadow annotation.
    """
    annotated = [r for r in rows if r.shadow_predicted_policy]
    if not annotated:
        return {}

    n = len(annotated)
    n_correct = sum(
        1 for r in annotated if r.shadow_policy_correct.strip().lower() == "true"
    )
    false_switches = sum(
        1 for r in annotated
        if r.shadow_false_switch_to_history_rescue_under_regime_switch.strip().lower() == "true"
    )
    missed_rescues = sum(
        1 for r in annotated
        if r.shadow_missed_history_rescue_under_false_trass.strip().lower() == "true"
    )
    history_history_wins_missed = sum(
        1 for r in annotated
        if r.shadow_history_history_wins_missed.strip().lower() == "true"
    )

    predicted_counts: dict[str, int] = defaultdict(int)
    actual_counts: dict[str, int] = defaultdict(int)
    for r in annotated:
        predicted_counts[r.shadow_predicted_policy] += 1
        actual_counts[r.shadow_actual_best_policy] += 1

    return {
        "n_predictions": n,
        "accuracy": n_correct / n,
        "false_switch_to_history_rescue_under_regime_switch": false_switches,
        "missed_history_rescue_under_false_trass": missed_rescues,
        "history_history_wins_missed": history_history_wins_missed,
        "predicted_policy": dict(predicted_counts),
        "actual_best_policy": dict(actual_counts),
    }


def print_shadow_selector_summary(summary: dict) -> None:
    print("\n=== 7. Shadow Policy Selector Summary ===")
    if not summary:
        print("  (no shadow annotation in TSV — re-run with current batch_run.py)")
        return
    print("  diagnostic only — no runtime policy switching")
    print(f"  n_predictions : {summary['n_predictions']}")
    print(f"  accuracy      : {summary['accuracy']:.3f}")
    print(
        f"  false_switch_to_history_rescue_under_regime_switch : "
        f"{summary['false_switch_to_history_rescue_under_regime_switch']}"
    )
    print(
        f"  missed_history_rescue_under_false_trass            : "
        f"{summary['missed_history_rescue_under_false_trass']}"
    )
    print(
        f"  history_history_wins_missed                        : "
        f"{summary['history_history_wins_missed']}"
    )
    print("  predicted_policy counts:")
    for policy, count in sorted(summary["predicted_policy"].items()):
        print(f"    {policy:<36} {count}")
    print("  actual_best_policy counts:")
    for policy, count in sorted(summary["actual_best_policy"].items()):
        print(f"    {policy:<36} {count}")


# ---------------------------------------------------------------------------
# 8. Baseline-only shadow selector summary
# ---------------------------------------------------------------------------

def compute_baseline_shadow_selector_summary(rows: list[Row]) -> dict:
    """Compute baseline-only shadow selector accuracy from per-row baseline fields.

    Each scope contributes one prediction (from sensitivity/none diagnostics).
    Returns an empty dict if the TSV was produced without baseline annotation.
    """
    annotated = [r for r in rows if r.baseline_shadow_predicted_policy]
    if not annotated:
        return {}

    # De-duplicate by scope: all rows in a scope carry the same baseline prediction,
    # so count each scope once to match the one-prediction-per-scope invariant.
    seen_scopes: set = set()
    unique: list[Row] = []
    for r in annotated:
        scope = (r.schedule, r.n_vars, r.cycles)
        if scope not in seen_scopes:
            seen_scopes.add(scope)
            unique.append(r)

    n = len(unique)
    n_correct = sum(
        1 for r in unique if r.baseline_shadow_correct.strip().lower() == "true"
    )
    false_switches = sum(
        1 for r in unique
        if r.baseline_shadow_false_switch_to_history_rescue_under_regime_switch.strip().lower() == "true"
    )
    missed_rescues = sum(
        1 for r in unique
        if r.baseline_shadow_missed_history_rescue_under_false_trass.strip().lower() == "true"
    )
    history_history_wins_missed = sum(
        1 for r in unique
        if r.baseline_shadow_history_history_wins_missed.strip().lower() == "true"
    )

    predicted_counts: dict[str, int] = defaultdict(int)
    actual_counts: dict[str, int] = defaultdict(int)
    for r in unique:
        predicted_counts[r.baseline_shadow_predicted_policy] += 1
        actual_counts[r.baseline_shadow_actual_best_policy] += 1

    return {
        "n_predictions": n,
        "accuracy": n_correct / n,
        "false_switch_to_history_rescue_under_regime_switch": false_switches,
        "missed_history_rescue_under_false_trass": missed_rescues,
        "history_history_wins_missed": history_history_wins_missed,
        "predicted_policy": dict(predicted_counts),
        "actual_best_policy": dict(actual_counts),
    }


def print_baseline_shadow_selector_summary(summary: dict) -> None:
    print("\n=== 8. Baseline-Only Shadow Policy Selector Summary ===")
    if not summary:
        print("  (no baseline shadow annotation in TSV — re-run with current batch_run.py)")
        return
    print("  diagnostic only — no runtime policy switching")
    print(f"  n_predictions : {summary['n_predictions']}")
    print(f"  accuracy      : {summary['accuracy']:.3f}")
    print(
        f"  false_switch_to_history_rescue_under_regime_switch : "
        f"{summary['false_switch_to_history_rescue_under_regime_switch']}"
    )
    print(
        f"  missed_history_rescue_under_false_trass            : "
        f"{summary['missed_history_rescue_under_false_trass']}"
    )
    print(
        f"  history_history_wins_missed                        : "
        f"{summary['history_history_wins_missed']}"
    )
    print("  predicted_policy counts:")
    for policy, count in sorted(summary["predicted_policy"].items()):
        print(f"    {policy:<36} {count}")
    print("  actual_best_policy counts:")
    for policy, count in sorted(summary["actual_best_policy"].items()):
        print(f"    {policy:<36} {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarize(path: str) -> None:
    rows = load_tsv(path)

    winners = compute_winner_table(rows)
    counts = compute_winner_counts(winners)
    means = compute_grouped_means(rows)
    dominated = compute_dominated_counts(rows)
    tradeoff = compute_iv_tradeoff(rows)
    recs = compute_recommendations(rows)
    shadow_summary = compute_shadow_selector_summary(rows)
    baseline_summary = compute_baseline_shadow_selector_summary(rows)

    print_winner_table(winners)
    print_winner_counts(counts)
    print_grouped_means(means)
    print_dominated_summary(dominated)
    print_iv_tradeoff(tradeoff)
    print_recommendations(recs)
    print_shadow_selector_summary(shadow_summary)
    print_baseline_shadow_selector_summary(baseline_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize policy_report.tsv offline.")
    parser.add_argument("--tsv", required=True, help="Path to policy_report.tsv")
    args = parser.parse_args()
    summarize(args.tsv)


if __name__ == "__main__":
    main()
