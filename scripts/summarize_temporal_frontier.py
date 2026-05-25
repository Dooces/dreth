from __future__ import annotations

"""Offline summarizer for temporal relative-authority frontier JSONL output.

Usage:
    python scripts/summarize_temporal_frontier.py \
      --jsonl reports/relative_authority_temporal_frontier_sweep.jsonl

WARNING: diagnostic only; this measures proposal-prior quality, not safe
exclusive filtering.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, TextIO


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if row.get("record_type") == "policy_report":
                continue
            rows.append(row)
    return rows


def policy_label(row: dict[str, Any]) -> str:
    parent_ranker = row.get("parent_ranker")
    probe_proposer = row.get("probe_proposer")
    if parent_ranker is not None and probe_proposer is not None:
        return f"{parent_ranker}/{probe_proposer}"
    return str(row.get("policy") or "unknown/unknown")


@dataclass(frozen=True)
class FrontierKey:
    schedule: str
    n_vars: int
    cycles: int
    policy: str
    warmup_cycles: int
    max_candidates: int
    max_depth: int


@dataclass
class FrontierSummary:
    key: FrontierKey
    runs: int = 0
    evals: int = 0
    frontier_size_total: float = 0.0
    visible_count_total: float = 0.0
    chosen_parent_hits: int = 0
    chosen_parent_total: int = 0
    revoked_hits: int = 0
    revoked_total: int = 0
    misses: int = 0

    @property
    def avg_frontier_size(self) -> float:
        return self.frontier_size_total / self.evals if self.evals else 0.0

    @property
    def avg_visible(self) -> float:
        return self.visible_count_total / self.evals if self.evals else 0.0

    @property
    def frontier_fraction(self) -> float:
        return self.avg_frontier_size / self.avg_visible if self.avg_visible else 0.0

    @property
    def chosen_parent_recall(self) -> float | None:
        if not self.chosen_parent_total:
            return None
        return self.chosen_parent_hits / self.chosen_parent_total

    @property
    def revoked_recall(self) -> float | None:
        if not self.revoked_total:
            return None
        return self.revoked_hits / self.revoked_total

    @property
    def random_recall_baseline(self) -> float:
        return self.frontier_fraction

    @property
    def recall_lift(self) -> float | None:
        recall = self.chosen_parent_recall
        if recall is None:
            return None
        return recall / max(self.frontier_fraction, 1e-9)

    @property
    def candidate_reduction_vs_visible(self) -> float:
        return 1.0 - self.frontier_fraction if self.evals else 0.0


def _row_key(row: dict[str, Any]) -> FrontierKey:
    return FrontierKey(
        schedule=str(row.get("schedule") or "unknown"),
        n_vars=_as_int(row.get("n_vars")),
        cycles=_as_int(row.get("cycles")),
        policy=policy_label(row),
        warmup_cycles=_as_int(
            row.get("warmup_cycles", row.get("temporal_frontier_warmup_cycles"))
        ),
        max_candidates=_as_int(
            row.get("max_candidates", row.get("temporal_frontier_max_candidates", 20))
        ),
        max_depth=_as_int(
            row.get("max_depth", row.get("temporal_frontier_max_depth", 2))
        ),
    )


def compute_frontier_summaries(
    rows: Iterable[dict[str, Any]],
) -> list[FrontierSummary]:
    groups: dict[FrontierKey, FrontierSummary] = {}
    for row in rows:
        key = _row_key(row)
        summary = groups.setdefault(key, FrontierSummary(key=key))
        evals = _as_int(row.get("temporal_frontier_evals"))
        summary.runs += 1
        summary.evals += evals
        summary.frontier_size_total += (
            _as_float(row.get("temporal_frontier_avg_size")) * evals
        )
        summary.visible_count_total += (
            _as_float(row.get("temporal_frontier_avg_visible_count")) * evals
        )
        summary.chosen_parent_hits += _as_int(
            row.get("temporal_frontier_chosen_parent_hits")
        )
        summary.chosen_parent_total += _as_int(
            row.get("temporal_frontier_chosen_parent_total")
        )
        summary.revoked_hits += _as_int(row.get("temporal_frontier_revoked_hits"))
        summary.revoked_total += _as_int(row.get("temporal_frontier_revoked_total"))
        summary.misses += _as_int(row.get("temporal_frontier_misses"))
    return [groups[key] for key in sorted(groups, key=lambda k: (
        k.schedule,
        k.n_vars,
        k.cycles,
        k.policy,
        k.warmup_cycles,
        k.max_candidates,
        k.max_depth,
    ))]


@dataclass(frozen=True)
class CurveSummary:
    label: int
    chosen_parent_recall: float | None
    recall_lift: float | None
    frontier_fraction: float


def _combine_summaries(summaries: Iterable[FrontierSummary]) -> FrontierSummary:
    combined = FrontierSummary(
        key=FrontierKey(
            schedule="all",
            n_vars=0,
            cycles=0,
            policy="all",
            warmup_cycles=0,
            max_candidates=0,
            max_depth=0,
        )
    )
    for summary in summaries:
        combined.runs += summary.runs
        combined.evals += summary.evals
        combined.frontier_size_total += summary.frontier_size_total
        combined.visible_count_total += summary.visible_count_total
        combined.chosen_parent_hits += summary.chosen_parent_hits
        combined.chosen_parent_total += summary.chosen_parent_total
        combined.revoked_hits += summary.revoked_hits
        combined.revoked_total += summary.revoked_total
        combined.misses += summary.misses
    return combined


def compute_warmup_curve(
    summaries: Iterable[FrontierSummary],
) -> list[tuple[int, int, CurveSummary]]:
    groups: dict[tuple[int, int], list[FrontierSummary]] = defaultdict(list)
    for summary in summaries:
        groups[(summary.key.cycles, summary.key.warmup_cycles)].append(summary)
    rows = []
    for (cycles, warmup_cycles), group in sorted(groups.items()):
        combined = _combine_summaries(group)
        rows.append((
            cycles,
            warmup_cycles,
            CurveSummary(
                label=warmup_cycles,
                chosen_parent_recall=combined.chosen_parent_recall,
                recall_lift=combined.recall_lift,
                frontier_fraction=combined.frontier_fraction,
            ),
        ))
    return rows


def compute_scale_curve(summaries: Iterable[FrontierSummary]) -> list[CurveSummary]:
    groups: dict[int, list[FrontierSummary]] = defaultdict(list)
    for summary in summaries:
        groups[summary.key.n_vars].append(summary)
    rows = []
    for n_vars, group in sorted(groups.items()):
        combined = _combine_summaries(group)
        rows.append(CurveSummary(
            label=n_vars,
            chosen_parent_recall=combined.chosen_parent_recall,
            recall_lift=combined.recall_lift,
            frontier_fraction=combined.frontier_fraction,
        ))
    return rows


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _print_scaling_table(summaries: list[FrontierSummary], out: TextIO) -> None:
    print("A. Temporal frontier scaling table:", file=out)
    print(
        f"  {'schedule':<16} {'policy':<32} {'n_vars':>6} {'cycles':>7} "
        f"{'warmup':>7} {'cap':>4} {'depth':>5} {'runs':>4} {'evals':>6} "
        f"{'avg_visible':>11} {'avg_frontier_size':>17} "
        f"{'frontier_fraction':>17} {'chosen_parent_recall':>22} "
        f"{'random_recall_baseline':>23} {'recall_lift':>11} "
        f"{'candidate_reduction_vs_visible':>30} {'misses':>7}",
        file=out,
    )
    for summary in summaries:
        key = summary.key
        print(
            f"  {key.schedule:<16} {key.policy:<32} {key.n_vars:>6} "
            f"{key.cycles:>7} {key.warmup_cycles:>7} "
            f"{key.max_candidates:>4} {key.max_depth:>5} "
            f"{summary.runs:>4} {summary.evals:>6} "
            f"{summary.avg_visible:>11.2f} {summary.avg_frontier_size:>17.2f} "
            f"{summary.frontier_fraction:>17.3f} "
            f"{_fmt_float(summary.chosen_parent_recall):>22} "
            f"{summary.random_recall_baseline:>23.3f} "
            f"{_fmt_float(summary.recall_lift):>11} "
            f"{summary.candidate_reduction_vs_visible:>30.3f} "
            f"{summary.misses:>7}",
            file=out,
        )
    if not summaries:
        print("  (no normal run records)", file=out)


def _print_warmup_curve(summaries: list[FrontierSummary], out: TextIO) -> None:
    print("\nB. Warmup/scaffold curve:", file=out)
    print(
        f"  {'cycles':>7} {'warmup_cycles':>13} "
        f"{'chosen_parent_recall':>22} {'recall_lift':>11} "
        f"{'frontier_fraction':>17}",
        file=out,
    )
    rows = compute_warmup_curve(summaries)
    for cycles, warmup_cycles, row in rows:
        print(
            f"  {cycles:>7} {warmup_cycles:>13} "
            f"{_fmt_float(row.chosen_parent_recall):>22} "
            f"{_fmt_float(row.recall_lift):>11} "
            f"{row.frontier_fraction:>17.3f}",
            file=out,
        )
    if not rows:
        print("  (no temporal frontier records)", file=out)


def _print_scale_curve(summaries: list[FrontierSummary], out: TextIO) -> None:
    print("\nC. Scale curve:", file=out)
    print(
        f"  {'n_vars':>6} {'chosen_parent_recall':>22} "
        f"{'recall_lift':>11} {'frontier_fraction':>17}",
        file=out,
    )
    rows = compute_scale_curve(summaries)
    for row in rows:
        print(
            f"  {row.label:>6} {_fmt_float(row.chosen_parent_recall):>22} "
            f"{_fmt_float(row.recall_lift):>11} {row.frontier_fraction:>17.3f}",
            file=out,
        )
    if not rows:
        print("  (no temporal frontier records)", file=out)


def print_report(rows: list[dict[str, Any]], out: TextIO | None = None) -> None:
    if out is None:
        out = sys.stdout
    summaries = compute_frontier_summaries(rows)

    _print_scaling_table(summaries, out)
    _print_warmup_curve(summaries, out)
    _print_scale_curve(summaries, out)
    print("\nD. Interpretation warning:", file=out)
    print(
        "  Diagnostic only; this measures proposal-prior quality, "
        "not safe exclusive filtering.",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize temporal relative-authority frontier JSONL output."
    )
    parser.add_argument("--jsonl", required=True, help="Path to a batch JSONL report")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.jsonl)
    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
