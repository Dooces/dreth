#!/usr/bin/env python3
from __future__ import annotations

"""Compare uncertainty consolidation off/shadow/assist JSONL runs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.learner.uncertainty_consolidation import (
    cluster_uncertainty_cases,
    extract_uncertainty_cases_from_rows,
    summarize_clusters,
)


BEHAVIOR_FIELDS = (
    "interventions",
    "full_audits",
    "revocations",
    "unique_fails",
    "quality_cost",
    "temporal_frontier_chosen_source_edge_recall",
    "temporal_frontier_recall_lift",
    "dormant_total",
    "vars_open_novelty",
)

LOWER_IS_BETTER = {
    "interventions",
    "full_audits",
    "revocations",
    "unique_fails",
    "quality_cost",
    "vars_open_novelty",
}
HIGHER_IS_BETTER = {
    "temporal_frontier_chosen_source_edge_recall",
    "temporal_frontier_recall_lift",
    "dormant_total",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
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


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _revocations(row: dict[str, Any]) -> float:
    if "revocations" in row:
        return _as_float(row.get("revocations"))
    revoked = row.get("revoked_by_dist")
    if isinstance(revoked, dict):
        return float(sum(_as_float(v) for v in revoked.values()))
    return 0.0


def _metric(row: dict[str, Any], field: str) -> float:
    if field == "revocations":
        return _revocations(row)
    if field == "unique_fails":
        return _as_float(row.get("unique_fails", row.get("total_unique_failures")))
    if field == "interventions":
        return _as_float(row.get("interventions", row.get("iv")))
    return _as_float(row.get(field))


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    config = row.get("config") if isinstance(row.get("config"), dict) else {}
    source = {**config, **row}
    return (
        source.get("schedule"),
        source.get("n_vars"),
        source.get("cycles"),
        source.get("seed"),
        source.get("settle_cycles"),
        source.get("noise_sigma"),
    )


def _by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        out[_key(row)] = row
    return out


def _avg_delta(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    rows: Iterable[dict[str, Any]],
    field: str,
) -> float:
    deltas = [
        _metric(row, field) - _metric(baseline[key], field)
        for row in rows
        for key in [_key(row)]
        if key in baseline
    ]
    return mean(deltas) if deltas else 0.0


def _off_shadow_equal(
    off: dict[tuple[Any, ...], dict[str, Any]],
    shadow: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for key in sorted(set(off) & set(shadow)):
        for field in BEHAVIOR_FIELDS:
            if _metric(off[key], field) != _metric(shadow[key], field):
                mismatches.append(
                    f"{key} {field}: off={_metric(off[key], field)} "
                    f"shadow={_metric(shadow[key], field)}"
                )
    missing = sorted(set(off) ^ set(shadow))
    for key in missing:
        mismatches.append(f"{key} missing from one side")
    return not mismatches, mismatches


def _policy(row: dict[str, Any]) -> str:
    return str(row.get("uncertainty_assist_policy") or "all")


def _cluster_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = extract_uncertainty_cases_from_rows(rows)
    clusters = cluster_uncertainty_cases(cases)
    return summarize_clusters(clusters)


def _assist_warning(deltas: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    for field in LOWER_IS_BETTER:
        if deltas.get(field, 0.0) > 0:
            warnings.append(f"{field} worsened by {deltas[field]:.3f}")
    for field in HIGHER_IS_BETTER:
        if deltas.get(field, 0.0) < 0:
            warnings.append(f"{field} worsened by {deltas[field]:.3f}")
    return warnings


def _benefit_attributable(deltas: dict[str, float], warnings: list[str]) -> bool:
    if warnings:
        return False
    return (
        deltas.get("quality_cost", 0.0) < 0
        or deltas.get("revocations", 0.0) < 0
        or deltas.get("unique_fails", 0.0) < 0
        or deltas.get("temporal_frontier_chosen_source_edge_recall", 0.0) > 0
        or deltas.get("temporal_frontier_recall_lift", 0.0) > 0
    )


def print_report(
    off_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    assist_rows: list[dict[str, Any]],
    out: TextIO,
) -> None:
    off = _by_key(off_rows)
    shadow = _by_key(shadow_rows)
    equal, mismatches = _off_shadow_equal(off, shadow)

    print("Uncertainty Consolidation Mode Comparison", file=out)
    print(file=out)
    print("A. off vs shadow equality", file=out)
    print(f"  equal: {equal}", file=out)
    if mismatches:
        for line in mismatches[:20]:
            print(f"  mismatch: {line}", file=out)
        if len(mismatches) > 20:
            print(f"  ... +{len(mismatches) - 20} more", file=out)
    print(file=out)

    print("B. assist delta versus off", file=out)
    overall_deltas = {field: _avg_delta(off, assist_rows, field) for field in BEHAVIOR_FIELDS}
    for field in BEHAVIOR_FIELDS:
        print(f"  delta_{field}: {overall_deltas[field]:.3f}", file=out)
    print(file=out)

    print("C. cluster size distribution", file=out)
    cluster_summary = _cluster_summary(shadow_rows + assist_rows)
    dist = cluster_summary.get("cluster_size_distribution", {})
    if dist:
        for size, count in sorted(dist.items()):
            print(f"  size_{size}: {count}", file=out)
    else:
        print("  none", file=out)
    print(file=out)

    print("D. giant clusters", file=out)
    print(f"  giant_cluster_count: {cluster_summary.get('giant_cluster_count', 0)}", file=out)
    print(
        "  suppressed_giant_clusters: "
        f"{sum(int(row.get('giant_clusters_suppressed') or 0) for row in assist_rows)}",
        file=out,
    )
    print(
        "  assists_suppressed_by_specificity_gate: "
        f"{sum(int(row.get('assists_suppressed_by_specificity_gate') or 0) for row in assist_rows)}",
        file=out,
    )
    print(file=out)

    print("E. assist cost/benefit by policy", file=out)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assist_rows:
        by_policy[_policy(row)].append(row)
    for policy, rows in sorted(by_policy.items()):
        deltas = {field: _avg_delta(off, rows, field) for field in BEHAVIOR_FIELDS}
        warnings = _assist_warning(deltas)
        print(f"  policy={policy} runs={len(rows)}", file=out)
        print(
            "    costs: "
            f"budget={sum(int(row.get('assist_extra_budget_total') or 0) for row in rows)} "
            f"probes={sum(int(row.get('assist_extra_probe_total') or 0) for row in rows)} "
            f"preserved={sum(int(row.get('assist_preserved_alternative_total') or 0) for row in rows)} "
            f"priority={sum(int(row.get('assist_priority_hint_total') or 0) for row in rows)}",
            file=out,
        )
        delta_text = " ".join(f"{field}={deltas[field]:.3f}" for field in BEHAVIOR_FIELDS)
        print(f"    deltas: {delta_text}", file=out)
        print(
            "    benefit_attributable: "
            + ("yes" if _benefit_attributable(deltas, warnings) else "no"),
            file=out,
        )
    print(file=out)

    print("F. warnings", file=out)
    warnings = _assist_warning(overall_deltas)
    if not equal:
        warnings.append("off and shadow are not behavior-identical")
    if not warnings:
        print("  none", file=out)
    for warning in warnings:
        print(f"  WARNING: {warning}", file=out)
    if not _benefit_attributable(overall_deltas, warnings):
        print("  WARNING: assist benefit is not attributable from this comparison", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare off/shadow/assist uncertainty consolidation JSONL files."
    )
    parser.add_argument("--off", required=True)
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--assist", required=True, nargs="+")
    args = parser.parse_args()

    off_rows = load_jsonl(args.off)
    shadow_rows = load_jsonl(args.shadow)
    assist_rows: list[dict[str, Any]] = []
    for path in args.assist:
        assist_rows.extend(load_jsonl(path))
    print_report(off_rows, shadow_rows, assist_rows, sys.stdout)


if __name__ == "__main__":
    main()
