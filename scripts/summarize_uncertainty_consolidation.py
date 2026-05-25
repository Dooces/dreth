#!/usr/bin/env python3
from __future__ import annotations

"""Summarize visible-evidence uncertainty consolidation from JSONL reports."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.uncertainty_consolidation import (
    cluster_uncertainty_cases,
    extract_uncertainty_cases_from_rows,
    summarize_clusters,
)


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


def _runtime_assist_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    fields = [
        "consolidation_assists_total",
        "assist_prioritize_attention",
        "assist_preserve_alternatives",
        "assist_request_probe",
        "assist_increase_monitoring",
        "assist_repair_priority_bonus",
        "assist_noops",
    ]
    out = {field: 0 for field in fields}
    for row in rows:
        for field in fields:
            out[field] += int(row.get(field) or 0)
    return out


def _relation_by_var(rows: Iterable[dict[str, Any]]) -> dict[int, Counter[str]]:
    rels: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        evaluation = row.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            continue
        behavior = evaluation.get("blind_challenge_behavior") or {}
        if not isinstance(behavior, dict):
            continue
        for item in behavior.get("per_var") or ():
            if not isinstance(item, dict):
                continue
            var = item.get("var")
            if var is None:
                continue
            rels[int(var)][str(item.get("relation_type") or "unknown")] += 1
    return rels


def print_report(rows: list[dict[str, Any]], out: TextIO) -> None:
    cases = extract_uncertainty_cases_from_rows(rows)
    clusters = cluster_uncertainty_cases(cases)
    summary = summarize_clusters(clusters)
    assist_summary = _runtime_assist_summary(rows)

    print("Uncertainty Consolidation Report", file=out)
    print("Warning: hidden truth is not used for clustering or runtime assist.", file=out)
    print(file=out)

    print("A. Raw uncertainty cases", file=out)
    print(f"  count: {len(cases)}", file=out)
    case_signals = Counter(sig for case in cases for sig in case.active_signals)
    for signal, count in case_signals.most_common():
        print(f"  {signal}: {count}", file=out)
    print(file=out)

    print("B. Consolidated clusters", file=out)
    print(f"  clusters: {summary['uncertainty_clusters']}", file=out)
    print(f"  max_cluster_size: {summary['max_cluster_size']}", file=out)
    print(f"  avg_cluster_size: {summary['avg_cluster_size']:.2f}", file=out)
    for kind, count in sorted(summary.get("handle_kinds", {}).items()):
        print(f"  {kind}: {count}", file=out)
    print(file=out)

    print("C. Cluster examples", file=out)
    if not clusters:
        print("  none", file=out)
    for cluster in clusters[:10]:
        print(
            f"  {cluster.cluster_id}: vars={list(cluster.vars)} "
            f"kind={cluster.proposed_handle_kind} "
            f"probe={cluster.proposed_next_probe_family}",
            file=out,
        )
        print(f"    evidence: {cluster.evidence_summary}", file=out)
    print(file=out)

    print("D. Compression ratio", file=out)
    raw_ratio = len(cases) / len(clusters) if clusters else 0.0
    print(f"  cases / clusters: {raw_ratio:.3f}", file=out)
    if clusters and len(cases) <= len(clusters):
        print("  No reduction observed; uncertainty remained mostly per-variable.", file=out)
    print(file=out)

    print("E. Runtime assist summary", file=out)
    for field, value in assist_summary.items():
        print(f"  {field}: {value}", file=out)
    modes = Counter(str(row.get("uncertainty_consolidation_mode") or "off") for row in rows)
    print("  modes: " + " ".join(f"{k}={v}" for k, v in sorted(modes.items())), file=out)
    print(file=out)

    print("F. Post-hoc interpretation", file=out)
    print("  Uses relation_type only after clustering.", file=out)
    rel_by_var = _relation_by_var(rows)
    if not clusters:
        print("  none", file=out)
    for cluster in clusters[:10]:
        rel_counts: Counter[str] = Counter()
        for var in cluster.vars:
            rel_counts.update(rel_by_var.get(var, {}))
        rel_text = " ".join(f"{rel}={count}" for rel, count in rel_counts.most_common())
        print(f"  {cluster.cluster_id}: {rel_text or 'unknown'}", file=out)
    print(file=out)

    print("G. Hidden-truth warning", file=out)
    print("  Clustering and runtime assists use visible agent evidence only.", file=out)
    print("  Post-hoc relation_type interpretation is separate from clustering.", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize uncertainty consolidation JSONL output."
    )
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    print_report(load_jsonl(args.jsonl), out=__import__("sys").stdout)


if __name__ == "__main__":
    main()
