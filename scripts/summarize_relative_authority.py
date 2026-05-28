from __future__ import annotations

"""Offline summarizer for --relative-authority-report JSONL output.

Usage:
    python scripts/summarize_relative_authority.py --jsonl reports/policy_report.jsonl

WARNING: diagnostic only; graph summaries are projected from existing ledger
artifacts and do not drive runtime behavior.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, TextIO


RELATION_TYPES = [
    "depends_on",
    "coactive_with",
    "exception_to",
    "substitutes_for",
    "conflicts_with",
    "shares_node",
    "beats_in_context",
    "loses_in_context",
]


@dataclass
class GroupSummary:
    schedule: str
    policy: str
    runs: int = 0
    total_nodes: float = 0.0
    total_relations: float = 0.0
    total_authority_records: float = 0.0
    relation_types: Counter[str] = field(default_factory=Counter)

    @property
    def avg_nodes(self) -> float:
        return self.total_nodes / self.runs if self.runs else 0.0

    @property
    def avg_relations(self) -> float:
        return self.total_relations / self.runs if self.runs else 0.0

    @property
    def avg_authority_records(self) -> float:
        return self.total_authority_records / self.runs if self.runs else 0.0

    @property
    def exception_density(self) -> float:
        return self.relation_types["exception_to"] / max(1.0, self.total_relations)

    @property
    def shares_node_density(self) -> float:
        return self.relation_types["shares_node"] / max(1.0, self.total_relations)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
    source_edge_ranker = row.get("source_edge_ranker")
    probe_proposer = row.get("probe_proposer")
    if source_edge_ranker is not None and probe_proposer is not None:
        return f"{source_edge_ranker}/{probe_proposer}"
    return str(row.get("policy") or "unknown/unknown")


def compute_group_summaries(rows: Iterable[dict[str, Any]]) -> list[GroupSummary]:
    groups: dict[tuple[str, str], GroupSummary] = {}
    for row in rows:
        schedule = str(row.get("schedule") or "unknown")
        policy = policy_label(row)
        key = (schedule, policy)
        summary = groups.setdefault(key, GroupSummary(schedule=schedule, policy=policy))
        summary.runs += 1
        summary.total_nodes += _as_float(row.get("relative_authority_nodes"))
        summary.total_relations += _as_float(row.get("relative_authority_relations"))
        summary.total_authority_records += _as_float(row.get("relative_authority_records"))
        relation_counts = row.get("relative_authority_relation_types") or {}
        if isinstance(relation_counts, dict):
            for relation_type, count in relation_counts.items():
                summary.relation_types[str(relation_type)] += int(_as_float(count))
    return [groups[key] for key in sorted(groups)]


def _node_id_from_example(example: Any) -> str:
    if isinstance(example, dict):
        return str(example.get("node_id") or example.get("id") or "")
    text = str(example)
    node_id, sep, maybe_score = text.rpartition(":")
    if sep:
        try:
            float(maybe_score)
            return node_id
        except ValueError:
            pass
    return text


def compute_top_example_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        examples = row.get("relative_authority_top_examples") or []
        if not isinstance(examples, list):
            continue
        for example in examples:
            node_id = _node_id_from_example(example)
            if node_id:
                counts[node_id] += 1
    return counts


def _print_group_size(summaries: list[GroupSummary], out: TextIO) -> None:
    print("A. Mean graph size by schedule/policy:", file=out)
    print(
        f"  {'schedule':<18} {'source_edge_ranker/probe_proposer':<34} "
        f"{'runs':>4} {'avg_nodes':>10} {'avg_relations':>13} "
        f"{'avg_authority_records':>22}",
        file=out,
    )
    for summary in summaries:
        print(
            f"  {summary.schedule:<18} {summary.policy:<34} "
            f"{summary.runs:>4} {summary.avg_nodes:>10.2f} "
            f"{summary.avg_relations:>13.2f} "
            f"{summary.avg_authority_records:>22.2f}",
            file=out,
        )
    if not summaries:
        print("  (no normal run records)", file=out)


def _print_relation_distribution(summaries: list[GroupSummary], out: TextIO) -> None:
    print("\nB. Relation type distribution by schedule/policy:", file=out)
    print(
        f"  {'schedule':<18} {'source_edge_ranker/probe_proposer':<34} "
        + " ".join(f"{name:>18}" for name in RELATION_TYPES),
        file=out,
    )
    for summary in summaries:
        counts = " ".join(
            f"{summary.relation_types[relation_type]:>18}"
            for relation_type in RELATION_TYPES
        )
        print(f"  {summary.schedule:<18} {summary.policy:<34} {counts}", file=out)
    if not summaries:
        print("  (no normal run records)", file=out)


def _print_top_examples(counts: Counter[str], out: TextIO) -> None:
    print("\nC. Top authority examples frequency:", file=out)
    print(f"  {'node_id':<54} {'count':>8}", file=out)
    for node_id, count in counts.most_common(20):
        print(f"  {node_id:<54} {count:>8}", file=out)
    if not counts:
        print("  (no top authority examples)", file=out)


def _print_density(
    title: str,
    column_name: str,
    summaries: list[GroupSummary],
    attr: str,
    out: TextIO,
) -> None:
    print(f"\n{title}", file=out)
    print(
        f"  {'schedule':<18} {'source_edge_ranker/probe_proposer':<34} "
        f"{column_name:>18}",
        file=out,
    )
    for summary in summaries:
        print(
            f"  {summary.schedule:<18} {summary.policy:<34} "
            f"{getattr(summary, attr):>18.4f}",
            file=out,
        )
    if not summaries:
        print("  (no normal run records)", file=out)


def print_report(rows: list[dict[str, Any]], out: TextIO | None = None) -> None:
    if out is None:
        out = sys.stdout
    summaries = compute_group_summaries(rows)
    top_examples = compute_top_example_counts(rows)

    _print_group_size(summaries, out)
    _print_relation_distribution(summaries, out)
    _print_top_examples(top_examples, out)
    _print_density(
        "D. Exception density:",
        "exception_to/relations",
        summaries,
        "exception_density",
        out,
    )
    _print_density(
        "E. Composite/shared-node density:",
        "shares_node/relations",
        summaries,
        "shares_node_density",
        out,
    )
    print("\nF. Interpretation warning:", file=out)
    print(
        "  Diagnostic only; graph is projected from existing ledger artifacts "
        "and does not drive behavior.",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize --relative-authority-report JSONL output offline."
    )
    parser.add_argument("--jsonl", required=True, help="Path to a batch JSONL report")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.jsonl)
    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
