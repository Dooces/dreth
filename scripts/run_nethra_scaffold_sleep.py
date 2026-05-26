#!/usr/bin/env python3
"""Run offline NethraScaffoldSleep over exported JSONL and memory records."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_scaffold_sleep import (  # noqa: E402
    NethraScaffoldSleep,
    NethraScaffoldSleepSummary,
    write_scaffold_sleep_jsonl,
)


def _fmt_summary(summary: NethraScaffoldSleepSummary) -> str:
    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append(f"── {title} {'─' * max(0, 60 - len(title))}")

    lines.append("NethraScaffoldSleep Report")
    lines.append("=" * 64)

    h("A. scaffold inventory")
    lines.append(f"  scaffold_nethras:       {summary.scaffold_nethras}")
    lines.append(f"  compositions:           {summary.compositions}")
    lines.append(f"  abstractions:           {summary.abstractions}")
    lines.append(f"  role_maps:              {summary.role_maps}")

    h("B. full-role coverage")
    lines.append(f"  tareth records seen:              {summary.tareth_records_seen}")
    lines.append(f"  trass records seen:               {summary.trass_records_seen}")
    lines.append(f"  background records seen:          {summary.background_records_seen}")
    lines.append(f"  unresolved records seen:          {summary.unresolved_records_seen}")
    lines.append(f"  authority/debt records seen:      {summary.authority_debt_records_seen}")
    lines.append(f"  stable/best_available records seen:{summary.stable_best_available_records_seen}")

    h("C. role maps")
    if summary.role_shift_examples:
        for i, ex in enumerate(summary.role_shift_examples[:8], 1):
            lines.append(
                f"  [{i}] {ex.get('scaffold_id')} "
                f"{ex.get('from_role')}@{ex.get('from_context')} -> "
                f"{ex.get('to_role')}@{ex.get('to_context')}"
            )
    else:
        lines.append("  (no same-scaffold role shifts observed)")
    lines.append("  trass here / tareth there examples are included above when present.")
    lines.append("  background but later active examples are included above when present.")

    h("D. composition")
    if summary.composition_examples:
        for i, ex in enumerate(summary.composition_examples[:8], 1):
            lines.append(
                f"  [{i}] {ex.get('composition_id')} lower={len(ex.get('lower_scaffold_ids') or [])} "
                f"higher={ex.get('higher_scaffold_id')} "
                f"confidence={ex.get('confidence_as_familiarity')}"
            )
            lines.append(f"      {str(ex.get('evidence_summary', ''))[:140]}")
    else:
        lines.append("  (no lower nethra composition observed)")
    lines.append("  nethra-of-nethra candidates are scaffold compositions only.")

    h("E. abstraction")
    if summary.abstraction_counts_by_kind:
        for kind, count in sorted(summary.abstraction_counts_by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {kind:<32} {count}")
    else:
        lines.append("  (no scaffold abstractions emitted)")
    wanted = Counter(summary.abstraction_counts_by_kind)
    lines.append(f"  operator families:          {wanted.get('operator_family', 0)}")
    lines.append(f"  parent-signature families:  {wanted.get('parent_signature_family', 0)}")
    lines.append(f"  context-role families:      {wanted.get('context_role_family', 0)}")
    lines.append(f"  mixed-role families:        {wanted.get('mixed_role_family', 0)}")

    h("F. anti-hoarding diagnostics")
    lines.append(f"  raw records read:                  {summary.raw_records_read}")
    lines.append(f"  scaffold nethras emitted:          {summary.scaffold_nethras}")
    lines.append(f"  abstractions emitted:              {summary.abstractions}")
    lines.append(f"  broad generic debt count:          {summary.broad_generic_debt_count}")
    lines.append(f"  broad generic debt useful count:   {summary.broad_generic_debt_useful_count}  (must be 0)")

    h("G. authority boundary")
    lines.append(f"  authority_allowed_count: {summary.authority_allowed_count}  (must be 0)")
    lines.append(f"  behavior_effects:        {summary.behavior_effects}  (must be 0)")
    if summary.hidden_truth_fields_seen:
        lines.append(f"  hidden truth/debug fields ignored: {summary.hidden_truth_fields_seen}")
    else:
        lines.append("  hidden truth/debug fields ignored: none")

    h("H. warning")
    lines.append(f"  {summary.warning}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build persistent nethra scaffold sleep records from visible exports."
    )
    parser.add_argument("--jsonl", action="append", default=[], help="Batch/export JSONL input")
    parser.add_argument("--memory", action="append", default=[], help="Nethra memory JSONL input")
    parser.add_argument("--out", required=True, help="Output scaffold sleep JSONL")
    parser.add_argument("--summary", required=True, help="Output summary TXT")
    parser.add_argument("--max-scaffolds", type=int, default=5000)
    parser.add_argument("--max-compositions", type=int, default=1000)
    parser.add_argument("--max-abstractions", type=int, default=2000)
    args = parser.parse_args()

    inputs = [Path(p) for p in args.jsonl + args.memory]
    if not inputs:
        print("ERROR: at least one --jsonl or --memory input is required", file=sys.stderr)
        sys.exit(1)

    sleep = NethraScaffoldSleep()
    rows = sleep.load_rows(*inputs)
    scaffold_nethras = sleep.extract_scaffold_nethras(rows, max_scaffolds=args.max_scaffolds)
    role_maps = sleep.build_role_maps(scaffold_nethras)
    compositions = sleep.build_compositions(
        scaffold_nethras,
        max_compositions=args.max_compositions,
    )
    abstractions = sleep.build_abstractions(
        scaffold_nethras,
        compositions,
        max_abstractions=args.max_abstractions,
    )
    summary = sleep.summarize(rows, scaffold_nethras, compositions, abstractions, role_maps)

    write_scaffold_sleep_jsonl(args.out, scaffold_nethras, role_maps, compositions, abstractions)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_fmt_summary(summary) + "\n")

    print(f"rows: {len(rows)}")
    print(f"scaffold_nethras: {summary.scaffold_nethras}")
    print(f"role_maps: {summary.role_maps}")
    print(f"compositions: {summary.compositions}")
    print(f"abstractions: {summary.abstractions}")
    print(f"authority_allowed_count: {summary.authority_allowed_count}")
    print(f"behavior_effects: {summary.behavior_effects}")
    print(f"broad_generic_debt_useful_count: {summary.broad_generic_debt_useful_count}")
    if summary.authority_allowed_count != 0 or summary.behavior_effects != 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
