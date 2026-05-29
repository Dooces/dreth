#!/usr/bin/env python3
"""Offline MemorySleep consolidator: builds scaffold proposals from exported runtime memory.

Usage:
    python scripts/run_memory_sleep.py \\
        --jsonl reports/background_nethra_compare_record.jsonl \\
        --out reports/memory_sleep_proposals.jsonl \\
        --summary reports/memory_sleep_summary.txt

Optional:
    --max-proposals 2000
    --max-sources-per-proposal 500
    --min-sources 2
    --posthoc-relation-type-report

Output:
    proposals JSONL  — one ScaffoldProposal per line
    summary TXT      — human-readable report

Core invariant:
  Sleep proposals are familiarity scaffolds only.
  They are not operational authority.
  They do not imply tareth in the current context.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.memory_sleep import MemorySleepConsolidator, MemorySleepSummary, ScaffoldProposal


# ── Summary text formatter ────────────────────────────────────────────────────

def _fmt_summary(summary: MemorySleepSummary, posthoc: bool = False) -> str:
    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append(f"── {title} {'─' * max(0, 60 - len(title))}")

    lines.append("MemorySleep Consolidation Report")
    lines.append("=" * 64)

    # ── A. Input inventory ────────────────────────────────────────────────────
    h("A. Input inventory")
    lines.append(f"  rows read:                    {summary.input_rows}")
    lines.append(f"  background records seen:      {summary.background_records_seen}")
    lines.append(f"  context-role records seen:    {summary.context_role_records_seen}")
    lines.append(f"  uncertainty records seen:     {summary.uncertainty_records_seen}")
    lines.append(f"  authority records seen:       {summary.authority_records_seen}")
    lines.append(f"  temporal records seen:        {summary.temporal_records_seen}")

    # ── B. Source field diagnostics ───────────────────────────────────────────
    h("B. Source field diagnostics")
    if not summary.zero_or_flat_source_fields:
        lines.append("  (no zero/missing/flat source fields detected)")
    else:
        for field_info in summary.zero_or_flat_source_fields:
            if field_info.startswith("MISMATCH:"):
                lines.append(f"  [MISMATCH] {field_info[9:].strip()}")
            else:
                lines.append(f"  [ZERO/FLAT] {field_info}")
    if summary.hidden_truth_fields_seen:
        lines.append(f"  hidden-truth-like fields present but ignored: "
                     f"{summary.hidden_truth_fields_seen}")
    else:
        lines.append("  hidden-truth-like fields present but ignored: none")
    if not posthoc:
        lines.append("  relation_type: not used (posthoc mode off)")

    # ── C. Scaffold proposals ─────────────────────────────────────────────────
    h("C. Scaffold proposals")
    lines.append(f"  proposal count:               {len(summary.proposals)}")
    lines.append(f"  proposals by kind:")
    for kind, count in sorted(summary.proposals_by_kind.items(), key=lambda x: -x[1]):
        lines.append(f"    {kind:<40} {count}")
    lines.append(f"  avg sources per proposal:     {summary.avg_sources_per_proposal:.4f}")
    lines.append(f"  max sources per proposal:     {summary.max_sources_per_proposal}")
    lines.append(f"  compression ratio:            {summary.compression_ratio:.4f}")
    lines.append(f"    (background_records_seen / proposal_count)")

    # ── D. Familiarity not authority ──────────────────────────────────────────
    h("D. Familiarity not authority")
    lines.append(f"  authority_allowed_count:      {summary.authority_allowed_count}  (must be 0)")
    if summary.authority_allowed_count != 0:
        lines.append(
            f"  *** INVARIANT VIOLATION: authority_allowed_count={summary.authority_allowed_count} ***"
        )
    else:
        lines.append("  INVARIANT OK: no proposal issued any authority.")
    # Runtime use distribution
    use_counts: Counter[str] = Counter(p.suggested_runtime_use for p in summary.proposals)
    lines.append(f"  suggested_runtime_use distribution:")
    for use, cnt in sorted(use_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {use:<30} {cnt}")
    # Action relevance distribution
    action_scores = [p.action_relevance_score for p in summary.proposals]
    if action_scores:
        lines.append(f"  action_relevance_score:")
        lines.append(f"    min={min(action_scores):.4f}  "
                     f"max={max(action_scores):.4f}  "
                     f"mean={sum(action_scores)/len(action_scores):.4f}")
    else:
        lines.append("  action_relevance_score: (no proposals)")

    # ── E. Proposal examples ──────────────────────────────────────────────────
    h("E. Proposal examples (top 10 by recurrence_count)")
    top = sorted(summary.proposals, key=lambda p: -p.recurrence_count)[:10]
    if not top:
        lines.append("  (no proposals)")
    for i, p in enumerate(top, 1):
        lines.append(f"  [{i:2d}] id={p.proposal_id}")
        lines.append(f"       kind={p.kind}")
        lines.append(f"       vars={p.vars[:10]}")
        lines.append(f"       contexts={p.contexts[:3]}")
        lines.append(f"       signatures={p.common_signatures[:3]}")
        lines.append(f"       source_edges={p.common_source_edges[:3]}")
        lines.append(f"       role_patterns={p.role_patterns[:4]}")
        lines.append(f"       source_kinds={p.source_kinds[:3]}")
        lines.append(f"       source_ids (sample)={p.source_record_ids[:3]}")
        lines.append(f"       recurrence_count={p.recurrence_count}  "
                     f"runs_seen={p.runs_seen}  seeds_seen={p.seeds_seen}")
        lines.append(f"       confidence_as_familiarity={p.confidence_as_familiarity:.4f}")
        lines.append(f"       suggested_runtime_use={p.suggested_runtime_use}")
        if p.warnings:
            lines.append(f"       warnings={p.warnings}")
        lines.append(f"       evidence={p.evidence_summary[:120]}")

    # ── F. Giant cluster decomposition ───────────────────────────────────────
    h("F. Giant cluster decomposition")
    giant_proposals = [p for p in summary.proposals if p.kind == 'giant_cluster_subfamily']
    giant_sources_total = sum(p.recurrence_count for p in giant_proposals)
    low_spec = [p for p in summary.proposals if 'low_specificity' in p.warnings]
    no_use = [p for p in summary.proposals if p.suggested_runtime_use == 'no_runtime_use']
    lines.append(f"  giant source records seen:    {giant_sources_total}")
    lines.append(f"  giant proposals emitted:      {len(giant_proposals)}")
    lines.append(f"  low-specificity proposals:    {len(low_spec)}")
    lines.append(f"  no_runtime_use proposals:     {len(no_use)}")
    if giant_proposals:
        giant_vars = sorted(set(v for p in giant_proposals for v in p.vars))
        lines.append(f"  subfamilies produced for vars: {giant_vars[:20]}")

    # ── G. Hidden-truth guard ─────────────────────────────────────────────────
    h("G. Hidden-truth guard")
    lines.append(f"  hidden truth fields ignored:  {summary.hidden_truth_fields_seen or 'none'}")
    if posthoc:
        lines.append("  relation_type: POSTHOC mode enabled — field read for report only, "
                     "not used in proposals")
    else:
        lines.append("  relation_type: ignored (posthoc mode off)")
    lines.append(f"  warning_count:                {summary.warning_count}")

    # ── H. Warning ────────────────────────────────────────────────────────────
    h("H. Warning")
    lines.append("  Sleep proposals are familiarity scaffolds only.")
    lines.append("  They are not operational authority.")
    lines.append("  They do not imply tareth in the current context.")
    lines.append("  recurrence/frequency is familiarity signal, not proof.")
    lines.append("  authority_allowed is always False on every proposal.")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline MemorySleep consolidator: build scaffold proposals from "
                    "exported runtime memory."
    )
    p.add_argument("--jsonl", required=True, help="Path to batch_run JSONL output file")
    p.add_argument("--out", required=True, help="Output path for proposals JSONL")
    p.add_argument("--summary", required=True, help="Output path for summary TXT")
    p.add_argument("--max-proposals", type=int, default=2000,
                   help="Maximum number of proposals to emit (default: 2000)")
    p.add_argument("--max-sources-per-proposal", type=int, default=500,
                   help="Maximum source records per proposal (default: 500)")
    p.add_argument("--min-sources", type=int, default=2,
                   help="Minimum source records to form a proposal (default: 2)")
    p.add_argument("--posthoc-relation-type-report", action="store_true",
                   help="Enable posthoc relation-type reporting (off by default)")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    consolidator = MemorySleepConsolidator()

    print(f"Loading rows from {jsonl_path} ...")
    rows = consolidator.load_jsonl_rows(jsonl_path)
    print(f"  Loaded {len(rows)} row(s).")

    print("Extracting records ...")
    bg = consolidator.extract_background_records(rows)
    cr = consolidator.extract_context_role_records(rows)
    unc = consolidator.extract_uncertainty_records(rows)
    auth = consolidator.extract_authority_records(rows)
    temp = consolidator.extract_temporal_records_if_available(rows)
    mem = consolidator.extract_nethra_memory_records(rows)
    exp = consolidator.extract_experience_events(rows)
    print(f"  background: {len(bg)}  context-role: {len(cr)}  "
          f"uncertainty: {len(unc)}  authority: {len(auth)}  temporal: {len(temp)} "
          f"memory: {len(mem)}  experience: {len(exp)}")

    print("Building proposals ...")
    proposals = consolidator.build_proposals(
        bg, cr, unc, auth, temp,
        min_sources=args.min_sources,
        max_proposals=args.max_proposals,
        max_sources_per_proposal=args.max_sources_per_proposal,
        posthoc_relation_type=args.posthoc_relation_type_report,
    )
    sleep_products = consolidator.build_sleep_products(
        mem,
        exp,
        min_sources=1,
        max_products=args.max_proposals,
    )
    print(f"  {len(proposals)} scaffold proposal(s) built.")
    print(f"  {len(sleep_products)} sleep product(s) built.")

    print("Summarizing ...")
    summary = consolidator.summarize(rows, bg, cr, unc, auth, temp, proposals)

    # Write proposals JSONL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        for prop in proposals:
            fh.write(json.dumps(prop.to_dict()) + "\n")
        for product in sleep_products:
            fh.write(json.dumps(product.to_dict()) + "\n")
    print(
        f"  Proposals written to {out_path}  "
        f"({len(proposals)} scaffold proposals, {len(sleep_products)} sleep products)"
    )

    # Write summary TXT
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = _fmt_summary(summary, posthoc=args.posthoc_relation_type_report)
    with open(summary_path, 'w') as fh:
        fh.write(summary_text + "\n")
    print(f"  Summary written to {summary_path}")

    # Print key invariant check
    print()
    print(f"authority_allowed_count: {summary.authority_allowed_count}  (must be 0)")
    if summary.authority_allowed_count != 0:
        print("*** INVARIANT VIOLATION: authority_allowed_count must be 0 ***",
              file=sys.stderr)
        sys.exit(2)
    print(f"compression_ratio: {summary.compression_ratio:.4f}")
    print(f"proposals: {len(proposals)}")
    print(f"sleep_products: {len(sleep_products)}")
    if summary.zero_or_flat_source_fields:
        print("source field diagnostics:")
        for f in summary.zero_or_flat_source_fields:
            print(f"  {f}")


if __name__ == "__main__":
    main()
