#!/usr/bin/env python3
"""Compact raw nethra memory JSONL into a canonical nethra mind graph.

Supports both full compaction and generation-delta compaction:

Full compaction (first generation, no prior mind):
    python scripts/compact_nethra_memory.py \\
        --delta-input  reports/gen0_delta.jsonl \\
        --out          reports/mind_gen0.jsonl \\
        --report       reports/mind_gen0_report.txt

Generation-delta compaction (load previous mind + ingest only new delta):
    python scripts/compact_nethra_memory.py \\
        --previous-mind reports/mind_gen0.jsonl \\
        --delta-input   reports/gen1_delta.jsonl \\
        --out           reports/mind_gen1.jsonl \\
        --report        reports/mind_gen1_report.txt

Legacy single-file mode (--input is an alias for --delta-input):
    python scripts/compact_nethra_memory.py \\
        --input  reports/raw_store.jsonl \\
        --out    reports/mind.jsonl \\
        --report reports/mind_report.txt

WARNING: the compacted mind is not authority. It may provide ranking/probe
hints only. authority_effect_count is always zero.

IMPORTANT: mind-derived rows (entry_kind in {nethra_mind_node, nethra_mind_edge,
nethra_mind_summary} or source in {mind, sleep_derived_mind, compacted}) are
silently rejected from the delta stream. They are compaction outputs, not evidence.
Do not re-ingest a previous mind as delta input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_mind_store import NethraMindStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact raw nethra memory JSONL into a canonical mind graph"
    )
    # Primary args
    parser.add_argument(
        "--delta-input",
        default="",
        metavar="PATH",
        help="World-backed delta JSONL to ingest (records + experience_events + sleep_products)",
    )
    parser.add_argument(
        "--previous-mind",
        default="",
        metavar="PATH",
        help="Path to previous compacted mind JSONL to load as base (not re-ingested as evidence)",
    )
    parser.add_argument("--out", required=True, help="Output compacted mind JSONL path")
    parser.add_argument("--report", required=True, help="Output report text path")
    # Legacy alias
    parser.add_argument(
        "--input",
        default="",
        metavar="PATH",
        help="Alias for --delta-input (legacy single-file mode)",
    )
    parser.add_argument(
        "--raw-sidecar",
        default="",
        metavar="PATH",
        help="Optional path to write raw experience_event rows read from delta",
    )
    args = parser.parse_args()

    # Resolve delta input: --delta-input takes precedence over --input
    delta_input = args.delta_input or args.input
    if not delta_input:
        print("error: --delta-input (or --input) is required", file=sys.stderr)
        sys.exit(1)

    delta_path = Path(delta_input)
    if not delta_path.exists():
        print(f"error: delta input file not found: {delta_path}", file=sys.stderr)
        sys.exit(1)

    store = NethraMindStore()

    # Load previous mind first (if given) — this does NOT count as fresh evidence
    if args.previous_mind:
        prev_path = Path(args.previous_mind)
        if prev_path.exists():
            prev_loaded = store.load(prev_path)
            print(f"Previous mind loaded: {prev_loaded:>8} entries  <- {prev_path}")
        else:
            print(
                f"warning: --previous-mind not found, starting from empty: {prev_path}",
                file=sys.stderr,
            )

    # Snapshot node/edge counts after loading previous mind, before delta ingestion
    store.snapshot_delta_start()

    generation = 0
    sidecar_rows: list[dict] = []

    with open(delta_path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            entry_kind = str(row.get("entry_kind", ""))
            if entry_kind == "record":
                store.ingest_record(row, line_no=line_no, generation=generation)
            elif entry_kind == "experience_event":
                store.ingest_experience_event(row, line_no=line_no, generation=generation)
                if args.raw_sidecar:
                    sidecar_rows.append(row)
            elif entry_kind == "sleep_product":
                store.ingest_sleep_product(row, line_no=line_no, generation=generation)
            elif entry_kind in ("run_summary", "sleep_result"):
                generation += 1
            elif entry_kind in ("nethra_mind_node", "nethra_mind_edge", "nethra_mind_summary"):
                # Mind-derived rows must never be re-ingested as delta evidence.
                # Route to ingest_record so the policy check increments the rejection counter.
                store.ingest_record(row, line_no=line_no, generation=generation)

    summary = store.write_compact(args.out)
    store.write_report(args.report)

    if args.raw_sidecar and sidecar_rows:
        sidecar_path = Path(args.raw_sidecar)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w") as fh:
            for row in sidecar_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Sidecar events:     {len(sidecar_rows):>8}  -> {sidecar_path}")

    print(f"Raw rows read:      {summary['raw_rows_read']:>8}")
    print(f"Rejected (mind):    {summary['rows_rejected_mind_derived']:>8}")
    print(f"Rejected (compact): {summary['rows_rejected_compacted']:>8}")
    print(f"Rows ingested:      {summary['rows_ingested']:>8}")
    print(f"Nodes before:       {summary['nodes_before']:>8}")
    print(f"Nodes after:        {summary['nodes_after']:>8}")
    print(f"Edges after:        {summary['edges_after']:>8}")
    print(f"Exact folds:        {summary['exact_folds']:>8}")
    print(f"Structural folds:   {summary['structural_folds']:>8}")
    print(f"Assimilation folds: {summary['assimilation_folds']:>8}")
    print(f"Sleep folded:       {summary['sleep_products_folded']:>8}")
    print(f"Nodes pruned:       {summary['nodes_pruned']:>8}")
    print(f"Residuals kept:     {summary['residuals_kept']:>8}")
    print(f"Compression:        {summary['compression_ratio']:>8.2f}x")
    print(f"Mind bytes:         {summary['active_mind_bytes']:>8}")
    print(f"Mind output:        {args.out}")
    print(f"Report:             {args.report}")
    astats = summary.get("assimilation_stats", {})
    if astats:
        print("Assimilation breakdown:")
        for k, v in sorted(astats.items()):
            print(f"  {k:<22} {v:>6}")
    print()
    print("WARNING: compacted mind is not authority.")
    print("  Assist mode may use mind nodes as ranking/probe hints only.")
    print("  authority_effects must remain zero in all runtime metrics.")


if __name__ == "__main__":
    main()
