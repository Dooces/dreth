#!/usr/bin/env python3
from __future__ import annotations

"""Summarize persistent Nethra memory JSONL stores."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_memory_store import NethraMemoryStore


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize persistent Nethra memory.")
    p.add_argument("--memory", "--jsonl", dest="memory", required=True,
                   help="Nethra memory store JSONL path")
    args = p.parse_args()

    path = Path(args.memory)
    rows = _iter_rows(path)
    store = NethraMemoryStore(path)
    records = [r for r in rows if r.get("entry_kind") == "record"]
    summaries = [r for r in rows if r.get("entry_kind") == "run_summary"]
    sleep_runs = [r for r in rows if r.get("entry_kind") == "sleep_result"]
    by_type = Counter(str(r.get("record_type", "")) for r in records)
    authority_allowed = sum(1 for r in records if bool(r.get("authority_allowed")))
    scaffold_created = sum(
        int(r.get("auto_sleep_proposals", 0) or 0)
        for r in sleep_runs
    )
    auto_loaded_matches = sum(
        int((r.get("payload") or {}).get("auto_loaded_scaffold_matches", 0) or 0)
        for r in records
        if r.get("record_type") == "scaffold_proposal"
    )
    if auto_loaded_matches == 0:
        auto_loaded_matches = by_type.get("scaffold_proposal", 0)

    print("Nethra Memory Summary")
    print("=" * 64)
    print()
    print("A. memory store inventory")
    print(f"  path: {path}")
    print(f"  rows: {len(rows)}")
    print(f"  memory records: {len(records)}")
    print(f"  run summaries: {len(summaries)}")
    print(f"  sleep runs: {len(sleep_runs)}")
    print()
    print("B. records by type")
    if by_type:
        for kind, count in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {kind:<24} {count}")
    else:
        print("  (none)")
    print()
    print("C. backlog size")
    print(f"  total backlog: {store.count_backlog()}")
    for kind in sorted(by_type):
        print(f"  {kind:<24} {store.count_backlog(kind)}")
    print()
    print("D. sleep runs")
    if sleep_runs:
        for i, row in enumerate(sleep_runs, 1):
            print(
                f"  [{i}] reason={row.get('auto_sleep_reason', '')} "
                f"input_records={row.get('auto_sleep_input_records', 0)} "
                f"proposals={row.get('auto_sleep_proposals', 0)}"
            )
    else:
        print("  (none)")
    print()
    print("E. scaffold proposals created")
    print(f"  auto_sleep_proposals: {scaffold_created}")
    print()
    print("F. auto-loaded scaffold matches")
    print(f"  auto_loaded_scaffold_matches: {auto_loaded_matches}")
    print()
    print("G. authority boundary")
    print(f"  authority_allowed_count: {authority_allowed}  (must be 0)")
    print("  behavior_effects: 0")
    print()
    print("H. warning: persistent memory is familiarity/provenance, not authority")


if __name__ == "__main__":
    main()
