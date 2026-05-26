#!/usr/bin/env python3
"""Summarize scaffold memory match records from a batch_run JSONL output file.

Usage:
    python scripts/summarize_scaffold_memory.py \\
        --jsonl reports/scaffold_memory_record.jsonl

Report sections:
    A. Loaded proposals
    B. Match counts and match rate
    C. Matches by kind
    D. Unmatched record count
    E. Match examples
    F. Authority boundary (authority_allowed_count must be 0; behavior_effects must be 0)
    G. Warning: scaffold memory is familiarity/provenance only

WARNING:
    Scaffold memory is familiarity/provenance telemetry.
    It does NOT issue authority.
    It does NOT affect runtime behavior.
    It does NOT imply tareth, trigger repair, or suppress skips.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sum_int(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(int(r.get(key, 0) or 0) for r in rows)


def _sum_float(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows)


def _mean_float(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _agg_kind_counts(rows: List[Dict[str, Any]], key: str) -> Counter:
    agg: Counter = Counter()
    for row in rows:
        d = row.get(key) or {}
        if isinstance(d, dict):
            agg.update({k: int(v) for k, v in d.items()})
    return agg


def _collect_examples(rows: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    seen: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for row in rows:
        for ex in (row.get("scaffold_memory_match_examples") or []):
            if not isinstance(ex, dict):
                continue
            pid = ex.get("matched_proposal_id", "")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            seen.append(ex)
            if len(seen) >= limit:
                return seen
    return seen


def _mode_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("scaffold_memory_mode") == mode]


def summarize(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows found in JSONL."

    record_rows = _mode_rows(rows, "record")
    assist_rows = _mode_rows(rows, "assist_feature")
    active_rows = record_rows + assist_rows
    n_total = len(rows)
    n_record = len(record_rows)
    n_assist = len(assist_rows)

    lines: List[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append(f"── {title} {'─' * max(0, 60 - len(title))}")

    lines.append(
        f"scaffold_memory summary  ({n_record} record rows / "
        f"{n_assist} assist_feature rows / {n_total} total rows)"
    )

    if not active_rows:
        lines.append("  No record or assist_feature rows found.")
        lines.append("  Run with --scaffold-memory <path> --scaffold-memory-mode record or assist_feature.")
        return "\n".join(lines)

    # ── A. Loaded proposals ───────────────────────────────────────────────────
    h("A. Loaded proposals")
    max_loaded = max(
        int(r.get("scaffold_memory_loaded_proposals", 0) or 0)
        for r in active_rows
    )
    lines.append(f"  scaffold_memory_loaded_proposals: {max_loaded}")
    if max_loaded == 0:
        lines.append("  *** WARNING: 0 proposals loaded — check --scaffold-memory path ***")
    else:
        lines.append("  PASS: loaded_proposals > 0")

    # ── B. Match counts and match rate ────────────────────────────────────────
    h("B. Match counts and match rate")
    total_attempts = _sum_int(active_rows, "scaffold_memory_match_attempts")
    total_matches = _sum_int(active_rows, "scaffold_memory_matches")
    total_useful = _sum_int(active_rows, "scaffold_memory_useful_matches")
    total_broad = _sum_int(active_rows, "scaffold_memory_broad_generic_debt_matches")
    total_unmatched = _sum_int(active_rows, "scaffold_memory_unmatched_records")
    match_rate = total_matches / max(1, total_attempts)

    lines.append(f"  match_attempts:              {total_attempts}")
    lines.append(f"  matches:                     {total_matches}")
    lines.append(f"  useful_matches:              {total_useful}")
    lines.append(f"  broad_generic_debt_matches:  {total_broad}")
    lines.append(f"  unmatched_records:           {total_unmatched}")
    lines.append(f"  match_rate:                  {match_rate:.4f}")

    if total_matches == 0:
        lines.append("  *** WARN: scaffold_memory_matches = 0 — check proposal file and record export ***")
    else:
        lines.append("  PASS: matches > 0")

    # Decision per spec:
    if total_matches > 0 and total_broad > total_useful:
        lines.append(
            f"  WARN: matches dominated by broad_generic_debt "
            f"({total_broad} broad vs {total_useful} useful) — "
            f"consider filtering or improving proposal specificity."
        )
    if total_useful > 0:
        lines.append(
            f"  PASS for future assist consideration: useful_matches={total_useful} > 0"
        )
    else:
        lines.append(
            f"  NOTE: useful_matches=0 — no non-broad match found; "
            f"scaffold assist consideration not yet warranted."
        )

    # ── C. Useful vs broad generic ────────────────────────────────────────────
    h("C. Useful vs broad generic")
    lines.append(f"  useful_matches:             {total_useful}")
    lines.append(f"  broad_generic_matches:      {total_broad}")
    broad_noops = _sum_int(active_rows, "scaffold_memory_broad_generic_noops")
    lines.append(f"  broad_generic_noops:        {broad_noops}")
    lines.append("  Broad generic debt is telemetry only and must not reorder candidates.")

    # ── D. Matches by kind ────────────────────────────────────────────────────
    h("D. Matches by kind")
    by_kind = _agg_kind_counts(active_rows, "scaffold_memory_matches_by_kind")
    useful_by_kind = _agg_kind_counts(active_rows, "scaffold_memory_useful_matches_by_kind")
    if not by_kind:
        lines.append("  (no matches)")
    else:
        lines.append(f"  {'kind':<40} {'total':>8} {'useful':>8}")
        for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            useful = useful_by_kind.get(kind, 0)
            lines.append(f"  {kind:<40} {count:>8} {useful:>8}")

    # ── E. Ranking applications ───────────────────────────────────────────────
    h("E. Ranking applications")
    ranking = _sum_int(active_rows, "scaffold_memory_ranking_applications")
    no_hook = _sum_int(active_rows, "scaffold_memory_no_runtime_hook_available")
    lines.append(f"  scaffold_memory_ranking_applications:     {ranking}")
    lines.append(f"  scaffold_memory_no_runtime_hook_available:{no_hook}")
    if assist_rows and ranking == 0 and no_hook == 0:
        lines.append("  WARN: assist_feature ran but no runtime ranking hook was exercised.")

    # ── F. Candidate reorder stats ────────────────────────────────────────────
    h("F. Candidate reorder stats")
    reordered = _sum_int(active_rows, "scaffold_memory_candidates_reordered")
    lines.append(f"  scaffold_memory_candidates_reordered: {reordered}")

    # ── G. Top1/topk support ──────────────────────────────────────────────────
    h("G. Top1/topk support")
    lines.append(f"  scaffold_memory_top1_supported: {_sum_int(active_rows, 'scaffold_memory_top1_supported')}")
    lines.append(f"  scaffold_memory_topk_supported: {_sum_int(active_rows, 'scaffold_memory_topk_supported')}")

    # ── H. Behavior boundary ──────────────────────────────────────────────────
    h("H. Behavior boundary")
    total_auth_allowed = _sum_int(active_rows, "scaffold_memory_authority_allowed_count")
    total_behavior_effects = _sum_int(active_rows, "scaffold_memory_behavior_effects")

    lines.append(f"  authority_allowed_count: {total_auth_allowed}  (must be 0)")
    lines.append(f"  behavior_effects:        {total_behavior_effects}  (must be 0)")

    if total_auth_allowed != 0:
        lines.append(
            f"  *** INVARIANT VIOLATION: authority_allowed_count={total_auth_allowed} ***"
        )
    else:
        lines.append("  INVARIANT OK: authority_allowed_count = 0")

    if total_behavior_effects != 0:
        lines.append(
            f"  *** INVARIANT VIOLATION: behavior_effects={total_behavior_effects} ***"
        )
    else:
        lines.append("  INVARIANT OK: behavior_effects = 0")

    # ── I. Comparison off/record/assist_feature ───────────────────────────────
    h("I. Mode comparison")
    by_mode = {
        mode: _mode_rows(rows, mode)
        for mode in ("off", "record", "assist_feature")
    }
    for mode, mode_rows in by_mode.items():
        if not mode_rows:
            continue
        lines.append(
            f"  {mode}: rows={len(mode_rows)} "
            f"loaded={max(int(r.get('scaffold_memory_loaded_proposals', 0) or 0) for r in mode_rows)} "
            f"matches={_sum_int(mode_rows, 'scaffold_memory_matches')} "
            f"ranking_applications={_sum_int(mode_rows, 'scaffold_memory_ranking_applications')} "
            f"reordered={_sum_int(mode_rows, 'scaffold_memory_candidates_reordered')}"
        )

    # ── Legacy detail: unmatched and examples ─────────────────────────────────
    h("Unmatched record count")
    lines.append(f"  scaffold_memory_unmatched_records: {total_unmatched}")
    if total_attempts > 0:
        unmatched_rate = total_unmatched / total_attempts
        lines.append(f"  unmatched_rate: {unmatched_rate:.4f}")

    # ── E. Match examples ─────────────────────────────────────────────────────
    h("Match examples (up to 10)")
    examples = _collect_examples(active_rows, limit=10)
    if not examples:
        lines.append("  (no examples)")
    else:
        for i, ex in enumerate(examples, 1):
            broad = ex.get("broad_generic_debt", False)
            broad_tag = " [BROAD_GENERIC_DEBT]" if broad else ""
            lines.append(
                f"  [{i:2d}] record={str(ex.get('record_nethra_id', ''))[:50]}"
                f"  kind={ex.get('record_kind', '')}"
            )
            lines.append(
                f"       → proposal={ex.get('matched_proposal_id', '')}  "
                f"kind={ex.get('matched_kind', '')}  "
                f"conf={ex.get('confidence', 0):.3f}{broad_tag}"
            )

    # Decision summary
    h("Decision")
    auth_ok = total_auth_allowed == 0
    behavior_ok = total_behavior_effects == 0
    loaded_ok = max_loaded > 0
    matches_ok = total_matches > 0
    if auth_ok and behavior_ok and loaded_ok and matches_ok:
        lines.append("  PASS: loaded_proposals > 0, matches > 0, behavior unchanged.")
    elif not loaded_ok:
        lines.append("  FAIL: loaded_proposals = 0")
    elif not matches_ok:
        lines.append("  WARN: matches = 0")
    if not auth_ok or not behavior_ok:
        lines.append("  FAIL: invariant violation detected (see Section F)")

    if total_matches > 0 and total_broad > total_useful:
        lines.append(
            "  WARN: matches dominated by broad_generic_debt — "
            "not yet useful for assist consideration."
        )
    if total_useful > 0:
        lines.append(
            "  PASS for future assist consideration: useful_matches > 0."
        )

    # ── J. Warning ────────────────────────────────────────────────────────────
    h("J. Warning")
    lines.append(
        "  Scaffold memory is consideration, not authority."
    )
    lines.append(
        "  It does NOT issue authority, revoke authority, suppress skips,"
    )
    lines.append(
        "  increase monitoring, increase repair priority, or add runtime budget."
    )
    lines.append(
        "  confidence_as_familiarity means 'familiar enough to recognize,'")
    lines.append(
        "  not 'safe enough to act on.'"
    )
    lines.append(
        "  broad_generic_debt proposals must not be used as ranking_hint or"
    )
    lines.append(
        "  clustering_prior without a local structural anchor."
    )

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize scaffold memory match records from a batch_run JSONL file."
    )
    p.add_argument(
        "--jsonl", required=True,
        help="Path to batch_run JSONL output file (with --scaffold-memory-mode record)"
    )
    args = p.parse_args()

    path = args.jsonl
    if not Path(path).exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(path)
    print(f"Loaded {len(rows)} rows from {path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
