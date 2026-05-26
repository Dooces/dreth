#!/usr/bin/env python3
"""Summarize BackgroundNethraIndex records from a batch_run JSONL output file.

Usage:
    python scripts/summarize_background_nethra.py --jsonl reports/background_nethra_record.jsonl
    python scripts/summarize_background_nethra.py --jsonl reports/background_nethra_record.jsonl | tee report.txt

Report sections:
    A. Background nethra size (records, edges, records by kind)
    B. Role/context structure (trass, unresolved, quarantined, role shifts, cross-context)
    C. Familiarity without authority (familiar_background_count vs operational_authority_count)
    D. Examples (top 20 background nethras)
    E. Giant cluster handling
    F. Feature-use accounting
    G. Warning: background nethras are not authority

WARNING in report:
    Background nethras are learned familiar structure.
    They are not active authority.
    They do not imply tareth in the current context.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def _get_arch(row: Dict[str, Any]) -> Dict[str, Any]:
    # Background nethra metrics are serialized as top-level JSONL keys.
    return row


def _sum_int(rows: List[Dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        arch = _get_arch(row)
        total += int(arch.get(key, 0) or 0)
    return total


def _mean_float(rows: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        arch = _get_arch(row)
        v = arch.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return sum(vals) / len(vals) if vals else 0.0


def _agg_kind_counts(rows: List[Dict[str, Any]]) -> Counter:
    agg: Counter = Counter()
    for row in rows:
        arch = _get_arch(row)
        by_kind = arch.get("background_nethra_by_kind") or {}
        if isinstance(by_kind, dict):
            agg.update(by_kind)
    return agg


def _collect_export_records(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Collect up to `limit` background nethra export records across runs."""
    seen: dict[str, Dict[str, Any]] = {}
    for row in rows:
        arch = _get_arch(row)
        export = arch.get("background_nethra_export") or {}
        if not isinstance(export, dict):
            continue
        for rec in export.get("records") or []:
            if not isinstance(rec, dict):
                continue
            nid = str(rec.get("nethra_id", ""))
            if not nid or nid in seen:
                continue
            seen[nid] = rec
            if len(seen) >= limit * 5:
                break
    recs = list(seen.values())
    recs.sort(key=lambda r: -float(r.get("cheap_recognition_score", 0) or 0))
    return recs[:limit]


def _collect_role_shift_examples(rows: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    seen: list[Dict[str, Any]] = []
    for row in rows:
        arch = _get_arch(row)
        export = arch.get("background_nethra_export") or {}
        if not isinstance(export, dict):
            continue
        for ex in export.get("role_shift_examples") or []:
            if isinstance(ex, dict):
                seen.append(ex)
                if len(seen) >= limit:
                    return seen
    return seen


def summarize(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows found in JSONL."

    n = len(rows)
    lines: List[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append(f"── {title} {'─' * max(0, 60 - len(title))}")

    # ── A. Background nethra size ─────────────────────────────────────────────
    h("A. Background nethra size")
    total_records = _sum_int(rows, "background_nethra_records")
    total_edges = _sum_int(rows, "background_nethra_edges")
    total_contexts = _sum_int(rows, "background_contexts_seen")
    kind_counts = _agg_kind_counts(rows)

    lines.append(f"  runs: {n}")
    lines.append(f"  total background_nethra_records (across runs): {total_records}")
    lines.append(f"  total background_nethra_edges:  {total_edges}")
    lines.append(f"  total background_contexts_seen: {total_contexts}")
    lines.append(f"  records by kind:")
    for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {kind:<40} {count}")

    # ── B. Role/context structure ─────────────────────────────────────────────
    h("B. Role / context structure")
    trass = _sum_int(rows, "background_trass_patterns")
    unresolved = _sum_int(rows, "background_unresolved_patterns")
    quarantined = _sum_int(rows, "background_quarantined_patterns")
    dormant = _sum_int(rows, "background_dormant_patterns")
    frontier = _sum_int(rows, "background_tied_frontier_patterns")
    role_shifts = _sum_int(rows, "background_role_shift_examples")

    lines.append(f"  trass_patterns:          {trass}")
    lines.append(f"  unresolved_patterns:     {unresolved}")
    lines.append(f"  quarantined_patterns:    {quarantined}")
    lines.append(f"  dormant_patterns:        {dormant}")
    lines.append(f"  tied_frontier_patterns:  {frontier}")
    lines.append(f"  role_shift_examples:     {role_shifts}")

    examples = _collect_role_shift_examples(rows)
    if examples:
        lines.append(f"  role shift sample (up to 10):")
        for ex in examples[:10]:
            lines.append(
                f"    x{ex.get('var','')}  {ex.get('from_role','')} → "
                f"{ex.get('to_role','')}  cycle={ex.get('cycle','')}"
                f"  id={str(ex.get('nethra_id',''))[:50]}"
            )

    # ── C. Familiarity without authority ──────────────────────────────────────
    h("C. Familiarity without authority")
    familiar = _sum_int(rows, "familiar_background_count")
    authority = _sum_int(rows, "operational_authority_count")
    recog_mean = _mean_float(rows, "background_recognition_score_mean")
    action_mean = _mean_float(rows, "background_action_relevance_score_mean")

    lines.append(f"  familiar_background_count:    {familiar}")
    lines.append(f"  operational_authority_count:  {authority}  (must be 0)")
    lines.append(f"  recognition_score_mean:       {recog_mean:.4f}")
    lines.append(f"  action_relevance_score_mean:  {action_mean:.4f}  (expected 0.0)")
    if authority != 0:
        lines.append(
            f"  *** INVARIANT VIOLATION: operational_authority_count={authority} "
            f"must be 0 ***"
        )
    else:
        lines.append("  INVARIANT OK: no background nethra issued any authority.")

    # ── D. Examples ───────────────────────────────────────────────────────────
    h("D. Top background nethra examples (up to 20)")
    top_recs = _collect_export_records(rows, limit=20)
    if not top_recs:
        lines.append("  (no export records found)")
    else:
        for i, rec in enumerate(top_recs, 1):
            nid = str(rec.get("nethra_id", ""))[:60]
            kind = str(rec.get("kind", ""))
            vars_ = rec.get("vars") or []
            ctx = (rec.get("context_keys") or [])[:3]
            roles = list(rec.get("source_roles") or [])[:4]
            parents = (rec.get("parent_sets") or [[]])[:1]
            signals = list(rec.get("recurring_signals") or [])[:4]
            recog = float(rec.get("cheap_recognition_score", 0) or 0)
            action = float(rec.get("action_relevance_score", 0) or 0)
            lines.append(
                f"  [{i:2d}] id={nid}"
            )
            lines.append(
                f"       kind={kind}  vars={vars_}  recog={recog:.3f}  action={action:.3f}"
            )
            lines.append(
                f"       contexts={ctx}  roles={roles}"
            )
            lines.append(
                f"       parents={parents}  signals={signals}"
            )

    # ── E. Giant cluster handling ─────────────────────────────────────────────
    h("E. Giant cluster handling")
    giant = _sum_int(rows, "background_giant_cluster_patterns")
    lines.append(f"  giant clusters recorded as background:   {giant}")
    lines.append(f"  giant clusters used as authority:        0  (invariant)")
    lines.append(f"  giant clusters used as repair/monitoring:0  (invariant)")
    if giant > 0:
        lines.append(
            f"  Giant clusters ({giant}) were recorded as low-specificity background "
            f"structure only."
        )

    # ── F. Feature-use accounting ─────────────────────────────────────────────
    h("F. Feature-use accounting")
    used = _sum_int(rows, "background_records_used_as_features")
    hits = _sum_int(rows, "background_feature_hits")
    noops = _sum_int(rows, "background_feature_noops")
    lines.append(f"  records_used_as_features: {used}")
    lines.append(f"  feature_hits:             {hits}")
    lines.append(f"  feature_noops:            {noops}")
    lines.append(
        f"  operational_authority_count: {authority}  (must remain 0)"
    )
    if used > 0 or hits > 0:
        lines.append(
            "  NOTE: assist_feature mode may expose familiarity metadata. "
            "Verify operational_authority_count=0 and no behavior-effect counters."
        )

    # ── G. Warning ────────────────────────────────────────────────────────────
    h("G. Warning")
    lines.append(
        "  Background nethras are learned familiar structure."
    )
    lines.append(
        "  They are NOT active authority."
    )
    lines.append(
        "  They do NOT imply tareth in the current context."
    )
    lines.append(
        "  background_confidence means 'familiar enough to recognize,' "
        "not 'safe enough to act.'"
    )

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize BackgroundNethraIndex records from a batch_run JSONL file."
    )
    p.add_argument("--jsonl", required=True, help="Path to batch_run JSONL output file")
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
