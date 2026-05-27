#!/usr/bin/env python3
from __future__ import annotations

"""Summarize ContextRoleIndex / NethraGraphIndex JSONL output."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def _iter_nodes(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        index = row.get("context_role_index") or row.get("nethra_reservoir") or {}
        for node in index.get("nodes") or index.get("records") or ():
            if isinstance(node, dict):
                yield node


def _iter_roles(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        index = row.get("context_role_index") or row.get("nethra_reservoir") or {}
        for role in index.get("roles") or ():
            if isinstance(role, dict):
                yield role


def _counter_from_rows(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for row in rows:
        value = row.get(field) or {}
        if isinstance(value, dict):
            out.update({str(k): int(v) for k, v in value.items()})
    return out


def _sum_int(rows: Iterable[dict[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _revocations(row: dict[str, Any]) -> float:
    if row.get("revocations") is not None:
        return _as_float(row.get("revocations"))
    value = row.get("revoked_by_dist") or {}
    if isinstance(value, dict):
        return sum(_as_float(v) for v in value.values())
    return 0.0


def _mode_key(row: dict[str, Any]) -> str:
    mode = str(row.get("context_role_index_mode") or row.get("nethra_reservoir_mode") or "off")
    policy = str(row.get("context_role_anchor_policy") or "off")
    uc = str(row.get("uncertainty_consolidation_mode") or "off")
    if mode == "assist_feature":
        if policy == "off" and int(row.get("context_role_matches_used_as_local_anchor") or 0) > 0:
            policy = "loose"
        return f"assist_feature+{policy}"
    if mode == "record":
        return "record"
    if uc == "off":
        return "off"
    return mode


def _avg(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(_as_float(row.get(field)) for row in rows) / len(rows)


def _avg_metric(rows: list[dict[str, Any]], field: str) -> float:
    if field == "revocations":
        return sum(_revocations(row) for row in rows) / len(rows) if rows else 0.0
    if field == "unique_fails":
        return _avg_alias(rows, "unique_fails", "total_unique_failures")
    if field == "iv":
        return _avg_alias(rows, "iv", "interventions")
    if field == "regime_sentinel_fails":
        return _avg_alias(rows, "regime_sentinel_fails", "regime_sentinel_fail")
    return _avg(rows, field)


def _avg_alias(rows: list[dict[str, Any]], first: str, second: str) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += _as_float(row.get(first, row.get(second)))
    return total / len(rows)


def _relation_by_var(rows: Iterable[dict[str, Any]]) -> dict[int, Counter[str]]:
    rels: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        behavior = ((row.get("evaluation") or {}).get("blind_challenge_behavior") or {})
        if not isinstance(behavior, dict):
            continue
        for item in behavior.get("per_var") or ():
            if not isinstance(item, dict) or item.get("var") is None:
                continue
            rels[int(item["var"])][str(item.get("relation_type") or "unknown")] += 1
    return rels


def print_report(rows: list[dict[str, Any]], out: TextIO) -> None:
    nodes = list(_iter_nodes(rows))
    roles = list(_iter_roles(rows))
    by_kind = Counter(str(n.get("kind") or "unknown") for n in nodes)
    by_source = Counter(str(n.get("source") or "unknown") for n in nodes)
    by_role = Counter(str(r.get("role") or "unknown") for r in roles)
    by_context = Counter(str(r.get("context_key") or "unknown") for r in roles)
    runtime_kind = _counter_from_rows(rows, "context_role_nodes_by_kind")
    runtime_source = _counter_from_rows(rows, "context_role_nodes_by_source")
    runtime_roles_by_context = _counter_from_rows(rows, "context_roles_by_context")
    runtime_roles_by_role = _counter_from_rows(rows, "context_roles_by_role")
    runtime_reasons = _counter_from_rows(rows, "context_role_top_match_reasons")
    if runtime_kind and not nodes:
        by_kind.update(runtime_kind)
    if runtime_source and not nodes:
        by_source.update(runtime_source)
    if runtime_roles_by_context and not roles:
        by_context.update(runtime_roles_by_context)
    if runtime_roles_by_role and not roles:
        by_role.update(runtime_roles_by_role)

    role_sets: dict[str, set[str]] = defaultdict(set)
    context_sets: dict[str, set[str]] = defaultdict(set)
    for role in roles:
        nid = str(role.get("nethra_id") or "")
        if not nid:
            continue
        role_sets[nid].add(str(role.get("role") or "unknown"))
        context_sets[nid].add(str(role.get("context_key") or "unknown"))

    print("ContextRoleIndex Report", file=out)
    print("Warning: nethra nodes are provenance, not truth; roles are context-indexed.", file=out)
    print(file=out)

    print("A. Index size", file=out)
    print(f"  runs: {len(rows)}", file=out)
    print(f"  exported_nodes: {len(nodes)}", file=out)
    print(f"  exported_context_roles: {len(roles)}", file=out)
    print(f"  runtime_nodes: {sum(int(row.get('context_role_index_nodes') or 0) for row in rows)}", file=out)
    print(f"  runtime_roles: {sum(int(row.get('context_role_records') or 0) for row in rows)}", file=out)
    print(file=out)

    print("B. Nethra nodes by kind/source", file=out)
    print("  kind:", " ".join(f"{k}={v}" for k, v in by_kind.most_common()) or "none", file=out)
    print("  source:", " ".join(f"{k}={v}" for k, v in by_source.most_common()) or "none", file=out)
    print(file=out)

    print("C. Context roles by role/context", file=out)
    print("  role:", " ".join(f"{k}={v}" for k, v in by_role.most_common()) or "none", file=out)
    for context, count in by_context.most_common(12):
        print(f"  {context}: {count}", file=out)
    print(file=out)

    print("D. Nethras with multiple roles across contexts", file=out)
    multi = [(nid, sorted(rs), sorted(context_sets[nid])) for nid, rs in role_sets.items() if len(rs) > 1]
    if not multi:
        print("  none", file=out)
    for nid, rs, contexts in multi[:10]:
        print(f"  {nid}: roles={','.join(rs)} contexts={len(contexts)}", file=out)
    print(file=out)

    print("E. Trass-in-one-context / tareth-in-another examples", file=out)
    examples = [(nid, sorted(context_sets[nid])) for nid, rs in role_sets.items() if {"trass", "tareth"} <= rs]
    if not examples:
        print("  none", file=out)
    for nid, contexts in examples[:10]:
        print(f"  {nid}: contexts={contexts[:3]}", file=out)
    print(file=out)

    print("F. Match pressure and anchor admission", file=out)
    print(f"  context_role_index_matches={_sum_int(rows, 'context_role_index_matches')}", file=out)
    print(f"  context_role_raw_matches={_sum_int(rows, 'context_role_raw_matches')}", file=out)
    print(f"  context_role_deduped_matches={_sum_int(rows, 'context_role_deduped_matches')}", file=out)
    print(
        "  suppressions: "
        f"weak={_sum_int(rows, 'context_role_matches_suppressed_weak')} "
        f"duplicate={_sum_int(rows, 'context_role_matches_suppressed_duplicate')} "
        f"cap={_sum_int(rows, 'context_role_matches_suppressed_cap')}",
        file=out,
    )
    print(
        "  context_role_matches_used_as_local_anchor="
        f"{_sum_int(rows, 'context_role_matches_used_as_local_anchor')}",
        file=out,
    )
    pressure = _sum_int(rows, "context_role_assist_pressure_events")
    if pressure == 0:
        pressure = _sum_int(rows, "context_role_assist_pressure_per_cycle")
    cycles = max(1, sum(int(row.get("cycles") or 0) for row in rows))
    print(f"  context_role_assist_feature_hits={_sum_int(rows, 'context_role_assist_feature_hits')}", file=out)
    print(f"  context_role_assist_pressure_per_cycle={pressure / cycles:.6f}", file=out)
    print("  top_match_reasons:", " ".join(f"{k}={v}" for k, v in runtime_reasons.most_common(8)) or "none", file=out)
    print(file=out)

    print("G. Role transition examples", file=out)
    transition_examples = []
    by_nid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for role in roles:
        by_nid[str(role.get("nethra_id") or "")].append(role)
    for nid, items in by_nid.items():
        ordered = sorted(items, key=lambda r: int(r.get("cycle") or 0))
        for prev, curr in zip(ordered, ordered[1:]):
            if prev.get("role") != curr.get("role"):
                transition_examples.append((nid, prev, curr))
                break
    if not transition_examples:
        print("  none", file=out)
    for nid, prev, curr in transition_examples[:10]:
        print(
            f"  {nid}: {prev.get('role')}@{prev.get('context_key')} -> "
            f"{curr.get('role')}@{curr.get('context_key')}",
            file=out,
        )
    print(file=out)

    print("H. Post-hoc interpretation", file=out)
    print("  Uses manifest relation_type only after index matching/reporting.", file=out)
    print("  context_role_anchor_precision_posthoc is report-only; runtime never reads it.", file=out)
    rel_by_var = _relation_by_var(rows)
    for node in nodes[:10]:
        components = [int(v) for v in node.get("components") or [] if isinstance(v, int)]
        rel_counts: Counter[str] = Counter()
        for var in components:
            rel_counts.update(rel_by_var.get(var, {}))
        rel_text = " ".join(f"{rel}={count}" for rel, count in rel_counts.most_common())
        print(f"  {node.get('nethra_id')}: {rel_text or 'unknown'}", file=out)
    if not nodes:
        print("  none", file=out)
    print(file=out)

    print(file=out)

    print("I. Four-way comparison", file=out)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_mode_key(row)].append(row)
    metric_fields = [
        ("quality_cost", "quality_cost"),
        ("iv", "iv"),
        ("audits", "full_audits"),
        ("revocations", "revocations"),
        ("unique_fails", "unique_fails"),
        ("regime_sentinel_fail", "regime_sentinel_fails"),
        ("passive_stress", "passive_stress_count"),
        ("dormant", "dormant_total"),
        ("raw_match_pressure", "context_role_raw_matches"),
        ("anchor_pressure", "context_role_matches_used_as_local_anchor"),
        ("assist_pressure", "context_role_assist_feature_hits"),
    ]
    off_rows = groups.get("off", [])
    for key in ("off", "record", "assist_feature+loose", "assist_feature+strict"):
        group = groups.get(key, [])
        if not group:
            print(f"  {key}: none", file=out)
            continue
        bits = []
        for label, field in metric_fields:
            value = _avg_metric(group, field)
            delta = value - _avg_metric(off_rows, field) if off_rows and key != "off" else 0.0
            bits.append(f"{label}={value:.3f} d={delta:+.3f}")
        print(f"  {key}: " + " ".join(bits), file=out)
    loose = groups.get("assist_feature+loose", [])
    strict = groups.get("assist_feature+strict", [])
    if off_rows and loose:
        loose_q = _avg_metric(loose, "quality_cost") - _avg_metric(off_rows, "quality_cost")
        loose_iv = _avg_metric(loose, "iv") - _avg_metric(off_rows, "iv")
        if loose_q > 0 or loose_iv > 0:
            print("  WARNING: loose matching worsens metrics relative to off.", file=out)
    if off_rows and strict and loose:
        strict_q = _avg_metric(strict, "quality_cost") - _avg_metric(off_rows, "quality_cost")
        loose_q = _avg_metric(loose, "quality_cost") - _avg_metric(off_rows, "quality_cost")
        if strict_q > 0 and strict_q >= loose_q:
            print("  WARNING: strict still worsens; keep assist_feature record/provenance-only.", file=out)
    print(file=out)

    print("J. Warning", file=out)
    print("  ContextRoleIndex is provenance and locality memory, not runtime truth.", file=out)
    print("  A trass role is not deletion and a tareth role is not global identity.", file=out)
    print(file=out)

    _print_surface_report(rows, out)


def _iter_role_surfaces(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        index = row.get("context_role_index") or row.get("nethra_reservoir") or {}
        for s in index.get("role_surfaces") or ():
            if isinstance(s, dict):
                yield s


def _iter_residual_buckets(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        index = row.get("context_role_index") or row.get("nethra_reservoir") or {}
        for b in index.get("residual_buckets") or ():
            if isinstance(b, dict):
                yield b


def _print_surface_report(rows: list[dict[str, Any]], out: TextIO) -> None:
    surfaces = list(_iter_role_surfaces(rows))
    buckets = list(_iter_residual_buckets(rows))

    surface_count = _sum_int(rows, "role_surface_count")
    load_bearing = _sum_int(rows, "load_bearing_surface_count")
    residual_surfaces = _sum_int(rows, "residual_surface_count")
    bucket_count = _sum_int(rows, "residual_bucket_count")
    pressure_total = sum(_as_float(row.get("residual_pressure_total")) for row in rows)
    unresolved_total = _sum_int(rows, "residual_unresolved_count")
    absorbed_total = _sum_int(rows, "residual_absorbed_count")
    regime_candidates = _sum_int(rows, "regime_transition_candidates_from_residuals")
    growth_windows = _sum_int(rows, "residual_pressure_persistent_growth_windows")

    print("K. Role surfaces (record-only)", file=out)
    print(f"  exported_surfaces: {len(surfaces)}", file=out)
    print(f"  runtime_role_surface_count: {surface_count}", file=out)
    print(f"  load_bearing_surface_count: {load_bearing}", file=out)
    print(f"  residual_surface_count: {residual_surfaces}", file=out)
    role_counts: Counter[str] = Counter(str(s.get("role_state") or "unknown") for s in surfaces)
    print("  by_role_state:", " ".join(f"{k}={v}" for k, v in role_counts.most_common()) or "none", file=out)
    print(file=out)

    print("L. Residual buckets (record-only)", file=out)
    print(f"  exported_buckets: {len(buckets)}", file=out)
    print(f"  runtime_bucket_count: {bucket_count}", file=out)
    print(f"  residual_pressure_total: {pressure_total:.4f}", file=out)
    print(f"  residual_unresolved_count: {unresolved_total}", file=out)
    print(f"  residual_absorbed_count: {absorbed_total}", file=out)
    print(f"  regime_transition_candidates: {regime_candidates}", file=out)
    print(f"  persistent_growth_windows: {growth_windows}", file=out)
    pressures = [_as_float(b.get("pressure")) for b in buckets]
    if pressures:
        print(
            f"  exported_pressure: max={max(pressures):.4f} mean={sum(pressures)/len(pressures):.4f}",
            file=out,
        )
    print("  Note: residual pressure is not authority. Pressure alone may not trigger any action.", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ContextRoleIndex JSONL output.")
    parser.add_argument("--jsonl", required=True, nargs="+")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.jsonl:
        rows.extend(load_jsonl(path))
    print_report(rows, sys.stdout)


if __name__ == "__main__":
    main()
