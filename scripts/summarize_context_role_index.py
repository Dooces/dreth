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

    print("F. Matches used by uncertainty consolidation", file=out)
    print(
        f"  context_role_index_matches={sum(int(row.get('context_role_index_matches') or 0) for row in rows)}",
        file=out,
    )
    print(
        "  context_role_matches_used_as_local_anchor="
        f"{sum(int(row.get('context_role_matches_used_as_local_anchor') or 0) for row in rows)}",
        file=out,
    )
    print(
        f"  context_role_assist_feature_hits={sum(int(row.get('context_role_assist_feature_hits') or 0) for row in rows)}",
        file=out,
    )
    print(file=out)

    print("G. Post-hoc interpretation", file=out)
    print("  Uses manifest relation_type only after index matching/reporting.", file=out)
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

    print("H. Warning", file=out)
    print("  ContextRoleIndex is provenance and locality memory, not runtime truth.", file=out)
    print("  A trass role is not deletion and a tareth role is not global identity.", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ContextRoleIndex JSONL output.")
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    print_report(load_jsonl(args.jsonl), sys.stdout)


if __name__ == "__main__":
    main()
