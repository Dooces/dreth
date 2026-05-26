#!/usr/bin/env python3
from __future__ import annotations

"""Summarize visible-evidence authority-strength JSONL output."""

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


def _iter_records(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        payload = row.get("authority_strength") or {}
        for record in payload.get("records") or ():
            if isinstance(record, dict):
                yield record


def _sum_int(rows: Iterable[dict[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def _counter_field(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
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
    records = list(_iter_records(rows))
    strength_counts = Counter(str(record.get("strength") or "unknown") for record in records)
    if not records:
        strength_counts.update({
            "strong": _sum_int(rows, "strength_strong"),
            "usable": _sum_int(rows, "strength_usable"),
            "weak": _sum_int(rows, "strength_weak"),
            "contested": _sum_int(rows, "strength_contested"),
            "insufficient": _sum_int(rows, "strength_insufficient"),
        })
        strength_counts += Counter()
    reason_counts = Counter(str(record.get("reason") or "unknown") for record in records)
    runtime_reason_counts = _counter_field(rows, "authority_strength_counts_by_reason")
    if runtime_reason_counts and not reason_counts:
        reason_counts.update(runtime_reason_counts)
    state_counts = Counter(
        str(record.get("authority_state") or "unknown") for record in records
    )
    runtime_state_counts = _counter_field(rows, "authority_state_counts")
    if runtime_state_counts:
        state_counts = runtime_state_counts
    elif not records:
        state_counts += Counter()

    best_by_strength = Counter(
        str(record.get("strength") or "unknown")
        for record in records
        if bool(record.get("best_available"))
    )
    trigger_counts = Counter()
    for record in records:
        for field in ("active_evidence", "contradictory_evidence", "uncertainty_signals"):
            for item in record.get(field) or ():
                trigger_counts[str(item).split("=", 1)[0]] += 1

    modes = Counter(str(row.get("authority_strength_mode") or "off") for row in rows)
    controllers = Counter(str(row.get("authority_strength_controller") or "state") for row in rows)

    print("Authority Strength Report", file=out)
    print("Warning: visible-evidence state, not truth.", file=out)
    print(file=out)

    print("A. strength distribution", file=out)
    print(f"  runs: {len(rows)}", file=out)
    print(f"  exported_records: {len(records)}", file=out)
    for strength in ("strong", "usable", "weak", "contested", "insufficient"):
        print(f"  {strength}: {strength_counts.get(strength, 0)}", file=out)
    print(file=out)

    print("B. authority state distribution", file=out)
    for state in (
        "strong",
        "usable",
        "contested_best_available",
        "quarantined_for_derivation",
        "repair_candidate",
        "insufficient",
    ):
        print(f"  {state}: {state_counts.get(state, 0)}", file=out)
    print(file=out)

    print("C. debt created / paid / outstanding", file=out)
    print(f"  authority_debt_created={_sum_int(rows, 'authority_debt_created')}", file=out)
    print(f"  authority_debt_paid={_sum_int(rows, 'authority_debt_paid')}", file=out)
    print(f"  authority_debt_outstanding={_sum_int(rows, 'authority_debt_outstanding')}", file=out)
    print(f"  future_evidence_requirements={_sum_int(rows, 'future_evidence_requirements')}", file=out)
    print(file=out)

    print("D. derivation quarantines", file=out)
    print(f"  derivation_quarantines={_sum_int(rows, 'derivation_quarantines')}", file=out)
    print(file=out)

    print("E. local-use-preserved counts", file=out)
    print(f"  local_use_preserved={_sum_int(rows, 'local_use_preserved')}", file=out)
    if records:
        print("  best_available by strength:", file=out)
        for strength in ("strong", "usable", "weak", "contested", "insufficient"):
            print(f"    {strength}: {best_by_strength.get(strength, 0)}", file=out)
    else:
        print(f"  weak_best_available={_sum_int(rows, 'weak_best_available')}", file=out)
        print(f"  contested_best_available={_sum_int(rows, 'contested_best_available')}", file=out)
    print(file=out)

    print("F. bounded runtime effects applied", file=out)
    print(
        f"  monitoring_increases_from_strength_applied="
        f"{_sum_int(rows, 'monitoring_increases_from_strength_applied')}",
        file=out,
    )
    print(f"  monitoring_hints_applied={_sum_int(rows, 'monitoring_hints_applied')}", file=out)
    print(
        f"  repair_priority_bumps_from_strength_applied="
        f"{_sum_int(rows, 'repair_priority_bumps_from_strength_applied')}",
        file=out,
    )
    print(f"  bounded_repairs_applied={_sum_int(rows, 'bounded_repairs_applied')}", file=out)
    print(
        f"  alternatives_preserved_from_strength="
        f"{_sum_int(rows, 'alternatives_preserved_from_strength')}",
        file=out,
    )
    print(file=out)

    print("G. suppressed candidate effects", file=out)
    for prefix in (
        "monitoring_increases_from_strength",
        "repair_priority_bumps_from_strength",
    ):
        print(f"  {prefix}_candidates={_sum_int(rows, prefix + '_candidates')}", file=out)
        print(f"  {prefix}_suppressed_by_state={_sum_int(rows, prefix + '_suppressed_by_state')}", file=out)
        print(f"  {prefix}_suppressed_by_cooldown={_sum_int(rows, prefix + '_suppressed_by_cooldown')}", file=out)
        print(f"  {prefix}_suppressed_by_budget={_sum_int(rows, prefix + '_suppressed_by_budget')}", file=out)
        print(f"  {prefix}_noops={_sum_int(rows, prefix + '_noops')}", file=out)
    print(file=out)

    print("H. transition examples", file=out)
    transitions: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("authority_strength") or {}
        controller = payload.get("controller") or {}
        for item in controller.get("authority_state_transitions_examples") or ():
            if isinstance(item, dict):
                transitions.append(item)
    if transitions:
        for item in transitions[:10]:
            print(
                f"  x{item.get('var')}: {item.get('previous_state')} -> "
                f"{item.get('current_state')} c{item.get('cycle')} "
                f"{','.join(item.get('reasons') or ())}",
                file=out,
            )
    else:
        print("  none", file=out)
    print(file=out)

    print("I. off/record/assist comparison", file=out)
    print("  modes: " + " ".join(f"{k}={v}" for k, v in sorted(modes.items())), file=out)
    print(
        "  controllers: "
        + " ".join(f"{k}={v}" for k, v in sorted(controllers.items())),
        file=out,
    )
    print(file=out)

    print("J. warning: visible-evidence state, not truth", file=out)
    print("  No cert issuance, revocation, skip suppression, or fit replacement is implied.", file=out)
    print(file=out)

    print("Supplement. contested/weak reasons", file=out)
    weak_contested = [
        record for record in records
        if record.get("strength") in {"weak", "contested"}
    ]
    weak_reason_counts = Counter(str(record.get("reason") or "unknown") for record in weak_contested)
    if weak_reason_counts:
        for reason, count in weak_reason_counts.most_common():
            print(f"  {reason}: {count}", file=out)
    else:
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}", file=out)
    if not weak_reason_counts and not reason_counts:
        print("  none", file=out)
    print(file=out)

    print("Supplement. visible-evidence triggers", file=out)
    if trigger_counts:
        for trigger, count in trigger_counts.most_common():
            print(f"  {trigger}: {count}", file=out)
    else:
        print("  none", file=out)
    print(file=out)

    print("Supplement. post-hoc interpretation may use manifest only after classification", file=out)
    rel_by_var = _relation_by_var(rows)
    if not records:
        print("  none", file=out)
    for record in records[:10]:
        rel_counts = rel_by_var.get(int(record.get("var") or -1), Counter())
        rel_text = " ".join(f"{rel}={count}" for rel, count in rel_counts.most_common())
        print(
            f"  {record.get('nethra_id')}: strength={record.get('strength')} "
            f"{rel_text or 'unknown'}",
            file=out,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize authority-strength JSONL output."
    )
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    print_report(load_jsonl(args.jsonl), out=sys.stdout)


if __name__ == "__main__":
    main()
