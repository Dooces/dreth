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


def _mean_float(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row.get(field) or 0.0) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _counter_field(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for row in rows:
        value = row.get(field) or {}
        if isinstance(value, dict):
            out.update({str(k): int(v) for k, v in value.items()})
    return out


def _controller_counter(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    out = _counter_field(rows, field)
    for row in rows:
        payload = row.get("authority_strength") or {}
        controller = payload.get("controller") or {}
        value = controller.get(field) or {}
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
    policies = Counter(str(row.get("authority_derivation_policy") or "off") for row in rows)

    print("Authority Strength Report", file=out)
    print("Warning: visible-evidence state, not truth.", file=out)
    print(file=out)

    print("A. state distribution", file=out)
    print(f"  runs: {len(rows)}", file=out)
    print(f"  exported_records: {len(records)}", file=out)
    print("  strength:", file=out)
    for strength in ("strong", "usable", "weak", "contested", "insufficient"):
        print(f"    {strength}: {strength_counts.get(strength, 0)}", file=out)
    print("  authority_state:", file=out)
    for state in (
        "strong",
        "usable",
        "contested_best_available",
        "quarantined_for_derivation",
        "repair_candidate",
        "insufficient",
    ):
        print(f"    {state}: {state_counts.get(state, 0)}", file=out)
    print(file=out)

    print("B. debt lifecycle", file=out)
    print(f"  authority_debt_created={_sum_int(rows, 'authority_debt_created')}", file=out)
    print(f"  authority_debt_persisted={_sum_int(rows, 'authority_debt_persisted')}", file=out)
    print(f"  authority_debt_paid={_sum_int(rows, 'authority_debt_paid')}", file=out)
    print(f"  authority_debt_escalated={_sum_int(rows, 'authority_debt_escalated')}", file=out)
    print(f"  authority_debt_deescalated={_sum_int(rows, 'authority_debt_deescalated')}", file=out)
    print(f"  authority_debt_outstanding={_sum_int(rows, 'authority_debt_outstanding')}", file=out)
    print(f"  debt_age_mean={_mean_float(rows, 'debt_age_mean'):.3f}", file=out)
    print(f"  debt_age_max={max((int(row.get('debt_age_max') or 0) for row in rows), default=0)}", file=out)
    print(f"  future_evidence_requirements={_sum_int(rows, 'future_evidence_requirements')}", file=out)
    print(file=out)

    print("C. derivation gate checks allowed/blocked", file=out)
    print(f"  derivation_quarantines={_sum_int(rows, 'derivation_quarantines')}", file=out)
    print(f"  derivation_gate_checks={_sum_int(rows, 'derivation_gate_checks')}", file=out)
    print(f"  derivation_gate_allowed={_sum_int(rows, 'derivation_gate_allowed')}", file=out)
    print(f"  derivation_gate_blocked={_sum_int(rows, 'derivation_gate_blocked')}", file=out)
    print(f"  derivation_gate_would_block={_sum_int(rows, 'derivation_gate_would_block')}", file=out)
    print(f"  derivation_gate_shadow_would_block={_sum_int(rows, 'derivation_gate_shadow_would_block')}", file=out)
    print(file=out)

    print("D. blocked handle kinds", file=out)
    handle_counts = _counter_field(rows, "derivation_gate_blocked_by_handle_kind")
    state_block_counts = _counter_field(rows, "derivation_gate_blocked_by_state")
    reason_block_counts = _counter_field(rows, "derivation_gate_blocked_by_reason")
    print("  by_handle_kind:", file=out)
    if handle_counts:
        for key, count in handle_counts.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print("  by_state:", file=out)
    if state_block_counts:
        for key, count in state_block_counts.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print("  by_reason:", file=out)
    if reason_block_counts:
        for key, count in reason_block_counts.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print(file=out)

    print("E. transitions", file=out)
    transition_edges = _controller_counter(rows, "authority_state_transitions_by_edge")
    transition_reasons = _controller_counter(rows, "authority_state_transitions_by_reason")
    transition_paid = _controller_counter(rows, "authority_state_transitions_later_paid_down")
    print(f"  authority_state_transitions={_sum_int(rows, 'authority_state_transitions')}", file=out)
    print(
        "  transitions_to_derivation_quarantine="
        f"{sum(int(((row.get('authority_strength') or {}).get('controller') or {}).get('authority_state_transitions_to_derivation_quarantine') or 0) for row in rows)}",
        file=out,
    )
    print("  by_previous_next:", file=out)
    if transition_edges:
        for key, count in transition_edges.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print("  by_reason:", file=out)
    if transition_reasons:
        for key, count in transition_reasons.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print("  later_paid_down:", file=out)
    if transition_paid:
        for key, count in transition_paid.most_common():
            print(f"    {key}: {count}", file=out)
    else:
        print("    none", file=out)
    print(file=out)

    print("F. applied vs suppressed effects", file=out)
    print(f"  authority_action_candidates={_sum_int(rows, 'authority_action_candidates')}", file=out)
    print(f"  authority_actions_applied={_sum_int(rows, 'authority_actions_applied')}", file=out)
    print(f"  authority_noop_state_not_permit={_sum_int(rows, 'authority_noop_state_not_permit')}", file=out)
    print(f"  authority_suppressed_cooldown={_sum_int(rows, 'authority_suppressed_cooldown')}", file=out)
    print(f"  authority_suppressed_budget={_sum_int(rows, 'authority_suppressed_budget')}", file=out)
    print(f"  authority_suppressed_local_use_only={_sum_int(rows, 'authority_suppressed_local_use_only')}", file=out)
    print(f"  authority_suppressed_derivation_only={_sum_int(rows, 'authority_suppressed_derivation_only')}", file=out)
    print(f"  local_use_preserved={_sum_int(rows, 'local_use_preserved')}", file=out)
    if records:
        print("  best_available by strength:", file=out)
        for strength in ("strong", "usable", "weak", "contested", "insufficient"):
            print(f"    {strength}: {best_by_strength.get(strength, 0)}", file=out)
    else:
        print(f"  weak_best_available={_sum_int(rows, 'weak_best_available')}", file=out)
        print(f"  contested_best_available={_sum_int(rows, 'contested_best_available')}", file=out)
    print(file=out)

    print("  bounded runtime effects applied:", file=out)
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
    print("  suppressed candidate effects:", file=out)
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

    print("G. off/record/state comparison", file=out)
    print("  modes: " + " ".join(f"{k}={v}" for k, v in sorted(modes.items())), file=out)
    print(
        "  controllers: "
        + " ".join(f"{k}={v}" for k, v in sorted(controllers.items())),
        file=out,
    )
    print(
        "  derivation_policies: "
        + " ".join(f"{k}={v}" for k, v in sorted(policies.items())),
        file=out,
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row.get("authority_strength_mode") or "off"),
            str(row.get("authority_strength_controller") or "state"),
            str(row.get("authority_derivation_policy") or "off"),
        )].append(row)
    for key, group in sorted(grouped.items()):
        mode, controller, policy = key
        print(
            f"  {mode}/{controller}/{policy}: "
            f"quality_cost_mean={_mean_float(group, 'quality_cost'):.3f} "
            f"iv_mean={_mean_float(group, 'iv'):.3f} "
            f"full_audits_mean={_mean_float(group, 'full_audits'):.3f} "
            f"blocked={_sum_int(group, 'derivation_gate_blocked')} "
            f"would_block={_sum_int(group, 'derivation_gate_would_block')}",
            file=out,
        )
    print(file=out)

    print("H. warning when derivation gating worsens amortization", file=out)
    off_rows = grouped.get(("off", "state", "off"), []) or [
        row for row in rows if str(row.get("authority_strength_mode") or "off") == "off"
    ]
    warnings: list[str] = []
    if off_rows:
        off_quality = _mean_float(off_rows, "quality_cost")
        off_iv = _mean_float(off_rows, "iv")
        off_audits = _mean_float(off_rows, "full_audits")
        for key, group in sorted(grouped.items()):
            if key[0] != "assist" or _sum_int(group, "derivation_gate_blocked") <= 0:
                continue
            worse = (
                _mean_float(group, "quality_cost") > off_quality
                or _mean_float(group, "iv") > off_iv
                or _mean_float(group, "full_audits") > off_audits
            )
            if worse:
                warnings.append("/".join(key))
    if warnings:
        for warning in warnings:
            print(f"  WARN: derivation gating worsened amortization for {warning}", file=out)
    else:
        print("  none", file=out)
    print(file=out)

    print("Supplement. transition examples", file=out)
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
                f"{item.get('next_state') or item.get('current_state')} c{item.get('cycle')} "
                f"{','.join(item.get('reasons') or ())}",
                file=out,
            )
    else:
        print("  none", file=out)
    print(file=out)

    print("Supplement. warning: visible-evidence state, not truth", file=out)
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
