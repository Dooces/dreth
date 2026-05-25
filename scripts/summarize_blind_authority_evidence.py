#!/usr/bin/env python3
from __future__ import annotations

"""Evidence-relative authority summary for blind_challenge JSONL output."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

# Allow importing dreth package when run directly from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CLASSIFICATIONS = (
    "evidence_supported_surrogate",
    "weakly_supported_surrogate",
    "contradicted_authority",
    "insufficient_evidence",
    "unknown",
)


@dataclass
class AuthorityEvidenceSummary:
    external_mismatch_cases: list[dict[str, Any]] = field(default_factory=list)
    by_classification: Counter[str] = field(default_factory=Counter)
    best_available_surrogates: list[dict[str, Any]] = field(default_factory=list)
    serious_mismatch_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Original per_var items for mismatch cases, used by section F throttle analysis.
    external_mismatch_items: list[dict[str, Any]] = field(default_factory=list)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
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


def _behavior(row: dict[str, Any]) -> dict[str, Any]:
    evaluation = row.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return {}
    behavior = evaluation.get("blind_challenge_behavior") or {}
    return behavior if isinstance(behavior, dict) else {}


def external_mismatch_under_authority(item: dict[str, Any]) -> bool:
    if not bool(item.get("authoritative")):
        return False
    learned = set(_as_int(v) for v in item.get("learned_parents", []) or [])
    truth = set(_as_int(v) for v in item.get("truth_parents", []) or [])
    truth.update(_as_int(v) for v in item.get("truth_delayed_parents", []) or [])
    if truth:
        return learned != truth
    if learned:
        return True
    truth_func = item.get("truth_func")
    learned_func = item.get("learned_func")
    return bool(truth_func and learned_func and truth_func != learned_func)


def _has_contradiction(item: dict[str, Any]) -> bool:
    return (
        _as_int(item.get("recent_revocations")) >= 2
        or _as_int(item.get("recent_detected_drift")) >= 2
        or _as_int(item.get("consecutive_sentinel_failures")) > 0
        or _as_int(item.get("open_novelty_observations")) > 0
        or (
            item.get("passive_stress_recent") is not None
            and _as_int(item.get("passive_stress_recent")) > 0
        )
    )


def _has_low_evidence(item: dict[str, Any]) -> bool:
    return (
        ("strong_observations" in item and _as_int(item.get("strong_observations")) <= 0)
        or ("sentinel_count" in item and _as_int(item.get("sentinel_count")) <= 0)
        or ("fit_history_count" in item and _as_int(item.get("fit_history_count")) <= 0)
        or (
            item.get("last_fit_margin") is not None
            and _as_int(item.get("last_fit_margin")) <= 0
        )
    )


def _has_stable_support(item: dict[str, Any]) -> bool:
    return (
        bool(item.get("repeatedly_stable_under_probes"))
        and _as_int(item.get("strong_observations")) >= 2
        and _as_int(item.get("sentinel_count")) > 0
        and _as_int(item.get("fit_history_count")) > 0
        and _as_int(item.get("last_fit_margin")) > 0
        and not _has_contradiction(item)
    )


def _no_better_alternative_visible(item: dict[str, Any]) -> bool:
    return (
        not bool(item.get("alternatives_existed"))
        and _as_int(item.get("last_fit_tie_count")) <= 1
        and _as_int(item.get("last_fit_near_tie_count")) <= 1
    )


def classify_evidence_support(item: dict[str, Any]) -> str:
    """Classify authority support from agent-visible evidence fields only."""
    if _has_contradiction(item):
        return "contradicted_authority"
    if _has_low_evidence(item):
        return "insufficient_evidence"
    if _has_stable_support(item) and _no_better_alternative_visible(item):
        return "evidence_supported_surrogate"
    if (
        _as_int(item.get("strong_observations")) > 0
        or _as_int(item.get("sentinel_count")) > 0
        or _as_int(item.get("last_fit_margin")) > 0
    ):
        return "weakly_supported_surrogate"
    return "unknown"


def _case_record(row: dict[str, Any], item: dict[str, Any], classification: str) -> dict[str, Any]:
    return {
        "seed": row.get("seed"),
        "var": item.get("var"),
        "relation_type": item.get("relation_type"),
        "classification": classification,
        "status": item.get("status"),
        "skip_role": item.get("skip_role"),
        "strong_observations": item.get("strong_observations"),
        "sentinel_count": item.get("sentinel_count"),
        "recent_revocations": item.get("recent_revocations"),
        "recent_detected_drift": item.get("recent_detected_drift"),
        "open_novelty": item.get("open_novelty"),
        "fit_history_count": item.get("fit_history_count"),
        "last_fit_margin": item.get("last_fit_margin"),
        "last_fit_tie_count": item.get("last_fit_tie_count"),
        "last_fit_near_tie_count": item.get("last_fit_near_tie_count"),
        "alternatives_existed": item.get("alternatives_existed"),
        "repeatedly_stable_under_probes": item.get("repeatedly_stable_under_probes"),
    }


def summarize(rows: Iterable[dict[str, Any]]) -> AuthorityEvidenceSummary:
    summary = AuthorityEvidenceSummary()
    for row in rows:
        behavior = _behavior(row)
        for item in behavior.get("per_var", []) or []:
            if not isinstance(item, dict):
                continue
            if not external_mismatch_under_authority(item):
                continue
            classification = classify_evidence_support(item)
            case = _case_record(row, item, classification)
            summary.external_mismatch_cases.append(case)
            summary.external_mismatch_items.append(item)
            summary.by_classification[classification] += 1
            if classification == "evidence_supported_surrogate":
                summary.best_available_surrogates.append(case)
            if classification in {"contradicted_authority", "insufficient_evidence"}:
                summary.serious_mismatch_candidates.append(case)
    return summary


def _print_cases(title: str, cases: list[dict[str, Any]], out: TextIO, limit: int = 20) -> None:
    print(title, file=out)
    if not cases:
        print("  (none)", file=out)
        return
    print(
        f"  {'seed':>6} {'var':>4} {'relation':<20} {'class':<30} "
        f"{'strong':>6} {'sent':>4} {'rev':>4} {'drift':>5} {'margin':>6}",
        file=out,
    )
    for case in cases[:limit]:
        print(
            f"  {str(case.get('seed')):>6} {str(case.get('var')):>4} "
            f"{str(case.get('relation_type')):<20} "
            f"{str(case.get('classification')):<30} "
            f"{_as_int(case.get('strong_observations')):>6} "
            f"{_as_int(case.get('sentinel_count')):>4} "
            f"{_as_int(case.get('recent_revocations')):>4} "
            f"{_as_int(case.get('recent_detected_drift')):>5} "
            f"{_as_int(case.get('last_fit_margin')):>6}",
            file=out,
        )
    if len(cases) > limit:
        print(f"  ... +{len(cases) - limit} more", file=out)


def _throttle_counts(
    items: list[dict[str, Any]], mode: str
) -> dict[str, int]:
    from dreth.shadow_authority_throttle import would_throttle_authority

    total = len(items)
    would_throttle = 0
    throttled_contradicted = 0
    throttled_insufficient = 0
    unthrottled_supported = 0
    unthrottled_weak = 0

    for item in items:
        decision = would_throttle_authority(item, mode=mode)
        if decision.would_throttle:
            would_throttle += 1
            if decision.evidence_class == "contradicted_authority":
                throttled_contradicted += 1
            elif decision.evidence_class == "insufficient_evidence":
                throttled_insufficient += 1
        else:
            if decision.evidence_class == "evidence_supported_surrogate":
                unthrottled_supported += 1
            elif decision.evidence_class == "weakly_supported_surrogate":
                unthrottled_weak += 1

    return {
        "external_mismatch_cases": total,
        "would_throttle": would_throttle,
        "would_not_throttle": total - would_throttle,
        "throttled_contradicted_authority": throttled_contradicted,
        "throttled_insufficient_evidence": throttled_insufficient,
        "unthrottled_supported_surrogate": unthrottled_supported,
        "unthrottled_weak_surrogate": unthrottled_weak,
        "estimated_mismatch_cases_avoided": would_throttle,
        "estimated_supported_surrogates_preserved": unthrottled_supported,
    }


def _print_throttle_section(
    summary: "AuthorityEvidenceSummary", mode: str, out: TextIO
) -> None:
    counts = _throttle_counts(summary.external_mismatch_items, mode)
    print(f"\nF. Shadow authority throttle (mode: {mode}):", file=out)
    print(
        "  Offline estimate only. No effect on runtime behavior, skip logic,\n"
        "  cert issuance, revocation, fit, sentinels, or route certs.",
        file=out,
    )
    w = 46
    for key, val in counts.items():
        print(f"  {key + ':':<{w}} {val:>6}", file=out)


def print_report(
    rows: list[dict[str, Any]],
    out: TextIO | None = None,
    throttle_mode: str = "conservative",
) -> None:
    if out is None:
        out = sys.stdout
    summary = summarize(rows)

    print("A. External mismatch under authority:", file=out)
    print(f"  cases={len(summary.external_mismatch_cases)}", file=out)
    _print_cases("  cases:", summary.external_mismatch_cases, out)

    print("\nB. Evidence support level:", file=out)
    for name in CLASSIFICATIONS:
        print(f"  {name:<32} {summary.by_classification[name]:>6}", file=out)

    print("\nC. Best-available surrogate candidates:", file=out)
    print(
        "  Hidden mismatch with stable residual/probe evidence and no visible better "
        "alternative is reported as a surrogate, not a failure.",
        file=out,
    )
    _print_cases("  candidates:", summary.best_available_surrogates, out)

    print("\nD. Serious authority/evidence mismatch candidates:", file=out)
    print(
        "  These are evidence-relative candidates: repeated stress/revocation/novelty "
        "or low observations while strong authority persisted.",
        file=out,
    )
    _print_cases("  candidates:", summary.serious_mismatch_candidates, out)

    print("\nE. Warning:", file=out)
    print(
        "  Hidden truth is used only for offline interpretation. Dreth must be "
        "judged by what was available to it.",
        file=out,
    )

    _print_throttle_section(summary, throttle_mode, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize blind_challenge authority/evidence support offline."
    )
    parser.add_argument("--jsonl", required=True, help="Path to a blind_challenge JSONL report")
    parser.add_argument(
        "--throttle-mode",
        default="conservative",
        choices=["conservative", "strict"],
        help="Shadow throttle mode for section F (default: conservative)",
    )
    args = parser.parse_args(argv)
    print_report(load_jsonl(args.jsonl), throttle_mode=args.throttle_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
