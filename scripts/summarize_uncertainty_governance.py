#!/usr/bin/env python3
from __future__ import annotations

"""Shadow-only uncertainty governance agenda summarizer for blind_challenge JSONL output.

Reads a blind_challenge JSONL report and builds per-var uncertainty governance
proposals from agent-visible evidence fields. Hidden truth is used only post-hoc
to select cases for report sections D and E.

WARNING: DIAGNOSTIC ONLY. No runtime behavior is changed by this script.
  - Agent behavior, fit_var, cert issuance, revocation, sentinels, skip logic,
    and authority records are not modified.
  - Proposals are NOT correct actions. They are shadow governance proposals only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreth.uncertainty_governance import (
    PROPOSAL_ACTIONS,
    SIGNAL_NAMES,
    UncertaintyGovernanceSummary,
    build_governance_summary,
)


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


def _print_cases(
    title: str,
    cases: list[dict[str, Any]],
    out: TextIO,
    limit: int = 20,
) -> None:
    print(title, file=out)
    if not cases:
        print("  (none)", file=out)
        return
    print(
        f"  {'seed':>6} {'var':>4} {'relation':<22} {'action':<32} signals",
        file=out,
    )
    for case in cases[:limit]:
        signals_str = " ".join(case.get("active_signals") or []) or "(none)"
        print(
            f"  {str(case.get('seed')):>6} {str(case.get('var')):>4} "
            f"{str(case.get('relation_type')):<22} "
            f"{str(case.get('action')):<32} "
            f"{signals_str}",
            file=out,
        )
    if len(cases) > limit:
        print(f"  ... +{len(cases) - limit} more", file=out)


def print_report(
    rows: list[dict[str, Any]],
    out: TextIO | None = None,
) -> None:
    """Print the uncertainty governance summary report.

    Sections:
      A. Signal counts
      B. Proposed action counts
      C. Proposal counts by relation_type (manifest used only post-hoc)
      D. Cases where governance would have recommended caution before external mismatch
      E. Supported surrogate cases where governance would not suppress action
      F. Warning: diagnostic only, no runtime behavior changed
    """
    if out is None:
        out = sys.stdout

    summary = build_governance_summary(rows)
    total_proposals = len(summary.proposals)

    print(
        "Uncertainty Governance Agenda — shadow-only diagnostic report",
        file=out,
    )
    print(
        f"Total proposals: {total_proposals}",
        file=out,
    )

    # A. Signal counts
    print("\nA. Signal counts:", file=out)
    print(
        "  Count of vars where each observable uncertainty signal was active.\n"
        "  Derived from agent-visible evidence fields only.",
        file=out,
    )
    w = 28
    all_signal_bases = [
        "open_novelty", "low_margin", "near_tie_count", "tie_count",
        "dormant_revival", "repeated_fit_churn", "sentinel_failures",
        "passive_stress", "recent_revocations", "alternatives_existed",
        "graph_frontier_miss",
    ]
    for sig in all_signal_bases:
        print(f"  {sig + ':':<{w}} {summary.signal_counts[sig]:>6}", file=out)

    # consequence_tier is informational — show distribution separately.
    tier_counts: Counter[str] = Counter()
    for proposal in summary.proposals:
        tier_counts[proposal.signal.consequence_tier] += 1
    print(f"\n  consequence_tier distribution:", file=out)
    for tier in sorted(tier_counts):
        print(f"    {tier + ':':<18} {tier_counts[tier]:>6}", file=out)

    # B. Proposed action counts
    print("\nB. Proposed action counts:", file=out)
    print(
        "  Shadow governance action proposed for each var. Proposals are not\n"
        "  correct actions and do not change runtime behavior.",
        file=out,
    )
    w2 = 36
    for action in PROPOSAL_ACTIONS:
        print(
            f"  {action + ':':<{w2}} {summary.action_counts[action]:>6}",
            file=out,
        )

    # C. Proposal counts by relation_type
    print("\nC. Proposal counts by relation_type:", file=out)
    print(
        "  relation_type is from per_var visible fields. Post-hoc manifest\n"
        "  interpretation explains structural meaning; not used for classification.",
        file=out,
    )
    if not summary.by_relation_type:
        print("  (no data)", file=out)
    else:
        rel_types = sorted(summary.by_relation_type)
        actions_present = [
            a for a in PROPOSAL_ACTIONS if any(
                summary.by_relation_type[r].get(a, 0) > 0 for r in rel_types
            )
        ]
        if actions_present:
            header = " ".join(f"{a[:10]:>11}" for a in actions_present)
            print(f"  {'relation_type':<26} {header}", file=out)
            for rel in rel_types:
                counts_str = " ".join(
                    f"{summary.by_relation_type[rel].get(a, 0):>11}"
                    for a in actions_present
                )
                print(f"  {rel:<26} {counts_str}", file=out)

    # D. Cases where governance would have recommended caution before mismatch
    print(
        "\nD. Cases where governance would have recommended caution before "
        "external mismatch:",
        file=out,
    )
    print(
        "  These vars were authoritative, externally mismatched (post-hoc),\n"
        "  and governance did NOT propose continue_best_available.\n"
        "  Hidden truth selects the cases; governance action comes from visible evidence.",
        file=out,
    )
    print(
        f"  caution_before_mismatch count: {len(summary.caution_before_mismatch)}",
        file=out,
    )
    _print_cases("  cases:", summary.caution_before_mismatch, out)

    # D sub-table: breakdown of caution actions
    if summary.caution_before_mismatch:
        caution_action_counts: Counter[str] = Counter(
            c["action"] for c in summary.caution_before_mismatch
        )
        print("\n  Caution action distribution:", file=out)
        for action, count in caution_action_counts.most_common():
            print(f"    {action + ':':<36} {count:>5}", file=out)

    # E. Supported surrogate cases where governance would not suppress action
    print(
        "\nE. Supported surrogate cases where governance would not suppress action:",
        file=out,
    )
    print(
        "  These vars were authoritative and externally mismatched (post-hoc),\n"
        "  yet governance proposed continue_best_available based on visible evidence.\n"
        "  They represent cases where best-available behavior was the right call.",
        file=out,
    )
    print(
        f"  supported_surrogate count: {len(summary.supported_surrogate_cases)}",
        file=out,
    )
    _print_cases("  cases:", summary.supported_surrogate_cases, out)

    # F. Warning
    print("\nF. Warning — DIAGNOSTIC ONLY:", file=out)
    print(
        "  This report is diagnostic only. No runtime behavior has been changed.\n"
        "  Agent behavior, fit_var, cert issuance, revocation, sentinels, skip\n"
        "  logic, authority records, and throttling are all unaffected.\n"
        "  Governance proposals are NOT correct actions. Hidden truth is used\n"
        "  only post-hoc in sections D and E to select cases for interpretation;\n"
        "  the governance classifier itself never reads truth_parents, truth_func,\n"
        "  truth_delayed_parents, truth_latents, or any hidden-world field.",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-only uncertainty governance agenda summarizer for "
            "blind_challenge JSONL output. DIAGNOSTIC ONLY."
        )
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to a blind_challenge JSONL report",
    )
    args = parser.parse_args(argv)
    print_report(load_jsonl(args.jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
