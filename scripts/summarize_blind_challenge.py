#!/usr/bin/env python3
from __future__ import annotations

"""Offline summarizer for blind_challenge batch JSONL output."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, TextIO


@dataclass
class BasicOutcome:
    runs: int = 0
    ok: int = 0
    invariant_failures: int = 0
    interventions: int = 0
    audits: int = 0
    skips: int = 0
    trass_skips: int = 0
    sentinel_skips: int = 0
    compression_skips: int = 0
    revocations: int = 0
    novelty_open: int = 0
    frontier_stress: int = 0
    graph_frontier_evals: int = 0
    graph_frontier_recall_total: float = 0.0
    temporal_frontier_evals: int = 0
    temporal_frontier_recall_total: float = 0.0
    temporal_frontier_lift_total: float = 0.0

    @property
    def invariants_pass(self) -> bool:
        return self.invariant_failures == 0


@dataclass
class DiscoverySummary:
    relation_types: Counter[str] = field(default_factory=Counter)
    appeared_learned: Counter[str] = field(default_factory=Counter)
    failed_to_learn: Counter[str] = field(default_factory=Counter)
    over_certified: Counter[str] = field(default_factory=Counter)
    uncertain_or_stressed: Counter[str] = field(default_factory=Counter)
    withheld: Counter[str] = field(default_factory=Counter)
    side_effect_rules: int = 0
    latent_count: int = 0
    manifests: int = 0


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


def _revocations(row: dict[str, Any]) -> int:
    revoked = row.get("revoked_by_dist") or {}
    if isinstance(revoked, dict):
        return sum(_as_int(v) for v in revoked.values())
    return 0


def compute_basic_outcome(rows: Iterable[dict[str, Any]]) -> BasicOutcome:
    out = BasicOutcome()
    for row in rows:
        out.runs += 1
        if row.get("ok"):
            out.ok += 1
        violations = row.get("violations") or []
        out.invariant_failures += len(violations) if isinstance(violations, list) else 0
        out.interventions += _as_int(row.get("interventions"))
        out.audits += _as_int(row.get("full_audits"))
        out.trass_skips += _as_int(row.get("trass_skips"))
        out.sentinel_skips += _as_int(row.get("sentinel_skips"))
        out.compression_skips += _as_int(row.get("compression_skips"))
        if out.trass_skips or out.sentinel_skips or out.compression_skips:
            out.skips = out.trass_skips + out.sentinel_skips + out.compression_skips
        else:
            out.skips += int(_as_float(row.get("skip_pct")) > 0.0)
        out.revocations += _revocations(row)
        out.novelty_open += _as_int(row.get("vars_open_novelty"))
        out.frontier_stress += _as_int(row.get("frontier_cleared"))
        out.graph_frontier_evals += _as_int(row.get("graph_frontier_evals"))
        out.graph_frontier_recall_total += _as_float(row.get("graph_frontier_chosen_parent_recall"))
        out.temporal_frontier_evals += _as_int(row.get("temporal_frontier_evals"))
        out.temporal_frontier_recall_total += _as_float(row.get("temporal_frontier_chosen_parent_recall"))
        out.temporal_frontier_lift_total += _as_float(row.get("temporal_frontier_recall_lift"))
    return out


def _evaluation(row: dict[str, Any]) -> dict[str, Any]:
    ev = row.get("evaluation") or {}
    return ev if isinstance(ev, dict) else {}


def _manifest(row: dict[str, Any]) -> dict[str, Any]:
    manifest = _evaluation(row).get("blind_challenge_manifest") or {}
    return manifest if isinstance(manifest, dict) else {}


def _behavior(row: dict[str, Any]) -> dict[str, Any]:
    behavior = _evaluation(row).get("blind_challenge_behavior") or {}
    return behavior if isinstance(behavior, dict) else {}


def compute_discovery_summary(rows: Iterable[dict[str, Any]]) -> DiscoverySummary:
    summary = DiscoverySummary()
    for row in rows:
        manifest = _manifest(row)
        behavior = _behavior(row)
        if manifest:
            summary.manifests += 1
            summary.side_effect_rules += len(manifest.get("intervention_side_effects", []) or [])
            summary.latent_count += len(manifest.get("latents", []) or [])
            for rel in manifest.get("relations", []) or []:
                if isinstance(rel, dict):
                    summary.relation_types[str(rel.get("relation_type") or "unknown")] += 1
        for item in behavior.get("per_var", []) or []:
            if not isinstance(item, dict):
                continue
            rel_type = str(item.get("relation_type") or "unknown")
            overlap = bool(item.get("learned_parent_overlap"))
            truth_parents = bool(item.get("truth_parents") or item.get("truth_delayed_parents"))
            compatible = bool(item.get("agent_func_compatible"))
            certified = item.get("status") == "certified" or bool(item.get("authoritative"))
            uncertain = item.get("status") in {"uncertain", "proposed"} or item.get("skip_role") == "untested"
            withheld = not bool(item.get("authoritative")) or item.get("skip_role") in {"untested", "noise_floor"}
            learned_parents = bool(item.get("learned_parents"))

            if certified and (overlap or (compatible and not truth_parents)):
                summary.appeared_learned[rel_type] += 1
            elif truth_parents and not overlap:
                summary.failed_to_learn[rel_type] += 1
            if certified and truth_parents and learned_parents and not overlap:
                summary.over_certified[rel_type] += 1
            if uncertain:
                summary.uncertain_or_stressed[rel_type] += 1
            if withheld:
                summary.withheld[rel_type] += 1
    return summary


def _print_counter(title: str, counter: Counter[str], out: TextIO) -> None:
    print(title, file=out)
    if not counter:
        print("  (none observed)", file=out)
        return
    for key, count in counter.most_common():
        print(f"  {key:<20} {count:>6}", file=out)


def print_report(rows: list[dict[str, Any]], out: TextIO | None = None) -> None:
    if out is None:
        out = sys.stdout
    basic = compute_basic_outcome(rows)
    discovery = compute_discovery_summary(rows)

    print("A. Basic outcome:", file=out)
    print(f"  runs={basic.runs} ok={basic.ok}", file=out)
    print(
        "  invariants="
        + ("PASS" if basic.invariants_pass else f"FAIL ({basic.invariant_failures} violations)"),
        file=out,
    )
    print(
        f"  interventions={basic.interventions} audits={basic.audits} "
        f"skips={basic.skips} revocations={basic.revocations}",
        file=out,
    )
    print(
        f"  skip_paths: trass={basic.trass_skips} sentinel={basic.sentinel_skips} "
        f"compression={basic.compression_skips}",
        file=out,
    )
    print(
        f"  novelty_open={basic.novelty_open} frontier_stress={basic.frontier_stress}",
        file=out,
    )
    if basic.graph_frontier_evals:
        avg = basic.graph_frontier_recall_total / max(1, basic.runs)
        print(f"  graph_frontier: evals={basic.graph_frontier_evals} avg_chosen_parent_recall={avg:.3f}", file=out)
    if basic.temporal_frontier_evals:
        avg_recall = basic.temporal_frontier_recall_total / max(1, basic.runs)
        avg_lift = basic.temporal_frontier_lift_total / max(1, basic.runs)
        print(
            f"  temporal_frontier: evals={basic.temporal_frontier_evals} "
            f"avg_recall={avg_recall:.3f} avg_lift={avg_lift:.3f}",
            file=out,
        )

    print("\nB. Post-hoc manifest comparison:", file=out)
    print(
        f"  manifests={discovery.manifests} latent_drivers={discovery.latent_count} "
        f"intervention_side_effect_rules={discovery.side_effect_rules}",
        file=out,
    )
    _print_counter("  generated relation types:", discovery.relation_types, out)

    print("\nC. Discovery report:", file=out)
    _print_counter("  structures Dreth appeared to learn:", discovery.appeared_learned, out)
    _print_counter("  structures Dreth failed to learn:", discovery.failed_to_learn, out)
    _print_counter("  structures Dreth over-certified:", discovery.over_certified, out)
    _print_counter("  structures Dreth left uncertain/stressed:", discovery.uncertain_or_stressed, out)
    if basic.graph_frontier_evals or basic.temporal_frontier_evals:
        print("  structures where graph locality helped:", file=out)
        if basic.graph_frontier_evals:
            print("    graph frontier produced post-hoc chosen-parent recall signal", file=out)
        if basic.temporal_frontier_evals:
            print("    temporal frontier produced post-hoc recall/lift signal", file=out)
    else:
        print("  structures where graph locality helped:", file=out)
        print("    (frontier metrics not enabled)", file=out)

    print("\nD. Scope report:", file=out)
    _print_counter("  where Dreth had grip:", discovery.appeared_learned, out)
    _print_counter("  where Dreth lost grip:", discovery.failed_to_learn, out)
    _print_counter("  where Dreth falsely trusted:", discovery.over_certified, out)
    _print_counter("  where Dreth properly withheld authority:", discovery.withheld, out)

    print("\nE. Warning:", file=out)
    print(
        "  This is not proof of real-world capability. It is a blind procedural "
        "stress test for scope discovery.",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize blind_challenge JSONL output offline."
    )
    parser.add_argument("--jsonl", required=True, help="Path to a batch JSONL report")
    args = parser.parse_args(argv)
    print_report(load_jsonl(args.jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
