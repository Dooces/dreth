from __future__ import annotations

"""Automatic offline sleep scheduling for persistent Nethra memory.

Auto sleep runs only at run boundaries in this first implementation. It calls
MemorySleepConsolidator, writes scaffold proposals and a summary, and never
mutates a live agent or current-run authority state.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory_sleep import MemorySleepConsolidator, MemorySleepSummary
from dreth.nethra_memory_store import NethraMemoryStore


@dataclass
class AutoSleepConfig:
    enabled: bool = False
    memory_path: str | Path = "reports/nethra_memory_store.jsonl"
    proposals_path: str | Path = "reports/auto_sleep_proposals.jsonl"
    summary_path: str | Path = "reports/auto_sleep_summary.txt"
    cycle_threshold: int = 0
    backlog_threshold: int = 0
    run_end: bool = False
    max_proposals: int = 2000
    max_sources_per_proposal: int = 500
    min_sources: int = 2


class AutoSleepScheduler:
    def __init__(self, config: AutoSleepConfig | None = None) -> None:
        self.config = config
        self.last_reason = ""

    def should_sleep(self, cycle: int, backlog_count: int, run_end: bool = False) -> bool:
        config = self.config or AutoSleepConfig(enabled=True, run_end=True)
        should, reason = self.should_schedule_boundary_sleep(
            config,
            cycle=cycle,
            backlog_count=backlog_count,
            run_end=run_end,
        )
        self.last_reason = reason
        return should

    def should_schedule_boundary_sleep(
        self,
        config: AutoSleepConfig,
        *,
        cycle: int,
        backlog_count: int,
        run_end: bool,
    ) -> tuple[bool, str]:
        if not config.enabled:
            return False, "disabled"
        if config.run_end and run_end:
            return True, "run_end"
        if run_end and config.cycle_threshold > 0 and cycle >= config.cycle_threshold:
            return True, "cycle_threshold"
        if run_end and config.backlog_threshold > 0 and backlog_count >= config.backlog_threshold:
            return True, "backlog_threshold"
        return False, "threshold_not_met"

    def run_sleep(
        self,
        memory_store: NethraMemoryStore,
        config: AutoSleepConfig,
    ) -> dict[str, Any]:
        rows = memory_store.to_sleep_rows()
        consolidator = MemorySleepConsolidator()
        bg = consolidator.extract_background_records(rows)
        cr = consolidator.extract_context_role_records(rows)
        unc = consolidator.extract_uncertainty_records(rows)
        auth = consolidator.extract_authority_records(rows)
        temp = consolidator.extract_temporal_records_if_available(rows)
        proposals = consolidator.build_proposals(
            bg,
            cr,
            unc,
            auth,
            temp,
            min_sources=config.min_sources,
            max_proposals=config.max_proposals,
            max_sources_per_proposal=config.max_sources_per_proposal,
        )
        summary = consolidator.summarize(rows, bg, cr, unc, auth, temp, proposals)

        proposals_path = Path(config.proposals_path)
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proposals_path, "w") as fh:
            for proposal in proposals:
                d = proposal.to_dict()
                d["authority_allowed"] = False
                fh.write(json.dumps(d, sort_keys=True) + "\n")

        summary_path = Path(config.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_fmt_auto_sleep_summary(summary), encoding="utf-8")

        return {
            "auto_sleep_triggered": 1,
            "auto_sleep_input_records": len(bg) + len(cr) + len(auth) + len(temp),
            "auto_sleep_proposals": len(proposals),
            "auto_sleep_authority_allowed_count": summary.authority_allowed_count,
            "auto_sleep_compression_ratio": summary.compression_ratio,
            "auto_sleep_behavior_effects": 0,
            "auto_sleep_proposals_path": str(proposals_path),
            "auto_sleep_summary_path": str(summary_path),
        }

    def record_sleep_result(
        self,
        memory_store: NethraMemoryStore,
        *,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        memory_store.append_sleep_result({
            "auto_sleep_reason": reason,
            **result,
            "authority_allowed": False,
        })


def empty_auto_sleep_metrics() -> dict[str, Any]:
    return {
        "auto_sleep_triggered": 0,
        "auto_sleep_reason": "",
        "auto_sleep_input_records": 0,
        "auto_sleep_proposals": 0,
        "auto_sleep_authority_allowed_count": 0,
        "auto_sleep_compression_ratio": 0.0,
        "auto_sleep_behavior_effects": 0,
    }


def _fmt_auto_sleep_summary(summary: MemorySleepSummary) -> str:
    lines = [
        "AutoSleep MemorySleep Summary",
        "=" * 64,
        "",
        "A. input inventory",
        f"  rows read: {summary.input_rows}",
        f"  background records seen: {summary.background_records_seen}",
        f"  context-role records seen: {summary.context_role_records_seen}",
        f"  uncertainty records seen: {summary.uncertainty_records_seen}",
        f"  authority records seen: {summary.authority_records_seen}",
        f"  temporal records seen: {summary.temporal_records_seen}",
        "",
        "B. scaffold proposals",
        f"  proposals: {len(summary.proposals)}",
        f"  proposals by kind: {summary.proposals_by_kind}",
        f"  compression ratio: {summary.compression_ratio}",
        "",
        "C. authority boundary",
        f"  authority_allowed_count: {summary.authority_allowed_count}",
        "  behavior_effects: 0",
        "",
        "D. warning",
        "  Persistent memory is familiarity/provenance infrastructure, not authority.",
    ]
    return "\n".join(lines) + "\n"
