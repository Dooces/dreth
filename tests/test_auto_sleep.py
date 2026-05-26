from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.auto_sleep import AutoSleepConfig, AutoSleepScheduler
from dreth.nethra_memory_store import NethraMemoryRecord, NethraMemoryStore
from dreth.scaffold_memory import ScaffoldMemoryIndex, compute_run_scaffold_metrics


def _bg_record(run_id: str, seed: int) -> NethraMemoryRecord:
    payload = {
        "nethra_id": "bg:x0:MAX(1)",
        "kind": "unresolved_pattern",
        "vars": [0],
        "context_keys": ["audit|x0|vis=2"],
        "fit_signatures": ["x0:MAX(1)"],
        "parent_sets": [[1]],
        "source_roles": ["unresolved"],
        "first_seen_cycle": 1,
        "last_seen_cycle": 10,
    }
    return NethraMemoryRecord(
        record_id=f"{run_id}:bg",
        record_type="background_nethra",
        run_id=run_id,
        seed=seed,
        schedule="blind_challenge",
        n_vars=2,
        cycle_start=0,
        cycle_end=10,
        vars=[0],
        contexts=["audit|x0|vis=2"],
        source_kind="unresolved_pattern",
        payload=payload,
    )


def test_auto_sleep_run_end_creates_proposals():
    with tempfile.TemporaryDirectory() as td:
        store = NethraMemoryStore(Path(td) / "memory.jsonl")
        store.append_records([_bg_record("run1", 1), _bg_record("run2", 2)])
        cfg = AutoSleepConfig(
            enabled=True,
            memory_path=store.path,
            proposals_path=Path(td) / "props.jsonl",
            summary_path=Path(td) / "summary.txt",
            run_end=True,
            min_sources=2,
        )
        result = AutoSleepScheduler().run_sleep(store, cfg)
        assert result["auto_sleep_triggered"] == 1
        assert result["auto_sleep_proposals"] > 0
        assert result["auto_sleep_authority_allowed_count"] == 0
        assert Path(cfg.proposals_path).exists()
        assert Path(cfg.summary_path).exists()


def test_auto_sleep_does_not_alter_current_run_behavior_metrics():
    with tempfile.TemporaryDirectory() as td:
        store = NethraMemoryStore(Path(td) / "memory.jsonl")
        store.append_records([_bg_record("run1", 1), _bg_record("run2", 2)])
        cfg = AutoSleepConfig(
            enabled=True,
            proposals_path=Path(td) / "props.jsonl",
            summary_path=Path(td) / "summary.txt",
            run_end=True,
            min_sources=2,
        )
        result = AutoSleepScheduler().run_sleep(store, cfg)
        assert result["auto_sleep_behavior_effects"] == 0


def test_auto_load_scaffold_memory_record_mode_loads_proposals():
    with tempfile.TemporaryDirectory() as td:
        store = NethraMemoryStore(Path(td) / "memory.jsonl")
        store.append_records([_bg_record("run1", 1), _bg_record("run2", 2)])
        proposals_path = Path(td) / "props.jsonl"
        AutoSleepScheduler().run_sleep(
            store,
            AutoSleepConfig(
                enabled=True,
                proposals_path=proposals_path,
                summary_path=Path(td) / "summary.txt",
                run_end=True,
                min_sources=2,
            ),
        )
        idx = ScaffoldMemoryIndex()
        loaded = idx.load_proposals(proposals_path)
        metrics = compute_run_scaffold_metrics(
            idx,
            {"records": [_bg_record("current", 3).payload]},
            {"records": []},
            {"records": []},
        )
        assert loaded > 0
        assert metrics["scaffold_memory_matches"] > 0
        assert metrics["scaffold_memory_behavior_effects"] == 0


def test_auto_loaded_scaffold_memory_behavior_equals_off():
    idx = ScaffoldMemoryIndex()
    off_metrics = {
        "skip_count": 10,
        "full_audits": 2,
        "interventions": 4,
    }
    loaded_metrics = {
        **off_metrics,
        **compute_run_scaffold_metrics(idx, {"records": []}, {"records": []}, {"records": []}),
    }
    assert loaded_metrics["skip_count"] == off_metrics["skip_count"]
    assert loaded_metrics["full_audits"] == off_metrics["full_audits"]
    assert loaded_metrics["interventions"] == off_metrics["interventions"]
    assert loaded_metrics["scaffold_memory_behavior_effects"] == 0


def test_threshold_decision_works_but_does_not_mutate_mid_cycle():
    cfg = AutoSleepConfig(enabled=True, cycle_threshold=10, backlog_threshold=5)
    scheduler = AutoSleepScheduler(cfg)
    assert scheduler.should_sleep(cycle=10, backlog_count=5, run_end=False) is False
    assert scheduler.last_reason == "threshold_not_met"
    assert scheduler.should_schedule_boundary_sleep(
        cfg, cycle=10, backlog_count=1, run_end=True
    ) == (True, "cycle_threshold")
    assert scheduler.should_schedule_boundary_sleep(
        cfg, cycle=1, backlog_count=5, run_end=True
    ) == (True, "backlog_threshold")


def test_broad_generic_debt_remains_telemetry_only():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "props.jsonl"
        proposal = {
            "proposal_id": "debt",
            "kind": "authority_debt_family",
            "source_record_ids": ["auth:x0"],
            "source_kinds": ["authority_strength"],
            "vars": [0],
            "contexts": [],
            "common_signatures": [],
            "common_parents": [],
            "role_patterns": ["contested_best_available"],
            "recurring_signals": [],
            "confidence_as_familiarity": 0.4,
            "authority_allowed": True,
        }
        path.write_text(json.dumps(proposal) + "\n")
        idx = ScaffoldMemoryIndex()
        idx.load_proposals(path)
        matches = idx.match_authority_strength_record({
            "var": 0,
            "authority_state": "contested_best_available",
            "reason": "active_visible_conflict",
        })
        assert matches[0].broad_generic_debt is True
        assert matches[0].authority_allowed is False
        assert idx.summarize_matches()["scaffold_memory_behavior_effects"] == 0
