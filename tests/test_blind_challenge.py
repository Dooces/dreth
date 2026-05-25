from __future__ import annotations

import inspect
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dreth.agent import ChainedAgent
from dreth.world import CausalWorld
from scripts.batch_run import RunConfig, _run_one
from summarize_blind_challenge import load_jsonl, print_report


def _cfg(schedule: str, n_vars: int = 8, cycles: int = 12, seed: int = 42) -> RunConfig:
    return RunConfig(
        n_vars=n_vars,
        cycles=cycles,
        seed=seed,
        schedule=schedule,
        settle_cycles=8,
        noise_sigma=0.02,
    )


def test_blind_challenge_runs_n30_cycles300_without_exceptions() -> None:
    result = _run_one(_cfg("blind_challenge", n_vars=30, cycles=300, seed=17))

    assert result.ok, result.error
    assert result.recorded_cycles == 300
    assert result.blind_challenge_evaluation is not None
    assert not result.violations


def test_blind_challenge_observed_values_remain_clipped() -> None:
    world = CausalWorld(20, random.Random(12), noise_sigma=0.03)
    world.prepare_schedule("blind_challenge")

    for cycle in range(1, 80):
        world.perturb_by_schedule(cycle, "blind_challenge")
        assert all(0.0 <= value <= 1.0 for value in world.visible_state)
        predicted = world.predict_under_intervention(0, 0.95)
        assert all(0.0 <= value <= 1.0 for value in predicted)


def test_different_seeds_produce_different_blind_manifests() -> None:
    world_a = CausalWorld(18, random.Random(100), noise_sigma=0.02)
    world_b = CausalWorld(18, random.Random(101), noise_sigma=0.02)
    world_a.prepare_schedule("blind_challenge")
    world_b.prepare_schedule("blind_challenge")

    assert world_a.debug_blind_challenge_manifest() != world_b.debug_blind_challenge_manifest()


def test_chained_agent_does_not_read_blind_debug_manifest() -> None:
    source = inspect.getsource(ChainedAgent)

    assert "debug_blind_challenge_manifest" not in source
    assert "blind_challenge_manifest" not in source


def test_existing_schedules_still_smoke() -> None:
    for schedule in [
        "incremental",
        "periodic_shifts",
        "novelty",
        "shaped",
        "rare_catastrophe",
        "regime_switch",
        "false_trass",
    ]:
        result = _run_one(_cfg(schedule, n_vars=8, cycles=16, seed=9))
        assert result.ok, f"{schedule}: {result.error}"


def test_summarize_blind_challenge_reads_synthetic_jsonl(tmp_path: Path, capsys) -> None:
    path = tmp_path / "blind.jsonl"
    row = {
        "schedule": "blind_challenge",
        "ok": True,
        "interventions": 12,
        "full_audits": 3,
        "trass_skips": 4,
        "sentinel_skips": 5,
        "compression_skips": 1,
        "revoked_by_dist": {"sentinel_failure": 2},
        "violations": [],
        "evaluation": {
            "blind_challenge_manifest": {
                "latents": [{"id": 0}],
                "relations": [
                    {"var": 0, "relation_type": "symbolic"},
                    {"var": 1, "relation_type": "delayed"},
                ],
                "intervention_side_effects": [{"source": 0, "targets": [1]}],
            },
            "blind_challenge_behavior": {
                "per_var": [
                    {
                        "var": 0,
                        "relation_type": "symbolic",
                        "truth_parents": [],
                        "truth_delayed_parents": [],
                        "agent_func_compatible": True,
                        "learned_parents": [],
                        "learned_parent_overlap": [],
                        "status": "certified",
                        "skip_role": "tareth",
                        "authoritative": True,
                    },
                    {
                        "var": 1,
                        "relation_type": "delayed",
                        "truth_parents": [0],
                        "truth_delayed_parents": [0],
                        "agent_func_compatible": False,
                        "learned_parents": [2],
                        "learned_parent_overlap": [],
                        "status": "certified",
                        "skip_role": "tareth",
                        "authoritative": True,
                        "strong_observations": 3,
                        "sentinel_count": 2,
                        "fit_history_count": 2,
                        "last_fit_margin": 2,
                    },
                ]
            },
        },
    }
    path.write_text(json.dumps(row) + "\n")

    rows = load_jsonl(str(path))
    print_report(rows)
    output = capsys.readouterr().out

    assert "Basic outcome" in output
    assert "Post-hoc manifest comparison" in output
    assert "external-truth mismatches under authority" in output
    assert "authority/evidence mismatch candidates" in output
    assert "falsely trusted" not in output
    assert "blind procedural stress test for scope discovery" in output
