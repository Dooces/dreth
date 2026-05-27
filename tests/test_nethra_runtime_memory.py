from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_memory_store import ExperienceEvent, NethraMemoryRecord, NethraMemoryStore
from dreth.nethra_runtime_memory import PersistentNethraIndex, SalienceScorer


def _handle(
    record_id: str,
    atoms: list[str],
    *,
    use_right: str = "ranking_hint",
    context: str = "parent_candidates|x0|vis=3",
    success: int = 1,
    failure: int = 0,
    salience: float = 0.0,
    source: str = "runtime",
) -> NethraMemoryRecord:
    return NethraMemoryRecord(
        record_id=record_id,
        record_type="nethra_handle",
        run_id="run-a",
        seed=1,
        schedule="blind_challenge",
        n_vars=3,
        cycle_start=0,
        cycle_end=10,
        vars=[int(a[1:]) for a in atoms if a.startswith("x") and a[1:].isdigit()],
        contexts=[context],
        nethra_id=record_id,
        touched_atoms=atoms,
        touched_structure_refs=["x0:MAX(1,2)", "parents:1,2"],
        member_nethras=[record_id],
        context_scope=context,
        role_history=[{"role": "best_available", "cycle": 1}],
        evidence_refs=[record_id],
        use_right=use_right,
        salience=salience,
        source=source,
        created_cycle=1,
        last_used_cycle=5,
        last_success_cycle=5,
        success_count=success,
        failure_count=failure,
        lift_history=[{"candidate_reduction_lift": 0.5, "audit_saved_lift": 0.25}],
        invalidators=[],
    )


def test_record_mode_loads_and_records_matches_without_reordering():
    index = PersistentNethraIndex(mode="record", run_id="run-b", seed=2)
    index.add_records([_handle("h2", ["x2"], salience=2.0)])
    original = (1, 2)
    ranked = index.rank_candidates(
        var=0,
        context_key="parent_candidates|x0|vis=3",
        candidates=original,
        hook="parent_candidates",
        cycle=10,
    )
    assert ranked == original
    assert index.runtime_metrics()["persistent_nethras_loaded"] == 1
    assert index.runtime_metrics()["nethra_memory_behavior_effects"] == 0
    assert len(index.export_experience_events()) == 1


def test_assist_mode_ranking_hint_reorders_existing_candidates_with_attribution():
    index = PersistentNethraIndex(mode="assist", run_id="run-b", seed=2)
    index.add_records([_handle("h2", ["x2"], salience=2.0)])
    ranked = index.rank_candidates(
        var=0,
        context_key="parent_candidates|x0|vis=3",
        candidates=(1, 2),
        hook="parent_candidates",
        cycle=10,
    )
    assert ranked == (2, 1)
    metrics = index.runtime_metrics()
    assert metrics["nethra_memory_behavior_effects"] == 1
    assert metrics["nethra_memory_authority_effects"] == 0
    event = index.export_experience_events()[-1]
    assert event["behavior_effect"] == 1
    assert event["authority_effect"] == 0
    assert event["candidates_before"] == [1, 2]
    assert event["candidates_after"] == [2, 1]


def test_sleep_hard_filter_is_rejected_on_load():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sleep.jsonl"
        path.write_text(json.dumps({
            "entry_kind": "sleep_product",
            "record_type": "sleep_product",
            "proposal_id": "sleep_bad",
            "member_nethras": ["h1"],
            "touched_atoms": ["x1"],
            "touched_structure_refs": ["parents:1"],
            "proposed_use_right": "hard_filter",
            "proposed_context_scope": "parent_candidates|x0|vis=3",
            "salience_delta": 1.0,
            "evidence_summary": "test",
            "invalidators": [],
            "reason": "test",
            "authority_allowed": True,
        }) + "\n")
        index = PersistentNethraIndex(mode="assist")
        index.load_path(path)
    assert index.records[0].use_right == "record_only"
    assert index.records[0].authority_allowed is False
    assert "sleep_hard_filter_rejected" in index.records[0].invalidators


def test_salience_specific_useful_handle_beats_frequent_broad_handle():
    scorer = SalienceScorer(current_cycle=10)
    broad = _handle(
        "broad",
        ["x0", "x1", "x2", "x3", "x4"],
        success=20,
        salience=0.2,
    )
    specific = _handle(
        "specific",
        ["x2"],
        success=2,
        salience=0.2,
    )
    b = scorer.score(broad, active_atoms=["x2"], context_key="parent_candidates|x0|vis=3")
    s = scorer.score(specific, active_atoms=["x2"], context_key="parent_candidates|x0|vis=3")
    assert s.score > b.score
    assert b.components["broad_atom_penalty"] < 0
    assert s.components["specificity"] > b.components["specificity"]


def test_failed_and_stale_handle_downranks():
    scorer = SalienceScorer(current_cycle=1000)
    useful = _handle("useful", ["x1"], success=2, failure=0)
    useful.last_used_cycle = 995
    useful.last_success_cycle = 995
    failed = _handle("failed", ["x1"], success=2, failure=4)
    stale = _handle("stale", ["x1"], success=2, failure=0)
    stale.last_used_cycle = 1
    stale.last_success_cycle = 1
    assert scorer.score(useful, active_atoms=["x1"], context_key="parent_candidates|x0|vis=3").score > scorer.score(failed, active_atoms=["x1"], context_key="parent_candidates|x0|vis=3").score
    assert scorer.score(useful, active_atoms=["x1"], context_key="parent_candidates|x0|vis=3").score > scorer.score(stale, active_atoms=["x1"], context_key="parent_candidates|x0|vis=3").score


def test_store_appends_and_reloads_experience_events():
    with tempfile.TemporaryDirectory() as td:
        store = NethraMemoryStore(Path(td) / "memory.jsonl")
        event = ExperienceEvent(
            run_id="run-b",
            seed=2,
            cycle=3,
            context_key="parent_candidates|x0|vis=3",
            active_atoms=["x0", "x2"],
            active_nethras=["h2"],
            hook="parent_candidates",
            use_right="ranking_hint",
            candidates_before=[1, 2],
            candidates_after=[2, 1],
            behavior_effect=1,
        )
        assert store.append_experience_events([event]) == 1
        loaded = store.load_experience_events()
        assert len(loaded) == 1
        assert loaded[0].hidden_truth_used is False
        assert loaded[0].behavior_effect == 1
