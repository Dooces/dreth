from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_memory_store import (
    NethraMemoryRecord,
    NethraMemoryStore,
    records_from_batch_record,
)


def test_memory_store_appends_and_reloads_records():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.jsonl"
        store = NethraMemoryStore(path)
        rec = NethraMemoryRecord(
            record_id="r1",
            record_type="background_nethra",
            run_id="run1",
            seed=42,
            schedule="blind_challenge",
            n_vars=3,
            cycle_start=0,
            cycle_end=10,
            vars=[1],
            contexts=["audit|x1"],
            source_kind="unresolved_pattern",
            payload={"nethra_id": "n1", "kind": "unresolved_pattern", "vars": [1]},
        )
        assert store.append_records([rec]) == 1
        loaded = store.load_records()
        assert len(loaded) == 1
        assert loaded[0].record_id == "r1"
        assert loaded[0].authority_allowed is False


def test_memory_store_never_sets_authority_allowed_true_by_default():
    rec = NethraMemoryRecord(
        record_id="r1",
        record_type="authority_strength",
        run_id="run1",
        seed=1,
        schedule="blind_challenge",
        n_vars=2,
        cycle_start=0,
        cycle_end=1,
        vars=[0],
        contexts=[],
        source_kind="contested",
        payload={"authority_allowed": True},
        authority_allowed=True,
    )
    assert rec.authority_allowed is False
    assert rec.to_dict()["authority_allowed"] is False


def test_hidden_truth_debug_manifest_fields_are_not_persisted():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.jsonl"
        store = NethraMemoryStore(path)
        recs = records_from_batch_record({
            "run_id": "run1",
            "seed": 1,
            "schedule": "blind_challenge",
            "n_vars": 2,
            "cycles": 5,
            "background_nethra_export": {
                "records": [{
                    "nethra_id": "n1",
                    "kind": "unresolved_pattern",
                    "vars": [0],
                    "truth_parents": [1],
                    "debug_blind_challenge_manifest": {"x": 1},
                    "payload": {"truth_func": "MAX", "visible": True},
                }]
            },
        })
        store.append_records(recs)
        row = json.loads(path.read_text().strip())
        payload = row["payload"]
        assert "truth_parents" not in payload
        assert "debug_blind_challenge_manifest" not in payload
        assert "truth_func" not in payload["payload"]


def test_load_scaffold_proposals_from_memory_records():
    with tempfile.TemporaryDirectory() as td:
        store = NethraMemoryStore(Path(td) / "memory.jsonl")
        store.append_records([
            NethraMemoryRecord(
                record_id="p1",
                record_type="scaffold_proposal",
                run_id="run1",
                seed=1,
                schedule="blind_challenge",
                n_vars=2,
                cycle_start=0,
                cycle_end=1,
                vars=[],
                contexts=[],
                source_kind="scaffold_memory_match",
                payload={"proposal_id": "prop_1", "authority_allowed": True},
            )
        ])
        props = store.load_scaffold_proposals()
        assert props == [{"proposal_id": "prop_1", "authority_allowed": True}]
        assert store.summarize()["authority_allowed_count"] == 0
