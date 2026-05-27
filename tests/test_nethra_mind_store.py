from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.nethra_mind_store import NethraMindStore, effective_use_right
from dreth.nethra_runtime_memory import PersistentNethraIndex


# ── helpers ───────────────────────────────────────────────────────────────────

def _unique_sleep_product(proposal_id: str, idx: int, **kwargs) -> dict:
    """Sleep product with atoms unique to idx so assimilator won't fold distinct nodes."""
    row = _sleep_product(proposal_id, **kwargs)
    row["touched_atoms"] = [f"xu{idx}_a", f"xu{idx}_b"]
    row["touched_structure_refs"] = [f"unique_ref_{idx}"]
    return row


def _sleep_product(
    proposal_id: str = "sp1",
    proposed_use_right: str = "ranking_hint",
    member_nethras: list[str] | None = None,
) -> dict:
    return {
        "entry_kind": "sleep_product",
        "proposal_id": proposal_id,
        "member_nethras": member_nethras or ["n1", "n2"],
        "touched_atoms": ["x0", "x1"],
        "touched_structure_refs": ["parents:0,1"],
        "proposed_use_right": proposed_use_right,
        "proposed_context_scope": "test_ctx",
        "salience_delta": 0.5,
        "evidence_summary": "test evidence",
        "invalidators": [],
        "reason": "test",
        "authority_allowed": False,
    }


def _record_row(
    nethra_id: str = "nid1",
    atoms: list[str] | None = None,
    *,
    context_scope: str = "parent_candidates|x0|vis=3",
    use_right: str = "ranking_hint",
    success_count: int = 0,
    salience: float = 0.0,
    cycle_start: int = 5,
    cycle_end: int = 50,
    lift_history: list[dict] | None = None,
) -> dict:
    atoms = atoms or ["x0", "x1"]
    return {
        "entry_kind": "record",
        "record_id": nethra_id,
        "record_type": "nethra_handle",
        "run_id": "run1",
        "seed": 1,
        "schedule": "blind_challenge",
        "n_vars": 3,
        "cycle_start": cycle_start,
        "cycle_end": cycle_end,
        "nethra_id": nethra_id,
        "touched_atoms": atoms,
        "touched_structure_refs": ["parents:0"],
        "member_nethras": [nethra_id],
        "contexts": [context_scope],
        "context_scope": context_scope,
        "created_cycle": cycle_start,
        "last_used_cycle": cycle_end,
        "use_right": use_right,
        "source": "runtime",
        "success_count": success_count,
        "salience": salience,
        "lift_history": lift_history or [],
        "invalidators": [],
    }


# ── tests ─────────────────────────────────────────────────────────────────────

def test_repeated_sleep_products_fold_into_one_node():
    store = NethraMindStore()
    n = 7
    for i in range(n):
        store.ingest_sleep_product(_sleep_product("sp1"), line_no=i + 1, generation=0)
    assert len(store._nodes) == 1
    node = store._nodes["sp1"]
    assert node.sleep_product_count == n
    assert node.evidence_count == n


def test_temporal_provenance_preserves_first_last_cycle_and_sample_lines():
    store = NethraMindStore()
    row1 = _record_row("nid1", cycle_start=5, cycle_end=50)
    row2 = dict(row1)
    row2["created_cycle"] = 100
    row2["last_used_cycle"] = 200
    store.ingest_record(row1, line_no=10, generation=1)
    store.ingest_record(row2, line_no=20, generation=2)

    node = store._nodes["nid1"]
    assert node.first_seen_cycle == 5
    assert node.last_seen_cycle == 200
    assert node.first_seen_line == 10
    assert node.last_seen_line == 20
    assert node.first_seen_generation == 1
    assert node.last_seen_generation == 2
    assert len(node.sample_evidence_refs) >= 1
    assert len(node.temporal_spans) >= 1


def test_member_nethras_survive_compaction():
    store = NethraMindStore()
    prod = _sleep_product("sp1", member_nethras=["n1", "n2", "n3"])
    store.ingest_sleep_product(prod, line_no=1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)
        store2 = NethraMindStore()
        store2.load(out)

    assert "sp1" in store2._nodes
    node = store2._nodes["sp1"]
    assert set(node.member_nethras) == {"n1", "n2", "n3"}


def test_backpointer_edges_survive_compaction():
    store = NethraMindStore()
    prod = _sleep_product("sp1", member_nethras=["n1", "n2"])
    store.ingest_sleep_product(prod, line_no=1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)
        store2 = NethraMindStore()
        store2.load(out)

    member_edges = [
        e for e in store2._edges.values()
        if e.src == "sp1" and e.relation == "member_of"
    ]
    assert len(member_edges) == 2
    dsts = {e.dst for e in member_edges}
    assert dsts == {"n1", "n2"}


def test_authority_allowed_always_false_in_node_and_dict():
    store = NethraMindStore()
    store.ingest_sleep_product(_sleep_product("sp1"), line_no=1, generation=0)
    node = store._nodes["sp1"]
    assert node.authority_effect_count == 0
    d = node.to_dict()
    assert d["authority_allowed"] is False
    assert d["authority_effect_count"] == 0


def test_hard_filter_from_sleep_is_rejected_and_downgraded():
    store = NethraMindStore()
    prod = _sleep_product("sp1", proposed_use_right="hard_filter")
    store.ingest_sleep_product(prod, line_no=1, generation=0)

    node = store._nodes["sp1"]
    assert "hard_filter" not in node.use_rights_seen
    assert "record_only" in node.use_rights_seen
    assert "sleep_hard_filter_rejected" in node.invalidators
    effective = effective_use_right(node.use_rights_seen)
    assert effective != "hard_filter"


def test_record_mode_equals_off_with_compacted_mind_loaded():
    store = NethraMindStore()
    store.ingest_sleep_product(_sleep_product("sp1"), line_no=1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)

        idx = PersistentNethraIndex(mode="record", run_id="test", seed=0)
        idx.load_path(out)

        original = (1, 2, 3)
        ranked = idx.rank_candidates(
            var=0,
            context_key="test_ctx",
            candidates=original,
            hook="parent_candidates",
            cycle=1,
        )
        assert tuple(ranked) == original
        assert idx.runtime_metrics()["nethra_memory_behavior_effects"] == 0


def test_assist_reorders_using_compacted_mind_with_behavior_effect_attribution():
    store = NethraMindStore()
    store.ingest_record(
        _record_row(
            "nid_x2",
            atoms=["x2"],
            context_scope="parent_candidates|x0|vis=3",
            use_right="ranking_hint",
            success_count=5,
            salience=2.0,
            cycle_start=1,
            cycle_end=100,
            lift_history=[{"candidate_reduction_lift": 1.0}],
        ),
        line_no=1,
        generation=0,
    )

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)

        idx = PersistentNethraIndex(mode="assist", run_id="test", seed=0)
        idx.load_path(out)

        ranked = idx.rank_candidates(
            var=0,
            context_key="parent_candidates|x0|vis=3",
            candidates=(1, 2),
            hook="parent_candidates",
            cycle=50,
        )
        # x2 (candidate 2) should be ranked first
        assert list(ranked)[0] == 2
        m = idx.runtime_metrics()
        assert m["nethra_memory_behavior_effects"] >= 1
        assert m["nethra_memory_authority_effects"] == 0


def test_authority_effects_remain_separate_and_zero():
    store = NethraMindStore()
    for i in range(4):
        store.ingest_sleep_product(_sleep_product(f"sp{i}"), line_no=i + 1, generation=0)

    for node in store._nodes.values():
        assert node.authority_effect_count == 0
        d = node.to_dict()
        assert d["authority_allowed"] is False
        assert d["authority_effect_count"] == 0

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)

        idx = PersistentNethraIndex(mode="assist", run_id="test", seed=0)
        idx.load_path(out)
        idx.rank_candidates(
            var=0, context_key="test_ctx", candidates=(0, 1), hook="pc", cycle=1
        )
        assert idx.runtime_metrics()["nethra_memory_authority_effects"] == 0


def test_hidden_truth_fields_are_not_ingested():
    store = NethraMindStore()

    row_with_truth = _record_row("bad_nid")
    row_with_truth["truth_parents"] = [1, 2]
    store.ingest_record(row_with_truth, line_no=1, generation=0)
    assert "bad_nid" not in store._nodes

    prod_with_debug = _sleep_product("bad_sp")
    prod_with_debug["debug_manifest"] = {"x": 1}
    store.ingest_sleep_product(prod_with_debug, line_no=2, generation=0)
    assert "bad_sp" not in store._nodes

    event_with_hidden_truth = {
        "entry_kind": "experience_event",
        "run_id": "r1",
        "seed": 1,
        "cycle": 5,
        "context_key": "ctx",
        "active_atoms": ["x0"],
        "active_nethras": [],
        "hook": "parent_candidates",
        "use_right": "ranking_hint",
        "behavior_effect": 1,
        "hidden_truth_used": True,
    }
    store.ingest_experience_event(event_with_hidden_truth, line_no=3, generation=0)
    assert store._skipped_hidden_truth >= 3


def test_compacted_file_much_smaller_than_raw_repeated_jsonl():
    n = 50
    raw_rows = [json.dumps(_sleep_product("sp_dup")) + "\n" for _ in range(n)]
    raw_size = sum(len(r.encode()) for r in raw_rows)

    store = NethraMindStore()
    for i in range(n):
        store.ingest_sleep_product(_sleep_product("sp_dup"), line_no=i + 1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)
        compact_size = out.stat().st_size

    assert compact_size < raw_size
    assert len(store._nodes) == 1
    node = store._nodes["sp_dup"]
    assert node.sleep_product_count == n


def test_write_compact_and_load_roundtrip():
    store = NethraMindStore()
    store.ingest_record(
        _record_row("nid_a", atoms=["x0", "x1"], use_right="probe_hint"),
        line_no=1,
        generation=1,
    )
    store.ingest_sleep_product(_sleep_product("sp_b"), line_no=2, generation=1)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        summary = store.write_compact(out)
        assert summary["canonical_nodes"] == 2
        assert summary["canonical_edges"] >= 2

        store2 = NethraMindStore()
        loaded = store2.load(out)
        assert loaded >= 2
        assert "nid_a" in store2._nodes
        assert "sp_b" in store2._nodes


def test_write_report_sections_present():
    store = NethraMindStore()
    store.ingest_sleep_product(_sleep_product("sp1"), line_no=1, generation=0)
    store.ingest_record(_record_row("nid1"), line_no=2, generation=0)

    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "report.txt"
        report = store.write_report(report_path)
        assert report_path.exists()

    for section in ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K."):
        assert section in report, f"missing report section {section}"
    assert "compacted mind is not authority" in report


def test_sleep_product_with_existing_invalidators_preserved():
    store = NethraMindStore()
    prod = _sleep_product("sp1")
    prod["invalidators"] = ["prior_failure", "context_mismatch"]
    store.ingest_sleep_product(prod, line_no=1, generation=0)

    node = store._nodes["sp1"]
    assert "prior_failure" in node.invalidators
    assert "context_mismatch" in node.invalidators


def test_structural_id_stable_across_repeated_ingestion():
    store = NethraMindStore()
    # Two sleep products with no proposal_id but same structure
    prod1 = {
        "entry_kind": "sleep_product",
        "member_nethras": ["m1"],
        "touched_atoms": ["x3"],
        "touched_structure_refs": ["parents:3"],
        "proposed_use_right": "feature_only",
        "proposed_context_scope": "ctx_a",
        "salience_delta": 0.2,
        "evidence_summary": "",
        "invalidators": [],
        "reason": "",
        "authority_allowed": False,
    }
    prod2 = dict(prod1)
    store.ingest_sleep_product(prod1, line_no=1, generation=0)
    store.ingest_sleep_product(prod2, line_no=2, generation=0)
    assert len(store._nodes) == 1
    node = list(store._nodes.values())[0]
    assert node.sleep_product_count == 2


def test_experience_events_update_existing_nodes_only():
    store = NethraMindStore()
    store.ingest_record(_record_row("nid_e"), line_no=1, generation=0)

    event = {
        "entry_kind": "experience_event",
        "run_id": "r1",
        "seed": 1,
        "cycle": 10,
        "context_key": "ctx",
        "active_atoms": ["x0"],
        "active_nethras": ["nid_e", "nid_ghost"],
        "hook": "parent_candidates",
        "use_right": "ranking_hint",
        "behavior_effect": 1,
        "success": True,
        "hidden_truth_used": False,
    }
    store.ingest_experience_event(event, line_no=2, generation=0)

    assert "nid_e" in store._nodes
    assert store._nodes["nid_e"].behavior_effect_count == 1
    assert store._nodes["nid_e"].success_count >= 1
    # nid_ghost was referenced but doesn't exist — must not be created
    assert "nid_ghost" not in store._nodes


def test_authority_effect_count_never_propagated_through_experience_events():
    store = NethraMindStore()
    store.ingest_record(_record_row("nid_auth"), line_no=1, generation=0)

    event = {
        "entry_kind": "experience_event",
        "run_id": "r1",
        "seed": 1,
        "cycle": 5,
        "context_key": "ctx",
        "active_atoms": ["x0"],
        "active_nethras": ["nid_auth"],
        "hook": "parent_candidates",
        "use_right": "ranking_hint",
        "behavior_effect": 1,
        "authority_effect": 99,
        "success": True,
        "hidden_truth_used": False,
    }
    store.ingest_experience_event(event, line_no=2, generation=0)
    assert store._nodes["nid_auth"].authority_effect_count == 0


def test_load_path_loads_mind_nodes_into_persistent_index():
    store = NethraMindStore()
    store.ingest_sleep_product(
        _sleep_product("sp_load", proposed_use_right="ranking_hint"),
        line_no=1,
        generation=0,
    )

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        store.write_compact(out)

        idx = PersistentNethraIndex(mode="off", run_id="r", seed=0)
        loaded = idx.load_path(out)
        assert loaded >= 1
        assert idx.runtime_metrics()["persistent_nethras_loaded"] >= 1


# ── new tests (bounded canonical mind) ────────────────────────────────────────

def test_same_sleep_product_1000_times_single_node():
    store = NethraMindStore()
    for i in range(1000):
        store.ingest_sleep_product(_sleep_product("sp_rep"), line_no=i + 1, generation=0)
    assert len(store._nodes) == 1
    node = store._nodes["sp_rep"]
    assert node.sleep_product_count == 1000
    assert node.evidence_count == 1000


def test_compacting_previous_mind_plus_no_delta_no_growth():
    store = NethraMindStore()
    for i in range(5):
        store.ingest_sleep_product(_sleep_product(f"sp{i}"), line_no=i + 1, generation=0)
    store.ingest_record(_record_row("nid_a"), line_no=10, generation=0)

    with tempfile.TemporaryDirectory() as td:
        mind0 = Path(td) / "mind0.jsonl"
        s0 = store.write_compact(mind0)
        nodes0 = s0["canonical_nodes"]
        edges0 = s0["canonical_edges"]

        # Load mind0 as previous mind + ingest no new delta rows
        store2 = NethraMindStore()
        store2.load(mind0)
        store2.snapshot_delta_start()
        mind1 = Path(td) / "mind1.jsonl"
        s1 = store2.write_compact(mind1)

        assert s1["canonical_nodes"] == nodes0
        assert s1["canonical_edges"] == edges0
        assert s1["rows_ingested"] == 0


def test_nethra_mind_node_rejected_as_fresh_evidence():
    store = NethraMindStore()
    mind_node_row = {
        "entry_kind": "nethra_mind_node",
        "nethra_id": "nid_mind",
        "kind": "sleep_product",
        "touched_atoms": ["x0"],
        "touched_structure_refs": [],
        "member_nethras": [],
        "contexts": [],
        "use_rights_seen": ["ranking_hint"],
        "evidence_count": 99,
        "source_counts": {"mind": 1},
        "salience": 1.0,
        "authority_allowed": False,
        "authority_effect_count": 0,
    }
    result = store.ingest_record(mind_node_row, line_no=1, generation=0)
    assert result is None
    assert "nid_mind" not in store._nodes
    assert store._rows_rejected_mind_derived == 1
    assert store._raw_row_count == 0


def test_source_mind_rejected_as_fresh_evidence():
    store = NethraMindStore()
    row = _record_row("nid_m")
    row["source"] = "mind"
    result = store.ingest_record(row, line_no=1, generation=0)
    assert result is None
    assert "nid_m" not in store._nodes
    assert store._rows_rejected_mind_derived == 1

    # Also test source=sleep_derived_mind
    row2 = _record_row("nid_sd")
    row2["source"] = "sleep_derived_mind"
    result2 = store.ingest_record(row2, line_no=2, generation=0)
    assert result2 is None
    assert store._rows_rejected_mind_derived == 2


def test_proposal_id_record_id_cycle_differ_no_new_node():
    store = NethraMindStore()
    base = _sleep_product("sp_stable")
    # Same structural content, same proposal_id → same node regardless of extra fields
    for i in range(5):
        row = dict(base)
        row["some_extra_cycle"] = i  # doesn't affect structural identity
        store.ingest_sleep_product(row, line_no=i + 1, generation=i)
    assert len(store._nodes) == 1
    assert store._nodes["sp_stable"].sleep_product_count == 5
    assert store._exact_folds == 4  # first creates, 4 subsequent fold


def test_max_node_cap_prunes_low_salience_nodes():
    from dreth.nethra_mind_store import _MAX_NODES

    store = NethraMindStore()
    # Create _MAX_NODES + 50 distinct nodes (unique atoms prevent assimilation folding)
    for i in range(_MAX_NODES + 50):
        row = _unique_sleep_product(f"sp_low_{i}", i)
        row["salience_delta"] = 0.0
        store.ingest_sleep_product(row, line_no=i + 1, generation=0)

    # Before prune
    assert len(store._nodes) == _MAX_NODES + 50

    store.prune_to_cap()
    assert len(store._nodes) <= _MAX_NODES
    assert store._nodes_pruned == 50


def test_invalidators_failure_counts_survive_pruning():
    from dreth.nethra_mind_store import _MAX_NODES

    store = NethraMindStore()
    # Create distinct nodes that exceed the cap (unique atoms prevent assimilation folding)
    for i in range(_MAX_NODES + 20):
        row = _unique_sleep_product(f"sp_bulk_{i}", i)
        row["salience_delta"] = 0.0
        row["invalidators"] = []
        store.ingest_sleep_product(row, line_no=i + 1, generation=0)

    # Add two safety nodes beyond the cap (also unique atoms)
    sp_fail = _unique_sleep_product("sp_fail", _MAX_NODES + 20)
    sp_fail["invalidators"] = ["critical_failure"]
    store.ingest_sleep_product(sp_fail, line_no=_MAX_NODES + 21, generation=0)
    store._nodes["sp_fail"].failure_count = 3

    sp_inv = _sleep_product("sp_inv")
    sp_inv["invalidators"] = ["drift_invalidator"]
    store.ingest_sleep_product(sp_inv, line_no=_MAX_NODES + 22, generation=0)

    store.prune_to_cap()
    # Safety nodes must survive
    assert "sp_fail" in store._nodes
    assert "sp_inv" in store._nodes
    assert store._nodes["sp_fail"].failure_count == 3
    assert "critical_failure" in store._nodes["sp_fail"].invalidators


def test_active_mind_100x_smaller_than_raw_duplicate_input():
    n = 1000
    raw_bytes = sum(len(json.dumps(_sleep_product("sp_dup")).encode()) for _ in range(n))

    store = NethraMindStore()
    for i in range(n):
        store.ingest_sleep_product(_sleep_product("sp_dup"), line_no=i + 1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        summary = store.write_compact(out)
        compact_bytes = summary["active_mind_bytes"]

    assert compact_bytes * 100 < raw_bytes, (
        f"compact({compact_bytes}B) should be 100x smaller than raw({raw_bytes}B)"
    )
    assert len(store._nodes) == 1
    assert store._nodes["sp_dup"].sleep_product_count == n


def test_mind_edge_entry_kind_rejected():
    store = NethraMindStore()
    edge_row = {
        "entry_kind": "nethra_mind_edge",
        "src": "a",
        "dst": "b",
        "relation": "member_of",
        "count": 10,
    }
    result = store.ingest_sleep_product(edge_row, line_no=1, generation=0)
    assert result is None
    assert store._rows_rejected_mind_derived == 1
    assert store._raw_row_count == 0


def test_write_compact_report_fields_present():
    store = NethraMindStore()
    store.ingest_sleep_product(_sleep_product("sp1"), line_no=1, generation=0)
    store.ingest_record(_record_row("nid1"), line_no=2, generation=0)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mind.jsonl"
        summary = store.write_compact(out)

    required = [
        "raw_rows_read", "rows_rejected_mind_derived", "rows_rejected_compacted",
        "rows_ingested", "nodes_before", "nodes_after", "edges_after",
        "exact_folds", "structural_folds", "sleep_products_folded",
        "nodes_pruned", "edges_pruned", "active_mind_bytes", "compression_ratio",
    ]
    for field in required:
        assert field in summary, f"missing summary field: {field}"


def test_delta_compaction_snapshot_tracks_nodes_before():
    store = NethraMindStore()
    # Create mind0 with 3 distinct nodes (unique atoms so assimilator doesn't fold them)
    for i in range(3):
        store.ingest_sleep_product(_unique_sleep_product(f"sp_base_{i}", i), line_no=i + 1, generation=0)

    with tempfile.TemporaryDirectory() as td:
        mind0 = Path(td) / "mind0.jsonl"
        store.write_compact(mind0)

        # Load mind0 into store2, then ingest 2 more nodes as delta
        store2 = NethraMindStore()
        store2.load(mind0)
        store2.snapshot_delta_start()

        assert store2._nodes_before_delta == 3

        store2.ingest_sleep_product(_unique_sleep_product("sp_delta_1", 10), line_no=1, generation=1)
        store2.ingest_sleep_product(_unique_sleep_product("sp_delta_2", 11), line_no=2, generation=1)

        mind1 = Path(td) / "mind1.jsonl"
        s1 = store2.write_compact(mind1)

        assert s1["nodes_before"] == 3
        assert s1["nodes_after"] == 5
        assert s1["rows_ingested"] == 2


def test_invalidator_counts_tracked_per_occurrence():
    store = NethraMindStore()
    prod = _sleep_product("sp_inv")
    prod["invalidators"] = ["reason_a"]
    store.ingest_sleep_product(prod, line_no=1, generation=0)
    # Second ingestion adds the same invalidator again
    store.ingest_sleep_product(prod, line_no=2, generation=0)

    node = store._nodes["sp_inv"]
    assert node.invalidator_counts.get("reason_a", 0) == 2
    # Still only one entry in the deduplicated list
    assert node.invalidators.count("reason_a") == 1


def test_write_report_sections_present_with_new_sections():
    store = NethraMindStore()
    store.ingest_sleep_product(_sleep_product("sp1"), line_no=1, generation=0)
    store.ingest_record(_record_row("nid1"), line_no=2, generation=0)

    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "report.txt"
        report = store.write_report(report_path)
        assert report_path.exists()

    for section in ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.", "L.", "M."):
        assert section in report, f"missing report section {section}"
    assert "compacted mind is not authority" in report
    assert "rows_rejected_mind_derived" in report
    assert "nodes_pruned" in report


# ── Step 2: surface fields in NethraMindNode ──────────────────────────────────

def test_mind_node_loads_without_surface_fields_from_old_file(tmp_path):
    """Old compact files that lack role_surfaces/residual_buckets load cleanly."""
    import json
    from dreth.nethra_mind_store import NethraMindStore

    old_node = {
        "entry_kind": "nethra_mind_node",
        "nethra_id": "old_n1",
        "touched_atoms": ["x1"],
        "contexts": ["ctx_A"],
        "use_rights_seen": ["ranking_hint"],
        "evidence_count": 3,
        # No role_surfaces, no residual_buckets, no surface_transitions
    }
    mind_file = tmp_path / "old_mind.jsonl"
    mind_file.write_text(json.dumps(old_node) + "\n")

    store = NethraMindStore()
    loaded = store.load(mind_file)
    assert loaded == 1

    node = store._nodes["old_n1"]
    assert node.role_surfaces == {}
    assert node.residual_buckets == {}
    assert node.surface_transitions == []
    assert node.authority_effect_count == 0


def test_mind_node_serializes_surface_fields_in_new_file(tmp_path):
    """New compact files include role_surfaces, residual_buckets, surface_transitions."""
    import json
    from dreth.nethra_mind_store import NethraMindStore, NethraMindNode

    store = NethraMindStore()
    store.upsert_node(
        "n_new",
        touched_atoms=["x1"],
        contexts=["ctx_B"],
        use_right="ranking_hint",
    )
    # Manually attach surface data to the node (as would be done by a future integration)
    node = store._nodes["n_new"]
    node.role_surfaces["ctx_B"] = {"role_state": "tareth", "load_bearing_score": 0.8}
    node.residual_buckets["ctx_B"] = {"pressure": 1.5, "unresolved_count": 3}
    node.surface_transitions.append({"operation": "PROMOTE_ROLE", "cycle": 5})

    out = tmp_path / "new_mind.jsonl"
    store.write_compact(out)

    # Load back and verify round-trip
    store2 = NethraMindStore()
    store2.load(out)
    node2 = store2._nodes["n_new"]
    assert "ctx_B" in node2.role_surfaces
    assert node2.role_surfaces["ctx_B"]["role_state"] == "tareth"
    assert "ctx_B" in node2.residual_buckets
    assert node2.residual_buckets["ctx_B"]["pressure"] == 1.5
    assert len(node2.surface_transitions) >= 1


def test_mind_compaction_serializes_surface_buckets(tmp_path):
    """Compacted mind output contains role_surfaces and residual_buckets on nodes."""
    import json
    from dreth.nethra_mind_store import NethraMindStore

    store = NethraMindStore()
    store.upsert_node("n_compact", touched_atoms=["x1"], use_right="ranking_hint")
    node = store._nodes["n_compact"]
    node.residual_buckets["ctx_X"] = {"pressure": 2.0, "absorbed_count": 5}

    out = tmp_path / "compact.jsonl"
    store.write_compact(out)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    mind_nodes = [r for r in rows if r.get("entry_kind") == "nethra_mind_node"]
    assert mind_nodes, "expected at least one nethra_mind_node"
    target = next((r for r in mind_nodes if r["nethra_id"] == "n_compact"), None)
    assert target is not None
    assert "residual_buckets" in target
    assert "ctx_X" in target["residual_buckets"]


def test_repeated_residuals_do_not_write_unbounded_raw_rows(tmp_path):
    """After many partial-overlap ingestions, residual rows stay bounded."""
    from dreth.nethra_mind_store import NethraMindStore
    import json

    store = NethraMindStore()
    # First: create a canonical node
    store.ingest_record(_record_row("seed", atoms=["xa", "xb", "xc"]), line_no=1, generation=1)

    # Now ingest many partial-overlap rows
    for i in range(50):
        partial = {
            "record_id": f"partial_{i}",
            "touched_atoms": ["xa"],   # partial overlap with seed (1/3 atoms)
            "context_scope": "ctx_partial",
            "use_right": "record_only",
            "source": "runtime",
        }
        store.ingest_record(partial, line_no=i + 2, generation=1)

    out = tmp_path / "mind.jsonl"
    store.write_compact(out)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    residual_rows = [r for r in rows if r.get("entry_kind") == "nethra_residual"]
    # Residual rows are bounded by ResidualIndex.max_size
    assert len(residual_rows) <= 200


def test_authority_allowed_remains_false_in_compacted_mind(tmp_path):
    """After compaction and reload, authority_allowed and authority_effect_count stay zero."""
    import json
    from dreth.nethra_mind_store import NethraMindStore

    store = NethraMindStore()
    store.upsert_node("n_auth", touched_atoms=["x1"], use_right="ranking_hint")
    # Even if we try to set authority_effect_count, it should reset on upsert
    store._nodes["n_auth"].authority_effect_count = 0  # invariant already enforced

    out = tmp_path / "auth_mind.jsonl"
    store.write_compact(out)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    node_rows = [r for r in rows if r.get("entry_kind") == "nethra_mind_node"]
    for row in node_rows:
        assert row.get("authority_allowed") is False
        assert row.get("authority_effect_count") == 0

    # Also verify after reload
    store2 = NethraMindStore()
    store2.load(out)
    for node in store2._nodes.values():
        assert node.authority_effect_count == 0
