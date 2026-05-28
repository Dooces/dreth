"""Tests for ProjectionEntry and ProjectionIndex."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from dreth.learner.nethra_projection import ProjectionIndex, ProjectionEntry, _HOOK_USE_RIGHTS


# ── helpers ───────────────────────────────────────────────────────────────────

def make_node(nethra_id, atoms=None, contexts=None, use_rights=None,
              salience=0.0, behavior=0, failure=0, success=0):
    node = MagicMock()
    node.nethra_id = nethra_id
    node.touched_atoms = atoms or []
    node.contexts = contexts or []
    node.use_rights_seen = use_rights or []
    node.salience = salience
    node.behavior_effect_count = behavior
    node.failure_count = failure
    node.success_count = success
    return node


def make_mind_row(nethra_id, atoms=None, contexts=None, use_rights=None,
                  salience=0.0, behavior=0, failure=0):
    return {
        "entry_kind": "nethra_mind_node",
        "nethra_id": nethra_id,
        "touched_atoms": atoms or [],
        "contexts": contexts or [],
        "use_rights_seen": use_rights or [],
        "salience": salience,
        "behavior_effect_count": behavior,
        "failure_count": failure,
        "success_count": 0,
    }


# ── basic indexing ────────────────────────────────────────────────────────────

def test_index_node_adds_entry():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x1"], use_rights=["ranking_hint"]))
    assert pi.size() == 1


def test_index_node_from_row_adds_entry():
    pi = ProjectionIndex()
    pi.index_node_from_row(make_mind_row("n1", atoms=["x1"], use_rights=["ranking_hint"]))
    assert pi.size() == 1


def test_index_node_from_row_ignores_non_mind_node():
    pi = ProjectionIndex()
    pi.index_node_from_row({"entry_kind": "record", "nethra_id": "n1"})
    assert pi.size() == 0


def test_index_node_from_row_ignores_missing_id():
    pi = ProjectionIndex()
    pi.index_node_from_row({"entry_kind": "nethra_mind_node"})
    assert pi.size() == 0


def test_remove_node_reduces_size():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x1"], use_rights=["ranking_hint"]))
    pi.remove_node("n1")
    assert pi.size() == 0


def test_remove_nonexistent_node_is_safe():
    pi = ProjectionIndex()
    pi.remove_node("nonexistent")  # should not raise


# ── reindex on second call ────────────────────────────────────────────────────

def test_reindex_on_second_index_node_call():
    """Calling index_node again with new atoms updates the entry."""
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x1"], use_rights=["ranking_hint"]))
    pi.index_node(make_node("n1", atoms=["x1", "x2"], use_rights=["ranking_hint"]))
    assert pi.size() == 1
    result = pi.query("x2", "", "ranking_hint")
    assert any(e.nethra_id == "n1" for e in result)


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_matching_node_by_atom():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x5"], use_rights=["ranking_hint"]))
    result = pi.query("x5", "", "ranking_hint")
    assert len(result) == 1
    assert result[0].nethra_id == "n1"


def test_query_filters_by_hook_use_right():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x5"], use_rights=["probe_hint"]))
    # probe_hint nodes should NOT appear for ranking_hint query
    result = pi.query("x5", "", "ranking_hint")
    assert not any(e.nethra_id == "n1" for e in result)


def test_query_soft_filter_appears_in_ranking_hook():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x5"], use_rights=["soft_filter"]))
    result = pi.query("x5", "", "ranking_hint")
    assert any(e.nethra_id == "n1" for e in result)


def test_query_soft_filter_appears_in_probe_hook():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x5"], use_rights=["soft_filter"]))
    result = pi.query("x5", "", "probe_hint")
    assert any(e.nethra_id == "n1" for e in result)


def test_query_context_bonus_raises_rank():
    pi = ProjectionIndex()
    pi.index_node(make_node("n_match", atoms=["x5"], contexts=["ctx_A"], use_rights=["ranking_hint"]))
    pi.index_node(make_node("n_other", atoms=["x5"], contexts=["ctx_B"], use_rights=["ranking_hint"]))
    result = pi.query("x5", "ctx_A", "ranking_hint")
    ids = [e.nethra_id for e in result]
    # n_match (matching context) should rank first
    assert ids.index("n_match") < ids.index("n_other")


def test_query_top_k_limits_results():
    pi = ProjectionIndex()
    for i in range(30):
        pi.index_node(make_node(f"n{i}", atoms=["x1"], use_rights=["ranking_hint"]))
    result = pi.query("x1", "", "ranking_hint", top_k=10)
    assert len(result) <= 10


def test_query_empty_index_returns_empty():
    pi = ProjectionIndex()
    result = pi.query("x1", "ctx", "ranking_hint")
    assert result == []


def test_query_unknown_hook_returns_all_with_any_use_right():
    pi = ProjectionIndex()
    pi.index_node(make_node("n1", atoms=["x1"], use_rights=["ranking_hint"]))
    # "unknown_hook" is not in _HOOK_USE_RIGHTS so allowed=None → no filter
    result = pi.query("x1", "", "unknown_hook")
    assert any(e.nethra_id == "n1" for e in result)


# ── scoring ───────────────────────────────────────────────────────────────────

def test_behavior_effect_bonus():
    pi = ProjectionIndex()
    pi.index_node(make_node("n_high", atoms=["x1"], use_rights=["ranking_hint"],
                             behavior=5, salience=0.0))
    pi.index_node(make_node("n_low", atoms=["x1"], use_rights=["ranking_hint"],
                             behavior=0, salience=0.0))
    result = pi.query("x1", "", "ranking_hint")
    ids = [e.nethra_id for e in result]
    assert ids.index("n_high") < ids.index("n_low")


def test_failure_count_penalty():
    pi = ProjectionIndex()
    pi.index_node(make_node("n_clean", atoms=["x1"], use_rights=["ranking_hint"],
                             failure=0))
    pi.index_node(make_node("n_failed", atoms=["x1"], use_rights=["ranking_hint"],
                             failure=3))
    result = pi.query("x1", "", "ranking_hint")
    ids = [e.nethra_id for e in result]
    assert ids.index("n_clean") < ids.index("n_failed")


# ── load roundtrip ────────────────────────────────────────────────────────────

def test_index_node_from_row_use_right_summary_fallback():
    """Handles compact mind format that has use_right_summary instead of use_rights_seen."""
    pi = ProjectionIndex()
    row = {
        "entry_kind": "nethra_mind_node",
        "nethra_id": "n1",
        "touched_atoms": ["x1"],
        "contexts": [],
        "use_rights_seen": [],
        "use_right_summary": "ranking_hint",
        "salience": 0.5,
    }
    pi.index_node_from_row(row)
    result = pi.query("x1", "", "ranking_hint")
    assert any(e.nethra_id == "n1" for e in result)


# ── integration with NethraMindStore ─────────────────────────────────────────

def test_projection_populated_after_store_load(tmp_path):
    """After loading a compact mind, ProjectionIndex inside the store is populated."""
    import json
    from dreth.learner.nethra_mind_store import NethraMindStore

    mind_file = tmp_path / "mind.jsonl"
    node_row = {
        "entry_kind": "nethra_mind_node",
        "nethra_id": "n1",
        "touched_atoms": ["x1", "x2"],
        "contexts": ["ctx_A"],
        "use_rights_seen": ["ranking_hint"],
        "salience": 0.8,
        "behavior_effect_count": 3,
        "failure_count": 0,
        "success_count": 5,
        "evidence_count": 10,
    }
    mind_file.write_text(json.dumps(node_row) + "\n")

    store = NethraMindStore()
    store.load(mind_file)
    assert store._projection.size() == 1


def test_projection_populated_after_upsert(tmp_path):
    """After upsert_node, the projection index contains the node."""
    from dreth.learner.nethra_mind_store import NethraMindStore

    store = NethraMindStore()
    store.upsert_node(
        "n1",
        touched_atoms=["x1", "x2"],
        contexts=["ctx_A"],
        use_right="ranking_hint",
        salience=0.5,
    )
    assert store._projection.size() == 1
    result = store._projection.query("x1", "ctx_A", "ranking_hint")
    assert any(e.nethra_id == "n1" for e in result)


# ── Step 2: role-surface gating ───────────────────────────────────────────────

def test_projection_index_gates_primary_projection_by_surface_role():
    """A trass-role node is not returned for primary projection hooks."""
    pi = ProjectionIndex()
    node = make_node("n_trass", atoms=["x1"], use_rights=["ranking_hint"])
    # Override via direct entry to set role_state
    pi.index_node(node)
    # Manually set the role_state to trass after indexing to simulate surface-gating
    pi._entries["n_trass"].role_state = "trass"

    result = pi.query("x1", "", "ranking_hint")
    assert not any(e.nethra_id == "n_trass" for e in result), \
        "trass surface must not emit primary ranking_hint projection"


def test_trass_surface_does_not_alter_primary_candidate_projection():
    """Trass surface node is excluded from source_edge_candidates projection."""
    pi = ProjectionIndex()
    node = make_node("n_trass2", atoms=["x2"], use_rights=["ranking_hint"])
    pi.index_node(node)
    pi._entries["n_trass2"].role_state = "trass"

    result_source_edge = pi.query("x2", "", "source_edge_candidates")
    assert not any(e.nethra_id == "n_trass2" for e in result_source_edge)

    result_probe = pi.query("x2", "", "probe_hint")
    assert not any(e.nethra_id == "n_trass2" for e in result_probe)


def test_best_available_projection_is_bounded():
    """Best-available surface nodes are NOT blocked; they can emit bounded weak projections."""
    pi = ProjectionIndex()
    node = make_node("n_ba", atoms=["x3"], use_rights=["ranking_hint"])
    pi.index_node(node)
    pi._entries["n_ba"].role_state = "best_available"

    result = pi.query("x3", "", "ranking_hint")
    assert any(e.nethra_id == "n_ba" for e in result), \
        "best_available must still emit for ranking_hint"


def test_tareth_surface_emits_normally():
    """Tareth-role nodes always emit for primary hooks."""
    pi = ProjectionIndex()
    node = make_node("n_ta", atoms=["x4"], use_rights=["ranking_hint"])
    pi.index_node(node)
    pi._entries["n_ta"].role_state = "tareth"

    result = pi.query("x4", "", "ranking_hint")
    assert any(e.nethra_id == "n_ta" for e in result)


def test_role_state_set_from_node_roles_by_context():
    """index_node reads roles_by_context and sets role_state on the entry."""
    pi = ProjectionIndex()
    node = make_node("n_rbc", atoms=["x5"], use_rights=["ranking_hint"])
    node.roles_by_context = {"ctx_X": ["trass"]}
    pi.index_node(node)
    assert pi._entries["n_rbc"].role_state == "trass"


def test_unresolved_surface_blocked_from_primary_hooks():
    """Unresolved surface is also blocked from primary projection hooks."""
    pi = ProjectionIndex()
    node = make_node("n_unres", atoms=["x6"], use_rights=["probe_hint"])
    pi.index_node(node)
    pi._entries["n_unres"].role_state = "unresolved"

    result = pi.query("x6", "", "probe_hint")
    assert not any(e.nethra_id == "n_unres" for e in result)
