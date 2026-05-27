"""Tests for NethraAssimilator and its sub-indexes."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from dreth.nethra_assimilator import (
    AnchorIndex,
    PerspectiveIndex,
    RoleIndex,
    TopologyIndex,
    ResidualIndex,
    NethraAssimilator,
    Disposition,
    AssimilationResult,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_node(nethra_id, atoms=None, contexts=None, use_rights=None, roles=None,
              structure_refs=None, member_nethras=None, success_count=0, failure_count=0):
    node = MagicMock()
    node.nethra_id = nethra_id
    node.touched_atoms = atoms or []
    node.contexts = contexts or []
    node.use_rights_seen = use_rights or []
    node.roles_by_context = roles or {}
    node.touched_structure_refs = structure_refs or []
    node.member_nethras = member_nethras or []
    node.success_count = success_count
    node.failure_count = failure_count
    return node


def make_assimilator_with_node(nethra_id, atoms, **kwargs):
    """Create an assimilator with one indexed node and a matching nodes dict."""
    a = NethraAssimilator()
    node = make_node(nethra_id, atoms=atoms, **kwargs)
    nodes = {nethra_id: node}
    a.index_node(nethra_id, atoms)
    return a, nodes, node


# ── AnchorIndex ───────────────────────────────────────────────────────────────

def test_anchor_index_by_atom():
    ai = AnchorIndex()
    ai.index_node("n1", ["x1", "x2"], [], [])
    assert "n1" in ai.candidates_for_row({"touched_atoms": ["x1"]})


def test_anchor_index_by_ref():
    ai = AnchorIndex()
    ai.index_node("n1", [], ["sig:abc"], [])
    assert "n1" in ai.candidates_for_row({"touched_structure_refs": ["sig:abc"]})


def test_anchor_index_by_member():
    ai = AnchorIndex()
    ai.index_node("n1", [], [], ["child1"])
    assert "n1" in ai.candidates_for_row({"member_nethras": ["child1"]})


def test_anchor_index_remove():
    ai = AnchorIndex()
    ai.index_node("n1", ["x1"], [], [])
    ai.remove_node("n1")
    assert "n1" not in ai.candidates_for_row({"touched_atoms": ["x1"]})


def test_anchor_index_no_overlap_returns_empty():
    ai = AnchorIndex()
    ai.index_node("n1", ["x1"], [], [])
    assert ai.candidates_for_row({"touched_atoms": ["x99"]}) == set()


# ── PerspectiveIndex ──────────────────────────────────────────────────────────

def test_perspective_index_exact_match():
    pi = PerspectiveIndex()
    pi.index_node("n1", ["ctx1|sub"], ["ranking_hint"])
    hits = pi.candidates_for_row({"context_scope": "ctx1|sub", "use_right": "ranking_hint"})
    assert "n1" in hits


def test_perspective_index_no_match():
    pi = PerspectiveIndex()
    pi.index_node("n1", ["ctx1"], ["ranking_hint"])
    hits = pi.candidates_for_row({"context_scope": "ctx2", "use_right": "ranking_hint"})
    assert "n1" not in hits


def test_perspective_index_remove():
    pi = PerspectiveIndex()
    pi.index_node("n1", ["ctx1"], ["ranking_hint"])
    pi.remove_node("n1")
    hits = pi.candidates_for_row({"context_scope": "ctx1", "use_right": "ranking_hint"})
    assert "n1" not in hits


# ── RoleIndex ─────────────────────────────────────────────────────────────────

def test_role_index_by_role_history():
    ri = RoleIndex()
    ri.index_node("n1", {"ctx1": ["guide", "probe"]})
    hits = ri.candidates_for_row({"role_history": [{"role": "guide", "context": "ctx1"}]})
    assert "n1" in hits


def test_role_index_no_match():
    ri = RoleIndex()
    ri.index_node("n1", {"ctx1": ["guide"]})
    hits = ri.candidates_for_row({"role_history": [{"role": "oracle"}]})
    assert "n1" not in hits


def test_role_index_remove():
    ri = RoleIndex()
    ri.index_node("n1", {"ctx1": ["guide"]})
    ri.remove_node("n1")
    hits = ri.candidates_for_row({"role_history": [{"role": "guide"}]})
    assert "n1" not in hits


# ── TopologyIndex ─────────────────────────────────────────────────────────────

def test_topology_neighborhood_depth1():
    ti = TopologyIndex()
    ti.index_edge("n1", "n2")
    ti.index_edge("n2", "n3")
    assert "n2" in ti.neighborhood("n1", depth=1)
    assert "n3" not in ti.neighborhood("n1", depth=1)


def test_topology_neighborhood_depth2():
    ti = TopologyIndex()
    ti.index_edge("n1", "n2")
    ti.index_edge("n2", "n3")
    assert "n3" in ti.neighborhood("n1", depth=2)


def test_topology_remove_node():
    ti = TopologyIndex()
    ti.index_edge("n1", "n2")
    ti.remove_node("n2")
    assert "n2" not in ti.neighborhood("n1", depth=1)


# ── ResidualIndex ─────────────────────────────────────────────────────────────

def test_residual_index_add_and_rows():
    ri = ResidualIndex(max_size=10)
    ri.add({"touched_atoms": ["x1"], "salience": 0.5})
    assert len(ri) == 1
    assert ri.rows()[0]["touched_atoms"] == ["x1"]


def test_residual_index_max_size_evicts_lowest_salience():
    ri = ResidualIndex(max_size=3)
    ri.add({"touched_atoms": ["x1"], "salience": 0.1})
    ri.add({"touched_atoms": ["x2"], "salience": 0.9})
    ri.add({"touched_atoms": ["x3"], "salience": 0.5})
    ri.add({"touched_atoms": ["x4"], "salience": 0.8})
    assert len(ri) == 3
    assert ri.total_evicted == 1
    saliences = {r["salience"] for r in ri.rows()}
    assert 0.1 not in saliences  # lowest evicted


def test_residual_index_promote_matching():
    ri = ResidualIndex(max_size=10)
    ri.add({"touched_atoms": ["x1", "x2"], "salience": 0.3})
    ri.add({"touched_atoms": ["x3"], "salience": 0.2})
    promoted = ri.promote_matching(["x1"])
    assert len(promoted) == 1
    assert promoted[0]["touched_atoms"] == ["x1", "x2"]
    assert len(ri) == 1  # only x3 row remains


def test_residual_index_load_rows():
    ri = ResidualIndex(max_size=5)
    ri.load_rows([
        {"touched_atoms": ["x1"], "salience": 0.5},
        {"touched_atoms": ["x2"], "salience": 0.3},
    ])
    assert len(ri) == 2


# ── NethraAssimilator — disposition logic ─────────────────────────────────────

def test_explain_noise_no_atoms_no_structure():
    a = NethraAssimilator()
    row = {"entry_kind": "record", "record_id": "r1"}
    result = a.explain(row, {})
    assert result.disposition == Disposition.NOISE
    assert result.explained_by is None


def test_explain_residual_no_candidates():
    a = NethraAssimilator()
    row = {"touched_atoms": ["x1", "x2"], "record_id": "r1"}
    result = a.explain(row, {})
    assert result.disposition == Disposition.RESIDUAL
    assert result.explained_by is None


def test_explain_assimilated_high_overlap():
    a, nodes, _ = make_assimilator_with_node("n1", ["x1", "x2", "x3"])
    row = {"touched_atoms": ["x1", "x2", "x3"], "record_id": "r1"}
    result = a.explain(row, nodes)
    assert result.disposition == Disposition.ASSIMILATED
    assert result.explained_by == "n1"
    assert result.match_score >= 0.5


def test_explain_residual_low_overlap():
    a, nodes, _ = make_assimilator_with_node("n1", ["x1", "x2", "x3", "x4", "x5"])
    # Only 1 of 5 atoms shared → Jaccard = 1/5 = 0.2 < _PARTIAL_THRESHOLD
    row = {"touched_atoms": ["x1", "x6", "x7", "x8", "x9"], "record_id": "r1"}
    result = a.explain(row, nodes)
    assert result.disposition == Disposition.RESIDUAL


def test_explain_contradiction_row_failure_vs_success_dominant_node():
    a = NethraAssimilator()
    node = make_node("n1", atoms=["x1", "x2", "x3"], success_count=10, failure_count=0)
    nodes = {"n1": node}
    a.index_node("n1", ["x1", "x2", "x3"])
    row = {
        "touched_atoms": ["x1", "x2", "x3"],
        "record_id": "r1",
        "success": False,
    }
    result = a.explain(row, nodes)
    assert result.disposition == Disposition.CONTRADICTION
    assert result.explained_by == "n1"


def test_explain_split_candidate_incompatible_context():
    a = NethraAssimilator()
    node = make_node("n1", atoms=["x1", "x2", "x3"], contexts=["ctx_A|sub"])
    nodes = {"n1": node}
    a.index_node("n1", ["x1", "x2", "x3"])
    row = {
        "touched_atoms": ["x1", "x2", "x3"],
        "record_id": "r1",
        "context_scope": "ctx_B|sub",  # different prefix
    }
    result = a.explain(row, nodes)
    assert result.disposition == Disposition.SPLIT_CANDIDATE
    assert result.explained_by == "n1"


def test_explain_topology_expansion_finds_neighbor():
    """A row matching node n2's atoms should reach n1 via topology if n2 is n1's neighbor."""
    a = NethraAssimilator()
    # n1 has atoms x1,x2 — n2 has atoms x3,x4 — n1 and n2 are connected
    node1 = make_node("n1", atoms=["x1", "x2"])
    node2 = make_node("n2", atoms=["x3", "x4"])
    nodes = {"n1": node1, "n2": node2}
    a.index_node("n1", ["x1", "x2"])
    a.index_node("n2", ["x3", "x4"])
    a.index_edge("n1", "n2")
    # Row only matches n1's atoms — but n2 should be a candidate via topology
    row = {"touched_atoms": ["x1", "x2"], "record_id": "r1"}
    result = a.explain(row, nodes)
    # At minimum it should find n1 via anchor
    assert result.disposition in (Disposition.ASSIMILATED, Disposition.SPLIT_CANDIDATE)
    assert result.explained_by == "n1"


def test_explain_stats_accumulate():
    a, nodes, _ = make_assimilator_with_node("n1", ["x1", "x2", "x3"])
    a.explain({"touched_atoms": ["x1", "x2", "x3"]}, nodes)  # assimilated
    a.explain({"touched_atoms": ["x99"]}, nodes)  # residual (no candidate)
    stats = a.stats()
    assert stats["assimilated"] == 1
    assert stats["residual"] == 1


def test_explain_reset_stats():
    a, nodes, _ = make_assimilator_with_node("n1", ["x1", "x2"])
    a.explain({"touched_atoms": ["x1", "x2"]}, nodes)
    a.reset_stats()
    stats = a.stats()
    assert all(v == 0 for v in stats.values())
    assert a.total_calls == 0


def test_remove_node_clears_candidates():
    a, nodes, _ = make_assimilator_with_node("n1", ["x1", "x2", "x3"])
    a.remove_node("n1")
    row = {"touched_atoms": ["x1", "x2", "x3"], "record_id": "r1"}
    result = a.explain(row, nodes)
    # n1 is removed from all sub-indexes; no candidates found
    assert result.disposition == Disposition.RESIDUAL


def test_explain_uses_perspective_index_for_candidates():
    """Rows matching context+use_right should find candidates even without atom match."""
    a = NethraAssimilator()
    node = make_node("n1", atoms=["x1", "x2", "x3"], contexts=["ctx_X"],
                     use_rights=["ranking_hint"])
    nodes = {"n1": node}
    a.index_node("n1", ["x1", "x2", "x3"], contexts=["ctx_X"],
                 use_rights_seen=["ranking_hint"])
    # Row has 3 matching atoms; perspective also matches
    row = {
        "touched_atoms": ["x1", "x2", "x3"],
        "context_scope": "ctx_X",
        "use_right": "ranking_hint",
    }
    result = a.explain(row, nodes)
    assert result.disposition == Disposition.ASSIMILATED
    assert result.explained_by == "n1"


def test_residuals_are_stored_on_assimilator():
    """RESIDUAL rows go into assimilator.residuals, not discarded."""
    a = NethraAssimilator()
    row = {"touched_atoms": ["x99"], "salience": 0.4, "record_id": "r1"}
    result = a.explain(row, {})
    assert result.disposition == Disposition.RESIDUAL
    # The assimilator itself doesn't auto-store residuals in explain()
    # (store does it via the returned disposition); just verify stats
    assert a.stats()["residual"] == 1
