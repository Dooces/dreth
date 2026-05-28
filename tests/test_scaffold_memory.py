from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().source_edges[1]
sys.path.insert(0, str(ROOT))

from dreth.scaffold_memory import (
    HIDDEN_TRUTH_LIKE_FIELDS,
    ScaffoldMemoryIndex,
    ScaffoldMemoryProposal,
    compute_run_scaffold_metrics,
    empty_scaffold_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _proposal_dict(
    proposal_id: str = "prop_000001_trass_family",
    kind: str = "trass_family",
    source_record_ids: list[str] | None = None,
    source_kinds: list[str] | None = None,
    vars: list[int] | None = None,
    contexts: list[str] | None = None,
    common_signatures: list[str] | None = None,
    common_source_edges: list[list[int]] | None = None,
    role_patterns: list[str] | None = None,
    recurring_signals: list[str] | None = None,
    confidence_as_familiarity: float = 0.3,
    authority_allowed: bool = False,
    suggested_runtime_use: str = "feature_only",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "kind": kind,
        "source_record_ids": source_record_ids or ["n1"],
        "source_kinds": source_kinds or ["trass_pattern"],
        "vars": vars or [0],
        "contexts": contexts or [],
        "common_signatures": common_signatures or [],
        "common_source_edges": common_source_edges or [],
        "role_patterns": role_patterns or [],
        "recurring_signals": recurring_signals or [],
        "confidence_as_familiarity": confidence_as_familiarity,
        "action_relevance_score": 0.06,
        "authority_allowed": authority_allowed,
        "suggested_runtime_use": suggested_runtime_use,
        "evidence_summary": "test",
        "warnings": warnings or [],
        "recurrence_count": 2,
        "runs_seen": 2,
        "seeds_seen": 2,
        "first_seen_cycle": 0,
        "last_seen_cycle": 100,
    }


def _make_jsonl(proposals: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as fh:
        for p in proposals:
            fh.write(json.dumps(p) + "\n")
        return fh.name


def _index_with(*proposals: dict) -> ScaffoldMemoryIndex:
    path = _make_jsonl(list(proposals))
    idx = ScaffoldMemoryIndex()
    idx.load_proposals(path)
    return idx


# ── Tests: load_proposals ─────────────────────────────────────────────────────

def test_load_proposals_from_sleep_output():
    p1 = _proposal_dict("prop_1", kind="trass_family", vars=[0])
    p2 = _proposal_dict("prop_2", kind="unresolved_family", vars=[1])
    idx = _index_with(p1, p2)
    assert idx.loaded_proposals_count == 2


def test_load_proposals_skips_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(_proposal_dict("p1")) + "\n")
        fh.write("not valid json{\n")
        fh.write(json.dumps(_proposal_dict("p2", vars=[1])) + "\n")
        path = fh.name
    idx = ScaffoldMemoryIndex()
    idx.load_proposals(path)
    assert idx.loaded_proposals_count == 2


def test_load_proposals_skips_hidden_truth_rows():
    hidden = _proposal_dict("p_hidden")
    hidden["truth_source_edges"] = {"0": [1, 2]}
    normal = _proposal_dict("p_normal", vars=[1])
    idx = _index_with(hidden, normal)
    assert idx.loaded_proposals_count == 1


def test_authority_allowed_always_false_on_load():
    p = _proposal_dict("p1", authority_allowed=True)  # file says True
    idx = _index_with(p)
    assert idx.loaded_proposals_count == 1
    assert idx._proposals[0].authority_allowed is False


# ── Tests: broad_generic_debt ─────────────────────────────────────────────────

def test_broad_generic_debt_marked_for_no_anchor_authority_debt():
    p = _proposal_dict(
        "p_broad", kind="authority_debt_family", vars=list(range(10)),
        contexts=[],
        common_signatures=[],
        common_source_edges=[],
        role_patterns=["contested_best_available", "active_visible_conflict"],
    )
    idx = _index_with(p)
    assert idx._proposals[0].broad_generic_debt is True


def test_non_authority_debt_not_broad_generic():
    p = _proposal_dict("p_trass", kind="trass_family", vars=[0])
    idx = _index_with(p)
    assert idx._proposals[0].broad_generic_debt is False


def test_authority_debt_with_context_not_broad():
    p = _proposal_dict(
        "p_auth_local", kind="authority_debt_family", vars=[0],
        contexts=["auth|x0|vis=100"],
    )
    idx = _index_with(p)
    assert idx._proposals[0].broad_generic_debt is False


def test_authority_debt_with_source_edges_not_broad():
    p = _proposal_dict(
        "p_auth_local", kind="authority_debt_family", vars=[0],
        common_source_edges=[[1, 2]],
    )
    idx = _index_with(p)
    assert idx._proposals[0].broad_generic_debt is False


# ── Tests: compute_run_scaffold_metrics ───────────────────────────────────────

def test_compute_run_metrics_authority_allowed_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {"records": []}, {}, {})
    assert m["scaffold_memory_authority_allowed_count"] == 0


def test_compute_run_metrics_behavior_effects_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {"records": []}, {}, {})
    assert m["scaffold_memory_behavior_effects"] == 0


def test_compute_run_metrics_loaded_proposals_count():
    p = _proposal_dict("p1", kind="trass_family", vars=[0])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {}, {}, {})
    assert m["scaffold_memory_loaded_proposals"] == 1


def test_compute_run_metrics_empty_exports():
    p = _proposal_dict("p1", kind="trass_family", vars=[0])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {}, {}, {})
    assert m["scaffold_memory_match_attempts"] == 0
    assert m["scaffold_memory_matched_records"] == 0
    assert m["scaffold_memory_behavior_effects"] == 0


# ── Tests: empty_scaffold_metrics ────────────────────────────────────────────

def test_empty_scaffold_metrics_returns_expected_keys():
    m = empty_scaffold_metrics()
    assert m["scaffold_memory_loaded_proposals"] == 0
    assert m["scaffold_memory_authority_allowed_count"] == 0
    assert m["scaffold_memory_behavior_effects"] == 0


# ── Tests: summarize_matches ──────────────────────────────────────────────────

def test_summarize_matches_loaded_proposals_count():
    p1 = _proposal_dict("p1", kind="trass_family", vars=[0])
    p2 = _proposal_dict("p2", kind="unresolved_family", vars=[1])
    idx = _index_with(p1, p2)
    s = idx.summarize_matches()
    assert s["scaffold_memory_loaded_proposals"] == 2
    assert s["scaffold_memory_authority_allowed_count"] == 0
    assert s["scaffold_memory_behavior_effects"] == 0


def test_summarize_matches_by_kind():
    p1 = _proposal_dict("p1", kind="trass_family", vars=[0])
    p2 = _proposal_dict("p2", kind="trass_family", vars=[1])
    p3 = _proposal_dict("p3", kind="unresolved_family", vars=[2])
    idx = _index_with(p1, p2, p3)
    s = idx.summarize_matches()
    assert s["scaffold_memory_proposals_by_kind"].get("trass_family") == 2
    assert s["scaffold_memory_proposals_by_kind"].get("unresolved_family") == 1


def test_summarize_matches_broad_generic_debt_count():
    broad = _proposal_dict(
        "p_broad", kind="authority_debt_family", vars=list(range(8)),
        contexts=[], common_signatures=[], common_source_edges=[],
    )
    non_broad = _proposal_dict("p_ok", kind="trass_family", vars=[0])
    idx = _index_with(broad, non_broad)
    s = idx.summarize_matches()
    assert s["scaffold_memory_broad_generic_debt_proposals"] == 1


# ── Tests: invariants ─────────────────────────────────────────────────────────

def test_authority_allowed_count_always_zero_in_compute_metrics():
    """All proposals must have authority_allowed=False; metric must be 0."""
    p = _proposal_dict("p1", kind="trass_family", vars=[0],
                       source_record_ids=["n1"], authority_allowed=True)
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {"records": []}, {}, {})
    assert m["scaffold_memory_authority_allowed_count"] == 0
    assert idx._proposals[0].authority_allowed is False


def test_behavior_effects_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {"records": []}, {}, {})
    assert m["scaffold_memory_behavior_effects"] == 0


def test_no_hidden_truth_manifest_read():
    """Proposals rows containing hidden truth fields must be skipped entirely."""
    proposals_with_truth = [
        _proposal_dict("p_hidden", kind="trass_family", vars=[0]),
        _proposal_dict("p_normal", kind="trass_family", vars=[1]),
    ]
    proposals_with_truth[0]["debug_blind_challenge_manifest"] = {"secret": True}
    proposals_with_truth[0]["truth_source_edges"] = {"0": [1, 2]}

    path = _make_jsonl(proposals_with_truth)
    idx = ScaffoldMemoryIndex()
    idx.load_proposals(path)
    assert idx.loaded_proposals_count == 1
    # p_hidden skipped → only p_normal (vars=[1]) loaded
    assert idx._proposals[0].vars == [1]


def test_no_imports_from_agent():
    """scaffold_memory module must not import from agent.py."""
    import dreth.scaffold_memory as sm_mod
    import inspect
    source = inspect.getsource(sm_mod)
    assert "from dreth.agent" not in source
    assert "from .agent" not in source
    assert "import agent" not in source
    assert "ChainedAgent" not in source


# ── Tests: from_dict ─────────────────────────────────────────────────────────

def test_from_dict_loads_all_fields():
    d = _proposal_dict(
        "prop_test", kind="unresolved_family",
        vars=[3, 7], contexts=["ctx:x3"],
        common_signatures=["x3:MAX(1,2)"],
        common_source_edges=[[1, 2]],
        role_patterns=["unresolved"],
        recurring_signals=["drift"],
        confidence_as_familiarity=0.45,
        suggested_runtime_use="clustering_prior",
        warnings=["low_specificity"],
    )
    p = ScaffoldMemoryProposal.from_dict(d)
    assert p.proposal_id == "prop_test"
    assert p.kind == "unresolved_family"
    assert p.vars == [3, 7]
    assert "ctx:x3" in p.contexts
    assert p.common_signatures == ["x3:MAX(1,2)"]
    assert p.common_source_edges == [[1, 2]]
    assert p.role_patterns == ["unresolved"]
    assert p.confidence_as_familiarity == 0.45
    assert p.authority_allowed is False
    assert p.broad_generic_debt is False


def test_from_dict_authority_allowed_forced_false():
    d = _proposal_dict("p1", authority_allowed=True)
    p = ScaffoldMemoryProposal.from_dict(d)
    assert p.authority_allowed is False


def test_from_dict_broad_generic_debt_detected():
    d = _proposal_dict(
        "p_broad", kind="authority_debt_family",
        vars=[0, 1, 2, 3, 4, 5, 6],
        contexts=[], common_signatures=[], common_source_edges=[],
    )
    p = ScaffoldMemoryProposal.from_dict(d)
    assert p.broad_generic_debt is True


def test_hidden_truth_fields_constant():
    assert "truth_source_edges" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "truth_func" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "debug_blind_challenge_manifest" in HIDDEN_TRUTH_LIKE_FIELDS
