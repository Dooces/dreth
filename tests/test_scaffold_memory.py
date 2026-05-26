from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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
    common_parents: list[list[int]] | None = None,
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
        "common_parents": common_parents or [],
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


def _bg_record(
    nethra_id: str = "n1",
    kind: str = "trass_pattern",
    vars: list[int] | None = None,
    context_keys: list[str] | None = None,
    fit_signatures: list[str] | None = None,
    parent_sets: list[list[int]] | None = None,
    source_roles: list[str] | None = None,
    payload: dict | None = None,
) -> dict[str, Any]:
    return {
        "nethra_id": nethra_id,
        "kind": kind,
        "vars": vars or [0],
        "context_keys": context_keys or [],
        "fit_signatures": fit_signatures or [],
        "parent_sets": parent_sets or [],
        "source_roles": source_roles or [],
        "payload": payload or {},
    }


def _auth_record(
    var: int = 0,
    nethra_id: str = "var_fit:x0:HIGH()",
    authority_state: str = "contested_best_available",
    reason: str = "active_visible_conflict",
) -> dict[str, Any]:
    return {
        "var": var,
        "nethra_id": nethra_id,
        "authority_state": authority_state,
        "reason": reason,
        "context_key": f"authority_strength|x{var}",
    }


def _cr_record(
    nethra_id: str = "var_fit:x0:MAX(1,2)",
    kind: str = "var_fit",
    target_var: int = 0,
    learned_parents: list[int] | None = None,
) -> dict[str, Any]:
    parents = learned_parents or [1, 2]
    return {
        "nethra_id": nethra_id,
        "kind": kind,
        "target_var": target_var,
        "learned_parents": parents,
        "signature": f"x{target_var}:MAX({','.join(map(str, parents))})",
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
    hidden["truth_parents"] = {"0": [1, 2]}
    normal = _proposal_dict("p_normal", vars=[1])
    idx = _index_with(hidden, normal)
    # Row with hidden truth field must be skipped
    assert idx.loaded_proposals_count == 1


def test_authority_allowed_always_false_on_load():
    p = _proposal_dict("p1", authority_allowed=True)  # file says True
    idx = _index_with(p)
    assert idx.loaded_proposals_count == 1
    props = idx.query_by_var(0)
    assert len(props) == 1
    assert props[0].authority_allowed is False


# ── Tests: match_background_record ───────────────────────────────────────────

def test_match_by_nethra_id():
    p = _proposal_dict("p1", kind="trass_family", source_record_ids=["n_exact"], vars=[0])
    idx = _index_with(p)
    rec = _bg_record(nethra_id="n_exact", vars=[0])
    matches = idx.match_background_record(rec)
    assert len(matches) == 1
    assert matches[0].proposal_id == "p1"


def test_match_by_shared_var_and_context():
    p = _proposal_dict(
        "p1", kind="trass_family", vars=[5],
        contexts=["audit|x5|vis=100"],
    )
    idx = _index_with(p)
    rec = _bg_record(nethra_id="other_id", vars=[5], context_keys=["audit|x5|vis=100"])
    matches = idx.match_background_record(rec)
    assert len(matches) == 1


def test_match_by_shared_var_and_signature():
    p = _proposal_dict(
        "p1", kind="unresolved_family", vars=[3],
        common_signatures=["x3:MAX(1,2)"],
    )
    idx = _index_with(p)
    rec = _bg_record(nethra_id="other", vars=[3], fit_signatures=["x3:MAX(1,2)"])
    matches = idx.match_background_record(rec)
    assert len(matches) == 1


def test_match_by_shared_var_and_parent():
    p = _proposal_dict(
        "p1", kind="trass_family", vars=[2],
        common_parents=[[1, 4]],
    )
    idx = _index_with(p)
    rec = _bg_record(nethra_id="other", vars=[2], parent_sets=[[1, 4]])
    matches = idx.match_background_record(rec)
    assert len(matches) == 1


def test_no_match_var_only_no_anchor():
    # Record shares var but proposal has no contexts/sigs/parents → no match via Phase 2
    p = _proposal_dict("p1", kind="trass_family", vars=[0])
    idx = _index_with(p)
    rec = _bg_record(nethra_id="diff_id", vars=[0])  # no context, no sig, no parent
    matches = idx.match_background_record(rec)
    assert matches == []


def test_unrelated_records_do_not_match():
    p = _proposal_dict("p1", kind="trass_family", vars=[5], contexts=["ctx:x5"])
    idx = _index_with(p)
    # Different var, different context
    rec = _bg_record(nethra_id="unrelated", vars=[99], context_keys=["ctx:x99"])
    matches = idx.match_background_record(rec)
    assert matches == []


def test_no_duplicate_matches_from_multiple_indices():
    p = _proposal_dict(
        "p1", kind="trass_family", vars=[0],
        contexts=["ctx:x0"],
        common_signatures=["x0:MAX(1,2)"],
    )
    idx = _index_with(p)
    rec = _bg_record(
        nethra_id="n1",  # matches by nethra_id and by context+sig
        vars=[0],
        context_keys=["ctx:x0"],
        fit_signatures=["x0:MAX(1,2)"],
        source_roles=[],
    )
    # source_record_ids contains "n1" so phase 1 matches; phase 2 also would
    matches = idx.match_background_record(rec)
    ids = [m.proposal_id for m in matches]
    assert ids.count("p1") == 1


# ── Tests: match_context_role_record ─────────────────────────────────────────

def test_match_context_role_by_var_and_parents():
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[0],
        role_patterns=["var_fit"],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    rec = _cr_record(target_var=0, kind="var_fit", learned_parents=[1, 2])
    matches = idx.match_context_role_record(rec)
    assert len(matches) == 1
    assert matches[0].proposal_id == "p_cr"


def test_match_context_role_by_var_no_parent_constraint():
    # Proposal has no common_parents → accepts any parents for matching var
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[3],
        role_patterns=["var_fit"],
        common_parents=[],
    )
    idx = _index_with(p)
    rec = _cr_record(target_var=3, kind="var_fit", learned_parents=[5, 6])
    matches = idx.match_context_role_record(rec)
    assert len(matches) == 1


def test_no_match_context_role_different_var():
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[0],
        role_patterns=["var_fit"],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    rec = _cr_record(target_var=99, kind="var_fit", learned_parents=[1, 2])
    matches = idx.match_context_role_record(rec)
    assert matches == []


def test_no_match_context_role_different_parents():
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[0],
        role_patterns=["var_fit"],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    # Different parents → no overlap
    rec = _cr_record(target_var=0, kind="var_fit", learned_parents=[7, 8])
    matches = idx.match_context_role_record(rec)
    assert matches == []


def test_no_match_context_role_wrong_kind():
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[0],
        role_patterns=["var_fit"],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    rec = _cr_record(target_var=0, kind="route_fit", learned_parents=[1, 2])
    matches = idx.match_context_role_record(rec)
    assert matches == []


# ── Tests: match_uncertainty_record ──────────────────────────────────────────

def test_match_uncertainty_record_delegates_to_bg():
    p = _proposal_dict(
        "p_unc", kind="giant_cluster_subfamily", vars=[1],
        contexts=["unc|x1"],
    )
    idx = _index_with(p)
    rec = _bg_record(
        nethra_id="bg_uc:x1",
        kind="recurring_low_salience_pattern",
        vars=[1],
        context_keys=["unc|x1"],
        payload={"is_giant": True},
    )
    matches = idx.match_uncertainty_record(rec)
    assert len(matches) == 1


# ── Tests: match_authority_strength_record ────────────────────────────────────

def test_match_authority_strength_by_var_and_state():
    p = _proposal_dict(
        "p_auth", kind="authority_debt_family", vars=[0],
        role_patterns=["contested_best_available", "active_visible_conflict"],
    )
    idx = _index_with(p)
    rec = _auth_record(
        var=0,
        authority_state="contested_best_available",
        reason="active_visible_conflict",
    )
    matches = idx.match_authority_strength_record(rec)
    assert len(matches) == 1
    assert matches[0].proposal_id == "p_auth"


def test_no_match_authority_different_var():
    p = _proposal_dict(
        "p_auth", kind="authority_debt_family", vars=[5],
        role_patterns=["contested_best_available"],
    )
    idx = _index_with(p)
    rec = _auth_record(var=99, authority_state="contested_best_available")
    matches = idx.match_authority_strength_record(rec)
    assert matches == []


def test_no_match_authority_different_state():
    p = _proposal_dict(
        "p_auth", kind="authority_debt_family", vars=[0],
        role_patterns=["strong"],
    )
    idx = _index_with(p)
    rec = _auth_record(var=0, authority_state="weak_best_available")
    matches = idx.match_authority_strength_record(rec)
    assert matches == []


# ── Tests: query methods ──────────────────────────────────────────────────────

def test_query_by_var_returns_all_proposals_for_var():
    p1 = _proposal_dict("p1", vars=[0])
    p2 = _proposal_dict("p2", vars=[0, 1])
    p3 = _proposal_dict("p3", vars=[1])
    idx = _index_with(p1, p2, p3)
    result = idx.query_by_var(0)
    ids = {p.proposal_id for p in result}
    assert "p1" in ids
    assert "p2" in ids
    assert "p3" not in ids


def test_query_by_context_returns_correct_proposals():
    p1 = _proposal_dict("p1", contexts=["ctx:x0"])
    p2 = _proposal_dict("p2", contexts=["ctx:x1"])
    idx = _index_with(p1, p2)
    result = idx.query_by_context("ctx:x0")
    assert len(result) == 1
    assert result[0].proposal_id == "p1"


def test_query_by_var_empty():
    idx = _index_with(_proposal_dict("p1", vars=[0]))
    assert idx.query_by_var(999) == []


# ── Tests: broad_generic_debt ─────────────────────────────────────────────────

def test_broad_generic_debt_marked_for_no_anchor_authority_debt():
    p = _proposal_dict(
        "p_broad", kind="authority_debt_family", vars=list(range(10)),
        contexts=[],
        common_signatures=[],
        common_parents=[],
        role_patterns=["contested_best_available", "active_visible_conflict"],
    )
    idx = _index_with(p)
    props = idx.query_by_var(0)
    assert len(props) == 1
    assert props[0].broad_generic_debt is True


def test_non_authority_debt_not_broad_generic():
    p = _proposal_dict("p_trass", kind="trass_family", vars=[0])
    idx = _index_with(p)
    props = idx.query_by_var(0)
    assert props[0].broad_generic_debt is False


def test_authority_debt_with_context_not_broad():
    p = _proposal_dict(
        "p_auth_local", kind="authority_debt_family", vars=[0],
        contexts=["auth|x0|vis=100"],
    )
    idx = _index_with(p)
    props = idx.query_by_var(0)
    assert props[0].broad_generic_debt is False


def test_authority_debt_with_parents_not_broad():
    p = _proposal_dict(
        "p_auth_local", kind="authority_debt_family", vars=[0],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    props = idx.query_by_var(0)
    assert props[0].broad_generic_debt is False


# ── Tests: compute_run_scaffold_metrics ───────────────────────────────────────

def test_compute_run_metrics_matches_bg_records():
    p = _proposal_dict(
        "p1", kind="trass_family", vars=[0],
        source_record_ids=["n_match"],
    )
    idx = _index_with(p)
    bg_export = {
        "records": [_bg_record(nethra_id="n_match", vars=[0])],
        "edges": [],
    }
    m = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    assert m["scaffold_memory_loaded_proposals"] == 1
    assert m["scaffold_memory_match_attempts"] == 1
    assert m["scaffold_memory_matches"] == 1
    assert m["scaffold_memory_unmatched_records"] == 0
    assert m["scaffold_memory_useful_matches"] == 1
    assert m["scaffold_memory_broad_generic_debt_matches"] == 0


def test_compute_run_metrics_unmatched_counted():
    p = _proposal_dict("p1", kind="trass_family", vars=[99], contexts=["ctx:x99"])
    idx = _index_with(p)
    bg_export = {
        "records": [_bg_record(nethra_id="other", vars=[0])],
    }
    m = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    assert m["scaffold_memory_matches"] == 0
    assert m["scaffold_memory_unmatched_records"] == 1


def test_compute_run_metrics_broad_generic_debt_not_useful():
    p = _proposal_dict(
        "p_broad", kind="authority_debt_family", vars=[0],
        role_patterns=["contested_best_available"],
    )
    idx = _index_with(p)
    auth_export = {
        "records": [_auth_record(var=0, authority_state="contested_best_available")],
    }
    m = compute_run_scaffold_metrics(idx, {}, {}, auth_export)
    assert m["scaffold_memory_matches"] == 1
    assert m["scaffold_memory_broad_generic_debt_matches"] == 1
    assert m["scaffold_memory_useful_matches"] == 0


def test_compute_run_metrics_match_rate():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    bg_export = {
        "records": [
            _bg_record(nethra_id="n1", vars=[0]),   # matches
            _bg_record(nethra_id="n_other", vars=[99]),  # no match
        ]
    }
    m = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    assert m["scaffold_memory_match_attempts"] == 2
    assert m["scaffold_memory_matches"] == 1
    assert abs(m["scaffold_memory_match_rate"] - 0.5) < 1e-6


def test_compute_run_metrics_authority_allowed_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(
        idx,
        {"records": [_bg_record(nethra_id="n1")]},
        {},
        {},
    )
    assert m["scaffold_memory_authority_allowed_count"] == 0


def test_compute_run_metrics_behavior_effects_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(
        idx,
        {"records": [_bg_record(nethra_id="n1")]},
        {},
        {},
    )
    assert m["scaffold_memory_behavior_effects"] == 0


def test_compute_run_metrics_match_examples_populated():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    bg_export = {"records": [_bg_record(nethra_id="n1", vars=[0])]}
    m = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    assert len(m["scaffold_memory_match_examples"]) >= 1
    ex = m["scaffold_memory_match_examples"][0]
    assert "matched_proposal_id" in ex
    assert "broad_generic_debt" in ex


def test_compute_run_metrics_cr_records():
    p = _proposal_dict(
        "p_cr", kind="context_role_recurrence", vars=[0],
        role_patterns=["var_fit"],
        common_parents=[[1, 2]],
    )
    idx = _index_with(p)
    cr_export = {
        "records": [_cr_record(target_var=0, kind="var_fit", learned_parents=[1, 2])],
    }
    m = compute_run_scaffold_metrics(idx, {}, cr_export, {})
    assert m["scaffold_memory_matches"] >= 1
    assert m["scaffold_memory_useful_matches"] >= 1


def test_compute_run_metrics_empty_exports_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(idx, {}, {}, {})
    assert m["scaffold_memory_match_attempts"] == 0
    assert m["scaffold_memory_matches"] == 0
    assert m["scaffold_memory_behavior_effects"] == 0


# ── Tests: record mode behavior equals off ────────────────────────────────────

def test_record_mode_does_not_modify_proposals():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    bg_export = {"records": [_bg_record(nethra_id="n1", vars=[0])]}
    # Running match multiple times must not change proposal state
    m1 = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    m2 = compute_run_scaffold_metrics(idx, bg_export, {}, {})
    assert m1["scaffold_memory_matches"] == m2["scaffold_memory_matches"]
    props = idx.query_by_var(0)
    assert props[0].authority_allowed is False  # unchanged
    assert props[0].broad_generic_debt is False


def test_empty_scaffold_metrics_off_mode():
    m = empty_scaffold_metrics()
    assert m["scaffold_memory_mode"] == "off"
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
        contexts=[], common_signatures=[], common_parents=[],
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
    m = compute_run_scaffold_metrics(
        idx,
        {"records": [_bg_record(nethra_id="n1", vars=[0])]},
        {},
        {},
    )
    assert m["scaffold_memory_authority_allowed_count"] == 0
    # Verify the loaded proposal has authority_allowed=False
    assert idx.query_by_var(0)[0].authority_allowed is False


def test_behavior_effects_always_zero():
    p = _proposal_dict("p1", kind="trass_family", vars=[0], source_record_ids=["n1"])
    idx = _index_with(p)
    m = compute_run_scaffold_metrics(
        idx,
        {"records": [_bg_record(nethra_id="n1")]},
        {},
        {},
    )
    assert m["scaffold_memory_behavior_effects"] == 0


def test_no_hidden_truth_manifest_read():
    """Proposals rows containing hidden truth fields must be skipped entirely."""
    proposals_with_truth = [
        _proposal_dict("p_hidden", kind="trass_family", vars=[0]),
        _proposal_dict("p_normal", kind="trass_family", vars=[1]),
    ]
    proposals_with_truth[0]["debug_blind_challenge_manifest"] = {"secret": True}
    proposals_with_truth[0]["truth_parents"] = {"0": [1, 2]}

    path = _make_jsonl(proposals_with_truth)
    idx = ScaffoldMemoryIndex()
    idx.load_proposals(path)
    # Hidden truth row must be skipped
    assert idx.loaded_proposals_count == 1
    assert idx.query_by_var(0) == []   # p_hidden skipped
    assert idx.query_by_var(1) != []   # p_normal loaded


def test_no_imports_from_agent():
    """scaffold_memory module must not import from agent.py."""
    import dreth.scaffold_memory as sm_mod
    import inspect
    source = inspect.getsource(sm_mod)
    assert "from dreth.agent" not in source
    assert "from .agent" not in source
    assert "import agent" not in source
    assert "ChainedAgent" not in source


# ── Tests: assist_feature ranking ────────────────────────────────────────────

def test_useful_local_scaffold_can_reorder_candidates():
    idx = _index_with(_proposal_dict(
        "p_rank", kind="unresolved_family", vars=[0],
        contexts=["parent_candidates|x0|vis=5"],
        common_parents=[[3]],
        suggested_runtime_use="ranking_hint",
    ))
    ranked = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    assert ranked == (3, 1, 2)
    metrics = idx.runtime_metrics()
    assert metrics["scaffold_memory_ranking_applications"] == 1
    assert metrics["scaffold_memory_candidates_reordered"] == 1


def test_unrelated_scaffold_does_not_reorder_candidates():
    idx = _index_with(_proposal_dict(
        "p_other", kind="unresolved_family", vars=[9],
        contexts=["parent_candidates|x9|vis=5"],
        common_parents=[[3]],
        suggested_runtime_use="ranking_hint",
    ))
    ranked = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    assert ranked == (1, 3, 2)
    assert idx.runtime_metrics()["scaffold_memory_ranking_applications"] == 0


def test_low_specificity_or_no_runtime_use_proposal_does_not_reorder():
    idx = _index_with(
        _proposal_dict(
            "p_low", kind="unresolved_family", vars=[0],
            contexts=["parent_candidates|x0|vis=5"],
            common_parents=[[3]],
            suggested_runtime_use="ranking_hint",
            warnings=["low_specificity"],
        ),
        _proposal_dict(
            "p_no_use", kind="unresolved_family", vars=[0],
            contexts=["parent_candidates|x0|vis=5"],
            common_parents=[[2]],
            suggested_runtime_use="no_runtime_use",
        ),
    )
    ranked = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    assert ranked == (1, 3, 2)
    assert idx.runtime_metrics()["scaffold_memory_ranking_applications"] == 0


def test_broad_generic_debt_does_not_affect_ordering():
    idx = _index_with(_proposal_dict(
        "p_broad", kind="authority_debt_family", vars=[0],
        contexts=[], common_signatures=[], common_parents=[],
        role_patterns=["active_visible_conflict"],
        suggested_runtime_use="ranking_hint",
    ))
    ranked = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    assert ranked == (1, 3, 2)
    metrics = idx.runtime_metrics()
    assert metrics["scaffold_memory_ranking_applications"] == 0
    assert metrics["scaffold_memory_broad_generic_noops"] > 0


def test_scaffold_ranking_is_deterministic():
    idx = _index_with(_proposal_dict(
        "p_rank", kind="unresolved_family", vars=[0],
        contexts=["parent_candidates|x0|vis=5"],
        common_parents=[[3]],
        suggested_runtime_use="ranking_hint",
    ))
    first = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    second = idx.rank_candidate_keys(0, "parent_candidates|x0|vis=5", (1, 3, 2))
    assert first == second == (3, 1, 2)


# ── Tests: from_dict ─────────────────────────────────────────────────────────

def test_from_dict_loads_all_fields():
    d = _proposal_dict(
        "prop_test", kind="unresolved_family",
        vars=[3, 7], contexts=["ctx:x3"],
        common_signatures=["x3:MAX(1,2)"],
        common_parents=[[1, 2]],
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
    assert p.common_parents == [[1, 2]]
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
        contexts=[], common_signatures=[], common_parents=[],
    )
    p = ScaffoldMemoryProposal.from_dict(d)
    assert p.broad_generic_debt is True


def test_hidden_truth_fields_constant():
    assert "truth_parents" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "truth_func" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "debug_blind_challenge_manifest" in HIDDEN_TRUTH_LIKE_FIELDS
