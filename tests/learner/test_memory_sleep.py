from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().source_edges[2]
sys.path.insert(0, str(ROOT))

from dreth.learner.memory_sleep import (
    HIDDEN_TRUTH_LIKE_FIELDS,
    MemorySleepConsolidator,
    MemorySleepSummary,
    ScaffoldProposal,
    _BgObs,
    _AuthObs,
    _CRObs,
    _TempObs,
    _bg_anchor_key,
    _parse_sig_source_edges,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _bg_rec(
    nethra_id: str = "n1",
    kind: str = "unresolved_pattern",
    vars: list[int] | None = None,
    context_keys: list[str] | None = None,
    source_roles: list[str] | None = None,
    fit_signatures: list[str] | None = None,
    source_edge_sets: list[list[int]] | None = None,
    recurring_signals: list[str] | None = None,
    cheap_recognition_score: float = 0.1,
    first_seen_cycle: int = 0,
    last_seen_cycle: int = 0,
    payload: dict | None = None,
) -> dict[str, Any]:
    return {
        "nethra_id": nethra_id,
        "kind": kind,
        "vars": vars or [0],
        "context_keys": context_keys or [],
        "source_roles": source_roles or ["unresolved"],
        "fit_signatures": fit_signatures or [],
        "source_edge_sets": source_edge_sets or [],
        "operation_roles": [],
        "recurring_signals": recurring_signals or [],
        "first_seen_cycle": first_seen_cycle,
        "last_seen_cycle": last_seen_cycle,
        "seen_count": 1,
        "stability_count": 0,
        "contexts_seen_count": 0,
        "cheap_recognition_score": cheap_recognition_score,
        "salience_score": 0.0,
        "action_relevance_score": 0.0,
        "background_confidence": 0.1,
        "relation_edges": [],
        "payload": payload or {},
    }


def _auth_rec(
    var: int = 0,
    nethra_id: str = "var_fit:x0:HIGH()",
    authority_state: str = "contested_best_available",
    reason: str = "active_visible_conflict",
    cycle: int = 100,
    uncertainty_signals: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "var": var,
        "nethra_id": nethra_id,
        "context_key": f"authority_strength|x{var}|vis=100",
        "cycle": cycle,
        "strength": "contested",
        "reason": reason,
        "authority_state": authority_state,
        "active_evidence": [],
        "contradictory_evidence": [],
        "required_future_evidence": [],
        "uncertainty_signals": uncertainty_signals or [],
        "prior_role": "tareth",
        "best_available": True,
        "evidence_epoch": 1,
    }


def _cr_rec(
    nethra_id: str = "var_fit:x0:MAX(1,2)",
    kind: str = "var_fit",
    target_var: int = 0,
    learned_source_edges: list[int] | None = None,
    signature: str = "x0:MAX(1,2)",
) -> dict[str, Any]:
    source_edges = learned_source_edges or [1, 2]
    return {
        "nethra_id": nethra_id,
        "kind": kind,
        "target_var": target_var,
        "components": [target_var] + source_edges,
        "learned_source_edges": source_edges,
        "learned_func": "MAX",
        "signature": signature,
        "first_seen_cycle": 0,
        "last_seen_cycle": 0,
        "observations": 1,
        "passive_evidence_count": 0,
        "active_probe_count": 0,
        "composition_links": [],
        "source": "operation_role",
    }


def _row(
    seed: int = 1,
    bg_records: list[dict] | None = None,
    auth_records: list[dict] | None = None,
    cr_records: list[dict] | None = None,
    bg_edges: list[dict] | None = None,
    extra_fields: dict | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "background_nethra_export": {
            "records": bg_records or [],
            "edges": bg_edges or [],
            "role_shift_examples": [],
        },
        "authority_strength": {
            "records": auth_records or [],
            "summary": {},
            "controller": {},
        },
        "context_role_index": {
            "nodes": {},
            "edges": [],
            "roles": {},
            "match_attribution": {},
            "records": cr_records or [],
        },
        "background_nethra_edges": 0,
        "background_nethra_records": len(bg_records or []),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _consolidator() -> MemorySleepConsolidator:
    return MemorySleepConsolidator()


# ── Tests: load_jsonl_rows ────────────────────────────────────────────────────

def test_load_jsonl_rows_reads_valid_lines():
    c = _consolidator()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as fh:
        fh.write(json.dumps({"seed": 1, "background_nethra_export": {"records": []}}) + "\n")
        fh.write(json.dumps({"seed": 2, "background_nethra_export": {"records": []}}) + "\n")
        path = fh.name
    rows = c.load_jsonl_rows(path)
    assert len(rows) == 2
    assert rows[0]["seed"] == 1


def test_load_jsonl_rows_skips_invalid_lines():
    c = _consolidator()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as fh:
        fh.write(json.dumps({"seed": 1}) + "\n")
        fh.write("not valid json{\n")
        fh.write(json.dumps({"seed": 3}) + "\n")
        path = fh.name
    rows = c.load_jsonl_rows(path)
    assert len(rows) == 2


# ── Tests: extract methods ────────────────────────────────────────────────────

def test_extract_background_records_returns_bg_obs():
    c = _consolidator()
    rec = _bg_rec("n1")
    rows = [_row(seed=7, bg_records=[rec])]
    result = c.extract_background_records(rows)
    assert len(result) == 1
    assert result[0].seed == 7
    assert result[0].run_idx == 0
    assert result[0].rec["nethra_id"] == "n1"


def test_extract_background_records_multiple_rows():
    c = _consolidator()
    rows = [
        _row(seed=1, bg_records=[_bg_rec("n1"), _bg_rec("n2")]),
        _row(seed=2, bg_records=[_bg_rec("n3")]),
    ]
    result = c.extract_background_records(rows)
    assert len(result) == 3
    seeds = {o.seed for o in result}
    assert seeds == {1, 2}


def test_extract_context_role_records():
    c = _consolidator()
    rec = _cr_rec("var_fit:x0:MAX(1,2)", target_var=0)
    rows = [_row(seed=7, cr_records=[rec])]
    result = c.extract_context_role_records(rows)
    assert len(result) == 1
    assert result[0].rec["target_var"] == 0


def test_extract_uncertainty_records_giant():
    c = _consolidator()
    giant = _bg_rec(
        "bg_uc:giant:0,1,2",
        kind="recurring_low_salience_pattern",
        vars=[0, 1, 2],
        payload={"is_giant": True},
    )
    normal = _bg_rec("bg_normal:x0", kind="unresolved_pattern", vars=[0])
    rows = [_row(seed=1, bg_records=[giant, normal])]
    result = c.extract_uncertainty_records(rows)
    ids = [o.rec["nethra_id"] for o in result]
    assert "bg_uc:giant:0,1,2" in ids


def test_extract_uncertainty_records_by_source_role():
    c = _consolidator()
    unc_rec = _bg_rec("bg_uc:x0", kind="unresolved_pattern", source_roles=["uncertainty_cluster"])
    rows = [_row(seed=1, bg_records=[unc_rec])]
    result = c.extract_uncertainty_records(rows)
    assert len(result) == 1


def test_extract_authority_records():
    c = _consolidator()
    rec = _auth_rec(var=0)
    rows = [_row(seed=1, auth_records=[rec])]
    result = c.extract_authority_records(rows)
    assert len(result) == 1
    assert result[0].rec["var"] == 0


def test_extract_temporal_records_if_available_empty():
    c = _consolidator()
    rows = [_row(seed=1, bg_records=[_bg_rec("n1", kind="unresolved_pattern")])]
    result = c.extract_temporal_records_if_available(rows)
    assert result == []


def test_extract_temporal_records_if_available_present():
    c = _consolidator()
    temp = _bg_rec("te_1", kind="temporal_cohort_pattern", payload={"temporal_event": "cohort_5"})
    rows = [_row(seed=1, bg_records=[temp])]
    result = c.extract_temporal_records_if_available(rows)
    assert len(result) == 1
    assert result[0].rec["kind"] == "temporal_cohort_pattern"


def test_sleep_consumes_experience_events_and_emits_proposal_only_product():
    c = _consolidator()
    rows = [
        {
            "entry_kind": "record",
            "record_type": "nethra_handle",
            "record_id": "h1",
            "nethra_id": "h1",
            "seed": 1,
            "touched_atoms": ["x1"],
            "touched_structure_refs": ["x0:MAX(1)", "source_edges:1"],
            "member_nethras": ["h1"],
            "context_scope": "source_edge_candidates|x0|vis=2",
            "use_right": "ranking_hint",
            "authority_allowed": False,
        },
        {
            "entry_kind": "experience_event",
            "run_id": "run-b",
            "seed": 1,
            "cycle": 5,
            "context_key": "source_edge_candidates|x0|vis=2",
            "active_atoms": ["x0", "x1"],
            "active_nethras": ["h1"],
            "hook": "source_edge_candidates",
            "use_right": "ranking_hint",
            "candidates_before": [0, 1],
            "candidates_after": [1, 0],
            "behavior_effect": 1,
            "authority_effect": 0,
            "success": True,
            "hidden_truth_used": False,
        },
    ]
    mem = c.extract_nethra_memory_records(rows)
    exp = c.extract_experience_events(rows)
    products = c.build_sleep_products(mem, exp)
    assert len(products) >= 1
    product = next(p.to_dict() for p in products if "source_edges:1" in p.to_dict()["touched_structure_refs"])
    assert product["entry_kind"] == "sleep_product"
    assert product["authority_allowed"] is False
    assert product["proposed_use_right"] != "hard_filter"
    assert "h1" in product["member_nethras"]
    assert "x1" in product["touched_atoms"]
    assert "source_edges:1" in product["touched_structure_refs"]


def test_sleep_negative_gate_requires_visible_failure_association():
    c = _consolidator()
    rows = [{
        "entry_kind": "experience_event",
        "run_id": "run-b",
        "seed": 1,
        "cycle": 5,
        "context_key": "source_edge_candidates|x0|vis=2",
        "active_atoms": ["x0", "x1"],
        "active_nethras": ["h1"],
        "hook": "source_edge_candidates",
        "use_right": "ranking_hint",
        "failure_reason": "quality_regression",
        "hidden_truth_used": False,
    }]
    products = c.build_sleep_products(
        c.extract_nethra_memory_records(rows),
        c.extract_experience_events(rows),
    )
    assert products
    product = products[0].to_dict()
    assert product["proposed_use_right"] == "feature_only"
    assert "visible_failure_association" in product["invalidators"]


# ── Tests: build_proposals ────────────────────────────────────────────────────

def test_visible_background_records_produce_proposals():
    """Background records from multiple runs produce scaffold proposals."""
    c = _consolidator()
    rec = _bg_rec("frontier:x0:DIFF(5,8)", kind="unresolved_pattern", vars=[0],
                  source_edge_sets=[[1, 8]], fit_signatures=["x0:MAX(1,8)"],
                  context_keys=["tied_frontier|x0|vis=100|source_edges=1,8"])
    rows = [
        _row(seed=1, bg_records=[rec]),
        _row(seed=2, bg_records=[rec]),
    ]
    bg = c.extract_background_records(rows)
    cr = c.extract_context_role_records(rows)
    unc = c.extract_uncertainty_records(rows)
    auth = c.extract_authority_records(rows)
    temp = c.extract_temporal_records_if_available(rows)
    proposals = c.build_proposals(bg, cr, unc, auth, temp, min_sources=2)
    assert len(proposals) > 0


def test_repeated_trass_records_group_together():
    """Same nethra_id with kind=trass_pattern seen in multiple runs → trass_family proposal."""
    c = _consolidator()
    rec = _bg_rec("n_trass", kind="trass_pattern", vars=[3],
                  fit_signatures=["x3:LOW(1,2)"], source_edge_sets=[[1, 2]])
    rows = [
        _row(seed=10, bg_records=[rec]),
        _row(seed=20, bg_records=[rec]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    trass = [p for p in proposals if p.kind == "trass_family"]
    assert len(trass) >= 1
    assert any("n_trass" in p.source_record_ids for p in trass)


def test_unresolved_groups_by_shared_var_not_kind_alone():
    """Records sharing var+source_edge group together; records sharing only kind do not."""
    c = _consolidator()
    # Two records for var=0, source_edge=(1,8) — should group
    rec_a = _bg_rec("frontier:x0:DIFF(5,8)", kind="unresolved_pattern", vars=[0],
                    source_edge_sets=[[1, 8]])
    rec_b = _bg_rec("frontier:x0:DIFF(3,5)", kind="unresolved_pattern", vars=[0],
                    source_edge_sets=[[1, 8]])
    # One record for var=99, source_edge=(1,2) — different var, should be separate
    rec_c = _bg_rec("frontier:x99:DIFF(1,2)", kind="unresolved_pattern", vars=[99],
                    source_edge_sets=[[1, 2]])
    rows = [_row(seed=1, bg_records=[rec_a, rec_b, rec_c])]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # The rec_a/rec_b group should produce a proposal for var=0, source_edges=(1,8)
    var0_proposals = [p for p in proposals if 0 in p.vars and 99 not in p.vars]
    assert len(var0_proposals) >= 1
    # rec_c is alone with min_sources=2, so no proposal for var=99
    var99_proposals = [p for p in proposals if 99 in p.vars and 0 not in p.vars]
    assert len(var99_proposals) == 0


def test_unresolved_alone_not_grouped_without_anchor():
    """Records sharing only kind=unresolved and different vars do not form one proposal."""
    c = _consolidator()
    # Each record has a unique var — nothing shared beyond "unresolved"
    recs = [
        _bg_rec(f"n_{v}", kind="unresolved_pattern", vars=[v], source_edge_sets=[[v + 10]])
        for v in range(5)
    ]
    rows = [_row(seed=1, bg_records=recs)]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # No proposal should cover all 5 vars in one unresolved group
    all_var_proposals = [p for p in proposals if len(p.vars) == 5]
    assert len(all_var_proposals) == 0


def test_quarantined_authority_groups_with_local_anchors():
    """Quarantined records sharing var+context group together."""
    c = _consolidator()
    rec_q1 = _bg_rec(
        "bg_auth:n_A", kind="quarantined_pattern", vars=[5],
        source_roles=["authority_state:quarantined_for_derivation"],
    )
    rec_q2 = _bg_rec(
        "bg_auth:n_B", kind="quarantined_pattern", vars=[5],
        source_roles=["authority_state:contested_best_available"],
    )
    rows = [_row(seed=1, bg_records=[rec_q1, rec_q2])]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # Both records share var=5 → should produce at least one quarantined_pattern proposal
    assert len(proposals) >= 1
    assert any(5 in p.vars for p in proposals)


def test_unrelated_records_remain_separate():
    """Records with different vars and source_edges do not merge into one proposal."""
    c = _consolidator()
    recs = [
        _bg_rec(f"frontier:x{v}:DIFF(0,{v+1})", kind="unresolved_pattern",
                vars=[v], source_edge_sets=[[v + 1]])
        for v in range(4)
    ]
    rows = [_row(seed=1, bg_records=recs)]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # No proposal should merge all 4 together
    merged = [p for p in proposals if len(p.vars) == 4]
    assert len(merged) == 0


def test_giant_clusters_split_by_var_and_marked_no_runtime_use():
    """Giant cluster records are split by var, marked giant_cluster_subfamily + no_runtime_use."""
    c = _consolidator()
    giant = _bg_rec(
        "bg_uc:giant:0,1,2",
        kind="recurring_low_salience_pattern",
        vars=[0, 1, 2],
        payload={"is_giant": True},
    )
    rows = [
        _row(seed=1, bg_records=[giant]),
        _row(seed=2, bg_records=[giant]),
    ]
    bg = c.extract_background_records(rows)
    unc = c.extract_uncertainty_records(rows)
    proposals = c.build_proposals(bg, [], unc, [], [], min_sources=2)
    giant_props = [p for p in proposals if p.kind == "giant_cluster_subfamily"]
    assert len(giant_props) >= 1
    # All giant proposals must be no_runtime_use
    for p in giant_props:
        assert p.suggested_runtime_use == "no_runtime_use"
        assert "low_specificity" in p.warnings
    # Each giant proposal covers at most a subset of vars (split by var)
    for p in giant_props:
        assert len(p.vars) == 1  # split by individual var


def test_one_all_var_unresolved_not_emitted_as_useful_scaffold():
    """An all-var unresolved group should not appear as feature_only/clustering_prior."""
    c = _consolidator()
    # Records all share kind=unresolved but different vars — no common anchor
    recs = [
        _bg_rec(f"n_{v}", kind="unresolved_pattern", vars=[v])
        for v in range(10)
    ]
    rows = [_row(seed=1, bg_records=recs * 2)]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # If an all-var proposal exists, it must be marked no_runtime_use
    large = [p for p in proposals if len(p.vars) >= 5]
    for p in large:
        assert p.suggested_runtime_use == "no_runtime_use"


def test_authority_allowed_count_is_always_zero():
    """All proposals must have authority_allowed=False."""
    c = _consolidator()
    recs = [
        _bg_rec(f"n_{i}", kind="trass_pattern", vars=[i % 5], source_edge_sets=[[i + 1]])
        for i in range(10)
    ]
    auth_recs = [_auth_rec(var=v) for v in range(5)]
    rows = [
        _row(seed=1, bg_records=recs, auth_records=auth_recs),
        _row(seed=2, bg_records=recs, auth_records=auth_recs),
    ]
    bg = c.extract_background_records(rows)
    cr = c.extract_context_role_records(rows)
    unc = c.extract_uncertainty_records(rows)
    auth = c.extract_authority_records(rows)
    temp = c.extract_temporal_records_if_available(rows)
    proposals = c.build_proposals(bg, cr, unc, auth, temp, min_sources=2)
    for p in proposals:
        assert p.authority_allowed is False, (
            f"proposal {p.proposal_id} has authority_allowed=True"
        )
    summary = c.summarize(rows, bg, cr, unc, auth, temp, proposals)
    assert summary.authority_allowed_count == 0


def test_hidden_truth_fields_are_ignored():
    """Records with hidden-truth-like fields must not affect proposals."""
    c = _consolidator()
    # Add a hidden truth field to a row
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    # Hidden field in the row (not in the bg record)
    row_with_hidden = _row(seed=1, bg_records=[rec],
                           extra_fields={"truth_source_edges": {"0": [1, 2]}})
    rows = [row_with_hidden, _row(seed=2, bg_records=[rec])]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # Proposals should be generated but not reference truth_source_edges content
    summary = c.summarize(rows, bg, [], [], [], [], proposals)
    assert "truth_source_edges" in summary.hidden_truth_fields_seen
    # The proposals themselves should not contain hidden truth data
    for p in proposals:
        assert "truth" not in str(p.to_dict()).lower() or "truth_source_edges" not in str(p.to_dict())


def test_relation_type_not_used_in_proposals():
    """relation_type field in records is not used in proposals (posthoc off by default)."""
    c = _consolidator()
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    # Embed a relation_type in the payload (simulating it being there)
    rec_with_rel = dict(rec)
    rec_with_rel["relation_type"] = "causal"
    rows = [
        _row(seed=1, bg_records=[rec_with_rel]),
        _row(seed=2, bg_records=[rec_with_rel]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2, posthoc_relation_type=False)
    # Proposals should still be generated
    assert len(proposals) >= 1
    # role_patterns should not contain "causal" (relation_type value)
    for p in proposals:
        assert "causal" not in p.role_patterns


def test_empty_input_produces_empty_summary():
    """Empty rows → no proposals, zero counts."""
    c = _consolidator()
    rows: list[dict] = []
    bg = c.extract_background_records(rows)
    cr = c.extract_context_role_records(rows)
    unc = c.extract_uncertainty_records(rows)
    auth = c.extract_authority_records(rows)
    temp = c.extract_temporal_records_if_available(rows)
    proposals = c.build_proposals(bg, cr, unc, auth, temp, min_sources=2)
    summary = c.summarize(rows, bg, cr, unc, auth, temp, proposals)
    assert summary.input_rows == 0
    assert summary.background_records_seen == 0
    assert len(summary.proposals) == 0
    assert summary.authority_allowed_count == 0


def test_proposals_contain_provenance_source_ids():
    """source_record_ids must be populated with actual identifiers."""
    c = _consolidator()
    rec = _bg_rec("frontier:x0:DIFF(5,8)", kind="unresolved_pattern", vars=[0],
                  source_edge_sets=[[1, 8]])
    rows = [
        _row(seed=1, bg_records=[rec]),
        _row(seed=2, bg_records=[rec]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    assert len(proposals) > 0
    for p in proposals:
        assert len(p.source_record_ids) > 0
        assert all(isinstance(sid, str) and sid for sid in p.source_record_ids)


def test_aggregate_zero_fields_diagnosed_when_export_has_per_record_data():
    """When aggregate fields are absent but export records have data, report mismatch."""
    c = _consolidator()
    rec = _bg_rec(
        "n1",
        context_keys=["audit|x0|vis=100"],
        cheap_recognition_score=0.2,
    )
    # Row has background_nethra_edges=0 (present), but no background_contexts_seen or
    # background_recognition_score_mean (simulating batch_run omission)
    row = _row(seed=1, bg_records=[rec])
    # Ensure these aggregate fields are absent (not added)
    assert "background_contexts_seen" not in row
    assert "background_recognition_score_mean" not in row
    rows = [row]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=1)
    summary = c.summarize(rows, bg, [], [], [], [], proposals)
    # Should report the mismatch for contexts_seen and recognition_score_mean
    diag = "\n".join(summary.zero_or_flat_source_fields)
    assert "background_contexts_seen" in diag
    assert "MISMATCH" in diag
    assert "background_recognition_score_mean" in diag


def test_no_imports_from_agent():
    """memory_sleep module must not import from agent.py."""
    import dreth.learner.memory_sleep as ms_mod
    import inspect
    source = inspect.getsource(ms_mod)
    assert "from dreth.agent" not in source
    assert "from .agent" not in source
    assert "import agent" not in source
    assert "ChainedAgent" not in source


def test_cross_run_recurrence_seeds_and_runs_tracked():
    """Cross-run proposals record correct runs_seen and seeds_seen."""
    c = _consolidator()
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    rows = [
        _row(seed=7, bg_records=[rec]),
        _row(seed=42, bg_records=[rec]),
        _row(seed=99, bg_records=[rec]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    p = next((p for p in proposals if "n1" in p.source_record_ids), None)
    assert p is not None
    assert p.runs_seen == 3
    assert p.seeds_seen == 3
    assert p.recurrence_count == 3


def test_min_sources_parameter_filters_small_groups():
    """With min_sources=3, groups of 2 are not emitted."""
    c = _consolidator()
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    rows = [
        _row(seed=1, bg_records=[rec]),
        _row(seed=2, bg_records=[rec]),  # only 2 observations
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=3)
    # With min_sources=3, the 2-observation group should not produce a proposal
    assert not any("n1" in p.source_record_ids for p in proposals)


def test_max_proposals_bounded():
    """max_proposals caps the number of proposals emitted."""
    c = _consolidator()
    # 20 unique nethra_ids, each seen in 2 runs
    recs = [_bg_rec(f"n_{i}", vars=[i], source_edge_sets=[[i + 1]]) for i in range(20)]
    rows = [
        _row(seed=1, bg_records=recs),
        _row(seed=2, bg_records=recs),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2, max_proposals=5)
    assert len(proposals) <= 5


def test_authority_debt_family_groups_same_state():
    """Authority records with same auth_state+reason group into authority_debt_family."""
    c = _consolidator()
    auth_recs = [
        _auth_rec(var=v, authority_state="contested_best_available",
                  reason="active_visible_conflict")
        for v in range(5)
    ]
    rows = [
        _row(seed=1, auth_records=auth_recs),
        _row(seed=2, auth_records=auth_recs),
    ]
    auth = c.extract_authority_records(rows)
    proposals = c.build_proposals([], [], [], auth, [], min_sources=2)
    debt_props = [p for p in proposals if p.kind == "authority_debt_family"]
    assert len(debt_props) >= 1
    assert all(p.authority_allowed is False for p in debt_props)


def test_context_role_recurrence_groups_same_var_kind():
    """Context-role records with same var+kind+source_edges group into context_role_recurrence."""
    c = _consolidator()
    cr_recs = [
        _cr_rec(f"var_fit:x0:MAX(1,2):run{i}", kind="var_fit", target_var=0,
                learned_source_edges=[1, 2])
        for i in range(3)
    ]
    rows = [
        _row(seed=1, cr_records=cr_recs),
        _row(seed=2, cr_records=cr_recs),
    ]
    cr = c.extract_context_role_records(rows)
    proposals = c.build_proposals([], cr, [], [], [], min_sources=2)
    cr_props = [p for p in proposals if p.kind == "context_role_recurrence"]
    assert len(cr_props) >= 1
    assert all(0 in p.vars for p in cr_props if p.vars)


def test_scaffold_proposal_to_dict_serializable():
    """ScaffoldProposal.to_dict() produces JSON-serializable output."""
    p = ScaffoldProposal(
        proposal_id="prop_000001_test",
        kind="unresolved_family",
        source_record_ids=["n1", "n2"],
        source_kinds=["unresolved_pattern"],
        vars=[0, 1],
        contexts=["ctx1"],
        common_signatures=["x0:MAX(1,2)"],
        common_source_edges=[[1, 2]],
        role_patterns=["unresolved"],
        recurring_signals=[],
        recurrence_count=2,
        runs_seen=2,
        seeds_seen=2,
        first_seen_cycle=0,
        last_seen_cycle=100,
        confidence_as_familiarity=0.3,
        action_relevance_score=0.06,
        authority_allowed=False,
        suggested_runtime_use="feature_only",
        evidence_summary="test evidence",
        warnings=[],
    )
    d = p.to_dict()
    # Must serialize without error
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["proposal_id"] == "prop_000001_test"
    assert loaded["authority_allowed"] is False
    assert loaded["kind"] == "unresolved_family"


def test_summary_compression_ratio_greater_than_one_when_grouped():
    """compression_ratio > 1 when records compress into fewer proposals."""
    c = _consolidator()
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    rows = [
        _row(seed=1, bg_records=[rec]),
        _row(seed=2, bg_records=[rec]),
        _row(seed=3, bg_records=[rec]),
    ]
    bg = c.extract_background_records(rows)  # 3 records
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    summary = c.summarize(rows, bg, [], [], [], [], proposals)
    # 3 bg records, 1 proposal → ratio > 1
    assert summary.compression_ratio > 1.0


def test_giant_proposal_not_marked_useful():
    """Giant cluster proposals must always be suggested_runtime_use=no_runtime_use."""
    c = _consolidator()
    giant = _bg_rec(
        "bg_uc:giant:0,1,2,3,4",
        kind="recurring_low_salience_pattern",
        vars=list(range(5)),
        payload={"is_giant": True},
    )
    rows = [
        _row(seed=1, bg_records=[giant]),
        _row(seed=2, bg_records=[giant]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    for p in proposals:
        if p.kind == "giant_cluster_subfamily":
            assert p.suggested_runtime_use == "no_runtime_use"
            assert "low_specificity" in p.warnings


def test_summarize_reports_proposal_by_kind():
    """MemorySleepSummary.proposals_by_kind has correct counts."""
    c = _consolidator()
    trass_rec = _bg_rec("n_trass", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    unres_rec1 = _bg_rec("n_unres1", kind="unresolved_pattern", vars=[1], source_edge_sets=[[2]])
    unres_rec2 = _bg_rec("n_unres2", kind="unresolved_pattern", vars=[1], source_edge_sets=[[2]])
    rows = [
        _row(seed=1, bg_records=[trass_rec, unres_rec1, unres_rec2]),
        _row(seed=2, bg_records=[trass_rec, unres_rec1]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    summary = c.summarize(rows, bg, [], [], [], [], proposals)
    assert isinstance(summary.proposals_by_kind, dict)
    assert len(summary.proposals) == len(proposals)


def test_parse_sig_source_edges_extracts_correctly():
    assert _parse_sig_source_edges("x0:MAX(1,8)") == frozenset({1, 8})
    assert _parse_sig_source_edges("x3:LOW(4,5,6)") == frozenset({4, 5, 6})
    assert _parse_sig_source_edges("x0:HIGH()") == frozenset()
    assert _parse_sig_source_edges("invalid") == frozenset()
    assert _parse_sig_source_edges("") == frozenset()


def test_bg_anchor_key_excludes_giant():
    giant = _bg_rec("bg_uc:giant", kind="recurring_low_salience_pattern", vars=[0, 1])
    assert _bg_anchor_key(giant) is None


def test_bg_anchor_key_excludes_no_var():
    no_var = _bg_rec("n1", vars=[])
    no_var_rec = dict(no_var)
    no_var_rec["vars"] = []
    assert _bg_anchor_key(no_var_rec) is None


def test_bg_anchor_key_extracts_source_edge_from_signature():
    rec = _bg_rec("n1", vars=[0], source_edge_sets=[], fit_signatures=["x0:MAX(1,8)"])
    key = _bg_anchor_key(rec)
    assert key is not None
    kind, vars_fs, source_edge_fs = key
    assert source_edge_fs == frozenset({1, 8})


def test_hidden_truth_like_fields_constant_is_correct():
    assert "truth_source_edges" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "truth_func" in HIDDEN_TRUTH_LIKE_FIELDS
    assert "debug_blind_challenge_manifest" in HIDDEN_TRUTH_LIKE_FIELDS


def test_proposals_do_not_use_hidden_truth_fields():
    """Even if rows contain hidden truth fields in nested records, proposals ignore them."""
    c = _consolidator()
    rec = _bg_rec("n1", kind="trass_pattern", vars=[0], source_edge_sets=[[1]])
    # Inject truth fields into the background record itself
    rec_with_truth = dict(rec)
    rec_with_truth["truth_source_edges"] = {"0": [1, 2]}
    rec_with_truth["truth_func"] = "MAX"
    rows = [
        _row(seed=1, bg_records=[rec_with_truth]),
        _row(seed=2, bg_records=[rec_with_truth]),
    ]
    bg = c.extract_background_records(rows)
    proposals = c.build_proposals(bg, [], [], [], [], min_sources=2)
    # Proposals should be generated (records still valid)
    assert len(proposals) >= 1
    # Summary should detect and report the hidden truth fields
    summary = c.summarize(rows, bg, [], [], [], [], proposals)
    assert "truth_source_edges" in summary.hidden_truth_fields_seen
    assert "truth_func" in summary.hidden_truth_fields_seen
    # authority_allowed must still be 0
    assert summary.authority_allowed_count == 0
