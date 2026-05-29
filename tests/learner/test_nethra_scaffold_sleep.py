from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dreth.learner.nethra_scaffold_sleep import (  # noqa: E402
    NethraScaffoldSleep,
    ScaffoldAbstraction,
    ScaffoldComposition,
    ScaffoldNethra,
    write_scaffold_sleep_jsonl,
)


def _bg_rec(
    nethra_id: str,
    *,
    kind: str = "context_role_pattern",
    vars: list[int] | None = None,
    contexts: list[str] | None = None,
    roles: list[str] | None = None,
    signatures: list[str] | None = None,
    source_edges: list[list[int]] | None = None,
) -> dict[str, Any]:
    return {
        "nethra_id": nethra_id,
        "kind": kind,
        "vars": vars if vars is not None else [0],
        "context_keys": contexts or ["audit|x0|vis=100"],
        "source_roles": roles or ["best_available"],
        "fit_signatures": signatures or ["x0:MAX(1,2)"],
        "source_edge_sets": source_edges if source_edges is not None else [[1, 2]],
        "first_seen_cycle": 1,
        "last_seen_cycle": 3,
        "payload": {},
    }


def _cr_node(
    nethra_id: str = "var_fit:x0:MAX(1,2)",
    *,
    var: int = 0,
    source_edges: list[int] | None = None,
    func: str = "MAX",
) -> dict[str, Any]:
    source_edge_vals = source_edges or [1, 2]
    return {
        "nethra_id": nethra_id,
        "kind": "var_fit",
        "target_var": var,
        "components": [var] + source_edge_vals,
        "learned_source_edges": source_edge_vals,
        "learned_func": func,
        "signature": f"x{var}:{func}({','.join(str(p) for p in source_edge_vals)})",
        "first_seen_cycle": 1,
        "last_seen_cycle": 5,
    }


def _cr_role(nethra_id: str, role: str, context: str) -> dict[str, Any]:
    return {
        "nethra_id": nethra_id,
        "context_key": context,
        "operation": context.split("|", 1)[0],
        "role": role,
        "cycle": 5,
    }


def _auth_rec(
    *,
    var: int = 0,
    nethra_id: str = "",
    authority_state: str = "contested_best_available",
    reason: str = "active_visible_conflict",
) -> dict[str, Any]:
    rec = {
        "var": var,
        "context_key": f"authority_strength|x{var}|vis=100",
        "authority_state": authority_state,
        "reason": reason,
        "cycle": 7,
        "best_available": True,
    }
    if nethra_id:
        rec["nethra_id"] = nethra_id
    return rec


def _row(
    *,
    seed: int = 1,
    bg: list[dict[str, Any]] | None = None,
    cr_records: list[dict[str, Any]] | None = None,
    cr_roles: list[dict[str, Any]] | None = None,
    auth: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "background_nethra_export": {"records": bg or [], "edges": [], "role_shift_examples": []},
        "context_role_index": {
            "records": cr_records or [],
            "roles": cr_roles or [],
            "edges": [],
            "match_attribution": [],
        },
        "authority_strength": {"records": auth or [], "summary": {}, "controller": {}},
    }
    if extra:
        row.update(extra)
    return row


def _run_sleep(rows: list[dict[str, Any]]):
    sleep = NethraScaffoldSleep()
    rows = sleep.load_rows(rows)
    nethras = sleep.extract_scaffold_nethras(rows)
    maps = sleep.build_role_maps(nethras)
    comps = sleep.build_compositions(nethras)
    abstractions = sleep.build_abstractions(nethras, comps)
    summary = sleep.summarize(rows, nethras, comps, abstractions, maps)
    return sleep, nethras, maps, comps, abstractions, summary


def test_stable_best_available_records_are_included_not_ignored():
    _, nethras, _, _, _, summary = _run_sleep([
        _row(bg=[_bg_rec("var_fit:x0:MAX(1,2)", roles=["best_available"])])
    ])
    assert nethras
    assert summary.stable_best_available_records_seen > 0
    assert any("best_available" in n.observed_roles for n in nethras)


def test_trass_records_become_scaffold_role_entries_not_deletion():
    _, nethras, maps, _, _, summary = _run_sleep([
        _row(bg=[_bg_rec(
            "var_fit:x1:LOW()",
            kind="trass_pattern",
            vars=[1],
            contexts=["skip|x1|vis=100"],
            roles=["trass"],
            signatures=["x1:LOW()"],
            source_edges=[],
        )])
    ])
    assert summary.trass_records_seen > 0
    assert any("trass" in n.observed_roles for n in nethras)
    assert any(m.trass_contexts for m in maps)


def test_same_structure_can_have_tareth_and_trass_roles_in_different_contexts():
    nid = "var_fit:x2:FIRST(1)"
    _, nethras, maps, _, _, _ = _run_sleep([
        _row(bg=[
            _bg_rec(nid, vars=[2], contexts=["audit|x2|vis=100"], roles=["tareth"], signatures=["x2:FIRST(1)"], source_edges=[[1]]),
            _bg_rec(nid, vars=[2], contexts=["skip|x2|vis=100"], roles=["trass"], signatures=["x2:FIRST(1)"], source_edges=[[1]]),
        ])
    ])
    scaffold = next(n for n in nethras if "x2:FIRST(1)" in n.signatures)
    assert {"tareth", "trass"}.issubset(set(scaffold.observed_roles))
    role_map = next(m for m in maps if m.scaffold_id == scaffold.scaffold_id)
    assert role_map.tareth_contexts == ["audit|x2|vis=100"]
    assert role_map.trass_contexts == ["skip|x2|vis=100"]
    assert role_map.role_shift_examples


def test_background_records_contribute_to_scaffold_nethras():
    _, nethras, _, _, _, summary = _run_sleep([
        _row(bg=[_bg_rec(
            "bg_uc:x3",
            kind="recurring_low_salience_pattern",
            vars=[3],
            contexts=["uncertainty_cluster|vis=100"],
            roles=["background"],
            signatures=["x3:HIGH()"],
            source_edges=[],
        )])
    ])
    assert summary.background_records_seen > 0
    assert any("background" in n.observed_roles for n in nethras)


def test_unresolved_records_group_only_with_local_structural_anchors():
    _, nethras, _, _, _, _ = _run_sleep([
        _row(bg=[
            _bg_rec("frontier:x0:MAX(1,2)", kind="unresolved_pattern", vars=[0], roles=["unresolved"], signatures=["x0:MAX(1,2)"], source_edges=[[1, 2]]),
            _bg_rec("frontier:x0:PROD(1,2)", kind="unresolved_pattern", vars=[0], roles=["unresolved"], signatures=["x0:PROD(1,2)"], source_edges=[[1, 2]]),
            _bg_rec("frontier:x9:LOW()", kind="unresolved_pattern", vars=[9], roles=["unresolved"], signatures=["x9:LOW()"], source_edges=[]),
        ])
    ])
    broad_unresolved = [
        n for n in nethras
        if set(n.vars) == {0, 9} and n.observed_roles == ["unresolved"]
    ]
    assert broad_unresolved == []


def test_broad_active_visible_conflict_does_not_define_useful_abstraction():
    _, _, _, _, abstractions, summary = _run_sleep([
        _row(seed=1, auth=[_auth_rec(var=0)]),
        _row(seed=2, auth=[_auth_rec(var=1)]),
    ])
    assert summary.broad_generic_debt_count == 2
    assert summary.broad_generic_debt_useful_count == 0
    useful_debt = [
        a for a in abstractions
        if a.kind == "authority_debt_family" and a.suggested_runtime_use != "no_runtime_use"
    ]
    assert useful_debt == []


def test_nethra_of_nethra_composition_created_from_shared_lower_structures():
    _, nethras, _, comps, _, _ = _run_sleep([
        _row(bg=[
            _bg_rec("var_fit:x4:MAX(1,2)", vars=[4], signatures=["x4:MAX(1,2)"], source_edges=[[1, 2]]),
            _bg_rec("var_fit:x5:MIN(1,2)", vars=[5], signatures=["x5:MIN(1,2)"], source_edges=[[1, 2]]),
        ])
    ])
    assert len(nethras) >= 2
    assert comps
    assert any(len(c.lower_scaffold_ids) >= 2 for c in comps)


def test_authority_allowed_count_remains_zero_for_all_outputs():
    _, nethras, maps, comps, abstractions, summary = _run_sleep([
        _row(bg=[_bg_rec("var_fit:x0:MAX(1,2)")], auth=[_auth_rec(nethra_id="var_fit:x0:MAX(1,2)")])
    ])
    assert summary.authority_allowed_count == 0
    assert summary.behavior_effects == 0
    assert all(n.authority_allowed is False for n in nethras)
    assert all(c.authority_allowed is False for c in comps)
    assert all(a.authority_allowed is False for a in abstractions)
    assert all(m.to_dict()["authority_allowed"] is False for m in maps)


def test_hidden_truth_debug_fields_are_ignored():
    hidden = _bg_rec(
        "var_fit:x0:MAX(1,2)",
        signatures=["x0:MAX(1,2)"],
        source_edges=[[1, 2]],
    )
    hidden["truth_source_edges"] = [99]
    hidden["payload"] = {"debug_blind_challenge_manifest": {"secret": True}, "truth_func": "LOW"}
    _, nethras, _, _, _, summary = _run_sleep([
        _row(bg=[hidden], extra={"truth_func": "LOW"})
    ])
    assert "truth_func" in summary.hidden_truth_fields_seen
    assert "truth_source_edges" in summary.hidden_truth_fields_seen
    assert "debug_blind_challenge_manifest" in summary.hidden_truth_fields_seen
    assert "truth" not in json.dumps([n.to_dict() for n in nethras])
    assert "debug_blind_challenge_manifest" not in json.dumps([n.to_dict() for n in nethras])


def test_no_runtime_behavior_changes_or_agent_imports():
    import inspect
    import dreth.learner.nethra_scaffold_sleep as mod

    source = inspect.getsource(mod)
    assert "from dreth.agent" not in source
    assert "from .agent" not in source
    assert "import agent" not in source
    _, _, _, _, _, summary = _run_sleep([_row(bg=[_bg_rec("var_fit:x0:MAX(1,2)")])])
    assert summary.behavior_effects == 0


def test_jsonl_writer_emits_all_record_types():
    n = ScaffoldNethra(
        scaffold_id="s1",
        source_ids=["r1"],
        source_types=["background_nethra"],
        vars=[0],
        signatures=["x0:HIGH()"],
        source_edge_sets=[],
        contexts=["audit|x0"],
        observed_roles=["best_available"],
        role_counts={"best_available": 1},
        role_contexts={"best_available": ["audit|x0"]},
        first_seen_cycle=0,
        last_seen_cycle=1,
        runs_seen=1,
        seeds_seen=1,
        familiarity_score=0.1,
        specificity_score=0.5,
        stability_score=0.3,
    )
    c = ScaffoldComposition(
        composition_id="c1",
        lower_scaffold_ids=["s1", "s2"],
        higher_scaffold_id="h1",
        shared_vars=[],
        shared_contexts=[],
        shared_signatures=[],
        evidence_summary="test",
        confidence_as_familiarity=0.4,
    )
    a = ScaffoldAbstraction(
        abstraction_id="a1",
        kind="operator_family",
        member_scaffold_ids=["s1", "s2"],
        common_structure={"group": "HIGH"},
        role_distribution={"best_available": 1},
        specificity_score=0.5,
        familiarity_score=0.4,
        suggested_runtime_use="feature_only",
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sleep.jsonl"
        write_scaffold_sleep_jsonl(path, [n], [], [c], [a])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["record_type"] for row in rows} == {
        "scaffold_nethra",
        "scaffold_composition",
        "scaffold_abstraction",
    }
    assert all(row["authority_allowed"] is False for row in rows)


def test_previous_sleep_proposal_rows_are_visible_inputs():
    proposal = {
        "proposal_id": "prop_1",
        "kind": "context_role_recurrence",
        "source_record_ids": ["var_fit:x0:HIGH()"],
        "source_kinds": ["context_role_pattern"],
        "vars": [0],
        "contexts": ["audit|x0|vis=100"],
        "common_signatures": ["x0:HIGH()"],
        "common_source_edges": [],
        "role_patterns": ["best_available"],
        "authority_allowed": True,
    }
    _, nethras, _, _, _, summary = _run_sleep([proposal])
    assert summary.raw_records_read == 1
    assert any("scaffold_proposal" in n.source_types for n in nethras)
    assert summary.authority_allowed_count == 0
