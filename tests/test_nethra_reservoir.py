from __future__ import annotations

import inspect
import random

import pytest

import dreth.agent as agent_mod
from dreth.agent import ChainedAgent
from dreth.context_role_index import (
    ContextRoleIndex,
    ContextRoleRecord,
    NethraNode,
)
from dreth.uncertainty_consolidation import UncertaintyCase
from dreth.world import CausalWorld


def _agent(
    *,
    index_mode: str = "off",
    uc_mode: str = "off",
    anchor_policy: str | None = None,
) -> ChainedAgent:
    world = CausalWorld(5, random.Random(3), noise_sigma=0.0)
    world.visible_count = 5
    return ChainedAgent(
        world=world,
        rng=random.Random(13003),
        sentinel_count=5,
        sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world.n_vars,
        uncertainty_consolidation_mode=uc_mode,
        uncertainty_assist_policy="local_only",
        context_role_index_mode=index_mode,
        context_role_anchor_policy=anchor_policy,
    )


def _case(var: int) -> UncertaintyCase:
    return UncertaintyCase(
        var=var,
        cycle=10,
        action="preserve_ambiguity",
        active_signals=("open_novelty", "near_tie_count"),
        learned_parents=(),
        near_tie_candidates=(),
        tied_frontier_info={"active": False},
        novelty_state="open",
        recent_fit_history=(),
        sentinels=(),
        consequence_tier="skip_tareth",
        graph_neighbors=(),
    )


def test_off_mode_preserves_behavior() -> None:
    off = _agent(index_mode="off")
    default = _agent()
    off.initialize()
    default.initialize()
    for cycle in range(1, 8):
        off.run_cycle(cycle)
        default.run_cycle(cycle)

    assert off.records == default.records
    assert off.skip_count == default.skip_count
    assert off.full_audit_count == default.full_audit_count
    assert off.total_interventions == default.total_interventions


def test_record_mode_records_nethras_but_does_not_change_behavior() -> None:
    off = _agent(index_mode="off")
    record = _agent(index_mode="record")
    off.initialize()
    record.initialize()
    for cycle in range(1, 5):
        off.run_cycle(cycle)
        record.run_cycle(cycle)

    assert off.records == record.records
    assert off.skip_count == record.skip_count
    assert off.total_interventions == record.total_interventions
    assert record.context_role_index_metrics()["context_role_index_nodes"] > 0


def test_loose_assist_feature_exposes_broad_index_match_as_local_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(index_mode="assist_feature", uc_mode="assist", anchor_policy="loose")
    agent.initialize()
    index = agent._context_role_index
    assert index is not None
    index.add_or_update_node(NethraNode(
        nethra_id="shape-handle",
        kind="var_fit",
        components=(0, 1),
        learned_parents=(),
        first_seen_cycle=1,
        last_seen_cycle=1,
        source="audit",
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="shape-handle",
        context_key="operation|x0|x1",
        operation="classification",
        role="trass",
        cycle=1,
        strong_observations=2,
        validity_scope=(0, 1),
    ))
    monkeypatch.setattr(
        agent_mod,
        "extract_uncertainty_cases_from_agent",
        lambda _agent, _cycle: [_case(0), _case(1)],
    )

    agent._run_uncertainty_consolidation(2)
    metrics = agent.context_role_index_metrics()

    assert agent._uncertainty_budget_bonus
    assert metrics["context_role_matches_used_as_local_anchor"] > 0
    assert metrics["context_role_assist_feature_hits"] > 0


def test_strict_assist_feature_rejects_broad_unresolved_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(index_mode="assist_feature", uc_mode="assist", anchor_policy="strict")
    agent.initialize()
    index = agent._context_role_index
    assert index is not None
    index.add_or_update_node(NethraNode(
        nethra_id="broad-unresolved",
        kind="var_fit",
        components=(0, 1),
        learned_parents=(),
        first_seen_cycle=1,
        last_seen_cycle=1,
        source="audit",
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="broad-unresolved",
        context_key="operation|x0|x1",
        operation="classification",
        role="unresolved",
        cycle=1,
        uncertainty_signals=("open_novelty", "near_tie_count"),
        validity_scope=(0, 1),
    ))
    monkeypatch.setattr(
        agent_mod,
        "extract_uncertainty_cases_from_agent",
        lambda _agent, _cycle: [_case(0), _case(1)],
    )

    agent._run_uncertainty_consolidation(2)
    metrics = agent.context_role_index_metrics()

    assert not agent._uncertainty_budget_bonus
    assert metrics["context_role_matches_used_as_local_anchor"] == 0
    assert metrics["context_role_matches_suppressed_weak"] > 0


def test_strict_exact_nethra_signature_context_qualifies() -> None:
    index = ContextRoleIndex()
    nid = "exact"
    signature = "x0:FIRST(2)"
    index.add_or_update_node(NethraNode(
        nethra_id=nid,
        kind="var_fit",
        target_var=0,
        components=(0, 2),
        learned_parents=(2,),
        learned_func="FIRST",
        signature=signature,
        first_seen_cycle=1,
        last_seen_cycle=1,
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id=nid,
        context_key="uncertainty_cluster|possible_missing_operator",
        operation="uncertainty_consolidation",
        role="tareth",
        cycle=1,
        validity_scope=(0, 2),
    ))
    cluster = type("Cluster", (), {
        "cluster_id": "uc1",
        "vars": (0, 1),
        "shared_parents": (2,),
        "shared_graph_neighbors": (),
        "shared_signals": ("near_tie_count",),
        "proposed_handle_kind": "possible_missing_operator",
        "learned_signature": signature,
        "is_giant_cluster": False,
    })()

    matches = index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=2,
    )

    assert [m.nethra_id for m in matches] == [nid]


def test_strict_shared_parent_and_context_family_qualifies() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(
        nethra_id="parent-family",
        kind="var_fit",
        target_var=0,
        components=(0, 2),
        learned_parents=(2,),
        first_seen_cycle=1,
        last_seen_cycle=1,
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="parent-family",
        context_key="uncertainty_cluster|shared_ambiguity",
        operation="uncertainty_consolidation",
        role="tareth",
        cycle=1,
        validity_scope=(0, 2),
    ))
    cluster = type("Cluster", (), {
        "cluster_id": "uc1",
        "vars": (0, 1),
        "shared_parents": (2,),
        "shared_graph_neighbors": (),
        "shared_signals": ("near_tie_count",),
        "proposed_handle_kind": "shared_ambiguity",
        "is_giant_cluster": False,
    })()

    assert index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=2,
    )


def test_strict_role_transition_qualifies() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(
        nethra_id="transition",
        kind="var_fit",
        target_var=0,
        components=(0,),
        first_seen_cycle=1,
        last_seen_cycle=3,
    ))
    for cycle, role in ((1, "trass"), (2, "tareth")):
        index.assign_context_role(ContextRoleRecord(
            nethra_id="transition",
            context_key="uncertainty_cluster|possible_latent_regime",
            operation="uncertainty_consolidation",
            role=role,
            cycle=cycle,
            validity_scope=(0,),
        ))
    cluster = type("Cluster", (), {
        "cluster_id": "uc1",
        "vars": (0, 1),
        "shared_parents": (),
        "shared_graph_neighbors": (),
        "shared_signals": ("sentinel_failures",),
        "proposed_handle_kind": "possible_latent_regime",
        "is_giant_cluster": False,
    })()

    assert index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=3,
    )


def test_cap_suppresses_excessive_anchors() -> None:
    index = ContextRoleIndex()
    index.max_local_anchors_per_cluster = 2
    for i in range(4):
        index.add_or_update_node(NethraNode(
            nethra_id=f"n{i}",
            kind="var_fit",
            target_var=0,
            components=(0, 2),
            learned_parents=(2,),
            first_seen_cycle=1,
            last_seen_cycle=1,
        ))
        index.assign_context_role(ContextRoleRecord(
            nethra_id=f"n{i}",
            context_key="uncertainty_cluster|shared_ambiguity",
            operation="uncertainty_consolidation",
            role="tareth",
            cycle=1,
            validity_scope=(0, 2),
        ))
    cluster = type("Cluster", (), {
        "cluster_id": "uc1",
        "vars": (0, 1),
        "shared_parents": (2,),
        "shared_graph_neighbors": (),
        "shared_signals": ("near_tie_count",),
        "proposed_handle_kind": "shared_ambiguity",
        "is_giant_cluster": False,
    })()

    matches = index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=2,
    )
    metrics = index.summarize()

    assert len(matches) == 2
    assert metrics["context_role_matches_suppressed_cap"] == 2


def test_duplicate_matches_are_suppressed_within_cycle() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(
        nethra_id="dup",
        kind="var_fit",
        target_var=0,
        components=(0, 2),
        learned_parents=(2,),
        first_seen_cycle=1,
        last_seen_cycle=1,
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="dup",
        context_key="uncertainty_cluster|shared_ambiguity",
        operation="uncertainty_consolidation",
        role="tareth",
        cycle=1,
        validity_scope=(0, 2),
    ))
    cluster = type("Cluster", (), {
        "cluster_id": "uc1",
        "vars": (0, 1),
        "shared_parents": (2,),
        "shared_graph_neighbors": (),
        "shared_signals": ("near_tie_count",),
        "proposed_handle_kind": "shared_ambiguity",
        "is_giant_cluster": False,
    })()

    assert len(index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=2,
    )) == 1
    assert index.query_for_uncertainty_cluster(
        cluster,
        anchor_policy="strict",
        current_cycle=2,
    ) == ()
    assert index.summarize()["context_role_matches_suppressed_duplicate"] == 1


def test_same_nethra_can_be_trass_and_tareth_in_different_contexts() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(nethra_id="shape", components=(1, 2)))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="shape", context_key="color", operation="classify",
        role="trass", cycle=1,
    ))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="shape", context_key="grasp", operation="plan",
        role="tareth", cycle=2,
    ))

    node, roles = index.query_by_nethra("shape")
    assert node is not None
    assert {role.role for role in roles} == {"trass", "tareth"}


def test_trass_role_does_not_delete_or_suppress_node() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(nethra_id="shape", components=(1,)))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="shape", context_key="color", operation="classify",
        role="trass", cycle=1,
    ))

    node, roles = index.query_by_nethra("shape")
    assert node is not None
    assert len(roles) == 1
    assert index.query_by_component(1)[0].nethra_id == "shape"


def test_broad_query_without_local_overlap_returns_no_match() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(nethra_id="remote", components=(9,)))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="remote", context_key="remote", operation="classify",
        role="tareth", cycle=1, strong_observations=3,
    ))

    assert index.query_for_uncertainty_cluster(type("Cluster", (), {
        "vars": (0, 1),
        "shared_parents": (),
        "shared_graph_neighbors": (),
        "shared_signals": ("open_novelty",),
        "proposed_handle_kind": "unknown",
    })()) == ()


def test_context_mismatch_blocks_weak_single_component_match() -> None:
    index = ContextRoleIndex()
    index.add_or_update_node(NethraNode(nethra_id="weak", components=(0,)))
    index.assign_context_role(ContextRoleRecord(
        nethra_id="weak", context_key="unrelated_context", operation="classify",
        role="tareth", cycle=1, strong_observations=3,
    ))

    matches = index.query_for_uncertainty_cluster(type("Cluster", (), {
        "vars": (0, 1),
        "shared_parents": (),
        "shared_graph_neighbors": (),
        "shared_signals": ("low_margin",),
        "proposed_handle_kind": "unknown",
    })())
    assert matches == ()


def test_hidden_truth_manifest_is_not_read_by_runtime_index() -> None:
    runtime_source = inspect.getsource(ChainedAgent._run_uncertainty_consolidation)
    index_source = inspect.getsource(ContextRoleIndex)
    banned = [
        "debug_blind_challenge_manifest",
        "blind_challenge_manifest",
        "truth_parents",
        "truth_func",
        "truth_delayed_parents",
        "truth_latents",
    ]
    for field in banned:
        assert field not in runtime_source
        assert field not in index_source


def test_index_does_not_issue_revoke_suppress_skip_or_replace_fit_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(index_mode="assist_feature", uc_mode="assist")
    agent.initialize()
    before_certs = sum(len(n.certificates) for n in agent.ledger.vars.values())
    before_roles = [agent.ledger.vars[v].role_for("skip") for v in range(agent.world.visible_count)]
    before_fits = [
        (agent.ledger.vars[v].parents, agent.ledger.vars[v].func)
        for v in range(agent.world.visible_count)
    ]
    monkeypatch.setattr(
        agent_mod,
        "extract_uncertainty_cases_from_agent",
        lambda _agent, _cycle: [_case(0), _case(1)],
    )

    agent._run_uncertainty_consolidation(2)

    after_certs = sum(len(n.certificates) for n in agent.ledger.vars.values())
    after_roles = [agent.ledger.vars[v].role_for("skip") for v in range(agent.world.visible_count)]
    after_fits = [
        (agent.ledger.vars[v].parents, agent.ledger.vars[v].func)
        for v in range(agent.world.visible_count)
    ]
    assert after_certs == before_certs
    assert after_roles == before_roles
    assert after_fits == before_fits
