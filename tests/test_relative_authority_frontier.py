from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative
from dreth.relative_authority import NethraGraphSnapshot, NethraNodeRef, NethraRelation
from dreth.relative_authority_frontier import (
    TemporalGraphFrontierEvaluator,
    evaluate_frontier_against_agent,
    evaluate_frontier_leave_one_out,
    propose_frontier,
)
from dreth.relative_authority_observer import build_snapshot_from_agent
from dreth.world import CausalWorld
from scripts.batch_run import RunConfig, _build_and_run_dreth, _run_one


ROOT = Path(__file__).resolve().parents[1]


def _node(node_id: str, var: int | None = None, kind: str = "nethra_var") -> NethraNodeRef:
    return NethraNodeRef(node_id=node_id, kind=kind, var=var)


def _make_snapshot() -> NethraGraphSnapshot:
    a = _node("var:0", 0)
    b = _node("var:1", 1)
    c = _node("var:2", 2)
    d = _node("var:3", 3)
    return NethraGraphSnapshot(
        nodes=(a, b, c, d),
        relations=(
            NethraRelation(a, b, "depends_on", "ctx"),
            NethraRelation(b, c, "coactive_with", "ctx"),
            NethraRelation(c, d, "shares_node", "ctx"),
        ),
        authority_records=(),
    )


def _make_initialized_agent():
    rng_w = random.Random(3)
    rng_a = random.Random(13003)
    world = CausalWorld(5, rng_w, noise_sigma=0.0)
    world.visible_count = 5
    agent = ChainedAgent(
        world,
        rng_a,
        sentinel_count=5,
        sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world.n_vars,
    )
    agent.initialize()
    return agent


def _force_next_cycle_audits(agent) -> None:
    agent._live_set = set(range(agent.world.visible_count))
    for var in range(agent.world.visible_count):
        nethra = agent.ledger.vars[var]
        nethra.certificates.clear()
        nethra.route_certs.clear()
        nethra.sentinels = []
        nethra.compressions = []
        nethra.status = "uncertain"


def _ledger_fingerprint(agent):
    return {
        var: (
            n.parents,
            n.func,
            n.status,
            tuple(
                sorted(
                    (op, cert.role, cert.revoked_by)
                    for op, cert in n.certificates.items()
                )
            ),
            tuple(
                sorted(
                    (candidate, cert.role, cert.revoked_by)
                    for candidate, cert in n.route_certs.items()
                )
            ),
            tuple(
                (alt.parents, alt.func, alt.last_score, alt.revival_count, tuple(sorted(alt.context_keys_seen)))
                for alt in n.dormant_alternatives
            ),
            n.skip_count,
            n.full_audits,
        )
        for var, n in sorted(agent.ledger.vars.items())
    }, len(agent.ledger.event_log)


def test_propose_frontier_returns_neighbors_within_max_depth() -> None:
    proposal = propose_frontier(_make_snapshot(), "var:0", max_depth=2)

    assert proposal.target_var == 0
    assert proposal.candidate_node_ids == ("var:1", "var:2")
    assert proposal.frontier_size == 2
    assert proposal.relation_types_used == ("coactive_with", "depends_on")


def test_propose_frontier_max_candidates_caps_output() -> None:
    proposal = propose_frontier(_make_snapshot(), "var:0", max_depth=3, max_candidates=1)

    assert proposal.candidate_node_ids == ("var:1",)
    assert proposal.frontier_size == 1


def test_chosen_parent_hit_works_on_tiny_agent_ledger() -> None:
    agent = _make_initialized_agent()
    agent.ledger.vars[2].parents = (0,)
    snapshot = build_snapshot_from_agent(agent)

    evals = {
        evaluation.target_var: evaluation
        for evaluation in evaluate_frontier_against_agent(snapshot, agent)
    }

    assert evals[2].chosen_parent_hits == 1
    assert evals[2].chosen_parent_total == 1


def test_revoked_cert_neighbor_hit_works_if_revoked_cert_exists() -> None:
    agent = _make_initialized_agent()
    agent.ledger.issue_cert(
        2,
        "skip",
        "untested",
        "none",
        context_parents=(),
        context_visible=5,
        context_cycle=1,
        targets=(2,),
        substitutions_tested=("test",),
        changes=0,
        trials=1,
        earned_by="substitution_test",
        revoked_by="sentinel_failure",
    )
    snapshot = build_snapshot_from_agent(agent)

    evals = {
        evaluation.target_var: evaluation
        for evaluation in evaluate_frontier_against_agent(snapshot, agent)
    }

    assert evals[2].revoked_neighbor_hits >= 1
    assert evals[2].revoked_total >= 1


def test_dormant_alternative_hit_works_if_dormant_alternatives_exist() -> None:
    agent = _make_initialized_agent()
    agent.ledger.vars[2].dormant_alternatives.append(
        DormantAlternative(
            parents=(1,),
            func="FIRST",
            last_score=7,
            revival_count=1,
            context_keys_seen={101},
            last_seen_cycle=5,
        )
    )
    snapshot = build_snapshot_from_agent(agent)

    evals = {
        evaluation.target_var: evaluation
        for evaluation in evaluate_frontier_against_agent(snapshot, agent)
    }

    assert evals[2].dormant_neighbor_hits == 1
    assert evals[2].dormant_total == 1


def test_leave_one_out_chosen_parent_uses_indirect_graph_path() -> None:
    source = _node("var:0", 0)
    parent = _node("var:1", 1)
    bridge = _node("var:2", 2)
    snapshot = NethraGraphSnapshot(
        nodes=(source, parent, bridge),
        relations=(
            NethraRelation(source, parent, "depends_on", "ctx"),
            NethraRelation(source, bridge, "shares_node", "ctx"),
            NethraRelation(bridge, parent, "coactive_with", "ctx"),
        ),
        authority_records=(),
    )
    agent = SimpleNamespace(
        ledger=SimpleNamespace(
            vars={
                0: SimpleNamespace(parents=(1,), certificates={}, route_certs={}),
            },
        ),
    )

    evaluation = evaluate_frontier_leave_one_out(snapshot, agent)[0]

    assert evaluation.chosen_parent_hits == 1
    assert evaluation.chosen_parent_total == 1


def test_leave_one_out_chosen_parent_misses_direct_only_edge() -> None:
    source = _node("var:0", 0)
    parent = _node("var:1", 1)
    snapshot = NethraGraphSnapshot(
        nodes=(source, parent),
        relations=(NethraRelation(source, parent, "depends_on", "ctx"),),
        authority_records=(),
    )
    agent = SimpleNamespace(
        ledger=SimpleNamespace(
            vars={
                0: SimpleNamespace(parents=(1,), certificates={}, route_certs={}),
            },
        ),
    )

    evaluation = evaluate_frontier_leave_one_out(snapshot, agent)[0]

    assert evaluation.chosen_parent_hits == 0
    assert evaluation.chosen_parent_total == 1


def test_leave_one_out_revoked_cert_uses_indirect_graph_path() -> None:
    source = _node("var:0", 0)
    cert = _node("cert:0:skip", 0, kind="certificate")
    bridge = _node("var:1", 1)
    snapshot = NethraGraphSnapshot(
        nodes=(source, cert, bridge),
        relations=(
            NethraRelation(source, cert, "coactive_with", "ctx"),
            NethraRelation(source, bridge, "shares_node", "ctx"),
            NethraRelation(bridge, cert, "shares_node", "ctx"),
        ),
        authority_records=(),
    )
    revoked = SimpleNamespace(revoked_by="sentinel_failure")
    agent = SimpleNamespace(
        ledger=SimpleNamespace(
            vars={
                0: SimpleNamespace(
                    parents=(),
                    certificates={"skip": revoked},
                    route_certs={},
                ),
            },
        ),
    )

    evaluation = evaluate_frontier_leave_one_out(snapshot, agent)[0]

    assert evaluation.revoked_neighbor_hits == 2
    assert evaluation.revoked_total == 2


def test_leave_one_out_dormant_uses_indirect_graph_path() -> None:
    source = _node("var:0", 0)
    dormant = _node("dormant:0:0:():LOW", 0, kind="dormant_alternative")
    bridge = _node("var:1", 1)
    snapshot = NethraGraphSnapshot(
        nodes=(source, dormant, bridge),
        relations=(
            NethraRelation(dormant, source, "substitutes_for", "ctx"),
            NethraRelation(source, bridge, "shares_node", "ctx"),
            NethraRelation(bridge, dormant, "shares_node", "ctx"),
        ),
        authority_records=(),
    )
    agent = SimpleNamespace(
        ledger=SimpleNamespace(
            vars={
                0: SimpleNamespace(parents=(), certificates={}, route_certs={}),
            },
        ),
    )

    evaluation = evaluate_frontier_leave_one_out(snapshot, agent)[0]

    assert evaluation.dormant_neighbor_hits == 1
    assert evaluation.dormant_total == 1


def test_evaluator_does_not_mutate_agent_or_ledger() -> None:
    agent = _make_initialized_agent()
    agent.ledger.vars[2].parents = (0,)
    snapshot = build_snapshot_from_agent(agent)
    before = _ledger_fingerprint(agent)

    _ = evaluate_frontier_against_agent(snapshot, agent)
    _ = evaluate_frontier_leave_one_out(snapshot, agent)

    assert _ledger_fingerprint(agent) == before


def test_temporal_frontier_records_no_proposals_before_warmup() -> None:
    agent = _make_initialized_agent()
    _force_next_cycle_audits(agent)
    observer = TemporalGraphFrontierEvaluator(warmup_cycles=10)
    agent._diagnostic_audit_observer = observer

    agent.run_cycle(1)

    assert observer.proposals == []
    assert observer.summary()["temporal_frontier_evals"] == 0


def test_temporal_frontier_records_proposals_after_warmup() -> None:
    agent = _make_initialized_agent()
    _force_next_cycle_audits(agent)
    observer = TemporalGraphFrontierEvaluator(warmup_cycles=1)
    agent._diagnostic_audit_observer = observer

    agent.run_cycle(1)

    assert observer.proposals
    assert observer.summary()["temporal_frontier_evals"] > 0


def test_temporal_graph_proposals_do_not_change_audit_behavior() -> None:
    base = dict(
        n_vars=5,
        cycles=6,
        seed=42,
        schedule="regime_switch",
        settle_cycles=8,
        noise_sigma=0.02,
        hybrid_control="interfaces",
        repair_agenda_enabled=True,
    )
    agent_off, _ = _build_and_run_dreth(RunConfig(**base))
    agent_on, _ = _build_and_run_dreth(
        RunConfig(
            **base,
            relative_authority_frontier_temporal_report=True,
            relative_authority_frontier_warmup_cycles=1,
        )
    )

    assert _ledger_fingerprint(agent_on) == _ledger_fingerprint(agent_off)
    assert agent_on.skip_count == agent_off.skip_count
    assert agent_on.full_audit_count == agent_off.full_audit_count
    assert agent_on.total_interventions == agent_off.total_interventions


def test_temporal_recall_uses_pre_audit_proposal_not_post_audit_snapshot() -> None:
    source = _node("var:0", 0)
    parent = _node("var:1", 1)
    pre_snapshot = NethraGraphSnapshot(
        nodes=(source, parent),
        relations=(),
        authority_records=(),
    )
    post_snapshot = NethraGraphSnapshot(
        nodes=(source, parent),
        relations=(NethraRelation(source, parent, "depends_on", "ctx"),),
        authority_records=(),
    )
    calls = {"count": 0}

    def _snapshot_builder(_agent):
        calls["count"] += 1
        return pre_snapshot if calls["count"] == 1 else post_snapshot

    agent = SimpleNamespace(
        world=SimpleNamespace(visible_count=2),
        ledger=SimpleNamespace(
            vars={
                0: SimpleNamespace(parents=(1,), certificates={}, route_certs={}),
            },
        ),
    )
    observer = TemporalGraphFrontierEvaluator(
        warmup_cycles=0,
        snapshot_builder=_snapshot_builder,
    )

    token = observer.before_audit(agent, target_var=0, cycle=10)
    observer.after_audit(
        agent,
        token,
        target_var=0,
        cycle=10,
        parents=(1,),
        func="LOW",
        sig_changed=True,
    )

    summary = observer.summary()
    assert summary["temporal_frontier_chosen_parent_total"] == 1
    assert summary["temporal_frontier_chosen_parent_hits"] == 0
    assert summary["temporal_frontier_chosen_parent_recall"] == 0.0


def test_temporal_report_flag_off_preserves_existing_behavior() -> None:
    base = dict(
        n_vars=5,
        cycles=5,
        seed=42,
        schedule="regime_switch",
        settle_cycles=8,
        noise_sigma=0.02,
        hybrid_control="interfaces",
        repair_agenda_enabled=True,
        relative_authority_report=True,
    )
    off = _run_one(RunConfig(**base))
    on = _run_one(
        RunConfig(
            **base,
            relative_authority_frontier_temporal_report=True,
            relative_authority_frontier_warmup_cycles=1,
        )
    )

    assert off.ok
    assert on.ok
    assert off.skip_pct == on.skip_pct
    assert off.interventions == on.interventions
    assert off.full_audits == on.full_audits
    assert off.arch.temporal_frontier_evals == 0
    assert on.arch.temporal_frontier_evals >= 0


def test_agent_does_not_import_frontier_or_relative_authority_modules() -> None:
    agent_source = (ROOT / "dreth" / "agent.py").read_text()

    assert "relative_authority_frontier" not in agent_source
    assert "relative_authority" not in agent_source
