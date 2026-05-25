from __future__ import annotations

import random
from pathlib import Path

from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative
from dreth.relative_authority import NethraGraphSnapshot, NethraNodeRef, NethraRelation
from dreth.relative_authority_frontier import (
    evaluate_frontier_against_agent,
    propose_frontier,
)
from dreth.relative_authority_observer import build_snapshot_from_agent
from dreth.world import CausalWorld


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


def test_evaluator_does_not_mutate_agent_or_ledger() -> None:
    agent = _make_initialized_agent()
    agent.ledger.vars[2].parents = (0,)
    snapshot = build_snapshot_from_agent(agent)
    before = _ledger_fingerprint(agent)

    _ = evaluate_frontier_against_agent(snapshot, agent)

    assert _ledger_fingerprint(agent) == before


def test_agent_does_not_import_frontier_or_relative_authority_modules() -> None:
    agent_source = (ROOT / "dreth" / "agent.py").read_text()

    assert "relative_authority_frontier" not in agent_source
    assert "relative_authority" not in agent_source
