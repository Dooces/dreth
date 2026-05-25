from __future__ import annotations

import random
from pathlib import Path

from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative
from dreth.relative_authority_observer import build_snapshot_from_agent
from dreth.world import CausalWorld
from scripts.batch_run import RunConfig, _run_one


ROOT = Path(__file__).resolve().parents[1]


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
    return agent, world


def _ledger_fingerprint(agent):
    return {
        var: (
            n.parents,
            n.func,
            n.status,
            tuple(sorted(n.certificates)),
            tuple(sorted(n.route_certs)),
            len(n.dormant_alternatives),
            n.skip_count,
            n.full_audits,
        )
        for var, n in sorted(agent.ledger.vars.items())
    }, len(agent.ledger.event_log)


def test_observer_builds_var_nodes_from_initialized_agent() -> None:
    agent, world = _make_initialized_agent()

    snapshot = build_snapshot_from_agent(agent)
    var_node_ids = {node.node_id for node in snapshot.nodes if node.kind == "nethra_var"}

    assert {f"var:{var}" for var in range(world.visible_count)} <= var_node_ids
    assert snapshot.node_count >= world.visible_count
    assert snapshot.authority_records


def test_depends_on_relations_reflect_ledger_parent_structure() -> None:
    agent, _ = _make_initialized_agent()
    agent.ledger.vars[2].parents = (0,)

    snapshot = build_snapshot_from_agent(agent)
    depends_edges = {
        (relation.source.node_id, relation.target.node_id)
        for relation in snapshot.relations
        if relation.relation_type == "depends_on"
    }

    assert ("var:2", "var:0") in depends_edges
    assert ("var:2", "var:1") not in depends_edges


def test_dormant_alternatives_become_nodes_and_relations() -> None:
    agent, _ = _make_initialized_agent()
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

    assert any(node.kind == "dormant_alternative" for node in snapshot.nodes)
    assert any(
        relation.relation_type == "substitutes_for"
        and relation.target.node_id == "var:2"
        for relation in snapshot.relations
    )
    assert snapshot.local_competitors("var:2")


def test_observer_does_not_mutate_agent_or_ledger() -> None:
    agent, _ = _make_initialized_agent()
    before = _ledger_fingerprint(agent)

    _ = build_snapshot_from_agent(agent)

    assert _ledger_fingerprint(agent) == before


def test_agent_does_not_import_relative_authority_observer() -> None:
    agent_source = (ROOT / "dreth" / "agent.py").read_text()

    assert "relative_authority" not in agent_source
    assert "relative_authority_observer" not in agent_source


def test_relative_authority_report_flag_off_preserves_metrics() -> None:
    base_kwargs = dict(
        n_vars=5,
        cycles=5,
        seed=42,
        schedule="regime_switch",
        settle_cycles=8,
        noise_sigma=0.02,
        hybrid_control="interfaces",
        repair_agenda_enabled=True,
    )
    off = _run_one(RunConfig(**base_kwargs))
    on = _run_one(RunConfig(**base_kwargs, relative_authority_report=True))

    assert off.ok
    assert on.ok
    assert off.skip_pct == on.skip_pct
    assert off.interventions == on.interventions
    assert off.full_audits == on.full_audits
    assert off.arch.relative_authority_nodes == 0
    assert off.arch.relative_authority_relations == 0
    assert off.arch.relative_authority_records == 0
    assert on.arch.relative_authority_nodes > 0
    assert on.arch.relative_authority_records > 0
