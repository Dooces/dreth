from __future__ import annotations

import pytest

from dreth import Context, ContextPattern, Dreth, Factor, UseState


def register_engine(dreth: Dreth, nethra_id: str = "engine"):
    return dreth.register(
        nethra_id=nethra_id,
        name=nethra_id,
        operation="start",
        scope=ContextPattern.make("start"),
        components={"combustion"},
    )


def test_authority_requires_a_prospective_prediction() -> None:
    dreth = Dreth()
    register_engine(dreth)
    context = Context.make("start", temperature="warm")

    assert dreth.can_reuse("engine", context, 1) is False
    with pytest.raises(ValueError, match="prospective"):
        dreth.commit(
            nethra_id="engine",
            context=context,
            role="tareth",
            expected="runs",
            cycle=2,
            horizon=0,
        )


def test_success_earns_exact_context_and_horizon_authority() -> None:
    dreth = Dreth()
    register_engine(dreth)
    warm = Context.make("start", temperature="warm")
    cold = Context.make("start", temperature="cold")

    dreth.commit(
        nethra_id="engine",
        context=warm,
        role="tareth",
        expected="runs",
        cycle=4,
        horizon=5,
    )
    dreth.observe(cycle=9, context=warm, observed="runs")

    assert dreth.can_reuse("engine", warm, 5)
    assert dreth.can_reuse("engine", warm, 6) is False
    assert dreth.can_reuse("engine", cold, 1) is False


def test_unexposed_prediction_earns_nothing() -> None:
    dreth = Dreth()
    register_engine(dreth)
    warm = Context.make("start", temperature="warm")

    dreth.commit(
        nethra_id="engine",
        context=warm,
        role="tareth",
        expected="runs",
        cycle=0,
        horizon=2,
    )
    expired = dreth.expire_unexposed(through_cycle=2)

    assert len(expired) == 1
    assert dreth.can_reuse("engine", warm, 1) is False


def test_failure_is_local_and_factorization_is_lazy() -> None:
    calls = []

    def factorizer(failure):
        calls.append(failure)
        return [
            Factor("fuel", frozenset({"fuel"})),
            Factor("spark", frozenset({"spark"})),
        ]

    dreth = Dreth(factorizer=factorizer)
    engine = register_engine(dreth)
    warm = Context.make("start", temperature="warm")
    cold = Context.make("start", temperature="cold")

    assert engine.children == set()
    dreth.commit(
        nethra_id="engine",
        context=warm,
        role="tareth",
        expected="runs",
        cycle=0,
        horizon=3,
    )
    dreth.observe(cycle=3, context=warm, observed="runs")
    assert engine.children == set()
    assert calls == []

    dreth.commit(
        nethra_id="engine",
        context=cold,
        role="tareth",
        expected="runs",
        cycle=4,
        horizon=2,
    )
    outcome = dreth.observe(cycle=6, context=cold, observed="stall")[0]
    boundary = dreth.graph.nethras[outcome.boundary_id]

    assert len(calls) == 1
    assert boundary.id in engine.children
    assert {dreth.graph.nethras[item].name for item in boundary.children} == {
        "fuel",
        "spark",
    }
    assert dreth.can_reuse("engine", warm, 3)
    assert dreth.can_reuse("engine", cold, 1) is False


def test_failure_at_long_horizon_preserves_shorter_horizon() -> None:
    dreth = Dreth()
    register_engine(dreth)
    context = Context.make("start", temperature="warm")

    dreth.commit(
        nethra_id="engine",
        context=context,
        role="tareth",
        expected="runs",
        cycle=0,
        horizon=5,
    )
    dreth.observe(cycle=5, context=context, observed="runs")
    dreth.commit(
        nethra_id="engine",
        context=context,
        role="tareth",
        expected="runs",
        cycle=6,
        horizon=4,
    )
    dreth.observe(cycle=10, context=context, observed="stall")

    authority = dreth.graph.authority("engine", "tareth")
    assert authority.usable_horizon(context) == 3
    assert authority.state_at(context, 3) is UseState.USABLE
    assert authority.state_at(context, 4) is UseState.UNUSABLE


def test_joint_failure_creates_relation_without_demoting_members() -> None:
    dreth = Dreth()
    fuel = dreth.register(
        nethra_id="fuel",
        name="fuel",
        operation="burn",
        scope=ContextPattern.make("burn"),
        components={"combustion", "fuel"},
    )
    spark = dreth.register(
        nethra_id="spark",
        name="spark",
        operation="burn",
        scope=ContextPattern.make("burn"),
        components={"combustion", "spark"},
    )
    context = Context.make("burn", temperature="cold")

    for nethra in (fuel, spark):
        dreth.commit(
            nethra_id=nethra.id,
            context=context,
            role="tareth",
            expected="ignition",
            cycle=0,
            horizon=1,
        )
    dreth.observe(cycle=1, context=context, observed="ignition")

    dreth.commit(
        nethra_id=fuel.id,
        implicated_ids=(fuel.id, spark.id),
        context=context,
        role="tareth",
        expected="ignition",
        cycle=2,
        horizon=1,
    )
    outcome = dreth.observe(cycle=3, context=context, observed="misfire")[0]
    composite = dreth.graph.nethras[outcome.boundary_id]

    assert composite.kind == "composite"
    assert composite.parents == {"fuel", "spark"}
    assert dreth.can_reuse("fuel", context, 1)
    assert dreth.can_reuse("spark", context, 1)
    assert dreth.can_reuse(composite.id, context, 1) is False


def test_common_nodes_retrieve_related_nethras_as_hints() -> None:
    dreth = Dreth()
    register_engine(dreth, "engine")
    dreth.register(
        nethra_id="combustion-repair",
        name="combustion repair",
        operation="repair",
        scope=ContextPattern.make("repair"),
        components={"combustion"},
    )
    context = Context.make("start", temperature="warm")

    considered = dreth.graph.consider(context, horizon=1, components={"combustion"})
    rows = {row.nethra_id: row for row in considered}

    assert rows["engine"].match == "local"
    assert rows["combustion-repair"].match == "shared"
    assert rows["combustion-repair"].state is UseState.UNUSABLE


def test_trass_authority_controls_collapse() -> None:
    dreth = Dreth()
    register_engine(dreth)
    context = Context.make("start", temperature="warm")

    dreth.commit(
        nethra_id="engine",
        context=context,
        role="trass",
        expected="unchanged",
        cycle=0,
        horizon=2,
    )
    dreth.observe(cycle=2, context=context, observed="unchanged")

    assert dreth.graph.can_collapse("engine", context, 2)
    assert dreth.graph.can_collapse("engine", context, 3) is False


def test_repeated_boundaries_can_be_consolidated_as_a_higher_nethra() -> None:
    dreth = Dreth()
    register_engine(dreth)
    cold = Context.make("start", temperature="cold")
    wet = Context.make("start", weather="wet")

    boundary_ids = []
    for cycle, context in ((0, cold), (2, wet)):
        dreth.commit(
            nethra_id="engine",
            context=context,
            role="tareth",
            expected="runs",
            cycle=cycle,
            horizon=1,
        )
        outcome = dreth.observe(cycle=cycle + 1, context=context, observed="stall")[0]
        boundary_ids.append(outcome.boundary_id)

    higher = dreth.consolidate(
        name="environmental start failures",
        boundary_ids=boundary_ids,
    )

    assert higher.kind == "consolidation"
    assert higher.parents == set(boundary_ids)
    assert all(higher.id in dreth.graph.nethras[item].children for item in boundary_ids)
    assert dreth.graph.authorities_for(higher.id) == []
