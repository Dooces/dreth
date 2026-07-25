from __future__ import annotations

import json

from .engine import Dreth
from .model import Context, ContextPattern, Factor, Failure


def engine_factorizer(failure: Failure) -> list[Factor]:
    return [
        Factor("fuel", frozenset({"fuel"})),
        Factor("spark", frozenset({"spark"})),
        Factor("air", frozenset({"air"})),
        Factor("compression", frozenset({"compression"})),
        Factor("sensor", frozenset({"sensor"})),
    ]


def run_demo() -> dict:
    dreth = Dreth(factorizer=engine_factorizer)
    engine = dreth.register(
        nethra_id="engine",
        name="engine",
        operation="start",
        scope=ContextPattern.make("start"),
        components={"combustion"},
    )
    warm = Context.make("start", temperature="warm")
    cold = Context.make("start", temperature="cold")

    dreth.commit(
        nethra_id=engine.id,
        context=warm,
        role="tareth",
        expected="runs",
        cycle=0,
        horizon=3,
    )
    dreth.observe(cycle=3, context=warm, observed="runs")

    dreth.commit(
        nethra_id=engine.id,
        context=cold,
        role="tareth",
        expected="runs",
        cycle=4,
        horizon=2,
    )
    outcome = dreth.observe(cycle=6, context=cold, observed="stall")[0]
    boundary = dreth.graph.nethras[outcome.boundary_id]

    return {
        "warm_reuse_h3": dreth.can_reuse("engine", warm, 3),
        "warm_reuse_h4": dreth.can_reuse("engine", warm, 4),
        "cold_reuse_h2": dreth.can_reuse("engine", cold, 2),
        "failure_boundary": boundary.id,
        "factor_children": sorted(
            dreth.graph.nethras[child_id].name for child_id in boundary.children
        ),
        "nethra_count": len(dreth.graph.nethras),
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))
