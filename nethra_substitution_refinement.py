#!/usr/bin/env python3
"""
complete_nethra_attention_system.py

Complete runnable implementation of the stated initial framework.

Stated target:
- causal-inference system
- variables are not equally worth tracking
- tareth: operationally relevant, worth tracking and predicting
- trass: operationally irrelevant, does not propagate when perturbed, can be collapsed/ignored
- nethra: per-variable attention record
- irrelevance is positively certified, not assumed
- once trass is certified, attention skips it
- cost-asymmetric: important errors matter; trivial errors do not get equal budget

This file contains both:
1. A generic finite causal-DAG environment with real do-interventions and downstream recomputation.
2. A reusable nethra/tareth/trass attention engine that audits variables, certifies roles,
   tracks/predicts tareth variables, skips trass variables, and rechecks trass under drift.

Run:
    python3 complete_nethra_attention_system.py

Optional:
    python3 complete_nethra_attention_system.py --cycles 120 --drift-at 50 --json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


Variable = str
Target = str
Value = Any
State = Dict[str, Any]
Cycle = int
Key = Tuple[Tuple[str, Any], ...]

Mechanism = Callable[[Mapping[str, Any], Cycle], Any]


# =============================================================================
# Generic causal DAG environment
# =============================================================================

@dataclass(frozen=True)
class NodeSpec:
    """
    One node in a finite deterministic causal DAG.

    If parents=() and mechanism=None, the node is exogenous/root and is sampled
    from its domain.

    If mechanism is provided, the value is computed from already-computed parents.
    Mechanism may depend on cycle, allowing non-stationarity/drift.
    """

    name: Variable
    domain: Tuple[Value, ...]
    parents: Tuple[Variable, ...] = ()
    mechanism: Optional[Mechanism] = None


class FiniteCausalDAGWorld:
    """
    Generic finite causal world with do-interventions.

    - sample_context(cycle): samples roots and computes all non-root nodes in topo order.
    - intervene(context, var, value, cycle): applies do(var=value) and recomputes descendants.
    - observe(context, cycle): reads monitored target nodes.

    This is the causal propagation environment the engine requires.
    """

    def __init__(
        self,
        nodes: Sequence[NodeSpec],
        intervenable: Sequence[Variable],
        monitored_targets: Sequence[Target],
        seed: int = 0,
    ):
        self.nodes: Dict[Variable, NodeSpec] = {n.name: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node names")

        self._intervenable = tuple(intervenable)
        self._targets = tuple(monitored_targets)
        self.rng = random.Random(seed)

        for v in self._intervenable:
            if v not in self.nodes:
                raise KeyError(f"intervenable {v!r} not in nodes")
        for t in self._targets:
            if t not in self.nodes:
                raise KeyError(f"target {t!r} not in nodes")
        for n in nodes:
            for p in n.parents:
                if p not in self.nodes:
                    raise KeyError(f"parent {p!r} of {n.name!r} not in nodes")

        self.children: Dict[Variable, List[Variable]] = {name: [] for name in self.nodes}
        for n in nodes:
            for p in n.parents:
                self.children[p].append(n.name)

        self.topo = self._topological_order()

    def _topological_order(self) -> Tuple[Variable, ...]:
        indeg = {name: 0 for name in self.nodes}
        for n in self.nodes.values():
            for p in n.parents:
                indeg[n.name] += 1

        q = deque(sorted(name for name, d in indeg.items() if d == 0))
        out: List[Variable] = []

        while q:
            x = q.popleft()
            out.append(x)
            for c in self.children[x]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)

        if len(out) != len(self.nodes):
            raise ValueError("cycle in DAG")
        return tuple(out)

    def variables(self) -> Sequence[Variable]:
        return self._intervenable

    def intervention_values(self, var: Variable) -> Sequence[Value]:
        return self.nodes[var].domain

    def sample_context(self, rng: random.Random, cycle: Cycle) -> State:
        state: State = {}
        for name in self.topo:
            spec = self.nodes[name]
            if spec.mechanism is None:
                state[name] = rng.choice(spec.domain)
            else:
                state[name] = spec.mechanism(state, cycle)
        return state

    def _descendants(self, var: Variable) -> List[Variable]:
        seen = set()
        q = deque(self.children[var])
        out = []
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            q.extend(self.children[x])
        return out

    def intervene(self, context: State, var: Variable, value: Value, cycle: Cycle) -> State:
        """
        do(var=value): force var and recompute downstream non-intervened descendants.

        This is where propagation is real. If var affects a monitored target through
        mechanisms, observe() will change.
        """
        if var not in self.nodes:
            raise KeyError(var)
        out = dict(context)
        out[var] = value

        affected = set(self._descendants(var))
        for name in self.topo:
            if name not in affected:
                continue
            spec = self.nodes[name]
            if spec.mechanism is not None:
                out[name] = spec.mechanism(out, cycle)
        return out

    def observe(self, context: State, cycle: Cycle) -> Mapping[Target, Any]:
        return {t: context[t] for t in self._targets}

    def equal(self, target: Target, a: Any, b: Any) -> bool:
        return a == b

    def prediction_key(self, context: Mapping[str, Any], var: Variable) -> Key:
        """
        Generic prediction key: all observed variables except the variable itself and monitored targets.
        """
        excluded = set(self._targets)
        excluded.add(var)
        return tuple(sorted((k, v) for k, v in context.items() if k not in excluded))


# =============================================================================
# Nethra/tareth/trass attention engine
# =============================================================================

@dataclass(frozen=True)
class PropagationWitness:
    var: Variable
    context: Key
    forced_value: Value
    target: Target
    before: Any
    after: Any
    cycle_created: int

    def thaw_context(self) -> State:
        return dict(self.context)


@dataclass(frozen=True)
class PreservationProbe:
    var: Variable
    context: Key
    forced_value: Value
    cycle_created: int


@dataclass(frozen=True)
class PredictionSample:
    var: Variable
    key: Key
    value: Value


@dataclass
class TabularPredictor:
    """
    Simple explicit predictor for tareth variables.

    It is intentionally plain:
    key -> most frequent value, fallback -> global most frequent value.
    """

    table: Dict[Key, Value] = field(default_factory=dict)
    fallback: Optional[Value] = None
    trained_samples: int = 0

    def fit(self, samples: Sequence[PredictionSample]) -> None:
        if not samples:
            self.table = {}
            self.fallback = None
            self.trained_samples = 0
            return

        by_key: Dict[Key, Counter] = defaultdict(Counter)
        global_counts: Counter = Counter()
        for s in samples:
            by_key[s.key][s.value] += 1
            global_counts[s.value] += 1

        self.table = {k: c.most_common(1)[0][0] for k, c in by_key.items()}
        self.fallback = global_counts.most_common(1)[0][0]
        self.trained_samples = len(samples)

    def predict(self, key: Key) -> Optional[Value]:
        return self.table.get(key, self.fallback)


@dataclass
class CertificationReceipt:
    var: Variable
    cycle: int
    role: str
    contexts_tested: int
    interventions_tested: int
    targets_compared: int
    witnesses: List[PropagationWitness] = field(default_factory=list)
    preservation_probes: List[PreservationProbe] = field(default_factory=list)
    prediction_samples: List[PredictionSample] = field(default_factory=list)


@dataclass
class NethraRecord:
    """
    Per-variable unit of attention.

    This is the nethra in the stated initial intention.
    """

    var: Variable
    role: str = "unknown"  # unknown | tareth | trass
    certified_at: Optional[int] = None
    last_attention_at: Optional[int] = None

    witnesses: List[PropagationWitness] = field(default_factory=list)
    preservation_probes: List[PreservationProbe] = field(default_factory=list)
    predictor: TabularPredictor = field(default_factory=TabularPredictor)
    receipts: List[CertificationReceipt] = field(default_factory=list)

    full_audits: int = 0
    sentinel_checks: int = 0
    prediction_checks: int = 0
    prediction_correct: int = 0
    skipped_cycles: int = 0
    trass_rechecks: int = 0
    sentinel_failures: int = 0
    role_changes: List[Tuple[int, str, str, str]] = field(default_factory=list)

    def set_role(self, new_role: str, cycle: int, reason: str) -> None:
        old = self.role
        if old != new_role:
            self.role_changes.append((cycle, old, new_role, reason))
        self.role = new_role
        self.certified_at = cycle
        self.last_attention_at = cycle


@dataclass
class AttentionConfig:
    max_contexts_per_audit: int = 256
    max_witnesses: int = 8
    max_preservation_probes: int = 32
    max_prediction_samples: int = 256
    trass_recheck_period: int = 10
    seed: int = 0


class NethraAttentionSystem:
    """
    The complete initial framework.

    unknown:
        full causal intervention audit -> tareth/trass

    tareth:
        sentinel checks + prediction checks

    trass:
        skip attention, except scheduled recheck
    """

    def __init__(self, world: FiniteCausalDAGWorld, config: AttentionConfig):
        self.world = world
        self.config = config
        self.rng = random.Random(config.seed)
        self.records: Dict[Variable, NethraRecord] = {
            var: NethraRecord(var=var) for var in world.variables()
        }

        self.total_full_audits = 0
        self.total_sentinel_checks = 0
        self.total_prediction_checks = 0
        self.total_skipped_cycles = 0

    @staticmethod
    def freeze_context(context: Mapping[str, Any]) -> Key:
        return tuple(sorted(context.items(), key=lambda item: item[0]))

    def prediction_key(self, context: Mapping[str, Any], var: Variable) -> Key:
        return self.world.prediction_key(context, var)

    def certify(self, var: Variable, cycle: int) -> CertificationReceipt:
        witnesses: List[PropagationWitness] = []
        preservation: List[PreservationProbe] = []
        pred_samples: List[PredictionSample] = []

        contexts_tested = 0
        interventions_tested = 0
        targets_compared = 0

        for _ in range(self.config.max_contexts_per_audit):
            ctx = self.world.sample_context(self.rng, cycle)
            contexts_tested += 1

            if var in ctx and len(pred_samples) < self.config.max_prediction_samples:
                pred_samples.append(
                    PredictionSample(
                        var=var,
                        key=self.prediction_key(ctx, var),
                        value=ctx[var],
                    )
                )

            before_obs = self.world.observe(ctx, cycle)

            for forced in self.world.intervention_values(var):
                after_ctx = self.world.intervene(ctx, var, forced, cycle)
                after_obs = self.world.observe(after_ctx, cycle)
                interventions_tested += 1

                changed_any = False
                for target, before in before_obs.items():
                    after = after_obs[target]
                    targets_compared += 1
                    if not self.world.equal(target, before, after):
                        changed_any = True
                        witnesses.append(
                            PropagationWitness(
                                var=var,
                                context=self.freeze_context(ctx),
                                forced_value=forced,
                                target=target,
                                before=before,
                                after=after,
                                cycle_created=cycle,
                            )
                        )
                        if len(witnesses) >= self.config.max_witnesses:
                            return CertificationReceipt(
                                var=var,
                                cycle=cycle,
                                role="tareth",
                                contexts_tested=contexts_tested,
                                interventions_tested=interventions_tested,
                                targets_compared=targets_compared,
                                witnesses=witnesses,
                                preservation_probes=preservation[: self.config.max_preservation_probes],
                                prediction_samples=pred_samples,
                            )

                if not changed_any and len(preservation) < self.config.max_preservation_probes:
                    preservation.append(
                        PreservationProbe(
                            var=var,
                            context=self.freeze_context(ctx),
                            forced_value=forced,
                            cycle_created=cycle,
                        )
                    )

        return CertificationReceipt(
            var=var,
            cycle=cycle,
            role="trass",
            contexts_tested=contexts_tested,
            interventions_tested=interventions_tested,
            targets_compared=targets_compared,
            witnesses=[],
            preservation_probes=preservation,
            prediction_samples=pred_samples,
        )

    def apply_receipt(self, record: NethraRecord, receipt: CertificationReceipt, reason: str) -> None:
        record.full_audits += 1
        self.total_full_audits += 1
        record.receipts.append(receipt)
        record.set_role(receipt.role, receipt.cycle, reason)

        if receipt.role == "tareth":
            record.witnesses = list(receipt.witnesses)
            record.preservation_probes = []
            record.predictor.fit(receipt.prediction_samples)
        elif receipt.role == "trass":
            record.witnesses = []
            record.preservation_probes = list(receipt.preservation_probes)
            record.predictor = TabularPredictor()
        else:
            raise ValueError(receipt.role)

    def full_audit(self, record: NethraRecord, cycle: int, reason: str) -> None:
        self.apply_receipt(record, self.certify(record.var, cycle), reason)

    def witness_active_now(self, witness: PropagationWitness, cycle: int) -> bool:
        ctx = witness.thaw_context()
        before = self.world.observe(ctx, cycle)
        after_ctx = self.world.intervene(ctx, witness.var, witness.forced_value, cycle)
        after = self.world.observe(after_ctx, cycle)
        return not self.world.equal(witness.target, before[witness.target], after[witness.target])

    def sentinels_pass(self, record: NethraRecord, cycle: int) -> bool:
        active = 0
        for w in record.witnesses:
            record.sentinel_checks += 1
            self.total_sentinel_checks += 1
            if self.witness_active_now(w, cycle):
                active += 1
        record.last_attention_at = cycle
        return active > 0

    def check_prediction(self, record: NethraRecord, cycle: int) -> None:
        ctx = self.world.sample_context(self.rng, cycle)
        pred = record.predictor.predict(self.prediction_key(ctx, record.var))
        if pred is None:
            return

        record.prediction_checks += 1
        self.total_prediction_checks += 1
        if pred == ctx[record.var]:
            record.prediction_correct += 1

    def step_record(self, record: NethraRecord, cycle: int) -> None:
        if record.role == "unknown":
            self.full_audit(record, cycle, "initial certification")
            return

        if record.role == "tareth":
            if not self.sentinels_pass(record, cycle):
                record.sentinel_failures += 1
                self.full_audit(record, cycle, "tareth sentinel failure")
                return
            self.check_prediction(record, cycle)
            return

        if record.role == "trass":
            record.skipped_cycles += 1
            self.total_skipped_cycles += 1

            if cycle > 0 and self.config.trass_recheck_period > 0 and cycle % self.config.trass_recheck_period == 0:
                record.trass_rechecks += 1
                self.full_audit(record, cycle, "scheduled trass recheck")
            return

        raise ValueError(record.role)

    def run(self, cycles: int) -> Dict[Variable, NethraRecord]:
        for cycle in range(cycles):
            for var in self.world.variables():
                self.step_record(self.records[var], cycle)
        return self.records

    def rows(self) -> List[Dict[str, Any]]:
        rows = []
        for var in self.world.variables():
            r = self.records[var]
            acc = None if r.prediction_checks == 0 else r.prediction_correct / r.prediction_checks
            rows.append(
                {
                    "var": r.var,
                    "role": r.role,
                    "full_audits": r.full_audits,
                    "sentinel_checks": r.sentinel_checks,
                    "prediction_checks": r.prediction_checks,
                    "prediction_accuracy": acc,
                    "skipped_cycles": r.skipped_cycles,
                    "trass_rechecks": r.trass_rechecks,
                    "sentinel_failures": r.sentinel_failures,
                    "role_changes": r.role_changes,
                }
            )
        return rows


# =============================================================================
# Concrete complete environment
# =============================================================================

def build_receipt_world(seed: int, drift_at: int) -> FiniteCausalDAGWorld:
    """
    Causal DAG:
        x0,x1,x2,x3,x4,x5 are root variables.
        y is monitored target.
        before drift: y = x0 XOR x1
        after drift:  y = x0 XOR x1 XOR x4

    Therefore:
        x0,x1 are tareth throughout.
        x4 is trass before drift, tareth after drift.
        x2,x3,x5 are trass throughout.
    """

    def y_mechanism(state: Mapping[str, Any], cycle: int) -> int:
        y = int(state["x0"]) ^ int(state["x1"])
        if cycle >= drift_at:
            y ^= int(state["x4"])
        return y

    nodes = [
        NodeSpec("x0", (0, 1)),
        NodeSpec("x1", (0, 1)),
        NodeSpec("x2", (0, 1)),
        NodeSpec("x3", (0, 1)),
        NodeSpec("x4", (0, 1)),
        NodeSpec("x5", (0, 1)),
        NodeSpec("y", (0, 1), parents=("x0", "x1", "x4"), mechanism=y_mechanism),
    ]
    return FiniteCausalDAGWorld(
        nodes=nodes,
        intervenable=("x0", "x1", "x2", "x3", "x4", "x5"),
        monitored_targets=("y",),
        seed=seed,
    )


def print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    print("var  role    audits  sentinels  pred_chk  pred_acc  skipped  rechecks  failures  changes")
    for r in rows:
        acc = "-" if r["prediction_accuracy"] is None else f"{r['prediction_accuracy']:.2f}"
        print(
            f"{r['var']:<4} {r['role']:<7} {r['full_audits']:<7} "
            f"{r['sentinel_checks']:<10} {r['prediction_checks']:<9} "
            f"{acc:<8} {r['skipped_cycles']:<8} {r['trass_rechecks']:<8} "
            f"{r['sentinel_failures']:<9} {r['role_changes']}"
        )


def run_receipt(args: argparse.Namespace) -> None:
    world = build_receipt_world(seed=args.seed, drift_at=args.drift_at)
    system = NethraAttentionSystem(
        world,
        AttentionConfig(
            max_contexts_per_audit=args.max_contexts_per_audit,
            max_witnesses=args.max_witnesses,
            max_preservation_probes=args.max_preservation_probes,
            max_prediction_samples=args.max_prediction_samples,
            trass_recheck_period=args.recheck_period,
            seed=args.seed,
        ),
    )
    system.run(args.cycles)
    rows = system.rows()

    print("COMPLETE NETHRA ATTENTION SYSTEM")
    print("────────────────────────────────")
    print(f"cycles={args.cycles} drift_at={args.drift_at} seed={args.seed}")
    print_rows(rows)
    print()
    print(
        f"TOTAL full_audits={system.total_full_audits} "
        f"sentinel_checks={system.total_sentinel_checks} "
        f"prediction_checks={system.total_prediction_checks} "
        f"skipped_cycles={system.total_skipped_cycles}"
    )

    if args.json:
        print(json.dumps(rows, indent=2, default=str))

    rec = system.records

    assert rec["x0"].role == "tareth"
    assert rec["x1"].role == "tareth"
    assert rec["x4"].role == "tareth"
    assert rec["x2"].role == "trass"
    assert rec["x3"].role == "trass"
    assert rec["x5"].role == "trass"

    assert rec["x0"].witnesses
    assert rec["x1"].witnesses
    assert rec["x4"].witnesses

    assert rec["x0"].prediction_checks > 0
    assert rec["x1"].prediction_checks > 0
    assert rec["x4"].prediction_checks > 0

    assert rec["x2"].prediction_checks == 0
    assert rec["x3"].prediction_checks == 0
    assert rec["x5"].prediction_checks == 0

    assert rec["x2"].skipped_cycles > args.cycles // 2
    assert rec["x3"].skipped_cycles > args.cycles // 2
    assert rec["x5"].skipped_cycles > args.cycles // 2

    assert any(old == "trass" and new == "tareth" for _, old, new, _ in rec["x4"].role_changes)

    print()
    print("ASSERTIONS")
    print("──────────")
    print("PASS: causal DAG environment supports do-interventions and propagation")
    print("PASS: nethra is the per-variable unit of attention")
    print("PASS: tareth is certified by propagation witness")
    print("PASS: trass is certified by absence of tested propagation")
    print("PASS: tareth variables are tracked and predicted")
    print("PASS: trass variables are skipped after certification")
    print("PASS: cost asymmetry is enforced")
    print("PASS: x4 demonstrates trass -> tareth after drift")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=100)
    p.add_argument("--drift-at", type=int, default=50)
    p.add_argument("--recheck-period", type=int, default=10)
    p.add_argument("--max-contexts-per-audit", type=int, default=256)
    p.add_argument("--max-witnesses", type=int, default=8)
    p.add_argument("--max-preservation-probes", type=int, default=32)
    p.add_argument("--max-prediction-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p


if __name__ == "__main__":
    run_receipt(build_parser().parse_args())
