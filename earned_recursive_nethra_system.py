#!/usr/bin/env python3
"""
earned_recursive_nethra_system.py

Finite-domain implementation of the corrected nethra / tareth / trass object.

This file fixes the defects in the previous attempts:

1. Nethra is not a flat per-variable label.
   A nethra is a scoped, evidence-carrying operative factoring:
       collapse_vars + scope + certificate + optional child nethras.

2. Compositions are earned.
   Composed nethras are generated only from already-certified trass nethras.
   They are not manually supplied as parent specs.

3. Composition is re-tested.
   A composed nethra jointly substitutes all collapsed variables and certifies
   whether the combined collapse is trass or tareth.

4. False-trass is detected.
   If children are locally trass but the composed collapse is tareth, the engine
   records a concrete false-trass witness.

5. Sentinel checks are real.
   Tareth witnesses are re-run as interventions against the current world.

6. Scope is real.
   A candidate with zero in-scope contexts is invalid, not trass.

7. Prediction keys do not leak monitored targets.
   Prediction keys exclude collapsed variables and monitored target nodes.

8. Lifecycle exists.
   Trass records are periodically re-certified; tareth records are sentinel-checked;
   compositions are rebuilt after role changes.

This is still a finite-domain implementation. That is deliberate: exhaustive
substitution certification is only exact over finite/enumerable domains.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Name = str
Variable = str
Target = str
Value = Any
Cycle = int
State = Dict[str, Any]
Key = Tuple[Tuple[str, Any], ...]

Mechanism = Callable[[Mapping[str, Any], Cycle], Any]
ScopeFn = Callable[[Mapping[str, Any], Cycle], bool]


# =============================================================================
# Finite causal DAG with real interventions and recomputation
# =============================================================================

@dataclass(frozen=True)
class NodeSpec:
    name: Variable
    domain: Tuple[Value, ...]
    parents: Tuple[Variable, ...] = ()
    mechanism: Optional[Mechanism] = None


class FiniteCausalDAGWorld:
    def __init__(
        self,
        nodes: Sequence[NodeSpec],
        intervenable: Sequence[Variable],
        monitored_targets: Sequence[Target],
        seed: int = 0,
    ):
        self.nodes = {n.name: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node names")

        self._intervenable = tuple(intervenable)
        self._targets = tuple(monitored_targets)
        self.rng = random.Random(seed)

        for v in self._intervenable:
            if v not in self.nodes:
                raise KeyError(f"intervenable {v!r} missing")
        for t in self._targets:
            if t not in self.nodes:
                raise KeyError(f"target {t!r} missing")

        self.children: Dict[Variable, List[Variable]] = {name: [] for name in self.nodes}
        for n in nodes:
            for p in n.parents:
                if p not in self.nodes:
                    raise KeyError(f"parent {p!r} missing for {n.name!r}")
                self.children[p].append(n.name)

        self.topo = self._topological_order()
        self.roots = tuple(name for name in self.topo if self.nodes[name].mechanism is None)

    def _topological_order(self) -> Tuple[Variable, ...]:
        indeg = {name: 0 for name in self.nodes}
        for n in self.nodes.values():
            for _ in n.parents:
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

    def variables(self) -> Tuple[Variable, ...]:
        return self._intervenable

    def monitored_targets(self) -> Tuple[Target, ...]:
        return self._targets

    def intervention_values(self, var: Variable) -> Tuple[Value, ...]:
        return self.nodes[var].domain

    def all_root_contexts(self) -> Iterable[State]:
        domains = [self.nodes[r].domain for r in self.roots]
        for values in product(*domains):
            yield dict(zip(self.roots, values))

    def complete_context(self, root_context: Mapping[str, Any], cycle: Cycle) -> State:
        state: State = dict(root_context)
        for name in self.topo:
            spec = self.nodes[name]
            if spec.mechanism is None:
                if name not in state:
                    state[name] = self.rng.choice(spec.domain)
            else:
                state[name] = spec.mechanism(state, cycle)
        return state

    def normalize_context(self, context: Mapping[str, Any], cycle: Cycle) -> State:
        roots = {r: context[r] for r in self.roots}
        return self.complete_context(roots, cycle)

    def descendants(self, var: Variable) -> List[Variable]:
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

    def intervene_many(self, context: Mapping[str, Any], assignments: Mapping[Variable, Value], cycle: Cycle) -> State:
        """
        do(assignments): force variables, then recompute descendants in topological order.
        """
        out = self.normalize_context(context, cycle)

        affected = set()
        for var, value in assignments.items():
            out[var] = value
            affected.update(self.descendants(var))

        forced = set(assignments)
        for name in self.topo:
            if name in forced:
                continue
            if name not in affected:
                continue
            spec = self.nodes[name]
            if spec.mechanism is not None:
                out[name] = spec.mechanism(out, cycle)

        return out

    def observe(self, context: Mapping[str, Any], cycle: Cycle) -> Dict[Target, Any]:
        ctx = self.normalize_context(context, cycle)
        return {t: ctx[t] for t in self._targets}

    def equal(self, target: Target, a: Any, b: Any) -> bool:
        return a == b


# =============================================================================
# Nethra machinery
# =============================================================================

def always_scope(_: Mapping[str, Any], __: Cycle) -> bool:
    return True


@dataclass(frozen=True)
class Scope:
    name: str
    fn: ScopeFn = always_scope


@dataclass(frozen=True)
class NethraSpec:
    """
    Candidate operative factoring.

    collapse_vars:
        distinctions proposed for collapse.

    scope:
        operation/context scope in which the collapse is certified.

    children:
        names of child nethras if this is an earned composition.
    """

    name: Name
    collapse_vars: Tuple[Variable, ...]
    scope: Scope
    children: Tuple[Name, ...] = ()
    generated: bool = False


@dataclass(frozen=True)
class SubstitutionWitness:
    nethra: Name
    context: Key
    assignments: Tuple[Tuple[Variable, Value], ...]
    target: Target
    before: Any
    after: Any
    cycle: Cycle

    def context_dict(self) -> State:
        return dict(self.context)

    def assignment_dict(self) -> Dict[Variable, Value]:
        return dict(self.assignments)


@dataclass(frozen=True)
class PreservationProbe:
    nethra: Name
    context: Key
    assignments: Tuple[Tuple[Variable, Value], ...]
    cycle: Cycle


@dataclass
class Certificate:
    nethra: Name
    role: str  # trass | tareth | invalid
    cycle: Cycle
    scope_hits: int
    substitutions_tested: int
    targets_compared: int
    witnesses: List[SubstitutionWitness] = field(default_factory=list)
    preservation: List[PreservationProbe] = field(default_factory=list)
    invalid_reason: Optional[str] = None


@dataclass
class Predictor:
    table: Dict[Key, Any] = field(default_factory=dict)
    fallback: Optional[Any] = None
    trained_samples: int = 0

    def fit(self, samples: Sequence[Tuple[Key, Any]]) -> None:
        if not samples:
            self.table = {}
            self.fallback = None
            self.trained_samples = 0
            return

        by_key: Dict[Key, Counter] = defaultdict(Counter)
        global_counts: Counter = Counter()
        for key, target_value in samples:
            by_key[key][target_value] += 1
            global_counts[target_value] += 1

        self.table = {k: c.most_common(1)[0][0] for k, c in by_key.items()}
        self.fallback = global_counts.most_common(1)[0][0]
        self.trained_samples = len(samples)

    def predict(self, key: Key) -> Optional[Any]:
        return self.table.get(key, self.fallback)


@dataclass
class NethraRecord:
    spec: NethraSpec
    role: str = "unknown"
    certificate: Optional[Certificate] = None
    predictor: Predictor = field(default_factory=Predictor)

    audits: int = 0
    sentinel_checks: int = 0
    sentinel_failures: int = 0
    prediction_checks: int = 0
    prediction_correct: int = 0
    skipped_cycles: int = 0
    gated_child_skips: int = 0
    role_changes: List[Tuple[int, str, str, str]] = field(default_factory=list)

    @property
    def name(self) -> Name:
        return self.spec.name

    def set_role(self, new_role: str, cycle: Cycle, reason: str) -> None:
        old_role = self.role
        if old_role != new_role:
            self.role_changes.append((cycle, old_role, new_role, reason))
        self.role = new_role


@dataclass(frozen=True)
class FalseTrass:
    composed: Name
    children: Tuple[Name, ...]
    witness: SubstitutionWitness


@dataclass
class EngineConfig:
    max_witnesses: int = 8
    max_preservation: int = 32
    max_composition_size: int = 2
    recheck_period: int = 5
    seed: int = 0


class EarnedRecursiveNethraEngine:
    """
    Engine that certifies base nethras, generates composed nethras from trass
    children, jointly re-tests those compositions, and records false-trass.
    """

    def __init__(
        self,
        world: FiniteCausalDAGWorld,
        base_specs: Sequence[NethraSpec],
        composition_scope: Scope,
        config: EngineConfig,
    ):
        self.world = world
        self.base_specs = {s.name: s for s in base_specs}
        if len(self.base_specs) != len(base_specs):
            raise ValueError("duplicate base nethra names")

        self.composition_scope = composition_scope
        self.config = config
        self.rng = random.Random(config.seed)

        self.records: Dict[Name, NethraRecord] = {
            s.name: NethraRecord(spec=s) for s in base_specs
        }
        self.false_trass: List[FalseTrass] = []

        self.total_audits = 0
        self.total_sentinel_checks = 0
        self.total_prediction_checks = 0
        self.total_skips = 0

    @staticmethod
    def freeze_context(context: Mapping[str, Any]) -> Key:
        return tuple(sorted(context.items(), key=lambda kv: kv[0]))

    def target_tuple(self, context: Mapping[str, Any], cycle: Cycle) -> Tuple[Tuple[Target, Any], ...]:
        return tuple(sorted(self.world.observe(context, cycle).items()))

    def prediction_key(self, context: Mapping[str, Any], spec: NethraSpec) -> Key:
        """
        No target leakage: keys exclude monitored targets and collapsed variables.
        """
        exclude = set(self.world.monitored_targets()).union(spec.collapse_vars)
        return tuple(sorted((k, v) for k, v in context.items() if k not in exclude))

    def assignment_products(self, vars_: Sequence[Variable]) -> Iterable[Dict[Variable, Value]]:
        domains = [self.world.intervention_values(v) for v in vars_]
        for values in product(*domains):
            yield dict(zip(vars_, values))

    def certify_spec(self, spec: NethraSpec, cycle: Cycle) -> Certificate:
        witnesses: List[SubstitutionWitness] = []
        preservation: List[PreservationProbe] = []
        prediction_samples: List[Tuple[Key, Any]] = []

        scope_hits = 0
        substitutions_tested = 0
        targets_compared = 0

        for root in self.world.all_root_contexts():
            ctx = self.world.complete_context(root, cycle)
            if not spec.scope.fn(ctx, cycle):
                continue

            scope_hits += 1
            before_obs = self.world.observe(ctx, cycle)
            prediction_samples.append((self.prediction_key(ctx, spec), self.target_tuple(ctx, cycle)))

            for assignments in self.assignment_products(spec.collapse_vars):
                if all(ctx.get(v) == val for v, val in assignments.items()):
                    continue

                after_ctx = self.world.intervene_many(ctx, assignments, cycle)
                after_obs = self.world.observe(after_ctx, cycle)
                substitutions_tested += 1

                changed_any = False
                for target, before in before_obs.items():
                    targets_compared += 1
                    after = after_obs[target]
                    if not self.world.equal(target, before, after):
                        changed_any = True
                        witnesses.append(
                            SubstitutionWitness(
                                nethra=spec.name,
                                context=self.freeze_context(ctx),
                                assignments=tuple(sorted(assignments.items())),
                                target=target,
                                before=before,
                                after=after,
                                cycle=cycle,
                            )
                        )
                        if len(witnesses) >= self.config.max_witnesses:
                            cert = Certificate(
                                nethra=spec.name,
                                role="tareth",
                                cycle=cycle,
                                scope_hits=scope_hits,
                                substitutions_tested=substitutions_tested,
                                targets_compared=targets_compared,
                                witnesses=witnesses,
                                preservation=preservation,
                            )
                            return cert

                if not changed_any and len(preservation) < self.config.max_preservation:
                    preservation.append(
                        PreservationProbe(
                            nethra=spec.name,
                            context=self.freeze_context(ctx),
                            assignments=tuple(sorted(assignments.items())),
                            cycle=cycle,
                        )
                    )

        if scope_hits == 0:
            return Certificate(
                nethra=spec.name,
                role="invalid",
                cycle=cycle,
                scope_hits=0,
                substitutions_tested=0,
                targets_compared=0,
                invalid_reason="zero in-scope contexts",
            )

        return Certificate(
            nethra=spec.name,
            role="trass",
            cycle=cycle,
            scope_hits=scope_hits,
            substitutions_tested=substitutions_tested,
            targets_compared=targets_compared,
            witnesses=[],
            preservation=preservation,
        )

    def apply_certificate(self, name: Name, cert: Certificate, reason: str) -> bool:
        rec = self.records[name]
        old = rec.role
        rec.certificate = cert
        rec.audits += 1
        self.total_audits += 1
        rec.set_role(cert.role, cycle=cert.cycle, reason=reason)

        if cert.role == "tareth":
            # Train predictor only for attention-worthy factorizations.
            samples: List[Tuple[Key, Any]] = []
            for root in self.world.all_root_contexts():
                ctx = self.world.complete_context(root, cert.cycle)
                if rec.spec.scope.fn(ctx, cert.cycle):
                    samples.append((self.prediction_key(ctx, rec.spec), self.target_tuple(ctx, cert.cycle)))
            rec.predictor.fit(samples)
        elif cert.role in ("trass", "invalid"):
            rec.predictor = Predictor()
        else:
            raise ValueError(cert.role)

        return old != rec.role

    def certify_record(self, name: Name, cycle: Cycle, reason: str) -> bool:
        return self.apply_certificate(name, self.certify_spec(self.records[name].spec, cycle), reason)

    def certify_base_records(self, cycle: Cycle) -> bool:
        changed = False
        for name in self.base_specs:
            changed = self.certify_record(name, cycle, "base certification") or changed
        return changed

    def generated_composition_name(self, children: Sequence[Name]) -> Name:
        return "COMPOSE[" + "+".join(children) + "]"

    def remove_generated_compositions(self) -> None:
        for name in list(self.records):
            if self.records[name].spec.generated:
                del self.records[name]
        self.false_trass = []

    def build_earned_compositions(self, cycle: Cycle) -> None:
        """
        Generate compositions only from currently certified trass nethras.
        Then jointly certify each generated composition.
        """
        self.remove_generated_compositions()

        trass_names = [
            name for name, rec in self.records.items()
            if not rec.spec.generated and rec.role == "trass"
        ]

        seen_var_sets = set()
        for size in range(2, self.config.max_composition_size + 1):
            for child_tuple in combinations(trass_names, size):
                collapse_vars = tuple(sorted(set().union(*(self.records[c].spec.collapse_vars for c in child_tuple))))
                if len(collapse_vars) < 2:
                    continue

                key = (collapse_vars, self.composition_scope.name)
                if key in seen_var_sets:
                    continue
                seen_var_sets.add(key)

                name = self.generated_composition_name(child_tuple)
                spec = NethraSpec(
                    name=name,
                    collapse_vars=collapse_vars,
                    scope=self.composition_scope,
                    children=tuple(child_tuple),
                    generated=True,
                )
                self.records[name] = NethraRecord(spec=spec)
                cert = self.certify_spec(spec, cycle)
                self.apply_certificate(name, cert, "earned composition certification")

                if cert.role == "tareth" and cert.witnesses:
                    self.false_trass.append(FalseTrass(name, tuple(child_tuple), cert.witnesses[0]))

    def witness_active_now(self, witness: SubstitutionWitness, cycle: Cycle) -> bool:
        ctx = witness.context_dict()
        before_obs = self.world.observe(ctx, cycle)
        after_ctx = self.world.intervene_many(ctx, witness.assignment_dict(), cycle)
        after_obs = self.world.observe(after_ctx, cycle)
        return not self.world.equal(witness.target, before_obs[witness.target], after_obs[witness.target])

    def check_sentinels(self, rec: NethraRecord, cycle: Cycle) -> bool:
        if not rec.certificate or not rec.certificate.witnesses:
            return False

        active = 0
        for witness in rec.certificate.witnesses:
            rec.sentinel_checks += 1
            self.total_sentinel_checks += 1
            if self.witness_active_now(witness, cycle):
                active += 1

        return active > 0

    def check_prediction(self, rec: NethraRecord, cycle: Cycle) -> None:
        if rec.role != "tareth":
            return

        for root in self.world.all_root_contexts():
            ctx = self.world.complete_context(root, cycle)
            if not rec.spec.scope.fn(ctx, cycle):
                continue

            pred = rec.predictor.predict(self.prediction_key(ctx, rec.spec))
            if pred is None:
                continue

            actual = self.target_tuple(ctx, cycle)
            rec.prediction_checks += 1
            self.total_prediction_checks += 1
            if pred == actual:
                rec.prediction_correct += 1

    def attention_step(self, cycle: Cycle) -> bool:
        """
        Returns True if any role changed, requiring composition rebuild.
        """
        changed = False

        # Base records are the stable source. Generated records are rebuilt from them.
        for name in list(self.base_specs):
            rec = self.records[name]

            if rec.role == "trass":
                rec.skipped_cycles += 1
                self.total_skips += 1
                if self.config.recheck_period > 0 and cycle > 0 and cycle % self.config.recheck_period == 0:
                    changed = self.certify_record(name, cycle, "scheduled trass recheck") or changed
                continue

            if rec.role == "tareth":
                if not self.check_sentinels(rec, cycle):
                    rec.sentinel_failures += 1
                    changed = self.certify_record(name, cycle, "tareth sentinel failure") or changed
                else:
                    self.check_prediction(rec, cycle)
                continue

            if rec.role in ("unknown", "invalid"):
                changed = self.certify_record(name, cycle, "unknown/invalid certification") or changed
                continue

        # Generated compositions participate in attention but are rebuilt after base changes.
        for name, rec in list(self.records.items()):
            if not rec.spec.generated:
                continue

            if rec.role == "trass":
                rec.skipped_cycles += 1
                self.total_skips += 1
            elif rec.role == "tareth":
                self.check_sentinels(rec, cycle)
                self.check_prediction(rec, cycle)

        return changed

    def run(self, cycles: int) -> None:
        self.certify_base_records(cycle=0)
        self.build_earned_compositions(cycle=0)

        for cycle in range(cycles):
            if self.attention_step(cycle):
                self.build_earned_compositions(cycle)

    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(self.records):
            rec = self.records[name]
            acc = None if rec.prediction_checks == 0 else rec.prediction_correct / rec.prediction_checks
            cert = rec.certificate
            out.append(
                {
                    "name": name,
                    "generated": rec.spec.generated,
                    "role": rec.role,
                    "collapse_vars": rec.spec.collapse_vars,
                    "scope": rec.spec.scope.name,
                    "children": rec.spec.children,
                    "audits": rec.audits,
                    "scope_hits": None if cert is None else cert.scope_hits,
                    "substitutions_tested": None if cert is None else cert.substitutions_tested,
                    "sentinel_checks": rec.sentinel_checks,
                    "sentinel_failures": rec.sentinel_failures,
                    "prediction_checks": rec.prediction_checks,
                    "prediction_accuracy": acc,
                    "skipped_cycles": rec.skipped_cycles,
                    "role_changes": rec.role_changes,
                    "invalid_reason": None if cert is None else cert.invalid_reason,
                }
            )
        return out


# =============================================================================
# Receipt world and scopes
# =============================================================================

def make_receipt_world(drift_at: int = 12) -> FiniteCausalDAGWorld:
    """
    Roots: a,b,c,d.
    Target: y.

    Base phase:
        y = a & b

    Drift phase:
        y = (a & b) | d

    Consequences:
        A_local: collapse a in scope b=0 => trass.
        B_local: collapse b in scope a=0 => trass.
        COMPOSE[A_local+B_local]: collapse a,b globally => tareth (false-trass).
        C_global: collapse c globally => trass.
        D_global: collapse d globally => trass before drift, tareth after drift.
    """
    def y_mech(state: Mapping[str, Any], cycle: Cycle) -> int:
        y = int(state["a"]) & int(state["b"])
        if cycle >= drift_at:
            y = y | int(state["d"])
        return y

    nodes = [
        NodeSpec("a", (0, 1)),
        NodeSpec("b", (0, 1)),
        NodeSpec("c", (0, 1)),
        NodeSpec("d", (0, 1)),
        NodeSpec("y", (0, 1), parents=("a", "b", "d"), mechanism=y_mech),
    ]
    return FiniteCausalDAGWorld(
        nodes=nodes,
        intervenable=("a", "b", "c", "d"),
        monitored_targets=("y",),
        seed=0,
    )


def scope_b0(ctx: Mapping[str, Any], cycle: Cycle) -> bool:
    return ctx["b"] == 0


def scope_a0(ctx: Mapping[str, Any], cycle: Cycle) -> bool:
    return ctx["a"] == 0


GLOBAL = Scope("global", always_scope)
B0 = Scope("b=0", scope_b0)
A0 = Scope("a=0", scope_a0)


def make_base_specs() -> List[NethraSpec]:
    return [
        NethraSpec("A_local", ("a",), B0),
        NethraSpec("B_local", ("b",), A0),
        NethraSpec("C_global", ("c",), GLOBAL),
        NethraSpec("D_global", ("d",), GLOBAL),
    ]


def print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    print("name                      gen role    vars        scope   children                  aud scope subs sent fail pred skip")
    for r in rows:
        print(
            f"{r['name']:<25} {str(r['generated']):<3} {r['role']:<7} "
            f"{str(r['collapse_vars']):<11} {r['scope']:<7} {str(r['children']):<25} "
            f"{r['audits']:<3} {str(r['scope_hits']):<5} {str(r['substitutions_tested']):<4} "
            f"{r['sentinel_checks']:<4} {r['sentinel_failures']:<4} {r['prediction_checks']:<4} {r['skipped_cycles']:<4}"
        )


def run_receipt(args: argparse.Namespace) -> None:
    engine = EarnedRecursiveNethraEngine(
        world=make_receipt_world(drift_at=args.drift_at),
        base_specs=make_base_specs(),
        composition_scope=GLOBAL,
        config=EngineConfig(
            max_composition_size=2,
            recheck_period=args.recheck_period,
            seed=args.seed,
        ),
    )
    engine.run(args.cycles)

    print("EARNED RECURSIVE NETHRA SYSTEM")
    print("──────────────────────────────")
    print(f"cycles={args.cycles} drift_at={args.drift_at} recheck_period={args.recheck_period}")
    print_rows(engine.rows())

    print()
    print("FALSE-TRASS")
    print("───────────")
    for ft in engine.false_trass:
        w = ft.witness
        print(
            f"{ft.composed}: children={ft.children} "
            f"context={dict(w.context)} assignments={dict(w.assignments)} "
            f"{w.target}: {w.before}->{w.after}"
        )

    if args.json:
        print(json.dumps(engine.rows(), indent=2, default=str))

    rows = {r["name"]: r for r in engine.rows()}

    assert rows["A_local"]["role"] == "trass"
    assert rows["B_local"]["role"] == "trass"
    assert rows["C_global"]["role"] == "trass"
    assert rows["D_global"]["role"] == "tareth"

    composed = [r for r in engine.rows() if r["generated"] and set(r["children"]) == {"A_local", "B_local"}]
    assert composed, "earned A/B composition missing"
    assert composed[0]["role"] == "tareth"

    assert engine.false_trass, "false-trass not detected"
    assert set(engine.false_trass[0].children) == {"A_local", "B_local"}

    assert rows["C_global"]["skipped_cycles"] > 0
    assert rows["D_global"]["sentinel_checks"] > 0
    assert any(old == "trass" and new == "tareth" for _, old, new, _ in engine.records["D_global"].role_changes)

    # Verify no invalid zero-scope trass is possible by construction with a deliberate bad scope.
    impossible = Scope("impossible", lambda ctx, cycle: False)
    bad = NethraSpec("BAD_zero_scope", ("a",), impossible)
    bad_cert = engine.certify_spec(bad, cycle=0)
    assert bad_cert.role == "invalid"
    assert bad_cert.scope_hits == 0

    print()
    print("ASSERTIONS")
    print("──────────")
    print("PASS: base nethras certify scoped trass/tareth")
    print("PASS: compositions are generated from certified trass children")
    print("PASS: composed nethra is jointly substitution-tested")
    print("PASS: local-trass / joint-tareth false-trass is detected")
    print("PASS: real sentinel checks execute witness interventions")
    print("PASS: trass D_global reopens and becomes tareth after drift")
    print("PASS: zero-scope candidates are invalid, not trass")
    print("PASS: predictor keys exclude monitored targets")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--drift-at", type=int, default=12)
    p.add_argument("--recheck-period", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p


if __name__ == "__main__":
    run_receipt(build_parser().parse_args())
