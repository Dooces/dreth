#!/usr/bin/env python3
"""
full_recursive_nethra_system.py

Full runnable implementation of the nethra / tareth / trass framework as the
corrected invariant requires.

Implemented meanings:

    nethra
        A vetted operative factoring. It may be a leaf factoring or a composed
        factoring built from previously certified nethras. It carries:
          - what distinctions it collapses
          - the scope/context in which certification was earned
          - the operation/targets preserved
          - witnesses or preservation certificate
          - children if composed

    trass
        A collapse certified safe in its scope:
            substituting/collapsing the distinction does not change monitored targets.

    tareth
        A distinction that must survive:
            substitution changes monitored targets, with a concrete witness.

    false-trass
        Local nethras are each trass in their own scopes, but their composition
        fails under the composition scope. This is detected by a joint substitution
        test and recorded as a counterexample.

    attention economy
        Certified trass nethras are skipped/collapsed.
        Tareth nethras keep sentinels/prediction checks.
        Composed trass nethras gate/collapse their children.

This is not a demo-only toy:
    - FiniteCausalDAGWorld is a reusable finite causal environment.
    - NethraEngine accepts arbitrary candidate nethras and scopes.
    - The included receipt world is only a verification harness.

Run:
    python3 full_recursive_nethra_system.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


Name = str
Variable = str
Target = str
Value = Any
State = Dict[str, Any]
Cycle = int
Key = Tuple[Tuple[str, Any], ...]
Mechanism = Callable[[Mapping[str, Any], Cycle], Any]
ScopeFn = Callable[[Mapping[str, Any], Cycle], bool]


# =============================================================================
# Generic finite causal DAG
# =============================================================================

@dataclass(frozen=True)
class NodeSpec:
    name: Variable
    domain: Tuple[Value, ...]
    parents: Tuple[Variable, ...] = ()
    mechanism: Optional[Mechanism] = None


class FiniteCausalDAGWorld:
    """
    Reusable finite causal DAG with real do-interventions.

    Root nodes have no mechanism and are enumerated/sampled from domains.
    Mechanism nodes are recomputed topologically.
    """

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
                raise KeyError(v)
        for t in self._targets:
            if t not in self.nodes:
                raise KeyError(t)

        self.children = {name: [] for name in self.nodes}
        for n in nodes:
            for p in n.parents:
                if p not in self.nodes:
                    raise KeyError(f"missing parent {p!r}")
                self.children[p].append(n.name)

        self.topo = self._topological_order()
        self.roots = tuple(name for name in self.topo if self.nodes[name].mechanism is None)

    def _topological_order(self) -> Tuple[Variable, ...]:
        indeg = {name: 0 for name in self.nodes}
        for n in self.nodes.values():
            for _ in n.parents:
                indeg[n.name] += 1
        q = deque(sorted(k for k, d in indeg.items() if d == 0))
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

    def all_root_contexts(self) -> Iterable[State]:
        domains = [self.nodes[r].domain for r in self.roots]
        for values in product(*domains):
            yield dict(zip(self.roots, values))

    def complete_context(self, root_context: Mapping[str, Any], cycle: Cycle) -> State:
        state: State = dict(root_context)
        for name in self.topo:
            spec = self.nodes[name]
            if spec.mechanism is not None:
                state[name] = spec.mechanism(state, cycle)
            elif name not in state:
                state[name] = self.rng.choice(spec.domain)
        return state

    def sample_context(self, rng: random.Random, cycle: Cycle) -> State:
        roots = {r: rng.choice(self.nodes[r].domain) for r in self.roots}
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

    def intervene_many(self, context: State, assignments: Mapping[Variable, Value], cycle: Cycle) -> State:
        """
        do(assignments): force one or more variables and recompute descendants.
        """
        out = dict(context)
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

    def observe(self, context: State, cycle: Cycle) -> Mapping[Target, Any]:
        return {t: context[t] for t in self._targets}

    def equal(self, target: Target, a: Any, b: Any) -> bool:
        return a == b


# =============================================================================
# Nethra certification / composition
# =============================================================================

def always_scope(_: Mapping[str, Any], __: Cycle) -> bool:
    return True


@dataclass(frozen=True)
class NethraSpec:
    """
    Candidate operative factoring.

    collapse_vars:
        variables whose distinctions may be collapsed/ignored/substituted.

    scope:
        certification context. This makes certificates explicitly scoped rather
        than ontological.

    children:
        if present, this candidate is a composition of prior nethras.
    """

    name: Name
    collapse_vars: Tuple[Variable, ...]
    scope: ScopeFn = always_scope
    children: Tuple[Name, ...] = ()


@dataclass(frozen=True)
class SubstitutionWitness:
    """
    Concrete counterexample: a proposed collapse changes operation output.
    """

    nethra: Name
    assignments: Tuple[Tuple[Variable, Value], ...]
    context: Key
    target: Target
    before: Any
    after: Any
    cycle: int

    def thaw_context(self) -> State:
        return dict(self.context)


@dataclass(frozen=True)
class PreservationProbe:
    nethra: Name
    assignments: Tuple[Tuple[Variable, Value], ...]
    context: Key
    cycle: int


@dataclass
class Certificate:
    nethra: Name
    role: str  # trass | tareth
    cycle: int
    scope_hits: int
    substitutions_tested: int
    targets_compared: int
    witnesses: List[SubstitutionWitness] = field(default_factory=list)
    preservation: List[PreservationProbe] = field(default_factory=list)


@dataclass
class Predictor:
    """
    Simple predictor used only for tareth nethras to show attention budget.
    Predicts observed target tuple from non-collapsed context key.
    """

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
        for k, y in samples:
            by_key[k][y] += 1
            global_counts[y] += 1
        self.table = {k: c.most_common(1)[0][0] for k, c in by_key.items()}
        self.fallback = global_counts.most_common(1)[0][0]
        self.trained_samples = len(samples)

    def predict(self, key: Key) -> Optional[Any]:
        return self.table.get(key, self.fallback)


@dataclass
class NethraRecord:
    """
    Certified operative factoring.

    Not merely a variable label. This is the proof-carrying record for one
    collapse/factoring, including composed nethras.
    """

    spec: NethraSpec
    role: str = "unknown"
    certificate: Optional[Certificate] = None
    predictor: Predictor = field(default_factory=Predictor)

    audits: int = 0
    sentinel_checks: int = 0
    prediction_checks: int = 0
    prediction_correct: int = 0
    skipped_cycles: int = 0
    child_gated_skips: int = 0
    role_changes: List[Tuple[int, str, str, str]] = field(default_factory=list)

    @property
    def name(self) -> Name:
        return self.spec.name

    @property
    def collapse_vars(self) -> Tuple[Variable, ...]:
        return self.spec.collapse_vars

    @property
    def children(self) -> Tuple[Name, ...]:
        return self.spec.children

    def set_role(self, role: str, cycle: int, reason: str) -> None:
        old = self.role
        if old != role:
            self.role_changes.append((cycle, old, role, reason))
        self.role = role


@dataclass
class FalseTrass:
    """
    Local trass certificates whose composed collapse is tareth.
    """

    composed: Name
    children: Tuple[Name, ...]
    witness: SubstitutionWitness


@dataclass
class EngineConfig:
    max_preservation: int = 32
    max_witnesses: int = 8
    seed: int = 0


class NethraEngine:
    """
    Earned recursive nethra engine.

    It:
      1. certifies candidate nethras by substitution/intervention tests
      2. composes certified nethras
      3. jointly re-tests composition
      4. records false-trass when local trass fails jointly
      5. uses role to gate attention
    """

    def __init__(self, world: FiniteCausalDAGWorld, specs: Sequence[NethraSpec], config: EngineConfig):
        self.world = world
        self.config = config
        self.rng = random.Random(config.seed)
        self.specs = self._normalize_specs(specs)
        self._cert_order = self._certification_order()
        self.records: Dict[Name, NethraRecord] = {
            s.name: NethraRecord(spec=s) for s in self.specs.values()
        }
        self.false_trass: List[FalseTrass] = []
        self.total_audits = 0
        self.total_skips = 0
        self.total_sentinel_checks = 0
        self.total_prediction_checks = 0

    def _normalize_specs(self, specs: Sequence[NethraSpec]) -> Dict[Name, NethraSpec]:
        by_name: Dict[Name, NethraSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate nethra spec {spec.name!r}")
            by_name[spec.name] = spec

        visiting: set[Name] = set()
        visited: set[Name] = set()

        def visit(name: Name) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"cycle in nethra composition at {name!r}")
            if name not in by_name:
                raise KeyError(f"missing nethra spec {name!r}")
            visiting.add(name)
            for child in by_name[name].children:
                visit(child)
            visiting.remove(name)
            visited.add(name)

        for name in list(by_name):
            visit(name)

        # A composed nethra's collapse set is the union of its certified
        # components. If the caller supplies the same set, accept it; if they
        # leave it empty, derive it; if they supply a different set, reject it.
        for name in list(by_name):
            spec = by_name[name]
            if not spec.children:
                continue
            child_vars = tuple(sorted({
                v
                for child in spec.children
                for v in by_name[child].collapse_vars
            }))
            if spec.collapse_vars and tuple(sorted(spec.collapse_vars)) != child_vars:
                raise ValueError(
                    f"composed nethra {name!r} collapse_vars {spec.collapse_vars!r} "
                    f"do not match children union {child_vars!r}"
                )
            by_name[name] = replace(spec, collapse_vars=child_vars)

        return by_name

    def _certification_order(self) -> Tuple[Name, ...]:
        order: List[Name] = []
        seen: set[Name] = set()

        def visit(name: Name) -> None:
            if name in seen:
                return
            for child in self.specs[name].children:
                visit(child)
            seen.add(name)
            order.append(name)

        for name in self.specs:
            visit(name)
        return tuple(order)

    @staticmethod
    def freeze_context(context: Mapping[str, Any]) -> Key:
        return tuple(sorted(context.items(), key=lambda kv: kv[0]))

    def _target_tuple(self, context: Mapping[str, Any], cycle: int) -> Tuple[Tuple[Target, Any], ...]:
        return tuple(sorted(self.world.observe(dict(context), cycle).items()))

    def _prediction_key(self, context: Mapping[str, Any], spec: NethraSpec) -> Key:
        excluded = set(spec.collapse_vars)
        excluded.update(self.world._targets)
        return tuple(sorted((k, v) for k, v in context.items() if k not in excluded))

    def _assignment_products(self, vars_: Sequence[Variable]) -> Iterable[Dict[Variable, Value]]:
        domains = [self.world.intervention_values(v) for v in vars_]
        for values in product(*domains):
            yield dict(zip(vars_, values))

    def certify(self, name: Name, cycle: int) -> Certificate:
        rec = self.records[name]
        spec = rec.spec

        witnesses: List[SubstitutionWitness] = []
        preservation: List[PreservationProbe] = []
        pred_samples: List[Tuple[Key, Any]] = []

        scope_hits = 0
        substitutions_tested = 0
        targets_compared = 0

        for root_ctx in self.world.all_root_contexts():
            ctx = self.world.complete_context(root_ctx, cycle)
            if not spec.scope(ctx, cycle):
                continue

            scope_hits += 1
            baseline = self.world.observe(ctx, cycle)
            pred_samples.append((self._prediction_key(ctx, spec), self._target_tuple(ctx, cycle)))

            for assignments in self._assignment_products(spec.collapse_vars):
                # Skip no-op assignment; it cannot prove anything.
                if all(ctx.get(v) == val for v, val in assignments.items()):
                    continue

                after_ctx = self.world.intervene_many(ctx, assignments, cycle)
                after = self.world.observe(after_ctx, cycle)
                substitutions_tested += 1

                changed_any = False
                for target, before in baseline.items():
                    targets_compared += 1
                    if not self.world.equal(target, before, after[target]):
                        changed_any = True
                        witnesses.append(
                            SubstitutionWitness(
                                nethra=spec.name,
                                assignments=tuple(sorted(assignments.items())),
                                context=self.freeze_context(ctx),
                                target=target,
                                before=before,
                                after=after[target],
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
                            rec.predictor.fit(pred_samples)
                            return cert

                if not changed_any and len(preservation) < self.config.max_preservation:
                    preservation.append(
                        PreservationProbe(
                            nethra=spec.name,
                            assignments=tuple(sorted(assignments.items())),
                            context=self.freeze_context(ctx),
                            cycle=cycle,
                        )
                    )

        if scope_hits == 0:
            return Certificate(
                nethra=spec.name,
                role="unscoped",
                cycle=cycle,
                scope_hits=0,
                substitutions_tested=0,
                targets_compared=0,
                witnesses=[],
                preservation=[],
            )

        if witnesses:
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
            rec.predictor.fit(pred_samples)
            return cert

        cert = Certificate(
            nethra=spec.name,
            role="trass",
            cycle=cycle,
            scope_hits=scope_hits,
            substitutions_tested=substitutions_tested,
            targets_compared=targets_compared,
            witnesses=[],
            preservation=preservation,
        )
        if cert.role == "tareth":
            rec.predictor.fit(pred_samples)
        return cert

    def apply_certificate(self, name: Name, cert: Certificate, reason: str) -> None:
        rec = self.records[name]
        rec.audits += 1
        self.total_audits += 1
        rec.certificate = cert
        rec.set_role(cert.role, cert.cycle, reason)

    def certify_all(self, cycle: int = 0) -> None:
        for name in self._cert_order:
            self.apply_certificate(name, self.certify(name, cycle), "certification")
            self._record_false_trass_if_needed(name)

    def _record_false_trass_if_needed(self, name: Name) -> None:
        rec = self.records[name]
        if not rec.children or rec.role != "tareth":
            return
        child_roles = tuple(self.records[c].role for c in rec.children if c in self.records)
        if child_roles and all(r == "trass" for r in child_roles):
            if rec.certificate and rec.certificate.witnesses:
                already_recorded = any(
                    ft.composed == name and ft.witness == rec.certificate.witnesses[0]
                    for ft in self.false_trass
                )
                if already_recorded:
                    return
                self.false_trass.append(
                    FalseTrass(
                        composed=name,
                        children=rec.children,
                        witness=rec.certificate.witnesses[0],
                    )
                )

    def compose_nethra(
        self,
        name: Name,
        children: Sequence[Name],
        scope: ScopeFn = always_scope,
        cycle: int = 0,
    ) -> NethraRecord:
        if name in self.records:
            raise ValueError(f"nethra {name!r} already exists")
        child_tuple = tuple(children)
        if not child_tuple:
            raise ValueError("composed nethra needs at least one child")
        for child in child_tuple:
            if child not in self.records:
                raise KeyError(f"missing child nethra {child!r}")
            if self.records[child].certificate is None or self.records[child].role in ("unknown", "unscoped"):
                raise ValueError(f"child nethra {child!r} is not vetted")

        collapse_vars = tuple(sorted({
            v
            for child in child_tuple
            for v in self.records[child].collapse_vars
        }))
        spec = NethraSpec(name=name, collapse_vars=collapse_vars, scope=scope, children=child_tuple)
        rec = NethraRecord(spec=spec)
        self.specs[name] = spec
        self.records[name] = rec
        self._cert_order = self._cert_order + (name,)
        self.apply_certificate(name, self.certify(name, cycle), "composition certification")
        self._record_false_trass_if_needed(name)
        return rec

    def witness_active_now(self, witness: SubstitutionWitness, cycle: int) -> bool:
        frozen = witness.thaw_context()
        root_ctx = {r: frozen[r] for r in self.world.roots if r in frozen}
        ctx = self.world.complete_context(root_ctx, cycle)
        baseline = self.world.observe(ctx, cycle)
        after_ctx = self.world.intervene_many(ctx, dict(witness.assignments), cycle)
        after = self.world.observe(after_ctx, cycle)
        return not self.world.equal(
            witness.target,
            baseline[witness.target],
            after[witness.target],
        )

    def recertify(self, name: Name, cycle: int, reason: str) -> None:
        cert = self.certify(name, cycle)
        self.apply_certificate(name, cert, reason)
        self._record_false_trass_if_needed(name)

    def step_attention(self, name: Name, cycle: int) -> None:
        """
        Attention accounting:
        - trass nethra skips itself and gates children.
        - tareth nethra keeps sentinels/prediction and descends.
        """
        rec = self.records[name]

        if rec.role == "trass":
            rec.skipped_cycles += 1
            self.total_skips += 1
            for child in rec.children:
                if child in self.records:
                    rec.child_gated_skips += 1
            return

        if rec.role == "tareth":
            if rec.certificate:
                alive = 0
                for witness in rec.certificate.witnesses:
                    rec.sentinel_checks += 1
                    self.total_sentinel_checks += 1
                    if self.witness_active_now(witness, cycle):
                        alive += 1
                if rec.certificate.witnesses and alive == 0:
                    self.recertify(name, cycle, "tareth witness failure")
                    rec = self.records[name]
                    if rec.role != "tareth":
                        return

            # Prediction check against all contexts in current scope.
            checked = 0
            correct = 0
            for root_ctx in self.world.all_root_contexts():
                ctx = self.world.complete_context(root_ctx, cycle)
                if not rec.spec.scope(ctx, cycle):
                    continue
                pred = rec.predictor.predict(self._prediction_key(ctx, rec.spec))
                actual = self._target_tuple(ctx, cycle)
                if pred is None:
                    continue
                checked += 1
                if pred == actual:
                    correct += 1
            rec.prediction_checks += checked
            rec.prediction_correct += correct
            self.total_prediction_checks += checked

            for child in rec.children:
                if child in self.records:
                    self.step_attention(child, cycle)
            return

        if rec.role in ("unknown", "unscoped"):
            self.recertify(name, cycle, f"{rec.role} recertification")
            return

    def run_attention(self, root_names: Sequence[Name], cycles: int) -> None:
        for cycle in range(cycles):
            for root in root_names:
                self.step_attention(root, cycle)

    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for name in self.records:
            rec = self.records[name]
            acc = None if rec.prediction_checks == 0 else rec.prediction_correct / rec.prediction_checks
            out.append(
                {
                    "name": name,
                    "collapse_vars": rec.collapse_vars,
                    "children": rec.children,
                    "role": rec.role,
                    "audits": rec.audits,
                    "scope_hits": None if rec.certificate is None else rec.certificate.scope_hits,
                    "subs_tested": None if rec.certificate is None else rec.certificate.substitutions_tested,
                    "sentinel_checks": rec.sentinel_checks,
                    "prediction_checks": rec.prediction_checks,
                    "prediction_accuracy": acc,
                    "skipped_cycles": rec.skipped_cycles,
                    "child_gated_skips": rec.child_gated_skips,
                    "role_changes": rec.role_changes,
                }
            )
        return out


# =============================================================================
# Receipt world demonstrating all required pieces
# =============================================================================

def make_receipt_world() -> FiniteCausalDAGWorld:
    """
    DAG:
        a,b,c are roots.
        y = a & b.

    Scopes:
        A_local is certified where b=0. There, collapsing a is trass.
        B_local is certified where a=0. There, collapsing b is trass.
        AB_composed is tested globally. Jointly collapsing a,b is tareth
        because do(a=1,b=1) changes y from 0 to 1.

    c is globally trass.
    """
    def y_mech(state: Mapping[str, Any], cycle: int) -> int:
        return int(state["a"]) & int(state["b"])

    nodes = [
        NodeSpec("a", (0, 1)),
        NodeSpec("b", (0, 1)),
        NodeSpec("c", (0, 1)),
        NodeSpec("y", (0, 1), parents=("a", "b"), mechanism=y_mech),
    ]
    return FiniteCausalDAGWorld(
        nodes=nodes,
        intervenable=("a", "b", "c"),
        monitored_targets=("y",),
        seed=0,
    )


def scope_b0(ctx: Mapping[str, Any], cycle: int) -> bool:
    return ctx["b"] == 0


def scope_a0(ctx: Mapping[str, Any], cycle: int) -> bool:
    return ctx["a"] == 0


def make_receipt_specs() -> List[NethraSpec]:
    return [
        NethraSpec("A_local", collapse_vars=("a",), scope=scope_b0),
        NethraSpec("B_local", collapse_vars=("b",), scope=scope_a0),
        NethraSpec("C_global", collapse_vars=("c",), scope=always_scope),
    ]


def never_scope(ctx: Mapping[str, Any], cycle: int) -> bool:
    return False


def make_vanishing_witness_world() -> FiniteCausalDAGWorld:
    def y_mech(state: Mapping[str, Any], cycle: int) -> int:
        if cycle == 0:
            return int(state["a"]) & int(state["b"])
        return 0

    return FiniteCausalDAGWorld(
        nodes=[
            NodeSpec("a", (0, 1)),
            NodeSpec("b", (0, 1)),
            NodeSpec("y", (0, 1), parents=("a", "b"), mechanism=y_mech),
        ],
        intervenable=("a", "b"),
        monitored_targets=("y",),
        seed=0,
    )


def run_semantic_regressions() -> None:
    # Composed certification must not depend on caller insertion order.
    reversed_engine = NethraEngine(
        world=make_receipt_world(),
        specs=list(reversed(make_receipt_specs())),
        config=EngineConfig(),
    )
    reversed_engine.certify_all(cycle=0)
    reversed_engine.compose_nethra("AB_composed", ("A_local", "B_local"), always_scope, cycle=0)
    assert reversed_engine.records["AB_composed"].role == "tareth"
    assert reversed_engine.false_trass

    # A composed nethra's collapse set must be derived from its children, not
    # silently accepted when hand-written inconsistently.
    try:
        NethraEngine(
            world=make_receipt_world(),
            specs=[
                NethraSpec("A_local", collapse_vars=("a",), scope=scope_b0),
                NethraSpec("B_local", collapse_vars=("b",), scope=scope_a0),
                NethraSpec("bad", collapse_vars=("a",), children=("A_local", "B_local")),
            ],
            config=EngineConfig(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent composed collapse_vars were accepted")

    # Empty scope is not a trass certificate. It is no certificate.
    empty_scope_engine = NethraEngine(
        world=make_receipt_world(),
        specs=[NethraSpec("empty", collapse_vars=("a",), scope=never_scope)],
        config=EngineConfig(),
    )
    empty_scope_engine.certify_all(cycle=0)
    assert empty_scope_engine.records["empty"].role == "unscoped"

    # Prediction cannot leak monitored targets or collapsed distinctions.
    leak_engine = NethraEngine(
        world=make_receipt_world(),
        specs=make_receipt_specs(),
        config=EngineConfig(),
    )
    leak_engine.certify_all(cycle=0)
    leak_engine.compose_nethra("AB_composed", ("A_local", "B_local"), always_scope, cycle=0)
    ctx = leak_engine.world.complete_context({"a": 1, "b": 1, "c": 0}, 0)
    key_names = {k for k, _ in leak_engine._prediction_key(ctx, leak_engine.specs["AB_composed"])}
    assert "y" not in key_names
    assert "a" not in key_names
    assert "b" not in key_names

    # Tareth sentinels must be actual witness rechecks. If the witness no longer
    # propagates, the nethra must recertify rather than just increment a counter.
    drift_engine = NethraEngine(
        world=make_vanishing_witness_world(),
        specs=[NethraSpec("AB", collapse_vars=("a", "b"))],
        config=EngineConfig(),
    )
    drift_engine.certify_all(cycle=0)
    assert drift_engine.records["AB"].role == "tareth"
    drift_engine.step_attention("AB", cycle=1)
    assert drift_engine.records["AB"].role == "trass"
    assert drift_engine.records["AB"].audits == 2


def print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    print("name         role    collapse  children              scope subs  sent pred skip gate changes")
    for r in rows:
        print(
            f"{r['name']:<12} {r['role']:<7} {str(r['collapse_vars']):<9} "
            f"{str(r['children']):<21} {str(r['scope_hits']):<5} {str(r['subs_tested']):<5} "
            f"{r['sentinel_checks']:<4} {r['prediction_checks']:<4} "
            f"{r['skipped_cycles']:<4} {r['child_gated_skips']:<4} {r['role_changes']}"
        )


def run_receipt(args: argparse.Namespace) -> None:
    engine = NethraEngine(
        world=make_receipt_world(),
        specs=make_receipt_specs(),
        config=EngineConfig(seed=args.seed),
    )
    engine.certify_all(cycle=0)
    engine.compose_nethra("AB_composed", ("A_local", "B_local"), always_scope, cycle=0)
    engine.run_attention(root_names=("AB_composed", "C_global"), cycles=args.cycles)

    print("FULL RECURSIVE NETHRA SYSTEM")
    print("────────────────────────────")
    print_rows(engine.rows())
    print()
    print(
        f"TOTAL audits={engine.total_audits} "
        f"sentinels={engine.total_sentinel_checks} "
        f"predictions={engine.total_prediction_checks} "
        f"skips={engine.total_skips}"
    )

    print()
    print("FALSE-TRASS")
    print("───────────")
    for ft in engine.false_trass:
        w = ft.witness
        print(
            f"{ft.composed}: children={ft.children} "
            f"witness context={dict(w.context)} assignments={dict(w.assignments)} "
            f"{w.target}: {w.before}->{w.after}"
        )

    if args.json:
        print(json.dumps(engine.rows(), indent=2, default=str))

    rec = engine.records

    assert rec["A_local"].role == "trass"
    assert rec["B_local"].role == "trass"
    assert rec["C_global"].role == "trass"
    assert rec["AB_composed"].role == "tareth"

    assert engine.false_trass
    assert engine.false_trass[0].composed == "AB_composed"
    assert engine.false_trass[0].children == ("A_local", "B_local")

    assert rec["C_global"].skipped_cycles == args.cycles
    assert rec["AB_composed"].sentinel_checks > 0
    assert rec["AB_composed"].prediction_checks > 0

    run_semantic_regressions()

    print()
    print("ASSERTIONS")
    print("──────────")
    print("PASS: nethra is a vetted operative factoring, not a flat variable label")
    print("PASS: local nethras A_local and B_local certify trass in their scopes")
    print("PASS: composed nethra AB_composed is jointly re-tested")
    print("PASS: local-trass / joint-tareth false-trass is detected")
    print("PASS: trass nethra C_global is skipped")
    print("PASS: tareth composed nethra keeps sentinels and prediction checks")
    print("PASS: semantic regressions cover ordering, scope, leakage, and sentinel recertification")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p


if __name__ == "__main__":
    run_receipt(build_parser().parse_args())
