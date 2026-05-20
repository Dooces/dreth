#!/usr/bin/env python3
"""
baseline_attention_budget.py

Plain implementation of the original idea:
- maintain per-variable model state
- classify variables as relevant or irrelevant by intervention
- spend audit budget only on relevant / uncertain variables
- keep cheap sentinels for already-certified relevant variables
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

Value = int
Probe = Dict[int, Value]
Fn = Callable[[Probe], Value]

TOL = 0
N_PROBES = 80
N_SENTINELS = 5
INTERVENTION_VALUES = [0, 1, 2, 3, 4]


# ---------- toy world ----------

def f_const(_: Probe) -> Value:
    return 1

def f_copy0(x: Probe) -> Value:
    return x[0]

def f_sum01(x: Probe) -> Value:
    return x[0] + x[1]

def f_max12(x: Probe) -> Value:
    return max(x[1], x[2])

TRUE_FUNCS: Dict[int, Fn] = {
    0: f_const,
    1: f_copy0,
    2: f_sum01,
    3: f_max12,
    4: f_const,     # irrelevant sink-like variable
}

HYPOTHESES: Dict[str, Fn] = {
    "CONST": f_const,
    "COPY0": f_copy0,
    "SUM01": f_sum01,
    "MAX12": f_max12,
}


def sample_context(rng: random.Random) -> Probe:
    return {i: rng.randint(0, 4) for i in TRUE_FUNCS}


def observe(var: int, ctx: Probe) -> Value:
    return TRUE_FUNCS[var](ctx)


# ---------- per-variable state ----------

@dataclass
class VarState:
    role: str = "unknown"          # unknown | relevant | irrelevant
    best_name: str | None = None
    best_score: int = -1
    sentinels: List[Probe] = field(default_factory=list)
    audits: int = 0
    sentinel_checks: int = 0
    failures: int = 0


# ---------- standard primitive 1: model discrimination / QBC fit ----------

def fit_var(var: int, rng: random.Random) -> Tuple[str, int, List[Probe]]:
    probes = [sample_context(rng) for _ in range(N_PROBES)]

    scores: Dict[str, int] = {}
    for name, hyp in HYPOTHESES.items():
        score = 0
        for p in probes:
            if abs(hyp(p) - observe(var, p)) <= TOL:
                score += 1
        scores[name] = score

    best_name = max(scores, key=scores.get)
    best_fn = HYPOTHESES[best_name]

    # QBC sentinel selection: keep probes where alternatives disagree with winner.
    ranked = []
    for p in probes:
        y_best = best_fn(p)
        disagree = sum(
            1 for name, hyp in HYPOTHESES.items()
            if name != best_name and abs(hyp(p) - y_best) > TOL
        )
        ranked.append((disagree, p))

    ranked.sort(key=lambda t: t[0], reverse=True)
    sentinels = [p for _, p in ranked[:N_SENTINELS]]
    return best_name, scores[best_name], sentinels


# ---------- standard primitive 2: active interventional relevance test ----------

def classify_relevance(var: int) -> str:
    """
    A variable is relevant iff intervening on it changes some other observable.
    This is the plain tareth/trass replacement.
    """
    for target in TRUE_FUNCS:
        if target == var:
            continue

        vals = []
        for forced in INTERVENTION_VALUES:
            ctx = {i: 1 for i in TRUE_FUNCS}
            ctx[var] = forced
            vals.append(observe(target, ctx))

        if max(vals) - min(vals) > TOL:
            return "relevant"

    return "irrelevant"


# ---------- cheap path + full audit loop ----------

def sentinel_ok(var: int, st: VarState) -> bool:
    if st.best_name is None:
        return False

    hyp = HYPOTHESES[st.best_name]
    for p in st.sentinels:
        st.sentinel_checks += 1
        if abs(hyp(p) - observe(var, p)) > TOL:
            st.failures += 1
            return False
    return True


def run(seed: int = 3, cycles: int = 100) -> None:
    rng = random.Random(seed)
    states = {v: VarState() for v in TRUE_FUNCS}

    for cycle in range(cycles):
        for var, st in states.items():
            if st.role == "unknown":
                st.role = classify_relevance(var)

            if st.role == "irrelevant":
                continue

            if sentinel_ok(var, st):
                continue

            best, score, sentinels = fit_var(var, rng)
            st.best_name = best
            st.best_score = score
            st.sentinels = sentinels
            st.audits += 1

    print("var  role        model   audits  sentinel_checks  failures")
    for v, st in states.items():
        print(
            f"{v:<4} {st.role:<11} {str(st.best_name):<7} "
            f"{st.audits:<7} {st.sentinel_checks:<16} {st.failures}"
        )


if __name__ == "__main__":
    run()