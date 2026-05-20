#!/usr/bin/env python3
"""
test_rare_catastrophe.py

Three demonstrations:

1. RARE DETECTION — a tareth var's function mutates rarely (rare_prob per cycle).
   Only var 0 carries high cost_weight (consequential). Other vars use default.
   Measures: cycles between mutation and detection (via detected_drift_vars only).

2. ANTI-FORGETTING — certs earned early survive many quiet cycles unchanged.
   Checks cert evidence (role, changes, trials) — not Python object identity,
   which changes on dataclasses.replace regardless of content.

3. INVERSION — the burden of proof is on ruling out, not confirming.
   All vars start untested. No var can skip its first cycle. Trass requires
   0/5 perturbation evidence. Default is full attention.

Run:
    python scripts/test_rare_catastrophe.py
    python scripts/test_rare_catastrophe.py --n-vars 10 --cycles 1000 --rare-prob 0.05
"""

import argparse
import random
from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent


def run_rare_detection(n_vars, cycles, seed, rare_prob, cost_weight, noise_sigma):
    rng_w = random.Random(seed)
    rng_a = random.Random(seed + 1)
    world = CausalWorld(n_vars, rng_w, noise_sigma=noise_sigma)
    world.visible_count = n_vars

    # High cost only on var 0 — the consequential var. Others use default (1.0).
    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=40,
        compression_discover_after=10,
        compression_promote_after=10,
        priority_audit_budget=n_vars,
        cost_weights={0: cost_weight},
    )
    agent.initialize()

    settle = 20
    for c in range(1, settle + 1):
        m = world.perturb_by_schedule(c, "rare_catastrophe",
                                      settle_cycles=settle,
                                      rare_var=0, rare_prob=0.0)
        agent.run_cycle(m)

    mutation_cycles = []
    detection_cycles = []

    for c in range(settle + 1, cycles + 1):
        m = world.perturb_by_schedule(c, "rare_catastrophe",
                                      settle_cycles=settle,
                                      rare_var=0, rare_prob=rare_prob)
        if m.rule_changed and m.affected_var == 0:
            mutation_cycles.append(c)

        agent.run_cycle(m)
        r = agent.records[-1]

        # detected_drift_vars: hypothesis changed after re-audit.
        # That is the only signal that the agent learned the causal relationship changed.
        # fully_audited_vars is NOT used — a var can be audited for cascade, deferred
        # queue, or initial cert reasons that have nothing to do with this mutation.
        if mutation_cycles and mutation_cycles[-1] not in detection_cycles:
            if 0 in r.detected_drift_vars:
                detection_cycles.append(c)

    latencies = [detection_cycles[i] - mutation_cycles[i]
                 for i in range(min(len(mutation_cycles), len(detection_cycles)))]
    undetected = len(mutation_cycles) - len(detection_cycles)

    return {
        "mutations": len(mutation_cycles),
        "detected": len(detection_cycles),
        "undetected": undetected,
        "latencies": latencies,
        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
    }


def _cert_evidence(cert):
    """The meaningful content of a cert: role and what evidence earned it."""
    if cert is None:
        return None
    return (cert.role, cert.changes, cert.trials)


def run_anti_forgetting(n_vars, cycles, seed):
    """Certify a world with zero noise, run many quiet cycles, check cert stability.
    Tracks cert evidence (role, changes, trials) — not Python object identity,
    which changes on every dataclasses.replace regardless of content change."""
    rng_w = random.Random(seed)
    rng_a = random.Random(seed + 1)
    world = CausalWorld(n_vars, rng_w, noise_sigma=0.0)
    world.visible_count = n_vars

    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=40,
        compression_discover_after=999,
        compression_promote_after=999,
        priority_audit_budget=n_vars,
    )
    agent.initialize()

    steady = lambda c: HiddenMutation(c, "VALUE", "steady", False, -1)
    for c in range(1, 21):
        agent.run_cycle(steady(c))

    snapshot = {v: _cert_evidence(agent.ledger.vars[v].certificates.get("skip"))
                for v in range(n_vars)}

    evidence_changes = 0
    change_log = []
    for c in range(21, cycles + 1):
        agent.run_cycle(steady(c))
        for v in range(n_vars):
            ev = _cert_evidence(agent.ledger.vars[v].certificates.get("skip"))
            if ev != snapshot[v]:
                evidence_changes += 1
                change_log.append((c, v, snapshot[v], ev))
                snapshot[v] = ev

    return {
        "quiet_cycles": cycles - 20,
        "evidence_changes": evidence_changes,
        "change_log": change_log[:10],
    }


def run_inversion(n_vars, seed, noise_sigma):
    """Show the inversion: default is full attention, trass requires evidence.

    All vars visible from the start. After initialize(), every var has been
    through a perturbation test. Show the verdict and the evidence that earned it.
    No var skips its first cycle — untested vars are audited.
    """
    rng_w = random.Random(seed)
    rng_a = random.Random(seed + 1)
    world = CausalWorld(n_vars, rng_w, noise_sigma=noise_sigma)
    world.visible_count = n_vars

    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=40,
        compression_discover_after=999,
        compression_promote_after=999,
        priority_audit_budget=n_vars,
    )

    # Before initialize: all vars are untested (no cert exists)
    untested_before = sum(
        1 for v in range(n_vars)
        if agent.ledger.vars[v].role_for("skip") == "untested"
    )

    agent.initialize()

    # After initialize: every var has been through the perturbation test
    results = []
    for v in range(n_vars):
        cert = agent.ledger.vars[v].certificates.get("skip")
        results.append({
            "var": v,
            "role": cert.role if cert else "untested",
            "changes": cert.changes if cert else None,
            "trials": cert.trials if cert else None,
        })

    return untested_before, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-vars",      type=int,   default=8)
    p.add_argument("--cycles",      type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--rare-prob",   type=float, default=0.03)
    p.add_argument("--cost-weight", type=float, default=3.0)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    args = p.parse_args()

    print("=" * 60)
    print("1. RARE EVENT DETECTION")
    print(f"   n={args.n_vars} cycles={args.cycles} seed={args.seed} "
          f"rare_prob={args.rare_prob} x0.cost_weight={args.cost_weight}")
    print("=" * 60)
    rd = run_rare_detection(args.n_vars, args.cycles, args.seed,
                            args.rare_prob, args.cost_weight, args.noise_sigma)
    print(f"  mutations:           {rd['mutations']}")
    print(f"  detected:            {rd['detected']}")
    print(f"  undetected:          {rd['undetected']}")
    print(f"  detection latencies: {rd['latencies']}")
    print(f"  avg latency:         {rd['avg_latency']}")
    print(f"  max latency:         {rd['max_latency']}")

    print()
    print("=" * 60)
    print("2. ANTI-FORGETTING")
    print(f"   n={args.n_vars} quiet_cycles={args.cycles} seed={args.seed} noise=0.0")
    print("=" * 60)
    af = run_anti_forgetting(args.n_vars, args.cycles, args.seed)
    print(f"  quiet cycles after cert:  {af['quiet_cycles']}")
    print(f"  cert evidence changes:    {af['evidence_changes']}")
    if af["change_log"]:
        for c, v, old, new in af["change_log"]:
            print(f"    c{c} x{v}: {old} → {new}")
    print(f"  (zero = no cert evidence changed without a failure)")

    print()
    print("=" * 60)
    print("3. INVERSION — default is watch, trass requires evidence")
    print(f"   n={args.n_vars} seed={args.seed}")
    print("=" * 60)
    untested_before, inv = run_inversion(args.n_vars, args.seed, args.noise_sigma)
    print(f"  vars untested before initialize(): {untested_before}/{args.n_vars}")
    print(f"  (no var can skip before the perturbation test runs)")
    print()
    print(f"  {'var':>4}  {'role':>8}  {'evidence':>10}  meaning")
    print("  " + "-" * 55)
    for r in inv:
        ev = f"{r['changes']}/{r['trials']}" if r["changes"] is not None else "none"
        meaning = ("skip — 0 propagating perturbations out of tested"
                   if r["role"] == "trass" else
                   "watch — perturbation propagated to monitored target"
                   if r["role"] == "tareth" else
                   "watch — deferred, not enough other vars to test")
        print(f"  x{r['var']:>3}  {r['role']:>8}  {ev:>10}  {meaning}")
    print()
    print(f"  tareth: {sum(1 for r in inv if r['role'] == 'tareth')}")
    print(f"  trass:  {sum(1 for r in inv if r['role'] == 'trass')}")
    print(f"  untested (deferred): {sum(1 for r in inv if r['role'] == 'untested')}")


if __name__ == "__main__":
    main()
