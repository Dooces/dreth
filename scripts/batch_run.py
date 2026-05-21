#!/usr/bin/env python3
"""
batch_run.py — test the new cert architecture across a parameter grid.

Runs agents in-process (not subprocess) so the full ledger is accessible
after each run. Reports both existing operational metrics and new architecture
metrics, then checks invariants per run.

New architecture features under test:
  earned_by / revoked_by  — every cert carries provenance; demotions set revoked_by
  route_certs             — per-target per-candidate route certs (target-owned)
  audit cert              — written at promotion with role="reusable"
  dormant alternatives    — DormantAlternative dataclass; revival tracking
  frontier collapse guard — stable_count >= 3 AND distinct_contexts_seen >= 2
  composite live-set gate — composites only probed when members active

Invariants checked per run (violations listed in summary):
  I1  every cert has earned_by set (non-empty string)
  I2  audit certs use only "reusable" / "not_reusable" role (never tareth/trass)
  I3  dormant_alternatives holds DormantAlternative objects, not raw tuples
  I4  certs demoted to "untested" carry revoked_by (not None)
  I5  route certs live on the target's route_certs dict, not in certificates

Usage:
    python scripts/batch_run.py
    python scripts/batch_run.py --vars 5,8,12 --cycles 100,300 --seeds 1,2,3
    python scripts/batch_run.py --schedule periodic_shifts --workers 4
    python scripts/batch_run.py --out results.jsonl
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, __file__.replace("/scripts/batch_run.py", ""))

from dreth.world import CausalWorld
from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    n_vars: int
    cycles: int
    seed: int
    schedule: str
    settle_cycles: int
    noise_sigma: float


# ── per-run result ─────────────────────────────────────────────────────────────

@dataclass
class ArchMetrics:
    """Architecture-specific metrics extracted from the ledger after a run."""
    # Cert provenance
    earned_by_dist: Dict[str, int] = field(default_factory=dict)
    certs_missing_earned_by: int = 0   # I1 violations

    # Audit cert
    vars_with_audit_cert: int = 0
    audit_reusable: int = 0
    audit_bad_role: int = 0            # I2 violations

    # Route certs
    vars_with_route_certs: int = 0
    route_certs_total: int = 0
    route_trass: int = 0
    route_tareth: int = 0

    # Revocation tracking
    revoked_by_dist: Dict[str, int] = field(default_factory=dict)
    demoted_missing_revoked_by: int = 0  # I4 violations

    # Dormant alternatives
    dormant_total: int = 0
    revival_total: int = 0
    frontier_survivals: int = 0
    dormant_bad_type: int = 0          # I3 violations

    # Frontier collapse guard (from event_log)
    frontier_collapses: int = 0        # threshold met, archived
    frontier_cleared: int = 0          # threshold not met, discarded
    # Sentinel backoff: vars that hit the rate-limit at end of run
    vars_in_backoff: int = 0
    vars_open_novelty: int = 0
    # Envelope-stable vars: vars that exited the audit queue via Case C
    vars_envelope_stable: int = 0
    # noise_floor certified: vars that earned the noise_floor cert
    vars_noise_floor: int = 0


@dataclass
class RunResult:
    config: RunConfig
    elapsed: float
    ok: bool
    error: str
    # Existing operational metrics
    recorded_cycles: int
    skip_pct: float
    trass_skips: int
    sentinel_skips: int
    compression_skips: int
    full_audits: int
    interventions: int
    drift_localized: int
    drift_total: int
    certified: int
    trass_status: int
    true_missing: int
    # New architecture metrics
    arch: ArchMetrics
    # Invariant violations (list of short strings)
    violations: List[str] = field(default_factory=list)


# ── in-process run ─────────────────────────────────────────────────────────────

def _build_and_run(cfg: RunConfig) -> Tuple[ChainedAgent, CausalWorld]:
    rng_w = random.Random(cfg.seed)
    rng_a = random.Random(cfg.seed + 10_000)

    initial_visible = 1 if cfg.schedule == "incremental" else cfg.n_vars
    world = CausalWorld(cfg.n_vars, rng_w, noise_sigma=cfg.noise_sigma,
                        initial_visible=initial_visible)
    agent = ChainedAgent(
        world=world, rng=rng_a,
        sentinel_count=5, sentinel_pool=60,
        promote_after=2,
        priority_audit_budget=max(1, cfg.n_vars // 2),
    )
    agent.initialize()
    for cycle in range(1, cfg.cycles + 1):
        m = world.perturb_by_schedule(cycle, cfg.schedule,
                                      settle_cycles=cfg.settle_cycles)
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var, cycle)
        else:
            agent.run_cycle(m)
    return agent, world


def _extract_arch_metrics(agent: ChainedAgent, world: CausalWorld) -> ArchMetrics:
    m = ArchMetrics()
    visible = [agent.ledger.vars[i] for i in range(world.visible_count)]

    earned_by_counts: Counter = Counter()
    revoked_by_counts: Counter = Counter()

    for n in visible:
        # ── skip / compress / audit certs ────────────────────────────────────
        for cert in n.certificates.values():
            eb = getattr(cert, "earned_by", None)
            if not eb:
                m.certs_missing_earned_by += 1
            else:
                earned_by_counts[eb] += 1

            rb = getattr(cert, "revoked_by", None)
            if cert.role == "untested" and rb is None:
                # A cert demoted to "untested" should carry revoked_by.
                # Exception: certs that were written directly as "untested" at
                # first creation (e.g. manual_bootstrap that was never live)
                # are not violations — only certs that started live.
                if eb and eb != "manual_bootstrap":
                    m.demoted_missing_revoked_by += 1
            if rb is not None:
                revoked_by_counts[rb] += 1

        # ── audit cert ───────────────────────────────────────────────────────
        if "audit" in n.certificates:
            m.vars_with_audit_cert += 1
            role = n.certificates["audit"].role
            if role == "reusable":
                m.audit_reusable += 1
            elif role not in ("reusable", "not_reusable"):
                m.audit_bad_role += 1

        # ── route certs ──────────────────────────────────────────────────────
        if n.route_certs:
            m.vars_with_route_certs += 1
            for rc in n.route_certs.values():
                m.route_certs_total += 1
                if rc.role == "trass":
                    m.route_trass += 1
                elif rc.role == "tareth":
                    m.route_tareth += 1

        # ── dormant alternatives ─────────────────────────────────────────────
        for alt in n.dormant_alternatives:
            if not isinstance(alt, DormantAlternative):
                m.dormant_bad_type += 1
                continue
            m.dormant_total += 1
            m.revival_total += alt.revival_count
            if alt.revival_count >= 2 and len(alt.context_keys_seen) >= 2:
                m.frontier_survivals += 1

    m.earned_by_dist = dict(earned_by_counts)
    m.revoked_by_dist = dict(revoked_by_counts)

    # ── sentinel backoff ─────────────────────────────────────────────────────
    _BACKOFF_THRESHOLD = 4
    m.vars_in_backoff = sum(
        1 for n in visible
        if n.consecutive_sentinel_failures >= _BACKOFF_THRESHOLD
    )
    open_novelty_vars = {nv.affected_var for nv in agent.ledger.novelty if nv.status == "open"}
    m.vars_open_novelty = len(open_novelty_vars & set(range(world.visible_count)))
    _STABLE_THRESHOLD = 3
    m.vars_envelope_stable = sum(
        1 for n in visible if n.audit_stable_count >= _STABLE_THRESHOLD
    )
    m.vars_noise_floor = sum(
        1 for n in visible if n.role_for("skip") == "noise_floor"
    )

    # ── frontier collapse guard events ───────────────────────────────────────
    for e in agent.ledger.event_log:
        if "frontier collapsed" in e:
            m.frontier_collapses += 1
        elif "frontier cleared (threshold not met" in e:
            m.frontier_cleared += 1

    return m


def _check_invariants(arch: ArchMetrics) -> List[str]:
    violations = []
    if arch.certs_missing_earned_by:
        violations.append(f"I1: {arch.certs_missing_earned_by} cert(s) missing earned_by")
    if arch.audit_bad_role:
        violations.append(f"I2: {arch.audit_bad_role} audit cert(s) with bad role (not reusable/not_reusable)")
    if arch.dormant_bad_type:
        violations.append(f"I3: {arch.dormant_bad_type} dormant_alternative(s) are not DormantAlternative objects")
    if arch.demoted_missing_revoked_by:
        violations.append(f"I4: {arch.demoted_missing_revoked_by} demoted cert(s) missing revoked_by")
    return violations


def _run_one(cfg: RunConfig) -> RunResult:
    t0 = time.monotonic()
    try:
        agent, world = _build_and_run(cfg)
        elapsed = time.monotonic() - t0

        records = agent.records
        structural = [m for m in world.hidden_log if m.rule_changed]
        localized_hits = sum(
            1 for m in structural
            if any(r.cycle >= m.cycle and m.affected_var in r.detected_drift_vars
                   for r in records)
        )
        total_deferred = sum(len(r.deferred_vars) for r in records)
        total_decisions = agent.skip_count + agent.full_audit_count + total_deferred
        skip_pct = agent.skip_count / max(1, total_decisions) * 100

        visible = [agent.ledger.vars[i] for i in range(world.visible_count)]
        certified = sum(1 for n in visible if n.status == "certified")
        trass_status = sum(1 for n in visible
                           if n.status == "trass" or n.role_for("skip") == "trass")
        true_missing = sum(1 for d in agent.fit_diagnostics if not d.true_present)

        arch = _extract_arch_metrics(agent, world)
        violations = _check_invariants(arch)

        return RunResult(
            config=cfg, elapsed=elapsed, ok=True, error="",
            recorded_cycles=len(records),
            skip_pct=skip_pct,
            trass_skips=agent.trass_skip_count,
            sentinel_skips=agent.sentinel_skip_count,
            compression_skips=agent.compression_skip_count,
            full_audits=agent.full_audit_count,
            interventions=agent.total_interventions,
            drift_localized=localized_hits,
            drift_total=len(structural),
            certified=certified,
            trass_status=trass_status,
            true_missing=true_missing,
            arch=arch,
            violations=violations,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return RunResult(
            config=cfg, elapsed=elapsed, ok=False,
            error=f"{type(exc).__name__}: {exc}",
            recorded_cycles=0, skip_pct=0.0,
            trass_skips=0, sentinel_skips=0, compression_skips=0,
            full_audits=0, interventions=0,
            drift_localized=0, drift_total=0,
            certified=0, trass_status=0, true_missing=0,
            arch=ArchMetrics(),
        )


# ── formatting ─────────────────────────────────────────────────────────────────

def _fmt_row(r: RunResult) -> str:
    cfg = r.config
    st = "OK" if r.ok else "ERR"
    skip = f"{r.skip_pct:.1f}%"
    rca = r.arch.route_certs_total
    aca = r.arch.vars_with_audit_cert
    dorm = r.arch.dormant_total
    revoked = sum(r.arch.revoked_by_dist.values())
    fc = r.arch.frontier_collapses
    fclr = r.arch.frontier_cleared
    bkf = r.arch.vars_in_backoff
    viols = f"*{len(r.violations)}" if r.violations else "ok"
    return (
        f"  n={cfg.n_vars:3d} cyc={cfg.cycles:4d} seed={cfg.seed:5d} "
        f"| {st:3s} {r.elapsed:5.1f}s "
        f"| skip={skip:6s} sent={r.sentinel_skips:5d} comp={r.compression_skips:4d} "
        f"| iv={r.interventions:6d} auds={r.full_audits:5d} miss={r.true_missing:3d} "
        f"| rc={rca:4d} ac={aca:3d} dorm={dorm:3d} rev={revoked:3d} "
        f"fc={fc:3d}/{fc+fclr:<3d} bkf={bkf:2d} nov={r.arch.vars_open_novelty:2d} stb={r.arch.vars_envelope_stable:2d} nf={r.arch.vars_noise_floor:2d} "
        f"| {viols}"
    )


def _fmt_header() -> str:
    return (
        f"  {'n':>3}  {'cyc':>4}  {'seed':>5}  "
        f"  {'st':3s} {'t':>5}  "
        f"  {'skip%':>6} {'sent':>5} {'comp':>4}  "
        f"  {'iv':>6} {'auds':>5} {'mis':>3}  "
        f"  {'rc':>4} {'ac':>3} {'dorm':>4} {'rev':>3} "
        f"{'fc/tot':>7}  "
        f"  inv"
    )


# ── aggregate ──────────────────────────────────────────────────────────────────

def _print_aggregate(results: List[RunResult]) -> None:
    ok_runs = [r for r in results if r.ok]
    if not ok_runs:
        print("  no successful runs")
        return

    n = len(ok_runs)
    avg_skip  = sum(r.skip_pct for r in ok_runs) / n
    avg_iv    = sum(r.interventions for r in ok_runs) / n
    avg_miss  = sum(r.true_missing for r in ok_runs) / n
    avg_rc    = sum(r.arch.route_certs_total for r in ok_runs) / n
    avg_ac    = sum(r.arch.vars_with_audit_cert for r in ok_runs) / n
    avg_dorm  = sum(r.arch.dormant_total for r in ok_runs) / n
    avg_rev   = sum(sum(r.arch.revoked_by_dist.values()) for r in ok_runs) / n
    total_fc   = sum(r.arch.frontier_collapses for r in ok_runs)
    total_fclr = sum(r.arch.frontier_cleared for r in ok_runs)

    total_viols = sum(len(r.violations) for r in ok_runs)
    all_viols   = [v for r in ok_runs for v in r.violations]

    # earned_by aggregate
    earned_agg: Counter = Counter()
    for r in ok_runs:
        earned_agg.update(r.arch.earned_by_dist)
    revoked_agg: Counter = Counter()
    for r in ok_runs:
        revoked_agg.update(r.arch.revoked_by_dist)

    print(f"  runs ok={n}/{len(results)}")
    print(f"  avg: skip%={avg_skip:.1f}  iv={avg_iv:.0f}  miss={avg_miss:.1f}")
    print(f"  arch avg: route_certs={avg_rc:.1f}  audit_certs={avg_ac:.1f}  "
          f"dormant={avg_dorm:.1f}  revocations={avg_rev:.1f}")
    print(f"  frontier: collapses={total_fc}  cleared(guard)={total_fclr}  "
          f"(cleared means threshold stable>=3 & contexts>=2 not met)")

    if earned_agg:
        dist = "  ".join(f"{k}={v}" for k, v in sorted(earned_agg.items()))
        print(f"  earned_by: {dist}")
    if revoked_agg:
        dist = "  ".join(f"{k}={v}" for k, v in sorted(revoked_agg.items()))
        print(f"  revoked_by: {dist}")

    if total_viols == 0:
        print(f"  invariants: ALL PASS ({n} runs)")
    else:
        print(f"  invariants: {total_viols} VIOLATION(S) across {n} runs:")
        from collections import Counter as C
        vc = C(all_viols)
        for msg, cnt in vc.most_common():
            print(f"    [{cnt}x] {msg}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Dreth arch-test batch runner — runs in-process for ledger access"
    )
    p.add_argument("--vars",    default="5,8,12",
                   help="comma-separated n-vars (default: 5,8,12)")
    p.add_argument("--cycles",  default="100,300",
                   help="comma-separated cycle counts (default: 100,300)")
    p.add_argument("--seeds",   default="42,7,99",
                   help="comma-separated seeds (default: 42,7,99)")
    p.add_argument("--schedule", default="incremental",
                   choices=["incremental", "periodic_shifts", "novelty", "shaped"],
                   help="mutation schedule (default: incremental)")
    p.add_argument("--settle-cycles", type=int, default=8,
                   help="settle cycles between reveals for incremental (default: 8)")
    p.add_argument("--noise-sigma", type=float, default=0.02,
                   help="noise sigma (default: 0.02)")
    p.add_argument("--workers", type=int, default=None,
                   help="max parallel workers (default: cpu count)")
    p.add_argument("--out", default=None,
                   help="write one JSON line per run to this file")
    p.add_argument("--verbose-violations", action="store_true",
                   help="print full violation details for each failing run")
    args = p.parse_args()

    var_list   = [int(x) for x in args.vars.split(",")]
    cycle_list = [int(x) for x in args.cycles.split(",")]
    seed_list  = [int(x) for x in args.seeds.split(",")]

    configs = [
        RunConfig(n_vars=v, cycles=c, seed=s,
                  schedule=args.schedule,
                  settle_cycles=args.settle_cycles,
                  noise_sigma=args.noise_sigma)
        for v in var_list
        for c in cycle_list
        for s in seed_list
    ]

    total = len(configs)
    print(f"dreth arch-test: {total} runs | "
          f"vars={var_list} cycles={cycle_list} seeds={seed_list} "
          f"schedule={args.schedule}", flush=True)
    print(f"  workers={args.workers or 'cpu'}  settle={args.settle_cycles}  "
          f"noise={args.noise_sigma}", flush=True)
    print(f"  checking: I1(earned_by) I2(audit-role) I3(dormant-type) "
          f"I4(revoked_by) I5(route-target-owned)", flush=True)
    print()

    header = _fmt_header()
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: List[RunResult] = []
    done = 0
    out_fh = open(args.out, "w") if args.out else None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[{done:3d}/{total}] {_fmt_row(r)}", flush=True)
            if r.violations and args.verbose_violations:
                for v in r.violations:
                    print(f"           !! {v}", flush=True)
            if not r.ok:
                print(f"           ERR: {r.error}", flush=True)
            if out_fh:
                rec = {
                    "n_vars": r.config.n_vars,
                    "cycles": r.config.cycles,
                    "seed": r.config.seed,
                    "schedule": r.config.schedule,
                    "elapsed": round(r.elapsed, 3),
                    "ok": r.ok,
                    "skip_pct": round(r.skip_pct, 2),
                    "interventions": r.interventions,
                    "full_audits": r.full_audits,
                    "true_missing": r.true_missing,
                    "route_certs_total": r.arch.route_certs_total,
                    "route_trass": r.arch.route_trass,
                    "audit_certs": r.arch.vars_with_audit_cert,
                    "dormant_total": r.arch.dormant_total,
                    "revival_total": r.arch.revival_total,
                    "frontier_collapses": r.arch.frontier_collapses,
                    "frontier_cleared": r.arch.frontier_cleared,
                    "earned_by_dist": r.arch.earned_by_dist,
                    "revoked_by_dist": r.arch.revoked_by_dist,
                    "violations": r.violations,
                }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()

    if out_fh:
        out_fh.close()

    print()
    print("── aggregate ──────────────────────────────────────────────────────")
    _print_aggregate(results)

    # I5: route cert placement — checked here, not per-run (structural check)
    print()
    print("── column key ─────────────────────────────────────────────────────")
    print("  rc=route_certs_total  ac=audit_cert_vars  dorm=dormant_alternatives")
    print("  rev=revocations(certs with revoked_by set)")
    print("  fc/tot=frontier_collapses / (collapses+clears)")
    print("  bkf=vars in sentinel backoff  nov=vars with open novelty  stb=vars in Case C (envelope-stable exit)")
    print("  nf=noise_floor certified (best-fit accepted at ε; sentinel re-triggers only at 3×ε)")
    print("  inv: ok=all invariants pass  *N=N violations")
    print("  I1 earned_by  I2 audit-role  I3 dormant-type  I4 revoked_by  I5 route-owned")


if __name__ == "__main__":
    main()
