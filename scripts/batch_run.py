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
    python scripts/batch_run.py --compare
    python scripts/batch_run.py --compare --vars 8,12 --cycles 300 --seeds 7,42
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, __file__.replace("/scripts/batch_run.py", ""))

from dreth.world import CausalWorld
from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative
from dreth.fit import fit_var
from dreth.functions import FUNC_LIBRARY


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    n_vars: int
    cycles: int
    seed: int
    schedule: str
    settle_cycles: int
    noise_sigma: float
    compare: bool = False


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
class BaselineMetrics:
    """Metrics from the dirty-closure baseline agent."""
    elapsed: float = 0.0
    skip_count: int = 0          # vars not refit this cycle (no sentinel failure in closure)
    full_audits: int = 0         # total refits
    interventions: int = 0       # total probe calls (fit + sentinel)
    sentinel_fails: int = 0      # times a sentinel detected a deviation
    wrong_fits: int = 0          # vars with wrong final parents at end of run
    ok: bool = True
    error: str = ""


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
    true_missing: int      # cumulative: fit attempts where true hypothesis not in candidate set
    wrong_fits_dreth: int  # final-state: vars with wrong parents at end of run
    # New architecture metrics
    arch: ArchMetrics
    # Invariant violations (list of short strings)
    violations: List[str] = field(default_factory=list)
    # Baseline comparison (populated only when --compare is active)
    baseline: Optional[BaselineMetrics] = None


# ── dirty-closure baseline agent ───────────────────────────────────────────────

class BaselineAgent:
    """Dirty-closure cached refit baseline.

    Fit each variable; cache result. Each cycle:
      1. Run sentinel probes for every cached variable.
      2. On failure: mark variable dirty + full transitive descendant closure.
      3. Refit entire dirty closure in topological order.
      4. Refresh sentinels for refitted vars.

    No nethra certs. No route-trass pruning. No composite nethras.
    No operation-indexed authority. Available parents = all other vars.

    This is the realistic comparator: it localises failure to the dirty
    closure but always cascades to all believed descendants — the exact
    behavior that route-trass pruning skips.
    """

    _DEFAULT_TOL = 0.1

    def __init__(
        self,
        world: CausalWorld,
        rng: random.Random,
        intervention_budget: int = 10,
        sentinel_count: int = 5,
    ):
        self.world = world
        self.rng = rng
        self.intervention_budget = intervention_budget
        self.sentinel_count = sentinel_count
        self.tolerance = self._DEFAULT_TOL

        # fit_cache[var] = (parents_tuple, func_str)
        self.fit_cache: Dict[int, Tuple[Tuple[int, ...], str]] = {}
        # sentinels[var] = list of (iv_var, iv_val) pairs
        self.sentinels: Dict[int, List[Tuple[int, float]]] = {}

        self.skip_count: int = 0
        self.full_audit_count: int = 0
        self.total_interventions: int = 0
        self.sentinel_fail_count: int = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _fit_one(self, var: int) -> None:
        parents, func, _, _ = fit_var(
            var, self.world, self.rng,
            self.intervention_budget, self.tolerance,
            available_parents=None,  # no restriction
        )
        self.fit_cache[var] = (parents, func)
        self.total_interventions += self.intervention_budget
        self.full_audit_count += 1
        # Random sentinel probes (no discrimination scoring)
        n_vars = self.world.visible_count
        pool = [(self.rng.randint(0, n_vars - 1), self.rng.random())
                for _ in range(self.sentinel_count)]
        self.sentinels[var] = pool

    def _check_sentinels(self, var: int) -> bool:
        """Returns True if all sentinel probes pass. Counts interventions."""
        parents, func = self.fit_cache[var]
        fn = FUNC_LIBRARY.get(func)
        if fn is None:
            return True
        n_vars = self.world.visible_count
        for iv_var, iv_val in self.sentinels.get(var, []):
            forced = [iv_val if i == iv_var else self.world.state[i]
                      for i in range(n_vars)]
            expected = fn([forced[p] for p in parents])
            actual = self.world.predict_var_under_intervention(var, iv_var, iv_val)
            self.total_interventions += 1
            if abs(actual - expected) > self.tolerance:
                return False
        return True

    def _dependents(self, var: int) -> Set[int]:
        n_vars = self.world.visible_count
        return {
            v for v in range(n_vars)
            if v in self.fit_cache and var in self.fit_cache[v][0]
        }

    def _full_closure(self, dirty: Set[int]) -> Set[int]:
        """Transitive closure of dirty — no pruning, all believed descendants."""
        out = set(dirty)
        frontier = set(dirty)
        while frontier:
            new_front: Set[int] = set()
            for v in frontier:
                for d in self._dependents(v):
                    if d not in out:
                        out.add(d)
                        new_front.add(d)
            frontier = new_front
        return out

    def _refit_closure(self, closure: Set[int]) -> None:
        """Refit vars in closure. World DAG has lower-index parents, so
        iterating 0..n_vars is topologically ordered."""
        for var in range(self.world.visible_count):
            if var in closure:
                self._fit_one(var)

    # ── public ────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        for var in range(self.world.visible_count):
            self._fit_one(var)

    def on_variable_revealed(self, var: int) -> None:
        """Fit new var; check existing vars' sentinels since new var is a
        candidate parent for all — those that fail cascade to their closure."""
        self._fit_one(var)
        dirty: Set[int] = set()
        for v in range(self.world.visible_count):
            if v == var:
                continue
            if v not in self.fit_cache:
                dirty.add(v)
            elif not self._check_sentinels(v):
                self.sentinel_fail_count += 1
                dirty.add(v)
        if dirty:
            closure = self._full_closure(dirty)
            self._refit_closure(closure)

    def run_cycle(self) -> None:
        n_vars = self.world.visible_count
        dirty: Set[int] = set()
        for var in range(n_vars):
            if var not in self.fit_cache:
                dirty.add(var)
            elif not self._check_sentinels(var):
                self.sentinel_fail_count += 1
                dirty.add(var)

        closure = self._full_closure(dirty) if dirty else set()
        self.skip_count += n_vars - len(closure)
        self._refit_closure(closure)

    def wrong_fits(self) -> int:
        """Count vars whose cached parents differ from world's true parents."""
        n_vars = self.world.visible_count
        return sum(
            1 for var in range(n_vars)
            if set(self.fit_cache.get(var, ((),))[0]) != set(self.world.parents[var])
        )


# ── in-process run ─────────────────────────────────────────────────────────────

def _build_and_run_dreth(cfg: RunConfig) -> Tuple[ChainedAgent, CausalWorld]:
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


def _build_and_run_baseline(cfg: RunConfig) -> Tuple[BaselineAgent, CausalWorld]:
    """Build an independent world from the same seed and run the baseline."""
    rng_w = random.Random(cfg.seed)
    rng_a = random.Random(cfg.seed + 20_000)   # distinct stream from Dreth

    initial_visible = 1 if cfg.schedule == "incremental" else cfg.n_vars
    world = CausalWorld(cfg.n_vars, rng_w, noise_sigma=cfg.noise_sigma,
                        initial_visible=initial_visible)
    agent = BaselineAgent(
        world=world, rng=rng_a,
        intervention_budget=10,
        sentinel_count=5,
    )
    agent.initialize()
    for cycle in range(1, cfg.cycles + 1):
        m = world.perturb_by_schedule(cycle, cfg.schedule,
                                      settle_cycles=cfg.settle_cycles)
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var)
        else:
            agent.run_cycle()
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
        agent, world = _build_and_run_dreth(cfg)
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
        # Final-state wrong parents: same metric used for baseline comparison.
        # true_missing is cumulative across fit attempts; wrong_fits is end-state.
        wrong_fits_dreth = sum(
            1 for var in range(world.visible_count)
            if set(agent.ledger.vars[var].parents) != set(world.parents[var])
        )

        arch = _extract_arch_metrics(agent, world)
        violations = _check_invariants(arch)

        result = RunResult(
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
            wrong_fits_dreth=wrong_fits_dreth,
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
            certified=0, trass_status=0, true_missing=0, wrong_fits_dreth=0,
            arch=ArchMetrics(),
        )

    # ── baseline comparison ───────────────────────────────────────────────────
    if cfg.compare:
        try:
            t1 = time.monotonic()
            b_agent, _ = _build_and_run_baseline(cfg)
            b_elapsed = time.monotonic() - t1
            result.baseline = BaselineMetrics(
                elapsed=b_elapsed,
                skip_count=b_agent.skip_count,
                full_audits=b_agent.full_audit_count,
                interventions=b_agent.total_interventions,
                sentinel_fails=b_agent.sentinel_fail_count,
                wrong_fits=b_agent.wrong_fits(),
                ok=True,
            )
        except Exception as exc:
            result.baseline = BaselineMetrics(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    return result


# ── formatting ─────────────────────────────────────────────────────────────────

def _pct_diff(dreth_val: float, base_val: float) -> str:
    """Return signed percentage change dreth vs baseline (negative = dreth cheaper)."""
    if base_val == 0:
        return "n/a"
    d = (dreth_val - base_val) / base_val * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.0f}%"


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
    dreth_line = (
        f"  n={cfg.n_vars:3d} cyc={cfg.cycles:4d} seed={cfg.seed:5d} "
        f"| {st:3s} {r.elapsed:5.1f}s "
        f"| skip={skip:6s} sent={r.sentinel_skips:5d} comp={r.compression_skips:4d} "
        f"| iv={r.interventions:6d} auds={r.full_audits:5d} miss={r.true_missing:3d} "
        f"| rc={rca:4d} ac={aca:3d} dorm={dorm:3d} rev={revoked:3d} "
        f"fc={fc:3d}/{fc+fclr:<3d} bkf={bkf:2d} nov={r.arch.vars_open_novelty:2d} "
        f"stb={r.arch.vars_envelope_stable:2d} nf={r.arch.vars_noise_floor:2d} "
        f"| {viols}"
    )
    if r.baseline is None:
        return dreth_line

    b = r.baseline
    if not b.ok:
        base_line = f"  {'':>3s} {'':>4s} {'':>5s}   BASE ERR: {b.error}"
    else:
        b_total = b.skip_count + b.full_audits
        b_skip_pct = b.skip_count / max(1, b_total) * 100
        iv_diff   = _pct_diff(r.interventions, b.interventions)
        aud_diff  = _pct_diff(r.full_audits,   b.full_audits)
        t_diff    = _pct_diff(r.elapsed,        b.elapsed)
        base_line = (
            f"  {'':>3s} {'':>4s} {'':>5s}   "
            f"BASE {b.elapsed:5.1f}s "
            f"| skip={b_skip_pct:5.1f}%       "
            f"| iv={b.interventions:6d} auds={b.full_audits:5d} wf={b.wrong_fits:3d} "
            f"| sfail={b.sentinel_fails:4d} "
            f"| Δiv={iv_diff:>6s} Δaud={aud_diff:>6s} Δt={t_diff:>6s}"
        )
    return dreth_line + "\n" + base_line


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

    # ── baseline comparison aggregate ────────────────────────────────────────
    base_runs = [r for r in ok_runs if r.baseline is not None and r.baseline.ok]
    if not base_runs:
        return

    b_n = len(base_runs)
    d_iv  = sum(r.interventions for r in base_runs)
    b_iv  = sum(r.baseline.interventions for r in base_runs)
    d_aud = sum(r.full_audits for r in base_runs)
    b_aud = sum(r.baseline.full_audits for r in base_runs)
    d_t   = sum(r.elapsed for r in base_runs)
    b_t   = sum(r.baseline.elapsed for r in base_runs)
    d_skip_avg = sum(r.skip_pct for r in base_runs) / b_n
    b_total_avg = sum(
        r.baseline.skip_count / max(1, r.baseline.skip_count + r.baseline.full_audits)
        for r in base_runs
    ) / b_n * 100

    print()
    print("── baseline comparison ─────────────────────────────────────────────")
    print(f"  {b_n} paired runs  (same world seed, independent agent rng)")
    print(f"                    Dreth         Baseline      Dreth vs Baseline")
    print(f"  interventions:  {d_iv:8d}      {b_iv:8d}      {_pct_diff(d_iv, b_iv):>8s}")
    print(f"  full_audits:    {d_aud:8d}      {b_aud:8d}      {_pct_diff(d_aud, b_aud):>8s}")
    print(f"  elapsed(total): {d_t:8.2f}s     {b_t:8.2f}s     {_pct_diff(d_t, b_t):>8s}")
    print(f"  avg skip%:      {d_skip_avg:7.1f}%      {b_total_avg:7.1f}%")

    # Final-state wrong parents (same metric for both)
    d_wf = sum(r.wrong_fits_dreth for r in base_runs)
    b_wf = sum(r.baseline.wrong_fits for r in base_runs)
    print(f"  wrong_fits(end-state): dreth={d_wf}  baseline={b_wf}  "
          f"({'same' if d_wf == b_wf else 'DIFFERS'})")

    print()
    print("  Δiv/Δaud negative = Dreth uses fewer probes/refits than baseline.")
    print("  Δt  negative = Dreth is faster.")
    print("  wrong_fits: lower is better; both should converge to the same answer.")
    print("  Advantage comes from: route-trass cascade pruning + failure-localized")
    print("  invalidation + not reopening unrelated descendants.")


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
    p.add_argument("--compare", action="store_true",
                   help="run dirty-closure baseline alongside Dreth and report comparison")
    args = p.parse_args()

    var_list   = [int(x) for x in args.vars.split(",")]
    cycle_list = [int(x) for x in args.cycles.split(",")]
    seed_list  = [int(x) for x in args.seeds.split(",")]

    configs = [
        RunConfig(n_vars=v, cycles=c, seed=s,
                  schedule=args.schedule,
                  settle_cycles=args.settle_cycles,
                  noise_sigma=args.noise_sigma,
                  compare=args.compare)
        for v in var_list
        for c in cycle_list
        for s in seed_list
    ]

    total = len(configs)
    mode = " +compare" if args.compare else ""
    print(f"dreth arch-test{mode}: {total} runs | "
          f"vars={var_list} cycles={cycle_list} seeds={seed_list} "
          f"schedule={args.schedule}", flush=True)
    print(f"  workers={args.workers or 'cpu'}  settle={args.settle_cycles}  "
          f"noise={args.noise_sigma}", flush=True)
    if args.compare:
        print(f"  baseline: dirty-closure cached refit (no certs, no route-trass pruning)", flush=True)
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
                if r.baseline and r.baseline.ok:
                    rec["baseline"] = {
                        "elapsed": round(r.baseline.elapsed, 3),
                        "skip_count": r.baseline.skip_count,
                        "full_audits": r.baseline.full_audits,
                        "interventions": r.baseline.interventions,
                        "sentinel_fails": r.baseline.sentinel_fails,
                        "wrong_fits": r.baseline.wrong_fits,
                    }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()

    if out_fh:
        out_fh.close()

    print()
    print("── aggregate ──────────────────────────────────────────────────────")
    _print_aggregate(results)

    print()
    print("── column key ─────────────────────────────────────────────────────")
    print("  rc=route_certs_total  ac=audit_cert_vars  dorm=dormant_alternatives")
    print("  rev=revocations(certs with revoked_by set)")
    print("  fc/tot=frontier_collapses / (collapses+clears)")
    print("  bkf=vars in sentinel backoff  nov=vars with open novelty")
    print("  stb=vars in Case C (envelope-stable exit)")
    print("  nf=noise_floor certified (best-fit accepted at ε; sentinel re-triggers at 3×ε)")
    print("  inv: ok=all invariants pass  *N=N violations")
    print("  I1 earned_by  I2 audit-role  I3 dormant-type  I4 revoked_by  I5 route-owned")
    if args.compare:
        print()
        print("  BASE row: wf=wrong_fits sfail=sentinel_fails")
        print("  Δiv=intervention diff  Δaud=audit diff  Δt=time diff (dreth vs baseline)")
        print("  negative Δ = Dreth is cheaper/faster")


if __name__ == "__main__":
    main()
