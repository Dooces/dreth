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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreth.world import CausalWorld
from dreth.agent import ChainedAgent
from dreth.ledger import DormantAlternative
from dreth.fit import fit_var
from dreth.functions import FUNC_LIBRARY
from dreth.hybrid import (
    SymbolicResidualPredictor,
    SensitivityParentRanker,
    DiscriminationProbeProposer,
    FuncLibraryRouter,
)
from dreth.learned_residual import ShadowLearnedResidualPredictor, OnlineResidualCalibrator


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
    ablate: bool = False
    log_interval: int = 0       # 0 = disabled; N = print progress every N cycles
    hybrid_control: str = "off" # "off" | "interfaces"
    repair_agenda_enabled: bool = False
    shadow_residual: str = "off"  # "off" | "online"


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

    # I5: route cert misplaced in certificates dict instead of route_certs
    bad_route_cert_location: int = 0

    # Composite nethra (nethra-of-nethra) handle metrics
    composite_skip_count: int = 0     # sentinel checks avoided by composite handle
    vars_under_composite: int = 0     # distinct vars covered by >= 1 composite
    active_composites: int = 0        # composites that passed their probe this run
    composite_revoked: int = 0        # composites revoked during the run
    composite_components: int = 0     # connected components in live composite graph
    composite_max_degree: int = 0     # highest composite membership count for any var
    composite_mean_degree: float = 0.0
    composite_duplicate_factor: float = 0.0  # raw_pair_passes / true_skip; >1 = overlap
    # HyperCompositeNethra (component) metrics
    component_live: int = 0
    component_revoked: int = 0
    component_members: int = 0
    component_skips: int = 0
    pairwise_fallbacks: int = 0
    duplicate_factor_before: float = 0.0
    duplicate_factor_after: float = 0.0

    # Regime-based sentinel amortization metrics
    regime_skip_count: int = 0        # leaf checks skipped when regime sentinel passed
    confirmed_regimes: int = 0        # confirmed regime signatures at end of run
    vars_under_regime: int = 0        # distinct vars covered by confirmed regimes
    # Regime sentinel pass/fail/no_sentinel (summed across all cycles in the run)
    regime_sentinel_passes: int = 0   # cycles where regime sentinel passed → rsk credited
    regime_sentinel_fails: int = 0    # cycles where regime sentinel failed → leaf checks ran
    regime_no_sentinel: int = 0       # cycles where regime had no active sentinel (annotate only)
    # Sentinel utility accounting (end-of-run snapshot from VarNethra fields)
    vars_with_unique_failures: int = 0   # vars whose leaf sentinel caught something higher missed
    vars_parkable: int = 0               # vars: covered by regime, 0 unique failures, ≥200 quiet cycles
    total_unique_failures: int = 0       # sum of unique_failures_caught across all vars
    total_higher_caught: int = 0         # sum of failures_also_caught_by_higher across all vars
    # Sentinel parking metrics
    parked_skip_count: int = 0           # sentinel checks skipped due to parking
    woken_count: int = 0                 # times a parked var was woken for revalidation
    # Passive residual monitoring metrics
    passive_saved_iv: int = 0            # IV calls saved by passive-OK skips
    passive_stress_count: int = 0        # var-cycles where passive was stressed

    # Shadow residual metrics (nonzero only when --shadow-residual online)
    shadow_residual_calls: int = 0
    shadow_residual_ok: int = 0
    shadow_residual_stressed: int = 0
    shadow_residual_insufficient: int = 0
    shadow_false_ok_vs_symbolic: int = 0
    shadow_false_stress_vs_symbolic: int = 0
    shadow_agree_symbolic: int = 0
    shadow_would_save_iv: int = 0
    shadow_would_miss_symbolic_stress: int = 0
    shadow_false_ok_vs_active_sentinel: int = 0
    shadow_would_miss_active_failure: int = 0

    # Hybrid control metrics (nonzero only when hybrid-control=interfaces)
    hybrid_residual_predictor_calls: int = 0
    hybrid_residual_ok: int = 0
    hybrid_residual_stressed: int = 0
    hybrid_parent_ranker_calls: int = 0
    hybrid_probe_proposer_calls: int = 0
    hybrid_expert_router_calls: int = 0
    hybrid_repair_agenda_items: int = 0
    hybrid_repair_agenda_scope_mean: float = 0.0
    hybrid_repair_agenda_scope_max: int = 0


@dataclass
class BaselineMetrics:
    """Metrics from the sparse_cached_refit baseline agent."""
    elapsed: float = 0.0
    skip_count: int = 0
    full_audits: int = 0
    interventions: int = 0
    sentinel_fails: int = 0
    candidate_refreshes: int = 0
    ok: bool = True
    error: str = ""


@dataclass
class TierMetrics:
    """Per-consequence-tier breakdown: var counts, avg sentinel count, avg cycles-to-cert."""
    n_t0: int = 0
    n_t1: int = 0
    n_t2: int = 0
    sent_t0: float = 0.0
    sent_t1: float = 0.0
    sent_t2: float = 0.0
    promo_t0: float = 0.0
    promo_t1: float = 0.0
    promo_t2: float = 0.0


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
    # New architecture metrics
    arch: ArchMetrics
    # Invariant violations (list of short strings)
    violations: List[str] = field(default_factory=list)
    # Baseline comparison (populated only when --compare is active)
    baseline: Optional[BaselineMetrics] = None
    # Consequence-weight tier breakdown (always populated)
    tier: TierMetrics = field(default_factory=TierMetrics)
    # Ablation: second run with CW disabled (populated only when cfg.ablate=True)
    tier_no_cw: Optional[TierMetrics] = None
    # Regime register summary (populated for all dreth runs)
    regime_summary: str = ""


# ── sparse-cached-refit baseline agent ────────────────────────────────────────

@dataclass
class SparseVarState:
    candidate_parents: List[int]
    parents: Tuple[int, ...]
    func: str
    residuals: List[float]
    last_refit_cycle: int
    refit_count: int
    candidate_refresh_count: int
    last_refresh_cycle: int


class SparseCachedRefitAgent:
    """Sparse-cached refit baseline (K=10, window=8, threshold=3).

    Per variable: maintains a top-K candidate parent set screened by
    intervention sensitivity (|predict(x=0.9) - predict(x=0.1)|). Each cycle,
    reads current world state to compute a residual. If the rolling window
    accumulates >= failure_threshold failures, refits using the candidate set.
    If still poor, refreshes the candidate set (rate-limited by
    candidate_refresh_interval) and refits again.

    No nethra certs. No route-trass pruning. No composite nethras.
    """

    K: int = 10
    VALIDATION_WINDOW: int = 8
    FAILURE_THRESHOLD: int = 3
    CANDIDATE_REFRESH_INTERVAL: int = 100
    _DEFAULT_TOL: float = 0.1

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
        self.tolerance = self._DEFAULT_TOL

        self._state: Dict[int, SparseVarState] = {}
        self._cycle: int = 0

        self.skip_count: int = 0
        self.full_audit_count: int = 0
        self.total_interventions: int = 0
        self.sentinel_fail_count: int = 0
        self.candidate_refresh_count: int = 0

    def _screen_candidates(self, y: int) -> List[int]:
        n = self.world.visible_count
        scores: List[Tuple[float, int]] = []
        for x in range(n):
            if x == y:
                continue
            lo = self.world.predict_var_under_intervention(y, x, 0.1)
            hi = self.world.predict_var_under_intervention(y, x, 0.9)
            self.total_interventions += 2
            scores.append((abs(hi - lo), x))
        scores.sort(reverse=True)
        return [x for _, x in scores[: self.K]]

    def _predict(self, y: int) -> float:
        vs = self._state[y]
        fn = FUNC_LIBRARY.get(vs.func)
        if fn is None or not vs.parents:
            return 0.0
        try:
            return fn([self.world.state[p] for p in vs.parents])
        except Exception:
            return 0.0

    def _do_refit(self, y: int, candidates: List[int]) -> None:
        available = set(candidates) if candidates else None
        parents, func, _, _ = fit_var(
            y, self.world, self.rng,
            self.intervention_budget, self.tolerance,
            available_parents=available,
        )
        self.total_interventions += self.intervention_budget
        self.full_audit_count += 1
        vs = self._state[y]
        vs.parents = tuple(parents)
        vs.func = func
        vs.last_refit_cycle = self._cycle
        vs.refit_count += 1

    def _current_residual(self, y: int) -> float:
        return abs(self.world.state[y] - self._predict(y))

    def _is_poor(self, y: int) -> bool:
        vs = self._state[y]
        window = vs.residuals[-self.VALIDATION_WINDOW:]
        return sum(1 for r in window if r > self.tolerance) >= self.FAILURE_THRESHOLD

    def _init_var(self, y: int) -> None:
        candidates = self._screen_candidates(y)
        self._state[y] = SparseVarState(
            candidate_parents=candidates,
            parents=(),
            func="",
            residuals=[],
            last_refit_cycle=0,
            refit_count=0,
            candidate_refresh_count=0,
            last_refresh_cycle=-self.CANDIDATE_REFRESH_INTERVAL,
        )
        self._do_refit(y, candidates)

    def initialize(self) -> None:
        for y in range(self.world.visible_count):
            self._init_var(y)

    def on_variable_revealed(self, var: int) -> None:
        self._init_var(var)
        for y in range(self.world.visible_count):
            if y == var or y not in self._state:
                continue
            candidates = self._screen_candidates(y)
            vs = self._state[y]
            vs.candidate_parents = candidates
            vs.candidate_refresh_count += 1
            vs.last_refresh_cycle = self._cycle
            self.candidate_refresh_count += 1
            self._do_refit(y, candidates)
            vs.residuals.clear()

    def run_cycle(self) -> None:
        self._cycle += 1
        for y in range(self.world.visible_count):
            if y not in self._state:
                self._init_var(y)
                continue
            vs = self._state[y]
            residual = self._current_residual(y)
            vs.residuals.append(residual)
            if len(vs.residuals) > self.VALIDATION_WINDOW:
                vs.residuals = vs.residuals[-self.VALIDATION_WINDOW:]
            if not self._is_poor(y):
                self.skip_count += 1
                continue
            self.sentinel_fail_count += 1
            self._do_refit(y, vs.candidate_parents)
            new_residual = self._current_residual(y)
            vs.residuals = [new_residual]
            if new_residual > self.tolerance:
                if self._cycle - vs.last_refresh_cycle >= self.CANDIDATE_REFRESH_INTERVAL:
                    new_candidates = self._screen_candidates(y)
                    vs.candidate_parents = new_candidates
                    vs.candidate_refresh_count += 1
                    vs.last_refresh_cycle = self._cycle
                    self.candidate_refresh_count += 1
                    self._do_refit(y, new_candidates)
                    vs.residuals = [self._current_residual(y)]


# ── in-process run ─────────────────────────────────────────────────────────────

def _compute_tier_metrics(agent: ChainedAgent, world: CausalWorld) -> TierMetrics:
    """Bucket visible vars by consequence tier and collect per-tier stats."""
    tm = TierMetrics()
    n_buckets  = [0, 0, 0]
    sent_sums  = [0.0, 0.0, 0.0]
    promo_sums = [0.0, 0.0, 0.0]
    promo_cnts = [0, 0, 0]

    for v in range(world.visible_count):
        t = min(agent._consequence_tier(v), 2)
        n = agent.ledger.vars[v]
        n_buckets[t] += 1
        sent_sums[t] += len(n.sentinels)
        fc = getattr(n, "first_certified_cycle", None)
        if fc is not None:
            promo_sums[t] += fc
            promo_cnts[t] += 1

    tm.n_t0, tm.n_t1, tm.n_t2 = n_buckets
    tm.sent_t0 = sent_sums[0] / max(1, n_buckets[0])
    tm.sent_t1 = sent_sums[1] / max(1, n_buckets[1])
    tm.sent_t2 = sent_sums[2] / max(1, n_buckets[2])
    tm.promo_t0 = promo_sums[0] / max(1, promo_cnts[0])
    tm.promo_t1 = promo_sums[1] / max(1, promo_cnts[1])
    tm.promo_t2 = promo_sums[2] / max(1, promo_cnts[2])
    return tm


def _build_and_run_dreth(
    cfg: RunConfig,
    consequence_weight: bool = True,
    log_interval: int = 0,
    log_tag: str = "",
    agent_seed_offset: int = 0,
) -> Tuple[ChainedAgent, CausalWorld]:
    """Returns (agent, world).

    When log_interval > 0, prints a one-line status every log_interval cycles.
    log_tag prefixes each line (useful to distinguish CW ON vs OFF).

    agent_seed_offset shifts rng_a independently of rng_w, so two runs on the
    same world (same rng_w seed) can have independent agent randomness.

    Hybrid control:
      off        — no providers installed; behavior identical to pre-hybrid.
      interfaces — symbolic default providers installed; behavior preserved
                   (SymbolicResidualPredictor reproduces inline path exactly).
    """
    rng_w = random.Random(cfg.seed)
    rng_a = random.Random(cfg.seed + 10_000 + agent_seed_offset)

    initial_visible = 1 if cfg.schedule == "incremental" else cfg.n_vars
    world = CausalWorld(cfg.n_vars, rng_w, noise_sigma=cfg.noise_sigma,
                        initial_visible=initial_visible)
    # Pre-install schedule-specific subgraph so agent.initialize() audits the
    # intended world structure, not the random base.
    world.prepare_schedule(cfg.schedule, cfg.settle_cycles)

    # Hybrid provider setup: only install when hybrid_control != "off".
    # In "off" mode no providers are created — zero overhead, identical behavior.
    # In "interfaces" mode all four symbolic default providers are installed;
    # they reproduce existing behavior so metrics remain compatible.
    _residual_predictor = None
    _parent_ranker = None
    _probe_proposer = None
    _expert_router = None
    if cfg.hybrid_control == "interfaces":
        _residual_predictor = SymbolicResidualPredictor()
        _parent_ranker = SensitivityParentRanker(world)
        _probe_proposer = DiscriminationProbeProposer()
        _expert_router = FuncLibraryRouter()

    _shadow_predictor = None
    _shadow_enabled = False
    if cfg.shadow_residual == "online":
        _shadow_predictor = ShadowLearnedResidualPredictor(OnlineResidualCalibrator())
        _shadow_enabled = True

    agent = ChainedAgent(
        world=world, rng=rng_a,
        sentinel_count=5, sentinel_pool=60,
        promote_after=2,
        priority_audit_budget=max(1, cfg.n_vars // 2),
        consequence_weight=consequence_weight,
        residual_predictor=_residual_predictor,
        parent_ranker=_parent_ranker,
        probe_proposer=_probe_proposer,
        expert_router=_expert_router,
        repair_agenda_enabled=cfg.repair_agenda_enabled,
        shadow_residual_predictor=_shadow_predictor,
        shadow_residual_enabled=_shadow_enabled,
    )
    agent.initialize()

    prev_iv = 0
    prev_sent_skip = 0

    def _snap_parents() -> Dict[int, tuple]:
        return {v: agent.ledger.vars[v].parents for v in range(world.visible_count)}

    for cycle in range(1, cfg.cycles + 1):
        if log_interval > 0:
            pre_parents = _snap_parents()

        m = world.perturb_by_schedule(cycle, cfg.schedule,
                                      settle_cycles=cfg.settle_cycles)
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var, cycle)
        else:
            agent.run_cycle(cycle)

        if log_interval > 0:
            changed = [
                v for v in range(world.visible_count)
                if agent.ledger.vars[v].parents != pre_parents.get(v)
            ]

            if cycle % log_interval == 0:
                iv_delta = agent.total_interventions - prev_iv
                sent_delta = agent.sentinel_skip_count - prev_sent_skip
                vis = world.visible_count
                tag = f"[{log_tag}] " if log_tag else ""
                n_cert = sum(
                    1 for v in range(vis)
                    if agent.ledger.vars[v].status in ("certified", "trass")
                )
                changed_str = (
                    " fits=[" + ",".join(
                        f"x{v}→{agent.ledger.vars[v].status}"
                        for v in changed
                    ) + "]"
                ) if changed else ""
                print(
                    f"  {tag}c{cycle:4d}/{cfg.cycles} vis={vis:3d} cert={n_cert:3d} "
                    f"Δiv={iv_delta:5d} Δsent={sent_delta:4d}"
                    f"{changed_str}",
                    flush=True,
                )
                prev_iv = agent.total_interventions
                prev_sent_skip = agent.sentinel_skip_count

    return agent, world


def _build_and_run_baseline(cfg: RunConfig) -> Tuple[SparseCachedRefitAgent, CausalWorld]:
    """Returns (agent, world)."""
    rng_w = random.Random(cfg.seed)
    rng_a = random.Random(cfg.seed + 20_000)

    initial_visible = 1 if cfg.schedule == "incremental" else cfg.n_vars
    world = CausalWorld(cfg.n_vars, rng_w, noise_sigma=cfg.noise_sigma,
                        initial_visible=initial_visible)
    agent = SparseCachedRefitAgent(world=world, rng=rng_a, intervention_budget=10)
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

            # I5: route certs belong in route_certs, not certificates
            if getattr(cert, "operation", None) == "route":
                m.bad_route_cert_location += 1

        if "audit" in n.certificates:
            m.vars_with_audit_cert += 1
            role = n.certificates["audit"].role
            if role == "reusable":
                m.audit_reusable += 1
            elif role not in ("reusable", "not_reusable"):
                m.audit_bad_role += 1

        if n.route_certs:
            m.vars_with_route_certs += 1
            for rc in n.route_certs.values():
                m.route_certs_total += 1
                if rc.role == "trass":
                    m.route_trass += 1
                elif rc.role == "tareth":
                    m.route_tareth += 1

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

    for e in agent.ledger.event_log:
        if "frontier collapsed" in e:
            m.frontier_collapses += 1
        elif "frontier cleared (threshold not met" in e:
            m.frontier_cleared += 1

    # Composite nethra (nethra-of-nethra) handle metrics
    m.composite_skip_count = agent.composite_skip_count
    _live_cns = agent.ledger.composites
    _degrees = {}
    for cn in _live_cns:
        for v in cn.members:
            _degrees[v] = _degrees.get(v, 0) + 1
    m.vars_under_composite = len(_degrees)
    m.active_composites = len(_live_cns)
    m.composite_revoked = len(agent.ledger.revoked_composites)
    _deg_vals = list(_degrees.values())
    m.composite_max_degree = max(_deg_vals) if _deg_vals else 0
    m.composite_mean_degree = sum(_deg_vals) / len(_deg_vals) if _deg_vals else 0.0
    _raw_pair = sum(cn.pass_count * len(cn.members) for cn in _live_cns)
    m.composite_duplicate_factor = _raw_pair / max(1, agent.composite_skip_count)
    # connected components via union-find
    _par = {v: v for v in _degrees}
    def _find_c(x: int) -> int:
        while _par[x] != x:
            _par[x] = _par[_par[x]]
            x = _par[x]
        return x
    for cn in _live_cns:
        ra, rb = _find_c(cn.members[0]), _find_c(cn.members[1])
        if ra != rb:
            _par[ra] = rb
    m.composite_components = len({_find_c(v) for v in _degrees}) if _degrees else 0
    # HyperCompositeNethra (component) metrics
    _live_hc = agent.ledger.hyper_composites
    m.component_live = len(_live_hc)
    m.component_revoked = len(agent.ledger.revoked_hyper_composites)
    m.component_members = len({v for hc in _live_hc for v in hc.members})
    m.component_skips = getattr(agent, "component_skip_count", 0)
    m.pairwise_fallbacks = getattr(agent, "pairwise_fallback_count", 0)
    _all_pair = list(_live_cns) + list(agent.ledger.absorbed_composites)
    _raw_before = sum(cn.pass_count * len(cn.members) for cn in _all_pair)
    _total_skips = agent.composite_skip_count + m.component_skips
    m.duplicate_factor_before = _raw_before / max(1, _total_skips)
    _raw_after = _raw_pair + m.component_skips
    m.duplicate_factor_after = _raw_after / max(1, _total_skips)

    # Regime-based skip metrics
    m.regime_skip_count = agent._regime_skip_count
    m.confirmed_regimes = len(agent.regime_register._confirmed)
    m.vars_under_regime = len({e.var for sig in agent.regime_register._confirmed for e in sig.events})
    m.regime_sentinel_passes = agent._regime_sentinel_passes
    m.regime_sentinel_fails = agent._regime_sentinel_fails
    m.regime_no_sentinel = agent._regime_no_sentinel

    # Sentinel utility accounting
    _PARK_W = 200
    m.vars_with_unique_failures = sum(1 for n in visible if n.unique_failures_caught > 0)
    m.vars_parkable = sum(
        1 for n in visible
        if n.unique_failures_caught == 0
        and n.covered_by_regime_id is not None
        and n.cycles_since_unique_failure >= _PARK_W
    )
    m.total_unique_failures = sum(n.unique_failures_caught for n in visible)
    m.total_higher_caught = sum(n.failures_also_caught_by_higher for n in visible)

    # Parking metrics
    m.parked_skip_count = agent._parked_skip_count
    m.woken_count = agent._woken_count

    # Passive residual monitoring metrics
    m.passive_saved_iv = agent._passive_saved_iv
    m.passive_stress_count = agent._passive_stress_count

    # Shadow residual metrics (zero when --shadow-residual off)
    m.shadow_residual_calls = getattr(agent, "_shadow_residual_calls", 0)
    m.shadow_residual_ok = getattr(agent, "_shadow_residual_ok", 0)
    m.shadow_residual_stressed = getattr(agent, "_shadow_residual_stressed", 0)
    m.shadow_residual_insufficient = getattr(agent, "_shadow_residual_insufficient", 0)
    m.shadow_false_ok_vs_symbolic = getattr(agent, "_shadow_false_ok_vs_symbolic", 0)
    m.shadow_false_stress_vs_symbolic = getattr(agent, "_shadow_false_stress_vs_symbolic", 0)
    m.shadow_agree_symbolic = getattr(agent, "_shadow_agree_symbolic", 0)
    m.shadow_would_save_iv = getattr(agent, "_shadow_would_save_iv", 0)
    m.shadow_would_miss_symbolic_stress = getattr(agent, "_shadow_would_miss_symbolic_stress", 0)
    m.shadow_false_ok_vs_active_sentinel = getattr(agent, "_shadow_false_ok_vs_active_sentinel", 0)
    m.shadow_would_miss_active_failure = getattr(agent, "_shadow_would_miss_active_failure", 0)

    # Hybrid control metrics (zero when hybrid-control=off)
    m.hybrid_residual_predictor_calls = getattr(agent, "_hybrid_residual_predictor_calls", 0)
    m.hybrid_residual_ok = getattr(agent, "_hybrid_residual_ok", 0)
    m.hybrid_residual_stressed = getattr(agent, "_hybrid_residual_stressed", 0)
    m.hybrid_parent_ranker_calls = getattr(agent, "_hybrid_parent_ranker_calls", 0)
    m.hybrid_probe_proposer_calls = getattr(agent, "_hybrid_probe_proposer_calls", 0)
    m.hybrid_expert_router_calls = getattr(agent, "_hybrid_expert_router_calls", 0)
    _agenda = getattr(agent, "_repair_agenda", None)
    if _agenda is not None:
        _as = _agenda.summary()
        m.hybrid_repair_agenda_items = _as["total_pushed"]
        m.hybrid_repair_agenda_scope_mean = _as.get("scope_mean", 0.0)
        m.hybrid_repair_agenda_scope_max = _as.get("scope_max", 0)

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
    if arch.bad_route_cert_location:
        violations.append(f"I5: {arch.bad_route_cert_location} route cert(s) misplaced in certificates (must live in route_certs)")
    return violations


def _run_one(cfg: RunConfig) -> RunResult:
    t0 = time.monotonic()
    try:
        agent, world = _build_and_run_dreth(
            cfg, log_interval=cfg.log_interval, log_tag="CW" if cfg.ablate else ""
        )
        elapsed = time.monotonic() - t0
        tier = _compute_tier_metrics(agent, world)

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

        arch = _extract_arch_metrics(agent, world)
        violations = _check_invariants(arch)
        regime_summary = agent.regime_register.summary()

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
            arch=arch,
            violations=violations,
            tier=tier,
            regime_summary=regime_summary,
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
            certified=0, trass_status=0,
            arch=ArchMetrics(),
        )

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
                candidate_refreshes=b_agent.candidate_refresh_count,
                ok=True,
            )
        except Exception as exc:
            result.baseline = BaselineMetrics(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    if cfg.ablate:
        try:
            a2, w2 = _build_and_run_dreth(
                cfg, consequence_weight=False,
                log_interval=cfg.log_interval, log_tag="CW-OFF",
                agent_seed_offset=5_000,
            )
            result.tier_no_cw = _compute_tier_metrics(a2, w2)
        except Exception:
            pass

    return result


# ── formatting ─────────────────────────────────────────────────────────────────

def _pct_diff(dreth_val: float, base_val: float) -> str:
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
    csk = r.arch.composite_skip_count
    rsk = r.arch.regime_skip_count
    dreth_line = (
        f"  n={cfg.n_vars:3d} cyc={cfg.cycles:4d} seed={cfg.seed:5d} "
        f"| {st:3s} {r.elapsed:5.1f}s "
        f"| skip={skip:6s} sent={r.sentinel_skips:5d} comp={r.compression_skips:4d} "
        f"csk={csk:4d} rsk={rsk:4d} "
        f"| iv={r.interventions:6d} auds={r.full_audits:5d} "
        f"| rc={rca:4d} ac={aca:3d} dorm={dorm:3d} rev={revoked:3d} "
        f"fc={fc:3d}/{fc+fclr:<3d} bkf={bkf:2d} nov={r.arch.vars_open_novelty:2d} "
        f"stb={r.arch.vars_envelope_stable:2d} nf={r.arch.vars_noise_floor:2d} "
        f"| {viols}"
    )
    tm = r.tier
    def _tier_line(t: TierMetrics, label: str) -> str:
        return (
            f"  {'':>3s} {'':>4s} {'':>5s}   {label}"
            f"tier n=({t.n_t0}/{t.n_t1}/{t.n_t2})  "
            f"sent=({t.sent_t0:.1f}/{t.sent_t1:.1f}/{t.sent_t2:.1f})  "
            f"promo=({t.promo_t0:.0f}/{t.promo_t1:.0f}/{t.promo_t2:.0f})"
        )
    tier_lines = _tier_line(tm, "[CW ON ] ")
    if r.tier_no_cw is not None:
        tier_lines += "\n" + _tier_line(r.tier_no_cw, "[CW OFF] ")

    regime_line = (
        "\n" + r.regime_summary
        if r.regime_summary and "confirmed" in r.regime_summary and "0 confirmed" not in r.regime_summary
        else ""
    )
    parking_line = ""
    if r.arch.parked_skip_count > 0 or r.arch.woken_count > 0:
        parking_line = (
            f"\n    parking: psk={r.arch.parked_skip_count:5d}  woken={r.arch.woken_count:4d}"
            f"  parkable={r.arch.vars_parkable:2d}  uniq_fail={r.arch.total_unique_failures:4d}"
            f"  higher_caught={r.arch.total_higher_caught:4d}"
        )
    if r.baseline is None:
        return dreth_line + "\n" + tier_lines + regime_line + parking_line

    b = r.baseline
    if not b.ok:
        base_line = f"  {'':>3s} {'':>4s} {'':>5s}   BASE ERR: {b.error}"
        return dreth_line + "\n" + tier_lines + "\n" + base_line

    b_total = b.skip_count + b.full_audits
    b_skip_pct = b.skip_count / max(1, b_total) * 100
    iv_diff  = _pct_diff(r.interventions, b.interventions)
    aud_diff = _pct_diff(r.full_audits,   b.full_audits)
    t_diff   = _pct_diff(r.elapsed,       b.elapsed)
    base_line = (
        f"  {'':>3s} {'':>4s} {'':>5s}   "
        f"BASE {b.elapsed:5.1f}s "
        f"| skip={b_skip_pct:5.1f}%       "
        f"| iv={b.interventions:6d} auds={b.full_audits:5d} "
        f"| rfail={b.sentinel_fails:4d} ref={b.candidate_refreshes:3d} "
        f"| Δiv={iv_diff:>6s} Δaud={aud_diff:>6s} Δt={t_diff:>6s}"
    )
    return dreth_line + "\n" + tier_lines + "\n" + base_line + parking_line


def _fmt_header() -> str:
    return (
        f"  {'n':>3}  {'cyc':>4}  {'seed':>5}  "
        f"  {'st':3s} {'t':>5}  "
        f"  {'skip%':>6} {'sent':>5} {'comp':>4}  "
        f"  {'iv':>6} {'auds':>5}  "
        f"  {'rc':>4} {'ac':>3} {'dorm':>4} {'rev':>3} "
        f"{'fc/tot':>7}  "
        f"  inv"
    )


# ── aggregate ──────────────────────────────────────────────────────────────────

def _print_tier_aggregate(ok_runs: List[RunResult]) -> None:
    if not ok_runs:
        return

    def _tier_sum(attr: str) -> Tuple[int, int, int]:
        return (
            sum(getattr(r.tier, attr + "_t0") for r in ok_runs),
            sum(getattr(r.tier, attr + "_t1") for r in ok_runs),
            sum(getattr(r.tier, attr + "_t2") for r in ok_runs),
        )

    n0, n1, n2 = _tier_sum("n")
    avg_sent0 = sum(r.tier.sent_t0 for r in ok_runs) / len(ok_runs)
    avg_sent1 = sum(r.tier.sent_t1 for r in ok_runs) / len(ok_runs)
    avg_sent2 = sum(r.tier.sent_t2 for r in ok_runs) / len(ok_runs)
    avg_pr0   = sum(r.tier.promo_t0 for r in ok_runs) / len(ok_runs)
    avg_pr1   = sum(r.tier.promo_t1 for r in ok_runs) / len(ok_runs)
    avg_pr2   = sum(r.tier.promo_t2 for r in ok_runs) / len(ok_runs)

    print()
    print("── consequence-weight tier breakdown ───────────────────────────────")
    print(f"  tier        T0(leaf)   T1(1-2dep)  T2(3+dep)")
    print(f"  var count   {n0:8d}   {n1:9d}  {n2:8d}")
    print(f"  avg sent    {avg_sent0:8.2f}   {avg_sent1:9.2f}  {avg_sent2:8.2f}")
    print(f"  avg promo   {avg_pr0:8.1f}   {avg_pr1:9.1f}  {avg_pr2:8.1f}  (cycles to first cert)")

    ablate_runs = [r for r in ok_runs if r.tier_no_cw is not None]
    if ablate_runs:
        sent0_off = sum(r.tier_no_cw.sent_t0 for r in ablate_runs) / len(ablate_runs)
        sent1_off = sum(r.tier_no_cw.sent_t1 for r in ablate_runs) / len(ablate_runs)
        sent2_off = sum(r.tier_no_cw.sent_t2 for r in ablate_runs) / len(ablate_runs)
        pr0_off   = sum(r.tier_no_cw.promo_t0 for r in ablate_runs) / len(ablate_runs)
        pr1_off   = sum(r.tier_no_cw.promo_t1 for r in ablate_runs) / len(ablate_runs)
        pr2_off   = sum(r.tier_no_cw.promo_t2 for r in ablate_runs) / len(ablate_runs)
        print()
        print(f"  ablation ({len(ablate_runs)} runs with CW disabled):")
        print(f"  CW OFF avg sent    {sent0_off:5.2f}   {sent1_off:9.2f}  {sent2_off:8.2f}")
        print(f"  CW ON  avg sent    {avg_sent0:5.2f}   {avg_sent1:9.2f}  {avg_sent2:8.2f}  ← should be > CW OFF for T1/T2")
        print(f"  CW OFF avg promo   {pr0_off:5.1f}   {pr1_off:9.1f}  {pr2_off:8.1f}")
        print(f"  CW ON  avg promo   {avg_pr0:5.1f}   {avg_pr1:9.1f}  {avg_pr2:8.1f}  ← should be > CW OFF for T1/T2")


def _print_aggregate(results: List[RunResult]) -> None:
    ok_runs = [r for r in results if r.ok]
    if not ok_runs:
        print("  no successful runs")
        return

    n = len(ok_runs)
    avg_skip = sum(r.skip_pct for r in ok_runs) / n
    avg_iv   = sum(r.interventions for r in ok_runs) / n
    avg_rc   = sum(r.arch.route_certs_total for r in ok_runs) / n
    avg_ac   = sum(r.arch.vars_with_audit_cert for r in ok_runs) / n
    avg_dorm = sum(r.arch.dormant_total for r in ok_runs) / n
    avg_rev  = sum(sum(r.arch.revoked_by_dist.values()) for r in ok_runs) / n
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

    avg_trass_sk = sum(r.trass_skips for r in ok_runs) / n
    avg_sent_sk  = sum(r.sentinel_skips for r in ok_runs) / n
    avg_comp_sk  = sum(r.arch.composite_skip_count for r in ok_runs) / n
    avg_compr_sk = sum(r.compression_skips for r in ok_runs) / n
    avg_rsk      = sum(r.arch.regime_skip_count for r in ok_runs) / n
    avg_psk_agg  = sum(r.arch.parked_skip_count for r in ok_runs) / n
    total_sk_avg = avg_trass_sk + avg_sent_sk + avg_comp_sk + avg_compr_sk + avg_rsk + avg_psk_agg
    handle_avg   = avg_comp_sk + avg_rsk + avg_psk_agg
    amort_pct    = 100.0 * handle_avg / total_sk_avg if total_sk_avg > 0 else 0.0

    print(f"  runs ok={n}/{len(results)}")
    print(f"  avg: skip%={avg_skip:.1f}  iv={avg_iv:.0f}")
    print(f"  handle amortization: {amort_pct:.1f}%  "
          f"(composite={avg_comp_sk:.0f} regime={avg_rsk:.0f} park={avg_psk_agg:.0f} of {total_sk_avg:.0f} avg total skips)")
    avg_comp_live = sum(r.arch.active_composites for r in ok_runs) / n
    avg_comp_rev  = sum(r.arch.composite_revoked for r in ok_runs) / n
    avg_comp_umem = sum(r.arch.vars_under_composite for r in ok_runs) / n
    avg_comp_comp = sum(r.arch.composite_components for r in ok_runs) / n
    avg_comp_maxd = sum(r.arch.composite_max_degree for r in ok_runs) / n
    avg_comp_meand = sum(r.arch.composite_mean_degree for r in ok_runs) / n
    avg_comp_dup  = sum(r.arch.composite_duplicate_factor for r in ok_runs) / n
    print(f"  composite overlap: live={avg_comp_live:.0f} rev={avg_comp_rev:.0f} "
          f"members={avg_comp_umem:.0f} components={avg_comp_comp:.1f} "
          f"deg(max={avg_comp_maxd:.0f} mean={avg_comp_meand:.1f}) "
          f"dup_factor={avg_comp_dup:.2f}x")
    avg_hc_live  = sum(r.arch.component_live for r in ok_runs) / n
    avg_hc_rev   = sum(r.arch.component_revoked for r in ok_runs) / n
    avg_hc_mem   = sum(r.arch.component_members for r in ok_runs) / n
    avg_hc_skip  = sum(r.arch.component_skips for r in ok_runs) / n
    avg_hc_fall  = sum(r.arch.pairwise_fallbacks for r in ok_runs) / n
    avg_dup_bef  = sum(r.arch.duplicate_factor_before for r in ok_runs) / n
    avg_dup_aft  = sum(r.arch.duplicate_factor_after for r in ok_runs) / n
    print(f"  component handle: live={avg_hc_live:.0f} rev={avg_hc_rev:.0f} "
          f"members={avg_hc_mem:.0f} skips={avg_hc_skip:.0f} fallbacks={avg_hc_fall:.0f} "
          f"dup_factor {avg_dup_bef:.2f}x→{avg_dup_aft:.2f}x")
    total_rpass = sum(r.arch.regime_sentinel_passes for r in ok_runs)
    total_rfail = sum(r.arch.regime_sentinel_fails for r in ok_runs)
    total_rno   = sum(r.arch.regime_no_sentinel for r in ok_runs)
    if total_rpass + total_rfail + total_rno > 0:
        print(f"  regime sentinel: pass={total_rpass}  fail={total_rfail}  no_sentinel={total_rno}")
    avg_unique_fail = sum(r.arch.total_unique_failures for r in ok_runs) / n
    avg_higher_caught = sum(r.arch.total_higher_caught for r in ok_runs) / n
    avg_parkable = sum(r.arch.vars_parkable for r in ok_runs) / n
    avg_uniq_vars = sum(r.arch.vars_with_unique_failures for r in ok_runs) / n
    print(f"  sentinel utility: unique_fails={avg_unique_fail:.0f}  higher_caught={avg_higher_caught:.0f}  "
          f"parkable_vars={avg_parkable:.1f}  vars_w_unique={avg_uniq_vars:.1f}")
    avg_psk = sum(r.arch.parked_skip_count for r in ok_runs) / n
    avg_woken = sum(r.arch.woken_count for r in ok_runs) / n
    if avg_psk > 0 or avg_woken > 0:
        print(f"  parking: avg_psk={avg_psk:.0f}  avg_woken={avg_woken:.1f}")
    avg_passive_iv = sum(r.arch.passive_saved_iv for r in ok_runs) / n
    avg_passive_stress = sum(r.arch.passive_stress_count for r in ok_runs) / n
    print(f"  passive monitor: saved_iv={avg_passive_iv:.0f}  stressed={avg_passive_stress:.0f}")

    _total_shadow_calls = sum(r.arch.shadow_residual_calls for r in ok_runs)
    if _total_shadow_calls > 0:
        _shadow_ok_t    = sum(r.arch.shadow_residual_ok for r in ok_runs)
        _shadow_str_t   = sum(r.arch.shadow_residual_stressed for r in ok_runs)
        _shadow_ins_t   = sum(r.arch.shadow_residual_insufficient for r in ok_runs)
        _shadow_agr_t   = sum(r.arch.shadow_agree_symbolic for r in ok_runs)
        _shadow_fok_t   = sum(r.arch.shadow_false_ok_vs_symbolic for r in ok_runs)
        _shadow_fst_t   = sum(r.arch.shadow_false_stress_vs_symbolic for r in ok_runs)
        _shadow_wsv_t   = sum(r.arch.shadow_would_save_iv for r in ok_runs)
        _shadow_wms_t   = sum(r.arch.shadow_would_miss_symbolic_stress for r in ok_runs)
        _shadow_fas_t   = sum(r.arch.shadow_false_ok_vs_active_sentinel for r in ok_runs)
        _shadow_wma_t   = sum(r.arch.shadow_would_miss_active_failure for r in ok_runs)
        print()
        print("── shadow residual ─────────────────────────────────────────────────")
        print(f"  calls={_total_shadow_calls}")
        print(f"  ok={_shadow_ok_t}  stressed={_shadow_str_t}  insufficient={_shadow_ins_t}")
        print(f"  agree_symbolic={_shadow_agr_t}")
        print(f"  false_ok_vs_symbolic={_shadow_fok_t}  false_stress_vs_symbolic={_shadow_fst_t}")
        print(f"  would_save_iv={_shadow_wsv_t}  would_miss_symbolic_stress={_shadow_wms_t}")
        print(f"  false_ok_vs_active={_shadow_fas_t}  would_miss_active_failure={_shadow_wma_t}")

    # Print hybrid metrics whenever any provider was active, even if some counts
    # are zero — zero counts expose wiring gaps immediately.
    _hybrid_res_calls = sum(r.arch.hybrid_residual_predictor_calls for r in ok_runs)
    _hybrid_pr_calls  = sum(r.arch.hybrid_parent_ranker_calls for r in ok_runs)
    _hybrid_pp_calls  = sum(r.arch.hybrid_probe_proposer_calls for r in ok_runs)
    _hybrid_er_calls  = sum(r.arch.hybrid_expert_router_calls for r in ok_runs)
    _any_hybrid = _hybrid_res_calls + _hybrid_pr_calls + _hybrid_pp_calls + _hybrid_er_calls
    if _any_hybrid > 0:
        _hybrid_ok  = sum(r.arch.hybrid_residual_ok for r in ok_runs)
        _hybrid_str = sum(r.arch.hybrid_residual_stressed for r in ok_runs)
        print(
            f"  hybrid residual_predictor: calls={_hybrid_res_calls}"
            f"  ok={_hybrid_ok}  stressed={_hybrid_str}"
        )
        print(
            f"  hybrid parent_ranker:      calls={_hybrid_pr_calls}"
            f"  probe_proposer: calls={_hybrid_pp_calls}"
            f"  expert_router: calls={_hybrid_er_calls}"
        )
        _agenda_tot = sum(r.arch.hybrid_repair_agenda_items for r in ok_runs)
        if _agenda_tot > 0:
            _scope_max  = max(r.arch.hybrid_repair_agenda_scope_max for r in ok_runs)
            _scope_mean = (
                sum(r.arch.hybrid_repair_agenda_scope_mean * max(r.arch.hybrid_repair_agenda_items, 1)
                    for r in ok_runs if r.arch.hybrid_repair_agenda_items > 0)
                / max(1, sum(r.arch.hybrid_repair_agenda_items for r in ok_runs if r.arch.hybrid_repair_agenda_items > 0))
            ) if any(r.arch.hybrid_repair_agenda_items > 0 for r in ok_runs) else 0.0
            print(
                f"  repair_agenda: total_pushed={_agenda_tot}"
                f"  scope_mean={_scope_mean:.1f}  scope_max={_scope_max}"
            )
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

    base_runs = [r for r in ok_runs if r.baseline is not None and r.baseline.ok]
    if not base_runs:
        _print_tier_aggregate(ok_runs)
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
    b_ref = sum(r.baseline.candidate_refreshes for r in base_runs)
    print(f"  baseline candidate_refreshes: {b_ref}")
    print()
    print("  Δiv/Δaud negative = Dreth uses fewer probes/refits than baseline.")
    print("  Δt  negative = Dreth is faster.")
    _print_tier_aggregate(ok_runs)


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
                   choices=["incremental", "periodic_shifts", "novelty", "shaped",
                            "rare_catastrophe", "regime_switch", "false_trass"],
                   help="mutation schedule (default: incremental)")
    p.add_argument("--settle-cycles", type=int, default=8,
                   help="settle cycles for incremental reveals or regime_switch initial window (default: 8)")
    p.add_argument("--noise-sigma", type=float, default=0.02,
                   help="noise sigma (default: 0.02)")
    p.add_argument("--workers", type=int, default=20,
                   help="max parallel workers (default: cpu count)")
    p.add_argument("--out", default=None,
                   help="write one JSON line per run to this file")
    p.add_argument("--verbose-violations", action="store_true",
                   help="print full violation details for each failing run")
    p.add_argument("--compare", action="store_true",
                   help="run sparse_cached_refit baseline alongside Dreth and report comparison")
    p.add_argument("--ablate-consequence", action="store_true",
                   help="re-run each config with consequence_weight=False and compare tier metrics")
    p.add_argument("--progress", type=int, default=0, metavar="N",
                   help="print a per-cycle status line every N cycles (forces --workers 1)")
    p.add_argument("--hybrid-control", default="off",
                   choices=["off", "interfaces"],
                   help="hybrid control mode: off=current behavior; interfaces=symbolic provider wrappers (default: off)")
    p.add_argument("--repair-agenda", action="store_true",
                   help="enable RepairAgenda: annotate needs_audit entries with scope/authority metadata")
    p.add_argument("--shadow-residual", default="off",
                   choices=["off", "online"],
                   help="shadow residual mode: off=disabled; online=shadow learned predictor (default: off)")
    args = p.parse_args()

    var_list   = [int(x) for x in args.vars.split(",")]
    cycle_list = [int(x) for x in args.cycles.split(",")]
    seed_list  = [int(x) for x in args.seeds.split(",")]

    n_workers = args.workers
    if args.progress > 0:
        n_workers = 1

    configs = [
        RunConfig(n_vars=v, cycles=c, seed=s,
                  schedule=args.schedule,
                  settle_cycles=args.settle_cycles,
                  noise_sigma=args.noise_sigma,
                  compare=args.compare,
                  ablate=args.ablate_consequence,
                  log_interval=args.progress,
                  hybrid_control=args.hybrid_control,
                  repair_agenda_enabled=args.repair_agenda,
                  shadow_residual=args.shadow_residual)
        for v in var_list
        for c in cycle_list
        for s in seed_list
    ]

    total = len(configs)
    mode = " +compare" if args.compare else ""
    if args.ablate_consequence:
        mode += " +ablate"
    if args.progress:
        mode += f" +progress({args.progress})"
    if args.hybrid_control != "off":
        mode += f" +hybrid({args.hybrid_control})"
    if args.repair_agenda:
        mode += " +repair-agenda"
    if args.shadow_residual != "off":
        mode += f" +shadow-residual({args.shadow_residual})"
    print(f"dreth arch-test{mode}: {total} runs | "
          f"vars={var_list} cycles={cycle_list} seeds={seed_list} "
          f"schedule={args.schedule}", flush=True)
    print(f"  workers={n_workers}  settle={args.settle_cycles}  "
          f"noise={args.noise_sigma}", flush=True)
    if args.compare:
        print(f"  baseline: sparse_cached_refit (K=10 window=8 threshold=3 refresh_interval=100)", flush=True)
    if args.progress:
        print(f"  progress: every {args.progress} cycles — cert count, Δiv, Δsent, fit changes", flush=True)
    print(f"  checking: I1(earned_by) I2(audit-role) I3(dormant-type) "
          f"I4(revoked_by) I5(route-target-owned)", flush=True)
    print()

    header = _fmt_header()
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: List[RunResult] = []
    done = 0
    out_fh = open(args.out, "w") if args.out else None

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
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
                        "candidate_refreshes": r.baseline.candidate_refreshes,
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
        print("  BASE row: rfail=sentinel_fails ref=candidate_refreshes")
        print("  Δiv=intervention diff  Δaud=audit diff  Δt=time diff (dreth vs baseline)")
        print("  negative Δ = Dreth is cheaper/faster")


if __name__ == "__main__":
    main()
