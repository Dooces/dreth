from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The authority-record control loop. ChainedAgent owns the full lifecycle.
#
# Per cycle, per variable, the agent dispatches to one of:
#   trass-skip        — operation_role==trass: do nothing
#   compression       — gate matches cached value: return it
#   sentinel cheap path — run sentinel probes, validate fit still holds
#   full audit        — re-enumerate, re-score, reinstall
#
# Key methods and what they actually do:
#   _full_audit_var       — calls fit_var, records FitDiagnostic
#   _install_var          — applies audit result: updates VarNethra status,
#                           tests operation role, attaches sentinels, promotes
#                           to certified status, manages TiedFrontier
#   _certify_operation_role — runs the substitution test: perturb this var,
#                           observe whether other vars change beyond tolerance.
#                           Returns tareth/trass/untested. This verdict gates
#                           all downstream hypothesis spaces.
#   _update_tied_frontier — maintains the ambiguity object on VarNethra
#   _collapse_tied_frontier — CURRENTLY PREMATURE: collapses when score
#                           landscape narrows to one candidate in a single
#                           audit. Should require regime-survival evidence
#                           (stable_count + distinct context_keys). Fix in P4.
#
# What makes authoritative nethras operative here:
#   available_parents in _full_audit_var is built only from vars with tareth
#   authority records. A variable's current authority directly controls what hypotheses are
#   enumerated for every variable that might depend on it.
#
# Active:
#   _adaptive_probe_budget — now activated in _full_audit_var (P1-A).
#     Scales probe budget up with hypothesis space; only scales UP from base.
#     RNG trajectory changes when pool size changes — downstream test seeds
#     that relied on the old flat budget will diverge (expected, not a bug).
#
# ════════════════════════════════════════════════════════════════════════════════
# CORE INVARIANT — READ BEFORE MODIFYING THIS FILE
#
# NETHRA: Not a label. A factoring that earned authority by surviving
#   intervention tests in a specific scope. Authoritative nethras are operative:
#   they become active filters deciding what later evidence counts as tareth
#   or trass. They do not passively describe — they gate future reasoning.
#
# TARETH / TRASS: Provisional verdicts from scope-specific substitution tests.
#   trass  — substituting the distinction leaves monitored targets unchanged
#   tareth — substitution changes monitored targets; a concrete witness exists
#   Certs fire by default; only observed failure or an active dependency event
#   earns revocation. Sentinel failure and downstream contradiction defeat cert
#   authority in the relevant scope. Structural or scope changes revoke only when they are themselves
#   dependency events (parent set changed, contradicting evidence in expanded
#   context). The verdict belongs to the scope, not the hypothesis.
#
# FALSE-TRASS: Two locally-trass nethras can jointly be tareth. Composition
#   requires a joint re-test. Local authority does not propagate upward.
#
# MORPHOLOGY ≠ CAUSE:
#   Morphology (same parents, same operator, close scores) is structural —
#   readable from candidate shape with no interventions required.
#   Cause (genuine equivalence, library gap, under-probing) requires
#   separating probes and regime-survival evidence across distinct regimes.
#   Pattern-matching on scores or parent structure is morphology, never cause.
#
# AMBIGUITY IS FIRST-CLASS: Insufficient evidence → TiedFrontier survives.
#   Collapse requires regime-survival proof. Score proximity does not justify
#   collapsing; it justifies recording the ambiguity and generating probes.
#
# This file: authority-record decisions live here. _install_var must not collapse
#   based on score proximity alone. _certify_operation_role must not promote
#   trass without regime-survival evidence. _collapse_tied_frontier as currently
#   implemented triggers on score-landscape narrowing — that is PREMATURE COLLAPSE
#   and must be replaced with stable_count + distinct_contexts_seen gating (P1.2).
# ════════════════════════════════════════════════════════════════════════════════

import dataclasses
import math
import random
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

from .functions import FUNC_LIBRARY
from .world import CausalWorld
from .ledger import (ChainedLedger, CompositeNethra, HyperCompositeNethra, Compression, LedgerEvent,
                     DEFAULT_TOLERANCE, DormantAlternative, NethraCertificate, Role, TiedFrontier)
from .fit import fit_var
from .sentinels import select_var_sentinels, check_var_sentinels_with_envelope
from .regime import RegimeRegister, CertEvent as RegimeCertEvent
from .records import CycleRecord, FitDiagnostic
from .summary import RunAnalyzer, SummaryRenderer
from .hybrid import (
    ResidualPredictor, ParentRanker, ProbeProposer, ExpertRouter,
    SymbolicResidualPredictor, SensitivityParentRanker,
    ParentProposalDiagnostics, ProbeProposalDiagnostics,
)
from .repair_agenda import RepairAgenda, RepairAgendaItem
from .uncertainty_consolidation import (
    cluster_has_specific_local_anchor,
    cluster_uncertainty_cases,
    extract_uncertainty_cases_from_agent,
    propose_consolidation_assists,
    summarize_clusters,
)
from .context_role_index import (
    ContextRoleRecord,
    ContextRoleIndex,
    NethraNode,
    candidate_id,
    context_key as nethra_context_key,
    var_fit_id,
)
from .authority_strength import (
    AuthorityStateController,
    compute_authority_strength_records,
    records_to_dicts as authority_strength_records_to_dicts,
    summarize_authority_strength_records,
    summary_to_dict as authority_strength_summary_to_dict,
)
from .learned_residual import (
    ResidualFeatureVector,
    ShadowLearnedResidualPredictor,
    ShadowResidualKeyAuthority,
)
from .background_nethra import BackgroundNethraIndex
from .scaffold_memory import ScaffoldMemoryIndex
from .nethra_runtime_memory import PersistentNethraIndex

# ── Trass authority thresholds ────────────────────────────────────────────────
# A trass cert suppresses future sentinel monitoring — the strongest operational
# authority the framework grants. It must require more evidence than any cert that merely
# prioritizes or routes attention (tareth, compression, dormancy).
#
# Newly issued trass certs are *provisional*: each hot-pass cycle that reaches
# the trass block without a prior cascade invalidation increments cert.sentinel_passes
# (a stable-cycle counter, not a probe counter). Only after _STRONG_TRASS_SENTINEL_PASSES
# stable cycles (scaled by consequence tier) does the cert earn hard-suppress authority.
# Trass vars are always counted as skips during the provisional period — they are NOT
# queued for full audit. If cascade invalidates the cert, provisional evidence resets.
#
# Consequence-tier scaling: _STRONG_TRASS_SENTINEL_PASSES + tier * 3
#   Tier 0 (leaf):  1 stable cycle
#   Tier 1 (1–2 deps): 4 stable cycles
#   Tier 2 (3+ deps): 7 stable cycles
_STRONG_TRASS_SENTINEL_PASSES = 1

# Sentinel parking thresholds.
# A leaf sentinel may be parked (skipped each cycle) when:
#   - var is covered by a confirmed regime with active_sentinel
#   - no unique failures in the last _PARK_W cycles
#   - the covering regime's sentinel has passed _PARK_K times
#     across at least 2 distinct recurrences (authority >= 2 already by
#     confirmed-regime invariant; _PARK_K checks pass count specifically)
# Wake on: covering regime sentinel fails, local sentinel fail (contradiction),
#   or sparse revalidation every _PARK_REVALIDATE_INTERVAL cycles.
_PARK_W = 200                    # unique-failure-free window required
_PARK_K = 4                      # regime sentinel passes required before parking
_PARK_REVALIDATE_INTERVAL = 500  # parked vars re-run full sentinel every N cycles

# Proactive joint false-trass scan interval.
# Every _JOINT_SCAN_INTERVAL cycles, scan all trass-var pairs for joint
# interactions without waiting for a downstream var's sentinel to fail.
# This catches PROD(TINY, TINY) patterns where neither var is individually
# salient but their joint effect on a tareth sentinel var is large.
_JOINT_SCAN_INTERVAL = 50

# Component promotion: when the live composite graph has a dense connected
# component, promote it to a HyperCompositeNethra that checks one joint
# sentinel instead of k² pairwise sentinels.
_COMPONENT_MIN_SIZE       = 4     # minimum member count to consider a component
_COMPONENT_MIN_DENSITY    = 0.4   # fraction of possible edges that must exist
_COMPONENT_MIN_PASSES     = 10    # individual pair pass threshold
_COMPONENT_MIN_PASSES_FRAC = 0.5  # fraction of pairs that must meet _COMPONENT_MIN_PASSES
_COMPONENT_PROMOTE_INTERVAL = 100 # how often to scan for promotable components

# Repair-authority escalation: if a sentinel fires and re-audit returns the same
# fit _REPAIR_FAILURE_ESCALATION_THRESHOLD times, the var's audit budget is
# multiplied by _BUDGET_ESCALATION_FACTOR (capped at _BUDGET_ESCALATION_CAP).
_REPAIR_FAILURE_ESCALATION_THRESHOLD = 3
_BUDGET_ESCALATION_FACTOR            = 4
_BUDGET_ESCALATION_CAP               = 400
# Inert re-screen: after this many repair failures on a var, probe the inert
# set for influence on that var. Fires every multiple of this threshold so
# re-screen recurs as failures accumulate (e.g. at 6, 12, 18...).
_INERT_RESCREEN_THRESHOLD            = 6
_BACKGROUND_RESIDUAL_BUDGET          = 10  # max residuals classified per cycle; never competes with focal audit

class AgentExtension(Protocol):
    derive_compressions: Callable[["ChainedAgent", int, int], List[Compression]]
    derive_equivalence_compressions: Callable[["ChainedAgent", int, int], List[Compression]]


class ChainedAgent:
    """The agent. Owns:
      - a CausalWorld reference (read-only window into the world's interventions)
      - a ChainedLedger holding per-variable nethras and novelty records
      - per-variable counters for instability, stability, defer streaks
      - cycle history (records) for offline diagnostic comparison

    Per-cycle behavior: for each visible variable, decide one of
      (a) trass-skip   — operation_role says irrelevant, no work
      (b) compression  — gate matches, return cached value
      (c) sentinel     — cheap-path validation against world
      (d) full audit   — re-fit hypothesis from scratch (expensive)
    Variables that need (d) get scheduled by topological+budget priority.

    Configuration parameters (all have sensible defaults):
      intervention_budget:    base probe budget; actual count scales with
                              hypothesis space via _adaptive_probe_budget
      full_margin_threshold:  unused since v28; kept for arg compat
      sentinel_count:         probes per sentinel set (5)
      sentinel_pool:          pool size for sentinel selection (60)
      promote_after:          consecutive matching audits to certify (2)
      novelty_weak_streak:    instability streak to fire vocabulary novelty (2)
      compression_discovery_budget: samples per compression test (8)
      compression_discover_after:   strong_obs needed before compression search (2)
      cost_weights:           per-var cost overrides (default 1.0)
      cost_low/high_threshold: dispatch boundaries
      envelope_certify_after: deltas needed before envelope certification
      priority_audit_budget:  max full audits per cycle (default n_vars//2)
      parent_screen_m:        per-target sensitivity screen top-M candidates
                              (0 = disabled, use certified-only pool; default 8)
    """

    def __init__(
        self,
        world: CausalWorld,
        rng: random.Random,
        intervention_budget: int = 30,
        full_margin_threshold: int = 4,
        sentinel_count: int = 5,
        sentinel_pool: int = 60,
        promote_after: int = 2,
        novelty_weak_streak: int = 2,
        compression_discovery_budget: int = 8,
        compression_discover_after: int = 2,
        compression_promote_after: int = 5,
        cost_weights: Optional[Dict[int, float]] = None,
        cost_low_threshold: float = 0.5,
        cost_high_threshold: float = 2.0,
        envelope_certify_after: int = 20,
        priority_audit_budget: Optional[int] = None,
        role_salience: str = "all-visible",
        salience_targets: Optional[Set[int]] = None,
        consequence_weight: bool = True,
        frontier_k: int = 4,
        parent_screen_m: int = 8,
        residual_predictor: Optional[ResidualPredictor] = None,
        parent_ranker: Optional[ParentRanker] = None,
        probe_proposer: Optional[ProbeProposer] = None,
        expert_router: Optional[ExpertRouter] = None,
        repair_agenda_enabled: bool = False,
        repair_agenda_ordering: str = "observe",
        shadow_residual_predictor: Optional[ShadowLearnedResidualPredictor] = None,
        shadow_residual_enabled: bool = False,
        shadow_key_authority: Optional[ShadowResidualKeyAuthority] = None,
        uncertainty_consolidation_mode: str = "off",
        uncertainty_assist_policy: str = "all",
        uncertainty_max_preserve_count: int = 3,
        context_role_index_mode: str = "off",
        context_role_anchor_policy: Optional[str] = None,
        nethra_reservoir_mode: Optional[str] = None,
        authority_strength_mode: str = "off",
        authority_strength_controller: str = "state",
        authority_derivation_policy: Optional[str] = None,
        background_nethra_mode: str = "off",
        scaffold_memory_mode: str = "off",
        scaffold_memory_index: Optional[ScaffoldMemoryIndex] = None,
        nethra_memory_mode: str = "off",
        nethra_memory_index: Optional[PersistentNethraIndex] = None,
    ):
        """Construct agent. Initializes empty ledger, zero counters, and
        applies any provided per-var cost weight overrides."""
        self.world = world
        self.rng = rng
        self.intervention_budget = intervention_budget
        self.full_margin_threshold = full_margin_threshold
        self.sentinel_count = sentinel_count
        self.sentinel_pool = sentinel_pool
        self.promote_after = promote_after
        self.novelty_weak_streak = novelty_weak_streak
        self.compression_discovery_budget = compression_discovery_budget
        self.compression_discover_after = compression_discover_after
        self.compression_promote_after = compression_promote_after
        self.cost_low_threshold = cost_low_threshold
        self.cost_high_threshold = cost_high_threshold
        self.envelope_certify_after = envelope_certify_after
        if role_salience not in {"all-visible", "live-frontier"}:
            raise ValueError(f"unknown role_salience: {role_salience}")
        self.role_salience = role_salience
        if salience_targets is not None:
            bad_targets = [t for t in salience_targets if t < 0 or t >= world.n_vars]
            if bad_targets:
                raise ValueError(f"salience target out of range: {bad_targets}")
            self.salience_targets = set(salience_targets)
        else:
            self.salience_targets = None
        if priority_audit_budget is None:
            self.priority_audit_budget = max(2, world.n_vars // 2)
        else:
            self.priority_audit_budget = priority_audit_budget
        # [CONSEQUENCE-WEIGHT] ablation gate — set False to disable all CW policy.
        # Revert: remove this line and the gate check in _consequence_tier.
        self._consequence_weight_enabled = consequence_weight

        self.ledger = ChainedLedger(world.n_vars)
        # weak_streak: consecutive cycles where the fit signature CHANGED
        # (instability — drives vocabulary novelty firing)
        self.weak_streak: Dict[int, int] = {i: 0 for i in range(world.n_vars)}
        # stable_streak: consecutive cycles where the fit DID NOT change
        # (drives novelty resolution after sustained stability)
        self.stable_streak: Dict[int, int] = {i: 0 for i in range(world.n_vars)}
        if cost_weights:
            for var, w in cost_weights.items():
                if var in self.ledger.vars:
                    self.ledger.vars[var].cost_weight = w

        self.records: List[CycleRecord] = []
        self.full_audit_count = 0
        # skip_count: total non-audit decisions (sum of below). Kept for
        # backward-compat reporting.
        self.skip_count = 0
        # Skip categories — separate so the summary can show what kind of
        # work the framework actually saved (vs trass-collapsed which is no
        # work at all, vs cheap-path which IS prediction work via sentinels).
        self.trass_skip_count = 0
        self.compression_skip_count = 0
        self.sentinel_skip_count = 0
        self.composite_skip_count = 0
        self.component_skip_count = 0    # var-skips via HyperCompositeNethra sentinel
        self.pairwise_fallback_count = 0 # times component sentinel failed → pairwise ran
        self.total_interventions = 0
        # Graded cascade event counters — track the four outcomes of sentinel
        # failure separately so logs can show the distinction the policy recovers:
        #   sentinel_miss_count:       any sentinel failure (case b, world-changed branch)
        #   local_reaudit_count:       sentinel-failed var queued for and completed full audit
        #   signature_changed_count:   re-audit produced a different fit (genuine change)
        #   descendant_cascade_count:  vars reached by ledger.invalidate after confirmed change
        #   noisy_miss_no_cascade_count: re-audit found same fit → no cascade (noisy miss)
        self.sentinel_miss_count = 0
        self.local_reaudit_count = 0
        self.signature_changed_count = 0
        self.descendant_cascade_count = 0
        self.noisy_miss_no_cascade_count = 0
        # Repair-authority tracking
        # oscillation_count: fit changed for a var that had already changed fit before
        #   (correct→wrong or wrong→correct reversal). Distinguishes repair failure from
        #   genuine world change.
        # budget_escalation_count: times any var's audit budget was stepped up due to
        #   repeated repair failures.
        self.oscillation_count = 0
        self.budget_escalation_count = 0
        # Per-var repair state (not on VarNethra — operative agent policy, not cert data)
        # _var_repair_failures: sentinel fired → same fit found. Resets on genuine change.
        # _var_budget_escalation: escalated intervention budget for that var (abs value).
        # _var_sig_changes: total fit changes for oscillation detection.
        self._var_repair_failures: Dict[int, int] = {}
        self._var_budget_escalation: Dict[int, int] = {}
        self._var_sig_changes: Dict[int, int] = {}
        # Attention contract: trass/compression paths are deliberately lazy.
        # Do not add periodic revalidation, salience polling, or compression
        # spot-checks just to protect hidden truth. Escalate only when observed
        # downstream behavior invalidates the cheap path.
        # Diagnostic ledger: one FitDiagnostic per full audit, used for
        # offline analysis. Truth fields filled here are never read by the
        # agent for action decisions.
        self.fit_diagnostics: List[FitDiagnostic] = []
        # _last_fit_diag removed — FitDiagnostic now passed explicitly via AuditResult
        # Deferral counters: distinguish "this var is hard to fit" from
        # "this var keeps getting bumped off the schedule by budget."
        self.defer_count: Dict[int, int] = {i: 0 for i in range(world.n_vars)}
        self.defer_streak: Dict[int, int] = {i: 0 for i in range(world.n_vars)}
        self.max_defer_streak: Dict[int, int] = {i: 0 for i in range(world.n_vars)}

        # Tie tracking: per-var, per-tie-set count of how often that exact set
        # of hypotheses tied for rank 1 across audits. Always-on diagnostic.
        # Used by extension module to detect stable equivalence classes.
        # tie_log[var][frozenset({(parents, func), ...})] = count
        self.tie_log: Dict[int, Dict[FrozenSet[Tuple[Tuple[int, ...], str]], int]] = {}

        # Probe retention cap: if >0, only keep per-probe arrays for the most
        # recent K FitDiagnostics per variable. 0 = keep all. Memory tradeoff.
        self.probe_retention_per_var: int = 0

        # Near-tie margin: hypotheses scoring within this many probes of the
        # best are treated as operationally equivalent under the current context
        # and tracked as a TiedFrontier on the VarNethra.
        self.near_tie_margin: int = 4

        # Extension dispatcher: extension module loaded via --mode v29.
        # When non-None, _install_var calls extension.derive_compressions(...)
        # after the existing _discover_compressions to add derived compressions.
        # Default None preserves v28-only behavior.
        self.extension: Optional[AgentExtension] = None
        self.extension_modes: Set[str] = set()

        # Topological order cache. Invalidated when parent structure changes.
        self._topo_cache: Optional[List[int]] = None
        self._topo_cache_visible_count: int = -1

        # Sparse init: how many vars to full-audit at cold start.
        self.frontier_k: int = frontier_k
        # Per-target parent screen: top-M candidates by sensitivity to this target.
        # 0 = disabled (use certified-only pool, old behavior).
        self.parent_screen_m: int = parent_screen_m
        # Vars screened as causally inert at init (no downstream movement across
        # 0.05/0.95 perturbation range). Not admitted to frontier unless woken by
        # a descendant sentinel failure or direct dependency event.
        self._inert_vars: Set[int] = set()

        # Dormant partition: certified+stable vars are removed from the hot
        # pass. Re-entry is failure-driven: only a sentinel failure (cascade
        # invalidation) wakes a dormant var. None until initialize() runs.
        self._live_set: Optional[Set[int]] = None
        # Minimum envelope age before a var is eligible for dormancy.
        self._min_dormant_cert_age: int = 100

        # Joint false-trass tracking: vars whose certs were invalidated by
        # sentinel failure this cycle. Cleared at start of each run_cycle.
        # Trass pairs where both appear here are candidates for joint test.
        self._uncertain_this_cycle: Set[int] = set()

        # Regime detection: collects per-cycle cert failure/repair events and
        # clusters them into recurring co-failure patterns (regimes).
        self.regime_register: RegimeRegister = RegimeRegister()
        self._regime_skip_count = 0
        self._regime_sentinel_passes = 0
        self._regime_sentinel_fails = 0
        self._regime_no_sentinel = 0
        self._last_regime_failed_vars: Set[int] = set()

        # Sentinel parking counters.
        self._parked_skip_count = 0   # cycles where parked var was skipped
        self._woken_count = 0         # times a parked var was woken

        # Passive residual monitoring counters.
        # Passive: compute expected-next-state from certified func(parents) against
        # actual world.state. If residual within envelope → skip active sentinel
        # (saves IVs). If stressed → run active sentinel. Stressed co-occurrences
        # feed the regime register as candidate evidence.
        self._passive_saved_iv = 0    # IVs saved by passive-OK skips
        self._passive_stress_count = 0  # vars where passive was stressed (active ran)

        # ── Hybrid control providers ──────────────────────────────────────────
        # When residual_predictor is None (default / hybrid-off), the passive
        # residual block runs the inline FUNC_LIBRARY path unchanged.
        # When set (hybrid-interfaces mode), the block delegates to the provider.
        # In either case, NO provider may issue certs or mutate ledger state.
        if residual_predictor is not None:
            self._residual_predictor: Optional[ResidualPredictor] = residual_predictor
        else:
            self._residual_predictor = None

        self._parent_ranker: Optional[ParentRanker] = parent_ranker
        self._probe_proposer: Optional[ProbeProposer] = probe_proposer
        self._expert_router: Optional[ExpertRouter] = expert_router

        # Hybrid counters — only incremented when providers are active.
        # Diagnostic only; never influence cert decisions.
        self._hybrid_residual_predictor_calls: int = 0
        self._hybrid_residual_ok: int = 0
        self._hybrid_residual_stressed: int = 0
        self._hybrid_parent_ranker_calls: int = 0
        self._hybrid_probe_proposer_calls: int = 0
        self._hybrid_expert_router_calls: int = 0
        self._parent_proposal_diagnostics = ParentProposalDiagnostics()
        self._probe_proposal_diagnostics = ProbeProposalDiagnostics()
        self._pending_parent_rankings: Dict[int, Tuple[int, ...]] = {}
        self._pending_parent_sources: Dict[int, Dict[int, str]] = {}
        # Optional diagnostic observer. It may inspect audit inputs/results, but
        # no authority path reads it and no decision is allowed to depend on it.
        self._diagnostic_audit_observer: Optional[Any] = None

        # Repair agenda: structural planning surface for pending repairs.
        # When disabled (default), needs_audit drives repair as before.
        # When enabled, each needs_audit entry also gets a RepairAgendaItem
        # with scope/authority metadata for later A*-style triage.
        self._repair_agenda_enabled: bool = repair_agenda_enabled
        # "observe" = current topo order (default); "priority" = agenda.pop() order.
        # Priority ordering is consequence-tier only (#SHORTCUT); A* reserved for later.
        self._repair_agenda_ordering: str = repair_agenda_ordering
        self._repair_agenda: RepairAgenda = RepairAgenda()

        # ── Shadow residual predictor (Stage 3A — shadow mode only) ──────────
        # Never used for gating, certs, skips, or any operative decision.
        # shadow_residual_enabled=True only when shadow_residual_predictor is set.
        self._shadow_residual_predictor: Optional[ShadowLearnedResidualPredictor] = shadow_residual_predictor
        self._shadow_residual_enabled: bool = (
            shadow_residual_enabled and shadow_residual_predictor is not None
        )
        self._shadow_key_authority: Optional[ShadowResidualKeyAuthority] = (
            shadow_key_authority if self._shadow_residual_enabled else None
        )
        # Shadow diagnostic counters — increment each cycle for vars where
        # shadow prediction runs. Never influence cert or skip decisions.
        self._shadow_residual_calls: int = 0
        self._shadow_residual_ok: int = 0
        self._shadow_residual_stressed: int = 0
        self._shadow_residual_insufficient: int = 0
        self._shadow_false_ok_vs_symbolic: int = 0
        self._shadow_false_stress_vs_symbolic: int = 0
        self._shadow_agree_symbolic: int = 0
        self._shadow_would_save_iv: int = 0
        self._shadow_would_miss_symbolic_stress: int = 0
        self._shadow_false_ok_vs_active_sentinel: int = 0
        self._shadow_would_miss_active_failure: int = 0
        # Feature-conditioned calibrator key-usage counters (nonzero only in feature mode)
        self._shadow_feature_key_func_var: int = 0
        self._shadow_feature_key_func_tier_parentcount: int = 0
        self._shadow_feature_key_func_tier: int = 0
        self._shadow_feature_key_func: int = 0
        self._shadow_feature_key_global: int = 0
        self._shadow_feature_key_insufficient: int = 0
        # false_ok per key (feature mode only; insufficient cannot produce false_ok)
        self._shadow_feature_fok_func_var: int = 0
        self._shadow_feature_fok_func_tier_parentcount: int = 0
        self._shadow_feature_fok_func_tier: int = 0
        self._shadow_feature_fok_func: int = 0
        self._shadow_feature_fok_global: int = 0

        # Uncertainty consolidation. Default is off, preserving behavior.
        # Shadow records only. Assist writes bounded hints for existing
        # attention/probe/repair surfaces; it never issues or revokes authority.
        if uncertainty_consolidation_mode not in {"off", "shadow", "assist"}:
            raise ValueError(
                "uncertainty_consolidation_mode must be off, shadow, or assist"
            )
        if uncertainty_assist_policy not in {
            "all",
            "budget_only",
            "probe_only",
            "preserve_only",
            "priority_only",
            "local_only",
        }:
            raise ValueError(
                "uncertainty_assist_policy must be all, budget_only, probe_only, "
                "preserve_only, priority_only, or local_only"
            )
        self._uncertainty_consolidation_mode = uncertainty_consolidation_mode
        self._uncertainty_assist_policy = uncertainty_assist_policy
        self._uncertainty_max_preserve_count = max(0, int(uncertainty_max_preserve_count))
        self._uncertainty_consolidation_interval = 1
        self._uncertainty_latest_cases = []
        self._uncertainty_latest_clusters = []
        self._uncertainty_latest_assists = []
        self._uncertainty_summary = {
            "uncertainty_clusters": 0,
            "uncertainty_cases_seen": 0,
            "uncertainty_compression_ratio": 0.0,
            "max_cluster_size": 0,
            "avg_cluster_size": 0.0,
        }
        self._uncertainty_assist_vars: Dict[int, Set[str]] = {}
        self._uncertainty_forced_probes: Dict[int, Tuple[Tuple[int, float], ...]] = {}
        self._uncertainty_budget_bonus: Dict[int, int] = {}
        self._uncertainty_preserve_vars: Set[int] = set()
        self._uncertainty_preserve_remaining: int = 0
        self._uncertainty_cases_seen_total = 0
        self._uncertainty_clusters_total = 0
        self._uncertainty_assists_total = 0
        self._uncertainty_assist_prioritize_attention = 0
        self._uncertainty_assist_preserve_alternatives = 0
        self._uncertainty_assist_request_probe = 0
        self._uncertainty_assist_increase_monitoring = 0
        self._uncertainty_assist_repair_priority_bonus = 0
        self._uncertainty_assist_noops = 0
        self._uncertainty_max_cluster_size = 0
        self._uncertainty_cluster_size_sum = 0
        self._uncertainty_cluster_observations = 0
        self._uncertainty_cluster_specificity_sum = 0.0
        self._uncertainty_giant_clusters_suppressed = 0
        self._uncertainty_assists_suppressed_by_specificity_gate = 0
        self._uncertainty_assists_applied_from_local_clusters = 0
        self._uncertainty_assists_applied_from_giant_clusters = 0
        self._uncertainty_assist_extra_budget_total = 0
        self._uncertainty_assist_extra_probe_total = 0
        self._uncertainty_assist_preserved_alternative_total = 0
        self._uncertainty_assist_priority_hint_total = 0

        # Context-role provenance for reusable learned nethras. Default off
        # preserves behavior and avoids recording overhead.
        if nethra_reservoir_mode is not None:
            context_role_index_mode = nethra_reservoir_mode
        if context_role_index_mode not in {"off", "record", "assist_feature"}:
            raise ValueError(
                "context_role_index_mode must be off, record, or assist_feature"
            )
        self._context_role_index_mode = context_role_index_mode
        if context_role_anchor_policy is None:
            context_role_anchor_policy = (
                "strict" if context_role_index_mode == "assist_feature" else "off"
            )
        if context_role_anchor_policy not in {"off", "strict", "loose"}:
            raise ValueError(
                "context_role_anchor_policy must be off, strict, or loose"
            )
        if context_role_index_mode != "assist_feature":
            context_role_anchor_policy = "off"
        self._context_role_anchor_policy = context_role_anchor_policy
        self._context_role_index = (
            ContextRoleIndex() if context_role_index_mode != "off" else None
        )
        if self._context_role_index is not None:
            self._context_role_index.anchor_policy = self._context_role_anchor_policy
        self._context_role_index_latest_matches = []

        if authority_strength_mode not in {"off", "record", "assist"}:
            raise ValueError(
                "authority_strength_mode must be off, record, or assist"
            )
        if authority_strength_controller not in {"legacy", "state"}:
            raise ValueError(
                "authority_strength_controller must be legacy or state"
            )
        if authority_derivation_policy is None:
            authority_derivation_policy = (
                "shadow"
                if authority_strength_mode == "assist"
                and authority_strength_controller == "state"
                else "off"
            )
        if authority_derivation_policy not in {
            "off",
            "quarantine_persistent",
            "quarantine_repair_only",
            "shadow",
        }:
            raise ValueError(
                "authority_derivation_policy must be off, quarantine_persistent, "
                "quarantine_repair_only, or shadow"
            )
        self._authority_strength_mode = authority_strength_mode
        self._authority_strength_controller_mode = authority_strength_controller
        self._authority_derivation_policy = authority_derivation_policy
        self._authority_strength_latest_records = []
        self._authority_strength_latest_summary = summarize_authority_strength_records([])
        self._authority_state_controller = AuthorityStateController(
            derivation_policy=authority_derivation_policy,  # type: ignore[arg-type]
        )
        self._authority_strength_budget_bonus: Dict[int, int] = {}
        self._authority_strength_preserve_vars: Set[int] = set()
        self._authority_strength_preserve_remaining: int = 0
        self._authority_strength_repair_priority_vars: Set[int] = set()
        self._authority_strength_future_requirement_vars: Set[int] = set()
        self._authority_strength_derivation_quarantined_vars: Set[int] = set()
        self._authority_strength_monitoring_increases_total = 0
        self._authority_strength_alternatives_preserved_total = 0
        self._authority_strength_future_requirements_total = 0
        self._authority_strength_repair_priority_bumps_total = 0

        # Passive background-familiarity index.  Default off preserves behavior.
        if background_nethra_mode not in {"off", "record", "assist_feature"}:
            raise ValueError(
                "background_nethra_mode must be off, record, or assist_feature"
            )
        self._background_nethra_mode = background_nethra_mode
        self._background_nethra_index: Optional[BackgroundNethraIndex] = (
            BackgroundNethraIndex(mode=background_nethra_mode)
            if background_nethra_mode != "off"
            else None
        )
        if scaffold_memory_mode not in {"off", "record", "assist_feature"}:
            raise ValueError(
                "scaffold_memory_mode must be off, record, or assist_feature"
            )
        self._scaffold_memory_mode = scaffold_memory_mode
        self._scaffold_memory_index = (
            scaffold_memory_index if scaffold_memory_mode != "off" else None
        )
        self._current_cycle_for_memory = 0
        if nethra_memory_mode not in {"off", "record", "assist"}:
            raise ValueError("nethra_memory_mode must be off, record, or assist")
        self._nethra_memory_mode = nethra_memory_mode
        self._nethra_memory_index = (
            nethra_memory_index if nethra_memory_mode != "off" else None
        )

    def scaffold_memory_metrics(self) -> Dict[str, Any]:
        if self._scaffold_memory_index is None:
            return {
                "scaffold_memory_ranking_applications": 0,
                "scaffold_memory_candidates_reordered": 0,
                "scaffold_memory_top1_supported": 0,
                "scaffold_memory_topk_supported": 0,
                "scaffold_memory_broad_generic_noops": 0,
                "scaffold_memory_no_runtime_hook_available": 0,
                "scaffold_memory_feature_examples": [],
            }
        return self._scaffold_memory_index.runtime_metrics()

    def nethra_memory_metrics(self) -> Dict[str, Any]:
        if self._nethra_memory_index is None:
            return {
                "persistent_nethras_loaded": 0,
                "persistent_nethras_used": 0,
                "sleep_products_loaded": 0,
                "sleep_products_used": 0,
                "nethra_memory_behavior_effects": 0,
                "nethra_memory_authority_effects": 0,
                "nethra_memory_candidate_reorders": 0,
                "nethra_memory_probe_reorders": 0,
                "nethra_memory_soft_filter_fallbacks": 0,
                "nethra_memory_hard_filter_rejected": 0,
                "nethra_memory_block_events": 0,
                "nethra_memory_lookups": 0,
                "nethra_memory_matches": 0,
                "nethra_memory_use_right_counts": {},
                "nethra_memory_examples": [],
            }
        return self._nethra_memory_index.runtime_metrics()

    def nethra_memory_experience_export(self) -> List[Dict[str, Any]]:
        if self._nethra_memory_index is None:
            return []
        return self._nethra_memory_index.export_experience_events()

    def _context_role_index_enabled(self) -> bool:
        return self._context_role_index is not None

    def _context_role_index_record_var_fit(
        self,
        var: int,
        cycle: int,
        source: str = "audit",
        fit_diag: Optional[FitDiagnostic] = None,
    ) -> str:
        if self._context_role_index is None:
            return ""
        n = self.ledger.vars[var]
        nid = var_fit_id(var, tuple(n.parents), n.func)
        self._context_role_index.add_or_update_node(NethraNode(
            nethra_id=nid,
            kind="var_fit",
            target_var=var,
            components=tuple(sorted((var, *n.parents))),
            learned_parents=tuple(n.parents),
            learned_func=n.func,
            signature=f"x{var}:{n.func}({','.join(map(str, n.parents))})",
            first_seen_cycle=cycle,
            last_seen_cycle=cycle,
            observations=1,
            active_probe_count=len(fit_diag.probes) if fit_diag is not None else 0,
            source=source,  # type: ignore[arg-type]
        ))
        return nid

    def _context_role_index_assign_role(
        self,
        nethra_id: str,
        *,
        var: int,
        cycle: int,
        operation: str,
        role: str,
        evidence_summary: str = "",
        witness_probes: Tuple = (),
        fit_diag: Optional[FitDiagnostic] = None,
        route_role: str = "",
        uncertainty_signals: Tuple[str, ...] = (),
        validity_scope: Tuple[int, ...] = (),
    ) -> None:
        # Passive background observation: runs independently of context_role_index.
        if nethra_id and self._background_nethra_index is not None:
            n = self.ledger.vars[var]
            self._background_nethra_index.add_or_update_from_context_role(
                nethra_id=nethra_id,
                role=role,
                var=var,
                context_key=nethra_context_key(
                    operation=operation,
                    var=var,
                    visible=self.world.visible_count,
                    parents=tuple(n.parents),
                ),
                cycle=cycle,
                operation_role=operation,
                fit_signature=(
                    f"x{var}:{n.func}({','.join(map(str, n.parents))})"
                ),
                parents=tuple(n.parents),
                signals=uncertainty_signals,
            )
        if self._context_role_index is None or not nethra_id:
            return
        if operation in {"route", "composite", "regime", "compression"}:
            for support_var in validity_scope:
                if support_var != var and not self._authority_strength_derivation_allowed(
                    support_var,
                    cycle=cycle,
                    blocked_handle_kind="context-role",
                    blocked_target=f"{operation}:{nethra_id}",
                ):
                    return
        n = self.ledger.vars[var]
        revoked = sum(
            1 for cert in list(n.certificates.values()) + list(n.route_certs.values())
            if getattr(cert, "revoked_by", None)
        )
        self._context_role_index.assign_context_role(ContextRoleRecord(
            nethra_id=nethra_id,
            context_key=nethra_context_key(
                operation=operation,
                var=var,
                visible=self.world.visible_count,
                parents=tuple(n.parents),
            ),
            operation=operation,
            role=role,  # type: ignore[arg-type]
            cycle=cycle,
            evidence_summary=evidence_summary,
            witness_probes=witness_probes,
            sentinel_passes=sum(
                int(getattr(cert, "sentinel_passes", 0) or 0)
                for cert in n.certificates.values()
            ),
            sentinel_failures=int(n.consecutive_sentinel_failures),
            fit_margin=fit_diag.margin if fit_diag is not None else None,
            tie_count=len(fit_diag.tie_set) if fit_diag is not None else 0,
            near_tie_count=len(fit_diag.near_tie_candidates) if fit_diag is not None else 0,
            strong_observations=int(n.strong_observations),
            revocations=revoked,
            skip_role=n.role_for("skip"),
            route_role=route_role,
            uncertainty_signals=uncertainty_signals,
            validity_scope=validity_scope or tuple(sorted((var, *n.parents))),
        ))

    def _context_role_index_record_candidate(
        self,
        *,
        prefix: str,
        kind: str,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        cycle: int,
        source: str,
        role: str,
        score: int = 0,
        context_operation: str = "candidate",
        fit_diag: Optional[FitDiagnostic] = None,
    ) -> str:
        if self._context_role_index is None:
            return ""
        nid = candidate_id(prefix, var, parents, func)
        self._context_role_index.add_or_update_node(NethraNode(
            nethra_id=nid,
            kind=kind,  # type: ignore[arg-type]
            target_var=var,
            components=tuple(sorted((var, *parents))),
            learned_parents=tuple(parents),
            learned_func=func,
            signature=f"x{var}:{func}({','.join(map(str, parents))})",
            first_seen_cycle=cycle,
            last_seen_cycle=cycle,
            observations=1,
            active_probe_count=len(fit_diag.probes) if fit_diag is not None else 0,
            source=source,  # type: ignore[arg-type]
        ))
        self._context_role_index_assign_role(
            nid,
            var=var,
            cycle=cycle,
            operation=context_operation,
            role=role,
            evidence_summary=f"score={score}",
            fit_diag=fit_diag,
            validity_scope=tuple(sorted((var, *parents))),
        )
        return nid

    def _reset_uncertainty_assist_surfaces(self) -> None:
        self._uncertainty_assist_vars = {}
        self._uncertainty_forced_probes = {}
        self._uncertainty_budget_bonus = {}
        self._uncertainty_preserve_vars = set()
        self._uncertainty_preserve_remaining = self._uncertainty_max_preserve_count

    def _run_uncertainty_consolidation(self, cycle: int) -> None:
        """Run visible-evidence consolidation for shadow/assist modes.

        Off mode never calls this method. Shadow mode records diagnostics only.
        Assist mode writes bounded hints consumed by existing repair/probe paths.
        """
        self._reset_uncertainty_assist_surfaces()
        if self._uncertainty_consolidation_mode == "off":
            return
        if cycle % self._uncertainty_consolidation_interval != 0:
            return

        cases = extract_uncertainty_cases_from_agent(self, cycle)
        if (
            self._scaffold_memory_mode == "assist_feature"
            and self._scaffold_memory_index is not None
            and len(cases) >= 2
        ):
            context = nethra_context_key(
                operation="uncertainty_cluster",
                visible=self.world.visible_count,
            )
            cases = list(
                self._scaffold_memory_index.rank_uncertainty_local_anchors(
                    -1,
                    context,
                    cases,
                )
            )
        clusters = cluster_uncertainty_cases(cases, visible_count=self.world.visible_count)
        assists = propose_consolidation_assists(clusters)
        summary = summarize_clusters(clusters)

        self._uncertainty_latest_cases = cases
        self._uncertainty_latest_clusters = clusters
        self._uncertainty_latest_assists = assists
        self._uncertainty_summary = summary
        index_matches_by_cluster: Dict[str, Tuple[Any, ...]] = {}
        if self._context_role_index is not None:
            for cluster in clusters:
                nid = f"uncertainty_cluster:{cluster.cluster_id}:{','.join(map(str, cluster.vars))}"
                self._context_role_index.add_or_update_node(NethraNode(
                    nethra_id=nid,
                    kind="unknown",
                    target_var=None,
                    components=tuple(sorted(set(cluster.vars) | set(cluster.shared_parents) | set(cluster.shared_graph_neighbors))),
                    learned_parents=tuple(cluster.shared_parents),
                    learned_func=str(cluster.proposed_handle_kind),
                    signature=f"{cluster.proposed_handle_kind}:{cluster.evidence_summary}",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                    observations=max(1, len(cluster.vars)),
                    source="uncertainty_cluster",
                ))
                self._context_role_index.assign_context_role(ContextRoleRecord(
                    nethra_id=nid,
                    context_key=nethra_context_key(operation="uncertainty_cluster", visible=self.world.visible_count),
                    operation="uncertainty_consolidation",
                    role="unresolved",
                    cycle=cycle,
                    evidence_summary=cluster.evidence_summary,
                    fit_margin=None,
                    near_tie_count=cluster.shared_near_tie_count,
                    uncertainty_signals=tuple(cluster.shared_signals),
                    validity_scope=tuple(cluster.vars),
                ))
                if (
                    self._context_role_index_mode == "assist_feature"
                    and self._context_role_anchor_policy != "off"
                    and not cluster.is_giant_cluster
                ):
                    index_matches_by_cluster[cluster.cluster_id] = (
                        self._context_role_index.query_for_uncertainty_cluster(
                            cluster,
                            anchor_policy=self._context_role_anchor_policy,
                            current_cycle=cycle,
                        )
                    )
            self._context_role_index_latest_matches = [
                match
                for matches in index_matches_by_cluster.values()
                for match in matches
            ]
        # Passive background observation of uncertainty clusters.
        if self._background_nethra_index is not None:
            for cluster in clusters:
                bg_nid = (
                    f"bg_uc:{cluster.proposed_handle_kind}:"
                    f"{','.join(map(str, cluster.shared_parents or cluster.vars[:4]))}"
                )
                self._background_nethra_index.add_or_update_from_uncertainty_cluster(
                    nethra_id=bg_nid,
                    vars=cluster.vars,
                    context_key=nethra_context_key(
                        operation="uncertainty_cluster",
                        visible=self.world.visible_count,
                    ),
                    cycle=cycle,
                    is_giant=cluster.is_giant_cluster,
                    signals=cluster.shared_signals,
                    parents=cluster.shared_parents,
                )

        self._uncertainty_cases_seen_total += len(cases)
        self._uncertainty_clusters_total += len(clusters)
        for cluster in clusters:
            size = len(cluster.vars)
            self._uncertainty_max_cluster_size = max(self._uncertainty_max_cluster_size, size)
            self._uncertainty_cluster_size_sum += size
            self._uncertainty_cluster_specificity_sum += cluster.shared_signal_specificity
            self._uncertainty_cluster_observations += 1

        if self._uncertainty_consolidation_mode != "assist":
            return

        visible = self.world.visible_count
        clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
        suppressed_giant_clusters: Set[str] = set()
        for assist in assists:
            applied = False
            target_vars = tuple(v for v in assist.target_vars if 0 <= v < visible)
            if not target_vars:
                self._uncertainty_assist_noops += 1
                continue
            cluster = clusters_by_id.get(assist.cluster_id)
            is_local_cluster = (
                cluster_has_specific_local_anchor(cluster)
                if cluster is not None else False
            )
            index_local_anchor = bool(
                cluster is not None
                and index_matches_by_cluster.get(cluster.cluster_id)
            )
            index_anchor_required = bool(index_local_anchor and not is_local_cluster)
            if index_local_anchor:
                is_local_cluster = True
            is_giant_cluster = bool(cluster and cluster.is_giant_cluster)
            if not is_local_cluster or (
                self._uncertainty_assist_policy == "local_only" and is_giant_cluster
            ):
                self._uncertainty_assists_suppressed_by_specificity_gate += 1
                if is_giant_cluster:
                    suppressed_giant_clusters.add(assist.cluster_id)
                continue
            self._uncertainty_assists_total += 1

            if assist.assist_kind == "prioritize_attention":
                self._uncertainty_assist_prioritize_attention += 1
                if self._uncertainty_assist_policy in {"all", "budget_only", "local_only"}:
                    for var in target_vars:
                        before = self._uncertainty_budget_bonus.get(var, 0)
                        after = max(
                            before,
                            min(2, max(1, int(math.ceil(assist.bounded_strength * 4)))),
                        )
                        self._uncertainty_budget_bonus[var] = after
                        self._uncertainty_assist_extra_budget_total += after - before
                    applied = True
            elif assist.assist_kind == "preserve_alternatives":
                self._uncertainty_assist_preserve_alternatives += 1
                if self._uncertainty_assist_policy in {"all", "preserve_only", "local_only"}:
                    self._uncertainty_preserve_vars.update(target_vars)
                    applied = bool(self._uncertainty_max_preserve_count > 0)
            elif assist.assist_kind == "request_separating_probe":
                self._uncertainty_assist_request_probe += 1
                if self._uncertainty_assist_policy in {"all", "probe_only", "local_only"}:
                    for var in target_vars:
                        frontier = self.ledger.vars[var].tied_frontier
                        probes = frontier.separating_probes if frontier is not None else ()
                        valid = tuple(
                            (iv_var, iv_val)
                            for iv_var, iv_val in probes
                            if 0 <= iv_var < visible and 0.0 <= iv_val <= 1.0
                        )[:3]
                        if valid:
                            self._uncertainty_forced_probes[var] = valid
                            self._uncertainty_assist_extra_probe_total += len(valid)
                            applied = True
            elif assist.assist_kind == "increase_monitoring":
                self._uncertainty_assist_increase_monitoring += 1
                if self._uncertainty_assist_policy in {"all", "budget_only", "local_only"}:
                    for var in target_vars:
                        before = self._uncertainty_budget_bonus.get(var, 0)
                        after = max(before, 1)
                        self._uncertainty_budget_bonus[var] = after
                        self._uncertainty_assist_extra_budget_total += after - before
                    applied = True
            elif assist.assist_kind == "repair_priority_bonus":
                self._uncertainty_assist_repair_priority_bonus += 1
                if self._uncertainty_assist_policy in {"all", "priority_only", "local_only"}:
                    self._uncertainty_assist_priority_hint_total += len(target_vars)
                    applied = True

            if applied:
                for var in target_vars:
                    self._uncertainty_assist_vars.setdefault(var, set()).add(assist.assist_kind)
                if (
                    index_anchor_required
                    and self._context_role_index is not None
                    and self._context_role_index.can_record_index_assist_for_cycle(cycle)
                ):
                    anchor = index_matches_by_cluster.get(assist.cluster_id, ())[0]
                    self._context_role_index.record_index_assist(
                        cycle=cycle,
                        assist_kind=assist.assist_kind,
                        cluster_id=assist.cluster_id,
                        nethra_id=anchor.nethra_id,
                        match_reason=anchor.match_reason,
                        changed_budget=assist.assist_kind in {
                            "prioritize_attention",
                            "increase_monitoring",
                        } and self._uncertainty_assist_policy in {
                            "all",
                            "budget_only",
                            "local_only",
                        },
                        changed_probes=assist.assist_kind == "request_separating_probe",
                        changed_preservation=assist.assist_kind == "preserve_alternatives",
                        changed_priority=assist.assist_kind == "repair_priority_bonus",
                        outcome=self._context_role_assist_outcome(target_vars),
                    )
                if is_giant_cluster:
                    self._uncertainty_assists_applied_from_giant_clusters += 1
                else:
                    self._uncertainty_assists_applied_from_local_clusters += 1

            if not applied:
                self._uncertainty_assist_noops += 1
        self._uncertainty_giant_clusters_suppressed += len(suppressed_giant_clusters)

    def _context_role_assist_outcome(self, target_vars: Tuple[int, ...]) -> Dict[str, Any]:
        sentinel_failure = False
        revocation = False
        novelty_persistence = False
        audit_count = 0
        fit_signatures: Set[Tuple[Tuple[int, ...], str]] = set()
        target_set = set(target_vars)
        for var in target_vars:
            if not (0 <= var < self.world.visible_count):
                continue
            n = self.ledger.vars[var]
            sentinel_failure = sentinel_failure or int(n.consecutive_sentinel_failures) > 0
            revocation = revocation or any(
                getattr(cert, "revoked_by", None)
                for cert in list(n.certificates.values()) + list(n.route_certs.values())
            )
            novelty_persistence = novelty_persistence or any(
                int(getattr(nv, "affected_var", -1)) == int(var)
                and getattr(nv, "status", "") == "open"
                for nv in getattr(self.ledger, "novelty", ()) or ()
            )
            audit_count += int(getattr(n, "full_audits", 0) or 0)
        for fd in getattr(self, "fit_diagnostics", ()) or ():
            if int(getattr(fd, "var", -1)) in target_set:
                fit_signatures.add((
                    tuple(int(p) for p in getattr(fd, "best_parents", ()) or ()),
                    str(getattr(fd, "best_func", "")),
                ))
        return {
            "sentinel_failure": sentinel_failure,
            "revocation": revocation,
            "fit_churn": len(fit_signatures) > 1,
            "novelty_persistence": novelty_persistence,
            "audit_count": audit_count,
        }

    def uncertainty_consolidation_metrics(self) -> Dict[str, Any]:
        avg_cluster = (
            self._uncertainty_cluster_size_sum / self._uncertainty_cluster_observations
            if self._uncertainty_cluster_observations else 0.0
        )
        avg_specificity = (
            self._uncertainty_cluster_specificity_sum / self._uncertainty_cluster_observations
            if self._uncertainty_cluster_observations else 0.0
        )
        cluster_count = (
            self._uncertainty_summary.get("uncertainty_clusters", 0)
            if self._uncertainty_consolidation_mode != "off" else 0
        )
        case_count = (
            self._uncertainty_summary.get("uncertainty_cases_seen", 0)
            if self._uncertainty_consolidation_mode != "off" else 0
        )
        return {
            "uncertainty_consolidation_mode": self._uncertainty_consolidation_mode,
            "uncertainty_assist_policy": self._uncertainty_assist_policy,
            "uncertainty_cases_seen": case_count,
            "uncertainty_clusters": cluster_count,
            "uncertainty_compression_ratio": (
                self._uncertainty_summary.get("uncertainty_compression_ratio", 0.0)
                if self._uncertainty_consolidation_mode != "off" else 0.0
            ),
            "consolidation_assists_total": self._uncertainty_assists_total,
            "assist_prioritize_attention": self._uncertainty_assist_prioritize_attention,
            "assist_preserve_alternatives": self._uncertainty_assist_preserve_alternatives,
            "assist_request_probe": self._uncertainty_assist_request_probe,
            "assist_increase_monitoring": self._uncertainty_assist_increase_monitoring,
            "assist_repair_priority_bonus": self._uncertainty_assist_repair_priority_bonus,
            "assist_noops": self._uncertainty_assist_noops,
            "max_cluster_size": self._uncertainty_max_cluster_size,
            "avg_cluster_size": avg_cluster,
            "cluster_specificity_mean": avg_specificity,
            "giant_cluster_count": int(self._uncertainty_summary.get("giant_cluster_count", 0)),
            "giant_clusters_suppressed": self._uncertainty_giant_clusters_suppressed,
            "assists_suppressed_by_specificity_gate": (
                self._uncertainty_assists_suppressed_by_specificity_gate
            ),
            "assists_applied_from_local_clusters": (
                self._uncertainty_assists_applied_from_local_clusters
            ),
            "assists_applied_from_giant_clusters": (
                self._uncertainty_assists_applied_from_giant_clusters
            ),
            "assist_extra_budget_total": self._uncertainty_assist_extra_budget_total,
            "assist_extra_probe_total": self._uncertainty_assist_extra_probe_total,
            "assist_preserved_alternative_total": (
                self._uncertainty_assist_preserved_alternative_total
            ),
            "assist_priority_hint_total": self._uncertainty_assist_priority_hint_total,
        }

    def context_role_index_metrics(self) -> Dict[str, Any]:
        if self._context_role_index is None:
            return {
                "context_role_index_mode": self._context_role_index_mode,
                "context_role_index_nodes": 0,
                "context_role_records": 0,
                "context_role_tareth": 0,
                "context_role_trass": 0,
                "context_role_unresolved": 0,
                "context_role_best_available": 0,
                "context_role_index_queries": 0,
                "context_role_index_matches": 0,
                "context_role_raw_matches": 0,
                "context_role_deduped_matches": 0,
                "context_role_matches_suppressed_weak": 0,
                "context_role_matches_suppressed_duplicate": 0,
                "context_role_matches_suppressed_cap": 0,
                "context_role_matches_used_as_local_anchor": 0,
                "context_role_assist_feature_hits": 0,
                "context_role_anchor_policy": self._context_role_anchor_policy,
                "context_role_assist_pressure_events": 0,
                "context_role_assist_pressure_per_cycle": 0,
                "context_role_top_match_reasons": {},
                "context_role_nodes_by_kind": {},
                "context_role_nodes_by_source": {},
                "context_roles_by_context": {},
                "context_roles_by_role": {},
                "nethra_reservoir_records": 0,
                "nethra_context_roles": 0,
                "nethra_role_tareth": 0,
                "nethra_role_trass": 0,
                "nethra_role_unresolved": 0,
                "nethra_role_best_available": 0,
                "reservoir_queries": 0,
                "reservoir_matches": 0,
                "reservoir_raw_matches": 0,
                "reservoir_deduped_matches": 0,
                "reservoir_matches_used_as_local_anchor": 0,
                "reservoir_assist_feature_hits": 0,
                "reservoir_records_by_kind": {},
                "reservoir_records_by_source": {},
                "reservoir_roles_by_context": {},
                "reservoir_roles_by_role": {},
                # Surface metric stubs for off mode
                "role_surface_count": 0,
                "load_bearing_surface_count": 0,
                "residual_surface_count": 0,
                "residual_bucket_count": 0,
                "residual_pressure_total": 0.0,
                "residual_pressure_mean": 0.0,
                "residual_recent_growth_total": 0.0,
                "residual_absorbed_count": 0,
                "residual_unresolved_count": 0,
                "residual_clarity_mean": 0.0,
                "regime_transition_candidates_from_residuals": 0,
                "residual_pressure_persistent_growth_windows": 0,
            }
        summary = self._context_role_index.summarize()
        summary["context_role_index_mode"] = self._context_role_index_mode
        summary["nethra_reservoir_mode"] = self._context_role_index_mode
        summary["context_role_anchor_policy"] = self._context_role_anchor_policy
        return summary

    def context_role_index_export(self, limit: int = 200) -> Dict[str, Any]:
        if self._context_role_index is None:
            return {"nodes": [], "edges": [], "roles": [], "records": []}
        return self._context_role_index.export_records(limit=limit)

    def nethra_reservoir_metrics(self) -> Dict[str, Any]:
        return self.context_role_index_metrics()

    def nethra_reservoir_export(self, limit: int = 200) -> Dict[str, Any]:
        return self.context_role_index_export(limit=limit)

    def _reset_authority_strength_assist_surfaces(self) -> None:
        self._authority_strength_budget_bonus = {}
        self._authority_strength_preserve_vars = set()
        self._authority_strength_preserve_remaining = 3
        self._authority_strength_repair_priority_vars = set()
        self._authority_strength_future_requirement_vars = set()
        self._authority_strength_derivation_quarantined_vars = set()

    def _authority_strength_derivation_allowed(
        self,
        var: int,
        *,
        context_key: Optional[str] = None,
        cycle: Optional[int] = None,
        blocked_handle_kind: str = "unknown",
        blocked_target: Optional[str] = None,
    ) -> bool:
        if (
            self._authority_strength_mode != "assist"
            or self._authority_strength_controller_mode != "state"
        ):
            return True
        if cycle is None:
            cycle = self.records[-1].cycle if self.records else 0
        return self._authority_state_controller.check_derivation_gate(
            var,
            context_key,
            cycle=int(cycle),
            blocked_handle_kind=blocked_handle_kind,
            blocked_target=blocked_target,
        )

    def _run_authority_strength(self, cycle: int) -> None:
        """Record visible-evidence strength and write bounded assist hints."""
        self._reset_authority_strength_assist_surfaces()
        if self._authority_strength_mode == "off":
            return

        records = compute_authority_strength_records(self, cycle)
        self._authority_strength_latest_records = records

        if (
            self._authority_strength_mode == "assist"
            and self._authority_strength_controller_mode == "legacy"
        ):
            for record in records:
                if not record.best_available:
                    continue
                if record.required_future_evidence:
                    self._authority_strength_future_requirement_vars.add(record.var)
                    self._authority_strength_future_requirements_total += 1
                if record.strength in {"weak", "contested"}:
                    before = self._authority_strength_budget_bonus.get(record.var, 0)
                    self._authority_strength_budget_bonus[record.var] = max(before, 1)
                    if before == 0:
                        self._authority_strength_monitoring_increases_total += 1
                if record.strength == "contested":
                    self._authority_strength_preserve_vars.add(record.var)
                    self._authority_strength_repair_priority_vars.add(record.var)
                    self._authority_strength_repair_priority_bumps_total += 1
        elif self._authority_strength_mode == "assist":
            result = self._authority_state_controller.process(records, cycle)
            self._authority_strength_budget_bonus = dict(result.monitoring_bonus_vars)
            self._authority_strength_preserve_vars = set(result.preserve_vars)
            self._authority_strength_preserve_remaining = (
                3 if result.preserve_vars else 0
            )
            self._authority_strength_repair_priority_vars = set(result.repair_priority_vars)
            self._authority_strength_future_requirement_vars = set(
                result.future_requirement_vars
            )
            self._authority_strength_derivation_quarantined_vars = set(
                result.derivation_quarantined_vars
            )
            summary = self._authority_state_controller.summary()
            self._authority_strength_monitoring_increases_total = int(
                summary.get("monitoring_increases_from_strength_applied", 0)
            )
            self._authority_strength_repair_priority_bumps_total = int(
                summary.get("repair_priority_bumps_from_strength_applied", 0)
            )
            self._authority_strength_future_requirements_total = int(
                summary.get("authority_debt_created", 0)
            )

        self._authority_strength_latest_summary = summarize_authority_strength_records(
            records,
            monitoring_increases=self._authority_strength_monitoring_increases_total,
            alternatives_preserved=self._authority_strength_alternatives_preserved_total,
        )

    def _run_background_nethra(self, cycle: int) -> None:
        """Passive per-cycle background-familiarity scan.

        Observes tied frontiers, dormant alternatives, and authority states
        from the current ledger snapshot. Never issues authority, revokes,
        suppresses skips, forces probes, or increases monitoring.
        """
        bgi = self._background_nethra_index
        if bgi is None:
            return
        visible = getattr(self.world, "visible_count", 0)
        ctx_key = nethra_context_key(
            operation="background_scan",
            visible=visible,
        )
        for var in range(visible):
            n = self.ledger.vars[var]
            parents = tuple(sorted(int(p) for p in (n.parents or ())))

            # Tied frontiers
            frontier = getattr(n, "tied_frontier", None)
            if frontier is not None:
                candidates = getattr(frontier, "candidates", ())
                nid = f"bg_frontier:x{var}:{n.func}({','.join(map(str, parents))})"
                bgi.add_or_update_from_tied_frontier(
                    nethra_id=nid,
                    var=var,
                    context_key=ctx_key,
                    cycle=cycle,
                    candidate_count=len(candidates),
                    stable_count=int(getattr(frontier, "stable_count", 0)),
                    parents=parents,
                )

            # Dormant alternatives
            for alt in getattr(n, "dormant_alternatives", ()) or ():
                alt_parents = tuple(sorted(
                    int(p) for p in (getattr(alt, "parents", ()) or ())
                ))
                alt_func = str(getattr(alt, "func", ""))
                nid = (
                    f"bg_dormant:x{var}:{alt_func}"
                    f"({','.join(map(str, alt_parents))})"
                )
                bgi.add_or_update_from_dormant_alternative(
                    nethra_id=nid,
                    var=var,
                    context_key=ctx_key,
                    cycle=cycle,
                    revival_count=int(getattr(alt, "revival_count", 0)),
                    parents=alt_parents,
                )

        # Authority state patterns (only if authority_strength ran)
        if self._authority_strength_mode != "off":
            for record in self._authority_strength_latest_records:
                if record.authority_state not in {
                    "contested_best_available",
                    "quarantined_for_derivation",
                    "repair_candidate",
                }:
                    continue
                var = record.var
                parents = tuple(sorted(
                    int(p) for p in (self.ledger.vars[var].parents or ())
                ))
                bgi.add_or_update_from_authority_debt(
                    nethra_id=f"bg_auth:{record.nethra_id}",
                    var=var,
                    context_key=str(record.context_key),
                    cycle=cycle,
                    authority_state=str(record.authority_state),
                    parents=parents,
                    signals=tuple(record.uncertainty_signals),
                )

    def _run_background_residual_classification(self, cycle: int) -> None:
        """Passive per-cycle residual pressure classification.

        Uses a strictly capped budget so it never competes with focal audit.
        Invariants: never issues certs, revokes certs, suppresses skips, forces
        probes, increases monitoring, or reads hidden truth fields.
        Record mode matches off mode on all operational metrics.
        """
        if self._context_role_index is None:
            return
        store = self._context_role_index.role_surfaces
        if not store._buckets:
            return
        budget = _BACKGROUND_RESIDUAL_BUDGET
        store.classify_background_residuals(cycle, budget)
        store.decay_residuals(cycle, budget)
        store.check_persistent_growth()
        store.regime_transition_candidates(
            cycle=cycle,
            min_pressure=2.0,
            min_growth=0.0,
            min_co_shift=2,
        )

    def background_nethra_metrics(self) -> Dict[str, Any]:
        if self._background_nethra_index is None:
            return {
                "background_nethra_mode": self._background_nethra_mode,
                "background_nethra_records": 0,
                "background_nethra_by_kind": {},
                "background_nethra_edges": 0,
                "background_contexts_seen": 0,
                "background_role_shift_examples": 0,
                "background_trass_patterns": 0,
                "background_unresolved_patterns": 0,
                "background_quarantined_patterns": 0,
                "background_giant_cluster_patterns": 0,
                "background_dormant_patterns": 0,
                "background_tied_frontier_patterns": 0,
                "background_recognition_score_mean": 0.0,
                "background_action_relevance_score_mean": 0.0,
                "background_records_used_as_features": 0,
                "background_feature_hits": 0,
                "background_feature_noops": 0,
                "familiar_background_count": 0,
                "operational_authority_count": 0,
            }
        summary = self._background_nethra_index.summarize()
        summary["background_nethra_mode"] = self._background_nethra_mode
        return summary

    def background_nethra_export(self, limit: int = 200) -> Dict[str, Any]:
        if self._background_nethra_index is None:
            return {"records": [], "edges": [], "role_shift_examples": []}
        return self._background_nethra_index.export_records(limit=limit)

    def authority_strength_metrics(self) -> Dict[str, Any]:
        zero_controller_metrics: Dict[str, Any] = {
            "authority_strength_controller": self._authority_strength_controller_mode,
            "authority_derivation_policy": self._authority_derivation_policy,
            "authority_state_counts": {},
            "authority_debt_created": 0,
            "authority_debt_persisted": 0,
            "authority_debt_paid": 0,
            "authority_debt_escalated": 0,
            "authority_debt_deescalated": 0,
            "authority_debt_outstanding": 0,
            "debt_age_mean": 0.0,
            "debt_age_max": 0,
            "authority_state_transitions": 0,
            "derivation_quarantines": 0,
            "derivation_gate_checks": 0,
            "derivation_gate_allowed": 0,
            "derivation_gate_blocked": 0,
            "derivation_gate_would_block": 0,
            "derivation_gate_shadow_would_block": 0,
            "derivation_gate_blocked_by_state": {},
            "derivation_gate_blocked_by_reason": {},
            "derivation_gate_blocked_by_handle_kind": {},
            "action_reason_specificity": {},
            "local_use_preserved": 0,
            "repair_candidates": 0,
            "bounded_repairs_applied": 0,
            "monitoring_hints_applied": 0,
            "monitoring_hints_suppressed": 0,
            "repair_hints_suppressed": 0,
            "debt_noops": 0,
            "authority_action_candidates": 0,
            "authority_actions_applied": 0,
            "authority_noop_state_not_permit": 0,
            "authority_suppressed_cooldown": 0,
            "authority_suppressed_budget": 0,
            "authority_suppressed_local_use_only": 0,
            "authority_suppressed_derivation_only": 0,
            "generic_contested_noop": 0,
            "authority_action_regime_sentinel_failure_attribution": 0,
            "authority_action_activated_failing_regime_sentinel": 0,
            "monitoring_increases_from_strength_candidates": 0,
            "monitoring_increases_from_strength_applied": 0,
            "monitoring_increases_from_strength_suppressed_by_state": 0,
            "monitoring_increases_from_strength_suppressed_by_cooldown": 0,
            "monitoring_increases_from_strength_suppressed_by_budget": 0,
            "monitoring_increases_from_strength_noops": 0,
            "repair_priority_bumps_from_strength_candidates": 0,
            "repair_priority_bumps_from_strength_applied": 0,
            "repair_priority_bumps_from_strength_suppressed_by_state": 0,
            "repair_priority_bumps_from_strength_suppressed_by_cooldown": 0,
            "repair_priority_bumps_from_strength_suppressed_by_budget": 0,
            "repair_priority_bumps_from_strength_noops": 0,
        }
        if self._authority_strength_mode == "off":
            return {
                "authority_strength_mode": "off",
                "authority_derivation_policy": self._authority_derivation_policy,
                "authority_strength_records": 0,
                "strength_strong": 0,
                "strength_usable": 0,
                "strength_weak": 0,
                "strength_contested": 0,
                "strength_insufficient": 0,
                "weak_best_available": 0,
                "contested_best_available": 0,
                "monitoring_increases_from_strength": 0,
                "alternatives_preserved_from_strength": 0,
                "future_evidence_requirements": 0,
                "repair_priority_bumps_from_strength": 0,
                "authority_strength_counts_by_reason": {},
                **zero_controller_metrics,
            }
        if not self._authority_strength_latest_records:
            self._authority_strength_latest_records = compute_authority_strength_records(
                self,
                self.records[-1].cycle if self.records else 0,
            )
            self._authority_strength_latest_summary = summarize_authority_strength_records(
                self._authority_strength_latest_records,
                monitoring_increases=self._authority_strength_monitoring_increases_total,
                alternatives_preserved=self._authority_strength_alternatives_preserved_total,
            )
        summary = authority_strength_summary_to_dict(self._authority_strength_latest_summary)
        counts = summary.get("counts_by_strength", {})
        controller_summary = (
            self._authority_state_controller.summary()
            if (
                self._authority_strength_mode == "assist"
                and self._authority_strength_controller_mode == "state"
            )
            else {}
        )
        if not controller_summary:
            controller_summary = {
                **zero_controller_metrics,
                "authority_state_counts": dict(
                    summary.get("counts_by_authority_state", {})
                ),
            }
        else:
            controller_summary = {
                **zero_controller_metrics,
                **controller_summary,
                "authority_strength_controller": self._authority_strength_controller_mode,
            }
        return {
            "authority_strength_mode": self._authority_strength_mode,
            "authority_strength_controller": self._authority_strength_controller_mode,
            "authority_derivation_policy": self._authority_derivation_policy,
            "authority_strength_records": len(self._authority_strength_latest_records),
            "strength_strong": int(counts.get("strong", 0)),
            "strength_usable": int(counts.get("usable", 0)),
            "strength_weak": int(counts.get("weak", 0)),
            "strength_contested": int(counts.get("contested", 0)),
            "strength_insufficient": int(counts.get("insufficient", 0)),
            "weak_best_available": int(summary.get("weak_best_available", 0)),
            "contested_best_available": int(summary.get("contested_best_available", 0)),
            "monitoring_increases_from_strength": (
                self._authority_strength_monitoring_increases_total
            ),
            "alternatives_preserved_from_strength": (
                self._authority_strength_alternatives_preserved_total
            ),
            "future_evidence_requirements": int(
                summary.get("future_evidence_requirements", 0)
            ),
            "repair_priority_bumps_from_strength": (
                self._authority_strength_repair_priority_bumps_total
            ),
            "authority_strength_counts_by_reason": dict(summary.get("counts_by_reason", {})),
            **controller_summary,
        }

    def authority_strength_export(self, limit: int = 300) -> Dict[str, Any]:
        if self._authority_strength_mode == "off":
            return {"records": [], "summary": authority_strength_summary_to_dict(
                summarize_authority_strength_records([])
            ), "controller": {}}
        if not self._authority_strength_latest_records:
            self._authority_strength_latest_records = compute_authority_strength_records(
                self,
                self.records[-1].cycle if self.records else 0,
            )
            self._authority_strength_latest_summary = summarize_authority_strength_records(
                self._authority_strength_latest_records,
                monitoring_increases=self._authority_strength_monitoring_increases_total,
                alternatives_preserved=self._authority_strength_alternatives_preserved_total,
            )
        limit = max(0, int(limit))
        return {
            "records": authority_strength_records_to_dicts(
                self._authority_strength_latest_records[:limit]
            ),
            "summary": authority_strength_summary_to_dict(
                self._authority_strength_latest_summary
            ),
            "controller": (
                self._authority_state_controller.summary()
                if (
                    self._authority_strength_mode == "assist"
                    and self._authority_strength_controller_mode == "state"
                )
                else {}
            ),
        }

    def _adaptive_probe_budget(self, n_hypotheses: int) -> int:
        """Return the number of probes to use for a hypothesis space of size
        n_hypotheses.

        Only scales UP from the base budget. The base is the floor: with
        small restricted spaces the probe-per-hypothesis ratio is already
        sufficient at the base value, and reducing it further makes 1-parent
        vs 2-parent discrimination unreliable. Scaling up for large unrestricted
        spaces compensates for the higher ambiguity there.

          n_hyp ≤ 225  → base probes   (default 30)
          n_hyp = 600  → 49 probes
          n_hyp = 2841 → 107 probes, capped at 2 × base (60 when base=30)

        The floor guarantees: restricted enumeration (typically ≤ 500 hyp
        for n_available ≤ 14) always runs at base. Unrestricted enumeration
        (n_vars²-scale hypothesis spaces) gets proportionally more budget,
        so early-cycle cold-start fits improve without undersampling the
        small-available-parent regime.

        Authority scales implicitly: for hypothesis spaces that exceed
        2 × base even at the cap, probe-per-hypothesis ratio still falls —
        score margins are narrower, promotion is slower, the fit carries
        lower inherent authority without any additional gating."""
        scaled = int(math.ceil(math.sqrt(n_hypotheses) * 2.0))
        return max(self.intervention_budget, min(self.intervention_budget * 2, scaled))

    def _full_audit_var(self, var: int, cycle: int) -> Tuple[Tuple[int, ...], str, int, int, FitDiagnostic]:
        """Run a full hypothesis-space search for one variable. Steps:
          1. Build available_parents set (exclude only cert-excluded candidates)
          2. Call fit_var which enumerates, scores, and ranks hypotheses
          3. Record FitDiagnostic for offline analysis
        Returns (best_parents, best_func, best_score, second_score, fit_diag).
        fit_diag is passed explicitly to _install_var — no side-channel.
        Increments full_audit_count and total_interventions."""
        cycle = int(getattr(cycle, "cycle", cycle))
        self.full_audit_count += 1
        n = self.ledger.vars[var]
        # available_parents: either from a per-target sensitivity screen (sparse
        # mode, parent_screen_m > 0) or from the certified/trass pool (full mode).
        #
        # Screen path: probe every candidate at 0.05/0.95, rank by |Δtarget|,
        # keep top M. Does not require candidates to be pre-certified — any var
        # can be a parent as long as it moves the target. Route certs (trass role
        # for this target) are respected: explicitly excluded vars are dropped.
        #
        # Certified pool path (legacy): include certified/trass/proposed+sentinel
        # vars that haven't been route-cert-excluded for this target.
        # Q4: no joint composition test. Individual route certs (when they exist)
        # won't guarantee the combination is non-redundant.
        if self.parent_screen_m > 0:
            available = self._screen_candidate_parents(var, self.parent_screen_m)
        else:
            available = {
                other_var for other_var, other_n in self.ledger.vars.items()
                if other_var != var
                and (n.route_certs.get(other_var) is None or n.route_certs[other_var].role != "trass")
                and (
                    other_n.status == "certified"
                    or other_n.status == "trass"
                    or other_n.role_for("skip") == "trass"
                    or (other_n.status == "proposed" and bool(other_n.sentinels))
                )
            }
        # Estimate hypothesis space size so the probe budget can scale with
        # ambiguity. Always use restricted formula: fit_var now uses restricted
        # enumeration for any explicitly-provided available set (even empty).
        # Empty available → _n_hyp = 2 (constants only). Full set → full restricted.
        _n_av = len(available)
        _n_hyp = 2 + _n_av + (_n_av * (_n_av - 1) // 2 * 5)
        # Adaptive probe budget: scale up with hypothesis space size.
        # Only scales UP from base (see _adaptive_probe_budget docstring).
        # Failure-earned repair escalation (_var_budget_escalation) can exceed
        # the adaptive value; take the max so escalation still dominates.
        # P1-A: activated — was previously computed but discarded.
        _adaptive = self._adaptive_probe_budget(_n_hyp)
        budget = max(self._var_budget_escalation.get(var, 0), _adaptive)
        budget += self._uncertainty_budget_bonus.get(var, 0)
        if self._authority_strength_mode == "assist":
            budget += self._authority_strength_budget_bonus.get(var, 0)
        diag_dict: Dict[str, object] = {
            "cycle": cycle,
            "var": var,
            "status_before": n.status,
            "role_before": n.role_for("skip"),
            "available_parents": tuple(sorted(available)),
        }
        # P1-B: if this var has an active TiedFrontier with separating probes,
        # inject them as forced inclusions so the tie has a chance to resolve.
        _frontier_probes: Tuple[Tuple[int, float], ...] = (
            n.tied_frontier.separating_probes
            if n.tied_frontier is not None and n.tied_frontier.separating_probes
            else ()
        )
        if self._probe_proposer is not None and hasattr(self._probe_proposer, "observe_frontier_probes"):
            self._probe_proposer.observe_frontier_probes(var, _frontier_probes)  # type: ignore[attr-defined]

        # ProbeProposer: provider may suggest additional forced probes.
        # Provider probes are merged with frontier probes; they do NOT certify
        # anything — scoring and cert decisions remain in the standard audit path.
        # Invalid probes (out-of-range var or value outside [0,1]) are dropped
        # and logged as structured diagnostic events, not fatal errors.
        _provider_probes: Tuple[Tuple[int, float], ...] = ()
        if self._probe_proposer is not None:
            _pp = self._probe_proposer.propose_probes(var, available, budget)
            self._hybrid_probe_proposer_calls += 1
            _valid: List[Tuple[int, float]] = []
            _invalid_count = 0
            for _iv_var, _iv_val in _pp.probes:
                if not (0 <= _iv_var < self.world.visible_count):
                    _invalid_count += 1
                    self.ledger.emit(LedgerEvent(
                        type="provider_diagnostic", var=var, cycle=cycle,
                        payload={"provider": "probe_proposer",
                                 "event": "invalid_probe_var",
                                 "iv_var": _iv_var, "iv_val": _iv_val},
                    ))
                    continue
                if not (0.0 <= _iv_val <= 1.0):
                    _invalid_count += 1
                    self.ledger.emit(LedgerEvent(
                        type="provider_diagnostic", var=var, cycle=cycle,
                        payload={"provider": "probe_proposer",
                                 "event": "invalid_probe_val",
                                 "iv_var": _iv_var, "iv_val": _iv_val},
                    ))
                    continue
                _valid.append((_iv_var, _iv_val))
            _provider_probes = tuple(_valid)
            self._probe_proposal_diagnostics.record_proposal(
                proposed=len(_pp.probes),
                valid=len(_provider_probes),
                invalid=_invalid_count,
            )

        # Merge provider probes with frontier probes; pass None when both are empty
        # so fit_var's default discrimination pool is used unchanged.
        _merged_probes: Optional[Tuple[Tuple[int, float], ...]] = None
        _consolidation_probes: Tuple[Tuple[int, float], ...] = (
            self._uncertainty_forced_probes.get(var, ())
            if self._uncertainty_consolidation_mode == "assist"
            else ()
        )
        if _provider_probes or _frontier_probes or _consolidation_probes:
            _merged_probes = _provider_probes + _frontier_probes + _consolidation_probes
        if (
            self._nethra_memory_index is not None
            and _merged_probes
        ):
            _merged_probes = self._nethra_memory_index.rank_probes(
                var=var,
                context_key=nethra_context_key(
                    operation="probe_candidates",
                    var=var,
                    visible=self.world.visible_count,
                    parents=tuple(sorted(available)),
                ),
                probes=tuple(_merged_probes),
                cycle=cycle,
            )

        result = fit_var(var, self.world, self.rng, budget,
                         n.current_tolerance, available_parents=available, diag=diag_dict,
                         near_tie_margin=self.near_tie_margin,
                         forced_probes=_merged_probes)
        self.total_interventions += budget

        # ExpertRouter: diagnostic call only. Output is NOT used to choose parents,
        # choose function, score hypotheses, authorize skips, or issue certs.
        # Route metadata is stored as a structured LedgerEvent for offline analysis.
        if self._expert_router is not None:
            _er_context: Dict = {
                "cycle": cycle,
                "budget": budget,
                "best_parents": tuple(result[0]),
                "best_func": result[1],
            }
            _, _er_meta = self._expert_router.route(var, available, _er_context)
            self._hybrid_expert_router_calls += 1
            self.ledger.emit(LedgerEvent(
                type="provider_diagnostic", var=var, cycle=cycle,
                payload={"provider": "expert_router", "route_meta": _er_meta},
            ))

        fd = FitDiagnostic(
            cycle=int(diag_dict["cycle"]),
            var=int(diag_dict["var"]),
            status_before=str(diag_dict["status_before"]),
            role_before=str(diag_dict["role_before"]),
            available_parents=tuple(diag_dict["available_parents"]),
            restricted=bool(diag_dict.get("restricted", False)),
            hypothesis_count=int(diag_dict.get("hypothesis_count", -1)),
            best_score=int(diag_dict.get("best_score", -1)),
            second_score=int(diag_dict.get("second_score", -1)),
            margin=int(diag_dict.get("margin", -1)),
            best_parents=tuple(diag_dict.get("best_parents", ())),
            best_func=str(diag_dict.get("best_func", "?")),
            failure_class=str(diag_dict.get("failure_class", "unknown")),
            probes=tuple(diag_dict.get("probes", ())),
            actuals=tuple(diag_dict.get("actuals", ())),
            pick_preds=tuple(diag_dict.get("pick_preds", ())),
            tie_set=diag_dict.get("tie_set", frozenset()),
            near_tie_candidates=tuple(diag_dict.get("near_tie_candidates", ())),
            near_tie_context_key=int(diag_dict.get("near_tie_context_key", 0)),
        )
        self.fit_diagnostics.append(fd)
        if self._parent_ranker is not None and var in self._pending_parent_rankings:
            _ranked = self._pending_parent_rankings.pop(var, ())
            _sources = self._pending_parent_sources.pop(var, {})
            self._parent_proposal_diagnostics.record_fit(_ranked, tuple(result[0]), _sources)
            if hasattr(self._parent_ranker, "observe_fit_result"):
                self._parent_ranker.observe_fit_result(var, tuple(result[0]), int(fd.margin))  # type: ignore[attr-defined]
            if hasattr(self._parent_ranker, "observe_probe_results"):
                self._parent_ranker.observe_probe_results(var, fd.probes, fd.actuals)  # type: ignore[attr-defined]
        if self._probe_proposer is not None:
            self._probe_proposal_diagnostics.record_fit(
                _provider_probes,
                fd.probes,
                int(fd.margin),
            )
        # Tie-tracking: bump count for this var's tie set if size > 1
        if len(fd.tie_set) > 1:
            self.tie_log.setdefault(var, {})
            self.tie_log[var][fd.tie_set] = self.tie_log[var].get(fd.tie_set, 0) + 1
        # Probe retention cap (default unlimited; --probe-retention K to cap)
        if self.probe_retention_per_var > 0:
            var_diags = [d for d in self.fit_diagnostics if d.var == var]
            if len(var_diags) > self.probe_retention_per_var:
                oldest = var_diags[-(self.probe_retention_per_var + 1)]
                oldest.probes = ()
                oldest.actuals = ()
                oldest.pick_preds = ()
                oldest.truth_preds = None
        parents, func, score, second = result
        return parents, func, score, second, fd

    def _certify_operation_role(self, var: int, cycle: int) -> str:
        """Substitution test: does perturbing `var` change other visible vars
        beyond their noise tolerances? Returns "tareth" (yes, track), "trass"
        (no, collapse), or "untested" (deferred — too few other visible vars).

        Method:
          1. If <2 other visible vars: defer (cannot test meaningfully).
          2. For each of 5 spread perturbations (0.05, 0.25, 0.5, 0.75, 0.95):
             skip if too close to current; else issue n=5 baseline samples and
             n=5 perturbed samples via the world's intervention path.
          3. For each other var j, compare the AVERAGE baseline vs AVERAGE
             perturbed (averaging cancels per-sample noise). If |Δavg| > j's
             current_tolerance, count as a change.
          4. Verdict: "tareth" if at least half the perturbations produced
             changes; otherwise "trass".

        Per-j tolerance (not var's tolerance) because each variable's noise
        envelope governs what counts as a real change in that variable.
        Multi-sample averaging is necessary because a single noisy probe
        comparison can show "change" purely from independent noise draws.
        """
        n = self.ledger.vars[var]
        if self.salience_targets is not None and var in self.salience_targets:
            if n.role_for("skip") != "tareth":
                self.ledger.emit(LedgerEvent(
                    type="role_changed", var=var, cycle=cycle,
                    payload={"from": n.role_for("skip"), "to": "tareth",
                             "reason": "declared_salience_target"},
                ))
                self.ledger.issue_cert(
                    var, "skip", "tareth", "skip",
                    context_parents=tuple(n.parents) if n.parents else (),
                    context_visible=self.world.visible_count, context_cycle=cycle,
                    targets=(), substitutions_tested=("declared_salience",),
                    changes=0, trials=0,
                    earned_by="manual_bootstrap",
                )
                nid = self._context_role_index_record_var_fit(var, cycle, source="operation_role")
                self._context_role_index_assign_role(
                    nid,
                    var=var,
                    cycle=cycle,
                    operation="skip",
                    role="tareth",
                    evidence_summary="declared_salience_target",
                    validity_scope=(var,),
                )
            return "tareth"

        n_other_visible = sum(1 for j in range(self.world.visible_count) if j != var)
        if n_other_visible < 2:
            if n.role_for("skip") == "untested":
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} operation_role test DEFERRED "
                    f"(only {n_other_visible} other visible vars; need ≥2)"
                )
                nid = self._context_role_index_record_var_fit(var, cycle, source="operation_role")
                self._context_role_index_assign_role(
                    nid,
                    var=var,
                    cycle=cycle,
                    operation="skip",
                    role="unresolved",
                    evidence_summary="operation_role_deferred",
                    validity_scope=(var,),
                )
            return "untested"

        # Q6a: scope of the cert — the targets actually tested.
        # WHAT IT OUGHT TO DO: the cert's authority extends exactly over the vars that
        # were tested. Including untested vars in the cert's scope would be overclaiming;
        # excluding tested vars would be underclaiming. Under live-frontier mode,
        # cert-trass vars are already shortcut-skipped (they don't propagate within the
        # tested regime) so excluding them from change-counting is correct: they are
        # outside the cert's meaningful scope. Gate is cert only (role_for("skip") ==
        # "trass") — not status. Status alone has no scope, no witnesses, no authority.
        # FILTER LEDGER: if scope later expands (vars become tareth), the shortcut keeps
        # firing by default. Sentinel catches actual failures. No proactive recertification.
        filtered_targets_list: List[int] = []
        for j in range(self.world.visible_count):
            if j == var:
                continue
            if self.salience_targets is not None and j not in self.salience_targets:
                continue
            if self.role_salience == "live-frontier":
                if self.ledger.vars[j].role_for("skip") == "trass":
                    continue
            filtered_targets_list.append(j)
        filtered_targets = tuple(filtered_targets_list)

        spread_perturbs = [0.05, 0.25, 0.5, 0.75, 0.95]
        n_trials = 0  # incremented only for probes actually run (not skipped)
        n_samples_per = 5
        changes = 0
        # Q5: witnesses are attribution handles, not confirmation tokens.
        # WHAT IT OUGHT TO DO: each witness is the (state_snapshot, iv_val) pair that
        # earned the tareth claim — the specific context and intervention that produced
        # propagation. They are stored so that when a sentinel FAILS, the system can
        # open the cert and ask: "did the basis for this cert expire, or did the world
        # genuinely change?" If replaying the witnesses shows no propagation: the cert's
        # authority has expired (recertify). If witnesses still propagate but the sentinel
        # failed: the world changed in a way the sentinel correctly caught (proceed with
        # invalidation). Witnesses are for attribution under failure, not for every-cycle
        # confirmation. LAZY DECOMPOSITION: no witness replay until failure earns it.
        witnesses: List = []
        saved = self.world.state
        state_snapshot = tuple(saved)  # immutable snapshot for witness storage
        var_tol = self.ledger.vars[var].current_tolerance
        for iv_val in spread_perturbs:
            n_trials += 1
            if abs(iv_val - saved[var]) <= var_tol:
                continue
            baseline_sum = [0.0] * self.world.visible_count
            perturbed_sum = [0.0] * self.world.visible_count
            for _ in range(n_samples_per):
                self.world.state = saved
                b = self.world.predict_under_intervention(var, saved[var])
                for j in range(self.world.visible_count):
                    baseline_sum[j] += b[j]
                self.world.state = saved
                p = self.world.predict_under_intervention(var, iv_val)
                for j in range(self.world.visible_count):
                    perturbed_sum[j] += p[j]
            for j in filtered_targets_list:
                j_tol = self.ledger.vars[j].current_tolerance
                avg_baseline = baseline_sum[j] / n_samples_per
                avg_perturbed = perturbed_sum[j] / n_samples_per
                if abs(avg_baseline - avg_perturbed) > j_tol:
                    changes += 1
                    witnesses.append((state_snapshot, iv_val))
                    break
        self.world.state = saved
        role = "tareth" if changes > 0 else "trass"
        n = self.ledger.vars[var]
        prev_role = n.role_for("skip")
        if prev_role != role:
            self.ledger.emit(LedgerEvent(
                type="role_changed", var=var, cycle=cycle,
                payload={"from": prev_role, "to": role,
                         "changes": changes, "trials": n_trials},
            ))
        self.ledger.issue_cert(
            var, "skip", role,
            "skip" if role == "tareth" else "none",
            context_parents=tuple(n.parents) if n.parents else (),
            context_visible=self.world.visible_count,
            context_cycle=cycle,
            targets=filtered_targets,
            substitutions_tested=("perturbation",),
            changes=changes,
            trials=n_trials,
            earned_by="substitution_test",
            witnesses=tuple(witnesses) if role == "tareth" else (),
            audits_at_issuance=n.full_audits,
        )
        nid = self._context_role_index_record_var_fit(var, cycle, source="operation_role")
        self._context_role_index_assign_role(
            nid,
            var=var,
            cycle=cycle,
            operation="skip",
            role=role,
            evidence_summary=f"substitution_test changes={changes} trials={n_trials}",
            witness_probes=tuple(witnesses) if role == "tareth" else (),
            validity_scope=filtered_targets,
        )
        return role

    def _provisional_trass_probe(self, var: int) -> bool:
        """One-shot cheap op-role check for provisional trass detection.

        Picks the spread perturbation furthest from the current state value
        among [0.05, 0.5, 0.95], runs one baseline + one perturbed world
        query, returns True if any visible target var changed beyond tolerance
        (cert is stale and should be invalidated).

        Cost: 2 world queries. Called once per provisional-trass var per cycle
        to prevent wrong-trass lock-in without running the full 5×5-sample
        op-role test.

        P1-C: provisional trass must not suppress detection. A single probe is
        the minimum that makes 'provisional' meaningfully different from the
        hard-suppress path.
        """
        saved = self.world.state
        var_tol = self.ledger.vars[var].current_tolerance
        candidates = [v for v in (0.05, 0.5, 0.95) if abs(v - saved[var]) > var_tol]
        if not candidates:
            return False
        iv_val = max(candidates, key=lambda v: abs(v - saved[var]))
        baseline = self.world.predict_under_intervention(var, saved[var])
        self.world.state = saved
        perturbed = self.world.predict_under_intervention(var, iv_val)
        self.world.state = saved
        for j in range(self.world.visible_count):
            if j == var:
                continue
            j_tol = self.ledger.vars[j].current_tolerance
            if abs(baseline[j] - perturbed[j]) > j_tol:
                return True
        return False

    def _test_joint_false_trass(self, var_a: int, var_b: int, cycle: int) -> str:
        """Joint substitution test for two individually-trass vars.

        This method uses the agent's believed parent structure, not the true
        causal structure. It will miss joint effects when the agent's hypothesis
        about ancestry is wrong.

        Procedure — 5 trials across spread values [0.05, 0.25, 0.5, 0.75, 0.95]:
          R0:  baseline (no intervention)
          RA:  predict_under_intervention(var_a, val_a)
          RB:  predict_under_intervention(var_b, val_b)
          RAB: predict_under_joint_intervention({var_a: val_a, var_b: val_b})

          For each downstream tareth var j:
            if |RAB[j] - R0[j]| > tol AND |RA[j] - R0[j]| <= tol
               AND |RB[j] - R0[j]| <= tol → interaction evidence this trial

          Verdict: jointly tareth if interaction evidence in >= half of trials.

        Returns: "tareth" if jointly tareth (false_trass), "trass" if not,
                 "untested" if no tareth downstream vars to test against.

        On jointly-tareth verdict: installs a CompositeNethra on the ledger
        (the durable cert for the joint relationship) and writes false_trass
        certs on both vars. The composite carries one representative probe
        (first interacting val_a, val_b, sentinel_j) for cheap per-cycle
        replay. No invalidate_certs call — the composite cert is the authority;
        revocation happens when the composite sentinel fails in _check_composites.
        """
        if (
            not self._authority_strength_derivation_allowed(
                var_a,
                cycle=cycle,
                blocked_handle_kind="composite",
                blocked_target=f"x{var_a},x{var_b}",
            )
            or not self._authority_strength_derivation_allowed(
                var_b,
                cycle=cycle,
                blocked_handle_kind="composite",
                blocked_target=f"x{var_a},x{var_b}",
            )
        ):
            return "untested"
        saved = self.world.state
        spread = [0.05, 0.25, 0.5, 0.75, 0.95]

        sentinel_vars = [
            j for j in range(self.world.visible_count)
            if j != var_a and j != var_b
            and self.ledger.vars[j].role_for("skip") == "tareth"
        ]
        if not sentinel_vars:
            return "untested"

        interaction_trials = 0
        total_trials = 0
        # First interacting probe — stored as the composite sentinel probe.
        first_probe: Optional[Tuple[float, float, int, float]] = None  # val_a, val_b, j, tol_j
        first_joint_values: Optional[Tuple[float, float, float, float]] = None

        for val_a, val_b in zip(spread, spread):
            if abs(val_a - saved[var_a]) <= self.ledger.vars[var_a].current_tolerance:
                continue
            if abs(val_b - saved[var_b]) <= self.ledger.vars[var_b].current_tolerance:
                continue
            self.world.state = saved
            R0 = list(self.world.predict_under_joint_intervention({}))
            self.world.state = saved
            RA = list(self.world.predict_under_intervention(var_a, val_a))
            self.world.state = saved
            RB = list(self.world.predict_under_intervention(var_b, val_b))
            self.world.state = saved
            RAB = list(self.world.predict_under_joint_intervention({var_a: val_a, var_b: val_b}))

            interaction = False
            for j in sentinel_vars:
                if first_joint_values is None:
                    first_joint_values = (R0[j], RA[j], RB[j], RAB[j])
                jt = self.ledger.vars[j].current_tolerance
                if (abs(RAB[j] - R0[j]) > jt
                        and abs(RA[j] - R0[j]) <= jt
                        and abs(RB[j] - R0[j]) <= jt):
                    interaction = True
                    if first_probe is None:
                        first_probe = (val_a, val_b, j, jt)
                    break
            total_trials += 1
            if interaction:
                interaction_trials += 1

        self.world.state = saved

        if total_trials == 0:
            return "untested"

        jointly_tareth = interaction_trials * 2 >= total_trials
        if not jointly_tareth:
            return "trass"

        self.ledger.event_log.append(
            f"c{cycle}: x{var_a},x{var_b} JOINT FALSE-TRASS "
            f"({interaction_trials}/{total_trials} trials showed interaction)"
        )

        # Install composite nethra — the durable cert for the joint relationship.
        # The composite sentinel probe is the first (val_a, val_b, j) that showed
        # interaction; _check_composites replays it each cycle.
        if first_probe is not None:
            probe_va, probe_vb, probe_j, probe_tol = first_probe
            self.ledger.install_composite(
                members=(var_a, var_b),
                sentinel_var=probe_j,
                probe_val_a=probe_va,
                probe_val_b=probe_vb,
                tol=probe_tol,
                changes=interaction_trials,
                trials=total_trials,
                certified_at_cycle=cycle,
                context_visible=self.world.visible_count,
            )
            if self._context_role_index is not None:
                nid = f"composite:x{var_a},x{var_b}->x{probe_j}"
                self._context_role_index.add_or_update_node(NethraNode(
                    nethra_id=nid,
                    kind="composite",
                    target_var=probe_j,
                    components=tuple(sorted((var_a, var_b, probe_j))),
                    learned_parents=(var_a, var_b),
                    learned_func="joint_interaction",
                    signature=f"x{var_a},x{var_b}->x{probe_j}",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                    observations=1,
                    active_probe_count=total_trials,
                    composition_links=(
                        var_fit_id(var_a, tuple(self.ledger.vars[var_a].parents), self.ledger.vars[var_a].func),
                        var_fit_id(var_b, tuple(self.ledger.vars[var_b].parents), self.ledger.vars[var_b].func),
                    ),
                    source="composite",
                ))
                self._context_role_index_assign_role(
                    nid,
                    var=probe_j,
                    cycle=cycle,
                    operation="composite",
                    role="tareth",
                    evidence_summary=f"joint_interaction {interaction_trials}/{total_trials}",
                    witness_probes=((var_a, probe_va), (var_b, probe_vb), (probe_j, probe_tol)),
                    validity_scope=tuple(sorted((var_a, var_b, probe_j))),
                )
            self.ledger.emit(LedgerEvent(
                type="composite_installed", var=var_a, cycle=cycle,
                payload={"members": (var_a, var_b), "sentinel_var": probe_j,
                         "interactions": f"{interaction_trials}/{total_trials}"},
            ))

        # Joint evidence invalidates the shortcut authority of the individual
        # trass certs in this composition scope. The composite carries the joint
        # witness; individual certs must be retested before they shortcut again.
        r0, ra, rb, rab = first_joint_values or (None, None, None, None)
        for _member in (var_a, var_b):
            _cert = self.ledger.vars[_member].certificates.get("skip")
            if _cert is not None:
                _joint_updates: Dict[str, Any] = {
                    "role": _cert.role,
                    "revoked_by": _cert.revoked_by,
                }
                if not _cert.context_parents:
                    _joint_updates = {
                        "role": "untested",
                        "revoked_by": "composite_failure",
                        "changes": interaction_trials,
                        "trials": total_trials,
                    }
                self.ledger.vars[_member].certificates["skip"] = dataclasses.replace(
                    _cert,
                    joint_members=(var_a, var_b),
                    joint_R0=r0,
                    joint_RA=ra,
                    joint_RB=rb,
                    joint_RAB=rab,
                    **_joint_updates,
                )
        return "tareth"

    def _is_ancestor(self, v: int, target: int, _visited: Optional[Set[int]] = None) -> bool:
        """True if v is in the causal ancestry of target per the agent's ledger.
        Uses the agent's believed parent structure, not the true causal graph."""
        if _visited is None:
            _visited = set()
        parents = self.ledger.vars[target].parents
        if not parents:
            return False
        if v in parents:
            return True
        for p in parents:
            if p not in _visited:
                _visited.add(p)
                if self._is_ancestor(v, p, _visited):
                    return True
        return False

    def _check_composites(self, cycle: int) -> Set[int]:
        """Check all composite nethras for this cycle. Returns the set of vars
        covered by composites whose joint interaction probe still passes.

        For each composite:
          - Stale (visible_count changed): revoke immediately.
          - Sentinel var went trass: revoke (no longer a valid witness).
          - Replay R0 and RAB at the stored probe values; if |RAB-R0| > tol
            the interaction is still present → both member vars skip this cycle.
          - Otherwise: revoke composite, reset both members to untested.

        Revocation resets both vars' skip certs via invalidate_certs so they
        re-enter the audit queue on the next cycle as untested. Vars whose
        composite passes are returned in passing_members and take the
        composite-skip path in run_cycle's first-pass loop.
        """
        passing_members: Set[int] = set()
        to_remove: List[Tuple[CompositeNethra, str]] = []
        for cn in self.ledger.composites:
            a, b = cn.members
            # Activation-scoped: if both members are dormant (not in live_set),
            # the composite interaction has no active consequence path this cycle.
            # Assume passing — no probe needed (invariant 70: polling only when
            # tied to an active consequence path).
            if (self._live_set is not None
                    and a not in self._live_set
                    and b not in self._live_set):
                cn.pass_count += 1
                passing_members.add(a)
                passing_members.add(b)
                continue
            if cn.context_visible != self.world.visible_count:
                to_remove.append((cn, "stale_context"))
                continue
            if self.ledger.vars[cn.sentinel_var].role_for("skip") == "trass":
                to_remove.append((cn, "sentinel_trass"))
                continue
            saved = self.world.state
            R0_val = list(self.world.predict_under_joint_intervention({}))[cn.sentinel_var]
            self.world.state = saved
            RAB_val = list(self.world.predict_under_joint_intervention(
                {a: cn.probe_val_a, b: cn.probe_val_b}
            ))[cn.sentinel_var]
            self.world.state = saved
            self.total_interventions += 2
            if abs(RAB_val - R0_val) > cn.tol:
                cn.pass_count += 1
                passing_members.add(a)
                passing_members.add(b)
            else:
                to_remove.append((cn, "interaction_lost"))
        for cn, reason in to_remove:
            cn.revoke_reason = reason
            cn.revoked_at_cycle = cycle
            self.ledger.composites.remove(cn)
            self.ledger.revoked_composites.append(cn)
            a, b = cn.members
            na, nb = self.ledger.vars[a], self.ledger.vars[b]
            na.invalidate_certs("false_trass_contradiction")
            nb.invalidate_certs("false_trass_contradiction")
            if na.status == "trass":
                na.status = "proposed"
            if nb.status == "trass":
                nb.status = "proposed"
            self.ledger.event_log.append(
                f"c{cycle}: x{a},x{b} composite REVOKED ({reason}) "
                f"(probe ({cn.probe_val_a:.2f},{cn.probe_val_b:.2f})"
                f" → x{cn.sentinel_var})"
            )
        return passing_members

    def _check_hyper_composites(self, cycle: int) -> Set[int]:
        """Check each live HyperCompositeNethra with one joint probe.

        2 world calls per component regardless of member count.
        Pass: all members added to covered set, pass_count incremented.
        Fail: absorbed pairwise composites restored to ledger.composites,
              component annotated and moved to revoked list, pairwise_fallback_count
              incremented on agent and component.

        Returns covered_vars — caller excludes them from pairwise _check_composites.
        """
        covered: Set[int] = set()
        to_revoke: List[HyperCompositeNethra] = []
        for hc in self.ledger.hyper_composites:
            if hc.context_visible != self.world.visible_count:
                hc.revoke_reason = "stale_context"
                hc.revoked_at_cycle = cycle
                to_revoke.append(hc)
                continue
            if self.ledger.vars[hc.sentinel_var].role_for("skip") == "trass":
                hc.revoke_reason = "sentinel_trass"
                hc.revoked_at_cycle = cycle
                to_revoke.append(hc)
                continue
            saved = self.world.state
            R0_val = list(self.world.predict_under_joint_intervention({}))[hc.sentinel_var]
            self.world.state = saved
            RAB_val = list(self.world.predict_under_joint_intervention(hc.probe_values))[hc.sentinel_var]
            self.world.state = saved
            self.total_interventions += 2
            if abs(RAB_val - R0_val) > hc.tol:
                hc.pass_count += 1
                for v in hc.members:
                    covered.add(v)
                    self.component_skip_count += 1
            else:
                hc.pairwise_fallback_count += 1
                self.pairwise_fallback_count += 1
                hc.revoke_reason = "interaction_lost"
                hc.revoked_at_cycle = cycle
                to_revoke.append(hc)
        for hc in to_revoke:
            self.ledger.hyper_composites.remove(hc)
            self.ledger.revoked_hyper_composites.append(hc)
            # Restore absorbed pairwise composites so _check_composites covers them.
            restored = [
                cn for cn in self.ledger.absorbed_composites
                if set(cn.members) <= set(hc.members)
            ]
            for cn in restored:
                self.ledger.absorbed_composites.remove(cn)
                self.ledger.composites.append(cn)
            self.ledger.event_log.append(
                f"c{cycle}: component C{hc.component_id} ({len(hc.members)} members) "
                f"REVOKED ({hc.revoke_reason}), restored {len(restored)} pairwise composites"
            )
        return covered

    def _promote_dense_components(self, cycle: int) -> None:
        """Scan live pairwise composites for dense connected components and
        promote qualifying ones to HyperCompositeNethra handles.

        Promotion criteria (all must hold for a connected component):
          - size >= _COMPONENT_MIN_SIZE
          - edge density >= _COMPONENT_MIN_DENSITY  (edges / max_possible_edges)
          - all constituent pairwise composites have pass_count >= _COMPONENT_MIN_PASSES
          - >= _COMPONENT_SENTINEL_FRAC of pairs share the dominant sentinel_var

        Cost: O(len(composites)) per call. Called every _COMPONENT_PROMOTE_INTERVAL cycles.
        """
        if len(self.ledger.composites) < _COMPONENT_MIN_SIZE:
            return

        # Union-find to identify connected components.
        parent: Dict[int, int] = {}
        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)
                x = parent[x]
            return x
        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for cn in self.ledger.composites:
            a, b = cn.members
            if a not in parent:
                parent[a] = a
            if b not in parent:
                parent[b] = b
            union(a, b)

        # Group composites by component root.
        from collections import defaultdict as _defaultdict, Counter as _Counter
        comp_pairs: Dict[int, List[CompositeNethra]] = _defaultdict(list)
        for cn in self.ledger.composites:
            comp_pairs[find(cn.members[0])].append(cn)

        for _, pairs in comp_pairs.items():
            members_set = {v for cn in pairs for v in cn.members}
            size = len(members_set)
            if size < _COMPONENT_MIN_SIZE:
                continue
            if any(
                not self._authority_strength_derivation_allowed(
                    member,
                    cycle=cycle,
                    blocked_handle_kind="composite",
                    blocked_target="dense_component",
                )
                for member in members_set
            ):
                continue
            max_edges = size * (size - 1) // 2
            density = len(pairs) / max_edges if max_edges > 0 else 0.0
            if density < _COMPONENT_MIN_DENSITY:
                continue
            passes_ok = sum(1 for cn in pairs if cn.pass_count >= _COMPONENT_MIN_PASSES)
            if passes_ok < _COMPONENT_MIN_PASSES_FRAC * len(pairs):
                continue

            # Build probe_values: one value per member from any constituent pair.
            probe_values: Dict[int, float] = {}
            for cn in pairs:
                a, b = cn.members
                if a not in probe_values:
                    probe_values[a] = cn.probe_val_a
                if b not in probe_values:
                    probe_values[b] = cn.probe_val_b

            # Verify a sentinel_var responds to the full joint probe on all members.
            # Try candidate sentinel_vars in descending frequency order — the most
            # commonly used pairwise sentinel is the most likely to respond.
            sentinel_counts = _Counter(cn.sentinel_var for cn in pairs)
            saved = self.world.state
            R0 = list(self.world.predict_under_joint_intervention({}))
            self.world.state = saved
            RAB = list(self.world.predict_under_joint_intervention(probe_values))
            self.world.state = saved
            self.total_interventions += 2

            chosen_sv: Optional[int] = None
            chosen_tol: float = 0.0
            for sv, _ in sentinel_counts.most_common():
                sv_tol = self.ledger.vars[sv].current_tolerance
                if abs(RAB[sv] - R0[sv]) > sv_tol:
                    chosen_sv = sv
                    chosen_tol = sv_tol
                    break
            if chosen_sv is None:
                continue  # no downstream var responds to full joint probe; skip

            members_tuple = tuple(sorted(members_set))
            self.ledger.install_hyper_composite(
                members=members_tuple,
                sentinel_var=chosen_sv,
                probe_values=probe_values,
                tol=chosen_tol,
                certified_at_cycle=cycle,
                context_visible=self.world.visible_count,
                absorbed_pairs=len(pairs),
            )
            # Move constituent pairwise composites to absorbed list.
            for cn in pairs:
                self.ledger.composites.remove(cn)
                self.ledger.absorbed_composites.append(cn)
            self.ledger.event_log.append(
                f"c{cycle}: promoted C{self.ledger._next_component_id - 1} "
                f"({size} members, {len(pairs)} pairs, density={density:.2f}, sv=x{chosen_sv})"
            )

    def _commission_regime_sentinel(self, regime_id: int) -> bool:
        """Try to build a cluster-level witness probe for a newly confirmed regime.

        Phase 1: searches the union of member vars' sentinel pools for a probe
        that elicits delta >= tol from >= 2 regime members. These probes are
        free (already selected); covers co-occurring vars via shared ancestors.

        Phase 2: if Phase 1 finds nothing, scans all visible vars with fixed
        probe values [0.1, 0.9] — a broader but bounded search (2 * n_vis world
        calls per value). Stops as soon as a qualifying probe is found.

        If neither phase finds a cluster-level witness, the regime annotates only
        and may not authorize rsk until a future commissioning succeeds (e.g. the
        world settles into a state where a cluster probe becomes available).

        Cost: up to 2 * (sentinel_pool + 2 * n_vis) world calls, amortized over
        all subsequent cycles where the sentinel replaces N leaf checks.
        """
        sig = next((s for s in self.regime_register._confirmed if s.regime_id == regime_id), None)
        if sig is None:
            return False

        member_set = {e.var for e in sig.events if e.var < self.world.visible_count}
        if len(member_set) < 2:
            return False

        tol = max(
            (self.ledger.vars[v].current_tolerance for v in member_set),
            default=0.05,
        )

        best_slot: Optional[int] = None
        best_val: float = 0.0
        best_target_vars: frozenset = frozenset()
        best_coverage = 0

        def _evaluate(iv_slot: int, iv_val: float) -> None:
            nonlocal best_slot, best_val, best_target_vars, best_coverage
            if iv_slot >= self.world.visible_count:
                return
            baseline_val = self.world.state[iv_slot]
            if abs(iv_val - baseline_val) < 1e-6:
                return  # probe collapses to identity — would not discriminate
            baseline = self.world.predict_under_intervention(iv_slot, baseline_val)
            intervened = self.world.predict_under_intervention(iv_slot, iv_val)
            self.total_interventions += 2
            responsive = frozenset(
                v for v in member_set
                if abs(intervened[v] - baseline[v]) >= tol
            )
            if len(responsive) >= 2 and len(responsive) > best_coverage:
                best_coverage = len(responsive)
                best_slot, best_val = iv_slot, iv_val
                best_target_vars = responsive

        # Phase 1: union of member vars' existing sentinel pools.
        seen_keys: set = set()
        for v in member_set:
            for iv_slot, iv_val in self.ledger.vars[v].sentinels:
                key = (iv_slot, round(iv_val, 3))
                if key not in seen_keys:
                    seen_keys.add(key)
                    _evaluate(iv_slot, iv_val)

        # Phase 2: scan visible vars with fixed probe values if phase 1 failed.
        if best_slot is None:
            for iv_slot in range(self.world.visible_count):
                for iv_val in (0.1, 0.9):
                    key = (iv_slot, iv_val)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        _evaluate(iv_slot, iv_val)
                    if best_slot is not None:
                        break  # take first qualifying probe — stop scanning
                if best_slot is not None:
                    break

        if best_slot is None:
            self.ledger.event_log.append(
                f"regime R{regime_id}: sentinel commissioning FAILED "
                f"(no probe covers >=2 of {len(member_set)} members with delta>=tol)"
            )
            return False

        self.regime_register.install_sentinel(regime_id, best_slot, best_val, best_target_vars, tol)
        self.ledger.event_log.append(
            f"regime R{regime_id}: sentinel commissioned "
            f"iv=x{best_slot}→{best_val:.3f} covers {len(best_target_vars)} of {len(member_set)} members"
        )
        return True

    def _find_joint_trass_candidates(self, cycle: int) -> None:
        """Called once per cycle after all per-var processing. Checks vars that
        transitioned to n.status=="uncertain" due to sentinel failure this cycle
        (tracked in self._uncertain_this_cycle, populated by the two sentinel
        failure sites in run_cycle). For each uncertain var, finds trass ancestors
        and runs joint false-trass test on pairs.

        This method uses the agent's believed parent structure, not the true
        causal structure. It will miss joint effects when the agent's hypothesis
        about ancestry is wrong.
        """
        for uvar in list(self._uncertain_this_cycle):
            trass_ancestors = [
                v for v in range(self.world.visible_count)
                if self.ledger.vars[v].role_for("skip") == "trass"
                and self._is_ancestor(v, uvar)
            ]
            for i in range(len(trass_ancestors)):
                for j in range(i + 1, len(trass_ancestors)):
                    result = self._test_joint_false_trass(
                        trass_ancestors[i], trass_ancestors[j], cycle
                    )
                    if result == "tareth":
                        na = self.ledger.vars[trass_ancestors[i]]
                        nb = self.ledger.vars[trass_ancestors[j]]
                        if na.status == "trass":
                            na.status = "proposed"
                        if nb.status == "trass":
                            nb.status = "proposed"

    def _proactive_joint_scan(self, cycle: int) -> None:
        """Scan all trass-var pairs for joint false-trass without waiting for a
        downstream sentinel to fail. Catches PROD(TINY, TINY)-style patterns where
        neither var is individually salient enough to trigger _find_joint_trass_candidates.

        Called every _JOINT_SCAN_INTERVAL cycles. Only considers pairs where neither
        var already has a CompositeNethra together (no re-testing known composites).
        """
        n_vis = self.world.visible_count
        trass_vars = [
            v for v in range(n_vis)
            if self.ledger.vars[v].role_for("skip") == "trass"
            or v in self._inert_vars
        ]
        if len(trass_vars) < 2:
            return
        # Build set of already-composite pairs to skip re-testing.
        existing_pairs: Set[FrozenSet[int]] = {
            frozenset(cn.members) for cn in self.ledger.composites
        }
        for i in range(len(trass_vars)):
            for j in range(i + 1, len(trass_vars)):
                va, vb = trass_vars[i], trass_vars[j]
                if frozenset((va, vb)) in existing_pairs:
                    continue
                result = self._test_joint_false_trass(va, vb, cycle)
                if result == "tareth":
                    na = self.ledger.vars[va]
                    nb = self.ledger.vars[vb]
                    if na.status == "trass":
                        na.status = "proposed"
                    if nb.status == "trass":
                        nb.status = "proposed"
                    existing_pairs.add(frozenset((va, vb)))

    def _retest_trass_vars(self, cycle: int) -> List[int]:
        """Re-run the operation_role test on all currently-trass variables.
        Trass classification is provisional: it depended on which vars were
        visible at classification time. When new vars are revealed, previously
        trass vars may now have visible dependents and become tareth.

        For each trass var:
          - reset operation_role to "untested" and re-run _certify_operation_role
          - if "tareth": flip status from trass→proposed, add to flipped list
          - if "untested" (still deferred): also flip status to proposed so
            the var gets normal audits next cycle (otherwise trass-skip would
            block any audit and the role stays untested permanently)
          - if "trass": leave classification as-is

        Returns the list of vars whose role/status was changed (flipped).
        Called from on_variable_revealed each time a new var is added.
        """
        flipped: List[int] = []
        for v in range(self.world.visible_count):
            n = self.ledger.vars[v]
            # FILTER LEDGER: trass-cert vars are the normal case — shortcut earned,
            # now re-testing whether the expanded scope invalidates it. Status-only-trass
            # vars (no cert, status=="trass") are a bypass of the filter ledger: the var
            # collapsed without earning a shortcut. Including them here is a recovery
            # path, not normal operation — it pulls vars back into the cert model so they
            # can re-earn their classification. Without this, status-only-trass vars are
            # permanently invisible to scope-expansion revalidation.
            if n.role_for("skip") != "trass" and n.status != "trass":
                continue
            n.certificates.pop("skip", None)
            new_role = self._certify_operation_role(v, cycle)
            if new_role == "tareth":
                if n.status == "trass":
                    n.status = "proposed"
                flipped.append(v)
                self.ledger.event_log.append(
                    f"c{cycle}: x{v} role REVISED trass→tareth (new visible vars exposed dependence)"
                )
            elif new_role == "untested":
                if n.status == "trass":
                    n.status = "proposed"
                flipped.append(v)
                self.ledger.event_log.append(
                    f"c{cycle}: x{v} role test deferred (still untested), "
                    f"status reverted trass→proposed for re-audit"
                )
            else:
                pass

        # Q6b: FILTER LEDGER — proactive cert invalidation on trass→tareth flip is wrong.
        # When a var flips trass→tareth, other certs that excluded it from their tested
        # scope should NOT be proactively flagged "untested." Their shortcuts keep firing
        # by default. If the scope expansion causes an actual failure, the sentinel catches
        # it at failure time. Pre-emptive cert scanning on every scope transition is the
        # positive-ledger pattern. The cert.targets accurately records what was tested
        # (Q6a); if that scope later proves insufficient, failure earns the recertification,
        # not a structural prediction of what might fail.

        return flipped

    def _discover_compressions(self, var: int, cycle: int) -> int:
        """Search for compressions: gate conditions under which the variable's
        prediction simplifies to a near-constant. Returns count added.

        Method:
          1. For each parent of var, treat it as a candidate gate variable.
          2. Try 3 anchor target values (low/mid/high: 0.15, 0.5, 0.85).
          3. Sample budget×{1..4} parent-value tuples where gate-var is near
             target ±tol and other parents are random over [0,1].
          4. Compute predictions; if all are within tolerance of the mean,
             this gate-condition produces a stable simplified value.
          5. Store as a Compression if not already present for this gate.

        Varying non-gate parents over the full range is essential — a
        compression must hold across all values of other parents, not just
        their current world-state values.
        """
        n = self.ledger.vars[var]
        if not n.parents:
            return 0
        if not self._authority_strength_derivation_allowed(
            var,
            cycle=cycle,
            blocked_handle_kind="compression",
            blocked_target=f"x{var}",
        ):
            return 0
        if any(
            not self._authority_strength_derivation_allowed(
                parent,
                cycle=cycle,
                blocked_handle_kind="compression",
                blocked_target=f"x{var}",
            )
            for parent in n.parents
        ):
            return 0

        budget = self.compression_discovery_budget
        candidate_gates = list(n.parents)
        added = 0

        # Sample candidate target values for each gating parent (3 anchors per parent)
        for gate_var in candidate_gates:
            gate_n = self.ledger.vars[gate_var]
            gate_tol = gate_n.current_tolerance
            # Anchor target values: low, mid, high range
            for target in (0.15, 0.5, 0.85):
                gate = ((gate_var, target, gate_tol),)
                samples = []
                attempts = 0
                while len(samples) < budget and attempts < budget * 4:
                    attempts += 1
                    # v28+: vary ALL parents (including gate_var near target).
                    # Previously held non-gate parents at current world.state
                    # which falsely declared compressions that only held at
                    # current values of other parents. A real compression
                    # must hold across the full range of other parent values
                    # when gate_var is near target.
                    par_vals_list = []
                    for p in n.parents:
                        if p == gate_var:
                            v = max(0.0, min(1.0, target + self.rng.uniform(-gate_tol, gate_tol)))
                        else:
                            v = self.rng.random()
                        par_vals_list.append(v)
                    pred = FUNC_LIBRARY[n.func](par_vals_list)
                    samples.append(pred)
                if len(samples) < 4:
                    continue
                # Are predictions tightly clustered (within this variable's tolerance)?
                tol = n.current_tolerance
                ref = samples[0]
                if all(abs(s - ref) <= tol for s in samples):
                    avg = sum(samples) / len(samples)
                    comp = Compression(
                        gate=gate,
                        simplified_value=avg,
                        certified_equivalence=len(samples),
                        discovery_cycle=cycle,
                    )
                    existing_gates = {c.gate for c in n.compressions}
                    if comp.gate not in existing_gates:
                        n.compressions.append(comp)
                        added += 1
                        self.ledger.event_log.append(
                            f"c{cycle}: x{var} compression discovered: {comp.display()}"
                        )
        return added

    def _try_compression(self, var: int) -> Optional[float]:
        """Try to use a matching compression for cheap prediction. Walks the
        var's compression list; returns the first matching compression's
        simplified_value, or None if no compression matches the current
        world state. Updates per-var hit/miss counters."""
        n = self.ledger.vars[var]
        for comp in n.compressions:
            if comp.pred_passes < self.compression_promote_after:
                continue
            if comp.gate_matches(self.world.state):
                n.compression_hits += 1
                n.compression_hits_lifetime += 1
                return comp.simplified_value
        n.compression_misses += 1
        n.compression_misses_lifetime += 1
        return None

    def _install_var(self, var: int, parents: Tuple[int, ...], func: str,
                     score: int, second: int, cycle: int,
                     fit_diag: Optional[FitDiagnostic] = None) -> bool:
        """Apply the result of a full audit. Returns semantic_changed (True if
        the new fit is not same-parent tied churn).

        Pipeline:
          1. update_var: archive old fit if signature changed; reset state
             only when the transition invalidates ledger state.
          2. If operation_role is "untested", run _certify_operation_role.
          3. If role becomes "trass": collapse, return early.
          4. Otherwise (role tareth or still-untested):
             a. Increment strong_observations (or reset to 1 if semantic_changed).
             b. If any current parent is trass-classified, force re-test of
                that parent's role (contradiction — fit depends on supposedly-
                irrelevant var).
             c. If no sentinels yet and strong_obs ≥ 1: select sentinels using
                current available_parents. They go live next cycle.
             d. If strong_obs ≥ promote_after AND sentinels exist: promote
                status to "certified" (informational confidence label).
                Else if status was "uncertain"/"quarantined": back to "proposed".
             e. If sentinels exist AND strong_obs ≥ compression_discover_after
                AND no compressions yet: discover compressions.
        """
        margin = score - second
        old_n = self.ledger.vars[var]
        old_parents = tuple(old_n.parents)
        old_func = old_n.func
        new_parents = tuple(parents)
        new_func = func
        syntactic_changed = (old_parents, old_func) != (new_parents, new_func)
        parents_changed = old_parents != new_parents
        old_hyp = (old_parents, old_func)
        new_hyp = (new_parents, new_func)
        tie_set = frozenset()
        near_tie_candidates: Tuple = ()
        near_tie_context_key: int = 0
        if fit_diag is None:
            fit_diag = getattr(self, "_last_fit_diag", None)
        if fit_diag is not None:
            tie_set = fit_diag.tie_set
            near_tie_candidates = fit_diag.near_tie_candidates
            near_tie_context_key = fit_diag.near_tie_context_key
        near_tie_set = frozenset((p, f) for p, f, _ in near_tie_candidates)
        same_parent_tied_churn = (
            syntactic_changed
            and not parents_changed
            and old_hyp in tie_set
            and new_hyp in tie_set
        )
        semantic_changed = syntactic_changed and not same_parent_tied_churn
        ledger_reset_needed = semantic_changed or parents_changed

        self.ledger.update_var(
            var, new_parents, new_func, cycle,
            reset_state=ledger_reset_needed,
        )

        # Parent structure, not operator churn, determines DAG topo invalidation.
        if parents_changed:
            self._invalidate_topo_cache()
            # Wake previously-inert parents now known to causally affect `var`.
            # Inert vars are skipped by the cheap salience screen at initialize time
            # because their individual effect is below DEFAULT_TOLERANCE. When a
            # structural event (e.g., JOINT_SHOCK) causes the parent screen to
            # discover them as real causes, they must be audited and certified.
            for _p in new_parents:
                if _p < self.world.visible_count and _p in self._inert_vars:
                    self._inert_vars.discard(_p)
                    if self._live_set is not None:
                        self._live_set.add(_p)
                    self.ledger.event_log.append(
                        f"c{cycle}: x{_p} woken from inert (new parent of x{var})"
                    )

        n = self.ledger.vars[var]
        if n.first_audited_cycle == 0:
            n.first_audited_cycle = cycle
        n.full_audits += 1
        n.margins.append(margin)
        audit_nethra_id = self._context_role_index_record_var_fit(
            var,
            cycle,
            source="audit",
            fit_diag=fit_diag,
        )
        self._context_role_index_assign_role(
            audit_nethra_id,
            var=var,
            cycle=cycle,
            operation="audit",
            role="best_available",
            evidence_summary="selected by fit_var",
            fit_diag=fit_diag,
            validity_scope=tuple(sorted((var, *new_parents))),
        )

        if n.role_for("skip") == "untested":
            self._certify_operation_role(var, cycle)

        if n.role_for("skip") == "trass":
            # Trass vars: no sentinel monitoring. Clear any sentinels that were
            # installed before this audit (e.g., from an earlier tareth period).
            # The hot-pass accumulates stable-cycle evidence in cert.sentinel_passes
            # and sets status="trass" when threshold is reached. Trass vars never
            # reach n.authoritative (which requires tareth/noise_floor role), so
            # sentinel_passes is cycle-counted in the hot-pass trass block, not here.
            n.sentinels = []
            n.expected_outcomes = []
            if fit_diag is not None:
                fit_diag.status_after = n.status
                fit_diag.role_after = n.role_for("skip")
            return semantic_changed

        # Provisional commitment: trust the best fit, let sentinels validate.
        if not semantic_changed:
            n.strong_observations += 1
        else:
            n.strong_observations = 1
            n.consecutive_sentinel_failures = 0  # world genuinely changed

        # v28+: if the current fit lists a parent that's currently trass, that's
        # a contradiction — we declared the parent "doesn't matter operationally"
        # but our fit for `var` depends on it. Force re-test of those trass vars.
        # Run on EVERY audit (not just sig_changed): the trass classification
        # could have happened after this var was fit, making it newly contradictory.
        for p in parents:
            if p < self.world.visible_count:
                pn = self.ledger.vars[p]
                if pn.role_for("skip") == "trass":
                    pn.invalidate_certs("false_trass_contradiction")
                    if pn.status == "trass":
                        pn.status = "proposed"
                    self.ledger.event_log.append(
                        f"c{cycle}: x{p} role re-test triggered "
                        f"(picked as parent by x{var} despite trass status)"
                    )

        # Sentinel attachment: as soon as the fit is stable for ONE cycle,
        # attach sentinels so the cheap path can fire next cycle. We don't wait
        # for promote_after — sentinels themselves validate provisionally;
        # let them work. Promotion to "certified" status happens later (just a
        # confidence label) and means the same fit has been stable for
        # promote_after consecutive cycles.
        if not n.sentinels and n.strong_observations >= 1:
            # Sentinel parent pool: include all vars not cert-excluded for route.
            # Per-target route certs gate exclusion (invariant 50 — route/include by default).
            available = {
                other_var for other_var, other_n in self.ledger.vars.items()
                if other_var != var
                and (n.route_certs.get(other_var) is None or n.route_certs[other_var].role != "trass")
                and (
                    other_n.status == "certified"
                    or other_n.status == "trass"
                    or other_n.role_for("skip") == "trass"
                    or (other_n.status == "proposed" and bool(other_n.sentinels))
                )
            }
            # [CONSEQUENCE-WEIGHT P1] sentinel count scaled by downstream consequence.
            # Original: self.sentinel_count (uniform).
            # Revert: replace _eff_sentinel_count arg with self.sentinel_count and delete this line.
            _eff_sentinel_count = self.sentinel_count + self._consequence_tier(var) * 2
            sentinels, expected = select_var_sentinels(
                var, parents, func, self.world, self.rng,
                _eff_sentinel_count, self.sentinel_pool, n.current_tolerance,
                available_parents=available,
            )
            if sentinels:
                n.sentinels = sentinels
                n.expected_outcomes = expected

        # Promotion to "certified" status (informational confidence label)
        just_promoted = False
        # [CONSEQUENCE-WEIGHT P2] promotion threshold scaled by downstream consequence.
        # Original: n.strong_observations >= self.promote_after
        # Revert: delete _eff_promote_after line, replace _eff_promote_after with self.promote_after.
        _eff_promote_after = self.promote_after + self._consequence_tier(var) * 2
        if n.status != "certified" and n.strong_observations >= _eff_promote_after and n.sentinels:
            n.status = "certified"
            just_promoted = True
            if n.first_certified_cycle == 0:
                n.first_certified_cycle = cycle
        elif n.status in ("quarantined", "uncertain"):
            # Audit produced a fit; revert to proposed so it can re-accumulate
            # observations toward promotion.
            n.status = "proposed"

        if just_promoted:
            # Route certs: counterfactual fit per non-parent candidate.
            # Earned at promotion — the fit is stable enough to trust the comparison.
            if self._authority_strength_derivation_allowed(
                var,
                cycle=cycle,
                blocked_handle_kind="route",
                blocked_target=f"x{var}",
            ):
                avail_for_route = {
                    other_var for other_var, other_n in self.ledger.vars.items()
                    if other_var != var
                    and self._authority_strength_derivation_allowed(
                        other_var,
                        cycle=cycle,
                        blocked_handle_kind="route",
                        blocked_target=f"x{var}",
                    )
                    and (other_n.status == "certified" or other_n.status == "trass"
                         or other_n.role_for("skip") == "trass"
                         or (other_n.status == "proposed" and bool(other_n.sentinels)))
                }
                self._certify_route_certs(var, new_parents, avail_for_route, cycle, fit_diag)
                # Audit cert: stable fit earned enough observations; mark as reusable.
                self.ledger.issue_cert(
                    var, "audit", "reusable", "guarded_reuse",
                    context_parents=new_parents,
                    context_visible=self.world.visible_count,
                    context_cycle=cycle,
                    targets=(),
                    substitutions_tested=("stable_audit",),
                    changes=0,
                    trials=self.promote_after,
                    earned_by="stable_audit",
                )
            # Dormant revival check: if any archived alternative now wins,
            # increment its revival_count and track the context.
            context_key = near_tie_context_key
            for alt in n.dormant_alternatives:
                if alt.parents == new_parents and alt.func == new_func:
                    alt.revival_count += 1
                    alt.context_keys_seen.add(context_key)
                    alt.last_seen_cycle = cycle
                    if alt.revival_count >= 2 and len(alt.context_keys_seen) >= 2:
                        self.ledger.event_log.append(
                            f"c{cycle}: x{var} dormant alternative "
                            f"{alt.func}({list(alt.parents)}) achieved frontier_survival "
                            f"(revivals={alt.revival_count} contexts={len(alt.context_keys_seen)})"
                        )

        # Compression discovery: triggered when the variable has stable sentinels
        # AND enough strong observations. Status label not the gate.
        if n.sentinels and n.strong_observations >= self.compression_discover_after \
           and not n.compressions:
            self._discover_compressions(var, cycle)

        # v29 extension hook: if an extension module is loaded, ask it to
        # derive additional compressions. The extension reads agent state
        # (read-only) and returns Compression objects to append. Existing
        # gates are deduplicated. Failure mode is bounded: a wrong derived
        # compression mismatches world output → sentinel fail → invalidation
        # cascade. The collapse mechanism handles bad derivations.
        if self.extension is not None and n.sentinels and n.role_for("skip") == "tareth":
            try:
                derived = []
                if "algebraic" in self.extension_modes:
                    derived.extend(self.extension.derive_compressions(self, var, cycle))
                if "equiv" in self.extension_modes:
                    derived.extend(self.extension.derive_equivalence_compressions(self, var, cycle))
                existing_gates = {c.gate for c in n.compressions}
                for d in derived:
                    if any(
                        not self._authority_strength_derivation_allowed(
                            int(gate_var),
                            cycle=cycle,
                            blocked_handle_kind="compression",
                            blocked_target=f"x{var}:extension",
                        )
                        for gate_var, _, _ in d.gate
                    ):
                        continue
                    if d.gate not in existing_gates:
                        n.compressions.append(d)
                        existing_gates.add(d.gate)
                        self.ledger.event_log.append(
                            f"c{cycle}: x{var} extension-derived compression added: {d.display()}"
                        )
            except Exception as e:
                # Extension failures must not break the agent. Log and continue.
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} extension error: {type(e).__name__}: {e}"
                )

        if fit_diag is not None:
            fit_diag.status_after = n.status
            fit_diag.role_after = n.role_for("skip")

        # Frontier management: maintain TiedFrontier on the VarNethra.
        # Skipped for trass (already returned early above).
        if len(near_tie_set) >= 2:
            scores_dict = {(p, f): s for p, f, s in near_tie_candidates}
            self._update_tied_frontier(var, cycle, near_tie_set,
                                       scores_dict, near_tie_context_key, fit_diag)
        elif n.tied_frontier is not None:
            winning = next(iter(near_tie_set)) if near_tie_set else None
            self._collapse_tied_frontier(var, winning, cycle)

        return semantic_changed

    def _certify_route_certs(
        self, var: int, parents: Tuple[int, ...], available: Set[int], cycle: int,
        fit_diag: Optional[FitDiagnostic] = None,
    ) -> None:
        """Issue per-candidate route certs for target `var` at promotion time.

        Only certifies candidates that were ACTIVELY COMPETING in the last audit
        (appeared in near_tie_candidates but not in the winner's parents). A clean
        fit with no near-ties earns no route certs — invariant 2: use succeeds → do
        nothing. Proactively scanning all available vars violates invariant 17.

        For each competing non-parent candidate P:
          - Fit `var` with P excluded from available.
          - Same winner as the baseline → P is route-trass (safe to exclude).
          - Different winner → P is route-tareth (P influences the ranking).

        Uses a reduced budget (intervention_budget // 3, min 6) because route cert
        fits are secondary evidence: the main fit already ran at full budget.

        Target-owned: cert is stored in n.route_certs[P], not on P.
        """
        if fit_diag is None or not fit_diag.near_tie_candidates:
            return  # clean fit, no competition → invariant 2, nothing earned

        # Build candidate pool: vars in near-tie parents that aren't in winner.
        parents_set = set(parents)
        competing = {
            p
            for cand_parents, _, _ in fit_diag.near_tie_candidates
            for p in cand_parents
            if p not in parents_set and p in available
            and self._authority_strength_derivation_allowed(
                p,
                cycle=cycle,
                blocked_handle_kind="route",
                blocked_target=f"x{var}",
            )
        }
        if not competing:
            return

        n = self.ledger.vars[var]
        # Skip candidates already certified in this parent context — re-promotion
        # does not earn a re-test if the evidence context hasn't changed.
        competing = {
            p for p in competing
            if p not in n.route_certs
            or n.route_certs[p].context_parents != tuple(parents)
        }
        if not competing:
            return

        rc_budget = max(6, self.intervention_budget // 3)
        base_parents = tuple(parents)
        base_func = n.func

        for p in competing:
            avail_excl = available - {p}
            if len(avail_excl) < len(parents_set):
                continue
            excl_parents, excl_func, _, _ = fit_var(
                var, self.world, self.rng, rc_budget,
                n.current_tolerance, available_parents=avail_excl,
            )
            same_winner = (base_parents == excl_parents and base_func == excl_func)
            role: Role = "trass" if same_winner else "tareth"
            self.ledger.issue_route_cert(
                var, p, role,
                context_parents=tuple(parents),
                context_visible=self.world.visible_count,
                context_cycle=cycle,
                targets=(var,),
                substitutions_tested=("counterfactual_fit",),
                changes=0 if same_winner else 1,
                trials=1,
                earned_by="counterfactual_fit",
            )
            if self._context_role_index is not None:
                nid = f"route:x{p}->x{var}:{base_func}({','.join(map(str, base_parents))})"
                self._context_role_index.add_or_update_node(NethraNode(
                    nethra_id=nid,
                    kind="route_handle",
                    target_var=var,
                    components=tuple(sorted((var, p, *base_parents))),
                    learned_parents=tuple(base_parents),
                    learned_func=base_func,
                    signature=f"x{p}->x{var}:{role}",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                    observations=1,
                    active_probe_count=rc_budget,
                    source="route_cert",
                ))
                self._context_role_index_assign_role(
                    nid,
                    var=var,
                    cycle=cycle,
                    operation="route",
                    role=role,
                    evidence_summary="counterfactual_fit same_winner=" + str(same_winner),
                    fit_diag=fit_diag,
                    route_role=role,
                    validity_scope=tuple(sorted((var, p, *base_parents))),
                )

    def _derive_separating_probes(
        self, frontier: "TiedFrontier",
        fit_diag: Optional[FitDiagnostic] = None,
    ) -> Tuple[Tuple[int, float], ...]:
        """Derive separating probes from the last FitDiagnostic for `var`.

        Phase 1: use existing audit probes (no new world calls). For each probe
        (iv_var, iv_val), compute the pairwise prediction disagreement across all
        frontier candidates using FUNC_LIBRARY. Retain the top 3 probes by max
        pairwise disagreement.

        Returns a tuple of (iv_var, iv_val) pairs (at most 3).
        """
        if fit_diag is None or not fit_diag.probes:
            return ()
        probes = fit_diag.probes  # Tuple[Tuple[int, float], ...]
        candidates = list(frontier.candidates)  # List[(parents, func)]
        if len(candidates) < 2:
            return ()
        from .functions import FUNC_LIBRARY
        state = self.world.state
        scored: List[Tuple[float, Tuple[int, float]]] = []
        for iv_var, iv_val in probes:
            # Build an intervened state snapshot
            intervened = list(state)
            intervened[iv_var] = iv_val
            preds = []
            for cand_parents, cand_func in candidates:
                fn = FUNC_LIBRARY.get(cand_func)
                if fn is None:
                    continue
                args = [intervened[p] for p in cand_parents]
                preds.append(fn(args) if args else 0.0)
            if len(preds) < 2:
                continue
            max_disagree = max(
                abs(preds[i] - preds[j])
                for i in range(len(preds))
                for j in range(i + 1, len(preds))
            )
            scored.append((max_disagree, (iv_var, iv_val)))
        scored.sort(key=lambda x: -x[0])
        return tuple(p for _, p in scored[:3])

    def _update_tied_frontier(
        self, var: int, cycle: int,
        near_tie_set: FrozenSet[Tuple[Tuple[int, ...], str]],
        scores_dict: Dict[Tuple[Tuple[int, ...], str], int],
        context_key: int,
        fit_diag: Optional[FitDiagnostic] = None,
    ) -> None:
        """Maintain the TiedFrontier on VarNethra `var`.

        If the context_key matches and the new candidate set is the same as
        the existing frontier, increment stable_count. If the set narrowed,
        archive the dropped candidates to dormant_alternatives and replace
        the frontier. If context changed or no frontier exists, start fresh.
        """
        n = self.ledger.vars[var]
        existing = n.tied_frontier
        frontier_candidates = tuple(near_tie_set)
        if (
            self._scaffold_memory_mode == "assist_feature"
            and self._scaffold_memory_index is not None
            and len(frontier_candidates) >= 2
        ):
            frontier_candidates = tuple(
                self._scaffold_memory_index.rank_frontier_candidates(
                    var,
                    nethra_context_key(
                        operation="tied_frontier",
                        var=var,
                        visible=self.world.visible_count,
                    ),
                    frontier_candidates,
                )
            )
        for cand_parents, cand_func in frontier_candidates:
            self._context_role_index_record_candidate(
                prefix="frontier",
                kind="tied_frontier_candidate",
                var=var,
                parents=tuple(cand_parents),
                func=cand_func,
                cycle=cycle,
                source="tied_frontier",
                role="unresolved",
                score=scores_dict.get((cand_parents, cand_func), 0),
                context_operation="tied_frontier",
                fit_diag=fit_diag,
            )
        if existing is None:
            new_frontier = TiedFrontier(
                candidates=near_tie_set,
                scores={h: scores_dict.get(h, 0) for h in near_tie_set},
                margin=self.near_tie_margin,
                context_key=context_key,
                collapse_sig=None,
                separating_probes=(),
                first_seen_cycle=cycle,
                last_seen_cycle=cycle,
                stable_count=1,
                distinct_contexts_seen=1,
            )
            # P1-B: derive separating probes from the just-completed audit so the
            # next audit can use them as forced inclusions.
            new_frontier.separating_probes = self._derive_separating_probes(new_frontier, fit_diag)
            n.tied_frontier = new_frontier
        elif near_tie_set == existing.candidates:
            existing.scores = {h: scores_dict.get(h, 0) for h in near_tie_set}
            existing.last_seen_cycle = cycle
            existing.stable_count += 1
            if context_key != existing.context_key:
                # Same candidates survived a context change — that is cross-context
                # evidence (invariant distinct_contexts_seen rule). Update key and count.
                existing.context_key = context_key
                existing.distinct_contexts_seen += 1
            # Refresh separating probes from latest audit — the last-used probe
            # set may discriminate better than the one derived at frontier creation.
            existing.separating_probes = self._derive_separating_probes(existing, fit_diag)
        elif existing.context_key != context_key:
            # Context changed AND candidate set differs — fresh frontier.
            # Archive dropped candidates from old frontier.
            for h in existing.candidates - near_tie_set:
                n.dormant_alternatives.append(
                    DormantAlternative(
                        parents=h[0], func=h[1],
                        last_score=existing.scores.get(h, 0),
                        last_seen_cycle=cycle,
                    )
                )
                self._context_role_index_record_candidate(
                    prefix="dormant",
                    kind="dormant_alternative",
                    var=var,
                    parents=h[0],
                    func=h[1],
                    cycle=cycle,
                    source="dormant_alternative",
                    role="unresolved",
                    score=existing.scores.get(h, 0),
                    context_operation="frontier_context_change",
                    fit_diag=fit_diag,
                )
            new_frontier = TiedFrontier(
                candidates=near_tie_set,
                scores={h: scores_dict.get(h, 0) for h in near_tie_set},
                margin=self.near_tie_margin,
                context_key=context_key,
                collapse_sig=None,
                separating_probes=(),
                first_seen_cycle=existing.first_seen_cycle,
                last_seen_cycle=cycle,
                stable_count=1,
                distinct_contexts_seen=1,
            )
            new_frontier.separating_probes = self._derive_separating_probes(new_frontier, fit_diag)
            n.tied_frontier = new_frontier
        else:
            # Same context, different candidate set — archive dropped candidates.
            for h in existing.candidates - near_tie_set:
                n.dormant_alternatives.append(
                    DormantAlternative(
                        parents=h[0], func=h[1],
                        last_score=existing.scores.get(h, 0),
                        last_seen_cycle=cycle,
                    )
                )
                self._context_role_index_record_candidate(
                    prefix="dormant",
                    kind="dormant_alternative",
                    var=var,
                    parents=h[0],
                    func=h[1],
                    cycle=cycle,
                    source="dormant_alternative",
                    role="unresolved",
                    score=existing.scores.get(h, 0),
                    context_operation="frontier_narrowed",
                    fit_diag=fit_diag,
                )
            new_frontier = TiedFrontier(
                candidates=near_tie_set,
                scores={h: scores_dict.get(h, 0) for h in near_tie_set},
                margin=self.near_tie_margin,
                context_key=context_key,
                collapse_sig=None,
                separating_probes=(),
                first_seen_cycle=existing.first_seen_cycle,
                last_seen_cycle=cycle,
                stable_count=1,
                distinct_contexts_seen=existing.distinct_contexts_seen,
            )
            new_frontier.separating_probes = self._derive_separating_probes(new_frontier, fit_diag)
            n.tied_frontier = new_frontier

    def _collapse_tied_frontier(
        self, var: int,
        winning_hyp: Optional[Tuple[Tuple[int, ...], str]],
        cycle: int,
    ) -> None:
        """Collapse the TiedFrontier for `var`.

        Guard: collapse only if stable_count >= 3 AND distinct_contexts_seen >= 2.
        If threshold not met, the tie is ambiguity, not resolved — clear without
        archiving (the candidates have not proven themselves across regimes).
        If threshold met, archive losing candidates as DormantAlternatives.
        """
        n = self.ledger.vars[var]
        if n.tied_frontier is None:
            return
        f = n.tied_frontier
        threshold_met = f.stable_count >= 3 and f.distinct_contexts_seen >= 2
        if threshold_met:
            for h in f.candidates:
                if h != winning_hyp:
                    n.dormant_alternatives.append(
                        DormantAlternative(
                            parents=h[0], func=h[1],
                            last_score=f.scores.get(h, 0),
                            last_seen_cycle=cycle,
                        )
                    )
                    self._context_role_index_record_candidate(
                        prefix="dormant",
                        kind="dormant_alternative",
                        var=var,
                        parents=h[0],
                        func=h[1],
                        cycle=cycle,
                        source="dormant_alternative",
                        role="unresolved",
                        score=f.scores.get(h, 0),
                        context_operation="frontier_collapse",
                    )
            if winning_hyp is not None:
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} frontier collapsed → "
                    f"{winning_hyp[1]}({list(winning_hyp[0])}); "
                    f"{len(n.dormant_alternatives)} candidates archived"
                )
        else:
            # Threshold not met: ambiguity is unresolved; discard without archiving.
            if (
                self._uncertainty_consolidation_mode == "assist"
                and var in self._uncertainty_preserve_vars
                and self._uncertainty_preserve_remaining > 0
            ):
                archived = 0
                for h in f.candidates:
                    if h == winning_hyp or self._uncertainty_preserve_remaining <= 0:
                        continue
                    n.dormant_alternatives.append(
                        DormantAlternative(
                            parents=h[0], func=h[1],
                            last_score=f.scores.get(h, 0),
                            last_seen_cycle=cycle,
                        )
                    )
                    self._context_role_index_record_candidate(
                        prefix="dormant",
                        kind="dormant_alternative",
                        var=var,
                        parents=h[0],
                        func=h[1],
                        cycle=cycle,
                        source="dormant_alternative",
                        role="unresolved",
                        score=f.scores.get(h, 0),
                        context_operation="frontier_preserve",
                    )
                    self._uncertainty_preserve_remaining -= 1
                    self._uncertainty_assist_preserved_alternative_total += 1
                    archived += 1
                if archived:
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} uncertainty consolidation preserved "
                        f"{archived} alternative(s) within cap"
                    )
            if (
                self._authority_strength_mode == "assist"
                and var in self._authority_strength_preserve_vars
                and self._authority_strength_preserve_remaining > 0
            ):
                existing_alt = {
                    (tuple(alt.parents), alt.func)
                    for alt in n.dormant_alternatives
                }
                archived = 0
                for h in f.candidates:
                    if h == winning_hyp or self._authority_strength_preserve_remaining <= 0:
                        continue
                    if (tuple(h[0]), h[1]) in existing_alt:
                        continue
                    n.dormant_alternatives.append(
                        DormantAlternative(
                            parents=h[0], func=h[1],
                            last_score=f.scores.get(h, 0),
                            last_seen_cycle=cycle,
                        )
                    )
                    self._context_role_index_record_candidate(
                        prefix="dormant",
                        kind="dormant_alternative",
                        var=var,
                        parents=h[0],
                        func=h[1],
                        cycle=cycle,
                        source="dormant_alternative",
                        role="unresolved",
                        score=f.scores.get(h, 0),
                        context_operation="authority_strength_preserve",
                    )
                    self._authority_strength_preserve_remaining -= 1
                    self._authority_strength_alternatives_preserved_total += 1
                    archived += 1
                if archived:
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} authority strength preserved "
                        f"{archived} alternative(s) within cap"
                    )
            self.ledger.event_log.append(
                f"c{cycle}: x{var} frontier cleared (threshold not met: "
                f"stable={f.stable_count} contexts={f.distinct_contexts_seen})"
            )
        n.tied_frontier = None

    def _maybe_novelty(self, var: int, score: int, second: int, cycle: int,
                       sig_changed: bool = False) -> bool:
        """Manage vocabulary-novelty firing and resolution for one variable.
        Returns True if a novelty was newly fired this cycle.

        Logic:
          - If var is operation_role=trass or status=trass: reset both
            streaks, return False (irrelevant).
          - If sig_changed (fit swung this audit): increment weak_streak,
            reset stable_streak. If weak_streak ≥ novelty_weak_streak,
            propose vocabulary novelty for this var.
          - If not sig_changed (fit stable this audit): increment
            stable_streak. If stable_streak ≥ novelty_weak_streak AND
            weak_streak > 0 (we previously fired): resolve the open
            novelty and zero weak_streak.

        The instability streak (sig_changed) is the trigger — repeatedly
        picking different best hypotheses across audits IS the signal that
        no library hypothesis stably explains the variable. Stable fits
        with low margin are observational ties, not vocabulary problems.
        """
        n = self.ledger.vars[var]
        if n.role_for("skip") == "trass" or n.status == "trass":
            self.weak_streak[var] = 0
            self.stable_streak[var] = 0
            return False

        if sig_changed:
            self.weak_streak[var] += 1
            self.stable_streak[var] = 0
        else:
            self.stable_streak[var] += 1
            if self.stable_streak[var] >= self.novelty_weak_streak and self.weak_streak[var] > 0:
                self.ledger.resolve_novelty(var, "vocabulary", cycle)
                self.weak_streak[var] = 0
            return False

        if self.weak_streak[var] < self.novelty_weak_streak:
            return False

        margin = score - second
        evidence = [
            f"c{cycle}: x{var} fit unstable; streak={self.weak_streak[var]}",
            f"current fit={n.func}({list(n.parents)}) score={score} second={second} margin={margin}",
            f"hypothesis keeps swinging across audits — library insufficient",
        ]
        self.ledger.propose_novelty(
            cycle, var, "vocabulary",
            f"x{var}: hypothesis library insufficient at certified noise tolerance",
            evidence,
        )
        return True

    def _tractability_score(self, var: int) -> float:
        """Heuristic: how easy is this var to fit decisively right now?
        Higher = audit first.

        Two factors:
          - base_size: rough size of var's hypothesis space (constants,
            1-parent, 2-parent options). Smaller is more tractable.
          - decided_frac: fraction of OTHER visible vars that are
            provisionally committed (certified or proposed-with-sentinels)
            or classified trass. Higher = more reference frame.

        Score = (1 + decided_frac) / log2(base_size + 1).
        Used as tie-breaker within topological-order audit scheduling.
        """
        n_total = self.world.visible_count
        n = self.ledger.vars[var]
        n_par = len(n.parents)
        base_size = 2 if n_par == 0 else (2 * (n_total - 1) if n_par == 1 else 5 * (n_total - 1) * (n_total - 2) // 2 + 4)
        decided = 0
        for other_var, other_n in self.ledger.vars.items():
            if other_var == var or other_var >= n_total: continue
            if (other_n.status == "certified"
                or other_n.status == "trass"
                or other_n.role_for("skip") == "trass"
                or (other_n.status == "proposed" and bool(other_n.sentinels))):
                decided += 1
        decided_frac = decided / max(1, n_total - 1)
        # Higher when base is small AND many others are decided
        return (1.0 + decided_frac) / math.log2(base_size + 1)

    def _audit_priority_order(self, vars_needing_audit: List[int]) -> List[int]:
        """Sort needs_audit list by cost_weight × tractability (high-first).
        Used at initialization where topological order is not yet meaningful.
        High-cost vars are prioritized so high-stakes fits are established
        early and provide reference frame for subsequent cheaper vars."""
        scored = [
            (self.ledger.vars[v].cost_weight * self._tractability_score(v), v)
            for v in vars_needing_audit
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [v for _, v in scored]

    # ── [CONSEQUENCE-WEIGHT] new method ──────────────────────────────────────────
    # To revert: delete this method and revert the three call sites below
    # (search "# [CONSEQUENCE-WEIGHT]" to find them all).
    def _consequence_tier(self, var: int) -> int:
        """Structural consequence tier under current beliefs.

        Counts how many vars directly list `var` as a parent (direct downstream
        dependents). Used to scale sentinel count, promotion threshold, and
        dormancy threshold — importance affects the *action policy*, not scoring.

        Tier 0: no dependents (leaf in current belief graph).
        Tier 1: 1–2 dependents.
        Tier 2: 3+ dependents.

        This is endogenous (uses agent beliefs, not world truth) and dynamic
        (updates automatically as fits change). It is NOT global importance
        (invariant #19); it is the repair-operation consequence of this var's
        fit being wrong right now.

        To revert this feature entirely, delete this method and replace all
        three [CONSEQUENCE-WEIGHT] call sites with their originals:
          P1 sentinel:  self.sentinel_count
          P2 promote:   self.promote_after
          P3 dormancy:  self._min_dormant_cert_age
        """
        # [CONSEQUENCE-WEIGHT ablation] — returning 0 here disables all three
        # policy effects (P1/P2/P3 add 0 to their respective thresholds).
        if not self._consequence_weight_enabled:
            return 0
        deps = len(self.ledger.variable_dependents(var))
        if deps <= 0:
            return 0
        elif deps <= 2:
            return 1
        else:
            return 2
    # ── [/CONSEQUENCE-WEIGHT] ─────────────────────────────────────────────────

    def _cost_biased_topo_audit_order(self, needs_audit_set: Set[int]) -> List[int]:
        """Return needs_audit vars in topological order (parents before children).
        Uses the cached DFS topo order, which groups parent+child adjacently so
        both are likely to land within the same cycle's audit budget.
        cost_weight priority for the audit queue is a future extension; for now
        the DFS order is preserved exactly to avoid budget-cutoff regressions."""
        topo = self._topological_order(self.world.visible_count)
        return [v for v in topo if v in needs_audit_set]

    def _topological_order(self, n_visible: int) -> List[int]:
        """Return visible variables in topological order based on current
        parent maps (parents before children). Used for in-cycle processing
        so sentinel failures can invalidate descendants BEFORE descendants
        take their own cheap-path skips. DFS-based; defensively handles
        cycles in current fits (shouldn't occur but doesn't loop).

        Caches the result. Invalidated by _invalidate_topo_cache
        whenever a variable's parents change (sig_changed in _install_var).
        """
        if (self._topo_cache is not None
            and self._topo_cache_visible_count == n_visible):
            return self._topo_cache
        result: List[int] = []
        visited: Set[int] = set()
        in_progress: Set[int] = set()

        def visit(v: int):
            """DFS helper. Marks v as in-progress, recurses into parents
            (only those <n_visible), then appends v to result on the way out."""
            if v in visited or v in in_progress: return
            in_progress.add(v)
            n = self.ledger.vars[v]
            for p in n.parents:
                if p < n_visible:
                    visit(p)
            in_progress.discard(v)
            visited.add(v)
            result.append(v)

        for v in range(n_visible):
            visit(v)
        self._topo_cache = result
        self._topo_cache_visible_count = n_visible
        return result

    def _invalidate_topo_cache(self) -> None:
        """Drop the cached topological order. Called from _install_var when
        a variable's signature changes (parents may have changed)."""
        self._topo_cache = None

    def _maybe_demote(self, var: int, cycle: int) -> None:
        """Move var into the dormant partition if it meets stability criteria.
        Trass and noise_floor demote unconditionally. Non-trass/non-noise_floor
        require certified+stable with no active watch state, weak streak, or
        defer streak."""
        if self._live_set is None:
            return
        n = self.ledger.vars[var]
        # P2: gate on cert only — status=="trass" is a write-only sync field and
        # must not trigger dormancy independently. A status-only-trass var has no
        # cert scope, no evidence, and no invalidation conditions; parking it via
        # status bypasses the cert-stability criteria below.
        if n.role_for("skip") == "trass":
            self._live_set.discard(var)
            return
        if n.role_for("skip") == "noise_floor":
            # Best fit accepted at noise floor — park immediately. Sentinel
            # re-triggers at 3×ε if the fit genuinely changes.
            self._live_set.discard(var)
            return
        # [CONSEQUENCE-WEIGHT P3] dormancy age floor scaled by downstream consequence.
        # Original: min_cert_age = self._min_dormant_cert_age (100 cycles, uniform).
        # Revert: replace right-hand side with just self._min_dormant_cert_age.
        min_cert_age = self._min_dormant_cert_age + self._consequence_tier(var) * 50
        if (n.status == "certified"
                and n.authoritative
                and self.weak_streak.get(var, 0) == 0
                and self.defer_streak.get(var, 0) == 0
                and len(n.envelope.deltas) >= 100
                and cycle - n.envelope.certified_at_cycle >= min_cert_age):
            self._live_set.discard(var)

    def _maybe_park_var(self, var: int, cycle: int) -> None:
        """Evaluate and apply parking eligibility for one authoritative var.

        A var earns parking when:
          - It is covered by a confirmed regime with an active_sentinel
          - It has zero unique failures in the last _PARK_W cycles
          - The covering regime's sentinel has passed at least _PARK_K times
            (authority >= 2 already, but _PARK_K checks accumulated passes)

        When parked, run_cycle skips the leaf sentinel each cycle unless a wake
        condition fires (regime sentinel fails, revalidation interval, or cert
        invalidation via parent_change/sentinel_failure resets parked=False).
        """
        n = self.ledger.vars[var]
        if n.parked:
            return  # already parked
        if not n.authoritative:
            return  # no leaf sentinel to park
        if n.covered_by_regime_id is None:
            return  # not under any confirmed regime
        if n.unique_failures_caught > 0:
            return  # has caught something no higher handle caught
        if n.cycles_since_unique_failure < _PARK_W:
            return  # not quiet long enough
        # Check the covering regime's pass count
        regime_id = n.covered_by_regime_id
        sig = next((s for s in self.regime_register._confirmed if s.regime_id == regime_id), None)
        if sig is None or sig.active_sentinel is None:
            return  # no active sentinel on covering regime
        if self._regime_sentinel_passes < _PARK_K:
            return  # regime hasn't accumulated enough passes
        n.parked = True
        n.park_cycle = cycle
        self.ledger.event_log.append(
            f"c{cycle}: x{var} PARKED (regime R{regime_id} coverage, "
            f"csuf={n.cycles_since_unique_failure}, rpass>={_PARK_K})"
        )

    def _screen_candidate_parents(self, target: int, m: int) -> Set[int]:
        """Per-target sensitivity screen. For each candidate var x, force x to
        0.05 and 0.95 and measure how much `target` moves. Return the top-M
        candidates by movement magnitude.

        Cost: 2 × (n_visible - 1) × predict_var_under_intervention calls.
        Replaces the certified-only `available` set when parent_screen_m > 0.

        Only vars with non-zero movement compete; if fewer than M have any
        movement at all, the returned set may be smaller than M. Route certs
        (trass role) for this target are respected: positively-excluded vars
        are dropped from the result even if they score highly.

        Provider path (when self._parent_ranker is set):
          Delegates ranking to the provider. ChainedAgent still applies route-cert
          exclusion AFTER ranking — providers must not touch certs. Intervention
          cost is accounted here for SensitivityParentRanker (2 per candidate);
          other providers are responsible for their own cost outside this counter.
        """
        n = self.ledger.vars[target]
        if self._parent_ranker is not None:
            candidates: Set[int] = {x for x in range(self.world.visible_count) if x != target}
            ranking = self._parent_ranker.rank_parents(target, candidates, m)
            self._hybrid_parent_ranker_calls += 1
            _ranking_diag = getattr(ranking, "diagnostics", {})
            self.total_interventions += int(_ranking_diag.get("sensitivity_rescue_interventions", 0))
            # Mirror intervention cost for the default SensitivityParentRanker path,
            # which runs 2 world calls per candidate (same as the inline loop below).
            if isinstance(self._parent_ranker, SensitivityParentRanker):
                self.total_interventions += 2 * len(candidates)
            # Route-cert exclusion: providers must not apply cert logic; ChainedAgent does it here.
            post_route = tuple(
                x for x in ranking.ranked
                if n.route_certs.get(x) is None or n.route_certs[x].role != "trass"
            )
            if (
                self._scaffold_memory_mode == "assist_feature"
                and self._scaffold_memory_index is not None
                and len(post_route) >= 2
            ):
                post_route = tuple(
                    self._scaffold_memory_index.rank_candidate_keys(
                        target,
                        nethra_context_key(
                            operation="parent_candidates",
                            var=target,
                            visible=self.world.visible_count,
                        ),
                        post_route,
                    )
                )
            if self._nethra_memory_index is not None:
                post_route = tuple(
                    self._nethra_memory_index.rank_candidates(
                        var=target,
                        context_key=nethra_context_key(
                            operation="parent_candidates",
                            var=target,
                            visible=self.world.visible_count,
                        ),
                        candidates=post_route,
                        hook="parent_candidates",
                        cycle=getattr(self, "_current_cycle_for_memory", 0),
                    )
                )
            excluded = tuple(x for x in ranking.ranked if x not in post_route)
            source_by_candidate = getattr(ranking, "source_by_candidate", {})
            post_route_sources = {
                x: source_by_candidate.get(x, "")
                for x in post_route
                if source_by_candidate.get(x, "")
            }
            self._parent_proposal_diagnostics.record_call(
                tuple(ranking.ranked), post_route, _ranking_diag
            )
            self._pending_parent_rankings[target] = post_route
            self._pending_parent_sources[target] = post_route_sources
            if excluded and hasattr(self._parent_ranker, "observe_route_exclusions"):
                self._parent_ranker.observe_route_exclusions(target, excluded)  # type: ignore[attr-defined]
            if self._probe_proposer is not None and hasattr(self._probe_proposer, "observe_parent_ranking_metadata"):
                self._probe_proposer.observe_parent_ranking_metadata(  # type: ignore[attr-defined]
                    target, post_route, post_route_sources
                )
            elif self._probe_proposer is not None and hasattr(self._probe_proposer, "observe_parent_ranking"):
                self._probe_proposer.observe_parent_ranking(target, post_route)  # type: ignore[attr-defined]
            return set(post_route)
        # Inline path: existing behavior unchanged when no provider is set.
        visible = self.world.visible_count
        scored: List[Tuple[float, int]] = []
        for x in range(visible):
            if x == target:
                continue
            lo = self.world.predict_var_under_intervention(target, x, 0.05)
            hi = self.world.predict_var_under_intervention(target, x, 0.95)
            self.total_interventions += 2
            delta = abs(hi - lo)
            if delta > DEFAULT_TOLERANCE:
                scored.append((delta, x))
        if (
            self._scaffold_memory_mode == "assist_feature"
            and self._scaffold_memory_index is not None
            and len(scored) >= 2
        ):
            context = nethra_context_key(
                operation="parent_candidates",
                var=target,
                visible=self.world.visible_count,
            )
            scaffold_scores = {
                x: sum(
                    score
                    for _, score, _ in self._scaffold_memory_index.useful_local_matches(
                        target,
                        context,
                        x,
                    )
                )
                for _, x in scored
            }
            if any(score > 0 for score in scaffold_scores.values()):
                self._scaffold_memory_index.rank_candidate_keys(
                    target,
                    context,
                    tuple(x for _, x in scored),
                )
            scored.sort(key=lambda row: (-row[0], -scaffold_scores.get(row[1], 0), row[1]))
        else:
            scored.sort(reverse=True)
        if self._nethra_memory_index is not None:
            context = nethra_context_key(
                operation="parent_candidates",
                var=target,
                visible=self.world.visible_count,
            )
            ranked_candidates = tuple(
                self._nethra_memory_index.rank_candidates(
                    var=target,
                    context_key=context,
                    candidates=tuple(x for _, x in scored),
                    hook="parent_candidates",
                    cycle=getattr(self, "_current_cycle_for_memory", 0),
                )
            )
            rank_pos = {x: i for i, x in enumerate(ranked_candidates)}
            scored.sort(key=lambda row: (rank_pos.get(row[1], len(scored)), -row[0], row[1]))
        return {
            x for _, x in scored[:m]
            if n.route_certs.get(x) is None or n.route_certs[x].role != "trass"
        }

    def _cheap_salience_screen(self) -> Set[int]:
        """Two-probe causal salience screen run once at cold start.

        For each visible var, force it to 0.05 and 0.95 and check whether
        any OTHER visible var moves by more than DEFAULT_TOLERANCE across
        that range. If yes → salient (potentially load-bearing cause).
        If no → inert (no observable downstream effect at this state).

        Cost: 2 × predict_under_intervention per var = O(n_vars) probes total,
        vs the O(n_vars × hypotheses × probes) cost of full auditing all vars.
        """
        salient: Set[int] = set()
        visible = self.world.visible_count
        for var in range(visible):
            state_lo = self.world.predict_under_intervention(var, 0.05)
            state_hi = self.world.predict_under_intervention(var, 0.95)
            self.total_interventions += 2
            for j in range(visible):
                if j == var:
                    continue
                if abs(state_lo[j] - state_hi[j]) > DEFAULT_TOLERANCE:
                    salient.add(var)
                    break
        return salient

    def _priority_score(self, var: int, cycle: int) -> float:
        """Frontier priority score. Higher = audit sooner.

          failure_signal         consecutive sentinel failures (most urgent)
        + consequence_weight     var's cost weight from ledger
        + dep_score              1 per live var that currently believes this var as parent
        + uncertainty_age        time since last change, if never audited (novel vars age in)
        - clean_passes           skip count × 0.1 (penalises boring stable vars)
        """
        cycle = int(getattr(cycle, "cycle", cycle))
        n = self.ledger.vars[var]
        failure_signal = n.consecutive_sentinel_failures * 2.0
        consequence = n.cost_weight
        dep_score = sum(
            1.0 for fv in (self._live_set or set())
            if var in self.ledger.vars[fv].parents
        )
        uncertainty_age = (cycle - n.last_changed_cycle) * 0.01 if n.full_audits == 0 else 0.0
        clean_passes = n.skip_count * 0.1
        return failure_signal + consequence + dep_score + uncertainty_age - clean_passes

    def _pick_initial_frontier(self, salient: Set[int], K: int) -> Set[int]:
        """Return top-K salient vars by priority score.

        K >= visible_count acts as a full-audit flag: return all visible vars
        regardless of salience (old behavior, useful in tests and small worlds).
        Falls back to all visible vars if the salience screen found nothing.
        """
        visible = self.world.visible_count
        if K >= visible:
            return set(range(visible))
        candidates = list(salient) if salient else list(range(visible))
        if len(candidates) <= K:
            return set(candidates)
        scored = sorted(candidates, key=lambda v: self._priority_score(v, 0), reverse=True)
        return set(scored[:K])

    def initialize(self) -> None:
        """Sparse initialization: cheap salience screen → pick K frontier vars → audit only those.

        Replaces the previous full audit of every visible var. Cost drops from
        O(n_vars × hypotheses × probes) to O(n_vars × 2) for the screen plus
        O(K × hypotheses × probes) for the K frontier audits.

        Vars screened as inert are stored in _inert_vars and skipped at startup.
        They remain eligible for wakeup via sentinel cascade or dependency events.
        """
        salient = self._cheap_salience_screen()
        self._inert_vars = set(range(self.world.visible_count)) - salient
        frontier = self._pick_initial_frontier(salient, self.frontier_k)
        first_pass_order = self._audit_priority_order(list(frontier))
        for var in first_pass_order:
            parents, func, score, second, fd = self._full_audit_var(var, 0)
            self._install_var(var, parents, func, score, second, 0, fd)
        self._live_set = set(frontier)

    def on_variable_revealed(self, new_var: int, cycle: int) -> None:
        """Hook fired when world reveals a new variable. Audits the new var;
        existing certs are untouched — filter ledger: no recert without failure.
        Trass-status vars are valid parent candidates (route certs, not skip certs,
        gate available_parents; no route certs exist → nothing excluded by cert).

        Invalidates the topo cache because visible_count grew."""
        self._invalidate_topo_cache()
        if self._live_set is not None:
            self._live_set.add(new_var)
        parents, func, score, second, fd = self._full_audit_var(new_var, cycle)
        self._install_var(new_var, parents, func, score, second, cycle, fd)
        self.ledger.event_log.append(
            f"c{cycle}: x{new_var} REVEALED — first audit complete; "
            f"available parents at reveal time: "
            f"{sorted(other for other, n in self.ledger.vars.items() if other != new_var and n.status in ('certified', 'trass', 'proposed'))}"
        )

    def run_cycle(self, cycle: int) -> None:
        """Process one cycle of agent operation. Steps:

          1. First pass over all visible vars in TOPOLOGICAL order:
             - if trass: count as skip, no work
             - if has compression and gate matches: count as compression-skip
             - if authoritative (sentinels+role): run sentinel check via
               cost-weighted dispatch; on fail, invalidate self+descendants
               and queue them for audit; on pass, count as sentinel-skip
             - otherwise: queue for full audit

          2. Audit pass: variables in needs_audit get full audit in
             topological-filtered order, up to priority_audit_budget.
             Excess vars are deferred to next cycle.

          3. Each audit calls _full_audit_var (fits) then _install_var
             (commits the result, attaches sentinels, possibly promotes/
             discovers compressions, possibly fires novelty).

          4. Append a CycleRecord with skipped/audited/deferred lists for
             offline diagnostic comparison.
        """
        cycle = int(getattr(cycle, "cycle", cycle))
        self._current_cycle_for_memory = cycle
        self._uncertain_this_cycle.clear()
        self._run_uncertainty_consolidation(cycle)
        self._run_authority_strength(cycle)
        self._run_background_nethra(cycle)
        self._run_background_residual_classification(cycle)

        skipped: List[int] = []
        audited: List[int] = []
        drift: List[int] = []
        deferred: List[int] = []
        novelty_fired = False
        _cert_events: List[RegimeCertEvent] = []
        _passive_stressed_vars: Set[int] = set()
        _shadow_ok_vars: Set[int] = set()  # vars where shadow said ok this cycle
        _shadow_ok_var_keys: Dict[int, Tuple[object, Optional[str]]] = {}
        _structural_change_this_cycle = False

        # First pass: cheap paths and queue full audits.
        needs_audit: List[int] = []
        # Graded cascade: sentinel failures in this cycle that have not yet been
        # confirmed by a local re-audit. Value = reason string for the cascade
        # event log. Cascade fires only after _install_var confirms sig_changed.
        _sentinel_failed_vars: Dict[int, Tuple[str, float]] = {}

        # v28+: process variables in dependency order (parents first) so a
        # parent's sentinel failure invalidates descendants BEFORE they take
        # cheap-path skips based on now-invalid parent assumptions. Previously
        # the loop went in numeric order, which happened to align with
        # topological order in this toy because _random_dag builds low-to-high,
        # but that's an accidental coincidence — intent says topological.
        topo_order = self._topological_order(self.world.visible_count)

        # Composite nethra check: replay each composite's joint probe. Returns
        # Component sentinels first (one probe per component regardless of size).
        # Covered vars are excluded from pairwise _check_composites below.
        _hyper_covered = self._check_hyper_composites(cycle)

        # Pairwise composite check for vars not already covered by a component.
        # Failing composites are revoked here — their members' certs are reset
        # before the first-pass loop so those vars fall through to the audit queue.
        composite_passing = _hyper_covered | self._check_composites(cycle)

        # Update stability horizons for display. Diagnostic only — no behavioral
        # consequence. Dormant wakeup is failure-driven (sentinel fail → cascade).
        for _v in topo_order:
            _vn = self.ledger.vars[_v]
            if (_vn.role_for("skip") != "trass"
                    and _vn.status != "trass"
                    and _vn.median_interval > 0):
                self.ledger.update_stability_horizon(_v, cycle)

        # First-pass loop runs in topological order over LIVE vars only.
        # Dormant vars (removed from _live_set by _maybe_demote) are skipped.
        # Re-entry is failure-driven: sentinel failure → cascade invalidation → live.
        # Build open-novelty set once per cycle (used in needs_audit rate-limit).
        _novelty_vars: Set[int] = {
            nv.affected_var for nv in self.ledger.novelty if nv.status == "open"
        }
        # Audit rate-limit constants (used in needs_audit gate and cert issuance).
        _BACKOFF_THRESHOLD = 4
        _BACKOFF_INTERVAL  = 8
        _NOVELTY_INTERVAL  = 5
        _STABLE_THRESHOLD  = 3

        # Regime sentinel check: replay each confirmed regime's active cluster
        # probe. Vars covered by a passing regime sentinel may skip individual
        # leaf checks this cycle — the cluster probe is the amortized signal.
        # Costs 2 world calls per regime with an active_sentinel.
        # Regimes without an active_sentinel annotate only; they do not gate here.
        _regime_covered, _rpass, _rfail, _rno, _regime_failed_vars = self.regime_register.check_sentinels(self.world)
        self._regime_sentinel_passes += _rpass
        self._regime_sentinel_fails += _rfail
        self._regime_no_sentinel += _rno
        self._last_regime_failed_vars = {int(v) for v in _regime_failed_vars}
        self.total_interventions += 2 * (_rpass + _rfail)
        regime_stable = _regime_covered

        # Sentinel utility accounting: update per-var coverage fields and
        # increment cycles_since_unique_failure for all visible vars each cycle.
        _regime_membership = self.regime_register.regime_membership()
        _composite_by_var: Dict[int, int] = {}
        for _cni, _cn in enumerate(self.ledger.composites):
            for _cm in _cn.members:
                if _cm not in _composite_by_var:
                    _composite_by_var[_cm] = _cni
        for _v in range(self.world.visible_count):
            _vn = self.ledger.vars[_v]
            _vn.covered_by_regime_id = _regime_membership.get(_v)
            _vn.covered_by_composite_id = _composite_by_var.get(_v)
            _vn.cycles_since_unique_failure += 1

        for var in topo_order:
            n = self.ledger.vars[var]

            # Composite nethra skip: var is covered by a joint interaction cert
            # whose probe passed this cycle in _check_composites. Checked BEFORE
            # the individual trass-skip so that individually-trass vars covered
            # by a composite are intercepted here, not collapsed to the trass path.
            # The composite sentinel (2 world calls per composite, done once above)
            # is the cheap-path check for both members jointly.
            # Composite evidence lives on the CompositeNethra, not on individual certs.
            # Individual certs remain accurate as individual claims.
            if var in composite_passing:
                self.skip_count += 1
                self.composite_skip_count += 1
                n.skip_count += 1
                skipped.append(var)
                continue

            # Trass: the skip shortcut fires — but only for CONFIRMED trass certs.
            # FILTER LEDGER: the trass cert IS the shortcut — "not otherwise excluded
            # from skip." It was earned by testing; it fires here by default. The cert
            # carries scope, witnesses, and invalidation conditions. When those conditions
            # trigger, the cert is revoked and the shortcut stops firing.
            # WHAT IT OUGHT TO DO: gate on cert only (role_for("skip") == "trass").
            # status=="trass" is a write-only sync field maintained for legacy reasons;
            # it must not be read as an independent skip authority. A var with
            # status=="trass" but no skip cert has no earned shortcut: no scope, no
            # witnesses, no invalidation conditions. Using it as a gate bypasses the
            # filter ledger entirely and makes the var permanently invisible to
            # _retest_trass_vars (scope-expansion revalidation).
            #
            # PROVISIONAL TRASS: a trass cert is provisional until a subsequent full
            # audit has occurred AFTER the cert was issued. cert.audits_at_issuance
            # records n.full_audits at issuance; confirmation requires n.full_audits
            # > cert.audits_at_issuance. This is cert-local, not a lifetime counter,
            # so it catches mid-run new certs (world changes → wrong fit → trass cert
            # at full_audits=N → provisional until audit N+1 occurs).
            #
            # Provisional path: on each cycle, run one cheap probe (P1-C).
            # If the probe detects propagation, the cert was wrong — invalidate
            # and queue for full audit this cycle. If no propagation, increment
            # sentinel_passes and count as a skip (shortcut is still valid).
            # Hard-suppress is only earned after _trass_strong_threshold quiet
            # probe cycles — until then detection is active, not suppressed.
            if n.role_for("skip") == "trass":
                cert = n.certificates["skip"]
                _trass_strong_threshold = (
                    _STRONG_TRASS_SENTINEL_PASSES + self._consequence_tier(var) * 3
                )
                if cert.confirmed and cert.sentinel_passes >= _trass_strong_threshold:
                    # Strong confirmed trass: cert has accumulated enough quiet probe
                    # cycles — hard-suppress future detection.
                    self.skip_count += 1
                    self.trass_skip_count += 1
                    n.skip_count += 1
                    skipped.append(var)
                    if self._live_set is not None:
                        self._live_set.discard(var)
                    continue
                # Provisional trass: run one cheap probe before crediting the skip.
                # P1-C: this is what separates provisional from hard-suppress.
                # Cost: 2 world queries. total_interventions accounts for them.
                self.total_interventions += 2
                if self._provisional_trass_probe(var):
                    # Probe detected propagation — cert is stale. Invalidate and
                    # queue for full audit. The full op-role test will re-certify.
                    n.certificates.pop("skip", None)
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} provisional_trass PROBE FAILED "
                        f"— cert invalidated, queued for audit"
                    )
                    needs_audit.append(var)
                    continue
                # Probe passed — no propagation detected. Increment stable counter.
                if not cert.confirmed:
                    _new_sp = 1
                    n.certificates["skip"] = dataclasses.replace(
                        cert, confirmed=True, sentinel_passes=_new_sp
                    )
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} provisional_trass confirmed "
                        f"(stable_cycles=1/{_trass_strong_threshold} needed)"
                    )
                else:
                    _new_sp = cert.sentinel_passes + 1
                    n.certificates["skip"] = dataclasses.replace(cert, sentinel_passes=_new_sp)
                if _new_sp >= _trass_strong_threshold:
                    n.status = "trass"
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} trass STRONG "
                        f"({_new_sp}/{_trass_strong_threshold} stable cycles — hard-suppress earned)"
                    )
                self.skip_count += 1
                self.trass_skip_count += 1
                n.skip_count += 1
                skipped.append(var)
                if _new_sp >= _trass_strong_threshold and self._live_set is not None:
                    self._live_set.discard(var)
                continue

            # Dormant gate: vars not in the live set are handled by the periodic
            # sweep. Skip here — sentinel and audit will fire when the sweep
            # detects a genuine deviation.
            if self._live_set is not None and var not in self._live_set:
                continue

            # Proposed compression trial: for each compression not yet promoted,
            # check gate+prediction against world state to accumulate pred_passes.
            # Runs regardless of what path the variable takes this cycle.
            if n.compressions and n.status in ("certified", "proposed"):
                tol = n.current_tolerance
                actual = self.world.state[var]
                for comp in n.compressions:
                    if comp.pred_passes >= self.compression_promote_after:
                        continue
                    if comp.gate_matches(self.world.state):
                        if abs(actual - comp.simplified_value) <= tol:
                            comp.pred_passes += 1
                            if comp.pred_passes == self.compression_promote_after:
                                # Q3: compression cert earned by accumulated evidence.
                                # FILTER LEDGER: this cert IS the compress shortcut.
                                # Once issued, _try_compression fires by default when
                                # the gate matches. The cert is revoked on prediction
                                # mismatch (below) — failure earns revocation.
                                # WHAT IT OUGHT TO DO:
                                #   targets: gate var indices — the scope under which
                                #     the simplification holds. Cert authority is
                                #     bounded to this gate condition.
                                #   witnesses: (state_snapshot, simplified_value) — the
                                #     attribution handle. If the compression is later
                                #     disputed, this is where descent starts.
                                #   changes == trials: every gate match was a confirmed
                                #     equivalence. The pred_passes count IS the evidence.
                                # LAZY DECOMPOSITION: no sub-structure pre-built. If
                                # the compress cert fails (mismatch), it is revoked and
                                # the compression must re-earn it. Decomposition into
                                # why it failed (gate boundary wrong? simplified_value
                                # drifted?) is only earned if failure recurs.
                                if "compress" not in n.certificates:
                                    self.ledger.issue_cert(
                                        var, "compress", "trass", "guarded_reuse",
                                        context_parents=tuple(n.parents) if n.parents else (),
                                        context_visible=self.world.visible_count,
                                        context_cycle=cycle,
                                        targets=tuple(gv for gv, _, _ in comp.gate),
                                        substitutions_tested=("compression_match",),
                                        changes=comp.pred_passes,
                                        trials=comp.pred_passes,
                                        earned_by="compression_equivalence",
                                        witnesses=((tuple(self.world.state), comp.simplified_value),),
                                    )
                        else:
                            # Prediction mismatch under the gate — failure signal earned.
                            # FILTER LEDGER: cert is revoked; shortcut stops firing.
                            # Reset pred_passes so the compression must re-earn its cert
                            # from scratch. If this failure recurs, that recurrence is
                            # the signal to decompose further (why does the gate fail?
                            # wrong boundary? simplified_value drifted? gate condition
                            # too coarse?). That decomposition is lazy — only earned by
                            # repeated failure, not pre-built.
                            comp.pred_passes = 0
                            n.certificates.pop("compress", None)

            # Cheapest path: matching compression
            if n.compressions and n.sentinels and n.status in ("certified", "proposed"):
                comp_pred = self._try_compression(var)
                if comp_pred is not None:
                    self.skip_count += 1
                    self.compression_skip_count += 1
                    n.skip_count += 1
                    skipped.append(var)
                    continue

            # Regime-stable skip: var is covered by a confirmed regime whose
            # active sentinel passed this cycle (check_sentinels result).
            # The regime-level probe (2 world calls, done once above) is the
            # authority — not quiescence. Only fires when the var is authoritative.
            if var in regime_stable and n.authoritative:
                self.skip_count += 1
                self._regime_skip_count += 1
                n.skip_count += 1
                skipped.append(var)
                self._maybe_park_var(var, cycle)
                continue

            # Passive residual check: compute expected-next-state from certified
            # func(parents) at current world state. No interventions — O(1).
            # If residual ≤ tolerance: passive says OK → skip active sentinel
            #   (save IVs; does NOT certify anything — only defers the probe).
            # If residual > tolerance: passive stressed → run active sentinel
            #   (stress record also feeds regime register as candidate evidence).
            # This implements passive-first monitoring: ordinary observation is
            # cheap, intervention is reserved for when authority must be earned
            # or repaired (residual stress OR downstream contradiction).
            #
            # When _residual_predictor is set (hybrid-interfaces mode), the
            # prediction is delegated to the provider. The provider MUST NOT
            # issue certs — it only returns a ResidualPrediction signal.
            # When None (default / hybrid-off), the inline FUNC_LIBRARY path
            # runs as before — identical behavior, zero overhead.
            _passive_ok = False
            _passive_stressed = False
            if n.authoritative:
                if self._residual_predictor is not None:
                    _parent_vals = [self.world.state[p] for p in n.parents]
                    _rp = self._residual_predictor.predict_residual(
                        var, n.parents, n.func,
                        _parent_vals, self.world.state[var], n.current_tolerance,
                    )
                    self._hybrid_residual_predictor_calls += 1
                    if _rp.ok:
                        _passive_ok = True
                        self._hybrid_residual_ok += 1
                    else:
                        _passive_stressed = True
                        self._passive_stress_count += 1
                        self._hybrid_residual_stressed += 1
                        _passive_stressed_vars.add(var)
                else:
                    # Inline path (hybrid-off / no provider): current behavior unchanged.
                    _f = FUNC_LIBRARY.get(n.func)
                    if _f is not None:
                        _parent_vals = [self.world.state[p] for p in n.parents]
                        _passive_pred = _f(_parent_vals)
                        _passive_residual = abs(self.world.state[var] - _passive_pred)
                        if _passive_residual <= n.current_tolerance:
                            _passive_ok = True
                        else:
                            _passive_stressed = True
                            self._passive_stress_count += 1
                            _passive_stressed_vars.add(var)

            if self._parent_ranker is not None and hasattr(self._parent_ranker, "observe_residual_event"):
                self._parent_ranker.observe_residual_event(var, cycle, _passive_stressed)  # type: ignore[attr-defined]
            if self._probe_proposer is not None and hasattr(self._probe_proposer, "observe_residual_event"):
                self._probe_proposer.observe_residual_event(var, cycle, _passive_stressed)  # type: ignore[attr-defined]

            # Shadow residual (Stage 3A): observe actual symbolic residual and
            # predict — NEVER used for gating, certs, skips, or any operative
            # decision. Only increments diagnostic counters.
            if n.authoritative and self._shadow_residual_enabled:
                _sf = FUNC_LIBRARY.get(n.func)
                if _sf is not None:
                    _shadow_pv = [self.world.state[p] for p in n.parents]
                    # Symbolic reference: always computed from FUNC_LIBRARY, never from the
                    # active residual provider. When _residual_predictor is set (hybrid mode),
                    # _passive_ok/_passive_stressed are provider-derived; comparing shadow
                    # against those would make false_ok_vs_symbolic meaningless in learned
                    # provider stages. The symbolic reference stays invariant across stages.
                    _symbolic_residual_value = abs(self.world.state[var] - _sf(_shadow_pv))
                    _symbolic_passive_ok = _symbolic_residual_value <= n.current_tolerance
                    _symbolic_passive_stressed = not _symbolic_passive_ok

                    # Build a ResidualFeatureVector for feature-conditioned calibration.
                    # Only uses already-visible agent/world state — no hidden truth.
                    _shadow_fv = ResidualFeatureVector(
                        var=var,
                        cycle=cycle,
                        parents=tuple(n.parents),
                        func=n.func,
                        parent_vals=tuple(_shadow_pv),
                        actual=self.world.state[var],
                        tolerance=n.current_tolerance,
                        consequence_tier=self._consequence_tier(var),
                        full_audits=n.full_audits,
                        sentinel_count=len(n.sentinels),
                        cert_age=(cycle - n.first_certified_cycle) if n.first_certified_cycle > 0 else 0,
                    )

                    # predict_shadow BEFORE observe — shadow must not train on the
                    # current sample before predicting it. This order is the core
                    # honesty guarantee: the predictor sees only history, not the future.
                    _sp = self._shadow_residual_predictor.predict_shadow(  # type: ignore[union-attr]
                        var, n.func, n.current_tolerance, fv=_shadow_fv
                    )
                    _sp_insufficient = self._shadow_residual_predictor._last_call_insufficient  # type: ignore[union-attr]

                    # observe AFTER predicting — calibrator learns from the actual
                    # symbolic residual using FUNC_LIBRARY (never from provider output).
                    self._shadow_residual_predictor.observe(  # type: ignore[union-attr]
                        n.func, _symbolic_residual_value, fv=_shadow_fv
                    )

                    # Track per-call counters
                    self._shadow_residual_calls += 1
                    if _sp_insufficient:
                        self._shadow_residual_insufficient += 1
                        self._shadow_residual_stressed += 1
                    elif _sp.ok:
                        self._shadow_residual_ok += 1
                        _shadow_ok_vars.add(var)
                    else:
                        self._shadow_residual_stressed += 1

                    # Compare shadow against FUNC_LIBRARY symbolic decision, not provider.
                    # false_ok_vs_symbolic = shadow predicted ok when FUNC_LIBRARY said stressed.
                    if _sp.ok == _symbolic_passive_ok:
                        self._shadow_agree_symbolic += 1
                    if _sp.ok and _symbolic_passive_stressed:
                        self._shadow_false_ok_vs_symbolic += 1
                    if _sp.stressed and _symbolic_passive_ok:
                        self._shadow_false_stress_vs_symbolic += 1
                    if _sp.ok:
                        self._shadow_would_save_iv += len(n.sentinels)
                        if _symbolic_passive_stressed:
                            self._shadow_would_miss_symbolic_stress += len(n.sentinels)

                    if self._shadow_key_authority is not None:
                        _meta = self._shadow_residual_predictor.last_prediction_metadata  # type: ignore[union-attr]
                        _key_used = _meta.get("key_used")
                        _key_type = _meta.get("key_type")
                        if self._shadow_residual_predictor._is_feature_mode:  # type: ignore[union-attr]
                            _authority_key = (
                                _key_used
                                if _key_used is not None
                                else ShadowResidualKeyAuthority.KEY_INSUFFICIENT
                            )
                            _authority_key_type = _key_type
                            self._shadow_key_authority.record_prediction(
                                key=_authority_key,
                                key_type=_authority_key_type,
                                shadow_ok=_sp.ok,
                                shadow_stressed=_sp.stressed,
                                symbolic_ok=_symbolic_passive_ok,
                                symbolic_stressed=_symbolic_passive_stressed,
                                would_save_iv=len(n.sentinels) if _sp.ok else 0,
                                would_miss_symbolic_stress=(
                                    len(n.sentinels)
                                    if _sp.ok and _symbolic_passive_stressed
                                    else 0
                                ),
                                cycle=cycle,
                            )
                            if _sp.ok:
                                _shadow_ok_var_keys[var] = (
                                    _authority_key,
                                    _authority_key_type,
                                )

                    # Feature-calibrator key-usage and false-ok-per-key counters.
                    if self._shadow_residual_predictor._is_feature_mode:  # type: ignore[union-attr]
                        _lku = self._shadow_residual_predictor._last_key_used  # type: ignore[union-attr]
                        if _lku is None:
                            self._shadow_feature_key_insufficient += 1
                        elif _lku[0] == "func_var":
                            self._shadow_feature_key_func_var += 1
                            if _sp.ok and _symbolic_passive_stressed:
                                self._shadow_feature_fok_func_var += 1
                        elif _lku[0] == "func_tier_parent":
                            self._shadow_feature_key_func_tier_parentcount += 1
                            if _sp.ok and _symbolic_passive_stressed:
                                self._shadow_feature_fok_func_tier_parentcount += 1
                        elif _lku[0] == "func_tier":
                            self._shadow_feature_key_func_tier += 1
                            if _sp.ok and _symbolic_passive_stressed:
                                self._shadow_feature_fok_func_tier += 1
                        elif _lku[0] == "func":
                            self._shadow_feature_key_func += 1
                            if _sp.ok and _symbolic_passive_stressed:
                                self._shadow_feature_fok_func += 1
                        else:  # "global"
                            self._shadow_feature_key_global += 1
                            if _sp.ok and _symbolic_passive_stressed:
                                self._shadow_feature_fok_global += 1

            # Parked sentinel skip: leaf cert redundant — higher handle covered
            # this var for _PARK_W+ cycles with no unique failures.
            # Wake conditions (any one overrides skip):
            #   - covering regime sentinel failed (var in _regime_failed_vars)
            #   - sparse revalidation due (cycle % _PARK_REVALIDATE_INTERVAL == 0)
            #   - passive residual stressed (cert may be stale — recheck)
            #   - cert was invalidated externally (parked reset on parent_change/sentinel_failure)
            if n.parked and n.authoritative:
                _regime_failed = var in _regime_failed_vars
                _revalidate = (cycle % _PARK_REVALIDATE_INTERVAL == 0)
                if not _regime_failed and not _revalidate and not _passive_stressed:
                    self.skip_count += 1
                    self._parked_skip_count += 1
                    n.skip_count += 1
                    skipped.append(var)
                    continue
                # Wake: run leaf sentinel this cycle
                self._woken_count += 1
                n.parked = False
                _wake_reason = (
                    "regime_failed" if _regime_failed
                    else "passive_stress" if _passive_stressed
                    else "revalidation"
                )
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} WOKEN from parking ({_wake_reason})"
                )

            # Passive OK gate: residual inside envelope → skip active sentinel.
            # Authority is NOT changed — this only defers the intervention probe
            # for this cycle. The cert remains valid; the shortcut fires because
            # passive evidence is sufficient THIS CYCLE, not because we certified.
            if _passive_ok:
                self._passive_saved_iv += len(n.sentinels)
                self.skip_count += 1
                n.skip_count += 1
                skipped.append(var)
                continue

            # Sentinel path.
            # FILTER LEDGER: sentinels are sparse exclusion monitors — cheap probes
            # that detect when the cert's boundary has been crossed. They do NOT confirm
            # every use. Sentinel passes → shortcut fires, no further accounting needed
            # (lazy decomposition). Sentinel fails → failure signal earned; open the cert,
            # replay witnesses, attribute the failure.
            #
            # WHAT IT OUGHT TO DO:
            #   sentinel passes → skip, done. No witness replay (that is positive-ledger).
            #   sentinel fails  → replay cert witnesses to distinguish two cases:
            #     (a) witnesses no longer propagate: cert authority has expired, not that
            #         the var became trass. Recertify; do not collapse. A new witness may
            #         re-establish tareth.
            #     (b) witnesses still propagate: the world changed in a way the sentinel
            #         correctly caught. Proceed with invalidation cascade.
            # The witness replay belongs in the failure branch. It is currently in the
            # pass branch — that is backwards. Replaying on pass is the positive-ledger
            # pattern: paying accounting cost where no failure signal exists.
            # Q5 DIVERGENCE: check_var_sentinels_with_envelope runs (iv_slot, iv_val)
            # probes from a discrimination pool. This is drift detection only. The
            # cert's original witnesses (stored in skip_cert.witnesses) are the
            # attribution handles for failure diagnosis — not for pass-path confirmation.
            if n.authoritative:
                self.total_interventions += len(n.sentinels)
                passed, _, _, reason, _sentinel_max_dev = check_var_sentinels_with_envelope(
                    var, n, self.world, cycle,
                    self.cost_low_threshold, self.cost_high_threshold,
                )
                if len(n.envelope.deltas) >= self.envelope_certify_after:
                    n.envelope.maybe_certify(cycle)

                if passed:
                    # Sentinel passed — shortcut fires. No accounting, no witness replay.
                    # Filter ledger: pass means not otherwise excluded; fire and continue.
                    n.consecutive_sentinel_failures = 0
                    self.skip_count += 1
                    self.sentinel_skip_count += 1
                    n.skip_count += 1
                    skipped.append(var)
                    if "TEMPORAL_TRASS" in reason:
                        self.ledger.event_log.append(f"c{cycle}: x{var} {reason}")
                    # For proposed vars, sentinel passes count as matching observations
                    # toward promotion. In sparse-init, cascade audits are rare so the
                    # second matching audit may never come via re-audit; sentinel
                    # evidence fills the same role.
                    if n.status == "proposed" and n.sentinels:
                        _eff_promote_after = self.promote_after + self._consequence_tier(var) * 2
                        if n.strong_observations < _eff_promote_after:
                            n.strong_observations += 1
                            if n.strong_observations >= _eff_promote_after:
                                n.status = "certified"
                                if n.first_certified_cycle == 0:
                                    n.first_certified_cycle = cycle
                    self._maybe_demote(var, cycle)
                    continue
                # Sentinel failed — failure signal earned. Replay witnesses to attribute.
                # Case (a) authority_expired: witnesses no longer propagate → cert basis
                #   gone, not that the var became trass. Recertify; do not cascade.
                # Case (b) world_changed: witnesses still propagate → world drifted past
                #   the cert's tested scope. Proceed with invalidation cascade.
                _skip_cert = n.certificates.get("skip")
                _authority_expired = False
                if _skip_cert and _skip_cert.witnesses:
                    _saved_state = self.world.state
                    _any_witness_live = False
                    for _wsnap, _wiv in _skip_cert.witnesses:
                        self.world.state = list(_wsnap)
                        _wb = self.world.predict_under_intervention(var, _wsnap[var])
                        _wp = self.world.predict_under_intervention(var, _wiv)
                        for _wj in _skip_cert.targets:
                            if _wj >= self.world.visible_count:
                                continue
                            if abs(_wp[_wj] - _wb[_wj]) > self.ledger.vars[_wj].current_tolerance:
                                _any_witness_live = True
                                break
                        if _any_witness_live:
                            break
                    self.world.state = _saved_state
                    if not _any_witness_live:
                        # Case (a): authority expired. Recertify — do not cascade.
                        n.certificates.pop("skip", None)
                        self._certify_operation_role(var, cycle)
                        self.ledger.event_log.append(
                            f"c{cycle}: x{var} tareth witness expired on sentinel fail — recertifying"
                        )
                        needs_audit.append(var)
                        _authority_expired = True
                if _authority_expired:
                    continue
                # Case (b): world changed — local demotion only (graded cascade).
                # Single sentinel miss earns local re-audit, not immediate
                # descendant cascade. Cascade fires only after _install_var
                # confirms sig_changed (genuine parent mutation). Noisy misses
                # that resolve to the same fit produce zero cascade work.
                n.consecutive_sentinel_failures += 1
                self.ledger.vars[var].invalidate_certs("sentinel_failure")
                self._uncertain_this_cycle.add(var)
                # Demote local var without touching descendants.
                if n.status == "certified":
                    n.status = "uncertain"
                    n.collapse_log.append(f"c{cycle}: sentinel — local demotion (pending re-audit)")
                elif n.status in ("proposed", "quarantined"):
                    n.strong_observations = 0
                    n.status = "uncertain"
                    n.collapse_log.append(f"c{cycle}: sentinel — local demotion (pending re-audit)")
                n.tied_frontier = None
                if self._live_set is not None:
                    self._live_set.add(var)
                _sentinel_failed_vars[var] = (f"sentinel: {reason}", _sentinel_max_dev)
                self.sentinel_miss_count += 1
                # Shadow false-OK vs active sentinel: shadow said ok but sentinel
                # failed this cycle. Increments without changing any decision.
                if var in _shadow_ok_vars:
                    self._shadow_false_ok_vs_active_sentinel += 1
                    self._shadow_would_miss_active_failure += len(n.sentinels)
                    if (
                        self._shadow_key_authority is not None
                        and var in _shadow_ok_var_keys
                    ):
                        _authority_key, _authority_key_type = _shadow_ok_var_keys[var]
                        self._shadow_key_authority.record_active_failure(
                            key=_authority_key,
                            key_type=_authority_key_type,
                            would_miss_active_failure=len(n.sentinels),
                            cycle=cycle,
                        )
                # Utility accounting: was a higher handle also firing for this var?
                if var in _regime_failed_vars:
                    n.failures_also_caught_by_higher += 1
                else:
                    n.unique_failures_caught += 1
                    n.cycles_since_unique_failure = 0
                _cert_events.append(RegimeCertEvent(
                    var=var, cert_key="skip", event_type="failed",
                    delta=_sentinel_max_dev, cert_age=n.full_audits,
                ))
            # Needs full audit — rate-limit in two cases:
            #
            # Case A: sentinel-stable loop. Sentinel failed but audit returns
            # the same fit every time (consecutive_sentinel_failures accumulated,
            # no sig_change to reset it). The sentinel failure is real but the
            # full audit learns nothing new. Only re-audit every BACKOFF_INTERVAL.
            #
            # Case B: open vocabulary novelty. _maybe_novelty fires when
            # weak_streak >= novelty_weak_streak — the fit keeps swinging,
            # meaning the library is insufficient or the world changes faster
            # than the audit can track. Re-auditing every cycle consumes
            # interventions without converging. Rate-limit to once per
            # NOVELTY_INTERVAL cycles. Invariant 7: threshold policy for
            # high-cost domains.
            if n.audit_stable_count >= _STABLE_THRESHOLD:
                continue  # Case C: envelope stable — best fit accepted at noise floor; sentinel re-opens
            if (n.consecutive_sentinel_failures >= _BACKOFF_THRESHOLD
                    and cycle % _BACKOFF_INTERVAL != 0):
                continue  # Case A: sentinel-stable loop
            if var in _novelty_vars and cycle % _NOVELTY_INTERVAL != 0:
                continue  # Case B: vocabulary gap — don't thrash
            needs_audit.append(var)

        # v25: audit budget. Order by tractability, audit up to budget.
        # v28+: audit in topological-then-tractability order. Parents in
        # needs_audit must be audited before their children so that when a
        # child's parent gets re-fit this cycle, the child sees the corrected
        # parent in available_parents. Within the same dependency level,
        # tractability ordering decides priority.
        needs_audit_set = set(needs_audit)

        # Repair agenda: when enabled, represent each needs_audit entry as a
        # RepairAgendaItem with scope/authority metadata for later A*-style
        # triage. Audit decisions are unchanged — this pass is structural only.
        # When repair_agenda_enabled=False (default), no items are created and
        # behavior is identical to the pre-hybrid path.
        if self._repair_agenda_enabled:
            self._repair_agenda.clear()
            for _ra_var in needs_audit:
                _ra_n = self.ledger.vars[_ra_var]
                _ra_scope = tuple(sorted(self.ledger.closure_descendants({_ra_var})))
                _ra_auth = sum(
                    len(self.ledger.vars[_v].certificates)
                    for _v in _ra_scope
                    if _v in self.ledger.vars
                )
                _ra_kind = (
                    "sentinel_failure" if _ra_var in _sentinel_failed_vars else "unknown"
                )
                # #SHORTCUT: priority is negated consequence tier so higher-consequence
                # vars sort first (RepairAgenda.pop() is a min-heap: lower value = more
                # urgent). T2 → -2.0, T1 → -1.0, T0 → 0.0. A*-style cost/benefit
                # weighting is reserved for a later stage.
                _ra_priority = -float(self._consequence_tier(_ra_var))
                if (
                    self._uncertainty_consolidation_mode == "assist"
                    and "repair_priority_bonus" in self._uncertainty_assist_vars.get(_ra_var, set())
                ):
                    _ra_priority -= 0.25
                if (
                    self._authority_strength_mode == "assist"
                    and _ra_var in self._authority_strength_repair_priority_vars
                ):
                    _ra_priority -= 0.10
                self._repair_agenda.push(RepairAgendaItem(
                    cycle=cycle,
                    target_var=_ra_var,
                    failure_kind=_ra_kind,
                    source="needs_audit",
                    covering_regime_id=_ra_n.covered_by_regime_id,
                    covering_composite_id=_ra_n.covered_by_composite_id,
                    scope_vars=_ra_scope,
                    authority_at_risk=_ra_auth,
                    estimated_probe_cost=self.intervention_budget,
                    priority=_ra_priority,
                    payload={
                        "cert_age": _ra_n.full_audits,
                        "consecutive_sentinel_failures": _ra_n.consecutive_sentinel_failures,
                        "status": _ra_n.status,
                    },
                ))

        # Audit order: priority mode uses RepairAgenda item priority (low value = urgent);
        # observe mode uses topological+tractability order (default, behavior unchanged).
        # Priority mode is consequence-tier only (#SHORTCUT — not full A* cost/benefit).
        if self._repair_agenda_enabled and self._repair_agenda_ordering == "priority":
            priority_order = [
                _item.target_var
                for _item in sorted(self._repair_agenda._items, key=lambda i: i.priority)
                if _item.target_var in needs_audit_set
            ]
        else:
            priority_order = self._cost_biased_topo_audit_order(needs_audit_set)
        budget = self.priority_audit_budget

        for i, var in enumerate(priority_order):
            if i >= budget:
                deferred.append(var)
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} audit DEFERRED (budget {budget} exhausted; "
                    f"topological rank {i+1}/{len(priority_order)})"
                )
                continue
            _pre_parents = tuple(self.ledger.vars[var].parents)
            _pre_func = self.ledger.vars[var].func
            _diag_token = None
            if self._diagnostic_audit_observer is not None:
                _diag_token = self._diagnostic_audit_observer.before_audit(
                    self,
                    var,
                    cycle,
                )
            parents, func, score, second, fd = self._full_audit_var(var, cycle)
            sig_changed = self._install_var(var, parents, func, score, second, cycle, fd)
            if self._diagnostic_audit_observer is not None:
                self._diagnostic_audit_observer.after_audit(
                    self,
                    _diag_token,
                    var,
                    cycle,
                    parents,
                    func,
                    sig_changed,
                )
            audited.append(var)
            if var in _sentinel_failed_vars:
                self.local_reaudit_count += 1
                _new_parents = tuple(parents)
                _p_changed = _new_parents != _pre_parents
                _f_changed = func != _pre_func
                if _p_changed and _f_changed:
                    _rshape = "full_change"
                elif _p_changed:
                    _rshape = "parent_change"
                elif _f_changed:
                    _rshape = "func_change"
                else:
                    _rshape = "stable"
                _cert_events.append(RegimeCertEvent(
                    var=var, cert_key="skip", event_type="repaired",
                    delta=_sentinel_failed_vars[var][1],
                    repair_shape=_rshape,
                    cert_age=self.ledger.vars[var].full_audits,
                ))
            if sig_changed:
                drift.append(var)
                self.ledger.record_drift(var, cycle)
                if self._live_set is not None:
                    for _dep in self.ledger.variable_dependents(var):
                        self._live_set.add(_dep)
                # Graded cascade: if this var had a sentinel failure this cycle
                # AND re-audit confirms the fit changed, cascade to descendants.
                # Noisy misses (sig_changed=False) produce no cascade.
                if var in _sentinel_failed_vars:
                    self.signature_changed_count += 1
                    _cascade_reason = _sentinel_failed_vars[var][0]
                    closure = self.ledger.invalidate({var}, cycle, _cascade_reason)
                    self.descendant_cascade_count += len(closure) - 1
                    if self._live_set is not None:
                        self._live_set.update(closure)
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} sentinel confirmed — cascading {len(closure)-1} descendants"
                    )
                    # Genuine change: reset repair-failure counter (the sentinel was right).
                    self._var_repair_failures.pop(var, None)
                    _structural_change_this_cycle = True
                # Oscillation: fit changed for a var that has changed before.
                # Distinguishes repair oscillation (wrong→correct→wrong) from
                # genuine world change (stable→changed once).
                _prev_changes = self._var_sig_changes.get(var, 0)
                self._var_sig_changes[var] = _prev_changes + 1
                if _prev_changes >= 1:
                    self.oscillation_count += 1
            else:
                if var in _sentinel_failed_vars:
                    self.noisy_miss_no_cascade_count += 1
                    # Repair failure: sentinel fired but re-audit found same fit.
                    # Accumulate toward budget escalation.
                    _rf = self._var_repair_failures.get(var, 0) + 1
                    self._var_repair_failures[var] = _rf
                    if _rf >= _REPAIR_FAILURE_ESCALATION_THRESHOLD:
                        _current_budget = self._var_budget_escalation.get(
                            var, self.intervention_budget
                        )
                        _new_budget = min(
                            _current_budget * _BUDGET_ESCALATION_FACTOR,
                            _BUDGET_ESCALATION_CAP,
                        )
                        if _new_budget > _current_budget:
                            self._var_budget_escalation[var] = _new_budget
                            self.budget_escalation_count += 1
                            self.ledger.event_log.append(
                                f"c{cycle}: x{var} audit budget escalated "
                                f"{_current_budget}→{_new_budget} "
                                f"(repair_failures={_rf})"
                            )
                    if _rf % _INERT_RESCREEN_THRESHOLD == 0 and self._inert_vars:
                        _woken: Set[int] = set()
                        for _ix in self._inert_vars:
                            _lo = self.world.predict_var_under_intervention(var, _ix, 0.05)
                            _hi = self.world.predict_var_under_intervention(var, _ix, 0.95)
                            self.total_interventions += 2
                            if abs(_hi - _lo) > DEFAULT_TOLERANCE:
                                _woken.add(_ix)
                        if _woken:
                            self._inert_vars -= _woken
                            if self._live_set is not None:
                                self._live_set.update(_woken)
                            self.ledger.event_log.append(
                                f"c{cycle}: x{var} inert rescreen woke "
                                f"{sorted(_woken)} (repair_failures={_rf})"
                            )
                self._maybe_demote(var, cycle)
                # Envelope stability: did this audit move ε at all?
                # sig_changed == False means same fit; now check if the noise
                # floor itself has shifted. If not AND no OOB cluster, the var
                # has converged — increment toward the Case C exit.
                _n = self.ledger.vars[var]
                if len(_n.envelope.deltas) >= self.envelope_certify_after:
                    _env_updated = _n.envelope.maybe_certify(cycle)
                    if not _env_updated and not _n.envelope.envelope_failing():
                        _n.audit_stable_count += 1
                        if _n.audit_stable_count == _STABLE_THRESHOLD:
                            # First time reaching the threshold: issue noise_floor cert.
                            # Carries ε and audit count as evidence; sentinel re-opens
                            # only on deviation > k×ε (genuine change, not tail noise).
                            _prev = _n.certificates.get("skip")
                            self.ledger.issue_cert(
                                var, "skip", "noise_floor", "guarded_reuse",
                                context_parents=tuple(_n.parents),
                                context_visible=self.world.visible_count,
                                context_cycle=cycle,
                                targets=_prev.targets if _prev else (),
                                substitutions_tested=("envelope_stable",),
                                changes=_n.audit_stable_count,
                                trials=_n.full_audits,
                                earned_by="envelope_stable",
                            )
                            self.ledger.emit(LedgerEvent(
                                type="cert_issued", var=var, cycle=cycle,
                                payload={"role": "noise_floor",
                                         "eps": _n.envelope.certified_eps,
                                         "audits": _n.full_audits},
                            ))
                    else:
                        _n.audit_stable_count = 0
            if self._maybe_novelty(var, score, second, cycle, sig_changed=sig_changed):
                novelty_fired = True

        # DIAGNOSTIC HANDLE v26d: maintain per-var deferral streaks.
        deferred_set = set(deferred)
        _streak_vars = (self._live_set if self._live_set is not None
                        else range(self.world.visible_count))
        for v in _streak_vars:
            if v in deferred_set:
                self.defer_count[v] += 1
                self.defer_streak[v] += 1
                self.max_defer_streak[v] = max(self.max_defer_streak[v], self.defer_streak[v])
            else:
                self.defer_streak[v] = 0

        # Inert rescreen on confirmed structural change: when a sentinel failure
        # led to a genuine fit change this cycle, re-probe every inert var.
        # Cost: 2 probes per inert var, fired at most once per cycle only on
        # confirmed shifts (not on noisy misses or value-only perturbations).
        if _structural_change_this_cycle and self._inert_vars:
            _newly_salient: Set[int] = set()
            for _ix in self._inert_vars:
                _s_lo = self.world.predict_under_intervention(_ix, 0.05)
                _s_hi = self.world.predict_under_intervention(_ix, 0.95)
                self.total_interventions += 2
                for _j in range(self.world.visible_count):
                    if _j != _ix and abs(_s_lo[_j] - _s_hi[_j]) > DEFAULT_TOLERANCE:
                        _newly_salient.add(_ix)
                        break
            if _newly_salient:
                self._inert_vars -= _newly_salient
                if self._live_set is not None:
                    self._live_set.update(_newly_salient)
                self.ledger.event_log.append(
                    f"c{cycle}: structural shift woke inert {sorted(_newly_salient)}"
                )

        # Frontier admission: when no never-audited vars remain in the live set,
        # admit the next highest-priority unknowns (up to frontier_k at a time).
        # "Fresh" = in live_set but full_audits == 0 (not yet looked at once).
        # Condition is much looser than "no unresolved": structural disruptions
        # keep live vars perpetually in uncertain/proposed, so waiting for full
        # resolution would stall admission indefinitely.
        # "Unknown" = never audited, not inert, not already live.
        if self._live_set is not None:
            fresh_in_live = {v for v in self._live_set if self.ledger.vars[v].full_audits == 0}
            if not fresh_in_live:
                unknown = [
                    v for v in range(self.world.visible_count)
                    if v not in self._live_set
                    and self.ledger.vars[v].full_audits == 0
                    and v not in self._inert_vars
                ]
                if unknown:
                    admit = sorted(unknown,
                                   key=lambda v: self._priority_score(v, cycle),
                                   reverse=True)[:self.frontier_k]
                    for v in admit:
                        self._live_set.add(v)
                    self.ledger.event_log.append(
                        f"c{cycle}: frontier admit {[f'x{v}' for v in admit]} "
                        f"({len(unknown)-len(admit)} unknown remaining)"
                    )

        if self._uncertain_this_cycle:
            self._find_joint_trass_candidates(cycle)

        if cycle % _JOINT_SCAN_INTERVAL == 0:
            self._proactive_joint_scan(cycle)

        if cycle % _COMPONENT_PROMOTE_INTERVAL == 0:
            self._promote_dense_components(cycle)

        if _cert_events:
            _n_failed = sum(1 for e in _cert_events if e.event_type == "failed")
            _mean_dev = (sum(e.delta for e in _cert_events) / len(_cert_events))
            _regime_id, _newly_confirmed = self.regime_register.observe(
                _cert_events,
                cycle,
                {"n_failed": float(_n_failed), "mean_delta": _mean_dev},
            )
            if _newly_confirmed and _regime_id is not None:
                self._commission_regime_sentinel(_regime_id)
                if self._context_role_index is not None:
                    members = tuple(sorted({int(e.var) for e in _cert_events}))
                    nid = f"regime:R{_regime_id}:{','.join(map(str, members))}"
                    self._context_role_index.add_or_update_node(NethraNode(
                        nethra_id=nid,
                        kind="regime_handle",
                        target_var=None,
                        components=members,
                        learned_parents=(),
                        learned_func="regime_cofailure",
                        signature=f"R{_regime_id}:{members}",
                        first_seen_cycle=cycle,
                        last_seen_cycle=cycle,
                        observations=len(_cert_events),
                        passive_evidence_count=len(_passive_stressed_vars),
                        active_probe_count=sum(1 for e in _cert_events if e.event_type == "failed"),
                        source="regime",
                    ))
                    self._context_role_index.assign_context_role(ContextRoleRecord(
                        nethra_id=nid,
                        context_key=nethra_context_key(operation="regime", visible=self.world.visible_count),
                        operation="regime",
                        role="unresolved",
                        cycle=cycle,
                        evidence_summary=f"confirmed_regime failed={_n_failed}",
                        validity_scope=members,
                    ))

        # Passive co-stress: if ≥ 2 vars were passive-stressed this cycle, feed
        # them to the regime register as a seeded candidate. Passive evidence
        # may not confirm a regime — only active failures do. seed_only=True
        # adds a candidate without matching; subsequent active failure events
        # can then match this candidate and promote it to confirmed.
        if len(_passive_stressed_vars) >= 2:
            _stressed_events = [
                RegimeCertEvent(
                    var=v, cert_key="skip", event_type="stressed",
                    delta=0.0, cert_age=self.ledger.vars[v].full_audits,
                )
                for v in _passive_stressed_vars
            ]
            self.regime_register.observe(
                _stressed_events, cycle,
                {"passive_stress": float(len(_passive_stressed_vars))},
                seed_only=True,
            )

        self.records.append(CycleRecord(
            cycle=cycle,
            detected_drift_vars=tuple(drift),
            skipped_vars=tuple(skipped),
            fully_audited_vars=tuple(audited),
            novelty_attention=novelty_fired,
            deferred_vars=tuple(deferred),
        ))

    def final_summary(self) -> str:
        """Build the multi-line end-of-run summary string."""
        return SummaryRenderer(RunAnalyzer(self)).render()
