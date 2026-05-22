from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The certification control loop. ChainedAgent owns the full lifecycle.
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
#                           to certified, manages TiedFrontier
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
# What makes certified nethras operative here:
#   available_parents in _full_audit_var is built only from tareth-certified
#   vars. A variable's certification directly controls what hypotheses are
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
# NETHRA: Not a label. A factoring that earned certification by surviving
#   intervention tests in a specific scope. Certified nethras are operative:
#   they become active filters deciding what later evidence counts as tareth
#   or trass. They do not passively describe — they gate future reasoning.
#
# TARETH / TRASS: Provisional verdicts from scope-specific substitution tests.
#   trass  — substituting the distinction leaves monitored targets unchanged
#   tareth — substitution changes monitored targets; a concrete witness exists
#   Certs fire by default; only observed failure or an active dependency event
#   earns revocation. Sentinel failure and downstream contradiction revoke cert
#   authority. Structural or scope changes revoke only when they are themselves
#   dependency events (parent set changed, contradicting evidence in expanded
#   context). The verdict belongs to the scope, not the hypothesis.
#
# FALSE-TRASS: Two locally-trass nethras can jointly be tareth. Composition
#   requires a joint re-test. Local certification does not propagate upward.
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
# This file: certification decisions live here. _install_var must not collapse
#   based on score proximity alone. _certify_operation_role must not promote
#   trass without regime-survival evidence. _collapse_tied_frontier as currently
#   implemented triggers on score-landscape narrowing — that is PREMATURE COLLAPSE
#   and must be replaced with stable_count + distinct_contexts_seen gating (P1.2).
# ════════════════════════════════════════════════════════════════════════════════

import dataclasses
import math
import random
from typing import Callable, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

from .functions import FUNC_LIBRARY
from .world import CausalWorld, HiddenMutation
from .ledger import (ChainedLedger, Compression, CompositeNethra,
                     DEFAULT_TOLERANCE, DormantAlternative, NethraCertificate, Role, TiedFrontier)
from .fit import fit_var
from .sentinels import select_var_sentinels, check_var_sentinels_with_envelope
from .records import CycleRecord, FitDiagnostic

# ── Trass authority thresholds ────────────────────────────────────────────────
# A trass cert suppresses future sentinel monitoring — the strongest authority
# the framework grants. It must require more evidence than any cert that merely
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

# Repair-authority escalation: if a sentinel fires and re-audit returns the same
# fit _REPAIR_FAILURE_ESCALATION_THRESHOLD times, the var's audit budget is
# multiplied by _BUDGET_ESCALATION_FACTOR (capped at _BUDGET_ESCALATION_CAP).
_REPAIR_FAILURE_ESCALATION_THRESHOLD = 3
_BUDGET_ESCALATION_FACTOR            = 4
_BUDGET_ESCALATION_CAP               = 400
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
        self._last_fit_diag: Optional[FitDiagnostic] = None
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

    def _full_audit_var(self, var: int, cycle: int) -> Tuple[Tuple[int, ...], str, int, int]:
        """Run a full hypothesis-space search for one variable. Steps:
          1. Build available_parents set (exclude only cert-excluded candidates)
          2. Call fit_var which enumerates, scores, and ranks hypotheses
          3. Record FitDiagnostic for offline analysis
        Returns (best_parents, best_func, best_score, second_score).
        Increments full_audit_count and total_interventions."""
        self.full_audit_count += 1
        n = self.ledger.vars[var]
        # available_parents: include by default (invariant 50 — route/include unless excluded by cert).
        # Exclusion is per-target: n.route_certs[P] with role "trass" means P was shown not to
        # change the fit winner for this target T. Absent cert → include.
        # Q4: no joint composition test. Individual route certs (when they exist) won't
        # guarantee the combination is non-redundant. Requires predict_under_joint_intervention.
        available = {
            other_var for other_var, other_n in self.ledger.vars.items()
            if other_var != var
            and (n.route_certs.get(other_var) is None or n.route_certs[other_var].role != "trass")
            and (
                other_n.status == "certified"
                or other_n.status == "trass"
                or other_n.role_for("skip") == "trass"  # provisional trass: status="proposed", role="trass"
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
        diag_dict: Dict[str, object] = {
            "cycle": cycle,
            "var": var,
            "status_before": n.status,
            "role_before": n.role_for("skip"),
            "available_parents": tuple(sorted(available)),
        }
        # P1-B: if this var has an active TiedFrontier with separating probes,
        # inject them as forced inclusions so the tie has a chance to resolve.
        _frontier_probes = (
            n.tied_frontier.separating_probes
            if n.tied_frontier is not None and n.tied_frontier.separating_probes
            else None
        )
        result = fit_var(var, self.world, self.rng, budget,
                         n.current_tolerance, available_parents=available, diag=diag_dict,
                         near_tie_margin=self.near_tie_margin,
                         forced_probes=_frontier_probes)
        self.total_interventions += budget
        fd = FitDiagnostic(
            cycle=int(diag_dict["cycle"]),
            var=int(diag_dict["var"]),
            status_before=str(diag_dict["status_before"]),
            role_before=str(diag_dict["role_before"]),
            available_parents=tuple(diag_dict["available_parents"]),
            restricted=bool(diag_dict.get("restricted", False)),
            hypothesis_count=int(diag_dict.get("hypothesis_count", -1)),
            true_parents=tuple(diag_dict.get("true_parents", ())),  
            true_func=str(diag_dict.get("true_func", "?")),
            true_present=bool(diag_dict.get("true_present", False)),
            true_rank=int(diag_dict.get("true_rank", -1)),
            true_score=int(diag_dict.get("true_score", -1)),
            best_score=int(diag_dict.get("best_score", -1)),
            second_score=int(diag_dict.get("second_score", -1)),
            margin=int(diag_dict.get("margin", -1)),
            best_parents=tuple(diag_dict.get("best_parents", ())),
            best_func=str(diag_dict.get("best_func", "?")),
            failure_class=str(diag_dict.get("failure_class", "unknown")),
            probes=tuple(diag_dict.get("probes", ())),
            actuals=tuple(diag_dict.get("actuals", ())),
            pick_preds=tuple(diag_dict.get("pick_preds", ())),
            truth_preds=diag_dict.get("truth_preds"),
            tie_set=diag_dict.get("tie_set", frozenset()),
            near_tie_candidates=tuple(diag_dict.get("near_tie_candidates", ())),
            near_tie_context_key=int(diag_dict.get("near_tie_context_key", 0)),
        )
        self.fit_diagnostics.append(fd)
        self._last_fit_diag = fd
        # Tie-tracking: bump count for this var's tie set if size > 1
        if len(fd.tie_set) > 1:
            self.tie_log.setdefault(var, {})
            self.tie_log[var][fd.tie_set] = self.tie_log[var].get(fd.tie_set, 0) + 1
        # Probe retention cap (default unlimited; --probe-retention K to cap)
        if self.probe_retention_per_var > 0:
            var_diags = [d for d in self.fit_diagnostics if d.var == var]
            if len(var_diags) > self.probe_retention_per_var:
                # Clear oldest var_diag's per-probe arrays to free memory
                oldest = var_diags[-(self.probe_retention_per_var + 1)]
                oldest.probes = ()
                oldest.actuals = ()
                oldest.pick_preds = ()
                oldest.truth_preds = None
        return result

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
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} operation_role: {n.role_for('skip')}->tareth "
                    f"(declared salience target)"
                )
                n.certificates["skip"] = NethraCertificate(
                    operation="skip", role="tareth", authority="skip",
                    context_parents=tuple(n.parents) if n.parents else (),
                    context_visible=self.world.visible_count, context_cycle=cycle,
                    targets=(), substitutions_tested=("declared_salience",),
                    changes=0, trials=0,
                    earned_by="manual_bootstrap",
                )
            return "tareth"

        n_other_visible = sum(1 for j in range(self.world.visible_count) if j != var)
        if n_other_visible < 2:
            if n.role_for("skip") == "untested":
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} operation_role test DEFERRED "
                    f"(only {n_other_visible} other visible vars; need ≥2)"
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
            if abs(iv_val - saved[var]) <= var_tol:
                continue
            n_trials += 1
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
            self.ledger.event_log.append(
                f"c{cycle}: x{var} operation_role: {prev_role}→{role} ({changes}/{n_trials} perturbations propagated)"
            )
        cert = NethraCertificate(
            operation="skip",
            role=role,
            authority="skip" if role == "tareth" else "none",
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
        n.certificates["skip"] = cert
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
            cn = CompositeNethra(
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
            # Deduplicate: replace any existing composite for this member pair.
            self.ledger.composites = [
                c for c in self.ledger.composites
                if set(c.members) != {var_a, var_b}
            ]
            self.ledger.composites.append(cn)

        # Individual certs are left unchanged. xA and xB proved individually trass
        # before this test ran — that evidence is still accurate. The composite
        # cert is the authority for the joint relationship; the individual trass
        # certs remain correct as individual claims. Writing false_trass on the
        # individual certs would conflate individual and joint evidence.
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
        to_remove = []
        for cn in self.ledger.composites:
            a, b = cn.members
            # Activation-scoped: if both members are dormant (not in live_set),
            # the composite interaction has no active consequence path this cycle.
            # Assume passing — no probe needed (invariant 70: polling only when
            # tied to an active consequence path).
            if (self._live_set is not None
                    and a not in self._live_set
                    and b not in self._live_set):
                passing_members.add(a)
                passing_members.add(b)
                continue
            if cn.context_visible != self.world.visible_count:
                to_remove.append(cn)
                continue
            if self.ledger.vars[cn.sentinel_var].role_for("skip") == "trass":
                to_remove.append(cn)
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
                passing_members.add(a)
                passing_members.add(b)
            else:
                to_remove.append(cn)
        for cn in to_remove:
            self.ledger.composites.remove(cn)
            a, b = cn.members
            na, nb = self.ledger.vars[a], self.ledger.vars[b]
            na.invalidate_certs("false_trass_contradiction")
            nb.invalidate_certs("false_trass_contradiction")
            if na.status == "trass":
                na.status = "proposed"
            if nb.status == "trass":
                nb.status = "proposed"
            self.ledger.event_log.append(
                f"c{cycle}: x{a},x{b} composite REVOKED "
                f"(interaction gone at probe ({cn.probe_val_a:.2f},{cn.probe_val_b:.2f})"
                f" → x{cn.sentinel_var})"
            )
        return passing_members

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
                     score: int, second: int, cycle: int) -> bool:
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
        if (self._last_fit_diag is not None
            and self._last_fit_diag.var == var
            and self._last_fit_diag.cycle == cycle):
            tie_set = self._last_fit_diag.tie_set
            near_tie_candidates = self._last_fit_diag.near_tie_candidates
            near_tie_context_key = self._last_fit_diag.near_tie_context_key
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

        n = self.ledger.vars[var]
        if n.first_audited_cycle == 0:
            n.first_audited_cycle = cycle
        n.full_audits += 1
        n.margins.append(margin)

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
            if self._last_fit_diag is not None and self._last_fit_diag.var == var and self._last_fit_diag.cycle == cycle:
                self._last_fit_diag.status_after = n.status
                self._last_fit_diag.role_after = n.role_for("skip")
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
            avail_for_route = {
                other_var for other_var, other_n in self.ledger.vars.items()
                if other_var != var
                and (other_n.status == "certified" or other_n.status == "trass"
                     or other_n.role_for("skip") == "trass"
                     or (other_n.status == "proposed" and bool(other_n.sentinels)))
            }
            self._certify_route_certs(var, new_parents, avail_for_route, cycle)
            # Audit cert: stable fit earned enough observations; mark as reusable.
            n.certificates["audit"] = NethraCertificate(
                operation="audit",
                role="reusable",
                authority="guarded_reuse",
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

        if self._last_fit_diag is not None and self._last_fit_diag.var == var and self._last_fit_diag.cycle == cycle:
            self._last_fit_diag.status_after = n.status
            self._last_fit_diag.role_after = n.role_for("skip")

        # Frontier management: maintain TiedFrontier on the VarNethra.
        # Skipped for trass (already returned early above).
        if len(near_tie_set) >= 2:
            scores_dict = {(p, f): s for p, f, s in near_tie_candidates}
            self._update_tied_frontier(var, cycle, near_tie_set,
                                       scores_dict, near_tie_context_key)
        elif n.tied_frontier is not None:
            winning = next(iter(near_tie_set)) if near_tie_set else None
            self._collapse_tied_frontier(var, winning, cycle)

        return semantic_changed

    def _certify_route_certs(
        self, var: int, parents: Tuple[int, ...], available: Set[int], cycle: int
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
        if (self._last_fit_diag is None
                or self._last_fit_diag.var != var
                or self._last_fit_diag.cycle != cycle):
            return
        diag = self._last_fit_diag
        if not diag.near_tie_candidates:
            return  # clean fit, no competition → invariant 2, nothing earned

        # Build candidate pool: vars in near-tie parents that aren't in winner.
        parents_set = set(parents)
        competing = {
            p
            for cand_parents, _, _ in diag.near_tie_candidates
            for p in cand_parents
            if p not in parents_set and p in available
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
            n.route_certs[p] = NethraCertificate(
                operation="route",
                role=role,
                authority="none" if role == "tareth" else "guarded_reuse",
                context_parents=tuple(parents),
                context_visible=self.world.visible_count,
                context_cycle=cycle,
                targets=(var,),
                substitutions_tested=("counterfactual_fit",),
                changes=0 if same_winner else 1,
                trials=1,
                earned_by="counterfactual_fit",
            )

    def _derive_separating_probes(
        self, var: int, frontier: "TiedFrontier"
    ) -> Tuple[Tuple[int, float], ...]:
        """Derive separating probes from the last FitDiagnostic for `var`.

        Phase 1: use existing audit probes (no new world calls). For each probe
        (iv_var, iv_val), compute the pairwise prediction disagreement across all
        frontier candidates using FUNC_LIBRARY. Retain the top 3 probes by max
        pairwise disagreement.

        Returns a tuple of (iv_var, iv_val) pairs (at most 3).
        """
        if (self._last_fit_diag is None
                or self._last_fit_diag.var != var
                or not self._last_fit_diag.probes):
            return ()
        probes = self._last_fit_diag.probes  # Tuple[Tuple[int, float], ...]
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
    ) -> None:
        """Maintain the TiedFrontier on VarNethra `var`.

        If the context_key matches and the new candidate set is the same as
        the existing frontier, increment stable_count. If the set narrowed,
        archive the dropped candidates to dormant_alternatives and replace
        the frontier. If context changed or no frontier exists, start fresh.
        """
        n = self.ledger.vars[var]
        existing = n.tied_frontier
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
            new_frontier.separating_probes = self._derive_separating_probes(var, new_frontier)
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
            existing.separating_probes = self._derive_separating_probes(var, existing)
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
            new_frontier.separating_probes = self._derive_separating_probes(var, new_frontier)
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
            new_frontier.separating_probes = self._derive_separating_probes(var, new_frontier)
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
            if winning_hyp is not None:
                self.ledger.event_log.append(
                    f"c{cycle}: x{var} frontier collapsed → "
                    f"{winning_hyp[1]}({list(winning_hyp[0])}); "
                    f"{len(n.dormant_alternatives)} candidates archived"
                )
        else:
            # Threshold not met: ambiguity is unresolved; discard without archiving.
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
            parents, func, score, second = self._full_audit_var(var, 0)
            self._install_var(var, parents, func, score, second, 0)
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
        parents, func, score, second = self._full_audit_var(new_var, cycle)
        self._install_var(new_var, parents, func, score, second, cycle)
        self.ledger.event_log.append(
            f"c{cycle}: x{new_var} REVEALED — first audit complete; "
            f"available parents at reveal time: "
            f"{sorted(other for other, n in self.ledger.vars.items() if other != new_var and n.status in ('certified', 'trass', 'proposed'))}"
        )

    def run_cycle(self, mutation: HiddenMutation) -> None:
        """Process one cycle of agent operation. Inputs: the world's mutation
        (purely informational — agent never reads structural fields, only
        cycle index). Steps:

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
        cycle = mutation.cycle
        self._uncertain_this_cycle.clear()


        truth_novelty = any(
            self.world.funcs[i] == "SIN"
            for i in range(self.world.visible_count)
        )

        skipped: List[int] = []
        audited: List[int] = []
        drift: List[int] = []
        deferred: List[int] = []
        novelty_fired = False

        # First pass: cheap paths and queue full audits.
        needs_audit: List[int] = []
        # Graded cascade: sentinel failures in this cycle that have not yet been
        # confirmed by a local re-audit. Value = reason string for the cascade
        # event log. Cascade fires only after _install_var confirms sig_changed.
        _sentinel_failed_vars: Dict[int, str] = {}

        # v28+: process variables in dependency order (parents first) so a
        # parent's sentinel failure invalidates descendants BEFORE they take
        # cheap-path skips based on now-invalid parent assumptions. Previously
        # the loop went in numeric order, which happened to align with
        # topological order in this toy because _random_dag builds low-to-high,
        # but that's an accidental coincidence — intent says topological.
        topo_order = self._topological_order(self.world.visible_count)

        # Composite nethra check: replay each composite's joint probe. Returns
        # the set of vars whose composite is still live. Failing composites are
        # revoked here — their members' certs are reset before the first-pass
        # loop so those vars fall through to the audit queue this cycle.
        composite_passing = self._check_composites(cycle)

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
                                    n.certificates["compress"] = NethraCertificate(
                                        operation="compress",
                                        role="trass",
                                        authority="guarded_reuse",
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
                passed, _, _, reason = check_var_sentinels_with_envelope(
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
                _sentinel_failed_vars[var] = f"sentinel: {reason}"
                self.sentinel_miss_count += 1
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
            parents, func, score, second = self._full_audit_var(var, cycle)
            sig_changed = self._install_var(var, parents, func, score, second, cycle)
            audited.append(var)
            if var in _sentinel_failed_vars:
                self.local_reaudit_count += 1
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
                    _cascade_reason = _sentinel_failed_vars[var]
                    closure = self.ledger.invalidate({var}, cycle, _cascade_reason)
                    self.descendant_cascade_count += len(closure) - 1
                    if self._live_set is not None:
                        self._live_set.update(closure)
                    self.ledger.event_log.append(
                        f"c{cycle}: x{var} sentinel confirmed — cascading {len(closure)-1} descendants"
                    )
                    # Genuine change: reset repair-failure counter (the sentinel was right).
                    self._var_repair_failures.pop(var, None)
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
                            _n.certificates["skip"] = NethraCertificate(
                                operation="skip",
                                role="noise_floor",
                                authority="guarded_reuse",
                                context_parents=tuple(_n.parents),
                                context_visible=self.world.visible_count,
                                context_cycle=cycle,
                                targets=_prev.targets if _prev else (),
                                substitutions_tested=("envelope_stable",),
                                changes=_n.audit_stable_count,
                                trials=_n.full_audits,
                                earned_by="envelope_stable",
                            )
                            self.ledger.event_log.append(
                                f"c{cycle}: x{var} noise_floor certified "
                                f"(ε={_n.envelope.certified_eps:.3f} audits={_n.full_audits})"
                            )
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

        # Frontier admission: when the live set has no unresolved vars, admit
        # the next highest-priority unknowns (up to frontier_k at a time) so the
        # agent keeps making forward progress without ever auditing everything upfront.
        # "Unresolved" = status not yet certified/trass and not dormant-eligible.
        # "Unknown" = never audited, not inert, not already live.
        if self._live_set is not None:
            unresolved = {
                v for v in self._live_set
                if (self.ledger.vars[v].status in ("uncertain", "proposed")
                    and self.ledger.vars[v].role_for("skip") != "trass")
            }
            if not unresolved:
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

        self.records.append(CycleRecord(
            cycle=cycle, truth_kind=mutation.kind,
            truth_rule_changed=mutation.rule_changed,
            truth_affected_var=mutation.affected_var,
            truth_novelty_active=truth_novelty,
            detected_drift_vars=tuple(drift),
            skipped_vars=tuple(skipped),
            fully_audited_vars=tuple(audited),
            novelty_attention=novelty_fired,
            deferred_vars=tuple(deferred),
        ))

    def final_summary(self) -> str:
        """Build the multi-line end-of-run summary string. Covers:
          - per-variable status counts (cert/quar/trass/uncert/prop)
          - authoritative count (vars actually running cheap-path)
          - skip rate and skip breakdown by category
          - total interventions consumed
          - rule-drift detection confusion matrix (TP/FN/FP/TN) at three
            granularities: any-drift, localized, op-relevant
          - novelty-attention confusion matrix
          - per-fit-diagnostic failure-class counts
          - the per-variable display lines

        Truth fields in CycleRecord are used here for offline scoring;
        they were not used by the agent during operation."""
        lines: List[str] = []
        n_cycles = len(self.records)
        n_var = self.world.visible_count
        visible = [self.ledger.vars[i] for i in range(self.world.visible_count)]
        cert = sum(1 for n in visible if n.status == "certified")
        quar = sum(1 for n in visible if n.status == "quarantined")
        uncert = sum(1 for n in visible if n.status == "uncertain")
        prop = sum(1 for n in visible if n.status == "proposed")
        trass = sum(1 for n in visible if n.status == "trass" or n.role_for("skip") == "trass")
        authoritative = sum(1 for n in visible if n.authoritative)

        total_deferred = sum(len(r.deferred_vars) for r in self.records)
        total_var_decisions = self.skip_count + self.full_audit_count + total_deferred
        var_skip_rate = self.skip_count / max(1, total_var_decisions) * 100

        latencies = []
        for i, r in enumerate(self.records):
            if r.truth_rule_changed and r.truth_affected_var >= 0:
                lat = -1
                for j in range(i, len(self.records)):
                    if r.truth_affected_var in self.records[j].detected_drift_vars:
                        lat = j - i
                        break
                latencies.append(lat)
        det = [l for l in latencies if l >= 0]
        mean_lat = sum(det) / max(1, len(det))
        max_lat = max(latencies, default=0)
        undet = sum(1 for l in latencies if l < 0)

        # HYGIENE FIX 3 (v24): split drift metrics into three distinct measures.
        # Old single confusion matrix conflated "any var drifted" with
        # "the actually-affected var was identified."

        # 3a) ANY drift detected (loose — anything changed signature)
        any_tp = any_fn = any_fp = any_tn = 0
        for r in self.records:
            d = bool(r.detected_drift_vars)
            if r.truth_rule_changed and d: any_tp += 1
            elif r.truth_rule_changed and not d: any_fn += 1
            elif not r.truth_rule_changed and d: any_fp += 1
            else: any_tn += 1

        # 3b) LOCALIZED drift — the affected variable was specifically detected
        loc_tp = loc_fn = loc_fp = loc_tn = 0
        for r in self.records:
            localized = (r.truth_rule_changed and r.truth_affected_var >= 0
                         and r.truth_affected_var in r.detected_drift_vars)
            if r.truth_rule_changed and localized: loc_tp += 1
            elif r.truth_rule_changed and not localized: loc_fn += 1
            elif not r.truth_rule_changed and bool(r.detected_drift_vars): loc_fp += 1
            else: loc_tn += 1

        # 3c) OPERATIONALLY-RELEVANT drift — affected var was in tareth set
        # (drift on trass variables doesn't matter for the operation)
        op_tp = op_fn = op_fp = op_tn = 0
        for r in self.records:
            affected_role = (self.ledger.vars[r.truth_affected_var].role_for("skip")
                             if r.truth_affected_var >= 0 else "trass")
            relevant = r.truth_rule_changed and affected_role == "tareth"
            localized = relevant and r.truth_affected_var in r.detected_drift_vars
            if relevant and localized: op_tp += 1
            elif relevant and not localized: op_fn += 1
            elif not relevant and bool(r.detected_drift_vars): op_fp += 1
            else: op_tn += 1

        ntp = nfn = nfp = ntn = 0
        for r in self.records:
            if r.truth_novelty_active and r.novelty_attention: ntp += 1
            elif r.truth_novelty_active and not r.novelty_attention: nfn += 1
            elif not r.truth_novelty_active and r.novelty_attention: nfp += 1
            else: ntn += 1

        nov_open = sum(1 for n in self.ledger.novelty if n.status == "open")
        nov_resolved = sum(1 for n in self.ledger.novelty if n.status == "resolved")
        #nov_voc = sum(1 for n in self.ledger.novelty if n.kind == "vocabulary") - never used

        # compression metrics — both live and lifetime per HYGIENE FIX 2
        total_comps = sum(len(n.compressions) for n in self.ledger.vars.values())
        total_comp_hits = sum(n.compression_hits for n in self.ledger.vars.values())
        total_comp_misses = sum(n.compression_misses for n in self.ledger.vars.values())
        total_comp_hits_lifetime = sum(n.compression_hits_lifetime for n in self.ledger.vars.values())
        total_comp_misses_lifetime = sum(n.compression_misses_lifetime for n in self.ledger.vars.values())

        # envelope metrics
        certified_envs = sum(1 for n in self.ledger.vars.values() if n.envelope.certified_eps > 0)
        total_oob = sum(n.envelope.out_of_band_count for n in self.ledger.vars.values())

        # temporal trass: how often did cost-based muting fire
        total_muted = sum(len(n.temporal_trass_log) for n in self.ledger.vars.values())
        muted_low = sum(1 for n in self.ledger.vars.values()
                        for e in n.temporal_trass_log if e.reason == "low_cost_dismissed")
        muted_outlier = sum(1 for n in self.ledger.vars.values()
                            for e in n.temporal_trass_log if e.reason == "outlier_within_tolerance")

        lines.append("\n── summary ────────────────────────────────────────────")
        lines.append(f"  cyc={n_cycles} vars={n_var} | "
                     f"st: cert={cert} prop={prop} unc={uncert} quar={quar} trass={trass} | "
                     f"auth={authoritative}/{n_var}")
        lines.append(f"  nov: {len(self.ledger.novelty)} (open={nov_open} res={nov_resolved})")
        lines.append(f"  comp: stored={total_comps} hit/miss live={total_comp_hits}/{total_comp_misses} "
                     f"life={total_comp_hits_lifetime}/{total_comp_misses_lifetime}")
        lines.append(f"  env: cert={certified_envs}/{n_var} oob={total_oob}")
        lines.append(f"  mute: total={total_muted} low={muted_low} outlier={muted_outlier}")
        lines.append(f"  audit: full={self.full_audit_count} skip={self.skip_count}/{total_var_decisions} "
                     f"({var_skip_rate:.1f}%) | trass={self.trass_skip_count} "
                     f"comp={self.compression_skip_count} sent={self.sentinel_skip_count}")
        lines.append(f"  iv={self.total_interventions}")
        lines.append(f"  drift any:    TP={any_tp} FN={any_fn} FP={any_fp} TN={any_tn} | "
                     f"latency mean={mean_lat:.2f} max={max_lat} undet={undet}/{len(latencies)}")
        lines.append(f"  drift loc:    TP={loc_tp} FN={loc_fn} FP={loc_fp} TN={loc_tn}")
        lines.append(f"  drift op:     TP={op_tp} FN={op_fn} FP={op_fp} TN={op_tn}")
        lines.append(f"  novelty att:  TP={ntp} FN={nfn} FP={nfp} TN={ntn}")

        total_deferred = sum(len(r.deferred_vars) for r in self.records)
        truth_deferred = sum(
            1 for r in self.records
            if r.truth_rule_changed and r.truth_affected_var >= 0 and r.truth_affected_var in r.deferred_vars
        )
        op_truth_deferred = sum(
            1 for r in self.records
            if r.truth_rule_changed and r.truth_affected_var >= 0
            and self.ledger.vars[r.truth_affected_var].role_for("skip") == "tareth"
            and r.truth_affected_var in r.deferred_vars
        )
        worst_deferred = sorted(self.defer_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
        worst_def_str = ", ".join(
            f"x{v}:n={c},max={self.max_defer_streak[v]}" for v, c in worst_deferred if c > 0
        ) or "none"
        lines.append(f"  defer: total={total_deferred} truth={truth_deferred} op={op_truth_deferred} "
                     f"worst={worst_def_str}")

        watch_count = sum(1 for n in visible if n.in_watch_state)
        lambda_vals = [n.poisson_rate for n in visible
                       if n.role_for("skip") != "trass" and n.poisson_rate > 0
                       and n.first_audited_cycle > 0]
        avg_lam = sum(lambda_vals) / len(lambda_vals) if lambda_vals else 0.0
        watch_queued = sum(1 for e in self.ledger.event_log if "in watch state)" in e)
        lines.append(f"  predict: watch={watch_count}/{n_var} λ_tracked={len(lambda_vals)} "
                     f"avg_λ={avg_lam:.3f} queued={watch_queued}")

        cert_cycles = sorted(
            n.first_certified_cycle for n in visible
            if n.first_certified_cycle > 0 and n.role_for("skip") != "trass"
        )
        if cert_cycles:
            p50 = cert_cycles[len(cert_cycles) // 2]
            lines.append(f"  coverage: n={len(cert_cycles)} "
                         f"first={cert_cycles[0]} p50={p50} last={cert_cycles[-1]}")

        from collections import Counter, defaultdict
        fit_class_counts = Counter(d.failure_class for d in self.fit_diagnostics)
        class_str = " ".join(f"{k}={v}" for k, v in fit_class_counts.most_common()) or "none"
        true_missing = sum(1 for d in self.fit_diagnostics if not d.true_present)
        true_present = sum(1 for d in self.fit_diagnostics if d.true_present)
        true_rank1 = sum(1 for d in self.fit_diagnostics if d.true_rank == 1)
        restricted_fits = sum(1 for d in self.fit_diagnostics if d.restricted)
        lines.append(f"  fit: aud={len(self.fit_diagnostics)} restr={restricted_fits} "
                     f"present={true_present} missing={true_missing} r1={true_rank1}")
        lines.append(f"    classes: {class_str}")

        by_var = defaultdict(list)
        for d in self.fit_diagnostics:
            by_var[d.var].append(d)
        rows = []
        for v, ds in by_var.items():
            latest = ds[-1]
            miss = sum(1 for d in ds if not d.true_present)
            rank1 = sum(1 for d in ds if d.true_rank == 1)
            common_class = Counter(d.failure_class for d in ds).most_common(1)[0][0]
            rows.append((len(ds), v, miss, rank1, common_class, latest.status_after, latest.role_after, latest.true_rank, latest.margin))
        rows.sort(reverse=True)
        top_rows = rows[:6]
        if top_rows:
            lines.append("    worst: " + " | ".join(
                f"x{v}:a={aud} m={miss} r1={rank1} {klass[:10]} {status[:4]}/{role[:4]} lr={rank} lm={margin}"
                for aud, v, miss, rank1, klass, status, role, rank, margin in top_rows
            ))

        # Tie tracking summary: which vars have stable tie sets vs transient.
        # A "stable" tie set is one that recurred multiple times — those are
        # candidates for equivalence-class compression (extension-derived).
        if self.tie_log:
            tie_summary = []
            for v, sets in sorted(self.tie_log.items()):
                # Most frequent tie set per var
                top_set, top_count = max(sets.items(), key=lambda kv: kv[1])
                size = len(top_set)
                tie_summary.append(f"x{v}:|set|={size} ×{top_count}")
            lines.append(f"  tie sets (top per var): {' '.join(tie_summary[:8])}"
                         + (f" ... +{len(tie_summary) - 8} more" if len(tie_summary) > 8 else ""))

        # ── compression amortization assessment ──
        # Framework intent: compressions cache cheap predictions under gate
        # conditions, amortizing audit cost over many cycles. The architecture
        # promises that vars with stable parents/funcs will accrue compressions
        # and reduce per-cycle work. This block reports whether that promise
        # is being kept on this run.
        lines.append("\n── compression amortization ───────────────────────────")
        eligible = [n for n in visible
                    if bool(n.parents) and n.role_for("skip") == "tareth"
                    and n.status in ("certified", "proposed") and n.sentinels]
        n_elig = len(eligible)
        n_with_comps = sum(1 for n in eligible if n.compressions)
        # Cost saved: each compression hit replaced a sentinel check
        # (sentinel_count interventions). Cost paid: each discovery ran
        # compression_discovery_budget * 3 anchors * up-to-len(parents) gates.
        cost_per_hit_saved = self.sentinel_count
        # Discovery cost upper bound — the actual sample count is internal
        # to _discover_compressions; we estimate from settings.
        est_disc_cost = self.compression_discovery_budget * 3
        total_disc_runs = sum(1 for n in visible if n.compression_hits_lifetime > 0
                              or n.compression_misses_lifetime > 0)
        est_disc_total = total_disc_runs * est_disc_cost * max(1, sum(len(n.parents) for n in eligible) // max(1, n_elig))
        hits_saved = total_comp_hits_lifetime * cost_per_hit_saved
        amort = "n/a"
        if est_disc_total > 0:
            amort = f"{hits_saved / est_disc_total:.2f}x"
        lines.append(f"  eligible vars (tareth+committed+has-parents): {n_elig}/{n_var}")
        lines.append(f"  with compressions: {n_with_comps}/{n_elig}")
        lines.append(f"  hits lifetime: {total_comp_hits_lifetime} | misses lifetime: {total_comp_misses_lifetime}")
        lines.append(f"  est cost saved by hits: {hits_saved} iv | "
                     f"est discovery cost: {est_disc_total} iv | amortization: {amort}")
        return "\n".join(lines)
