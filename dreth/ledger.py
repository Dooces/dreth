from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# All certified state lives here. The data structures, not the logic.
#
# Active objects:
#   NoiseEnvelope   — certifies ε for one variable. The envelope IS the
#                     variable's current judgment of what counts as a real
#                     deviation vs. noise. All match/fail comparisons use it.
#   VarNethra       — per-variable certification handle. operation_role gates
#                     whether this variable enters downstream hypothesis spaces.
#                     status + sentinels gate whether the cheap path is open.
#                     It is operative: it changes what the agent considers next.
#   TiedFrontier    — ambiguity object. Records near-tied candidates that are
#                     currently indistinguishable under the current probe set.
#                     It accumulates candidates but does not yet drive any
#                     decisions. Morphology only until separating probes and
#                     regime-survival tracking are added (TODO P1.2).
#   Compression     — memoized prediction for a specific gate condition.
#   NoveltyNethra   — records that the hypothesis library appears insufficient.
#   ChainedLedger   — owns all VarNethras, novelty log, event log.
#
# Vestigial or incomplete:
#   TiedFrontier.separating_probes — field exists, always empty. Not yet
#                     generated or consumed. Placeholder for P3.
#   dormant_alternatives — candidates archive on collapse, but nothing reads
#                     or revives them yet (TODO P6.2).
#   TemporalTrassEntry — active: audit trail for dismissed deviations. Used
#                     in the cost-dispatch sentinel check path.
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
# This file: VarNethra stores PROVISIONAL certification state. tied_frontier
#   is the ambiguity object — it must persist until evidence justifies collapse,
#   not until score-landscape narrowing happens to produce a single candidate.
# ════════════════════════════════════════════════════════════════════════════════

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal, Optional, Sequence, Set, Tuple

from .functions import State

DEFAULT_TOLERANCE = 0.1   # bootstrap tolerance before any envelope certifies

def values_match(a: float, b: float, tolerance: float) -> bool:
    """True if |a-b| ≤ tolerance. The single comparison primitive used
    everywhere — fits, sentinels, compressions, op_role tests."""
    return abs(a - b) <= tolerance


# ── NETHRA CERTIFICATE ───────────────────────────────────────────────────────

Operation = Literal["skip", "route", "compress", "audit", "reexamine"]
# NARETH DIVERGENCE (Q3 — two layers):
# Layer 1 — code taxonomy vs conceptual taxonomy:
#   Conceptual (DRETH_TAXONOMY.md): observe / skip / route / compress / audit-reuse
#   Code (here):                    skip / route / compress / audit / reexamine
#   "observe" absent (correct — passive, not shortcut-authorizing).
#   "audit-reuse" → "audit" in code; "reexamine" added (not in conceptual taxonomy).
#   These are not the same taxonomy. Treat them separately when reasoning about certs.
# Layer 2 — live certification status per code operation:
#   skip:      LIVE — _certify_operation_role issues certificates["skip"]
#   route:     PARTIAL — _certify_form_role issues form.form_certificates["route"] at
#              form level only; no instance-level route cert exists
#   compress:  NOT LIVE — _discover_compressions uses pred_passes frequency count only;
#              certificates["compress"] is never populated (invalidate_certs clears it
#              but nothing writes it)
#   audit:     NOT LIVE — declared, cleared by invalidate_certs, never populated
#   reexamine: NOT LIVE — same
# Ought to: _discover_compressions should issue a certificates["compress"] cert scoped
# to the gate predicate, with witnesses from the observed (state, value) pairs, so that
# invalidate_certs("sentinel_failure") clears a real cert rather than a no-op.
# untested:    no substitution test has run yet, OR evidence is stale and retest required
# tareth:      tested; substitution propagates; distinction is load-bearing
# trass:       tested; substitution does not propagate; collapse allowed
# false_trass: locally trass but jointly tareth with another var; composition invalidated
Role = Literal["tareth", "trass", "untested", "false_trass"]
Authority = Literal["none", "prefer", "guarded_reuse", "skip", "propagate"]

@dataclass
class NethraCertificate:
    """A certified claim scoped to a named operation. Carries the context under
    which it was tested, the scope of what was checked, and the evidence counts.
    Role and authority are provisional — the cert fires by default; invalidation
    requires observed failure or an active dependency event (parent set changed,
    sentinel contradiction, composite revoked). Structural context alone (e.g.
    new variable visible) does not revoke unless it is itself a dependency event.
    See DRETH_TAXONOMY.md for full semantics."""
    operation: Operation
    role: Role
    authority: Authority
    context_parents: Tuple[int, ...]   # parent tuple of this var at cert time
    context_visible: int               # visible_count at cert time
    context_cycle: int                 # cycle when certified
    targets: Tuple[int, ...]           # vars actually tested for downstream change (filtered set)
    substitutions_tested: Tuple[str, ...]  # what was swapped
    changes: int                       # how many substitutions propagated
    trials: int                        # total substitutions tested
    joint_members: Optional[Tuple[int, ...]] = None
    joint_R0: Optional[float] = None
    joint_RA: Optional[float] = None
    joint_RB: Optional[float] = None
    joint_RAB: Optional[float] = None
    witnesses: Tuple = ()
    # For skip certs: Tuple of (state_snapshot: Tuple[float,...], iv_val: float) pairs.
    # Each pair is a specific (world context, intervention) that produced a propagation
    # during _certify_operation_role. The sentinel path replays these each cycle.
    # For compress certs: Tuple of one (state_snapshot: Tuple[float,...], simplified_value: float).
    # Trass certs carry no witnesses (no intervention that produced change to store).


# ── PER-VARIABLE NETHRA ───────────────────────────────────────────────────────

@dataclass
class NoiseEnvelope:
    """Empirical noise model for one variable. Tracks the running list of
    observed |predicted - actual| deltas and certifies an ε such that ~95%
    of recent deltas fall within it. The envelope IS the variable's tolerance.

    Re-examinable: when out-of-band events cluster recently (envelope_failing),
    the envelope is treated as drifted and re-certifies on next audit cycle.

    Fields:
      deltas:               last 200 observed |pred-actual| values
      certified_eps:        currently-certified ε; 0 if not yet certified
      certified_at_cycle:   cycle at which current ε was certified
      samples_at_cert:      delta count at that certification (for diagnostics)
      out_of_band_count:    deltas exceeding ε since current certification
      out_of_band_log:      (cycle, delta) pairs for recent OOB events
    """
    deltas: List[float] = field(default_factory=list)
    certified_eps: float = 0.0
    certified_at_cycle: int = 0
    samples_at_cert: int = 0
    out_of_band_count: int = 0
    out_of_band_log: List[Tuple[int, float]] = field(default_factory=list)

    def add_delta(self, delta: float, cycle: int) -> bool:
        """Record one observed deviation. Returns True if within current ε
        (or if not yet certified). False if out-of-band, in which case the
        cycle and magnitude are logged. Maintains a rolling window of 200
        deltas and 50 OOB events."""
        abs_delta = abs(delta)
        self.deltas.append(abs_delta)
        if len(self.deltas) > 200:
            self.deltas = self.deltas[-200:]
        if self.certified_eps > 0:
            within = abs_delta <= self.certified_eps
            if not within:
                self.out_of_band_count += 1
                self.out_of_band_log.append((cycle, abs_delta))
                if len(self.out_of_band_log) > 50:
                    self.out_of_band_log = self.out_of_band_log[-50:]
            return within
        return True  # not yet certified, accept anything

    def maybe_certify(self, cycle: int, min_samples: int = 20, percentile: float = 0.95) -> bool:
        """Certify a new ε if ≥min_samples have accumulated AND the new
        percentile-of-deltas is substantially different (>20%) from current ε.
        Resets OOB tracking on new certification because old OOB events were
        measured against the old ε and don't carry semantic meaning under
        the new ε. Returns True if ε was updated."""
        if len(self.deltas) < min_samples:
            return False
        sorted_d = sorted(self.deltas)
        idx = min(int(percentile * len(sorted_d)), len(sorted_d) - 1)
        new_eps = sorted_d[idx]
        # Only update if substantially different from current certification
        if self.certified_eps == 0 or abs(new_eps - self.certified_eps) / max(self.certified_eps, 1e-6) > 0.2:
            self.certified_eps = new_eps
            self.certified_at_cycle = cycle
            self.samples_at_cert = len(self.deltas)
            # v28+: re-certifying means tolerance changed. Old OOB events
            # were measured against old ε and don't carry semantic meaning
            # under the new ε. Reset OOB tracking — fresh start for the new
            # envelope.
            self.out_of_band_count = 0
            self.out_of_band_log = []
            return True
        return False

    def envelope_failing(self, threshold_count: int = 5, recent_window: int = 30) -> bool:
        """True if the envelope has been broken at least `threshold_count` times
        within the last `recent_window` cycles. Used by cost-weighted dispatch
        to decide whether to escalate persistent failures. Returns False if
        ε not yet certified or no OOB events recorded."""
        if self.certified_eps == 0: return False
        if not self.out_of_band_log: return False
        latest_cycle = self.out_of_band_log[-1][0]
        recent_oob = [c for c, _ in self.out_of_band_log if c >= latest_cycle - recent_window]
        return len(recent_oob) >= threshold_count

    def display(self) -> str:
        """One-line summary for per-variable status display."""
        return (f"ε={self.certified_eps:.3f} samples={len(self.deltas)} "
                f"oob={self.out_of_band_count}")


@dataclass
class Compression:
    """A cached simplified prediction for a variable, valid only when a gate
    condition holds. The gate is a list of (var_idx, target_value, tolerance)
    triples — when ALL listed variables are within tolerance of their target,
    the compression's simplified_value is used as the prediction (cheap path).
    Otherwise the fit's full computation runs.

    Discovered by `_discover_compressions` after a fit has been stable enough
    to be trusted. Cleared when the fit's signature changes.

    Fields:
      gate:                  conjunction of (var, target, tol) conditions
      simplified_value:      cached prediction when gate matches
      certified_equivalence: number of samples that confirmed equivalence
      discovery_cycle:       when this compression was added
    """
    gate: Tuple[Tuple[int, float, float], ...]
    simplified_value: float
    certified_equivalence: int
    discovery_cycle: int
    pred_passes: int = 0

    def gate_matches(self, state: State) -> bool:
        """True if every gate condition holds in the given state. Single AND
        of |state[var_idx] - target| ≤ tol over all gate entries."""
        for var_idx, target, tol in self.gate:
            if abs(state[var_idx] - target) > tol:
                return False
        return True

    def display(self) -> str:
        """One-line summary for the compression list display."""
        gate_str = ",".join(f"x{v}≈{t:.2f}±{tol:.2f}" for v, t, tol in self.gate)
        status = f"pp={self.pred_passes}"
        return f"if [{gate_str}] → {self.simplified_value:.3f} [{status}]"


@dataclass
class CompositeNethra:
    """Joint interaction cert earned by two individually-trass vars.

    Two vars A and B jointly change a monitored sentinel var j, but neither
    does alone. The cert authorizes a composite-skip shortcut: each cycle,
    replay the stored (probe_val_a, probe_val_b) joint intervention and check
    whether |RAB[j] - R0[j]| > tol. If yes: interaction still present, both
    vars skip. If no: revoke this cert, reset both vars to untested.

    Scope: context_visible encodes what was visible at cert time. If visible
    count changes, the scope is stale and the cert is revoked before the next
    check so the joint test can re-run in the expanded scope.

    Fields:
      members:           (var_a, var_b) — the two jointly-tareth vars
      sentinel_var:      downstream j that showed the interaction
      probe_val_a:       intervention value for var_a used as sentinel probe
      probe_val_b:       intervention value for var_b used as sentinel probe
      tol:               tolerance for comparison (sentinel_var's tol at cert time)
      changes:           interaction_trials at certification
      trials:            total_trials at certification
      certified_at_cycle:
      context_visible:   visible_count at cert time
    """
    members: Tuple[int, int]
    sentinel_var: int
    probe_val_a: float
    probe_val_b: float
    tol: float
    changes: int
    trials: int
    certified_at_cycle: int
    context_visible: int


@dataclass
class TemporalTrassEntry:
    """One audit-log entry recording a deliberately-muted attention event.
    When cost-weighted dispatch chooses NOT to escalate a deviation (because
    the variable's cost weight is low or the deviation is within tolerance),
    this entry records what was muted and why. The record is the audit trail
    for later credit assignment if downstream operations show the dismissal
    mattered.

    Fields:
      cycle:        when the mute was applied
      var:          which variable was muted
      delta:        magnitude of the dismissed deviation
      cost_weight:  the variable's cost weight at mute time
      reason:       category — low_cost_dismissed, low_cost_dismissed_persistent,
                    or outlier_within_tolerance
    """
    cycle: int
    var: int
    delta: float
    cost_weight: float
    reason: str


@dataclass
class TiedFrontier:
    """Near-tied hypothesis constellation for one variable. Captures all
    candidates within `near_tie_margin` probes of the best score.

    A frontier persists until either:
    - One candidate pulls decisively ahead (drops below threshold)
    - The available-parents context changes (context_key shifts)
    - A signature change with parents_changed=True (structural shift)

    When collapsed, the losing candidates move to `dormant_alternatives`
    on the VarNethra. Archived there as (parents, func, last_score) triples.

    Fields:
      candidates:        frozenset of (parents, func) in the near-tie constellation
      scores:            last-seen score per candidate
      margin:            near_tie_margin used when this frontier was built
      context_key:       hash(frozenset(available_parents)) at creation;
                         stale when context changes and frontier should reset
      collapse_sig:      (parents, func) that collapsed the frontier, or None
      separating_probes: (iv_var, iv_val) pairs that discriminate members best
      first_seen_cycle:  cycle when this frontier was first established
      last_seen_cycle:   cycle of most recent audit that confirmed/updated it
      stable_count:      consecutive audits where the same candidate set returned
    """
    candidates: FrozenSet[Tuple[Tuple[int, ...], str]]
    scores: Dict[Tuple[Tuple[int, ...], str], int]
    margin: int
    context_key: int
    collapse_sig: Optional[Tuple[Tuple[int, ...], str]]
    separating_probes: Tuple[Tuple[int, float], ...]
    first_seen_cycle: int
    last_seen_cycle: int
    stable_count: int = 1


@dataclass
class VarNethra:
    """Per-variable handle. Holds the agent's current hypothesis, its
    operational role, the supporting sentinels, the noise envelope, any
    discovered compressions, and the audit trails.

    Status (lifecycle):
      proposed:    has a current fit but not yet promoted; uses cheap-path
                   if sentinels attached
      certified:   stable fit confirmed by promote_after consecutive matching
                   audits (informational confidence label)
      quarantined: legacy state from old margin-gate logic; not used in v28+
      uncertain:   was certified, sentinel failed → demoted; needs re-audit
      trass:       operation_role test concluded var doesn't matter; collapsed

    Operation role:
      untested:    not yet decided (deferred when too few visible peers)
      tareth:      perturbing this var changes other vars beyond their tolerance
      trass:       perturbing this var has no effect on other vars within tol

    Fields:
      var, parents, func:       current fit (parents tuple, function name)
      status, operation_role:   the two lifecycle dimensions above
      strong_observations:      consecutive cycles same fit was returned
      sentinels:                list of (intervention_var, intervention_val)
                                probes used for cheap-path validation
      expected_outcomes:        legacy field; sentinel check now computes
                                expected from current state
      margins:                  diagnostic record of (best - second) score gaps
      skip_count, full_audits:  per-var work counters
      last_changed_cycle:       last cycle the fit signature changed
      collapse_log:             history of invalidations affecting this var
      compressions:             list of currently-valid Compressions
      compression_hits/misses:  live counters since last compression clear
      compression_*_lifetime:   counters across all compression generations
      cost_weight:              attention weight for cost-dispatch (default 1)
      envelope:                 NoiseEnvelope object for this variable
      temporal_trass_log:       audit trail of dismissed deviations
    """
    var: int
    parents: Tuple[int, ...] = field(default_factory=tuple)
    func: str = "LOW"
    status: str = "proposed"
    strong_observations: int = 0
    sentinels: List[Tuple[int, float]] = field(default_factory=list)
    expected_outcomes: List[float] = field(default_factory=list)
    margins: List[int] = field(default_factory=list)
    skip_count: int = 0
    full_audits: int = 0
    last_changed_cycle: int = 0
    collapse_log: List[str] = field(default_factory=list)
    compressions: List[Compression] = field(default_factory=list)
    compression_hits: int = 0
    compression_misses: int = 0
    cost_weight: float = 1.0
    envelope: NoiseEnvelope = field(default_factory=NoiseEnvelope)
    temporal_trass_log: List[TemporalTrassEntry] = field(default_factory=list)
    # HYGIENE FIX 2 (v24): split compression counters.
    # compression_hits / misses are LIVE — reset when compressions are cleared
    # (signature change). compression_hits_lifetime / misses_lifetime accumulate
    # across all compression generations for the variable.
    compression_hits_lifetime: int = 0
    compression_misses_lifetime: int = 0
    # Predictive failure fields (prospective reliability modeling)
    drift_cycles: List[int] = field(default_factory=list)
    poisson_rate: float = 0.0          # diagnostic display only; not used for horizon
    first_audited_cycle: int = 0
    first_certified_cycle: int = 0    # cycle when status first reached "certified"
    stability_horizon: int = 9999
    in_watch_state: bool = False
    _watch_propagated: bool = False    # True after children queued for current watch episode
    median_interval: int = 0           # cached median inter-invalidation interval
    watch_threshold_cycles: int = 0    # cached: int(median_interval * 0.3), min 1
    # Near-tie frontier: hypothesis constellation that is operationally
    # equivalent under the current context. Cleared on structural reset
    # (parents_changed) and on invalidation. Losing candidates archived to
    # dormant_alternatives when the frontier collapses.
    tied_frontier: Optional["TiedFrontier"] = None
    dormant_alternatives: List[Tuple[Tuple[int, ...], str, int]] = field(default_factory=list)
    # Per-operation certificates. Keys are Operation literals ("skip", "route", etc.).
    # Populated by _certify_operation_role (skip) and form cert functions (route).
    # During migration, falls back to legacy operation_role if key absent.
    certificates: Dict[str, NethraCertificate] = field(default_factory=dict)

    def role_for(self, operation: Operation) -> Role:
        """Return the certified role for a named operation, or 'untested' if
        no cert exists for this operation yet."""
        if operation in self.certificates:
            return self.certificates[operation].role
        return "untested"

    def authority_for(self, operation: Operation) -> Authority:
        if operation in self.certificates:
            return self.certificates[operation].authority
        return "none"

    def invalidate_certs(self, event: str) -> None:
        """Invalidate per-operation certificates based on the event type.
        Named invalidate_certs (not invalidate) to avoid collision with
        ChainedLedger.invalidate(vars, cycle, reason) which cascades status
        through descendants — a completely separate operation.

        event: "parent_change" | "sentinel_failure" | "structural_mutation" |
               "drift" | "false_trass_contradiction"

        Cascade logic:
          parent_change:
            - clears predict/compress/audit certs (function operating on new inputs)
            - FLAGS skip cert as untested — skip cert validity depends on downstream
              propagation structure, not this var's own parents. No infrastructure
              exists to verify downstream is unchanged, so flag for retest.
              Do NOT add keep_skip_cert parameter — it would never be called True.

          sentinel_failure | false_trass_contradiction:
            - clears audit/compress certs
            - FLAGS skip cert as untested (world changed or contradiction detected)

          structural_mutation:
            - clears ALL certs. Graph changed. Nothing is valid.

          drift:
            - FLAGS all certs as untested. Preserve records, require retest.
        """
        if event == "structural_mutation":
            self.certificates.clear()
            return
        if event == "parent_change":
            self.certificates.pop("predict", None)
            self.certificates.pop("compress", None)
            self.certificates.pop("audit", None)
            if "skip" in self.certificates:
                # Parent change is an active dependency event — cert was scoped
                # to the old parent set; the old evidence no longer applies.
                self.certificates["skip"] = dataclasses.replace(
                    self.certificates["skip"], role="untested"
                )
        if event in ("sentinel_failure", "false_trass_contradiction"):
            self.certificates.pop("audit", None)
            self.certificates.pop("compress", None)
            if "skip" in self.certificates:
                # Observed failure (sentinel) or composite contradiction earns revocation.
                self.certificates["skip"] = dataclasses.replace(
                    self.certificates["skip"], role="untested"
                )
        if event == "drift":
            for op in list(self.certificates):
                self.certificates[op] = dataclasses.replace(
                    self.certificates[op], role="untested"
                )

    @property
    def authoritative(self) -> bool:
        """True if this variable can run cheap-path (sentinel skip) right now.
        Requires: tareth for skip, sentinels attached, status not collapsed."""
        return (
            self.role_for("skip") == "tareth"
            and bool(self.sentinels)
            and self.status in ("certified", "proposed")
        )

    @property
    def current_tolerance(self) -> float:
        """The variable's current matching threshold. Returns the certified ε
        if envelope has certified, else DEFAULT_TOLERANCE."""
        return self.envelope.certified_eps if self.envelope.certified_eps > 0 else DEFAULT_TOLERANCE

    def display(self) -> str:
        """One-line per-variable summary printed at end of run. Marks sentinels
        as STALE if status indicates the variable shouldn't be using them."""
        sent_label = f"sent={len(self.sentinels)}"
        if self.status == "quarantined" and self.sentinels:
            sent_label = f"sent={len(self.sentinels)}/STALE"
        watch_tag = " WATCH" if self.in_watch_state else ""
        horiz = f"{self.stability_horizon}" if self.stability_horizon < 9999 else "?"
        return (f"x{self.var}={self.func}({list(self.parents)}) "
                f"[{self.status}|{self.role_for('skip')}] cost={self.cost_weight:.1f} "
                f"strong={self.strong_observations} "
                f"{sent_label} skip={self.skip_count} "
                f"comp={len(self.compressions)} hits={self.compression_hits} "
                f"env={self.envelope.display()} "
                f"muted={len(self.temporal_trass_log)} "
                f"λ={self.poisson_rate:.3f} horiz={horiz}{watch_tag}")


@dataclass
class NoveltyNethra:
    """A persistent record that the agent's hypothesis library appears
    insufficient for some variable. Created when a variable's fit keeps
    swinging across audits (instability streak ≥ novelty_weak_streak),
    indicating no library hypothesis is stable. Resolved when the variable
    has been stable for the same number of consecutive audits.

    Fields:
      id:             unique identifier (NOV_001, NOV_002, ...)
      created_cycle:  when first proposed
      last_cycle:     last cycle this novelty was updated/observed
      status:         "open" or "resolved"
      affected_var:   which variable triggered this novelty
      kind:           "vocabulary" — only kind currently used
      reason:         short description of why novelty was raised
      evidence:       last ~12 supporting log lines
      observations:   times this novelty was reinforced
    """
    id: str
    created_cycle: int
    last_cycle: int
    status: str
    affected_var: int
    kind: str
    reason: str
    evidence: List[str] = field(default_factory=list)
    observations: int = 1

    def add(self, cycle: int, ev: Sequence[str]) -> None:
        """Reinforce an existing open novelty. Increments observation count,
        appends to evidence (capped at 12), updates last_cycle."""
        self.observations += 1
        self.last_cycle = cycle
        self.evidence.extend(ev)
        if len(self.evidence) > 12:
            self.evidence = self.evidence[-12:]

    def display(self) -> str:
        """One-line summary for novelty report."""
        return (f"{self.id} {self.status.upper()} {self.kind} x{self.affected_var} "
                f"obs={self.observations}")


class ChainedLedger:
    """The agent's persistent state: one VarNethra per variable, plus a list
    of open/resolved NoveltyNethra records, plus an event log. The "ledger"
    in the framework's design — append-only audit trail. All status changes,
    invalidations, certifications go through methods on this object.

    Attributes:
      n_vars:          total number of slots (same as world.n_vars)
      vars:            dict {var_idx: VarNethra}
      history:         per-var archive of past VarNethra states (on signature change)
      novelty:         list of all NoveltyNethra (open + resolved)
      next_novelty_id: counter for ID generation
      event_log:       chronological list of significant events
    """

    def __init__(self, n_vars: int):
        """Initialize empty ledger with one default-state VarNethra per slot."""
        self.n_vars = n_vars
        self.vars: Dict[int, VarNethra] = {i: VarNethra(var=i) for i in range(n_vars)}
        self.history: Dict[int, List[VarNethra]] = {i: [] for i in range(n_vars)}
        self.novelty: List[NoveltyNethra] = []
        self.next_novelty_id = 1
        self.event_log: List[str] = []
        self.composites: List[CompositeNethra] = []

    def variable_dependents(self, var: int) -> Set[int]:
        """Return all variables whose CURRENT FIT lists `var` as a parent.
        This is the agent's belief about dependency; may be wrong if a fit
        is wrong. Used by closure_descendants for invalidation cascade."""
        return {i for i, n in self.vars.items() if var in n.parents}

    def closure_descendants(self, changed: Set[int]) -> Set[int]:
        """Transitive closure: all variables that the agent BELIEVES depend
        (directly or indirectly) on any var in `changed`, including the
        original set. Computed by repeatedly expanding via variable_dependents
        until no new vars are added. Used for cascading invalidation."""
        out = set(changed)
        frontier = set(changed)
        while frontier:
            new_front = set()
            for v in frontier:
                for d in self.variable_dependents(v):
                    if d not in out:
                        out.add(d)
                        new_front.add(d)
            frontier = new_front
        return out

    def update_var(
        self,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        cycle: int,
        reset_state: bool = True,
    ) -> bool:
        """Apply the result of a full audit to a variable. If (parents, func)
        differs from current fit (signature change), archive the old
        VarNethra to history and reset the variable: fit replaced, status
        back to "proposed", strong_observations=0, sentinels/compressions/
        envelope cleared. Returns True if signature changed, False otherwise.

        The reset on signature change is essential: old sentinels were
        selected against the old fit; old envelope deltas were measured
        against old fit's predictions; old compressions were valid only
        for the old fit. None of these carry over.
        """
        n = self.vars[var]
        changed = (n.parents, n.func) != (parents, func)
        if changed:
            self.history[var].append(VarNethra(
                var=n.var, parents=n.parents, func=n.func, status=n.status,
                strong_observations=n.strong_observations,
                sentinels=list(n.sentinels), expected_outcomes=list(n.expected_outcomes),
                margins=list(n.margins), skip_count=n.skip_count,
                full_audits=n.full_audits, last_changed_cycle=n.last_changed_cycle,
                compressions=list(n.compressions),
                compression_hits=n.compression_hits,
                compression_misses=n.compression_misses,
                cost_weight=n.cost_weight,
            ))
            n.parents = parents
            n.func = func
            if reset_state:
                n.status = "proposed"
                n.strong_observations = 0
                n.sentinels = []
                n.expected_outcomes = []
                n.compressions = []
                n.compression_hits = 0
                n.compression_misses = 0
                n.envelope = NoiseEnvelope()
                n.last_changed_cycle = cycle
                n.tied_frontier = None
                n.dormant_alternatives = []
                self.event_log.append(f"c{cycle}: x{var} signature changed; compressions and envelope reset")
            else:
                self.event_log.append(f"c{cycle}: x{var} tied signature churn; state preserved")
        return changed

    def invalidate(self, vars: Set[int], cycle: int, reason: str) -> Set[int]:
        """Cascade an invalidation through `vars` and their descendant closure.
        For each affected variable:
          - if certified: demote to "uncertain" (was using cheap-path; can't
            anymore until re-audit confirms)
          - if proposed/quarantined: reset strong_observations to 0 and
            demote to "uncertain" if it's a descendant (not the originally
            failed var) — descendants' progress was based on a now-invalid
            parent assumption
        Returns the closure set (originally-failed vars + all descendants).
        Triggered by sentinel failures.
        """
        closure = self.closure_descendants(vars)
        for v in closure:
            n = self.vars[v]
            n.tied_frontier = None  # invalidation changes context; old ties are stale
            if n.status == "certified":
                n.collapse_log.append(f"c{cycle}: invalidated — {reason}")
                n.status = "uncertain"
                self.event_log.append(f"c{cycle}: x{v} invalidated ({reason})")
            elif n.status in ("proposed", "quarantined"):
                if n.strong_observations > 0:
                    n.strong_observations = 0
                    n.collapse_log.append(
                        f"c{cycle}: strong_observations reset (parent invalidated): {reason}"
                    )
                if v not in vars:
                    n.status = "uncertain"
                    n.collapse_log.append(
                        f"c{cycle}: descendant demoted to uncertain — parent invalidated"
                    )
        return closure

    def propose_novelty(
        self, cycle: int, var: int, kind: str, reason: str, evidence: Sequence[str]
    ) -> Tuple[NoveltyNethra, bool]:
        """Raise a novelty for `var` of given `kind`. If an open novelty
        already exists for that (var, kind) pair, reinforce it with new
        evidence. Otherwise create a fresh NoveltyNethra with a generated
        ID. Returns (the novelty, True if newly created)."""
        for n in self.novelty:
            if n.status == "open" and n.affected_var == var and n.kind == kind:
                n.add(cycle, evidence)
                return n, False
        n = NoveltyNethra(
            id=f"NOV_{self.next_novelty_id:03d}",
            created_cycle=cycle, last_cycle=cycle, status="open",
            affected_var=var, kind=kind, reason=reason,
            evidence=list(evidence)[-12:],
        )
        self.next_novelty_id += 1
        self.novelty.append(n)
        return n, True

    def resolve_novelty(self, var: int, kind: str, cycle: int) -> None:
        """Mark any open novelty for (var, kind) as resolved. Called when
        the variable's fit has been stable for novelty_weak_streak consecutive
        cycles, indicating the apparent vocabulary insufficiency was
        resolved by finding a stable hypothesis."""
        for n in self.novelty:
            if n.status == "open" and n.affected_var == var and n.kind == kind:
                n.status = "resolved"
                n.last_cycle = cycle

    def record_drift(self, var: int, cycle: int) -> None:
        """Record a fit-signature change event for `var` at `cycle`.
        Appends to drift_cycles (capped at 200), updates poisson_rate
        (diagnostic display), and caches median_interval and
        watch_threshold_cycles from the empirical inter-invalidation
        distribution. Not reset on signature change."""
        n = self.vars[var]
        n.drift_cycles.append(cycle)
        if len(n.drift_cycles) > 200:
            n.drift_cycles = n.drift_cycles[-200:]
        if n.first_audited_cycle > 0:
            observed = cycle - n.first_audited_cycle + 1
            if observed > 0:
                n.poisson_rate = len(n.drift_cycles) / observed
        # Cache median interval once we have at least two drift events.
        # Sorted here so update_stability_horizon is O(1) per cycle.
        if len(n.drift_cycles) >= 2:
            intervals = [n.drift_cycles[i + 1] - n.drift_cycles[i]
                         for i in range(len(n.drift_cycles) - 1)]
            intervals.sort()
            n.median_interval = intervals[len(intervals) // 2]
            n.watch_threshold_cycles = max(1, int(n.median_interval * 0.3))

    def update_stability_horizon(self, var: int, cycle: int) -> None:
        """Recompute stability horizon and watch state from the empirical
        inter-invalidation distribution. Uses the cached median_interval
        (set by record_drift); no Poisson assumption, no hardcoded threshold.

        horizon = max(0, median_interval - cycles_since_last_change)
        watch  ← horizon ≤ watch_threshold_cycles  (30% of median_interval)

        Suppressed until at least two drift events have been observed so
        the estimate is grounded in actual evidence, not a prior."""
        n = self.vars[var]
        if n.median_interval == 0 or n.first_audited_cycle == 0:
            n.stability_horizon = 9999
            if n.in_watch_state:
                n.in_watch_state = False
                n._watch_propagated = False
            return
        since_last = cycle - n.last_changed_cycle
        n.stability_horizon = max(0, n.median_interval - since_last)
        prev_watch = n.in_watch_state
        n.in_watch_state = n.stability_horizon <= n.watch_threshold_cycles
        if n.in_watch_state != prev_watch:
            n._watch_propagated = False
