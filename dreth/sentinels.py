from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The cheap-path validation layer. Two functions, both active.
#
# select_var_sentinels:
#   Called once after a fit stabilizes. Generates a pool of candidate
#   (var, val) intervention probes, scores each by how many alternative
#   hypotheses would produce a different prediction, and returns the top-K
#   most discriminating ones. These become the variable's sentinels.
#   Goal: pick probes that are hard for a wrong hypothesis to pass.
#
# check_var_sentinels_with_envelope:
#   Called every cycle instead of a full audit. Issues the sentinel probes
#   against the current world state, compares predictions to actuals within
#   the variable's noise envelope, and returns pass/fail. Cost-dispatch
#   (high/mid/low cost_weight) controls how strictly partial failures
#   escalate vs. get logged as TEMPORAL_TRASS and dismissed.
#
# Nothing in this file is vestigial. The cost-dispatch paths are all active.
# TEMPORAL_TRASS logging (TemporalTrassEntry) is the audit trail for
# dismissed deviations — used for credit assignment if dismissal turns out
# to have been wrong.
# ─────────────────────────────────────────────────────────────────────────────

import random
from typing import List, Optional, Set, Tuple

from .world import CausalWorld
from .ledger import DEFAULT_TOLERANCE, TemporalTrassEntry, VarNethra, values_match
from .fit import predict_var, enumerate_var_hypotheses, enumerate_var_hypotheses_restricted
from .functions import FUNC_LIBRARY

# Max alternative hypotheses to sample for discrimination scoring. The full
# space is O(n_vars²) which becomes dominant for n_vars≥20. A sample of 200
# preserves probe ranking without exhaustive evaluation — discrimination is a
# selection heuristic, not a correctness requirement.
_MAX_DISC_SAMPLE = 200

def select_var_sentinels(
    var: int, parents: Tuple[int, ...], func: str,
    world: CausalWorld, rng: random.Random,
    count: int, pool: int, tolerance: float,
    available_parents: Optional[Set[int]] = None,
) -> Tuple[List[Tuple[int, float]], List[float]]:
    """Pick `count` intervention probes that will be used as sentinels for
    cheap-path validation. Returns (probes, expected_outcomes).

    Steps:
      1. Generate `pool` random (var, val) candidate probes.
      2. For each probe, compute discrimination score = number of alternative
         hypotheses (from the agent's available enumeration) whose prediction
         differs from the chosen fit's prediction. Higher = better probe.
      3. Sort by discrimination, take top `count`.

    Discrimination is preferred but NOT required: a sentinel's job is to
    test the chosen fit against the world, not against alternatives. The
    expected_outcomes are computed at selection time but are not used at
    check time (sentinel-check recomputes against current state). They're
    kept as a legacy field.
    """
    candidates = [(rng.randint(0, world.visible_count - 1), rng.random()) for _ in range(pool)]
    if available_parents is None:
        neighbors = enumerate_var_hypotheses(var, world.visible_count)
    else:
        neighbors = enumerate_var_hypotheses_restricted(
            var, available_parents,
        )
    # Pre-resolve function lookups and filter self-hypothesis once; reuse per probe.
    # For large hypothesis spaces (n_vars≥20), this inner loop dominates runtime.
    # Sampling is intentionally avoided: rng.sample would shift the RNG sequence
    # and alter the agent's subsequent probe choices, which degrades iv quality.
    chosen_fn = FUNC_LIBRARY.get(func)
    neighbor_fns: List[Tuple[object, Tuple[int, ...]]] = [
        (FUNC_LIBRARY[nf], np_)
        for np_, nf in neighbors
        if (np_, nf) != (parents, func) and nf in FUNC_LIBRARY
    ]
    # Sample a deterministic stride-based subset when hypothesis space is large.
    # stride ≥ 2 only kicks in above _MAX_DISC_SAMPLE; below it we use all entries.
    if len(neighbor_fns) > _MAX_DISC_SAMPLE:
        stride = len(neighbor_fns) // _MAX_DISC_SAMPLE
        neighbor_fns = neighbor_fns[::stride][:_MAX_DISC_SAMPLE]
    seen = set()
    scored: List[Tuple[int, Tuple[int, float], float]] = []
    for iv in candidates:
        key = (iv[0], round(iv[1], 3))
        if key in seen: continue
        seen.add(key)
        # Build forced state once per probe; reuse for every neighbor hypothesis.
        forced = tuple(iv[1] if i == iv[0] else world.state[i]
                       for i in range(world.visible_count))
        my_pred = chosen_fn([forced[p] for p in parents]) if chosen_fn is not None else float("nan")
        disagree = 0
        for n_fn, n_par in neighbor_fns:
            np = n_fn([forced[p] for p in n_par])
            if not values_match(np, my_pred, tolerance):
                disagree += 1
        scored.append((disagree, iv, my_pred))
    scored.sort(reverse=True, key=lambda x: x[0])
    chosen: List[Tuple[int, float]] = []
    expected: List[float] = []
    for _, iv, pred in scored[:count]:
        chosen.append(iv)
        expected.append(pred)
    return chosen, expected


def check_var_sentinels_with_envelope(
    var: int, n: VarNethra, world: CausalWorld, cycle: int,
    cost_low_threshold: float, cost_high_threshold: float,
) -> Tuple[bool, int, int, str]:
    """Run sentinel checks for one variable. Returns (passed, score, total,
    reason). This is the cheap-path validation: cheaper than full audit
    (only uses sentinel-count probes vs intervention_budget) and requires
    no hypothesis re-enumeration.

    For each sentinel probe:
      - compute expected = current fit's prediction at this probe
      - issue intervention to world, get actual
      - delta = |expected - actual|; feed envelope; count match if within
        envelope ε (or default tolerance if not yet certified)

    Dispatch by cost_weight (asymmetric attention to mistakes):
      cost >= cost_high_threshold:
        Strict — any miss is sentinel failure. Used for high-stakes vars
        where false negatives are expensive (e.g., is-this-a-child).
      cost_low_threshold <= cost < cost_high_threshold:
        Standard — all-pass returns OK; partial-pass triggers escalation
        only if envelope_failing (clustering of OOB events recently).
        Tolerated outliers logged as TEMPORAL_TRASS.
      cost < cost_low_threshold:
        Permissive — failures are dismissed regardless of clustering.
        Logged as TEMPORAL_TRASS. The intent: low-cost variables can
        absorb persistent inaccuracy. If downstream shows this mattered,
        re-weight; don't punish the dismissal.

    All deltas are added to envelope regardless of dispatch decision.
    """
    score = 0
    total = len(n.sentinels)
    deviations = []
    tol_fallback = DEFAULT_TOLERANCE
    # noise_floor vars: widen re-trigger threshold. The 5% statistical tail of
    # a certified noise floor should not count as signal. Only fail on
    # deviations that exceed the noise floor by a meaningful margin (3×ε).
    _base_eps = n.envelope.certified_eps if n.envelope.certified_eps > 0 else tol_fallback
    _NOISE_FLOOR_K = 3.0
    effective_tol = _NOISE_FLOOR_K * _base_eps if n.role_for("skip") == "noise_floor" else _base_eps
    for iv, _stale_exp in zip(n.sentinels, n.expected_outcomes):
        expected = predict_var(n.parents, n.func, world.state, iv[0], iv[1], world.visible_count)
        # Only compute the target var's output. The previous full-state path
        # discarded n_vars-1 outputs per sentinel probe.
        actual = world.predict_var_under_intervention(var, iv[0], iv[1])
        delta = abs(actual - expected)
        deviations.append(delta)
        if delta <= effective_tol:
            score += 1
        n.envelope.add_delta(delta, cycle)

    if score == total:
        return True, score, total, "all_within"

    out_of_band = total - score
    max_dev = max(deviations) if deviations else 0.0

    if n.cost_weight >= cost_high_threshold:
        return False, score, total, f"high_cost_strict (max_dev={max_dev:.3f})"

    if n.cost_weight < cost_low_threshold:
        n.temporal_trass_log.append(TemporalTrassEntry(
            cycle=cycle, var=var, delta=max_dev, cost_weight=n.cost_weight,
            reason="low_cost_dismissed_persistent" if n.envelope.envelope_failing(threshold_count=8) else "low_cost_dismissed",
        ))
        return True, score, total, f"low_cost_dismissed (max_dev={max_dev:.3f}) #TEMPORAL_TRASS"

    if n.envelope.envelope_failing(threshold_count=5):
        return False, score, total, f"mid_cost_persistent (max_dev={max_dev:.3f})"
    else:
        n.temporal_trass_log.append(TemporalTrassEntry(
            cycle=cycle, var=var, delta=max_dev, cost_weight=n.cost_weight,
            reason="outlier_within_tolerance",
        ))
        return True, score, total, f"mid_cost_outlier (max_dev={max_dev:.3f}) #TEMPORAL_TRASS"
