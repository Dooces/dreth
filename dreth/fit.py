from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Runs the full audit for one variable. Everything here is morphology output.
#
# fit_var:
#   1. Enumerates (parents, func) hypotheses — either restricted to
#      tareth-certified available_parents, or full space if too few.
#   2. Runs interventional probes, scores each hypothesis by how often its
#      prediction matches the world's actual output within tolerance.
#   3. Returns the best hypothesis plus diagnostics.
#
# What this file produces:
#   - scores, tie sets, near-tie candidates — these are structural
#     observations about which candidates survived the current probe set.
#     They say nothing about WHY candidates tie or what the tie means.
#   - near_tie_candidates: all hypotheses within near_tie_margin probes of
#     best score. Passed to TiedFrontier in agent.py. Morphology only.
#
# Active and load-bearing:
#   fit_var, score_var_hypothesis, enumerate_var_hypotheses,
#   enumerate_var_hypotheses_restricted, predict_var
#
# Active:
#   _adaptive_probe_budget — activated in agent.py _full_audit_var (P1-A).
#   forced_probes param (P1-B) — when a TiedFrontier exists, its separating
#     probes are injected before the discrimination pool so they are always
#     included in the audit regardless of pool randomness.
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
#   Neither verdict is permanent. World drift, scope change, or sentinel failure
#   revokes certification. The verdict belongs to the scope, not the hypothesis.
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
# This file: produces MORPHOLOGY observables only — scores, near-tie candidates,
#   tie sets. Nothing returned here is a cause classification. Two candidates
#   sharing scores tells you their shapes are indistinguishable to the current
#   probe set, not why they tie. Cause requires separating probes (P1.2).
# ════════════════════════════════════════════════════════════════════════════════

import random
import numpy as np
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .functions import State, FUNC_LIBRARY
from .world import CausalWorld
from .ledger import values_match

def predict_var(
    parents: Tuple[int, ...], func: str,
    state: State, intervention_var: int, intervention_val: float,
    n_vars: int,
) -> float:
    """Agent's prediction for `var` under (parents, func) when intervention
    forces `intervention_var` = `intervention_val`. Builds the forced state
    tuple, extracts parent values, and applies the agent-side function.
    Returns NaN if `func` is not in the agent's library (defensive — the
    agent should never enumerate such hypotheses)."""
    if func not in FUNC_LIBRARY:
        return float("nan")
    forced = tuple(intervention_val if i == intervention_var else state[i]
                   for i in range(n_vars))
    par_vals = [forced[p] for p in parents]
    return FUNC_LIBRARY[func](par_vals)


def score_var_hypothesis(
    var: int, parents: Tuple[int, ...], func: str,
    world: CausalWorld, interventions: Sequence[Tuple[int, float]],
    tolerance: float,
) -> int:
    """Pure-python scoring for one hypothesis. For each intervention probe,
    compares hypothesis prediction vs world's actual output and counts
    matches within tolerance. Returns the integer match count.
    Kept as fallback for the vectorized scorer; not on the hot path."""
    score = 0
    saved = world.state
    for iv_var, iv_val in interventions:
        world.state = saved
        actual = world.predict_under_intervention(iv_var, iv_val)[var]
        predicted = predict_var(parents, func, saved, iv_var, iv_val, world.visible_count)
        if values_match(predicted, actual, tolerance):
            score += 1
    world.state = saved
    return score


def enumerate_var_hypotheses(var: int, n_vars: int, max_parents: int = 2
                             ) -> List[Tuple[Tuple[int, ...], str]]:
    """Generate the full hypothesis space for `var`: (parents_tuple, func_name)
    pairs. Includes:
      - 2 constant hypotheses: ((), LOW), ((), HIGH)
      - n_vars - 1 single-parent FIRST hypotheses (each non-self var as parent)
      - if max_parents >= 2: 5 functions × C(n_vars-1, 2) two-parent combinations
    Used when restricted enumeration falls back (too few settled parents)
    and by sentinel selection."""
    out = [((), "LOW"), ((), "HIGH"), ((), "TINY")]
    for p in range(n_vars):
        if p == var: continue
        out.append(((p,), "FIRST"))
    if max_parents >= 2:
        for p1 in range(n_vars):
            if p1 == var: continue
            for p2 in range(p1+1, n_vars):
                if p2 == var: continue
                for fn in ["MEAN", "MAX", "MIN", "PROD", "DIFF"]:
                    out.append(((p1, p2), fn))
    return out


def enumerate_var_hypotheses_restricted(
    var: int,
    available_parents: Set[int],
    max_parents: int = 2,
) -> List[Tuple[Tuple[int, ...], str]]:
    """Restricted hypothesis enumeration — same shape as the full enumerator
    but candidate parents drawn only from `available_parents`. Constants
    (LOW/HIGH) are always included since they need no parents. This is the
    main hypothesis-space reduction: only consider parents the framework
    has already provisionally committed to.
    Caller falls back to full enumeration if available_parents has < 2
    candidates (insufficient to form 2-parent hypotheses)."""
    out: List[Tuple[Tuple[int, ...], str]] = [((), "LOW"), ((), "HIGH"), ((), "TINY")]
    candidate_parents = sorted(p for p in available_parents if p != var)
    for p in candidate_parents:
        out.append(((p,), "FIRST"))
    if max_parents >= 2:
        for i, p1 in enumerate(candidate_parents):
            for p2 in candidate_parents[i+1:]:
                for fn in ["MEAN", "MAX", "MIN", "PROD", "DIFF"]:
                    out.append(((p1, p2), fn))
    return out


# ── NUMPY-BATCHED FUNCTION EVALUATION ─────────────────────────────────────────

def _func_apply_batch(func: str, parent_vals: np.ndarray) -> np.ndarray:
    """Vectorized version of FUNC_LIBRARY for hot-path scoring. Computes
    `func` applied across N parent-value rows in one numpy call.
    Input shape: (N, k) where k is parent count. Output: (N,).
    Strictly matches FUNC_LIBRARY — no SIN or hidden functions.
    Used by score_hypotheses_batched to score all hypotheses against
    all interventions in two outer loops with numpy in the middle."""
    if func == "LOW":
        return np.full(parent_vals.shape[0], 0.2)
    if func == "HIGH":
        return np.full(parent_vals.shape[0], 0.8)
    if func == "TINY":
        return np.full(parent_vals.shape[0], 0.1)
    if parent_vals.shape[1] == 0:
        return np.zeros(parent_vals.shape[0])
    if func == "FIRST":
        return parent_vals[:, 0]
    if func == "MEAN":
        return parent_vals.mean(axis=1)
    if func == "MAX":
        return parent_vals.max(axis=1)
    if func == "MIN":
        return parent_vals.min(axis=1)
    if func == "PROD":
        return parent_vals.prod(axis=1)
    if func == "DIFF":
        if parent_vals.shape[1] < 2:
            return parent_vals[:, 0]
        return np.abs(parent_vals[:, 0] - parent_vals[:, 1])
    raise ValueError(f"unknown agent func: {func}")


def score_hypotheses_batched(
    var: int, hypotheses: List[Tuple[Tuple[int, ...], str]],
    world: CausalWorld, interventions: Sequence[Tuple[int, float]],
    tolerance: float,
    return_preds: bool = False,
):
    """Vectorized hypothesis scoring. For each (parents, func) hypothesis,
    counts how many of `interventions` produce predictions within tolerance
    of the world's actual output. Returns one int per hypothesis.

    If return_preds=True, returns (scores, actuals, all_preds) instead of
    just scores. all_preds is shape (n_hypotheses, n_interventions). Used
    by fit_var to populate FitDiagnostic per-probe arrays for downstream
    analysis (cross-audit, cross-variable, tie characterization).

    Implementation:
      1. Run each intervention probe once, record actual output and forced state
      2. For each hypothesis, extract parent columns from forced states,
         apply func via _func_apply_batch, count |pred - actual| ≤ tol
    World state is restored after probes so the test is non-destructive.
    """
    n_iv = len(interventions)
    if n_iv == 0:
        return (np.zeros(len(hypotheses), dtype=int), np.zeros(0), np.zeros((len(hypotheses), 0))) if return_preds else np.zeros(len(hypotheses), dtype=int)

    saved = world.state
    n_vars = world.visible_count

    actuals = np.empty(n_iv)
    forced_states = np.empty((n_iv, n_vars))
    for k, (iv_var, iv_val) in enumerate(interventions):
        world.state = saved
        actuals[k] = world.predict_under_intervention(iv_var, iv_val)[var]
        forced_states[k] = [iv_val if i == iv_var else saved[i] for i in range(n_vars)]
    world.state = saved

    scores = np.empty(len(hypotheses), dtype=int)
    if return_preds:
        all_preds = np.empty((len(hypotheses), n_iv))
    for h_idx, (parents, func) in enumerate(hypotheses):
        if not parents:
            par_vals = np.zeros((n_iv, 0))
        else:
            par_vals = forced_states[:, list(parents)]
        preds = _func_apply_batch(func, par_vals)
        scores[h_idx] = int(np.sum(np.abs(preds - actuals) <= tolerance))
        if return_preds:
            all_preds[h_idx] = preds
    if return_preds:
        return scores, actuals, all_preds
    return scores


def _discrimination_counts(rounded: np.ndarray) -> np.ndarray:
    """Count distinct rounded prediction buckets per intervention column."""
    if rounded.shape[0] == 0:
        return np.zeros(rounded.shape[1], dtype=int)
    sorted_by_probe = np.sort(rounded.T, axis=1)
    return 1 + np.count_nonzero(np.diff(sorted_by_probe, axis=1), axis=1)


def fit_var(
    var: int, world: CausalWorld, rng: random.Random,
    intervention_budget: int, tolerance: float,
    targeted: bool = True,
    available_parents: Optional[Set[int]] = None,
    diag: Optional[Dict[str, object]] = None,
    near_tie_margin: int = 0,
    forced_probes: Optional[Tuple[Tuple[int, float], ...]] = None,
) -> Tuple[Tuple[int, ...], str, int, int]:
    """Find the best (parents, func) for one variable. Returns the tuple
    (parents, func, best_score, second_best_score).

    Steps:
      1. Enumerate hypothesis space (restricted if available_parents is
         provided and large enough, else full).
      2. Build intervention pool. If targeted=True, generate 4×budget candidates,
         score each by hypothesis-discrimination (number of distinct
         predictions across hypotheses), pick top-budget. Else random sample.
         If forced_probes is given (P1-B), those (iv_var, iv_val) pairs are
         injected first; remaining slots filled from discrimination pool.
      3. Score all hypotheses against chosen interventions (vectorized).
      4. Return rank-1 hypothesis and its margin to rank-2.

    The `diag` dict is filled with diagnostic data (true rank, available
    parents, restricted_used flag, scores, etc.) for offline analysis.
    Diagnostics never affect fit selection.
    """
    n_vars = world.visible_count
    # Use unrestricted enumeration only when available_parents was not provided
    # (legacy/no-constraint callers). An explicitly empty set means "no committed
    # parents yet" → restrict to constants; do NOT fall back to the full n_vars²
    # hypothesis space which causes blowup when the agent is bootstrapping.
    restricted_used = available_parents is not None
    if not restricted_used:
        hypotheses = enumerate_var_hypotheses(var, n_vars)
    else:
        hypotheses = enumerate_var_hypotheses_restricted(var, available_parents)

    # P1-B: forced_probes from TiedFrontier.separating_probes are guaranteed
    # inclusions — budget slots they consume are unavailable to the pool.
    _forced: List[Tuple[int, float]] = list(forced_probes) if forced_probes else []
    _forced_count = min(len(_forced), intervention_budget)
    _forced = _forced[:_forced_count]
    _remaining_budget = intervention_budget - _forced_count

    if targeted:
        pool_size = max(max(_remaining_budget, 1) * 4, 40)
        candidates = [(rng.randint(0, n_vars - 1), rng.random())
                      for _ in range(pool_size)]
        # Use batched evaluation for discrimination too
        n_pool = len(candidates)
        forced_pool = np.empty((n_pool, n_vars))
        for k, (iv_var, iv_val) in enumerate(candidates):
            forced_pool[k] = [iv_val if i == iv_var else world.state[i] for i in range(n_vars)]
        # For each hypothesis, predict at every pool state — shape (H, P)
        H = len(hypotheses)
        all_preds = np.empty((H, n_pool))
        for h_idx, (parents, func) in enumerate(hypotheses):
            if not parents:
                par_vals = np.zeros((n_pool, 0))
            else:
                par_vals = forced_pool[:, list(parents)]
            all_preds[h_idx] = _func_apply_batch(func, par_vals)
        # Discrimination per pool entry: count distinct predictions within tolerance
        # (simple approach: number of unique values rounded to tolerance)
        rounded = np.round(all_preds / max(tolerance, 1e-6)).astype(int)
        # discriminations[k] = number of distinct hypothesis-buckets at pool entry k
        discrim = _discrimination_counts(rounded)
        # Frontier bias (not yet activated): when near-tied candidates from a
        # prior audit are known, probes that split those candidates could be
        # upweighted here. Disabled for the same reason as _adaptive_probe_budget:
        # Pick top _remaining_budget by discrimination; prepend forced probes.
        # P1-B: forced_probes (from TiedFrontier.separating_probes) are guaranteed
        # inclusions — they don't compete in the discrimination pool so the pool
        # randomness is preserved for the non-forced slots.
        order = np.argsort(-discrim)
        pool_selected = [candidates[idx] for idx in order[:_remaining_budget]]
        while len(pool_selected) < _remaining_budget:
            pool_selected.append((rng.randint(0, n_vars - 1), rng.random()))
        interventions = _forced + pool_selected
    else:
        pool_random = [(rng.randint(0, n_vars - 1), rng.random())
                       for _ in range(_remaining_budget)]
        interventions = _forced + pool_random

    # Score all hypotheses with per-probe data captured (used for diagnostic
    # classification AND for downstream extension consumers).
    scores, actuals_arr, all_preds = score_hypotheses_batched(
        var, hypotheses, world, interventions, tolerance, return_preds=True
    )
    order = np.argsort(-scores)
    best_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else best_idx
    best_parents, best_func = hypotheses[best_idx]

    # Tie set: all hypotheses scoring equal to the best
    best_score_val = int(scores[best_idx])
    tie_indices = [int(i) for i in np.where(scores == best_score_val)[0]]
    tie_set = frozenset(hypotheses[i] for i in tie_indices)

    # Near-tie constellation: all hypotheses within near_tie_margin of best.
    # Stored as ((parents, func, score), ...) sorted by score desc so the
    # agent can maintain TiedFrontier state across audits.
    near_tie_threshold = best_score_val - max(0, near_tie_margin)
    near_tie_candidates_out: Tuple = tuple(
        sorted(
            ((hypotheses[i][0], hypotheses[i][1], int(scores[i]))
             for i in range(len(hypotheses))
             if int(scores[i]) >= near_tie_threshold),
            key=lambda x: -x[2],
        )
    )
    if available_parents is not None:
        near_tie_context_key_out = hash(frozenset(available_parents))
    else:
        near_tie_context_key_out = hash(frozenset(range(n_vars)))

    if diag is not None:
        best_score = int(scores[best_idx])
        second_score = int(scores[second_idx]) if second_idx != best_idx else 0
        margin = best_score - second_score
        pick_preds_arr = all_preds[best_idx]

        if margin >= 4:
            failure_class = "fit_clean"
        elif margin > 0:
            failure_class = "fit_close"
        else:
            failure_class = "fit_tied"

        diag.update({
            "restricted": restricted_used,
            "hypothesis_count": len(hypotheses),
            "best_score": best_score,
            "second_score": second_score,
            "margin": margin,
            "best_parents": tuple(best_parents),
            "best_func": best_func,
            "failure_class": failure_class,
            "probes": tuple(interventions),
            "actuals": tuple(float(a) for a in actuals_arr),
            "pick_preds": tuple(float(p) for p in pick_preds_arr),
            "tie_set": tie_set,
            "near_tie_candidates": near_tie_candidates_out,
            "near_tie_context_key": near_tie_context_key_out,
        })

    return (best_parents, best_func,
            int(scores[best_idx]),
            int(scores[second_idx]) if second_idx != best_idx else 0)
