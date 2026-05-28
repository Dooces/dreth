"""
DRETH CAUSAL v29 — extension module.

Loaded conditionally by dreth_causal_v28.py when --mode is v29 (or a v29-only
sub-mode). All functions here are pure: they read agent state, return new
Compression objects to be appended. No mutation. The agent's existing collapse
mechanism (sentinel failure → invalidate → re-audit) handles wrong derivations,
so this module is bounded in failure.

Two paths exposed:

  derive_compressions(agent, var, cycle) → List[Compression]
      Algebraic full-collapse: detects narrow operator + source_edge-fit
      combinations that genuinely simplify to a constant. Conservative.
      Only emits when the simplified value is a single float (matching
      Compression.simplified_value's existing shape).

  derive_equivalence_compressions(agent, var, cycle) → List[Compression]
      Tie-set based: when the same set of hypotheses keeps tying for rank 1
      across audits (recorded in agent.tie_log), the equivalence class IS
      the compressible structure. Picks any class member's prediction at a
      representative state as the simplified value, gates broadly. Lets the
      sentinel mechanism validate — if the gate is wrong, sentinel fails.

Public surface kept stable so v28 doesn't need to change when this file does.
Both functions:
  - take agent (ChainedAgent), var (int), cycle (int)
  - return List[Compression]
  - never mutate agent state
  - never raise (catch and return [] on internal error so v28's try/except
    is a backstop, not the primary defense)
"""

from typing import List, Tuple


# ─── algebraic collapse ───────────────────────────────────────────────────────

def derive_compressions(agent, var: int, cycle: int):
    """Detect operator+source_edge combinations that algebraically reduce to a
    single constant value. Conservative — only emits for cases where the
    collapse is unambiguous and the result is a single float.

    Cases handled (full collapse only):
      FIRST(p)             — if p has compression-to-constant, var = that constant
      FIRST(p, q)          — same as FIRST(p); q is ignored
      MAX(p, q) HIGH cap   — if p is fit-HIGH (0.8) and q ∈ [0,1], MAX = 0.8
                             (since 0.8 ≥ q always under domain constraint)
      MIN(p, q) LOW floor  — if p is fit-LOW (0.2) and q ∈ [0,1], MIN = 0.2
                             (only if 0.2 ≤ q always; not always true,
                             so additionally require q's envelope ≥ 0.2)
      DIFF(p, q) equal     — if both p and q are fit to the same constant,
                             DIFF = 0
      MEAN(p, q) constants — if both fit-constant, MEAN = (p_c + q_c)/2

    NOT handled (would need richer Compression representation):
      PROD with one constant — yields scaled other source_edge, not a constant
      MEAN with one constant — yields linear in other source_edge
      MAX with non-extreme constant — depends on other source_edge
      Operator-equivalence ties — handled by derive_equivalence_compressions

    Args:
      agent: ChainedAgent (read-only access to ledger.vars)
      var:   variable index to derive compressions for
      cycle: current cycle number (for Compression.discovery_cycle field)

    Returns:
      List of Compression objects to append to agent.ledger.vars[var].compressions.
      Each Compression's gate is a tuple of (gate_var, target_value, tolerance).
      Empty list if no algebraic collapse applies.
    """
    try:
        # Import here to avoid circular import at load time
        from dreth_causal_v28 import Compression

        n = agent.ledger.vars[var]
        if not n.source_edges:
            return []

        # Helper: is source_edge p effectively a constant under its current fit?
        # Returns (is_const, value, tolerance) where tolerance is p's envelope ε.
        def source_edge_constant_value(p_idx):
            pn = agent.ledger.vars[p_idx]
            tol = pn.current_tolerance
            # Direct LOW/HIGH fit
            if pn.func == "LOW" and not pn.source_edges:
                return True, 0.2, tol
            if pn.func == "HIGH" and not pn.source_edges:
                return True, 0.8, tol
            # FIRST passthrough where the source itself is constant
            if pn.func == "FIRST" and pn.source_edges:
                source = pn.source_edges[0]
                if source < agent.world.visible_count:
                    return source_edge_constant_value(source)
            # Has a derived compression that's itself a constant + the source_edge
            # is currently in a state where that compression's gate matches
            # (peek at compression list)
            for comp in pn.compressions:
                if not comp.gate:
                    return True, comp.simplified_value, tol
            return False, None, tol

        derived: List = []
        source_edges = list(n.source_edges)
        func = n.func

        if func == "FIRST" and source_edges:
            # FIRST(p, ...) ignores everything after first source_edge.
            # If p is constant, FIRST(p) = that constant.
            p = source_edges[0]
            is_const, p_val, p_tol = source_edge_constant_value(p)
            if is_const:
                gate = ((p, p_val, p_tol),)
                derived.append(Compression(
                    gate=gate, simplified_value=p_val,
                    certified_equivalence=1, discovery_cycle=cycle,
                ))

        elif func == "DIFF" and len(source_edges) == 2:
            # DIFF(p, q) = |p - q|. If both are equal constants, DIFF = 0.
            p, q = source_edges
            p_const, p_val, p_tol = source_edge_constant_value(p)
            q_const, q_val, q_tol = source_edge_constant_value(q)
            if p_const and q_const and abs(p_val - q_val) <= max(p_tol, q_tol):
                gate = ((p, p_val, p_tol), (q, q_val, q_tol))
                derived.append(Compression(
                    gate=gate, simplified_value=0.0,
                    certified_equivalence=1, discovery_cycle=cycle,
                ))

        elif func == "MEAN" and len(source_edges) == 2:
            # MEAN(p, q) = (p + q) / 2. Constants on both → constant mean.
            p, q = source_edges
            p_const, p_val, p_tol = source_edge_constant_value(p)
            q_const, q_val, q_tol = source_edge_constant_value(q)
            if p_const and q_const:
                gate = ((p, p_val, p_tol), (q, q_val, q_tol))
                derived.append(Compression(
                    gate=gate, simplified_value=(p_val + q_val) / 2.0,
                    certified_equivalence=1, discovery_cycle=cycle,
                ))

        elif func == "MAX" and len(source_edges) == 2:
            # MAX(p, q): if p is fit-HIGH (0.8) AND q's envelope keeps q ≤ 0.8
            # (which we approximate by checking q has had no observation > 0.8 + q_tol
            # within recent cycles — proxy: q is also fit constant ≤ 0.8 OR q is LOW).
            for hi_idx, lo_idx in [(0, 1), (1, 0)]:
                hi = source_edges[hi_idx]
                lo = source_edges[lo_idx]
                hi_const, hi_val, hi_tol = source_edge_constant_value(hi)
                if hi_const and hi_val >= 0.8 - hi_tol:
                    lo_const, lo_val, _ = source_edge_constant_value(lo)
                    # Only safe if lo is constant ≤ hi_val
                    if lo_const and lo_val <= hi_val + hi_tol:
                        gate = ((hi, hi_val, hi_tol),)
                        derived.append(Compression(
                            gate=gate, simplified_value=hi_val,
                            certified_equivalence=1, discovery_cycle=cycle,
                        ))
                    break  # don't double-emit

        elif func == "MIN" and len(source_edges) == 2:
            # MIN(p, q): if p is fit-LOW (0.2) AND q's value never falls below it.
            for lo_idx, other_idx in [(0, 1), (1, 0)]:
                lo = source_edges[lo_idx]
                other = source_edges[other_idx]
                lo_const, lo_val, lo_tol = source_edge_constant_value(lo)
                if lo_const and lo_val <= 0.2 + lo_tol:
                    other_const, other_val, _ = source_edge_constant_value(other)
                    if other_const and other_val >= lo_val - lo_tol:
                        gate = ((lo, lo_val, lo_tol),)
                        derived.append(Compression(
                            gate=gate, simplified_value=lo_val,
                            certified_equivalence=1, discovery_cycle=cycle,
                        ))
                    break

        elif func == "PROD" and len(source_edges) == 2:
            # PROD with both constants → constant product. Don't emit for
            # one-constant case (would be linear, not constant).
            p, q = source_edges
            p_const, p_val, p_tol = source_edge_constant_value(p)
            q_const, q_val, q_tol = source_edge_constant_value(q)
            if p_const and q_const:
                gate = ((p, p_val, p_tol), (q, q_val, q_tol))
                derived.append(Compression(
                    gate=gate, simplified_value=p_val * q_val,
                    certified_equivalence=1, discovery_cycle=cycle,
                ))

        return derived

    except Exception:
        # Defensive: never propagate exceptions to v28.
        return []


# ─── equivalence-class compression (tie-based) ────────────────────────────────

# A tie set must recur this many times before being treated as a stable
# equivalence class (durable, not transient probe accident).
EQUIV_STABILITY_THRESHOLD = 3

def derive_equivalence_compressions(agent, var: int, cycle: int):
    """Tie-set based compression — fixed gate-validity check.

    Bug in prior version: cached one simplified_value at audit-time anchors,
    but ties only meant "agreed on probes seen," not "agreed everywhere."
    When world state moved within the gate's tolerance, the cached value
    became stale → sentinel failures → cascading re-audits → net cost spike.

    Fix: before emitting a compression, verify that ALL tied hypotheses
    produce predictions within the variable's current_tolerance of each
    other AT THE GATE'S ANCHOR VALUES. If they don't, the tie was probe-
    artifact, not durable equivalence — skip. If they do, the cached
    simplified_value is safe to use because by construction every class
    member predicts within tolerance at that anchor.

    Additionally: only emit when the prediction is genuinely close to
    constant under perturbation of the current state within each source_edge's
    envelope. This catches the case where ties are coincidental at one
    state but the underlying functions diverge as state moves.

    Args:
      agent: ChainedAgent (read-only)
      var:   variable index
      cycle: current cycle (for Compression.discovery_cycle)

    Returns:
      List[Compression]. Empty if no tie set passes the validity check.
    """
    try:
        from dreth_causal_v28 import Compression, FUNC_LIBRARY

        if var not in agent.tie_log:
            return []

        n = agent.ledger.vars[var]
        if not n.source_edges:
            return []

        var_tol = n.current_tolerance
        derived = []

        for tie_set, count in agent.tie_log[var].items():
            if count < EQUIV_STABILITY_THRESHOLD:
                continue
            if len(tie_set) < 2:
                continue

            # Build candidate gate from current source_edge values.
            gate_parts = []
            for p in n.source_edges:
                if p >= agent.world.visible_count:
                    continue
                pn = agent.ledger.vars[p]
                gate_parts.append((p, agent.world.state[p], pn.current_tolerance))
            if not gate_parts:
                continue
            gate = tuple(gate_parts)

            # Validity check #1: at the gate's anchor values, do all tied
            # hypotheses produce within-tolerance predictions of each other?
            anchor_state = list(agent.world.state)
            anchor_preds = []
            valid = True
            for hyp_source_edges, hyp_func in tie_set:
                if hyp_func not in FUNC_LIBRARY:
                    valid = False
                    break
                try:
                    par_vals = [anchor_state[p] for p in hyp_source_edges]
                    pred = FUNC_LIBRARY[hyp_func](par_vals)
                    anchor_preds.append(pred)
                except (IndexError, KeyError, ValueError):
                    valid = False
                    break
            if not valid or not anchor_preds:
                continue

            # All tied hypotheses must agree at anchor (within var's tolerance).
            anchor_pred = anchor_preds[0]
            if not all(abs(p - anchor_pred) <= var_tol for p in anchor_preds):
                continue

            # Validity check #2: perturb each source_edge within its envelope ε
            # and confirm tied hypotheses STILL agree. If they diverge under
            # perturbation, the tie was state-specific and the compression
            # would return stale values when state moved within the gate.
            # Sample 4 perturbed corner states.
            perturbations_ok = True
            for trial in range(4):
                perturbed_state = list(anchor_state)
                for (p_idx, p_val, p_tol) in gate_parts:
                    # Alternate corners: ±tol per source_edge per trial
                    sign = 1.0 if (trial >> (p_idx % 4)) & 1 else -1.0
                    perturbed_state[p_idx] = max(0.0, min(1.0, p_val + sign * p_tol))
                trial_preds = []
                for hyp_source_edges, hyp_func in tie_set:
                    par_vals = [perturbed_state[p] for p in hyp_source_edges]
                    try:
                        trial_preds.append(FUNC_LIBRARY[hyp_func](par_vals))
                    except (IndexError, KeyError, ValueError):
                        perturbations_ok = False
                        break
                if not perturbations_ok:
                    break
                first = trial_preds[0]
                if not all(abs(p - first) <= var_tol for p in trial_preds):
                    perturbations_ok = False
                    break
            if not perturbations_ok:
                continue

            # All checks passed: the tie is durable across the gate's
            # tolerance neighborhood. The cached simplified_value is the
            # average of tied hypotheses' anchor predictions (any single
            # one would be within tolerance; averaging reduces bias).
            simplified = sum(anchor_preds) / len(anchor_preds)
            derived.append(Compression(
                gate=gate, simplified_value=simplified,
                certified_equivalence=count, discovery_cycle=cycle,
            ))

        return derived

    except Exception:
        return []