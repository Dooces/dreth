from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The operator vocabulary. Two sets:
#   FUNC_LIBRARY      — operators the agent can use to construct hypotheses.
#                       This is a closed set. If the world uses an operator
#                       not here, the agent cannot converge and fires novelty.
#   HIDDEN_FUNC_LIBRARY — FUNC_LIBRARY + SIN. The world can assign SIN to a
#                       variable; the agent cannot fit it. Used to test that
#                       vocabulary-novelty detection fires correctly.
#
# Nothing in this file is vestigial. The set of operators here directly bounds
# what hypotheses the agent will ever consider. Extending this set widens the
# hypothesis space for every variable on every audit.
# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import Callable, Dict, List, Tuple

State = Tuple[float, ...]


# ── FUNCTION LIBRARY (continuous) ─────────────────────────────────────────────
# These are the agent's available hypothesis functions. Each takes a list of
# parent values (floats in [0,1]) and returns one float in [0,1]. No noise
# applied here — noise is added by the world wrapper, not by these primitives.

def f_mean(parents: List[float]) -> float:
    """Mean of parent values. Returns 0.0 if no parents."""
    return sum(parents) / len(parents) if parents else 0.0

def f_max(parents: List[float]) -> float:
    """Max of parent values. Returns 0.0 if no parents."""
    return max(parents) if parents else 0.0

def f_min(parents: List[float]) -> float:
    """Min of parent values. Returns 0.0 if no parents."""
    return min(parents) if parents else 0.0

def f_product(parents: List[float]) -> float:
    """Product of parent values. Returns 0.0 if no parents."""
    out = 1.0
    for p in parents: out *= p
    return out if parents else 0.0

def f_diff(parents: List[float]) -> float:
    """Absolute difference of first two parents. With <2 parents, returns first or 0."""
    if len(parents) < 2: return parents[0] if parents else 0.0
    return abs(parents[0] - parents[1])

def f_first(parents: List[float]) -> float:
    """Identity on first parent. Returns 0.0 if no parents."""
    return parents[0] if parents else 0.0

def f_const_low(parents: List[float]) -> float:
    """Constant 0.2. Used for variables with no parents in the agent's hypothesis."""
    return 0.2

def f_const_high(parents: List[float]) -> float:
    """Constant 0.8. Used for variables with no parents in the agent's hypothesis."""
    return 0.8

def f_const_tiny(parents: List[float]) -> float:
    """Constant 0.1. Used as baseline for joint-false-trass test worlds where
    two vars are individually below the salience threshold but jointly tareth."""
    return 0.1

# Agent's available functions, addressed by name. The agent enumerates
# (parents, func) hypotheses using only these names.
FUNC_LIBRARY: Dict[str, Callable[[List[float]], float]] = {
    "MEAN":  f_mean,
    "MAX":   f_max,
    "MIN":   f_min,
    "PROD":  f_product,
    "DIFF":  f_diff,
    "FIRST": f_first,
    "LOW":   f_const_low,
    "HIGH":  f_const_high,
    "TINY":  f_const_tiny,
}

def f_sin_gate(parents: List[float]) -> float:
    """Sinusoidal function of mean(parents). NOT in agent's library — used only
    by the hidden world to test vocabulary novelty detection. Returns 0.5
    when no parents (constant, won't trigger novelty in that case)."""
    if not parents: return 0.5
    s = sum(parents) / len(parents)
    return 0.5 + 0.5 * math.sin(2 * math.pi * s)

# Hidden world's function set: agent's library plus SIN. The world can
# generate variables using SIN; the agent cannot fit them and must report
# vocabulary novelty (instability streak) when this happens.
HIDDEN_FUNC_LIBRARY: Dict[str, Callable[[List[float]], float]] = dict(FUNC_LIBRARY)
HIDDEN_FUNC_LIBRARY["SIN"] = f_sin_gate
