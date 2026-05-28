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
# source_edge values (floats in [0,1]) and returns one float in [0,1]. No noise
# applied here — noise is added by the world wrapper, not by these primitives.

def f_mean(source_edges: List[float]) -> float:
    """Mean of source_edge values. Returns 0.0 if no source_edges."""
    return sum(source_edges) / len(source_edges) if source_edges else 0.0

def f_max(source_edges: List[float]) -> float:
    """Max of source_edge values. Returns 0.0 if no source_edges."""
    return max(source_edges) if source_edges else 0.0

def f_min(source_edges: List[float]) -> float:
    """Min of source_edge values. Returns 0.0 if no source_edges."""
    return min(source_edges) if source_edges else 0.0

def f_product(source_edges: List[float]) -> float:
    """Product of source_edge values. Returns 0.0 if no source_edges."""
    out = 1.0
    for p in source_edges: out *= p
    return out if source_edges else 0.0

def f_diff(source_edges: List[float]) -> float:
    """Absolute difference of first two source_edges. With <2 source_edges, returns first or 0."""
    if len(source_edges) < 2: return source_edges[0] if source_edges else 0.0
    return abs(source_edges[0] - source_edges[1])

def f_first(source_edges: List[float]) -> float:
    """Identity on first source_edge. Returns 0.0 if no source_edges."""
    return source_edges[0] if source_edges else 0.0

def f_const_low(source_edges: List[float]) -> float:
    """Constant 0.2. Used for variables with no source_edges in the agent's hypothesis."""
    return 0.2

def f_const_high(source_edges: List[float]) -> float:
    """Constant 0.8. Used for variables with no source_edges in the agent's hypothesis."""
    return 0.8

def f_const_tiny(source_edges: List[float]) -> float:
    """Constant 0.1. Used as baseline for joint-false-trass test worlds where
    two vars are individually below the salience threshold but jointly tareth."""
    return 0.1

# Agent's available functions, addressed by name. The agent enumerates
# (source_edges, func) hypotheses using only these names.
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

def f_sin_gate(source_edges: List[float]) -> float:
    """Sinusoidal function of mean(source_edges). NOT in agent's library — used only
    by the hidden world to test vocabulary novelty detection. Returns 0.5
    when no source_edges (constant, won't trigger novelty in that case)."""
    if not source_edges: return 0.5
    s = sum(source_edges) / len(source_edges)
    return 0.5 + 0.5 * math.sin(2 * math.pi * s)

# Hidden world's function set: agent's library plus SIN. The world can
# generate variables using SIN; the agent cannot fit them and must report
# vocabulary novelty (instability streak) when this happens.
HIDDEN_FUNC_LIBRARY: Dict[str, Callable[[List[float]], float]] = dict(FUNC_LIBRARY)
HIDDEN_FUNC_LIBRARY["SIN"] = f_sin_gate
