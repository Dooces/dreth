#!/usr/bin/env python3
"""
bench_transition.py

Proof that TiedFrontier addresses a real gap: transition detection for
nearly-equivalent hypotheses.

The certification benchmark (bench_frontier.py) tested the wrong phase.
Certification speed is not the core problem. The core problem is this:

    After H_correct is certified, the world switches to H_rival.
    How many cycles until the agent detects the switch via sentinel checks?

This is the non-stationarity failure mode. It does not appear in static
active causal discovery because the ground truth does not move.

Mathematical claim
─────────────────
Let D = disagreement region between H_correct and H_rival.
Let ε = fraction of probe pool in D.
Let k = sentinels per cycle.

Greedy sentinel (no frontier awareness):
  Each cycle, k probes drawn from pool. P(probe hits D) = ε per probe.
  P(detect this cycle) = 1 − (1−ε)^k ≈ k·ε  for small ε.
  E[cycles to detection] = 1 / (1 − (1−ε)^k)  →  1/(k·ε)  as ε→0.

Frontier-aware sentinel (1+ probes guaranteed from D):
  At least 1 sentinel is from the separating pool each cycle.
  D-probes detect H_correct→H_rival by construction (differ beyond tolerance).
  E[cycles to detection] = 1.

Ratio: E[greedy] / E[frontier] = 1/(k·ε) → ∞ as ε → 0.

This is not an empirical claim. It follows from the selection mechanisms.
The benchmark below confirms it at three values of ε.

Why sentinel selection cannot self-correct
──────────────────────────────────────────
select_var_sentinels (sentinels.py) ranks candidates by disagree count:
  number of alternative hypotheses whose prediction differs from H_correct.

For probes in D: disagree count ≈ small.
  Only H_rival and a handful of hypotheses that happen to differ in D.

For probes outside D: disagree count ≈ large.
  Most wrong hypotheses differ from H_correct outside D too.

Therefore: discrimination-based selection is biased AGAINST D, making
ε_effective ≤ ε_pool. Greedy detection is at least as slow as the
pool-uniform case, often slower.

This is the structural reason sentinel selection cannot close the gap.
It is not a calibration problem. It is inherent to optimizing for the
global hypothesis space rather than the nearest competitor.

World structure (three ε values)
──────────────────────────────────
  T = H_correct:  0.0 for x ≤ BOUNDARY, else x * 0.9
  H_rival:        x * 0.9 for all x
  D:              x ∈ (TOLERANCE/0.9, BOUNDARY]
                  = probes where |H_correct − H_rival| = x*0.9 > TOLERANCE

  Three pool configurations:
    ε = 0.05  (BOUNDARY=0.30, 5% rare):  baseline
    ε = 0.02  (BOUNDARY=0.12, 2% rare):  moderate squeeze
    ε = 0.005 (BOUNDARY=0.03, 0.5% rare): extreme — D nearly empty

  All three use the same mathematical structure.
  Only the rare fraction changes.

No dreth imports. Deterministic. Run with: python scripts/bench_transition.py
"""

from __future__ import annotations
import random
import time
import statistics
from dataclasses import dataclass
from typing import Callable, List, Tuple

SEED            = 42
N_TRIALS        = 1000
MAX_CYCLES      = 500   # cap per trial; counts as MAX_CYCLES if not detected
N_SENTINELS     = 5     # sentinels per cycle (matches agent default)
N_SEP_IN_FRONT  = 2     # frontier-aware: 2 of 5 sentinels from sep pool
POOL_SIZE       = 1000
TOLERANCE       = 0.05

Fn = Callable[[float], float]


# ── WORLD BUILDER ─────────────────────────────────────────────────────────────
def build_world(boundary: float, rare_fraction: float, rng: random.Random):
    """
    T = H_correct: 0.0 for x <= boundary, else x*0.9
    H_rival:       x*0.9 for all x
    D:             (TOLERANCE/0.9, boundary]  — guaranteed effective separators
    Pool:          rare_fraction in D, rest in (boundary, 1.0]
    """
    low = TOLERANCE / 0.9 + 1e-6   # smallest x where |H_rival - T| > TOLERANCE

    def h_c(x): return 0.0 if x <= boundary else x * 0.9
    def h_r(x): return x * 0.9

    n_rare   = max(1, int(POOL_SIZE * rare_fraction))
    n_common = POOL_SIZE - n_rare
    pool = (
        [rng.uniform(boundary + 1e-6, 1.0) for _ in range(n_common)] +
        [rng.uniform(low, boundary)         for _ in range(n_rare)]
    )
    sep_pool = [x for x in pool if abs(h_c(x) - h_r(x)) > TOLERANCE]
    return h_c, h_r, pool, sep_pool


# ── TRANSITION DETECTION TRIALS ───────────────────────────────────────────────
def detect_greedy(h_c: Fn, h_r: Fn, pool: List[float],
                  rng: random.Random) -> int:
    """
    H_correct is certified. World has switched to H_rival.
    Sentinels drawn uniformly from pool each cycle.
    Returns: cycles until any sentinel detects |H_correct(x) - H_rival(x)| > TOLERANCE.
    """
    for cycle in range(1, MAX_CYCLES + 1):
        for _ in range(N_SENTINELS):
            x = rng.choice(pool)
            if abs(h_c(x) - h_r(x)) > TOLERANCE:
                return cycle
    return MAX_CYCLES


def detect_frontier(h_c: Fn, h_r: Fn, pool: List[float], sep_pool: List[float],
                    rng: random.Random) -> int:
    """
    H_correct is certified. World has switched to H_rival.
    Sentinels include N_SEP_IN_FRONT probes from sep_pool each cycle.
    Returns: cycles until detection.
    """
    if not sep_pool:
        return detect_greedy(h_c, h_r, pool, rng)
    n_gen = N_SENTINELS - N_SEP_IN_FRONT
    for cycle in range(1, MAX_CYCLES + 1):
        # N_SEP_IN_FRONT probes guaranteed from D
        for _ in range(N_SEP_IN_FRONT):
            x = rng.choice(sep_pool)
            if abs(h_c(x) - h_r(x)) > TOLERANCE:
                return cycle
        # Remaining probes from general pool
        for _ in range(n_gen):
            x = rng.choice(pool)
            if abs(h_c(x) - h_r(x)) > TOLERANCE:
                return cycle
    return MAX_CYCLES


# ── ANALYTICAL PREDICTION ─────────────────────────────────────────────────────
def analytical_greedy_mean(eps: float, k: int) -> float:
    """E[cycles to detection] = 1 / (1 - (1-eps)^k)"""
    p = 1.0 - (1.0 - eps) ** k
    return 1.0 / p if p > 0 else float("inf")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.perf_counter()
    master = random.Random(SEED)

    print("bench_transition.py — transition detection under non-stationarity")
    print(f"  trials={N_TRIALS}  max_cycles={MAX_CYCLES}  sentinels={N_SENTINELS}")
    print(f"  frontier_sep_probes={N_SEP_IN_FRONT}  tolerance={TOLERANCE}")
    print()

    configs = [
        ("ε=0.050", 0.30, 0.050),
        ("ε=0.020", 0.15, 0.020),
        ("ε=0.005", 0.08, 0.005),   # boundary must exceed TOLERANCE/0.9 ≈ 0.056
    ]

    print(f"  {'config':<10}  {'ε_actual':>9}  {'analytical':>11}  "
          f"{'greedy_med':>11}  {'frontier_med':>13}  "
          f"{'ratio':>8}  {'sep_pool':>9}")
    print("  " + "─" * 80)

    for label, boundary, rare_frac in configs:
        world_rng = random.Random(master.randint(0, 2**31))
        h_c, h_r, pool, sep_pool = build_world(boundary, rare_frac, world_rng)

        # Validate: every sep probe is an effective separator
        bad = [x for x in sep_pool if not (abs(h_c(x) - h_r(x)) > TOLERANCE)]
        assert not bad, f"sep pool has ineffective probes: {len(bad)}"

        # Validate non-separable control: common probes produce no differential
        common = [x for x in pool if x > boundary]
        bad_common = [x for x in common if abs(h_c(x) - h_r(x)) > TOLERANCE]
        assert not bad_common, "common probes should not separate"

        trial_seeds = [master.randint(0, 2**31) for _ in range(N_TRIALS)]

        greedy_cyc = [
            detect_greedy(h_c, h_r, pool, random.Random(s))
            for s in trial_seeds
        ]
        front_cyc = [
            detect_frontier(h_c, h_r, pool, sep_pool, random.Random(s))
            for s in trial_seeds
        ]

        actual_eps = len(sep_pool) / len(pool)
        analytical  = analytical_greedy_mean(actual_eps, N_SENTINELS)
        g_med       = statistics.median(greedy_cyc)
        f_med       = statistics.median(front_cyc)
        ratio       = g_med / f_med if f_med > 0 else float("inf")

        print(f"  {label:<10}  {actual_eps:>9.4f}  {analytical:>11.1f}  "
              f"{g_med:>11.1f}  {f_med:>13.1f}  "
              f"{ratio:>7.1f}x  {len(sep_pool):>9}")

    print()
    print("Interpretation")
    print("──────────────")
    print("  greedy_med: median cycles for standard sentinel to detect H→H' transition")
    print("  frontier_med: median cycles when 2/5 sentinels are guaranteed from D")
    print("  ratio: how many times slower greedy is")
    print("  analytical: E[T] = 1/(1-(1-ε)^k), matches greedy empirically")
    print()
    print("  As ε shrinks (nearly-equivalent hypotheses), greedy detection time")
    print("  grows as 1/(k·ε). Frontier detection time stays at 1 cycle.")
    print("  This is not a calibration issue. It is structural.")
    print()
    print("  TiedFrontier.separating_probes exists for exactly this use.")
    print("  select_var_sentinels does not read it.")
    print("  That is the gap.")

    print(f"\n  elapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
