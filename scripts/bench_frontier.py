#!/usr/bin/env python3
"""
bench_frontier.py

Standalone, deterministic benchmark for TiedFrontier separating-probe value.

Question: can frontier-guided separating probes certify the correct hypothesis
faster than a greedy baseline on a rare-separator world (World A), without
causing false certification on a non-separable world (World B)?

No dreth imports. No edits to agent.py, fit.py, or ledger.py.
Run with:  python scripts/bench_frontier.py

Two synthetic worlds
────────────────────
World A — rare separator
  T(x)      = 0.0 if x ≤ 0.3 else x · 0.9
  H_correct = T
  H_rival   = x · 0.9  (ignores the zero-region)
  Pool      = 95 % x ∈ (0.30, 1.0],  5 % x ∈ (0.056, 0.30]
  Separator = any x > 0.056 in the rare region:
              |H_rival(x) − T(x)| = x·0.9 > 0.05 = TOLERANCE → score differential

World B — non-separable
  T = H_correct = H_rival = x · 0.9  (algebraically identical)
  Pool = uniform over [0, 1]
  No probe can produce a score differential; neither hypothesis can certify.

Conditions
──────────
  greedy   — draw uniformly from pool; no tie awareness
  frontier — when gap < CERT_MARGIN, prefer probes from the
             pre-computed separating pool (inputs where |H_c−H_r| > SEP_GAP)

Win criteria
────────────
  World A: frontier cert_rate ≥ greedy + 0.20
           AND median probes (imputed) reduction ≥ 30 %
           AND sep hit rate ≥ 0.80
  World B: neither condition produces any certification  (< 2 % tolerance)
"""

from __future__ import annotations
import random
import time
import statistics
from dataclasses import dataclass
from typing import Callable, List, Optional

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SEED          = 42
N_TRIALS      = 500
PROBE_BUDGET  = 200
TOLERANCE     = 0.05    # |h(x) − T(x)| ≤ TOLERANCE → match
CERT_MARGIN   = 8       # must lead rival by this many matches to certify
MIN_SCORE     = 15      # must have ≥ this matches before certifying
SEP_GAP       = TOLERANCE  # |h_c(x) − h_r(x)| > SEP_GAP → sep probe candidate
POOL_SIZE     = 500     # probe pool size (built once, shared between conditions)
RARE_FRACTION = 0.05    # fraction of World A pool in the separator region

# TIE_DELTA = CERT_MARGIN - 1: "in tie" ≡ "not yet decided" ≡ gap < CERT_MARGIN
# Written here as documentation; used below as the literal expression.
TIE_DELTA = CERT_MARGIN - 1

Fn = Callable[[float], float]


# ── SCORING ───────────────────────────────────────────────────────────────────
def matches(h: Fn, truth: Fn, x: float) -> bool:
    return abs(h(x) - truth(x)) <= TOLERANCE


def check_cert(sc: int, sr: int) -> Optional[str]:
    if sc >= MIN_SCORE and sc - sr >= CERT_MARGIN:
        return "correct"
    if sr >= MIN_SCORE and sr - sc >= CERT_MARGIN:
        return "rival"
    return None


# ── SEP PROBE POOL ─────────────────────────────────────────────────────────────
def build_sep_pool(h_c: Fn, h_r: Fn, pool: List[float]) -> List[float]:
    """Inputs from pool where |h_c(x) − h_r(x)| > SEP_GAP."""
    return [x for x in pool if abs(h_c(x) - h_r(x)) > SEP_GAP]


# ── TRIAL RESULT ──────────────────────────────────────────────────────────────
@dataclass
class Trial:
    certified: Optional[str]  # 'correct' | 'rival' | None
    probes: int
    sep_hits: int    # sep probes that produced an actual score differential
    sep_misses: int  # sep probes that produced no differential
    waste: int       # non-sep probes drawn while gap < CERT_MARGIN (greedy only)


# ── GREEDY CONDITION ──────────────────────────────────────────────────────────
def run_greedy(
    truth: Fn, h_c: Fn, h_r: Fn,
    pool: List[float],
    rng: random.Random,
) -> Trial:
    sc = sr = waste = 0
    for i in range(PROBE_BUDGET):
        was_undecided = abs(sc - sr) < CERT_MARGIN
        x = rng.choice(pool)
        hit_c = matches(h_c, truth, x)
        hit_r = matches(h_r, truth, x)
        sc += hit_c
        sr += hit_r
        if was_undecided and hit_c == hit_r:
            waste += 1
        cert = check_cert(sc, sr)
        if cert:
            return Trial(cert, i + 1, 0, 0, waste)
    return Trial(None, PROBE_BUDGET, 0, 0, waste)


# ── FRONTIER CONDITION ────────────────────────────────────────────────────────
def run_frontier(
    truth: Fn, h_c: Fn, h_r: Fn,
    pool: List[float],
    sep_pool: List[float],
    rng: random.Random,
) -> Trial:
    sc = sr = sep_hits = sep_misses = 0
    for i in range(PROBE_BUDGET):
        in_tie = abs(sc - sr) < CERT_MARGIN  # = gap ≤ TIE_DELTA
        if in_tie and sep_pool:
            x = rng.choice(sep_pool)
            hit_c = matches(h_c, truth, x)
            hit_r = matches(h_r, truth, x)
            if hit_c != hit_r:
                sep_hits += 1
            else:
                sep_misses += 1
        else:
            x = rng.choice(pool)
            hit_c = matches(h_c, truth, x)
            hit_r = matches(h_r, truth, x)
        sc += hit_c
        sr += hit_r
        cert = check_cert(sc, sr)
        if cert:
            return Trial(cert, i + 1, sep_hits, sep_misses, 0)
    return Trial(None, PROBE_BUDGET, sep_hits, sep_misses, 0)


# ── WORLD BUILDERS ────────────────────────────────────────────────────────────
def build_world_a(rng: random.Random):
    """
    Rare-separator world.

    Lower bound 0.057 for the rare region ensures x·0.9 > 0.05 = TOLERANCE
    for every rare probe, so all rare inputs are guaranteed effective separators.
    Common region (0.3, 1.0]: both hypotheses agree ⇒ no differential.
    """
    def truth(x): return 0.0 if x <= 0.3 else x * 0.9
    def h_c(x):   return 0.0 if x <= 0.3 else x * 0.9
    def h_r(x):   return x * 0.9

    n_rare   = int(POOL_SIZE * RARE_FRACTION)
    n_common = POOL_SIZE - n_rare
    pool = (
        [rng.uniform(0.301, 1.00) for _ in range(n_common)] +
        [rng.uniform(0.057, 0.30) for _ in range(n_rare)]
    )
    return truth, h_c, h_r, pool


def build_world_b(rng: random.Random):
    """
    Non-separable world.

    H_correct and H_rival are algebraically identical — (x+x)·0.45 = x·0.9.
    |h_c(x) − h_r(x)| = 0 everywhere; sep_pool will be empty.
    Neither hypothesis can ever earn a score lead; certification is impossible.
    """
    def truth(x): return x * 0.9
    def h_c(x):   return x * 0.9
    def h_r(x):   return (x + x) * 0.45

    pool = [rng.uniform(0.0, 1.0) for _ in range(POOL_SIZE)]
    return truth, h_c, h_r, pool


# ── VALIDATION (failure-mode checks before running trials) ─────────────────────
def validate(
    truth_a: Fn, h_c_a: Fn, h_r_a: Fn, pool_a: List[float], sep_pool_a: List[float],
    truth_b: Fn, h_c_b: Fn, h_r_b: Fn, pool_b: List[float], sep_pool_b: List[float],
) -> bool:
    print("\n── VALIDATION ───────────────────────────────────────────────────────")
    ok = True

    # F1 — pool composition (World A)
    rare_a   = [x for x in pool_a if x <= 0.30]
    common_a = [x for x in pool_a if x > 0.30]
    frac = len(rare_a) / len(pool_a)
    f1 = 0.03 <= frac <= 0.08
    print(f"  F1 pool split:        {len(common_a)} common / {len(rare_a)} rare "
          f"({frac:.1%})  {'OK' if f1 else 'WARN: expected ~5%'}")
    ok = ok and f1

    # F4 — every rare probe is an effective separator (score differential guaranteed)
    bad_rare = [x for x in rare_a if not (matches(h_c_a, truth_a, x) != matches(h_r_a, truth_a, x))]
    f4 = len(bad_rare) == 0
    print(f"  F4 rare eff. sep:     {len(rare_a) - len(bad_rare)}/{len(rare_a)} produce "
          f"score differential  {'OK' if f4 else 'WARN: some rare probes non-discriminating'}")
    ok = ok and f4

    # sep pool coverage
    sp_ok = len(sep_pool_a) > 0
    print(f"  sep pool (World A):   {len(sep_pool_a)} probes  {'OK' if sp_ok else 'FAIL: empty'}")
    ok = ok and sp_ok

    # F3 — World B hypotheses numerically identical
    max_diff_b = max(abs(h_c_b(x) - h_r_b(x)) for x in pool_b)
    f3 = max_diff_b < 1e-12
    print(f"  F3 World B identity:  max|h_c−h_r| = {max_diff_b:.2e}  {'OK' if f3 else 'FAIL'}")
    ok = ok and f3

    # sep pool empty for World B
    sp_b_ok = len(sep_pool_b) == 0
    print(f"  sep pool (World B):   {len(sep_pool_b)} probes  {'OK: empty (non-separable)' if sp_b_ok else 'FAIL: should be empty'}")
    ok = ok and sp_b_ok

    if not ok:
        print("  VALIDATION FAILED — results may be invalid; see warnings above")
    else:
        print("  all checks pass")
    return ok


# ── SUMMARIZE TRIALS ──────────────────────────────────────────────────────────
def summarize(label: str, results: List[Trial]) -> dict:
    n = len(results)
    cert_c = [r for r in results if r.certified == "correct"]
    cert_r = [r for r in results if r.certified == "rival"]

    cert_rate       = len(cert_c) / n
    false_cert_rate = len(cert_r) / n

    cert_probes = [r.probes for r in cert_c]
    median_p_cert = statistics.median(cert_probes) if cert_probes else None
    mean_p_cert   = statistics.mean(cert_probes)   if cert_probes else None

    # Imputed: non-certified trials count as PROBE_BUDGET
    all_probes          = [r.probes if r.certified == "correct" else PROBE_BUDGET
                           for r in results]
    median_p_imputed    = statistics.median(all_probes)

    sep_tot   = sum(r.sep_hits + r.sep_misses for r in results)
    sep_hits  = sum(r.sep_hits for r in results)
    sep_hr    = sep_hits / sep_tot if sep_tot > 0 else None

    waste_vals   = [r.waste for r in results]
    median_waste = statistics.median(waste_vals)

    return dict(
        label=label, n=n,
        cert_rate=cert_rate, false_cert_rate=false_cert_rate,
        median_probes_cert=median_p_cert, mean_probes_cert=mean_p_cert,
        median_probes_imputed=median_p_imputed,
        sep_hit_rate=sep_hr, sep_total=sep_tot,
        median_waste=median_waste,
    )


def print_summary(s: dict):
    print(f"\n  {s['label']}")
    print(f"    cert_rate (correct):      {s['cert_rate']:.3f}")
    print(f"    false_cert_rate (rival):  {s['false_cert_rate']:.3f}")
    mp_c = (f"{s['median_probes_cert']:.0f}"
            if s['median_probes_cert'] is not None else "N/A (none certified)")
    me_c = (f"{s['mean_probes_cert']:.1f}"
            if s['mean_probes_cert'] is not None else "N/A")
    print(f"    median probes (cert-only):  {mp_c}")
    print(f"    mean   probes (cert-only):  {me_c}")
    print(f"    median probes (imputed):    {s['median_probes_imputed']:.0f}")
    shr  = f"{s['sep_hit_rate']:.3f}" if s['sep_hit_rate'] is not None else "N/A"
    print(f"    sep hit rate:               {shr}  (n_sep={s['sep_total']})")
    print(f"    median waste probes:        {s['median_waste']:.1f}")


# ── VERDICT ───────────────────────────────────────────────────────────────────
def verdict(wa_g: dict, wa_f: dict, wb_g: dict, wb_f: dict) -> bool:
    print("\n── VERDICT ──────────────────────────────────────────────────────────")

    cr_gain = wa_f["cert_rate"] - wa_g["cert_rate"]
    mp_g    = wa_g["median_probes_imputed"]
    mp_f    = wa_f["median_probes_imputed"]
    probe_reduction = (mp_g - mp_f) / mp_g if mp_g else 0.0

    shr        = wa_f["sep_hit_rate"]
    sep_ok     = shr is not None and shr >= 0.80

    wb_any_g = wb_g["cert_rate"] + wb_g["false_cert_rate"]
    wb_any_f = wb_f["cert_rate"] + wb_f["false_cert_rate"]
    world_b_safe = wb_any_g < 0.02 and wb_any_f < 0.02

    a_cert_pass  = cr_gain >= 0.20
    a_probe_pass = probe_reduction >= 0.30

    def ok(flag): return "PASS" if flag else "FAIL"
    print(f"\n  [World A] cert_rate:  greedy={wa_g['cert_rate']:.3f}  "
          f"frontier={wa_f['cert_rate']:.3f}  gain={cr_gain:+.3f}  "
          f"{ok(a_cert_pass)} (need ≥+0.20)")
    print(f"  [World A] med probes (imputed): greedy={mp_g:.0f}  "
          f"frontier={mp_f:.0f}  reduction={probe_reduction:.1%}  "
          f"{ok(a_probe_pass)} (need ≥30%)")
    print(f"  [World A] sep hit rate:  {shr if shr is not None else 'N/A'}  "
          f"{ok(sep_ok)} (need ≥0.80)")
    print(f"  [World B] any cert rate: greedy={wb_any_g:.3f}  "
          f"frontier={wb_any_f:.3f}  {ok(world_b_safe)} (both need <0.02)")

    overall = a_cert_pass and a_probe_pass and sep_ok and world_b_safe
    banner  = "FRONTIER EARNS NEXT STEP" if overall else "DO NOT WIRE INTO AGENT"
    print(f"\n  ══ {banner} ══")
    return overall


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.perf_counter()

    print("bench_frontier.py")
    print(f"  trials={N_TRIALS}  budget={PROBE_BUDGET}  seed={SEED}")
    print(f"  tolerance={TOLERANCE}  cert_margin={CERT_MARGIN}  min_score={MIN_SCORE}")
    print(f"  sep_gap={SEP_GAP}  rare_fraction={RARE_FRACTION}  pool_size={POOL_SIZE}")

    master = random.Random(SEED)

    # Worlds built once; pool is shared between greedy and frontier (controlled var)
    truth_a, h_c_a, h_r_a, pool_a = build_world_a(random.Random(master.randint(0, 2**31)))
    sep_pool_a = build_sep_pool(h_c_a, h_r_a, pool_a)

    truth_b, h_c_b, h_r_b, pool_b = build_world_b(random.Random(master.randint(0, 2**31)))
    sep_pool_b = build_sep_pool(h_c_b, h_r_b, pool_b)

    valid = validate(
        truth_a, h_c_a, h_r_a, pool_a, sep_pool_a,
        truth_b, h_c_b, h_r_b, pool_b, sep_pool_b,
    )
    if not valid:
        print("\n  Aborting: fix validation failures before reading results.")
        return

    # Same trial seeds for both conditions: reproducible, fair population comparison
    trial_seeds = [master.randint(0, 2**31) for _ in range(N_TRIALS)]

    # ── World A ──────────────────────────────────────────────────────────────
    print("\n── WORLD A (rare separator) ─────────────────────────────────────────")
    wa_g = [run_greedy(truth_a, h_c_a, h_r_a, pool_a, random.Random(s))
            for s in trial_seeds]
    wa_f = [run_frontier(truth_a, h_c_a, h_r_a, pool_a, sep_pool_a, random.Random(s))
            for s in trial_seeds]
    s_wa_g = summarize("World A / greedy",   wa_g)
    s_wa_f = summarize("World A / frontier", wa_f)
    print_summary(s_wa_g)
    print_summary(s_wa_f)

    # ── World B ──────────────────────────────────────────────────────────────
    print("\n── WORLD B (non-separable) ──────────────────────────────────────────")
    wb_g = [run_greedy(truth_b, h_c_b, h_r_b, pool_b, random.Random(s))
            for s in trial_seeds]
    wb_f = [run_frontier(truth_b, h_c_b, h_r_b, pool_b, sep_pool_b, random.Random(s))
            for s in trial_seeds]
    s_wb_g = summarize("World B / greedy",   wb_g)
    s_wb_f = summarize("World B / frontier", wb_f)
    print_summary(s_wb_g)
    print_summary(s_wb_f)

    verdict(s_wa_g, s_wa_f, s_wb_g, s_wb_f)

    print(f"\n  elapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
