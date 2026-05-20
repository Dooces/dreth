#!/usr/bin/env python3
"""
batch_run.py — run dreth CLI configurations concurrently and tabulate results.

Usage:
    python scripts/batch_run.py [options]

Examples:
    # default grid: vars=[5,8,12,20] x cycles=[50,150,300] x seeds=[42,7,99]
    python scripts/batch_run.py

    # custom grid
    python scripts/batch_run.py --vars 5,10,20 --cycles 100,500 --seeds 1,2,3

    # different schedule
    python scripts/batch_run.py --schedule periodic_shifts --vars 8,15 --cycles 200

    # cap concurrency (default = cpu count)
    python scripts/batch_run.py --workers 4

    # write raw output lines to file
    python scripts/batch_run.py --out results.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RunConfig:
    n_vars: int
    cycles: int
    seed: int
    schedule: str
    settle_cycles: int
    noise_sigma: float
    extra: tuple  # passthrough args


@dataclass
class RunResult:
    config: RunConfig
    elapsed: float
    returncode: int
    # parsed from quiet output
    recorded_cycles: Optional[int]
    vars_visible: Optional[int]
    skip_pct: Optional[float]
    trass_skips: Optional[int]
    sentinel_skips: Optional[int]
    compression_skips: Optional[int]
    full_audits: Optional[int]
    interventions: Optional[int]
    drift_detected: Optional[int]
    certified: Optional[int]
    trass_status: Optional[int]
    true_missing: Optional[int]
    raw: str


def _run_one(cfg: RunConfig) -> RunResult:
    cmd = [
        sys.executable, "-m", "dreth.cli",
        "--n-vars",       str(cfg.n_vars),
        "--cycles",       str(cfg.cycles),
        "--seed",         str(cfg.seed),
        "--schedule",     cfg.schedule,
        "--settle-cycles", str(cfg.settle_cycles),
        "--noise-sigma",  str(cfg.noise_sigma),
        "--quiet",
    ]
    for arg in cfg.extra:
        cmd.append(arg)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    raw = proc.stdout + proc.stderr

    def find(pattern, text, group=1, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(group)) if m else None

    recorded_cycles = find(r"requested=\d+ recorded=(\d+)", raw, cast=int)
    vars_visible    = find(r"vars=(\d+)/\d+", raw, cast=int)
    skip_pct        = find(r"skips=\d+/\d+ \(([\d.]+)%\)", raw)
    trass_skips     = find(r"trass=(\d+) sentinel=", raw, cast=int)
    sentinel_skips  = find(r"sentinel=(\d+) compression=", raw, cast=int)
    compression_skips = find(r"compression=(\d+);", raw, cast=int)
    full_audits     = find(r"full_audits=(\d+)", raw, cast=int)
    interventions   = find(r"interventions=(\d+)", raw, cast=int)
    drift_detected  = find(r"localized=(\d+)/", raw, cast=int)
    certified       = find(r"certified=(\d+)", raw, cast=int)
    trass_status    = find(r"trass=(\d+) authoritative=", raw, cast=int)
    true_missing    = find(r"true_missing=(\d+)", raw, cast=int)

    return RunResult(
        config=cfg,
        elapsed=elapsed,
        returncode=proc.returncode,
        recorded_cycles=recorded_cycles,
        vars_visible=vars_visible,
        skip_pct=skip_pct,
        trass_skips=trass_skips,
        sentinel_skips=sentinel_skips,
        compression_skips=compression_skips,
        full_audits=full_audits,
        interventions=interventions,
        drift_detected=drift_detected,
        certified=certified,
        trass_status=trass_status,
        true_missing=true_missing,
        raw=raw,
    )


def _fmt_row(r: RunResult) -> str:
    cfg = r.config
    ok = "OK" if r.returncode == 0 else f"ERR({r.returncode})"
    skip = f"{r.skip_pct:.1f}%" if r.skip_pct is not None else "?"
    trass = r.trass_skips if r.trass_skips is not None else "?"
    sent  = r.sentinel_skips if r.sentinel_skips is not None else "?"
    comp  = r.compression_skips if r.compression_skips is not None else "?"
    iv    = r.interventions if r.interventions is not None else "?"
    drift = r.drift_detected if r.drift_detected is not None else "?"
    miss  = r.true_missing if r.true_missing is not None else "?"
    cyc   = r.recorded_cycles if r.recorded_cycles is not None else "?"
    return (
        f"  n={cfg.n_vars:3d} cyc={cfg.cycles:4d} seed={cfg.seed:5d} "
        f"sched={cfg.schedule:16s} "
        f"| {ok:8s} {r.elapsed:5.1f}s "
        f"| skip={skip:6s} trass={trass:5} sent={sent:5} comp={comp:4} "
        f"| iv={iv:6} audits={r.full_audits or '?':5} drift={drift:4} miss={miss:4} "
        f"| rec={cyc}"
    )


def main():
    p = argparse.ArgumentParser(description="Concurrent dreth batch runner")
    p.add_argument("--vars",    default="5,8,12,20",
                   help="comma-separated n-vars values (default: 5,8,12,20)")
    p.add_argument("--cycles",  default="50,150,300",
                   help="comma-separated cycle counts (default: 50,150,300)")
    p.add_argument("--seeds",   default="42,7,99",
                   help="comma-separated seeds (default: 42,7,99)")
    p.add_argument("--schedule", default="incremental",
                   choices=["incremental", "periodic_shifts", "novelty", "shaped"],
                   help="mutation schedule (default: incremental)")
    p.add_argument("--settle-cycles", type=int, default=8,
                   help="settle cycles between reveals for incremental (default: 8)")
    p.add_argument("--noise-sigma", type=float, default=0.02,
                   help="noise sigma (default: 0.02)")
    p.add_argument("--workers", type=int, default=None,
                   help="max parallel workers (default: cpu count)")
    p.add_argument("--out", default=None,
                   help="write one JSON line per run to this file")
    p.add_argument("--verbose", action="store_true",
                   help="print full CLI output for each run after the table")
    args, extra = p.parse_known_args()

    var_list    = [int(x) for x in args.vars.split(",")]
    cycle_list  = [int(x) for x in args.cycles.split(",")]
    seed_list   = [int(x) for x in args.seeds.split(",")]

    configs = [
        RunConfig(n_vars=v, cycles=c, seed=s,
                  schedule=args.schedule,
                  settle_cycles=args.settle_cycles,
                  noise_sigma=args.noise_sigma,
                  extra=tuple(extra))
        for v in var_list
        for c in cycle_list
        for s in seed_list
    ]

    total = len(configs)
    print(f"dreth batch: {total} runs | "
          f"vars={var_list} cycles={cycle_list} seeds={seed_list} "
          f"schedule={args.schedule}", flush=True)
    print(f"  workers={args.workers or 'cpu'} settle={args.settle_cycles} "
          f"noise={args.noise_sigma}", flush=True)
    print()

    header = (
        f"  {'n':>3}  {'cyc':>4}  {'seed':>5}  {'schedule':16s}  "
        f"  {'st':8s} {'t':>5}  "
        f"  {'skip%':>6} {'trass':>5} {'sent':>5} {'comp':>4}  "
        f"  {'iv':>6} {'auds':>5} {'drft':>4} {'miss':>4}  "
        f"  rec"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    done = 0
    out_fh = open(args.out, "w") if args.out else None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[{done:3d}/{total}] {_fmt_row(r)}", flush=True)
            if out_fh:
                rec = asdict(r)
                rec.pop("raw")  # keep jsonl compact
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()

    if out_fh:
        out_fh.close()

    print()
    print("── aggregate ──────────────────────────────────────────")
    ok_runs = [r for r in results if r.returncode == 0]
    if ok_runs:
        avg_skip = sum(r.skip_pct for r in ok_runs if r.skip_pct) / len(ok_runs)
        avg_iv   = sum(r.interventions for r in ok_runs if r.interventions) / len(ok_runs)
        avg_miss = sum(r.true_missing for r in ok_runs if r.true_missing is not None) / len(ok_runs)
        print(f"  ok={len(ok_runs)}/{total}  avg_skip%={avg_skip:.1f}  "
              f"avg_iv={avg_iv:.0f}  avg_true_missing={avg_miss:.1f}")

    if args.verbose:
        print()
        print("── full output ─────────────────────────────────────────")
        for r in sorted(results, key=lambda r: (r.config.n_vars, r.config.cycles, r.config.seed)):
            cfg = r.config
            print(f"\n{'='*60}")
            print(f"n={cfg.n_vars} cyc={cfg.cycles} seed={cfg.seed} sched={cfg.schedule}")
            print(r.raw)


if __name__ == "__main__":
    main()
