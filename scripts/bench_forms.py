#!/usr/bin/env python3
"""Paired forms-off/forms-on benchmark harness.

Verdict logic — forms are judged on two joint signals:
  help    : comp_hits_delta > 0  AND  full_audit_delta < 0
            (compressions up, audit work down — the ideal outcome)
  hurt    : comp_hits_delta < 0  AND  full_audit_delta > 0
            (compression degraded, audit work rose — the bad outcome)
  neutral : neither clear pattern (mixed signals or no change)

iv_Δ/c/v is retained as a secondary efficiency metric but is NOT the verdict.

Predictive-failure (VarNethra) columns — from the new `predict:` summary line:
  λ_avg   : average poisson_rate across tareth vars with drift history (forms-on run)
  watch_Q : total "queued (parent X in watch state)" pre-audits fired (forms-on run)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RAW_FIELDS = [
    "seed", "n_vars", "cycles", "mode", "forms", "cost_weight_mode",
    "iv", "iv_per_cycle",
    "full_audits", "skip_rate",
    "comp_stored", "comp_hits_life", "comp_misses_life", "amortization",
    "forms_active", "forms_total", "form_instances", "form_collapse_events",
    "auth_num", "auth_den", "env_cert_num", "env_cert_den",
    "drift_fp", "novelty_fp", "defer_total",
    # predictive-failure fields (from `predict:` summary line)
    "watch_vars", "avg_lambda", "watch_queued", "form_sibling_watches",
    "cooldown_blocks", "gated_pairs",
    # wall-clock timing
    "elapsed_ms", "ms_per_cycle",
    # form hypothesis shortcut (from `form_hyp:` summary line)
    "form_hyp_shortcuts",
    # cycles-to-coverage (from `coverage:` summary line)
    "cov_n", "cov_first", "cov_p50", "cov_last",
]

PAIR_FIELDS = [
    "iv_delta", "iv_delta_pct",
    "iv_pcpv_off", "iv_pcpv_on", "iv_pcpv_delta",
    "full_audit_delta",
    "drift_fp_delta", "novelty_fp_delta",
    "comp_hits_delta", "comp_hits_delta_pct",
    "amortization_delta",
    "watch_queued_delta", "avg_lambda_delta",
    # wall-clock timing pair fields
    "ms_pcyc_off", "ms_pcyc_on", "time_delta_pct",
    # cycles-to-coverage pair field
    "cov_p50_delta",
    # defer delta (suppress/churn diagnostic)
    "defer_delta",
    "verdict",
]

EXTRA_FIELDS = [
    "row_type", "parse_error", "command", "returncode",
    "forms_off_iv", "forms_on_iv", "best_case", "worst_case",
    "total_pairs", "wins", "losses", "neutral",
    "median_comp_hits_delta", "median_full_audit_delta",
    "median_iv_delta_pct", "median_iv_pcpv_delta",
]

CSV_FIELDS = EXTRA_FIELDS + RAW_FIELDS + PAIR_FIELDS


def parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_cost_weight_modes(raw: str) -> List[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    for m in modes:
        if m not in ("uniform", "mixed"):
            raise argparse.ArgumentTypeError(f"unknown cost_weight_mode: {m!r}; choose uniform or mixed")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("3,20,42"))
    parser.add_argument("--n-vars", type=parse_int_list, default=parse_int_list("15,25,35"))
    parser.add_argument("--cycles", type=parse_int_list, default=parse_int_list("500,1000,2000,3000"))
    parser.add_argument("--mode", default="v29")
    parser.add_argument("--out", default="bench_forms.csv")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument(
        "--cost-weight-modes",
        type=parse_cost_weight_modes,
        default=["uniform"],
        metavar="MODES",
        help="comma-separated list of cost_weight modes: uniform,mixed (default: uniform)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="maximum subprocesses to run in parallel",
    )
    return parser.parse_args()


def search(pattern: str, text: str, flags: int = re.MULTILINE) -> Optional[re.Match[str]]:
    return re.search(pattern, text, flags)


def to_int(value: str) -> int:
    return int(value.replace(",", ""))


def to_float_or_none(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value == "n/a":
        return None
    return float(value.rstrip("x"))


def require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = search(pattern, text)
    if match is None:
        raise ValueError(f"missing {label}")
    return match


def parse_stdout(stdout: str, base: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(base)

    auth = require(r"^\s+cyc=(\d+)\s+vars=(\d+).*?\|\s+auth=(\d+)/(\d+)", stdout, "summary/auth")
    row["auth_num"] = to_int(auth.group(3))
    row["auth_den"] = to_int(auth.group(4))

    comp = require(
        r"^\s+comp:\s+stored=(\d+)\s+hit/miss\s+live=(\d+)/(\d+)\s+life=(\d+)/(\d+)",
        stdout,
        "compression summary",
    )
    row["comp_stored"] = to_int(comp.group(1))
    row["comp_hits_life"] = to_int(comp.group(4))
    row["comp_misses_life"] = to_int(comp.group(5))

    env = require(r"^\s+env:\s+cert=(\d+)/(\d+)\s+oob=(\d+)", stdout, "environment summary")
    row["env_cert_num"] = to_int(env.group(1))
    row["env_cert_den"] = to_int(env.group(2))

    audit = require(
        r"^\s+audit:\s+full=(\d+)\s+skip=\d+/\d+\s+\(([0-9.]+)%\)",
        stdout,
        "audit summary",
    )
    row["full_audits"] = to_int(audit.group(1))
    row["skip_rate"] = float(audit.group(2))

    iv = require(r"^\s+iv=(\d+)", stdout, "intervention count")
    row["iv"] = to_int(iv.group(1))
    row["iv_per_cycle"] = row["iv"] / max(1, int(row["cycles"]))

    drift = require(r"^\s+drift any:\s+TP=\d+\s+FN=\d+\s+FP=(\d+)\s+TN=\d+", stdout, "drift FP")
    row["drift_fp"] = to_int(drift.group(1))

    novelty = require(r"^\s+novelty att:\s+TP=\d+\s+FN=\d+\s+FP=(\d+)\s+TN=\d+", stdout, "novelty FP")
    row["novelty_fp"] = to_int(novelty.group(1))

    defer = require(r"^\s+defer:\s+total=(\d+)", stdout, "defer total")
    row["defer_total"] = to_int(defer.group(1))

    amort = search(r"amortization:\s+([0-9.]+x|n/a)", stdout)
    row["amortization"] = to_float_or_none(amort.group(1)) if amort else None

    forms_summary = search(r"forms:\s+(\d+)\s+active,\s+(\d+)\s+total", stdout)
    if forms_summary:
        row["forms_active"] = to_int(forms_summary.group(1))
        row["forms_total"] = to_int(forms_summary.group(2))
    elif search(r"forms:\s+none active", stdout):
        row["forms_active"] = 0
        row["forms_total"] = 0
    else:
        row["forms_active"] = 0
        row["forms_total"] = 0

    row["form_instances"] = sum(
        to_int(match)
        for match in re.findall(r"\binstances=(\d+)", stdout)
    )
    row["form_collapse_events"] = len(
        re.findall(r"^\s+c\d+:\s+.*(?:unbound|RETIRED)", stdout, re.MULTILINE)
    )

    # predictive-failure metrics (from `predict:` summary line)
    predict = search(
        r"^\s+predict:\s+watch=(\d+)/\d+\s+λ_tracked=\d+\s+avg_λ=([0-9.]+)"
        r"\s+queued=(\d+)\s+form_sib=(\d+)"
        r"(?:\s+cd_blocks=(\d+)\s+gated=(\d+))?",
        stdout,
    )
    if predict:
        row["watch_vars"] = to_int(predict.group(1))
        row["avg_lambda"] = float(predict.group(2))
        row["watch_queued"] = to_int(predict.group(3))
        row["form_sibling_watches"] = to_int(predict.group(4))
        row["cooldown_blocks"] = to_int(predict.group(5)) if predict.group(5) else 0
        row["gated_pairs"] = to_int(predict.group(6)) if predict.group(6) else 0
    else:
        row["watch_vars"] = 0
        row["avg_lambda"] = 0.0
        row["watch_queued"] = 0
        row["form_sibling_watches"] = 0
        row["cooldown_blocks"] = 0
        row["gated_pairs"] = 0

    cov = search(r"coverage:\s+n=(\d+)\s+first=(\d+)\s+p50=(\d+)\s+last=(\d+)", stdout)
    if cov:
        row["cov_n"] = to_int(cov.group(1))
        row["cov_first"] = to_int(cov.group(2))
        row["cov_p50"] = to_int(cov.group(3))
        row["cov_last"] = to_int(cov.group(4))
    else:
        row["cov_n"] = 0
        row["cov_first"] = 0
        row["cov_p50"] = 0
        row["cov_last"] = 0

    hyp = search(r"form_hyp:\s+shortcuts=(\d+)", stdout)
    row["form_hyp_shortcuts"] = to_int(hyp.group(1)) if hyp else 0

    row["parse_error"] = ""
    return row


def make_cost_weights(n_vars: int, seed: int) -> str:
    """Deterministic mixed weights: ~20% high (3.0), ~20% low (0.3), rest 1.0.
    Uses a seed offset so weight assignments don't correlate with world structure."""
    rng = random.Random(seed ^ 0xDEADBEEF)
    weights = []
    for _ in range(n_vars):
        r = rng.random()
        if r < 0.20:
            weights.append("3.0")
        elif r < 0.40:
            weights.append("0.3")
        else:
            weights.append("1.0")
    return ",".join(weights)


def build_command(
    seed: int, n_vars: int, cycles: int, mode: str, forms: bool,
    cost_weights_str: Optional[str] = None,
) -> List[str]:
    cmd = [
        "python",
        "dreth_causal_v28.py",
        "--seed", str(seed),
        "--n-vars", str(n_vars),
        "--cycles", str(cycles),
        "--mode", mode,
    ]
    if not forms:
        cmd.append("--disable-forms")
    if cost_weights_str:
        cmd += ["--cost-weights", cost_weights_str]
    return cmd


def run_one(
    seed: int, n_vars: int, cycles: int, mode: str, forms: bool,
    cost_weight_mode: str = "uniform",
) -> Dict[str, Any]:
    cost_weights_str = make_cost_weights(n_vars, seed) if cost_weight_mode == "mixed" else None
    cmd = build_command(seed, n_vars, cycles, mode, forms, cost_weights_str)
    base: Dict[str, Any] = {
        "row_type": "raw",
        "seed": seed,
        "n_vars": n_vars,
        "cycles": cycles,
        "mode": mode,
        "forms": int(forms),
        "cost_weight_mode": cost_weight_mode,
        "command": " ".join(cmd),
    }
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except OSError as exc:
        row = dict(base)
        row["returncode"] = ""
        row["parse_error"] = f"exec failed: {exc}"
        return row

    base["returncode"] = proc.returncode
    base["elapsed_ms"] = round(elapsed_ms, 1)
    base["ms_per_cycle"] = round(elapsed_ms / max(1, int(base["cycles"])), 3)
    if proc.returncode != 0:
        row = dict(base)
        row["parse_error"] = f"returncode={proc.returncode}: {proc.stderr.strip()[:500]}"
        return row

    try:
        return parse_stdout(proc.stdout, base)
    except Exception as exc:
        row = dict(base)
        row["parse_error"] = f"{exc}; command={' '.join(cmd)}"
        return row


def numeric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None):
        return default
    return float(value)


def pair_rows(off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "row_type": "pair",
        "seed": off["seed"],
        "n_vars": off["n_vars"],
        "cycles": off["cycles"],
        "mode": off["mode"],
        "forms": "paired",
        "cost_weight_mode": off.get("cost_weight_mode", "uniform"),
        "forms_off_iv": off.get("iv", ""),
        "forms_on_iv": on.get("iv", ""),
    }
    errors = [err for err in (off.get("parse_error"), on.get("parse_error")) if err]
    if errors:
        row["parse_error"] = " | ".join(errors)
        row["verdict"] = "parse_error"
        return row

    iv_off = numeric(off, "iv")
    iv_on = numeric(on, "iv")
    iv_delta = iv_on - iv_off
    iv_delta_pct = 100.0 * iv_delta / iv_off if iv_off else 0.0
    row["iv_delta"] = int(iv_delta)
    row["iv_delta_pct"] = iv_delta_pct

    n_vars = int(off.get("n_vars") or 1)
    iv_pcpv_off = numeric(off, "iv_per_cycle") / n_vars
    iv_pcpv_on  = numeric(on,  "iv_per_cycle") / n_vars
    iv_pcpv_delta = iv_pcpv_on - iv_pcpv_off
    row["iv_pcpv_off"]   = round(iv_pcpv_off, 4)
    row["iv_pcpv_on"]    = round(iv_pcpv_on, 4)
    row["iv_pcpv_delta"] = round(iv_pcpv_delta, 4)

    row["full_audit_delta"] = int(numeric(on, "full_audits") - numeric(off, "full_audits"))
    row["drift_fp_delta"] = int(numeric(on, "drift_fp") - numeric(off, "drift_fp"))
    row["novelty_fp_delta"] = int(numeric(on, "novelty_fp") - numeric(off, "novelty_fp"))

    off_hits = numeric(off, "comp_hits_life")
    on_hits  = numeric(on,  "comp_hits_life")
    comp_hits_delta = int(on_hits - off_hits)
    row["comp_hits_delta"] = comp_hits_delta
    if off_hits > 0:
        row["comp_hits_delta_pct"] = round(100.0 * comp_hits_delta / off_hits, 1)
    else:
        row["comp_hits_delta_pct"] = 100.0 if on_hits > 0 else 0.0

    row["amortization_delta"] = numeric(on, "amortization") - numeric(off, "amortization")
    row["watch_queued_delta"] = int(numeric(on, "watch_queued") - numeric(off, "watch_queued"))
    row["avg_lambda_delta"] = round(numeric(on, "avg_lambda") - numeric(off, "avg_lambda"), 4)

    ms_off = numeric(off, "ms_per_cycle")
    ms_on  = numeric(on,  "ms_per_cycle")
    row["ms_pcyc_off"] = round(ms_off, 2)
    row["ms_pcyc_on"]  = round(ms_on, 2)
    row["time_delta_pct"] = round(100.0 * (ms_on - ms_off) / ms_off, 1) if ms_off else 0.0

    cov_p50_off = numeric(off, "cov_p50")
    cov_p50_on  = numeric(on,  "cov_p50")
    row["cov_p50_delta"] = int(cov_p50_on - cov_p50_off) if (cov_p50_off > 0 or cov_p50_on > 0) else ""

    defer_delta = int(numeric(on, "defer_total") - numeric(off, "defer_total"))
    row["defer_delta"] = defer_delta

    # Verdict: compressions are the primary signal; IV direction catches the
    # defer/suppress failure mode where audit counts drop because work was
    # deferred rather than done (audits fall, IV rises, hits fall — economically hurt).
    # help  = compressions improved  AND  audit work fell
    # hurt  = compressions degraded  AND  (audit work rose  OR  IV rose)
    # neutral = mixed signals or negligible change
    hit_up   = comp_hits_delta > 0
    hit_down = comp_hits_delta < 0
    aud_down = row["full_audit_delta"] < 0
    aud_up   = row["full_audit_delta"] > 0
    iv_up    = iv_delta > 0
    if hit_up and aud_down:
        row["verdict"] = "help"
    elif hit_down and (aud_up or iv_up):
        row["verdict"] = "hurt"
    else:
        row["verdict"] = "neutral"

    row["parse_error"] = ""
    return row


def case_label(row: Dict[str, Any]) -> str:
    cw = row.get("cost_weight_mode", "uniform")
    cw_s = f" cw={cw}" if cw != "uniform" else ""
    hit_d = int(row.get("comp_hits_delta") or 0)
    aud_d = int(row.get("full_audit_delta") or 0)
    return (
        f"seed={row.get('seed')} n_vars={row.get('n_vars')} "
        f"cycles={row.get('cycles')}{cw_s} "
        f"hit_Δ={hit_d:+d} aud_Δ={aud_d:+d}"
    )


def summary_row(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [p for p in pairs if not p.get("parse_error")]
    pct_deltas        = [float(p["iv_delta_pct"])       for p in valid]
    pcpv_deltas       = [float(p["iv_pcpv_delta"])       for p in valid]
    comp_hits_deltas  = [float(p["comp_hits_delta"])     for p in valid]
    full_audit_deltas = [float(p["full_audit_delta"])    for p in valid]
    wins    = sum(1 for p in valid if p.get("verdict") == "help")
    losses  = sum(1 for p in valid if p.get("verdict") == "hurt")
    neutral = sum(1 for p in valid if p.get("verdict") == "neutral")
    best  = min(valid, key=lambda p: float(p.get("comp_hits_delta") or 0) * -1 +
                                     float(p.get("full_audit_delta") or 0)) if valid else None
    worst = max(valid, key=lambda p: float(p.get("comp_hits_delta") or 0) * -1 +
                                     float(p.get("full_audit_delta") or 0)) if valid else None
    return {
        "row_type": "summary",
        "total_pairs": len(valid),
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "median_comp_hits_delta":  statistics.median(comp_hits_deltas)  if comp_hits_deltas  else "",
        "median_full_audit_delta": statistics.median(full_audit_deltas) if full_audit_deltas else "",
        "median_iv_delta_pct":     statistics.median(pct_deltas)        if pct_deltas        else "",
        "median_iv_pcpv_delta":    statistics.median(pcpv_deltas)       if pcpv_deltas       else "",
        "best_case":  case_label(best)  if best  else "",
        "worst_case": case_label(worst) if worst else "",
        "parse_error": "" if len(valid) == len(pairs) else f"{len(pairs) - len(valid)} pair(s) had parse errors",
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def print_pair_table(pairs_with_on: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
    VERDICT_ORDER = {"hurt": 0, "neutral": 1, "help": 2, "parse_error": 3}
    sorted_pairs = sorted(
        pairs_with_on,
        key=lambda x: (
            VERDICT_ORDER.get(x[0].get("verdict", ""), 9),
            -float(x[0].get("comp_hits_delta") or 0),
        ),
    )
    hdr = (
        f"{'seed':>4} {'nv':>3} {'cyc':>5}  "
        f"{'hit_Δ':>8}  {'hit_Δ%':>7}  {'aud_Δ':>7}  {'iv_Δ/c/v':>9}  "
        f"{'defer_Δ':>8}  {'ms/cyc':>7}  {'t_Δ%':>6}  verdict"
    )
    print(hdr)
    print("─" * len(hdr))
    for pair, on in sorted_pairs:
        if pair.get("parse_error"):
            print(f"  parse_error: {pair['parse_error'][:80]}")
            continue
        hit_d      = int(pair.get("comp_hits_delta") or 0)
        hit_d_pct  = float(pair.get("comp_hits_delta_pct") or 0)
        aud_d      = int(pair.get("full_audit_delta") or 0)
        iv_pcpv_d  = float(pair.get("iv_pcpv_delta") or 0)
        defer_d    = int(pair.get("defer_delta") or 0)
        ms_on      = float(pair.get("ms_pcyc_on") or 0)
        t_delta    = float(pair.get("time_delta_pct") or 0)
        verdict    = pair.get("verdict", "?")
        iv_sign    = "+" if iv_pcpv_d > 0 else ""
        print(
            f"{pair.get('seed'):>4} {pair.get('n_vars'):>3} {pair.get('cycles'):>5}  "
            f"{hit_d:>+8}  {hit_d_pct:>+6.1f}%  {aud_d:>+7}  "
            f"{iv_sign}{iv_pcpv_d:>8.4f}  "
            f"{defer_d:>+8}  {ms_on:>7.2f}  {t_delta:>+5.1f}%  {verdict}"
        )


def print_causal_diagnostic(
    pairs_with_on: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> None:
    """Break down help/hurt/neutral groups, showing what drove the outcome
    and how active predictive-failure tracking was."""
    groups: Dict[str, List[Tuple[Dict, Dict]]] = {"help": [], "hurt": [], "neutral": []}
    for pair, on in pairs_with_on:
        v = pair.get("verdict", "")
        if v in groups:
            groups[v].append((pair, on))
    if not any(groups.values()):
        return

    def _avg(rows: List[Tuple[Dict, Dict]], fn) -> str:
        vals = [fn(p, on) for p, on in rows]
        vals = [v for v in vals if v is not None]
        return f"{sum(vals)/len(vals):+.1f}" if vals else "n/a"

    print()
    print("causal breakdown by verdict:")
    hdr2 = (f"  {'':8}  {'n':>3}  {'avg hit_Δ':>10}  {'avg hit_Δ%':>11}  "
            f"{'avg aud_Δ':>10}  {'avg iv_Δ/c/v':>13}  {'avg λ_avg':>10}  "
            f"{'avg watch_Q':>11}  {'avg cd_blk':>10}  {'avg gated':>10}")
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for label in ("help", "hurt", "neutral"):
        rows = groups[label]
        if not rows:
            continue
        n = len(rows)
        avg_hit   = _avg(rows, lambda p, _: float(p.get("comp_hits_delta") or 0))
        avg_hitp  = _avg(rows, lambda p, _: float(p.get("comp_hits_delta_pct") or 0))
        avg_aud   = _avg(rows, lambda p, _: float(p.get("full_audit_delta") or 0))
        avg_iv    = _avg(rows, lambda p, _: float(p.get("iv_pcpv_delta") or 0))
        avg_lam   = _avg(rows, lambda _, on: numeric(on, "avg_lambda"))
        avg_wq    = _avg(rows, lambda _, on: numeric(on, "watch_queued"))
        avg_cdb   = _avg(rows, lambda _, on: numeric(on, "cooldown_blocks"))
        avg_gated = _avg(rows, lambda _, on: numeric(on, "gated_pairs"))
        print(f"  {label:8}  {n:>3}  {avg_hit:>10}  {avg_hitp:>11}%  {avg_aud:>10}  "
              f"{avg_iv:>13}  {avg_lam:>10}  {avg_wq:>11}  {avg_cdb:>10}  {avg_gated:>10}")

    # Highlight neutral rows with mixed signals (one direction helps, other hurts)
    mixed = [
        (p, on) for p, on in groups["neutral"]
        if (float(p.get("comp_hits_delta") or 0) > 0) != (float(p.get("full_audit_delta") or 0) < 0)
        and float(p.get("comp_hits_delta") or 0) != 0
        and float(p.get("full_audit_delta") or 0) != 0
    ]
    if mixed:
        print(f"\n  mixed-signal neutral rows ({len(mixed)}):")
        for p, on in mixed[:5]:
            hit_d = int(p.get("comp_hits_delta") or 0)
            aud_d = int(p.get("full_audit_delta") or 0)
            print(
                f"    seed={p.get('seed')} nv={p.get('n_vars')} cyc={p.get('cycles')}"
                f"  hit_Δ={hit_d:+d}  aud_Δ={aud_d:+d}"
                f"  (hit {'↑' if hit_d > 0 else '↓'} but aud {'↓' if aud_d < 0 else '↑'})"
            )


def main() -> int:
    args = parse_args()
    cw_modes: List[str] = args.cost_weight_modes

    jobs: List[Tuple[int, int, int, str, bool, str]] = []
    for cw_mode in cw_modes:
        for seed in args.seeds:
            for n_vars in args.n_vars:
                for cycles in args.cycles:
                    jobs.append((seed, n_vars, cycles, args.mode, False, cw_mode))
                    jobs.append((seed, n_vars, cycles, args.mode, True,  cw_mode))

    results: Dict[Tuple[int, int, int, bool, str], Dict[str, Any]] = {}
    max_workers = max(1, int(args.jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(run_one, seed, n_vars, cycles, mode, forms, cw_mode):
            (seed, n_vars, cycles, forms, cw_mode)
            for seed, n_vars, cycles, mode, forms, cw_mode in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            seed, n_vars, cycles, forms, cw_mode = future_to_job[future]
            try:
                results[(seed, n_vars, cycles, forms, cw_mode)] = future.result()
            except Exception as exc:
                results[(seed, n_vars, cycles, forms, cw_mode)] = {
                    "row_type": "raw",
                    "seed": seed,
                    "n_vars": n_vars,
                    "cycles": cycles,
                    "mode": args.mode,
                    "forms": int(forms),
                    "cost_weight_mode": cw_mode,
                    "parse_error": f"worker failed: {exc}",
                }

    raw_rows: List[Dict[str, Any]] = []
    all_pairs: List[Dict[str, Any]] = []
    pairs_by_mode: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {m: [] for m in cw_modes}

    for cw_mode in cw_modes:
        for seed in args.seeds:
            for n_vars in args.n_vars:
                for cycles in args.cycles:
                    off = results[(seed, n_vars, cycles, False, cw_mode)]
                    on  = results[(seed, n_vars, cycles, True,  cw_mode)]
                    raw_rows.extend([off, on])
                    pair = pair_rows(off, on)
                    all_pairs.append(pair)
                    pairs_by_mode[cw_mode].append((pair, on))

    summary = summary_row(all_pairs)
    all_rows = raw_rows + all_pairs + [summary]

    write_csv(Path(args.out), all_rows)
    if args.jsonl:
        write_jsonl(Path(args.jsonl), all_rows)

    med_hit = summary["median_comp_hits_delta"]
    med_aud = summary["median_full_audit_delta"]
    med_hit_s = f"{med_hit:+.0f}" if isinstance(med_hit, float) else str(med_hit)
    med_aud_s = f"{med_aud:+.0f}" if isinstance(med_aud, float) else str(med_aud)
    print(
        f"\nbench_forms  {summary['total_pairs']} pairs | "
        f"{summary['wins']} help  {summary['losses']} hurt  {summary['neutral']} neutral | "
        f"median hit_Δ {med_hit_s}  aud_Δ {med_aud_s}"
    )
    print(f"  best   {summary['best_case']}")
    print(f"  worst  {summary['worst_case']}")

    for cw_mode in cw_modes:
        pairs_with_on = pairs_by_mode[cw_mode]
        mode_pairs = [p for p, _ in pairs_with_on]
        mode_sum = summary_row(mode_pairs)
        mh = mode_sum["median_comp_hits_delta"]
        ma = mode_sum["median_full_audit_delta"]
        mh_s = f"{mh:+.0f}" if isinstance(mh, float) else str(mh)
        ma_s = f"{ma:+.0f}" if isinstance(ma, float) else str(ma)
        print(
            f"\n── {cw_mode}  "
            f"({mode_sum['wins']}H {mode_sum['losses']}U {mode_sum['neutral']}N  "
            f"median hit_Δ {mh_s}  aud_Δ {ma_s}) ──"
        )
        print_pair_table(pairs_with_on)
        print_causal_diagnostic(pairs_with_on)

    print()
    print(f"  jobs={max_workers}  csv={args.out}", end="")
    if args.jsonl:
        print(f"  jsonl={args.jsonl}", end="")
    print()
    if summary.get("parse_error"):
        print(f"  parse_error={summary['parse_error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
