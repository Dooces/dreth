#!/usr/bin/env python3
from __future__ import annotations

"""Run Dreth comparison suites without changing runtime behavior."""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
AUTHORITY_STRENGTH_JOBS = (
    ("off", "off", "state", "off"),
    ("record", "record", "state", "off"),
    ("assist", "assist", "state", "shadow"),
    ("assist_quarantine_persistent", "assist", "state", "quarantine_persistent"),
    ("assist_quarantine_repair_only", "assist", "state", "quarantine_repair_only"),
    ("assist_legacy", "assist", "legacy", "off"),
)
AUTHORITY_STRENGTH_MODES = tuple(label for label, _, _, _ in AUTHORITY_STRENGTH_JOBS)
SUMMARY_SPECS = (
    ("authority_strength", "summarize_authority_strength.py", "authority_strength_summary"),
    ("context_role", "summarize_context_role_index.py", "context_role_summary"),
    ("uncertainty", "summarize_uncertainty_consolidation.py", "uncertainty_summary"),
    ("authority_evidence", "summarize_blind_authority_evidence.py", "authority_evidence"),
)
BEHAVIOR_FIELDS = (
    "skip_pct",
    "iv",
    "quality_cost",
    "full_audits",
    "revocations",
    "unique_fails",
    "regime_sentinel_fail",
    "no_sentinel",
)
OPERATIONAL_WARN_FIELDS = ("quality_cost", "iv", "full_audits", "revocations", "unique_fails")


@dataclass(frozen=True)
class CommandJob:
    label: str
    command: list[str]
    log_path: Path


@dataclass
class RunningJob:
    job: CommandJob
    process: subprocess.Popen[str]
    log_handle: TextIO
    reader: threading.Thread


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Dreth comparison suite.")
    parser.add_argument("--suite", required=True, choices=["authority_strength"])
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--suite-workers", type=int, default=3)
    parser.add_argument("--summary-workers", type=int, default=4)
    parser.add_argument("--sequential", action="store_true")

    parser.add_argument("--vars", default="5,8,12")
    parser.add_argument("--cycles", default="100,300")
    parser.add_argument("--seeds", default="42,7,99")
    parser.add_argument("--schedule", default="blind_challenge")
    parser.add_argument("--settle-cycles", type=int, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--challenge-blind", action="store_true")
    parser.add_argument("--hybrid-control", default=None)
    parser.add_argument("--repair-agenda", action="store_true")
    parser.add_argument("--parent-ranker", default=None)
    parser.add_argument("--probe-proposer", default=None)
    parser.add_argument("--uncertainty-consolidation", default=None)
    parser.add_argument("--uncertainty-assist-policy", default=None)
    parser.add_argument("--context-role-index", default=None)
    parser.add_argument("--context-role-anchor-policy", default=None)
    parser.add_argument("--relative-authority-report", action="store_true")
    parser.add_argument("--relative-authority-frontier-report", action="store_true")
    parser.add_argument("--relative-authority-frontier-temporal-report", action="store_true")
    parser.add_argument("--relative-authority-frontier-warmup-cycles", type=int, default=None)
    parser.add_argument("--relative-authority-frontier-max-candidates", type=int, default=None)
    parser.add_argument("--relative-authority-frontier-max-depth", type=int, default=None)
    return parser


def _append_value(args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def batch_passthrough_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, attr in (
        ("--vars", "vars"),
        ("--cycles", "cycles"),
        ("--seeds", "seeds"),
        ("--schedule", "schedule"),
        ("--settle-cycles", "settle_cycles"),
        ("--noise-sigma", "noise_sigma"),
        ("--workers", "workers"),
        ("--hybrid-control", "hybrid_control"),
        ("--parent-ranker", "parent_ranker"),
        ("--probe-proposer", "probe_proposer"),
        ("--uncertainty-consolidation", "uncertainty_consolidation"),
        ("--uncertainty-assist-policy", "uncertainty_assist_policy"),
        ("--context-role-index", "context_role_index"),
        ("--context-role-anchor-policy", "context_role_anchor_policy"),
        ("--relative-authority-frontier-warmup-cycles", "relative_authority_frontier_warmup_cycles"),
        ("--relative-authority-frontier-max-candidates", "relative_authority_frontier_max_candidates"),
        ("--relative-authority-frontier-max-depth", "relative_authority_frontier_max_depth"),
    ):
        _append_value(out, flag, getattr(args, attr))
    for flag, attr in (
        ("--challenge-blind", "challenge_blind"),
        ("--repair-agenda", "repair_agenda"),
        ("--relative-authority-report", "relative_authority_report"),
        ("--relative-authority-frontier-report", "relative_authority_frontier_report"),
        ("--relative-authority-frontier-temporal-report", "relative_authority_frontier_temporal_report"),
    ):
        if getattr(args, attr):
            out.append(flag)
    return out


def suite_paths(out_prefix: str) -> dict[str, dict[str, Path]]:
    prefix = Path(out_prefix)
    return {
        mode: {
            "jsonl": prefix.with_name(f"{prefix.name}_{mode}.jsonl"),
            "log": prefix.with_name(f"{prefix.name}_{mode}.log"),
        }
        for mode in AUTHORITY_STRENGTH_MODES
    }


def build_authority_strength_jobs(args: argparse.Namespace) -> list[CommandJob]:
    paths = suite_paths(args.out_prefix)
    common = batch_passthrough_args(args)
    jobs = []
    for label, mode, controller, policy in AUTHORITY_STRENGTH_JOBS:
        command = [
            sys.executable,
            str(SCRIPTS / "batch_run.py"),
            *common,
            "--authority-strength",
            mode,
            "--authority-strength-controller",
            controller,
            "--authority-derivation-policy",
            policy,
            "--out",
            str(paths[label]["jsonl"]),
        ]
        jobs.append(CommandJob(label, command, paths[label]["log"]))
    return jobs


def summary_output_path(out_prefix: str, mode: str, suffix: str) -> Path:
    prefix = Path(out_prefix)
    return prefix.with_name(f"{prefix.name}_{mode}_{suffix}.txt")


def build_summary_jobs(out_prefix: str) -> list[CommandJob]:
    paths = suite_paths(out_prefix)
    jobs: list[CommandJob] = []
    for mode in AUTHORITY_STRENGTH_MODES:
        if mode == "off":
            continue
        jsonl = paths[mode]["jsonl"]
        for summary_name, script_name, suffix in SUMMARY_SPECS:
            output_path = summary_output_path(out_prefix, mode, suffix)
            command = [sys.executable, str(SCRIPTS / script_name), "--jsonl", str(jsonl)]
            jobs.append(CommandJob(f"{mode}:{summary_name}", command, output_path))
    return jobs


def _stream_output(
    job: CommandJob,
    process: subprocess.Popen[str],
    log_handle: TextIO,
    terminal: TextIO,
    terminal_lock: threading.Lock,
) -> None:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line == "":
            break
        log_handle.write(line)
        log_handle.flush()
        if not line.endswith("\n"):
            line += "\n"
        with terminal_lock:
            terminal.write(f"[{job.label}] {line}")
            terminal.flush()


def _launch_job(
    job: CommandJob,
    terminal: TextIO,
    terminal_lock: threading.Lock,
    popen_factory: Any,
) -> RunningJob:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = job.log_path.open("w")
    process = popen_factory(
        job.command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader = threading.Thread(
        target=_stream_output,
        args=(job, process, log_handle, terminal, terminal_lock),
        daemon=True,
    )
    reader.start()
    return RunningJob(job, process, log_handle, reader)


def _finish_job(running: RunningJob) -> None:
    running.reader.join(timeout=5)
    running.log_handle.close()


def _terminate_running(running_jobs: Iterable[RunningJob]) -> None:
    for running in running_jobs:
        if running.process.poll() is None:
            running.process.terminate()
    for running in running_jobs:
        try:
            running.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            running.process.kill()
            running.process.wait(timeout=5)
        _finish_job(running)


def run_labeled_commands(
    jobs: list[CommandJob],
    *,
    max_workers: int,
    sequential: bool = False,
    terminal: TextIO | None = None,
    popen_factory: Any = subprocess.Popen,
) -> int:
    if terminal is None:
        terminal = sys.stdout
    workers = 1 if sequential else max(1, max_workers)
    pending = list(jobs)
    running: list[RunningJob] = []
    terminal_lock = threading.Lock()

    try:
        while pending or running:
            while pending and len(running) < workers:
                running.append(_launch_job(pending.pop(0), terminal, terminal_lock, popen_factory))

            completed: list[tuple[RunningJob, int]] = []
            for item in list(running):
                code = item.process.poll()
                if code is not None:
                    completed.append((item, code))

            if not completed:
                time.sleep(0.05)
                continue

            for item, code in completed:
                if item in running:
                    running.remove(item)
                _finish_job(item)
                if code != 0:
                    _terminate_running(running)
                    return int(code or 1)
    except KeyboardInterrupt:
        _terminate_running(running)
        raise
    return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_log_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    match = re.search(r"runs ok=([0-9]+/[0-9]+)", text)
    if match:
        metrics["runs_ok"] = match.group(1)
    match = re.search(r"invariants:\s*([^\n]+)", text)
    if match:
        metrics["invariants"] = match.group(1).strip()

    aliases = {
        "skip%": "skip_pct",
        "iv": "iv",
        "quality_cost": "quality_cost",
        "full_audits": "full_audits",
        "audits": "full_audits",
        "revocations": "revocations",
        "unique_fails": "unique_fails",
        "regime_sentinel_fail": "regime_sentinel_fail",
        "no_sentinel": "no_sentinel",
        "saved_iv": "passive_saved_iv",
        "stressed": "passive_stressed",
        "route_certs": "route_certs",
        "audit_certs": "audit_certs",
        "dormant": "dormant",
        "chosen_parent_recall": "chosen_parent_recall",
        "recall_lift": "recall_lift",
        "candidate_reduction_vs_visible": "candidate_reduction_vs_visible",
    }
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*|skip%)=([-+]?[0-9]+(?:\.[0-9]+)?)")
    for key, raw in pattern.findall(text):
        if key in aliases:
            metrics[aliases[key]] = _as_float(raw)
    return metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") != "policy_report":
                rows.append(row)
    return rows


def aggregate_jsonl_metrics(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        return {}
    n = len(rows)
    metrics: dict[str, Any] = {
        "runs_ok": f"{sum(1 for row in rows if row.get('ok', False))}/{n}",
        "skip_pct": sum(_as_float(row.get("skip_pct")) for row in rows) / n,
        "iv": sum(_as_float(row.get("interventions")) for row in rows) / n,
        "quality_cost": sum(_as_float(row.get("quality_cost")) for row in rows) / n,
        "full_audits": sum(_as_float(row.get("full_audits")) for row in rows) / n,
        "revocations": sum(
            sum(_as_float(v) for v in (row.get("revoked_by_dist") or {}).values())
            for row in rows
        )
        / n,
        "unique_fails": sum(
            _as_float(row.get("total_unique_failures", row.get("unique_fails")))
            for row in rows
        )
        / n,
        "regime_sentinel_fail": sum(
            _as_float(row.get("regime_sentinel_fail", row.get("regime_sentinel_fails")))
            for row in rows
        )
        / n,
        "no_sentinel": sum(_as_float(row.get("regime_sentinel_no_sentinel")) for row in rows)
        / n,
        "passive_saved_iv": sum(_as_float(row.get("passive_saved_iv")) for row in rows) / n,
        "passive_stressed": sum(_as_float(row.get("passive_stress_count")) for row in rows)
        / n,
        "route_certs": sum(_as_float(row.get("route_certs_total")) for row in rows) / n,
        "audit_certs": sum(_as_float(row.get("audit_certs")) for row in rows) / n,
        "dormant": sum(_as_float(row.get("dormant_total")) for row in rows) / n,
    }
    frontier_aliases = (
        ("chosen_parent_recall", "temporal_frontier_chosen_parent_recall"),
        ("recall_lift", "temporal_frontier_recall_lift"),
        ("candidate_reduction_vs_visible", "temporal_frontier_candidate_reduction_vs_visible"),
    )
    for out_key, row_key in frontier_aliases:
        values = [_as_float(row.get(row_key)) for row in rows if row.get(row_key) is not None]
        if values:
            metrics[out_key] = sum(values) / len(values)
    return metrics


def parse_authority_strength_summary(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("strong", "usable", "weak", "contested", "insufficient"):
        match = re.search(rf"^\s*{key}:\s*([0-9]+)", text, re.MULTILINE)
        if match:
            out[key] = int(match.group(1))
    out["weak_contested_authority"] = out.get("weak", 0) + out.get("contested", 0)
    for key in (
        "monitoring_increases_from_strength",
        "alternatives_preserved_from_strength",
        "future_evidence_requirements",
        "repair_priority_bumps_from_strength",
    ):
        match = re.search(rf"{key}=([0-9]+)|{key}:\s*([0-9]+)", text)
        if match:
            out[key] = int(match.group(1) or match.group(2))
    return out


def parse_blind_authority_evidence_summary(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    match = re.search(r"^\s*cases=([0-9]+)\s*$", text, re.MULTILINE)
    if match:
        out["external_mismatch_cases"] = int(match.group(1))
    for key in (
        "evidence_supported_surrogate",
        "weakly_supported_surrogate",
        "contradicted_authority",
        "insufficient_evidence",
    ):
        match = re.search(rf"^\s*{key}\s+([0-9]+)\s*$", text, re.MULTILINE)
        if match:
            out[key] = int(match.group(1))
    for key in (
        "would_throttle",
        "would_not_throttle",
        "estimated_supported_surrogates_preserved",
        "unthrottled_supported_surrogate",
    ):
        match = re.search(rf"^\s*{key}:\s+([0-9]+)\s*$", text, re.MULTILINE)
        if match:
            out[key] = int(match.group(1))
    out["supported_surrogates_preserved"] = out.get(
        "estimated_supported_surrogates_preserved",
        out.get("unthrottled_supported_surrogate", 0),
    )
    return out


def evidence_metrics_from_jsonl(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPTS))
    from summarize_blind_authority_evidence import summarize  # type: ignore
    from summarize_blind_authority_evidence import _throttle_counts  # type: ignore

    summary = summarize(load_jsonl(path))
    throttle = _throttle_counts(summary.external_mismatch_items, mode="conservative")
    return {
        "external_mismatch_cases": len(summary.external_mismatch_cases),
        "evidence_supported_surrogate": summary.by_classification["evidence_supported_surrogate"],
        "weakly_supported_surrogate": summary.by_classification["weakly_supported_surrogate"],
        "contradicted_authority": summary.by_classification["contradicted_authority"],
        "insufficient_evidence": summary.by_classification["insufficient_evidence"],
        "would_throttle": throttle["would_throttle"],
        "would_not_throttle": throttle["would_not_throttle"],
        "supported_surrogates_preserved": throttle["estimated_supported_surrogates_preserved"],
    }


def collect_mode_metrics(out_prefix: str) -> dict[str, dict[str, Any]]:
    paths = suite_paths(out_prefix)
    result: dict[str, dict[str, Any]] = {}
    for mode, mode_paths in paths.items():
        metrics = aggregate_jsonl_metrics(mode_paths["jsonl"])
        if mode_paths["log"].exists():
            metrics.update(parse_log_metrics(mode_paths["log"].read_text()))
        evidence = evidence_metrics_from_jsonl(mode_paths["jsonl"])
        metrics.update(evidence)
        result[mode] = metrics
    for mode in AUTHORITY_STRENGTH_MODES:
        if mode == "off":
            continue
        auth_path = summary_output_path(out_prefix, mode, "authority_strength_summary")
        evidence_path = summary_output_path(out_prefix, mode, "authority_evidence")
        if auth_path.exists():
            result[mode]["authority_strength_summary"] = parse_authority_strength_summary(
                auth_path.read_text()
            )
        if evidence_path.exists():
            result[mode].update(parse_blind_authority_evidence_summary(evidence_path.read_text()))
    return result


def behavior_equal(off: dict[str, Any], record: dict[str, Any]) -> bool:
    for field in BEHAVIOR_FIELDS:
        if abs(_as_float(off.get(field)) - _as_float(record.get(field))) > 1e-9:
            return False
    return True


def decision_lines(metrics: dict[str, dict[str, Any]]) -> list[str]:
    off = metrics.get("off", {})
    record = metrics.get("record", {})
    assist = metrics.get("assist", {})
    lines: list[str] = []
    if behavior_equal(off, record):
        lines.append("PASS: off and record match on behavior metrics.")
    else:
        changed = [
            field
            for field in BEHAVIOR_FIELDS
            if abs(_as_float(off.get(field)) - _as_float(record.get(field))) > 1e-9
        ]
        lines.append(f"FAIL: record differs from off on behavior metrics: {', '.join(changed)}.")

    worsened = [
        field
        for field in OPERATIONAL_WARN_FIELDS
        if _as_float(assist.get(field)) > _as_float(off.get(field))
    ]
    if worsened:
        lines.append(f"WARN: assist worsens operational metrics vs off: {', '.join(worsened)}.")

    hidden_help = (
        _as_float(assist.get("contradicted_authority"))
        < _as_float(off.get("contradicted_authority"))
        or _as_float(assist.get("external_mismatch_cases"))
        < _as_float(off.get("external_mismatch_cases"))
    )
    preserved = _as_float(assist.get("supported_surrogates_preserved")) >= _as_float(
        off.get("supported_surrogates_preserved")
    )
    if hidden_help and preserved:
        lines.append(
            "PASS: assist reduces contradicted authority or mismatch cases without losing supported surrogates."
        )
    if hidden_help and worsened:
        lines.append("WARN: assist helps hidden mismatch but costs more operationally.")
    lines.append("WARN: hidden truth is offline interpretation only; runtime does not use it.")
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _delta(metrics: dict[str, Any], base: dict[str, Any], field: str) -> float:
    return _as_float(metrics.get(field)) - _as_float(base.get(field))


def render_comparison(metrics: dict[str, dict[str, Any]]) -> str:
    lines = ["Dreth Comparison Suite: authority_strength", ""]
    lines.append("Log Metrics")
    log_fields = (
        "runs_ok",
        "invariants",
        "skip_pct",
        "iv",
        "quality_cost",
        "full_audits",
        "revocations",
        "unique_fails",
        "regime_sentinel_fail",
        "no_sentinel",
        "passive_saved_iv",
        "passive_stressed",
        "route_certs",
        "audit_certs",
        "dormant",
        "chosen_parent_recall",
        "recall_lift",
        "candidate_reduction_vs_visible",
    )
    for mode in AUTHORITY_STRENGTH_MODES:
        bits = [f"{field}={_fmt(metrics.get(mode, {}).get(field))}" for field in log_fields]
        lines.append(f"  {mode}: " + " ".join(bits))

    lines.extend(["", "Authority Strength Summaries"])
    for mode in AUTHORITY_STRENGTH_MODES:
        if mode == "off":
            continue
        summary = metrics.get(mode, {}).get("authority_strength_summary", {})
        lines.append(
            f"  {mode}: "
            f"strong={summary.get('strong', 0)} usable={summary.get('usable', 0)} "
            f"weak={summary.get('weak', 0)} contested={summary.get('contested', 0)} "
            f"insufficient={summary.get('insufficient', 0)} "
            f"weak_contested={summary.get('weak_contested_authority', 0)} "
            f"monitoring_budget_hints={summary.get('monitoring_increases_from_strength', 0)} "
            f"alternative_preservation_hints={summary.get('alternatives_preserved_from_strength', 0)} "
            f"repair_priority_hints={summary.get('repair_priority_bumps_from_strength', 0)} "
            f"future_evidence_requirements={summary.get('future_evidence_requirements', 0)}"
        )

    lines.extend(["", "Blind Authority Evidence"])
    evidence_fields = (
        "external_mismatch_cases",
        "evidence_supported_surrogate",
        "weakly_supported_surrogate",
        "contradicted_authority",
        "insufficient_evidence",
        "would_throttle",
        "would_not_throttle",
        "supported_surrogates_preserved",
    )
    for mode in AUTHORITY_STRENGTH_MODES:
        bits = [f"{field}={_fmt(metrics.get(mode, {}).get(field))}" for field in evidence_fields]
        lines.append(f"  {mode}: " + " ".join(bits))

    table_fields = (
        "iv",
        "quality_cost",
        "full_audits",
        "revocations",
        "unique_fails",
        "external_mismatch_cases",
        "contradicted_authority",
        "supported_surrogates_preserved",
    )
    header = (
        "mode        iv      qcost      audits   rev   unique_fails   "
        "mismatch   contradicted   supported_surrogate"
    )
    lines.extend(["", "Comparison Table", header])
    for mode in AUTHORITY_STRENGTH_MODES:
        row = metrics.get(mode, {})
        lines.append(
            f"{mode:<11} "
            f"{_fmt(row.get('iv')):<7} "
            f"{_fmt(row.get('quality_cost')):<10} "
            f"{_fmt(row.get('full_audits')):<8} "
            f"{_fmt(row.get('revocations')):<5} "
            f"{_fmt(row.get('unique_fails')):<14} "
            f"{_fmt(row.get('external_mismatch_cases')):<10} "
            f"{_fmt(row.get('contradicted_authority')):<14} "
            f"{_fmt(row.get('supported_surrogates_preserved'))}"
        )
    off = metrics.get("off", {})
    for label, mode in (("assist_delta_vs_off", "assist"), ("record_delta_vs_off", "record")):
        row = metrics.get(mode, {})
        lines.append(
            f"{label:<11} "
            f"{_fmt(_delta(row, off, 'iv')):<7} "
            f"{_fmt(_delta(row, off, 'quality_cost')):<10} "
            f"{_fmt(_delta(row, off, 'full_audits')):<8} "
            f"{_fmt(_delta(row, off, 'revocations')):<5} "
            f"{_fmt(_delta(row, off, 'unique_fails')):<14} "
            f"{_fmt(_delta(row, off, 'external_mismatch_cases')):<10} "
            f"{_fmt(_delta(row, off, 'contradicted_authority')):<14} "
            f"{_fmt(_delta(row, off, 'supported_surrogates_preserved'))}"
        )

    lines.extend(["", "Decision Block", *decision_lines(metrics)])
    lines.append("")
    return "\n".join(lines)


def write_comparison(out_prefix: str) -> tuple[Path, str]:
    comparison_path = Path(out_prefix).with_name(f"{Path(out_prefix).name}_comparison.txt")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_comparison(collect_mode_metrics(out_prefix))
    comparison_path.write_text(text)
    return comparison_path, text


def validate_outputs_exist(out_prefix: str) -> None:
    missing = [
        path
        for mode_paths in suite_paths(out_prefix).values()
        for path in (mode_paths["jsonl"], mode_paths["log"])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing batch outputs: " + ", ".join(str(p) for p in missing))


def run_authority_strength_suite(args: argparse.Namespace) -> int:
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    batch_jobs = build_authority_strength_jobs(args)
    code = run_labeled_commands(
        batch_jobs,
        max_workers=args.suite_workers,
        sequential=args.sequential,
    )
    if code != 0:
        return code

    validate_outputs_exist(args.out_prefix)
    summary_jobs = build_summary_jobs(args.out_prefix)
    code = run_labeled_commands(
        summary_jobs,
        max_workers=args.summary_workers,
        sequential=args.sequential,
    )
    if code != 0:
        return code

    comparison_path, text = write_comparison(args.out_prefix)
    print(f"[comparison] wrote {comparison_path}")
    in_decision = False
    for line in text.splitlines():
        if line == "Decision Block":
            in_decision = True
            print(line)
            continue
        if in_decision and line:
            print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.suite_workers < 1:
        parser.error("--suite-workers must be >= 1")
    if args.summary_workers < 1:
        parser.error("--summary-workers must be >= 1")
    if args.suite == "authority_strength":
        return run_authority_strength_suite(args)
    parser.error(f"unsupported suite: {args.suite}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
