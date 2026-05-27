#!/usr/bin/env python3
"""Multi-generation dreth loop with correct delta-based mind compaction.

Correct loop invariants:
  - Each generation produces its own run JSONL (world-backed evidence).
  - Sleep produces NEW sleep products from that generation's evidence only.
  - A per-generation delta file combines the run JSONL + sleep products.
  - compact receives --previous-mind=mindN-1 + --delta-input=deltaN → mindN.
  - Previous mind nodes are loaded directly, never re-ingested as evidence.
  - The cumulative raw archive is kept separately but NEVER used as compact input.

This prevents the recursive diary expansion bug where compacted mind nodes
were re-ingested as fresh evidence causing exponential file growth.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    "experience_events_written",
    "nethra_memory_records_written",
    "persistent_nethras_loaded",
    "sleep_products_loaded",
    "sleep_products_used",
    "nethra_memory_behavior_effects",
    "nethra_memory_authority_effects",
    "nethra_memory_candidate_reorders",
    "nethra_memory_probe_reorders",
    "full_audits",
    "interventions",
    "quality_cost",
    "revocations",
    "unique_fails",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_and_tee(cmd: list[str], log_path: Path, *, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()

    if rc != 0:
        raise SystemExit(f"command failed with exit code {rc}: {' '.join(cmd)}")


def batch_cmd(
    *,
    vars_: int,
    cycles: int,
    seeds: str,
    schedule: str,
    memory_mode: str,
    memory_path: Path,
    out_path: Path,
    workers: int | None,
    extra_flags: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/batch_run.py",
        "--vars",
        str(vars_),
        "--cycles",
        str(cycles),
        "--seeds",
        seeds,
        "--schedule",
        schedule,
        "--background-nethra",
        "record",
        "--context-role-index",
        "record",
        "--authority-strength",
        "record",
        "--nethra-memory",
        memory_mode,
        "--nethra-memory-path",
        str(memory_path),
        "--out",
        str(out_path),
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    cmd.extend(extra_flags)
    return cmd


def build_run_sleep_delta(
    *,
    run_path: Path,
    delta_path: Path,
    proposal_path: Path,
    min_sources: int,
    label: str,
    cwd: Path,
    archive_store_path: Path | None = None,
) -> dict[str, int]:
    """Build per-generation delta = run_path records + new sleep products.

    Writes delta_path as a combined JSONL of all run_path lines plus the
    generated sleep product lines. This file is the correct input for
    compact --delta-input (world-backed only, no prior mind nodes).

    Optionally also appends sleep products to archive_store_path for
    debugging/archival purposes (never used as compact input).
    """
    sys.path.insert(0, str(cwd))

    from dreth.memory_sleep import MemorySleepConsolidator
    from dreth.nethra_memory_store import NethraMemoryStore

    c = MemorySleepConsolidator()
    rows = c.load_jsonl_rows(run_path)

    bg = c.extract_background_records(rows)
    cr = c.extract_context_role_records(rows)
    unc = c.extract_uncertainty_records(rows)
    auth = c.extract_authority_records(rows)
    temp = c.extract_temporal_records_if_available(rows)
    exp = c.extract_experience_events(rows)
    mem = c.extract_nethra_memory_records(rows)

    proposals = c.build_proposals(bg, cr, unc, auth, temp, min_sources=min_sources)
    products = c.build_sleep_products(mem, exp)

    # Write proposals to proposal_path
    with proposal_path.open("w", encoding="utf-8") as fh:
        for p in proposals:
            d = p.to_dict() if hasattr(p, "to_dict") else vars(p)
            fh.write(json.dumps(d, sort_keys=True) + "\n")

    # Build delta: copy run_path lines + append sleep product lines
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", encoding="utf-8") as fh:
        # Copy all world-backed run records
        if run_path.exists():
            with run_path.open(encoding="utf-8") as src:
                for line in src:
                    fh.write(line)
        # Append sleep products as fresh delta evidence
        for p in products:
            d = p.to_dict() if hasattr(p, "to_dict") else vars(p)
            fh.write(json.dumps(d, sort_keys=True) + "\n")

    # Optionally append to cumulative archive (for debug only, never compacted)
    if archive_store_path is not None:
        store = NethraMemoryStore(archive_store_path)
        store.append_sleep_products(products)

    result = {
        "bg": len(bg),
        "cr": len(cr),
        "unc": len(unc),
        "auth": len(auth),
        "temp": len(temp),
        "exp": len(exp),
        "mem": len(mem),
        "proposals": len(proposals),
        "products": len(products),
        "authority_allowed_products": sum(
            1 for p in products if bool(getattr(p, "authority_allowed", False))
        ),
        "authority_allowed_proposals": sum(
            1 for p in proposals if bool(getattr(p, "authority_allowed", False))
        ),
    }

    print(f"\n=== SLEEP {label} ===")
    print(
        f"sleep_input: bg={result['bg']} cr={result['cr']} unc={result['unc']} "
        f"auth={result['auth']} temp={result['temp']} exp={result['exp']} mem={result['mem']}"
    )
    print(
        f"sleep_output: proposals={result['proposals']} products={result['products']}"
    )
    print(
        f"authority_allowed: products={result['authority_allowed_products']} "
        f"proposals={result['authority_allowed_proposals']}"
    )

    if result["authority_allowed_products"] or result["authority_allowed_proposals"]:
        raise SystemExit("sleep emitted authority_allowed=True; aborting")

    return result


def compact_mind(
    *,
    delta_input_path: Path,
    mind_path: Path,
    report_path: Path,
    cwd: Path,
    previous_mind_path: Path | None = None,
) -> None:
    """Compact delta_input into mind_path, optionally loading previous_mind_path as base.

    Never re-ingests previous mind as evidence. Uses --previous-mind to load
    the prior canonical nodes, then ingests only new delta rows.
    """
    cmd = [
        sys.executable,
        "scripts/compact_nethra_memory.py",
        "--delta-input",
        str(delta_input_path),
        "--out",
        str(mind_path),
        "--report",
        str(report_path),
    ]
    if previous_mind_path is not None and previous_mind_path.exists():
        cmd.extend(["--previous-mind", str(previous_mind_path)])
    run_and_tee(cmd, report_path.with_suffix(".compact.log"), cwd=cwd)


def totals(path: Path) -> dict[str, int | float]:
    out: dict[str, int | float] = {k: 0 for k in FIELDS}
    rows = 0
    if not path.exists():
        out["rows"] = 0
        return out

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for k in FIELDS:
                v = row.get(k, 0)
                if isinstance(v, (int, float)):
                    out[k] = out.get(k, 0) + v

    out["rows"] = rows
    return out


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _read_mind_summary(path: Path) -> dict[str, Any]:
    """Extract nethra_mind_summary entry from a compacted mind JSONL."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("entry_kind") == "nethra_mind_summary":
                return row
    return {}


def _kb(n: int) -> str:
    return f"{n // 1024}kB"


def write_summary(
    *,
    paths: dict[str, Path],
    archive_store_path: Path | None,
    mind_paths: dict[str, Path],
    delta_paths: dict[str, Path],
    summary_path: Path,
) -> None:
    data = {name: totals(path) for name, path in paths.items()}
    mind_summaries = {name: _read_mind_summary(path) for name, path in mind_paths.items()}

    with summary_path.open("w", encoding="utf-8") as fh:
        def emit(s: str = "") -> None:
            print(s)
            fh.write(s + "\n")

        # ── per-generation run table ────────────────────────────────────────
        emit("=== run results ===")
        header = f"{'run':<22} {'rows':>6} {'loaded':>7} {'sleep_used':>10} {'behavior':>8} {'authority':>9} {'cand_reord':>10} {'audits':>6} {'iv':>4}"
        emit(header)
        emit("-" * len(header))
        for name, d in data.items():
            emit(
                f"{name:<22} "
                f"{int(d['rows']):>6} "
                f"{int(d['persistent_nethras_loaded']):>7} "
                f"{int(d['sleep_products_used']):>10} "
                f"{int(d['nethra_memory_behavior_effects']):>8} "
                f"{int(d['nethra_memory_authority_effects']):>9} "
                f"{int(d['nethra_memory_candidate_reorders']):>10} "
                f"{int(d['full_audits']):>6} "
                f"{int(d['interventions']):>4}"
            )

        # ── compaction table ────────────────────────────────────────────────
        emit()
        emit("=== compaction ===")
        comp_header = f"{'gen':<8} {'nodes_before':>12} {'nodes_after':>11} {'exact':>6} {'struct':>7} {'assim':>6} {'pruned':>7} {'residuals':>9} {'mind_kB':>8}"
        emit(comp_header)
        emit("-" * len(comp_header))
        for gen_name, ms in sorted(mind_summaries.items()):
            if not ms:
                continue
            emit(
                f"{gen_name:<8} "
                f"{int(ms.get('nodes_before', 0)):>12} "
                f"{int(ms.get('nodes_after', 0)):>11} "
                f"{int(ms.get('exact_folds', 0)):>6} "
                f"{int(ms.get('structural_folds', 0)):>7} "
                f"{int(ms.get('assimilation_folds', 0)):>6} "
                f"{int(ms.get('nodes_pruned', 0)):>7} "
                f"{int(ms.get('residuals_kept', 0)):>9} "
                f"{file_size(mind_paths[gen_name]) // 1024:>8}"
            )

        # ── file sizes ──────────────────────────────────────────────────────
        emit()
        emit("=== sizes ===")
        if archive_store_path:
            emit(f"  raw_archive        {_kb(file_size(archive_store_path)):>10}")
        for name, path in sorted(delta_paths.items()):
            emit(f"  delta_{name:<10} {_kb(file_size(path)):>10}")
        for name, path in sorted(mind_paths.items()):
            emit(f"  mind_{name:<11} {_kb(file_size(path)):>10}")

        # ── gates ───────────────────────────────────────────────────────────
        off = data.get("control_off", {})
        record = data.get("control_record", {})

        off_record_behavior_equal = bool(off) and bool(record) and all(
            off.get(k, 0) == record.get(k, 0)
            for k in ["full_audits", "interventions", "quality_cost"]
        )
        record_behavior_effects_zero = record.get("nethra_memory_behavior_effects", 0) == 0
        authority_effects_zero_all = all(
            d.get("nethra_memory_authority_effects", 0) == 0
            for d in data.values()
        )
        any_assist_behavior = any(
            d.get("nethra_memory_behavior_effects", 0) > 0
            for name, d in data.items()
            if "assist" in name
        )
        any_sleep_used = any(
            d.get("sleep_products_used", 0) > 0
            for name, d in data.items()
            if "assist" in name
        )
        mind_sizes_bounded = all(
            file_size(p) < 50 * 1024 * 1024  # 50MB hard cap
            for p in mind_paths.values()
            if p.exists()
        )

        def gate(label: str, passed: bool) -> str:
            return f"  {'PASS' if passed else 'FAIL'}  {label}"

        emit()
        emit("=== gates ===")
        emit(gate("authority_effects_zero (all modes)", authority_effects_zero_all))
        emit(gate("record_mode_no_behavior_change", record_behavior_effects_zero))
        emit(gate("off_record_identical_outcomes", off_record_behavior_equal))
        emit(gate("assist_has_behavior_effects", any_assist_behavior))
        emit(gate("assist_uses_sleep_products", any_sleep_used))
        emit(gate("mind_files_under_50MB", mind_sizes_bounded))

        # ── generation-over-generation assist trend ─────────────────────────
        assist_names = sorted(
            [n for n in data if n.startswith("run") and n.endswith("_assist")],
            key=lambda x: int(x.removeprefix("run").removesuffix("_assist")),
        )
        if assist_names:
            emit()
            emit("=== assist trend (Δ vs prev gen) ===")
            prev = None
            for name in assist_names:
                cur = data[name]
                if prev is None:
                    emit(f"  {name}: baseline")
                else:
                    db = int(cur["nethra_memory_behavior_effects"] - prev["nethra_memory_behavior_effects"])
                    ds = int(cur["sleep_products_used"] - prev["sleep_products_used"])
                    da = int(cur["full_audits"] - prev["full_audits"])
                    sign_b = "+" if db >= 0 else ""
                    sign_s = "+" if ds >= 0 else ""
                    sign_a = "+" if da >= 0 else ""
                    emit(f"  {name}: Δbehavior={sign_b}{db}  Δsleep_used={sign_s}{ds}  Δaudits={sign_a}{da}")
                prev = cur

    print(f"\nsummary written: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-generation Dreth loop with delta-based mind compaction.\n"
            "Each generation compacts only its own delta; previous mind is loaded\n"
            "directly, never re-ingested as evidence."
        )
    )
    p.add_argument("--vars", type=int, default=50)
    p.add_argument("--cycles", type=int, default=3000)
    p.add_argument("--seeds", default="42,99,7")
    p.add_argument("--schedule", default="blind_challenge")
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--min-sources", type=int, default=2)
    p.add_argument("--out-prefix", default="reports/dreth_multimind")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep cumulative sleep-product archive (never used as compact input)",
    )
    p.add_argument("batch_flags", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cwd = repo_root()

    prefix = Path(args.out_prefix)
    if not prefix.is_absolute():
        prefix = cwd / prefix

    reports_dir = prefix.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = prefix.with_name(prefix.name + "_summary.txt")
    archive_store: Path | None = None
    if args.keep_archive:
        archive_store = prefix.with_name(prefix.name + "_raw_archive.jsonl")

    extra_flags = list(args.batch_flags or [])
    if extra_flags and extra_flags[0] == "--":
        extra_flags = extra_flags[1:]

    if args.force:
        for path in reports_dir.glob(prefix.name + "*"):
            if path.is_file():
                path.unlink()

    paths: dict[str, Path] = {}
    mind_paths: dict[str, Path] = {}
    delta_paths: dict[str, Path] = {}

    # ── GEN 0: record-only bootstrap ────────────────────────────────────────
    print("=== RUN 0: record-only bootstrap writes raw evidence ===")
    run0 = prefix.with_name(prefix.name + "_run0_record.jsonl")
    paths["run0_record"] = run0

    # For gen0, --nethra-memory-path is unused (record mode writes to --out)
    # Pass a throwaway path since batch_run requires it
    _scratch0 = prefix.with_name(prefix.name + "_scratch0.tmp")

    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="record",
            memory_path=_scratch0,
            out_path=run0,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_run0_record.log"),
        cwd=cwd,
    )

    delta0 = prefix.with_name(prefix.name + "_delta_gen0.jsonl")
    delta_paths["gen0"] = delta0
    build_run_sleep_delta(
        run_path=run0,
        delta_path=delta0,
        proposal_path=prefix.with_name(prefix.name + "_sleep0_proposals.jsonl"),
        min_sources=args.min_sources,
        label="0",
        cwd=cwd,
        archive_store_path=archive_store,
    )

    mind0 = prefix.with_name(prefix.name + "_mind_gen0.jsonl")
    mind_paths["gen0"] = mind0
    compact_mind(
        delta_input_path=delta0,
        mind_path=mind0,
        report_path=prefix.with_name(prefix.name + "_mind_gen0_report.txt"),
        cwd=cwd,
        previous_mind_path=None,
    )

    current_mind: Path = mind0

    # ── GEN 1..N: assist loads compact mind genN-1 ──────────────────────────
    for gen in range(1, args.gens + 1):
        print(f"\n=== RUN {gen}: assist loads compact mind gen{gen - 1} ===")

        run_path = prefix.with_name(prefix.name + f"_run{gen}_assist.jsonl")
        paths[f"run{gen}_assist"] = run_path

        run_and_tee(
            batch_cmd(
                vars_=args.vars,
                cycles=args.cycles,
                seeds=args.seeds,
                schedule=args.schedule,
                memory_mode="assist",
                memory_path=current_mind,
                out_path=run_path,
                workers=args.workers,
                extra_flags=extra_flags,
            ),
            prefix.with_name(prefix.name + f"_run{gen}_assist.log"),
            cwd=cwd,
        )

        print(f"\n=== SLEEP {gen}: build delta from gen{gen} run ===")
        delta_path = prefix.with_name(prefix.name + f"_delta_gen{gen}.jsonl")
        delta_paths[f"gen{gen}"] = delta_path
        build_run_sleep_delta(
            run_path=run_path,
            delta_path=delta_path,
            proposal_path=prefix.with_name(prefix.name + f"_sleep{gen}_proposals.jsonl"),
            min_sources=args.min_sources,
            label=str(gen),
            cwd=cwd,
            archive_store_path=archive_store,
        )

        next_mind = prefix.with_name(prefix.name + f"_mind_gen{gen}.jsonl")
        mind_paths[f"gen{gen}"] = next_mind
        compact_mind(
            delta_input_path=delta_path,
            mind_path=next_mind,
            report_path=prefix.with_name(prefix.name + f"_mind_gen{gen}_report.txt"),
            cwd=cwd,
            previous_mind_path=current_mind,
        )
        current_mind = next_mind

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    print("\n=== CONTROL: off ignores compact mind ===")
    off_path = prefix.with_name(prefix.name + "_control_off.jsonl")
    paths["control_off"] = off_path

    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="off",
            memory_path=current_mind,
            out_path=off_path,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_control_off.log"),
        cwd=cwd,
    )

    print("\n=== CONTROL: record loads compact mind but must not change behavior ===")
    record_path = prefix.with_name(prefix.name + "_control_record.jsonl")
    paths["control_record"] = record_path

    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="record",
            memory_path=current_mind,
            out_path=record_path,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_control_record.log"),
        cwd=cwd,
    )

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    write_summary(
        paths=paths,
        archive_store_path=archive_store,
        mind_paths=mind_paths,
        delta_paths=delta_paths,
        summary_path=summary,
    )

    print("\nDone.")
    print(f"current mind: {current_mind}")
    print(f"summary:      {summary}")
    if archive_store:
        print(f"raw archive:  {archive_store}")


if __name__ == "__main__":
    main()
