#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def run_sleep(
    *,
    run_path: Path,
    store_path: Path,
    proposal_path: Path,
    min_sources: int,
    label: str,
    cwd: Path,
) -> dict[str, int]:
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

    store = NethraMemoryStore(store_path)
    written = store.append_sleep_products(products)

    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    with proposal_path.open("w", encoding="utf-8") as fh:
        for p in proposals:
            d = p.to_dict() if hasattr(p, "to_dict") else vars(p)
            fh.write(json.dumps(d, sort_keys=True) + "\n")

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
        "appended_products": written,
        "authority_allowed_products": sum(
            1 for p in products if bool(getattr(p, "authority_allowed", False))
        ),
        "authority_allowed_proposals": sum(
            1 for p in proposals if bool(getattr(p, "authority_allowed", False))
        ),
    }

    print(f"\n=== SLEEP {label} ===")
    print(
        "sleep_input: "
        f"bg={result['bg']} cr={result['cr']} unc={result['unc']} "
        f"auth={result['auth']} temp={result['temp']} exp={result['exp']} mem={result['mem']}"
    )
    print(
        "sleep_output: "
        f"proposals={result['proposals']} products={result['products']} "
        f"appended_products={result['appended_products']}"
    )
    print(
        "authority_allowed: "
        f"products={result['authority_allowed_products']} "
        f"proposals={result['authority_allowed_proposals']}"
    )

    if result["authority_allowed_products"] or result["authority_allowed_proposals"]:
        raise SystemExit("sleep emitted authority_allowed=True; aborting")

    return result


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


def write_summary(paths: dict[str, Path], summary_path: Path) -> None:
    data = {name: totals(path) for name, path in paths.items()}

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        def emit(s: str = "") -> None:
            print(s)
            fh.write(s + "\n")

        emit("=== dreth multisleep loop summary ===")
        emit(
            "mode rows events records loaded sleep_loaded sleep_used "
            "behavior authority cand_reorders probe_reorders audits iv qcost rev unique"
        )

        for name, d in data.items():
            emit(
                f"{name} "
                f"{int(d['rows'])} "
                f"{int(d['experience_events_written'])} "
                f"{int(d['nethra_memory_records_written'])} "
                f"{int(d['persistent_nethras_loaded'])} "
                f"{int(d['sleep_products_loaded'])} "
                f"{int(d['sleep_products_used'])} "
                f"{int(d['nethra_memory_behavior_effects'])} "
                f"{int(d['nethra_memory_authority_effects'])} "
                f"{int(d['nethra_memory_candidate_reorders'])} "
                f"{int(d['nethra_memory_probe_reorders'])} "
                f"{int(d['full_audits'])} "
                f"{int(d['interventions'])} "
                f"{int(d['quality_cost'])} "
                f"{int(d['revocations'])} "
                f"{int(d['unique_fails'])}"
            )

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
        any_assist_behavior_effects = any(
            d.get("nethra_memory_behavior_effects", 0) > 0
            for name, d in data.items()
            if "assist" in name
        )
        any_sleep_products_used = any(
            d.get("sleep_products_used", 0) > 0
            for name, d in data.items()
            if "assist" in name
        )

        emit()
        emit("=== gates ===")
        emit(f"off_record_behavior_equal: {off_record_behavior_equal}")
        emit(f"record_behavior_effects_zero: {record_behavior_effects_zero}")
        emit(f"authority_effects_zero_all: {authority_effects_zero_all}")
        emit(f"any_assist_behavior_effects: {any_assist_behavior_effects}")
        emit(f"any_sleep_products_used: {any_sleep_products_used}")

        emit()
        emit("=== generation deltas vs previous assist generation ===")
        assist_names = [name for name in data if name.startswith("run") and name.endswith("_assist")]
        assist_names.sort(key=lambda x: int(x.removeprefix("run").removesuffix("_assist")))

        prev: dict[str, int | float] | None = None
        for name in assist_names:
            cur = data[name]
            if prev is None:
                emit(f"{name}: baseline assist generation")
            else:
                emit(
                    f"{name}: "
                    f"Δbehavior={int(cur['nethra_memory_behavior_effects'] - prev['nethra_memory_behavior_effects'])} "
                    f"Δsleep_used={int(cur['sleep_products_used'] - prev['sleep_products_used'])} "
                    f"Δaudits={int(cur['full_audits'] - prev['full_audits'])} "
                    f"Δiv={int(cur['interventions'] - prev['interventions'])} "
                    f"Δqcost={int(cur['quality_cost'] - prev['quality_cost'])}"
                )
            prev = cur

    print(f"\nsummary written: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-generation Dreth RUN→SLEEP→RUN memory loop."
    )
    p.add_argument("--vars", type=int, default=50)
    p.add_argument("--cycles", type=int, default=3000)
    p.add_argument("--seeds", default="42,99,7")
    p.add_argument("--schedule", default="blind_challenge")
    p.add_argument("--gens", type=int, default=4)
    p.add_argument("--out-prefix", default="reports/dreth_multisleep")
    p.add_argument("--min-sources", type=int, default=2)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true", help="delete old files with this prefix first")

    p.add_argument(
        "batch_flags",
        nargs=argparse.REMAINDER,
        help="extra flags passed to scripts/batch_run.py after --",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cwd = repo_root()

    prefix = Path(args.out_prefix)
    if not prefix.is_absolute():
        prefix = cwd / prefix

    reports_dir = prefix.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    store = prefix.with_name(prefix.name + "_memory.jsonl")
    summary = prefix.with_name(prefix.name + "_summary.txt")

    extra_flags = list(args.batch_flags or [])
    if extra_flags and extra_flags[0] == "--":
        extra_flags = extra_flags[1:]

    if args.force:
        for path in reports_dir.glob(prefix.name + "*"):
            if path.is_file():
                path.unlink()
        if store.exists():
            store.unlink()

    paths: dict[str, Path] = {}

    print("=== RUN 0: record-only bootstrap ===")
    run0 = prefix.with_name(prefix.name + "_run0_record.jsonl")
    paths["run0_record"] = run0
    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="record",
            memory_path=store,
            out_path=run0,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_run0_record.log"),
        cwd=cwd,
    )

    run_sleep(
        run_path=run0,
        store_path=store,
        proposal_path=prefix.with_name(prefix.name + "_sleep0_proposals.jsonl"),
        min_sources=args.min_sources,
        label="0",
        cwd=cwd,
    )

    for gen in range(1, args.gens + 1):
        print(f"\n=== RUN {gen}: assist using accumulated memory ===")
        run_path = prefix.with_name(prefix.name + f"_run{gen}_assist.jsonl")
        paths[f"run{gen}_assist"] = run_path

        run_and_tee(
            batch_cmd(
                vars_=args.vars,
                cycles=args.cycles,
                seeds=args.seeds,
                schedule=args.schedule,
                memory_mode="assist",
                memory_path=store,
                out_path=run_path,
                workers=args.workers,
                extra_flags=extra_flags,
            ),
            prefix.with_name(prefix.name + f"_run{gen}_assist.log"),
            cwd=cwd,
        )

        run_sleep(
            run_path=run_path,
            store_path=store,
            proposal_path=prefix.with_name(prefix.name + f"_sleep{gen}_proposals.jsonl"),
            min_sources=args.min_sources,
            label=str(gen),
            cwd=cwd,
        )

    print("\n=== CONTROL: off after accumulated memory ===")
    off_path = prefix.with_name(prefix.name + "_control_off.jsonl")
    paths["control_off"] = off_path
    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="off",
            memory_path=store,
            out_path=off_path,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_control_off.log"),
        cwd=cwd,
    )

    print("\n=== CONTROL: record after accumulated memory ===")
    record_path = prefix.with_name(prefix.name + "_control_record.jsonl")
    paths["control_record"] = record_path
    run_and_tee(
        batch_cmd(
            vars_=args.vars,
            cycles=args.cycles,
            seeds=args.seeds,
            schedule=args.schedule,
            memory_mode="record",
            memory_path=store,
            out_path=record_path,
            workers=args.workers,
            extra_flags=extra_flags,
        ),
        prefix.with_name(prefix.name + "_control_record.log"),
        cwd=cwd,
    )

    print("\n=== SUMMARY ===")
    write_summary(paths, summary)

    print("\nDone.")
    print(f"memory store: {store}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()