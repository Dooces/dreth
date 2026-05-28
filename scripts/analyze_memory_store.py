#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


VOLATILE_KEYS = {
    "record_id",
    "proposal_id",
    "run_id",
    "seed",
    "cycle",
    "created_at",
    "updated_at",
    "timestamp",
    "line",
    "first_seen_cycle",
    "last_seen_cycle",
    "last_action_cycle",
    "evidence_epoch",
}

PROVENANCE_KEYS = {
    "evidence_refs",
    "evidence_summary",
    "reason",
    "invalidators",
    "failure_reason",
    "active_nethras",
}

OUTCOME_KEYS = {
    "success",
    "behavior_effect",
    "authority_effect",
    "full_audit_delta",
    "intervention_delta",
    "candidate_reduction_delta",
    "quality_delta",
}


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def strip_keys(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): strip_keys(v, keys)
            for k, v in obj.items()
            if str(k) not in keys
        }
    if isinstance(obj, list):
        return [strip_keys(v, keys) for v in obj]
    return obj


def normalize_lists(obj: Any) -> Any:
    """Sort lists only for known set-like fields; preserve ordered event/probe/candidate fields."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in {
                "touched_atoms",
                "touched_structure_refs",
                "member_nethras",
                "vars",
                "active_atoms",
                "invalidators",
                "required_future_evidence",
                "contradictory_evidence",
                "active_evidence",
            } and isinstance(v, list):
                out[k] = sorted(normalize_lists(x) for x in v)
            else:
                out[k] = normalize_lists(v)
        return out
    if isinstance(obj, list):
        return [normalize_lists(v) for v in obj]
    return obj


def signature_payloads(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    exact = normalize_lists(row)

    semantic = normalize_lists(strip_keys(row, VOLATILE_KEYS))

    structural = normalize_lists(strip_keys(row, VOLATILE_KEYS | PROVENANCE_KEYS | OUTCOME_KEYS))

    # The "role" signature is intentionally coarse: it answers whether the file is
    # repeating the same kind of handle/use/action over and over.
    role = {
        "entry_kind": row.get("entry_kind", row.get("record_type", "")),
        "source": row.get("source", ""),
        "use_right": row.get("use_right", row.get("proposed_use_right", "")),
        "context_scope": row.get("context_scope", row.get("proposed_context_scope", "")),
        "nethra_id": row.get("nethra_id", ""),
        "touched_atoms": sorted(row.get("touched_atoms") or []),
        "touched_structure_refs": sorted(row.get("touched_structure_refs") or []),
        "member_nethras": sorted(row.get("member_nethras") or []),
        "hook": row.get("hook", ""),
    }

    return {
        "exact": exact,
        "semantic": semantic,
        "structural": structural,
        "role": role,
    }


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_kind": row.get("entry_kind", row.get("record_type", "")),
        "source": row.get("source", ""),
        "use_right": row.get("use_right", row.get("proposed_use_right", "")),
        "hook": row.get("hook", ""),
        "cycle": row.get("cycle", None),
        "run_id": row.get("run_id", ""),
        "seed": row.get("seed", ""),
    }


def cycle_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.source_edge.mkdir(source_edges=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            sig_type TEXT NOT NULL,
            sig TEXT NOT NULL,
            count INTEGER NOT NULL,
            bytes_sum INTEGER NOT NULL,
            first_line INTEGER NOT NULL,
            last_line INTEGER NOT NULL,
            first_cycle INTEGER,
            last_cycle INTEGER,
            entry_kind TEXT,
            source TEXT,
            use_right TEXT,
            hook TEXT,
            sample_json TEXT NOT NULL,
            temporal_samples_json TEXT NOT NULL,
            PRIMARY KEY (sig_type, sig)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS kind_counts (
            entry_kind TEXT NOT NULL,
            source TEXT NOT NULL,
            use_right TEXT NOT NULL,
            hook TEXT NOT NULL,
            count INTEGER NOT NULL,
            bytes_sum INTEGER NOT NULL,
            PRIMARY KEY (entry_kind, source, use_right, hook)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS occurrences (
            sig_type TEXT NOT NULL,
            sig TEXT NOT NULL,
            line INTEGER NOT NULL,
            cycle INTEGER,
            run_id TEXT,
            seed TEXT
        )
    """)
    return con


def upsert_group(
    con: sqlite3.Connection,
    *,
    sig_type: str,
    sig: str,
    row_bytes: int,
    line_no: int,
    cyc: int | None,
    meta: dict[str, Any],
    sample_json: str,
    temporal_sample: dict[str, Any],
    temporal_limit: int,
    keep_occurrences: bool,
) -> None:
    cur = con.execute(
        """
        SELECT count, bytes_sum, first_line, last_line, first_cycle, last_cycle,
               temporal_samples_json
        FROM groups
        WHERE sig_type=? AND sig=?
        """,
        (sig_type, sig),
    )
    found = cur.fetchone()

    if found is None:
        temporal_samples = [temporal_sample]
        con.execute(
            """
            INSERT INTO groups (
                sig_type, sig, count, bytes_sum, first_line, last_line,
                first_cycle, last_cycle, entry_kind, source, use_right, hook,
                sample_json, temporal_samples_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig_type,
                sig,
                1,
                row_bytes,
                line_no,
                line_no,
                cyc,
                cyc,
                str(meta["entry_kind"]),
                str(meta["source"]),
                str(meta["use_right"]),
                str(meta["hook"]),
                sample_json,
                json.dumps(temporal_samples, sort_keys=True),
            ),
        )
    else:
        count, bytes_sum, first_line, _last_line, first_cycle, last_cycle, samples_json = found
        samples = json.loads(samples_json)
        if len(samples) < temporal_limit:
            samples.append(temporal_sample)

        if cyc is not None:
            if first_cycle is None or cyc < first_cycle:
                first_cycle = cyc
            if last_cycle is None or cyc > last_cycle:
                last_cycle = cyc

        con.execute(
            """
            UPDATE groups
            SET count=?, bytes_sum=?, last_line=?, first_cycle=?, last_cycle=?,
                temporal_samples_json=?
            WHERE sig_type=? AND sig=?
            """,
            (
                int(count) + 1,
                int(bytes_sum) + row_bytes,
                line_no,
                first_cycle,
                last_cycle,
                json.dumps(samples, sort_keys=True),
                sig_type,
                sig,
            ),
        )

    if keep_occurrences:
        con.execute(
            """
            INSERT INTO occurrences(sig_type, sig, line, cycle, run_id, seed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sig_type,
                sig,
                line_no,
                cyc,
                str(meta.get("run_id", "")),
                str(meta.get("seed", "")),
            ),
        )


def upsert_kind(con: sqlite3.Connection, meta: dict[str, Any], row_bytes: int) -> None:
    key = (
        str(meta["entry_kind"]),
        str(meta["source"]),
        str(meta["use_right"]),
        str(meta["hook"]),
    )
    cur = con.execute(
        """
        SELECT count, bytes_sum
        FROM kind_counts
        WHERE entry_kind=? AND source=? AND use_right=? AND hook=?
        """,
        key,
    )
    found = cur.fetchone()
    if found is None:
        con.execute(
            """
            INSERT INTO kind_counts(entry_kind, source, use_right, hook, count, bytes_sum)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (*key, 1, row_bytes),
        )
    else:
        con.execute(
            """
            UPDATE kind_counts
            SET count=?, bytes_sum=?
            WHERE entry_kind=? AND source=? AND use_right=? AND hook=?
            """,
            (int(found[0]) + 1, int(found[1]) + row_bytes, *key),
        )


def analyze(
    *,
    input_path: Path,
    db_path: Path,
    temporal_limit: int,
    keep_occurrences: bool,
    commit_every: int,
) -> None:
    con = connect(db_path)
    total_rows = 0
    total_bytes = 0
    bad_rows = 0
    start = time.time()

    with input_path.open("rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            total_rows += 1
            row_bytes = len(raw)
            total_bytes += row_bytes

            try:
                row = json.loads(raw)
            except Exception:
                bad_rows += 1
                continue

            if not isinstance(row, dict):
                bad_rows += 1
                continue

            meta = row_meta(row)
            cyc = cycle_int(meta.get("cycle"))
            sample = {
                "line": line_no,
                "cycle": cyc,
                "run_id": meta.get("run_id", ""),
                "seed": meta.get("seed", ""),
            }

            upsert_kind(con, meta, row_bytes)

            payloads = signature_payloads(row)
            for sig_type, payload in payloads.items():
                s = stable_json(payload)
                upsert_group(
                    con,
                    sig_type=sig_type,
                    sig=sha(s),
                    row_bytes=row_bytes,
                    line_no=line_no,
                    cyc=cyc,
                    meta=meta,
                    sample_json=s,
                    temporal_sample=sample,
                    temporal_limit=temporal_limit,
                    keep_occurrences=keep_occurrences,
                )

            if total_rows % commit_every == 0:
                con.commit()
                elapsed = max(0.001, time.time() - start)
                mb = total_bytes / (1024 * 1024)
                print(
                    f"scanned rows={total_rows:,} bad={bad_rows:,} "
                    f"size={mb:,.1f} MiB rate={mb / elapsed:,.1f} MiB/s",
                    flush=True,
                )

    con.commit()
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_groups_type_count
        ON groups(sig_type, count DESC)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_occ_sig
        ON occurrences(sig_type, sig)
    """)
    con.commit()
    con.close()

    print(f"done: rows={total_rows:,} bad_rows={bad_rows:,} bytes={total_bytes:,}")


def write_report(db_path: Path, report_path: Path, *, top: int) -> None:
    con = sqlite3.connect(str(db_path))
    report_path.source_edge.mkdir(source_edges=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as out:
        def emit(s: str = "") -> None:
            print(s)
            out.write(s + "\n")

        emit("Memory Store Redundancy / Compression Report")
        emit("=" * 60)
        emit()

        emit("A. Row classes")
        for row in con.execute("""
            SELECT entry_kind, source, use_right, hook, count, bytes_sum
            FROM kind_counts
            ORDER BY count DESC
        """):
            entry_kind, source, use_right, hook, count, bytes_sum = row
            emit(
                f"{count:12,d} rows  {bytes_sum / (1024*1024):10.1f} MiB  "
                f"entry_kind={entry_kind or '-'} source={source or '-'} "
                f"use_right={use_right or '-'} hook={hook or '-'}"
            )
        emit()

        emit("B. Signature cardinality")
        for sig_type in ["exact", "semantic", "structural", "role"]:
            count_groups, rows_total, bytes_total, duplicate_rows = con.execute(
                """
                SELECT COUNT(*), SUM(count), SUM(bytes_sum),
                       SUM(CASE WHEN count > 1 THEN count - 1 ELSE 0 END)
                FROM groups
                WHERE sig_type=?
                """,
                (sig_type,),
            ).fetchone()
            count_groups = count_groups or 0
            rows_total = rows_total or 0
            bytes_total = bytes_total or 0
            duplicate_rows = duplicate_rows or 0
            compression_ratio = (rows_total / count_groups) if count_groups else 0.0
            emit(
                f"{sig_type:10s} groups={count_groups:12,d} "
                f"rows={rows_total:12,d} duplicate_rows={duplicate_rows:12,d} "
                f"ratio={compression_ratio:8.2f}x bytes={bytes_total/(1024*1024):10.1f} MiB"
            )
        emit()

        emit("C. Top repeated semantic groups")
        for row in con.execute(
            """
            SELECT count, bytes_sum, first_line, last_line, first_cycle, last_cycle,
                   entry_kind, source, use_right, hook, sample_json, temporal_samples_json
            FROM groups
            WHERE sig_type='semantic'
            ORDER BY count DESC
            LIMIT ?
            """,
            (top,),
        ):
            (
                count,
                bytes_sum,
                first_line,
                last_line,
                first_cycle,
                last_cycle,
                entry_kind,
                source,
                use_right,
                hook,
                sample_json,
                samples_json,
            ) = row
            samples = json.loads(samples_json)
            emit(
                f"\ncount={count:,} bytes={bytes_sum/(1024*1024):.2f} MiB "
                f"lines={first_line}-{last_line} cycles={first_cycle}-{last_cycle} "
                f"entry_kind={entry_kind} source={source} use_right={use_right} hook={hook}"
            )
            emit(f"temporal_samples={samples}")
            emit(f"sample={sample_json[:1200]}{'...' if len(sample_json) > 1200 else ''}")
        emit()

        emit("D. Top repeated structural groups")
        for row in con.execute(
            """
            SELECT count, bytes_sum, first_line, last_line, first_cycle, last_cycle,
                   entry_kind, source, use_right, hook, sample_json, temporal_samples_json
            FROM groups
            WHERE sig_type='structural'
            ORDER BY count DESC
            LIMIT ?
            """,
            (top,),
        ):
            (
                count,
                bytes_sum,
                first_line,
                last_line,
                first_cycle,
                last_cycle,
                entry_kind,
                source,
                use_right,
                hook,
                sample_json,
                samples_json,
            ) = row
            samples = json.loads(samples_json)
            emit(
                f"\ncount={count:,} bytes={bytes_sum/(1024*1024):.2f} MiB "
                f"lines={first_line}-{last_line} cycles={first_cycle}-{last_cycle} "
                f"entry_kind={entry_kind} source={source} use_right={use_right} hook={hook}"
            )
            emit(f"temporal_samples={samples}")
            emit(f"sample={sample_json[:1200]}{'...' if len(sample_json) > 1200 else ''}")
        emit()

        emit("E. Optimization implications")
        emit("- exact duplicates can be losslessly folded into one row plus count/provenance.")
        emit("- semantic duplicates can usually become one structure plus temporal provenance, if volatile IDs/cycles are not needed as separate objects.")
        emit("- structural duplicates indicate repeated same-shaped advice/experience; these are candidates for nethra-level consolidation, decay, or credit assignment.")
        emit("- role duplicates indicate broad generic repetition; these are usually dangerous as live assist unless backed by success/lift evidence.")
        emit("- this script does not rewrite the store; it identifies compression opportunities.")

    con.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-analyze a large Dreth nethra memory JSONL store for redundancy."
    )
    p.add_argument("jsonl", type=Path, help="Path to dreth_multisleep_memory.jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("reports/memory_store_analysis"))
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--temporal-samples", type=int, default=12)
    p.add_argument("--commit-every", type=int, default=5000)
    p.add_argument(
        "--keep-occurrences",
        action="store_true",
        help="Store every occurrence line/cycle in SQLite. More complete, but DB can be large.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.jsonl

    if not input_path.exists():
        raise SystemExit(f"not found: {input_path}")

    out_dir = args.out_dir
    out_dir.mkdir(source_edges=True, exist_ok=True)

    db_path = args.db or out_dir / (input_path.stem + "_analysis.sqlite")
    report_path = args.report or out_dir / (input_path.stem + "_redundancy_report.txt")

    print(f"input:  {input_path}")
    print(f"db:     {db_path}")
    print(f"report: {report_path}")
    print("mode:   streaming, disk-backed sqlite aggregation")
    print()

    analyze(
        input_path=input_path,
        db_path=db_path,
        temporal_limit=max(0, args.temporal_samples),
        keep_occurrences=bool(args.keep_occurrences),
        commit_every=max(1, args.commit_every),
    )
    write_report(db_path, report_path, top=max(1, args.top))

    print()
    print(f"wrote report: {report_path}")
    print(f"wrote db:     {db_path}")


if __name__ == "__main__":
    main()