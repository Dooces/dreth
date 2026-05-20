#!/usr/bin/env python3
"""
drive_inventory.py

Concurrent persistent metadata inventory for mounted drives.

What it does:
- scans multiple roots concurrently into one SQLite database
- generates biggest-file and possible-duplicate reports
- searches paths instantly after scanning
- compares two folder trees by relative path + size
- optionally hashes narrowed candidate lists concurrently

It is metadata-first. It does not hash whole drives unless you explicitly feed it paths to hash.

Typical use:

  python3 drive_inventory.py scan --reset --workers 7 \
    /mnt/1CC86BE5C86BBC20 \
    /mnt/420C7B340C7B2259 \
    /mnt/wwn-0x5001b448b5319a5c-part2 \
    /mnt/wwn-0x5001b448b5319a5c-part3 \
    /mnt/71683CA872B57B21 \
    /media/dooces/44968C2C968C2112 \
    /mnt/seagate2tb

  python3 drive_inventory.py stats
  python3 drive_inventory.py biggest --limit 300
  python3 drive_inventory.py dupnames --limit 1000
  python3 drive_inventory.py search "Clair Obscur"

  python3 drive_inventory.py compare-trees \
    "/mnt/1CC86BE5C86BBC20/1TB WD 1.7.2026 copy/Games/Clair Obscur Expedition 33" \
    "/mnt/wwn-0x5001b448b5319a5c-part2/Games/Clair Obscur Expedition 33"

  python3 drive_inventory.py hash-list --workers 4 --input ~/drive-audit/reports/compare_same_files.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import queue
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

DEFAULT_DB = Path.home() / "drive-audit" / "files.db"
DEFAULT_REPORT_DIR = Path.home() / "drive-audit" / "reports"

BATCH_SIZE = 5000
QUEUE_MAX_BATCHES = 64


@dataclass(frozen=True)
class FileRow:
    root: str
    path: str
    relpath: str
    name: str
    ext: str
    size: int
    mtime_ns: int
    mtime_iso: str


@dataclass
class RootScanResult:
    root: str
    files: int
    bytes_total: int
    seconds: float
    errors: int


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect(db: Path) -> sqlite3.Connection:
    ensure_parent(db)
    con = sqlite3.connect(str(db), timeout=60)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA cache_size=-200000;")
    return con


def init_db(con: sqlite3.Connection, reset: bool) -> None:
    if reset:
        con.executescript(
            """
            DROP TABLE IF EXISTS files;
            DROP TABLE IF EXISTS scan_roots;
            DROP TABLE IF EXISTS hashes;
            """
        )

    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS scan_roots (
            root TEXT PRIMARY KEY,
            scanned_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            root TEXT NOT NULL,
            path TEXT PRIMARY KEY,
            relpath TEXT NOT NULL,
            name TEXT NOT NULL,
            ext TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            mtime_iso TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS hashes (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            hashed_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_files_size ON files(size);
        CREATE INDEX IF NOT EXISTS idx_files_name_size ON files(name, size);
        CREATE INDEX IF NOT EXISTS idx_files_root ON files(root);
        CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);
        CREATE INDEX IF NOT EXISTS idx_hashes_sha256 ON hashes(sha256);
        """
    )
    con.commit()


def format_size(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} B"
        x /= 1024
    return f"{n} B"


def mtime_iso(ns: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ns / 1_000_000_000))


def iter_files(root: Path) -> Iterator[FileRow]:
    root = root.resolve()
    root_s = str(root)
    try:
        root_dev = os.stat(root_s).st_dev
    except OSError as e:
        print(f"SKIP cannot stat root {root_s}: {e}", file=sys.stderr)
        return

    for dirpath, dirnames, filenames in os.walk(root_s, topdown=True, followlinks=False):
        try:
            if os.stat(dirpath).st_dev != root_dev:
                dirnames[:] = []
                continue
        except OSError:
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in {".Trash-1000", "$RECYCLE.BIN", "System Volume Information"}]

        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            if not os.path.isfile(p):
                continue

            path_s = str(p)
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = path_s

            yield FileRow(
                root=root_s,
                path=path_s,
                relpath=rel,
                name=p.name,
                ext=p.suffix.lower(),
                size=int(st.st_size),
                mtime_ns=int(st.st_mtime_ns),
                mtime_iso=mtime_iso(int(st.st_mtime_ns)),
            )


def scanner_worker(root: Path, out_q: queue.Queue, stop_event: threading.Event) -> RootScanResult:
    start = time.time()
    root = root.resolve()
    root_s = str(root)
    batch: list[tuple] = []
    count = 0
    total_bytes = 0
    errors = 0
    last_report = start

    try:
        for r in iter_files(root):
            if stop_event.is_set():
                break

            batch.append((r.root, r.path, r.relpath, r.name, r.ext, r.size, r.mtime_ns, r.mtime_iso))
            count += 1
            total_bytes += r.size

            if len(batch) >= BATCH_SIZE:
                out_q.put(("batch", root_s, batch))
                batch = []

            now = time.time()
            if now - last_report >= 10:
                rate = count / max(now - start, 0.001)
                print(f"[{root_s}] {count:,} files, {format_size(total_bytes)}, {rate:,.0f} files/sec")
                last_report = now

    except Exception as e:
        errors += 1
        out_q.put(("error", root_s, repr(e)))

    if batch:
        out_q.put(("batch", root_s, batch))

    seconds = max(time.time() - start, 0.001)
    out_q.put(("root_done", root_s, count, total_bytes, seconds, errors))
    return RootScanResult(root=root_s, files=count, bytes_total=total_bytes, seconds=seconds, errors=errors)


def writer_worker(db: Path, in_q: queue.Queue, expected_roots: int, done_event: threading.Event) -> None:
    con = connect(db)
    inserted = 0
    completed_roots = 0
    last_commit = time.time()

    try:
        while completed_roots < expected_roots:
            item = in_q.get()
            kind = item[0]

            if kind == "batch":
                _, root_s, batch = item
                con.executemany(
                    """
                    INSERT OR REPLACE INTO files
                    (root, path, relpath, name, ext, size, mtime_ns, mtime_iso)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                inserted += len(batch)

                now = time.time()
                if now - last_commit >= 2:
                    con.commit()
                    last_commit = now
                    print(f"[writer] inserted {inserted:,} rows")

            elif kind == "root_done":
                _, root_s, count, bytes_total, seconds, errors = item
                con.execute("INSERT OR REPLACE INTO scan_roots(root, scanned_at) VALUES (?, ?)", (root_s, int(time.time())))
                con.commit()
                completed_roots += 1
                print(f"[done] {root_s}: {count:,} files, {format_size(bytes_total)}, {seconds:.1f}s, errors={errors}")

            elif kind == "error":
                _, root_s, msg = item
                print(f"[error] {root_s}: {msg}", file=sys.stderr)

            in_q.task_done()

        con.commit()
    finally:
        con.close()
        done_event.set()


def scan_roots(db: Path, roots: list[str], reset: bool, workers: int) -> None:
    cleaned_roots: list[Path] = []
    for root_arg in roots:
        root = Path(root_arg)
        if not root.exists():
            print(f"SKIP missing: {root}", file=sys.stderr)
            continue
        if not root.is_dir():
            print(f"SKIP not directory: {root}", file=sys.stderr)
            continue
        cleaned_roots.append(root.resolve())

    if not cleaned_roots:
        fail("no valid roots to scan")

    if workers < 1:
        workers = 1
    workers = min(workers, len(cleaned_roots))

    con = connect(db)
    init_db(con, reset=reset)
    for root in cleaned_roots:
        con.execute("DELETE FROM files WHERE root = ?", (str(root),))
        con.execute("DELETE FROM scan_roots WHERE root = ?", (str(root),))
    con.commit()
    con.close()

    print(f"Scanning {len(cleaned_roots)} roots with {workers} concurrent scanner workers")
    print(f"DB: {db}")

    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_BATCHES)
    stop_event = threading.Event()
    writer_done = threading.Event()

    writer = threading.Thread(target=writer_worker, args=(db, q, len(cleaned_roots), writer_done), daemon=True)
    writer.start()

    start = time.time()
    results: list[RootScanResult] = []

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(scanner_worker, root, q, stop_event) for root in cleaned_roots]
            for fut in as_completed(futs):
                results.append(fut.result())
    except KeyboardInterrupt:
        print("Interrupted; asking scanners to stop...", file=sys.stderr)
        stop_event.set()
        raise
    finally:
        writer_done.wait(timeout=300)

    elapsed = max(time.time() - start, 0.001)
    total_files = sum(r.files for r in results)
    total_bytes = sum(r.bytes_total for r in results)
    print()
    print("Finished scan")
    print(f"  roots: {len(results):,}")
    print(f"  files: {total_files:,}")
    print(f"  bytes: {format_size(total_bytes)}")
    print(f"  time:  {elapsed:.1f}s")
    print(f"  rate:  {total_files / elapsed:,.0f} files/sec")


def write_tsv(rows: Iterable[tuple], headers: list[str], out: Path) -> None:
    ensure_parent(out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"Wrote: {out}")


def report_biggest(db: Path, limit: int, outdir: Path) -> None:
    con = connect(db)
    rows = con.execute(
        """
        SELECT size, root, path, mtime_iso
        FROM files
        ORDER BY size DESC
        LIMIT ?
        """,
        (limit,),
    )
    out = outdir / f"biggest_{limit}.tsv"
    write_tsv(((format_size(s), s, root, path, mt) for s, root, path, mt in rows),
              ["human_size", "bytes", "root", "path", "mtime"], out)
    con.close()


def report_dupnames(db: Path, limit: int, outdir: Path, min_size: int) -> None:
    con = connect(db)
    rows = con.execute(
        """
        WITH d AS (
          SELECT name, size, COUNT(*) AS copies
          FROM files
          WHERE size >= ?
          GROUP BY name, size
          HAVING copies > 1
          ORDER BY size DESC
          LIMIT ?
        )
        SELECT f.size, d.copies, f.name, f.root, f.path, f.mtime_iso
        FROM d
        JOIN files f ON f.name = d.name AND f.size = d.size
        ORDER BY f.size DESC, f.name COLLATE NOCASE, f.path COLLATE NOCASE
        """,
        (min_size, limit),
    )
    out = outdir / f"possible_duplicates_by_name_size_top_{limit}.tsv"
    write_tsv(((format_size(s), s, copies, name, root, path, mt) for s, copies, name, root, path, mt in rows),
              ["human_size", "bytes", "copies", "name", "root", "path", "mtime"], out)
    con.close()


def search_inventory(db: Path, term: str, limit: int, outdir: Path) -> None:
    con = connect(db)
    rows = con.execute(
        """
        SELECT size, root, path, mtime_iso
        FROM files
        WHERE path LIKE ?
        ORDER BY size DESC
        LIMIT ?
        """,
        (f"%{term}%", limit),
    )
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in term)[:80]
    out = outdir / f"search_{safe}.tsv"
    write_tsv(((format_size(s), s, root, path, mt) for s, root, path, mt in rows),
              ["human_size", "bytes", "root", "path", "mtime"], out)
    con.close()


def collect_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    d: dict[str, tuple[int, int, str]] = {}
    for row in iter_files(root):
        d[row.relpath] = (row.size, row.mtime_ns, row.path)
    return d


def compare_trees(a: str, b: str, outdir: Path, workers: int) -> None:
    pa = Path(a).resolve()
    pb = Path(b).resolve()
    if not pa.is_dir():
        fail(f"not a directory: {pa}")
    if not pb.is_dir():
        fail(f"not a directory: {pb}")

    print(f"Collecting A and B concurrently")
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as ex:
        fa = ex.submit(collect_tree, pa)
        fb = ex.submit(collect_tree, pb)
        da = fa.result()
        dbb = fb.result()

    only_a = sorted(set(da) - set(dbb))
    only_b = sorted(set(dbb) - set(da))
    common = sorted(set(da) & set(dbb))
    same = []
    different_size = []

    for rel in common:
        sa, _, path_a = da[rel]
        sb, _, path_b = dbb[rel]
        if sa == sb:
            same.append((rel, sa, path_a, path_b))
        else:
            different_size.append((rel, sa, sb, path_a, path_b))

    outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(((rel, format_size(size), size, a_path, b_path) for rel, size, a_path, b_path in same),
              ["relpath", "human_size", "bytes", "path_a", "path_b"], outdir / "compare_same_files.tsv")
    write_tsv(((rel, format_size(sa), sa, format_size(sb), sb, a_path, b_path)
               for rel, sa, sb, a_path, b_path in different_size),
              ["relpath", "a_human_size", "a_bytes", "b_human_size", "b_bytes", "path_a", "path_b"],
              outdir / "compare_different_size.tsv")
    write_tsv(((rel, da[rel][2]) for rel in only_a), ["relpath", "path"], outdir / "compare_only_a.tsv")
    write_tsv(((rel, dbb[rel][2]) for rel in only_b), ["relpath", "path"], outdir / "compare_only_b.tsv")

    bytes_same = sum(x[1] for x in same)
    print()
    print("Summary")
    print(f"  A files:           {len(da):,}")
    print(f"  B files:           {len(dbb):,}")
    print(f"  same relpath+size: {len(same):,} ({format_size(bytes_same)})")
    print(f"  different sizes:   {len(different_size):,}")
    print(f"  only A:            {len(only_a):,}")
    print(f"  only B:            {len(only_b):,}")
    if not only_a and not only_b and not different_size:
        print("  VERDICT: same file tree by relative path + size.")
    else:
        print("  VERDICT: not identical by metadata.")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_one(path_s: str) -> Optional[tuple[str, str, int, int]]:
    path = Path(path_s)
    try:
        st = path.stat()
        digest = sha256_file(path)
        return (path_s, digest, int(st.st_size), int(st.st_mtime_ns))
    except OSError as e:
        print(f"SKIP {path}: {e}", file=sys.stderr)
        return None


def hash_list(db: Path, input_file: str, outdir: Path, workers: int) -> None:
    p = Path(input_file)
    if not p.exists():
        fail(f"input not found: {p}")

    paths: list[str] = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            fail("empty input")
        for row in reader:
            for cell in row:
                if cell.startswith("/") and Path(cell).exists():
                    paths.append(cell)

    seen = set()
    paths = [x for x in paths if not (x in seen or seen.add(x))]

    con = connect(db)
    init_db(con, reset=False)

    cached_rows: list[tuple[str, str, int, int]] = []
    to_hash: list[str] = []

    for path_s in paths:
        try:
            st = Path(path_s).stat()
        except OSError:
            continue
        cached = con.execute(
            "SELECT sha256 FROM hashes WHERE path=? AND size=? AND mtime_ns=?",
            (path_s, int(st.st_size), int(st.st_mtime_ns)),
        ).fetchone()
        if cached:
            cached_rows.append((path_s, cached[0], int(st.st_size), int(st.st_mtime_ns)))
        else:
            to_hash.append(path_s)

    print(f"Paths: {len(paths):,}; cached: {len(cached_rows):,}; hashing: {len(to_hash):,}; workers={workers}")

    new_rows: list[tuple[str, str, int, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(hash_one, p) for p in to_hash]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is None:
                continue
            new_rows.append(res)
            path_s, digest, size, mtime_ns = res
            con.execute(
                """
                INSERT OR REPLACE INTO hashes(path, size, mtime_ns, sha256, hashed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (path_s, size, mtime_ns, digest, int(time.time())),
            )
            if i % 100 == 0:
                con.commit()
                print(f"  hashed {i:,}/{len(to_hash):,}")
    con.commit()

    rows = cached_rows + new_rows
    out = outdir / "sha256_hashes.tsv"
    ensure_parent(out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sha256", "human_size", "bytes", "path"])
        for path_s, digest, size, _mtime_ns in sorted(rows, key=lambda x: (x[1], x[0])):
            w.writerow([digest, format_size(size), size, path_s])

    print(f"Wrote: {out}")
    con.close()


def roots(db: Path) -> None:
    con = connect(db)
    for row in con.execute("SELECT root, datetime(scanned_at, 'unixepoch', 'localtime') FROM scan_roots ORDER BY root"):
        print(f"{row[1]}\t{row[0]}")
    con.close()


def stats(db: Path) -> None:
    con = connect(db)
    total = con.execute("SELECT COUNT(*), COALESCE(SUM(size),0) FROM files").fetchone()
    print(f"Files: {total[0]:,}")
    print(f"Bytes: {format_size(total[1])}")
    print()
    for root, count, size in con.execute(
        "SELECT root, COUNT(*), COALESCE(SUM(size),0) FROM files GROUP BY root ORDER BY size DESC"
    ):
        print(f"{format_size(size):>12}  {count:>10,}  {root}")
    con.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrent persistent drive inventory and duplicate triage.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite database path. Default: {DEFAULT_DB}")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR, help=f"Report directory. Default: {DEFAULT_REPORT_DIR}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="Scan roots into SQLite metadata database concurrently.")
    p.add_argument("roots", nargs="+")
    p.add_argument("--reset", action="store_true", help="Delete existing DB tables before scan.")
    p.add_argument("--workers", type=int, default=4, help="Concurrent root scanners. Use 1 to force serial scan.")

    sub.add_parser("stats", help="Show inventory stats.")
    sub.add_parser("roots", help="Show scanned roots.")

    p = sub.add_parser("biggest", help="Write biggest files report.")
    p.add_argument("--limit", type=int, default=200)

    p = sub.add_parser("dupnames", help="Write possible duplicate report by exact filename + exact size.")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--min-size", type=int, default=1024 * 1024, help="Ignore files smaller than this many bytes.")

    p = sub.add_parser("search", help="Search inventory paths.")
    p.add_argument("term")
    p.add_argument("--limit", type=int, default=1000)

    p = sub.add_parser("compare-trees", help="Compare two folder trees by relative path and size.")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--workers", type=int, default=2)

    p = sub.add_parser("hash-list", help="SHA256 hash paths found in a TSV report concurrently.")
    p.add_argument("--input", required=True)
    p.add_argument("--workers", type=int, default=4)

    args = parser.parse_args(argv)

    if args.cmd == "scan":
        scan_roots(args.db, args.roots, reset=args.reset, workers=args.workers)
    elif args.cmd == "stats":
        stats(args.db)
    elif args.cmd == "roots":
        roots(args.db)
    elif args.cmd == "biggest":
        report_biggest(args.db, args.limit, args.reports)
    elif args.cmd == "dupnames":
        report_dupnames(args.db, args.limit, args.reports, args.min_size)
    elif args.cmd == "search":
        search_inventory(args.db, args.term, args.limit, args.reports)
    elif args.cmd == "compare-trees":
        compare_trees(args.a, args.b, args.reports, workers=args.workers)
    elif args.cmd == "hash-list":
        hash_list(args.db, args.input, args.reports, workers=args.workers)
    else:
        fail(f"unknown command: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
