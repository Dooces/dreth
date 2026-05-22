#!/usr/bin/env python3
"""
Rich dashboard wrapper for scripts/batch_run.py.

The underlying batch runner stays unchanged. This script passes argv through to
batch_run.py exactly as provided, captures stdout, and renders the stream as a
static terminal dashboard instead of line-by-line output.

Examples:
    python scripts/batch_tui.py
    python scripts/batch_tui.py --compare --vars 8,12
    python scripts/batch_tui.py --vars 5,8 --cycles 100 --seeds 1,2,3
    python scripts/batch_tui.py --vars 8 --cycles 100 --seeds 1 --progress 10
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


ROOT = Path(__file__).resolve().parents[1]
BATCH_RUN = ROOT / "scripts" / "batch_run.py"

RUN_RE = re.compile(r"^\[\s*(?P<done>\d+)/(?P<total>\d+)\]\s*(?P<body>.*)$")
PROGRESS_RE = re.compile(
    r"^\s*(?P<tag>\[[^\]]+\]\s*)?c\s*(?P<cycle>\d+)/(?P<cycles>\d+)\s+"
    r"vis=\s*(?P<vis>\d+)\s+wrong=(?P<wrong>\S+)\s+"
    r"Δiv=\s*(?P<iv>\d+)\s+Δsent=\s*(?P<sent>\d+)(?P<extra>.*)$"
)


class BatchScreen:
    def __init__(self, tail_size: int) -> None:
        self.tail_size = tail_size
        self.started_at = time.monotonic()
        self.banner: List[str] = []
        self.header: Optional[str] = None
        self.run_rows: Deque[Tuple[str, str]] = deque(maxlen=max(1, tail_size))
        self.progress_rows: Deque[str] = deque(maxlen=max(3, tail_size))
        self.messages: Deque[Tuple[str, str]] = deque(maxlen=max(4, tail_size))
        self.aggregate: List[str] = []
        self.done = 0
        self.total = 0
        self.current_cycle: Optional[int] = None
        self.current_cycles: Optional[int] = None
        self.in_aggregate = False
        self.in_column_key = False
        self.return_code: Optional[int] = None

    def consume(self, raw_line: str) -> None:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if "── aggregate " in line:
            self.in_aggregate = True
            self.in_column_key = False
            self.aggregate = [line]
            return
        if "── column key " in line:
            self.in_column_key = True
            self.in_aggregate = False
            self.messages.append(("info", line))
            return

        if self.in_aggregate:
            self.aggregate.append(line)
            return
        if self.in_column_key:
            if stripped:
                self.messages.append(("info", line))
            return

        run_match = RUN_RE.match(line)
        if run_match:
            self.done = int(run_match.group("done"))
            self.total = int(run_match.group("total"))
            body = run_match.group("body")
            style = "green" if "| OK" in body else "red"
            self.run_rows.append((style, body))
            return

        progress_match = PROGRESS_RE.match(line)
        if progress_match:
            self.current_cycle = int(progress_match.group("cycle"))
            self.current_cycles = int(progress_match.group("cycles"))
            self.progress_rows.append(stripped)
            return

        if stripped.startswith("dreth arch-test") or stripped.startswith("workers="):
            self.banner.append(stripped)
            header_match = re.search(r":\s*(\d+)\s+runs\b", stripped)
            if header_match:
                self.total = int(header_match.group(1))
            return
        if stripped.startswith("baseline:") or stripped.startswith("progress:") or stripped.startswith("checking:"):
            self.banner.append(stripped)
            return

        if stripped.startswith("n") or stripped.startswith("-"):
            self.header = stripped
            return

        if stripped.startswith("!!"):
            self.messages.append(("yellow", stripped))
            return
        if stripped.startswith("ERR:") or "Traceback" in line:
            self.messages.append(("red", stripped or line))
            return
        if stripped:
            self.messages.append(("dim", line))

    def render(self) -> Group:
        elapsed = time.monotonic() - self.started_at
        status_style = "cyan"
        status = "running"
        if self.return_code is not None:
            status = "complete" if self.return_code == 0 else f"failed ({self.return_code})"
            status_style = "green" if self.return_code == 0 else "red"

        title = Text.assemble(
            ("dreth batch dashboard", "bold white"),
            ("  "),
            (status, f"bold {status_style}"),
        )

        overview = Table.grid(expand=True)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_row(
            f"[bold]runs[/bold] {self.done}/{self.total or '?'}",
            f"[bold]cycle[/bold] {self._cycle_label()}",
            f"[bold]elapsed[/bold] {elapsed:0.1f}s",
        )
        if self.banner:
            overview.add_row("[dim]" + "\n".join(self.banner[-4:]) + "[/dim]", "", "")

        body = [
            Panel(overview, title=title, border_style=status_style, box=box.ROUNDED),
            self._progress_panel(),
            self._results_panel(),
            self._aggregate_panel(),
            self._messages_panel(),
        ]
        return Group(*body)

    def _cycle_label(self) -> str:
        if self.current_cycle is None or self.current_cycles is None:
            return "-"
        return f"{self.current_cycle}/{self.current_cycles}"

    def _progress_panel(self) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("recent progress", style="cyan", overflow="fold")
        if not self.progress_rows:
            table.add_row("[dim]waiting for progress output[/dim]")
        else:
            for row in self.progress_rows:
                style = "yellow" if "WRG" in row or "wrong={" in row else "cyan"
                table.add_row(f"[{style}]{_escape(row)}[/{style}]")
        return Panel(table, title="Live", border_style="cyan")

    def _results_panel(self) -> Panel:
        table = Table(expand=True, box=box.SIMPLE, show_lines=False)
        table.add_column("#", justify="right", width=4, style="dim")
        table.add_column("result", overflow="fold")
        if not self.run_rows:
            table.add_row("-", "[dim]no completed runs yet[/dim]")
        else:
            offset = max(0, self.done - len(self.run_rows))
            for idx, (style, body) in enumerate(self.run_rows, start=offset + 1):
                table.add_row(str(idx), f"[{style}]{_escape(body)}[/{style}]")
        return Panel(table, title="Completed Runs", border_style="green")

    def _aggregate_panel(self) -> Panel:
        if not self.aggregate:
            content = Align.left("[dim]aggregate appears when batch_run.py finishes[/dim]")
        else:
            content = Text("\n".join(self.aggregate[-18:]), style="white")
        return Panel(content, title="Aggregate", border_style="magenta")

    def _messages_panel(self) -> Panel:
        table = Table(expand=True, box=None, show_header=False)
        table.add_column("messages", overflow="fold")
        if not self.messages:
            table.add_row("[dim]no warnings or extra output[/dim]")
        else:
            for style, msg in self.messages:
                table.add_row(f"[{style}]{_escape(msg)}[/{style}]")
        return Panel(table, title="Messages", border_style="yellow")


def _escape(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _command(argv: Sequence[str]) -> List[str]:
    return [sys.executable, str(BATCH_RUN), *argv]


def main(argv: Optional[Sequence[str]] = None) -> int:
    batch_args = list(argv if argv is not None else sys.argv[1:])
    screen = BatchScreen(tail_size=_env_int("DRETH_TUI_TAIL", 10))
    cmd = _command(batch_args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    interrupted = False

    def _stop_child(signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        if proc.poll() is None:
            proc.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, _stop_child)
    old_term = signal.signal(signal.SIGTERM, _stop_child)
    try:
        with Live(
            screen.render(),
            console=Console(),
            refresh_per_second=max(1.0, _env_float("DRETH_TUI_REFRESH_RATE", 8.0)),
            screen=False,
            transient=False,
        ) as live:
            assert proc.stdout is not None
            for line in proc.stdout:
                screen.consume(line)
                live.update(screen.render())
            screen.return_code = proc.wait()
            live.update(screen.render())
            if screen.return_code != 0:
                time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if proc.poll() is None:
            proc.terminate()

    if interrupted and screen.return_code == 0:
        return 130
    return int(screen.return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
