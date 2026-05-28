from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().source_edges[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_comparison_suite as suite
import batch_run


def _args(tmp_path: Path):
    return suite.build_parser().parse_args(
        [
            "--suite",
            "authority_strength",
            "--schedule",
            "blind_challenge",
            "--challenge-blind",
            "--vars",
            "100",
            "--cycles",
            "10000",
            "--seeds",
            "42,99,7",
            "--hybrid-control",
            "interfaces",
            "--repair-agenda",
            "--source_edge-ranker",
            "history_rescue",
            "--probe-proposer",
            "history_rescue",
            "--uncertainty-consolidation",
            "assist",
            "--uncertainty-assist-policy",
            "local_only",
            "--context-role-index",
            "assist_feature",
            "--out-prefix",
            str(tmp_path / "reports" / "authority_strength_compare"),
        ]
    )


def test_command_construction_for_authority_strength_suite(tmp_path: Path) -> None:
    args = _args(tmp_path)
    jobs = suite.build_authority_strength_jobs(args)

    assert [job.label for job in jobs] == [
        "off",
        "record",
        "assist_state_shadow",
        "assist_quarantine_persistent",
        "assist_quarantine_repair_only",
        "assist_legacy",
    ]
    expected = [
        ("off", "state", "off"),
        ("record", "state", "off"),
        ("assist", "state", "shadow"),
        ("assist", "state", "quarantine_persistent"),
        ("assist", "state", "quarantine_repair_only"),
        ("assist", "legacy", "off"),
    ]
    for job, (mode, controller, policy) in zip(jobs, expected):
        assert job.command[:2] == [sys.executable, str(suite.SCRIPTS / "batch_run.py")]
        assert job.command[job.command.index("--authority-strength") + 1] == mode
        assert job.command[job.command.index("--authority-strength-controller") + 1] == controller
        assert job.command[job.command.index("--authority-derivation-policy") + 1] == policy
        assert "--challenge-blind" in job.command
        assert "--repair-agenda" in job.command
        assert job.command[job.command.index("--vars") + 1] == "100"
        assert job.command[job.command.index("--cycles") + 1] == "10000"
        assert job.command[job.command.index("--seeds") + 1] == "42,99,7"


def test_output_filenames_are_deterministic(tmp_path: Path) -> None:
    paths = suite.suite_paths(str(tmp_path / "reports" / "authority_strength_compare"))

    assert paths["off"]["jsonl"].name == "authority_strength_compare_off.jsonl"
    assert paths["off"]["log"].name == "authority_strength_compare_off.log"
    assert paths["record"]["jsonl"].name == "authority_strength_compare_record.jsonl"
    assert (
        paths["assist_state_shadow"]["log"].name
        == "authority_strength_compare_assist_state_shadow.log"
    )
    assert (
        paths["assist_quarantine_repair_only"]["jsonl"].name
        == "authority_strength_compare_assist_quarantine_repair_only.jsonl"
    )
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "authority_strength_compare"),
            "record",
            "authority_evidence",
        ).name
        == "authority_strength_compare_record_authority_evidence.txt"
    )


def test_batch_cli_exposes_authority_derivation_policy() -> None:
    proc = subprocess.run(
        [sys.executable, str(suite.SCRIPTS / "batch_run.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--authority-derivation-policy" in proc.stdout
    assert "quarantine_repair_only" in proc.stdout


def test_run_config_resolves_authority_derivation_policy_defaults() -> None:
    base = {
        "n_vars": 3,
        "cycles": 1,
        "seed": 1,
        "schedule": "blind_challenge",
        "settle_cycles": 0,
        "noise_sigma": 0.0,
    }
    off = batch_run.RunConfig(**base)
    assist = batch_run.RunConfig(
        **base,
        authority_strength="assist",
        authority_strength_controller="state",
    )
    legacy = batch_run.RunConfig(
        **base,
        authority_strength="assist",
        authority_strength_controller="legacy",
    )
    explicit = batch_run.RunConfig(
        **base,
        authority_strength="assist",
        authority_strength_controller="state",
        authority_derivation_policy="quarantine_repair_only",
    )

    assert off.authority_derivation_policy == "off"
    assert assist.authority_derivation_policy == "shadow"
    assert legacy.authority_derivation_policy == "off"
    assert explicit.authority_derivation_policy == "quarantine_repair_only"


def test_summaries_are_invoked_after_runs(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    phases: list[str] = []

    def fake_run(jobs, *, max_workers, sequential=False, terminal=None, popen_factory=None):
        if jobs and jobs[0].label in set(suite.AUTHORITY_STRENGTH_MODES):
            phases.append("batch")
            for mode_paths in suite.suite_paths(args.out_prefix).values():
                mode_paths["jsonl"].source_edge.mkdir(source_edges=True, exist_ok=True)
                mode_paths["jsonl"].write_text("")
                mode_paths["log"].write_text("")
        else:
            assert phases == ["batch"]
            phases.append("summary")
            for job in jobs:
                assert job.log_path.with_suffix(".txt").exists() or job.log_path.suffix == ".txt"
                job.log_path.write_text("")
        return 0

    def fake_write(out_prefix):
        assert phases == ["batch", "summary"]
        path = Path(out_prefix).with_name(f"{Path(out_prefix).name}_comparison.txt")
        path.write_text("Decision Block\nWARN: hidden truth is offline interpretation only\n")
        return path, path.read_text()

    monkeypatch.setattr(suite, "run_labeled_commands", fake_run)
    monkeypatch.setattr(suite, "write_comparison", fake_write)

    assert suite.run_authority_strength_suite(args) == 0
    assert phases == ["batch", "summary"]


def test_comparison_parser_extracts_key_metrics_from_fake_logs() -> None:
    text = """
  runs ok=3/3
  avg: skip%=12.5  iv=40
  quality_cost=900 iv=40 audits=7 revocations=2 unique_fails=1 regime_sentinel_fail=0 no_sentinel=0
  passive monitor: saved_iv=30  stressed=4
  arch avg: route_certs=5.0  audit_certs=6.0  dormant=2.0
relative_authority_frontier_temporal:
  chosen_source_edge_recall=0.750
  recall_lift=2.500
  candidate_reduction_vs_visible=0.600
  invariants: ALL PASS (3 runs)
"""
    metrics = suite.parse_log_metrics(text)

    assert metrics["runs_ok"] == "3/3"
    assert metrics["invariants"] == "ALL PASS (3 runs)"
    assert metrics["skip_pct"] == 12.5
    assert metrics["quality_cost"] == 900
    assert metrics["passive_saved_iv"] == 30
    assert metrics["chosen_source_edge_recall"] == 0.75
    assert metrics["candidate_reduction_vs_visible"] == 0.6


def test_blind_authority_summary_parser_uses_section_totals() -> None:
    text = """
A. External mismatch under authority:
  cases=2
  cases:
      42    0 latent_additive      contradicted_authority              4    9    0     0      0

B. Evidence support level:
  evidence_supported_surrogate          1
  weakly_supported_surrogate            0
  contradicted_authority               29
  insufficient_evidence                 2

F. Shadow authority throttle (mode: conservative):
  would_throttle:                                    31
  would_not_throttle:                                 2
  unthrottled_supported_surrogate:                    1
  estimated_supported_surrogates_preserved:           1
"""

    metrics = suite.parse_blind_authority_evidence_summary(text)

    assert metrics["external_mismatch_cases"] == 2
    assert metrics["contradicted_authority"] == 29
    assert metrics["would_throttle"] == 31
    assert metrics["supported_surrogates_preserved"] == 1


def test_off_vs_record_equality_check_works() -> None:
    off = {field: 1.0 for field in suite.BEHAVIOR_FIELDS}
    record = dict(off)
    assert suite.behavior_equal(off, record)

    record["unique_fails"] = 2.0
    assert not suite.behavior_equal(off, record)
    assert "FAIL: record differs" in "\n".join(
        suite.decision_lines({
            "off": off,
            "record": record,
            "assist_state_shadow": dict(off),
        })
    )


def test_assist_worse_warning_works() -> None:
    off = {field: 1.0 for field in suite.BEHAVIOR_FIELDS}
    off.update({"contradicted_authority": 2, "external_mismatch_cases": 2, "supported_surrogates_preserved": 1})
    assist = dict(off)
    assist["quality_cost"] = 2.0
    assist["contradicted_authority"] = 1

    lines = suite.decision_lines({
        "off": off,
        "record": dict(off),
        "assist_state_shadow": assist,
    })

    assert any(line.startswith("WARN: assist worsens") for line in lines)
    assert any(line.startswith("WARN: assist helps hidden mismatch") for line in lines)


class _FakeStdout(io.StringIO):
    pass


class _FakePopen:
    instances: list["_FakePopen"] = []
    max_active = 0
    active = 0

    def __init__(self, command, **kwargs):
        self.command = command
        self.returncode = 0
        self.done = False
        self.terminated = False
        self.stdout = _FakeStdout(f"{command[-1]} output\n")
        type(self).instances.append(self)
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)

    def poll(self):
        if not self.done:
            self.done = True
            type(self).active -= 1
        return self.returncode

    def wait(self, timeout=None):
        return self.poll()

    def terminate(self):
        self.terminated = True
        if not self.done:
            self.done = True
            type(self).active -= 1

    def kill(self):
        self.terminate()


def test_three_mode_commands_are_launched_concurrently_when_workers_allow(tmp_path: Path) -> None:
    _FakePopen.instances = []
    _FakePopen.max_active = 0
    _FakePopen.active = 0
    jobs = [
        suite.CommandJob(mode, ["cmd", mode], tmp_path / f"{mode}.log")
        for mode in ("off", "record", "assist")
    ]

    code = suite.run_labeled_commands(
        jobs, max_workers=3, terminal=io.StringIO(), popen_factory=_FakePopen
    )

    assert code == 0
    assert _FakePopen.max_active == 3


def test_sequential_launches_one_at_a_time(tmp_path: Path) -> None:
    _FakePopen.instances = []
    _FakePopen.max_active = 0
    _FakePopen.active = 0
    jobs = [
        suite.CommandJob(mode, ["cmd", mode], tmp_path / f"{mode}.log")
        for mode in ("off", "record", "assist")
    ]

    code = suite.run_labeled_commands(
        jobs,
        max_workers=3,
        sequential=True,
        terminal=io.StringIO(),
        popen_factory=_FakePopen,
    )

    assert code == 0
    assert _FakePopen.max_active == 1


class _FailurePopen(_FakePopen):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.returncode = 1 if command[-1] == "record" else None

    def poll(self):
        if self.returncode is None:
            return None
        return super().poll()

    def wait(self, timeout=None):
        if self.returncode is None:
            self.terminate()
            return -15
        return super().wait(timeout=timeout)


def test_failure_in_one_mode_terminates_remaining_modes(tmp_path: Path) -> None:
    _FailurePopen.instances = []
    _FailurePopen.max_active = 0
    _FailurePopen.active = 0
    jobs = [
        suite.CommandJob(mode, ["cmd", mode], tmp_path / f"{mode}.log")
        for mode in ("off", "record", "assist")
    ]

    code = suite.run_labeled_commands(
        jobs, max_workers=3, terminal=io.StringIO(), popen_factory=_FailurePopen
    )

    assert code == 1
    assert any(proc.terminated for proc in _FailurePopen.instances if proc.command[-1] != "record")


def test_terminal_output_is_prefixed_and_logs_remain_separate(tmp_path: Path) -> None:
    terminal = io.StringIO()
    jobs = [
        suite.CommandJob("off", ["cmd", "off"], tmp_path / "off.log"),
        suite.CommandJob("record", ["cmd", "record"], tmp_path / "record.log"),
    ]

    code = suite.run_labeled_commands(
        jobs, max_workers=2, terminal=terminal, popen_factory=_FakePopen
    )

    assert code == 0
    assert "[off] off output" in terminal.getvalue()
    assert "[record] record output" in terminal.getvalue()
    assert (tmp_path / "off.log").read_text() == "off output\n"
    assert (tmp_path / "record.log").read_text() == "record output\n"


def test_final_comparison_waits_for_all_summaries(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    seen_summary_outputs = []

    def fake_run(jobs, *, max_workers, sequential=False, terminal=None, popen_factory=None):
        if jobs and jobs[0].label in set(suite.AUTHORITY_STRENGTH_MODES):
            for mode_paths in suite.suite_paths(args.out_prefix).values():
                mode_paths["jsonl"].source_edge.mkdir(source_edges=True, exist_ok=True)
                mode_paths["jsonl"].write_text("")
                mode_paths["log"].write_text("")
        else:
            for job in jobs:
                job.log_path.write_text(job.label)
                seen_summary_outputs.append(job.log_path)
        return 0

    def fake_write(out_prefix):
        assert len(seen_summary_outputs) == 20
        path = Path(out_prefix).with_name(f"{Path(out_prefix).name}_comparison.txt")
        path.write_text("Decision Block\nWARN: hidden truth is offline interpretation only\n")
        return path, path.read_text()

    monkeypatch.setattr(suite, "run_labeled_commands", fake_run)
    monkeypatch.setattr(suite, "write_comparison", fake_write)

    assert suite.run_authority_strength_suite(args) == 0


def test_subprocess_failure_returns_nonzero(tmp_path: Path) -> None:
    jobs = [suite.CommandJob("record", ["cmd", "record"], tmp_path / "record.log")]

    code = suite.run_labeled_commands(
        jobs, max_workers=1, terminal=io.StringIO(), popen_factory=_FailurePopen
    )

    assert code == 1


# ── background_nethra suite tests ─────────────────────────────────────────────


def _bn_args(tmp_path: Path):
    return suite.build_parser().parse_args(
        [
            "--suite",
            "background_nethra",
            "--schedule",
            "blind_challenge",
            "--challenge-blind",
            "--vars",
            "100",
            "--cycles",
            "10000",
            "--seeds",
            "42,99,7",
            "--hybrid-control",
            "interfaces",
            "--repair-agenda",
            "--source_edge-ranker",
            "history_rescue",
            "--probe-proposer",
            "history_rescue",
            "--uncertainty-consolidation",
            "assist",
            "--uncertainty-assist-policy",
            "local_only",
            "--context-role-index",
            "assist_feature",
            "--authority-strength",
            "record",
            "--out-prefix",
            str(tmp_path / "reports" / "background_nethra_compare"),
        ]
    )


def test_suite_accepts_background_nethra() -> None:
    args = suite.build_parser().parse_args(
        ["--suite", "background_nethra", "--out-prefix", "reports/bg_compare"]
    )
    assert args.suite == "background_nethra"


def test_background_nethra_builds_off_and_record_jobs(tmp_path: Path) -> None:
    args = _bn_args(tmp_path)
    jobs = suite.build_background_nethra_jobs(args)

    assert [j.label for j in jobs] == ["off", "record"]
    off_job, record_job = jobs
    assert off_job.command[:2] == [sys.executable, str(suite.SCRIPTS / "batch_run.py")]
    assert off_job.command[off_job.command.index("--background-nethra") + 1] == "off"
    assert record_job.command[record_job.command.index("--background-nethra") + 1] == "record"
    assert "--challenge-blind" in off_job.command
    assert "--repair-agenda" in off_job.command
    assert off_job.command[off_job.command.index("--authority-strength") + 1] == "record"
    assert record_job.command[record_job.command.index("--authority-strength") + 1] == "record"


def test_background_nethra_output_filenames_are_deterministic(tmp_path: Path) -> None:
    paths = suite.background_nethra_suite_paths(
        str(tmp_path / "reports" / "background_nethra_compare")
    )

    assert paths["off"]["jsonl"].name == "background_nethra_compare_off.jsonl"
    assert paths["off"]["log"].name == "background_nethra_compare_off.log"
    assert paths["record"]["jsonl"].name == "background_nethra_compare_record.jsonl"
    assert paths["record"]["log"].name == "background_nethra_compare_record.log"
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "background_nethra_compare"),
            "record",
            "background_nethra_summary",
        ).name
        == "background_nethra_compare_record_background_nethra_summary.txt"
    )
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "background_nethra_compare"),
            "record",
            "context_role_summary",
        ).name
        == "background_nethra_compare_record_context_role_summary.txt"
    )
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "background_nethra_compare"),
            "record",
            "uncertainty_summary",
        ).name
        == "background_nethra_compare_record_uncertainty_summary.txt"
    )
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "background_nethra_compare"),
            "record",
            "authority_strength_summary",
        ).name
        == "background_nethra_compare_record_authority_strength_summary.txt"
    )


def test_background_nethra_jobs_run_concurrently_when_workers_allow(tmp_path: Path) -> None:
    _FakePopen.instances = []
    _FakePopen.max_active = 0
    _FakePopen.active = 0
    args = _bn_args(tmp_path)
    jobs = suite.build_background_nethra_jobs(args)

    assert len(jobs) == 2
    code = suite.run_labeled_commands(
        jobs, max_workers=2, terminal=io.StringIO(), popen_factory=_FakePopen
    )

    assert code == 0
    assert _FakePopen.max_active == 2


def test_background_nethra_summaries_run_after_batch(tmp_path: Path, monkeypatch) -> None:
    args = _bn_args(tmp_path)
    phases: list[str] = []

    def fake_run(jobs, *, max_workers, sequential=False, terminal=None, popen_factory=None):
        if jobs and jobs[0].label in set(suite.BACKGROUND_NETHRA_MODES):
            phases.append("batch")
            for mode_paths in suite.background_nethra_suite_paths(args.out_prefix).values():
                mode_paths["jsonl"].source_edge.mkdir(source_edges=True, exist_ok=True)
                mode_paths["jsonl"].write_text("")
                mode_paths["log"].write_text("")
        else:
            assert phases == ["batch"]
            phases.append("summary")
            for job in jobs:
                job.log_path.write_text("")
        return 0

    def fake_write(out_prefix):
        assert phases == ["batch", "summary"]
        path = Path(out_prefix).with_name(f"{Path(out_prefix).name}_comparison.txt")
        path.write_text("Decision Block\nPASS: test\n")
        return path, path.read_text()

    monkeypatch.setattr(suite, "run_labeled_commands", fake_run)
    monkeypatch.setattr(suite, "write_background_nethra_comparison", fake_write)

    assert suite.run_background_nethra_suite(args) == 0
    assert phases == ["batch", "summary"]


def test_background_nethra_four_summary_jobs_created(tmp_path: Path) -> None:
    jobs = suite.build_background_nethra_summary_jobs(
        str(tmp_path / "reports" / "bg_compare")
    )
    assert len(jobs) == 4
    scripts = [Path(job.command[1]).name for job in jobs]
    assert "summarize_background_nethra.py" in scripts
    assert "summarize_context_role_index.py" in scripts
    assert "summarize_uncertainty_consolidation.py" in scripts
    assert "summarize_authority_strength.py" in scripts


def test_background_nethra_comparison_parser_detects_off_record_equality() -> None:
    off = {field: 1.0 for field in suite.BN_BEHAVIOR_FIELDS}
    record = dict(off)
    record["background_nethra_records"] = 10
    record["familiar_background_count"] = 5
    record["operational_authority_count"] = 0
    metrics = {"off": off, "record": record}

    lines = suite.background_nethra_decision_lines(metrics)

    assert any("PASS" in line and "off and record match" in line for line in lines)
    assert any("PASS" in line and "background records > 0" in line for line in lines)
    assert any(
        "PASS" in line and "familiar_background_count > 0" in line for line in lines
    )


def test_background_nethra_comparison_parser_detects_behavior_leak() -> None:
    off = {field: 1.0 for field in suite.BN_BEHAVIOR_FIELDS}
    record = dict(off)
    record["skip_pct"] = 2.0
    record["background_nethra_records"] = 10
    record["familiar_background_count"] = 5
    record["operational_authority_count"] = 0
    metrics = {"off": off, "record": record}

    lines = suite.background_nethra_decision_lines(metrics)

    assert any("FAIL" in line and "skip_pct" in line for line in lines)


def test_background_nethra_comparison_parser_detects_zero_background_records() -> None:
    off = {field: 0.0 for field in suite.BN_BEHAVIOR_FIELDS}
    record = dict(off)
    record["background_nethra_records"] = 0
    record["familiar_background_count"] = 0
    record["operational_authority_count"] = 0
    metrics = {"off": off, "record": record}

    lines = suite.background_nethra_decision_lines(metrics)

    assert any("FAIL" in line and "background records == 0" in line for line in lines)


def test_background_nethra_comparison_parser_detects_operational_authority() -> None:
    off = {field: 0.0 for field in suite.BN_BEHAVIOR_FIELDS}
    record = dict(off)
    record["background_nethra_records"] = 10
    record["familiar_background_count"] = 5
    record["operational_authority_count"] = 3
    metrics = {"off": off, "record": record}

    lines = suite.background_nethra_decision_lines(metrics)

    assert any("FAIL" in line and "operational_authority_count" in line for line in lines)


def _sm_args(tmp_path: Path):
    return suite.build_parser().parse_args(
        [
            "--suite",
            "scaffold_memory",
            "--schedule",
            "blind_challenge",
            "--challenge-blind",
            "--vars",
            "100",
            "--cycles",
            "10000",
            "--seeds",
            "3,11,13",
            "--hybrid-control",
            "interfaces",
            "--repair-agenda",
            "--source_edge-ranker",
            "history_rescue",
            "--probe-proposer",
            "history_rescue",
            "--uncertainty-consolidation",
            "assist",
            "--uncertainty-assist-policy",
            "local_only",
            "--context-role-index",
            "assist_feature",
            "--authority-strength",
            "record",
            "--background-nethra",
            "record",
            "--scaffold-memory",
            "reports/auto_sleep_proposals.jsonl",
            "--out-prefix",
            str(tmp_path / "reports" / "scaffold_memory_compare"),
        ]
    )


def test_scaffold_memory_builds_off_record_assist_jobs(tmp_path: Path) -> None:
    args = _sm_args(tmp_path)
    jobs = suite.build_scaffold_memory_jobs(args)

    assert [j.label for j in jobs] == ["off", "record", "assist_feature"]
    for job, mode in zip(jobs, ["off", "record", "assist_feature"]):
        assert job.command[:2] == [sys.executable, str(suite.SCRIPTS / "batch_run.py")]
        assert job.command[job.command.index("--scaffold-memory-mode") + 1] == mode
        assert job.command[job.command.index("--scaffold-memory") + 1] == "reports/auto_sleep_proposals.jsonl"
        assert job.command[job.command.index("--background-nethra") + 1] == "record"


def test_scaffold_memory_output_filenames_are_deterministic(tmp_path: Path) -> None:
    paths = suite.scaffold_memory_suite_paths(
        str(tmp_path / "reports" / "scaffold_memory_compare")
    )

    assert paths["off"]["jsonl"].name == "scaffold_memory_compare_off.jsonl"
    assert paths["record"]["log"].name == "scaffold_memory_compare_record.log"
    assert paths["assist_feature"]["jsonl"].name == "scaffold_memory_compare_assist_feature.jsonl"
    assert (
        suite.summary_output_path(
            str(tmp_path / "reports" / "scaffold_memory_compare"),
            "assist_feature",
            "scaffold_memory_summary",
        ).name
        == "scaffold_memory_compare_assist_feature_scaffold_memory_summary.txt"
    )


def test_scaffold_memory_comparison_detects_behavior_leaks() -> None:
    off = {field: 1.0 for field in suite.BEHAVIOR_FIELDS}
    record = dict(off)
    record["scaffold_memory_loaded_proposals"] = 10
    record["scaffold_memory_matches"] = 5
    assist = dict(record)
    assist["scaffold_memory_ranking_applications"] = 1
    assist["scaffold_memory_authority_allowed_count"] = 0
    assist["scaffold_memory_behavior_effects"] = 1
    metrics = {"off": off, "record": record, "assist_feature": assist}

    lines = suite.scaffold_memory_decision_lines(metrics)

    assert any("PASS" in line and "off == record" in line for line in lines)
    assert any("FAIL" in line and "behavior_effects" in line for line in lines)


def test_reports_directory_is_created(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    reports_dir = Path(args.out_prefix).source_edge

    def fake_run(jobs, *, max_workers, sequential=False, terminal=None, popen_factory=None):
        for mode_paths in suite.suite_paths(args.out_prefix).values():
            mode_paths["jsonl"].write_text("")
            mode_paths["log"].write_text("")
        for job in jobs:
            if job.log_path.suffix == ".txt":
                job.log_path.write_text("")
        return 0

    def fake_write(out_prefix):
        path = Path(out_prefix).with_name(f"{Path(out_prefix).name}_comparison.txt")
        path.write_text("Decision Block\nWARN: hidden truth is offline interpretation only\n")
        return path, path.read_text()

    monkeypatch.setattr(suite, "run_labeled_commands", fake_run)
    monkeypatch.setattr(suite, "write_comparison", fake_write)

    assert suite.run_authority_strength_suite(args) == 0
    assert reports_dir.is_dir()
