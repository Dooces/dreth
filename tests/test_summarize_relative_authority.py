from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from summarize_relative_authority import (  # noqa: E402
    compute_group_summaries,
    compute_top_example_counts,
    load_jsonl,
    print_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "relative_authority.jsonl"
    _write_jsonl(
        path,
        [
            {
                "record_type": "policy_report",
                "schedule": "false_trass",
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "relative_authority_nodes": 999,
                "relative_authority_relations": 999,
                "relative_authority_records": 999,
                "relative_authority_relation_types": {
                    "exception_to": 999,
                    "shares_node": 999,
                },
                "relative_authority_top_examples": ["ignored:999.0"],
            },
            {
                "schedule": "false_trass",
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "relative_authority_nodes": 10,
                "relative_authority_relations": 5,
                "relative_authority_records": 7,
                "relative_authority_relation_types": {
                    "depends_on": 2,
                    "exception_to": 1,
                    "shares_node": 2,
                },
                "relative_authority_top_examples": ["var:1:10.0", "cert:2:add:5.0"],
            },
            {
                "schedule": "false_trass",
                "parent_ranker": "sensitivity",
                "probe_proposer": "none",
                "relative_authority_nodes": 20,
                "relative_authority_relations": 10,
                "relative_authority_records": 9,
                "relative_authority_relation_types": {
                    "depends_on": 3,
                    "exception_to": 2,
                    "shares_node": 1,
                    "conflicts_with": 4,
                },
                "relative_authority_top_examples": ["var:1:8.0", "var:3:1.0"],
            },
            {
                "schedule": "regime_switch",
                "parent_ranker": "history_rescue",
                "probe_proposer": "history_rescue",
                "relative_authority_nodes": 8,
                "relative_authority_relations": 4,
                "relative_authority_records": 5,
                "relative_authority_relation_types": {
                    "depends_on": 1,
                    "coactive_with": 2,
                    "shares_node": 1,
                },
                "relative_authority_top_examples": ["var:4:2.0"],
            },
        ],
    )
    return load_jsonl(str(path))


def _summary(rows, schedule: str, policy: str):
    summaries = compute_group_summaries(rows)
    return next(s for s in summaries if s.schedule == schedule and s.policy == policy)


def test_ignores_policy_report_rows(rows):
    assert len(rows) == 3
    counts = compute_top_example_counts(rows)
    assert "ignored" not in counts


def test_groups_by_schedule_and_policy(rows):
    summaries = compute_group_summaries(rows)
    keys = {(s.schedule, s.policy) for s in summaries}

    assert keys == {
        ("false_trass", "sensitivity/none"),
        ("regime_switch", "history_rescue/history_rescue"),
    }
    false_trass = _summary(rows, "false_trass", "sensitivity/none")
    assert false_trass.runs == 2
    assert false_trass.avg_nodes == pytest.approx(15.0)
    assert false_trass.avg_relations == pytest.approx(7.5)
    assert false_trass.avg_authority_records == pytest.approx(8.0)


def test_aggregates_relation_types_correctly(rows):
    false_trass = _summary(rows, "false_trass", "sensitivity/none")

    assert false_trass.relation_types["depends_on"] == 5
    assert false_trass.relation_types["exception_to"] == 3
    assert false_trass.relation_types["shares_node"] == 3
    assert false_trass.relation_types["conflicts_with"] == 4
    assert false_trass.relation_types["coactive_with"] == 0


def test_computes_exception_density(rows):
    false_trass = _summary(rows, "false_trass", "sensitivity/none")

    assert false_trass.exception_density == pytest.approx(3 / 15)


def test_computes_shares_node_density(rows):
    false_trass = _summary(rows, "false_trass", "sensitivity/none")

    assert false_trass.shares_node_density == pytest.approx(3 / 15)


def test_top_authority_examples_count_repeated_node_ids(rows):
    counts = compute_top_example_counts(rows)

    assert counts["var:1"] == 2
    assert counts["cert:2:add"] == 1
    assert counts["var:3"] == 1


def test_prints_diagnostic_only_warning(rows, capsys):
    print_report(rows)
    output = capsys.readouterr().out

    assert "A. Mean graph size by schedule/policy:" in output
    assert "B. Relation type distribution by schedule/policy:" in output
    assert "Diagnostic only" in output
    assert "does not drive behavior" in output
