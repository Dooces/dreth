from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_blind_authority_evidence import (
    classify_evidence_support,
    load_jsonl,
    print_report,
    summarize,
)


def _item(**overrides):
    base = {
        "var": 0,
        "relation_type": "delayed",
        "truth_parents": [1],
        "truth_delayed_parents": [],
        "learned_parents": [2],
        "learned_func": "FIRST",
        "authoritative": True,
        "status": "certified",
        "skip_role": "tareth",
    }
    base.update(overrides)
    return base


def _row(item):
    return {
        "seed": 123,
        "schedule": "blind_challenge",
        "evaluation": {
            "blind_challenge_behavior": {
                "per_var": [item],
            }
        },
    }


def test_hidden_mismatch_alone_is_not_classified_as_failure() -> None:
    item = _item()

    assert classify_evidence_support(item) == "unknown"
    summary = summarize([_row(item)])
    assert summary.by_classification["unknown"] == 1
    assert not summary.serious_mismatch_candidates


def test_stable_evidence_hidden_mismatch_is_supported_surrogate() -> None:
    item = _item(
        strong_observations=4,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        alternatives_existed=False,
        repeatedly_stable_under_probes=True,
        recent_revocations=0,
        recent_detected_drift=0,
        open_novelty_observations=0,
    )

    assert classify_evidence_support(item) == "evidence_supported_surrogate"
    summary = summarize([_row(item)])
    assert len(summary.best_available_surrogates) == 1
    assert not summary.serious_mismatch_candidates


def test_repeated_stress_authority_is_contradicted_authority() -> None:
    item = _item(
        strong_observations=5,
        sentinel_count=3,
        fit_history_count=3,
        last_fit_margin=5,
        recent_revocations=2,
        recent_detected_drift=0,
    )

    assert classify_evidence_support(item) == "contradicted_authority"
    summary = summarize([_row(item)])
    assert summary.by_classification["contradicted_authority"] == 1
    assert len(summary.serious_mismatch_candidates) == 1


def test_low_observations_authority_is_insufficient_evidence() -> None:
    item = _item(
        strong_observations=0,
        sentinel_count=2,
        fit_history_count=2,
        last_fit_margin=3,
    )

    assert classify_evidence_support(item) == "insufficient_evidence"
    summary = summarize([_row(item)])
    assert summary.by_classification["insufficient_evidence"] == 1
    assert len(summary.serious_mismatch_candidates) == 1


def test_report_avoids_false_trust_from_hidden_truth_only(tmp_path: Path, capsys) -> None:
    path = tmp_path / "authority.jsonl"
    path.write_text(json.dumps(_row(_item())) + "\n")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out

    assert "External mismatch under authority" in output
    assert "unknown" in output
    assert "falsely trusted" not in output
    assert "false trust" not in output.lower()
    assert "over-certified" not in output
