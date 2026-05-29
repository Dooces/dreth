from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_blind_challenge import load_jsonl, print_report


def test_blind_challenge_summary_uses_evidence_relative_terms(tmp_path: Path, capsys) -> None:
    path = tmp_path / "blind.jsonl"
    row = {
        "schedule": "blind_challenge",
        "ok": True,
        "interventions": 10,
        "full_audits": 2,
        "trass_skips": 1,
        "sentinel_skips": 2,
        "compression_skips": 0,
        "violations": [],
        "evaluation": {
            "blind_challenge_manifest": {
                "latents": [],
                "relations": [{"var": 0, "relation_type": "delayed"}],
                "intervention_side_effects": [],
            },
            "blind_challenge_behavior": {
                "per_var": [
                    {
                        "var": 0,
                        "relation_type": "delayed",
                        "truth_source_edges": [1],
                        "truth_delayed_source_edges": [],
                        "learned_source_edges": [2],
                        "learned_source_edge_overlap": [],
                        "status": "certified",
                        "skip_role": "tareth",
                        "authoritative": True,
                        "strong_observations": 2,
                        "sentinel_count": 2,
                        "fit_history_count": 2,
                        "last_fit_margin": 2,
                    }
                ]
            },
        },
    }
    path.write_text(json.dumps(row) + "\n")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out

    assert "external-truth mismatches under authority" in output
    assert "authority/evidence mismatch candidates" in output
    assert "structures Dreth over-certified" not in output
    assert "where Dreth falsely trusted" not in output
    assert "false trust" not in output.lower()
