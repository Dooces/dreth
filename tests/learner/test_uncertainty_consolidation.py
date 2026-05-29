from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dreth.learner.uncertainty_consolidation import (
    UncertaintyCase,
    cluster_uncertainty_cases,
    extract_uncertainty_cases_from_rows,
    summarize_clusters,
)
from compare_uncertainty_consolidation_modes import (
    load_jsonl as load_compare_jsonl,
    print_report as print_compare_report,
)
from summarize_uncertainty_consolidation import load_jsonl, print_report


def _case(var: int, **kwargs) -> UncertaintyCase:
    base = {
        "cycle": 10,
        "action": "preserve_ambiguity",
        "active_signals": ("open_novelty", "near_tie_count"),
        "learned_source_edges": (1, 2),
        "near_tie_candidates": ((1, 2),),
        "tied_frontier_info": {"active": True, "stable_count": 1},
        "novelty_state": "open",
        "recent_fit_history": (
            {"best_source_edges": [1], "best_func": "FIRST"},
            {"best_source_edges": [2], "best_func": "FIRST"},
        ),
        "sentinels": (),
        "consequence_tier": "skip_tareth",
        "graph_neighbors": (1, 2, 3),
    }
    base.update(kwargs)
    return UncertaintyCase(var=var, **base)


def test_clustering_ignores_hidden_fields_in_rows() -> None:
    row = {
        "cycles": 100,
        "evaluation": {
            "blind_challenge_behavior": {
                "per_var": [
                    {
                        "var": 0,
                        "learned_source_edges": [1, 2],
                        "last_fit_margin": 0,
                        "last_fit_near_tie_count": 4,
                        "open_novelty": True,
                        "recent_fit_history": [{"best_source_edges": [1, 2], "near_tie_count": 4}],
                    }
                ]
            }
        },
    }
    row_with_hidden = json.loads(json.dumps(row))
    item = row_with_hidden["evaluation"]["blind_challenge_behavior"]["per_var"][0]
    item.update({
        "truth_source_edges": [99],
        "truth_func": "HIDDEN",
        "truth_delayed_source_edges": [98],
        "truth_latents": [97],
    })

    clusters_a = cluster_uncertainty_cases(extract_uncertainty_cases_from_rows([row]))
    clusters_b = cluster_uncertainty_cases(extract_uncertainty_cases_from_rows([row_with_hidden]))

    assert clusters_a == clusters_b


def test_cases_sharing_source_edges_and_signals_cluster_together() -> None:
    clusters = cluster_uncertainty_cases([
        _case(0),
        _case(1, learned_source_edges=(1, 2), graph_neighbors=(1, 4)),
    ])

    assert len(clusters) == 1
    assert clusters[0].vars == (0, 1)


def test_unrelated_cases_remain_separate() -> None:
    clusters = cluster_uncertainty_cases([
        _case(
            0,
            learned_source_edges=(1,),
            graph_neighbors=(1,),
            active_signals=("low_margin",),
            near_tie_candidates=(),
            tied_frontier_info={"active": False},
            recent_fit_history=(),
            consequence_tier="none",
        ),
        _case(
            9,
            learned_source_edges=(7,),
            graph_neighbors=(8,),
            active_signals=("sentinel_failures",),
            near_tie_candidates=(),
            tied_frontier_info={"active": False},
            novelty_state="closed",
            recent_fit_history=(),
            consequence_tier="none",
        ),
    ])

    assert len(clusters) == 2


def test_broad_open_novelty_can_consolidate_into_fewer_clusters() -> None:
    cases = [
        _case(0, learned_source_edges=(2,), graph_neighbors=(2, 5)),
        _case(1, learned_source_edges=(2,), graph_neighbors=(2, 6)),
        _case(2, learned_source_edges=(2,), graph_neighbors=(2, 7)),
    ]
    clusters = cluster_uncertainty_cases(cases)

    assert len(clusters) < len(cases)
    assert clusters[0].proposed_handle_kind in {
        "possible_missing_operator",
        "dense_fanin_candidate",
        "shared_ambiguity",
    }


def test_summary_reports_compression_ratio() -> None:
    clusters = cluster_uncertainty_cases([_case(0), _case(1)])
    summary = summarize_clusters(clusters)

    assert summary["uncertainty_compression_ratio"] == pytest.approx(2.0)
    assert "cluster_specificity_mean" in summary


def test_posthoc_relation_type_interpretation_is_separate(tmp_path: Path, capsys) -> None:
    path = tmp_path / "uc.jsonl"
    row = {
        "ok": True,
        "uncertainty_consolidation_mode": "shadow",
        "evaluation": {
            "blind_challenge_behavior": {
                "cycles_observed": 100,
                "per_var": [
                    {
                        "var": 0,
                        "relation_type": "delayed",
                        "learned_source_edges": [1],
                        "open_novelty": True,
                        "last_fit_near_tie_count": 3,
                        "last_fit_margin": 0,
                    },
                    {
                        "var": 1,
                        "relation_type": "proxy_confounded",
                        "learned_source_edges": [1],
                        "open_novelty": True,
                        "last_fit_near_tie_count": 2,
                        "last_fit_margin": 0,
                    },
                ],
            }
        },
    }
    path.write_text(json.dumps(row) + "\n")

    print_report(load_jsonl(str(path)), sys.stdout)
    out = capsys.readouterr().out

    assert "F. Post-hoc interpretation" in out
    assert "Uses relation_type only after clustering" in out
    assert "Warning: hidden truth is not used" in out


def test_compare_script_detects_assist_worse_than_off(tmp_path: Path, capsys) -> None:
    off = tmp_path / "off.jsonl"
    shadow = tmp_path / "shadow.jsonl"
    assist = tmp_path / "assist.jsonl"
    base_row = {
        "schedule": "blind_challenge",
        "n_vars": 50,
        "cycles": 3000,
        "seed": 42,
        "settle_cycles": 25,
        "noise_sigma": 0.02,
        "interventions": 100,
        "full_audits": 10,
        "revoked_by_dist": {"sentinel": 1},
        "total_unique_failures": 2,
        "quality_cost": 1000,
        "temporal_frontier_chosen_source_edge_recall": 0.5,
        "temporal_frontier_recall_lift": 1.2,
        "dormant_total": 3,
        "vars_open_novelty": 1,
    }
    assist_row = dict(base_row)
    assist_row.update({
        "uncertainty_consolidation_mode": "assist",
        "uncertainty_assist_policy": "all",
        "interventions": 120,
        "full_audits": 12,
        "quality_cost": 1500,
        "assist_extra_budget_total": 7,
    })
    off.write_text(json.dumps(base_row) + "\n")
    shadow.write_text(json.dumps({**base_row, "uncertainty_consolidation_mode": "shadow"}) + "\n")
    assist.write_text(json.dumps(assist_row) + "\n")

    print_compare_report(
        load_compare_jsonl(str(off)),
        load_compare_jsonl(str(shadow)),
        load_compare_jsonl(str(assist)),
        sys.stdout,
    )
    out = capsys.readouterr().out

    assert "A. off vs shadow equality" in out
    assert "equal: True" in out
    assert "WARNING: quality_cost worsened" in out
    assert "assist benefit is not attributable" in out
