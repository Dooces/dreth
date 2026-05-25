from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dreth.agent import ChainedAgent
from dreth.ledger import TiedFrontier
from dreth.uncertainty_consolidation import (
    UncertaintyCase,
    cluster_uncertainty_cases,
    extract_uncertainty_cases_from_rows,
    propose_consolidation_assists,
    summarize_clusters,
)
from dreth.world import CausalWorld
from summarize_uncertainty_consolidation import load_jsonl, print_report


def _case(var: int, **kwargs) -> UncertaintyCase:
    base = {
        "cycle": 10,
        "action": "preserve_ambiguity",
        "active_signals": ("open_novelty", "near_tie_count"),
        "learned_parents": (1, 2),
        "near_tie_candidates": ((1, 2),),
        "tied_frontier_info": {"active": True, "stable_count": 1},
        "novelty_state": "open",
        "recent_fit_history": (
            {"best_parents": [1], "best_func": "FIRST"},
            {"best_parents": [2], "best_func": "FIRST"},
        ),
        "sentinels": (),
        "consequence_tier": "skip_tareth",
        "graph_neighbors": (1, 2, 3),
    }
    base.update(kwargs)
    return UncertaintyCase(var=var, **base)


def _agent(*, mode: str = "off", repair_agenda: bool = False) -> ChainedAgent:
    world = CausalWorld(5, random.Random(3), noise_sigma=0.0)
    world.visible_count = 5
    return ChainedAgent(
        world=world,
        rng=random.Random(13003),
        sentinel_count=5,
        sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world.n_vars,
        repair_agenda_enabled=repair_agenda,
        uncertainty_consolidation_mode=mode,
    )


def test_clustering_ignores_hidden_fields_in_rows() -> None:
    row = {
        "cycles": 100,
        "evaluation": {
            "blind_challenge_behavior": {
                "per_var": [
                    {
                        "var": 0,
                        "learned_parents": [1, 2],
                        "last_fit_margin": 0,
                        "last_fit_near_tie_count": 4,
                        "open_novelty": True,
                        "recent_fit_history": [{"best_parents": [1, 2], "near_tie_count": 4}],
                    }
                ]
            }
        },
    }
    row_with_hidden = json.loads(json.dumps(row))
    item = row_with_hidden["evaluation"]["blind_challenge_behavior"]["per_var"][0]
    item.update({
        "truth_parents": [99],
        "truth_func": "HIDDEN",
        "truth_delayed_parents": [98],
        "truth_latents": [97],
    })

    clusters_a = cluster_uncertainty_cases(extract_uncertainty_cases_from_rows([row]))
    clusters_b = cluster_uncertainty_cases(extract_uncertainty_cases_from_rows([row_with_hidden]))

    assert clusters_a == clusters_b


def test_runtime_consolidation_does_not_reference_hidden_manifest_or_truth_fields() -> None:
    source = inspect.getsource(ChainedAgent._run_uncertainty_consolidation)
    banned = [
        "debug_blind_challenge_manifest",
        "blind_challenge_manifest",
        "truth_parents",
        "truth_func",
        "truth_delayed_parents",
        "truth_latents",
    ]
    for field in banned:
        assert field not in source


def test_cases_sharing_parents_and_signals_cluster_together() -> None:
    clusters = cluster_uncertainty_cases([
        _case(0),
        _case(1, learned_parents=(1, 2), graph_neighbors=(1, 4)),
    ])

    assert len(clusters) == 1
    assert clusters[0].vars == (0, 1)


def test_unrelated_cases_remain_separate() -> None:
    clusters = cluster_uncertainty_cases([
        _case(
            0,
            learned_parents=(1,),
            graph_neighbors=(1,),
            active_signals=("low_margin",),
            near_tie_candidates=(),
            tied_frontier_info={"active": False},
            recent_fit_history=(),
            consequence_tier="none",
        ),
        _case(
            9,
            learned_parents=(7,),
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
        _case(0, learned_parents=(2,), graph_neighbors=(2, 5)),
        _case(1, learned_parents=(2,), graph_neighbors=(2, 6)),
        _case(2, learned_parents=(2,), graph_neighbors=(2, 7)),
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


def test_off_mode_matches_default_behavior() -> None:
    default = _agent()
    explicit_off = _agent(mode="off")
    default.initialize()
    explicit_off.initialize()
    for cycle in range(1, 8):
        default.run_cycle(cycle)
        explicit_off.run_cycle(cycle)

    assert default.records == explicit_off.records
    assert default.skip_count == explicit_off.skip_count
    assert default.full_audit_count == explicit_off.full_audit_count
    assert default.total_interventions == explicit_off.total_interventions


def test_shadow_mode_records_clusters_but_behavior_matches_off() -> None:
    off = _agent(mode="off")
    shadow = _agent(mode="shadow")
    off.initialize()
    shadow.initialize()
    for var in (0, 1):
        for agent in (off, shadow):
            n = agent.ledger.vars[var]
            n.parents = (2,)
            n.consecutive_sentinel_failures = 1
    for cycle in range(1, 4):
        off.run_cycle(cycle)
        shadow.run_cycle(cycle)

    assert off.records == shadow.records
    assert shadow.uncertainty_consolidation_metrics()["uncertainty_clusters"] >= 1
    assert shadow.uncertainty_consolidation_metrics()["consolidation_assists_total"] == 0


def test_assist_priority_hint_does_not_issue_or_revoke_certs() -> None:
    agent = _agent(mode="assist", repair_agenda=True)
    agent.initialize()
    before = sum(len(n.certificates) for n in agent.ledger.vars.values())
    for var in (0, 1):
        n = agent.ledger.vars[var]
        n.parents = (2,)
        n.consecutive_sentinel_failures = 1

    agent._run_uncertainty_consolidation(1)
    after = sum(len(n.certificates) for n in agent.ledger.vars.values())

    assert after == before
    assert any(
        "repair_priority_bonus" in assists
        for assists in agent._uncertainty_assist_vars.values()
    )


def test_assist_preserves_alternatives_only_within_cap() -> None:
    agent = _agent(mode="assist")
    agent.initialize()
    agent._uncertainty_max_preserve_count = 2
    agent._uncertainty_preserve_remaining = 2
    agent._uncertainty_preserve_vars = {0}
    n = agent.ledger.vars[0]
    n.tied_frontier = TiedFrontier(
        candidates=frozenset({
            ((1,), "FIRST"),
            ((2,), "FIRST"),
            ((3,), "FIRST"),
            ((4,), "FIRST"),
        }),
        scores={((1,), "FIRST"): 10, ((2,), "FIRST"): 9, ((3,), "FIRST"): 9, ((4,), "FIRST"): 9},
        margin=4,
        context_key=1,
        collapse_sig=None,
        separating_probes=(),
        first_seen_cycle=1,
        last_seen_cycle=1,
        stable_count=1,
        distinct_contexts_seen=1,
    )

    agent._collapse_tied_frontier(0, ((1,), "FIRST"), 2)

    assert len(n.dormant_alternatives) == 2


def test_assist_requests_probes_only_through_forced_probe_surface() -> None:
    agent = _agent(mode="assist")
    agent.initialize()
    for var in (0, 1):
        n = agent.ledger.vars[var]
        n.parents = (2,)
        n.consecutive_sentinel_failures = 1
        n.tied_frontier = TiedFrontier(
            candidates=frozenset({((2,), "FIRST"), ((3,), "FIRST")}),
            scores={((2,), "FIRST"): 10, ((3,), "FIRST"): 10},
            margin=4,
            context_key=1,
            collapse_sig=None,
            separating_probes=((2, 0.05),),
            first_seen_cycle=1,
            last_seen_cycle=1,
        )

    agent._run_uncertainty_consolidation(1)

    assert agent._uncertainty_forced_probes
    assert all(probes == ((2, 0.05),) for probes in agent._uncertainty_forced_probes.values())


def test_assist_does_not_hard_suppress_existing_trass_skips() -> None:
    agent = _agent(mode="assist")
    agent.initialize()
    trass_vars = [
        v for v in range(agent.world.visible_count)
        if agent.ledger.vars[v].role_for("skip") == "trass"
    ]
    assert trass_vars

    agent.run_cycle(1)

    assert any(v in agent.records[-1].skipped_vars for v in trass_vars)
    assert all(agent.ledger.vars[v].role_for("skip") == "trass" for v in trass_vars)


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
                        "learned_parents": [1],
                        "open_novelty": True,
                        "last_fit_near_tie_count": 3,
                        "last_fit_margin": 0,
                    },
                    {
                        "var": 1,
                        "relation_type": "proxy_confounded",
                        "learned_parents": [1],
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
