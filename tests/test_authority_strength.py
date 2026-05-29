from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dreth.authority_strength as authority_mod
from dreth.agent import ChainedAgent
from dreth.authority_strength import compute_authority_strength_records
from dreth.ledger import TiedFrontier
from dreth.records import FitDiagnostic
from dreth.world import CausalWorld


def _agent(
    mode: str = "off",
    *,
    repair_agenda: bool = False,
    controller: str = "state",
    derivation_policy: str | None = None,
) -> ChainedAgent:
    world = CausalWorld(5, random.Random(3), noise_sigma=0.0)
    world.visible_count = 5
    agent = ChainedAgent(
        world=world,
        rng=random.Random(13003),
        sentinel_count=5,
        sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world.n_vars,
        repair_agenda_enabled=repair_agenda,
        authority_strength_mode=mode,
        authority_strength_controller=controller,
        authority_derivation_policy=derivation_policy,
    )
    agent.initialize()
    return agent


def _fit_diag(
    var: int,
    *,
    margin: int = 5,
    near_ties: int = 0,
    cycle: int = 1,
    best_source_edges: tuple[int, ...] = (),
    best_func: str = "LOW",
) -> FitDiagnostic:
    near = tuple(((i,), "FIRST", 10 - i) for i in range(near_ties))
    return FitDiagnostic(
        cycle=cycle,
        var=var,
        status_before="proposed",
        role_before="untested",
        available_source_edges=(),
        restricted=True,
        hypothesis_count=2,
        best_score=10,
        second_score=10 - margin,
        margin=margin,
        best_source_edges=best_source_edges,
        best_func=best_func,
        failure_class="fit_clean",
        probes=((0, 0.05),),
        actuals=(0.0,),
        pick_preds=(0.0,),
        tie_set=frozenset({((), "LOW")}),
        near_tie_candidates=near,
    )


def _operational_snapshot(agent: ChainedAgent) -> tuple[int, int, int, int, int, tuple[tuple[str, str], ...]]:
    visible = range(agent.world.visible_count)
    return (
        agent.skip_count,
        agent.trass_skip_count,
        agent.sentinel_skip_count,
        agent.full_audit_count,
        agent.total_interventions,
        tuple((agent.ledger.vars[v].status, agent.ledger.vars[v].role_for("skip")) for v in visible),
    )


def _run_small(mode: str) -> ChainedAgent:
    rng_w = random.Random(7)
    rng_a = random.Random(1007)
    world = CausalWorld(6, rng_w, noise_sigma=0.0)
    world.prepare_schedule("shaped", 0)
    agent = ChainedAgent(
        world=world,
        rng=rng_a,
        sentinel_count=3,
        sentinel_pool=12,
        priority_audit_budget=3,
        frontier_k=world.n_vars,
        authority_strength_mode=mode,
    )
    agent.initialize()
    for cycle in range(1, 15):
        world.perturb_by_schedule(cycle, "shaped", settle_cycles=0)
        agent.run_cycle(cycle)
    return agent


def test_hidden_truth_is_not_read_by_runtime_authority_strength() -> None:
    source = inspect.getsource(authority_mod)
    source += inspect.getsource(ChainedAgent._run_authority_strength)
    banned = [
        "debug_blind_challenge_manifest",
        "truth_source_edges",
        "truth_func",
        "truth_delayed_source_edges",
        "truth_latents",
    ]
    for field in banned:
        assert field not in source


def test_off_mode_preserves_behavior_and_records_nothing() -> None:
    agent = _run_small("off")

    assert agent.authority_strength_metrics()["authority_strength_records"] == 0
    assert agent.authority_strength_export()["records"] == []


def test_record_mode_behavior_equals_off() -> None:
    off = _run_small("off")
    record = _run_small("record")

    assert _operational_snapshot(record) == _operational_snapshot(off)
    assert record.authority_strength_metrics()["authority_strength_records"] > 0


def test_stable_evidence_yields_strong_or_usable() -> None:
    agent = _agent("record")
    n = agent.ledger.vars[0]
    n.strong_observations = 4
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 0
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=6))

    record = next(r for r in compute_authority_strength_records(agent, 1) if r.var == 0)

    assert record.strength in {"strong", "usable"}


def test_low_observations_yields_weak() -> None:
    agent = _agent("record")
    n = agent.ledger.vars[0]
    n.strong_observations = 1
    n.sentinels = []
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5))

    record = next(r for r in compute_authority_strength_records(agent, 1) if r.var == 0)

    assert record.strength == "weak"
    assert record.best_available


def test_novelty_churn_revocation_yields_contested() -> None:
    agent = _agent("record")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5, near_ties=3))

    record = next(r for r in compute_authority_strength_records(agent, 1) if r.var == 0)

    assert record.strength == "contested"
    assert record.best_available
    assert record.authority_state == "repair_candidate"


def test_best_available_can_be_weak_or_contested_without_revocation() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    before_role = n.role_for("skip")
    n.strong_observations = 1
    n.sentinels = []
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=0))

    agent._run_authority_strength(1)
    record = next(r for r in agent._authority_strength_latest_records if r.var == 0)

    assert record.best_available
    assert record.strength in {"weak", "contested"}
    assert n.role_for("skip") == before_role


def test_contested_once_creates_debt_but_no_monitoring_spam() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 1
    n.sentinels = []
    before = _operational_snapshot(agent)
    role_before = n.role_for("skip")
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=1))

    agent._run_authority_strength(1)

    metrics = agent.authority_strength_metrics()
    assert agent._authority_strength_budget_bonus.get(0, 0) == 0
    assert metrics["authority_debt_created"] > 0
    assert metrics["monitoring_increases_from_strength_candidates"] == 0
    assert metrics["generic_contested_noop"] > 0
    assert metrics["local_use_preserved"] > 0
    assert n.role_for("skip") == role_before
    assert _operational_snapshot(agent) == before


def test_giant_uncertainty_cluster_alone_creates_debt_not_action() -> None:
    agent = _agent("assist")
    agent._uncertainty_latest_clusters = [
        SimpleNamespace(
            shared_signals=(),
            is_giant_cluster=True,
            cluster_size=4,
            vars=(0, 1, 2, 3),
        )
    ]
    agent.fit_diagnostics.clear()

    agent._run_authority_strength(1)
    metrics = agent.authority_strength_metrics()

    assert metrics["authority_debt_created"] > 0
    assert metrics["authority_action_candidates"] == 0
    assert metrics["authority_actions_applied"] == 0
    assert metrics["generic_contested_noop"] > 0


def test_non_giant_local_uncertainty_cluster_can_trigger_bounded_action() -> None:
    agent = _agent("assist")
    agent._uncertainty_latest_clusters = [
        SimpleNamespace(
            shared_signals=(),
            is_giant_cluster=False,
            cluster_size=2,
            vars=(0, 1),
        )
    ]
    agent.fit_diagnostics.clear()

    agent._run_authority_strength(1)
    metrics = agent.authority_strength_metrics()

    assert metrics["action_reason_specificity"]["non_giant_local_uncertainty_cluster"] > 0
    assert metrics["authority_action_candidates"] > 0
    assert metrics["monitoring_hints_applied"] > 0


def test_regime_sentinel_failure_attribution_for_authority_action() -> None:
    agent = _agent("assist")
    agent._last_regime_failed_vars = {0}
    agent.fit_diagnostics.clear()

    agent._run_authority_strength(1)
    metrics = agent.authority_strength_metrics()

    assert metrics["action_reason_specificity"]["regime_sentinel_failure"] > 0
    assert metrics["authority_action_regime_sentinel_failure_attribution"] > 0
    assert metrics["authority_action_activated_failing_regime_sentinel"] > 0


def test_persistent_contested_requires_evidence_and_preserves_alternatives() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.tied_frontier = TiedFrontier(
        candidates=frozenset({((), "LOW"), ((1,), "FIRST")}),
        scores={((), "LOW"): 10, ((1,), "FIRST"): 9},
        margin=4,
        context_key=1,
        collapse_sig=None,
        separating_probes=(),
        first_seen_cycle=1,
        last_seen_cycle=1,
    )
    agent.fit_diagnostics.clear()
    n.full_audits = 1
    agent.fit_diagnostics.append(_fit_diag(0, margin=1, near_ties=2, cycle=1))
    agent._run_authority_strength(1)
    n.full_audits = 2
    agent.fit_diagnostics.append(
        _fit_diag(0, margin=1, near_ties=2, cycle=2, best_source_edges=(1,), best_func="FIRST")
    )
    agent._run_authority_strength(2)

    before_fit = (n.source_edges, n.func)
    agent._collapse_tied_frontier(0, ((), "LOW"), 2)

    assert (n.source_edges, n.func) == before_fit
    assert n.dormant_alternatives
    metrics = agent.authority_strength_metrics()
    assert metrics["authority_debt_created"] > 0
    assert metrics["alternatives_preserved_from_strength"] > 0


def test_local_use_allowed_for_contested_best_available() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 1
    n.sentinels = []
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=1))

    agent._run_authority_strength(1)
    debt = next(
        debt for debt in agent._authority_state_controller.debts.values()
        if debt.var == 0
    )

    assert debt.current_state == "contested_best_available"
    assert debt.local_use_allowed
    assert debt.derivation_allowed


def test_open_novelty_and_churn_quarantines_derivation() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    agent.fit_diagnostics.clear()
    n.full_audits = 1
    agent.fit_diagnostics.append(_fit_diag(0, margin=1, near_ties=2, cycle=1))
    agent.ledger.propose_novelty(1, 0, "vocabulary", "test", ["test"])
    agent._run_authority_strength(1)
    n.full_audits = 2
    agent.fit_diagnostics.append(
        _fit_diag(
            0,
            margin=1,
            near_ties=2,
            cycle=2,
            best_source_edges=(1,),
            best_func="FIRST",
        )
    )

    agent._run_authority_strength(2)
    debt = next(
        debt for debt in agent._authority_state_controller.debts.values()
        if debt.var == 0
    )

    assert debt.current_state == "quarantined_for_derivation"
    assert debt.local_use_allowed
    assert debt.derivation_would_block
    assert debt.derivation_allowed
    assert agent._authority_strength_derivation_allowed(
        0,
        cycle=2,
        blocked_handle_kind="route",
        blocked_target="x0",
    )
    metrics = agent.authority_strength_metrics()
    assert metrics["authority_derivation_policy"] == "shadow"
    assert metrics["derivation_gate_shadow_would_block"] > 0
    assert metrics["derivation_gate_blocked"] == 0


def test_quarantine_persistent_blocks_only_after_persistence_threshold() -> None:
    agent = _agent("assist", derivation_policy="quarantine_persistent")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    agent.fit_diagnostics.clear()
    n.full_audits = 1
    agent.fit_diagnostics.append(_fit_diag(0, margin=1, near_ties=2, cycle=1))
    agent.ledger.propose_novelty(1, 0, "vocabulary", "test", ["test"])
    agent._run_authority_strength(1)

    assert agent._authority_strength_derivation_allowed(
        0,
        cycle=1,
        blocked_handle_kind="route",
        blocked_target="x0",
    )

    n.full_audits = 2
    agent.fit_diagnostics.append(
        _fit_diag(0, margin=1, near_ties=2, cycle=2, best_source_edges=(1,), best_func="FIRST")
    )
    agent._run_authority_strength(2)

    assert not agent._authority_strength_derivation_allowed(
        0,
        cycle=2,
        blocked_handle_kind="route",
        blocked_target="x0",
    )
    metrics = agent.authority_strength_metrics()
    assert metrics["derivation_gate_blocked"] > 0
    assert metrics["derivation_gate_blocked_by_handle_kind"]["route"] > 0


def test_off_derivation_policy_never_blocks_derivation() -> None:
    agent = _agent("assist", derivation_policy="off")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.full_audits = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=1, near_ties=2, cycle=1))
    agent.ledger.propose_novelty(1, 0, "vocabulary", "test", ["test"])
    agent._run_authority_strength(1)
    n.full_audits = 2
    agent.fit_diagnostics.append(
        _fit_diag(0, margin=1, near_ties=2, cycle=2, best_source_edges=(1,), best_func="FIRST")
    )
    agent._run_authority_strength(2)

    assert agent._authority_strength_derivation_allowed(
        0,
        cycle=2,
        blocked_handle_kind="route",
        blocked_target="x0",
    )
    metrics = agent.authority_strength_metrics()
    assert metrics["authority_derivation_policy"] == "off"
    assert metrics["derivation_gate_blocked"] == 0


def test_quarantine_repair_only_blocks_repair_candidate() -> None:
    agent = _agent("assist", derivation_policy="quarantine_repair_only")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5, near_ties=2))
    agent._run_authority_strength(1)

    assert not agent._authority_strength_derivation_allowed(
        0,
        cycle=1,
        blocked_handle_kind="route",
        blocked_target="x0",
    )
    debt = next(debt for debt in agent._authority_state_controller.debts.values() if debt.var == 0)
    assert debt.current_state == "repair_candidate"
    assert debt.local_use_allowed


def test_sentinel_failure_promotes_repair_candidate_with_bounded_hint() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5, near_ties=2))

    agent._run_authority_strength(1)
    metrics = agent.authority_strength_metrics()

    assert agent._authority_strength_budget_bonus.get(0, 0) == 1
    assert 0 in agent._authority_strength_repair_priority_vars
    assert metrics["repair_candidates"] > 0
    assert metrics["bounded_repairs_applied"] > 0


def test_cooldown_prevents_repeated_per_cycle_hints() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5, near_ties=2, cycle=1))

    agent._run_authority_strength(1)
    first = agent.authority_strength_metrics()["bounded_repairs_applied"]
    agent._run_authority_strength(2)
    second_metrics = agent.authority_strength_metrics()

    assert second_metrics["bounded_repairs_applied"] == first
    assert second_metrics["repair_priority_bumps_from_strength_suppressed_by_cooldown"] > 0


def test_debt_can_be_paid_down_by_stable_evidence() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.strong_observations = 1
    n.sentinels = []
    n.full_audits = 1
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=1, cycle=1))
    agent._run_authority_strength(1)

    n.strong_observations = 4
    n.sentinels = [(1, 0.05)]
    n.full_audits = 2
    agent.fit_diagnostics.append(_fit_diag(0, margin=6, cycle=2))
    agent._run_authority_strength(2)
    metrics = agent.authority_strength_metrics()

    assert metrics["authority_debt_paid"] > 0


def test_legacy_controller_reproduces_pressure_for_comparison() -> None:
    agent = _agent("assist", controller="legacy")
    n = agent.ledger.vars[0]
    n.strong_observations = 1
    n.sentinels = []
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=1))

    agent._run_authority_strength(1)

    assert agent._authority_strength_budget_bonus.get(0, 0) >= 1


def test_state_controller_does_not_issue_revoke_suppress_skip_or_replace_fit() -> None:
    agent = _agent("assist")
    n = agent.ledger.vars[0]
    n.source_edges = (1,)
    n.func = "FIRST"
    n.strong_observations = 3
    n.sentinels = [(1, 0.05)]
    n.consecutive_sentinel_failures = 1
    before_fit = (n.source_edges, n.func)
    before_role = n.role_for("skip")
    before_certs = {
        v: (set(agent.ledger.vars[v].certificates), set(agent.ledger.vars[v].route_certs))
        for v in range(agent.world.visible_count)
    }
    agent.fit_diagnostics.clear()
    agent.fit_diagnostics.append(_fit_diag(0, margin=5, near_ties=2))

    agent._run_authority_strength(1)

    after_certs = {
        v: (set(agent.ledger.vars[v].certificates), set(agent.ledger.vars[v].route_certs))
        for v in range(agent.world.visible_count)
    }
    assert (n.source_edges, n.func) == before_fit
    assert n.role_for("skip") == before_role
    assert after_certs == before_certs


def test_strength_is_context_specific_not_global() -> None:
    agent = _agent("record")
    n = agent.ledger.vars[0]
    n.source_edges = ()
    first = next(r for r in compute_authority_strength_records(agent, 1) if r.var == 0)

    n.source_edges = (1,)
    second = next(r for r in compute_authority_strength_records(agent, 2) if r.var == 0)

    assert first.nethra_id != second.nethra_id
    assert first.context_key != second.context_key
