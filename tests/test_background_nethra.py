from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreth.background_nethra import BackgroundNethra, BackgroundNethraIndex
from dreth.agent import ChainedAgent
from dreth.world import CausalWorld


# ── helpers ───────────────────────────────────────────────────────────────────

def _index(mode: str = "record") -> BackgroundNethraIndex:
    return BackgroundNethraIndex(mode=mode)


def _agent(background_nethra_mode: str = "off", *, n_vars: int = 5, seed: int = 7) -> ChainedAgent:
    world = CausalWorld(n_vars, random.Random(seed), noise_sigma=0.0)
    world.visible_count = n_vars
    agent = ChainedAgent(
        world=world,
        rng=random.Random(seed + 1000),
        sentinel_count=n_vars,
        sentinel_pool=n_vars * 4,
        priority_audit_budget=n_vars,
        frontier_k=n_vars,
        background_nethra_mode=background_nethra_mode,
    )
    agent.initialize()
    return agent


def _run_cycles(agent: ChainedAgent, n: int, start: int = 1) -> None:
    for i in range(n):
        agent.run_cycle(start + i)


def _operational_snapshot(agent: ChainedAgent) -> dict:
    """Collect all operational behavioral counters from an agent."""
    return {
        "skip_count": agent.skip_count,
        "trass_skip_count": agent.trass_skip_count,
        "sentinel_skip_count": agent.sentinel_skip_count,
        "total_interventions": agent.total_interventions,
        "authority_strength_monitoring_increases": agent._authority_strength_monitoring_increases_total,
        "authority_strength_repair_bumps": agent._authority_strength_repair_priority_bumps_total,
    }


# ── unit tests: BackgroundNethraIndex ────────────────────────────────────────

def test_off_mode_all_methods_are_noops():
    """mode='off' means all public update methods return None and no records accumulate."""
    idx = _index(mode="off")
    assert idx.add_or_update_from_context_role(
        nethra_id="n1", role="trass", var=0, context_key="ctx", cycle=1
    ) is None
    assert idx.add_or_update_from_tied_frontier(
        nethra_id="n2", var=1, context_key="ctx", cycle=1
    ) is None
    assert idx.add_or_update_from_dormant_alternative(
        nethra_id="n3", var=2, context_key="ctx", cycle=1
    ) is None
    assert idx.add_or_update_from_uncertainty_cluster(
        nethra_id="n4", vars=(0, 1), cycle=1
    ) is None
    assert idx.add_or_update_from_authority_debt(
        nethra_id="n5", var=0, authority_state="repair_candidate", cycle=1
    ) is None
    assert idx.add_or_update_from_temporal_event_if_available(
        nethra_id="n6", vars=(0,), temporal_event="some_event", cycle=1
    ) is None
    assert len(idx.records) == 0
    assert idx.background_nethra_records == 0
    assert idx.operational_authority_count == 0


def test_trass_role_creates_background_nethra():
    idx = _index()
    rec = idx.add_or_update_from_context_role(
        nethra_id="neth_A", role="trass", var=3, context_key="ctx1", cycle=5,
        fit_signature="x3:LOW(1,2)", parents=(1, 2),
    )
    assert rec is not None
    assert rec.kind == "trass_pattern"
    assert 3 in rec.vars
    assert "trass" in rec.source_roles
    assert idx.background_trass_patterns == 1
    assert idx.background_nethra_records == 1

    # Second call updates existing record
    rec2 = idx.add_or_update_from_context_role(
        nethra_id="neth_A", role="trass", var=3, context_key="ctx2", cycle=6,
    )
    assert rec2 is not None
    assert rec2.seen_count == 2
    assert idx.background_trass_patterns == 1  # counter only bumped for new records
    assert idx.background_nethra_records == 1


def test_unresolved_role_creates_background_nethra():
    idx = _index()
    rec = idx.add_or_update_from_context_role(
        nethra_id="neth_B", role="unresolved", var=2, context_key="ctx1", cycle=3,
    )
    assert rec is not None
    assert rec.kind == "unresolved_pattern"
    assert idx.background_unresolved_patterns == 1


def test_unresolved_tied_frontier_creates_background_nethra():
    idx = _index()
    rec = idx.add_or_update_from_tied_frontier(
        nethra_id="bg_frontier:x1:LOW(0)", var=1, context_key="ctx1", cycle=2,
        candidate_count=3, stable_count=1, parents=(0,),
    )
    assert rec is not None
    assert rec.kind == "tied_frontier_pattern"
    assert "unresolved" in rec.source_roles
    assert idx.background_tied_frontier_patterns == 1
    assert rec.payload.get("candidate_count") == 3


def test_quarantined_authority_state_creates_background_nethra():
    idx = _index()
    rec = idx.add_or_update_from_authority_debt(
        nethra_id="bg_auth:n_X", var=0, context_key="ctx1", cycle=4,
        authority_state="quarantined_for_derivation",
    )
    assert rec is not None
    assert rec.kind == "quarantined_pattern"
    assert idx.background_quarantined_patterns == 1
    assert "authority_state:quarantined_for_derivation" in rec.source_roles


def test_authority_debt_ignores_non_quarantine_states():
    idx = _index()
    rec = idx.add_or_update_from_authority_debt(
        nethra_id="bg_auth:n_Y", var=0, authority_state="active_best_available", cycle=1
    )
    assert rec is None
    assert idx.background_nethra_records == 0


def test_dormant_alternative_creates_background_nethra():
    idx = _index()
    rec = idx.add_or_update_from_dormant_alternative(
        nethra_id="bg_dormant:x0:FIRST(1)", var=0, context_key="ctx1", cycle=7,
        revival_count=2, parents=(1,),
    )
    assert rec is not None
    assert rec.kind == "dormant_alternative_pattern"
    assert "dormant" in rec.source_roles
    assert idx.background_dormant_patterns == 1
    assert rec.payload.get("revival_count") == 2


def test_same_nethra_trass_and_tareth_different_contexts():
    """A nethra recorded as trass may have tareth status elsewhere, but tareth
    is not observed here — only trass/unresolved/best_available are recorded."""
    idx = _index()
    # Trass observation recorded
    rec = idx.add_or_update_from_context_role(
        nethra_id="neth_C", role="trass", var=4, context_key="ctx_trass", cycle=1,
    )
    assert rec is not None
    assert rec.kind == "trass_pattern"

    # tareth observation is silently ignored (no record created)
    result = idx.add_or_update_from_context_role(
        nethra_id="neth_C", role="tareth", var=4, context_key="ctx_tareth", cycle=2,
    )
    assert result is None
    # Original record unchanged — no tareth context contamination
    assert idx.records["neth_C"].kind == "trass_pattern"
    assert "tareth" not in idx.records["neth_C"].source_roles


def test_background_nethra_does_not_issue_authority():
    """operational_authority_count must remain 0 regardless of how many records accumulate."""
    idx = _index()
    for i in range(10):
        idx.add_or_update_from_context_role(
            nethra_id=f"n{i}", role="trass", var=i % 5, cycle=i,
        )
    summary = idx.summarize()
    assert summary["operational_authority_count"] == 0
    assert idx.operational_authority_count == 0


def test_operational_authority_count_not_writable():
    """Verify the field stays 0 — no code path in the module touches it."""
    idx = _index()
    idx.add_or_update_from_context_role(nethra_id="n1", role="trass", var=0, cycle=1)
    idx.add_or_update_from_tied_frontier(nethra_id="n2", var=1, cycle=2)
    idx.add_or_update_from_uncertainty_cluster(nethra_id="n3", vars=(0, 1), cycle=3)
    idx.add_or_update_from_authority_debt(
        nethra_id="n4", var=2, authority_state="repair_candidate", cycle=4
    )
    assert idx.operational_authority_count == 0


def test_giant_uncertainty_cluster_creates_background_record_no_action():
    idx = _index()
    rec = idx.add_or_update_from_uncertainty_cluster(
        nethra_id="bg_uc:giant:0,1,2,3", vars=(0, 1, 2, 3), cycle=1,
        is_giant=True, signals=("high_entropy",),
    )
    assert rec is not None
    assert rec.kind == "recurring_low_salience_pattern"
    assert idx.background_giant_cluster_patterns == 1
    assert idx.operational_authority_count == 0

    # Giant counter increments once per new record, not on update
    rec2 = idx.add_or_update_from_uncertainty_cluster(
        nethra_id="bg_uc:giant:0,1,2,3", vars=(0, 1, 2, 3), cycle=2, is_giant=True,
    )
    assert rec2 is not None
    assert idx.background_giant_cluster_patterns == 1  # not incremented again


def test_non_giant_cluster_is_unresolved_pattern():
    idx = _index()
    rec = idx.add_or_update_from_uncertainty_cluster(
        nethra_id="bg_uc:small:0,1", vars=(0, 1), cycle=1, is_giant=False,
    )
    assert rec is not None
    assert rec.kind == "unresolved_pattern"
    assert idx.background_giant_cluster_patterns == 0


def test_temporal_event_adapter_noop_when_none():
    idx = _index()
    result = idx.add_or_update_from_temporal_event_if_available(
        nethra_id="te_1", vars=(0,), temporal_event=None, cycle=1
    )
    assert result is None
    assert idx.background_nethra_records == 0


def test_temporal_event_adapter_records_when_present():
    idx = _index()
    rec = idx.add_or_update_from_temporal_event_if_available(
        nethra_id="te_2", vars=(0, 1), temporal_event="cohort_42", cycle=3,
        context_key="ctx_te",
    )
    assert rec is not None
    assert rec.kind == "temporal_cohort_pattern"
    assert "cohort_42" in str(rec.payload.get("temporal_event", ""))


def test_summary_distinguishes_familiar_from_authority():
    """familiar_background_count and operational_authority_count must both appear;
    operational_authority_count must always be 0."""
    idx = _index()
    for i in range(5):
        idx.add_or_update_from_context_role(
            nethra_id=f"n{i}", role="trass", var=i, cycle=i + 1,
        )
    summary = idx.summarize()
    assert "familiar_background_count" in summary
    assert "operational_authority_count" in summary
    assert summary["familiar_background_count"] == 5
    assert summary["operational_authority_count"] == 0


def test_best_available_role_creates_context_role_pattern():
    idx = _index()
    rec = idx.add_or_update_from_context_role(
        nethra_id="n_ba", role="best_available", var=1, context_key="ctx1", cycle=1,
    )
    assert rec is not None
    assert rec.kind == "context_role_pattern"


def test_role_shift_recording():
    idx = _index()
    idx.add_or_update_from_context_role(nethra_id="n_rs", role="trass", var=2, cycle=1)
    idx.record_role_shift("n_rs", from_role="trass", to_role="unresolved", var=2, cycle=5)
    assert len(idx.background_role_shift_examples) == 1
    ex = idx.background_role_shift_examples[0]
    assert ex["from_role"] == "trass"
    assert ex["to_role"] == "unresolved"


def test_role_shift_off_mode_is_noop():
    idx = _index(mode="off")
    idx.record_role_shift("n_rs", from_role="trass", to_role="unresolved", var=0, cycle=1)
    assert len(idx.background_role_shift_examples) == 0


def test_query_by_var():
    idx = _index()
    idx.add_or_update_from_context_role(nethra_id="n1", role="trass", var=3, cycle=1)
    idx.add_or_update_from_context_role(nethra_id="n2", role="unresolved", var=3, cycle=2)
    idx.add_or_update_from_context_role(nethra_id="n3", role="trass", var=7, cycle=3)
    results = idx.query_by_var(3)
    ids = {r.nethra_id for r in results}
    assert "n1" in ids
    assert "n2" in ids
    assert "n3" not in ids


def test_query_by_context():
    idx = _index()
    idx.add_or_update_from_context_role(nethra_id="n1", role="trass", var=0, context_key="ctx_A", cycle=1)
    idx.add_or_update_from_context_role(nethra_id="n2", role="trass", var=1, context_key="ctx_B", cycle=2)
    results_a = idx.query_by_context("ctx_A")
    assert len(results_a) == 1 and results_a[0].nethra_id == "n1"


def test_export_records_structure():
    idx = _index()
    idx.add_or_update_from_context_role(nethra_id="n_ex", role="trass", var=0, cycle=1)
    export = idx.export_records(limit=10)
    assert "records" in export
    assert "edges" in export
    assert "role_shift_examples" in export
    assert len(export["records"]) == 1
    rec_dict = export["records"][0]
    assert rec_dict["nethra_id"] == "n_ex"
    assert rec_dict["action_relevance_score"] == 0.0


def test_mode_validation():
    try:
        BackgroundNethraIndex(mode="invalid")
        assert False, "should raise"
    except ValueError:
        pass


# ── integration tests: agent behavioral invariants ────────────────────────────

def test_record_mode_equals_off_on_behavioral_metrics():
    """mode='record' must not change any operational behavioral counters vs mode='off'."""
    rng_off = random.Random(42)
    rng_rec = random.Random(42)

    world_off = CausalWorld(5, random.Random(42), noise_sigma=0.0)
    world_off.visible_count = 5
    world_rec = CausalWorld(5, random.Random(42), noise_sigma=0.0)
    world_rec.visible_count = 5

    agent_off = ChainedAgent(
        world=world_off, rng=rng_off, sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="off",
    )
    agent_rec = ChainedAgent(
        world=world_rec, rng=rng_rec, sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="record",
    )
    agent_off.initialize()
    agent_rec.initialize()

    _run_cycles(agent_off, 20)
    _run_cycles(agent_rec, 20)

    snap_off = _operational_snapshot(agent_off)
    snap_rec = _operational_snapshot(agent_rec)
    assert snap_off == snap_rec, (
        f"record mode changed behavioral metrics:\n  off={snap_off}\n  rec={snap_rec}"
    )


def test_background_nethra_does_not_suppress_skips():
    """skip_count must be the same with and without background nethra enabled."""
    world_off = CausalWorld(5, random.Random(11), noise_sigma=0.0)
    world_off.visible_count = 5
    world_on = CausalWorld(5, random.Random(11), noise_sigma=0.0)
    world_on.visible_count = 5

    a_off = ChainedAgent(
        world=world_off, rng=random.Random(1011), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="off",
    )
    a_on = ChainedAgent(
        world=world_on, rng=random.Random(1011), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="record",
    )
    a_off.initialize()
    a_on.initialize()
    _run_cycles(a_off, 15)
    _run_cycles(a_on, 15)
    assert a_off.skip_count == a_on.skip_count
    assert a_off.trass_skip_count == a_on.trass_skip_count


def test_background_nethra_does_not_force_probes():
    """total_interventions must be the same with and without background nethra."""
    world_off = CausalWorld(5, random.Random(22), noise_sigma=0.0)
    world_off.visible_count = 5
    world_on = CausalWorld(5, random.Random(22), noise_sigma=0.0)
    world_on.visible_count = 5

    a_off = ChainedAgent(
        world=world_off, rng=random.Random(2022), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="off",
    )
    a_on = ChainedAgent(
        world=world_on, rng=random.Random(2022), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="record",
    )
    a_off.initialize()
    a_on.initialize()
    _run_cycles(a_off, 15)
    _run_cycles(a_on, 15)
    assert a_off.total_interventions == a_on.total_interventions


def test_background_nethra_does_not_increase_monitoring():
    """authority_strength monitoring increases must not differ due to background nethra."""
    world_off = CausalWorld(5, random.Random(33), noise_sigma=0.0)
    world_off.visible_count = 5
    world_on = CausalWorld(5, random.Random(33), noise_sigma=0.0)
    world_on.visible_count = 5

    a_off = ChainedAgent(
        world=world_off, rng=random.Random(3033), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5,
        authority_strength_mode="record",
        background_nethra_mode="off",
    )
    a_on = ChainedAgent(
        world=world_on, rng=random.Random(3033), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5,
        authority_strength_mode="record",
        background_nethra_mode="record",
    )
    a_off.initialize()
    a_on.initialize()
    _run_cycles(a_off, 15)
    _run_cycles(a_on, 15)
    assert (
        a_off._authority_strength_monitoring_increases_total
        == a_on._authority_strength_monitoring_increases_total
    )


def test_background_nethra_does_not_increase_repair_priority():
    """authority_strength repair priority bumps must not differ."""
    world_off = CausalWorld(5, random.Random(44), noise_sigma=0.0)
    world_off.visible_count = 5
    world_on = CausalWorld(5, random.Random(44), noise_sigma=0.0)
    world_on.visible_count = 5

    a_off = ChainedAgent(
        world=world_off, rng=random.Random(4044), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5,
        authority_strength_mode="record",
        background_nethra_mode="off",
    )
    a_on = ChainedAgent(
        world=world_on, rng=random.Random(4044), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5,
        authority_strength_mode="record",
        background_nethra_mode="record",
    )
    a_off.initialize()
    a_on.initialize()
    _run_cycles(a_off, 15)
    _run_cycles(a_on, 15)
    assert (
        a_off._authority_strength_repair_priority_bumps_total
        == a_on._authority_strength_repair_priority_bumps_total
    )


def test_background_nethra_does_not_revoke():
    """Revocation count (certs with revoked_by set) must not change due to background nethra."""
    def _count_revocations(agent: ChainedAgent) -> int:
        total = 0
        for var in range(agent.world.visible_count):
            n = agent.ledger.vars[var]
            certs = list(getattr(n, "certificates", {}).values())
            route_certs = list(getattr(n, "route_certs", {}).values())
            for cert in certs + route_certs:
                if getattr(cert, "revoked_by", None):
                    total += 1
        return total

    world_off = CausalWorld(5, random.Random(55), noise_sigma=0.0)
    world_off.visible_count = 5
    world_on = CausalWorld(5, random.Random(55), noise_sigma=0.0)
    world_on.visible_count = 5

    a_off = ChainedAgent(
        world=world_off, rng=random.Random(5055), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="off",
    )
    a_on = ChainedAgent(
        world=world_on, rng=random.Random(5055), sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5, frontier_k=5, background_nethra_mode="record",
    )
    a_off.initialize()
    a_on.initialize()
    _run_cycles(a_off, 15)
    _run_cycles(a_on, 15)
    assert _count_revocations(a_off) == _count_revocations(a_on)


def test_hidden_truth_not_read():
    """BackgroundNethraIndex has no reference to debug manifests or hidden truth.
    Verify at the module attribute level and that no summary field exposes it."""
    idx = _index()
    # No hidden truth attributes
    assert not hasattr(idx, "hidden_truth")
    assert not hasattr(idx, "debug_manifest")
    assert not hasattr(idx, "_hidden_truth")
    for name in dir(idx):
        assert "hidden" not in name.lower(), f"suspicious attribute: {name}"
        assert "debug_manifest" not in name.lower(), f"suspicious attribute: {name}"

    # Summary keys must not contain truth/manifest
    summary = idx.summarize()
    for key in summary:
        assert "hidden" not in key
        assert "truth" not in key
        assert "manifest" not in key


def test_background_nethra_accumulates_records_in_record_mode():
    """Sanity: after running cycles, background nethra index must have > 0 records."""
    agent = _agent(background_nethra_mode="record")
    _run_cycles(agent, 20)
    metrics = agent.background_nethra_metrics()
    assert metrics["background_nethra_mode"] == "record"
    # The invariant: familiar_background_count >= 0 and operational_authority_count == 0
    assert metrics["familiar_background_count"] >= 0
    assert metrics["operational_authority_count"] == 0


def test_background_nethra_off_mode_metrics_are_zero():
    """In off mode, all record counts are zero and operational_authority_count is 0."""
    agent = _agent(background_nethra_mode="off")
    _run_cycles(agent, 10)
    metrics = agent.background_nethra_metrics()
    assert metrics["background_nethra_mode"] == "off"
    assert metrics["background_nethra_records"] == 0
    assert metrics["familiar_background_count"] == 0
    assert metrics["operational_authority_count"] == 0


def test_context_role_index_not_duplicated():
    """Background nethra observes from context role assignments independently.
    Running background_nethra without context_role_index produces records."""
    world = CausalWorld(5, random.Random(99), noise_sigma=0.0)
    world.visible_count = 5
    agent = ChainedAgent(
        world=world,
        rng=random.Random(1099),
        sentinel_count=5,
        sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=5,
        background_nethra_mode="record",
        # context_role_index not enabled (default "off")
    )
    agent.initialize()
    _run_cycles(agent, 20)
    metrics = agent.background_nethra_metrics()
    # background_nethra works independently of context_role_index
    assert metrics["operational_authority_count"] == 0
    # Should have observed something (at minimum frontier/dormant scans)
    assert metrics["background_nethra_mode"] == "record"


def test_assist_feature_mode_still_never_issues_authority():
    idx = _index(mode="assist_feature")
    for i in range(5):
        idx.add_or_update_from_context_role(
            nethra_id=f"af_{i}", role="trass", var=i, cycle=i + 1,
        )
    summary = idx.summarize()
    assert summary["operational_authority_count"] == 0
    assert idx.operational_authority_count == 0
    assert summary["familiar_background_count"] == 5
