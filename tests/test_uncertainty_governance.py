from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dreth.uncertainty_governance import (
    PROPOSAL_ACTIONS,
    SIGNAL_NAMES,
    UncertaintyGovernanceProposal,
    UncertaintyGovernanceSummary,
    UncertaintySignal,
    build_governance_summary,
    classify_governance_proposal,
    extract_uncertainty_signals,
)
from summarize_uncertainty_governance import load_jsonl, print_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(**overrides):
    """Minimal visible-field item; no hidden-world fields by default."""
    base = {
        "var": 0,
        "relation_type": "symbolic",
        "status": "certified",
        "skip_role": "tareth",
        "authoritative": True,
    }
    base.update(overrides)
    return base


def _stable_item(**overrides):
    """An evidence-supported-surrogate item: stable, no contradictions."""
    base = {
        "var": 0,
        "relation_type": "symbolic",
        "status": "certified",
        "skip_role": "tareth",
        "authoritative": True,
        "strong_observations": 4,
        "sentinel_count": 3,
        "fit_history_count": 3,
        "last_fit_margin": 5,
        "last_fit_tie_count": 1,
        "last_fit_near_tie_count": 1,
        "alternatives_existed": False,
        "repeatedly_stable_under_probes": True,
        "recent_revocations": 0,
        "recent_detected_drift": 0,
        "consecutive_sentinel_failures": 0,
        "open_novelty": False,
        "open_novelty_observations": 0,
        "dormant_alternatives": 0,
        "passive_stress_recent": None,
        "passive_stress_available": False,
        "frontier_active": False,
        "frontier_stable_count": 0,
        "route_certs": 0,
        "recent_fit_history": [],
    }
    base.update(overrides)
    return base


def _row(item, seed=42):
    return {
        "seed": seed,
        "schedule": "blind_challenge",
        "evaluation": {
            "blind_challenge_behavior": {
                "per_var": [item],
            }
        },
    }


# ---------------------------------------------------------------------------
# Invariant: governance classifier never reads hidden-world fields
# ---------------------------------------------------------------------------


def test_governance_ignores_truth_fields() -> None:
    """Classifier must produce identical proposals with or without truth fields."""
    item = _stable_item()
    item_with_truth = {
        **item,
        "truth_source_edges": [1, 2, 3],
        "truth_func": "HIDDEN_FUNC",
        "truth_delayed_source_edges": [4],
        "truth_latents": [5],
    }
    proposal_a = classify_governance_proposal(item)
    proposal_b = classify_governance_proposal(item_with_truth)
    assert proposal_a.action == proposal_b.action
    assert proposal_a.reason == proposal_b.reason


def test_extract_signals_ignores_truth_fields() -> None:
    item = _stable_item(truth_source_edges=[1, 2], truth_func="HIDDEN", truth_delayed_source_edges=[3])
    item_clean = {k: v for k, v in item.items() if "truth" not in k}
    sig_a = extract_uncertainty_signals(item)
    sig_b = extract_uncertainty_signals(item_clean)
    assert sig_a == sig_b


def test_governance_module_does_not_reference_hidden_fields() -> None:
    """Source code of uncertainty_governance must not read any hidden-world field."""
    from dreth import uncertainty_governance as mod

    source = inspect.getsource(mod)
    banned = [
        '"truth_source_edges"',
        '"truth_func"',
        '"truth_delayed_source_edges"',
        '"truth_latents"',
    ]
    # _external_mismatch_under_authority is permitted to reference them for
    # post-hoc case selection; but classify_governance_proposal and
    # extract_uncertainty_signals must not.
    # We check the specific function sources instead.
    classify_src = inspect.getsource(classify_governance_proposal)
    extract_src = inspect.getsource(extract_uncertainty_signals)
    for field in banned:
        assert field not in classify_src, (
            f"classify_governance_proposal must not read {field}"
        )
        assert field not in extract_src, (
            f"extract_uncertainty_signals must not read {field}"
        )


# ---------------------------------------------------------------------------
# open_novelty → schedule_separating_probe or attempt_consolidation
# ---------------------------------------------------------------------------


def test_open_novelty_with_ties_produces_separating_probe() -> None:
    item = _item(
        open_novelty=True,
        open_novelty_observations=2,
        last_fit_tie_count=5,
        last_fit_near_tie_count=12,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        frontier_active=False,
        last_fit_margin=0,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "schedule_separating_probe"
    assert "open_novelty" in proposal.active_signals


def test_open_novelty_without_ties_produces_attempt_consolidation() -> None:
    item = _item(
        open_novelty=True,
        open_novelty_observations=1,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        frontier_active=False,
        last_fit_margin=3,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "attempt_consolidation"
    assert "open_novelty" in proposal.active_signals


def test_open_novelty_via_observations_field_only() -> None:
    """open_novelty signal fires from open_novelty_observations even if open_novelty key absent."""
    item = _item(
        open_novelty_observations=3,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        last_fit_margin=3,
    )
    signal = extract_uncertainty_signals(item)
    assert signal.open_novelty is True
    proposal = classify_governance_proposal(item)
    assert proposal.action in ("schedule_separating_probe", "attempt_consolidation")


# ---------------------------------------------------------------------------
# low_margin + near_ties → preserve_alternative
# ---------------------------------------------------------------------------


def test_low_margin_near_ties_produces_preserve_alternative() -> None:
    item = _item(
        open_novelty=False,
        open_novelty_observations=0,
        last_fit_margin=0,
        last_fit_near_tie_count=4,
        last_fit_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "preserve_alternative"
    assert "low_margin" in proposal.active_signals


def test_low_margin_alone_does_not_produce_preserve_alternative() -> None:
    """Low margin without near-ties does not warrant preserve_alternative."""
    item = _item(
        open_novelty=False,
        open_novelty_observations=0,
        last_fit_margin=0,
        last_fit_near_tie_count=1,  # not > 1
        last_fit_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action != "preserve_alternative"


def test_dormant_revival_with_alternatives_produces_preserve_alternative() -> None:
    item = _item(
        open_novelty=False,
        open_novelty_observations=0,
        last_fit_margin=5,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=2,
        alternatives_existed=True,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "preserve_alternative"
    assert "dormant_revival" in proposal.active_signals


# ---------------------------------------------------------------------------
# sentinel failures → increase_monitoring / prioritize_repair
# ---------------------------------------------------------------------------


def test_sentinel_failures_with_revocations_produces_prioritize_repair() -> None:
    item = _item(
        consecutive_sentinel_failures=1,
        recent_revocations=2,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "prioritize_repair"
    assert f"sentinel_failures=1" in proposal.active_signals


def test_sentinel_failures_alone_produces_increase_monitoring() -> None:
    item = _item(
        consecutive_sentinel_failures=1,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action in ("increase_monitoring", "prioritize_repair")


def test_repeated_sentinel_failures_produce_monitoring_or_repair() -> None:
    """Any consecutive_sentinel_failures > 0 must yield increase_monitoring or prioritize_repair."""
    for fail_count in (1, 2, 3):
        for rev_count in (0, 1, 2, 3):
            item = _item(
                consecutive_sentinel_failures=fail_count,
                recent_revocations=rev_count,
                open_novelty=False,
                open_novelty_observations=0,
                passive_stress_recent=None,
                dormant_alternatives=0,
                alternatives_existed=False,
                last_fit_margin=3,
                last_fit_near_tie_count=1,
                last_fit_tie_count=1,
                frontier_active=False,
            )
            proposal = classify_governance_proposal(item)
            assert proposal.action in (
                "increase_monitoring", "prioritize_repair"
            ), (
                f"sentinel_failures={fail_count}, revocations={rev_count}: "
                f"got {proposal.action!r}"
            )


# ---------------------------------------------------------------------------
# evidence-supported surrogate → continue_best_available
# ---------------------------------------------------------------------------


def test_evidence_supported_surrogate_remains_continue_best_available() -> None:
    """Stable evidence with no contradictions must yield continue_best_available
    even when hidden truth would show a mismatch (truth fields are ignored)."""
    item = _stable_item(
        # Hidden truth fields that the classifier must not read.
        truth_source_edges=[1, 2],
        truth_func="HIDDEN",
        truth_delayed_source_edges=[3],
        truth_latents=[4],
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "continue_best_available"


def test_stable_item_without_truth_fields_also_continues() -> None:
    item = _stable_item()
    proposal = classify_governance_proposal(item)
    assert proposal.action == "continue_best_available"
    assert proposal.active_signals == [] or all(
        "consequence_tier" in s for s in proposal.active_signals
    )


def test_evidence_supported_no_contradiction_signals() -> None:
    item = _stable_item()
    signal = extract_uncertainty_signals(item)
    assert signal.sentinel_failures == 0
    assert not signal.open_novelty
    assert not signal.passive_stress
    assert signal.recent_revocations == 0
    assert not signal.low_margin
    assert not signal.dormant_revival


# ---------------------------------------------------------------------------
# passive_stress → increase_monitoring
# ---------------------------------------------------------------------------


def test_passive_stress_produces_increase_monitoring() -> None:
    item = _item(
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=2,
        passive_stress_available=True,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=False,
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "increase_monitoring"
    assert "passive_stress" in proposal.active_signals


# ---------------------------------------------------------------------------
# graph_frontier_miss + consequence_tier → commission_higher_handle_shadow
# ---------------------------------------------------------------------------


def test_graph_frontier_miss_high_consequence_produces_commission() -> None:
    item = _item(
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=True,
        frontier_stable_count=0,  # miss
        route_certs=1,            # route consequence
        skip_role="tareth",
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "commission_higher_handle_shadow"
    assert "graph_frontier_miss" in proposal.active_signals


def test_graph_frontier_miss_low_consequence_does_not_commission() -> None:
    item = _item(
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=True,
        frontier_stable_count=0,
        route_certs=0,
        skip_role=None,  # consequence_tier == "none"
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action != "commission_higher_handle_shadow"


# ---------------------------------------------------------------------------
# reduce_skip_strength_shadow
# ---------------------------------------------------------------------------


def test_low_margin_alternatives_tareth_produces_reduce_skip() -> None:
    item = _item(
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=True,
        last_fit_margin=0,
        last_fit_near_tie_count=1,  # near_tie_count not > 1, so preserve_alternative won't fire
        last_fit_tie_count=1,
        frontier_active=False,
        route_certs=0,
        skip_role="tareth",  # consequence_tier = skip_tareth
    )
    proposal = classify_governance_proposal(item)
    assert proposal.action == "reduce_skip_strength_shadow"


# ---------------------------------------------------------------------------
# UncertaintySignal helpers
# ---------------------------------------------------------------------------


def test_signal_active_strings_empty_for_clean_item() -> None:
    item = _stable_item(route_certs=0, skip_role=None)
    signal = extract_uncertainty_signals(item)
    assert signal.active_signal_strings() == []


def test_signal_active_strings_include_all_active() -> None:
    item = _item(
        open_novelty=True,
        open_novelty_observations=3,
        last_fit_margin=0,
        last_fit_near_tie_count=5,
        last_fit_tie_count=4,
        dormant_alternatives=1,
        consecutive_sentinel_failures=2,
        passive_stress_recent=1,
        passive_stress_available=True,
        recent_revocations=3,
        alternatives_existed=True,
        frontier_active=True,
        frontier_stable_count=0,
        route_certs=0,
        skip_role="tareth",
        recent_fit_history=[
            {"best_source_edges": [0], "margin": 0},
            {"best_source_edges": [1], "margin": 0},
        ],
    )
    signal = extract_uncertainty_signals(item)
    parts = signal.active_signal_strings()
    assert "open_novelty" in parts
    assert "low_margin" in parts
    assert any("near_tie_count=" in p for p in parts)
    assert any("tie_count=" in p for p in parts)
    assert "dormant_revival" in parts
    assert any("sentinel_failures=" in p for p in parts)
    assert "passive_stress" in parts
    assert any("recent_revocations=" in p for p in parts)
    assert "alternatives_existed" in parts
    assert "graph_frontier_miss" in parts


# ---------------------------------------------------------------------------
# repeated_fit_churn detection
# ---------------------------------------------------------------------------


def test_repeated_fit_churn_detected_when_source_edges_change() -> None:
    item = _item(
        open_novelty=False,
        open_novelty_observations=0,
        last_fit_margin=0,
        last_fit_near_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        frontier_active=False,
        recent_fit_history=[
            {"best_source_edges": [0], "margin": 0},
            {"best_source_edges": [1], "margin": 0},
        ],
    )
    signal = extract_uncertainty_signals(item)
    assert signal.repeated_fit_churn is True


def test_repeated_fit_churn_not_detected_when_source_edges_stable() -> None:
    item = _item(
        recent_fit_history=[
            {"best_source_edges": [2], "margin": 5},
            {"best_source_edges": [2], "margin": 3},
        ],
    )
    signal = extract_uncertainty_signals(item)
    assert signal.repeated_fit_churn is False


def test_repeated_fit_churn_not_detected_with_single_entry() -> None:
    item = _item(
        recent_fit_history=[{"best_source_edges": [0], "margin": 5}],
    )
    signal = extract_uncertainty_signals(item)
    assert signal.repeated_fit_churn is False


# ---------------------------------------------------------------------------
# build_governance_summary
# ---------------------------------------------------------------------------


def test_build_governance_summary_counts_proposals() -> None:
    rows = [
        _row(_stable_item(var=0), seed=1),
        _row(_item(var=1, open_novelty=True, open_novelty_observations=1,
                   last_fit_tie_count=1, last_fit_near_tie_count=1,
                   consecutive_sentinel_failures=0, recent_revocations=0,
                   passive_stress_recent=None, dormant_alternatives=0,
                   alternatives_existed=False, last_fit_margin=3,
                   frontier_active=False), seed=1),
    ]
    summary = build_governance_summary(rows)
    assert len(summary.proposals) == 2
    assert summary.action_counts["continue_best_available"] == 1
    assert summary.action_counts["attempt_consolidation"] == 1


def test_build_governance_summary_signal_counts() -> None:
    item = _item(
        open_novelty=True,
        open_novelty_observations=1,
        last_fit_tie_count=1,
        last_fit_near_tie_count=1,
        consecutive_sentinel_failures=0,
        recent_revocations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        frontier_active=False,
    )
    summary = build_governance_summary([_row(item)])
    assert summary.signal_counts["open_novelty"] == 1


def test_build_governance_summary_by_relation_type() -> None:
    item = _stable_item(var=0, relation_type="symbolic")
    summary = build_governance_summary([_row(item)])
    assert "symbolic" in summary.by_relation_type
    assert summary.by_relation_type["symbolic"]["continue_best_available"] == 1


def test_build_governance_summary_caution_before_mismatch_uses_truth_post_hoc() -> None:
    """caution_before_mismatch uses truth_source_edges only for case selection."""
    # Item with a visible signal (sentinel failure) AND a hidden mismatch.
    item = _item(
        var=0,
        relation_type="delayed",
        authoritative=True,
        truth_source_edges=[1],
        truth_delayed_source_edges=[],
        learned_source_edges=[2],
        consecutive_sentinel_failures=1,
        recent_revocations=0,
        open_novelty=False,
        open_novelty_observations=0,
        passive_stress_recent=None,
        dormant_alternatives=0,
        alternatives_existed=False,
        last_fit_margin=3,
        last_fit_near_tie_count=1,
        last_fit_tie_count=1,
        frontier_active=False,
    )
    summary = build_governance_summary([_row(item)])
    assert len(summary.caution_before_mismatch) == 1
    case = summary.caution_before_mismatch[0]
    assert case["action"] == "increase_monitoring"
    # Governance classifier must not have used truth_source_edges — verified by
    # test_governance_ignores_truth_fields above; here we just confirm the
    # case made it into the right bucket.


def test_build_governance_summary_supported_surrogate_uses_truth_post_hoc() -> None:
    """supported_surrogate_cases uses truth_source_edges only for case selection."""
    item = _stable_item(
        var=0,
        relation_type="symbolic",
        authoritative=True,
        truth_source_edges=[1],         # mismatch: learned=[] vs truth=[1]
        truth_delayed_source_edges=[],
        learned_source_edges=[],
    )
    summary = build_governance_summary([_row(item)])
    assert len(summary.supported_surrogate_cases) == 1
    assert summary.supported_surrogate_cases[0]["action"] == "continue_best_available"


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def test_report_prints_diagnostic_warning(tmp_path: Path, capsys) -> None:
    path = tmp_path / "gov.jsonl"
    path.write_text(json.dumps(_row(_stable_item())) + "\n")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out

    assert "DIAGNOSTIC ONLY" in output
    assert "no runtime behavior" in output.lower() or "not changed" in output.lower()


def test_report_prints_all_sections(tmp_path: Path, capsys) -> None:
    items = [
        _stable_item(var=0),
        _item(var=1, open_novelty=True, open_novelty_observations=2,
              last_fit_tie_count=5, last_fit_near_tie_count=10,
              consecutive_sentinel_failures=0, recent_revocations=0,
              passive_stress_recent=None, dormant_alternatives=0,
              alternatives_existed=False, last_fit_margin=0,
              frontier_active=False),
    ]
    rows = [{"seed": 1, "schedule": "blind_challenge", "evaluation": {
        "blind_challenge_behavior": {"per_var": items}}}]
    path = tmp_path / "gov.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out

    assert "A. Signal counts" in output
    assert "B. Proposed action counts" in output
    assert "C. Proposal counts by relation_type" in output
    assert "D. Cases where governance would have recommended caution" in output
    assert "E. Supported surrogate cases" in output
    assert "F. Warning" in output


def test_report_proposal_actions_listed(tmp_path: Path, capsys) -> None:
    path = tmp_path / "gov.jsonl"
    path.write_text(json.dumps(_row(_stable_item())) + "\n")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out

    assert "continue_best_available" in output


def test_report_empty_jsonl(tmp_path: Path, capsys) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    print_report(load_jsonl(str(path)))
    output = capsys.readouterr().out
    assert "DIAGNOSTIC ONLY" in output


def test_report_skips_policy_report_rows(tmp_path: Path, capsys) -> None:
    row = {"record_type": "policy_report", "seed": 1}
    path = tmp_path / "pol.jsonl"
    path.write_text(json.dumps(row) + "\n")

    rows = load_jsonl(str(path))
    assert rows == []


# ---------------------------------------------------------------------------
# Dataclass exports
# ---------------------------------------------------------------------------


def test_proposal_actions_tuple_exported() -> None:
    assert "continue_best_available" in PROPOSAL_ACTIONS
    assert "schedule_separating_probe" in PROPOSAL_ACTIONS
    assert "prioritize_repair" in PROPOSAL_ACTIONS
    assert "preserve_alternative" in PROPOSAL_ACTIONS
    assert "attempt_consolidation" in PROPOSAL_ACTIONS
    assert "increase_monitoring" in PROPOSAL_ACTIONS
    assert "reduce_skip_strength_shadow" in PROPOSAL_ACTIONS
    assert "commission_higher_handle_shadow" in PROPOSAL_ACTIONS
    assert len(PROPOSAL_ACTIONS) == 8


def test_signal_names_tuple_exported() -> None:
    assert "open_novelty" in SIGNAL_NAMES
    assert "low_margin" in SIGNAL_NAMES
    assert "near_tie_count" in SIGNAL_NAMES
    assert "tie_count" in SIGNAL_NAMES
    assert "dormant_revival" in SIGNAL_NAMES
    assert "repeated_fit_churn" in SIGNAL_NAMES
    assert "sentinel_failures" in SIGNAL_NAMES
    assert "passive_stress" in SIGNAL_NAMES
    assert "recent_revocations" in SIGNAL_NAMES
    assert "alternatives_existed" in SIGNAL_NAMES
    assert "graph_frontier_miss" in SIGNAL_NAMES
    assert "consequence_tier" in SIGNAL_NAMES
    assert len(SIGNAL_NAMES) == 12


def test_uncertainty_signal_dataclass_fields() -> None:
    sig = UncertaintySignal(
        open_novelty=True,
        low_margin=False,
        near_tie_count=3,
        tie_count=2,
        dormant_revival=False,
        repeated_fit_churn=True,
        sentinel_failures=1,
        passive_stress=False,
        recent_revocations=0,
        alternatives_existed=True,
        graph_frontier_miss=False,
        consequence_tier="skip_tareth",
    )
    assert sig.open_novelty is True
    assert sig.near_tie_count == 3
    assert sig.consequence_tier == "skip_tareth"


def test_uncertainty_governance_proposal_fields() -> None:
    sig = UncertaintySignal(
        open_novelty=False, low_margin=False, near_tie_count=0, tie_count=0,
        dormant_revival=False, repeated_fit_churn=False, sentinel_failures=0,
        passive_stress=False, recent_revocations=0, alternatives_existed=False,
        graph_frontier_miss=False, consequence_tier="none",
    )
    proposal = UncertaintyGovernanceProposal(
        var=7,
        action="continue_best_available",
        reason="no signals",
        active_signals=[],
        signal=sig,
    )
    assert proposal.var == 7
    assert proposal.action == "continue_best_available"
