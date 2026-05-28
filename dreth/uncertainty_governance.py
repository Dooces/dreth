from __future__ import annotations

"""Shadow-only uncertainty governance agenda for blind_challenge offline analysis.

Records which variables show observable uncertainty signals and proposes shadow
governance actions. Proposals explain why they were made using only agent-visible
evidence. No hidden truth is used by the classifier.

INVARIANTS (enforced by design, not runtime):
  - Diagnostic only. No effect on agent behavior, cert issuance, skip logic,
    fit, sentinels, route certs, authority records, or defaults.
  - Classification uses agent-visible evidence fields only. truth_source_edges,
    truth_func, truth_delayed_source_edges, truth_latents, and all other hidden-world
    fields are never read by this module.
  - Never imported by agent.py, ChainedAgent, or any runtime path.
  - Proposals are NOT correct actions. They are shadow governance proposals only.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


SIGNAL_NAMES = (
    "open_novelty",
    "low_margin",
    "near_tie_count",
    "tie_count",
    "dormant_revival",
    "repeated_fit_churn",
    "sentinel_failures",
    "passive_stress",
    "recent_revocations",
    "alternatives_existed",
    "graph_frontier_miss",
    "consequence_tier",
)

PROPOSAL_ACTIONS = (
    "continue_best_available",
    "increase_monitoring",
    "schedule_separating_probe",
    "prioritize_repair",
    "preserve_alternative",
    "attempt_consolidation",
    "reduce_skip_strength_shadow",
    "commission_higher_handle_shadow",
)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _compute_repeated_fit_churn(item: dict[str, Any]) -> bool:
    history = item.get("recent_fit_history") or []
    if len(history) < 2:
        return False
    # Churn: best_source_edges changed between the last two fit entries.
    last_source_edges = tuple(sorted(history[-1].get("best_source_edges") or []))
    prev_source_edges = tuple(sorted(history[-2].get("best_source_edges") or []))
    return last_source_edges != prev_source_edges


def _consequence_tier(item: dict[str, Any]) -> str:
    if _as_int(item.get("route_certs")) > 0:
        return "route"
    skip_role = item.get("skip_role") or ""
    if skip_role == "tareth":
        return "skip_tareth"
    if skip_role == "trass":
        return "skip_trass"
    return "none"


@dataclass
class UncertaintySignal:
    """Observable uncertainty signals derived from agent-visible evidence only."""

    open_novelty: bool
    low_margin: bool
    near_tie_count: int
    tie_count: int
    dormant_revival: bool
    repeated_fit_churn: bool
    sentinel_failures: int
    passive_stress: bool
    recent_revocations: int
    alternatives_existed: bool
    graph_frontier_miss: bool
    consequence_tier: str  # "route" | "skip_tareth" | "skip_trass" | "none"

    def active_signal_strings(self) -> list[str]:
        """Return human-readable strings for all active signals."""
        parts: list[str] = []
        if self.open_novelty:
            parts.append("open_novelty")
        if self.low_margin:
            parts.append("low_margin")
        if self.near_tie_count > 1:
            parts.append(f"near_tie_count={self.near_tie_count}")
        if self.tie_count > 1:
            parts.append(f"tie_count={self.tie_count}")
        if self.dormant_revival:
            parts.append("dormant_revival")
        if self.repeated_fit_churn:
            parts.append("repeated_fit_churn")
        if self.sentinel_failures > 0:
            parts.append(f"sentinel_failures={self.sentinel_failures}")
        if self.passive_stress:
            parts.append("passive_stress")
        if self.recent_revocations > 0:
            parts.append(f"recent_revocations={self.recent_revocations}")
        if self.alternatives_existed:
            parts.append("alternatives_existed")
        if self.graph_frontier_miss:
            parts.append("graph_frontier_miss")
        if self.consequence_tier != "none":
            parts.append(f"consequence_tier={self.consequence_tier}")
        return parts

    def has_any_caution_signal(self) -> bool:
        return bool(self.active_signal_strings()) and not (
            not self.open_novelty
            and not self.low_margin
            and self.near_tie_count <= 1
            and self.tie_count <= 1
            and not self.dormant_revival
            and not self.repeated_fit_churn
            and self.sentinel_failures == 0
            and not self.passive_stress
            and self.recent_revocations == 0
            and not self.alternatives_existed
            and not self.graph_frontier_miss
        )


@dataclass
class UncertaintyGovernanceProposal:
    """Shadow governance proposal for one variable.

    Proposals are diagnostic only. They are NOT correct actions and must not
    change agent behavior, authority, skips, or cert issuance.
    """

    var: int
    action: str
    reason: str
    active_signals: list[str]
    signal: UncertaintySignal


@dataclass
class UncertaintyGovernanceSummary:
    """Aggregate governance proposals across all vars in a JSONL report."""

    proposals: list[UncertaintyGovernanceProposal] = field(default_factory=list)
    signal_counts: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    by_relation_type: dict[str, Counter[str]] = field(default_factory=dict)
    # Cases where governance proposed caution and there was an external mismatch.
    # truth_source_edges / truth_func are used here ONLY for post-hoc case selection;
    # the governance classifier itself never reads them.
    caution_before_mismatch: list[dict[str, Any]] = field(default_factory=list)
    # Cases where governance proposed continue_best_available despite external mismatch.
    supported_surrogate_cases: list[dict[str, Any]] = field(default_factory=list)


def extract_uncertainty_signals(item: dict[str, Any]) -> UncertaintySignal:
    """Extract observable uncertainty signals from a per_var evidence item.

    Reads only agent-visible fields. Never reads truth_source_edges, truth_func,
    truth_delayed_source_edges, truth_latents, or any other hidden-world field.
    """
    open_novelty = (
        bool(item.get("open_novelty"))
        or _as_int(item.get("open_novelty_observations")) > 0
    )

    last_fit_margin = item.get("last_fit_margin")
    low_margin = (
        last_fit_margin is not None and _as_int(last_fit_margin) <= 0
    )

    near_tie_count = _as_int(item.get("last_fit_near_tie_count"))
    tie_count = _as_int(item.get("last_fit_tie_count"))

    dormant_revival = _as_int(item.get("dormant_alternatives")) > 0

    repeated_fit_churn = _compute_repeated_fit_churn(item)

    sentinel_failures = _as_int(item.get("consecutive_sentinel_failures"))

    passive_stress = (
        item.get("passive_stress_recent") is not None
        and _as_int(item.get("passive_stress_recent")) > 0
    )

    recent_revocations = _as_int(item.get("recent_revocations"))

    alternatives_existed = bool(item.get("alternatives_existed"))

    frontier_active = bool(item.get("frontier_active"))
    frontier_stable_count = _as_int(item.get("frontier_stable_count"))
    graph_frontier_miss = frontier_active and frontier_stable_count == 0

    consequence_tier = _consequence_tier(item)

    return UncertaintySignal(
        open_novelty=open_novelty,
        low_margin=low_margin,
        near_tie_count=near_tie_count,
        tie_count=tie_count,
        dormant_revival=dormant_revival,
        repeated_fit_churn=repeated_fit_churn,
        sentinel_failures=sentinel_failures,
        passive_stress=passive_stress,
        recent_revocations=recent_revocations,
        alternatives_existed=alternatives_existed,
        graph_frontier_miss=graph_frontier_miss,
        consequence_tier=consequence_tier,
    )


def classify_governance_proposal(item: dict[str, Any]) -> UncertaintyGovernanceProposal:
    """Classify one per_var item into a shadow governance proposal.

    Priority order ensures the most urgent signal drives the proposal.
    Never reads truth_source_edges, truth_func, or any hidden-world field.
    """
    var = _as_int(item.get("var", 0))
    signal = extract_uncertainty_signals(item)
    active = signal.active_signal_strings()

    # Repair priority: sentinel failures combined with cert revocations.
    if signal.sentinel_failures > 0 and signal.recent_revocations >= 2:
        return UncertaintyGovernanceProposal(
            var=var,
            action="prioritize_repair",
            reason=(
                "consecutive sentinel failures combined with cert revocations "
                "indicate structural instability requiring repair"
            ),
            active_signals=active,
            signal=signal,
        )

    # Open novelty + significant ambiguity: schedule a separating probe.
    if signal.open_novelty and (signal.tie_count > 1 or signal.near_tie_count > 1):
        return UncertaintyGovernanceProposal(
            var=var,
            action="schedule_separating_probe",
            reason=(
                "open novelty with near-tied hypotheses cannot be resolved "
                "without a targeted separating probe"
            ),
            active_signals=active,
            signal=signal,
        )

    # Open novelty without dominant competing alternatives: attempt consolidation.
    if signal.open_novelty:
        return UncertaintyGovernanceProposal(
            var=var,
            action="attempt_consolidation",
            reason=(
                "open novelty with no dominant competing hypothesis; "
                "consolidation may settle ambiguity"
            ),
            active_signals=active,
            signal=signal,
        )

    # Low margin + significant near-ties: preserve alternatives.
    if signal.low_margin and signal.near_tie_count > 1:
        return UncertaintyGovernanceProposal(
            var=var,
            action="preserve_alternative",
            reason=(
                "low fit margin with near-tied alternatives; "
                "current choice has not pulled decisively ahead"
            ),
            active_signals=active,
            signal=signal,
        )

    # Dormant alternatives with historically active alternatives: preserve them.
    if signal.dormant_revival and signal.alternatives_existed:
        return UncertaintyGovernanceProposal(
            var=var,
            action="preserve_alternative",
            reason=(
                "dormant alternatives exist and were historically active; "
                "a context change may revive a better hypothesis"
            ),
            active_signals=active,
            signal=signal,
        )

    # Sentinel failures (without cert revocations): increase monitoring.
    if signal.sentinel_failures > 0:
        return UncertaintyGovernanceProposal(
            var=var,
            action="increase_monitoring",
            reason=(
                "consecutive sentinel failures indicate fit instability "
                "requiring closer monitoring"
            ),
            active_signals=active,
            signal=signal,
        )

    # Passive stress, repeated revocations, or fit churn: increase monitoring.
    if signal.passive_stress or signal.recent_revocations >= 2 or signal.repeated_fit_churn:
        return UncertaintyGovernanceProposal(
            var=var,
            action="increase_monitoring",
            reason=(
                "passive stress, repeated cert revocations, or fit churn "
                "observed in visible evidence"
            ),
            active_signals=active,
            signal=signal,
        )

    # Shadow skip reduction: low margin + alternatives in a tareth skip role.
    if (
        signal.low_margin
        and signal.alternatives_existed
        and signal.consequence_tier == "skip_tareth"
    ):
        return UncertaintyGovernanceProposal(
            var=var,
            action="reduce_skip_strength_shadow",
            reason=(
                "low fit margin with visible alternatives in a tareth skip role; "
                "shadow-only skip reduction proposed"
            ),
            active_signals=active,
            signal=signal,
        )

    # Graph frontier miss in a high-consequence role: shadow commission.
    if signal.graph_frontier_miss and signal.consequence_tier in ("route", "skip_tareth"):
        return UncertaintyGovernanceProposal(
            var=var,
            action="commission_higher_handle_shadow",
            reason=(
                "frontier search found no stable winner in a high-consequence role; "
                "shadow commission of a higher handle proposed"
            ),
            active_signals=active,
            signal=signal,
        )

    # Default: no observable signals warrant intervention.
    return UncertaintyGovernanceProposal(
        var=var,
        action="continue_best_available",
        reason="no observable uncertainty signals warrant governance intervention",
        active_signals=active,
        signal=signal,
    )


def _external_mismatch_under_authority(item: dict[str, Any]) -> bool:
    """Post-hoc selection: was this var authoritative and externally mismatched?

    truth_source_edges / truth_func are used ONLY here for case selection. The
    governance classifier (classify_governance_proposal) never calls this.
    """
    if not bool(item.get("authoritative")):
        return False
    learned = {_as_int(v) for v in item.get("learned_source_edges") or []}
    truth = {_as_int(v) for v in item.get("truth_source_edges") or []}
    truth.update(_as_int(v) for v in item.get("truth_delayed_source_edges") or [])
    if truth:
        return learned != truth
    if learned:
        return True
    truth_func = item.get("truth_func")
    learned_func = item.get("learned_func")
    return bool(truth_func and learned_func and truth_func != learned_func)


def build_governance_summary(
    rows: Iterable[dict[str, Any]],
) -> UncertaintyGovernanceSummary:
    """Build an UncertaintyGovernanceSummary from blind_challenge JSONL rows.

    Iterates over all per_var items across all rows. Governance proposals are
    classified using only agent-visible fields. truth_source_edges / truth_func are
    used post-hoc only to select caution_before_mismatch and
    supported_surrogate_cases for the report.
    """
    summary = UncertaintyGovernanceSummary()

    for row in rows:
        evaluation = row.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            continue
        behavior = evaluation.get("blind_challenge_behavior") or {}
        if not isinstance(behavior, dict):
            continue
        seed = row.get("seed")

        for item in behavior.get("per_var") or []:
            if not isinstance(item, dict):
                continue

            proposal = classify_governance_proposal(item)
            relation_type = str(item.get("relation_type") or "unknown")

            summary.proposals.append(proposal)
            summary.action_counts[proposal.action] += 1

            if relation_type not in summary.by_relation_type:
                summary.by_relation_type[relation_type] = Counter()
            summary.by_relation_type[relation_type][proposal.action] += 1

            for sig_str in proposal.active_signals:
                # Count each active signal by its base name (strip =value suffix).
                sig_name = sig_str.split("=")[0]
                summary.signal_counts[sig_name] += 1

            # Post-hoc: use truth fields only to select report cases.
            if _external_mismatch_under_authority(item):
                case = {
                    "seed": seed,
                    "var": item.get("var"),
                    "relation_type": relation_type,
                    "action": proposal.action,
                    "reason": proposal.reason,
                    "active_signals": proposal.active_signals,
                }
                if proposal.action != "continue_best_available":
                    summary.caution_before_mismatch.append(case)
                else:
                    summary.supported_surrogate_cases.append(case)

    return summary
