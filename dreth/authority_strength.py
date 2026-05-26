from __future__ import annotations

"""Visible-evidence authority-strength metadata.

Authority strength is a runtime classification over evidence the agent already
has in hand. It is not a certificate, does not revoke certificates, and does
not replace the current best fit.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AuthorityStrength = Literal[
    "strong",
    "usable",
    "weak",
    "contested",
    "insufficient",
]

AuthorityState = Literal[
    "strong",
    "usable",
    "contested_best_available",
    "quarantined_for_derivation",
    "repair_candidate",
    "insufficient",
]

AuthorityDerivationPolicy = Literal[
    "off",
    "quarantine_persistent",
    "quarantine_repair_only",
    "shadow",
]


@dataclass(frozen=True)
class AuthorityStrengthRecord:
    var: int
    nethra_id: str
    context_key: str
    cycle: int
    strength: AuthorityStrength
    reason: str
    active_evidence: tuple[str, ...] = ()
    contradictory_evidence: tuple[str, ...] = ()
    required_future_evidence: tuple[str, ...] = ()
    uncertainty_signals: tuple[str, ...] = ()
    prior_role: str = ""
    best_available: bool = False
    authority_state: AuthorityState = "usable"
    evidence_epoch: int = 0


@dataclass
class AuthorityDebt:
    var: int
    context_key: str
    current_state: AuthorityState
    active_reasons: tuple[str, ...] = ()
    first_seen_cycle: int = 0
    persistence_count: int = 0
    last_seen_cycle: int = 0
    last_action_cycle: int = -1_000_000
    evidence_required: int = 0
    evidence_paid: int = 0
    debt_score: int = 0
    derivation_allowed: bool = True
    derivation_would_block: bool = False
    local_use_allowed: bool = True
    repair_attention_allowed: bool = False
    last_evidence_epoch: int = -1


@dataclass(frozen=True)
class AuthorityStateTransition:
    var: int
    context_key: str
    cycle: int
    previous_state: AuthorityState
    current_state: AuthorityState
    next_state: AuthorityState
    reason: str = ""
    reasons: tuple[str, ...] = ()
    action: str = "state_update"


@dataclass(frozen=True)
class DerivationGateAttribution:
    var: int
    context_key: str
    authority_state: AuthorityState
    debt_score: int
    active_reasons: tuple[str, ...]
    blocked_handle_kind: str
    blocked_target: str
    cycle: int
    local_use_allowed: bool
    derivation_allowed: bool
    derivation_would_block: bool
    authority_derivation_policy: AuthorityDerivationPolicy


@dataclass(frozen=True)
class AuthorityControllerResult:
    monitoring_bonus_vars: dict[int, int] = field(default_factory=dict)
    preserve_vars: set[int] = field(default_factory=set)
    repair_priority_vars: set[int] = field(default_factory=set)
    future_requirement_vars: set[int] = field(default_factory=set)
    derivation_quarantined_vars: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class AuthorityStrengthSummary:
    counts_by_strength: dict[str, int] = field(default_factory=dict)
    counts_by_reason: dict[str, int] = field(default_factory=dict)
    counts_by_authority_state: dict[str, int] = field(default_factory=dict)
    weak_best_available: int = 0
    contested_best_available: int = 0
    monitoring_increases: int = 0
    alternatives_preserved: int = 0
    future_evidence_requirements: int = 0


def _evidence_keys(items: tuple[str, ...]) -> set[str]:
    return {str(item).split("=", 1)[0] for item in items}


def proposed_authority_state(
    *,
    strength: AuthorityStrength,
    contradictory_evidence: tuple[str, ...],
    best_available: bool,
) -> AuthorityState:
    keys = _evidence_keys(contradictory_evidence)
    if "sentinel_failures" in keys or "recent_revocations" in keys:
        return "repair_candidate"
    if strength == "strong":
        return "strong"
    if strength == "usable":
        return "usable"
    if strength == "insufficient":
        return "insufficient"
    if best_available and strength in {"weak", "contested"}:
        return "contested_best_available"
    return "insufficient"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _latest_fit_by_var(agent: Any) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = defaultdict(list)
    for fd in getattr(agent, "fit_diagnostics", ()) or ():
        out[int(getattr(fd, "var", -1))].append(fd)
    return dict(out)


def _recent_churn(fits: list[Any]) -> bool:
    if len(fits) < 2:
        return False
    recent = fits[-4:]
    signatures = {
        (
            tuple(int(p) for p in getattr(fd, "best_parents", ()) or ()),
            str(getattr(fd, "best_func", "")),
        )
        for fd in recent
    }
    return len(signatures) > 1


def _recent_role_for_nethra(agent: Any, nethra_id: str) -> str:
    index = getattr(agent, "_context_role_index", None)
    if index is None:
        return ""
    roles = getattr(index, "_roles_by_nethra", {}).get(nethra_id, ())
    if not roles:
        return ""
    return str(roles[-1].role)


def _cluster_signals_by_var(agent: Any) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for cluster in getattr(agent, "_uncertainty_latest_clusters", ()) or ():
        signals = {str(sig) for sig in getattr(cluster, "shared_signals", ()) or ()}
        if getattr(cluster, "is_giant_cluster", False):
            signals.add("uncertainty_giant_cluster")
        if getattr(cluster, "cluster_size", 0):
            signals.add("uncertainty_cluster")
        for var in getattr(cluster, "vars", ()) or ():
            out[int(var)].update(signals)
    return out


def _open_novelty_vars(agent: Any) -> set[int]:
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return set()
    return {
        int(getattr(item, "affected_var", -1))
        for item in getattr(ledger, "novelty", ()) or ()
        if getattr(item, "status", "") == "open"
    }


def _var_nethra_id(var: int, parents: tuple[int, ...], func: str) -> str:
    return f"var_fit:x{int(var)}:{func}({','.join(str(int(p)) for p in parents)})"


def _context_key(var: int, visible: int, parents: tuple[int, ...]) -> str:
    bits = [f"authority_strength", f"x{int(var)}", f"vis={int(visible)}"]
    if parents:
        bits.append("parents=" + ",".join(str(int(p)) for p in parents))
    return "|".join(bits)


def _authority_flags(state: AuthorityState) -> tuple[bool, bool, bool]:
    local_use_allowed = state != "insufficient"
    derivation_allowed = state != "quarantined_for_derivation"
    repair_attention_allowed = state == "repair_candidate"
    return derivation_allowed, local_use_allowed, repair_attention_allowed


def _debt_score(
    state: AuthorityState,
    persistence_count: int,
    evidence_required: int,
    evidence_paid: int,
) -> int:
    base = {
        "strong": 0,
        "usable": 0,
        "contested_best_available": 2,
        "quarantined_for_derivation": 4,
        "repair_candidate": 5,
        "insufficient": 3,
    }[state]
    return max(0, base + max(0, persistence_count - 1) + evidence_required - evidence_paid)


class AuthorityStateController:
    """Persistent visible-evidence authority-state controller.

    The controller turns strength metadata into bounded runtime hints. It does
    not issue certificates, revoke certificates, suppress skips, or replace fits.
    """

    def __init__(
        self,
        *,
        monitoring_budget_per_cycle: int = 2,
        repair_budget_per_cycle: int = 2,
        preserve_budget_per_cycle: int = 3,
        cooldown_cycles: int = 50,
        derivation_policy: AuthorityDerivationPolicy = "shadow",
        attribution_limit: int = 1000,
    ) -> None:
        if derivation_policy not in {
            "off",
            "quarantine_persistent",
            "quarantine_repair_only",
            "shadow",
        }:
            raise ValueError(
                "derivation_policy must be off, quarantine_persistent, "
                "quarantine_repair_only, or shadow"
            )
        self.monitoring_budget_per_cycle = max(0, int(monitoring_budget_per_cycle))
        self.repair_budget_per_cycle = max(0, int(repair_budget_per_cycle))
        self.preserve_budget_per_cycle = max(0, int(preserve_budget_per_cycle))
        self.cooldown_cycles = max(1, int(cooldown_cycles))
        self.derivation_policy = derivation_policy
        self.attribution_limit = max(0, int(attribution_limit))
        self.debts: dict[tuple[int, str], AuthorityDebt] = {}
        self.transitions: list[AuthorityStateTransition] = []
        self.derivation_gate_attributions: list[DerivationGateAttribution] = []
        self.derivation_gate_blocked_by_state: Counter[str] = Counter()
        self.derivation_gate_blocked_by_reason: Counter[str] = Counter()
        self.derivation_gate_blocked_by_handle_kind: Counter[str] = Counter()
        self.metrics: Counter[str] = Counter()
        self._latest_result = AuthorityControllerResult()

    @staticmethod
    def _reasons(record: AuthorityStrengthRecord) -> tuple[str, ...]:
        reasons = [record.reason]
        reasons.extend(str(item).split("=", 1)[0] for item in record.contradictory_evidence)
        return tuple(dict.fromkeys(reason for reason in reasons if reason))

    def _next_state(
        self,
        record: AuthorityStrengthRecord,
        debt: AuthorityDebt | None,
        persistence_count: int,
    ) -> AuthorityState:
        proposed = record.authority_state
        keys = _evidence_keys(record.contradictory_evidence)
        if proposed == "contested_best_available":
            if (
                persistence_count >= 2
                and "open_novelty" in keys
                and "repeated_fit_churn" in keys
            ):
                return "quarantined_for_derivation"
        if proposed in {"strong", "usable"} and debt is not None:
            if debt.evidence_required > debt.evidence_paid + 1:
                return "usable"
        return proposed

    @staticmethod
    def _reason_keys(reasons: tuple[str, ...]) -> set[str]:
        return {str(reason).split("=", 1)[0] for reason in reasons}

    def _policy_would_block(
        self,
        state: AuthorityState,
        reasons: tuple[str, ...],
        policy: AuthorityDerivationPolicy | None = None,
    ) -> bool:
        policy = self.derivation_policy if policy is None else policy
        if policy == "off":
            return False
        if policy in {"shadow", "quarantine_persistent"}:
            return state == "quarantined_for_derivation"
        if policy == "quarantine_repair_only":
            keys = self._reason_keys(reasons)
            return (
                state == "repair_candidate"
                or "sentinel_failures" in keys
                or "recent_revocations" in keys
            )
        return False

    def _effective_derivation_allowed(
        self,
        state: AuthorityState,
        reasons: tuple[str, ...],
    ) -> tuple[bool, bool]:
        would_block = self._policy_would_block(state, reasons)
        if self.derivation_policy == "shadow":
            return True, would_block
        return not would_block, would_block

    @staticmethod
    def _debt_outstanding(debt: AuthorityDebt) -> bool:
        return (
            debt.current_state not in {"strong", "usable"}
            or debt.evidence_paid < debt.evidence_required
        )

    def process(
        self,
        records: list[AuthorityStrengthRecord],
        cycle: int,
    ) -> AuthorityControllerResult:
        monitoring_bonus: dict[int, int] = {}
        preserve_vars: set[int] = set()
        repair_vars: set[int] = set()
        future_vars: set[int] = set()
        quarantined_vars: set[int] = set()
        monitoring_budget = self.monitoring_budget_per_cycle
        repair_budget = self.repair_budget_per_cycle
        preserve_budget = self.preserve_budget_per_cycle

        for record in records:
            key = (int(record.var), str(record.context_key))
            previous = self.debts.get(key)
            evidence_changed = (
                previous is None
                or previous.last_evidence_epoch != int(record.evidence_epoch)
            )
            persistence = previous.persistence_count if previous is not None else 0
            if evidence_changed:
                if (
                    previous is not None
                    and previous.current_state == record.authority_state
                    and record.authority_state in {
                        "contested_best_available",
                        "quarantined_for_derivation",
                        "repair_candidate",
                        "insufficient",
                    }
                ):
                    persistence += 1
                elif record.authority_state in {
                    "contested_best_available",
                    "repair_candidate",
                    "insufficient",
                }:
                    persistence = 1
                else:
                    persistence = 0

            state = self._next_state(record, previous, persistence)
            reasons = self._reasons(record)
            _, local_use_allowed, repair_allowed = _authority_flags(state)
            derivation_allowed, derivation_would_block = (
                self._effective_derivation_allowed(state, reasons)
            )
            if local_use_allowed and record.best_available:
                self.metrics["local_use_preserved"] += 1

            if previous is None:
                evidence_required = len(record.required_future_evidence)
                evidence_paid = 0
                self.metrics["authority_debt_created"] += int(state not in {"strong", "usable"})
            else:
                evidence_required = previous.evidence_required
                evidence_paid = previous.evidence_paid

            if evidence_changed and state == "contested_best_available" and persistence >= 2:
                evidence_required += 1
                future_vars.add(record.var)
                self.metrics["authority_debt_created"] += 1
                if preserve_budget > 0:
                    preserve_vars.add(record.var)
                    preserve_budget -= 1

            if state in {"strong", "usable"} and previous is not None:
                paid = 0
                evidence_keys = _evidence_keys(record.active_evidence)
                if "sentinel_coverage" in evidence_keys or "sentinel_passes" in evidence_keys:
                    paid += 1
                if "fit_margin" in evidence_keys:
                    paid += 1
                if paid:
                    before_paid = evidence_paid
                    evidence_paid = min(evidence_required, evidence_paid + paid)
                    self.metrics["authority_debt_paid"] += max(0, evidence_paid - before_paid)
                    if evidence_paid >= evidence_required:
                        state = "strong" if record.strength == "strong" else "usable"
                        derivation_allowed, derivation_would_block = (
                            self._effective_derivation_allowed(state, reasons)
                        )

            if state == "quarantined_for_derivation":
                quarantined_vars.add(record.var)
                if previous is None or previous.current_state != state:
                    self.metrics["derivation_quarantines"] += 1
            if state == "repair_candidate":
                self.metrics["repair_candidates"] += 1

            debt = AuthorityDebt(
                var=int(record.var),
                context_key=str(record.context_key),
                current_state=state,
                active_reasons=reasons,
                first_seen_cycle=(
                    previous.first_seen_cycle if previous is not None else int(cycle)
                ),
                persistence_count=persistence,
                last_seen_cycle=int(cycle),
                last_action_cycle=(
                    previous.last_action_cycle if previous is not None else -1_000_000
                ),
                evidence_required=max(0, int(evidence_required)),
                evidence_paid=max(0, int(evidence_paid)),
                debt_score=_debt_score(state, persistence, evidence_required, evidence_paid),
                derivation_allowed=derivation_allowed,
                derivation_would_block=derivation_would_block,
                local_use_allowed=local_use_allowed,
                repair_attention_allowed=repair_allowed,
                last_evidence_epoch=int(record.evidence_epoch),
            )

            previous_outstanding = (
                previous is not None and self._debt_outstanding(previous)
            )
            current_outstanding = self._debt_outstanding(debt)
            if previous_outstanding and current_outstanding:
                self.metrics["authority_debt_persisted"] += 1
            if previous is not None:
                if debt.debt_score > previous.debt_score:
                    self.metrics["authority_debt_escalated"] += 1
                elif debt.debt_score < previous.debt_score:
                    self.metrics["authority_debt_deescalated"] += 1
                if previous_outstanding and not current_outstanding:
                    self.metrics["authority_debt_paid"] += 1

            if previous is not None and previous.current_state != state:
                self.transitions.append(AuthorityStateTransition(
                    var=record.var,
                    context_key=record.context_key,
                    cycle=int(cycle),
                    previous_state=previous.current_state,
                    current_state=state,
                    next_state=state,
                    reason=",".join(reasons),
                    reasons=reasons,
                ))
                self.metrics["authority_state_transitions"] += 1

            # Split legacy pressure candidates from bounded controller effects.
            wants_monitoring = record.strength in {"weak", "contested"}
            wants_repair = record.strength == "contested"
            if wants_monitoring:
                self.metrics["monitoring_increases_from_strength_candidates"] += 1
                self.metrics["authority_action_candidates"] += 1
            if wants_repair:
                self.metrics["repair_priority_bumps_from_strength_candidates"] += 1
                self.metrics["authority_action_candidates"] += 1

            if wants_monitoring and state != "repair_candidate":
                self.metrics["monitoring_increases_from_strength_suppressed_by_state"] += 1
                self.metrics["monitoring_hints_suppressed"] += 1
                self.metrics["authority_noop_state_not_permit"] += 1
            if wants_repair and state != "repair_candidate":
                self.metrics["repair_priority_bumps_from_strength_suppressed_by_state"] += 1
                self.metrics["repair_hints_suppressed"] += 1
                self.metrics["authority_noop_state_not_permit"] += 1

            can_act = int(cycle) - debt.last_action_cycle >= self.cooldown_cycles
            acted = False
            if state == "repair_candidate":
                if not can_act:
                    if wants_monitoring:
                        self.metrics["monitoring_increases_from_strength_suppressed_by_cooldown"] += 1
                        self.metrics["monitoring_hints_suppressed"] += 1
                        self.metrics["authority_suppressed_cooldown"] += 1
                    if wants_repair:
                        self.metrics["repair_priority_bumps_from_strength_suppressed_by_cooldown"] += 1
                        self.metrics["repair_hints_suppressed"] += 1
                        self.metrics["authority_suppressed_cooldown"] += 1
                else:
                    if wants_monitoring:
                        if monitoring_budget > 0:
                            monitoring_bonus[record.var] = 1
                            monitoring_budget -= 1
                            acted = True
                            self.metrics["monitoring_increases_from_strength_applied"] += 1
                            self.metrics["monitoring_hints_applied"] += 1
                            self.metrics["authority_actions_applied"] += 1
                        else:
                            self.metrics["monitoring_increases_from_strength_suppressed_by_budget"] += 1
                            self.metrics["monitoring_hints_suppressed"] += 1
                            self.metrics["authority_suppressed_budget"] += 1
                    if wants_repair:
                        if repair_budget > 0:
                            repair_vars.add(record.var)
                            repair_budget -= 1
                            acted = True
                            self.metrics["repair_priority_bumps_from_strength_applied"] += 1
                            self.metrics["bounded_repairs_applied"] += 1
                            self.metrics["authority_actions_applied"] += 1
                        else:
                            self.metrics["repair_priority_bumps_from_strength_suppressed_by_budget"] += 1
                            self.metrics["repair_hints_suppressed"] += 1
                            self.metrics["authority_suppressed_budget"] += 1
                if acted:
                    debt.last_action_cycle = int(cycle)

            if wants_monitoring and record.var not in monitoring_bonus and state == "repair_candidate":
                self.metrics["monitoring_increases_from_strength_noops"] += int(not acted)
            if wants_repair and record.var not in repair_vars and state == "repair_candidate":
                self.metrics["repair_priority_bumps_from_strength_noops"] += int(not acted)
            if not acted:
                self.metrics["debt_noops"] += 1

            self.debts[key] = debt

        self._latest_result = AuthorityControllerResult(
            monitoring_bonus_vars=monitoring_bonus,
            preserve_vars=preserve_vars,
            repair_priority_vars=repair_vars,
            future_requirement_vars=future_vars,
            derivation_quarantined_vars=quarantined_vars,
        )
        return self._latest_result

    def check_derivation_gate(
        self,
        var: int,
        context_key: str | None = None,
        *,
        cycle: int = 0,
        blocked_handle_kind: str = "unknown",
        blocked_target: str | None = None,
    ) -> bool:
        self.metrics["derivation_gate_checks"] += 1
        matching = [
            debt for (debt_var, debt_context), debt in self.debts.items()
            if debt_var == int(var) and (context_key is None or debt_context == context_key)
        ]
        if not matching:
            self.metrics["derivation_gate_allowed"] += 1
            return True
        debt = max(matching, key=lambda item: item.debt_score)
        blocked = any(not item.derivation_allowed for item in matching)
        would_block = any(item.derivation_would_block for item in matching)
        allowed = not blocked
        if allowed:
            self.metrics["derivation_gate_allowed"] += 1
        else:
            self.metrics["derivation_gate_blocked"] += 1
        if would_block:
            self.metrics["derivation_gate_would_block"] += 1
            if self.derivation_policy == "shadow":
                self.metrics["derivation_gate_shadow_would_block"] += 1
        if blocked:
            self.derivation_gate_blocked_by_state[debt.current_state] += 1
            self.derivation_gate_blocked_by_handle_kind[str(blocked_handle_kind)] += 1
            for reason in debt.active_reasons:
                self.derivation_gate_blocked_by_reason[str(reason).split("=", 1)[0]] += 1
            if debt.local_use_allowed:
                self.metrics["authority_suppressed_local_use_only"] += 1
            self.metrics["authority_suppressed_derivation_only"] += 1
        if (blocked or would_block) and len(self.derivation_gate_attributions) < self.attribution_limit:
            self.derivation_gate_attributions.append(DerivationGateAttribution(
                var=int(var),
                context_key=str(debt.context_key),
                authority_state=debt.current_state,
                debt_score=int(debt.debt_score),
                active_reasons=tuple(debt.active_reasons),
                blocked_handle_kind=str(blocked_handle_kind),
                blocked_target=str(blocked_target or ""),
                cycle=int(cycle),
                local_use_allowed=bool(debt.local_use_allowed),
                derivation_allowed=bool(allowed),
                derivation_would_block=bool(would_block),
                authority_derivation_policy=self.derivation_policy,
            ))
        return allowed

    def derivation_allowed(self, var: int, context_key: str | None = None) -> bool:
        return self.check_derivation_gate(var, context_key)

    def summary(self) -> dict[str, Any]:
        state_counts = Counter(debt.current_state for debt in self.debts.values())
        outstanding = sum(
            1 for debt in self.debts.values()
            if debt.current_state not in {"strong", "usable"}
            or debt.evidence_paid < debt.evidence_required
        )
        debt_ages = [
            max(0, int(debt.last_seen_cycle) - int(debt.first_seen_cycle) + 1)
            for debt in self.debts.values()
            if self._debt_outstanding(debt)
        ]
        transitions_by_edge = Counter(
            f"{item.previous_state}->{item.next_state}" for item in self.transitions
        )
        transitions_by_reason: Counter[str] = Counter()
        for item in self.transitions:
            for reason in item.reasons:
                transitions_by_reason[str(reason).split("=", 1)[0]] += 1
        paid_down_keys = {
            (debt.var, debt.context_key)
            for debt in self.debts.values()
            if not self._debt_outstanding(debt)
        }
        transitions_later_paid_down = Counter(
            f"{item.previous_state}->{item.next_state}"
            for item in self.transitions
            if (item.var, item.context_key) in paid_down_keys
        )
        out: dict[str, Any] = {
            "authority_derivation_policy": self.derivation_policy,
            "authority_state_counts": dict(state_counts),
            "authority_debt_outstanding": outstanding,
            "debt_age_mean": (
                sum(debt_ages) / len(debt_ages) if debt_ages else 0.0
            ),
            "debt_age_max": max(debt_ages) if debt_ages else 0,
            "authority_debts": [asdict(debt) for debt in self.debts.values()],
            "authority_state_transitions_examples": [
                asdict(item) for item in self.transitions[-20:]
            ],
            "authority_state_transitions_by_edge": dict(transitions_by_edge),
            "authority_state_transitions_by_reason": dict(transitions_by_reason),
            "authority_state_transitions_to_derivation_quarantine": int(
                sum(
                    1 for item in self.transitions
                    if item.next_state == "quarantined_for_derivation"
                )
            ),
            "authority_state_transitions_later_paid_down": dict(
                transitions_later_paid_down
            ),
            "derivation_gate_attributions": [
                asdict(item) for item in self.derivation_gate_attributions
            ],
            "derivation_gate_blocked_by_state": dict(
                self.derivation_gate_blocked_by_state
            ),
            "derivation_gate_blocked_by_reason": dict(
                self.derivation_gate_blocked_by_reason
            ),
            "derivation_gate_blocked_by_handle_kind": dict(
                self.derivation_gate_blocked_by_handle_kind
            ),
        }
        for name in (
            "authority_debt_created",
            "authority_debt_persisted",
            "authority_debt_paid",
            "authority_debt_escalated",
            "authority_debt_deescalated",
            "authority_state_transitions",
            "derivation_quarantines",
            "derivation_gate_checks",
            "derivation_gate_allowed",
            "derivation_gate_blocked",
            "derivation_gate_would_block",
            "derivation_gate_shadow_would_block",
            "local_use_preserved",
            "repair_candidates",
            "bounded_repairs_applied",
            "monitoring_hints_applied",
            "monitoring_hints_suppressed",
            "repair_hints_suppressed",
            "debt_noops",
            "authority_action_candidates",
            "authority_actions_applied",
            "authority_noop_state_not_permit",
            "authority_suppressed_cooldown",
            "authority_suppressed_budget",
            "authority_suppressed_local_use_only",
            "authority_suppressed_derivation_only",
            "monitoring_increases_from_strength_candidates",
            "monitoring_increases_from_strength_applied",
            "monitoring_increases_from_strength_suppressed_by_state",
            "monitoring_increases_from_strength_suppressed_by_cooldown",
            "monitoring_increases_from_strength_suppressed_by_budget",
            "monitoring_increases_from_strength_noops",
            "repair_priority_bumps_from_strength_candidates",
            "repair_priority_bumps_from_strength_applied",
            "repair_priority_bumps_from_strength_suppressed_by_state",
            "repair_priority_bumps_from_strength_suppressed_by_cooldown",
            "repair_priority_bumps_from_strength_suppressed_by_budget",
            "repair_priority_bumps_from_strength_noops",
        ):
            out[name] = int(self.metrics.get(name, 0))
        return out


def _revocation_count(n: Any) -> int:
    certs = list(getattr(n, "certificates", {}).values())
    certs += list(getattr(n, "route_certs", {}).values())
    return sum(1 for cert in certs if getattr(cert, "revoked_by", None))


def _sentinel_passes(n: Any) -> int:
    return sum(
        _as_int(getattr(cert, "sentinel_passes", 0))
        for cert in getattr(n, "certificates", {}).values()
    )


def _classify(
    *,
    strong_observations: int,
    sentinel_count: int,
    sentinel_passes: int,
    fit_margin: int | None,
    near_tie_count: int,
    tie_count: int,
    open_novelty: bool,
    churn: bool,
    recent_revocations: int,
    sentinel_failures: int,
    passive_stress: int,
    dormant_alternatives: int,
    cluster_signals: set[str],
    best_available: bool,
) -> tuple[AuthorityStrength, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    active: list[str] = []
    contradictory: list[str] = []
    future: list[str] = []

    if strong_observations > 0:
        active.append(f"strong_observations={strong_observations}")
    if sentinel_count > 0:
        active.append(f"sentinel_coverage={sentinel_count}")
    if sentinel_passes > 0:
        active.append(f"sentinel_passes={sentinel_passes}")
    if fit_margin is not None and fit_margin >= 3:
        active.append(f"fit_margin={fit_margin}")

    if fit_margin is None:
        contradictory.append("fit_margin_missing")
    elif fit_margin <= 1:
        contradictory.append(f"low_fit_margin={fit_margin}")
    if near_tie_count > 1:
        contradictory.append(f"near_ties={near_tie_count}")
    if tie_count > 1:
        contradictory.append(f"tied_frontier={tie_count}")
    if open_novelty:
        contradictory.append("open_novelty")
    if churn:
        contradictory.append("repeated_fit_churn")
    if recent_revocations > 0:
        contradictory.append(f"recent_revocations={recent_revocations}")
    if sentinel_failures > 0:
        contradictory.append(f"sentinel_failures={sentinel_failures}")
    if passive_stress > 0:
        contradictory.append(f"passive_stress={passive_stress}")
    if dormant_alternatives > 0:
        contradictory.append(f"dormant_alternatives={dormant_alternatives}")
    if cluster_signals:
        contradictory.extend(sorted(cluster_signals))

    if sentinel_count == 0:
        future.append("sentinel_coverage")
    if strong_observations < 2:
        future.append("more_stable_observations")
    if fit_margin is None or fit_margin <= 1:
        future.append("larger_fit_margin")
    if near_tie_count > 1 or tie_count > 1 or dormant_alternatives > 0:
        future.append("separating_evidence")
    if open_novelty or churn or recent_revocations or sentinel_failures:
        future.append("stability_after_uncertainty")

    contested_markers = {
        "open_novelty",
        "repeated_fit_churn",
        "uncertainty_cluster",
        "uncertainty_giant_cluster",
    }
    has_contested_marker = (
        any(item.split("=", 1)[0] in contested_markers for item in contradictory)
        or near_tie_count > 1
        or tie_count > 1
        or dormant_alternatives > 0
        or recent_revocations > 0
        or sentinel_failures > 0
    )
    if has_contested_marker:
        return (
            "contested",
            "active_visible_conflict",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    if strong_observations <= 0 and sentinel_count <= 0 and fit_margin is None:
        return (
            "insufficient",
            "too_little_visible_evidence",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    low_observation = strong_observations < 2
    low_margin = fit_margin is not None and fit_margin <= 1
    limited_sentinel = sentinel_count <= 0
    if best_available and (low_observation or low_margin or limited_sentinel):
        return (
            "weak",
            "best_available_but_limited_evidence",
            tuple(active),
            tuple(dict.fromkeys(contradictory)),
            tuple(dict.fromkeys(future)),
        )

    if (
        strong_observations >= 3
        and sentinel_count > 0
        and (fit_margin is None or fit_margin >= 2)
        and not contradictory
    ):
        return (
            "strong",
            "stable_visible_evidence",
            tuple(active),
            tuple(),
            tuple(),
        )

    return (
        "usable",
        "best_available_mostly_stable",
        tuple(active),
        tuple(dict.fromkeys(contradictory)),
        tuple(dict.fromkeys(future)),
    )


def compute_authority_strength_records(agent: Any, cycle: int) -> list[AuthorityStrengthRecord]:
    """Classify current handles using visible runtime evidence only."""
    ledger = getattr(agent, "ledger", None)
    world = getattr(agent, "world", None)
    if ledger is None or world is None:
        return []
    visible = _as_int(getattr(world, "visible_count", 0))
    fit_by_var = _latest_fit_by_var(agent)
    novelty_vars = _open_novelty_vars(agent)
    cluster_signals = _cluster_signals_by_var(agent)
    records: list[AuthorityStrengthRecord] = []

    for var in range(visible):
        n = ledger.vars[var]
        parents = tuple(int(p) for p in getattr(n, "parents", ()) or ())
        func = str(getattr(n, "func", ""))
        fits = fit_by_var.get(var, [])
        last_fit = fits[-1] if fits else None
        fit_margin = (
            int(getattr(last_fit, "margin"))
            if last_fit is not None and getattr(last_fit, "margin", None) is not None
            else None
        )
        near_tie_count = len(getattr(last_fit, "near_tie_candidates", ()) or ()) if last_fit else 0
        tie_count = len(getattr(last_fit, "tie_set", ()) or ()) if last_fit else 0
        frontier = getattr(n, "tied_frontier", None)
        if frontier is not None:
            tie_count = max(tie_count, len(getattr(frontier, "candidates", ()) or ()))
        nid = _var_nethra_id(var, parents, func)
        strength, reason, active, contradictory, future = _classify(
            strong_observations=_as_int(getattr(n, "strong_observations", 0)),
            sentinel_count=len(getattr(n, "sentinels", ()) or ()),
            sentinel_passes=_sentinel_passes(n),
            fit_margin=fit_margin,
            near_tie_count=near_tie_count,
            tie_count=tie_count,
            open_novelty=var in novelty_vars,
            churn=_recent_churn(fits),
            recent_revocations=_revocation_count(n),
            sentinel_failures=_as_int(getattr(n, "consecutive_sentinel_failures", 0)),
            passive_stress=_as_int(getattr(getattr(n, "envelope", None), "out_of_band_count", 0)),
            dormant_alternatives=len(getattr(n, "dormant_alternatives", ()) or ()),
            cluster_signals=cluster_signals.get(var, set()),
            best_available=True,
        )
        authority_state = proposed_authority_state(
            strength=strength,
            contradictory_evidence=contradictory,
            best_available=True,
        )
        evidence_epoch = max(
            _as_int(getattr(n, "full_audits", 0)),
            _as_int(getattr(last_fit, "cycle", 0)) if last_fit is not None else 0,
        )
        records.append(AuthorityStrengthRecord(
            var=var,
            nethra_id=nid,
            context_key=_context_key(var, visible, parents),
            cycle=int(cycle),
            strength=strength,
            reason=reason,
            active_evidence=active,
            contradictory_evidence=contradictory,
            required_future_evidence=future,
            uncertainty_signals=tuple(sorted(cluster_signals.get(var, set()))),
            prior_role=_recent_role_for_nethra(agent, nid),
            best_available=True,
            authority_state=authority_state,
            evidence_epoch=evidence_epoch,
        ))
    return records


def summarize_authority_strength_records(
    records: list[AuthorityStrengthRecord],
    *,
    monitoring_increases: int = 0,
    alternatives_preserved: int = 0,
) -> AuthorityStrengthSummary:
    strength_counts = Counter(record.strength for record in records)
    reason_counts = Counter(record.reason for record in records)
    state_counts = Counter(record.authority_state for record in records)
    return AuthorityStrengthSummary(
        counts_by_strength=dict(strength_counts),
        counts_by_reason=dict(reason_counts),
        counts_by_authority_state=dict(state_counts),
        weak_best_available=sum(
            1 for record in records
            if record.best_available and record.strength == "weak"
        ),
        contested_best_available=sum(
            1 for record in records
            if record.best_available and record.strength == "contested"
        ),
        monitoring_increases=max(0, int(monitoring_increases)),
        alternatives_preserved=max(0, int(alternatives_preserved)),
        future_evidence_requirements=sum(
            1 for record in records if record.required_future_evidence
        ),
    )


def records_to_dicts(records: list[AuthorityStrengthRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def summary_to_dict(summary: AuthorityStrengthSummary) -> dict[str, Any]:
    return asdict(summary)
