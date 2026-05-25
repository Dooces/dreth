from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Hybrid control interface layer: Protocol definitions and default symbolic
# implementations.
#
# INVARIANT (enforced by design, not runtime):
#   Provider outputs are prediction/ranking surfaces only.
#   No provider may:
#     - Create a NethraCertificate
#     - Mutate ledger.vars[*].certificates or route_certs
#     - Mark a var tareth / trass
#     - Authorize a skip
#     - Bypass sentinels
#   Cert authority remains an earned, defeasible ledger record on the Dreth
#   ledger/cert/sentinel path. Provider confidence values are NEVER treated as
#   cert authority.
#
# Stage 1: interface separation + default symbolic wrappers.
# Future stages plug in neural/MoE components by implementing the Protocols
# without touching the authority-record path.
# ─────────────────────────────────────────────────────────────────────────────

import dataclasses
from collections import defaultdict, deque
from typing import Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclasses.dataclass
class ResidualPrediction:
    """Output from a ResidualPredictor for one variable in one cycle.
    Does not carry cert authority — only signals whether active sentinel is needed.
    """
    var: int
    ok: bool          # True → residual within tolerance; skip active sentinel this cycle
    stressed: bool    # True → residual exceeds tolerance; run active sentinel
    residual: float   # |actual - predicted|
    predicted: float  # provider's predicted value
    actual: float     # observed world state


@dataclasses.dataclass
class ParentRanking:
    """Output from a ParentRanker for one target variable.
    Ranked list only — does NOT exclude candidates via cert logic.
    """
    target: int
    ranked: Tuple[int, ...]      # candidate vars in descending priority order
    scores: Dict[int, float]     # sensitivity score per candidate
    source_by_candidate: Dict[int, str] = dataclasses.field(default_factory=dict)
    diagnostics: Dict[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ProbeProposal:
    """Output from a ProbeProposer for one variable.
    Proposals only — does NOT create earned authority records or update cert state.
    """
    var: int
    probes: Tuple[Tuple[int, float], ...]   # (iv_var, iv_val) pairs


@dataclasses.dataclass
class ExpertPrediction:
    """Output from an Expert for one hypothesis evaluation.
    Confidence is diagnostic only — NEVER treated as cert authority.
    """
    parents: Tuple[int, ...]
    func: str
    score: float        # predicted score (not authoritative)
    confidence: float   # self-reported confidence [0, 1]; diagnostic only
    route_key: str      # which expert produced this


@dataclasses.dataclass
class RepairEvent:
    """Diagnostic record of one provider call during a repair cycle."""
    cycle: int
    var: int
    provider: str    # "residual_predictor" | "parent_ranker" | "probe_proposer" | "expert_router"
    call_count: int
    outcome: str     # "ok" | "stressed" | "proposal_issued" | "ranked" | "routed"


@dataclasses.dataclass
class ParentProposalDiagnostics:
    """Diagnostic-only quality counters for ParentRanker proposals."""
    calls: int = 0
    proposed_total: int = 0
    proposed_in_final_fit: int = 0
    proposed_excluded_by_route_cert: int = 0
    proposed_not_used: int = 0
    miss_chosen_parent_count: int = 0
    _rank_sum: int = 0
    _rank_count: int = 0
    rank_of_chosen_parent_max: int = 0
    history_ranker_calls: int = 0
    sensitivity_rescue_calls: int = 0
    sensitivity_rescue_interventions: int = 0
    rescue_candidates_added: int = 0
    rescue_chosen_parent_hits: int = 0
    chosen_parent_from_history: int = 0
    chosen_parent_from_rescue: int = 0

    def record_call(
        self,
        ranked: Tuple[int, ...],
        post_route: Tuple[int, ...],
        diagnostics: Optional[Dict[str, int]] = None,
    ) -> None:
        self.calls += 1
        self.proposed_total += len(ranked)
        self.proposed_excluded_by_route_cert += max(0, len(ranked) - len(post_route))
        if diagnostics:
            self.history_ranker_calls += int(diagnostics.get("history_ranker_calls", 0))
            self.sensitivity_rescue_calls += int(diagnostics.get("sensitivity_rescue_calls", 0))
            self.sensitivity_rescue_interventions += int(
                diagnostics.get("sensitivity_rescue_interventions", 0)
            )
            self.rescue_candidates_added += int(diagnostics.get("rescue_candidates_added", 0))

    def record_fit(
        self,
        ranked_post_route: Tuple[int, ...],
        chosen_parents: Tuple[int, ...],
        source_by_candidate: Optional[Dict[int, str]] = None,
    ) -> None:
        rank_index = {p: i for i, p in enumerate(ranked_post_route)}
        chosen = set(chosen_parents)
        for parent in chosen_parents:
            rank = rank_index.get(parent)
            if rank is None:
                self.miss_chosen_parent_count += 1
                continue
            self.proposed_in_final_fit += 1
            source = (source_by_candidate or {}).get(parent, "")
            if "history" in source:
                self.chosen_parent_from_history += 1
            if "rescue" in source:
                self.chosen_parent_from_rescue += 1
                self.rescue_chosen_parent_hits += 1
            self._rank_sum += rank
            self._rank_count += 1
            self.rank_of_chosen_parent_max = max(self.rank_of_chosen_parent_max, rank)
        self.proposed_not_used += sum(1 for p in ranked_post_route if p not in chosen)

    @property
    def rank_of_chosen_parent_mean(self) -> float:
        return self._rank_sum / self._rank_count if self._rank_count else 0.0

    @property
    def chosen_parent_hit_rate(self) -> float:
        denom = self.proposed_in_final_fit + self.miss_chosen_parent_count
        return self.proposed_in_final_fit / denom if denom else 0.0


@dataclasses.dataclass
class ProbeProposalDiagnostics:
    """Diagnostic-only quality counters for ProbeProposer proposals."""
    provider_probes_proposed: int = 0
    provider_probes_valid: int = 0
    provider_probes_invalid: int = 0
    provider_probes_used_by_fit: int = 0
    provider_probe_improved_margin_count: int = 0
    provider_probe_no_effect_count: int = 0

    def record_proposal(self, proposed: int, valid: int, invalid: int) -> None:
        self.provider_probes_proposed += proposed
        self.provider_probes_valid += valid
        self.provider_probes_invalid += invalid

    def record_fit(
        self,
        valid_probes: Tuple[Tuple[int, float], ...],
        fit_probes: Tuple[Tuple[int, float], ...],
        margin: int,
    ) -> None:
        if not valid_probes:
            return
        used = sum(1 for probe in valid_probes if probe in fit_probes)
        self.provider_probes_used_by_fit += used
        if used > 0 and margin > 0:
            self.provider_probe_improved_margin_count += 1
        else:
            self.provider_probe_no_effect_count += 1


# ── Protocols ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ResidualPredictor(Protocol):
    """Predicts whether a variable's current state is consistent with its
    currently authoritative hypothesis.

    CONTRACT: Must NOT issue certs or mutate ledger state.
    Called once per authoritative variable per cycle (passive monitoring path).
    """

    def predict_residual(
        self,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        parent_vals: List[float],
        actual: float,
        tolerance: float,
    ) -> ResidualPrediction:
        ...


@runtime_checkable
class ParentRanker(Protocol):
    """Ranks candidate parent variables for a target.

    CONTRACT: Must NOT certify any variable or exclude candidates via cert logic.
    Cert-based exclusion (route certs) is applied by ChainedAgent AFTER ranking.
    """

    def rank_parents(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> ParentRanking:
        ...


@runtime_checkable
class ProbeProposer(Protocol):
    """Proposes discriminating probes for a variable.

    CONTRACT: Must NOT use probe results to certify hypotheses or update cert state.
    Returned probes are injected as forced_probes into fit_var; scoring and cert
    decisions remain in the standard audit path.
    """

    def propose_probes(
        self,
        var: int,
        available_parents: Set[int],
        budget: int,
    ) -> ProbeProposal:
        ...


@runtime_checkable
class Expert(Protocol):
    """Evaluates hypothesis quality for (var, parents, func).

    CONTRACT: Must NOT issue certs or alter ledger state.
    Confidence values are diagnostic only.
    """

    def evaluate(
        self,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        context: Dict,
    ) -> ExpertPrediction:
        ...


@runtime_checkable
class ExpertRouter(Protocol):
    """Selects which Expert to use for a variable and records routing metadata.

    CONTRACT: Must NOT use routing decisions to authorize skips or grant cert authority.
    Routing metadata is diagnostic only.
    """

    def route(
        self,
        var: int,
        available_parents: Set[int],
        context: Dict,
    ) -> Tuple["Expert", Dict]:   # (expert, route_metadata)
        ...


# ── Default symbolic implementations ─────────────────────────────────────────

class SymbolicResidualPredictor:
    """Default ResidualPredictor: reproduces the passive residual logic using
    currently authoritative parents + FUNC_LIBRARY.

    Extracts the current agent behavior as a provider. May NOT issue certs —
    it only computes and returns the residual signal for the agent to act on.

    Counters (call_count, ok_count, stressed_count) are diagnostic only.
    """

    def __init__(self) -> None:
        from .functions import FUNC_LIBRARY
        self._func_lib = FUNC_LIBRARY
        self.call_count: int = 0
        self.ok_count: int = 0
        self.stressed_count: int = 0

    def predict_residual(
        self,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        parent_vals: List[float],
        actual: float,
        tolerance: float,
    ) -> ResidualPrediction:
        self.call_count += 1
        _f = self._func_lib.get(func)
        if _f is None:
            # Unknown func — cannot predict; treat as stressed so active sentinel runs.
            self.stressed_count += 1
            return ResidualPrediction(
                var=var, ok=False, stressed=True,
                residual=float("inf"), predicted=float("nan"), actual=actual,
            )
        predicted = _f(list(parent_vals))
        residual = abs(actual - predicted)
        ok = residual <= tolerance
        if ok:
            self.ok_count += 1
        else:
            self.stressed_count += 1
        return ResidualPrediction(
            var=var, ok=ok, stressed=not ok,
            residual=residual, predicted=predicted, actual=actual,
        )


class SensitivityParentRanker:
    """Default ParentRanker: wraps the current per-target sensitivity screen.

    For each candidate var, perturbs it to 0.05 / 0.95 and measures |Δtarget|.
    Returns candidates ranked by movement without certifying anything.

    This is an adapter over ChainedAgent._screen_candidate_parents logic.
    Cert-based exclusion (route_certs / trass role) is applied by ChainedAgent
    AFTER ranking returns; this class never touches certs.
    """

    def __init__(self, world) -> None:
        self._world = world
        self.call_count: int = 0

    def rank_parents(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> ParentRanking:
        self.call_count += 1
        eligible = [c for c in candidates if c != target]
        scores: Dict[int, float] = {}
        for cand in eligible:
            lo = self._world.predict_var_under_intervention(target, cand, 0.05)
            hi = self._world.predict_var_under_intervention(target, cand, 0.95)
            scores[cand] = abs(hi - lo)
        ranked = tuple(sorted(scores, key=lambda c: scores[c], reverse=True)[:top_m])
        return ParentRanking(target=target, ranked=ranked, scores=scores)


class HistoryParentRanker:
    """ParentRanker using only agent-visible audit and residual history.

    It learns from prior Dreth fit results and residual co-stress observations
    supplied by ChainedAgent. It returns ranked candidates only; route-cert
    filtering and all certification remain outside the provider.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self._fit_parent_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._co_stress_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._route_exclusion_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._cycle: Optional[int] = None
        self._stressed_this_cycle: Set[int] = set()
        self._recent_stressed = deque(maxlen=128)

    def observe_residual_event(self, var: int, cycle: int, stressed: bool) -> None:
        if self._cycle != cycle:
            self._cycle = cycle
            self._stressed_this_cycle = set()
        if not stressed:
            return
        for other in self._stressed_this_cycle:
            if other == var:
                continue
            self._co_stress_counts[(var, other)] += 1
            self._co_stress_counts[(other, var)] += 1
        self._stressed_this_cycle.add(var)
        self._recent_stressed.append(var)

    def observe_fit_result(self, target: int, parents: Tuple[int, ...], margin: int = 0) -> None:
        weight = 2 if margin > 0 else 1
        for parent in parents:
            self._fit_parent_counts[(target, parent)] += weight

    def observe_route_exclusions(self, target: int, excluded: Tuple[int, ...]) -> None:
        for parent in excluded:
            self._route_exclusion_counts[(target, parent)] += 1

    def rank_parents(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> ParentRanking:
        self.call_count += 1
        eligible = [c for c in candidates if c != target]
        recent_counts: Dict[int, int] = defaultdict(int)
        for v in self._recent_stressed:
            recent_counts[v] += 1
        scores: Dict[int, float] = {}
        for cand in eligible:
            fit_score = 10.0 * self._fit_parent_counts[(target, cand)]
            co_stress = 2.0 * self._co_stress_counts[(target, cand)]
            recent = 0.25 * recent_counts[cand]
            route_penalty = 4.0 * self._route_exclusion_counts[(target, cand)]
            scores[cand] = fit_score + co_stress + recent - route_penalty
        ranked = tuple(
            sorted(eligible, key=lambda c: (-scores[c], c))[:top_m]
        )
        return ParentRanking(target=target, ranked=ranked, scores=scores)


class HistoryRescueParentRanker(HistoryParentRanker):
    """History ranker with a small sensitivity rescue pool.

    History proposes the first tranche. Sensitivity probes only a bounded pool
    built from visible agent-side history: recent stress, prior chosen parents,
    and round-robin visible candidates. The provider returns candidates only.
    ChainedAgent still applies route-cert exclusion and owns all authority.
    """

    def __init__(self, world) -> None:
        super().__init__()
        self._world = world
        self.history_ranker_calls: int = 0
        self.sensitivity_rescue_calls: int = 0
        self.sensitivity_rescue_interventions: int = 0
        self.rescue_candidates_added: int = 0
        self._rr_cursor: Dict[int, int] = defaultdict(int)
        self._recent_rescue_tested: Dict[int, deque] = defaultdict(lambda: deque(maxlen=64))
        self._last_rescue_parent_by_target: Dict[int, Optional[int]] = {}

    def _history_scores(self, target: int, candidates: Set[int]) -> Dict[int, float]:
        recent_counts: Dict[int, int] = defaultdict(int)
        for v in self._recent_stressed:
            recent_counts[v] += 1
        scores: Dict[int, float] = {}
        for cand in candidates:
            if cand == target:
                continue
            fit_score = 10.0 * self._fit_parent_counts[(target, cand)]
            co_stress = 2.0 * self._co_stress_counts[(target, cand)]
            recent = 0.25 * recent_counts[cand]
            route_penalty = 4.0 * self._route_exclusion_counts[(target, cand)]
            scores[cand] = fit_score + co_stress + recent - route_penalty
        return scores

    def _round_robin_candidates(
        self,
        target: int,
        eligible: List[int],
        selected: Set[int],
        limit: int,
    ) -> List[int]:
        if not eligible or limit <= 0:
            return []
        recent_tested = set(self._recent_rescue_tested[target])
        out: List[int] = []
        start = self._rr_cursor[target] % len(eligible)
        for offset in range(len(eligible)):
            cand = eligible[(start + offset) % len(eligible)]
            if cand in selected or cand in recent_tested or cand in out:
                continue
            out.append(cand)
            if len(out) >= limit:
                break
        if len(out) < limit:
            for offset in range(len(eligible)):
                cand = eligible[(start + offset) % len(eligible)]
                if cand in selected or cand in out:
                    continue
                out.append(cand)
                if len(out) >= limit:
                    break
        self._rr_cursor[target] = (start + max(1, len(out))) % len(eligible)
        return out

    def rank_parents(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> ParentRanking:
        self.call_count += 1
        self.history_ranker_calls += 1
        eligible = [c for c in sorted(candidates) if c != target]
        if top_m <= 0 or not eligible:
            return ParentRanking(
                target=target, ranked=(), scores={},
                diagnostics={"history_ranker_calls": 1},
            )

        rescue_r = top_m - (top_m // 2)
        if top_m >= 4:
            rescue_r = max(2, rescue_r)
        rescue_r = min(top_m, rescue_r)
        history_h = max(0, top_m - rescue_r)

        history_scores = self._history_scores(target, set(eligible))
        history_ranked = tuple(
            sorted(eligible, key=lambda c: (-history_scores[c], c))[:history_h]
        )
        selected = set(history_ranked)

        recent_counts: Dict[int, int] = defaultdict(int)
        for v in self._recent_stressed:
            recent_counts[v] += 1
        stress_pool = [
            c for c, _ in sorted(
                ((c, recent_counts[c]) for c in eligible if c not in selected and recent_counts[c] > 0),
                key=lambda item: (-item[1], item[0]),
            )[:rescue_r]
        ]
        prior_fit_pool = [
            c for c, _ in sorted(
                ((c, self._fit_parent_counts[(target, c)]) for c in eligible
                 if c not in selected and self._fit_parent_counts[(target, c)] > 0),
                key=lambda item: (-item[1], item[0]),
            )[:rescue_r]
        ]
        rr_pool = self._round_robin_candidates(target, eligible, selected, rescue_r)

        rescue_pool: List[int] = []
        for cand in stress_pool + prior_fit_pool + rr_pool:
            if cand not in selected and cand not in rescue_pool:
                rescue_pool.append(cand)

        rescue_scores: Dict[int, float] = {}
        for cand in rescue_pool:
            lo = self._world.predict_var_under_intervention(target, cand, 0.05)
            hi = self._world.predict_var_under_intervention(target, cand, 0.95)
            rescue_scores[cand] = abs(hi - lo)
            self._recent_rescue_tested[target].append(cand)

        rescue_interventions = 2 * len(rescue_pool)
        self.sensitivity_rescue_calls += 1
        self.sensitivity_rescue_interventions += rescue_interventions

        rescue_ranked = tuple(
            sorted(rescue_scores, key=lambda c: (-rescue_scores[c], c))[:rescue_r]
        )
        self._last_rescue_parent_by_target[target] = rescue_ranked[0] if rescue_ranked else None

        ranked: List[int] = []
        source_by_candidate: Dict[int, str] = {}
        for cand in history_ranked:
            ranked.append(cand)
            source_by_candidate[cand] = "history"
        added_from_rescue = 0
        for cand in rescue_ranked:
            if cand in source_by_candidate:
                source_by_candidate[cand] = "history_rescue"
                continue
            ranked.append(cand)
            source_by_candidate[cand] = "rescue"
            added_from_rescue += 1

        self.rescue_candidates_added += added_from_rescue
        scores = dict(history_scores)
        scores.update(rescue_scores)
        return ParentRanking(
            target=target,
            ranked=tuple(ranked[:top_m]),
            scores=scores,
            source_by_candidate=source_by_candidate,
            diagnostics={
                "history_ranker_calls": 1,
                "sensitivity_rescue_calls": 1,
                "sensitivity_rescue_interventions": rescue_interventions,
                "rescue_candidates_added": added_from_rescue,
            },
        )


class DiscriminationProbeProposer:
    """Default ProbeProposer: wraps the current separating-probe logic.

    When a TiedFrontier with separating_probes is available it will be injected
    by ChainedAgent directly (existing path). This default returns an empty
    proposal so fit_var falls back to its standard discrimination pool.

    Future neural probe proposers override this with learned probes without
    touching the cert path.
    """

    def __init__(self) -> None:
        self.call_count: int = 0

    def propose_probes(
        self,
        var: int,
        available_parents: Set[int],
        budget: int,
    ) -> ProbeProposal:
        self.call_count += 1
        # Default: empty — ChainedAgent's existing forced_probes path handles
        # TiedFrontier separating probes; this provider adds nothing new.
        return ProbeProposal(var=var, probes=())


class HistoryProbeProposer:
    """ProbeProposer using agent-visible ambiguity, stress, and ranking history."""

    def __init__(self, max_probes: int = 3) -> None:
        self.call_count: int = 0
        self.max_probes = max(0, max_probes)
        self._frontier_probes: Dict[int, Tuple[Tuple[int, float], ...]] = {}
        self._parent_rankings: Dict[int, Tuple[int, ...]] = {}
        self._cycle: Optional[int] = None
        self._stressed_this_cycle: Set[int] = set()
        self._recent_stressed = deque(maxlen=128)

    def observe_frontier_probes(self, var: int, probes: Tuple[Tuple[int, float], ...]) -> None:
        if probes:
            self._frontier_probes[var] = probes

    def observe_parent_ranking(self, var: int, ranked: Tuple[int, ...]) -> None:
        self._parent_rankings[var] = ranked

    def observe_parent_ranking_metadata(
        self,
        var: int,
        ranked: Tuple[int, ...],
        source_by_candidate: Dict[int, str],
    ) -> None:
        self.observe_parent_ranking(var, ranked)

    def observe_residual_event(self, var: int, cycle: int, stressed: bool) -> None:
        if self._cycle != cycle:
            self._cycle = cycle
            self._stressed_this_cycle = set()
        if stressed:
            self._stressed_this_cycle.add(var)
            self._recent_stressed.append(var)

    def propose_probes(
        self,
        var: int,
        available_parents: Set[int],
        budget: int,
    ) -> ProbeProposal:
        self.call_count += 1
        limit = min(self.max_probes, max(0, budget))
        out: List[Tuple[int, float]] = []

        def add(probe: Tuple[int, float]) -> None:
            if len(out) < limit and probe not in out:
                out.append(probe)

        for probe in self._frontier_probes.get(var, ()):
            add(probe)
        values = (0.1, 0.9)
        for i, parent in enumerate(self._parent_rankings.get(var, ())):
            if parent in available_parents:
                add((parent, values[i % len(values)]))
        for i, stressed_var in enumerate(reversed(self._recent_stressed)):
            if stressed_var in available_parents:
                add((stressed_var, values[(i + 1) % len(values)]))

        return ProbeProposal(var=var, probes=tuple(out))


class HistoryRescueProbeProposer(HistoryProbeProposer):
    """History probe proposer that appends one probe from a rescue parent."""

    def __init__(self, max_probes: int = 4) -> None:
        super().__init__(max_probes=max_probes)
        self._rescue_parent_by_target: Dict[int, int] = {}

    def observe_parent_ranking_metadata(
        self,
        var: int,
        ranked: Tuple[int, ...],
        source_by_candidate: Dict[int, str],
    ) -> None:
        super().observe_parent_ranking_metadata(var, ranked, source_by_candidate)
        for cand in ranked:
            if "rescue" in source_by_candidate.get(cand, ""):
                self._rescue_parent_by_target[var] = cand
                break

    def propose_probes(
        self,
        var: int,
        available_parents: Set[int],
        budget: int,
    ) -> ProbeProposal:
        base = super().propose_probes(var, available_parents, budget)
        out = list(base.probes)
        rescue_parent = self._rescue_parent_by_target.get(var)
        if (
            rescue_parent is not None
            and rescue_parent in available_parents
            and len(out) < min(self.max_probes, max(0, budget))
        ):
            probe = (rescue_parent, 0.9)
            if probe not in out:
                out.append(probe)
        return ProbeProposal(var=var, probes=tuple(out))


class FuncLibraryExpert:
    """Default Expert: evaluates (parents, func) pairs using FUNC_LIBRARY.

    Wraps the existing hypothesis evaluation logic without accessing hidden world
    structure.  Does NOT issue certs or alter ledger state.
    """

    KEY = "func_library"

    def __init__(self) -> None:
        from .functions import FUNC_LIBRARY
        self._func_lib = FUNC_LIBRARY

    def evaluate(
        self,
        var: int,
        parents: Tuple[int, ...],
        func: str,
        context: Dict,
    ) -> ExpertPrediction:
        parent_vals = context.get("parent_vals", [])
        _f = self._func_lib.get(func)
        score = 0.0
        if _f is not None and parent_vals:
            try:
                score = float(_f(list(parent_vals)))
            except Exception:
                score = 0.0
        return ExpertPrediction(
            parents=parents, func=func,
            score=score, confidence=1.0,
            route_key=self.KEY,
        )


class FuncLibraryRouter:
    """Default ExpertRouter: always routes to FuncLibraryExpert.

    Records routing metadata for diagnostics but does NOT use routing decisions
    to authorize skips or grant cert authority.
    """

    def __init__(self) -> None:
        self._expert = FuncLibraryExpert()
        self.call_count: int = 0

    def route(
        self,
        var: int,
        available_parents: Set[int],
        context: Dict,
    ) -> Tuple[FuncLibraryExpert, Dict]:
        self.call_count += 1
        meta = {"expert": FuncLibraryExpert.KEY, "var": var}
        return self._expert, meta
