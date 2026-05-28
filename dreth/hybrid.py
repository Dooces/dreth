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
class source_edgeRanking:
    """Output from a source_edgeRanker for one target variable.
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
    source_edges: Tuple[int, ...]
    func: str
    score: float        # predicted score (not authoritative)
    confidence: float   # self-reported confidence [0, 1]; diagnostic only
    route_key: str      # which expert produced this


@dataclasses.dataclass
class RepairEvent:
    """Diagnostic record of one provider call during a repair cycle."""
    cycle: int
    var: int
    provider: str    # "residual_predictor" | "source_edge_ranker" | "probe_proposer" | "expert_router"
    call_count: int
    outcome: str     # "ok" | "stressed" | "proposal_issued" | "ranked" | "routed"


@dataclasses.dataclass
class source_edgeProposalDiagnostics:
    """Diagnostic-only quality counters for source_edgeRanker proposals."""
    calls: int = 0
    proposed_total: int = 0
    proposed_in_final_fit: int = 0
    proposed_excluded_by_route_cert: int = 0
    proposed_not_used: int = 0
    miss_chosen_source_edge_count: int = 0
    _rank_sum: int = 0
    _rank_count: int = 0
    rank_of_chosen_source_edge_max: int = 0
    history_ranker_calls: int = 0
    sensitivity_rescue_calls: int = 0
    sensitivity_rescue_interventions: int = 0
    rescue_candidates_added: int = 0
    rescue_chosen_source_edge_hits: int = 0
    chosen_source_edge_from_history: int = 0
    chosen_source_edge_from_rescue: int = 0

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
        chosen_source_edges: Tuple[int, ...],
        source_by_candidate: Optional[Dict[int, str]] = None,
    ) -> None:
        rank_index = {p: i for i, p in enumerate(ranked_post_route)}
        chosen = set(chosen_source_edges)
        for source_edge in chosen_source_edges:
            rank = rank_index.get(source_edge)
            if rank is None:
                self.miss_chosen_source_edge_count += 1
                continue
            self.proposed_in_final_fit += 1
            source = (source_by_candidate or {}).get(source_edge, "")
            if "history" in source:
                self.chosen_source_edge_from_history += 1
            if "rescue" in source:
                self.chosen_source_edge_from_rescue += 1
                self.rescue_chosen_source_edge_hits += 1
            self._rank_sum += rank
            self._rank_count += 1
            self.rank_of_chosen_source_edge_max = max(self.rank_of_chosen_source_edge_max, rank)
        self.proposed_not_used += sum(1 for p in ranked_post_route if p not in chosen)

    @property
    def rank_of_chosen_source_edge_mean(self) -> float:
        return self._rank_sum / self._rank_count if self._rank_count else 0.0

    @property
    def chosen_source_edge_hit_rate(self) -> float:
        denom = self.proposed_in_final_fit + self.miss_chosen_source_edge_count
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
        source_edges: Tuple[int, ...],
        func: str,
        source_edge_vals: List[float],
        actual: float,
        tolerance: float,
    ) -> ResidualPrediction:
        ...


@runtime_checkable
class source_edgeRanker(Protocol):
    """Ranks candidate source_edge variables for a target.

    CONTRACT: Must NOT certify any variable or exclude candidates via cert logic.
    Cert-based exclusion (route certs) is applied by ChainedAgent AFTER ranking.
    """

    def rank_source_edges(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> source_edgeRanking:
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
        available_source_edges: Set[int],
        budget: int,
    ) -> ProbeProposal:
        ...


@runtime_checkable
class Expert(Protocol):
    """Evaluates hypothesis quality for (var, source_edges, func).

    CONTRACT: Must NOT issue certs or alter ledger state.
    Confidence values are diagnostic only.
    """

    def evaluate(
        self,
        var: int,
        source_edges: Tuple[int, ...],
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
        available_source_edges: Set[int],
        context: Dict,
    ) -> Tuple["Expert", Dict]:   # (expert, route_metadata)
        ...


# ── Default symbolic implementations ─────────────────────────────────────────

class SymbolicResidualPredictor:
    """Default ResidualPredictor: reproduces the passive residual logic using
    currently authoritative source_edges + FUNC_LIBRARY.

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
        source_edges: Tuple[int, ...],
        func: str,
        source_edge_vals: List[float],
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
        predicted = _f(list(source_edge_vals))
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


class Sensitivitysource_edgeRanker:
    """Default source_edgeRanker: wraps the current per-target sensitivity screen.

    For each candidate var, perturbs it to 0.05 / 0.95 and measures |Δtarget|.
    Returns candidates ranked by movement without certifying anything.

    This is an adapter over ChainedAgent._screen_candidate_source_edges logic.
    Cert-based exclusion (route_certs / trass role) is applied by ChainedAgent
    AFTER ranking returns; this class never touches certs.
    """

    def __init__(self, world) -> None:
        self._world = world
        self.call_count: int = 0

    def rank_source_edges(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> source_edgeRanking:
        self.call_count += 1
        eligible = [c for c in candidates if c != target]
        scores: Dict[int, float] = {}
        for cand in eligible:
            lo = self._world.predict_var_under_intervention(target, cand, 0.05)
            hi = self._world.predict_var_under_intervention(target, cand, 0.95)
            scores[cand] = abs(hi - lo)
        ranked = tuple(sorted(scores, key=lambda c: scores[c], reverse=True)[:top_m])
        return source_edgeRanking(target=target, ranked=ranked, scores=scores)


class Historysource_edgeRanker:
    """source_edgeRanker using only agent-visible audit and residual history.

    It learns from prior Dreth fit results and residual co-stress observations
    supplied by ChainedAgent. It returns ranked candidates only; route-cert
    filtering and all certification remain outside the provider.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self._fit_source_edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
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

    def observe_fit_result(self, target: int, source_edges: Tuple[int, ...], margin: int = 0) -> None:
        weight = 2 if margin > 0 else 1
        for source_edge in source_edges:
            self._fit_source_edge_counts[(target, source_edge)] += weight

    def observe_route_exclusions(self, target: int, excluded: Tuple[int, ...]) -> None:
        for source_edge in excluded:
            self._route_exclusion_counts[(target, source_edge)] += 1

    def rank_source_edges(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> source_edgeRanking:
        self.call_count += 1
        eligible = [c for c in candidates if c != target]
        recent_counts: Dict[int, int] = defaultdict(int)
        for v in self._recent_stressed:
            recent_counts[v] += 1
        scores: Dict[int, float] = {}
        for cand in eligible:
            fit_score = 10.0 * self._fit_source_edge_counts[(target, cand)]
            co_stress = 2.0 * self._co_stress_counts[(target, cand)]
            recent = 0.25 * recent_counts[cand]
            route_penalty = 4.0 * self._route_exclusion_counts[(target, cand)]
            scores[cand] = fit_score + co_stress + recent - route_penalty
        ranked = tuple(
            sorted(eligible, key=lambda c: (-scores[c], c))[:top_m]
        )
        return source_edgeRanking(target=target, ranked=ranked, scores=scores)


class HistoryRescuesource_edgeRanker(Historysource_edgeRanker):
    """History ranker with a small sensitivity rescue pool.

    History proposes the first tranche. Sensitivity probes only a bounded pool
    built from visible agent-side history: recent stress, prior chosen source_edges,
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
        self._last_rescue_source_edge_by_target: Dict[int, Optional[int]] = {}

    def _history_scores(self, target: int, candidates: Set[int]) -> Dict[int, float]:
        recent_counts: Dict[int, int] = defaultdict(int)
        for v in self._recent_stressed:
            recent_counts[v] += 1
        scores: Dict[int, float] = {}
        for cand in candidates:
            if cand == target:
                continue
            fit_score = 10.0 * self._fit_source_edge_counts[(target, cand)]
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

    def rank_source_edges(
        self,
        target: int,
        candidates: Set[int],
        top_m: int,
    ) -> source_edgeRanking:
        self.call_count += 1
        self.history_ranker_calls += 1
        eligible = [c for c in sorted(candidates) if c != target]
        if top_m <= 0 or not eligible:
            return source_edgeRanking(
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
                ((c, self._fit_source_edge_counts[(target, c)]) for c in eligible
                 if c not in selected and self._fit_source_edge_counts[(target, c)] > 0),
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
        self._last_rescue_source_edge_by_target[target] = rescue_ranked[0] if rescue_ranked else None

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
        return source_edgeRanking(
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
        available_source_edges: Set[int],
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
        self._source_edge_rankings: Dict[int, Tuple[int, ...]] = {}
        self._cycle: Optional[int] = None
        self._stressed_this_cycle: Set[int] = set()
        self._recent_stressed = deque(maxlen=128)

    def observe_frontier_probes(self, var: int, probes: Tuple[Tuple[int, float], ...]) -> None:
        if probes:
            self._frontier_probes[var] = probes

    def observe_source_edge_ranking(self, var: int, ranked: Tuple[int, ...]) -> None:
        self._source_edge_rankings[var] = ranked

    def observe_source_edge_ranking_metadata(
        self,
        var: int,
        ranked: Tuple[int, ...],
        source_by_candidate: Dict[int, str],
    ) -> None:
        self.observe_source_edge_ranking(var, ranked)

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
        available_source_edges: Set[int],
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
        for i, source_edge in enumerate(self._source_edge_rankings.get(var, ())):
            if source_edge in available_source_edges:
                add((source_edge, values[i % len(values)]))
        for i, stressed_var in enumerate(reversed(self._recent_stressed)):
            if stressed_var in available_source_edges:
                add((stressed_var, values[(i + 1) % len(values)]))

        return ProbeProposal(var=var, probes=tuple(out))


class NeuralHistorysource_edgeRanker:
    """source_edgeRanker with online-learned per-variable embeddings over observation/intervention history.

    Learns from three sources supplied by ChainedAgent:
      - interventional probe results (observe_probe_results): direct causal evidence
        from Dreth audits — the spread of target actuals when each candidate was probed
      - completed Dreth fit results (observe_fit_result): confirmed source_edge structure
      - residual co-stress events (observe_residual_event): correlated instability

    Score(target, candidate) =
        w_embed  * W[target] · W[candidate]           (learned embedding similarity)
      + w_fit    * fit_count(target, candidate)        (confirmed-source_edge frequency)
      + w_iv     * spread(actuals | iv_var=candidate)  (interventional response range)
      + w_co     * co_stress_count(target, candidate)  (co-stress co-occurrence)

    Embeddings W are updated online from fit results: confirmed source_edges are pulled toward
    the target embedding; non-source_edges in the candidate set are lightly repelled.

    CONTRACT: no certs, no ledger mutations, no hidden-truth access.
    All inputs come from agent-visible surfaces only.
    """

    _W_EMBED = 5.0
    _W_FIT   = 10.0
    _W_IV    = 8.0
    _W_CO    = 2.0
    _NEG_FACTOR = 0.05   # repulsion weight for non-source_edges (kept small to avoid thrash)

    def __init__(self, n_vars: int, embed_dim: int = 16, lr: float = 0.05) -> None:
        import numpy as _np
        self._np = _np
        self.n_vars = n_vars
        self.embed_dim = embed_dim
        self.lr = lr
        rng = _np.random.default_rng(seed=0)
        self.W = rng.normal(0.0, 1.0 / embed_dim ** 0.5, (n_vars, embed_dim)).astype(_np.float32)
        norms = _np.linalg.norm(self.W, axis=1, keepdims=True)
        self.W /= _np.where(norms > 1e-9, norms, 1.0)
        # Fit counts: (target, candidate) -> weighted count of confirmed source_edge observations
        self._fit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        # Probe actuals: (target, iv_var) -> deque of observed target values when iv_var was probed
        self._probe_actuals: Dict[Tuple[int, int], "deque[float]"] = {}
        # Co-stress: (target, candidate) -> count of co-stressed cycles
        self._co_stress: Dict[Tuple[int, int], int] = defaultdict(int)
        self._stressed_this_cycle: Set[int] = set()
        self._current_cycle: Optional[int] = None
        self._recent_stressed: "deque[int]" = deque(maxlen=128)
        # Diagnostics
        self.call_count: int = 0
        self.fit_observations: int = 0
        self.probe_observations: int = 0

    def observe_fit_result(self, target: int, source_edges: Tuple[int, ...], margin: int = 0) -> None:
        """Update from a completed Dreth fit (called by ChainedAgent after _install_var)."""
        self.fit_observations += 1
        weight = max(1, margin)
        for source_edge in source_edges:
            self._fit_counts[(target, source_edge)] += weight
        if not source_edges or target >= self.n_vars:
            return
        np = self._np
        t_emb = self.W[target].copy()
        for source_edge in source_edges:
            if source_edge >= self.n_vars:
                continue
            self.W[source_edge] += self.lr * (t_emb - self.W[source_edge])
            n = float(np.linalg.norm(self.W[source_edge]))
            if n > 1e-9:
                self.W[source_edge] /= n

    def observe_probe_results(
        self,
        target: int,
        probes: Tuple[Tuple[int, float], ...],
        actuals: Tuple[float, ...],
    ) -> None:
        """Update from probe responses in a Dreth audit.

        Each (iv_var, iv_val) probe with its corresponding observed target value
        provides direct interventional evidence about causal influence.
        Called by ChainedAgent after each full audit, before _install_var returns.
        """
        if not probes or not actuals:
            return
        self.probe_observations += 1
        for (iv_var, _iv_val), actual in zip(probes, actuals):
            if iv_var == target or iv_var >= self.n_vars:
                continue
            key = (target, iv_var)
            if key not in self._probe_actuals:
                self._probe_actuals[key] = deque(maxlen=64)
            self._probe_actuals[key].append(actual)

    def observe_residual_event(self, var: int, cycle: int, stressed: bool) -> None:
        """Track co-stress patterns between variables."""
        if self._current_cycle != cycle:
            self._current_cycle = cycle
            self._stressed_this_cycle = set()
        if not stressed:
            return
        for other in self._stressed_this_cycle:
            self._co_stress[(var, other)] += 1
            self._co_stress[(other, var)] += 1
        self._stressed_this_cycle.add(var)
        self._recent_stressed.append(var)

    def observe_route_exclusions(self, target: int, excluded: Tuple[int, ...]) -> None:
        pass  # route exclusion belongs to Dreth; no action here

    def rank_source_edges(self, target: int, candidates: Set[int], top_m: int) -> source_edgeRanking:
        self.call_count += 1
        eligible = [c for c in candidates if c != target and c < self.n_vars]
        if not eligible or top_m <= 0:
            return source_edgeRanking(
                target=target,
                ranked=(),
                scores={},
                source_by_candidate={},
                diagnostics={"neural_history_ranker_calls": 1},
            )
        np = self._np
        t_emb = self.W[target] if target < self.n_vars else np.zeros(self.embed_dim, dtype=np.float32)
        scores: Dict[int, float] = {}
        for cand in eligible:
            embed_sim = float(np.dot(t_emb, self.W[cand]))
            fit_freq = float(self._fit_counts.get((target, cand), 0))
            iv_hist = self._probe_actuals.get((target, cand))
            if iv_hist and len(iv_hist) >= 2:
                arr = list(iv_hist)
                iv_spread = float(max(arr) - min(arr))
            elif iv_hist:
                iv_spread = float(iv_hist[0])
            else:
                iv_spread = 0.0
            co = float(self._co_stress.get((target, cand), 0))
            scores[cand] = (
                self._W_EMBED * embed_sim
                + self._W_FIT  * fit_freq
                + self._W_IV   * iv_spread
                + self._W_CO   * co
            )
        ranked = tuple(sorted(eligible, key=lambda c: (-scores[c], c))[:top_m])
        return source_edgeRanking(
            target=target,
            ranked=ranked,
            scores=scores,
            source_by_candidate={c: "neural_history" for c in ranked},
            diagnostics={"neural_history_ranker_calls": 1},
        )


class HistoryRescueProbeProposer(HistoryProbeProposer):
    """History probe proposer that appends one probe from a rescue source_edge."""

    def __init__(self, max_probes: int = 4) -> None:
        super().__init__(max_probes=max_probes)
        self._rescue_source_edge_by_target: Dict[int, int] = {}

    def observe_source_edge_ranking_metadata(
        self,
        var: int,
        ranked: Tuple[int, ...],
        source_by_candidate: Dict[int, str],
    ) -> None:
        super().observe_source_edge_ranking_metadata(var, ranked, source_by_candidate)
        for cand in ranked:
            if "rescue" in source_by_candidate.get(cand, ""):
                self._rescue_source_edge_by_target[var] = cand
                break

    def propose_probes(
        self,
        var: int,
        available_source_edges: Set[int],
        budget: int,
    ) -> ProbeProposal:
        base = super().propose_probes(var, available_source_edges, budget)
        out = list(base.probes)
        rescue_source_edge = self._rescue_source_edge_by_target.get(var)
        if (
            rescue_source_edge is not None
            and rescue_source_edge in available_source_edges
            and len(out) < min(self.max_probes, max(0, budget))
        ):
            probe = (rescue_source_edge, 0.9)
            if probe not in out:
                out.append(probe)
        return ProbeProposal(var=var, probes=tuple(out))


class FuncLibraryExpert:
    """Default Expert: evaluates (source_edges, func) pairs using FUNC_LIBRARY.

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
        source_edges: Tuple[int, ...],
        func: str,
        context: Dict,
    ) -> ExpertPrediction:
        source_edge_vals = context.get("source_edge_vals", [])
        _f = self._func_lib.get(func)
        score = 0.0
        if _f is not None and source_edge_vals:
            try:
                score = float(_f(list(source_edge_vals)))
            except Exception:
                score = 0.0
        return ExpertPrediction(
            source_edges=source_edges, func=func,
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
        available_source_edges: Set[int],
        context: Dict,
    ) -> Tuple[FuncLibraryExpert, Dict]:
        self.call_count += 1
        meta = {"expert": FuncLibraryExpert.KEY, "var": var}
        return self._expert, meta
