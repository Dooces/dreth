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
#   Cert authority remains exclusively on the Dreth ledger/cert/sentinel path.
#   Provider confidence values are NEVER treated as cert authority.
#
# Stage 1: interface separation + default symbolic wrappers.
# Future stages plug in neural/MoE components by implementing the Protocols
# without touching the certification path.
# ─────────────────────────────────────────────────────────────────────────────

import dataclasses
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


@dataclasses.dataclass
class ProbeProposal:
    """Output from a ProbeProposer for one variable.
    Proposals only — does NOT certify hypotheses or update cert state.
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


# ── Protocols ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ResidualPredictor(Protocol):
    """Predicts whether a variable's current state is consistent with its
    certified hypothesis.

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
    certified parents + FUNC_LIBRARY.

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
