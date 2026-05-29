from __future__ import annotations

"""NethraExpression algebra, ActiveSlice compiler, recognition-collapse detection,
and expression-assist attribution.

This module provides the missing architectural layer between offline sleep expression
mining and the runtime search loop.

Contents:
  NethraExpression       — union/intersection/difference/gated/coactive algebra over nethras
  ActiveSlice            — compiled runtime product (filters, rank hints, probe hints, provenance)
  ExpressionAssistEvent  — attribution record for one expression-driven runtime influence
  RecognitionCollapseDetector — first-class subsystem for active coverage failure detection
  NethraExpressionIndex  — offline index with compile_active_slice() and ranking application
  mine_expressions_from_proposals() — mines overlap/subset/gated/coactivation from proposals

Design invariants:
  - Expressions do not inherit the strongest member use-right (anti-authority-laundering)
  - Mined expressions start at feature_only; runtime evidence upgrades toward ranking_hint
  - Hard_filter and block require explicit evidence events beyond positive outcome accumulation
  - Recognition collapse opens a regime-boundary candidate; does not itself authorize action
  - record/feature_only expressions annotate only; they cannot reorder or exclude
"""

import dataclasses
import hashlib
from collections import defaultdict, deque
from typing import Any, Optional

# ── Valid operation and use-right sets ────────────────────────────────────────

VALID_OPS: frozenset[str] = frozenset({
    "intersection", "union", "difference", "gated", "coactive"
})

VALID_USE_RIGHTS: frozenset[str] = frozenset({
    "record_only", "feature_only", "ranking_hint",
    "soft_filter", "hard_filter", "block",
})

_USE_RIGHT_RANK: dict[str, int] = {
    "record_only": 0, "feature_only": 1, "ranking_hint": 2,
    "soft_filter": 3, "hard_filter": 4, "block": 5,
}

# Mined expressions start here. Must earn stronger rights through outcome evidence.
_MINING_DEFAULT_USE_RIGHT = "feature_only"

# Positive outcomes needed before a coactive expression earns ranking_hint.
_COACTIVATION_RANKING_THRESHOLD = 5

# Recognition-collapse detection thresholds
_COLLAPSE_COVERAGE_THRESHOLD = 0.5      # certified fraction below which collapse fires
_COLLAPSE_SENTINEL_FAIL_RATE = 0.4      # sentinel fail fraction triggering collapse
_COLLAPSE_WINDOW = 20                   # rolling window in cycles


# ── NethraExpression ──────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class NethraExpression:
    """An algebraic expression over nethra operands.

    op:
      intersection — touched_vars = A.vars ∩ B.vars (shared structure)
      union        — touched_vars = A.vars ∪ B.vars (broad coverage)
      difference   — touched_vars = A.vars - B.vars (A blocked/contradicted by B)
      gated        — A is active only when gate nethra/condition is active
      coactive     — A and B repeatedly become useful together; stable basin candidate

    use_right starts at feature_only for mined expressions. Expressions never inherit
    the strongest operand use-right. They earn stronger rights through runtime evidence.

    evidence_cycles: positive outcome attributions accumulated at runtime.
    coactivation_count: cycles where all operands were simultaneously active.
    provenance: mining source tag (e.g. "mined_overlap", "mined_coactivation").
    """
    expr_id: str
    op: str                          # see VALID_OPS
    operands: tuple[str, ...]        # proposal_ids or nethra_ids
    gate: Optional[str]              # gate operand for "gated" op; None otherwise
    touched_vars: frozenset[int]     # vars covered by this expression
    use_right: str                   # see VALID_USE_RIGHTS
    evidence_cycles: int
    coactivation_count: int
    provenance: str
    first_seen_cycle: int
    last_seen_cycle: int

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op: {self.op!r}")
        if self.use_right not in VALID_USE_RIGHTS:
            raise ValueError(f"unknown use_right: {self.use_right!r}")

    def with_use_right(self, new_right: str) -> "NethraExpression":
        if new_right not in VALID_USE_RIGHTS:
            raise ValueError(f"unknown use_right: {new_right!r}")
        return dataclasses.replace(self, use_right=new_right)

    def upgraded_use_right(self) -> str:
        """Return the next use-right level above current, capped at soft_filter for auto-upgrade.

        Hard_filter and block require explicit grant events, not accumulated outcomes.
        """
        rank = _USE_RIGHT_RANK.get(self.use_right, 0)
        cap = _USE_RIGHT_RANK["soft_filter"]
        if rank < cap:
            for right, r in _USE_RIGHT_RANK.items():
                if r == rank + 1:
                    return right
        return self.use_right


# ── ActiveSlice ───────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ActiveSlice:
    """Compiled runtime product of active nethra expressions.

    The runtime receives this bounded slice instead of the full historical structure
    graph each cycle. It contains only what current evidence authorizes.

    hard_filters: vars to exclude from candidate enumeration (hard_filter expressions)
    soft_filters: vars to deprioritize but not exclude (soft_filter expressions)
    rank_hints:   ordered vars suggested as high-priority candidates
    probe_hints:  (var, value) pairs suggested as informative interventions
    blockers:     vars whose derivation should be blocked in scope (block expressions)
    provenance:   (expr_id:use_right) strings for attribution
    expression_ids: all expressions contributing to this slice
    compiled_at_cycle: cycle when this slice was compiled
    """
    hard_filters: frozenset[int]
    soft_filters: frozenset[int]
    rank_hints: tuple[int, ...]
    probe_hints: tuple[tuple[int, float], ...]
    blockers: frozenset[int]
    provenance: tuple[str, ...]
    expression_ids: tuple[str, ...]
    compiled_at_cycle: int

    @classmethod
    def empty(cls, cycle: int = 0) -> "ActiveSlice":
        return cls(
            hard_filters=frozenset(),
            soft_filters=frozenset(),
            rank_hints=(),
            probe_hints=(),
            blockers=frozenset(),
            provenance=(),
            expression_ids=(),
            compiled_at_cycle=cycle,
        )

    def is_empty(self) -> bool:
        return (
            not self.hard_filters and not self.soft_filters
            and not self.rank_hints and not self.probe_hints
            and not self.blockers
        )


# ── ExpressionAssistEvent ─────────────────────────────────────────────────────

@dataclasses.dataclass
class ExpressionAssistEvent:
    """Attribution record for one expression-driven runtime influence.

    Shows which expression changed which aspect of search (candidates/probes/filters)
    and whether the subsequent cycle outcome improved. This is required by the design:
    any runtime use of nethra expressions must show which expression changed
    ordering/probes/filters and whether that improved outcomes.

    outcome_score, outcome_improved, outcome_margin: filled retroactively after
    the audit result is known. None until record_assist_outcome() is called.
    """
    cycle: int
    var: int
    expr_id: str
    op: str
    use_right: str
    changed_ordering: bool
    changed_probes: bool
    changed_filters: bool
    n_candidates_before: int
    n_candidates_after: int
    outcome_score: Optional[int] = None
    outcome_improved: Optional[bool] = None
    outcome_margin: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ── RecognitionCollapseDetector ───────────────────────────────────────────────

class RecognitionCollapseDetector:
    """First-class subsystem for detecting when active nethra coverage has failed.

    The agent needs explicit metrics for when active nethra coverage has collapsed.
    Without this subsystem, the agent has no way to know that its current nethra set
    no longer covers the context: predictions degrade silently, sentinels keep firing
    without a coherent recovery signal.

    Signals tracked per cycle:
      coverage          — fraction of visible vars with certified/non-untested roles
      sentinel_fail_rate — ratio of sentinel failures to checks in this cycle
      prediction_error   — mean fit score delta vs prior cycle (higher = worse fit)
      rank_lift          — 1.0 if certified top candidate held; 0.0 if displaced

    Collapse condition: any of
      - mean_coverage < _COLLAPSE_COVERAGE_THRESHOLD over the rolling window
      - mean_sentinel_fail_rate > _COLLAPSE_SENTINEL_FAIL_RATE over the rolling window

    Collapse does not authorize action. It opens a regime-boundary candidate and
    signals that the active nethra set may need to be rebuilt from local overlap.
    """

    def __init__(self, window: int = _COLLAPSE_WINDOW) -> None:
        self._window = int(window)
        self._coverage: deque[float] = deque(maxlen=self._window)
        self._sentinel_fail_rate: deque[float] = deque(maxlen=self._window)
        self._prediction_error: deque[float] = deque(maxlen=self._window)
        self._rank_lift: deque[float] = deque(maxlen=self._window)
        self._collapse_events: list[dict[str, Any]] = []
        self.total_collapses: int = 0
        self._currently_collapsed: bool = False
        self._collapse_start_cycle: Optional[int] = None

    def record_cycle(
        self,
        *,
        coverage: float,
        sentinel_fail_rate: float,
        prediction_error: float = 0.0,
        rank_lift: float = 1.0,
        cycle: int,
    ) -> bool:
        """Record one cycle's signals. Returns True if currently in a collapse state."""
        self._coverage.append(float(coverage))
        self._sentinel_fail_rate.append(float(sentinel_fail_rate))
        self._prediction_error.append(float(prediction_error))
        self._rank_lift.append(float(rank_lift))

        # Require at least a quarter-window of history to avoid false early alarms.
        if len(self._coverage) < max(3, self._window // 4):
            return False

        mean_coverage = sum(self._coverage) / len(self._coverage)
        mean_fail_rate = sum(self._sentinel_fail_rate) / len(self._sentinel_fail_rate)
        is_collapsed = (
            mean_coverage < _COLLAPSE_COVERAGE_THRESHOLD
            or mean_fail_rate > _COLLAPSE_SENTINEL_FAIL_RATE
        )

        was_collapsed = self._currently_collapsed
        if is_collapsed and not was_collapsed:
            self.total_collapses += 1
            self._currently_collapsed = True
            self._collapse_start_cycle = cycle
            self._collapse_events.append({
                "cycle": cycle,
                "mean_coverage": round(mean_coverage, 4),
                "mean_sentinel_fail_rate": round(mean_fail_rate, 4),
                "mean_prediction_error": round(
                    sum(self._prediction_error) / max(1, len(self._prediction_error)), 4
                ),
                "reason": (
                    "coverage_below_threshold"
                    if mean_coverage < _COLLAPSE_COVERAGE_THRESHOLD
                    else "sentinel_fail_rate_above_threshold"
                ),
            })
        elif not is_collapsed and was_collapsed:
            self._currently_collapsed = False
            self._collapse_start_cycle = None

        return is_collapsed

    def is_collapsed(self) -> bool:
        return self._currently_collapsed

    def current_coverage(self) -> float:
        if not self._coverage:
            return 1.0
        return sum(self._coverage) / len(self._coverage)

    def current_sentinel_fail_rate(self) -> float:
        if not self._sentinel_fail_rate:
            return 0.0
        return sum(self._sentinel_fail_rate) / len(self._sentinel_fail_rate)

    def summary(self) -> dict[str, Any]:
        return {
            "recognition_collapse_total": self.total_collapses,
            "recognition_collapse_currently_active": self._currently_collapsed,
            "recognition_collapse_start_cycle": self._collapse_start_cycle,
            "recognition_collapse_mean_coverage": round(self.current_coverage(), 4),
            "recognition_collapse_mean_sentinel_fail_rate": round(
                self.current_sentinel_fail_rate(), 4
            ),
            "recognition_collapse_events": list(self._collapse_events[-10:]),
            "recognition_collapse_window": self._window,
        }


# ── Expression mining ─────────────────────────────────────────────────────────

def _expr_id(op: str, operands: tuple[str, ...], gate: Optional[str] = None) -> str:
    """Deterministic expression id from its defining structure."""
    key = f"{op}:{'|'.join(sorted(operands))}"
    if gate:
        key += f":gate={gate}"
    return "expr:" + hashlib.sha1(key.encode()).hexdigest()[:16]


def mine_expressions_from_proposals(
    proposals: list[Any],  # list[ScaffoldMemoryProposal] — avoid circular import
    *,
    cycle: int = 0,
    max_expressions: int = 500,
) -> list[NethraExpression]:
    """Mine NethraExpression objects from ScaffoldMemoryProposal objects.

    Produces four expression types from flat proposal groups:

    intersection — pairs with >= 2 shared vars (shared structure bridge).
    union        — pairs with >= 3 total vars (broad coverage; starts record_only).
    difference   — strict subset pairs: unique structure of the superset proposal.
    coactive     — pairs sharing >= 1 context key (co-activated in same context).
    gated        — proposal A whose context is a subset of B's (A may be gated by B).

    All mined expressions start at feature_only (union at record_only). They must
    earn stronger use-rights through runtime evidence, not by inheriting from operands.
    """
    expressions: list[NethraExpression] = []
    if not proposals:
        return expressions

    proposal_vars: list[frozenset[int]] = [frozenset(int(v) for v in p.vars) for p in proposals]
    proposal_ctxs: list[frozenset[str]] = [frozenset(p.contexts) for p in proposals]
    proposal_ids: list[str] = [p.proposal_id for p in proposals]

    seen_ids: set[str] = set()

    def _add(expr: NethraExpression) -> None:
        if expr.expr_id not in seen_ids and len(expressions) < max_expressions:
            seen_ids.add(expr.expr_id)
            expressions.append(expr)

    n = len(proposals)
    for i in range(n):
        vars_a = proposal_vars[i]
        pid_a = proposal_ids[i]
        ctxs_a = proposal_ctxs[i]
        if not vars_a:
            continue

        for j in range(i + 1, n):
            vars_b = proposal_vars[j]
            pid_b = proposal_ids[j]
            ctxs_b = proposal_ctxs[j]
            if not vars_b:
                continue

            shared = vars_a & vars_b

            # Intersection: shared structure bridge
            if len(shared) >= 2:
                _add(NethraExpression(
                    expr_id=_expr_id("intersection", (pid_a, pid_b)),
                    op="intersection",
                    operands=(pid_a, pid_b),
                    gate=None,
                    touched_vars=shared,
                    use_right=_MINING_DEFAULT_USE_RIGHT,
                    evidence_cycles=0,
                    coactivation_count=0,
                    provenance="mined_overlap",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))

            # Union: broad coverage — starts conservative at record_only
            union_vars = vars_a | vars_b
            if len(union_vars) >= 3:
                _add(NethraExpression(
                    expr_id=_expr_id("union", (pid_a, pid_b)),
                    op="union",
                    operands=(pid_a, pid_b),
                    gate=None,
                    touched_vars=union_vars,
                    use_right="record_only",
                    evidence_cycles=0,
                    coactivation_count=0,
                    provenance="mined_union",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))

            # Difference (subset): unique structure of the larger proposal
            if vars_a < vars_b:
                _add(NethraExpression(
                    expr_id=_expr_id("difference", (pid_b, pid_a)),
                    op="difference",
                    operands=(pid_b, pid_a),
                    gate=None,
                    touched_vars=vars_b - vars_a,
                    use_right=_MINING_DEFAULT_USE_RIGHT,
                    evidence_cycles=0,
                    coactivation_count=0,
                    provenance="mined_subset",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))
            elif vars_b < vars_a:
                _add(NethraExpression(
                    expr_id=_expr_id("difference", (pid_a, pid_b)),
                    op="difference",
                    operands=(pid_a, pid_b),
                    gate=None,
                    touched_vars=vars_a - vars_b,
                    use_right=_MINING_DEFAULT_USE_RIGHT,
                    evidence_cycles=0,
                    coactivation_count=0,
                    provenance="mined_subset",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))

            # Coactivation: proposals sharing context keys → repeated co-presence
            shared_ctxs = ctxs_a & ctxs_b
            if shared_ctxs:
                _add(NethraExpression(
                    expr_id=_expr_id("coactive", (pid_a, pid_b)),
                    op="coactive",
                    operands=(pid_a, pid_b),
                    gate=None,
                    touched_vars=vars_a | vars_b,
                    use_right=_MINING_DEFAULT_USE_RIGHT,
                    evidence_cycles=0,
                    coactivation_count=1,
                    provenance="mined_coactivation",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))

            # Gated: A may be active only when B's context condition holds
            # Detected when B has context strings that A does not share.
            if ctxs_b and ctxs_a and (ctxs_b - ctxs_a) and len(vars_a) >= 2:
                _add(NethraExpression(
                    expr_id=_expr_id("gated", (pid_a,), gate=pid_b),
                    op="gated",
                    operands=(pid_a,),
                    gate=pid_b,
                    touched_vars=vars_a,
                    use_right="record_only",  # gated starts conservative
                    evidence_cycles=0,
                    coactivation_count=0,
                    provenance="mined_gated",
                    first_seen_cycle=cycle,
                    last_seen_cycle=cycle,
                ))

    return expressions


# ── NethraExpressionIndex ─────────────────────────────────────────────────────

class NethraExpressionIndex:
    """Offline index of mined nethra expressions with a runtime ActiveSlice compiler.

    This is the bridge between offline sleep mining and the runtime search loop.

    Responsibilities:
      1. Store mined NethraExpression objects indexed by operand and var coverage.
      2. Compile ActiveSlice from active nethra/proposal ids and stored expressions.
      3. Apply ranking hints from the active slice to candidate lists.
      4. Record ExpressionAssistEvent attribution for each runtime influence.
      5. Upgrade expression use-rights when positive outcomes accumulate.

    Use-right application rules (what each level contributes to the slice):
      record_only / feature_only → provenance annotation only; no runtime effect
      ranking_hint   → contributes to rank_hints (reorders; never excludes)
      soft_filter    → contributes to soft_filters (deprioritizes; never excludes)
      hard_filter    → contributes to hard_filters (excludes; only with earned evidence)
      block          → contributes to blockers (prevents derivation in scope)
    """

    def __init__(self) -> None:
        self.expressions: dict[str, NethraExpression] = {}
        self._by_operand: dict[str, set[str]] = defaultdict(set)
        self._by_var: dict[int, set[str]] = defaultdict(set)
        self._by_op: dict[str, list[str]] = defaultdict(list)
        self._attribution: list[ExpressionAssistEvent] = []
        self._mined_count: int = 0
        self._last_slice: Optional[ActiveSlice] = None

        # Outcome counters for reporting
        self.ranking_applications: int = 0
        self.filter_applications: int = 0
        self.probe_hint_applications: int = 0
        self.assist_events: int = 0
        self.positive_outcomes: int = 0
        self.negative_outcomes: int = 0

    # ── Loading ────────────────────────────────────────────────────────────────

    def add_expression(self, expr: NethraExpression) -> None:
        """Add or merge an expression into the index."""
        existing = self.expressions.get(expr.expr_id)
        if existing is not None:
            merged = dataclasses.replace(
                existing,
                evidence_cycles=existing.evidence_cycles + expr.evidence_cycles,
                coactivation_count=existing.coactivation_count + expr.coactivation_count,
                last_seen_cycle=max(existing.last_seen_cycle, expr.last_seen_cycle),
            )
            self.expressions[expr.expr_id] = merged
        else:
            self.expressions[expr.expr_id] = expr
            self._mined_count += 1
            for op_id in expr.operands:
                self._by_operand[op_id].add(expr.expr_id)
            if expr.gate:
                self._by_operand[expr.gate].add(expr.expr_id)
            for v in expr.touched_vars:
                self._by_var[int(v)].add(expr.expr_id)
            self._by_op[expr.op].append(expr.expr_id)

    def load_from_proposals(self, proposals: list[Any], *, cycle: int = 0) -> int:
        """Mine expressions from scaffold proposals and load them. Returns count added."""
        exprs = mine_expressions_from_proposals(proposals, cycle=cycle)
        before = len(self.expressions)
        for expr in exprs:
            self.add_expression(expr)
        return len(self.expressions) - before

    # ── Compilation ───────────────────────────────────────────────────────────

    def compile_active_slice(
        self,
        active_ids: set[str],
        cycle: int,
    ) -> ActiveSlice:
        """Compile an ActiveSlice from currently-active nethra/proposal ids.

        Only expressions where operands overlap with active_ids contribute.
        Gated expressions additionally require the gate operand in active_ids.
        Coactive expressions require all operands in active_ids.

        Use-right gates what each expression contributes to the slice.
        record_only and feature_only contribute to provenance only.
        """
        hard_filters: set[int] = set()
        soft_filters: set[int] = set()
        rank_hints: list[int] = []
        probe_hints: list[tuple[int, float]] = []
        blockers: set[int] = set()
        provenance: list[str] = []
        contributing_ids: list[str] = []

        candidate_eids: set[str] = set()
        for nid in active_ids:
            candidate_eids.update(self._by_operand.get(nid, ()))

        for eid in sorted(candidate_eids):
            expr = self.expressions.get(eid)
            if expr is None:
                continue

            # Gated: gate operand must be in active set
            if expr.op == "gated" and expr.gate and expr.gate not in active_ids:
                continue
            # Coactive: all operands must be active for ranking_hint+
            if expr.op == "coactive" and expr.use_right != "record_only":
                if not all(op in active_ids for op in expr.operands):
                    continue

            use_right = expr.use_right
            provenance.append(f"{eid[:20]}:{use_right}")
            contributing_ids.append(eid)

            if use_right in {"record_only", "feature_only"}:
                continue

            if use_right == "ranking_hint":
                for v in sorted(expr.touched_vars):
                    if v not in rank_hints:
                        rank_hints.append(v)
            elif use_right == "soft_filter":
                soft_filters.update(expr.touched_vars)
                self.filter_applications += 1
            elif use_right == "hard_filter":
                hard_filters.update(expr.touched_vars)
                self.filter_applications += 1
            elif use_right == "block":
                blockers.update(expr.touched_vars)

        self._last_slice = ActiveSlice(
            hard_filters=frozenset(hard_filters),
            soft_filters=frozenset(soft_filters),
            rank_hints=tuple(rank_hints),
            probe_hints=tuple(probe_hints),
            blockers=frozenset(blockers),
            provenance=tuple(provenance),
            expression_ids=tuple(contributing_ids),
            compiled_at_cycle=cycle,
        )
        return self._last_slice

    # ── Runtime application ────────────────────────────────────────────────────

    def apply_ranking_hints(
        self,
        candidates: list[tuple[Any, ...]],
        active_slice: ActiveSlice,
        *,
        var: int,
        cycle: int,
    ) -> tuple[list[tuple[Any, ...]], Optional[ExpressionAssistEvent]]:
        """Reorder candidates using rank_hints from the active slice.

        Candidates whose source_edges overlap with hinted vars move to front.
        All candidates are preserved — this never excludes. Fallback is unconditional.

        Returns (reordered_candidates, assist_event_or_None). The assist_event
        records which expression triggered the reorder for outcome attribution.
        """
        if not active_slice.rank_hints or not candidates:
            return candidates, None

        hint_set = set(active_slice.rank_hints)
        front: list[tuple[Any, ...]] = []
        back: list[tuple[Any, ...]] = []
        for cand in candidates:
            # Candidates are (source_edges_tuple, func_str, ...) by convention
            source_edges = cand[0] if cand and isinstance(cand[0], (tuple, list, frozenset)) else ()
            if hint_set & set(source_edges):
                front.append(cand)
            else:
                back.append(cand)

        reordered = front + back
        if reordered == candidates or not active_slice.expression_ids:
            return candidates, None

        first_eid = active_slice.expression_ids[0]
        first_expr = self.expressions.get(first_eid)
        evt = ExpressionAssistEvent(
            cycle=cycle,
            var=var,
            expr_id=first_eid,
            op=first_expr.op if first_expr else "unknown",
            use_right="ranking_hint",
            changed_ordering=True,
            changed_probes=False,
            changed_filters=False,
            n_candidates_before=len(candidates),
            n_candidates_after=len(reordered),
        )
        if len(self._attribution) < 1000:
            self._attribution.append(evt)
        self.assist_events += 1
        self.ranking_applications += 1
        return reordered, evt

    # ── Attribution ───────────────────────────────────────────────────────────

    def record_assist_outcome(
        self,
        event: ExpressionAssistEvent,
        *,
        score_after: int,
        score_before: int,
    ) -> None:
        """Fill in the outcome for a prior ExpressionAssistEvent.

        If score improved, accumulate evidence on the expression and potentially
        upgrade its use_right. Negative outcomes do not downgrade use_right but
        are counted so drift can be detected.
        """
        improved = score_after > score_before
        margin = score_after - score_before

        for evt in reversed(self._attribution):
            if (evt.expr_id == event.expr_id
                    and evt.cycle == event.cycle
                    and evt.var == event.var):
                evt.outcome_score = score_after
                evt.outcome_improved = improved
                evt.outcome_margin = margin
                break

        expr = self.expressions.get(event.expr_id)
        if expr is None:
            return

        if improved:
            self.positive_outcomes += 1
            new_evidence = expr.evidence_cycles + 1
            new_expr = dataclasses.replace(expr, evidence_cycles=new_evidence)
            # Auto-upgrade feature_only → ranking_hint after threshold
            if (new_evidence >= _COACTIVATION_RANKING_THRESHOLD
                    and expr.use_right == "feature_only"):
                new_expr = dataclasses.replace(new_expr, use_right="ranking_hint")
            self.expressions[event.expr_id] = new_expr
        else:
            self.negative_outcomes += 1

    # ── Queries ────────────────────────────────────────────────────────────────

    def expressions_for_var(self, var: int) -> tuple[NethraExpression, ...]:
        eids = self._by_var.get(int(var), set())
        return tuple(
            self.expressions[eid] for eid in sorted(eids) if eid in self.expressions
        )

    def expressions_by_op(self, op: str) -> tuple[NethraExpression, ...]:
        eids = self._by_op.get(op, [])
        return tuple(self.expressions[eid] for eid in eids if eid in self.expressions)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        op_counts: dict[str, int] = defaultdict(int)
        use_right_counts: dict[str, int] = defaultdict(int)
        for expr in self.expressions.values():
            op_counts[expr.op] += 1
            use_right_counts[expr.use_right] += 1
        return {
            "expression_index_total": len(self.expressions),
            "expression_index_mined_total": self._mined_count,
            "expression_index_by_op": dict(op_counts),
            "expression_index_by_use_right": dict(use_right_counts),
            "expression_assist_events": self.assist_events,
            "expression_assist_positive_outcomes": self.positive_outcomes,
            "expression_assist_negative_outcomes": self.negative_outcomes,
            "expression_ranking_applications": self.ranking_applications,
            "expression_filter_applications": self.filter_applications,
            "expression_probe_hint_applications": self.probe_hint_applications,
            "expression_attribution_records": len(self._attribution),
        }

    def export_attribution(self, limit: int = 50) -> list[dict[str, Any]]:
        return [evt.to_dict() for evt in self._attribution[-limit:]]

    def export_expressions(self, limit: int = 100) -> list[dict[str, Any]]:
        out = []
        for expr in list(self.expressions.values())[:limit]:
            d = dataclasses.asdict(expr)
            d["touched_vars"] = sorted(d["touched_vars"])
            out.append(d)
        return out
