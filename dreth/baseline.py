from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Naive comparison agent. RefitBaseline refits every visible variable from
# scratch every cycle with no sentinels, no certification, no compressions,
# no operation-role filtering. Pure brute force.
#
# Purpose: cost ratio baseline. After a run, compare ChainedAgent's
# total_interventions to RefitBaseline's total_interventions on the same
# world. The ratio is the concrete savings the certification machinery bought.
#
# Not part of the certification system. Does not use nethras, tareth/trass,
# or any form of accumulated belief. Its only job is to be a credible lower
# bound on what a non-certifying agent would spend.
# ─────────────────────────────────────────────────────────────────────────────

import random
from typing import List, Tuple

from .world import CausalWorld, HiddenMutation
from .ledger import DEFAULT_TOLERANCE
from .fit import fit_var

class RefitBaseline:
    """Strawman comparison agent: refits every visible variable from scratch
    every cycle. No nethras, no sentinels, no compressions, no operation_role
    classification, no scheduling. Just full hypothesis-space search per
    variable per cycle.

    Used to compute the cost ratio "how much work would a naive refit-everything
    agent do vs the framework?" Same world is replayed in parallel with this
    baseline; total_interventions is compared at end.

    Records (cycle, truth_rule_changed, baseline_thinks_something_changed) for
    optional comparison of detection accuracy."""

    def __init__(self, world: CausalWorld, rng: random.Random, intervention_budget: int = 30):
        """Construct empty baseline. intervention_budget mirrors the agent's
        per-audit budget so cost comparison is apples-to-apples."""
        self.world = world; self.rng = rng
        self.intervention_budget = intervention_budget
        self.last: List[Tuple[Tuple[int,...], str]] = []
        self.total_interventions = 0
        self.records: List[Tuple[int, bool, bool]] = []

    def initialize(self) -> None:
        """First-time fits for all currently-visible vars."""
        self.last = []
        for v in range(self.world.visible_count):
            parents, func, _, _ = fit_var(v, self.world, self.rng, self.intervention_budget, DEFAULT_TOLERANCE)
            self.last.append((parents, func))
            self.total_interventions += self.intervention_budget

    def run_cycle(self, mutation: HiddenMutation) -> None:
        """Refit EVERY visible var this cycle. If new vars were revealed
        since last cycle, fit them too. Compare new fits to last cycle's
        fits and record whether anything changed.

        Cost: visible_count × intervention_budget per cycle, every cycle."""
        new = []
        any_changed = False
        while len(self.last) < self.world.visible_count:
            v = len(self.last)
            parents, func, _, _ = fit_var(v, self.world, self.rng, self.intervention_budget, DEFAULT_TOLERANCE)
            self.last.append((parents, func))
            self.total_interventions += self.intervention_budget
        for v in range(self.world.visible_count):
            parents, func, _, _ = fit_var(v, self.world, self.rng, self.intervention_budget, DEFAULT_TOLERANCE)
            new.append((parents, func))
            self.total_interventions += self.intervention_budget
            if (parents, func) != self.last[v]:
                any_changed = True
        self.last = new
        self.records.append((mutation.cycle, mutation.rule_changed, any_changed))
