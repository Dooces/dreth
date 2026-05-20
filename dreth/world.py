from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The hidden causal world — the oracle the agent tests against.
#
# CausalWorld holds:
#   - a random DAG (parent sets per variable)
#   - one operator per variable from HIDDEN_FUNC_LIBRARY
#   - current continuous state
#   - mutation history (never visible to the agent directly)
#
# The agent's only access to ground truth is through interventions:
#   predict_under_intervention(var, val) — fix var=val, run causal update,
#                                          return resulting full state.
#   predict_var_under_intervention(var, iv_var, iv_val) — same but returns
#                                          only the target variable's output.
#
# The world is the substitution test oracle. Every certification in the
# framework — tareth/trass verdicts, sentinel checks, compression tests —
# is ultimately a sequence of calls into this file.
#
# HiddenMutation records are diagnostic only. The test harness compares them
# to agent actions for confusion-matrix analysis. The agent never reads them.
#
# Nothing in this file is vestigial. harm attribution (per-variable cost
# weight seeding) is active for cost-dispatch demonstrations.
# ─────────────────────────────────────────────────────────────────────────────

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .functions import State, FUNC_LIBRARY, HIDDEN_FUNC_LIBRARY

@dataclass(frozen=True)
class HiddenMutation:
    """Record of a world-state change. Visible to the test harness only — the
    agent never sees these directly, only their effects through interventions.
    Used for ground-truth comparison in the diagnostic confusion matrix.

    Fields:
      cycle:         when the mutation happened
      kind:          one of VALUE, EDGE, FUNC, NOVELTY, REVEAL
      description:   human-readable summary
      rule_changed:  True if structure (edges/funcs) changed; False for value perturbation
      affected_var:  the variable whose structure changed (-1 for value-only mutations)
    """
    cycle: int
    kind: str
    description: str
    rule_changed: bool
    affected_var: int


class CausalWorld:
    """The hidden ground-truth dynamical system the agent is trying to learn.

    Holds:
      - A random DAG (each var has 0-2 parents from lower indices)
      - One function name per variable from HIDDEN_FUNC_LIBRARY
      - Current state (one float per variable)
      - History of all mutations applied (hidden_log)

    Visibility: agent only sees the first `visible_count` variables. World
    always evaluates the full state for self-consistent dynamics, regardless
    of visibility. This separation lets us test incremental introduction
    without changing world dynamics.
    """

    def __init__(self, n_vars: int, rng: random.Random, noise_sigma: float = 0.02,
                 initial_visible: Optional[int] = None):
        """Construct random DAG, assign random functions per var, randomize
        initial state. Run 3 burn-in steps to settle. If initial_visible is
        None, agent sees everything immediately; otherwise agent starts with
        the first `initial_visible` vars (used for incremental schedule)."""
        self.n_vars = n_vars
        self.rng = rng
        self.noise_sigma = noise_sigma
        self.parents: List[List[int]] = self._random_dag()
        self.funcs: List[str] = []
        for i in range(n_vars):
            if self.parents[i]:
                self.funcs.append(rng.choice([k for k in FUNC_LIBRARY if k not in {"LOW", "HIGH"}]))
            else:
                self.funcs.append(rng.choice(["LOW", "HIGH"]))
        # ── harm attribution (Flavor A) ─────────────────────────────────
        # Each variable gets a static harm value drawn at world creation.
        # The agent doesn't see this directly; the test harness uses it to
        # initialize cost_weights, demonstrating that cost-weighted dispatch
        # routes attention away from inert variables and onto dangerous ones.
        # Distribution: ~40% inert (0), ~40% mild (0.3-0.7), ~20% high (1.5-3.0).
        # These ranges align with the agent's cost_low_threshold (0.5) and
        # cost_high_threshold (2.0) so dispatch buckets fire as designed.
        # Preserve the RNG stream so harm attribution does not alter dynamics.
        saved_rng_state = self.rng.getstate()
        self.harm: List[float] = []
        for i in range(n_vars):
            roll = self.rng.random()
            if roll < 0.4:
                self.harm.append(0.0)
            elif roll < 0.8:
                self.harm.append(0.3 + self.rng.random() * 0.4)
            else:
                self.harm.append(1.5 + self.rng.random() * 1.5)
        self.rng.setstate(saved_rng_state)
        self.state: State = tuple(rng.random() for _ in range(n_vars))
        self.hidden_log: List[HiddenMutation] = []
        self.visible_count: int = initial_visible if initial_visible is not None else n_vars
        for _ in range(3):
            self._step_world()

    @property
    def visible_state(self) -> State:
        """Agent-visible portion of state. First visible_count entries."""
        return self.state[:self.visible_count]

    def reveal_next_variable(self, cycle: int) -> Optional[HiddenMutation]:
        """Make one more variable visible to the agent. Returns the REVEAL
        mutation, or None if all variables already visible. Used by the
        incremental schedule to stagger variable introductions."""
        if self.visible_count >= self.n_vars:
            return None
        new_idx = self.visible_count
        self.visible_count += 1
        m = HiddenMutation(
            cycle=cycle, kind="REVEAL",
            description=f"REVEAL x{new_idx} (now {self.visible_count}/{self.n_vars} visible)",
            rule_changed=False, affected_var=new_idx,
        )
        self.hidden_log.append(m)
        return m

    def _random_dag(self) -> List[List[int]]:
        """Build a random DAG by giving each variable 0-2 parents drawn from
        lower-indexed variables. Var 0 always has 0 parents (root). Subsequent
        vars draw 0-min(2,i) parents from {0..i-1}. Ensures acyclic by
        construction."""
        parents = []
        for i in range(self.n_vars):
            n_par = 0 if i == 0 else self.rng.randint(0, min(2, i))
            par = sorted(self.rng.sample(range(i), n_par))
            parents.append(par)
        return parents

    def _step_world(self) -> None:
        """Advance state by one time step. Each variable's new value is
        func(parent values from current state) + Gaussian noise, clipped to
        [0,1]. Mutates self.state."""
        new = []
        for i in range(self.n_vars):
            par_vals = [self.state[p] for p in self.parents[i]]
            v = HIDDEN_FUNC_LIBRARY[self.funcs[i]](par_vals)
            v += self.rng.gauss(0, self.noise_sigma)
            v = max(0.0, min(1.0, v))
            new.append(v)
        self.state = tuple(new)

    def predict_under_intervention(self, var: int, val: float) -> State:
        """Return the next state if `var` were forced to `val`. Computes
        one world-step starting from current state with var=val, applying
        full DAG and adding fresh noise. Does NOT mutate self.state. The
        agent uses this as its only observation channel — every probe is
        an intervention call."""
        forced = tuple(val if i == var else self.state[i] for i in range(self.n_vars))
        new = []
        for i in range(self.n_vars):
            par_vals = [forced[p] for p in self.parents[i]]
            v = HIDDEN_FUNC_LIBRARY[self.funcs[i]](par_vals)
            v += self.rng.gauss(0, self.noise_sigma)
            v = max(0.0, min(1.0, v))
            new.append(v)
        return tuple(new)

    def predict_under_joint_intervention(self, forced: Mapping[int, float]) -> Tuple[float, ...]:
        """Return the next state with all vars in `forced` simultaneously set.
        Semantics identical to predict_under_intervention but for multiple
        forced vars at once. Used by the joint false-trass test (T4.2)."""
        forced_state = tuple(forced.get(i, self.state[i]) for i in range(self.n_vars))
        new = []
        for i in range(self.n_vars):
            par_vals = [forced_state[p] for p in self.parents[i]]
            v = HIDDEN_FUNC_LIBRARY[self.funcs[i]](par_vals)
            v += self.rng.gauss(0, self.noise_sigma)
            v = max(0.0, min(1.0, v))
            new.append(v)
        return tuple(new)

    def predict_var_under_intervention(self, target_var: int, iv_var: int, iv_val: float) -> float:
        """Compute ONLY the new value of `target_var` if `iv_var` were forced
        to `iv_val`. Skips the full state recomputation that
        predict_under_intervention does.

        Computes only O(|parents of target_var|) function/parent work instead
        of predict_under_intervention's O(n_vars) full-state recomputation.
        It still consumes the same number of noise draws as the full path so
        repeated calls preserve the world's RNG stream.

        Behaves identically to predict_under_intervention(iv_var, iv_val)[target_var]
        from a noise/randomness perspective: same single noise draw applied
        to the target's output. Other vars not computed.
        """
        if iv_var == target_var:
            par_vals = [self.state[p] for p in self.parents[target_var]]
        else:
            par_vals = [
                iv_val if p == iv_var else self.state[p]
                for p in self.parents[target_var]
            ]
        for _ in range(target_var):
            self.rng.gauss(0, self.noise_sigma)
        v = HIDDEN_FUNC_LIBRARY[self.funcs[target_var]](par_vals)
        v += self.rng.gauss(0, self.noise_sigma)
        for _ in range(target_var + 1, self.n_vars):
            self.rng.gauss(0, self.noise_sigma)
        v = max(0.0, min(1.0, v))
        return v

    def perturb_value(self, cycle: int) -> HiddenMutation:
        """Pick a random variable, replace its value with a fresh random.
        Doesn't change DAG structure or functions — just shifts state.
        Returns a VALUE mutation with rule_changed=False."""
        i = self.rng.randint(0, self.n_vars - 1)
        old = self.state[i]
        new = self.rng.random()
        self.state = tuple(new if j == i else self.state[j] for j in range(self.n_vars))
        m = HiddenMutation(cycle, "VALUE", f"VALUE x{i}: {old:.2f}→{new:.2f}", False, -1)
        self.hidden_log.append(m); return m

    def perturb_edge(self, cycle: int) -> HiddenMutation:
        """Pick a non-root variable, modify its parent set: either drop a
        current parent, swap one for a new candidate, or add a new parent.
        Adjusts the function if needed (e.g., LOW/HIGH only for no-parent
        case). Returns an EDGE mutation with rule_changed=True."""
        i = self.rng.randint(1, self.n_vars - 1)
        old_par = list(self.parents[i])
        candidates = list(range(i))
        if not candidates: return self.perturb_value(cycle)
        target = self.rng.choice(candidates)
        if target in old_par:
            new_par = sorted([p for p in old_par if p != target])
        else:
            if len(old_par) >= 2:
                drop = self.rng.choice(old_par)
                new_par = sorted([p for p in old_par if p != drop] + [target])
            else:
                new_par = sorted(old_par + [target])
        self.parents[i] = new_par
        if not new_par:
            self.funcs[i] = self.rng.choice(["LOW", "HIGH"])
        elif self.funcs[i] in {"LOW", "HIGH"}:
            self.funcs[i] = self.rng.choice(["MEAN", "MAX", "MIN", "DIFF"])
        m = HiddenMutation(cycle, "EDGE", f"EDGE x{i} parents: {old_par}→{new_par}", True, i)
        self.hidden_log.append(m); return m

    def perturb_func(self, cycle: int) -> HiddenMutation:
        """Pick a random variable, swap its function for a different one of
        appropriate type (LOW/HIGH for no-parent, others for has-parent).
        Returns a FUNC mutation with rule_changed=True."""
        i = self.rng.randint(0, self.n_vars - 1)
        old = self.funcs[i]
        if self.parents[i]:
            choices = [k for k in FUNC_LIBRARY if k != old and k not in {"LOW", "HIGH"}]
        else:
            choices = [k for k in ["LOW", "HIGH"] if k != old]
        new = self.rng.choice(choices)
        self.funcs[i] = new
        m = HiddenMutation(cycle, "FUNC", f"FUNC x{i}: {old}→{new}", True, i)
        self.hidden_log.append(m); return m

    def introduce_sin(self, cycle: int) -> HiddenMutation:
        """Replace some variable's function with SIN (out-of-library). Picks
        the first variable that has parents and isn't already SIN. Returns a
        NOVELTY mutation. Falls back to perturb_value if no eligible vars.
        Used to test vocabulary-novelty detection."""
        for i in range(self.n_vars):
            if self.funcs[i] != "SIN" and len(self.parents[i]) >= 1:
                old = self.funcs[i]
                self.funcs[i] = "SIN"
                m = HiddenMutation(cycle, "NOVELTY",
                                   f"NOVELTY x{i}: {old}→SIN (outside library)",
                                   True, i)
                self.hidden_log.append(m); return m
        return self.perturb_value(cycle)

    def perturb_func_var(self, cycle: int, var: int) -> HiddenMutation:
        """Swap the function of a specific variable. Like perturb_func but
        targets a named var rather than picking at random."""
        old = self.funcs[var]
        if self.parents[var]:
            choices = [k for k in FUNC_LIBRARY if k != old and k not in {"LOW", "HIGH"}]
        else:
            choices = [k for k in ["LOW", "HIGH"] if k != old]
        if not choices:
            return self.perturb_value(cycle)
        new = self.rng.choice(choices)
        self.funcs[var] = new
        m = HiddenMutation(cycle, "FUNC", f"FUNC x{var}: {old}→{new}", True, var)
        self.hidden_log.append(m); return m

    def perturb_by_schedule(self, cycle: int, schedule: str,
                            settle_cycles: int = 25,
                            rare_var: int = 0,
                            rare_prob: float = 0.02) -> HiddenMutation:
        """Apply one mutation per cycle according to a named test schedule.
        The schedule controls when structural changes happen vs ordinary
        value drift, providing reproducible test scenarios.

        Schedules:
          shaped: edges and func swaps at fixed early cycles (2,5,8,11,13);
                  pure value drift otherwise.
          periodic_shifts: shaped early changes plus a structural change
                           every 50 cycles after cycle 15.
          novelty: SIN injection at cycle 10; some early edges; value drift.
          incremental: reveal one variable per `settle_cycles` cycles; no
                       structural changes. Tests bootstrapping behavior.
          rare_catastrophe: value drift each cycle; with probability rare_prob,
                            mutate rare_var's function (structural change on a
                            specific variable). Mutations are permanent — no
                            reversion. settle_cycles suppresses rare mutations
                            for the first N cycles so certs can establish.
        """
        if schedule == "shaped":
            structural = {2:"edge", 5:"func", 8:"edge", 11:"func", 13:"edge"}
            kind = structural.get(cycle, "value")
        elif schedule == "periodic_shifts":
            structural = {2:"edge", 5:"func", 8:"edge", 11:"func", 13:"edge"}
            kind = structural.get(cycle, "value")
            if cycle > 15 and cycle % 50 == 0:
                kind = "edge" if (cycle // 50) % 2 == 0 else "func"
        elif schedule == "novelty":
            kind = "novelty" if cycle == 10 else ("edge" if cycle in {2,5} else "value")
        elif schedule == "incremental":
            if self.visible_count < self.n_vars and cycle % settle_cycles == 0:
                m = self.reveal_next_variable(cycle)
                if m is not None:
                    return m
            kind = "value"
        elif schedule == "rare_catastrophe":
            if cycle <= settle_cycles:
                kind = "value"
            elif self.rng.random() < rare_prob:
                return self.perturb_func_var(cycle, rare_var)
            else:
                kind = "value"
        else:
            kind = "value"
        return {"edge": self.perturb_edge, "func": self.perturb_func,
                "novelty": self.introduce_sin,
                "value": self.perturb_value}[kind](cycle)
