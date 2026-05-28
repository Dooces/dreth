from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# The hidden causal world — the oracle the agent tests against.
#
# CausalWorld holds:
#   - a random DAG (source_edge sets per variable)
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

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
      - A random DAG (each var has 0-2 source_edges from lower indices)
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
        self.source_edges: List[List[int]] = self._random_dag()
        self.funcs: List[str] = []
        for i in range(n_vars):
            if self.source_edges[i]:
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
        self._blind_challenge_enabled: bool = False
        self._blind_challenge_manifest: Optional[Dict[str, Any]] = None
        self._blind_specs: List[Dict[str, Any]] = []
        self._blind_side_effects: List[Dict[str, Any]] = []
        self._blind_latents: List[Dict[str, Any]] = []
        self._blind_history: List[State] = []
        self._blind_cycle: int = 0
        self._blind_max_lag: int = 1
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
        """Build a random DAG by giving each variable 0-2 source_edges drawn from
        lower-indexed variables. Var 0 always has 0 source_edges (root). Subsequent
        vars draw 0-min(2,i) source_edges from {0..i-1}. Ensures acyclic by
        construction."""
        source_edges = []
        for i in range(self.n_vars):
            n_par = 0 if i == 0 else self.rng.randint(0, min(2, i))
            par = sorted(self.rng.sample(range(i), n_par))
            source_edges.append(par)
        return source_edges

    def _step_world(self) -> None:
        """Advance state by one time step. Each variable's new value is
        func(source_edge values from current state) + Gaussian noise, clipped to
        [0,1]. Mutates self.state."""
        if self._blind_challenge_enabled:
            self._step_blind_challenge_world()
            return
        new = []
        for i in range(self.n_vars):
            par_vals = [self.state[p] for p in self.source_edges[i]]
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
        if self._blind_challenge_enabled:
            return self._predict_blind_challenge({var: val})
        forced = tuple(val if i == var else self.state[i] for i in range(self.n_vars))
        new = []
        for i in range(self.n_vars):
            par_vals = [forced[p] for p in self.source_edges[i]]
            v = HIDDEN_FUNC_LIBRARY[self.funcs[i]](par_vals)
            v += self.rng.gauss(0, self.noise_sigma)
            v = max(0.0, min(1.0, v))
            new.append(v)
        return tuple(new)

    def predict_under_joint_intervention(self, forced: Mapping[int, float]) -> Tuple[float, ...]:
        """Return the next state with all vars in `forced` simultaneously set.
        Semantics identical to predict_under_intervention but for multiple
        forced vars at once. Used by the joint false-trass test (T4.2)."""
        if self._blind_challenge_enabled:
            return self._predict_blind_challenge(forced)
        forced_state = tuple(forced.get(i, self.state[i]) for i in range(self.n_vars))
        new = []
        for i in range(self.n_vars):
            par_vals = [forced_state[p] for p in self.source_edges[i]]
            v = HIDDEN_FUNC_LIBRARY[self.funcs[i]](par_vals)
            v += self.rng.gauss(0, self.noise_sigma)
            v = max(0.0, min(1.0, v))
            new.append(v)
        return tuple(new)

    def predict_var_under_intervention(self, target_var: int, iv_var: int, iv_val: float) -> float:
        """Compute ONLY the new value of `target_var` if `iv_var` were forced
        to `iv_val`. Skips the full state recomputation that
        predict_under_intervention does.

        Computes only O(|source_edges of target_var|) function/source_edge work instead
        of predict_under_intervention's O(n_vars) full-state recomputation.
        It still consumes the same number of noise draws as the full path so
        repeated calls preserve the world's RNG stream.

        Behaves identically to predict_under_intervention(iv_var, iv_val)[target_var]
        from a noise/randomness perspective: same single noise draw applied
        to the target's output. Other vars not computed.
        """
        if self._blind_challenge_enabled:
            return self._predict_blind_challenge({iv_var: iv_val})[target_var]
        if iv_var == target_var:
            par_vals = [self.state[p] for p in self.source_edges[target_var]]
        else:
            par_vals = [
                iv_val if p == iv_var else self.state[p]
                for p in self.source_edges[target_var]
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
        """Pick a non-root variable, modify its source_edge set: either drop a
        current source_edge, swap one for a new candidate, or add a new source_edge.
        Adjusts the function if needed (e.g., LOW/HIGH only for no-source_edge
        case). Returns an EDGE mutation with rule_changed=True."""
        i = self.rng.randint(1, self.n_vars - 1)
        old_par = list(self.source_edges[i])
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
        self.source_edges[i] = new_par
        if not new_par:
            self.funcs[i] = self.rng.choice(["LOW", "HIGH"])
        elif self.funcs[i] in {"LOW", "HIGH"}:
            self.funcs[i] = self.rng.choice(["MEAN", "MAX", "MIN", "DIFF"])
        m = HiddenMutation(cycle, "EDGE", f"EDGE x{i} source_edges: {old_par}→{new_par}", True, i)
        self.hidden_log.append(m); return m

    def perturb_func(self, cycle: int) -> HiddenMutation:
        """Pick a random variable, swap its function for a different one of
        appropriate type (LOW/HIGH for no-source_edge, others for has-source_edge).
        Returns a FUNC mutation with rule_changed=True."""
        i = self.rng.randint(0, self.n_vars - 1)
        old = self.funcs[i]
        if self.source_edges[i]:
            choices = [k for k in FUNC_LIBRARY if k != old and k not in {"LOW", "HIGH"}]
        else:
            choices = [k for k in ["LOW", "HIGH"] if k != old]
        new = self.rng.choice(choices)
        self.funcs[i] = new
        m = HiddenMutation(cycle, "FUNC", f"FUNC x{i}: {old}→{new}", True, i)
        self.hidden_log.append(m); return m

    def introduce_sin(self, cycle: int) -> HiddenMutation:
        """Replace some variable's function with SIN (out-of-library). Picks
        the first variable that has source_edges and isn't already SIN. Returns a
        NOVELTY mutation. Falls back to perturb_value if no eligible vars.
        Used to test vocabulary-novelty detection."""
        for i in range(self.n_vars):
            if self.funcs[i] != "SIN" and len(self.source_edges[i]) >= 1:
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
        if self.source_edges[var]:
            choices = [k for k in FUNC_LIBRARY if k != old and k not in {"LOW", "HIGH"}]
        else:
            choices = [k for k in ["LOW", "HIGH"] if k != old]
        if not choices:
            return self.perturb_value(cycle)
        new = self.rng.choice(choices)
        self.funcs[var] = new
        m = HiddenMutation(cycle, "FUNC", f"FUNC x{var}: {old}→{new}", True, var)
        self.hidden_log.append(m); return m

    def _init_false_trass(self, settle_cycles: int, shock_interval: int = 50) -> None:
        """One-time setup for the false_trass schedule.

        Wires a minimal joint-false-trass subgraph alongside the random world:
          xA (TINY≈0.1, root) — individually below salience threshold
          xB (TINY≈0.1, root) — individually below salience threshold
          T  (PROD([xA,xB])) — ≈0.01 at baseline; TINY constant fits within tol
          C  (FIRST([T]))    — tareth sentinel: responds to T; has collector dep
          D  (MEAN([C]))     — makes C tareth (C has a downstream dependent)

        Jointly: PROD(0.95,0.95)=0.9025 >> 0.01 baseline → strong joint interaction
        at any tareth var downstream of T.

        Every shock_interval cycles (after settle), both xA and xB jump to HIGH
        for one cycle. T spikes, triggering T's re-audit with source_edge_screen discovering
        xA and xB as source_edges. They are woken from inert state, audited, and certified
        trass. The proactive joint scan then detects the PROD(TINY,TINY) interaction
        and installs a CompositeNethra.

        shock_interval=50 (default) ensures the first shock fires within any test
        window of ≥ 100 cycles (first shock at settle_cycles+50).

        Layout uses the LAST five vars to avoid colliding with random structure:
          n-5=xA, n-4=xB, n-3=T, n-2=C, n-1=D
        """
        n = self.n_vars
        if n < 5:
            self._ft_initialized = True
            return
        self._ft_xA: int = n - 5
        self._ft_xB: int = n - 4
        self._ft_T: int = n - 3
        self._ft_C: int = n - 2
        self._ft_D: int = n - 1
        self._ft_settle: int = settle_cycles
        self._ft_shock_interval: int = shock_interval
        # Wire the false-trass subgraph
        self.funcs[self._ft_xA] = "TINY"
        self.source_edges[self._ft_xA] = []
        self.funcs[self._ft_xB] = "TINY"
        self.source_edges[self._ft_xB] = []
        self.funcs[self._ft_T] = "PROD"
        self.source_edges[self._ft_T] = [self._ft_xA, self._ft_xB]
        self.funcs[self._ft_C] = "FIRST"
        self.source_edges[self._ft_C] = [self._ft_T]
        self.funcs[self._ft_D] = "MEAN"
        self.source_edges[self._ft_D] = [self._ft_C]
        self._ft_shock_active: bool = False
        for _ in range(5):
            self._step_world()
        self._ft_initialized: bool = True

    def _init_regime_switch(self, settle_cycles: int, regime_length: int = 1000) -> None:
        """One-time setup for the regime_switch schedule.

        Wires a cluster of target vars to share a single causal carrier.
        Two carriers (LOW≈0.2, HIGH≈0.8) alternate every regime_length cycles.
        A collector var is wired downstream of the target cluster so all targets
        have at least one dependent — this gives them role="tareth" and a proper
        sentinel cert instead of a leaf-trass cert (which would hard-suppress
        them before any phase switch is observed).

        Layout (var indices):
          0 = carrier_a (LOW, ~0.2)
          1 = carrier_b (HIGH, ~0.8)
          2..2+k-1 = target cluster (MEAN([active_carrier]))
          2+k      = collector (MEAN([target[0], target[1]]))
          remaining vars: left in their original random structure

        cluster_size = max(2, min(3, n_vars - 3)), leaving room for collector.
        """
        self._rs_carrier_a: int = 0
        self._rs_carrier_b: int = 1
        cluster_size = max(2, min(3, self.n_vars - 3))
        self._rs_targets: List[int] = list(range(2, 2 + cluster_size))
        self._rs_collector: Optional[int] = (2 + cluster_size) if (2 + cluster_size) < self.n_vars else None
        self._rs_phase: int = 0      # 0 = A (carrier_a active), 1 = B (carrier_b active)
        self._rs_settle: int = settle_cycles
        self._rs_length: int = regime_length
        # Force both carriers to root constants with distinct steady-state values
        self.funcs[self._rs_carrier_a] = "LOW"
        self.source_edges[self._rs_carrier_a] = []
        self.funcs[self._rs_carrier_b] = "HIGH"
        self.source_edges[self._rs_carrier_b] = []
        # Wire all targets to carrier_a; collector depends on first two targets
        for t in self._rs_targets:
            self.source_edges[t] = [self._rs_carrier_a]
            self.funcs[t] = "MEAN"
        if self._rs_collector is not None:
            self.source_edges[self._rs_collector] = [self._rs_targets[0], self._rs_targets[1]]
            self.funcs[self._rs_collector] = "MEAN"
        self._rs_source_edges_a: Dict[int, List[int]] = {t: [self._rs_carrier_a] for t in self._rs_targets}
        self._rs_source_edges_b: Dict[int, List[int]] = {t: [self._rs_carrier_b] for t in self._rs_targets}
        for _ in range(5):
            self._step_world()
        self._rs_initialized: bool = True

    def _init_blind_challenge(self) -> None:
        """Generate a seed-driven blind stress world.

        The agent still sees only clipped scalar observations and intervention
        responses. The generated manifest is diagnostic-only and returned only
        by debug_blind_challenge_manifest().
        """
        if self._blind_challenge_enabled:
            return

        n = self.n_vars
        self.visible_count = n
        self._blind_cycle = 0
        self._blind_history = [tuple(self.state)]
        self._blind_latents = []
        self._blind_specs = []
        self._blind_side_effects = []

        n_latents = max(2, min(7, 2 + n // 9))
        for lid in range(n_latents):
            kind = self.rng.choice(["sine", "saw", "pulse", "drift"])
            period = self.rng.randint(45, 420)
            self._blind_latents.append({
                "id": lid,
                "kind": kind,
                "period": period,
                "phase": round(self.rng.random(), 6),
                "amplitude": round(0.12 + self.rng.random() * 0.35, 6),
                "center": round(0.35 + self.rng.random() * 0.3, 6),
                "drift": round((self.rng.random() - 0.5) * 0.004, 6),
            })

        symbolic_funcs = ["LOW", "HIGH", "TINY", "FIRST", "MEAN", "MAX", "MIN", "PROD", "DIFF"]
        relation_types = [
            "symbolic",
            "smooth_nonlinear",
            "latent_additive",
            "delayed",
            "proxy_confounded",
            "dense_mixed",
            "weak_noisy",
        ]

        max_lag = 1
        for i in range(n):
            lower = list(range(i))
            if not lower or self.rng.random() < 0.12:
                rel_type = self.rng.choice(["symbolic", "latent_additive", "weak_noisy"])
                source_edge_count = 0
            else:
                rel_type = self.rng.choice(relation_types)
                if rel_type == "dense_mixed" and len(lower) >= 3:
                    source_edge_count = self.rng.randint(3, min(6, len(lower)))
                elif rel_type in {"delayed", "proxy_confounded"}:
                    source_edge_count = self.rng.randint(1, min(4, len(lower)))
                else:
                    source_edge_count = self.rng.randint(1, min(3, len(lower)))
            source_edges = sorted(self.rng.sample(lower, source_edge_count)) if source_edge_count else []
            latent_count = self.rng.randint(0, min(2, n_latents))
            if rel_type in {"latent_additive", "proxy_confounded"}:
                latent_count = max(1, latent_count)
            latents = sorted(self.rng.sample(range(n_latents), latent_count)) if latent_count else []
            weights = [round(self.rng.uniform(-1.2, 1.2), 6) for _ in source_edges]
            latent_weights = [round(self.rng.uniform(-0.8, 0.8), 6) for _ in latents]
            bias = round(self.rng.uniform(-0.35, 0.35), 6)
            gain = round(self.rng.uniform(0.65, 2.4), 6)
            func = self.rng.choice(symbolic_funcs if source_edges else ["LOW", "HIGH", "TINY"])
            delayed_edges = []
            if source_edges and (rel_type == "delayed" or self.rng.random() < 0.22):
                for pidx in self.rng.sample(source_edges, self.rng.randint(1, min(2, len(source_edges)))):
                    lag = self.rng.randint(1, 9)
                    max_lag = max(max_lag, lag)
                    delayed_edges.append({
                        "source_edge": pidx,
                        "lag": lag,
                        "weight": round(self.rng.uniform(-1.0, 1.0), 6),
                    })
            phase = None
            if self.rng.random() < 0.34:
                phase = {
                    "latent": self.rng.randrange(n_latents),
                    "period": self.rng.randint(70, 520),
                    "flip_weight": round(self.rng.uniform(-1.0, 1.0), 6),
                    "bias_shift": round(self.rng.uniform(-0.28, 0.28), 6),
                }
            noise_scale = round(self.noise_sigma * self.rng.uniform(0.4, 2.8), 6)
            if rel_type == "weak_noisy":
                noise_scale = round(max(noise_scale, 0.035 + self.rng.random() * 0.04), 6)
                gain = round(gain * self.rng.uniform(0.12, 0.35), 6)

            self._blind_specs.append({
                "var": i,
                "relation_type": rel_type,
                "source_edges": source_edges,
                "weights": weights,
                "func": func,
                "latents": latents,
                "latent_weights": latent_weights,
                "bias": bias,
                "gain": gain,
                "delayed_edges": delayed_edges,
                "phase": phase,
                "noise_scale": noise_scale,
                "agent_func_compatible": rel_type == "symbolic" and len(source_edges) <= 2,
            })

        side_effect_count = self.rng.randint(max(1, n // 12), max(2, n // 5))
        for _ in range(side_effect_count):
            source = self.rng.randrange(n)
            possible_targets = [v for v in range(n) if v != source]
            target_count = self.rng.randint(1, min(3, len(possible_targets)))
            self._blind_side_effects.append({
                "source": source,
                "targets": sorted(self.rng.sample(possible_targets, target_count)),
                "trigger": round(self.rng.uniform(0.15, 0.85), 6),
                "direction": self.rng.choice(["above", "below"]),
                "effect": round(self.rng.uniform(-0.22, 0.22), 6),
                "mode": self.rng.choice(["additive", "damped"]),
            })

        self.source_edges = [list(spec["source_edges"]) for spec in self._blind_specs]
        self.funcs = [
            spec["func"] if spec["relation_type"] == "symbolic" else spec["relation_type"].upper()
            for spec in self._blind_specs
        ]
        self._blind_max_lag = max_lag
        self._blind_challenge_manifest = {
            "version": 1,
            "n_vars": n,
            "interface": "observed_scalar_clipped_0_1",
            "latents": copy.deepcopy(self._blind_latents),
            "relations": copy.deepcopy(self._blind_specs),
            "intervention_side_effects": copy.deepcopy(self._blind_side_effects),
        }
        self._blind_challenge_enabled = True
        for _ in range(max_lag + 3):
            self._step_blind_challenge_world()

    def _blind_latent_values(self, cycle: int) -> List[float]:
        values = []
        for spec in self._blind_latents:
            period = max(1, int(spec["period"]))
            t = (cycle / period + float(spec["phase"])) % 1.0
            amp = float(spec["amplitude"])
            center = float(spec["center"]) + float(spec["drift"]) * cycle
            kind = spec["kind"]
            if kind == "sine":
                raw = center + amp * math.sin(2.0 * math.pi * t)
            elif kind == "saw":
                raw = center + amp * (2.0 * t - 1.0)
            elif kind == "pulse":
                raw = center + (amp if t < 0.18 else -0.35 * amp)
            else:
                raw = center + amp * math.sin(math.pi * t) + float(spec["drift"]) * cycle
            values.append(max(0.0, min(1.0, raw)))
        return values

    def _delayed_value(self, source_edge: int, lag: int, fallback: State) -> float:
        if lag <= 0 or len(self._blind_history) < lag:
            return fallback[source_edge]
        return self._blind_history[-lag][source_edge]

    def _eval_blind_spec(
        self,
        spec: Dict[str, Any],
        forced_state: State,
        latents: List[float],
        include_noise: bool,
    ) -> float:
        source_edges = spec["source_edges"]
        vals = [forced_state[p] for p in source_edges]
        rel_type = spec["relation_type"]
        bias = float(spec["bias"])
        gain = float(spec["gain"])
        latent_term = sum(
            float(w) * (latents[lid] - 0.5)
            for lid, w in zip(spec["latents"], spec["latent_weights"])
        )
        phase = spec.get("phase")
        if phase is not None:
            phase_period = max(1, int(phase["period"]))
            phase_on = ((self._blind_cycle // phase_period) % 2) == 1
            if phase_on:
                bias += float(phase["bias_shift"])
                gain += float(phase["flip_weight"])

        if rel_type == "symbolic":
            fn = HIDDEN_FUNC_LIBRARY[spec["func"]]
            raw = fn(vals)
        elif rel_type == "smooth_nonlinear":
            drive = bias + latent_term + sum(float(w) * (v - 0.5) for w, v in zip(spec["weights"], vals))
            raw = 1.0 / (1.0 + math.exp(-gain * drive))
        elif rel_type == "latent_additive":
            base = sum(vals) / len(vals) if vals else 0.5
            raw = base + bias + latent_term
        elif rel_type == "delayed":
            drive = bias + latent_term
            for edge in spec["delayed_edges"]:
                drive += float(edge["weight"]) * (
                    self._delayed_value(edge["source_edge"], edge["lag"], forced_state) - 0.5
                )
            drive += sum(float(w) * (v - 0.5) for w, v in zip(spec["weights"], vals))
            raw = 0.5 + 0.5 * math.tanh(gain * drive)
        elif rel_type == "proxy_confounded":
            source_edge_term = sum(float(w) * v for w, v in zip(spec["weights"], vals))
            raw = 0.35 + 0.35 * source_edge_term + 0.55 * latent_term + bias
        elif rel_type == "dense_mixed":
            weighted = sum(float(w) * (v - 0.5) for w, v in zip(spec["weights"], vals))
            interaction = 0.0
            if len(vals) >= 2:
                interaction = (vals[0] - 0.5) * (vals[-1] - 0.5)
            raw = 0.5 + 0.5 * math.tanh(gain * (weighted + latent_term + bias + interaction))
        else:  # weak_noisy
            weak = sum(float(w) * (v - 0.5) for w, v in zip(spec["weights"], vals))
            raw = 0.5 + gain * weak + 0.35 * latent_term + bias * 0.25

        for edge in spec["delayed_edges"]:
            if rel_type != "delayed":
                raw += 0.08 * float(edge["weight"]) * (
                    self._delayed_value(edge["source_edge"], edge["lag"], forced_state) - 0.5
                )
        if include_noise:
            raw += self.rng.gauss(0.0, float(spec["noise_scale"]))
        return max(0.0, min(1.0, raw))

    def _apply_blind_side_effects(
        self,
        values: List[float],
        forced: Mapping[int, float],
    ) -> None:
        if not forced:
            return
        for rule in self._blind_side_effects:
            source = int(rule["source"])
            if source not in forced:
                continue
            forced_val = float(forced[source])
            trigger = float(rule["trigger"])
            fires = forced_val >= trigger if rule["direction"] == "above" else forced_val <= trigger
            if not fires:
                continue
            effect = float(rule["effect"])
            for target in rule["targets"]:
                if rule["mode"] == "damped":
                    values[target] = values[target] * (1.0 - abs(effect)) + max(0.0, effect)
                else:
                    values[target] += effect
                values[target] = max(0.0, min(1.0, values[target]))

    def _predict_blind_challenge(self, forced: Mapping[int, float]) -> State:
        forced_state = tuple(
            max(0.0, min(1.0, float(forced.get(i, self.state[i]))))
            for i in range(self.n_vars)
        )
        latents = self._blind_latent_values(self._blind_cycle)
        values = [
            self._eval_blind_spec(spec, forced_state, latents, include_noise=True)
            for spec in self._blind_specs
        ]
        self._apply_blind_side_effects(values, forced)
        return tuple(max(0.0, min(1.0, v)) for v in values)

    def _step_blind_challenge_world(self) -> None:
        self._blind_cycle += 1
        self._blind_history.append(tuple(self.state))
        if len(self._blind_history) > self._blind_max_lag + 2:
            self._blind_history = self._blind_history[-(self._blind_max_lag + 2):]
        self.state = self._predict_blind_challenge({})

    def debug_blind_challenge_manifest(self) -> Dict[str, Any]:
        """Return generated blind-world facts for post-run analysis only."""
        if self._blind_challenge_manifest is None:
            return {}
        manifest = copy.deepcopy(self._blind_challenge_manifest)
        manifest["cycles_elapsed"] = self._blind_cycle
        return manifest

    def prepare_schedule(self, schedule: str, settle_cycles: int = 25) -> None:
        """Pre-install schedule-specific subgraph structure BEFORE the agent is
        created. Call this after CausalWorld() but before ChainedAgent.initialize()
        so the agent's first audits see the intended world (not the random base).

        Safe to call multiple times — guards against double-init."""
        if schedule == "false_trass":
            if not getattr(self, "_ft_initialized", False):
                self._init_false_trass(settle_cycles)
        elif schedule == "regime_switch":
            if not getattr(self, "_rs_initialized", False):
                self._init_regime_switch(settle_cycles)
        elif schedule == "blind_challenge":
            self._init_blind_challenge()

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
                           every 1000 cycles after cycle 100.
          novelty: SIN injection at cycle 10; some early edges; value drift.
          incremental: reveal one variable per `settle_cycles` cycles; no
                       structural changes. Tests bootstrapping behavior.
          rare_catastrophe: value drift each cycle; with probability rare_prob,
                            mutate rare_var's function (structural change on a
                            specific variable). Mutations are permanent — no
                            reversion. settle_cycles suppresses rare mutations
                            for the first N cycles so certs can establish.
          regime_switch: a cluster of vars share a causal carrier that alternates
                         every 1000 cycles between two root constants (LOW/HIGH).
                         settle_cycles controls the initial stable window before
                         the first phase switch. Phase boundaries simultaneously
                         rewire all cluster members, producing multi-var co-failure
                         bursts that the regime register can confirm and commission
                         a cluster-level sentinel for.
          blind_challenge: seed-generated scalar world with hidden latents,
                           delayed/proxy/dense/weak relations, recurring phases,
                           and intervention side effects. Debug facts are exposed
                           only via debug_blind_challenge_manifest().
        """
        if schedule == "blind_challenge":
            if not self._blind_challenge_enabled:
                self._init_blind_challenge()
            self._step_blind_challenge_world()
            phase_hits = []
            for spec in self._blind_specs:
                phase = spec.get("phase")
                if phase is None:
                    continue
                period = max(1, int(phase["period"]))
                if self._blind_cycle % period == 0:
                    phase_hits.append(spec["var"])
            if phase_hits:
                m = HiddenMutation(
                    cycle=cycle,
                    kind="BLIND_PHASE",
                    description=f"BLIND_PHASE vars={phase_hits[:8]} count={len(phase_hits)}",
                    rule_changed=True,
                    affected_var=phase_hits[0],
                )
            else:
                m = HiddenMutation(cycle, "VALUE", f"blind_step c{cycle}", False, -1)
            self.hidden_log.append(m)
            return m
        if schedule == "shaped":
            structural = {2:"edge", 5:"func", 8:"edge", 11:"func", 13:"edge"}
            kind = structural.get(cycle, "value")
        elif schedule == "periodic_shifts":
            structural = {2:"edge", 5:"func", 8:"edge", 11:"func", 13:"edge"}
            kind = structural.get(cycle, "value")
            if cycle > 100 and cycle % 1000 == 0:
                kind = "edge" if (cycle // 1000) % 2 == 0 else "func"
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
        elif schedule == "regime_switch":
            if not getattr(self, "_rs_initialized", False):
                self._init_regime_switch(settle_cycles)
            if cycle < self._rs_settle:
                return self.perturb_value(cycle)
            elapsed = cycle - self._rs_settle
            if elapsed % self._rs_length == 0:
                self._rs_phase = 1 - self._rs_phase
                new_source_edges = self._rs_source_edges_b if self._rs_phase == 1 else self._rs_source_edges_a
                carrier = self._rs_carrier_b if self._rs_phase == 1 else self._rs_carrier_a
                for t, par in new_source_edges.items():
                    self.source_edges[t] = par
                phase_name = "B" if self._rs_phase == 1 else "A"
                m = HiddenMutation(
                    cycle=cycle, kind="REGIME_SWITCH",
                    description=(
                        f"REGIME_SWITCH phase→{phase_name} carrier=x{carrier} "
                        f"targets={self._rs_targets}"
                    ),
                    rule_changed=True, affected_var=carrier,
                )
                self.hidden_log.append(m)
                return m
            return self.perturb_value(cycle)
        elif schedule == "false_trass":
            if not getattr(self, "_ft_initialized", False):
                self._init_false_trass(settle_cycles)
            if cycle <= self._ft_settle or self.n_vars < 5:
                # Use causal stepping even during settle so the subgraph stays
                # consistent with its wired functions (xA/xB=TINY, T=PROD, ...).
                # perturb_value would randomly corrupt the subgraph state.
                self._step_world()
                m = HiddenMutation(cycle, "VALUE", f"ft_settle c{cycle}", False, -1)
                self.hidden_log.append(m)
                return m
            elapsed = cycle - self._ft_settle
            # Restore on the cycle AFTER shock before checking for a new one.
            if self._ft_shock_active:
                self.funcs[self._ft_xA] = "TINY"
                self.funcs[self._ft_xB] = "TINY"
                self._ft_shock_active = False
            # Fire new shock every shock_interval cycles (skip elapsed=0).
            if elapsed > 0 and elapsed % self._ft_shock_interval == 0:
                self.funcs[self._ft_xA] = "HIGH"
                self.funcs[self._ft_xB] = "HIGH"
                self._ft_shock_active = True
                self._step_world()
                m = HiddenMutation(
                    cycle=cycle, kind="JOINT_SHOCK",
                    description=(
                        f"JOINT_SHOCK xA=x{self._ft_xA} xB=x{self._ft_xB} "
                        f"→HIGH (T=x{self._ft_T} will spike)"
                    ),
                    rule_changed=True, affected_var=self._ft_T,
                )
                self.hidden_log.append(m)
                return m
            # Normal step: advance state from causal funcs so xA/xB track TINY.
            self._step_world()
            m = HiddenMutation(cycle, "VALUE", f"ft_step c{cycle}", False, -1)
            self.hidden_log.append(m)
            return m
        else:
            kind = "value"
        return {"edge": self.perturb_edge, "func": self.perturb_func,
                "novelty": self.introduce_sin,
                "value": self.perturb_value}[kind](cycle)
