# Codex Task: dreth Attention & Cost Optimizations

## Repository
https://github.com/Dooces/dreth (branch: `main`)

## Context

dreth is a causal-discovery simulation where an agent discovers hidden causal structure through earned authority. The agent observes a hidden causal world (DAG of variables with causal functions + noise), proposes hypotheses, and earns the right to certify causal relationships only by surviving structured intervention tests. The core pipeline is: Observe → Frontier → Audit → Fit → Sentinel → Certify → Predict.

**Key invariants you MUST NOT violate:**
1. Authority is earned, never assumed. Only the audit→fit→sentinel→certify pipeline grants authority.
2. Shadow/diagnostic layers observe but NEVER mutate agent or ledger state.
3. Provider confidence is never treated as cert authority.
4. The ledger (`NethraCertificate`, `VarNethra`) is the single source of truth.
5. `MORPHOLOGY ≠ CAUSE` — score proximity is structural observation, not causal classification.
6. Ambiguity is first-class — `TiedFrontier` must survive until regime-survival proof justifies collapse.

**Key types (all in `dreth/ledger.py`):**
- `VarNethra`: per-variable authority handle. Has `.parents`, `.func`, `.sentinels`, `.expected_outcomes`, `.tied_frontier`, `.certificates` dict, `.cost_weight`, `.envelope` (NoiseEnvelope), `.temporal_trass_log`, `.role_for(op)`, `.authority_for(op)`, `.invalidate_certs(event)`, `.consecutive_sentinel_failures`, `.strong_observations`, `.dormant_alternatives` (list of `DormantAlternative`).
- `TiedFrontier`: ambiguity object on VarNethra. Has `.candidates` (frozenset of (parents, func, score) tuples), `.separating_probes` (currently always empty tuple), `.context_key`, `.stable_count`, `.distinct_contexts_seen`.
- `NethraCertificate`: frozen cert with operation, role, authority, context, scope, evidence, witnesses.
- `NoiseEnvelope`: tracks deltas, has `.certified_eps`, `.add_delta()`, `.envelope_failing()`.
- `DormantAlternative`: archived hypothesis that was collapsed out. Has `.parents`, `.func`, `.score`, `.cycle_archived`.
- `ChainedLedger`: owns all VarNethras. Has `.vars` dict, `.invalidate()` which cascades through descendants.

**Key agent state (all in `dreth/agent.py` on `ChainedAgent`):**
- `self._live_set: Optional[Set[int]]` — hot partition of variables. Dormant vars removed from this set.
- `self._inert_vars: Set[int]` — variables screened as causally inert at init.
- `self._uncertain_this_cycle: Set[int]` — vars whose certs were invalidated by sentinel failure this cycle.
- `self.regime_register: RegimeRegister` — detects recurring co-failure patterns.
- `self.near_tie_margin: int = 4` — hypotheses within this many probes of best are near-ties.
- `self.priority_audit_budget: int` — max full audits per cycle.
- `self.frontier_k: int` — sparse init frontier size.
- `self.total_interventions: int` — running counter.
- `self.defer_count`, `self.defer_streak`, `self.weak_streak`, `self.stable_streak` — per-var counters.
- `self._var_repair_failures: Dict[int, int]` — sentinel fired → same fit found count.

**Key functions:**
- `fit_var()` in `dreth/fit.py`: enumerates hypotheses, scores them, returns `(parents, func, best_score, second_score)`. Fills `diag` dict with `near_tie_candidates`, `tie_set`, `margin`, `probes`, etc.
- `select_var_sentinels()` in `dreth/sentinels.py`: picks discriminating intervention probes for cheap-path validation.
- `check_var_sentinels_with_envelope()` in `dreth/sentinels.py`: runs sentinel probes, returns `(passed, score, total, reason, max_dev)`.
- `predict_var()` in `dreth/fit.py`: predicts a variable's value using certified Nethra.
- `ledger.invalidate()` in `dreth/ledger.py`: cascades invalidation through descendants.

---

## Task 1: Populate `TiedFrontier.separating_probes` during `fit_var`

### Problem
`TiedFrontier.separating_probes` is always an empty tuple. The field exists but is never populated. This means sentinel selection (`select_var_sentinels`) and frontier attention have no information about which probes would discriminate between tied hypotheses.

### What to do

**File: `dreth/fit.py`**

In `fit_var()`, after computing `near_tie_candidates_out` (around the end of the function), add logic to identify separating probes from the intervention pool that was already evaluated:

1. From the `all_preds` array (shape `(n_hypotheses, n_interventions)`) and the `interventions` list, identify which interventions produce the most divergent predictions among the near-tie candidates specifically.
2. For each intervention index `k`, compute how many distinct prediction buckets (rounded to `tolerance`) exist among just the near-tie hypothesis indices. Call this `tie_discrimination[k]`.
3. Select the top 3 interventions by `tie_discrimination` (breaking ties by overall discrimination). These become the separating probes.
4. Store them as `separating_probes` in the `diag` dict alongside `near_tie_candidates`:
   ```python
   diag["separating_probes"] = tuple((interventions[k][0], interventions[k][1]) for k in top_3_indices)
   ```

**File: `dreth/agent.py`**

In `_install_var` (or wherever `TiedFrontier` is constructed/updated from `FitDiagnostic`), read `diag.separating_probes` and pass it to the `TiedFrontier` constructor:
```python
tied_frontier = TiedFrontier(
    candidates=...,
    separating_probes=tuple(diag.separating_probes) if diag.separating_probes else (),
    ...
)
```

**File: `dreth/ledger.py`**

Verify that `TiedFrontier` accepts `separating_probes` in its constructor. It likely already has the field — confirm it's stored and not discarded.

### Constraints
- Do NOT change the return signature of `fit_var`.
- `separating_probes` is morphology — it says which probes split tied candidates, not why they tie.
- The probes must come from the intervention pool already evaluated in `fit_var` — do not run additional interventions.

---

## Task 2: Use `TiedFrontier` to influence frontier priority

### Problem
The agent's frontier selector picks which variable to attend to next, but it doesn't consider whether a variable has unresolved tied hypotheses. Variables with `TiedFrontier` entries are exactly the ones where more information would be most discriminating, but they get no priority boost.

### What to do

**File: `dreth/agent.py`**

Find the frontier selection logic in `run_cycle` (the part that decides which variables get full audits within the `priority_audit_budget`). This likely involves topological ordering and/or priority scoring.

Add a tiebreaker/boost for variables that have a non-None `tied_frontier` on their `VarNethra`:

1. When scoring variables for audit priority, add a small bonus (e.g., +1 to priority score) for any variable where `self.ledger.vars[var].tied_frontier is not None` and `len(self.ledger.vars[var].tied_frontier.candidates) > 1`.
2. This bonus should be a tiebreaker, not a dominant factor — it should not override cost-weight dispatch or consequence-tier ordering.
3. If `uncertainty_consolidation_mode == "assist"` and the variable also has an uncertainty budget bonus (`self._uncertainty_budget_bonus.get(var, 0) > 0`), the tied-frontier bonus stacks additively.

### Constraints
- This is attention routing, not authority. The bonus affects which variable gets audited next, not whether it gets certified.
- Do not change the dormancy logic (`_live_set`). Dormant vars stay dormant regardless of tied frontier.
- Do not change the trass-skip path. Trass vars are still skipped.

---

## Task 3: Feed `TiedFrontier.separating_probes` into sentinel selection

### Problem
`select_var_sentinels` picks discriminating probes generically — it doesn't know which probes would specifically separate the tied hypotheses. The `forced_probes` parameter in `fit_var` already supports injecting probes from `TiedFrontier.separating_probes`, but sentinel selection itself doesn't use them.

### What to do

**File: `dreth/agent.py`**

In the code path that calls `select_var_sentinels` (likely in `_install_var` or after certification), check if the variable's `VarNethra` has a `tied_frontier` with non-empty `separating_probes`. If so, pass them as a `priority_probes` hint:

1. Before calling `select_var_sentinels`, extract `n.tied_frontier.separating_probes` if available.
2. After `select_var_sentinels` returns its `(probes, expected)` list, prepend the separating probes (up to 2) to the sentinel list, removing any duplicates. Cap total sentinels at `self.sentinel_count` — separating probes displace the least-discriminating generic probes, not add to the total.

**File: `dreth/sentinels.py`**

No changes needed to `select_var_sentinels` itself — the injection happens at the call site in `agent.py`. But verify that the sentinel probes list is a plain `List[Tuple[int, float]]` that can be prepended to.

### Constraints
- Sentinel count stays at `self.sentinel_count`. Separating probes replace the weakest generic probes, they don't increase the total.
- The separating probes must be valid: `0 <= iv_var < world.visible_count` and `0.0 <= iv_val <= 1.0`. Filter invalid ones.

---

## Task 4: Cheap pre-check before cascade re-audit

### Problem
When a sentinel fails and `world_changed` is attributed, `ledger.invalidate()` cascades through all descendants, and all of them get queued for full re-audit. But many descendants may still have correct fits — their predictions still match observations. A full re-audit is expensive (runs `fit_var` with full intervention budget). A cheap prediction check could skip unnecessary re-audits.

### What to do

**File: `dreth/agent.py`**

Find the sentinel-failure handling code — the branch where `world_changed` is determined and `ledger.invalidate()` is called, followed by queueing descendants for re-audit. This is in `run_cycle`, in the sentinel-fail path.

After `ledger.invalidate()` cascades and before descendants are queued for full audit, add a prediction pre-check:

1. For each descendant `d` that was invalidated:
   a. Get `nd = self.ledger.vars[d]`
   b. If `nd.func` is not set or `nd.parents` is empty, skip pre-check (queue for audit — it needs one).
   c. Compute `predicted = predict_var(nd.parents, nd.func, self.world.state, d, self.world.state[d], self.world.visible_count)` — but this isn't right because `predict_var` takes an intervention. Instead, compute the prediction directly: `predicted = FUNC_LIBRARY[nd.func]([self.world.state[p] for p in nd.parents])`.
   d. Get `actual = self.world.state[d]`.
   e. Get `eps = nd.envelope.certified_eps if nd.envelope.certified_eps > 0 else DEFAULT_TOLERANCE`.
   f. If `abs(predicted - actual) <= eps * 3.0` (generous margin — 3× envelope to avoid false negatives), the descendant's fit is likely still correct. **Do not queue it for full audit.** Instead, just mark its certs as needing revalidation (which `invalidate_certs` already did) and let the normal sentinel path re-earn authority on the next cycle.
   g. If the prediction deviates, queue for full audit as before.

2. Add a counter `self.cascade_precheck_skipped: int = 0` to `__init__` and increment it each time a descendant passes the pre-check and avoids re-audit.

### Constraints
- The pre-check reads `self.world.state` which is the current observation — this is agent-visible data, not hidden truth. This is legal.
- The pre-check does NOT re-certify anything. It only decides whether to queue a full audit. Authority still flows through the pipeline.
- If in doubt, queue for audit. The pre-check is an optimization, not a correctness gate. Use the generous 3× margin.
- Do not skip the pre-check for high-cost variables (`cost_weight >= cost_high_threshold`). Those always get re-audited.

---

## Task 5: Revive dormant alternatives on regime change

### Problem
`VarNethra.dormant_alternatives` stores hypotheses that were collapsed out, but nothing reads them. When a regime change is detected (the regime layer fires), the agent re-audits from scratch, re-enumerating the full hypothesis space. But the dormant alternatives are a shortlist of previously-plausible hypotheses that could be checked cheaply first.

### What to do

**File: `dreth/agent.py`**

Find the regime-change handling code — where `self.regime_register` detects a confirmed regime and triggers re-audits. This is likely in `run_cycle` after the regime register processes `CertEvent`s.

When a regime is confirmed and variables are queued for re-audit:

1. For each affected variable `var`, check `self.ledger.vars[var].dormant_alternatives`.
2. If dormant alternatives exist (list is non-empty), before running the full `fit_var`, do a quick score check:
   a. For each `DormantAlternative` in the list (up to 5), compute its score against the current sentinel probes (reuse the variable's existing `n.sentinels` as a cheap probe set):
      ```python
      score = 0
      for iv, _ in zip(n.sentinels, n.expected_outcomes):
          predicted = predict_var(da.parents, da.func, self.world.state, iv[0], iv[1], self.world.visible_count)
          actual = self.world.predict_var_under_intervention(var, iv[0], iv[1])
          if values_match(predicted, actual, DEFAULT_TOLERANCE):
              score += 1
      ```
   b. If any dormant alternative scores perfectly (score == len(n.sentinels)) or better than the current fit's sentinel score, flag it as a `revival_candidate`.
   c. Pass `revival_candidate` parents into `fit_var` via `available_parents` to ensure they're included in the hypothesis space (they may have been excluded if they were trass-certified before the regime change).

3. Add a counter `self.dormant_revival_count: int = 0` to `__init__` and increment when a dormant alternative is revived as a candidate.

### Constraints
- Dormant revival does NOT skip the audit pipeline. It only ensures the dormant hypothesis is *included* in the enumeration. It still must survive fit→sentinel→certify.
- Do not revive dormant alternatives outside of regime-change contexts. Normal sentinel failures use the existing cascade path.
- Cap at 5 dormant alternatives checked per variable to bound cost.

---

## Task 6: Re-screen inert variables on regime change

### Problem
Variables screened as causally inert at initialization (`self._inert_vars`) are permanently excluded from the frontier unless woken by a descendant sentinel failure. But regime changes can make previously inert variables active. The regime layer already detects regime changes — but doesn't trigger inert re-screening.

### What to do

**File: `dreth/agent.py`**

There is already an `_INERT_RESCREEN_THRESHOLD` constant and some re-screening logic tied to repair failures. Add a regime-triggered re-screen:

1. Find where confirmed regimes are processed (after `self.regime_register` promotes a regime).
2. When a regime is confirmed, collect the variables involved in the regime's co-failure signature.
3. For each inert variable `iv` in `self._inert_vars`:
   a. Check if any of the regime's co-failure variables are potential descendants or parents of `iv` in the current topological order. If the relationship is unclear (which it will be for inert vars since they were never audited), do a cheap perturbation check:
      - Intervene on `iv` at 0.05 and 0.95 (the same test used at init for inert screening).
      - Check if any of the regime's co-failure variables change beyond `DEFAULT_TOLERANCE`.
   b. If they do, remove `iv` from `self._inert_vars`, add it to `self._live_set`, and queue it for audit.

4. Add a counter `self.regime_inert_wakeup_count: int = 0` to `__init__`.

### Constraints
- Only re-screen inert vars when a regime is *confirmed* (promoted), not on every co-failure event.
- Cap the re-screen to at most `min(len(self._inert_vars), 10)` variables per regime confirmation to bound cost.
- The perturbation check uses 2 interventions per inert variable — this is cheap relative to a full audit.

---

## Task 7: Add cumulative intervention cost tracking

### Problem
`self.total_interventions` counts interventions but doesn't weight them by cost. There's no way to measure whether cost-dispatch is actually saving budget vs. the `RefitBaseline`.

### What to do

**File: `dreth/agent.py`**

1. Add `self.total_weighted_intervention_cost: float = 0.0` to `__init__`.
2. Everywhere `self.total_interventions` is incremented, also add:
   ```python
   self.total_weighted_intervention_cost += self.ledger.vars[var].cost_weight
   ```
   where `var` is the variable being probed. If the intervention is a generic probe not tied to a specific variable (e.g., in `fit_var`), use `1.0` as the weight.

**File: `dreth/summary.py`**

3. In `RunAnalyzer` and/or `SummaryRenderer`, add `total_weighted_intervention_cost` to the end-of-run metrics output so it appears in the summary alongside `total_interventions`.

### Constraints
- This is purely diagnostic. The cost counter does not influence any agent decisions.
- Do not change the `RefitBaseline` — it already tracks its own intervention count.

---

## Testing

All existing tests in `tests/` must continue to pass. Run `pytest tests/` after all changes.

For the new behavior, add targeted tests:

1. **Task 1 test**: Run `fit_var` on a scenario with known tied hypotheses (e.g., two single-parent FIRST hypotheses that score identically). Verify `diag["separating_probes"]` is non-empty and contains valid `(iv_var, iv_val)` tuples.

2. **Task 4 test**: Set up a world where a sentinel fails on variable X, X has descendants Y and Z, but only Y's fit is actually wrong. Verify that Z is not queued for full audit (pre-check passes) while Y is.

3. **Task 5 test**: Set up a world with a regime change. Before the change, variable V has parent A. After the change, V's true parent switches to B. B was previously a dormant alternative. Verify that B appears in the hypothesis space during re-audit after regime detection.

Put new tests in `tests/test_attention_optimizations.py`.

## Summary of files to modify
- `dreth/fit.py` — Task 1 (separating probes computation)
- `dreth/agent.py` — Tasks 1-7 (all changes to agent logic)
- `dreth/ledger.py` — Task 1 (verify TiedFrontier field)
- `dreth/summary.py` — Task 7 (cost metric in summary)
- `tests/test_attention_optimizations.py` — new test file

## What NOT to do
- Do not change the authority pipeline (audit→fit→sentinel→certify).
- Do not let shadow/diagnostic layers mutate agent state.
- Do not change `fit_var`'s return signature.
- Do not change `select_var_sentinels`'s signature.
- Do not add new CLI arguments (these are internal optimizations).
- Do not change the `RefitBaseline`.
- Do not collapse `TiedFrontier` based on score proximity — that requires regime-survival evidence.
- Do not read hidden-world fields (`truth_parents`, `truth_func`) in any agent-visible code path.
