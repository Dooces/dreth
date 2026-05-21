# DRETH TODO — ground-truth current state and execution order

This TODO is the operational source of truth.

Do not use `dreth/STATE.md` as authoritative until it is repaired or deleted.

This file distinguishes:
- current executable state
- safe bookkeeping changes
- behavior-changing changes
- design-only future work

Do not promote design intent into implementation claims.

---

## Current state

The project is an executable research prototype.

Working spine:

- `dreth/agent.py`: causal audit loop, sentinel cheap path, invalidation, promotion, compression discovery.
- `dreth/fit.py`: hypothesis enumeration, probe selection, batched scoring, exact tie diagnostics.
- `dreth/ledger.py`: `VarNethra`, status/role state, drift tracking, invalidation closure.
- `tests/test_fit_transition_boundaries.py`: fast pytest coverage.
- `tests/test_regime_recall.py`: standalone receipt/stress harness, not normal pytest coverage.

REMOVED: `dreth/forms.py` and `dreth/v30_integration.py` — forms layer removed. No replacement yet (N3 in TODO.md).

Partially implemented / scaffolded:

- `TiedFrontier` tracking is wired:
  - dataclass exists in `ledger.py`
  - fields exist on `VarNethra`
  - lifecycle management exists in `_install_var`
  - `_update_tied_frontier` exists
  - `_collapse_tied_frontier` exists
  - `near_tie_candidates` and `near_tie_context_key` exist in `FitDiagnostic`
  - `frontier_candidates` and `near_tie_margin` are threaded through `fit_var`
  - frontier clears on invalidation and structural reset
  - collapsed losers archive to `dormant_alternatives`

- `TiedFrontier` is not yet operationally consumed:
  - no active decision path reads the frontier to change behavior
  - frontier-biased probe selection is disabled
  - semantic suppression for frontier churn was implemented and removed (perturbed form discovery timing; forms layer since removed)
  - dormant alternatives are archived but not revived

- Adaptive probe budget exists but is intentionally disabled because changing probe pool size perturbs probe selection / RNG trajectory.

- Dormant partition: dormant vars (noise_floor cert or otherwise parked) are excluded from the main loop. Re-entry is failure-driven only — sentinel failure triggers cascade invalidation which adds the var back to _live_set. The previous rationale ("envelope deltas feed form signatures") was a dead dependency on forms.py, which no longer exists.

- NN proposers train as diagnostics but are not active decision components.

- Cold storage exists but should be treated as experimental until key serialization is made invertible.

Current stress baseline:

```bash
PASS=6  PARTIAL=1  SKIP=17  FAIL=0  total=24
```

---

## Do not claim

Do not claim:

- `v28` mode is pure v28 while v30 forms are default-enabled.
- `total_interventions` is a full intervention-observation counter.
- NN proposers are operational.
- `TiedFrontier` consumers are operational.
- `TiedFrontier` currently performs abstraction discovery.
- raw ties imply abstraction.
- tie cause can be read from hypothesis morphology alone.
- same-parent / different-operator means FUNC_LIBRARY defect.
- different-parent / same-operator means structural symmetry.
- `frontier_same_parent_churn` currently suppresses `semantic_changed`.
- context-key survival is proof of regime survival.

Allowed claim:

- `TiedFrontier` currently preserves near-tie frontier state and dormant alternatives, but it does not yet use that state to guide probing, collapse, suppression, revival, or abstraction.

---

## Core invariant for TiedFrontier

A tie is evidence of unresolved operational ambiguity.

A tie is not proof of sameness.

A tie is not proof of abstraction.

A tie is not proof of library defect.

A tie is not proof of under-probing.

The correct sequence is:

```text
tie observed
→ store frontier
→ label morphology
→ generate candidate-disagreement diagnostics
→ generate separating probes
→ test survival across explicit operational contexts
→ only then infer cause
→ only then collapse, suppress, revive, or feed abstraction machinery
```

Morphology is structural.

Cause is evidential.

Collapse requires evidence.

---

# P0 — repair measurement before arguing speed or structure

## P0.1 Replace `total_interventions` with explicit probe counters

Problem:

`total_interventions` currently counts fit budgets and sentinel counts, but misses operation-role probes and form hypothesis probes. It therefore undercounts work and can over-credit forms.

Add counters on `ChainedAgent`:

```python
self.fit_probe_calls = 0
self.role_probe_calls = 0
self.sentinel_probe_calls = 0
self.form_probe_calls = 0
self.compression_probe_calls = 0
```

Make `total_interventions` either deprecated or a derived compatibility alias.

Add:

```python
@property
def total_probe_calls(self) -> int:
    return (
        self.fit_probe_calls
        + self.role_probe_calls
        + self.sentinel_probe_calls
        + self.form_probe_calls
        + self.compression_probe_calls
    )
```

Count sites:

- `_full_audit_var`: increment `fit_probe_calls` by actual `len(diag_dict["probes"])`, not requested budget.
- `_certify_operation_role`: increment `role_probe_calls` for every `predict_under_intervention` call.
- sentinel checks: increment `sentinel_probe_calls` by actual sentinel checks executed.
- `_check_form_hypothesis`: increment `form_probe_calls` per `predict_var_under_intervention` call.
- compression discovery/checks: increment `compression_probe_calls` wherever world intervention calls are used.

Acceptance:

- Add a monkeypatch test that wraps `world.predict_under_intervention` and `world.predict_var_under_intervention`.
- Assert derived total equals observed calls for at least one short CLI run.
- Existing fast pytest passes.
- Regime receipt remains non-FAIL.
- Stress baseline must not change except by explained counter-reporting.

---

## P0.2 Fix mode semantics

Problem:

`--mode v28` can still attach v30 forms because forms are default-enabled.

Choose one and implement consistently:

Option A:

- `--mode v28` disables v30 unless `--enable-forms` is explicitly passed.

Option B:

- If forms are active, report the active stack as `v28-core+forms`, not pure `v28`.

Acceptance:

- CLI summary reports the active stack, not only the mode label.
- One test or smoke command proves the chosen behavior.
- Documentation uses the same naming.

---

# P1 — make TiedFrontier bookkeeping truthful and non-causal

P1 is safe bookkeeping only.

Do not change audit behavior yet.

Do not suppress semantic changes yet.

Do not activate frontier-biased probes yet.

Do not feed ties into forms yet.

The purpose of P1 is to make the frontier’s stored state explicit enough that later consumers cannot confuse morphology with cause.

---

## P1.1 Add tie morphology labels

Add a computed morphology object to `TiedFrontier`, or compute/populate it inside `_update_tied_frontier`.

Recommended structure:

```python
@dataclass(frozen=True)
class TieMorphology:
    exactness: str          # "exact" | "near" | "mixed"
    parent_shape: str       # "same_parent" | "mixed_parent"
    operator_shape: str     # "same_operator" | "mixed_operator"
    candidate_count: int
    score_span: int
    best_score: int
```

Definitions:

- `exact`: all candidates have score equal to best score.
- `near`: at least one candidate is below best but within `near_tie_margin`.
- `mixed`: use only if exact and near candidates are both explicitly represented.
- `same_parent`: all candidates share the same parent tuple.
- `mixed_parent`: candidate parent tuples differ.
- `same_operator`: all candidates share the same function/operator name.
- `mixed_operator`: candidate function/operator names differ.
- `score_span`: `best_score - worst_score` across candidates.

These are descriptive labels only.

They do not imply cause.

They must not trigger collapse, suppression, revival, or abstraction by themselves.

Acceptance:

- Unit test morphology computation on synthetic candidate sets:
  - same-parent / same-operator / exact
  - same-parent / mixed-operator / exact
  - mixed-parent / same-operator / near
  - mixed-parent / mixed-operator / near
- No change to audit decisions.
- No change to stress baseline except added diagnostics.

---

## P1.2 Track frontier history without consuming it

Add history fields to `TiedFrontier` so later collapse gating can be evidence-based.

Recommended additions:

```python
context_keys_seen: FrozenSet[int]
winner_history: Tuple[HypothesisKey, ...]
morphology_history: Tuple[TieMorphology, ...]
last_score_by_candidate: Mapping[HypothesisKey, int]
```

If full histories are too large, cap them with a small rolling window.

Do not use these fields yet to alter behavior.

Definitions:

- `context_keys_seen`: set of observed `near_tie_context_key` values while frontier persisted.
- `winner_history`: best candidate per audit while frontier persisted.
- `morphology_history`: morphology over time.
- `last_score_by_candidate`: latest observed score for each candidate.

Important:

`context_key` is an operational context signature, not proof of a hidden causal regime.

Do not call it regime proof.

Acceptance:

- Frontiers retain context history across repeated audits.
- Histories clear on invalidation and structural reset.
- Histories archive when collapse occurs.
- No behavior changes.
- Fast pytest passes.
- Regime receipt remains non-FAIL.

---

# P2 — decouple probe-pool generation from probe selection

This must happen before activating frontier-biased probes or adaptive budgets.

Problem:

Current probe selection is coupled to probe pool size and RNG trajectory. Changing which probes are selected can perturb downstream form discovery, making behavior changes hard to interpret.

Goal:

Generate a fixed candidate probe pool independent of the number of probes selected or the scoring policy used to rank them.

Required behavior:

```text
same seed + same hidden world + same audit context
→ same generated candidate probe pool

different selection policy
→ different selected probes only
→ no unrelated RNG drift
```

Implementation route:

- Use a local RNG or deterministic pool-generation seed for audit probe pool construction.
- Generate a fixed-size candidate pool independent of requested budget.
- Score/rank the fixed pool by a selection policy.
- Select top K after scoring.
- Do not let K change the generated pool.
- Do not let frontier-bias activation change the generated pool.
- Do not let adaptive-budget activation change the generated pool.

Acceptance:

- With frontier bias disabled, results match previous baseline.
- Same seed and hidden world produce identical candidate pools across:
  - baseline selection
  - frontier-biased selection
  - adaptive-budget selection
- Different selection policies change selected probes but not pool generation.
- Fast pytest passes.
- Stress baseline remains non-FAIL.
- If stress baseline changes, the change must be attributable to selected probes, not RNG drift.

---

# P3 — add separating-probe diagnostics before changing behavior

P3 computes disagreement information.

It should first run diagnostically.

Do not yet use it to change audit outcomes by default.

---

## P3.1 Add `separating_probes` and candidate predictions to diagnostics

Extend `FitDiagnostic` with:

```python
separating_probes: Tuple[Probe, ...]
frontier_candidate_preds: Tuple[CandidateProbePreds, ...]
frontier_disagreement_scores: Tuple[int, ...]
```

Use a compact representation if needed.

For each candidate probe in the fixed pool:

- evaluate predictions for frontier candidates
- compute disagreement count or prediction entropy
- identify probes that maximally split the frontier

Do not infer cause from this.

This only identifies where frontier candidates disagree.

Acceptance:

- `fit_var` can report separating probes when `frontier_candidates` are present.
- Diagnostics show at least:
  - candidate count
  - top separating probes
  - disagreement score per selected separating probe
  - candidate predictions on those probes
- With diagnostic-only mode, selected audit probes do not change.
- Fast pytest passes.
- Stress baseline unchanged except diagnostics.

---

## P3.2 Activate frontier-biased probe selection behind an explicit flag

After P3.1 is stable, add a flag:

```bash
--enable-frontier-probe-bias
```

Default must remain off.

When enabled:

- rank candidate probe pool partly by frontier disagreement
- choose probes that separate current frontier candidates
- preserve deterministic pool generation from P2

Do not combine this with adaptive budgets yet unless explicitly requested.

Acceptance:

- Flag off: behavior identical to baseline.
- Flag on: selected probes differ only through ranked selection, not pool RNG drift.
- Diagnostics show which probes were selected because of frontier bias.
- Stress run with flag on must be reported separately from baseline.

---

# P4 — conservative frontier collapse gating

Current problem:

`_collapse_tied_frontier` can collapse when the score landscape narrows to one candidate in a single audit.

That is not enough evidence that the ambiguity has been resolved.

P4 replaces immediate score-narrowing collapse with conservative retention.

Do this only after P1 and P3 diagnostics exist.

---

## P4.1 Define collapse evidence gates

A frontier may collapse only if all required gates pass.

Minimum proposed gates:

```text
stable_count >= regime_survival_threshold
len(context_keys_seen) >= min_context_signatures
dominant_candidate appears as winner across recent audits
separating probes no longer preserve meaningful ambiguity
```

Suggested defaults:

```python
regime_survival_threshold = 5
min_context_signatures = 2
winner_window = 5
dominance_required = 4
```

Caution:

These are heuristics.

They are not proof of hidden-regime equivalence.

Name them accordingly:

- `context_signatures_seen`, not `regimes_proven`
- `collapse_evidence_met`, not `equivalence_proven`

Acceptance:

- Collapse does not occur solely because one audit produces a single winner.
- Frontier candidates are retained when gates are unmet.
- Archived dormant alternatives include morphology/history metadata.
- Fast pytest passes.
- Stress baseline remains non-FAIL.

---

## P4.2 Retain unresolved prior candidates when current audit narrows

When a current audit reports only one near-tie candidate but collapse gates are unmet:

- keep prior frontier candidates as unresolved
- update their last known scores
- mark missing candidates as stale or unobserved, not dead
- do not discard them into dormant alternatives yet

Recommended state:

```python
candidate_state: Literal["active", "stale", "dominated", "collapsed"]
```

Rules:

- `active`: candidate remains in current near-tie set.
- `stale`: candidate was in prior frontier but did not appear in current near-tie set; collapse gates unmet.
- `dominated`: candidate repeatedly loses across evidence windows but collapse gates still pending.
- `collapsed`: candidate archived after collapse gates pass.

Acceptance:

- Prior frontier candidates are not lost on one narrowed audit.
- Dormant alternatives receive only candidates collapsed after gates pass.
- Diagnostics distinguish stale from collapsed.
- Fast pytest passes.
- Regime receipt remains non-FAIL.

---

# P5 — cause classification, after evidence exists

Do not implement cause classification until P1/P3/P4 exist.

Cause labels are evidential labels, not morphology labels.

Possible cause labels:

```python
TieCause = Literal[
    "unresolved",
    "under_probed",
    "operational_equivalence",
    "contextual_equivalence",
    "duplicate_handle_candidate",
    "structural_symmetry_candidate",
    "library_defect_candidate",
    "irrelevant_trass",
]
```

Important definitions:

- `unresolved`: default. Evidence insufficient.
- `under_probed`: no adequate separating probes have been generated or executed across relevant context signatures.
- `operational_equivalence`: candidates remain prediction-equivalent over the defined intervention/probe regime.
- `contextual_equivalence`: candidates are equivalent under some context signatures and separable under others.
- `duplicate_handle_candidate`: different handles behave interchangeably under validated contexts.
- `structural_symmetry_candidate`: same operator with different bindings remains interchangeable across validated contexts.
- `library_defect_candidate`: existing candidate library repeatedly fails to produce a stable, separating explanation across multiple variables/contexts; do not infer from one same-parent/mixed-operator tie.
- `irrelevant_trass`: candidates differ syntactically but role/intervention tests show the difference is irrelevant to downstream behavior in the current regime.

Cause-classification requirements:

- Must consume morphology.
- Must consume separating-probe diagnostics.
- Must consume frontier history.
- Must consume context signature history.
- Must record why a cause label was assigned.
- Must allow downgrade back to `unresolved`.

Acceptance:

- No cause label assigned from morphology alone.
- Every non-`unresolved` cause label includes evidence fields.
- Tests show same morphology can produce different cause labels under different probe histories.
- Fast pytest passes.
- Stress baseline remains non-FAIL.

---

# P6 — semantic suppression and dormant revival

Do not reconnect these before P4/P5.

---

## P6.1 Reconnect semantic suppression for frontier churn

Previous attempt:

`frontier_same_parent_churn` was implemented and removed because early near-tie sets were too large, making suppression too aggressive and perturbing form discovery timing.

Reconnect only when frontier evidence is mature.

Suppression gate:

```text
old_frontier.collapse_evidence_met OR
old_frontier.stable_count >= regime_survival_threshold
AND old_frontier.cause in allowed_suppression_causes
```

Allowed suppression causes should be conservative:

```python
allowed_suppression_causes = {
    "operational_equivalence",
    "contextual_equivalence",
    "irrelevant_trass",
}
```

Do not suppress on:

- raw near-tie membership
- morphology alone
- large early frontiers
- unresolved cause
- library-defect candidate
- structural-symmetry candidate before form validation

Acceptance:

- Canonical run shows fewer unnecessary `strong_observations` resets only for mature frontiers.
- Form discovery timing is not degraded.
- Stress baseline remains non-FAIL.
- Any PASS/PARTIAL/SKIP changes are reported.

---

## P6.2 Dormant alternative revival

`n.dormant_alternatives` currently accumulates collapsed candidates but nothing reads them.

Connect after collapse gates and cause labels exist.

Revival use cases:

- form recovery probe fallback
- re-audit warm-start after invalidation
- diagnostic display in `final_summary`

Revival priority should depend on cause:

- `contextual_equivalence`: revive when context signature changes.
- `operational_equivalence`: low priority unless downstream failure appears.
- `library_defect_candidate`: preserve as novelty evidence, not immediate candidate.
- `structural_symmetry_candidate`: route to form-binding diagnostics.
- `under_probed`: revive when separating probes become available.

Acceptance:

- Dormant candidates are not blindly reintroduced.
- Revival reason is logged.
- Re-audit warm-start can use dormant alternatives without changing baseline when disabled.
- Fast pytest passes.
- Stress baseline remains non-FAIL.

---

# P7 — abstraction discovery from validated equivalences only

This is design work until P5/P6 exist.

Do not feed raw ties into forms.

Do not feed raw near-ties into function-library extension.

Only validated, cause-classified equivalences may feed abstraction machinery.

---

## P7.1 Structural symmetry to form-binding hints

Candidate path:

- same operator
- different parent bindings
- persistent interchangeability across separating probes
- stable across context signatures
- cause classified as `structural_symmetry_candidate`

Then:

- emit binding-equivalence hint to forms
- do not auto-certify form
- require existing form validation path to accept/reject

Acceptance:

- Hints are diagnostics unless enabled by explicit flag.
- Form discovery must not degrade on baseline.
- A rejected hint does not corrupt frontier state.

---

## P7.2 Library-defect novelty

Candidate path:

- repeated frontier instability across variables/contexts
- no existing operator gives stable explanation
- targeted probes fail to resolve into any candidate in FUNC_LIBRARY
- cause classified as `library_defect_candidate`

Then:

- raise vocabulary novelty diagnostic
- do not auto-create new operator
- do not rewrite FUNC_LIBRARY automatically
- require separate novelty experiment before function extraction

Acceptance:

- Library-defect is never assigned from one variable alone.
- Novelty diagnostic includes evidence:
  - variables affected
  - morphology history
  - separating-probe failures
  - context signatures
  - candidate score traces

---

# P8 — put receipt/stress under normal test discipline

Problem:

`tests/test_regime_recall.py` is a standalone harness, not pytest coverage.

Add either:

- `tests/test_regime_recall_pytest.py` with one bounded smoke receipt, or
- a pytest marker `@pytest.mark.slow` wrapper around selected receipt cases.

Acceptance:

- `python3 -m pytest -q` covers at least one regime-recall smoke path.
- Slow/stress run remains manual:

```bash
python3 tests/test_regime_recall.py --stress
```

Every behavior-changing frontier patch must report:

```text
fast pytest result
receipt result
stress result
baseline comparison
known PASS/PARTIAL/SKIP/FAIL delta
```

---

# P9 — make docs truthful

## P9.1 Replace `dreth/STATE.md`

Current file is obsolete.

Replace with:

- active modules
- active CLI modes
- active default flags
- scaffold-only components
- intentionally disabled components
- validated commands
- current stress baseline

## P9.2 Replace `run_flowchart.md`

Use `run_flowchart_better.md` as replacement.

Keep it focused on current executable paths, not historical narrative.

Include:

- current audit path
- current frontier tracking path
- disabled frontier consumers
- current form discovery path
- current invalidation/reset behavior

---

# P10 — later work, not next implementation

These are not immediate tasks.

Do not implement until measurement, probe-pool decoupling, frontier diagnostics, and collapse gating are stable.

---

## P10.1 Adaptive budgets

Current state:

Adaptive probe budget exists but is disabled.

Do not activate raw budget scaling until P2 probe-pool decoupling is proven.

After P2:

- rank fixed pool
- select K by adaptive budget
- ensure K changes probe count only, not pool generation
- report separately from baseline

Acceptance:

- Same seed + same hidden world + different budget policy changes only probe count/selection.
- No unrelated form trajectory drift.
- Stress baseline with adaptive budget enabled is reported separately.

---

## P10.2 NN proposers

Current state:

NN proposers are passive diagnostics.

Either:

- wire scores into audit priority / role testing / candidate ordering behind an explicit flag, or
- rename as diagnostics and remove active-sounding claims.

Do not mix NN proposer activation with frontier-consumer activation in the same experiment.

---

## P10.3 Form-as-VarNethra

Only after measurement is clean and frontier/form interactions are stable.

Minimal future step:

- give `Form` an envelope over instance/template prediction deltas
- track form-level watch state
- propagate form watch state to instances

Do not build forms-of-forms until form-level envelope is real.

---

# Global acceptance invariant

For every change:

```bash
python3 -m pytest -q
python3 tests/test_regime_recall.py
python3 tests/test_regime_recall.py --stress
```

Report:

```text
changed files
behavioral intent
whether behavior is supposed to change
fast pytest result
receipt result
stress result
PASS/PARTIAL/SKIP/FAIL delta
explanation for any delta
```

No TODO item is complete if it only adds plausible terminology.

A TODO item is complete only when it changes one of:

- measured counters
- stored state
- diagnostic output
- probe selection
- collapse behavior
- suppression behavior
- revival behavior
- form/function downstream behavior

and the corresponding acceptance checks pass.
