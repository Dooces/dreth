# TODO

## Current Read

- ✅ Core Dreth pipeline exists: world, audit, fit, sentinel, ledger, summaries, batch harness.
- ✅ Hybrid provider interfaces exist and remain advisory rather than authority-granting.
- ✅ Shadow/diagnostic layers exist for policy, residuals, authority/evidence, relative authority, temporal frontier, uncertainty governance, uncertainty consolidation, context-role indexing, and authority strength.
- ✅ `ContextRoleIndex` records nethra nodes and context-indexed roles without changing behavior in record mode.
- ✅ The context-role model is conceptually corrected: a nethra is learned structure; `tareth`/`trass` are context-dependent roles, not the identity of the nethra.
- ✅ Strict context-role match gating, deduplication, attribution, and anchor policy plumbing have been added.
- ✅ Authority strength now has record/assist modes, a state controller, evidence debt, derivation policies, and comparison-suite support.
- ⚠️ Authority-action narrowing has just been implemented and must be tested before adding new runtime machinery.
- ⚠️ Runtime assist layers have repeatedly failed by turning broad visible uncertainty into excess attention/repair pressure. Treat broad uncertainty as debt/provenance first, not automatic work.
- ⚠️ Delayed causality and temporal structure remain underrepresented. A passive temporal ledger is relevant, but should be added only after authority-action narrowing is validated.

---

## Immediate Next Work

- [ ] Run the six-way authority-strength comparison suite after authority-action narrowing.
- [ ] Verify `off == record` remains true.
- [ ] Verify `assist_legacy` still reproduces the old broad-pressure failure.
- [ ] Verify `assist_state_shadow` no longer blows up from generic contested records.
- [ ] Compare `assist_quarantine_persistent` and `assist_quarantine_repair_only` against `off` on IV, quality cost, audits, revocations, unique failures, regime sentinel failures, and passive stress.
- [ ] Confirm generic contested / giant-cluster authority records become debt/no-op rather than monitoring/repair work.
- [ ] If all assist modes still worsen behavior, keep authority strength as record/state reporting only and do not add new runtime assist layers.
- [ ] If narrowed authority assist becomes neutral or useful, then proceed to the passive temporal event ledger in Appendix A.

Suggested command:

```bash
mkdir -p reports

python scripts/run_comparison_suite.py \
  --suite authority_strength \
  --suite-workers 3 \
  --summary-workers 4 \
  --workers 8 \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 100 \
  --cycles 10000 \
  --seeds 42,99,7 \
  --hybrid-control interfaces \
  --repair-agenda \
  --parent-ranker history_rescue \
  --probe-proposer history_rescue \
  --uncertainty-consolidation assist \
  --uncertainty-assist-policy local_only \
  --context-role-index assist_feature \
  --out-prefix reports/authority_action_narrowing_compare
```

---

## Deferred / Reference Work

- [ ] TemporalEventLedger / passive temporal observers for delayed causality scaffolding. See Appendix A.
- [ ] Weighted intervention-cost tracking is low-risk and can be cherry-picked later as diagnostic-only.
- [ ] Tied-frontier separating probes are likely useful, but should wait until current authority/context-role assist attribution remains stable.
- [ ] Dormant alternative revival is conceptually aligned, but should wait until retrieval gates are strict enough.
- [ ] Regime-triggered inert re-screening may be valuable, but should not be mixed with authority-action narrowing.
- [ ] Cheap cascade pre-checks are higher-risk because they alter re-audit behavior. Do not implement before attribution tools are stable.
- [ ] Learned ranker/factorizer is not next. Use it only after deterministic match gating, authority narrowing, and temporal observers show that rule-based retrieval cannot separate useful anchors from broad noise.

---

## Non-Negotiable Invariants

- [ ] Authority is earned by evidence, not provider confidence, graph proximity, morphology, index membership, or temporal correlation.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, temporal observers, or assist logic.
- [ ] Shadow/diagnostic/passive layers may observe; they must not mutate authority unless explicitly promoted through a separately tested bounded runtime path.
- [ ] `ContextRoleIndex` records provenance and role history; it is not truth.
- [ ] `tareth`/`trass` are context roles, not global properties of a nethra.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, or generic uncertainty signals must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Temporal proposals are not authority, not certs, not revocations, and not skip suppressors.
- [ ] Passive temporal observers must be bounded by caps/ring buffers/summaries; do not create unbounded event-history growth.

---

# Appendix A — Deferred Codex Prompt: TemporalEventLedger and Passive Temporal Observers

Use this only after the authority-action narrowing comparison has been run and interpreted. The temporal layer is relevant because delayed causality and lagged structure are underrepresented, but adding it before validating authority-action narrowing would blur attribution.

```text
You are working in the Dooces/dreth repo.

Current diagnosis:
Dreth now has ContextRoleIndex, uncertainty consolidation, strict context-role gating, and shadow/assist machinery.
But delayed causality and temporal structure are still underrepresented.
CycleRecord exists, but it is too thin: it records per-cycle skipped/audited/drift/novelty fields, not a rich event timeline with salience, interventions, lagged effects, or delayed-causality candidates.

Goal:
Add a TemporalEventLedger and passive temporal observers.

Purpose:
Track when salient events happen, which variables were involved, what intervention context existed, and what later events may be temporally related.
Allow passive observers to build scaffold for delayed causality without intervening and without issuing authority.

Core invariant:
Default behavior unchanged.
No hidden truth in runtime.
Passive observers cannot intervene.
Passive observers cannot issue certs.
Passive observers cannot revoke authority.
Passive observers cannot suppress skips.
Passive observers cannot replace fit_var.
Passive observer output is proposal/metadata only.

Add module:
  dreth/temporal_event_ledger.py

Define:

TemporalEvent:
  cycle
  event_type:
    - intervention
    - sentinel_failed
    - sentinel_passed
    - fit_changed
    - fit_repaired
    - revocation
    - novelty_opened
    - novelty_resolved
    - frontier_opened
    - frontier_updated
    - context_role_changed
    - uncertainty_cluster_seen
    - authority_strength_changed
    - passive_residual_stressed
    - passive_residual_ok
  vars
  source_vars
  target_vars
  intervention_var
  intervention_value
  context_key
  nethra_ids
  role_before
  role_after
  fit_signature_before
  fit_signature_after
  local_graph_neighbors
  payload

TemporalEventLedger:
  add_event(event)
  events_in_window(start_cycle, end_cycle)
  events_for_var(var, window=None)
  events_near_vars(vars, window=None)
  lagged_events(source_var, target_var, min_lag, max_lag)
  summarize()

Add module:
  dreth/passive_temporal_observers.py

Define protocol:
  PassiveTemporalObserver:
    observe_event(event, ledger)
    observe_cycle(cycle, state, intervention, ledger)
    proposals()

Observers:
  RollingWindowObserver:
    tracks rolling mean/variance/residual shifts per var.

  LaggedCorrelationObserver:
    tracks whether changes/failures in target vars repeatedly follow interventions or changes in source vars at lag k.

  DelayedResidualObserver:
    tracks whether current residual stress is better explained by past source values/interventions.

  EventCohortObserver:
    tracks groups of vars that repeatedly fail/repair/change within temporal windows.

Proposal dataclass:
  TemporalProposal:
    proposal_type:
      - delayed_parent_candidate
      - lag_window_candidate
      - shared_temporal_cohort
      - delayed_residual_explanation
      - monitoring_candidate
    source_vars
    target_vars
    lag
    score
    evidence_summary
    cycles_seen

Runtime integration:
Add CLI:
  --temporal-events off|record
  --passive-temporal-observers off|basic

Defaults:
  off

record:
  records TemporalEvents only; no behavior change.

basic:
  records TemporalEvents and runs passive observers; observer proposals are exported and may be read by offline summaries only in this pass.

Do not yet feed proposals into runtime authority, fit, or skip behavior.

Where to emit events:
  - each intervention
  - sentinel pass/failure
  - fit changed / repaired
  - revocation
  - novelty opened/resolved
  - tied frontier opened/updated/collapsed
  - context-role role changes
  - uncertainty cluster creation/update
  - passive residual stress/ok if available

Reports:
  scripts/summarize_temporal_events.py

Sections:
A. event counts by type
B. most salient vars by event count
C. intervention-to-failure lag candidates
D. repeated delayed-parent candidates
E. temporal cohorts
F. delayed residual candidates
G. warning: passive proposals are not authority

Metrics:
  temporal_events_total
  temporal_event_types
  passive_temporal_proposals
  delayed_parent_candidates
  lag_window_candidates
  temporal_cohorts
  delayed_residual_candidates

Tests:
  - off mode preserves behavior
  - record mode preserves behavior
  - basic observer mode preserves behavior
  - intervention event includes intervention var/value
  - sentinel failure event records target var and cycle
  - lagged observer detects synthetic source→target lag
  - observer proposals do not issue/revoke certs
  - observer proposals do not suppress skips
  - no hidden truth/debug manifest read in runtime observers

Verification:
python -m pytest tests/test_temporal_event_ledger.py -q
python -m pytest tests/test_passive_temporal_observers.py -q
python -m pytest tests/test_cycle_mechanics.py -q
python -m pytest tests/test_blind_challenge.py -q

Smoke:
python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50 \
  --cycles 3000 \
  --seeds 42,99,7 \
  --hybrid-control interfaces \
  --repair-agenda \
  --temporal-events record \
  --passive-temporal-observers basic \
  --out reports/temporal_events_basic.jsonl \
  2>&1 | tee reports/temporal_events_basic.log

python scripts/summarize_temporal_events.py \
  --jsonl reports/temporal_events_basic.jsonl \
  | tee reports/temporal_events_basic_summary.txt

Expected:
  - invariants pass
  - off/record/basic behavior remains unchanged except diagnostics
  - report shows whether delayed-parent/lag candidates appear
  - no authority behavior changes
```

---

## Appendix A Warnings

- Do not implement this before the authority-action narrowing comparison is interpreted.
- Do not let temporal correlation become authority.
- Do not let passive temporal observers intervene, revoke, issue certs, suppress skips, or replace `fit_var`.
- Do not read hidden truth/debug manifest fields in temporal observers.
- Do not store unbounded event history. Use ring buffers, rolling summaries, caps, or windowed indexes.
- Do not feed temporal proposals into runtime behavior in the first implementation pass.
- Do not treat delayed-parent candidates as causal parents; they are proposal metadata only.
- Do not mix this with learned factorizer/ranker work. Temporal observers should first expose whether delayed scaffolding exists in visible evidence.

---

## Ledger

```text
[D:O!,N,V,E,B,C,A,L,S,Q,M,P0,Cav↓,R0,NB,LG
|R:AuthorityAnxietyPatternUnderRetest,GenericContestedOveractionPatchPending,TemporalPromptDeferredButRelevant,RegimeSentinelFragility,TemporalAttributionRisk,UnboundedEventLogRisk
|F:test-authority-action-narrowing-before-temporal-event-ledger-append-temporal-prompt-as-deferred-reference
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
