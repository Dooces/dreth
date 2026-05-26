# TODO

## Current Read

- ✅ Core Dreth pipeline exists: world, audit, fit, sentinel, ledger, summaries, batch harness.
- ✅ Hybrid provider interfaces exist and remain advisory rather than authority-granting.
- ✅ Shadow/diagnostic layers exist for policy, residuals, authority/evidence, relative authority, temporal frontier, uncertainty governance, uncertainty consolidation, context-role indexing, authority strength, and background nethra.
- ✅ `ContextRoleIndex` records nethra nodes and context-indexed roles without changing behavior in record mode.
- ✅ The context-role model is conceptually corrected: a nethra is learned structure; `tareth`/`trass` are context-dependent roles, not the identity of the nethra.
- ✅ Strict context-role match gating, deduplication, attribution, and anchor policy plumbing have been added.
- ✅ Authority strength now has record/assist modes, a state controller, evidence debt, derivation policies, and comparison-suite support.
- ✅ BackgroundNethraIndex exists to record familiar-but-non-authoritative structure.
- ⚠️ Background nethra record/export behavior must be smoke-tested through the comparison suite before building on it.
- ⚠️ Memory/sleep consolidation is the next conceptual bridge if background-nethra record mode is behavior-neutral and produces nonzero familiar structure.
- ⚠️ Runtime assist layers have repeatedly failed by turning broad visible uncertainty into excess attention/repair pressure. Treat broad uncertainty as debt/provenance first, not automatic work.
- ⚠️ Delayed causality and temporal structure remain underrepresented. A passive temporal ledger is relevant, but should follow memory/sleep scaffolding unless a test specifically demands temporal instrumentation first.

---

## Immediate Next Work

- [ ] Run the `background_nethra` comparison suite.
- [ ] Verify `off == record` remains true.
- [ ] Verify `background_nethra_records > 0`.
- [ ] Verify `familiar_background_count > 0`.
- [ ] Verify `operational_authority_count == 0`.
- [ ] If background-nethra record mode passes, implement the offline MemorySleepConsolidator in Appendix A.
- [ ] Do not wire sleep proposals into runtime behavior in the first pass.
- [ ] Do not proceed to TemporalEventLedger until either MemorySleepConsolidator is complete or a concrete temporal-only diagnostic need is identified.

Suggested background-nethra suite command:

```bash
mkdir -p reports

python scripts/run_comparison_suite.py \
  --suite background_nethra \
  --suite-workers 2 \
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
  --authority-strength record \
  --out-prefix reports/background_nethra_compare
```

---

## Deferred / Reference Work

- [ ] MemorySleepConsolidator / offline scaffold persistence. See Appendix A. This is next if background-nethra record mode passes.
- [ ] TemporalEventLedger / passive temporal observers for delayed causality scaffolding. See Appendix B.
- [ ] Weighted intervention-cost tracking is low-risk and can be cherry-picked later as diagnostic-only.
- [ ] Tied-frontier separating probes are likely useful, but should wait until current authority/context-role/background indexing remains stable.
- [ ] Dormant alternative revival is conceptually aligned, but should wait until retrieval gates are strict enough.
- [ ] Regime-triggered inert re-screening may be valuable, but should not be mixed with authority-action narrowing or background-nethra validation.
- [ ] Cheap cascade pre-checks are higher-risk because they alter re-audit behavior. Do not implement before attribution tools are stable.
- [ ] Learned ranker/factorizer is not next. Use it only after deterministic match gating, authority narrowing, background memory, and temporal observers show that rule-based retrieval cannot separate useful anchors from broad noise.

---

## Non-Negotiable Invariants

- [ ] Authority is earned by evidence, not provider confidence, graph proximity, morphology, index membership, recurrence, sleep proposal, or temporal correlation.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, temporal observers, sleep consolidation, or assist logic.
- [ ] Shadow/diagnostic/passive layers may observe; they must not mutate authority unless explicitly promoted through a separately tested bounded runtime path.
- [ ] `ContextRoleIndex` records provenance and role history; it is not truth.
- [ ] Background nethras are familiar structure, not operational authority.
- [ ] Sleep proposals are scaffold proposals, not authority, not certs, not revocations, and not skip suppressors.
- [ ] `tareth`/`trass` are context roles, not global properties of a nethra.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, generic uncertainty signals, or repeated background familiarity must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Passive temporal observers must be bounded by caps/ring buffers/summaries; do not create unbounded event-history growth.

---

# Appendix A — Deferred Codex Prompt: MemorySleepConsolidator / Offline Scaffold Persistence

Use this after the background-nethra comparison suite passes. The purpose is to let Dreth consolidate familiar background structure while the program is not running, without laundering that familiarity into authority.

```text
You are working in the Dooces/dreth repo.

Current context:
Dreth now has BackgroundNethraIndex for passive familiar structure.
A nethra can be learned even when its current role is trass/unresolved/quarantined.
The next question is whether offline memory consolidation can group these passive records into higher scaffold proposals without changing runtime authority.

Goal:
Add an offline MemorySleepConsolidator.

Purpose:
Read exported runtime memory after a run and build scaffold proposals:
  - background nethra groups
  - trass families
  - unresolved/tied-frontier families
  - dormant alternative families
  - authority-debt families
  - uncertainty subclusters
  - context-role recurrence groups
  - possible temporal cohorts if temporal records exist

Core invariant:
Offline sleep may create proposals only.
It must not issue authority.
It must not revoke authority.
It must not suppress skips.
It must not replace fit.
It must not increase monitoring.
It must not increase repair priority.
It must not use hidden truth/debug manifest fields.
It must not treat recurrence/frequency as proof.

Add module:
  dreth/memory_sleep.py

Define:

ScaffoldProposal:
  proposal_id
  kind
  source_record_ids
  vars
  contexts
  common_signatures
  common_parents
  role_patterns
  recurrence_count
  runs_seen
  seeds_seen
  first_seen_cycle
  last_seen_cycle
  confidence_as_familiarity
  authority_allowed: bool = False
  suggested_runtime_use:
    - no_runtime_use
    - feature_only
    - clustering_prior
    - ranking_hint
  evidence_summary
  warnings

MemorySleepSummary:
  input_rows
  proposals
  proposals_by_kind
  avg_sources_per_proposal
  largest_proposals
  authority_allowed_count
  hidden_truth_fields_seen
  warning_count

MemorySleepConsolidator:
  load_jsonl_rows(...)
  extract_background_records(...)
  extract_context_role_records(...)
  extract_uncertainty_records(...)
  extract_authority_debt_records(...)
  extract_temporal_records_if_available(...)
  build_proposals(...)
  summarize(...)

Grouping rules:
Use visible/exported fields only:
  - vars overlap
  - contexts overlap
  - role pattern overlap
  - parent/signature overlap
  - recurring signal overlap
  - source kind overlap
  - temporal proximity if available

Do not use:
  truth_parents
  truth_func
  truth_delayed_parents
  truth_latents
  debug_blind_challenge_manifest
  relation_type except in a separate post-hoc report section

Add script:
  scripts/run_memory_sleep.py

CLI:
  python scripts/run_memory_sleep.py \
    --jsonl reports/background_nethra_compare_record.jsonl \
    --out reports/memory_sleep_proposals.jsonl \
    --summary reports/memory_sleep_summary.txt

Output:
  proposals JSONL
  human summary

Summary sections:
A. Inputs
  rows read
  background records
  context-role records
  uncertainty records
  authority-debt records
  temporal records if present

B. Scaffold proposals
  count
  by kind
  avg source records per proposal
  largest proposals

C. Familiarity not authority
  authority_allowed_count must be 0
  suggested_runtime_use distribution

D. Examples
  top proposals with vars, contexts, roles, signatures, sources, warnings

E. Hidden-truth guard
  hidden truth fields ignored
  report if any hidden-truth-like fields were present in input

F. Warning
  Sleep proposals are familiarity scaffolds only.
  They are not operational authority.

Optional:
  Add --posthoc-relation-type-report
  This may use relation_type/debug fields only for offline interpretation.
  It must be disabled by default.

Do not wire sleep proposals into ChainedAgent in this task.
Do not add --scaffold-memory runtime loading yet.

Tests:
  - visible background records produce scaffold proposals
  - repeated trass records group together
  - unresolved/tied-frontier records group together
  - quarantined authority-debt records group together
  - unrelated records remain separate
  - authority_allowed_count is always 0
  - hidden truth fields are ignored
  - relation_type is not used unless posthoc mode is explicitly enabled
  - empty input produces empty summary
  - proposals contain provenance source ids
  - no imports from agent.py
  - no runtime behavior changes

Verification:
  python -m pytest tests/test_memory_sleep.py -q
  python -m pytest tests/test_background_nethra.py -q
  python -m pytest tests/test_cycle_mechanics.py -q
  git diff --check

Smoke:
  First run the background_nethra comparison suite.
  Then run:

  python scripts/run_memory_sleep.py \
    --jsonl reports/background_nethra_compare_record.jsonl \
    --out reports/memory_sleep_proposals.jsonl \
    --summary reports/memory_sleep_summary.txt

Expected:
  proposals > 0 if background records exist
  authority_allowed_count = 0
  proposal examples show grouped trass/unresolved/quarantined/background structure
  no runtime behavior change
```

## Appendix A Warnings

- Do not implement this before the background-nethra comparison suite passes.
- Do not let offline grouping become authority.
- Do not treat recurrence/frequency as proof.
- Do not use hidden truth/debug manifest fields in sleep consolidation.
- Do not wire sleep proposals into `ChainedAgent` in the first pass.
- Do not allow sleep proposals to issue certs, revoke, suppress skips, replace fits, increase monitoring, or increase repair priority.
- Do not call this runtime learning. It is offline scaffold/proposal generation.
- Do not store unbounded memory. Use summaries, caps, provenance IDs, and bounded examples.

---

# Appendix B — Deferred Codex Prompt: TemporalEventLedger and Passive Temporal Observers

Use this after the authority-action narrowing comparison has been run and interpreted, and after the background/sleep path is stable unless a concrete temporal-only diagnostic need appears. The temporal layer is relevant because delayed causality and lagged structure are underrepresented, but adding it before validating current memory scaffolding would blur attribution.

```text
You are working in the Dooces/dreth repo.

Current diagnosis:
Dreth now has ContextRoleIndex, uncertainty consolidation, strict context-role gating, background nethra, and authority strength machinery.
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

## Appendix B Warnings

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
|R:BackgroundNethraSmokePending,MemorySleepNowTopTodo,AuthorityLaunderingRisk,TemporalPromptDeferredButRelevant,UnboundedEventLogRisk
|F:run-background-suite-then-implement-offline-memory-sleep-consolidator-before-temporal-runtime-work
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
