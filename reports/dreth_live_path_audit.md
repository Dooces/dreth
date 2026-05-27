# Dreth Live Path Audit

This audit identifies the current live runtime path and the changes required
to make persistent nethra memory live without granting authority.

## dreth.agent.ChainedAgent

- `__init__`: live runtime. Constructs ledger, counters, optional context role,
  background, authority-strength, scaffold memory surfaces. Existing scaffold
  mode is telemetry/feature-oriented and only partially reaches ordering hooks.
  Reuse by adding a loaded persistent-memory index and explicit use-right mode.
  Test: next run loads memory and record/off equivalence holds.
- `_screen_candidate_parents`: live runtime candidate parent consideration.
  It observes current visible state by intervention probes, orders candidate
  parents, applies route-cert exclusions, then returns the parent slice used by
  `fit_var`. It already has scaffold ranking hooks, but provider ordering is
  mostly lost when converted to a set and inline scaffold effects are secondary.
  Reuse by applying persistent nethra use-rights here before the top-M slice and
  recording behavior attribution. Test: ranking_hint reorders existing parent
  candidates only; record mode does not change the returned set.
- `_full_audit_var`: live runtime audit/probe path. It chooses available parents,
  builds forced probe lists from frontier/provider/consolidation hints, calls
  `fit_var`, then records `FitDiagnostic`. Reuse by letting loaded nethras
  propose probe ordering through use-rights before `fit_var`. Test: probe_hint
  changes probe ordering only in assist mode and increments behavior effects.
- `run_cycle`: live runtime dispatcher. Observes world changes through sentinels,
  passive residuals, composites, regimes, and full audits. It installs audit
  results through `_install_var`, records `CycleRecord`, and updates ledger state.
  Reuse by exporting experience events gathered by candidate/probe hooks. Test:
  assist effects are attributed and authority effects remain separate.
- `_install_var`: live runtime ledger update. Installs learned parents/function,
  route certs, tied frontiers, compressions, context roles, sentinels, and drift
  state. Reuse for runtime memory record generation through existing context role
  and background exports. Test: runtime writes nethra handles with primitive,
  member, and structure backpointers.
- `_context_role_index_record_var_fit`, `_context_role_index_assign_role`,
  `_context_role_index_record_candidate`: live runtime record-only provenance.
  Reuse to produce persistent nethra handles and role history. Test: same nethra
  can be trass in one context and tareth in another.
- `_run_background_nethra`: live runtime record-only summarization. Reuse as a
  source for persistent memory; it must not become authority. Test: trass is not
  deletion and background records have zero authority effects.
- `scaffold_memory_metrics`, `context_role_index_metrics`,
  `background_nethra_metrics`: live metric writers. Reuse to expose loaded,
  used, behavior_effect, and authority_effect counters. Test: batch metrics
  include persistent_nethras_loaded/used and behavior attribution.

## dreth.fit

- `enumerate_var_hypotheses_restricted`: live runtime candidate enumeration.
  It enumerates hypotheses from the available parent set. It can be reused but
  should not read memory directly; memory applies before this through use-rights.
  Test: ranking_hint cannot add candidates.
- `fit_var`: live runtime fit/audit/probe engine. It selects discrimination
  probes, scores hypotheses, and emits morphology diagnostics. Reuse by adding a
  probe-order input only if needed; current forced probe input is sufficient for
  probe_hint. Test: memory probe hints cannot certify and do not affect authority.

## dreth.ledger

- `VarNethra`, `NethraCertificate`, `TiedFrontier`, `CompositeNethra`,
  `HyperCompositeNethra`, `ChainedLedger`: live runtime authority record state.
  Reuse only as visible evidence/backpointers for persistent nethra records.
  Loaded memory must not write certificates or mutate authority. Test: sleep and
  loaded memory cannot create authority or hard filters.

## dreth.context_role_index

- `NethraNode`, `ContextRoleRecord`, `ContextRoleIndex`: live record/assist
  feature surface. It stores nethra graph nodes, role histories, and local
  matching metadata. Reuse for persistent handles and role backpointers. Current
  assist pressure is limited to uncertainty anchors, not the main parent/probe
  path. Test: grouped nethra preserves primitive/member/structure backpointers.

## dreth.background_nethra

- `BackgroundNethraIndex`: live record-only familiarity over recurring structure.
  It does not issue authority and exports useful runtime-visible records. Reuse
  as a source of persistent nethra records and salience evidence. Test: loaded
  familiarity is not authority.

## dreth.scaffold_memory

- `ScaffoldMemoryIndex`: runtime loader for offline proposals. Currently marked
  as familiarity/provenance telemetry, with metrics forcing behavior_effects=0.
  It can rank existing candidates in assist_feature mode but does not provide a
  full use-right enforcement path. Reuse and modify to load sleep products,
  enforce use-rights, expose behavior/authority metrics, and reject hard_filter
  from sleep. Test: sleep products load next run and affect assist only through
  allowed use-rights.

## dreth.nethra_memory_store

- `NethraMemoryStore`: persistence/offline storage. Existing records are coarse
  provenance records and run summaries; it does not persist the required
  experience-event loop. Reuse JSONL append/load mechanics and add actual
  nethra handle and experience event rows. Test: Run A writes nethra memory and
  ExperienceEvents; Run B loads them.

## dreth.memory_sleep

- `MemorySleepConsolidator`: offline sleep. It consumes exported background,
  context-role, authority, uncertainty, and temporal records, then emits
  proposal-only scaffold proposals. It currently does not consume
  ExperienceEvents as first-class input. Reuse by adding experience/memory
  extraction and SleepProduct output with authority_allowed=false and no
  hard_filter. Test: sleep consumes ExperienceEvents and emits proposal-only
  grouped products with backpointers.

## dreth.nethra_scaffold_sleep

- `NethraScaffoldSleep`: offline scaffold abstraction over visible records. It
  preserves role maps and composition records without authority. Reuse as a
  reference for grouped backpointer preservation; do not make it the live path.
  Test: grouped proposal keeps primitive/member/structure backpointers.

## dreth.world

- `CausalWorld`: live oracle observed only through visible state and
  interventions. Hidden parents/functions/logs are diagnostic only. Reuse only
  via the existing agent intervention methods. Test: hidden truth/debug fields
  ignored by runtime and sleep.

## scripts.batch_run

- `_run_one`: runtime setup. It constructs world, optional memory/scaffold
  loaders, `ChainedAgent`, then runs cycles. Reuse by loading persistent memory
  and sleep products before agent initialization. Test: Run B loads persistent
  memory before runtime.
- `_extract_arch_metrics` and JSONL writer: metric/export path. Reuse to persist
  nethra memory records and experience events after runs. Test: batch output
  includes loaded/used/effect counters.
- CLI `--nethra-memory`: currently record-only. Modify to support record and
  assist loading while preserving off/record behavior. Test: record/off
  equivalence.

## scripts.run_memory_sleep

- Offline consolidator CLI. Reuse to consume persistent memory JSONL and emit
  SleepProducts. Test: sleep output can be loaded next run.

## scripts.run_nethra_scaffold_sleep

- Offline scaffold sleep CLI. Diagnostic/offline. Reuse only if needed for
  scaffold abstractions, not for live authority. Test: no authority/hard_filter
  emitted.

## scripts.summarize_scaffold_memory

- Diagnostic summary. Reuse for reports only; not a behavior path. Test: no
  claims of authority or completion.

## Tests To Prove The Changes

- Persistence: runtime writes `NethraMemoryRecord` and `ExperienceEvent`; next
  run loads them; loaded nethra affects assist only.
- Sleep: consumes events and memory records; emits `SleepProduct` rows with
  `authority_allowed=false`; rejects sleep hard_filter; preserves backpointers.
- Live consideration: active atoms retrieve candidate nethras; salience ranks
  specific useful handles above frequent broad ones; parent/probe order changes
  are attributed.
- Use-right: record_only and feature_only do not change behavior; ranking_hint
  and probe_hint reorder only; soft_filter preserves fallback; hard_filter
  requires local evidence and is rejected for sleep.
- Hidden truth: runtime and sleep ignore truth/debug fields; hidden data appears
  only in offline evaluation fields.
