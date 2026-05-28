# TODO

This file lists remaining work only. Completed baseline items are intentionally not repeated as TODO entries.

Steps 1, 2, and 3 of the nethra role-surface model are complete. Remaining work is deferred until the role-surface/residual system is measured.

---

# Step 3 — Peripheral Residual Classification, Regime Candidates, and Bounded Assist Experiments (COMPLETE)

All three steps are complete and verified. See git history for implementation details.

---

## Non-Negotiable Invariants

- [ ] A nethra is a learned operating handle/lens over structure, not the structure itself.
- [ ] Multiple nethras may touch the same structure.
- [ ] Structure overlap is not proof.
- [ ] Cross-context overlap is downgraded to hint/proposal until local evidence earns stronger use.
- [ ] `tareth`/`trass` are context roles/surfaces, not global identities.
- [ ] Trass is not deletion.
- [ ] Trass is not dormant/inactive; it organizes residuals while non-load-bearing.
- [ ] Tareth constrains focal reasoning only where earned.
- [ ] Best-available is fallback, not hard authority.
- [ ] Unresolved preserves ambiguity; it does not license action authority.
- [ ] Residual pressure is not authority.
- [ ] Residual pressure should decline into clarity in stable closed contexts.
- [ ] Persistent unresolved residual growth in a stable closed context is a warning/failure signal.
- [ ] Regime candidates are not regime switches.
- [ ] Regimes, if used later, are emergent active expressions over co-active nethras, not global enum modes.
- [ ] A composed nethra/expression must not inherit the strongest authority of its members.
- [ ] Authority is earned by visible evidence, not provider confidence, graph proximity, morphology, index membership, recurrence, sleep proposal, pressure, or temporal correlation.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, temporal observers, sleep consolidation, residual classification, or assist logic.
- [ ] Shadow/diagnostic/passive layers may observe; they must not mutate authority unless explicitly promoted through a separately tested bounded runtime path.
- [ ] Record-only indexes must match off-mode behavior.
- [ ] Assist-feature paths must be attributed: which handle/surface/expression changed ordering/probes/filters and whether the outcome improved.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, generic uncertainty signals, or repeated background familiarity must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Passive temporal observers must be bounded by caps/ring buffers/summaries; do not create unbounded event-history growth.
- [ ] Representative residual examples must be capped.
- [ ] Residual classification must not become a second unbounded audit loop.

---

## Deferred / Reference Work After the Three Steps

These are not immediate implementation targets. Do not start them until the role-surface/residual system is implemented and measured.

- [ ] `NethraExpressionIndex` / offline expression mining over role surfaces and compacted nethra identities.
- [ ] `ActiveSlice` compiler from expressions/surfaces into bounded runtime rank/probe/filter/block surfaces.
- [ ] Recognition-collapse metrics over active surface coverage.
- [ ] TemporalEventLedger / passive temporal observers for delayed causality scaffolding.
- [ ] Weighted intervention-cost tracking as diagnostic-only.
- [ ] Dormant alternative revival through strict retrieval/expression gates.
- [ ] Regime-triggered inert re-screening, but not as a global regime switch.
- [ ] Learned ranker/factorizer only after deterministic surface/expression attribution shows rule-based retrieval cannot separate useful anchors from broad noise.

---

# Appendix A — Deferred Prompt: NethraExpressionIndex / Offline Expression Mining

Use only after Step 1 through Step 3 are complete and measured.

```text
You are working in the Dooces/dreth repository.

Current context:
Dreth distinguishes shared structure from nethras. A nethra is a scoped, evidence-bearing, context-activated operating handle over shared structure. Multiple nethras can touch the same structure. Roles such as tareth/trass are context-indexed operating surfaces, not identities. Regimes should emerge as active expressions over co-active nethras, not as predeclared world labels.

Goal:
Add an offline NethraExpressionIndex design/implementation pass.

Purpose:
Read exported runtime/scaffold/mind/surface memory and build proposal-only expressions over overlapping nethras:
  - overlap bridges
  - subset/superset relations
  - union expressions
  - intersection expressions
  - difference expressions
  - gated activations
  - negative gates
  - coactivation clusters
  - recognition-collapse candidates
  - emergent-regime candidates
  - active-slice candidates

Core invariant:
Offline expression mining may create proposals only.
It must not issue authority.
It must not revoke authority.
It must not suppress skips.
It must not replace fit.
It must not increase monitoring.
It must not increase repair priority.
It must not use hidden truth/debug manifest fields.
It must not treat recurrence/frequency/overlap as proof.
It must not create global regime switches.
It must not assign hard_filter use-rights from sleep alone.

Add module:
  dreth/nethra_expression_index.py

Define:

NethraExpressionProposal:
  expression_id
  expression_kind:
    - overlap_bridge
    - subset_relation
    - union_expression
    - intersection_expression
    - difference_expression
    - gated_activation
    - negative_gate
    - coactivation_cluster
    - recognition_collapse_candidate
    - emergent_regime_candidate
    - active_slice_candidate
  member_nethra_ids
  touched_structure_ids
  vars
  contexts
  common_signatures
  common_source_edges
  operation:
    - union
    - intersection
    - difference
    - gated
    - negative_gate
    - coactive
    - recognition_collapse
  activation_gate
  negative_gate
  source_record_ids
  runs_seen
  seeds_seen
  first_seen_cycle
  last_seen_cycle
  evidence_summary
  suggested_use_right:
    - record_only
    - feature_only
    - ranking_hint
    - probe_hint
    - soft_filter_candidate
    - block_candidate
  invalidators
  authority_allowed: bool = False
  warnings

NethraExpressionSummary:
  input_rows
  structure_nodes_seen
  nethra_handles_seen
  expressions
  expressions_by_kind
  expressions_by_suggested_use
  overlap_bridges
  subset_relations
  gated_activations
  negative_gates
  coactivation_clusters
  recognition_collapse_candidates
  emergent_regime_candidates
  authority_allowed_count
  hidden_truth_fields_seen
  warning_count

NethraExpressionIndexBuilder:
  load_jsonl_rows(...)
  extract_structure_records(...)
  extract_nethra_handles(...)
  build_overlap_bridges(...)
  build_subset_relations(...)
  build_union_intersection_candidates(...)
  build_gated_activation_candidates(...)
  build_negative_gate_candidates(...)
  build_coactivation_clusters(...)
  build_recognition_collapse_candidates(...)
  build_active_slice_candidates(...)
  summarize(...)

Use visible/exported fields only:
  - nethra id
  - vars/components touched
  - contexts/context families
  - role histories/surfaces
  - source_edge/signature overlap
  - recurring signal overlap
  - source kind overlap
  - sentinel/revocation/failure counters if visible
  - coactivation across cycles/runs
  - residual bucket summaries
  - temporal proximity if temporal records exist

Do not use:
  - truth_source_edges
  - truth_func
  - truth_delayed_source_edges
  - truth_latents
  - debug_blind_challenge_manifest
  - relation_type except in a separate post-hoc report section disabled by default

Runtime boundary:
Do not wire expressions into ChainedAgent in this task.
Do not add behavior-changing use.
Output proposals and a summary only.

Add script:
  scripts/run_nethra_expression_sleep.py

Summary sections:
A. Inputs
B. Expressions
C. Overlap and gates
D. Recognition/regime candidates
E. Familiarity not authority
F. Examples
G. Hidden-truth guard
H. Warning that expression proposals are search/familiarity scaffolds only

Tests:
  - repeated overlap creates overlap_bridge proposal
  - subset touched-structure relation creates subset_relation proposal
  - coactive nethras create coactivation_cluster proposal
  - gated activation requires visible gate evidence
  - negative gate requires visible contradiction/failure association
  - recognition collapse is recorded as candidate, not regime proof
  - no expression has authority_allowed=True
  - no expression gets hard_filter from sleep alone
  - hidden truth fields are ignored
  - unrelated nethras remain separate
  - empty input produces empty summary
  - no imports from agent.py
  - no runtime behavior changes

Verification:
  python -m pytest tests/test_nethra_expression_index.py -q
  python -m pytest tests/test_memory_sleep.py -q
  python -m pytest tests/test_cycle_mechanics.py -q
  git diff --check
```

---

# Appendix B — Deferred Prompt: TemporalEventLedger and Passive Temporal Observers

Temporal work remains relevant because delayed causality and lagged structure are underrepresented. It should be integrated as another source of structure and gate evidence for expression mining, not as a separate authority path.

Core invariant:
Default behavior unchanged. Passive observers may record and propose. They may not intervene, issue authority, revoke authority, suppress skips, replace fit, or read hidden truth.

Future temporal records should feed:

- gated activations;
- negative gates;
- lagged coactivation clusters;
- delayed residual explanations;
- recognition-collapse candidates;
- emergent-regime candidates.

---

## Ledger

```text
[D:O!,N,V,E,B,C,A,L,S,Q,M,P0,Cav↓,R0,NB,LG
|R:RoleSurfaceMissing,TrassResidualBucketUnderspecified,ResidualPressureAuthorityLaunderingRisk,PeripheralClassificationInterferenceRisk,ClosedSystemResidualGrowthRisk,ProjectionGatingRisk
|F:split-role-surface-work-into-record-only-surfaces-then-memory-projection-integration-then-peripheral-residual-classification
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
