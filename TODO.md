# TODO

## Current Read

- ✅ Core Dreth pipeline exists: world, audit, fit, sentinel, ledger, summaries, batch harness.
- ✅ Hybrid provider interfaces exist and remain advisory rather than authority-granting.
- ✅ Shadow/diagnostic layers exist for policy, residuals, authority/evidence, relative authority, temporal frontier, uncertainty governance, uncertainty consolidation, context-role indexing, authority strength, background nethra, and scaffold memory.
- ✅ `ContextRoleIndex` records nethra nodes and context-indexed roles without changing behavior in record mode.
- ✅ The context-role model corrected one earlier error: `tareth`/`trass` are context-dependent roles, not identities of a nethra.
- ⚠️ The repo has now corrected a deeper error: a nethra is not the structure itself. A nethra is a scoped, evidence-bearing, context-activated lens over shared structure.
- ⚠️ Current `ContextRoleIndex` is provenance/indexing. It is not yet a full expression algebra over overlapping nethras.
- ⚠️ Current memory/sleep code groups scaffold proposals. That is useful but too flat for the corrected model.
- ⚠️ Runtime assist layers have repeatedly failed by turning broad visible uncertainty into excess attention/repair pressure. Treat broad uncertainty as debt/provenance first, not automatic work.
- ⚠️ Delayed causality and temporal structure remain underrepresented, but temporal work should connect to nethra-expression mining rather than become another flat proposal stream.

---

## Corrected Conceptual Target

The target object is no longer simply "offline memory sleep groups familiar records."

The target object is:

> Sleep should mine the overlap graph between shared structure and nethras, then propose bounded nethra expressions that runtime can compile into active search slices.

Definitions:

- `StructureGraph`: shared substrate of variables, relations, operators, candidate fits, probes, sentinels, experts, frontiers, dormant alternatives, residual patterns, role histories, temporal traces, and scaffold records.
- `Nethra`: scoped lens/handle touching part of the structure graph with evidence, activation conditions, use-rights, invalidators, and role history.
- `NethraExpression`: union/intersection/difference/gated/coactive expression over nethras.
- `ActiveSlice`: compiled runtime product containing filters, rank hints, probe hints, blockers, invalidators, and provenance.
- `EmergentRegime`: stable active-expression basin over co-active nethras that improves prediction/search; not a predeclared world label.

---

## Immediate Next Work

- [ ] Update or add design docs/code comments where they still say "nethra is learned structure" without the structure/lens distinction.
- [ ] Audit current `ContextRoleIndex`, `BackgroundNethraIndex`, `memory_sleep.py`, `nethra_scaffold_sleep.py`, and `scaffold_memory.py` against the corrected ontology.
- [ ] Identify which existing records can become `StructureGraph` nodes versus `Nethra` lenses.
- [ ] Identify which current proposal fields can support expression mining: touched vars/components, contexts, signatures, parent sets, role history, source kind, evidence counters, sentinel failures, revocations, coactivation, and temporal windows.
- [ ] Design `NethraExpressionIndex` as an offline/record-only module first.
- [ ] Keep runtime behavior unchanged in the first implementation.
- [ ] Do not add global regime switching. Regimes must be active expressions over lenses, not enum modes.
- [ ] Do not let sleep proposals produce authority, skip suppression, fit replacement, monitoring increases, or repair priority by default.

---

## NethraExpressionIndex Requirements

Add a design/implementation target for:

```text
NethraExpressionIndex
```

Input sources:

- scaffold nethras / scaffold proposals,
- context-role records,
- background nethra records,
- authority/debt records,
- uncertainty records,
- tied frontiers,
- dormant alternatives,
- temporal records if present,
- outcome metrics where available.

Output proposal types:

- `overlap_bridge`: two or more nethras touch the same structure.
- `subset_relation`: one nethra's touched structure is mostly contained inside another.
- `union_expression`: two or more lenses jointly cover a useful region.
- `intersection_expression`: overlap between lenses is more specific than either lens alone.
- `difference_expression`: one lens excludes or contradicts part of another.
- `gated_activation`: nethra A is useful only when nethra B or signal X is active.
- `negative_gate`: nethra A tends to fail when nethra B or signal X is active.
- `coactivation_cluster`: nethras repeatedly become useful together.
- `recognition_collapse_candidate`: active lens set loses coverage.
- `emergent_regime_candidate`: recurrent coactivation plus renewed predictability after recognition collapse.
- `active_slice_candidate`: bounded runtime slice containing rank hints, probe hints, blockers, and possible filters.

Each expression proposal must include:

- expression id,
- expression kind,
- member nethra ids,
- touched structure ids/components,
- operation: union/intersection/difference/gated/coactive,
- activation gate if any,
- negative gate if any,
- evidence summary,
- runs/seeds/cycles seen,
- suggested use-right,
- invalidators,
- warnings,
- authority_allowed: false.

Suggested use-rights:

- `record_only`,
- `feature_only`,
- `ranking_hint`,
- `probe_hint`,
- `soft_filter_candidate`,
- `block_candidate`.

No expression should default to `hard_filter` from sleep alone.

---

## Non-Negotiable Invariants

- [ ] A nethra is a lens over structure, not the structure itself.
- [ ] Multiple nethras may touch the same structure.
- [ ] Structure overlap is not proof.
- [ ] Cross-context overlap is downgraded to hint/proposal until local evidence earns stronger use.
- [ ] `tareth`/`trass` are context roles, not global properties of a nethra.
- [ ] Trass is not deletion.
- [ ] Recognition collapse is a signal that active coverage failed, not proof of a new world/regime.
- [ ] Regimes are emergent active expressions over co-active nethras, not global switches.
- [ ] A nethra expression must not inherit the strongest authority of its members.
- [ ] Authority is earned by visible evidence, not provider confidence, graph proximity, morphology, index membership, recurrence, sleep proposal, or temporal correlation.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, temporal observers, sleep consolidation, or assist logic.
- [ ] Shadow/diagnostic/passive layers may observe; they must not mutate authority unless explicitly promoted through a separately tested bounded runtime path.
- [ ] Record-only indexes must match off-mode behavior.
- [ ] Assist-feature paths must be attributed: which expression changed ordering/probes/filters, and whether the outcome improved.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, generic uncertainty signals, or repeated background familiarity must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Passive temporal observers must be bounded by caps/ring buffers/summaries; do not create unbounded event-history growth.

---

## Deferred / Reference Work

- [ ] `NethraExpressionIndex` / offline expression mining. This supersedes flat MemorySleepConsolidator as the next conceptual target.
- [ ] `ActiveSlice` compiler. Runtime should eventually compile active expressions into bounded rank/probe/filter/block surfaces.
- [ ] Recognition-collapse metrics: measure active lens coverage failure without assuming a named new world.
- [ ] TemporalEventLedger / passive temporal observers for delayed causality scaffolding. See Appendix B, but connect it to expression mining.
- [ ] Weighted intervention-cost tracking is low-risk and can be cherry-picked later as diagnostic-only.
- [ ] Dormant alternative revival is conceptually aligned, but should wait until retrieval/expression gates are strict enough.
- [ ] Regime-triggered inert re-screening may be valuable, but should not be implemented as a global regime switch.
- [ ] Learned ranker/factorizer is not next. Use it only after deterministic expression mining and attribution show that rule-based retrieval cannot separate useful anchors from broad noise.

---

# Appendix A — Deferred Codex Prompt: NethraExpressionIndex / Offline Expression Mining

Use this instead of the older flat MemorySleepConsolidator prompt. The purpose is to let Dreth consolidate overlapping nethra lenses while the program is not running, without laundering familiarity into authority.

```text
You are working in the Dooces/dreth repo.

Current context:
Dreth now distinguishes shared structure from nethras.
A nethra is not the structure itself. It is a scoped, evidence-bearing, context-activated lens over shared structure.
Multiple nethras can touch the same structure.
Roles such as tareth/trass are context roles, not identities.
Regimes should emerge as active expressions over co-active nethras, not as predeclared world labels.

Goal:
Add an offline NethraExpressionIndex design/implementation pass.

Purpose:
Read exported runtime/scaffold memory and build proposal-only expressions over overlapping nethras:
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
  common_parents
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

Grouping/mining rules:
Use visible/exported fields only:
  - nethra id
  - vars/components touched
  - contexts/context families
  - role histories
  - parent/signature overlap
  - recurring signal overlap
  - source kind overlap
  - sentinel/revocation/failure counters if visible
  - coactivation across cycles/runs
  - temporal proximity if temporal records exist

Do not use:
  truth_parents
  truth_func
  truth_delayed_parents
  truth_latents
  debug_blind_challenge_manifest
  relation_type except in a separate post-hoc report section disabled by default

Runtime boundary:
Do not wire expressions into ChainedAgent in this task.
Do not add behavior-changing use.
Output proposals and a summary only.

Add script:
  scripts/run_nethra_expression_sleep.py

CLI:
  python scripts/run_nethra_expression_sleep.py \
    --jsonl reports/background_nethra_compare_record.jsonl \
    --scaffold reports/nethra_scaffold_sleep_gen2.jsonl \
    --out reports/nethra_expression_proposals.jsonl \
    --summary reports/nethra_expression_summary.txt

Summary sections:
A. Inputs
  rows read
  structure records seen
  nethra handles seen
  scaffold records seen

B. Expressions
  count
  by kind
  by suggested use-right
  largest expressions

C. Overlap and gates
  overlap bridges
  subset relations
  gated activations
  negative gates
  coactivation clusters

D. Recognition/regime candidates
  recognition-collapse candidates
  emergent-regime candidates
  warning that these are not regime switches

E. Familiarity not authority
  authority_allowed_count must be 0
  no hard_filter from sleep alone

F. Examples
  top proposals with member nethras, touched structure, contexts, gates, invalidators, warnings

G. Hidden-truth guard
  hidden truth fields ignored

H. Warning
  Expression proposals are search/familiarity scaffolds only.
  They are not operational authority.

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

## Appendix A Warnings

- Do not collapse structure and nethra.
- Do not let overlap become authority.
- Do not let expression specificity become proof.
- Do not emit unbounded expression combinations.
- Do not create global regime switches.
- Do not let old-context use-rights transfer without downgrade.
- Do not let sleep expressions issue certs, revoke, suppress skips, replace fits, increase monitoring, or increase repair priority.
- Do not store unbounded memory. Use summaries, caps, provenance IDs, and bounded examples.

---

# Appendix B — Deferred Prompt: TemporalEventLedger and Passive Temporal Observers

Temporal work remains relevant because delayed causality and lagged structure are underrepresented. It should be integrated as another source of structure and gate evidence for expression mining, not as a separate authority path.

Core invariant:
Default behavior unchanged. Passive observers may record and propose. They may not intervene, issue authority, revoke authority, suppress skips, replace fit, or read hidden truth.

Future temporal records should feed:

- gated activations,
- negative gates,
- lagged coactivation clusters,
- delayed residual explanations,
- recognition-collapse candidates,
- and emergent-regime candidates.

---

## Ledger

```text
[D:O!,N,V,E,B,C,A,L,S,Q,M,P0,Cav↓,R0,NB,LG
|R:StructureNethraConflationRisk,ExpressionBlowupRisk,InheritedAuthorityLaunderingRisk,GlobalRegimeSwitchDrift,TemporalFlatProposalRisk
|F:update-docs-to-nethra-as-lens-over-structure-and-retarget-next-work-to-expression-mining
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
