# Design Understanding

This document describes the current Dreth design frame. It replaces older wording that treated `tareth` as hidden-world truth, `trass` as falsehood, and `NethraCertificate` as absolute certification.

The current model is different:

> A nethra is learned structure. `tareth`, `trass`, `unresolved`, and `best_available` are context-indexed roles assigned to that structure under evidence and operation.

Dreth is a simulation framework for testing whether explicit, evidence-shaped, context-indexed structure can help an agent allocate limited attention under uncertainty. It is not a claim of solved AGI and not a production causal discovery package.

## 1. What Dreth is testing

The central problem is not simply “find the true causal graph.” The generated worlds do have hidden truth for evaluation, but the agent is not allowed to read it. The runtime problem is evidence-relative:

- What structure has the agent learned?
- What evidence supports it?
- In which context does it matter?
- When should it be used as the current best available handle?
- When should it be treated as operationally irrelevant in this context?
- When should ambiguity, novelty, or failure change attention?
- When should a higher handle reduce work, and when is it merely a broad alarm?

The architecture studies these questions using a hidden world, an intervention interface, a ledger of learned structure, sentinel checks, frontiers, uncertainty consolidation, context-role indexing, and diagnostic reports.

## 2. Core terms

### Nethra

A **nethra** is a learned reusable structure: a handle over a relation, fit, candidate, component, regime pattern, frontier alternative, composite, or higher-order organization.

It is not a label handed down by the world. It is also not necessarily a final truth. It is a structure the system can reuse, compare, compose, monitor, preserve, or demote in scope.

A nethra may be composed of other nethras. That is the basis for nethras-of-nethras: higher handles built from lower ones.

### Context role

A context role says how a nethra functions in a particular context or operation.

Current roles:

- `tareth`: this nethra matters operationally in this context.
- `trass`: this nethra is operationally equivalent or irrelevant in this context.
- `unresolved`: available evidence preserves ambiguity or instability.
- `best_available`: this is the current working handle, even if uncertainty remains.

These are not identities. A nethra can be trass in one context and tareth in another.

Example:

- A shape-like nethra can be trass for color classification.
- The same shape-like nethra can be tareth for grasp planning.
- Under a novel regime, it may be unresolved.
- Under weak evidence, it may be best_available because nothing better exists.

### Authority record

The code still uses legacy names such as `NethraCertificate`, `certificates`, and `certified_eps`. In current design language, these should be read as **authority records**: evidence-bounded commitments, not absolute proof.

Authority is graded by evidence, scope, survival, revocation history, sentinels, alternatives, and context. A current authority record is not a global statement that the structure is true forever.

### Hidden truth

The world has hidden causal structure so reports can evaluate what happened after the run. Runtime must not use it.

Hidden truth is allowed only in post-hoc interpretation sections, such as blind challenge summaries or relation-type breakdowns. It is not allowed in clustering, runtime matching, assist decisions, authority assignment, skip behavior, or fit selection.

## 3. World model

`dreth/world.py` provides hidden causal worlds. These worlds expose scalar state and intervention interfaces. They include noise, clipping, schedule-driven drift, regime changes, and blind challenge structure.

Important schedules include:

- `regime_switch`: designed to produce recurring regime-like structural shifts.
- `false_trass`: stresses proxy/false operational equivalence behavior.
- `blind_challenge`: mixes symbolic, smooth nonlinear, latent, delayed, proxy, dense, and weak/noisy relations.

The world is an oracle only through observations and interventions. The agent observes effects; it does not inspect hidden parent/function fields.

The purpose of the more complex worlds is not to guarantee success. Failure is useful when it identifies which structures the current architecture cannot learn or use.

## 4. Runtime loop

`dreth/agent.py` owns the main `ChainedAgent` loop.

A typical cycle involves:

1. Observe current state.
2. Determine which variables or handles need attention.
3. Use existing authority records, sentinels, composites, roles, and repair agenda state to avoid unnecessary full audits where justified.
4. Run full audits when needed.
5. Fit candidate parent/function hypotheses.
6. Preserve ties, near-ties, dormant alternatives, novelty, and uncertainty signals.
7. Install the best available working structure.
8. Record role/provenance information where enabled.
9. Optionally run shadow or assist layers.

The loop should not be understood as “find truth, certify truth, skip forever.” It is closer to:

> maintain usable structure under limited attention, while making failures and uncertainty explicit enough to guide future repair.

## 5. Fitting and sentinels

`dreth/fit.py` enumerates and scores parent/function hypotheses using the agent’s visible vocabulary. It can find best fits, margins, ties, and near-ties. A best fit is not automatically hidden truth.

`dreth/sentinels.py` selects and runs probes that test whether a learned structure still behaves as expected. Sentinels are the main cheap-path check for whether a current authority record remains usable.

A sentinel pass is not metaphysical proof. It means the checked evidence surface did not currently contradict the handle.

A sentinel failure should create repair pressure, revocation/demotion, frontier activity, uncertainty signals, or context-role changes depending on scope.

## 6. Ledger and authority

`dreth/ledger.py` stores variable nethras, authority records, noise envelopes, tied frontiers, dormant alternatives, composites, revocations, and related state.

The ledger is the main authority/provenance surface. The long-term design direction is to make authority changes explicit ledger transactions rather than inline agent-side object construction.

The invariant is:

> Authority should be earned by visible evidence and changed through explicit ledger pathways.

The current code still has legacy names and some agent/ledger coupling. That is an implementation hygiene issue, not a conceptual requirement.

## 7. Tied frontiers and dormant alternatives

Ambiguity is not supposed to disappear merely because one candidate is temporarily selected.

Tied and near-tied hypotheses are tracked by `TiedFrontier`. Alternatives can become dormant. Dormant does not mean false, deleted, or irrelevant forever. It means not currently selected as the operative path.

The current design direction is:

- preserve ambiguity when evidence is insufficient,
- use it to guide future probes or context-role indexing,
- avoid letting dormant alternatives become global pressure everywhere.

## 8. ContextRoleIndex

`dreth/context_role_index.py` implements the corrected nethra/context-role model.

It defines:

- `NethraNode`: learned structure node.
- `NethraEdge`: relation between nethras.
- `ContextRoleRecord`: role assignment in a specific context.
- `ContextRoleIndex`: retrieval/index view over nethra graph provenance.

The index is not a trass reservoir. It is not a bucket of discarded distinctions. It is a graph/index view over nethras and their context roles.

The important distinction:

- The nethra persists as structure.
- The role changes by context.
- Trass is not deletion.
- Tareth is not global identity.

Current reports show that record mode can remain behavior-neutral while recording nethra nodes and context roles. That is the minimum safety condition for provenance indexing.

Runtime `assist_feature` use is more dangerous. It can feed context-role matches into uncertainty consolidation as locality anchors. At scale, loose matching has overconnected and worsened metrics. This means the index is useful as provenance, but runtime use requires strict match quality, deduplication, and attribution.

## 9. Uncertainty governance and consolidation

`dreth/uncertainty_governance.py` is shadow-only. It extracts visible uncertainty signals and proposes diagnostic actions. It does not change runtime behavior.

`dreth/uncertainty_consolidation.py` can group repeated uncertainty cases into candidate higher handles. In `assist` mode, it can feed bounded reversible hints into existing surfaces:

- attention priority,
- repair agenda priority,
- probe requests,
- monitoring pressure,
- alternative preservation.

It must not directly revoke authority, suppress skips, replace `fit_var`, or issue authority records.

Current lessons:

- Broad uncertainty is not automatically bad. It may indicate shared unresolved structure.
- Broad uncertainty is not automatically useful. If it collapses into giant global clusters, it becomes pressure rather than structure.
- Compression ratio alone is not success.
- Useful consolidation requires specificity: local anchors, shared parents/components, shared graph neighborhoods, role transitions, co-failure timing, or strong context overlap.

## 10. Repair agenda

`dreth/repair_agenda.py` is a planning surface for repair work.

It should help separate “this needs attention” from “this has authority.” Repair agenda items do not themselves authorize repairs or role changes. They prioritize where evidence-seeking should happen.

Current priority logic is intentionally limited. Better cost/benefit scheduling is unfinished.

## 11. Relative authority graph diagnostics

`relative_authority.py`, `relative_authority_observer.py`, and `relative_authority_frontier.py` are diagnostic graph tools.

They are used to ask questions like:

- Do existing authority records imply useful local neighborhoods?
- Can graph-local frontiers reduce candidate search while preserving recall?
- Are relative authority records forming useful proposal priors?

The frontier evaluator showed useful lift as a proposal prior, but not enough recall for exclusive filtering. The correct interpretation is:

> Graph frontiers may rank or propose; they should not hard-exclude without fallback.

## 12. Hybrid providers

`dreth/hybrid.py` defines provider interfaces:

- residual predictor,
- parent ranker,
- probe proposer,
- expert,
- expert router,
- repair event surface.

Providers can advise. They cannot create authority records or mutate ledger state as authority.

Provider outputs should be treated as proposals, rankings, or diagnostics unless and until they pass through visible evidence machinery.

## 13. Learned and shadow components

`learned_residual.py`, `shadow_policy.py`, and `shadow_authority_throttle.py` are shadow/diagnostic layers.

Their purpose is to determine whether learned or diagnostic signals could become useful later. Their outputs should not be read as authority.

A learner or NN should not be added merely because deterministic logic failed. First, the target must be clear:

- rank assist usefulness,
- predict future revocation,
- predict cluster quality,
- choose probes,
- select attention under uncertainty,
- or identify reusable factors.

Without a clear target, a learner hides the failure rather than explaining it.

## 14. Regime and composite handles

Composite handles can be legitimate nethras-of-nethras when they have active witnesses that test a higher relation.

Regime handles are still less mature. A recurring co-failure pattern is not enough by itself to justify reduced monitoring. A regime handle should only buy down leaf checks when it has an active witness or sentinel that validates the cluster-level invariant.

Important distinction:

- Quiescence is absence of observed failure.
- Sentinel validation is an active check.

A regime handle based only on quiescence should be diagnostic or weak authority at most.

## 15. Quality metrics and interpretation

`quality_cost` and related metrics are diagnostic. They are useful for comparing policies, but they are not runtime authority.

Important metrics include:

- intervention count / IV,
- full audits,
- revocations,
- unique sentinel failures,
- regime sentinel failures,
- passive saved/stressed counts,
- frontier recall/lift,
- context-role matches and assist pressure,
- dormant alternatives,
- quality cost.

Passing invariants means safety boundaries held. It does not mean a behavior path is useful.

A runtime assist path is only promising if it improves or preserves relevant metrics without creating broad pressure, excessive audits, excessive revocations, or degraded recall.

## 16. Current known results

Recent experimental status, as of the current design frame:

1. `ContextRoleIndex record` is conceptually correct and behavior-neutral in reported sweeps.
2. The index can show nethras with multiple roles across contexts, including trass-in-one-context / tareth-in-another examples.
3. `ContextRoleIndex assist_feature` has produced nonzero local-anchor use, so wiring exists.
4. At scale, loose context-role matching overconnected and worsened behavior: IV, quality cost, audits, revocations, unique failures, and regime sentinel failures increased.
5. Therefore ContextRoleIndex should be kept as provenance, while runtime assist use requires stricter match gating and attribution.
6. Uncertainty consolidation must be judged by off/shadow/assist comparisons, not by whether counters fire.
7. Blind challenge is useful because it exposes the limits of current structure, not because Dreth is expected to solve every generated relation family.

## 17. How to read test modes

Several modes exist and should not be confused.

- `off`: feature disabled.
- `record`: record/index/report only. Should match off in behavior.
- `shadow`: run diagnostics only. Should match off in behavior.
- `assist`: bounded reversible runtime hints. May change behavior, but must be judged by outcome metrics.
- `assist_feature`: uses an index or feature as part of an assist path. Must prove attribution and avoid overuse.

If `record` differs from `off`, there is a leak.

If `shadow` differs from `off`, there is a leak.

If `assist` differs from `off`, that is expected, but it may be harmful.

## 18. Commands for current checks

### Core tests

```bash
python -m pytest tests/test_cycle_mechanics.py -q
python -m pytest tests/test_blind_challenge.py -q
python -m pytest tests/test_uncertainty_consolidation.py -q
python -m pytest tests/test_nethra_reservoir.py -q
```

`tests/test_nethra_reservoir.py` is a compatibility filename; the semantics now target `ContextRoleIndex`.

### Context-role record check

```bash
python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50 \
  --cycles 3000 \
  --seeds 42,99,7 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index record \
  --out reports/context_role_record_check.jsonl \
  2>&1 | tee reports/context_role_record_check.log

python scripts/summarize_context_role_index.py \
  --jsonl reports/context_role_record_check.jsonl \
  | tee reports/context_role_record_check_summary.txt
```

### Context-role off/record/assist comparison

```bash
python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index off \
  --out reports/context_role_sweep_off.jsonl \
  2>&1 | tee reports/context_role_sweep_off.log

python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index record \
  --out reports/context_role_sweep_record.jsonl \
  2>&1 | tee reports/context_role_sweep_record.log

python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation assist \
  --uncertainty-assist-policy local_only \
  --context-role-index assist_feature \
  --out reports/context_role_sweep_assist_feature.jsonl \
  2>&1 | tee reports/context_role_sweep_assist_feature.log
```

Interpretation:

- `off == record`: provenance indexing is clean.
- `assist_feature` improves: check match attribution before claiming the index helped.
- `assist_feature` worsens: keep the index as provenance, but restrict or disable runtime assist use.

## 19. Design risks to keep visible

Current risks:

- **Absolute-certification drift**: reading authority records as proof rather than scoped evidence.
- **Role identity confusion**: treating tareth/trass as object identity instead of context role.
- **Trass-as-deletion**: losing reusable structure because it was operationally irrelevant once.
- **Broad uncertainty pressure**: using generic uncertainty signals as if they were local structure.
- **Context-role overmatching**: letting too many provenance matches become local anchors.
- **Shadow loop stall**: recording reports forever without controlled runtime integration.
- **Premature learner insertion**: adding an NN before the learning target is identifiable.
- **Regime quiescence error**: confusing no recent failure with an active higher sentinel pass.

## 20. Current next design target

The immediate high-value target is not more ontology.

The next useful work is strict gating and attribution for context-role-assisted consolidation:

- raw vs deduped context-role matches,
- weak-match suppression,
- duplicate suppression,
- per-cluster anchor caps,
- assist pressure per cycle,
- match-quality fields,
- strict vs loose comparison,
- off/record/assist comparison.

If strict matching still worsens behavior, then `ContextRoleIndex` should remain record/provenance only until a better factorizer or learned ranker has a clear target.

## 21. Bottom line

Dreth is currently best understood as an experimental ledgered attention system:

- nethras are reusable learned structures,
- roles are context-indexed,
- authority is evidence-bounded,
- uncertainty must become useful structure before it should drive behavior,
- and negative results are valuable when they identify exactly where a handle is too broad, too weak, or too expensive.
