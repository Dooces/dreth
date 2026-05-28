# Design Understanding

This document describes the current Dreth design frame. It replaces older wording that treated `tareth` as hidden-world truth, `trass` as falsehood, `NethraCertificate` as absolute certification, or a nethra as identical to learned structure.

The current model is:

> Structure is the shared substrate. A nethra is a scoped, evidence-bearing, context-activated lens over that substrate. Roles are context-indexed. Regimes are emergent active expressions over overlapping nethras, not predeclared world labels.

Dreth is a simulation framework for testing whether explicit, evidence-shaped, context-indexed structure can help an agent allocate limited attention and search under uncertainty. It is not a claim of solved AGI and not a production causal discovery package.

## 1. What Dreth is testing

The central problem is not simply "find the true causal graph." The generated worlds do have hidden truth for evaluation, but the agent is not allowed to read it. The runtime problem is evidence-relative:

- What structure has the agent learned or observed?
- Which nethra lenses touch that structure?
- What evidence supports each lens?
- In which context is each lens active?
- What use-right does each lens currently have?
- When should a lens annotate, rank, filter, block, preserve, or reopen structure?
- When should overlap with prior structure become a hint, and when should it remain inert?
- When does recognition collapse indicate that the active lens set no longer covers the current context?
- When do co-active lenses form an emergent regime expression?

The architecture studies these questions using hidden worlds, an intervention interface, a ledger of learned structure and authority records, sentinel checks, frontiers, uncertainty consolidation, context-role indexing, background/scaffold memory, and diagnostic reports.

## 2. Corrected core ontology

### StructureGraph

The **StructureGraph** is the reusable substrate: variables, relations, operators, candidate fits, source_edge sets, frontiers, dormant alternatives, probes, experts, sentinels, residual patterns, temporal traces, role histories, and scaffold/sleep products.

Structure is not itself authority. Structure is what nethras touch.

### Nethra

A **nethra** is a scoped handle/lens over structure. It does not own the structure and is not identical to the structure. Multiple nethras can touch the same node, edge, operator, candidate, probe, or expert.

A nethra should carry:

- touched structure,
- evidence/provenance,
- activation conditions,
- use-rights,
- invalidators,
- scope/context,
- role history,
- and relations to other nethras.

This distinction matters because the same structure may be relevant, irrelevant, familiar, blocked, unresolved, or useful as a rank hint in different contexts.

### Context role

A context role says how a nethra functions in a particular operation/context.

Current roles include:

- `tareth`: this nethra matters operationally in this context.
- `trass`: this nethra is operationally equivalent or irrelevant in this context.
- `unresolved`: available evidence preserves ambiguity or instability.
- `best_available`: this is the current working handle, even if uncertainty remains.

These are not identities. A nethra can be trass in one context and tareth in another. Trass is not deletion.

### Use-rights

A nethra's use-right controls how it may influence runtime search:

- `record_only`: stored/reported only.
- `feature_only`: may annotate, but cannot reorder or exclude.
- `ranking_hint`: may reorder existing candidates or probes.
- `soft_filter`: may prioritize touched structure while preserving fallback.
- `hard_filter`: may exclude structure only when local evidence has earned that permission.
- `block`: may prevent use/derivation in a scope.

Authority must be earned by visible evidence. Recognition, recurrence, graph proximity, and overlap are not authority.

### NethraExpression

A **NethraExpression** is a union/intersection/difference/gated/coactive expression over nethras.

Examples:

- `A ∩ B`: touched structure common to both lenses.
- `A ∪ B`: touched structure from either lens.
- `A - B`: structure touched by A but blocked/contradicted by B.
- `A if gate_B`: A is active only if B or a signal condition is active.
- `A coactive-with B`: A and B repeatedly become useful together.

A nethra expression does not inherit the strongest member authority. It must earn its own use-right in the current scope.

### ActiveSlice

An **ActiveSlice** is the compiled runtime product of active nethras and nethra expressions:

- hard filters,
- soft filters,
- rank hints,
- probe hints,
- blockers,
- invalidators,
- and provenance.

Runtime should not need the full historical structure graph every cycle. It should receive a bounded active slice.

### Emergent regime

A regime is not a world label. The agent does not need to know that it moved from "world A" to "world B." It only needs to detect that active recognition has collapsed: predictions degrade, familiar handles stop matching, sentinels fail, rank lift drops, or old filters no longer apply.

That opens a regime-boundary candidate. Old nethras are downgraded to hints. Local overlap with existing structure is tested. If recurrent overlap improves prediction, source_edge ranking, probe choice, or repair localization, local bridge nethras form. If several bridges co-activate and remain useful, a regime nethra emerges as a stable active expression basin.

## 3. World model

`dreth/world.py` provides hidden causal worlds. These worlds expose scalar state and intervention interfaces. They include noise, clipping, schedule-driven drift, regime changes, and blind challenge structure.

Important schedules include:

- `regime_switch`: produces recurring regime-like structural shifts.
- `false_trass`: stresses proxy/false operational equivalence behavior.
- `blind_challenge`: mixes symbolic, smooth nonlinear, latent, delayed, proxy, dense, and weak/noisy relations.

The world is an oracle only through observations and interventions. The agent observes effects; it does not inspect hidden source_edge/function fields.

The purpose of richer worlds is not to guarantee success. Failure is useful when it identifies which structures, lenses, or expressions the current architecture cannot learn or use.

## 4. Runtime loop

`dreth/agent.py` owns the main `ChainedAgent` loop.

A typical cycle involves:

1. Observe current state.
2. Determine which variables, handles, or structures need attention.
3. Use existing authority records, sentinels, composites, roles, and repair agenda state to avoid unnecessary full audits where justified.
4. Run full audits when needed.
5. Fit candidate source_edge/function hypotheses. Each candidate commits a prediction for the next relevant observation before that observation is read. Scoring then measures the accuracy of that precommitted prediction, not a post-hoc fit.
6. Preserve ties, near-ties, dormant alternatives, novelty, and uncertainty signals.
7. Install the best available working structure.
8. Record role/provenance information where enabled.
9. Optionally run shadow or assist layers.

The loop should not be understood as "find truth, certify truth, skip forever." It is closer to:

> maintain usable structure under limited attention, while making failures and uncertainty explicit enough to guide future search and repair.

## 5. Prediction commitment and credit assignment

**Intervention is not the judge. Precommitted prediction-vs-observation is the judge.** Intervention is a high-resolution diagnostic instrument — selected because rival handles already made different conditional predictions, not as the first cause of credit assignment.

### The correct Dreth loop

1. Proposal machinery emits multiple candidate handles/theories.
2. Each theory makes an explicit, precommitted prediction before observation or intervention result is known.
3. Observations arrive.
4. Each theory receives credit or debit by prediction quality against its precommitted value.
5. If theories remain tied or ambiguity matters, choose an intervention that best separates them.
6. Interventions are scored against the predictions each theory made beforehand.
7. Authority shifts toward handles that predicted well under the relevant temporal and context scope.

No intervention was required to resolve step 4. Intervention enters only at step 5, when precommitted predictions are already on the ledger and rival handles are still tied.

### Precommitment invariant

> A theory must predict before the observation or intervention result is known. Only precommitted predictions earn credit. Post-hoc explanations may be stored as proposals but do not earn authority until they predict future observations correctly.

This is non-negotiable. A handle that accounts for an observation it did not predict is not evidence of predictive authority. It is a proposal candidate. It earns forward credit by predicting the next cycle correctly.

### PredictionCommitment record

Each committed prediction should be a named, structured record:

```
PredictionCommitment:
  handle_id
  predicted_target
  premise / context
  time_index / horizon / lag
  predicted_value or distribution
  tolerance / envelope
  observation_channel
  operation_relevance
```

The `lag` field is not optional. Without a temporal commitment, delayed causality reads as weak or noisy same-time causality. A handle claiming x3[t+1] = f(x1[t-3]) must commit to the specific lag before the observation arrives; only then can the residual be correctly attributed.

### Concrete failure record

Failure is not:

> x3 was wrong.

Failure is:

> H predicted x3[t+5] under context C; observation O contradicted it by delta D.

That structured record — handle, predicted target, temporal scope, context, residual magnitude — is what supports repair, recurrence mining, sleep/scaffold grouping, and future intervention choice.

### Nareth as repair search structure

In this frame, the nareth-like repair surface is a set of:

- rival handles that made predictions,
- their temporal scopes,
- their residuals,
- their disagreement points,
- proposed separating observations or interventions.

**Tareth** is the handle or distinction whose precommitted prediction actually mattered for the operation and context — not globally true, but specifically predictive under the relevant scope.

### Intervention as separating diagnostic

Rival handles may make different conditional predictions. For example:

- H1 predicts x3[t+1] = MEAN(x1[t], x2[t]) → 0.62
- H3 predicts x3[t+1] = MIN(x1[t-3], x2[t]) → 0.39

If x3[t+1] = 0.41, H3 earns credit and H1 loses it. No intervention was needed.

If H1 and H3 later both predict well in different contexts, intervention can test the gating condition:

> If H1 is right, changing x1 now should affect x3 next cycle. If H3 is right, changing x1 now should not affect x3 until t+3.

The intervention is chosen because the theories already committed to different conditional predictions. Its result is scored against those precommitted predictions, not used as a standalone judge.

### Reward and task value

Reward or task value can determine which prediction failures matter more. It does not define what the proposal engine is allowed to notice or which precommitted predictions count as failures. Credit assignment is prediction-first; task weighting is applied afterward.

## 6. Fitting and sentinels

`dreth/fit.py` enumerates and scores source_edge/function hypotheses using the agent's visible vocabulary. It can find best fits, margins, ties, and near-ties. A best fit is not automatically hidden truth.

`dreth/sentinels.py` selects and runs probes that test whether a learned structure still behaves as expected. Sentinels are cheap-path checks for whether a current authority record or lens remains usable.

A sentinel pass is not metaphysical proof. It means the checked evidence surface did not currently contradict the handle.

A sentinel failure should create repair pressure, revocation/demotion, frontier activity, uncertainty signals, recognition-collapse evidence, or context-role changes depending on scope.

## 7. Ledger and authority

`dreth/ledger.py` stores variable handles, authority records, noise envelopes, tied frontiers, dormant alternatives, composites, revocations, and related state.

The ledger is the main authority/provenance surface. The long-term design direction is to make authority changes explicit ledger transactions rather than inline agent-side object construction.

The invariant is:

> Authority should be earned by visible evidence and changed through explicit ledger pathways.

Specifically, authority is earned by precommitted prediction accuracy — predictions declared before the observation is known. Visible evidence cannot retroactively upgrade a post-hoc explanation to authority. Post-hoc accounts are ledgered as proposals and must earn forward credit by predicting future observations correctly.

The current code still has legacy names such as `NethraCertificate`, `certificates`, and `certified_eps`. In current design language, these should be read as authority records or evidence-bounded commitments, not absolute proof.

## 8. Tied frontiers and dormant alternatives

Ambiguity is not supposed to disappear merely because one candidate is temporarily selected.

Tied and near-tied hypotheses are tracked by `TiedFrontier`. Alternatives can become dormant. Dormant does not mean false, deleted, or irrelevant forever. It means not currently selected as the operative path.

Dormant alternatives are structure. Nethras may touch them later as familiar, ranked, blocked, or reopened depending on context and evidence.

## 9. ContextRoleIndex

`dreth/context_role_index.py` currently implements a provenance index over learned nethra graph structure and context roles.

It defines:

- `NethraNode`: current representation of a learned structure node/handle.
- `NethraEdge`: relation between nodes.
- `ContextRoleRecord`: role assignment in a specific context.
- `ContextRoleIndex`: retrieval/index view over graph provenance and role history.

This is useful but incomplete under the corrected ontology. `ContextRoleIndex` is not yet a full nethra-expression compiler. It does not yet represent active expression algebra, activation gates, cross-context downgrade rules, or runtime active-slice compilation.

The important retained distinction:

- Structure persists.
- Nethras touch structure.
- Roles change by context.
- Trass is not deletion.
- Tareth is not global identity.

Runtime `assist_feature` use is dangerous because it can affect behavior by admitting matches as local anchors or reordering candidates. It must be judged by strict attribution and outcome metrics.

## 10. Uncertainty governance and consolidation

`dreth/uncertainty_governance.py` is shadow-only. It extracts visible uncertainty signals and proposes diagnostic actions. It does not change runtime behavior.

`dreth/uncertainty_consolidation.py` can group repeated uncertainty cases into candidate higher handles. In `assist` mode, it can feed bounded reversible hints into existing surfaces:

- attention priority,
- repair agenda priority,
- probe requests,
- monitoring pressure,
- alternative preservation.

It must not directly revoke authority, suppress skips, replace `fit_var`, or issue authority records.

Current lesson:

- Broad uncertainty is not automatically bad.
- Broad uncertainty is not automatically useful.
- Compression ratio alone is not success.
- Useful consolidation requires specificity: local anchors, shared source_edges/components, shared graph neighborhoods, role transitions, co-failure timing, context overlap, or later utility.

## 11. Repair agenda

`dreth/repair_agenda.py` is a planning surface for repair work.

It should help separate "this needs attention" from "this has authority." Repair agenda items do not themselves authorize repairs or role changes. They prioritize where evidence-seeking should happen.

Current priority logic is intentionally limited. Better cost/benefit scheduling is unfinished.

## 12. Relative authority graph diagnostics

`relative_authority.py`, `relative_authority_observer.py`, and `relative_authority_frontier.py` are diagnostic graph tools.

They are used to ask questions like:

- Do existing authority records imply useful local neighborhoods?
- Can graph-local frontiers reduce candidate search while preserving recall?
- Are relative authority records forming useful proposal priors?

The frontier evaluator showed useful lift as a proposal prior, but not enough recall for exclusive filtering. The correct interpretation is:

> Graph frontiers may rank or propose; they should not hard-exclude without fallback.

## 13. Hybrid providers

`dreth/hybrid.py` defines provider interfaces:

- residual predictor,
- source_edge ranker,
- probe proposer,
- expert,
- expert router,
- repair event surface.

Providers can advise. They cannot create authority records or mutate ledger state as authority.

Provider outputs should be treated as proposals, rankings, or diagnostics unless and until they pass through visible evidence machinery.

## 14. Learned and shadow components

`learned_residual.py`, `shadow_policy.py`, and `shadow_authority_throttle.py` are shadow/diagnostic layers.

Their purpose is to determine whether learned or diagnostic signals could become useful later. Their outputs should not be read as authority.

A learner or NN should not be added merely because deterministic logic failed. First, the target must be clear:

- rank assist usefulness,
- predict future revocation,
- predict cluster quality,
- choose probes,
- select attention under uncertainty,
- identify reusable factors,
- or mine nethra-expression utility.

Without a clear target, a learner hides the failure rather than explaining it.

## 15. Regime and composite handles

Composite handles can be legitimate nethras-of-nethras when they have active witnesses that test a higher relation.

Regime handles should be reunderstood as emergent expressions over overlapping/coactive nethras. A recurring co-failure pattern is not enough by itself to justify reduced monitoring. A regime handle should only buy down work when the expression has active witnesses/sentinels or locally measured utility.

Important distinction:

- Quiescence is absence of observed failure.
- Sentinel validation is an active check.
- Recognition collapse is a boundary signal, not a regime proof.
- Renewed clustered predictability is what begins to earn a regime expression.

Intervention into a regime-boundary candidate is a separating diagnostic: it is chosen because rival handles already committed to different conditional predictions about what the intervention will produce. Its result is scored against those precommitted predictions. It is not the judge; it is the best available probe given existing disagreement.

## 16. Sleep, scaffold memory, and expression mining

Current sleep/scaffold code groups familiar records into proposals. That is useful but flat.

The stronger target is offline expression mining over the structure/nethra overlap graph:

- overlap bridges,
- subset/superset relations,
- union/intersection/difference expressions,
- gated activations,
- negative gates,
- coactivation clusters,
- candidate active slices,
- and emergent regime-expression candidates.

Sleep can be parallelized across variables, operator families, context keys, nethra kinds, time windows, runs/seeds, and structure-node partitions.

Sleep output must remain proposal-only until runtime evidence upgrades use-rights.

## 17. Test modes

Several modes exist and should not be confused.

- `off`: feature disabled.
- `record`: record/index/report only. Should match off in behavior.
- `shadow`: run diagnostics only. Should match off in behavior.
- `assist`: bounded reversible runtime hints. May change behavior, but must be judged by outcome metrics.
- `assist_feature`: uses an index or feature as part of an assist path. Must prove attribution and avoid overuse.

If `record` differs from `off`, there is a leak.

If `shadow` differs from `off`, there is a leak.

If `assist` differs from `off`, that is expected, but it may be harmful.

## 18. Design risks to keep visible

Current risks:

- **Structure/nethra conflation**: treating the lens as the substrate.
- **Absolute-authority drift**: reading authority records as proof rather than scoped evidence.
- **Role identity confusion**: treating tareth/trass as object identity instead of context role.
- **Trass-as-deletion**: losing reusable structure because it was operationally irrelevant once.
- **Broad uncertainty pressure**: using generic uncertainty signals as if they were local structure.
- **Context-role overmatching**: letting too many provenance matches become local anchors.
- **Expression blowup**: emitting every possible union/intersection instead of utility-backed expressions.
- **Inherited authority laundering**: letting a composite expression inherit the strongest member use-right.
- **Global regime switch drift**: replacing one crude world label with another.
- **Regime quiescence error**: confusing no recent failure with active higher validation.
- **Premature learner insertion**: adding an NN before the learning target is identifiable.
- **Post-hoc authority laundering**: awarding credit to an explanation that was fit after the observation arrived. The explanation accounts for the observation but never predicted it. Storing it as a proposal is correct; installing it as a certified authority record is not.

## 19. Current next design target

The immediate high-value target is to update the memory/sleep path from flat scaffold grouping toward **NethraExpressionIndex** design.

Required next conceptual object:

- `NethraExpressionIndex`: offline index of overlap bridges, subset relations, gated activations, negative gates, coactivation clusters, and active-slice candidates.

Required runtime boundary:

- `record`: report active expressions only.
- `assist_feature`: may use active expressions only for ordering/probe hints unless local evidence earns stronger use-rights.

No authority, skip suppression, fit replacement, or monitoring increase should come from sleep expression proposals by default.

## 20. Bottom line

Dreth is currently best understood as an experimental ledgered attention/search system:

- structure is the shared substrate,
- nethras are scoped lenses over structure,
- roles are context-indexed,
- authority is earned by precommitted prediction accuracy — not by post-hoc fit,
- intervention is a separating diagnostic chosen because rival handles made different conditional predictions,
- regimes are emergent active expressions over overlapping nethras,
- uncertainty must become useful structure before it should drive behavior,
- and negative results are valuable when they identify exactly where a handle is too broad, too weak, too global, or too expensive.
