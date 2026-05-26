# dreth

`dreth` is an experimental architecture for studying how an agent can build, use, revise, relate, and search learned structure under limited attention.

The project is not a production ML system and not a claim of solved AGI. It is a simulation framework for testing a narrower question:

> Can an agent keep useful, evidence-shaped structure available while avoiding global trust, global deletion, global regime switches, and flat brute-force rechecking?

## Current design correction

The current conceptual correction is stronger than the older wording "a nethra is learned structure."

A **nethra is not identical to the structure itself**. A nethra is a scoped, evidence-bearing, context-activated **handle/lens over shared structure**.

The reusable substrate is the structure graph: variables, relations, operators, candidate fits, frontiers, dormant alternatives, probes, experts, role histories, residual patterns, and prior observations.

A nethra touches part of that substrate and carries the conditions under which that touched structure may be used. Multiple nethras may touch the same structure. The same structure can therefore be ranked, filtered, ignored, preserved, blocked, or reopened differently depending on the active context.

The useful separation is:

| Layer | Meaning |
|---|---|
| `StructureGraph` | Shared substrate of learned or observed nodes, edges, operators, candidates, probes, experts, and histories. |
| `Nethra` | Scoped lens/handle touching part of that structure with evidence, activation conditions, use-rights, and invalidators. |
| `ContextRole` | The role a nethra currently has in a particular operation/context: `tareth`, `trass`, `unresolved`, `best_available`, etc. |
| `NethraExpression` | Union/intersection/difference/gated/coactive expression over nethras. |
| `ActiveSlice` | Compiled runtime slice: filters, rank hints, probe hints, blockers, and provenance for the current evidence state. |
| `EmergentRegime` | A stable active-expression basin over co-active nethras that improves prediction/search; not a predeclared world label. |

## Core terms

### StructureGraph

The structure graph is the reusable substrate. It contains learned and observed structure: variables, functions, parent sets, candidate hypotheses, tied frontiers, dormant alternatives, probes, sentinels, experts, residual patterns, temporal traces, context-role records, and scaffold/sleep products.

Structure can be shared across contexts. For example, two different apparent regimes may both touch an `ADD` operator family, a delayed-effect pattern, or a sensor-noise structure.

### Nethra

A nethra is a scoped lens over structure. It does not own the structure. It touches structure and records how that touched structure may be used.

A nethra should carry at least:

- touched structure ids or components,
- evidence/provenance,
- activation conditions,
- use-rights,
- invalidators,
- scope/context,
- role history,
- and relation to other nethras.

A nethra may annotate, rank, filter, block, propose probes, or preserve ambiguity only according to its current use-rights. Recognition is not authority. Recurrence is not proof. Cross-context overlap is at most a downgraded hint until local evidence earns stronger use.

### Context roles: tareth, trass, unresolved, best_available

`tareth` and `trass` are context-indexed roles, not identities of a structure or nethra.

- `tareth`: this nethra currently matters for the operation/context being considered.
- `trass`: this nethra is currently operationally equivalent or irrelevant for that operation/context.
- `unresolved`: available evidence preserves ambiguity or instability.
- `best_available`: the current working handle when nothing better is available, even if uncertainty remains.

Trass is not deletion. A nethra can be trass in one context and tareth in another. A nethra can also remain familiar but not operationally usable.

### Nethra expressions

A regime should not be a monolithic mode switch. It should be a computed expression over active nethras.

Examples:

- `A ∩ B`: structure touched by both nethra A and nethra B.
- `A ∪ B`: structure touched by either nethra A or nethra B.
- `A - B`: structure touched by A but blocked or contradicted by B.
- `A if gate_B`: nethra A is active only when B or a signal condition is active.
- `A coactive-with B`: A and B repeatedly become useful together.

This lets a current context reuse part of one prior substrate, part of another, and locally new structure without declaring a global world switch.

### Emergent regimes

The agent does not need to know that it moved from "world A" to "world B." It only needs to detect that the active nethra set lost recognition power: predictions degrade, old filters stop matching, sentinels fail, ranking lift drops, or familiar handles no longer retrieve usable structure.

That recognition collapse opens a regime-boundary candidate. Old nethras are downgraded to hints. Any local overlap with existing structure is tested. If recurrent overlap improves prediction, parent ranking, probe choice, or repair localization, a new local bridge forms. If multiple bridges co-activate and keep improving search, a regime nethra emerges.

A regime is therefore an active, stable expression over nethras, not a label handed to the agent.

## Active architecture

| Module | Role |
|---|---|
| `dreth/world.py` | Hidden causal worlds and schedules. Provides observed scalar variables, interventions, drift/regime schedules, and blind challenge generation. Hidden debug state is for offline analysis only. |
| `dreth/agent.py` | `ChainedAgent`, the main control loop. Runs audits, fits, sentinels, frontier handling, repair agenda integration, uncertainty consolidation assists, and context-role recording. |
| `dreth/ledger.py` | Core data structures for variable handles, authority records, tied frontiers, dormant alternatives, composites, envelopes, and revocation state. |
| `dreth/fit.py` | Enumerates and scores parent/function hypotheses under the agent vocabulary. Produces best fits, ties, near-ties, and diagnostics. |
| `dreth/sentinels.py` | Selects and checks sentinel probes used to cheaply test whether prior structure still holds. |
| `dreth/regime.py` | Tracks recurring co-failure/regime patterns. Existing regime handling is still less mature than the lens/expression model described here. |
| `dreth/context_role_index.py` | Current provenance index over nethra graph nodes and context roles. Useful, but not yet a full nethra-expression compiler. |
| `dreth/memory_sleep.py` / `dreth/nethra_scaffold_sleep.py` | Offline scaffold grouping. Useful as proposal generation, but still flatter than the desired overlap/subset/gated-expression sleep model. |

## What the current system does

At a high level, the agent repeatedly:

1. Observes current scalar world state.
2. Chooses what needs attention under a limited audit budget.
3. Runs full audits when a variable or handle needs repair or initial structure.
4. Fits candidate parent/function hypotheses using agent-visible probes.
5. Records ties, near-ties, margins, alternatives, novelty, uncertainty, and role changes.
6. Installs the best available working structure into the ledger.
7. Uses sentinels, composites, and role records to reduce repeated work when evidence supports it.
8. Revokes or demotes when sentinels or other visible evidence contradict prior authority.
9. Builds provenance over nethras and roles so learned structure can remain available even when trass, unresolved, or dormant in a current context.
10. Optionally runs shadow/assist layers for uncertainty consolidation, context-role local anchors, background familiarity, and scaffold memory.

The design intentionally separates:

- structure learned or observed,
- nethra handles/lenses over that structure,
- role in a particular context,
- authority/use-rights,
- and runtime influence over search.

## Runtime influence and use-rights

A nethra or nethra expression may have different use-rights:

- `record_only`: may be stored and reported; cannot affect runtime behavior.
- `feature_only`: may annotate or expose provenance; cannot reorder or exclude.
- `ranking_hint`: may reorder existing candidates or probes.
- `soft_filter`: may prioritize touched structure but must preserve fallback.
- `hard_filter`: may exclude structure only when current local evidence has earned that permission.
- `block`: may prevent derivation/use in a scope, for example quarantine.

Cross-context reuse must be downgraded by default. A strong handle in one context becomes a hint in a context where recognition has collapsed until local evidence earns stronger use.

## Offline sleep target

The current sleep code groups familiar records into scaffold proposals. The next design target is stronger: sleep should mine the overlap graph of nethras and structure.

Sleep should eventually propose:

- overlap bridges,
- subset/superset relations,
- union/intersection/difference expressions,
- gated activations,
- negative gates,
- coactivation clusters,
- candidate active slices,
- and emergent regime-expression candidates.

Sleep may propose expressions only. It must not issue authority, revoke authority, suppress skips, replace `fit_var`, increase monitoring, increase repair priority, or treat recurrence as proof.

Runtime should compile only a bounded active slice from the sleep products: hard filters, soft filters, rank hints, probe hints, blockers, invalidators, and provenance. Record mode should remain behavior-neutral. Assist modes must be judged by outcome metrics.

## Core invariants

- Hidden truth/debug manifest fields are offline interpretation only; they must not drive runtime matching, clustering, assist, authority, fit, or skip behavior.
- Authority is earned by visible evidence, not provider confidence, graph proximity, morphology, index membership, recurrence, sleep proposal, or temporal correlation.
- A nethra is not global truth; it is a scoped lens over structure.
- A role is not identity; `tareth` and `trass` are context roles.
- Trass is not deletion.
- Recognition collapse is not proof of a new regime; it is a signal that active coverage failed.
- Cross-context overlap is not authority; it exposes downgraded hints.
- A nethra expression does not inherit the strongest authority of its members. It must earn its own use-rights.
- Record/shadow modes must not change behavior.
- Assist modes may change behavior, but must be bounded, attributed, reversible, and judged by off/record/assist comparisons.

## Known unfinished work

1. **Nethra-expression representation**: the repo currently has provenance indexes and scaffold groupers, not a full expression algebra over overlapping nethras.
2. **ActiveSlice compiler**: runtime lacks a clean compiler from active nethra expressions into filters, rank hints, probe hints, blockers, and provenance.
3. **Sleep expression mining**: offline sleep should move from flat grouping toward overlap/subset/gated/coactivation mining.
4. **Recognition-collapse detection**: the agent needs explicit metrics for when active nethra coverage has failed without assuming an externally named new world.
5. **Regime emergence**: regime handles should emerge from clustered recognition failure plus renewed clustered predictability, not from labels or quiescence alone.
6. **Assist attribution**: any runtime use of nethra expressions must show which expression changed ordering/probes/filters and whether that improved outcomes.

## Useful command shape

Record-only indexing should match off behavior. Assist modes may differ, but must be interpreted through attribution and outcome metrics.

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
```

Interpretation:

- `off == record`: provenance indexing is clean.
- `assist_feature` improves: check match attribution before claiming the index helped.
- `assist_feature` worsens: keep the index as provenance, or restrict runtime use until representation and gating improve.

## Bottom line

Dreth is best understood as a ledgered attention and search system. Its core promise is not merely storing learned structure. Its core promise is making structure searchable through scoped, overlapping, evidence-bounded lenses whose runtime use is context-dependent and revocable.
