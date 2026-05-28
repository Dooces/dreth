# Nethra Local Ledger — 2026-05-20

This ledger records the current working interpretation of nethra after reconciling the repository-facing framing with the corrected conceptual invariant from recent design review.

The public README currently contains only a minimal project marker. This file is therefore not a rewrite of established repo documentation; it is a new local ledger for the architecture as currently understood.

## Core invariant

A nethra is a learned handle that permits cheap reuse until consequential prediction/control failure earns repair work.

A nethra is not a proof object that must be globally recertified whenever the surrounding world changes. It is closer to a working habit or shortcut: use it if the current path reaches it and it is not specifically excluded. Repair is earned by failure, not by speculative hygiene.

The judge is not external conceptual correctness. The judge is observed prediction/control performance under the relevant operation.

## What nethra is not

Nethra is not ATMS-style full context labeling.

It does not carry every assumption on every shortcut and recheck every label on every use. That positive-ledger model creates bookkeeping explosion and is the wrong comparison target.

Nethra is also not morphology clustering. A group of variables that look structurally similar is not automatically a nethra-of-nethras. Structural resemblance can be useful as a heuristic, but it does not confer shortcut authority.

Nethra is not proactive risk management. A reachable but unused handle is not a problem merely because it might be stale. Authority changes when use, failure, or a directly active interface makes it relevant.

Nethra is not a raw-feature ledger. It should not attempt to map every failure surface over pixels, tokens, activations, or all variables. Existing models and tools — CNNs, transformers, simulators, planners, policies, retrieval systems, or symbolic modules — can themselves be treated as handles. Dreth governs when those handles are trusted, when their failure earns decomposition, and when a composite/interface nethra is formed.

## Anti-mistake guardrail

Repeated review mistakes came from importing a managerial or verification prior into a failure-earned architecture.

Wrong prior:

```text
something might be stale, costly, misleading, high-dimensional, or dangerous
therefore proactively inspect, prune, invalidate, recertify, or globally manage it
```

Nethra prior:

```text
use the reached handle if it is not specifically excluded
if use succeeds or no consequential failure is observed, do nothing
if use fails past the operation threshold, repair/factorize/compose locally
if repeated local patches reveal shared structure, consolidate offline
```

The mistake was made because common adjacent frames point in the wrong direction:

```text
ledger        -> ATMS-style bookkeeping explosion
shortcut      -> EBL utility collapse
hierarchy     -> predictive-coding / HRL comparison
diagnosis     -> MBD completeness assumptions
failure       -> proactive safety/verification reflex
stale handle  -> external garbage-collection or cleanup reflex
scale         -> raw-feature dimensionality objection
```

Each frame names a real historical failure mode, but it becomes wrong when applied before checking the nethra invariant. The required check is:

```text
What observed failure, sentinel miss, contradiction, active dependency/interface event, or repeated local patch earned this update?
```

If the answer is none, do not add a proactive invalidation, pruning, recertification, or cleanup rule. Treat the concern as speculation, not architecture.

## Lifecycle

Normal path:

```text
reach nethra
use shortcut
compare expected vs observed only through cheap sentinels / active consequences
if no consequential failure, do nothing
```

Failure path:

```text
failure crosses operation threshold
open implicated nethra path
if subnethras exist, descend
if no useful substructure exists, lazily factorize
patch the smallest boundary that changes the failed prediction/control outcome
leave unrelated nethras untouched
```

Consolidation path:

```text
store local patches first
watch recurrence
when multiple local patches share structure, consolidate offline
create higher nethra only when recurrence earns it
```

## Nethra as shortcut and repair structure

The important shift is that nethra is both:

1. the shortcut that makes ordinary operation cheap, and
2. the search structure used to repair the shortcut when it fails.

Nethra-of-nethra is therefore not just compression. It is also fault-localization topology.

The source_edge nethra does not need a complete internal decomposition before failure. If the source_edge works, no internal accounting is needed. If it fails, the failure earns the cost of decomposition. Newly created subnethras then reduce future repair cost.

## Scaling clarification

Dreth does not scale by ledgering more raw detail. It scales only if consequential failures compress into reusable handles.

A large learned model can be a nethra. A specialist module can be a nethra. A composed interface between tools can be a nethra. The ledger should govern authority at the handle/interface level, not replace the underlying computation.

The real scaling question is:

```text
Do failures compress into reusable handles faster than they fragment into active exceptions?
```

If a failure is random, non-recurring, non-compressible, or below consequence threshold, it should not become an active structure. If local failures recur and share a pattern, that recurrence earns a higher nethra. Fragmentation is a problem only when non-compressible exceptions remain active and keep consuming search/repair attention.

## Granularity

Granularity is discovered by failure localization. It is not preselected globally.

Too high a patch damages unrelated understanding. Too low a patch fails to generalize. The right patch is the smallest internal boundary that explains the consequential failure.

This does not require knowing the correct factorization in advance. A factorization is only wrong in the operational sense if it causes detection failure, attribution failure, or utility failure.

## Exhaustive conceptual constraints

At the conceptual level, the remaining constraints are:

1. Detection failure — the system fails in a decision-relevant way but does not notice.
2. Attribution failure — the system notices failure but repairs the wrong boundary.
3. Utility failure — the repair/checking/consolidation cost exceeds the value saved.

Other objections generally collapse into one of these. There is no independent external judge of a 'wrong factorization.' Wrongness is measured by prediction/control failure and repair utility.

## Composite nethras

Relational failures should create or update composite nethras rather than being forced into one child.

Example:

```text
fuel looks fine
spark looks fine
temperature changes both together
engine fails only under the relation
```

The correct repair locus is not fuel alone or spark alone. It is a composite/interface nethra such as:

```text
cold_start_combustion = relation(fuel, spark, temperature)
```

Composite nethras are normal nethra-of-nethra behavior, not a special exception.

## Operation-indexed authority

The same entity can hold different verdicts for different operations.

```text
x.skip = trass
x.route = tareth
x.compress = unknown
```

A skip verdict is not a route verdict. Using tareth-for-skip as a route proxy is a conflation unless an explicit interface cert authorizes that substitution.

## Certification semantics

Certification means earned shortcut authority for an operation in the tested regime. It does not mean eternal proof, and it does not mean global invalidation whenever nearby structure changes.

If x1 earned a trass skip cert while x0 was excluded from the tested target set, x0 later becoming tareth does not automatically make x1 wrong. x1 keeps firing. If x1 now fails, the failure earns repair. Proactive staling merely because a role changed is positive-ledger behavior unless an active interface says that role transition itself is a failure condition.

## Reachability, authority, and cleanup

Reachability is not the same as active authority.

A nethra can be reachable because a source_edge, log, old cert, index, or archive references it. That does not require proactive cleanup. It only matters operationally when the nethra is activated for shortcutting, routing, repair, composition, or consolidation.

Correct handling:

```text
unreferenced handle -> ordinary garbage-collection candidate
referenced but not activated -> no operational issue
activated and succeeds -> keep authority
activated and fails -> repair, demote, exclude, factorize, or compose as earned by the failure
```

Do not demote or prune a nethra because an external evaluator predicts it might be stale. Authority changes through use, sentinel failure, contradiction, active dependency/interface evidence, or repeated failed repairs.

## Forms

Forms, as morphology buckets, are wrong as an architectural primitive.

A form that groups variables because they share operator, arity, or behavioral signature is not a certified nethra-of-nethras. It is pattern grouping. It may provide useful heuristics, but it should not carry nethra authority.

The replacement primitive is CompositeNethra or an equivalent earned composite handle, created by failure, interaction, recurring local patches, or tested interchangeability — not by resemblance alone.

## Current implementation status to preserve

The live implementation has historically centered on skip certification.

Other declared operations — route, compress, audit, reexamine — should not be described as fully nethra-governed until they have operation-specific cert lifecycles.

Compression based on `pred_passes` is memoization/frequency gating, not nethra authority. It becomes nethra-aligned only when compression has explicit operation scope, evidence, and failure-earned revocation.

Joint false-trass detection becomes nethra-aligned only when it installs durable composite authority instead of immediately discarding the evidence through invalidation.

## Testing implication

Do not judge cold-start cost as failure. The right test is whether cost decays after warm-up and whether later costs are tied to consequential failures.

Useful measurements:

```text
warm-up cost
post-warm-up cost
skip rate by same seed across cycle lengths
failure-triggered audit cost
recurring same-boundary failures
composite creation and later composite-skip use
revocation only after composite/sentinel failure
```

A high cold-start cost is acceptable if it earns reusable structure. A high steady-state cost without failure pressure is suspicious.

## Short operational summary

```text
use shortcut unless specifically excluded
ignore non-consequential error
repair only after consequential failure
factorize lazily when the failed nethra lacks useful children
patch locally before abstracting
create composite nethras for relational failures
consolidate repeated local patches offline
treat tools/models as handles, not rivals
judge by prediction/control failure and utility, not by external correctness
```
