# Dreth

Dreth is a failure-shaped control graph.

It keeps successful behavior closed and cheap. A consequential prediction failure opens the exact boundary that produced it, earns decomposition there, and leaves the same handle intact everywhere else.

## Core object

A **nethra** is an operative handle with four jobs:

1. commit a prediction before its outcome;
2. earn context-relative authority when that prediction succeeds;
3. reduce future work within the horizon it has already predicted successfully;
4. identify the boundary to open when a consequential prediction fails.

Nethras reference other nethras recursively. Shared component nodes connect related handles and make nearby alternatives available for consideration.

## Roles and state

Roles belong to a nethra in a context and operation:

- **tareth**: preserving the distinction changes the predicted consequence;
- **trass**: collapsing the distinction preserves the predicted consequence.

Every authority boundary has two operational states:

- **usable**
- **unusable**

Trust remains graded and relative. A handle gains weight from successful prospective predictions, longer proven horizons, successful use across contexts, and reuse by higher nethras. Runtime permission still requires successful evidence in the current context.

## Runtime

```text
commit prediction at cycle t for horizon h
→ expose it to the outcome at t+h
→ success earns authority through h in that context
→ authority permits reuse through h
→ consequential failure blocks that local boundary
→ failure creates child factors or a relational composite
→ later recurrence may be consolidated explicitly
```

Normal operation performs prediction commitment and outcome comparison. Decomposition, repair indexing, and graph growth happen when failure earns them.

Joint failures create composite nethras over the implicated handles. Individual handles retain their existing authority. Recursive nethras use the same object and the same evidence rules at every level.

The nethra node persists through failure. Each failure blocks its context/horizon authority edge and adds the repair boundary beneath that node.

## Success criterion

Dreth succeeds when failure-shaped authority reduces future work while preserving detection, localizing repair to the responsible boundary, and costing less than the work it saves.

The exhaustive failure classes are:

1. **detection** — a consequential failure passes unseen;
2. **attribution** — the failure opens the wrong boundary;
3. **utility** — prediction checks and repair cost more than the work they remove.

## Example

```bash
python -m dreth
```

The example earns three-cycle authority for a working engine handle, produces a local cold-start failure, lazily creates fuel/spark/air/compression/sensor children, and shows that the warm-context authority remains usable.

## Tests

```bash
python -m pytest -q
```

The tests exercise prospective authority, horizon gating, graph retrieval, local failure boundaries, lazy factorization, relational composites, recursive consolidation, and the two-state authority surface.
