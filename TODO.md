# Dreth implementation order

The authoritative design, pseudocode, function contracts, and acceptance tests live in
[`docs/KERNEL_SPEC.md`](docs/KERNEL_SPEC.md).

## Repository state

- `docs/KERNEL_SPEC.md` is ready for implementation.
- `README.md` describes the current caller-driven executable accurately.
- The existing package remains available as a compact record of the rejected API boundary.
- Neutral identifiers and graph storage can be reused after their semantics match the
  specification.

## Pass 1 — prospective dispatch vertical slice

Create:

- `dreth/protocols.py`
- `dreth/records.py`
- `dreth/regions.py`
- `dreth/event_log.py`
- `dreth/graph.py`
- `dreth/evidence.py`
- `dreth/commitments.py`
- `dreth/authority.py`
- `dreth/dispatch.py`
- `dreth/settlement.py`
- `dreth/accounting.py`
- `dreth/runtime.py`

Deliver one fake operation and provider showing:

1. Dreth invokes the provider.
2. The provider creates its forecast.
3. Dreth freezes the forecast before domain execution.
4. Exact target obligations settle once.
5. An endpoint horizon grants only that endpoint.
6. A prefix trajectory grants only its successfully exposed prefix.
7. One exact offer earns binary authority under the registered evidence policy.
8. Dispatch physically removes named work units.
9. Every removed unit records its complete authority-source set.
10. A harmless miss earns zero extension and opens zero repair.
11. A consequential miss blocks the exact local use or bundle.
12. Runtime accounting reconciles baseline, executed, suppressed, and control work.

Implement acceptance tests 1–13, 19–36, 53–55, and 58 from the kernel specification.

## Pass 2 — contextual roles

Implement `dreth/roles.py`, substitution trials, operation-specific role requirements,
context regions, local exceptions, and exact/prefix horizon behavior.

Implement acceptance tests 14–18.

## Pass 3 — failure-shaped repair

Implement `dreth/repair.py`, failure boundaries, attribution frontiers, lazy factorization,
separating trials, local patches, and relational candidates.

Implement acceptance tests 37–45.

## Pass 4 — recursive ownership

Implement:

- `dreth/coverage.py`
- `dreth/recurrence.py`
- `experiments/higher_sentinel_regression.py`

Recreate the accepted mechanism:

```text
lower co-failure
→ recurrent boundary evidence
→ unusable higher candidate
→ prospective higher authority
→ channel-by-channel coverage
→ physical lower-sentinel suppression
→ local lower-check reopening on higher failure
```

Implement acceptance tests 46–52 and 56–57.

The primary ablation changes one switch:

```text
Dreth full
versus
the same world, seed, providers, lower certs, audits, and sentinels
with higher-owned subordinate suppression disabled
```

Report the result produced by the experiment.

## Pass 5 — offline proposals

Implement complete-scaffold snapshots, recurrence mining, passive background-trass
compilation, and topology-pruning proposals. Every offline product enters as an unusable
candidate. Active authority paths, failure boundaries, failure channels, and repair
provenance survive pruning.

## Completion rule

A pass is complete when its causal behavior executes, its raw accounting reconciles, its
acceptance tests pass, and README states the implemented subset exactly.
