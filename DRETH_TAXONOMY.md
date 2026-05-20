# DRETH Operational Taxonomy

DRETH terms are scoped to a decision inside the control loop, not to the whole
cycle. A nethra is not simply true or false; it is certified for an operation,
under a context, with bounded authority.

## Nethra

A filter that has earned the right to reduce the search space for a specific
decision in a specific context. Not a code object. Not a label. An operative
reduction: "this distinction matters here, keep it" or "this distinction does
not change the outcome here, collapse it."

## Form

A reusable filter shape — a pattern of operator + arity + behavior signature
that recurs across variables. A form is not automatically a nethra. It becomes
an operative nethra when certified as actually helping the agent decide what to
audit, skip, predict, or collapse.

Form-level role is separate from instance-level role:

```
instance trass ≠ form trass
form trass ≠ instance trass
```

A trass form is a quiet collapse verdict. It stays registered. Its morphology
and history are kept. Its operative shortcut paths are disabled for the current
operation. Instances are not automatically demoted. It only becomes a revocation
trigger when a later joint test or downstream dependency contradicts the trass
certificate.

## The five decisions per cycle

| Decision | Question | Example operative claim |
|---|---|---|
| Observe | What is current state? | — |
| Skip-or-audit | Can I trust the current shortcut? | tareth-for-skip |
| Predict/route | Which form/context handles this? | tareth-for-route |
| Compress | Is there a valid simplification? | tareth-for-compression |
| Reexamine/repair | Which claims need rechecking? | tareth-for-audit |

## Authorization failure conditions

Each tareth-form authorization is a claim that form membership transfers a
result from one instance to another. Failure = transferred result exceeds the
noise envelope when verified directly.

| Authorization | Fails when |
|---|---|
| Shared sentinels | Check discriminative for A does not track output for B |
| Shared compression | B's compressed prediction exceeds noise envelope vs full |
| Instance routing | Full audit on routed instance yields different hypothesis |
| Deferred reassessment | Sentinel failure on any var in this form's scope |

## Operation

The decision target for which a nethra is being used.

- `observe`: read current visible/output state.
- `skip`: decide whether a shortcut may replace a full audit.
- `route`: assign an instance to a form/context handler.
- `compress`: use a simplified prediction under gate conditions.
- `audit_reuse`: reuse prior probes, sentinels, or form evidence.
- `reexamine`: decide which claims must be repaired after failure.

Rule: every tareth/trass claim must name its operation. `trass_for_skip` does
not imply `trass_for_route` or `trass_for_compress`.

## Role

Whether the distinction is load-bearing for a named operation.

- `tareth`: substitution changes the operation outcome; preserve the distinction.
- `trass`: tested substitution does not change the operation outcome; collapse is allowed.
- `uncertain`: evidence is missing, stale, conflicting, or out of scope.
- `false_trass`: local trass claims fail when composed; the interaction is tareth.

Rule: role is an output of a substitution test, not an identity label.

## Context

The concrete conditions under which a role was tested.

Examples:

- parent signature
- form/operator identity
- route family
- gate values
- noise envelope
- visible variable set
- regime/window id
- current operation

Rule: if the evidence class that justified the role changes, the role must be
re-certified before it authorizes reuse.

## Scope

The boundary of validity for a certificate.

Scope should record:

- `operation`: which decision this certificate serves
- `context_predicate`: when it applies
- `targets`: which outputs/decisions were compared
- `substitutions`: which distinctions were tested
- `cycle/window`: when the evidence was earned

Rule: empty scope is not trass. It is no certificate.

## Authority

What the certificate is allowed to do.

- `none`: cannot affect control flow.
- `prefer`: may bias ranking or routing, but cannot skip audit.
- `guarded_reuse`: may reuse if sentinels still pass.
- `skip`: may avoid audit for this operation in scope.
- `propagate`: may be composed into higher nethras.

Rule: certification and authority are separate. A tareth form may authorize
routing but not compression; a trass form may authorize skip but not propagation.

## Joint Test

A test for interaction effects that local tests miss.

For two candidate collapses `A` and `B`:

- `R0`: baseline operation result
- `RA`: result after substituting `A`
- `RB`: result after substituting `B`
- `RAB`: result after substituting both

Interpretation:

- if `RA == R0` and `RB == R0` and `RAB != R0`, the pair is `false_trass`
- if `RAB` differs more than either alone, record interaction evidence
- if all are unchanged, the composed collapse may be trass in scope

Rule: composition must be certified by joint substitution, not inferred from
local trass certificates.

## Re-Certification Triggers

Run role certification when role evidence may have changed:

- first encounter
- parent/signature change
- sentinel failure
- compression mismatch
- routing mismatch
- relevant world/context drift
- reentry after quarantine/collapse
- composition changes

Do not run it merely because a cycle elapsed or an unrelated neighbor changed.

## Tareth Authorizations

A tareth nethra can authorize bounded reuse, depending on authority:

- shared sentinels
- shared compression
- instance routing
- lower audit priority
- propagation into composed nethras

Each authorization must carry its own operation, context, scope, and failure
condition.

## Minimal Record Shape

```python
NethraCertificate = {
    "operation": "skip" | "route" | "compress" | "audit_reuse" | "reexamine",
    "role": "tareth" | "trass" | "uncertain" | "false_trass",
    "authority": "none" | "prefer" | "guarded_reuse" | "skip" | "propagate",
    "context": {...},
    "scope": {
        "predicate": ...,
        "targets": (...),
        "substitutions": (...),
        "window": (...),
    },
    "witnesses": (...),
    "preservation_probes": (...),
    "joint_test": None | {
        "members": (...),
        "R0": ...,
        "RA": ...,
        "RB": ...,
        "RAB": ...,
    },
}
```

