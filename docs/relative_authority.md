# Relative Authority And The Nethra Graph

## A. Problem with the old frame

The older "certified" language made Dreth authority sound absolute, as if a
nethra had been proven true once and for all. That is the wrong frame.

A binary uncertified -> certified -> revoked story is too brittle. Useful but
fallible nethras should not be discarded when they remain the best available
search filter in a local scope. A local failure should usually create a
boundary, exception, or narrower scope instead of universally invalidating the
structure.

## B. Corrected frame

A cert is an earned authority record, not proof of truth. It records that a
nethra or handle has earned more trust than local alternatives for a specific
scope, context, and consequence budget.

Authority is context-indexed and relative to alternatives. "Usable" and
"not usable" are operational decisions; trust is graded. High-reuse and
high-stability nethras are generally more authoritative, but still defeasible.

Nethras should reference each other and share common nodes. A stable nethra is
not isolated proof; it is a graph-mediated handle whose authority comes from
reuse, contrast, support, failure history, and local competitors.

## C. NethraGraph concept

- nethra node: a graph node representing a nethra, handle, scope, or reusable
  filter structure.
- relation edge: a directed relation between nethra nodes, such as dependency,
  conflict, substitution, coactivity, contextual win/loss, or exception.
- context key: the local regime, scope, target, intervention family, or
  consequence budget in which evidence was observed.
- relative authority: the current trust assigned to a node compared with local
  alternatives in the same context.
- local competitor set: the alternatives that could plausibly replace or beat a
  node for the current context.
- shared node/common component: a reused substructure that connects multiple
  nethras and helps evidence transfer without making authority absolute.
- dormant contrast source: a failed or beaten nethra retained as useful contrast,
  baseline, or search guidance.
- exception boundary: a local limit that says where a nethra should not apply,
  without revoking all of its authority elsewhere.

## D. Failure handling

Failure adds evidence and stress; it does not automatically prove that a nethra
is globally useless. The first response should be to search nearby graph
alternatives and compare candidates by consequence-weighted error and repair
cost.

When possible, failure should be localized into a narrowed scope or exception
boundary. Global revocation should be reserved for evidence that the nethra has
broad misleading authority across contexts, not just one local loss.

Dormant alternatives should remain available as contrast and search references.
A nethra that loses in one context may still be the best available filter in
another.

## E. What remains unchanged for now

`NethraCertificate` remains the current implementation record. Route certs,
audit certs, sentinels, `fit_var`, and policy selection remain behaviorally
unchanged.

This document changes semantics and future direction only. It does not define a
new runtime algorithm.
