# Nethra Invariants — 2026-05-20

This file records the current invariant checklist for Dreth/nethra. It is deliberately operational: each item is meant to prevent a recurring implementation or interpretation error.

## Authority and failure

1. Observed prediction/control failure is the judge. No external correctness oracle decides whether a nethra is wrong.
2. Use succeeds -> do nothing. Success does not earn extra bookkeeping.
3. No consequential failure -> no repair. Do not repair because something might be stale, risky, old, ugly, suspicious, or high-dimensional.
4. Repair is earned by failure. A nethra is opened, patched, demoted, excluded, factored, or composed only after a relevant failure event.
5. Failure must be consequential. Noise, harmless mismatch, or irrelevant difference does not force repair.
6. Low-cost bounded failure is allowed. It is the learning channel, not a defect to preemptively eliminate.
7. High-cost domains require stricter thresholds, sandboxing, or external constraint. That is threshold policy, not a refutation.

## What nethra is

8. A nethra is a handle, not a full proof. It permits cheap reuse; it is not a complete model of all assumptions.
9. A nethra is both shortcut and repair index. The same structure that enables cheap reuse is the structure opened when failure occurs.
10. Nethra-of-nethra is still nethra. Higher handles obey the same use/failure/repair semantics.
11. Composite nethras are normal nethras. They are not special exceptions.
12. Tools can be nethras. CNNs, transformers, planners, simulators, policies, retrieval systems, symbolic modules, or other specialist tools can be wrapped as handles.
13. Dreth governs trust, failure, decomposition, and composition. It does not replace all underlying computation.

## What nethra is not

14. Nethra is not a raw-feature ledger. Do not ledger every pixel, token, activation, or variable.
15. Nethra is not ATMS-style full context labeling. It does not carry every assumption on every shortcut and recheck every label on every use.
16. Nethra is not morphology clustering. Similar shape does not confer authority.
17. Nethra is not proactive risk management. The system is not an external janitor deciding what might be worth pruning or recertifying.
18. Trass is not an intractable graveyard. It means operational collapse/equivalence for an operation.
19. Tareth is not global importance. It means a distinction matters for a named operation.

## Lazy decomposition and granularity

20. Do not prebuild full decomposition. If the source_edge works, leave it closed.
21. Lazy decomposition: if a source_edge fails and has no useful children, failure earns factorization.
22. Granularity is discovered by failure localization. It is not preselected globally.
23. Patch the smallest boundary that explains the consequential failure.
24. A source_edge failure does not automatically invalidate siblings or require global cascade.
25. A nethra can fail locally without damaging unrelated understanding.
26. A weird factorization is not wrong unless it fails operationally.
27. There is no independent judge of a wrong factorization. Wrongness collapses into detection, attribution, or utility failure.

## Local patching and consolidation

28. Local first, abstract later. Store a local patch first; consolidate only if recurrence reveals shared structure.
29. Offline consolidation is earned by recurrence. Do not force abstraction at first failure.
30. Repeated local failures can create a higher nethra.
31. If failures are random, non-recurring, non-compressible, or below consequence threshold, do not active-structure them.
32. Fragmentation is fatal only if non-compressible exceptions remain active and keep consuming search/repair attention.
33. Dreth scales by compression, not enumeration.
34. The real scaling question: do consequential failures compress into reusable handles faster than they fragment into active exceptions?

## Composite/interface nethras

35. Relational failure creates or updates a composite nethra.
36. If failure lives in A x B, do not force it into A or B alone.
37. Composite authority should not pretend children are individually proven.
38. The relation may be tareth while individual children remain trass for that operation.
39. Durable joint evidence should create/update composite nethra. Do not detect interaction and immediately discard it.
40. Child certs do not automatically compose upward. Composition requires joint evidence.

## Operation-indexed authority

41. Authority is operation-indexed.
42. A cert for one operation is not a cert for another.
43. Skip cert is not route cert.
44. Tareth-for-skip cannot automatically stand in for tareth-for-route.
45. Compression cert is not a frequency counter.
46. pred_passes is memoization evidence, not nethra authority by itself.
47. Declared operations need live cert lifecycles.
48. Do not describe declared-but-unimplemented operations as live.
49. The current live cert surface has historically centered on skip; other operations must not be described as fully nethra-governed until implemented.
50. Route/include by default unless explicitly excluded by a route cert.
51. Legacy global labels must not bypass operation certs. status == trass is not equivalent to role_for("skip") == trass.

## Certification semantics

52. Certs are scoped authority, not eternal truth.
53. Certification means earned shortcut authority for an operation in the tested regime.
54. Certification does not mean global invalidation whenever nearby structure changes.
55. Scope change alone does not automatically invalidate.
56. Only active dependency/interface evidence or actual failure earns revocation.
57. No proactive scope-ledger invalidation.
58. Do not recertify merely because the world expanded unless the expansion matters through use, failure, or an active interface.
59. A cert must carry enough evidence to explain its authority: operation, role, scope/targets, trials, changes, and witnesses where applicable.
60. Evidence accounting must be honest.
61. Trials count only probes actually run.
62. A skipped probe is not evidence.
63. No positive-ledger drift. Do not turn every structural change into a recertification event.

## Sentinels and witnesses

64. Sentinels are cheap failure detectors. They do not prove the whole nethra every cycle.
65. Witnesses are attribution handles. Replay/open them when failure earns it, not as periodic proof hygiene.
66. No periodic revalidation just to protect hidden truth.
67. Cheap paths stay cheap unless observed behavior invalidates them.
68. No salience polling as default.
69. No compression spot-checks by default.
70. Polling or spot-checking is valid only when tied to an active consequence path.

## Reachability and cleanup

71. Reachability is not active authority.
72. A referenced nethra is not automatically in the active search/use path.
73. An unreferenced handle is an ordinary garbage-collection candidate.
74. A referenced but inactive handle is not an operational problem.
75. Do not prune because something might be stale.
76. Demote/prune only when activation, failure, or interface evidence earns it.

## Conceptual constraint set

77. Detection failure: the system fails in a relevant way and does not notice.
78. Attribution failure: the system notices failure but repairs the wrong boundary.
79. Utility failure: repair/check/consolidation costs more than it saves.
80. Those three are the conceptual constraint set.
81. Any proposed objection should reduce to detection, attribution, utility, or be discarded.

## Testing and interpretation

82. Cold-start cost is not failure.
83. A high early cost is acceptable if it earns reusable structure.
84. Steady-state cost without failure pressure is suspicious.
85. Judge cost after warm-up, not only total runtime.
86. Same seed across cycle lengths tests amortization.
87. Skip rate rising with cycles supports cold-start interpretation.
88. High skip rate alone is not enough; distinguish warm-up, sentinel maintenance, failure repair, and recurring unresolved boundary.
89. Test one premise at a time.
90. One mechanism, one invariant, one expected success condition, one expected failure condition.
91. LLM-generated analogies are not tests.
92. Do not test invented cartoons.
93. If a toy adds global intractable graveyards, raw Euclidean failure radii, no composition, or no operation certs, it is not testing Dreth.
94. Do not call implementation debt a conceptual refutation.
95. Do not call conceptual coherence an implementation success.
96. Documented code behavior must be separated from conceptual architecture.

## Qualia and scope boundary

97. No qualia goalpost shift. If Dreth gives an operational relational account, do not demand an extra context-free phenomenal substance unless Dreth claims to solve that metaphysical problem.
98. The qualia account is a scope boundary, not a fatal gap. It may not solve the hard problem; that does not refute the operational account.

## Review guardrail

99. Every update must answer: what observed failure, sentinel miss, contradiction, active dependency/interface event, or repeated local patch earned this update?
100. If no event earned it, do not add the rule.
