# DRETH TODO

Reflects state as of the forms/v30 removal. Items ordered by dependency.
Do not implement a later item before its prerequisites are confirmed done.
An item is done when a test would fail if the new structure were removed — not when
the structure exists but nothing reads it.

---

## What is actually done

These are live in the code, not aspirational.

**Filter ledger / negative authority** — shortcuts fire by default. Trass-skip requires no confirmation. Sentinel-skip runs the sentinel probes; only failure triggers the failure path. This is the correct pattern and it is implemented.

**NethraCertificate** — defined in `ledger.py`. Has operation, role, authority, context (parents at cert time, visible count, cycle), scope (targets), evidence (changes/trials), witnesses (perturbation snapshots).

**VarNethra.certificates dict** — `role_for(op)` and `authority_for(op)` read from it. Falls back to "untested" if no cert for that operation. `invalidate_certs(event)` handles parent_change, sentinel_failure, structural_mutation, drift, false_trass_contradiction.

**`operation_role` legacy field is gone** — removed from VarNethra entirely. `role_for("skip")` is the only path. T5 completed.

**Skip certification** — `_certify_operation_role` runs perturbation tests at 5 iv_vals, stores witnesses, writes `certificates["skip"]` with scope and evidence. Tareth cert carries witnesses as attribution handles. Trass cert has no witnesses (none propagated).

**Sentinel attribution on failure** — when sentinel fails: witness replay checks if cert basis is still live. `world_changed` → invalidation cascade to descendants. `authority_expired` → drop cert, recertify, queue audit for this var only. This is in the sentinel-fail branch, not the pass branch (Q5 fixed).

**Invalidation cascade** — `ledger.invalidate()` cascades status through descendants. `invalidate_certs("sentinel_failure")` flags the skip cert as untested. Both sentinel failure paths (main loop and dormant sweep) populate `_uncertain_this_cycle`.

**Variable reveal scope expansion** — `_retest_trass_vars` runs after `on_variable_revealed`, re-earns each trass cert against the wider target set.

**Topological order, priority budget, deferred queue** — live.

**Compression cert** — `certificates["compress"]` is written by `pred_passes` accumulation and gates the compression-skip path. Evidence is count-based, not perturbation-based. This is approximate (see N1 below).

**Joint false-trass test** — `_test_joint_false_trass(var_a, var_b)` and `_find_joint_trass_candidates` exist and are wired. `_uncertain_this_cycle` is populated at both sentinel failure sites and consumed at end of `run_cycle`.

---

## What is approximate but acknowledged

These are real code behaviors with known limitations. Not bugs to fix speculatively — fix them when a test case shows they produce wrong outcomes.

**Q7 — skip-as-route proxy** — `available_parents` in `_full_audit_var` uses `role_for("skip") == "tareth"` as proxy for route eligibility. No instance-level route certs exist. `role_for("route")` always returns "untested". The filter logic checks it (`!= "trass"`) but can never gate anything out. The structural fallback (keep current parents regardless) partially compensates by keeping existing parents in scope.

**Q4 — no joint composition test in certification** — individual skip certs accumulate into available_parents without testing whether the combination is actually needed. `predict_under_joint_intervention` exists in the world and is called by `_test_joint_false_trass`, but is not used in `_certify_operation_role`.

**N1 — compression cert is count-based not perturbation-based** — `pred_passes` threshold earns `certificates["compress"]`, but without scope, witnesses, or a propagation test. It is a frequency-threshold shortcut, not a nareth cert. The cert object exists for structural consistency.

**Q2 — status="trass" label** — `n.status` is still set to "trass" by `_install_var` when skip cert says trass, as a convenience label. The cert is the authority; the status label is derived. They are kept in sync on the common path.

---

## What is missing — concrete gaps

### N2 — Route certs do not exist at instance level

`role_for("route")` always returns "untested". The `available_parents` filter checks it but can never exclude anything. The skip-as-route proxy (Q7) is the actual gate.

A route cert answers: "is this var a valid parent candidate for other vars' hypothesis enumeration?" That is a distinct question from "does perturbing this var propagate downstream?" (skip cert). A var could be skip-trass (doesn't propagate under current regime) but route-tareth (it is a genuine parent of some var in the learned structure).

**When to implement**: when a test case exists where the proxy produces the wrong available_parents set and it matters. Until then the proxy is honest and sufficient.

### N3 — Nethra-of-nethra does not exist

Forms were removed because they were morphological bookkeeping, not earned-by-joint-test. Nothing replaced them. The pseudocode's top-down recursion path — composed cert earned by jointly testing all children's collapses, `step_attention` recursing downward — has no implementation. The hierarchy is flat: one cert per var, one level.

A nethra-of-nethra cert would require `predict_under_joint_intervention` called with all children simultaneously, not just pairs. The joint false-trass test (pairs only) is a building block, not the full structure.

**When to implement**: this is the next major work. It requires a concrete scenario: a group of vars that together warrant a shortcut, where individually they don't, and the cert is earned by jointly testing their collapses. The test must exist before the structure is built — not the other way around.

### N4 — Offline consolidation does not exist

No utility gating (`check_cost + failure_cost > saved_cost → demote shortcut`), no abstraction discovery from repeated local patches, no scheduler. Local patches accumulate but never merge into higher abstractions. The DRETH_PSEUDOCODE.py `consolidate()` function is entirely absent.

**When to implement**: after N3. Consolidation merges local certs into composed certs. Without composed certs (N3), there is nothing to consolidate into.

### N5 — Nethra descent on prediction failure is implicit only

When a sentinel fails and world_changed is attributed, the code re-audits the var and its descendants. That's the right direction. The pseudocode's descent protocol — walk the implicated nethra path, localize the failed boundary at the smallest failing leaf, patch only that leaf, leave siblings intact — doesn't exist. The cascade re-audits all descendants, not just the subtree that actually failed.

This only matters once nethras are recursive (N3). With a flat single-level cert structure, "descend the nethra path" and "re-audit the var" are the same thing.

**When to implement**: after N3.

---

## Next concrete step

**N3 is the next real work.** Before writing any code, define one test scenario where nethra-of-nethra earns its cert by evidence, not morphology:

- What group of vars forms the candidate nethra?
- What joint test certifies the group's shortcut?
- What failure localizes to which member?
- What does the cert record look like?

Write the test first. Then build the structure to pass it. Do not build the structure speculatively and then write tests that accept whatever structure was built.

---

## Global acceptance invariant

```bash
python -m pytest tests/ -q
```

27 passing, 1 skipped. This is the baseline. Any item claimed done must leave the suite at ≥ this count with no new failures.
