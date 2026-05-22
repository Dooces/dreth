# Implementation Log

Session: 2026-05-22
Plan: /home/dooces/.claude/plans/understand-the-invariants-and-cryptic-reddy.md

Each entry: what changed, which invariant it addresses, what the before/after behavior is.

---

## P1-A — Activate adaptive probe budget
**File:** `dreth/agent.py` — `_full_audit_var`
**Invariant:** #79 (utility failure: repair costs must not exceed savings)
**Before:** `_adaptive_probe_budget(_n_hyp)` computed, result discarded; all audits use flat `self.intervention_budget` regardless of hypothesis space size.
**After:** Adaptive budget is the floor for all audits; repair escalation (`_var_budget_escalation`) takes the max, so failure-earned escalation still dominates when active.
**Status:** DONE — `agent.py:369–375`. `_adaptive = self._adaptive_probe_budget(_n_hyp); budget = max(self._var_budget_escalation.get(var, 0), _adaptive)`. Header + fit.py comments updated to reflect activation.

---

## P1-B — Wire separating probes into next audit
**Files:** `dreth/agent.py` — `_update_tied_frontier`, `_full_audit_var`; `dreth/fit.py` — `fit_var`
**Invariant:** Attribution failure (cross-audit frontier memory, ambiguity is first-class)
**Before:** `TiedFrontier.separating_probes` always `()`. `fit_var` has `frontier_candidates` bias commented out. Frontier state accumulates but never influences the next audit's probe choices.
**After:** `_update_tied_frontier` calls `_derive_separating_probes` (already exists) and stores result on the frontier. `fit_var` gains optional `forced_probes` parameter; when provided, those probes are guaranteed to be included, with remaining budget filled by discrimination pool. `_full_audit_var` passes `frontier.separating_probes` when the var has an active frontier.
**Status:** DONE
- `fit.py:252–258` — added `forced_probes: Optional[Tuple[Tuple[int, float], ...]] = None` param.
- `fit.py` — `_forced` injected before discrimination pool; `_remaining_budget` slots filled by pool.
- `agent.py:_update_tied_frontier` — all four frontier-creation paths now call `_derive_separating_probes` and store result. On same-candidate-set update, separating probes refreshed from latest audit.
- `agent.py:_full_audit_var` — passes `frontier.separating_probes` as `forced_probes` when frontier exists.

---

## P1-C — Provisional trass: lightweight detection check
**File:** `dreth/agent.py` — trass dispatch block (~line 1821)
**Invariant:** #89-96 (provisional cert should not have hard-suppress authority yet); Detection failure
**Before:** Provisional trass exits with `continue` before sentinel/audit path. Comment at line 1803 says "queue for audit" but code does not queue. No detection during provisional period beyond cascade.
**After:** Provisional trass runs one lightweight op-role probe (single perturbation point) each cycle. If it shows propagation → cert immediately invalidated, var queued for full audit. If not → increment provisional counter as before. Misleading "queue for audit" comment replaced with accurate description.
**Status:** DONE
- Added `_provisional_trass_probe(var)` helper (~line 577 after existing helpers): picks spread perturbation furthest from current state, 2 world queries, returns True if any target changes beyond tolerance.
- Modified provisional trass block (~line 1858): hard-suppress path unchanged; provisional path now runs probe before crediting the skip. Probe failure → cert invalidated, var added to `needs_audit`, continue. Probe pass → sentinel_passes increment as before.
- `total_interventions += 2` on each provisional probe run.
- Misleading "queue for audit" comment replaced with accurate description of the new behavior.

---

## P2 — _maybe_demote status leak
**File:** `dreth/agent.py` — `_maybe_demote`
**Invariant:** #51 (legacy labels must not bypass operation certs)
**Before:** `_maybe_demote` at line 1634: `n.role_for("skip") == "trass" or n.status == "trass"` — status-only-trass vars moved to dormant immediately without cert-stability evidence.
**After:** Gate on cert only: `n.role_for("skip") == "trass"`. Status field not read for dormancy decisions.
**Status:** DONE — `agent.py:_maybe_demote`: removed `or n.status == "trass"` from dormancy gate. Comment explains why.

---

## Test Results

**29 passed, 1 skipped, 3 pre-existing failures.**

Pre-existing failures (not introduced by this session):
- `test_certify_math.py::test_tareth_cert_changes_are_analytically_exact` — asserts `cert.trials==5` but code counts only non-skipped probes (4 when v=0.5 is near current state). In `_certify_operation_role`, untouched.
- `test_certify_math.py::test_trass_cert_changes_are_analytically_zero` — same `trials` accounting issue.
- `test_certify_math.py::test_false_trass_joint_evidence_is_analytically_correct` — asserts cert role set to `untested` after joint test; implementation leaves it `trass`. In `_test_joint_false_trass`, untouched.

Also fixed latent bug in `_derive_separating_probes` (line 1348): `fn(*args)` → `fn(args)`. FUNC_LIBRARY functions take `List[float]`, not varargs. This bug was dormant because `separating_probes` was always `()` before P1-B activated the path.
