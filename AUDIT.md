# Dreth Audit — What Is Wrong, What It Does, What It Should Do

The recurring failure: Dreth keeps absorbing representation work. The boundary
is simple — Dreth governs use-rights on learned handles; it does not do the
learning. Every module below violates or blurs that boundary.

---

## 1. REMOVE — No Dreth Role

### `background_nethra.py`
**What it does:** Records "familiar" recurring structures (trass patterns,
dormant alternatives, tied frontiers, uncertainty debt) across cycles and
contexts. Computes cheap recognition scores and familiarity confidence.

**Why it is wrong:** Zero authority. Zero gating. Pure observation and
accumulation — it notices familiar patterns but cannot act on them. Its own
docstring says: "background_confidence means 'familiar enough to recognize,'
not 'safe enough to act.'" That is a description of a feature store, not a
permission system.

**What to do:** Remove. If familiarity-based pattern recognition is needed,
it belongs in a learner that earns a use-right and hands it to Dreth.

---

### `learned_residual.py`
**What it does:** Shadow learning module. Tracks rolling stats on symbolic
residuals (per-func and global), predicts ok/stressed, audits itself against
the passive symbolic baseline. Self-revokes when too many false positives.

**Why it is wrong:** The invariant block in the file says it explicitly: "may
observe, predict, and report diagnostics... must NOT change run behavior, issue
certs, mutate ledger, mark tareth/trass, authorize skips, bypass sentinels."
It is a learning module that was placed inside Dreth while waiting to be
promoted. That wait should happen outside Dreth.

**What to do:** Remove or move to a separate learner directory. Stage 3B
promotion, if it happens, should mean the learner's handle gets a cert from
Dreth — not that the learner moves into Dreth.

---

### `relative_authority.py`
**What it does:** Defines vocabulary and scoring helpers for a future
graph-based nethra authority model (`NethraNodeRef`, `NethraRelation`,
`RelativeAuthorityRecord`, `NethraGraphSnapshot`). Not integrated. Its own
docstring: "intentionally not integrated with the runtime agent... must not
affect skips, cert issuance, revocation."

**Why it is wrong:** It is dead code inside the authority system. Unintegrated
design vocabulary does not belong in the governance layer.

**What to do:** Remove. If the graph-based authority model is needed, design
it outside Dreth first and integrate it through a cert.

---

## 2. REPRESENTATION IN THE WRONG PLACE

### `uncertainty_consolidation.py`
**What it does:** Extracts uncertainty signals from each var, clusters vars by
signal overlap (union-find), proposes what hidden structure might explain them
(`possible_latent_regime`, `possible_missing_operator`, `proxy_confounding`,
etc.), and emits `ConsolidationAssist` hints for monitoring and repair.

**Why it is wrong:** This is pattern recognition and clustering from experience.
It observes fit churn, sentinel failures, and tied frontiers, and infers hidden
structure from those observations. That is what a learner does. The assists it
produces ("increase monitoring here", "this looks like a latent variable") are
representation proposals, not authority decisions. Dreth should receive the
result of this work as a certified handle with earned scope — not perform the
work itself.

**What to do:** Move to a learner. Its output (a handle claiming there is a
latent regime here) should earn a cert from Dreth via the normal substitution
path before it can influence anything.

---

### `nethra_mind_store.py`
**What it does:** Builds and maintains a "canonical persistent nethra graph"
by folding repeated records, sleep products, and experience events into stable
`NethraMindNode` structures across generations. Has caps (`_MAX_NODES = 500`,
`_MAX_EDGES = 2000`). Its own invariant: "authority_allowed and
authority_effect_count are always zero."

**Why it is wrong:** A representation graph where authority is always zero is
a learner artifact. It accumulates provenance, source counts, behavior effect
counts, lift history, role history, and invalidator counts — all representation
data — under the governance namespace. The fact that it never issues authority
is the admission that it does not belong here.

**What to do:** Move to a separate learner pipeline. The mind graph should
produce handles; those handles should earn certs from Dreth. Dreth should not
own the graph.

---

### `memory_sleep.py` (the consolidation and proposal parts)
**What it does:** Offline sleep consolidation. Reads background nethras,
context-role records, uncertainty cases, and authority debt from run outputs.
Groups them by recurrence across runs and seeds. Builds `ScaffoldProposal`
objects describing pattern families (trass families, unresolved families,
dormant alternatives, tied frontiers). Proposes use_rights.

**Why it is wrong:** Pattern extraction, recurrence counting, family grouping,
and proposal generation are learning work. The fact that it runs "offline"
does not make it governance. Proposals are representation artifacts. Dreth's
boundary is not defined by when the computation runs; it is defined by what
the computation is.

**What to do:** Move to a learner pipeline. The `HIDDEN_TRUTH_LIKE_FIELDS`
filter (what fields must never be read) is the one governance-relevant piece
and can stay as a shared constant.

---

### `ResidualBucket` and pressure tracking in `nethra_role_surface.py`
**What it does:** `ResidualBucket` accumulates residual counts, unresolved
counts, absorbed counts, pressure, and recent growth per nethra per context.
`charge_residual`, `absorb_residual`, `decay_residuals`, and
`classify_background_residuals` maintain the pressure state. `promotion_candidates`
surfaces nethras whose pressure exceeds a threshold.

**Why it is wrong:** Residual pressure is a representation metric — it tracks
how much unabsorbed signal a nethra is carrying. That is a property of the
learner's state, not of the authority record. Dreth should know whether a
handle has earned authority; it should not track how much residual it is
carrying.

**What to do:** Remove `ResidualBucket` and all pressure tracking from the
role surface store. The `NethraRoleSurface` record itself (what role a nethra
occupies in a given context: tareth/trass/unresolved/best_available) is
governance-correct and should stay. The residual pressure mechanics should
move to the learner.

---

## 3. MIXED — Governance Frame Correct, Representation Parts Must Leave

### `nethra_runtime_memory.py` — `SalienceScorer`
**What it does:** Multi-component salience scoring over loaded nethra records:
`context_match`, `atom_overlap`, `recency`, `prior_success`, `prior_lift`,
`candidate_reduction_lift`, `probe_lift`, `audit_saved`, `sentinel_survival`,
`specificity`, `failure_count`, `revocation_count`, `stale_evidence`,
`broad_atom_penalty`, `quality_regression`, `sentinel_failure`.

**Why it is wrong:** This is a learning algorithm. It uses success/failure
history, lift history, atom overlap, and temporal recency to rank candidates.
That is exactly what a ranker module does. It does not gate authority — it
computes what to prefer.

**What to do:** Remove `SalienceScorer` from this file. The rest of
`PersistentNethraIndex` (use-right enforcement, `query()` filtering by
use_right, `_record_event` for attribution, `RuntimeMemoryMetrics`) is correct
governance and stays. The ranking signal should come from a learner handle
that has earned ranking authority, not from a scoring function embedded in the
authority layer.

---

### `scaffold_memory.py` — proposal matching
**What it does:** Loads offline proposals and matches runtime records against
them by atom overlap, context overlap, and signature overlap. Produces
familiarity scores. Gates by `authority_allowed=False` and demotes `hard_filter`
to `record_only`.

**Why it is wrong:** The matching logic (familiarity scoring, overlap
computation, confidence_as_familiarity) is representation work. The gating
(authority_allowed=False enforcement, hard_filter downgrade, broad generic
debt filter) is correct governance.

**What to do:** Strip the matching and familiarity scoring out. The governance
piece — enforcing that proposals cannot claim authority_allowed=True, blocking
hard_filter use_rights — stays.

---

### `context_role_index.py` — node explosion
**What it does:** Tracks every hypothesis form ever seen (`var_fit`,
`tied_frontier_candidate`, `dormant_alternative`, `composite`, etc.) as graph
nodes, indexed by var, parent, component. Grows without bound until the
per-var cap added in the recent fix. Also stores `roles` (every role assignment
ever made), `role_surfaces`, and `surface_transitions`.

**Why it is wrong:** A node for every candidate hypothesis that was ever
considered is representation bookkeeping. Dreth needs to know what cert a var
has earned in what scope. It does not need a graph of every hypothesis form
that appeared during exploration.

**What to do:** The cert provenance record (what role earned, in what context,
what scope, what invalidates it) is governance and stays. The `tied_frontier_candidate`
nodes should not persist here — tied frontier state lives on `VarNethra` in
`ledger.py` where it belongs. Reduce this module to: current cert per
(var, operation) and the context scope that cert was earned in.

---

## 4. WRONG BEHAVIOR IN `agent.py`

### `run_cycle` calls representation modules every cycle
In `run_cycle` (lines 4120-4127), every cycle runs:
- `_run_uncertainty_consolidation(cycle)` — clusters uncertainty signals
- `_run_authority_strength(cycle)` — authority debt classification
- `_run_background_nethra(cycle)` — pattern observation
- `_run_background_residual_classification(cycle)` — residual classification

The first, third, and fourth are representation work being called as if they
are part of the governance loop. Dreth's cycle should be: check certs, run
sentinels, audit as needed, issue/revoke certs. Pattern observation and
residual classification should not be in this loop.

**What to do:** Remove `_run_uncertainty_consolidation`, `_run_background_nethra`,
and `_run_background_residual_classification` from `run_cycle`. If these
operations are needed, they should run in a separate learner loop.
`_run_authority_strength` should stay — it gates derivation.

---

### `_context_role_index_record_var_fit` called on every audit
Every time a var is audited, `_context_role_index_record_var_fit` is called,
creating or updating a node in the context role index for every hypothesis form
seen. This drives the node explosion.

**What to do:** Record only when a cert is issued or revoked, not on every
audit. The cert event already carries the context and scope — that is the
provenance record Dreth needs.

---

### `_collapse_tied_frontier` — documented premature (agent.py:22-25)
Collapses the tied frontier when the score landscape narrows to one candidate
in a single audit. Should require regime-survival evidence (stable_count +
distinct context_keys ≥ 2). This is a known bug, acknowledged in the module
header.

**What to do:** Fix the collapse condition: require `stable_count >= 2` AND
`distinct_context_keys >= 2` before collapsing.

---

## 5. STRUCTURAL — Output Bloat Sources

### `experience_events` accumulation (partially fixed)
`PersistentNethraIndex.events` now capped at 200 via `deque(maxlen=200)`.
But the events themselves (`active_nethras`, `candidates_before/after`,
`probes_before/after`, `behavior_effect`, `authority_effect`) are training
signal data — the kind of feedback a learner needs to improve its rankings.
This is not governance provenance.

**What to do:** The cap prevents unbounded growth. But longer term, experience
events should not accumulate inside the authority index at all. They should
go directly to a learner feedback channel.

---

### `evaluation` field (235KB per 100-var run)
The `blind_challenge_behavior` and `blind_challenge_manifest` fields in the
evaluation export contain the full oracle manifest and per-cycle behavioral
delta. These are diagnostic artifacts, not authority records.

**What to do:** Write evaluation data to a separate file, not inline in the
run JSONL record. The run record should be authority state only.

---

## What Dreth Actually Needs

The correct Dreth state for a world of N vars is:

```
per var:
  - current cert (operation, role, scope, earned_by, what_invalidates)
  - sentinels (probe tuples that validate the cert)
  - noise envelope (certified tolerance ε)
  - tied frontier if unresolved (competing candidates with supporting evidence)

per composite (jointly-tareth var pairs):
  - composite cert (members, joint probe, scope)
```

That is it. For 10 vars: dozens of small records. For 100 vars: hundreds.
Everything else in the current codebase is either representation work that
belongs in a learner, or diagnostic output that belongs in a separate file.

---

## Summary Table

| Module | Status | Action |
|--------|--------|--------|
| `ledger.py` | Correct | Keep |
| `sentinels.py` | Correct | Keep |
| `agent.py` core | Correct | Keep (fix `_collapse_tied_frontier`) |
| `authority_strength.py` | Correct | Keep |
| `regime.py` | Governance-adjacent | Keep (cert co-failure tracking) |
| `fit.py`, `world.py` | Correct | Keep |
| `records.py` | Diagnostic write-only | Keep |
| `shadow_policy.py` | Diagnostic predictor | Keep (never mutates authority) |
| `background_nethra.py` | Representation | **Remove** |
| `learned_residual.py` | Learning, zero authority | **Remove** |
| `relative_authority.py` | Unintegrated dead code | **Remove** |
| `uncertainty_consolidation.py` | Pattern clustering | **Move to learner** |
| `nethra_mind_store.py` | Representation graph, authority=0 | **Move to learner** |
| `memory_sleep.py` | Offline learning pipeline | **Move to learner** |
| `auto_sleep.py` | Calls memory_sleep | **Move to learner** |
| `nethra_memory_store.py` | Mixed — strip use_right assignment | **Fix** |
| `nethra_runtime_memory.py` | Mixed — strip SalienceScorer | **Fix** |
| `scaffold_memory.py` | Mixed — strip matching logic | **Fix** |
| `context_role_index.py` | Mixed — strip node explosion | **Fix** |
| `nethra_role_surface.py` | Mixed — strip residual pressure | **Fix** |
| `nethra_assimilator.py` | Representation — answers "what nethra explains this row" | **Move to learner** |
| `nethra_projection.py` | Index for mind store lookups | **Move to learner** (with mind store) |
| `nethra_scaffold_sleep.py` | Sleep + scaffold pipeline | **Move to learner** |
