# TODO

This file lists remaining work only. Completed baseline items are intentionally not repeated as TODO entries.

Current priority: implement Dreth's nethra role-surface model in three bounded steps. Do not attempt the entire architecture in one pass.

---

## Current Target: Nethra Role Surfaces and Peripheral Residual Classification

Dreth's next object is not a generic knowledge graph, not a flat memory sleep pass, and not a global regime switch.

The target object is:

> A nethra is a persistent learned handle with context-indexed operating surfaces. Those surfaces determine whether the nethra can constrain focal reasoning, organize peripheral residuals, compose upward, or remain blocked/contested.

The graph/store remains the backing substrate. The operating surface is the runtime-relevant interface.

Core distinction:

- `tareth`: load-bearing surface. It may constrain search, candidate generation, parent ranking, probe choice, prediction, or action-relevant consideration.
- `best_available`: weak/load-bearing fallback surface. It may provide bounded ranking or probe hints, but must not become hard authority.
- `trass`: non-load-bearing surface. It must not constrain primary action/search as authority. It must actively organize residuals, ignored distinctions, low-priority peripheral structure, and context-switch evidence. Trass is not deletion, dormancy, or inactivity.
- `unresolved`: contested or insufficiently organized surface. It may preserve ambiguity/evidence/probe needs, but must not project action authority.

The key missing operation is `CHARGE_RESIDUAL`.

Residuals are not supposed to accumulate forever. Residual pressure should behave like peripheral vision: always doing low-cost classification when spare capacity exists, never interfering with focal work, and reducing unresolved pressure over time in a stable closed context.

Expected stable-context dynamics:

1. Residual count may initially rise.
2. Repeated residuals should become classified, absorbed, summarized, or composed.
3. Clarity should rise.
4. Pressure and recent growth should flatten or fall.
5. Monotonic unresolved residual growth in a closed system is a failure signal, not success.

---

# Step 1 — Record-Only Role Surfaces and Residual Buckets

## Purpose

Create the role-surface substrate without changing runtime behavior.

This is the first Codex pass. It must be record-only/provenance-only. It must prove that the role-surface model can exist beside current `ContextRoleIndex` without changing audit, skip, probe, repair, ranking, or projection behavior.

## Scope

Touch only what is required for passive storage, metrics, export, and tests.

Primary files:

- `dreth/nethra_role_surface.py` — new module.
- `dreth/context_role_index.py` — integrate passive role-surface store.
- `scripts/summarize_context_role_index.py` — report new passive surface metrics.
- `tests/test_nethra_role_surface.py` — new tests.
- `tests/test_nethra_reservoir.py` — keep existing behavior and compatibility tests passing.
- `README.md` and/or `DESIGN_UNDERSTANDING.md` — short record-only model note if needed.

Do not modify `agent.py` behavior in Step 1.
Do not modify `ProjectionIndex` runtime behavior in Step 1.
Do not add runtime assist behavior in Step 1.
Do not add new CLI flags unless tests absolutely require them.

## Required new module

Add:

```text
dreth/nethra_role_surface.py
```

Define, with type hints and docstrings:

- `NethraRoleSurface`
- `ResidualBucket`
- `EvidenceAccount`
- `ProjectionPermission`
- `RoleSurfaceTransition`
- `RegimeTransitionCandidate`
- `NethraRoleSurfaceStore`

Suggested shapes:

```python
ContextKey = str
NethraId = str
```

```python
@dataclass
class NethraRoleSurface:
    nethra_id: str
    context_key: str
    context_family: str
    role_state: Literal[
        "tareth",
        "trass",
        "best_available",
        "unresolved",
        "blocked",
        "contested",
    ]
    load_bearing_score: float = 0.0
    residual_score: float = 0.0
    projection_allowed: bool = False
    residual_collection_allowed: bool = False
    composition_allowed: bool = False
    last_updated_cycle: int = 0
    support_count: int = 0
    failure_count: int = 0
    use_count: int = 0
    helped_count: int = 0
    hurt_count: int = 0
```

```python
@dataclass
class ResidualBucket:
    nethra_id: str
    context_key: str
    residual_count: int = 0
    unresolved_count: int = 0
    absorbed_count: int = 0
    pressure: float = 0.0
    recent_growth: float = 0.0
    clarity: float = 0.0
    co_shift_nethras: dict[str, int] = field(default_factory=dict)
    representative_examples: list[dict[str, Any]] = field(default_factory=list)
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
```

```python
@dataclass
class EvidenceAccount:
    support_count: int = 0
    failure_count: int = 0
    invalidator_counts: dict[str, int] = field(default_factory=dict)
    prediction_lift: float = 0.0
    use_cost: float = 0.0
    helped_count: int = 0
    hurt_count: int = 0
```

```python
@dataclass
class ProjectionPermission:
    nethra_id: str
    context_key: str
    operation_hook: str
    allowed: bool
    strength: Literal["none", "background", "weak", "normal"] = "none"
    reason: str = ""
```

```python
@dataclass
class RoleSurfaceTransition:
    nethra_id: str
    context_key: str
    operation: Literal[
        "ABSORB",
        "CHARGE_RESIDUAL",
        "PROMOTE_ROLE",
        "DEMOTE_ROLE",
        "FRACTURE_IDENTITY",
        "COMPOSE",
        "SPAWN_RESIDUAL",
        "DECAY_RESIDUAL",
    ]
    cycle: int
    reason: str
    pressure_before: float = 0.0
    pressure_after: float = 0.0
    role_before: str = ""
    role_after: str = ""
```

```python
@dataclass
class RegimeTransitionCandidate:
    context_key: str
    cycle: int
    source_nethras: tuple[str, ...]
    pressure: float
    recent_growth: float
    co_shift_count: int
    evidence_refs: tuple[str, ...]
    reason: str
```

## Required store API

Implement `NethraRoleSurfaceStore` with at least:

```python
add_or_update_identity(...)
assign_surface(...)
assign_surface_from_context_role(...)
surface_for(nethra_id, context_key)
charge_residual(nethra_id, context_key, row, cycle, coactive_nethras=())
absorb_residual(nethra_id, context_key, evidence_ref, cycle)
decay_residuals(cycle, budget)
classify_background_residuals(cycle, budget)
promotion_candidates(cycle, min_pressure, min_growth, min_co_shift)
regime_transition_candidates(cycle, min_pressure, min_growth, min_co_shift)
projection_allowed(nethra_id, context_key, operation_hook)
projection_entries(context_key, operation_hook)
summarize()
export_records(limit=200)
```

Step 1 may implement `projection_allowed()` and `projection_entries()` as passive/diagnostic methods only. They must not be wired into runtime projection.

## Step 1 behavioral rules

- `tareth` and `best_available` surfaces can be represented as projection-capable.
- `trass` surfaces must not be primary-projection-capable.
- `trass` surfaces must be residual-collection-capable.
- `unresolved` surfaces preserve ambiguity and evidence, but no primary projection authority.
- Residual pressure must be bounded.
- Representative examples must be capped.
- No hidden-truth/debug fields may be read.
- Record mode must not alter operational behavior.
- Pressure alone must not issue authority, revoke authority, suppress skips, force probes, increase monitoring, or increase repair priority.

## ContextRoleIndex integration

Extend `ContextRoleIndex`; do not replace it.

Add:

```python
self.role_surfaces = NethraRoleSurfaceStore()
```

In `assign_context_role()`, after recording the `ContextRoleRecord`, call:

```python
self.role_surfaces.assign_surface_from_context_role(role_record, node)
```

Mapping:

- `role == "tareth"`
  - increase/maintain `load_bearing_score`
  - `projection_allowed=True` diagnostically
  - `residual_collection_allowed=False` by default

- `role == "best_available"`
  - weak/moderate `load_bearing_score`
  - weak projection permission only
  - no hard authority

- `role == "trass"`
  - low/zero `load_bearing_score`
  - `projection_allowed=False` for primary hooks
  - `residual_collection_allowed=True`

- `role == "unresolved"`
  - `projection_allowed=False` for primary hooks
  - `residual_collection_allowed=True`
  - `composition_allowed=True` only as evidence/proposal, not authority

Keep all existing metrics and aliases currently emitted by `ContextRoleIndex.summarize()`.

Add surface metrics without deleting existing fields:

```text
role_surface_count
load_bearing_surface_count
residual_surface_count
residual_bucket_count
residual_pressure_total
residual_pressure_mean
residual_recent_growth_total
residual_absorbed_count
residual_unresolved_count
residual_clarity_mean
regime_transition_candidates_from_residuals
residual_pressure_persistent_growth_windows
```

Extend `export_records()` with:

```text
role_surfaces
residual_buckets
surface_transitions
regime_transition_candidates
```

## Step 1 tests

Create `tests/test_nethra_role_surface.py`.

Required tests:

- `test_trass_surface_collects_residual_without_projection`
  - create trass surface;
  - charge residual;
  - bucket pressure/count rises;
  - primary projection query returns none.

- `test_tareth_surface_can_be_projection_capable`
  - create tareth surface;
  - diagnostic permission says projection can be allowed.

- `test_best_available_projection_is_weak`
  - best_available permits weak/soft metadata only.

- `test_unresolved_surface_preserves_without_authority`
  - unresolved stores surface/evidence;
  - no primary projection authority.

- `test_residual_bucket_caps_representative_examples`
  - repeated residual rows do not create unbounded examples.

- `test_background_decay_can_reduce_pressure`
  - pressure can decline and clarity can rise under bounded classification/decay.

- `test_correlated_trass_buckets_emit_candidate_not_authority`
  - co-shift creates candidate record;
  - no tareth role is assigned directly.

- `test_context_role_index_backcompat_metrics_survive`
  - all existing reservoir/context-role summary keys still exist.

- `test_context_role_index_export_includes_surfaces`
  - export includes nodes, edges, roles, records, role_surfaces, residual_buckets, surface_transitions.

## Step 1 verification

Run at minimum:

```bash
python -m pytest tests/test_nethra_role_surface.py -q
python -m pytest tests/test_nethra_reservoir.py -q
python -m pytest tests/test_cycle_mechanics.py -q
git diff --check
```

## Step 1 Codex prompt

```text
You are working in the Dooces/dreth repository.

Implement Step 1 only: record-only nethra role surfaces and residual buckets.

Do not modify runtime behavior. Do not modify agent audit/skip/probe/repair/ranking behavior. Do not wire role surfaces into ProjectionIndex yet. Do not add assist behavior.

Add dreth/nethra_role_surface.py with NethraRoleSurface, ResidualBucket, EvidenceAccount, ProjectionPermission, RoleSurfaceTransition, RegimeTransitionCandidate, and NethraRoleSurfaceStore.

Integrate NethraRoleSurfaceStore into ContextRoleIndex as passive storage updated from assign_context_role(). Preserve every existing public method, summary key, compatibility alias, and export shape. Add new surface/bucket metrics and export sections without deleting old ones.

Implement CHARGE_RESIDUAL as a bounded residual-bucket operation. Trass surfaces may collect residuals but must not emit primary projections. Tareth/best_available surfaces may record projection permission diagnostically only. Unresolved surfaces may preserve ambiguity but must not grant action authority.

Add tests/test_nethra_role_surface.py with the required tests listed in TODO.md. Keep tests/test_nethra_reservoir.py and tests/test_cycle_mechanics.py passing.

No hidden-truth/debug fields may be read. Record mode must remain behavior-neutral. Residual pressure must not issue authority, revoke authority, suppress skips, force probes, increase monitoring, or increase repair priority.

Report files changed, new classes, metrics added, tests run, and any #SHORTCUT markers.
```

---

# Step 2 — Mind Store, Assimilator, Projection Gating, and Compaction Integration

## Purpose

Move role surfaces from passive provenance into compact memory and projection eligibility, while still avoiding broad runtime behavior changes.

This is the second Codex pass. It should happen only after Step 1 tests pass.

## Scope

Primary files:

- `dreth/nethra_assimilator.py`
- `dreth/nethra_mind_store.py`
- `dreth/nethra_projection.py`
- `scripts/compact_nethra_memory.py`
- `tests/test_nethra_mind_store.py`
- `tests/test_nethra_projection.py`
- `tests/test_nethra_role_surface.py`
- docs if needed

Do not modify agent behavior yet except through existing memory/projection surfaces if already explicitly invoked by current code paths.
Do not let trass alter primary candidate ranking.
Do not add broad assist.

## Assimilator refactor

Keep existing `Disposition` values for compatibility:

```text
ASSIMILATED
RESIDUAL
CONTRADICTION
SPLIT_CANDIDATE
NOISE
```

Internally add a Dreth-native decision layer:

```text
ABSORB
CHARGE_RESIDUAL
FRACTURE_IDENTITY
SPAWN_RESIDUAL
CONTRADICTION_TO_ACCOUNT
```

Mapping:

- Strong match to existing node -> `ABSORB` -> old `ASSIMILATED`.
- Partial overlap with known trass/unresolved/best_available surface -> `CHARGE_RESIDUAL` -> old `RESIDUAL` boundary result.
- Evidence against success-dominant node -> `CONTRADICTION_TO_ACCOUNT` -> old `CONTRADICTION`.
- Same apparent identity but different authority trajectory or co-shift behavior -> `FRACTURE_IDENTITY` -> old `SPLIT_CANDIDATE`.
- No organizing handle at all -> `SPAWN_RESIDUAL` or new node path.
- No atoms/refs/members/context signal -> `NOISE`.

Do not let residual rows pile up as raw rows if they can be assigned to a trass residual bucket.

## Mind store integration

Extend `NethraMindNode` without breaking old compact files.

Add fields:

```python
role_surfaces: dict[str, dict[str, Any]]
residual_buckets: dict[str, dict[str, Any]]
surface_transitions: list[dict[str, Any]]
```

Keep existing compatibility summaries:

```text
contexts
roles_by_context
use_rights_seen
source_counts
invalidator_counts
evidence_count
behavior_effect_count
authority_effect_count = 0
```

Rules:

- Old compact files must still load.
- New compact files must write surface/bucket summaries.
- Repeated residual rows should compact into bounded bucket summaries where possible.
- `authority_allowed` must remain false.
- `authority_effect_count` must remain zero.
- Hidden truth-like fields must still be rejected.
- Mind-derived rows must still be rejected from delta ingestion as appropriate.

When writing compact mind:

- write `nethra_mind_node` entries with role-surface summaries;
- write compact residual-bucket summaries instead of repeated raw residual rows where possible;
- cap representative examples;
- include residual pressure and clarity metrics in `nethra_mind_summary`.

## ProjectionIndex integration

Refactor `ProjectionIndex` so projection emission can be gated by role surfaces.

Rules:

- `tareth`: may emit normal projection entries.
- `best_available`: may emit weak/ranking/probe projection entries.
- `trass`: must not emit primary `parent_candidates`, `ranking_hint`, or `probe_hint` projections.
- `trass`: may emit background-recognition projections only if explicitly queried by a background/peripheral hook.
- `unresolved`: may emit probe-request metadata only through existing bounded assist paths, not direct authority.

Keep existing projection tests passing or update them to reflect explicit role-surface gating.

## Step 2 tests

Add/update tests:

- `test_assimilator_maps_partial_overlap_to_charge_residual`
- `test_assimilator_keeps_old_disposition_compatibility`
- `test_mind_node_loads_without_surface_fields_from_old_file`
- `test_mind_node_serializes_surface_fields_in_new_file`
- `test_mind_compaction_serializes_surface_buckets`
- `test_repeated_residuals_do_not_write_unbounded_raw_rows`
- `test_projection_index_gates_primary_projection_by_surface_role`
- `test_trass_surface_does_not_alter_primary_candidate_projection`
- `test_best_available_projection_is_bounded`
- `test_authority_allowed_remains_false_in_compacted_mind`

## Step 2 verification

Run at minimum:

```bash
python -m pytest tests/test_nethra_role_surface.py -q
python -m pytest tests/test_nethra_mind_store.py -q
python -m pytest tests/test_nethra_projection.py -q
python -m pytest tests/test_nethra_reservoir.py -q
python -m pytest tests/test_memory_sleep.py -q
git diff --check
```

## Step 2 Codex prompt

```text
You are working in the Dooces/dreth repository.

Implement Step 2 only: integrate role surfaces with the nethra assimilator, compact mind store, and projection eligibility.

Prerequisite: Step 1 record-only role surfaces must already exist and tests must pass.

Refactor NethraAssimilator to keep the existing Disposition enum for compatibility while internally distinguishing ABSORB, CHARGE_RESIDUAL, FRACTURE_IDENTITY, SPAWN_RESIDUAL, and CONTRADICTION_TO_ACCOUNT. Partial overlap with an existing trass/unresolved/best_available organizing handle should charge a residual bucket rather than accumulate unlimited raw residual rows.

Extend NethraMindNode serialization/deserialization with role_surfaces, residual_buckets, and surface_transitions. Old compact files must still load. New compact files must write bounded surface/bucket summaries. authority_allowed must remain false and authority_effect_count must remain zero.

Refactor ProjectionIndex so primary projection emission is gated by role-surface permission. Tareth can emit normal projection entries. Best_available can emit bounded weak projections. Trass cannot emit primary parent_candidates/ranking_hint/probe_hint projections. Unresolved cannot emit direct authority projection.

Do not implement agent-side peripheral classification yet. Do not add broad assist. Do not let residual pressure affect audit, skip, probe, repair, or ranking except through explicit projection tests already covered by this step.

Add/update the tests listed under Step 2 in TODO.md. Run the required verification commands. Report files changed, compatibility handling, projection gating behavior, compact output changes, tests run, and any #SHORTCUT markers.
```

---

# Step 3 — Peripheral Residual Classification, Regime Candidates, and Bounded Assist Experiments

## Purpose

Add the low-cost peripheral classification loop and connect correlated residual pressure to existing uncertainty/regime surfaces without letting pressure become authority.

This is the third Codex pass. It should happen only after Step 1 and Step 2 pass.

## Scope

Primary files:

- `dreth/agent.py`
- `dreth/nethra_role_surface.py`
- `dreth/context_role_index.py`
- `dreth/background_nethra.py`
- `dreth/uncertainty_consolidation.py` only if required
- `scripts/batch_run.py`
- `scripts/summarize_context_role_index.py`
- tests for off/record behavior equality and bounded assist attribution

## Agent-side peripheral classification

Add to `ChainedAgent`:

```python
_run_background_residual_classification(cycle: int) -> None
```

It should:

- run only if role-surface storage exists;
- use surplus or strictly capped budget;
- never consume focal audit budget in a way that changes off/record behavior;
- never issue certs;
- never revoke certs;
- never suppress skips;
- never force probes;
- never increase monitoring directly in record mode;
- classify/absorb residual bucket contents when cheap matches are available;
- decay pressure when residuals become familiar;
- increase clarity as repeated residuals are classified;
- emit `RegimeTransitionCandidate` records when correlated trass buckets grow together.

This must mirror the invariant style of `BackgroundNethraIndex`: passive familiarity is not operational authority.

## Residual pressure mechanics

Implement stable-context behavior:

- pressure rises for novel unresolved residuals;
- pressure rises more when co-shifting residuals recur across related trass buckets;
- pressure decays when repeated residuals are classified;
- clarity rises when residuals are absorbed/summarized;
- monotonic unresolved pressure growth over a closed repeated window is counted as a failure/warning metric.

Do not over-decay useful signals. Decay should reflect classification/familiarity, not silent deletion.

## Regime candidate integration

Residual co-shift should produce candidate records only:

```text
RegimeTransitionCandidate
```

A residual regime candidate may be visible to existing uncertainty/regime/repair summaries, but it must not directly promote trass to tareth.

Promotion still requires existing visible evidence paths:

- sentinel failure;
- separating probes;
- full audit result;
- recurring context evidence;
- explicit context-role assignment;
- bounded assist path with attribution, if later enabled.

Pressure alone must not change action behavior.

## Optional CLI / batch flags

If needed, add conservative flags:

```text
--role-surface-mode off|record|assist_feature
--residual-classification-mode off|record
```

Defaults must preserve current behavior.

Record mode must match off-mode operational behavior.

Assist mode, if implemented, must be bounded, attributable, reversible, and reported. If assist is too large for this step, leave it unimplemented and document it as deferred.

## Reporting

Update reports with sections for:

- role surfaces;
- residual buckets;
- residual pressure dynamics;
- closed-system residual trend;
- co-shift regime candidates;
- projection gating by role;
- record/off behavior comparison;
- assist attribution, if assist mode exists.

## Step 3 tests

Add/update tests:

- `test_background_residual_classification_does_not_change_record_mode_behavior`
- `test_background_classification_reduces_pressure_in_closed_context`
- `test_residual_pressure_persistent_growth_window_is_warning`
- `test_co_shift_residuals_emit_regime_candidate_only`
- `test_regime_candidate_does_not_promote_role_directly`
- `test_record_mode_equals_off_for_small_seeded_run`
- `test_assist_mode_if_present_is_bounded_and_attributed`
- `test_no_hidden_truth_used_by_residual_classification`
- `test_no_unbounded_residual_history_growth`

## Step 3 verification

Run at minimum:

```bash
python -m pytest tests/test_nethra_role_surface.py -q
python -m pytest tests/test_nethra_reservoir.py -q
python -m pytest tests/test_nethra_mind_store.py -q
python -m pytest tests/test_nethra_projection.py -q
python -m pytest tests/test_shadow_residual.py -q
python -m pytest tests/test_uncertainty_consolidation.py -q
python -m pytest tests/test_cycle_mechanics.py -q
git diff --check
```

Also run an off-vs-record deterministic smoke comparison through the existing batch harness if the repo already has an appropriate command.

## Step 3 Codex prompt

```text
You are working in the Dooces/dreth repository.

Implement Step 3 only: peripheral residual classification, residual co-shift regime candidates, and reporting. Do not implement this until Step 1 and Step 2 are already complete and passing.

Add ChainedAgent._run_background_residual_classification(cycle). It must use only surplus or strictly capped budget. It must never issue certs, revoke certs, suppress skips, force probes, increase monitoring directly in record mode, replace fits, or change off/record operational behavior.

Residual pressure should rise for novel unresolved residuals and correlated co-shifts, but repeated stable residuals should become classified/summarized so clarity rises and pressure/recent_growth flattens or falls. Monotonic unresolved pressure growth in a closed repeated context is a warning metric.

Residual co-shifts should emit RegimeTransitionCandidate records only. Pressure alone must not promote trass to tareth. Promotion still requires existing visible evidence paths such as sentinel failure, separating probes, audit result, recurring context evidence, or explicit role assignment.

Add conservative CLI/batch flags only if necessary, with defaults preserving current behavior. If assist mode is too large, leave it deferred. Record mode must equal off mode on operational metrics.

Update summaries/reports with role-surface and residual-pressure sections. Add/update the tests listed under Step 3 in TODO.md. Run the required verification commands. Report files changed, how the peripheral loop is bounded, off/record comparison, tests run, and any #SHORTCUT markers.
```

---

## Non-Negotiable Invariants

- [ ] A nethra is a learned operating handle/lens over structure, not the structure itself.
- [ ] Multiple nethras may touch the same structure.
- [ ] Structure overlap is not proof.
- [ ] Cross-context overlap is downgraded to hint/proposal until local evidence earns stronger use.
- [ ] `tareth`/`trass` are context roles/surfaces, not global identities.
- [ ] Trass is not deletion.
- [ ] Trass is not dormant/inactive; it organizes residuals while non-load-bearing.
- [ ] Tareth constrains focal reasoning only where earned.
- [ ] Best-available is fallback, not hard authority.
- [ ] Unresolved preserves ambiguity; it does not license action authority.
- [ ] Residual pressure is not authority.
- [ ] Residual pressure should decline into clarity in stable closed contexts.
- [ ] Persistent unresolved residual growth in a stable closed context is a warning/failure signal.
- [ ] Regime candidates are not regime switches.
- [ ] Regimes, if used later, are emergent active expressions over co-active nethras, not global enum modes.
- [ ] A composed nethra/expression must not inherit the strongest authority of its members.
- [ ] Authority is earned by visible evidence, not provider confidence, graph proximity, morphology, index membership, recurrence, sleep proposal, pressure, or temporal correlation.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, temporal observers, sleep consolidation, residual classification, or assist logic.
- [ ] Shadow/diagnostic/passive layers may observe; they must not mutate authority unless explicitly promoted through a separately tested bounded runtime path.
- [ ] Record-only indexes must match off-mode behavior.
- [ ] Assist-feature paths must be attributed: which handle/surface/expression changed ordering/probes/filters and whether the outcome improved.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, generic uncertainty signals, or repeated background familiarity must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Passive temporal observers must be bounded by caps/ring buffers/summaries; do not create unbounded event-history growth.
- [ ] Representative residual examples must be capped.
- [ ] Residual classification must not become a second unbounded audit loop.

---

## Deferred / Reference Work After the Three Steps

These are not immediate implementation targets. Do not start them until the role-surface/residual system is implemented and measured.

- [ ] `NethraExpressionIndex` / offline expression mining over role surfaces and compacted nethra identities.
- [ ] `ActiveSlice` compiler from expressions/surfaces into bounded runtime rank/probe/filter/block surfaces.
- [ ] Recognition-collapse metrics over active surface coverage.
- [ ] TemporalEventLedger / passive temporal observers for delayed causality scaffolding.
- [ ] Weighted intervention-cost tracking as diagnostic-only.
- [ ] Dormant alternative revival through strict retrieval/expression gates.
- [ ] Regime-triggered inert re-screening, but not as a global regime switch.
- [ ] Learned ranker/factorizer only after deterministic surface/expression attribution shows rule-based retrieval cannot separate useful anchors from broad noise.

---

# Appendix A — Deferred Prompt: NethraExpressionIndex / Offline Expression Mining

Use only after Step 1 through Step 3 are complete and measured.

```text
You are working in the Dooces/dreth repository.

Current context:
Dreth distinguishes shared structure from nethras. A nethra is a scoped, evidence-bearing, context-activated operating handle over shared structure. Multiple nethras can touch the same structure. Roles such as tareth/trass are context-indexed operating surfaces, not identities. Regimes should emerge as active expressions over co-active nethras, not as predeclared world labels.

Goal:
Add an offline NethraExpressionIndex design/implementation pass.

Purpose:
Read exported runtime/scaffold/mind/surface memory and build proposal-only expressions over overlapping nethras:
  - overlap bridges
  - subset/superset relations
  - union expressions
  - intersection expressions
  - difference expressions
  - gated activations
  - negative gates
  - coactivation clusters
  - recognition-collapse candidates
  - emergent-regime candidates
  - active-slice candidates

Core invariant:
Offline expression mining may create proposals only.
It must not issue authority.
It must not revoke authority.
It must not suppress skips.
It must not replace fit.
It must not increase monitoring.
It must not increase repair priority.
It must not use hidden truth/debug manifest fields.
It must not treat recurrence/frequency/overlap as proof.
It must not create global regime switches.
It must not assign hard_filter use-rights from sleep alone.

Add module:
  dreth/nethra_expression_index.py

Define:

NethraExpressionProposal:
  expression_id
  expression_kind:
    - overlap_bridge
    - subset_relation
    - union_expression
    - intersection_expression
    - difference_expression
    - gated_activation
    - negative_gate
    - coactivation_cluster
    - recognition_collapse_candidate
    - emergent_regime_candidate
    - active_slice_candidate
  member_nethra_ids
  touched_structure_ids
  vars
  contexts
  common_signatures
  common_parents
  operation:
    - union
    - intersection
    - difference
    - gated
    - negative_gate
    - coactive
    - recognition_collapse
  activation_gate
  negative_gate
  source_record_ids
  runs_seen
  seeds_seen
  first_seen_cycle
  last_seen_cycle
  evidence_summary
  suggested_use_right:
    - record_only
    - feature_only
    - ranking_hint
    - probe_hint
    - soft_filter_candidate
    - block_candidate
  invalidators
  authority_allowed: bool = False
  warnings

NethraExpressionSummary:
  input_rows
  structure_nodes_seen
  nethra_handles_seen
  expressions
  expressions_by_kind
  expressions_by_suggested_use
  overlap_bridges
  subset_relations
  gated_activations
  negative_gates
  coactivation_clusters
  recognition_collapse_candidates
  emergent_regime_candidates
  authority_allowed_count
  hidden_truth_fields_seen
  warning_count

NethraExpressionIndexBuilder:
  load_jsonl_rows(...)
  extract_structure_records(...)
  extract_nethra_handles(...)
  build_overlap_bridges(...)
  build_subset_relations(...)
  build_union_intersection_candidates(...)
  build_gated_activation_candidates(...)
  build_negative_gate_candidates(...)
  build_coactivation_clusters(...)
  build_recognition_collapse_candidates(...)
  build_active_slice_candidates(...)
  summarize(...)

Use visible/exported fields only:
  - nethra id
  - vars/components touched
  - contexts/context families
  - role histories/surfaces
  - parent/signature overlap
  - recurring signal overlap
  - source kind overlap
  - sentinel/revocation/failure counters if visible
  - coactivation across cycles/runs
  - residual bucket summaries
  - temporal proximity if temporal records exist

Do not use:
  - truth_parents
  - truth_func
  - truth_delayed_parents
  - truth_latents
  - debug_blind_challenge_manifest
  - relation_type except in a separate post-hoc report section disabled by default

Runtime boundary:
Do not wire expressions into ChainedAgent in this task.
Do not add behavior-changing use.
Output proposals and a summary only.

Add script:
  scripts/run_nethra_expression_sleep.py

Summary sections:
A. Inputs
B. Expressions
C. Overlap and gates
D. Recognition/regime candidates
E. Familiarity not authority
F. Examples
G. Hidden-truth guard
H. Warning that expression proposals are search/familiarity scaffolds only

Tests:
  - repeated overlap creates overlap_bridge proposal
  - subset touched-structure relation creates subset_relation proposal
  - coactive nethras create coactivation_cluster proposal
  - gated activation requires visible gate evidence
  - negative gate requires visible contradiction/failure association
  - recognition collapse is recorded as candidate, not regime proof
  - no expression has authority_allowed=True
  - no expression gets hard_filter from sleep alone
  - hidden truth fields are ignored
  - unrelated nethras remain separate
  - empty input produces empty summary
  - no imports from agent.py
  - no runtime behavior changes

Verification:
  python -m pytest tests/test_nethra_expression_index.py -q
  python -m pytest tests/test_memory_sleep.py -q
  python -m pytest tests/test_cycle_mechanics.py -q
  git diff --check
```

---

# Appendix B — Deferred Prompt: TemporalEventLedger and Passive Temporal Observers

Temporal work remains relevant because delayed causality and lagged structure are underrepresented. It should be integrated as another source of structure and gate evidence for expression mining, not as a separate authority path.

Core invariant:
Default behavior unchanged. Passive observers may record and propose. They may not intervene, issue authority, revoke authority, suppress skips, replace fit, or read hidden truth.

Future temporal records should feed:

- gated activations;
- negative gates;
- lagged coactivation clusters;
- delayed residual explanations;
- recognition-collapse candidates;
- emergent-regime candidates.

---

## Ledger

```text
[D:O!,N,V,E,B,C,A,L,S,Q,M,P0,Cav↓,R0,NB,LG
|R:RoleSurfaceMissing,TrassResidualBucketUnderspecified,ResidualPressureAuthorityLaunderingRisk,PeripheralClassificationInterferenceRisk,ClosedSystemResidualGrowthRisk,ProjectionGatingRisk
|F:split-role-surface-work-into-record-only-surfaces-then-memory-projection-integration-then-peripheral-residual-classification
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
