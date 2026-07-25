# Dreth Kernel Specification

Status: implementation source of truth for the next kernel pass.

The current Python package is a caller-driven ledger skeleton. This specification defines
the causal boundary, records, functions, state transitions, and acceptance tests required
for an implementation of Dreth.

## 1. Success criterion

Dreth research has two separate success measurements.

### 1.1 Valid experiment

An experiment is valid when:

- Dreth performs every authority, dispatch, failure, attribution, and coverage transition;
- every operational claim was committed before its exposure;
- the runtime receives visible observations and declared operation semantics;
- the evaluator alone receives hidden world state;
- actual work, control work, failures, and unresolved outcomes are reported;
- the implementation follows this specification.

A run that saves zero work, misses failures, localizes poorly, or costs more than it saves
is a successful experiment when it tests this mechanism faithfully and reports that result.

### 1.2 Effective Dreth behavior

A Dreth run is effective when earned authority removes actual consideration or computation,
consequential failures remain detectable, repair reaches the responsible local boundary, and
saved work exceeds Dreth's control cost.

The three mechanism failure classes are:

1. `detection`: a consequential failure remains unseen;
2. `attribution`: a detected failure opens or repairs the wrong boundary;
3. `utility`: control and repair cost exceed the work removed.

Passing unit tests establishes implementation fidelity. Performance experiments establish
the behavior of the mechanism in a tested world.

## 2. Concept lock

The implementation must preserve these statements.

1. Dreth is an economy of consideration. It governs when distinctions, computations,
   monitors, models, and candidate paths receive operational work.
2. A nethra is a scoped operative handle over learned or supplied structure. It binds a
   provider, a touched structure slice, activation scope, evidence, use claims, and repair
   topology.
3. The same nethra enables cheap use and provides the path opened during repair.
4. Nethras reference nethras recursively. A higher nethra follows the same prospective
   evidence and local failure rules as a lower nethra.
5. Shared structure permits local retrieval of related nethras. Retrieval produces
   candidates with their existing authority state.
6. Providers emit proposals and predictions. Every provider output enters Dreth as a
   candidate.
7. Prospective exposure is the source of operational authority. A commitment exists in
   the immutable event log before its target outcome or intervention result is available.
8. Tareth and trass are roles of a distinction for one operation, context region,
   substitution, target set, and horizon coverage.
9. Tareth records a consequential difference under tested substitution. Trass records
   consequential equivalence under tested substitution.
10. Authority belongs to an exact executable use claim. Runtime permission resolves to
    `USABLE` or `UNUSABLE`.
11. Successful prospective evidence can earn authority. Harmless mismatch records evidence
    and leaves the operative graph closed. Consequential failure spends repair work.
12. Consequential failure blocks the exact local dispatch path that produced it and opens
    a failure boundary there.
13. Factorization begins after a consequential failure reaches an undecomposed boundary.
14. Factor and repair proposals begin `UNUSABLE` and earn their own authority through
    later prospective exposure.
15. Relational attribution begins after individual candidates fail to account for the
    consequence or after a precommitted joint substitution exposes an interaction.
16. Recurring local boundaries can propose a higher nethra. Recurrence supplies a proposal;
    prospective evidence supplies authority.
17. A higher nethra suppresses subordinate checks after it earns coverage of the subordinate
    failure channels being suppressed.
18. A newly discovered subordinate failure channel immediately returns its check to the
    active set until higher coverage is earned for that channel.
19. Authority changes through successful prospective exposure, consequential failure, or an
    active dependency or interface event. A changed behavior registers as a new revision
    with zero grants.
20. Runtime operation stays closed and cheap while the active consequence path remains
    within its earned boundary.
21. Earned authority persists at its exact revision, operation, action, region, targets, and
    horizons until a consequential failure or active interface event creates a local
    exclusion.

## 3. Corrections to the previous pseudocode

The prior pseudocode already contained a new layer of conceptual drift. These corrections
are binding.

| Prior construction | Correct kernel construction |
|---|---|
| `Nethra.structure: ExecutableLearnedStructure` made the nethra identical to its substrate. | `Nethra` is the operative handle that references a provider and a shared structure slice. |
| Every role required `role_pair()` and a paired preserve/collapse execution. | A substitution trial is one cold-path evidence protocol. Normal operation reuses earned claims. |
| A caller chose a public `plan` and passed it into commitment issuance. | Dreth asks the operation adapter for the baseline and asks providers for candidate use offers. |
| A provider forecast outside an acceptance envelope caused immediate authority subtraction. | An authoritative forecast may choose full work or reopen consideration. Realized consequential failure controls local exclusion and repair. |
| A harmless predictive miss subtracted an authority exception. | A harmless miss earns zero extension, updates relative evidence, and leaves binary permission unchanged. |
| One endpoint success at horizon `h` authorized every horizon through `h`. | Authority covers the explicit forecast obligations that settled successfully. Prefix coverage requires obligations at every prefix. |
| Every failure became a nethra node. | A failure first creates a `FailureBoundary` evidence record. An operative learned handle becomes a nethra after prospective success. |
| Failure immediately installed provider factors. | Failure triggers factor proposals; every proposed child begins `UNUSABLE`. |
| Caller-supplied or immediately inferred member lists created composites. | Dreth retains an attribution frontier, tests individual repairs, then tests relational proposals. |
| A multiple-handle failure threatened each member grant. | Dreth blocks the exact failed authority combination and preserves member grants outside that combination. |
| `acceptable(observed)` acted as a generic outcome gate. | The registered operation contract assesses the realized consequence of the exact dispatch and target obligations. |
| `maximum_horizon` represented all temporal evidence. | `HorizonCoverage` stores the exact settled offsets or declared prefix interval. |
| `context_cell` silently mixed a witnessed point with a broad scope. | The commitment freezes a context region before exposure; failures add local exception regions. |
| Provider repair predictions selected the repair. | Dreth freezes competing predictions, chooses a separating exposure, settles them, and maintains the unresolved frontier. |
| Higher coverage was a Boolean over lower nethra IDs. | Coverage is earned per subordinate failure-channel revision, operation, region, and horizon set. |
| Passing bookkeeping tests established a successful core. | End-to-end tests must show provider invocation, precommitment, real dispatch changes, local failure, candidate promotion, and measured work. |

## 4. Ownership boundary

### 4.1 Domain adapter

The domain adapter is registered before the run. It supplies:

- visible operation context;
- target-channel identities and outcome resolution;
- operation-specific consequence semantics;
- a full baseline work plan;
- execution of a selected work plan;
- actual work receipts;
- controlled exposure or intervention mechanics for experiments;
- static dependency metadata required to map plan changes to target channels.

The domain adapter supplies semantics and observations. It does not issue authority, select a
nethra, identify a failed nethra, install a factor, or promote a relation.

### 4.2 Nethra provider

A provider supplies:

- a use offer from a registered nethra;
- a prospective forecast attached to that offer;
- candidate factor specifications when Dreth requests factorization;
- candidate relational specifications when Dreth requests relation induction;
- candidate higher-handle specifications when Dreth requests recurrence compression.

A provider returns data. Every returned nethra specification is registered with zero
operational authority.

### 4.3 Dreth kernel

Dreth owns:

- provider invocation;
- offer validation and freezing;
- immutable prediction commitment;
- role derivation from exposed substitution consequences;
- binary authority resolution;
- candidate competition;
- dispatch selection;
- actual work suppression;
- commitment settlement by exact target identity;
- local failure quarantine;
- attribution-frontier construction;
- timing of factorization and relation induction;
- registration of candidates as `UNUSABLE`;
- prospective promotion;
- failure-channel creation;
- higher coverage and subordinate suppression;
- event accounting.

### 4.4 Evaluator

The evaluator owns:

- hidden causal truth;
- oracle labels used only for scoring detection or attribution;
- same-seed ablations;
- performance summaries;
- experimental pass/fail judgments.

The `dreth/` package receives no evaluator or hidden-world object.

## 5. Repository layout

```text
dreth/
    protocols.py       # domain/provider interfaces and operation contracts
    records.py         # immutable event and state record types
    regions.py         # context-region matching and local exception overlays
    event_log.py       # append-only event storage and replay
    graph.py           # nethra handles, shared structure, boundaries, relations
    evidence.py        # bounded scheduling of unusable candidates for exposure
    commitments.py     # provider invocation, offer freezing, commitment issuance
    roles.py           # substitution trials and role derivation
    authority.py       # grants, exceptions, exact horizon coverage, binary state
    dispatch.py        # offer competition, plan compilation, real work control
    settlement.py      # exact target resolution and consequence assessment
    repair.py          # quarantine, attribution, factorization, relations, patches
    coverage.py        # failure channels and higher ownership
    recurrence.py      # local-boundary recurrence and higher candidate proposals
    accounting.py      # honest work/failure/repair measurements
    runtime.py         # fixed orchestration lifecycle
    adapters/          # concrete domain and provider adapters

tests/
    fakes.py
    test_commitments.py
    test_roles.py
    test_authority.py
    test_dispatch.py
    test_failure_repair.py
    test_coverage.py
    test_runtime.py

experiments/
    higher_sentinel_regression.py
```

The first implementation remains an in-memory kernel. Event replay supplies deterministic
state reconstruction. Storage systems, background daemons, memory pipelines, and ranking
services belong after the kernel passes its control tests.

Normal dispatch invokes reached active handles plus explicitly scheduled candidates within a
measured evidence budget. Shared-structure expansion occurs on the repair path or through a
scheduled trial. Candidate discovery therefore remains visible in Dreth's control cost.
The reference evidence queue is deterministic FIFO by scheduling event ID, with open
consequential-failure repairs ahead of general candidates. Queue order changes exposure
opportunity and creates no authority.

## 6. Core records

The pseudocode uses Python-like types. Implementations may use dataclasses, protocols, and
enums.

### 6.1 Identity and state

```python
type NethraId = str
type NethraRevision = int
type OperationId = str
type EpisodeId = str
type TargetChannelId = str
type WorkUnitId = str
type DistinctionId = str
type FailureChannelId = str
type EventId = int
type TimePoint = int
type HorizonOffset = int

enum UseState:
    USABLE
    UNUSABLE

enum Role:
    TARETH
    TRASS
    UNRESOLVED

enum ActionKind:
    PRESERVE
    COLLAPSE
    SKIP
    REUSE
    ROUTE
    COMPRESS
    AUDIT_REUSE
    REEXAMINE
    MONITOR_SUPPRESS

enum CommitmentPurpose:
    SHADOW
    ACTIVE_USE
    ROLE_TRIAL
    REPAIR_TRIAL
    RELATION_TRIAL
    COVERAGE_TRIAL
    HIGHER_TRIAL
```

### 6.2 Context regions

```python
record Context:
    operation_id: OperationId
    facts: FrozenMap[str, Hashable]

protocol ContextRegion:
    id: str

    def contains(context: Context) -> bool
    def intersects(other: ContextRegion) -> bool
    def specificity_key() -> Comparable

record PointRegion(ContextRegion):
    context: Context

record ConjunctiveRegion(ContextRegion):
    operation_id: OperationId
    required_facts: FrozenMap[str, Hashable]

record ExceptionOverlay:
    grant_id: str
    region: ContextRegion
    horizons: HorizonExclusion
    action_signature: str
    failure_event_id: EventId
```

A provider or operation adapter selects the proposed region before exposure. The commitment
freezes it. A failure can subtract a point or another observed local region. Region
generalization is itself a candidate claim and follows prospective evidence.

### 6.3 Targets and forecasts

```python
record TargetRef:
    episode_id: EpisodeId
    channel_id: TargetChannelId
    entity_key: Hashable
    due_at: TimePoint

record ForecastEnvelope:
    predicted: Any
    tolerance_or_matcher_id: str

record ForecastObligation:
    offset: HorizonOffset
    target: TargetRef
    envelope: ForecastEnvelope
    required_plan_digest: str

record ForecastTrajectory:
    obligations: tuple[ForecastObligation, ...]
    prefix_claim: bool
```

Rules:

- every offset is positive;
- every target includes episode, channel, entity, and due time;
- duplicate target identities are rejected;
- settlement requires execution of the obligation's frozen plan digest;
- `prefix_claim=True` requires one obligation at every offset from `1` through the maximum;
- an endpoint-only trajectory covers its explicit endpoint;
- settlement occurs once per obligation.

### 6.4 Plans and provider offers

```python
record WorkUnit:
    id: WorkUnitId
    operation_id: OperationId
    dependencies: frozenset[WorkUnitId]
    target_channels: frozenset[TargetChannelId]
    declared_cost: float

record WorkPlan:
    id: str
    units: tuple[WorkUnit, ...]

record OperationRequest:
    operation_id: OperationId
    episode_id: EpisodeId
    requested_offsets: frozenset[HorizonOffset]
    payload: FrozenMap[str, Any]

record ResolvedOperationRequest(OperationRequest):
    context: Context

record ExecutionReceipt:
    plan_digest: str
    executed_work_unit_ids: tuple[WorkUnitId, ...]
    actual_work_cost: float
    emitted_target_refs: tuple[TargetRef, ...]
    sentinel_signals: FrozenMap[FailureChannelId, Any]

record DistinctionEffect:
    distinction_id: DistinctionId
    action: Literal[PRESERVE, COLLAPSE]

record UseOffer:
    offer_id: str
    nethra_id: NethraId
    nethra_revision: NethraRevision
    operation_id: OperationId
    context_region: ContextRegion
    action_kind: ActionKind
    baseline_plan_digest: str
    candidate_plan: WorkPlan
    distinction_effects: tuple[DistinctionEffect, ...]
    forecast: ForecastTrajectory

record PreparedOffer:
    offer: UseOffer
    commitment_id: EventId
    action_signature: str

record PreparedDispatch:
    request: OperationRequest
    trace: DispatchTrace
    required_channels: tuple[FailureChannel, ...]
```

`UseOffer` contains no role, authority, implicated nethra, boundary, or promotion field.

```python
def action_signature(offer: UseOffer) -> str:
    return digest(
        offer.nethra_id,
        offer.nethra_revision,
        offer.operation_id,
        offer.action_kind,
        normalized_plan_difference(
            offer.baseline_plan_digest,
            offer.candidate_plan,
        ),
        offer.distinction_effects,
        sorted(
            (
                target_schema_identity(o.target),
                o.offset,
                o.required_plan_digest,
            )
            for o in offer.forecast.obligations
        ),
    )
```

Authority applies only to an identical action signature.

`target_schema_identity` contains the channel and entity schema used by an operation. Episode
IDs and due times remain commitment identities and stay outside reusable authority keys.

### 6.5 Nethra

```python
record Nethra:
    id: NethraId
    revision: NethraRevision
    name: str
    provider_id: str
    provider_handle: Hashable
    activation_region: ContextRegion
    touched_structure_ids: frozenset[str]
    child_nethra_ids: frozenset[NethraId]
    relation_nethra_ids: frozenset[NethraId]
    parent_nethra_ids: frozenset[NethraId]
    provenance_event_ids: tuple[EventId, ...]

record NethraProposal:
    name: str
    provider_id: str
    provider_handle: Hashable
    activation_region: ContextRegion
    touched_structure_ids: frozenset[str]
    parent_nethra_ids: frozenset[NethraId]
    proposed_relation_member_ids: frozenset[NethraId]
    provenance_event_ids: tuple[EventId, ...]
```

The provider may wrap a model, rule, program, planner, simulator, policy, retrieval system,
tool, or composite. Shared structure lives in the graph and can be touched by several
nethras.

An explicit behavior change creates a new nethra revision. Existing authority remains bound
to the old revision and action signature.

### 6.6 Commitments and settlements

```python
record PredictionCommitment:
    id: EventId
    offer: UseOffer
    purpose: CommitmentPurpose
    issued_at: TimePoint
    request_context: Context

record CommitmentSelection:
    commitment_id: EventId
    dispatch_trace_id: EventId
    authority_sources: tuple[str, ...]

record ControlledUseTrial:
    id: EventId
    commitment_id: EventId
    exposure_recipe_id: str
    context: Context

record ObligationSettlement:
    obligation: ForecastObligation
    observed: Any
    matched: bool
    exposure_event_id: EventId

record ConsequenceAssessment:
    operation_id: OperationId
    target_channels: frozenset[TargetChannelId]
    value: float
    boundary: float
    breached: bool
    explanation: FrozenMap[str, Any]

record CommitmentSettlement:
    commitment_id: EventId
    obligation_results: tuple[ObligationSettlement, ...]
    prediction_succeeded: bool
    consequence: ConsequenceAssessment
    fully_exposed: bool
```

Prediction success means every required exposed obligation matched. Consequence is measured
from the selected dispatch, its target observations, and the operation contract.

### 6.7 Role evidence

```python
record SubstitutionTrial:
    id: EventId
    nethra_id: NethraId
    nethra_revision: NethraRevision
    operation_id: OperationId
    context_region: ContextRegion
    distinction_ids: frozenset[DistinctionId]
    preserved_commitment_id: EventId
    substituted_commitment_id: EventId
    exposure_recipe_id: str

record RoleEvidence:
    trial_id: EventId
    role: Role
    operation_id: OperationId
    context_region: ContextRegion
    distinction_ids: frozenset[DistinctionId]
    horizons: HorizonCoverage
    prospectively_supported: bool
    preserved_consequence: ConsequenceAssessment
    substituted_consequence: ConsequenceAssessment
```

Both forecasts and the exposure recipe are frozen before either outcome becomes available.

### 6.8 Exact horizon coverage

```python
record HorizonCoverage:
    exact_offsets: frozenset[HorizonOffset]
    prefix_through: HorizonOffset | None

    def covers(requested_offsets: frozenset[HorizonOffset]) -> bool:
        covered = set(exact_offsets)
        if prefix_through is not None:
            covered.update(range(1, prefix_through + 1))
        return requested_offsets <= covered

record HorizonExclusion:
    exact_offsets: frozenset[HorizonOffset]
    prefix_from: HorizonOffset | None

    def blocks(requested_offsets: frozenset[HorizonOffset]) -> bool:
        if exact_offsets & requested_offsets:
            return True
        if prefix_from is not None:
            return any(offset >= prefix_from for offset in requested_offsets)
        return False
```

```python
def coverage_from_success(trajectory, settled) -> HorizonCoverage:
    matched_offsets = {
        result.obligation.offset
        for result in settled.obligation_results
        if result.matched
    }

    if trajectory.prefix_claim:
        maximum = max(o.offset for o in trajectory.obligations)
        if matched_offsets == set(range(1, maximum + 1)):
            return HorizonCoverage(
                exact_offsets=frozenset(),
                prefix_through=maximum,
            )

    return HorizonCoverage(
        exact_offsets=frozenset(matched_offsets),
        prefix_through=None,
    )
```

```python
def exclusion_from_failure(
    granted: HorizonCoverage,
    failed_offsets: frozenset[HorizonOffset],
) -> HorizonExclusion:
    failed = failed_offsets & (
        granted.exact_offsets
        | (
            frozenset(range(1, granted.prefix_through + 1))
            if granted.prefix_through is not None
            else frozenset()
        )
    )

    prefix_from = None
    exact = set(failed)
    if granted.prefix_through is not None and failed:
        prefix_from = min(failed)
        exact -= set(range(prefix_from, granted.prefix_through + 1))

    return HorizonExclusion(
        exact_offsets=frozenset(exact),
        prefix_from=prefix_from,
    )
```

An endpoint failure excludes the exposed endpoint. A failed prefix offset excludes that
offset and longer prefix uses covered by the same grant.

### 6.9 Authority

```python
record AuthorityKey:
    nethra_id: NethraId
    nethra_revision: NethraRevision
    operation_id: OperationId
    action_signature: str
    target_channels: frozenset[TargetChannelId]

record AuthorityGrant:
    id: str
    key: AuthorityKey
    context_region: ContextRegion
    horizons: HorizonCoverage
    role_evidence_ids: tuple[EventId, ...]
    earning_settlement_ids: tuple[EventId, ...]
    evidence_policy_id: str

record DispatchException:
    authority_grant_ids: frozenset[str]
    action_signatures: frozenset[str]
    context_region: ContextRegion
    horizons: HorizonExclusion
    target_channels: frozenset[TargetChannelId]
    failure_event_id: EventId

record ActiveInterfaceEvent:
    id: EventId
    operation_id: OperationId
    interface_id: str
    prior_interface_revision: int
    current_interface_revision: int
    context_region: ContextRegion
    affected_action_signatures: frozenset[str]
    visible_witness: Any
```

The evidence policy is registered before the run. It can require one successful exposure,
several predeclared exposures, or a declared trial structure. It receives visible
settlements and cannot inspect evaluator truth.

An active interface event is emitted only by an executed dependency or operation interface
whose registered revision changed. Its affected action signatures are declared by that
interface. Dreth adds local exceptions to those signatures and leaves every other grant
unchanged.

### 6.10 Dispatch trace

```python
record WorkSuppression:
    work_unit_id: WorkUnitId
    authority_grant_ids: frozenset[str]
    action_signatures: frozenset[str]
    nethra_ids: frozenset[NethraId]

record DispatchTrace:
    id: EventId
    operation_id: OperationId
    episode_id: EpisodeId
    context: Context
    requested_horizons: HorizonCoverage
    baseline_plan: WorkPlan
    selected_plan: WorkPlan
    selected_offer_ids: tuple[str, ...]
    active_commitment_ids: tuple[EventId, ...]
    shadow_commitment_ids: tuple[EventId, ...]
    unexposed_candidate_commitment_ids: tuple[EventId, ...]
    suppressions: tuple[WorkSuppression, ...]
    executed_sentinel_channels: frozenset[FailureChannelId]
    suppressed_sentinel_channels: frozenset[FailureChannelId]
    coverage_grant_ids: tuple[str, ...]
```

Every omitted work unit has a complete nonempty authority-source set. A combination decision
records all participating sources.

### 6.11 Failure and repair records

```python
record FailureEvent:
    id: EventId
    settlement_ids: tuple[EventId, ...]
    dispatch_trace_id: EventId
    operation_id: OperationId
    context_region: ContextRegion
    horizons: HorizonCoverage
    target_channels: frozenset[TargetChannelId]
    failed_authority_grant_ids: frozenset[str]
    failed_coverage_grant_ids: frozenset[str]
    failed_action_signatures: frozenset[str]
    consequence: ConsequenceAssessment
    topology_signature: str

record FailureBoundary:
    id: str
    parent_nethra_ids: frozenset[NethraId]
    failure_event_ids: tuple[EventId, ...]
    context_region: ContextRegion
    horizons: HorizonCoverage
    target_channels: frozenset[TargetChannelId]
    topology_signature: str
    frontier_id: str

record RepairCandidate:
    nethra_id: NethraId
    source: Literal["child", "neighbor", "dormant", "factor", "relation"]
    state: UseState
    proposal_event_id: EventId

record AttributionFrontier:
    id: str
    boundary_id: str
    candidates: tuple[RepairCandidate, ...]
    surviving_nethra_ids: frozenset[NethraId]
    trial_ids: tuple[EventId, ...]
    status: Literal["UNRESOLVED", "LOCALIZED", "RELATIONAL"]
```

The topology signature uses operation, target channels, selected action signatures,
suppressed work dependencies, and failure-channel identities. Raw observed scalar values
remain evidence fields and do not split recurrence identity by themselves.

### 6.12 Failure channels and coverage

```python
record FailureChannel:
    id: FailureChannelId
    revision: int
    owner_nethra_id: NethraId
    operation_id: OperationId
    target_channels: frozenset[TargetChannelId]
    context_region: ContextRegion
    horizons: HorizonCoverage
    detector_recipe_id: str
    source_failure_event_ids: tuple[EventId, ...]

record CoverageGrant:
    id: str
    higher_nethra_id: NethraId
    higher_revision: NethraRevision
    lower_nethra_id: NethraId
    failure_channel_id: FailureChannelId
    failure_channel_revision: int
    operation_id: OperationId
    context_region: ContextRegion
    horizons: HorizonCoverage
    earning_settlement_ids: tuple[EventId, ...]

record CoverageException:
    coverage_grant_id: str
    context_region: ContextRegion
    horizons: HorizonExclusion
    failure_event_id: EventId
```

Coverage applies to one channel revision. A new channel or revision adds an uncovered active
check.

Repeated exposure of the same detector semantics appends evidence to the existing revision.
A revision increments when the detector recipe, target set, context region, or horizon
coverage changes.

## 7. Protocols

### 7.1 Operation contract

```python
protocol OperationContract:
    id: OperationId
    evidence_policy_id: str
    root_nethra_id: NethraId

    def visible_context(request: Any) -> Context

    def baseline_plan(context: Context) -> WorkPlan

    def validate_offer(
        offer: UseOffer,
        baseline: WorkPlan,
        context: Context,
    ) -> None

    def matcher(
        matcher_id: str,
        predicted: Any,
        observed: Any,
    ) -> bool

    def assess_consequence(
        commitment: PredictionCommitment,
        obligation_results: tuple[ObligationSettlement, ...],
        dispatch: DispatchTrace,
        receipt: ExecutionReceipt,
    ) -> ConsequenceAssessment

    def assess_dispatch_consequence(
        dispatch: DispatchTrace,
        active_settlements: tuple[CommitmentSettlement, ...],
        receipt: ExecutionReceipt,
    ) -> ConsequenceAssessment

    def compare_substitution_consequences(
        preserved: ConsequenceAssessment,
        substituted: ConsequenceAssessment,
    ) -> Role

    def required_role(
        action_kind: ActionKind,
        distinction_effects: tuple[DistinctionEffect, ...],
    ) -> Role | None

    def compile_bundle(
        baseline: WorkPlan,
        selected_offers: tuple[PreparedOffer, ...],
        context: Context,
    ) -> WorkPlan

    def offers_compatible(
        left: PreparedOffer,
        right: PreparedOffer,
        context: Context,
    ) -> bool

    def forecasted_bundle_consequence(
        offers: tuple[PreparedOffer, ...],
        context: Context,
    ) -> ConsequenceAssessment

    def declared_plan_cost(plan: WorkPlan) -> float
```

The contract supplies fixed operation semantics. Dreth chooses among usable compatible
offers. `compile_bundle` verifies compatibility and produces one executable plan.

### 7.2 Domain adapter

```python
protocol DomainAdapter:
    def resolve_target(target: TargetRef) -> Any | Unavailable

    def execute(
        plan: WorkPlan,
        context: Context,
        sentinel_recipes: tuple[str, ...],
    ) -> ExecutionReceipt

    def expose_substitution_trial(
        preserved_plan: WorkPlan,
        substituted_plan: WorkPlan,
        frozen_recipe_id: str,
        context: Context,
    ) -> PairExposureReceipt

    def expose_repair_trial(
        plans: tuple[WorkPlan, ...],
        frozen_recipe_id: str,
        context: Context,
    ) -> TrialExposureReceipt

    def expose_candidate_use_trial(
        plan: WorkPlan,
        frozen_recipe_id: str,
        context: Context,
    ) -> TrialExposureReceipt
```

### 7.3 Provider

```python
protocol NethraProvider:
    def propose_use(
        nethra: Nethra,
        request: OperationRequest,
        baseline: WorkPlan,
        visible_state: FrozenMap[str, Any],
    ) -> UseOffer | None

    def propose_substitution_trial(
        nethra: Nethra,
        request: OperationRequest,
        baseline: WorkPlan,
        distinction_ids: frozenset[DistinctionId],
    ) -> SubstitutionProposal | None

    def propose_factors(
        boundary: FailureBoundary,
        evidence: FailureEvidenceView,
    ) -> tuple[NethraProposal, ...]

    def propose_relations(
        frontier: AttributionFrontier,
        evidence: FailureEvidenceView,
    ) -> tuple[NethraProposal, ...]

    def propose_higher(
        recurrence: RecurrenceEvidence,
    ) -> tuple[NethraProposal, ...]
```

The evidence views contain visible events, plan provenance, graph neighborhoods, and prior
commitments. They expose no hidden evaluator state.

### 7.4 Public runtime API

```python
class DrethRuntime:
    def register_operation(
        self,
        contract: OperationContract,
        domain: DomainAdapter,
    ) -> None

    def register_provider(
        self,
        provider_id: str,
        provider: NethraProvider,
    ) -> None

    def register_candidate(
        self,
        proposal: NethraProposal,
    ) -> NethraId

    def run_operation(
        self,
        request: OperationRequest,
    ) -> ExecutionReceipt

    def settle_due(
        self,
        now: TimePoint,
    ) -> tuple[CommitmentSettlement, ...]
```

`register_candidate` creates a handle and schedules evidence. `run_operation` controls
dispatch. `settle_due` resolves previously frozen target obligations.

## 8. Module API

| Module | Function | Exact responsibility |
|---|---|---|
| `event_log.py` | `append(event)` | Assign an ID and durably append an immutable event. |
| `event_log.py` | `replay()` | Reconstruct graph, authority, coverage, and accounting state from events. |
| `graph.py` | `register_candidate(proposal)` | Register a nethra revision with zero grants. |
| `graph.py` | `active_for_dispatch(context, operation)` | Return reached handles on the operation's current active paths. |
| `graph.py` | `related_for_repair(boundary)` | Retrieve children and shared-structure neighbors after failure opens repair. |
| `evidence.py` | `schedule_candidate(nethra_id, purpose)` | Queue a bounded prospective opportunity for an unusable candidate. |
| `evidence.py` | `due_candidates(context, operation, budget)` | Return explicitly scheduled candidates within the current evidence budget. |
| `evidence.py` | `schedule_controlled_use_trial(nethra_id, request)` | Freeze a candidate offer and exposure recipe before testing a plan that normal dispatch cannot select. |
| `evidence.py` | `execute_controlled_use_trial(trial_id)` | Expose the frozen candidate plan and settle it without changing normal dispatch. |
| `commitments.py` | `solicit_offer(nethra_id, request, baseline, purpose)` | Invoke the registered provider, validate the offer, freeze it, and append its commitment. |
| `commitments.py` | `mark_selected(commitment_id, dispatch_id, authority_sources)` | Append selection metadata while preserving the original commitment. |
| `commitments.py` | `mark_unexposed(commitment_id, dispatch_id, reason)` | Close a commitment whose required exposure plan was not executed. |
| `commitments.py` | `close_overdue_unexposed(through_time)` | Close overdue unresolved obligations with zero authority effect. |
| `roles.py` | `schedule_substitution_trial(...)` | Freeze preserved and substituted forecasts plus exposure recipe. |
| `roles.py` | `execute_substitution_trial(trial_id)` | Expose the frozen pair and settle both commitments. |
| `roles.py` | `derive_role(trial_id)` | Derive tareth/trass/unresolved from settled prospective consequences. |
| `authority.py` | `observe_success(settlement_id)` | Apply the predeclared evidence policy and create or extend an exact grant. |
| `authority.py` | `state_for_offer(offer, context, requested_offsets)` | Resolve `USABLE` or `UNUSABLE` for the exact action signature. |
| `authority.py` | `bundle_state(offers, context, requested_offsets)` | Reject an exact authority combination covered by a dispatch exception. |
| `authority.py` | `relative_bundle_key(offers, context)` | Rank usable bundles from directly comparable prospective exposure records. |
| `authority.py` | `add_local_exception(failure)` | Exclude a uniquely implicated grant in the failed region and horizons. |
| `authority.py` | `add_dispatch_exception(failure)` | Exclude the exact failed multi-grant combination. |
| `authority.py` | `apply_interface_event(event)` | Exclude the declared affected action signatures in the event's local region. |
| `authority.py` | `recover_exception(settlement_id)` | Reopen an excluded local claim after its predeclared recovery evidence succeeds prospectively. |
| `authority.py` | `recover_dispatch_exception(trial_id)` | Reopen an excluded combination after a controlled prospective bundle trial satisfies recovery policy. |
| `dispatch.py` | `choose_bundle(usable, baseline, context)` | Select a compatible usable bundle by consequence, relative evidence, and work cost. |
| `dispatch.py` | `prepare(request)` | Solicit offers, resolve authority, choose a plan, compute sentinel requirements, and append a pre-execution trace. |
| `dispatch.py` | `execute(prepared)` | Execute the selected plan and record actual work. |
| `settlement.py` | `settle_due(now)` | Resolve each exact target obligation once and assess prediction and consequence. |
| `settlement.py` | `apply_settlement(settlement)` | Extend evidence on safe success or record a harmless miss. |
| `settlement.py` | `finalize_ready_dispatches()` | Assess each selected bundle once and invoke consequential failure handling once. |
| `repair.py` | `open_boundary(failure)` | Create or update the local boundary and attribution frontier. |
| `repair.py` | `collect_candidates(boundary)` | Retrieve existing children, neighbors, dormant alternatives, and prior local repairs. |
| `repair.py` | `request_factorization(boundary)` | Ask providers for factors after an undecomposed failure and register them `UNUSABLE`. |
| `repair.py` | `schedule_separating_trial(frontier)` | Freeze competing repair predictions and select a disagreement-maximizing exposure. |
| `repair.py` | `settle_attribution_trial(trial)` | Retain supported candidates, localize a patch, or request relational proposals. |
| `repair.py` | `request_relations(frontier)` | Register relation candidates `UNUSABLE` after individual attribution evidence warrants them. |
| `coverage.py` | `register_failure_channel(failure)` | Create or revise a cheap active consequence channel. |
| `coverage.py` | `schedule_coverage_trial(higher, channel)` | Freeze a higher-handle prediction against one lower channel. |
| `coverage.py` | `add_local_exception(failure)` | Block implicated higher coverage in the failed local region and reactivate its lower channel. |
| `coverage.py` | `required_channels(trace, context, horizons)` | Return active lower channels lacking usable higher coverage. |
| `coverage.py` | `subordinate_suppression_state(...)` | Authorize lower work suppression only when use authority and channel coverage both resolve usable. |
| `recurrence.py` | `find_recurrence(boundaries)` | Group repeated local topology into proposal evidence. |
| `recurrence.py` | `request_higher_candidates(recurrence)` | Ask providers for higher handles and register each `UNUSABLE`. |
| `accounting.py` | `record_dispatch(trace, receipt)` | Record baseline, executed, suppressed, sentinel, and control work. |
| `accounting.py` | `record_failure(failure)` | Record detection, consequence, latency, and active work source. |
| `accounting.py` | `report()` | Produce raw counters and derived metrics with explicit denominators. |
| `runtime.py` | `run_operation(request)` | Orchestrate the fixed runtime lifecycle. |

## 9. Provider invocation and commitment

The public runtime API receives an operation request. It contains no expected value, role,
implicated nethra IDs, factorization result, relation members, or boundary IDs.

```python
def solicit_offer(
    nethra_id: NethraId,
    request: OperationRequest,
    baseline: WorkPlan,
    purpose: CommitmentPurpose,
) -> PreparedOffer | None:
    nethra = graph.current_nethra(nethra_id)
    provider = providers.get(nethra.provider_id)
    contract = operations.get(request.operation_id)

    raw_offer = provider.propose_use(
        nethra=nethra,
        request=request,
        baseline=baseline,
        visible_state=freeze(domain.visible_state()),
    )

    if raw_offer is None:
        return None

    assert raw_offer.nethra_id == nethra.id
    assert raw_offer.nethra_revision == nethra.revision
    assert raw_offer.operation_id == request.operation_id
    assert raw_offer.baseline_plan_digest == digest(baseline)
    assert raw_offer.context_region.contains(request.context)
    validate_trajectory(raw_offer.forecast)
    contract.validate_offer(raw_offer, baseline, request.context)

    frozen_offer = deep_freeze(raw_offer)
    commitment = event_log.append(PredictionCommitment(
        id=UNASSIGNED,
        offer=frozen_offer,
        purpose=purpose,
        issued_at=clock.now(),
        request_context=freeze(request.context),
    ))

    return PreparedOffer(
        offer=frozen_offer,
        commitment_id=commitment.id,
        action_signature=action_signature(frozen_offer),
    )
```

The domain executes after this function returns and after `DispatchPrepared` is appended.

### 9.1 Controlled evidence for an unusable plan

```python
def schedule_controlled_use_trial(
    nethra_id: NethraId,
    request: ResolvedOperationRequest,
) -> ControlledUseTrial:
    contract = operations.get(request.operation_id)
    baseline = contract.baseline_plan(request.context)
    prepared = commitments.solicit_offer(
        nethra_id=nethra_id,
        request=request,
        baseline=baseline,
        purpose=CommitmentPurpose.SHADOW,
    )
    recipe = evidence_recipes.for_candidate_offer(
        prepared.offer,
        request,
    )
    frozen_recipe = exposure_recipes.freeze(recipe)
    return event_log.append(ControlledUseTrial(
        id=UNASSIGNED,
        commitment_id=prepared.commitment_id,
        exposure_recipe_id=frozen_recipe.id,
        context=request.context,
    ))

def execute_controlled_use_trial(
    trial_id: EventId,
) -> CommitmentSettlement:
    trial = event_log.controlled_use_trial(trial_id)
    commitment = event_log.commitment(trial.commitment_id)
    assert event_log.precedes(commitment.id, trial.id)

    receipt = domain.expose_candidate_use_trial(
        plan=commitment.offer.candidate_plan,
        frozen_recipe_id=trial.exposure_recipe_id,
        context=trial.context,
    )
    event_log.append(ControlledUseExposed(
        trial_id=trial.id,
        receipt=freeze(receipt),
    ))
    settled = settlement.settle_trial_commitment(
        commitment.id,
        receipt.targets,
    )

    if settled.prediction_succeeded and not settled.consequence.breached:
        authority.observe_success(settled.id)
    return settled
```

This is the authority-earning path for a candidate whose plan differs from the current
dispatch. The evidence budget and domain exposure policy bound its cost.

## 10. Binary authority

```python
def state_for_offer(
    offer: UseOffer,
    context: Context,
    requested_offsets: frozenset[HorizonOffset],
) -> UseState:
    key = AuthorityKey(
        nethra_id=offer.nethra_id,
        nethra_revision=offer.nethra_revision,
        operation_id=offer.operation_id,
        action_signature=action_signature(offer),
        target_channels=frozenset(
            obligation.target.channel_id
            for obligation in offer.forecast.obligations
        ),
    )

    grants = authority_store.grants_for(key)

    for grant in grants:
        if not grant.context_region.contains(context):
            continue
        if not grant.horizons.covers(requested_offsets):
            continue
        if role_requirement_unsatisfied(offer, grant):
            continue
        if grant_has_matching_exception(grant, offer, context, requested_offsets):
            continue
        return UseState.USABLE

    return UseState.UNUSABLE
```

```python
def bundle_state(
    offers: tuple[PreparedOffer, ...],
    context: Context,
    requested_offsets: frozenset[HorizonOffset],
) -> UseState:
    if any(
        state_for_offer(p.offer, context, requested_offsets)
        is UseState.UNUSABLE
        for p in offers
    ):
        return UseState.UNUSABLE

    grant_ids = authority_grant_ids_for(offers, context, requested_offsets)
    signatures = frozenset(p.action_signature for p in offers)
    if dispatch_exception_store.matches(
        grant_ids=frozenset(grant_ids),
        action_signatures=signatures,
        context=context,
        requested_offsets=requested_offsets,
    ):
        return UseState.UNUSABLE

    return UseState.USABLE
```

```python
def add_local_exception(failure: FailureEvent) -> ExceptionOverlay:
    grant_id = only(failure.failed_authority_grant_ids)
    grant = authority_store.grant(grant_id)
    exception = ExceptionOverlay(
        grant_id=grant.id,
        region=failure.context_region,
        horizons=exclusion_from_failure(
            grant.horizons,
            failure.horizons.exact_offsets,
        ),
        action_signature=grant.key.action_signature,
        failure_event_id=failure.id,
    )
    event_log.append(AuthorityExceptionAdded(exception))
    return exception

def add_dispatch_exception(failure: FailureEvent) -> DispatchException:
    grants = tuple(
        authority_store.grant(grant_id)
        for grant_id in failure.failed_authority_grant_ids
    )
    exception = DispatchException(
        authority_grant_ids=failure.failed_authority_grant_ids,
        action_signatures=failure.failed_action_signatures,
        context_region=failure.context_region,
        horizons=merge_exclusions(
            exclusion_from_failure(
                grant.horizons,
                failure.horizons.exact_offsets,
            )
            for grant in grants
        ),
        target_channels=failure.target_channels,
        failure_event_id=failure.id,
    )
    event_log.append(DispatchExceptionAdded(exception))
    return exception
```

Relative evidence can rank several usable offers. Ranking cannot create a grant. Shared-node
retrieval cannot create a grant. Provider confidence cannot create a grant.

## 11. Role derivation

Role evidence is required when an operation claim depends on preserving or collapsing a
distinction.

```python
def schedule_substitution_trial(
    nethra_id: NethraId,
    request: OperationRequest,
    distinction_ids: frozenset[DistinctionId],
) -> SubstitutionTrial:
    nethra = graph.current_nethra(nethra_id)
    provider = providers.get(nethra.provider_id)
    contract = operations.get(request.operation_id)
    baseline = contract.baseline_plan(request.context)

    proposal = provider.propose_substitution_trial(
        nethra=nethra,
        request=request,
        baseline=baseline,
        distinction_ids=distinction_ids,
    )
    validate_substitution_proposal(proposal, baseline, distinction_ids)

    preserved = freeze_and_commit(
        proposal.preserved_offer,
        purpose=CommitmentPurpose.ROLE_TRIAL,
    )
    substituted = freeze_and_commit(
        proposal.substituted_offer,
        purpose=CommitmentPurpose.ROLE_TRIAL,
    )
    frozen_recipe = exposure_recipes.freeze(proposal.exposure_recipe)

    return event_log.append(SubstitutionTrial(
        id=UNASSIGNED,
        nethra_id=nethra.id,
        nethra_revision=nethra.revision,
        operation_id=request.operation_id,
        context_region=proposal.context_region,
        distinction_ids=distinction_ids,
        preserved_commitment_id=preserved.id,
        substituted_commitment_id=substituted.id,
        exposure_recipe_id=frozen_recipe.id,
    ))
```

```python
def execute_substitution_trial(
    trial_id: EventId,
) -> PairExposureReceipt:
    trial = event_log.substitution_trial(trial_id)
    preserved = event_log.commitment(trial.preserved_commitment_id)
    substituted = event_log.commitment(trial.substituted_commitment_id)
    assert event_log.precedes(preserved.id, trial.id)
    assert event_log.precedes(substituted.id, trial.id)

    receipt = domain.expose_substitution_trial(
        preserved_plan=preserved.offer.candidate_plan,
        substituted_plan=substituted.offer.candidate_plan,
        frozen_recipe_id=trial.exposure_recipe_id,
        context=preserved.request_context,
    )
    event_log.append(SubstitutionExposed(
        trial_id=trial.id,
        receipt=freeze(receipt),
    ))
    settlement.settle_trial_commitment(
        preserved.id,
        receipt.preserved_targets,
    )
    settlement.settle_trial_commitment(
        substituted.id,
        receipt.substituted_targets,
    )
    return receipt
```

```python
def derive_role(trial_id: EventId) -> RoleEvidence:
    trial = event_log.substitution_trial(trial_id)
    preserved = settlements.require_complete(trial.preserved_commitment_id)
    substituted = settlements.require_complete(trial.substituted_commitment_id)
    contract = operations.get(trial.operation_id)

    role = contract.compare_substitution_consequences(
        preserved.consequence,
        substituted.consequence,
    )
    prospectively_supported = (
        preserved.prediction_succeeded
        and substituted.prediction_succeeded
    )

    evidence = event_log.append(RoleEvidence(
        trial_id=trial.id,
        role=role,
        operation_id=trial.operation_id,
        context_region=trial.context_region,
        distinction_ids=trial.distinction_ids,
        horizons=intersection(
            exposed_horizons(preserved),
            exposed_horizons(substituted),
        ),
        prospectively_supported=prospectively_supported,
        preserved_consequence=preserved.consequence,
        substituted_consequence=substituted.consequence,
    ))

    if (
        evidence.role is not Role.UNRESOLVED
        and evidence.prospectively_supported
    ):
        authority.reconsider_trial_successes(
            settlement_ids=(preserved.id, substituted.id),
            role_evidence_id=evidence.id,
        )

    return evidence
```

An operation can use natural matched exposures, replayable simulations, sandboxed trials, or
interventions. The exposure mechanism is declared and frozen before results arrive.
Role records describe the exposed substitution relation. Authority queries consume only role
records whose `prospectively_supported` field is true.

## 12. Dispatch and real work suppression

```python
def choose_bundle(
    usable: tuple[PreparedOffer, ...],
    baseline: WorkPlan,
    context: Context,
    requested_offsets: frozenset[HorizonOffset],
) -> tuple[PreparedOffer, ...]:
    contract = operations.get(context.operation_id)
    candidates = all_pairwise_compatible_subsets(
        usable,
        compatible=lambda left, right: contract.offers_compatible(
            left,
            right,
            context,
        ),
    )

    admissible = []
    for bundle in candidates:
        if authority.bundle_state(
            bundle,
            context,
            requested_offsets,
        ) is UseState.UNUSABLE:
            continue
        predicted = contract.forecasted_bundle_consequence(
            bundle,
            context,
        )
        if predicted.breached:
            continue
        plan = contract.compile_bundle(baseline, bundle, context)
        saved = (
            contract.declared_plan_cost(baseline)
            - contract.declared_plan_cost(plan)
        )
        admissible.append((bundle, saved))

    if not admissible:
        return ()

    return max(
        admissible,
        key=lambda row: (
            row[1],
            authority.relative_bundle_key(row[0], context),
            bundle_region_specificity(row[0]),
            stable_bundle_id(row[0]),
        ),
    )[0]
```

Binary authority supplies admissibility. The reference selector maximizes declared work
removed among forecast-safe bundles, then uses directly exposed relative evidence, local
scope specificity, and a stable ID tie-break. Selector variants are explicit experiments and
leave authority unchanged.

```python
def prepare(request: OperationRequest) -> PreparedDispatch:
    contract = operations.get(request.operation_id)
    context = contract.visible_context(request)
    baseline = contract.baseline_plan(context)
    requested_offsets = request.requested_offsets

    active_ids = graph.active_for_dispatch(
        context,
        request.operation_id,
    )
    trial_ids = evidence.due_candidates(
        context=context,
        operation=request.operation_id,
        budget=runtime_config.evidence_budget,
    )
    candidate_ids = ordered_union(active_ids, trial_ids)
    prepared_offers = []

    for nethra_id in candidate_ids:
        prepared = commitments.solicit_offer(
            nethra_id=nethra_id,
            request=request.with_context(context),
            baseline=baseline,
            purpose=(
                CommitmentPurpose.SHADOW
                if nethra_id in trial_ids
                else CommitmentPurpose.ACTIVE_USE
            ),
        )
        if prepared is not None:
            prepared_offers.append(prepared)

    usable = []
    for prepared in prepared_offers:
        state = authority.state_for_offer(
            prepared.offer,
            context,
            requested_offsets,
        )
        purpose = event_log.commitment(prepared.commitment_id).purpose
        if (
            state is UseState.USABLE
            and purpose is CommitmentPurpose.ACTIVE_USE
        ):
            usable.append(prepared)

    selected = choose_bundle(
        usable=tuple(usable),
        baseline=baseline,
        context=context,
        requested_offsets=requested_offsets,
    )

    selected_plan = contract.compile_bundle(
        baseline=baseline,
        selected_offers=selected,
        context=context,
    )
    nonselected = tuple(
        prepared
        for prepared in prepared_offers
        if prepared not in selected
    )
    shadow = tuple(
        prepared
        for prepared in nonselected
        if all(
            obligation.required_plan_digest == digest(selected_plan)
            for obligation in prepared.offer.forecast.obligations
        )
    )
    unexposed = tuple(
        prepared
        for prepared in nonselected
        if prepared not in shadow
    )

    suppressions = map_removed_units_to_authority(
        baseline=baseline,
        selected_offers=selected,
    )
    active_path = graph.execution_path(
        selected_nethra_ids=tuple(p.offer.nethra_id for p in selected),
        suppressions=suppressions,
    )
    required_channels = coverage.required_channels(
        active_path=active_path,
        context=context,
        horizons=HorizonCoverage(
            exact_offsets=requested_offsets,
            prefix_through=None,
        ),
    )

    trace = event_log.append(DispatchTrace(
        id=UNASSIGNED,
        operation_id=request.operation_id,
        episode_id=request.episode_id,
        context=context,
        requested_horizons=HorizonCoverage(
            exact_offsets=requested_offsets,
            prefix_through=None,
        ),
        baseline_plan=baseline,
        selected_plan=selected_plan,
        selected_offer_ids=tuple(p.offer.offer_id for p in selected),
        active_commitment_ids=tuple(p.commitment_id for p in selected),
        shadow_commitment_ids=tuple(p.commitment_id for p in shadow),
        unexposed_candidate_commitment_ids=tuple(
            p.commitment_id for p in unexposed
        ),
        suppressions=suppressions,
        executed_sentinel_channels=frozenset(c.id for c in required_channels),
        suppressed_sentinel_channels=coverage.suppressed_channel_ids(
            active_path,
            context,
            requested_offsets,
        ),
        coverage_grant_ids=coverage.grant_ids_for_suppression(
            active_path,
            context,
            requested_offsets,
        ),
    ))

    for prepared in selected:
        commitments.mark_selected(
            prepared.commitment_id,
            dispatch_id=trace.id,
            authority_sources=tuple(
                unique(
                    grant_id
                    for s in suppressions
                    if prepared.offer.nethra_id in s.nethra_ids
                    for grant_id in s.authority_grant_ids
                )
            ),
        )
    for prepared in unexposed:
        commitments.mark_unexposed(
            prepared.commitment_id,
            dispatch_id=trace.id,
            reason="required_plan_not_executed",
        )

    return PreparedDispatch(
        request=request,
        trace=trace,
        required_channels=required_channels,
    )
```

```python
def execute(prepared: PreparedDispatch) -> ExecutionReceipt:
    assert event_log.contains(prepared.trace.id)
    assert all(
        event_log.contains(commitment_id)
        for commitment_id in prepared.trace.active_commitment_ids
    )

    recipes = tuple(
        channel.detector_recipe_id
        for channel in prepared.required_channels
    )
    receipt = domain.execute(
        plan=prepared.trace.selected_plan,
        context=prepared.trace.context,
        sentinel_recipes=recipes,
    )

    event_log.append(DispatchCompleted(
        dispatch_trace_id=prepared.trace.id,
        receipt=freeze(receipt),
    ))
    accounting.record_dispatch(prepared.trace, receipt)
    return receipt
```

An unusable offer can earn evidence through a compatible visible shadow target or through a
scheduled controlled trial. It cannot alter normal dispatch.

## 13. Settlement

```python
def settle_due(now: TimePoint) -> tuple[CommitmentSettlement, ...]:
    completed = []

    for commitment in event_log.open_commitments_with_due_obligations(now):
        if event_log.commitment_marked_unexposed(commitment.id):
            continue
        if commitment.purpose not in {
            CommitmentPurpose.ACTIVE_USE,
            CommitmentPurpose.SHADOW,
        }:
            continue

        trace = event_log.dispatch_for_commitment(commitment.id)
        executed_plan_digest = digest(trace.selected_plan)
        results = list(
            event_log.obligation_settlements(commitment.id)
        )

        for obligation in commitment.offer.forecast.obligations:
            if obligation_already_settled(obligation, results):
                continue
            if obligation.target.due_at > now:
                continue
            if obligation.required_plan_digest != executed_plan_digest:
                continue

            observed = domain.resolve_target(obligation.target)
            if observed is Unavailable:
                continue

            matched = operations.get(
                commitment.offer.operation_id
            ).matcher(
                obligation.envelope.tolerance_or_matcher_id,
                obligation.envelope.predicted,
                observed,
            )

            exposure = event_log.append(Exposure(
                commitment_id=commitment.id,
                target=obligation.target,
                observed=freeze(observed),
            ))
            result = ObligationSettlement(
                obligation=obligation,
                observed=freeze(observed),
                matched=matched,
                exposure_event_id=exposure.id,
            )
            event_log.append(ObligationSettled(
                commitment_id=commitment.id,
                result=result,
            ))
            results.append(result)

        fully_exposed = all_obligations_resolved(commitment, results)
        if not fully_exposed:
            continue

        receipt = event_log.receipt_for_dispatch(trace.id)
        contract = operations.get(commitment.offer.operation_id)
        consequence = contract.assess_consequence(
            commitment=commitment,
            obligation_results=tuple(results),
            dispatch=trace,
            receipt=receipt,
        )

        settlement = event_log.append(CommitmentSettlement(
            commitment_id=commitment.id,
            obligation_results=tuple(results),
            prediction_succeeded=all(r.matched for r in results),
            consequence=consequence,
            fully_exposed=True,
        ))
        apply_settlement(settlement)
        completed.append(settlement)

    commitments.close_overdue_unexposed(through_time=now)
    finalize_ready_dispatches()
    return tuple(completed)
```

```python
def apply_settlement(settlement: CommitmentSettlement) -> None:
    commitment = event_log.commitment(settlement.commitment_id)

    if settlement.prediction_succeeded and not settlement.consequence.breached:
        authority.observe_success(settlement.id)
        relative_evidence.record_success(settlement.id)
        return

    if settlement.consequence.breached:
        relative_evidence.record(
            settlement.id,
            prediction_succeeded=settlement.prediction_succeeded,
            consequence_breached=True,
        )
        return

    relative_evidence.record_miss(settlement.id)
    event_log.append(HarmlessPredictionMiss(
        settlement_id=settlement.id,
        existing_permission_changed=False,
        repair_opened=False,
    ))
```

A harmless miss creates no authority extension. Existing binary authority remains in place.
The miss can change relative ranking among usable alternatives.

`close_overdue_unexposed` closes a commitment only after the latest frozen obligation due
time has passed and at least one required target or plan exposure remains unavailable. It
creates zero success, failure, role, authority, or repair evidence.

```python
def finalize_ready_dispatches() -> None:
    for trace in event_log.unfinalized_completed_dispatches():
        active_or_none = tuple(
            settlements.complete_or_none(commitment_id)
            for commitment_id in trace.active_commitment_ids
        )
        if any(item is None for item in active_or_none):
            if all(
                event_log.commitment_is_closed(commitment_id)
                for commitment_id in trace.active_commitment_ids
            ):
                event_log.append(DispatchOutcomeUnresolved(
                    dispatch_trace_id=trace.id,
                ))
                event_log.append(DispatchFinalized(
                    dispatch_trace_id=trace.id,
                ))
            continue
        active = tuple(item for item in active_or_none if item is not None)

        receipt = event_log.receipt_for_dispatch(trace.id)
        contract = operations.get(trace.operation_id)
        consequence = contract.assess_dispatch_consequence(
            dispatch=trace,
            active_settlements=active,
            receipt=receipt,
        )
        event_log.append(DispatchConsequenceAssessed(
            dispatch_trace_id=trace.id,
            consequence=consequence,
        ))

        if consequence.breached:
            failure = repair.quarantine_consequential_failure(
                dispatch_trace_id=trace.id,
                settlement_ids=tuple(s.id for s in active),
                consequence=consequence,
            )
            coverage.register_failure_channel(failure)
            repair.open_boundary(failure)

        event_log.append(DispatchFinalized(dispatch_trace_id=trace.id))
```

A selected bundle receives one consequence assessment and at most one failure event.
Shadow commitments update predictive evidence and cannot quarantine the selected bundle.

## 14. Earning authority

```python
def observe_success(settlement_id: EventId) -> AuthorityGrant | None:
    settlement = event_log.settlement(settlement_id)
    assert settlement.prediction_succeeded
    assert not settlement.consequence.breached
    commitment = event_log.commitment(settlement.commitment_id)
    offer = commitment.offer
    contract = operations.get(offer.operation_id)
    policy = evidence_policies.get(contract.evidence_policy_id)

    evidence = evidence_store.comparable_successes(
        nethra_id=offer.nethra_id,
        nethra_revision=offer.nethra_revision,
        operation_id=offer.operation_id,
        action_signature=action_signature(offer),
        context_region=offer.context_region,
    )
    evidence = evidence + (settlement,)

    if not policy.qualifies(evidence):
        return None

    required_role = contract.required_role(
        offer.action_kind,
        offer.distinction_effects,
    )
    role_ids = roles.supporting_evidence_ids(
        nethra_id=offer.nethra_id,
        operation_id=offer.operation_id,
        action_signature=action_signature(offer),
        context_region=offer.context_region,
        required_role=required_role,
    )
    if required_role is not None and not role_ids:
        return None

    key = AuthorityKey(
        nethra_id=offer.nethra_id,
        nethra_revision=offer.nethra_revision,
        operation_id=offer.operation_id,
        action_signature=action_signature(offer),
        target_channels=frozenset(
            o.target.channel_id
            for o in offer.forecast.obligations
        ),
    )
    earned_horizons = coverage_from_success(offer.forecast, settlement)
    existing = authority_store.find_same_grant(
        key=key,
        context_region=offer.context_region,
        evidence_policy_id=policy.id,
    )

    if existing is not None:
        blocking = authority_store.matching_exceptions(
            grant_id=existing.id,
            context=settlement_context(settlement),
            horizons=earned_horizons,
        )
        if blocking:
            recovery_evidence = evidence_store.successes_after(
                event_id=latest_event_id(
                    e.failure_event_id for e in blocking
                ),
                key=key,
                context_region=offer.context_region,
            )
            if not policy.qualifies_recovery(recovery_evidence):
                return None
            event_log.append(AuthorityExceptionsRecovered(
                grant_id=existing.id,
                exception_event_ids=tuple(e.failure_event_id for e in blocking),
                earning_settlement_id=settlement.id,
            ))

        extended = replace(
            existing,
            horizons=union(existing.horizons, earned_horizons),
            role_evidence_ids=ordered_union(
                existing.role_evidence_ids,
                role_ids,
            ),
            earning_settlement_ids=ordered_union(
                existing.earning_settlement_ids,
                tuple(s.id for s in evidence),
            ),
        )
        event_log.append(AuthorityExtended(grant=extended))
        return extended

    grant = AuthorityGrant(
        id=ids.next_grant(),
        key=key,
        context_region=offer.context_region,
        horizons=earned_horizons,
        role_evidence_ids=role_ids,
        earning_settlement_ids=tuple(s.id for s in evidence),
        evidence_policy_id=policy.id,
    )
    event_log.append(AuthorityGranted(grant=grant))
    return grant
```

The reference kernel can use a one-success evidence policy for a point region and exact
action signature. Experiments may register stricter predeclared policies. Policy comparisons
are separate experiments.

## 15. Consequential failure and local quarantine

```python
def quarantine_consequential_failure(
    dispatch_trace_id: EventId,
    settlement_ids: tuple[EventId, ...],
    consequence: ConsequenceAssessment,
) -> FailureEvent:
    trace = event_log.dispatch(dispatch_trace_id)

    sources = authority_sources_relevant_to_failed_targets(
        trace=trace,
        failed_targets=consequence.target_channels,
    )
    coverage_sources = coverage_sources_relevant_to_failed_targets(
        trace=trace,
        failed_targets=consequence.target_channels,
    )
    region = PointRegion(trace.context)
    horizons = trace.requested_horizons

    failure = event_log.append(FailureEvent(
        id=UNASSIGNED,
        settlement_ids=settlement_ids,
        dispatch_trace_id=trace.id,
        operation_id=trace.operation_id,
        context_region=region,
        horizons=horizons,
        target_channels=consequence.target_channels,
        failed_authority_grant_ids=frozenset(s.grant_id for s in sources),
        failed_coverage_grant_ids=frozenset(
            s.coverage_grant_id for s in coverage_sources
        ),
        failed_action_signatures=frozenset(s.action_signature for s in sources),
        consequence=consequence,
        topology_signature=failure_topology_signature(
            trace,
            sources,
            coverage_sources,
        ),
    ))

    if len(sources) == 1:
        authority.add_local_exception(failure)
    elif len(sources) > 1:
        authority.add_dispatch_exception(failure)
    if coverage_sources:
        coverage.add_local_exception(failure)

    accounting.record_failure(failure)
    return failure
```

```python
def open_boundary(failure: FailureEvent) -> FailureBoundary:
    existing = graph.boundary_by_topology(
        failure.topology_signature,
        failure.context_region,
        failure.horizons,
    )
    if existing is not None:
        return graph.append_failure_to_boundary(existing.id, failure.id)

    parent_ids = graph.nethras_on_dispatch(
        event_log.dispatch(failure.dispatch_trace_id)
    )
    if not parent_ids:
        parent_ids = frozenset({
            operations.get(failure.operation_id).root_nethra_id
        })
    frontier = AttributionFrontier(
        id=ids.next_frontier(),
        boundary_id=PENDING,
        candidates=(),
        surviving_nethra_ids=frozenset(),
        trial_ids=(),
        status="UNRESOLVED",
    )
    boundary = FailureBoundary(
        id=ids.next_boundary(),
        parent_nethra_ids=parent_ids,
        failure_event_ids=(failure.id,),
        context_region=failure.context_region,
        horizons=failure.horizons,
        target_channels=failure.target_channels,
        topology_signature=failure.topology_signature,
        frontier_id=frontier.id,
    )
    frontier = replace(frontier, boundary_id=boundary.id)
    event_log.append(BoundaryOpened(boundary, frontier))

    candidates = collect_candidates(boundary)
    if candidates:
        graph.set_frontier_candidates(frontier.id, candidates)
        schedule_separating_trial(frontier.id)
    else:
        request_factorization(boundary.id)

    return boundary
```

A boundary records the failed local authority path. It becomes a recurring structure source
only through later evidence.

## 16. Candidate collection and lazy factorization

```python
def collect_candidates(
    boundary: FailureBoundary,
) -> tuple[RepairCandidate, ...]:
    related_ids = ordered_union(
        graph.children_of(boundary.parent_nethra_ids),
        graph.shared_structure_neighbors(boundary.parent_nethra_ids),
        graph.dormant_alternatives(boundary.parent_nethra_ids),
        graph.prior_repairs(boundary.topology_signature),
    )

    return tuple(
        RepairCandidate(
            nethra_id=nethra_id,
            source=graph.candidate_source(nethra_id, boundary),
            state=authority.any_state_for_nethra(
                nethra_id,
                boundary.context_region,
                boundary.horizons,
            ),
            proposal_event_id=graph.proposal_event_id(nethra_id),
        )
        for nethra_id in related_ids
    )
```

```python
def request_factorization(
    boundary_id: str,
) -> tuple[Nethra, ...]:
    boundary = graph.boundary(boundary_id)
    if graph.factorization_already_requested(boundary_id):
        return ()

    evidence = repair_views.for_boundary(boundary_id)
    proposals = []
    for parent_id in boundary.parent_nethra_ids:
        parent = graph.current_nethra(parent_id)
        provider = providers.get(parent.provider_id)
        proposals.extend(provider.propose_factors(boundary, evidence))

    registered = []
    for proposal in deduplicate_proposals(proposals):
        nethra = graph.register_candidate(proposal)
        assert authority.grants_for_nethra(nethra.id) == ()
        event_log.append(FactorCandidateRegistered(
            boundary_id=boundary_id,
            nethra_id=nethra.id,
            initial_state=UseState.UNUSABLE,
        ))
        registered.append(nethra)

    graph.set_frontier_candidates(
        boundary.frontier_id,
        tuple(
            RepairCandidate(
                nethra_id=n.id,
                source="factor",
                state=UseState.UNUSABLE,
                proposal_event_id=graph.proposal_event_id(n.id),
            )
            for n in registered
        ),
    )
    schedule_separating_trial(boundary.frontier_id)
    return tuple(registered)
```

## 17. Attribution and relation induction

```python
def schedule_separating_trial(frontier_id: str) -> AttributionTrial:
    frontier = graph.frontier(frontier_id)
    boundary = graph.boundary(frontier.boundary_id)
    evidence = repair_views.for_boundary(boundary.id)
    offers = []

    for candidate in frontier.candidates:
        prepared = commitments.solicit_repair_offer(
            candidate_id=candidate.nethra_id,
            boundary=boundary,
            evidence=evidence,
        )
        if prepared is not None:
            offers.append(prepared)

    recipe = choose_maximum_disagreement_exposure(
        forecasts=tuple(p.offer.forecast for p in offers),
        available_recipes=evidence.available_exposure_recipes,
    )
    frozen_recipe = exposure_recipes.freeze(recipe)

    return event_log.append(AttributionTrial(
        id=UNASSIGNED,
        frontier_id=frontier.id,
        commitment_ids=tuple(p.commitment_id for p in offers),
        exposure_recipe_id=frozen_recipe.id,
    ))
```

```python
def settle_attribution_trial(trial_id: EventId) -> None:
    trial = event_log.attribution_trial(trial_id)
    settlements = settlement.settle_trial(trial)
    supported = {
        event_log.commitment(s.commitment_id).offer.nethra_id
        for s in settlements
        if s.prediction_succeeded and not s.consequence.breached
    }
    frontier = graph.frontier(trial.frontier_id)

    if len(supported) == 1:
        winner_id = only(supported)
        graph.localize_frontier(frontier.id, winner_id, trial.id)
        schedule_candidate_authority_trials(
            winner_id,
            graph.boundary(frontier.boundary_id),
        )
        return

    if len(supported) > 1:
        graph.retain_frontier(frontier.id, supported, trial.id)
        schedule_separating_trial(frontier.id)
        return

    request_relations(frontier.id)
```

```python
def request_relations(
    frontier_id: str,
) -> tuple[Nethra, ...]:
    frontier = graph.frontier(frontier_id)
    assert individual_trial_evidence_exhausted(frontier)
    evidence = repair_views.for_frontier(frontier.id)
    proposals = []

    for provider in providers.relevant_to(frontier):
        proposals.extend(provider.propose_relations(frontier, evidence))

    relations = []
    for proposal in deduplicate_proposals(proposals):
        relation = graph.register_candidate(proposal)
        assert authority.grants_for_nethra(relation.id) == ()
        graph.link_relation_candidate(
            relation_id=relation.id,
            member_ids=proposal.member_nethra_ids,
            frontier_id=frontier.id,
        )
        event_log.append(RelationCandidateRegistered(
            frontier_id=frontier.id,
            nethra_id=relation.id,
            initial_state=UseState.UNUSABLE,
        ))
        schedule_relation_trial(relation.id, frontier.id)
        relations.append(relation)

    return tuple(relations)
```

A relation becomes operational through the same `observe_success` authority path used by
every nethra.

`individual_trial_evidence_exhausted` requires one completed boundary-matched prospective
repair trial for every individual candidate retained in the frontier, with zero candidate
both predicting and restoring the protected consequence. An explicitly exposed joint
substitution can also establish relational evidence directly.

## 18. Failure-channel coverage and higher ownership

```python
def register_failure_channel(
    failure: FailureEvent,
) -> FailureChannel:
    owner = smallest_current_owner(failure)
    existing = graph.matching_failure_channel(
        owner_nethra_id=owner.id,
        topology_signature=failure.topology_signature,
    )

    if existing is None:
        channel = FailureChannel(
            id=ids.next_failure_channel(),
            revision=1,
            owner_nethra_id=owner.id,
            operation_id=failure.operation_id,
            target_channels=failure.target_channels,
            context_region=failure.context_region,
            horizons=failure.horizons,
            detector_recipe_id=derive_detector_recipe(failure),
            source_failure_event_ids=(failure.id,),
        )
    else:
        channel = revise_channel_with_failure(existing, failure)

    event_log.append(FailureChannelRegistered(channel))
    return channel
```

```python
def schedule_coverage_trial(
    higher_id: NethraId,
    channel_id: FailureChannelId,
) -> PredictionCommitment:
    higher = graph.current_nethra(higher_id)
    channel = graph.failure_channel(channel_id)
    request = coverage_request_for(channel)

    prepared = commitments.solicit_offer(
        nethra_id=higher.id,
        request=request,
        baseline=operations.get(channel.operation_id).baseline_plan(
            request.context
        ),
        purpose=CommitmentPurpose.COVERAGE_TRIAL,
    )
    assert offer_predicts_channel(prepared.offer, channel)
    return event_log.commitment(prepared.commitment_id)
```

```python
def grant_channel_coverage(
    settlement_id: EventId,
    lower_id: NethraId,
    channel_id: FailureChannelId,
) -> CoverageGrant | None:
    settlement = event_log.settlement(settlement_id)
    commitment = event_log.commitment(settlement.commitment_id)
    channel = graph.failure_channel(channel_id)

    if not settlement.prediction_succeeded:
        return None
    if not detected_or_correctly_predicted_channel(settlement, channel):
        return None

    grant = CoverageGrant(
        id=ids.next_coverage_grant(),
        higher_nethra_id=commitment.offer.nethra_id,
        higher_revision=commitment.offer.nethra_revision,
        lower_nethra_id=lower_id,
        failure_channel_id=channel.id,
        failure_channel_revision=channel.revision,
        operation_id=channel.operation_id,
        context_region=commitment.offer.context_region,
        horizons=coverage_from_success(
            commitment.offer.forecast,
            settlement,
        ),
        earning_settlement_ids=(settlement.id,),
    )
    event_log.append(CoverageGranted(grant))
    return grant
```

```python
def add_local_exception(failure: FailureEvent) -> None:
    for grant_id in failure.failed_coverage_grant_ids:
        event_log.append(CoverageExceptionAdded(
            exception=CoverageException(
                coverage_grant_id=grant_id,
                context_region=failure.context_region,
                horizons=exclusion_from_failure(
                    coverage_store.grant(grant_id).horizons,
                    failure.horizons.exact_offsets,
                ),
                failure_event_id=failure.id,
            )
        ))
        channel = coverage_store.grant(grant_id).failure_channel_id
        event_log.append(LowerChannelReactivated(
            failure_channel_id=channel,
            context_region=failure.context_region,
            horizons=failure.horizons,
        ))
```

```python
def required_channels(
    active_path: tuple[NethraId, ...],
    context: Context,
    horizons: HorizonCoverage,
) -> tuple[FailureChannel, ...]:
    required = []

    for channel in graph.active_failure_channels(active_path, context):
        covering_grant = coverage_store.find_usable(
            failure_channel_id=channel.id,
            failure_channel_revision=channel.revision,
            context=context,
            horizons=horizons,
            higher_nethra_ids=frozenset(active_path),
            require_active_use_authority=True,
        )
        if covering_grant is None:
            required.append(channel)

    return tuple(required)
```

```python
def subordinate_suppression_state(
    higher_offer: UseOffer,
    lower_id: NethraId,
    context: Context,
    requested_offsets: frozenset[HorizonOffset],
) -> UseState:
    if authority.state_for_offer(
        higher_offer,
        context,
        requested_offsets,
    ) is UseState.UNUSABLE:
        return UseState.UNUSABLE

    channels = graph.active_failure_channels_for_lower(lower_id, context)
    if not channels:
        return UseState.UNUSABLE
    for channel in channels:
        if coverage_store.find_usable(
            higher_nethra_id=higher_offer.nethra_id,
            lower_nethra_id=lower_id,
            failure_channel_id=channel.id,
            failure_channel_revision=channel.revision,
            context=context,
            horizons=HorizonCoverage(
                exact_offsets=requested_offsets,
                prefix_through=None,
            ),
        ) is None:
            return UseState.UNUSABLE

    return UseState.USABLE
```

This is the recursive ownership rule: higher authority plus complete current channel coverage
permits subordinate suppression.

## 19. Recurrence and higher candidates

```python
def find_recurrence(
    boundaries: tuple[FailureBoundary, ...],
) -> tuple[RecurrenceEvidence, ...]:
    groups = group_by(
        boundaries,
        key=lambda b: (
            b.topology_signature,
            b.target_channels,
            normalized_region_shape(b.context_region),
            b.horizons,
        ),
    )

    return tuple(
        RecurrenceEvidence.from_group(group)
        for group in groups
        if contains_repeated_local_boundaries(group)
    )
```

```python
def request_higher_candidates(
    recurrence: RecurrenceEvidence,
) -> tuple[Nethra, ...]:
    proposals = []
    for provider in providers.relevant_to(recurrence):
        proposals.extend(provider.propose_higher(recurrence))

    higher = []
    for proposal in deduplicate_proposals(proposals):
        nethra = graph.register_candidate(proposal)
        assert authority.grants_for_nethra(nethra.id) == ()
        graph.link_higher_candidate(
            higher_id=nethra.id,
            boundary_ids=recurrence.boundary_ids,
        )
        event_log.append(HigherCandidateRegistered(
            nethra_id=nethra.id,
            recurrence_id=recurrence.id,
            initial_state=UseState.UNUSABLE,
        ))
        schedule_standard_use_trials(nethra.id)
        higher.append(nethra)

    return tuple(higher)
```

After standard authority is earned, channel coverage trials determine which lower checks the
higher nethra can own.

## 20. Fixed runtime lifecycle

```python
def run_operation(request: OperationRequest) -> ExecutionReceipt:
    prepared = dispatch.prepare(request)
    receipt = dispatch.execute(prepared)
    settlement.settle_due(clock.now())
    repair.advance_ready_frontiers()
    recurrence.process_new_boundaries()
    return receipt
```

Expanded event order:

```text
1. domain supplies visible context and full baseline plan
2. graph retrieves locally applicable and shared-structure candidate nethras
3. Dreth invokes providers
4. Dreth freezes offers and prospective target obligations
5. authority resolves each exact offer to usable or unusable
6. operation contract selects among usable offers or baseline
7. coverage resolves required lower failure channels
8. Dreth appends the dispatch trace
9. domain executes the selected plan and required sentinels
10. exact target identities settle once
11. successful prospective evidence may earn or extend authority
12. harmless misses update evidence and keep the graph closed
13. consequential failure quarantines the exact local dispatch path
14. failure opens or updates a boundary and active failure channel
15. existing candidates receive separating trials
16. an undecomposed boundary requests factor proposals
17. individual attribution failure can request relational proposals
18. repeated boundaries can request higher proposals
19. every candidate returns through prospective authority earning
20. higher candidates suppress lower checks only through usable channel coverage
```

## 21. Accounting

`DrethAccounting` records raw values first.

```python
record DrethAccounting:
    commitments_issued
    obligations_exposed
    obligations_unexposed
    active_uses
    shadow_trials

    baseline_work_units
    executed_work_units
    baseline_work_cost
    executed_work_cost
    dreth_control_cost
    repair_cost
    sentinel_cost

    suppressed_work_units
    suppressed_work_by_grant
    suppressed_lower_sentinels
    higher_coverage_uses

    observed_consequence_breaches
    failure_events_opened
    harmless_prediction_misses

    local_exceptions
    dispatch_combination_exceptions
    boundaries_opened
    factor_candidates_registered
    relation_candidates_registered
    higher_candidates_registered

    attribution_trials
    attribution_resolved
    attribution_unresolved
    attribution_latency
```

The evaluator maintains a separate record:

```python
record EvaluationAccounting:
    oracle_consequential_failures
    detected_oracle_failures
    missed_oracle_failures
    correctly_localized_failures
    incorrectly_localized_failures
    hidden_truth_queries_by_runtime
```

Derived metrics:

```python
gross_saved_work = baseline_work_cost - executed_work_cost
net_saved_work = gross_saved_work - dreth_control_cost - repair_cost

detection_rate = (
    detected_oracle_failures / oracle_consequential_failures
)

higher_owned_work_reduction = (
    cost_of_suppressed_subordinate_work / baseline_work_cost
)

historical_handle_amortization = (
    composite_skip_count
    + regime_skip_count
    + parked_skip_count
) / total_skip_count

repair_cost_per_observed_breach = (
    repair_cost / observed_consequence_breaches
)
```

Every denominator is reported. `historical_handle_amortization` preserves the earlier batch
runner formula. `higher_owned_work_reduction` measures actual cost. Detection and attribution
correctness come from evaluator accounting after the run. An unavailable counterfactual
baseline is reported as `unknown`. Same-seed ablations provide the primary experimental
baseline.

## 22. Historical higher-sentinel regression

Conversation history records an accepted `regime_switch` experiment with:

```text
12 runs
n = 8, 12
handle amortization = 13.8%
regime sentinel pass = 9,186
regime sentinel fail = 11,214
no_sentinel = 0
runs ok = 12/12
```

The accepted mechanism was:

```text
lower nethras fail together
→ recurrence proposes a higher handle
→ a higher sentinel is commissioned and exposed
→ higher coverage becomes usable
→ lower sentinel work is physically skipped
→ a higher failure reopens lower checks
```

The originating report is absent from the current branch. The regression harness must
recreate the same world family and same-object ablation:

```text
Dreth full
versus
the same world, seed, providers, lower certs, audits, and sentinels
with higher-owned subordinate suppression disabled
```

The harness reports whatever result occurs. The 13.8% figure is historical provenance and
an experimental reproduction target. The pass condition is faithful execution, complete
accounting, and the exact one-switch ablation.

## 23. Acceptance tests

### 23.1 Causal ownership

1. `test_public_api_accepts_no_expected_value`
2. `test_public_api_accepts_no_role`
3. `test_public_api_accepts_no_implicated_ids`
4. `test_public_api_accepts_no_factorization_result`
5. `test_public_api_accepts_no_relation_members`
6. `test_dreth_invokes_provider_before_domain_execution`
7. `test_commitment_is_immutable_before_exposure`

### 23.2 Target and temporal identity

8. `test_episode_and_channel_prevent_cross_settlement`
9. `test_obligation_settles_once`
10. `test_unexposed_obligation_earns_zero_authority`
11. `test_endpoint_h3_success_covers_h3_only`
12. `test_prefix_h3_success_covers_h1_h2_h3`
13. `test_failed_h2_obligation_excludes_exact_h2_region`

### 23.3 Roles and authority

14. `test_role_is_derived_from_substitution_consequences`
15. `test_equivalent_substitution_yields_trass`
16. `test_consequential_substitution_difference_yields_tareth`
17. `test_unresolved_forecast_yields_unresolved_role`
18. `test_role_evidence_is_operation_and_context_scoped`
19. `test_provider_confidence_creates_zero_authority`
20. `test_shared_structure_retrieval_creates_zero_authority`
21. `test_runtime_permission_has_two_states`
22. `test_action_signature_change_requires_new_authority`

### 23.4 Real dispatch

23. `test_usable_offer_physically_removes_work_unit`
24. `test_every_removed_unit_has_authority_source`
25. `test_unusable_offer_runs_shadow_only`
26. `test_baseline_runs_when_every_offer_is_unusable`
27. `test_authoritative_forecast_can_select_full_work`
28. `test_dispatch_trace_precedes_execution`

### 23.5 Settlement and local failure

29. `test_success_can_earn_exact_offer_authority`
30. `test_harmless_miss_earns_no_extension`
31. `test_harmless_miss_opens_no_boundary`
32. `test_harmless_miss_preserves_existing_binary_permission`
33. `test_consequential_single_source_failure_adds_local_exception`
34. `test_consequential_combination_failure_blocks_exact_combination`
35. `test_combination_failure_preserves_member_grants_elsewhere`
36. `test_failure_signature_ignores_raw_scalar_fragmentation`

### 23.6 Repair and relations

37. `test_boundary_record_is_separate_from_nethra`
38. `test_factorization_waits_for_consequential_undecomposed_failure`
39. `test_factor_candidates_begin_unusable`
40. `test_separating_trial_commitments_precede_exposure`
41. `test_tied_frontier_remains_unresolved`
42. `test_unique_prospective_repair_localizes_boundary`
43. `test_individual_trials_precede_relation_request`
44. `test_relation_candidate_begins_unusable`
45. `test_relation_earns_authority_through_standard_path`

### 23.7 Higher ownership

46. `test_recurrence_registers_unusable_higher_candidate`
47. `test_recurrence_alone_grants_zero_authority`
48. `test_higher_use_authority_alone_suppresses_zero_lower_channels`
49. `test_channel_coverage_is_earned_prospectively`
50. `test_complete_current_coverage_suppresses_lower_sentinel`
51. `test_new_lower_failure_channel_reactivates_lower_sentinel`
52. `test_higher_failure_reopens_local_lower_checks`

### 23.8 Experimental validity

53. `test_runtime_package_has_no_hidden_truth_dependency`
54. `test_raw_work_counters_reconcile`
55. `test_failed_mechanism_run_produces_complete_report`
56. `test_higher_sentinel_ablation_changes_one_switch`
57. `test_same_seed_ablation_uses_identical_domain_events`
58. `test_controlled_unusable_plan_can_earn_without_normal_dispatch_effect`

## 24. Implementation order

### Pass 1: prospective dispatch vertical slice

Implement:

- `protocols.py`
- `records.py`
- `regions.py`
- `event_log.py`
- `graph.py`
- `evidence.py`
- `commitments.py`
- `authority.py`
- `dispatch.py`
- `settlement.py`
- `accounting.py`
- `runtime.py`

Use this reference policy:

```text
context authority region: precommitted point region
initial authority: one fully exposed, prediction-successful, consequence-safe commitment
recovery: one fully exposed post-failure success for the same exact claim
horizon: explicit offsets; prefix closure only for complete prefix trajectories
expiry: event-driven local exclusions; zero clock-driven expiration
bundle selection: exhaustive compatible subsets in the bounded fake domain
candidate evidence: one controlled trial slot per fake-domain cycle
```

Deliver one fake domain where:

- the provider produces its own forecast;
- Dreth commits it before execution;
- one exact offer earns authority;
- later dispatch physically removes named work units;
- a harmless miss leaves the graph closed;
- a consequential miss blocks the exact local use;
- work counters reconcile.

Acceptance tests: 1-13, 19-36, 53-55, and 58.

### Pass 2: contextual roles and substitution

Implement:

- `roles.py`
- role requirements in `OperationContract`
- point and conjunctive regions
- exact and prefix horizon coverage

Acceptance tests: 14-18 and 11-13.

### Pass 3: failure-shaped repair

Implement:

- `repair.py`
- failure boundaries
- attribution frontiers
- lazy factorization
- separating trials
- relational candidates

Acceptance tests: 37-45.

### Pass 4: recursive ownership

Implement:

- `coverage.py`
- `recurrence.py`
- higher candidate trials
- channel-specific subordinate suppression
- `experiments/higher_sentinel_regression.py`

Acceptance tests: 46-52 and 56-57.

### Pass 5: offline proposal generation

Add proposal-only functions:

```python
def snapshot_complete_scaffold() -> ScaffoldSnapshot
def mine_recurrent_intersections(snapshot) -> tuple[NethraProposal, ...]
def compile_background_trass(snapshot) -> tuple[NethraProposal, ...]
def propose_topology_pruning(snapshot) -> tuple[PruningProposal, ...]
def apply_safe_pruning(proposal) -> PruningReceipt
```

The snapshot includes stable nethras, contextual role histories, dormant alternatives,
authority regions, exception boundaries, failure channels, unresolved frontiers, local
repairs, composites, and higher coverage.

Offline outputs enter through `graph.register_candidate()` with zero grants. Pruning
preserves active authority paths, failure boundaries, failure channels, and repair
provenance.

## 25. Current branch replacement map

| Current file or API | Next implementation |
|---|---|
| `dreth/model.py` caller-authored `PredictionCommitment` | `records.py` provider-authored frozen `UseOffer` inside Dreth-issued commitment |
| `dreth/model.py` `Authority.successes/failures` | `authority.py` exact grants, horizon coverage, local and combination exceptions |
| `dreth/graph.py::consider` | candidate retrieval only; authority resolution remains separate |
| `dreth/engine.py::commit` | remove; use `commitments.solicit_offer` |
| `dreth/engine.py::observe` | replace with exact target `settlement.settle_due` |
| `dreth/engine.py::can_reuse` | replace with `dispatch.prepare` and actual plan execution |
| `dreth/engine.py::_open_failure` | replace with quarantine, boundary, frontier, and lazy repair |
| `dreth/engine.py::consolidate` | replace with recurrence evidence and unusable higher proposals |
| `dreth/demo.py` Boolean ledger demonstration | replace with work-plan execution and reconciled accounting demonstration |
| `tests/test_dreth.py` caller-driven ledger tests | replace with the acceptance tests in section 23 |

Neutral identifiers and basic graph storage can be reused after their semantics match these
records.

## 26. Definition of done for the next implementation pass

The next pass is complete when:

- the public API contains operation requests and candidate registration;
- Dreth invokes providers and freezes forecasts;
- commitments precede every exposure;
- exact targets settle once;
- horizon coverage matches explicit obligations;
- dispatch changes the executed work plan;
- every removed work unit has an authority source;
- harmless misses leave repair closed;
- consequential failures block the exact local path;
- accounting reports raw work and failure counters;
- the selected acceptance tests pass;
- README describes the implemented subset exactly;
- the demo prints causal receipts and measured work;
- every unimplemented phase remains explicitly listed.
