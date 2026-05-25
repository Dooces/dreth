# Dreth: Comprehensive System Analysis

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Core Problem & Design Philosophy](#2-core-problem--design-philosophy)
- [3. Architecture Overview](#3-architecture-overview)
- [4. The Hidden Causal World (world.py)](#4-the-hidden-causal-world-worldpy)
- [5. The Operator Vocabulary (functions.py)](#5-the-operator-vocabulary-functionspy)
- [6. Hypothesis Search — The Full Audit (fit.py)](#6-hypothesis-search--the-full-audit-fitpy)
- [7. The Certification Ledger (ledger.py)](#7-the-certification-ledger-ledgerpy)
- [8. Sentinel Validation — The Cheap Path (sentinels.py)](#8-sentinel-validation--the-cheap-path-sentinelspy)
- [9. Regime Detection (regime.py)](#9-regime-detection-regimepy)
- [10. The Certification Control Loop (agent.py)](#10-the-certification-control-loop-agentpy)
- [11. Batch Runner — The Test Harness (batch_run.py)](#11-batch-runner--the-test-harness-batch_runpy)
- [12. Supporting Modules](#12-supporting-modules)
- [13. Key Concepts & Terminology](#13-key-concepts--terminology)
- [14. Data Flow & Lifecycle](#14-data-flow--lifecycle)
- [15. The Multi-Level Certification Hierarchy](#15-the-multi-level-certification-hierarchy)
- [16. Invariants & Correctness Properties](#16-invariants--correctness-properties)
- [17. Performance Optimization Layers](#17-performance-optimization-layers)
- [18. Philosophical Foundations](#18-philosophical-foundations)
- [19. Current Limitations & Remaining Burden](#19-current-limitations--remaining-burden)

---

## 1. Executive Summary

**Dreth** is a streaming causal structure learning framework that builds a recursive hierarchy of **certified beliefs** ("nethras") about a hidden dynamical system. The agent cannot observe the system's structure directly. It can observe current values passively and can issue **interventions** (set a variable to a value and observe what happens). Passive residuals are used as cheap stress signals; interventions remain the authority-producing proof channel. Each intervention has a cost, so the central challenge is: *how do you build up certified knowledge that lets you pay less on future cycles without accumulating hidden errors?*

The framework's answer is a **certification machinery** where each belief earns its authority through intervention tests, survives through sentinel monitoring, and is revoked when contradicting evidence appears. Certified beliefs are **operative** — they actively gate what the agent considers in future reasoning, not merely describe what it knows.

**Key innovation**: The distinction between *variation* (different states exist) and *earned contrast* (differences that have been shaped by consequence). Only failure-tested, correction-shaped beliefs earn the right to reduce future work; passive observation can defer probes, but authority still comes from witnessed consequence.

---

## 2. Core Problem & Design Philosophy

### The Problem

A hidden causal world consists of N variables connected in a DAG (Directed Acyclic Graph). Each variable has:
- A set of parent variables (0–2 parents)
- A function that computes its value from parent values (MEAN, MAX, PROD, etc.)
- Gaussian noise added each time step

The world **drifts**: edges can change, functions can swap, values can shift. The agent must continuously produce accurate predictions for every variable while minimizing the number of costly interventions.

### The Design Philosophy

1. **Certification as currency**: Work is only saved when something *earned* the right to skip it. No shortcuts without proof.
2. **Operative, not descriptive**: A certified belief changes what the agent considers — it's an active filter on hypothesis spaces, not a passive label.
3. **Provisional authority**: Every verdict (tareth, trass, certified) is scoped to specific conditions and can be revoked when those conditions change.
4. **Ambiguity is first-class**: When evidence is insufficient, the system records ambiguity (TiedFrontier) rather than making a premature choice.
5. **Morphology ≠ Cause**: Observing that two hypotheses score similarly (morphology) is different from knowing *why* they score similarly (cause). The system tracks both separately.
6. **Failure-shaped topology**: The ledger preserves failure history so that future traversal is constrained by what has previously mattered. This does not mathematically prevent hallucination in all domains; it blocks **authority-bearing** traversal where missing coverage, residual stress, tests, contradiction, or other feedback channels expose that a handle has not earned scope.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        batch_run.py                                 │
│  (Test harness: parameter grid, invariant checks, baseline compare) │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ creates & runs
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ChainedAgent (agent.py)                         │
│  The certification control loop. Per cycle, per variable:           │
│    → trass-skip | compression | sentinel cheap-path | full audit    │
│                                                                     │
│  Owns: ChainedLedger, RegimeRegister, counters, diagnostics        │
└───┬──────────┬───────────┬──────────┬──────────┬────────────────────┘
    │          │           │          │          │
    ▼          ▼           ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐
│world.py│ │fit.py  │ │sentinel │ │ledger  │ │regime.py │
│        │ │        │ │.py      │ │.py     │ │          │
│Hidden  │ │Full    │ │Cheap    │ │All     │ │Recurring │
│causal  │ │audit:  │ │path:    │ │cert    │ │pattern   │
│world:  │ │enum &  │ │sentinel │ │state:  │ │detection │
│IV+obs  │ │score   │ │checks   │ │VarNet- │ │& cluster │
│        │ │hyps    │ │envelope │ │hra,    │ │sentinels │
│        │ │        │ │cost-    │ │compos- │ │          │
│        │ │        │ │dispatch │ │ites,   │ │          │
│        │ │        │ │         │ │events  │ │          │
└────────┘ └────────┘ └─────────┘ └────────┘ └──────────┘
    ▲
    │ uses
┌────────────┐
│functions.py│
│Operator    │
│vocabulary  │
│FUNC_LIBRARY│
│+ HIDDEN    │
└────────────┘
```

### Module Dependency Chain

```
functions.py  ← world.py ← ledger.py ← fit.py ← sentinels.py ← agent.py ← batch_run.py
                                    ↑                              ↑
                              regime.py ──────────────────────────┘
                              records.py ─────────────────────────┘
                              summary.py ─────────────────────────┘
```

---

## 4. The Hidden Causal World (`world.py`)

### Purpose
The **oracle** the agent tests against. It holds the ground-truth causal structure that the agent is trying to learn, but the agent can never see it directly.

### Key Data Structures

| Field | Description |
|-------|-------------|
| `parents: List[List[int]]` | Random DAG — each variable has 0–2 parents from lower indices |
| `funcs: List[str]` | One function per variable from `HIDDEN_FUNC_LIBRARY` |
| `state: Tuple[float, ...]` | Current continuous state (one float ∈ [0,1] per variable) |
| `visible_count: int` | How many variables the agent can see (supports incremental reveal) |
| `hidden_log: List[HiddenMutation]` | Complete mutation history (diagnostic only — agent never reads) |
| `harm: List[float]` | Per-variable harm attribution for cost-weighted dispatch |
| `noise_sigma: float` | Gaussian noise added to each variable each step (default 0.02) |

### Agent's Access Interface

The agent has two different evidence channels:

1. **Passive observation / residual monitoring**: The agent can read current visible values and compare them against certified predictions. This is a cheap stress signal. It can say "this handle appears unstressed this cycle" or "this handle needs active checking," but it does **not** by itself issue tareth/trass authority.
2. **Interventional proof**: The agent can issue counterfactual probes into the world. This is the authority-producing channel used for substitution tests, route certificates, composite witnesses, regime sentinel commissioning, and repair attribution.

```python
predict_under_intervention(var, val)           # Set var=val, run one step, return full state
predict_var_under_intervention(target, iv, val) # Same but returns only target var (cheaper)
predict_under_joint_intervention({v1: val1, v2: val2, ...})  # Multi-var intervention
```

Each intervention call has explicit cost. The current architecture should be read as **passive-first for monitoring, intervention-backed for certification**: ordinary observation can reduce unnecessary probes, but a shortcut only earns authority through witnessed consequence.

### Mutation Schedules

The world supports multiple test schedules that control when and how the causal structure changes:

| Schedule | Description |
|----------|-------------|
| `incremental` | Reveal variables one at a time, value drift only |
| `shaped` | Fixed structural changes at specific early cycles, then value drift |
| `periodic_shifts` | Shaped + structural changes every 1000 cycles |
| `novelty` | SIN injection (out-of-library function) at cycle 10 |
| `rare_catastrophe` | Occasional function swaps on a specific variable |
| `regime_switch` | Cluster of vars alternates between two causal carriers |
| `false_trass` | Wires a specific joint-false-trass subgraph for testing |

### HiddenMutation Record

```python
@dataclass(frozen=True)
class HiddenMutation:
    cycle: int          # When
    kind: str           # VALUE, EDGE, FUNC, NOVELTY, REVEAL, REGIME_SWITCH
    description: str    # Human-readable summary
    rule_changed: bool  # True = structural change; False = value drift only
    affected_var: int   # Which variable's structure changed (-1 for value-only)
```

---

## 5. The Operator Vocabulary (`functions.py`)

Two parallel function libraries define what the world can use vs. what the agent can hypothesize:

### Agent's Library (`FUNC_LIBRARY`)

| Function | Formula | Domain |
|----------|---------|--------|
| `LOW` | Constant 0.2 | No-parent variables |
| `HIGH` | Constant 0.8 | No-parent variables |
| `TINY` | Constant 0.1 | Joint false-trass baseline |
| `FIRST` | `parents[0]` | Single-parent identity |
| `MEAN` | `sum(parents)/len(parents)` | Multi-parent average |
| `MAX` | `max(parents)` | Multi-parent maximum |
| `MIN` | `min(parents)` | Multi-parent minimum |
| `PROD` | `∏ parents` | Multi-parent product |
| `DIFF` | `|parents[0] - parents[1]|` | Two-parent difference |

### Hidden Library (`HIDDEN_FUNC_LIBRARY`)

Everything in `FUNC_LIBRARY` plus:

| Function | Formula | Purpose |
|----------|---------|---------|
| `SIN` | `0.5 + 0.5·sin(2π·mean(parents))` | Out-of-library; tests vocabulary novelty detection |

When the world assigns SIN to a variable, the agent **cannot** fit it (SIN is not in `FUNC_LIBRARY`). The agent's fit will oscillate unstably, triggering a **NoveltyNethra** — a record that the hypothesis library appears insufficient.

---

## 6. Hypothesis Search — The Full Audit (`fit.py`)

### Purpose
The most expensive operation: enumerate all possible `(parents, func)` hypotheses for one variable, score each against interventional probes, and return the best.

### `fit_var()` — The Core Algorithm

```
Input:  target variable, world reference, RNG, probe budget, tolerance
Output: (best_parents, best_func, best_score, second_best_score)
```

**Step 1 — Hypothesis Enumeration**:
- **Restricted** (when `available_parents` provided): Only hypotheses using certified parent candidates
  - Constants: `((), LOW)`, `((), HIGH)`, `((), TINY)`
  - Single-parent: `((p,), FIRST)` for each available parent p
  - Two-parent: `((p1, p2), func)` for each pair × 5 functions
- **Full** (fallback): All variables as potential parents — O(n²) hypotheses

**Step 2 — Intervention Pool Construction** (targeted mode):
- Generate 4× budget random `(var, val)` candidate probes
- Score each by **discrimination**: how many hypotheses produce distinct predictions
- Select top-budget probes that maximize hypothesis separation
- Forced probes from TiedFrontier are guaranteed inclusion (P1-B)

**Step 3 — Vectorized Scoring** (`score_hypotheses_batched`):
- For each probe, query the world for the actual outcome
- For each hypothesis, predict the outcome and compare within tolerance
- Score = count of matching probes (integer out of budget)
- Uses numpy batched operations for performance

**Step 4 — Result Extraction**:
- Best hypothesis by score
- Tie set: all hypotheses with score == best_score
- Near-tie candidates: all hypotheses within `near_tie_margin` of best
- Context key: hash of available_parents for staleness detection

### Adaptive Probe Budget

The agent scales probe budget UP with hypothesis space size:
- Small spaces (≤225 hypotheses): base budget (30 probes)
- Large unrestricted spaces: up to 2× base, scaling as `√(n_hypotheses) × 2`
- Never scales DOWN — ensures small restricted spaces always get adequate probing

---

## 7. The Certification Ledger (`ledger.py`)

### Purpose
All certified state lives here — the data structures, not the logic. The ledger is the system's **causal memory**: it preserves failure history so that certified beliefs actively constrain future inference.

### Core Data Structures

#### `VarNethra` — Per-Variable Certification Handle

The central object. One per variable. **Operative**, not descriptive — its current state modifies every downstream inference.

| Field | Description |
|-------|-------------|
| `parents, func` | Current best hypothesis for this variable |
| `status` | Lifecycle state: `proposed` → `certified` → `uncertain` (on failure) |
| `certificates: Dict[str, NethraCertificate]` | Per-operation certified claims |
| `route_certs: Dict[int, NethraCertificate]` | Per-candidate-parent route certs |
| `sentinels` | List of `(iv_var, iv_val)` probes for cheap-path validation |
| `envelope: NoiseEnvelope` | Empirical noise model — certifies tolerance ε |
| `compressions: List[Compression]` | Cached predictions under specific gate conditions |
| `tied_frontier: Optional[TiedFrontier]` | Near-tied hypotheses constellation |
| `dormant_alternatives: List[DormantAlternative]` | Archived hypotheses from collapsed frontiers |
| `cost_weight: float` | Attention weight for cost-dispatched sentinel checking |
| `strong_observations: int` | Consecutive matching audits (drives promotion to certified) |
| `parked: bool` | Whether leaf sentinel is parked due to redundancy with higher handles |

**Status lifecycle**:
```
proposed ──(promote_after matching audits)──→ certified
    ↑                                            │
    │                                    (sentinel failure)
    │                                            ↓
    └───────────(re-audit succeeds)────── uncertain
```

#### `NethraCertificate` — Scoped Certified Claim

Every certification claim carries full provenance:

```python
@dataclass
class NethraCertificate:
    operation: Operation    # skip | route | compress | audit
    role: Role              # tareth | trass | untested | false_trass | noise_floor
    authority: Authority    # none | prefer | guarded_reuse | skip | propagate
    context_parents: Tuple  # Parent set at cert time
    context_visible: int    # Visible count at cert time
    context_cycle: int      # When certified
    targets: Tuple[int]     # Vars actually tested for downstream change
    earned_by: str          # Provenance (substitution_test, joint_interaction, etc.)
    revoked_by: Optional[str]  # Set on demotion (sentinel_failure, parent_change, etc.)
    witnesses: Tuple        # (state_snapshot, iv_val) pairs that produced propagation
    sentinel_passes: int    # Stable cycles accumulated after issuance
```

#### `NoiseEnvelope` — Empirical Tolerance Certification

Tracks observed |predicted - actual| deltas and certifies an ε such that ~95% of recent observations fall within it. This ε becomes the variable's tolerance for all comparison operations.

- Rolling window of 200 deltas
- Certifies when ≥20 samples accumulated and new ε differs by >20% from current
- Tracks out-of-band events for `envelope_failing` detection
- Resets OOB tracking on re-certification (old events measured against old ε)

#### `TiedFrontier` — Ambiguity as First-Class Object

Records near-tied hypotheses that are currently indistinguishable:

```python
@dataclass
class TiedFrontier:
    candidates: FrozenSet    # Set of (parents, func) in the near-tie constellation
    scores: Dict             # Last-seen score per candidate
    context_key: int         # Hash of available_parents at creation
    stable_count: int        # Consecutive audits returning same candidate set
    distinct_contexts_seen: int  # Distinct context_keys survived (for collapse gating)
    separating_probes: Tuple # Probes that discriminate members (placeholder)
```

**Collapse requires**: `stable_count ≥ 3 AND distinct_contexts_seen ≥ 2` — the frontier must prove itself across distinct evidence contexts, not just a single regime.

#### `CompositeNethra` — Joint False-Trass Certificate

Certifies that two individually-trass variables are *jointly* tareth:

```python
@dataclass
class CompositeNethra:
    members: Tuple[int, int]     # The two jointly-tareth vars
    sentinel_var: int            # Downstream witness
    probe_val_a, probe_val_b: float  # Joint intervention values
    tol: float                   # Comparison tolerance
    pass_count: int              # Cycles where joint probe passed
```

Per cycle, the composite sentinel replays the joint intervention. If the interaction persists, both members can be covered by the composite path. If it disappears, both are reset to untested.

Important metric distinction: a composite **pass** is pairwise evidence. A composite **skip** is a unique variable-cycle skip. If one hub variable appears in dozens of passing pairwise composites, the hub is still skipped once that cycle. Therefore raw sums such as `sum(cn.pass_count * len(cn.members))` are overlap evidence, not independent saved work.

#### `HyperCompositeNethra` — Component-Level Handle

When pairwise composites form dense connected components, they are compressed into a single component handle:
- One component-level probe can cover many overlapping pairwise composites
- Cost: 2 world calls per component vs. 2 per pairwise edge
- Falls back to pairwise composites on failure
- Exists to prevent O(k²) pairwise composite explosion around hub variables

This is a nethra-of-nethra move: many lower composite certs become evidence for a higher component boundary. The higher handle owns the shared boundary until it fails; failure descends back into the pairwise frontier.

#### `ChainedLedger` — The Central Registry

```python
class ChainedLedger:
    vars: Dict[int, VarNethra]              # One per variable
    history: Dict[int, List[VarNethra]]     # Archived past states per variable
    novelty: List[NoveltyNethra]            # Open/resolved vocabulary novelty records
    composites: List[CompositeNethra]       # Active joint-interaction certs
    hyper_composites: List[HyperCompositeNethra]  # Component-level handles
    events: List[LedgerEvent]              # Structured event log
    event_log: List[str]                   # Human-readable event log
```

**Key methods**:
- `update_var()` — Apply audit result; archives old state on signature change
- `invalidate()` — Cascade invalidation through descendant closure
- `closure_descendants()` — Transitive closure of believed dependency (with route-trass pruning)
- `issue_cert()` / `issue_route_cert()` — Authority transaction methods
- `install_composite()` / `install_hyper_composite()` — Composite cert installation

---

## 8. Sentinel Validation — The Cheap Path (`sentinels.py`)

### Purpose
The cheap-path validation layer. Two functions, both active and load-bearing.

### `select_var_sentinels()` — Sentinel Selection

Called once after a fit stabilizes. Picks the most discriminating intervention probes:

1. Generate `pool` random `(var, val)` candidate probes
2. For each probe, compute discrimination = number of alternative hypotheses that would predict differently from the chosen fit
3. Sort by discrimination, take top `count` (default 5)

**Goal**: Pick probes that are hard for a wrong hypothesis to pass. If the world structure changes but the sentinel probes still pass, the fit is likely still correct.

### `check_var_sentinels_with_envelope()` — Per-Cycle Validation

Called every cycle instead of a full audit. Cost: `sentinel_count` (5) world calls vs. `intervention_budget` (30+) for a full audit.

For each sentinel probe:
1. Compute expected value from current fit
2. Issue intervention, get actual value from world
3. Compare within the variable's noise envelope ε

**Cost-dispatched behavior** (asymmetric attention):

| Cost Weight | Behavior | Rationale |
|-------------|----------|-----------|
| ≥ `cost_high_threshold` | Strict: any miss = failure | High-stakes variables |
| mid-range | Standard: escalate only if `envelope_failing` (clustering of OOB events) | Default behavior |
| < `cost_low_threshold` | Permissive: failures dismissed as `TEMPORAL_TRASS` | Low-cost variables can absorb inaccuracy |

All deviations are added to the noise envelope regardless of dispatch decision. Dismissed deviations are logged as `TemporalTrassEntry` for later credit assignment.

---

## 9. Regime Detection (`regime.py`)

### Purpose
Identifies recurring patterns of cert behavior — co-failure patterns that repeat across different world states. A "regime" is not a world-state cluster; it's a pattern of *certified authorities failing, repairing, or surviving together*.

### Core Objects

#### `CertEvent` — One cert's behavior at a specific cycle

```python
@dataclass
class CertEvent:
    var: int              # Which variable
    cert_key: str         # Which certificate
    event_type: str       # "failed" | "repaired" | "stressed"
    repair_shape: str     # "stable" | "parent_change" | "func_change" | "full_change"
    cert_age: int         # Maturity proxy (n.full_audits at event time)
```

#### `RegimeSignature` — Confirmed recurring pattern

```python
@dataclass
class RegimeSignature:
    regime_id: int
    authority: int              # How many distinct windows matched this pattern
    events: List[CertEvent]     # Merged representative event set
    active_sentinel: Optional[Tuple]  # Commissioned cluster-level probe
```

### `RegimeRegister` — Pattern Matching Engine

**Matching**: Weighted Jaccard over `(var, cert_key, event_type)` triples, weighted by cert maturity (`min(cert_age/10, 1.0)`). Established certs failing count more than freshly-issued ones.

**Lifecycle**:
```
First occurrence → candidate (stored, waiting for match)
Second matching occurrence → confirmed regime (authority=2)
Subsequent matches → increment authority
Stale candidates (>1000 cycles) → pruned as noise
```

**Sentinel Commissioning**: Once a regime is confirmed, the agent searches for a cluster-level witness probe — a single intervention that elicits a response from ≥2 regime members. If found and active, this probe can replace N individual leaf checks for the covered members. A confirmed regime without an active/checkable sentinel is historical structure only; it should not authorize skip or parking by quiescence alone.

**Sentinel Mode**:
- `strict=True` (confirmed pool): Only "failed" events match
- `strict=False` (candidate pool): "failed" + "stressed" events contribute — passive stress can seed candidates but not confirm regimes

---

## 10. The Certification Control Loop (`agent.py`)

### Purpose
The heart of the system. `ChainedAgent` owns the full certification lifecycle and makes all dispatch decisions.

### Per-Cycle Decision Flow

For each visible variable, the agent dispatches to one of four paths:

```
┌─────────────────────────────────────────────────────────┐
│                    Per Variable                          │
│                                                          │
│  1. Trass skip?  ─── role=="trass" ──→ SKIP (no work)   │
│         │ no                                             │
│         ▼                                                │
│  2. Compression? ─── gate matches ──→ SKIP (cached)     │
│         │ no                                             │
│         ▼                                                │
│  3. Passive residual OK? ───────────→ SKIP active probe │
│         │ stressed                                       │
│         ▼                                                │
│  4. Sentinel?    ─── probes pass  ──→ SKIP (validated)  │
│         │ fail                                           │
│         ▼                                                │
│  5. Full audit   ─── re-enumerate & score ──→ UPDATE    │
│         │                                                │
│         ▼                                                │
│  6. Install result → update certs, sentinels, frontier  │
└─────────────────────────────────────────────────────────┘
```

### Key Agent Methods

#### `_certify_operation_role(var, cycle)` — The Substitution Test

Determines whether a variable is **tareth** (load-bearing) or **trass** (collapsible):

1. For each of 5 spread perturbation values (0.05, 0.25, 0.5, 0.75, 0.95):
   - Compute 5 baseline samples and 5 perturbed samples
   - Average cancels per-sample noise
   - For each other visible var j: if |Δavg| > j's tolerance → change detected
2. Verdict: "tareth" if any perturbation produced downstream changes; "trass" otherwise

**Scoping**: The verdict is scoped to:
- Current visible variables
- Current noise tolerances
- Current intervention targets (if `role_salience` set)
- Route-trass filtering (trass vars excluded from target list in `live-frontier` mode)

#### `_test_joint_false_trass(var_a, var_b, cycle)` — Composition Test

Two individually-trass variables may be *jointly* tareth:
```
R0:  baseline (no intervention)
RA:  perturb var_a only
RB:  perturb var_b only
RAB: perturb both simultaneously

If |RAB[j] - R0[j]| > tol AND |RA[j]| ≤ tol AND |RB[j]| ≤ tol → interaction!
```

Jointly tareth if interaction in ≥ half of trials. Installs a pairwise `CompositeNethra` on the ledger. Dense overlap among many such pairs is promoted into `HyperCompositeNethra` component handles rather than treated as independent saved work.

#### `_install_var(var, parents, func, ...)` — Post-Audit Installation

After a full audit returns a result:
1. Check for **signature change** (parents or func differ from current)
2. If changed: archive old state, invalidate dependent vars, cascade
3. Test operation role via `_certify_operation_role`
4. Assign new sentinels via `select_var_sentinels`
5. Update TiedFrontier from near-tie candidates
6. If consecutive matching audits ≥ `promote_after` → promote to "certified"
7. Discover compressions for stable fits
8. Record FitDiagnostic

#### Graded Cascade on Sentinel Failure

When a sentinel fails:
1. **Local re-audit**: Re-run full audit on the failed variable
2. If same fit returned (noisy miss): increment `consecutive_sentinel_failures`, no cascade
3. If different fit (genuine change):
   - Record drift event
   - `ledger.invalidate()` cascades to all descendants
   - Track `descendant_cascade_count`
   - Check for false-trass via `_find_joint_trass_candidates`

#### Consequence-Weighted Tiers

Variables are bucketed by their consequence tier (number of dependents):

| Tier | Dependents | Sentinel Count | Trass Threshold |
|------|-----------|----------------|-----------------|
| T0 (leaf) | 0 | base | 1 stable cycle |
| T1 | 1–2 | base + tier×scaling | 4 stable cycles |
| T2 | 3+ | base + tier×scaling | 7 stable cycles |

Higher-tier variables get more sentinels and require more evidence for trass authority.

#### Sentinel Parking

When a leaf sentinel is provably redundant:
- Covered by a confirmed regime or component handle with an active/checkable sentinel
- No unique failures in the last 200 cycles
- Higher sentinel passed ≥4 times

The leaf probe is **parked** (skipped each cycle). Wake conditions: regime sentinel fails, parent change, sparse revalidation every 500 cycles.

#### Passive Residual Monitoring

Before running active sentinels, compute expected-next-state from the certified fit:
- If residual is within envelope → skip the active sentinel for this cycle (saves intervention cost, but does not create new authority)
- If stressed → run active sentinel / repair logic
- Stressed co-occurrences can seed regime candidates, but passive stress alone should not confirm a regime

### Topological Scheduling

Full audits are scheduled by **topological order** (parents before children) with budget priority:
- `priority_audit_budget` caps max full audits per cycle
- Deferred variables tracked for defer-streak reporting
- Watch-state variables from stability horizon get priority

### Dormancy Management

Certified+stable variables are moved to a **dormant partition** (removed from hot pass):
- Re-entry only via sentinel failure (cascade invalidation wakes a dormant var)
- Minimum envelope age before eligibility: 100 cycles
- Composite/component handles: if all covered members are dormant and no active consequence path is stressed, the handle can be treated as passing for that cycle without polling

---

## 11. Batch Runner — The Test Harness (`batch_run.py`)

### Purpose
Tests the certification architecture across a parameter grid, checking invariants per run and optionally comparing against a baseline agent.

### Entry Point: `main()`

```
python batch_run.py [--vars 5,8,12] [--cycles 100,300] [--seeds 42,7,99]
                    [--schedule incremental] [--compare] [--ablate-consequence]
                    [--workers 4] [--out results.jsonl]
```

Creates a Cartesian product of `(n_vars × cycles × seeds)` configurations and runs each in parallel.

### Per-Run Flow: `_run_one(cfg)`

1. **Build and run dreth**: `_build_and_run_dreth(cfg)`
   - Create `CausalWorld` with seed
   - Prepare schedule-specific subgraph
   - Create `ChainedAgent` with parameters
   - Run `agent.initialize()` then `cycles` iterations of `world.perturb_by_schedule()` + `agent.run_cycle()`

2. **Extract metrics**: `_extract_arch_metrics(agent, world)`
   - Cert provenance distribution (`earned_by_dist`)
   - Audit cert analysis
   - Route cert counts
   - Dormant alternative counts
   - Composite/hyper-composite metrics
   - Regime skip/pass/fail counts
   - Parking metrics
   - Passive monitoring metrics

3. **Check invariants**: `_check_invariants(arch)`
   - I1: Every cert has `earned_by` set
   - I2: Audit certs use only `reusable`/`not_reusable` role
   - I3: Dormant alternatives are `DormantAlternative` objects
   - I4: Demoted certs carry `revoked_by`
   - I5: Route certs live on target's `route_certs`, not in `certificates`

4. **Optional baseline comparison**: `_build_and_run_baseline(cfg)`
   - `SparseCachedRefitAgent`: K=10 candidates, window=8, threshold=3
   - No nethras, no route-trass pruning, no composites
   - Reports Δiv, Δaudit, Δtime vs dreth

5. **Optional consequence-weight ablation**: Re-run with `consequence_weight=False` and compare tier metrics

### `SparseCachedRefitAgent` — The In-File Baseline

A lightweight diagnostic baseline that:
- Screens top-K candidate parents by intervention sensitivity
- Maintains per-variable residual windows
- Refits when failures exceed threshold
- Refreshes candidate set when refit fails (rate-limited)
- No certification, no composites, no regimes

This baseline is useful for smoke tests and rough intervention comparisons. It is not a decisive same-object ablation. The cleaner empirical comparison is: same hierarchy, same capacity, same routing budget, but with statistically learned routing instead of explicit certs, witnesses, and revocation conditions.

### Result Reporting

Each run produces a `RunResult` with:
- Operational metrics: skip%, interventions, full audits, drift detection
- Architecture metrics: cert provenance, route certs, dormant alternatives, composites
- Tier metrics: per-consequence-tier breakdown
- Invariant violations
- Baseline comparison (when enabled)
- Regime summary

Aggregate reporting computes means, distributions, and amortization ratios across all runs.

---

## 12. Supporting Modules

### `records.py` — Diagnostic Data Structures

- **`CycleRecord`**: Per-cycle snapshot of agent behavior (what was audited/skipped/deferred). Used for offline confusion-matrix analysis.
- **`FitDiagnostic`**: Per-audit record of hypothesis space, scores, ties, per-probe arrays. Write-only from agent — never feeds back into fit selection.

### `summary.py` — Run-End Analysis

- **`RunAnalyzer`**: Reads completed agent state and computes all summary metrics (pure reads, no writes)
- **`SummaryRenderer`**: Formats `RunAnalyzer` output as human-readable multi-section report

### `baseline.py` — Naive Comparison Agent

- **`RefitBaseline`**: Refits every visible variable from scratch every cycle. No sentinels, no certification. Pure brute force cost baseline.

### `cli.py` — CLI Entrypoint

Command-line interface for running single configurations with detailed output.

---

## 13. Key Concepts & Terminology

### Nethra (नेत्र)
A **factoring that earned certification by surviving intervention tests in a specific scope**. Not a label. Operative: certified nethras become active filters deciding what later evidence counts as tareth or trass.

### Tareth
**Substitution changes the operation outcome; preserve the distinction.** A concrete witness exists — the specific intervention that produced propagation. Tested, load-bearing.

### Trass
**Substitution does not change the operation outcome; collapse is allowed.** Provisional: revocable when scope changes or contradiction appears.

### False-Trass
**Two locally-trass nethras that jointly are tareth.** Composition requires a joint re-test. Local certification does not propagate upward. The PROD(TINY, TINY) pattern: each tiny value is individually below salience threshold, but their product can be much larger.

### Earned Contrast
**A difference that has been sorted by consequence.** Not mere variation — a distinction becomes contrast when the system has learned that the difference changes prediction, recovery, action, or repair. This supports hallucination resistance by forcing unsupported traversal to remain uncertified when the relevant feedback channel exposes missing coverage.

### Morphology vs. Cause
- **Morphology**: Structural observations readable from candidate shape (same parents, close scores). No interventions required.
- **Cause**: Why candidates tie (genuine equivalence, library gap, under-probing). Requires separating probes and regime-survival evidence.

### Operation
The decision target for which a nethra is being used: `skip`, `route`, `compress`, `audit`, `reexamine`. Every tareth/trass claim must name its operation — trass-for-skip does not imply trass-for-route.

### Authority
What a certificate is allowed to do: `none` → `prefer` → `guarded_reuse` → `skip` → `propagate`. Certification and authority are separate dimensions.

---

## 14. Data Flow & Lifecycle

### Variable Lifecycle

```
      ┌──────────────────────────────────────────────────────────┐
      │                    Per Variable                          │
      │                                                          │
      │  BOOT: screen candidates → first audit → "proposed"      │
      │           │                                              │
      │           ▼                                              │
      │  STABILIZE: sentinel check each cycle                    │
      │    ├── pass → increment strong_observations              │
      │    │          if ≥ promote_after → "certified"           │
      │    │                                                     │
      │    └── fail → re-audit                                   │
      │         ├── same fit → noisy miss (backoff)              │
      │         └── diff fit → cascade invalidation              │
      │                        weak_streak++                     │
      │                        if ≥ threshold → NoveltyNethra    │
      │                                                          │
      │  CERTIFIED: sentinel cheap-path each cycle               │
      │    ├── pass → skip (no work)                             │
      │    │          maybe enter dormancy                       │
      │    └── fail → demote to "uncertain" → re-audit           │
      │                                                          │
      │  TRASS: operation_role test showed no propagation        │
      │    └── skip entirely (no sentinel even)                  │
      │        but: provisional → needs sentinel_passes to earn  │
      │             hard-suppress authority                       │
      └──────────────────────────────────────────────────────────┘
```

### Certification Cascade

```
Sentinel failure on var V
  → ledger.invalidate({V})
    → closure_descendants({V}) — all vars that believe they depend on V
      → for each descendant D:
          if certified → demote to "uncertain"
          if proposed → reset strong_observations
    → route-trass prune: if D has route-trass cert for V, skip D
  → re-audit V
    → if fit changed:
        → record drift
        → _find_joint_trass_candidates (scan for false-trass)
        → regime_register.observe(cert_events)
    → if fit unchanged:
        → noisy miss; increment consecutive_sentinel_failures
        → if threshold exceeded → backoff / budget escalation
```

---

## 15. The Multi-Level Certification Hierarchy

Dreth implements (or envisions) certification at multiple nested levels:

### Level 0: Variable Fits
Each variable's `(parents, func)` hypothesis earns certification through:
- Repeated matching audits → `status = "certified"`
- Sentinel monitoring validates continued correctness
- Operation role (`tareth`/`trass`) tested via substitution

### Level 0.5: Composite Nethras
Pairs of trass variables tested for joint interaction:
- `CompositeNethra` — pairwise joint-false-trass cert
- `HyperCompositeNethra` — component-level handle compressing dense cliques / connected components of overlapping pairwise certs
- Component handles prevent O(k²) overlap from being reported as independent saved work
- Failure of the component handle descends back into the pairwise frontier

### Level 1: Regime Detection
Recurring co-failure patterns across variables:
- `RegimeSignature` — confirmed recurring pattern
- Active cluster-level sentinel can replace N leaf checks
- Confirmed regime without active sentinel is diagnostic/history, not skip authority
- Sentinel parking: leaf probes parked only when a higher handle has checkable authority and the leaf has no unique failures

### Level 1.5: Consequence-Weighted Tiers
Variables bucketed by downstream dependency count:
- Different sentinel density and promotion thresholds per tier
- Higher-tier variables require stronger evidence for authority

### Level 2 (Envisioned): Forms
Shared operator patterns across variables — a form becomes an operative nethra when certified through form-level substitution tests. Not yet fully implemented but structurally anticipated.

---

## 16. Invariants & Correctness Properties

### Batch-Checked Invariants

| ID | Property | Description |
|----|----------|-------------|
| I1 | `earned_by` provenance | Every cert has a non-empty `earned_by` string |
| I2 | Audit cert vocabulary | Audit certs use only `reusable`/`not_reusable` role |
| I3 | Dormant type safety | `dormant_alternatives` holds `DormantAlternative` objects |
| I4 | Revocation provenance | Demoted certs carry `revoked_by` (not None) |
| I5 | Route cert ownership | Route certs live on target's `route_certs`, not in `certificates` |

### Design Invariants (from code comments)

| # | Property |
|---|----------|
| Core | Certified nethras are operative — they gate future reasoning |
| Core | Tareth/trass verdicts are scoped to context, not permanent labels |
| Core | Morphology ≠ Cause — score proximity doesn't justify collapse |
| Core | Ambiguity is first-class — insufficient evidence → TiedFrontier survives |
| Core | Composition requires joint re-test — local trass doesn't propagate |
| 7 | High-cost domains require stricter sentinel thresholds |
| 20 | Collapse requires `stable_count ≥ 3 AND distinct_contexts_seen ≥ 2` |
| 28-30 | Dormant alternatives revived when they win a later audit |
| 31 | Revival ≥ 2 + distinct contexts ≥ 2 → frontier_survival evidence |
| 36 | Composite sentinels monitor the relation, not individual members |
| 50 | Route/include by default unless excluded by cert |
| 70 | Composite polling only when tied to an active consequence path |

---

## 17. Performance Optimization Layers

The framework implements multiple layers of work reduction, each earned through certification:

### Layer 1: Trass Skip
Variables whose substitution doesn't propagate skip entirely. No sentinel check, no work. **Strongest authority** — requires the most evidence (sentinel_passes scaled by consequence tier).

### Layer 2: Compression
Cached predictions valid under specific gate conditions. If gate matches → use cached value. Otherwise run full computation. Gate = conjunction of `(var, target_value, tolerance)` conditions.

### Layer 3: Sentinel Cheap Path
Run 5 sentinel probes instead of 30+ audit probes. If all pass, accept current fit. Cost: ~6× cheaper than full audit per cycle.

### Layer 4: Composite Handles
Pairwise joint-interaction probes that cover member variables when their joint relation persists. Report unique variable skips separately from raw pairwise pass counts; overlapping hub variables can otherwise inflate apparent savings.

### Layer 5: HyperComposite (Component) Handles
Dense pairwise composite cliques / connected components compressed to component handles. These prevent pairwise explosion and should report lower checks suppressed separately from unique variable-cycle skips.

### Layer 6: Regime Sentinels
Cluster-level probes that cover regime members only when the regime has an active/checkable sentinel. 2 world calls per active regime sentinel vs. many member leaf checks. Confirmed-but-unsentinelled regimes annotate history only.

### Layer 7: Sentinel Parking
Leaf sentinels parked when covered by higher handles and provably redundant. Zero work per cycle.

### Layer 8: Passive Residual Monitoring
Pre-sentinel check using certified fit predictions. If residual within envelope → skip active sentinel entirely. Saves intervention cost.

### Layer 9: Dormancy
Certified+stable variables removed from hot pass entirely. Re-entry only via cascade invalidation.

### Amortization Metrics (from batch_run.py aggregate)

```
handle_amortization = (composite_skips + regime_skips + parking_skips) / total_skips
```

Reports what fraction of unique skip decisions came from higher handles vs. individual sentinel passes. For composite/component layers, raw lower-check suppression and raw pairwise pass counts must be reported separately from unique variable-cycle skips.

---

## 18. Philosophical Foundations

### From `earned_contrast.md` — The Hallucination-Resistance Theory

Dreth embeds a specific theory about what makes inference *grounded* vs. *hallucinated*:

1. **Variation vs. Contrast**: Different states ≠ meaningful distinctions. Contrast requires consequence — knowing which differences change prediction, recovery, or action.

2. **Causal Ledger**: Not archival (stores what happened) but **causal** (changes what the system can compress, skip, trust, or infer). A failure record that doesn't affect future traversal is memory, not grounding.

3. **Hallucination = ungrounded traversal**: Moving fluently through plausible variation without enough earned contrast to know which distinctions are load-bearing.

4. **Authority-gated hallucination resistance**: Dreth does not guarantee hallucination elimination. It can withhold authority, halt, or trigger repair when no certified handle covers a case and a reliable monitor/test/feedback channel exposes that gap. In ambiguous domains, the monitor can fail; the vulnerability shifts to the monitoring and authority layer.

5. **Divergence test**: Same outward competence reached via failure-and-recovery vs. guided-success should diverge under perturbation, ambiguity, adversarial shortcuts, and reduced audit budget. If no divergence → ledger isn't doing real work.

### The Attention Economy

The framework closes a **feedback loop**: certified Level-0 nethras reduce the hypothesis space for further discovery. Certified composites and regimes reduce monitoring cost. Each level of certification feeds back into the cost of the next level. The loop compounds — savings are multiplicative, not additive.

### Operative vs. Descriptive

This is Dreth's core philosophical commitment: a belief earns the right to reduce work only through **observed consequence**. No prior, no assumption, no structural similarity justifies a shortcut. Only surviving intervention tests in scope, persisting through sentinel monitoring, and being revoked when contradicted.

---

## 19. Current Limitations & Remaining Burden

This analysis is a system map, not proof that every architectural claim has been secured. The remaining burden is specific:

1. **Feedback channel reliability**: Dreth needs some reliable constraint channel — passive residual, test suite, simulator, proof checker, sensor threshold, user correction, or downstream contradiction. Without a constraint signal, no agent can know that a boundary has failed.

2. **Passive monitoring is not authority**: Passive residuals can save interventions and seed suspicion. They do not by themselves certify tareth/trass, route authority, regime authority, or component authority. Authority still requires witnessed consequence or an equivalent hard check.

3. **Composite overlap accounting**: Pairwise composite passes must not be reported as independent saved work. The clean metrics are unique variable-cycle skips, component-level lower checks suppressed, raw pairwise pass evidence, overlap degree, and duplicate factor.

4. **Regime authority requires an active sentinel**: A confirmed co-failure pattern without a working cluster-level witness is historical structure. It may guide search, but it should not suppress lower checks or park sentinels by itself.

5. **Formal compute bounds are conditional**: Bounded repair search requires assumptions such as sparse failures, bounded branching, valid covering certificates, reliable monitors, and bounded probe cost. In adversarial or rapidly changing worlds, the repair frontier can expand toward the flat case.

6. **Current baselines are diagnostic, not decisive**: The decisive comparison is not against a naive full-refit or lightweight cached refit alone. The cleaner ablation is an equivalent hierarchy with the same capacity and routing budget but without explicit certs, witnesses, and revocation conditions.

7. **System-level forgetting avoidance is not weight-level editing**: Dreth can freeze and route around failed behavior when a behavior is behind a certified boundary. It does not directly solve interference inside dense shared weights.

8. **OOD hallucination is not mathematically eliminated**: Dreth can block uncertified outputs from becoming authoritative when coverage failure is detected. If the monitor misses the anomaly or hallucinates confidence, the fault can bypass the repair frontier.

9. **Long-horizon credit localization is conditional**: Certified macro-boundaries can turn some long-horizon failures into graph traversal over authority. This requires that the failed causal path is covered by the active certificate graph and that the relevant boundary failure is observable.

The strongest defensible thesis is therefore: Dreth gives systems explicit, revocable trust boundaries and a repair-routing discipline. Its value is measured by whether those boundaries improve localization, reduce collateral damage, and reduce repair/search cost relative to equivalent systems without certified authority.

---

*Analysis generated from Dooces/dreth repository. All module references correspond to the `dreth/` package and `scripts/` directory.*
