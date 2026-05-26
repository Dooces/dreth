# Design Understanding

## What dreth Is

dreth is a simulation of **causal discovery under uncertainty**. A hidden world contains variables connected by causal functions (a DAG). An agent observes variable values and must figure out the causal structure — but it can never look at the ground truth directly. It must *earn* its knowledge through a disciplined pipeline of observation, hypothesis generation, intervention, and certification.

The central question dreth explores: **how does an agent build reliable causal knowledge when it can only see effects, never causes directly?**

## The Hidden World (`world.py`)

The world is a directed acyclic graph of variables. Each variable has:
- A **true parent** (Tareth) that causally determines it via a function from `HIDDEN_FUNC_LIBRARY`.
- **Noise** added to the causal signal.
- Possible **regime changes** where the causal function or parent can shift mid-run.

The world exposes two interfaces to the agent:
1. **Observation**: read current variable values (includes noise).
2. **Intervention**: force a variable to a specific value and observe downstream effects.

The world also includes `SIN` in its hidden function library specifically to test whether the agent can handle functional forms it doesn't know about (novelty).

## The Earned-Authority Model

### Why "Earned"?

Most causal discovery systems label relationships: "X causes Y" becomes a fact once some statistical threshold is met. dreth rejects this. Instead:

- A **VarNethra** is a structured record that a specific parent→child link, with a specific function form and noise envelope, survived intervention testing within a specific cycle range (scope).
- A **NethraCertificate** is the final frozen artifact: proof that the Nethra earned authority.
- Authority is **scoped** — it applies to the cycle range in which it was tested, not forever.
- Authority can be **invalidated** by regime changes or new contradictory evidence.

### The Pipeline

```
Observe → Frontier → Audit → Fit → Sentinel → Certify → Predict
```

Each stage has a specific role and cannot be skipped:

1. **Observe**: gather raw data.
2. **Frontier**: decide which variable deserves attention (budget-constrained).
3. **Audit** (`_full_audit_var`): check if existing authority is still valid; if not, trigger re-evaluation.
4. **Fit** (`fit_var`): enumerate all parent×function combinations, score each against observed data. Output is morphology (scores, tie sets), never a causal claim.
5. **Sentinel** (`select_var_sentinels` + `check_var_sentinels_with_envelope`): design and execute targeted interventions. This is where Tareth is distinguished from Trass. The sentinel system uses cost-dispatch logic — cheap checks first, expensive interventions only when needed.
6. **Certify** (`_certify_operation_role`): if the hypothesis survives sentinels, issue a `NethraCertificate`. This is the *only* point where authority is granted.
7. **Predict** (`predict_var`): use certified Nethras to predict values. Predictions are only as good as the authority behind them.

### The False-Trass Problem

The hardest problem in dreth: a Trass (false parent) can correlate perfectly with the child under normal observation. Only interventions can reveal the difference. The sentinel system is specifically designed to catch this:

- `TEMPORAL_TRASS` is logged when an intervention reveals that a previously plausible parent is actually false.
- The `TiedFrontier` tracks cases where multiple hypotheses score equally well, requiring further intervention to resolve.

## The Hybrid Provider Layer (`hybrid.py`)

dreth supports pluggable "providers" that can advise the agent:

- **ResidualPrediction**: predict residuals from certified Nethras.
- **ParentRanking**: suggest which parents to try first.
- **ProbeProposal**: suggest which interventions to run.
- **ExpertPrediction**: provide direct predictions for variables.
- **RepairEvent**: suggest when to re-audit.

**Critical invariant**: providers *advise only*. They never:
- Create `NethraCertificate`s
- Mutate ledger state
- Have their confidence treated as cert authority

Default implementations are symbolic (e.g., `FuncLibraryExpert`, `SensitivityParentRanker`, `HistoryProbeProposer`). The architecture supports swapping in learned providers, but the authority boundary is enforced regardless.

## The Regime Layer (`regime.py`)

The world can change its causal structure mid-run (regime changes). The regime layer detects this:

- `CertEvent`: tracks certification behavior (successes, failures, revocations).
- `RegimeSignature`: captures recurring patterns of co-failure across variables.
- `RegimeRegister`: accumulates signatures and promotes them when they recur enough.

Regime detection is important because authority is scoped — a Nethra certified under one regime may be invalid under another.

## Context & Provenance (`context_role_index.py`)

The `ContextRoleIndex` provides a structured view over the Nethra graph:

- `NethraKind`: what type of Nethra (direct, composite, etc.)
- `NethraSource`: where the Nethra came from (fit, sentinel, repair, etc.)
- `EdgeKind`: the type of edge in the graph
- `ContextRole`: the role a variable plays in a specific context
- `NethraNode`: a node in the provenance graph

The index is a *view* over existing ledger data, not separate storage. It enables queries like "show me all Nethras where variable X plays a parent role in context Y."

## Repair (`repair_agenda.py`)

When authority breaks down (regime change, contradictory evidence), the `RepairAgenda` tracks what needs fixing:

- `RepairAgendaItem`: a single variable that needs re-evaluation, with reason and priority.
- `RepairAgenda`: the collection of pending items.

**Key invariant**: the agenda is a *planning surface only*. Items do not authorize repair — they must still go through the full audit→fit→sentinel→certify pipeline.

## Diagnostic / Shadow Layers

dreth has extensive offline analysis infrastructure. All of it is strictly read-only:

### Relative Authority (`relative_authority.py`, `relative_authority_frontier.py`, `relative_authority_observer.py`)
- Records for future NethraGraph work: `NethraNodeRef`, `NethraRelation`, `RelativeAuthorityRecord`, `NethraGraphSnapshot`.
- Shadow frontier evaluator proposes bounded candidate sets from graph snapshots.
- Post-run observer builds sparse relative-authority graphs from existing artifacts.
- **Not integrated with runtime agent.** Must not affect core behaviors.

### Shadow Policy (`shadow_policy.py`)
- `ShadowPolicySelector` predicts which provider policy would have performed better.
- Uses `DiagnosticFeatures` extracted from completed runs.
- Does not change the active policy or touch agent state.

### Shadow Authority Throttle (`shadow_authority_throttle.py`)
- Estimates whether downgrading authority when evidence was weak would have helped.
- Uses only agent-visible evidence; never reads hidden-world fields.
- Defines `EvidenceTriggers` and `AuthorityThrottleDecision`.

### Uncertainty Consolidation (`uncertainty_consolidation.py`)
- Factors repeated uncertainty signals into candidate higher-level handles.
- `UncertaintyCase`, `UncertaintyCluster`, `ConsolidationAssist`.
- Conservative: uses only agent-visible evidence.

### Uncertainty Governance (`uncertainty_governance.py`)
- Shadow governance agenda: records observable uncertainty signals, proposes shadow actions.
- Proposals explain their reasoning using only agent-visible evidence.
- Proposals are not actual actions.

### Learned Residual (`learned_residual.py`)
- Stage 3A shadow learned-residual predictor.
- Diagnostic/shadow only.

### Quality (`quality.py`)
- `QualityWeights` and `RunQualityScore` for provider policy scoring.
- Arithmetic only; not read by agent policy.

### Records (`records.py`)
- `CycleRecord`: per-cycle diagnostic snapshot.
- `FitDiagnostic`: per-fit diagnostic data. `near_tie_candidates` is the only field that feeds back into agent state (for `TiedFrontier`).

## Key Design Principles

1. **Separation of earned vs. diagnostic**: the agent's runtime state (ledger, certs, frontier) is strictly separated from diagnostic/shadow layers. Shadow layers observe but never mutate.

2. **Authority is scoped and revocable**: a Nethra is valid for a specific cycle range. Regime changes or new evidence can invalidate it.

3. **No shortcuts**: every causal claim must survive the full pipeline. There is no "fast path" to authority.

4. **Providers advise, pipeline decides**: hybrid providers can suggest, but only the core audit→fit→sentinel→certify pipeline can grant authority.

5. **Cost-aware**: the sentinel system uses cost-dispatch (cheap checks first). The `RefitBaseline` exists specifically to measure the cost of the naive approach.

6. **Novelty-robust**: the hidden world can use functions (`SIN`) that the agent doesn't know about, testing whether the system degrades gracefully.

7. **Observable invariants**: the system logs extensively (`TEMPORAL_TRASS`, regime signatures, diagnostic records) so that post-run analysis can verify that invariants held.
