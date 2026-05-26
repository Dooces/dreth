# dreth

**dreth** is a causal-discovery simulation framework built around *earned authority*. An agent observes a hidden causal world, proposes hypotheses about variable relationships, and earns the right to certify those relationships only by surviving structured intervention tests. Authority is never assumed or labelled — it is *factored from evidence*.

## Core Concepts

### Nethra (VarNethra)
A **Nethra** is not a label. It is a factoring that earned authority by surviving intervention tests in a specific scope. A `VarNethra` records the earned parent→child causal link, the function form, the noise envelope, and the scope (cycle range) in which it was validated.

### Tareth and Trass
- **Tareth**: the *true* causal parent of a variable in the hidden world.
- **Trass**: a *false* parent — a variable that correlates with the child but is not its actual cause.
- The **False-Trass Problem**: the agent must distinguish Tareth from Trass using only observational and interventional evidence, never by peeking at ground truth.

### Earned Authority vs. Assumed Labels
The system enforces a strict boundary: no component may treat a hypothesis as true until it has survived the full audit→fit→sentinel→certify pipeline. Diagnostic and shadow layers exist for offline analysis but are forbidden from mutating agent or ledger state.

## Architecture

### Runtime Core (agent-visible, state-mutating)

| Module | Role |
|---|---|
| `world.py` | Hidden causal world: DAG of variables with causal functions, noise, and regime changes. Provides observation and intervention interfaces. |
| `agent.py` | `ChainedAgent` — the authority-record control loop. Manages the attention frontier, dispatches audit/fit/sentinel/certify cycles, and maintains the ledger. |
| `ledger.py` | Immutable earned-authority records: `NoiseEnvelope`, `VarNethra`, `TiedFrontier`, `NethraCertificate`. The ledger is the single source of truth for what the agent has earned. |
| `fit.py` | `fit_var` — enumerates and scores parent×function hypotheses. Produces morphology output (scores, tie sets) but never cause classification. `predict_var` generates predictions from certified Nethras. |
| `sentinels.py` | `select_var_sentinels` chooses intervention probes; `check_var_sentinels_with_envelope` validates via cheap-path dispatch. Logs `TEMPORAL_TRASS` when interventions reveal false parents. |
| `functions.py` | `FUNC_LIBRARY` (agent vocabulary) and `HIDDEN_FUNC_LIBRARY` (world vocabulary, includes `SIN` for novelty testing). |
| `regime.py` | Regime detection: `CertEvent` tracks cert behavior, `RegimeSignature` captures recurring co-failure patterns, `RegimeRegister` tracks and promotes regimes. |
| `records.py` | `CycleRecord` and `FitDiagnostic` — diagnostic-only per-cycle data. `FitDiagnostic.near_tie_candidates` is the only field that feeds back into agent state (for `TiedFrontier`). |

### Hybrid Provider Layer

| Module | Role |
|---|---|
| `hybrid.py` | Protocol definitions (`ResidualPrediction`, `ParentRanking`, `ProbeProposal`, `ExpertPrediction`, `RepairEvent`) and default symbolic implementations. Providers *advise* but never create `NethraCertificate`s or mutate ledger state. Provider confidence is never treated as cert authority. |
| `learned_residual.py` | Stage 3A shadow learned-residual predictor. Diagnostic/shadow only. |

### Context & Provenance

| Module | Role |
|---|---|
| `context_role_index.py` | Context-indexed provenance over the Nethra graph. Defines `NethraKind`, `NethraSource`, `EdgeKind`, `ContextRole`, `NethraNode`. The index is a *view*, not separate storage. |
| `repair_agenda.py` | `RepairAgenda` — structural representation of pending repair work. A planning surface only; items do not authorize repair. |

### Diagnostic / Shadow Layers (never mutate runtime state)

| Module | Role |
|---|---|
| `relative_authority.py` | Diagnostic-only relative authority records for future NethraGraph work. Not integrated with runtime agent. |
| `relative_authority_frontier.py` | Shadow-only frontier evaluator for diagnostic NethraGraph snapshots. Proposes bounded local candidate sets. |
| `relative_authority_observer.py` | Post-run diagnostic NethraGraph observer. Builds sparse relative-authority graph from existing artifacts. |
| `shadow_authority_throttle.py` | Shadow-only authority throttle evaluator. Estimates whether downgrading authority would have reduced mismatches. Uses only agent-visible evidence; never reads hidden-world fields. |
| `shadow_policy.py` | `ShadowPolicySelector` — diagnostic-only offline policy predictor. Predicts which provider policy would have had lower `quality_cost`. Does not change active policy. |
| `uncertainty_consolidation.py` | Factors repeated uncertainty signals into candidate higher handles. Conservative; uses only agent-visible evidence. |
| `uncertainty_governance.py` | Shadow-only uncertainty governance agenda. Records observable uncertainty signals and proposes shadow governance actions (proposals are not actual actions). |
| `quality.py` | `QualityWeights` and `RunQualityScore` — diagnostic-only provider policy scoring. Arithmetic only; not read by agent policy. |

### Tooling

| Module | Role |
|---|---|
| `cli.py` | Command-line entry point. Parses arguments and runs the simulation. |
| `baseline.py` | `RefitBaseline` — naive agent that refits every visible variable every cycle. Used for cost comparison. |
| `summary.py` | `RunAnalyzer` and `SummaryRenderer` — end-of-run metrics computation and formatting. |

### Scripts (`scripts/`)

Batch runners, benchmarks, and summarization utilities:
`batch_run.py`, `batch_tui.py`, `bench_forms.py`, `bench_frontier.py`, `bench_transition.py`, `baseline_attention_budget.py`, `composite_churn.py`, `compare_uncertainty_consolidation_modes.py`, `test_rare_catastrophe.py`, `visualize.py`, and various `summarize_*.py` scripts for offline analysis of authority, policy, frontier, governance, and consolidation artifacts.

### Tests (`tests/`)

Comprehensive test suite covering core logic, edge cases, regime changes, tied frontiers, sentinel behavior, hybrid providers, shadow layers, and more.

## The Agent Lifecycle (per cycle)

1. **Observe** — read current variable values from the world.
2. **Frontier selection** — pick which variable to attend to (attention budget).
3. **Audit** (`_full_audit_var`) — if the variable has no Nethra or its cert is stale, run a full audit.
4. **Fit** (`fit_var`) — enumerate parent×function hypotheses, score them, identify ties.
5. **Sentinel probing** (`select_var_sentinels`, `check_var_sentinels_with_envelope`) — choose and execute interventions to distinguish Tareth from Trass.
6. **Certify** (`_certify_operation_role`) — if evidence survives, issue a `NethraCertificate`. Authority is earned.
7. **Predict** (`predict_var`) — use certified Nethras to predict variable values.
8. **Record** — log `CycleRecord` and `FitDiagnostic` for offline analysis.

## Installation & Usage

```bash
# Clone
git clone https://github.com/Dooces/dreth.git
cd dreth

# Install
pip install -e .

# Run a simulation
python -m dreth.cli --num-vars 5 --num-cycles 100

# Run tests
pytest tests/
```

## Key Design Invariants

- **No peeking**: the agent never reads hidden-world ground truth. All evidence is observational or interventional.
- **Earned, not assumed**: authority flows only through the audit→fit→sentinel→certify pipeline.
- **Shadow layers are read-only**: diagnostic modules observe but never mutate agent or ledger state.
- **Provider confidence ≠ cert authority**: hybrid providers advise; only the core pipeline certifies.
- **Ledger is immutable**: once a `NethraCertificate` is issued, it is a frozen record of earned authority.
