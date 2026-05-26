# TODO

## Status Key
- ✅ Done — implemented and tested
- 🔶 Approximate — implemented but may need refinement
- ❌ Missing — not yet implemented
- 🔧 Scaffolded — structure exists but logic is placeholder/stub

---

## Core Pipeline

### ✅ Done
- `world.py` — hidden causal world with DAG, causal functions, noise, observation, and intervention interfaces
- `agent.py` — `ChainedAgent` with full lifecycle: frontier selection, audit, fit, sentinel, certify, predict
- `ledger.py` — immutable earned-authority records: `NoiseEnvelope`, `VarNethra`, `TiedFrontier`, `NethraCertificate`
- `fit.py` — `fit_var` hypothesis enumeration and scoring; `predict_var` prediction from certified Nethras; batched scoring
- `sentinels.py` — `select_var_sentinels` probe selection; `check_var_sentinels_with_envelope` validation with cost-dispatch; `TEMPORAL_TRASS` logging
- `functions.py` — `FUNC_LIBRARY` (agent vocabulary) and `HIDDEN_FUNC_LIBRARY` (world vocabulary with `SIN` for novelty)
- `records.py` — `CycleRecord` and `FitDiagnostic` diagnostic records; `near_tie_candidates` feedback to `TiedFrontier`
- `cli.py` — command-line entry point with argument parsing
- `baseline.py` — `RefitBaseline` naive agent for cost comparison
- `summary.py` — `RunAnalyzer` and `SummaryRenderer` for end-of-run metrics

### ✅ Done — Regime Detection
- `regime.py` — `CertEvent`, `RegimeSignature`, `RegimeRegister` for detecting and tracking causal regime changes

### ✅ Done — Hybrid Provider Layer
- `hybrid.py` — protocol definitions (`ResidualPrediction`, `ParentRanking`, `ProbeProposal`, `ExpertPrediction`, `RepairEvent`) and default symbolic implementations
- Provider boundary enforced: providers advise only, never create certs or mutate ledger

### ✅ Done — Context & Provenance
- `context_role_index.py` — `NethraKind`, `NethraSource`, `EdgeKind`, `ContextRole`, `NethraNode` for context-indexed provenance
- `repair_agenda.py` — `RepairAgendaItem` and `RepairAgenda` as planning surface (does not authorize repair)

---

## Diagnostic / Shadow Layers

### ✅ Done — Shadow Analysis Infrastructure
- `quality.py` — `QualityWeights` and `RunQualityScore` (diagnostic-only scoring)
- `shadow_policy.py` — `ShadowPolicySelector` offline policy predictor
- `shadow_authority_throttle.py` — shadow authority throttle evaluator (agent-visible evidence only)
- `uncertainty_consolidation.py` — uncertainty signal clustering and consolidation assists
- `uncertainty_governance.py` — shadow governance agenda with observable-evidence proposals

### 🔧 Scaffolded — Relative Authority / NethraGraph
- `relative_authority.py` — diagnostic-only records (`NethraNodeRef`, `NethraRelation`, `RelativeAuthorityRecord`, `NethraGraphSnapshot`). **Not integrated with runtime agent.**
- `relative_authority_frontier.py` — shadow frontier evaluator for graph snapshots. Diagnostic only.
- `relative_authority_observer.py` — post-run graph observer. Observational only.

### 🔧 Scaffolded — Learned Residual
- `learned_residual.py` — Stage 3A shadow learned-residual predictor. Diagnostic/shadow only.

---

## Scripts & Tooling

### ✅ Done
- `scripts/batch_run.py` — batch simulation runner
- `scripts/batch_tui.py` — terminal UI for batch runs
- `scripts/bench_forms.py` — function form benchmarks
- `scripts/bench_frontier.py` — frontier selection benchmarks
- `scripts/bench_transition.py` — regime transition benchmarks
- `scripts/baseline_attention_budget.py` — baseline attention budget analysis
- `scripts/composite_churn.py` — composite churn analysis
- `scripts/compare_uncertainty_consolidation_modes.py` — consolidation mode comparison
- `scripts/test_rare_catastrophe.py` — rare catastrophe scenario testing
- `scripts/visualize.py` — run visualization
- `scripts/summarize_blind_authority_evidence.py` — authority evidence summarization
- `scripts/summarize_blind_challenge.py` — blind challenge summarization
- `scripts/summarize_context_role_index.py` — context role index summarization
- `scripts/summarize_nethra_reservoir.py` — Nethra reservoir summarization
- `scripts/summarize_policy_report.py` — policy report summarization
- `scripts/summarize_relative_authority.py` — relative authority summarization
- `scripts/summarize_temporal_frontier.py` — temporal frontier summarization
- `scripts/summarize_uncertainty_consolidation.py` — uncertainty consolidation summarization
- `scripts/summarize_uncertainty_governance.py` — uncertainty governance summarization

---

## Tests

### ✅ Done
- Comprehensive test suite in `tests/` covering core pipeline, edge cases, regime changes, tied frontiers, sentinel behavior, hybrid providers, shadow layers, and diagnostic records.

---

## What's Next

### N3: NethraGraph Integration (Next Real Work)
The relative authority modules (`relative_authority.py`, `relative_authority_frontier.py`, `relative_authority_observer.py`) are scaffolded with data structures and diagnostic-only logic, but are **not integrated with the runtime agent**. The next major milestone is:

1. **Connect `NethraGraphSnapshot` to the agent's decision loop** — allow the agent to use relative authority information when selecting frontiers and prioritizing audits.
2. **Promote `RelativeAuthorityRecord` from diagnostic to runtime** — with appropriate earned-authority guards (records must still go through the certification pipeline).
3. **Integrate `TemporalGraphFrontierEvaluator` with frontier selection** — use graph-based candidate proposals to inform attention allocation.

### Learned Residual Promotion
- `learned_residual.py` is Stage 3A (shadow only). Promoting it to influence provider selection or repair prioritization is a future milestone, contingent on the authority boundary being maintained.

### Shadow → Runtime Promotion Path
Several shadow modules contain useful logic that could eventually inform runtime decisions:
- `ShadowPolicySelector` → could inform automatic provider policy switching
- `AuthorityThrottleDecision` → could inform conservative authority granting
- `UncertaintyGovernance` → could inform repair prioritization

Each promotion must preserve the invariant: **shadow insights advise, they do not certify.**
