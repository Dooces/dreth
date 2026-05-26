# dreth

`dreth` is an experimental architecture for studying how an agent can build, use, revise, and relate learned structure under limited attention.

The current project is **not** a production ML system and not a claim of solved AGI. It is a simulation framework for testing a narrower question:

> Can an agent keep useful, evidence-shaped structure available while avoiding global trust, global deletion, and flat brute-force rechecking?

The central object is a **nethra**: a learned, reusable structure or handle. A nethra may be composed from other nethras. Whether that nethra is operationally important is not its identity; it is a **context role** assigned under a particular operation, evidence state, and regime.

## Correct current semantics

### Nethra

A **nethra** is learned structure: a reusable handle over an observed regularity, relation, candidate fit, component, regime, frontier candidate, composite, or higher-order pattern.

A nethra is not equivalent to hidden truth. It is also not a disposable label. It is something the system can use, compare, compose, revisit, or demote in scope as evidence changes.

### Context roles: tareth, trass, unresolved, best_available

`tareth` and `trass` are **context-indexed roles**, not global identities.

- `tareth`: this nethra currently matters for the operation/context being considered.
- `trass`: this nethra is currently operationally equivalent or irrelevant for that operation/context.
- `unresolved`: evidence preserves multiple live alternatives or instability.
- `best_available`: the current working structure when nothing better is available, even if uncertainty remains.

A nethra can be `trass` in one context and `tareth` in another. For example, a shape-like handle may be trass for color classification and tareth for grasp planning. The implementation now treats this as a graph/index problem, not as deletion.

### Authority records, not absolute certainty

The code still contains legacy implementation names such as `NethraCertificate`, `certificates`, and `certified_eps`. These should be read as **authority records** or **evidence-bounded commitments**, not as absolute proof.

Authority is graded by provenance, survival, scope, revocation history, sentinels, alternatives, and context. The system should never assume that a structure is globally true merely because it is currently useful.

### Hidden truth vs. agent-visible evidence

Generated worlds contain hidden truth for offline evaluation. The agent must not use it. Hidden truth can be used after a run to interpret failures, but not as the runtime metric for whether an authority record was reasonable.

The internal question is evidence-relative:

> Given what the agent could observe or probe, was this nethra a reasonable best-available handle, a context-role assignment, or an authority/evidence mismatch?

## Active architecture

### Runtime core

| Module | Role |
|---|---|
| `dreth/world.py` | Hidden causal worlds and schedules. Provides scalar observed variables, interventions, drift/regime schedules, and blind challenge generation. Hidden debug state is for offline analysis only. |
| `dreth/agent.py` | `ChainedAgent`, the main control loop. Runs audits, fits, sentinel checks, frontier handling, repair agenda integration, uncertainty consolidation assists, and context-role recording. |
| `dreth/ledger.py` | Core data structures for variable nethras, authority records, tied frontiers, dormant alternatives, composites, envelopes, and revocation state. |
| `dreth/fit.py` | Enumerates and scores parent/function hypotheses under the agent vocabulary. Produces best fits, ties, near-ties, and diagnostics. |
| `dreth/sentinels.py` | Selects and checks sentinel probes used to cheaply test whether prior structure still holds. |
| `dreth/regime.py` | Tracks recurring co-failure/regime patterns. Regime handling is still experimental; regime handles must be backed by active witness logic before they should buy broad skip authority. |
| `dreth/records.py` | Cycle and fit diagnostics. Mostly offline, but near-tie diagnostics feed tied-frontier bookkeeping. |
| `dreth/summary.py` | Run analysis and rendering. Keeps large reporting logic out of the agent. |

### Context-role and uncertainty layers

| Module | Role |
|---|---|
| `dreth/context_role_index.py` | Context-indexed provenance over the nethra graph. Defines nethra nodes, edges, context-role records, and retrieval matches. This is a graph/index view, not a trass reservoir. |
| `dreth/uncertainty_governance.py` | Shadow-only extraction/classification of visible uncertainty signals into proposed governance actions. It does not change behavior. |
| `dreth/uncertainty_consolidation.py` | Groups repeated visible uncertainty into candidate higher handles and, in assist mode, can feed bounded attention/probe/repair hints. This path is experimental and must be judged by off/shadow/assist comparisons. |
| `dreth/repair_agenda.py` | Structural surface for pending repairs. Current priority logic is intentionally limited. |
| `dreth/relative_authority.py` | Diagnostic relative-authority records. |
| `dreth/relative_authority_observer.py` | Post-run graph observer over existing ledger artifacts. |
| `dreth/relative_authority_frontier.py` | Shadow evaluator for graph-frontier proposal priors. |
| `dreth/shadow_authority_throttle.py` | Shadow-only authority/evidence throttle analysis from visible evidence. |
| `dreth/learned_residual.py` | Shadow learned-residual/calibration experiments. Not runtime authority. |
| `dreth/hybrid.py` | Provider interfaces and symbolic defaults for residual prediction, parent ranking, probe proposals, experts, and routing. Providers advise; they do not issue authority records. |
| `dreth/quality.py` | Diagnostic quality-cost scoring used for policy/report comparison. |

## What the current system does

At a high level, the agent repeatedly:

1. Observes current scalar world state.
2. Chooses what needs attention under a limited audit budget.
3. Runs a full audit when a variable or handle needs repair or initial structure.
4. Fits candidate parent/function hypotheses using agent-visible probes.
5. Records ties, near-ties, margins, alternatives, and novelty.
6. Installs the best available working structure into the ledger.
7. Uses sentinels, composites, and route/role records to reduce repeated work when evidence supports it.
8. Revokes or demotes when sentinels or other visible evidence contradict prior authority.
9. Builds context-role provenance over nethras so a structure can remain learned even when trass in a particular context.
10. Optionally runs shadow/assist layers for uncertainty consolidation and context-role local anchors.

The current design intentionally separates:

- learning a reusable structure,
- using it in a given context,
- recording its role in that context,
- and deciding whether it should influence attention or repair.

## Current experimental status

The project has moved beyond the original toy-only regime, but the richer paths are still experimental.

Important current findings:

- `blind_challenge` creates mixed symbolic, nonlinear, latent, delayed, proxy, dense, and weak/noisy structure. It is used to expose where current Dreth has grip and where the current substrate loses grip.
- Uncertainty signals are broad in `blind_challenge`; this is not automatically a bug. It may indicate shared unresolved structure, but broad signals must be consolidated into useful local handles before they should drive attention.
- `uncertainty_consolidation` assist mode is invariant-safe but has produced harmful broad pressure when clustering is too loose. Compression ratio alone is not success.
- `ContextRoleIndex` record mode has been behavior-neutral in sweeps: it can record nethra/context-role provenance without changing runtime behavior.
- `ContextRoleIndex` also demonstrates the corrected semantics: the same learned structure can have multiple roles across contexts, including trass in one context and tareth in another.
- `ContextRoleIndex assist_feature` has shown both promising small-run behavior and harmful scale behavior when matches are admitted too broadly. This path needs strict match gating, deduplication, and attribution before it should be trusted.

Do not treat any assist path as validated merely because invariants pass. Invariants show safety boundaries; they do not prove usefulness.

## Core invariants

The repository tests and batch checks track several invariants. The important conceptual ones are:

- **No hidden-truth runtime access**: generated-world truth is offline evaluation only.
- **Authority must flow through explicit evidence paths**: direct control-flow construction of authority is a bug risk.
- **Providers advise, they do not authorize**.
- **Shadow means no behavior change**: shadow mode must match off-mode operational metrics except for diagnostics.
- **Record-only indexes must not change behavior**.
- **Context roles are local**: `tareth` and `trass` are roles in context, not object identities.
- **Trass is not deletion**: a nethra can be learned and retained even when operationally trass in the current context.
- **Assist features must be reversible and bounded**: probes, monitoring, alternative preservation, repair priority; not direct revocation, skip suppression, or fit replacement.

## Common commands

### Install

```bash
git clone https://github.com/Dooces/dreth.git
cd dreth
pip install -e .
```

### Run tests

```bash
python -m pytest tests/test_cycle_mechanics.py -q
python -m pytest tests/test_blind_challenge.py -q
python -m pytest tests/test_uncertainty_consolidation.py -q
python -m pytest tests/test_nethra_reservoir.py -q
```

`tests/test_nethra_reservoir.py` is currently a compatibility filename. It asserts the corrected `ContextRoleIndex` semantics.

### Basic batch run

```bash
python scripts/batch_run.py \
  --schedule regime_switch \
  --vars 75 \
  --cycles 7500 \
  --seeds 42,99,7 \
  --hybrid-control interfaces \
  --repair-agenda
```

### Blind challenge with context-role recording

```bash
python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50 \
  --cycles 3000 \
  --seeds 42,99,7 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index record \
  --out reports/context_role_record_check.jsonl \
  2>&1 | tee reports/context_role_record_check.log

python scripts/summarize_context_role_index.py \
  --jsonl reports/context_role_record_check.jsonl \
  | tee reports/context_role_record_check_summary.txt
```

### Compare off / record / assist behavior

```bash
python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index off \
  --out reports/context_role_sweep_off.jsonl \
  2>&1 | tee reports/context_role_sweep_off.log

python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation shadow \
  --context-role-index record \
  --out reports/context_role_sweep_record.jsonl \
  2>&1 | tee reports/context_role_sweep_record.log

python scripts/batch_run.py \
  --schedule blind_challenge \
  --challenge-blind \
  --vars 50,75,100 \
  --cycles 3000,7500 \
  --seeds 42,99,7,3,11,13,17,23,29,31 \
  --hybrid-control interfaces \
  --repair-agenda \
  --uncertainty-consolidation assist \
  --uncertainty-assist-policy local_only \
  --context-role-index assist_feature \
  --out reports/context_role_sweep_assist_feature.jsonl \
  2>&1 | tee reports/context_role_sweep_assist_feature.log
```

Interpretation:

- `off == record` means indexing is behavior-neutral.
- `assist_feature` improving metrics is only meaningful if match counters show the index was used and broad-match pressure is controlled.
- `assist_feature` worsening metrics means the index may still be valuable as provenance, but not yet as runtime attention input.

## Scripts

Useful summarizers include:

- `scripts/summarize_blind_challenge.py`
- `scripts/summarize_blind_authority_evidence.py`
- `scripts/summarize_uncertainty_governance.py`
- `scripts/summarize_uncertainty_consolidation.py`
- `scripts/summarize_context_role_index.py`
- `scripts/summarize_relative_authority.py`
- `scripts/summarize_temporal_frontier.py`
- `scripts/compare_uncertainty_consolidation_modes.py`

Most reports are diagnostic. A good report does not imply a runtime path is safe.

## Known unfinished work

These are active design gaps, not polish items:

1. **Context-role assist gating**: runtime use of `ContextRoleIndex` currently needs stricter match quality, deduplication, and attribution.
2. **Uncertainty consolidation specificity**: broad uncertainty must not collapse into giant global clusters without local anchors.
3. **Regime witness semantics**: regime handles should not buy monitoring reduction unless backed by active witnesses, not merely quiescence.
4. **Authority transaction hygiene**: authority object creation should continue moving toward explicit ledger transactions rather than inline agent construction.
5. **Learner integration**: learned/NN components should rank or propose attention/factorization only after deterministic attribution makes the learning target clear.

## What this repository is for

Use this repo to test whether explicit, context-indexed, evidence-shaped structure can help an agent allocate limited attention under changing worlds.

The useful outcomes are not only successes. A good Dreth run should also expose when a handle is too broad, when an assist path creates pressure without benefit, when a nethra is only best-available, and when context roles are being confused with identity.
