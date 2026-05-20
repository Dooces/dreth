# DRETH run flowchart — current executable paths

This chart separates the normal CLI run, the regime-recall receipt harness, and the inactive/scaffold paths. It is meant to replace historical `run_flowchart.md` notes that mixed implemented behavior with future design.

## Entry points

```bash
python3 dreth_causal_v28.py [flags]
python3 -m pytest -q
python3 tests/test_regime_recall.py
python3 tests/test_regime_recall.py --stress
python3 scripts/bench_forms.py [flags]
```

## Current invariant summary

- `dreth_causal_v28.py` is a compatibility wrapper into `dreth.cli.run`.
- `--mode v29-algebraic-only` is the strongest currently demonstrated efficiency path.
- `--mode v29-equiv-only` exists but may be neutral depending on seed/horizon.
- v30 forms are default-enabled unless disabled, so `--mode v28` is not necessarily pure v28.
- Dormant vars are skipped for proactive audit queueing, not for all sentinel work.
- Adaptive budget is computed but intentionally not used.
- `TiedFrontier` exists as a dataclass only; it is not a live nareth/frontier mechanism yet.
- `total_interventions` is not a complete intervention-observation counter.

## CLI path

```mermaid
flowchart TD
    A[dreth_causal_v28.py] --> B[dreth.cli.run]
    B --> C[parse_args]
    C --> D[CausalWorld]
    D --> E[ChainedAgent]
    E --> F{mode starts v29?}
    F -- yes --> G[load dreth_extensions]
    F -- no --> H[no extension]
    G --> I{forms enabled?}
    H --> I
    I -- yes/default --> J[V30Bundle attached]
    I -- no/--disable-forms --> K[v30=None]
    J --> L[agent.initialize]
    K --> L
    L --> M[cycle loop]

    M --> N[world perturb/reveal]
    N --> O{reveal?}
    O -- yes --> P[agent.on_variable_revealed]
    O -- no --> Q[agent.run_cycle]
    P --> Q

    Q --> R[v30.on_cycle_start if attached]
    R --> S[dormant maintenance]
    S --> T[topological order]
    T --> U[per-var cheap path]

    U --> V{trass?}
    V -- yes --> V1[trass skip]
    V -- no --> W{compression valid?}
    W -- yes --> W1[compression skip]
    W -- no --> X{sentinels pass?}
    X -- yes --> X1[sentinel skip + envelope delta]
    X -- no --> Y[queue full audit / invalidate closure]

    Y --> Z[priority audit budget]
    Z --> AA[_try_form_hypothesis if forms]
    AA --> AB{form match?}
    AB -- yes --> AC[_install_var form hypothesis]
    AB -- no --> AD[_full_audit_var -> fit_var]
    AD --> AE[_install_var]
    AC --> AF[v30 hooks / extension hooks / metrics]
    AE --> AF
    AF --> M
```

## Full audit path

```mermaid
flowchart TD
    A[_full_audit_var] --> B[compute available tareth/proposed parents]
    B --> C[estimate hypothesis count]
    C --> D[_adaptive_probe_budget computed but not used]
    D --> E[budget = self.intervention_budget]
    E --> F[fit_var]
    F --> G[enumerate hypotheses]
    G --> H[build candidate probe pool]
    H --> I[rank probes by discrimination]
    I --> J[score hypotheses batched]
    J --> K[diag: best/second/margin/tie_set/probes/preds]
    K --> L[return best parents/function/score]
```

Current gap: this path records exact `tie_set`, but not `near_tie_set`, `frontier_candidate_preds`, or separating probes. `TiedFrontier` is therefore not operational.

## Install path

```mermaid
flowchart TD
    A[_install_var] --> B[old_hyp/new_hyp]
    B --> C[exact tie_set from last FitDiagnostic]
    C --> D{same-parent exact-tied churn?}
    D -- yes --> E[preserve state]
    D -- no --> F[semantic reset if syntactic change]
    E --> G[ledger.update_var]
    F --> G
    G --> H{parents changed?}
    H -- yes --> I[invalidate topo cache]
    H -- no --> J[continue]
    I --> J
    J --> K[v30.on_signature_change]
    K --> L[certify operation role if untested]
    L --> M{role trass?}
    M -- yes --> N[collapse to trass]
    M -- no --> O[increment strong observations]
    O --> P[attach sentinels if absent]
    P --> Q[promote if stable]
    Q --> R[discover compressions]
    R --> S[v29 extension compressions]
    S --> T[v30 audit-complete hook]
```

Needed change: extend exact-tie churn suppression to same-parent near-tie frontier churn, while still treating parent changes as structural.

## Dormant partition path

```mermaid
flowchart TD
    A[run_cycle] --> B[topo_order over all visible vars]
    B --> C[cheap path still runs for all vars]
    C --> D[sentinel checks add envelope deltas]
    D --> E{stable certified authoritative enough?}
    E -- yes --> F[_maybe_demote removes from live set]
    E -- no --> G[remains live]
    F --> H[dormant means no proactive audit queue]
    H --> I[dormant safety sweep every N cycles]
    I --> J{sentinel failure / form unhealthy / watch state?}
    J -- yes --> K[promote back to live]
    J -- no --> H
```

Invariant: do not replace the first pass with `live_topo`. That broke form behavior signatures because envelope deltas stopped accumulating.

## Regime-recall receipt path

```mermaid
flowchart TD
    A[tests/test_regime_recall.py] --> B{--stress?}
    B -- no --> C[canonical run_receipt]
    B -- yes --> D[run_stress only]
    C --> D

    C --> E[make_agent]
    E --> F[CausalWorld + ChainedAgent]
    F --> G[V30Bundle enable_forms=True]
    G --> H[agent.initialize]
    H --> I[warm-up shaped cycles]
    I --> J[discover eligible high/alt 2-ary forms]
    J --> K{valid form pair?}
    K -- no --> L[SKIP]
    K -- yes --> M[Receipt 1: quarantine high form]
    M --> N[probe recovery]
    N --> O{reactivated?}
    O -- no --> P[FAIL]
    O -- yes --> Q[Receipt 2/3: quarantine pair]
    Q --> R[compare A* selector vs flat selector]
    R --> S{A* slower?}
    S -- yes --> T[PARTIAL]
    S -- no --> U[PASS]
```

Receipt semantics:
- R1 recovery failure is a correctness failure.
- A* slower than flat is PARTIAL, not FAIL.
- SKIP means the required form pair did not emerge in that seed/horizon.

## Measurement warnings

The following counters must not be used as final efficiency proof until P0 accounting is fixed:

- `total_interventions`
- form on/off intervention delta
- v29/v30 intervention savings

Use full-audit count as a rough signal only. For probe economy, count actual world intervention method calls.

## Non-paths / scaffolds

These exist in code but should not be described as active capabilities:

- `TiedFrontier`: dataclass only, no active lifecycle.
- `near_tie_set`: not in diagnostics yet.
- `frontier_candidate_preds`: not in diagnostics yet.
- NN proposer scores: trained/passive, not used for decisions.
- Cold storage: experimental; key loading needs repair.
- Adaptive budget: computed, not applied.
