# TODO

## Current Read

- ✅ Core Dreth pipeline exists: world, audit, fit, sentinel, ledger, summaries, batch harness.
- ✅ Hybrid provider interfaces exist and remain advisory rather than authority-granting.
- ✅ Shadow/diagnostic layers exist for policy, residuals, authority/evidence, relative authority, temporal frontier, uncertainty governance, and uncertainty consolidation.
- ✅ `ContextRoleIndex` now records nethra nodes and context-indexed roles without changing behavior in record mode.
- ✅ The context-role model is conceptually corrected: a nethra is learned structure; `tareth`/`trass` are context-dependent roles, not the identity of the nethra.
- ✅ Record-mode context indexing is clean: `off == record` in the 60-run sweep.
- ⚠️ `assist_feature` is live and connected, but harmful at scale. It over-connects context-role matches and increases cost/failures.
- ⚠️ Runtime assist from context-role matches must be gated, deduped, and attributed before any further runtime expansion.
- ❌ Do not run the broad attention/cost optimization bundle yet. It touches too many runtime paths and would destroy attribution.

---

## Immediate Next Work

- [ ] Implement strict `ContextRoleIndex` match gating, deduplication, and attribution.
- [ ] Add `ContextRoleMatchQuality` with visible-only match-quality fields.
- [ ] Add `--context-role-anchor-policy off|strict|loose`.
- [ ] Make `strict` the default policy when `--context-role-index assist_feature` is enabled.
- [ ] Preserve `loose` only for reproducing current over-connection behavior.
- [ ] Add raw/deduped/suppressed/capped match counters.
- [ ] Add assist attribution by match reason, cluster, nethra id, and assist kind.
- [ ] Update `scripts/summarize_context_role_index.py` with strict-vs-loose match pressure reporting.
- [ ] Verify `off == record` remains true.
- [ ] Verify `loose` reproduces the current broad harmful behavior.
- [ ] Verify `strict` sharply reduces match pressure and does not worsen metrics as badly as `loose`.
- [ ] If `strict` still worsens metrics, keep `ContextRoleIndex` as provenance/record-only for now.

---

## Deferred / Reference Work

- [ ] Weighted intervention-cost tracking is low-risk and can be cherry-picked later as diagnostic-only.
- [ ] Tied-frontier separating probes are likely useful, but should wait until context-role assist attribution is controlled.
- [ ] Dormant alternative revival is conceptually aligned, but should wait until retrieval gates are strict enough.
- [ ] Regime-triggered inert re-screening may be valuable, but should not be mixed with context-role gating changes.
- [ ] Cheap cascade pre-checks are higher-risk because they alter re-audit behavior. Do not implement before attribution tools are stable.
- [ ] Learned ranker/factorizer is not next. Use it only after deterministic match gating and ablation show that rule-based retrieval cannot separate useful anchors from broad noise.

---

## Non-Negotiable Invariants

- [ ] Authority is earned by evidence, not provider confidence, graph proximity, morphology, or index membership.
- [ ] Hidden truth/debug manifest must not be read by runtime matching, clustering, or assist logic.
- [ ] Shadow/diagnostic layers may observe; they must not mutate authority.
- [ ] `ContextRoleIndex` records provenance and role history; it is not truth.
- [ ] `tareth`/`trass` are context roles, not global properties of a nethra.
- [ ] No context-role match may directly issue certs, revoke certs, suppress skips, or replace `fit_var`.
- [ ] Broad unresolved status, broad role equality, or generic uncertainty signals must not qualify as local anchors by themselves.

---

# Appendix A — Next Codex Prompt: Strict ContextRoleIndex Match Gating

```text
You are working in the Dooces/dreth repo.

Current result:
ContextRoleIndex record mode is behavior-neutral and conceptually correct:
- off == record over 60 runs
- index records nethra nodes and context roles
- record mode shows trass-in-one-context / tareth-in-another examples

But ContextRoleIndex assist_feature is behaviorally worse:
- iv increased
- quality_cost increased
- full_audits increased
- revocations increased
- unique_fails increased
- regime_sentinel_fail increased sharply
- passive stress increased
- dormant alternatives dropped to 0.0
- context_role_index_matches and local-anchor hits are very large

Diagnosis:
The index is not failing to connect.
It is over-connecting.
Context-role matches are being admitted as local anchors too broadly and producing harmful assist pressure.

Goal:
Add strict ContextRoleIndex match gating, deduplication, and attribution before allowing index matches to influence uncertainty consolidation assists.

Core invariant:
Default behavior unchanged.
Record mode remains behavior-neutral.
No hidden truth in runtime matching.
No authority revocation.
No skip suppression.
No fit_var replacement.
No cert issuance from index matches.
No broad assist from generic provenance.

Tasks:

1. Add ContextRoleMatchQuality.

For every match, compute visible-only fields:
- shared_var
- shared_target_var
- shared_parent_count
- shared_component_count
- shared_context_exact
- shared_context_family
- shared_role_transition
- shared_uncertainty_signal_count
- recent_cycle_distance
- prior_role
- current_context
- match_score
- match_reason

2. Add strict local-anchor rule.

A context-role match may become a consolidation local anchor only if at least one strong anchor holds:

- same target var and same learned signature
- shared parent/component count > 0 AND context family matches
- prior role transition exists for same nethra/context family
- same uncertainty cluster shares a specific nethra id
- recent role change near the same graph neighborhood

Weak evidence alone must not qualify:
- same broad role only
- same generic uncertainty signal
- same visible count
- same frontier kind
- same unresolved status
- old unrelated best_available record

3. Add deduplication.

Prevent match explosion:
- dedupe by (cluster_id, nethra_id, context_family, target_var)
- cap local anchors per cluster
- cap assists derived from index matches per cycle
- record suppressed duplicate count

Metrics:
- context_role_raw_matches
- context_role_deduped_matches
- context_role_matches_suppressed_weak
- context_role_matches_suppressed_duplicate
- context_role_matches_suppressed_cap
- context_role_matches_used_as_local_anchor
- context_role_anchor_precision_posthoc only in report, not runtime
- context_role_assist_pressure_per_cycle

4. Add assist attribution.

For each assist generated because of ContextRoleIndex:
- assist_kind
- cluster_id
- nethra_id
- match_reason
- whether it changed budget/probes/preservation/priority
- later local outcome if already available:
  - sentinel failure
  - revocation
  - fit churn
  - novelty persistence
  - audit count

Do not claim causality; report association only.

5. Add assist policy:
--context-role-anchor-policy off|strict|loose

Default:
  strict when context-role-index assist_feature is enabled.
  off when context-role-index is off/record.

`loose` can preserve current behavior for comparison.

6. Update reports.

scripts/summarize_context_role_index.py should print:
A. raw vs deduped matches
B. weak/duplicate/cap suppressions
C. local-anchor count
D. top match reasons
E. assist pressure per cycle
F. role transition examples
G. warning when loose matching worsens metrics

7. Add comparison run support or script output.

Compare:
- off
- record
- assist_feature + loose
- assist_feature + strict

Report:
- quality_cost delta
- iv delta
- audits delta
- revocations delta
- unique_fails delta
- regime_sentinel_fail delta
- passive stress delta
- dormant delta
- match pressure delta

8. Tests.

Add/extend tests:
- record mode remains behavior-neutral
- same broad unresolved role does not qualify as local anchor
- exact same nethra/signature/context qualifies
- shared parent + context family qualifies
- role transition qualifies
- duplicate matches are suppressed
- cap suppresses excessive anchors
- loose policy reproduces previous broad behavior
- strict policy reduces anchor count
- hidden truth/debug manifest is not read in runtime matching
- no authority/revocation/skip suppression paths are touched

Verification:
python -m pytest tests/test_nethra_reservoir.py -q
python -m pytest tests/test_uncertainty_consolidation.py -q
python -m pytest tests/test_cycle_mechanics.py -q
python -m pytest tests/test_blind_challenge.py -q

Smoke:
Run 4-way comparison on blind_challenge:

off:
  --uncertainty-consolidation shadow
  --context-role-index off

record:
  --uncertainty-consolidation shadow
  --context-role-index record

assist loose:
  --uncertainty-consolidation assist
  --uncertainty-assist-policy local_only
  --context-role-index assist_feature
  --context-role-anchor-policy loose

assist strict:
  --uncertainty-consolidation assist
  --uncertainty-assist-policy local_only
  --context-role-index assist_feature
  --context-role-anchor-policy strict

Use:
  vars=50,75,100
  cycles=3000,7500
  seeds=42,99,7,3,11,13,17,23,29,31

Expected:
- off == record
- loose reproduces current overuse/worse behavior
- strict sharply reduces match pressure
- strict must not worsen metrics relative to off as badly as loose
- if strict still worsens, disable assist_feature and keep ContextRoleIndex as record/provenance only
```

---

# Appendix B — Alternate Prompt Reference: Attention & Cost Optimizations

Use this as a reference source for later task items. Do **not** run it as the immediate next prompt because it mixes too many runtime changes and would destroy attribution while the context-role matching layer is still unresolved.

```text
# Codex Task: dreth Attention & Cost Optimizations

## Repository
https://github.com/Dooces/dreth (branch: `main`)

## Context

dreth is a causal-discovery simulation where an agent discovers hidden causal structure through earned authority. The agent observes a hidden causal world (DAG of variables with causal functions + noise), proposes hypotheses, and earns the right to certify causal relationships only by surviving structured intervention tests. The core pipeline is: Observe → Frontier → Audit → Fit → Sentinel → Certify → Predict.

Key invariants you MUST NOT violate:
1. Authority is earned, never assumed. Only the audit→fit→sentinel→certify pipeline grants authority.
2. Shadow/diagnostic layers observe but NEVER mutate agent or ledger state.
3. Provider confidence is never treated as cert authority.
4. The ledger (`NethraCertificate`, `VarNethra`) is the single source of truth.
5. `MORPHOLOGY ≠ CAUSE` — score proximity is structural observation, not causal classification.
6. Ambiguity is first-class — `TiedFrontier` must survive until regime-survival proof justifies collapse.

## Task 1: Populate `TiedFrontier.separating_probes` during `fit_var`

Problem:
`TiedFrontier.separating_probes` is always an empty tuple. The field exists but is never populated. This means sentinel selection and frontier attention have no information about which probes would discriminate between tied hypotheses.

What to do:
- In `dreth/fit.py`, after computing `near_tie_candidates_out`, identify separating probes from the intervention pool already evaluated.
- Use predictions among near-tie candidates to select top probes that split tied hypotheses.
- Store them in `diag["separating_probes"]`.
- In `dreth/agent.py`, when constructing/updating `TiedFrontier`, carry those probes into `TiedFrontier.separating_probes`.
- In `dreth/ledger.py`, verify the field is preserved.

Constraints:
- Do not change `fit_var` return signature.
- Separating probes are morphology, not causal proof.
- Do not run additional interventions for this.

## Task 2: Use `TiedFrontier` to influence frontier priority

Problem:
Variables with unresolved tied hypotheses have no audit-priority boost even though additional information would be discriminating.

What to do:
- In `dreth/agent.py`, add a small audit-priority tiebreaker for variables whose `VarNethra.tied_frontier` has more than one candidate.
- The bonus must not dominate cost-weight dispatch or consequence-tier ordering.
- If uncertainty consolidation assist already gives a budget bonus, allow the tied-frontier bonus to stack additively.

Constraints:
- This is attention routing, not authority.
- Do not change dormancy logic.
- Do not change trass-skip behavior.

## Task 3: Feed `TiedFrontier.separating_probes` into sentinel selection

Problem:
`select_var_sentinels` selects generic discriminating probes but does not prioritize probes known to separate tied hypotheses.

What to do:
- In `dreth/agent.py`, before calling `select_var_sentinels`, extract up to 2 valid `n.tied_frontier.separating_probes`.
- Prepend them to the selected sentinels, removing duplicates.
- Keep total sentinel count capped at `self.sentinel_count`; separating probes displace generic probes rather than adding cost.

Constraints:
- Do not change `select_var_sentinels` signature.
- Validate probe variable and value bounds.

## Task 4: Cheap pre-check before cascade re-audit

Problem:
Cascade invalidation queues descendants for full re-audit even when some descendants may still predict correctly.

What to do:
- In sentinel-failure handling, after `ledger.invalidate()` and before descendant re-audit queueing, perform a cheap current-state prediction pre-check for descendants.
- If predicted value matches current observed state within generous envelope margin, avoid immediate full audit and let normal sentinel/revalidation paths handle it.
- Count `cascade_precheck_skipped`.

Constraints:
- Uses current observed state only, not hidden truth.
- Does not re-certify anything.
- If unsure, queue audit.
- High-cost variables should still be re-audited.

## Task 5: Revive dormant alternatives on regime change

Problem:
`VarNethra.dormant_alternatives` stores plausible collapsed hypotheses, but regime-change re-audits do not use them.

What to do:
- In regime-change handling, for affected variables with dormant alternatives, score up to 5 dormant alternatives against current sentinel probes.
- If a dormant alternative scores perfectly or better than current fit's sentinel score, mark it as a revival candidate.
- Ensure revival candidate parents are included in `fit_var` enumeration.
- Count `dormant_revival_count`.

Constraints:
- Dormant revival does not skip audit.
- Only revive in regime-change contexts.
- Cap checked alternatives to bound cost.

## Task 6: Re-screen inert variables on regime change

Problem:
Variables screened as inert at initialization may become active under regime changes.

What to do:
- On confirmed regime promotion, collect co-failure variables.
- Re-screen up to `min(len(_inert_vars), 10)` inert vars with cheap perturbation probes.
- If an inert var affects regime co-failure variables beyond tolerance, remove it from `_inert_vars`, add it to `_live_set`, and queue audit.
- Count `regime_inert_wakeup_count`.

Constraints:
- Only on confirmed regimes.
- Cap cost.
- Uses intervention-visible effects, not hidden truth.

## Task 7: Add cumulative intervention cost tracking

Problem:
`self.total_interventions` counts probes but does not weight them by cost.

What to do:
- Add `self.total_weighted_intervention_cost` to `ChainedAgent`.
- Increment it wherever `total_interventions` increments, using var cost weight when tied to a var and `1.0` for generic probes.
- Add it to `RunAnalyzer` / `SummaryRenderer`.

Constraints:
- Diagnostic only.
- Do not change `RefitBaseline`.

## Testing

All existing tests in `tests/` must continue to pass.

Add `tests/test_attention_optimizations.py` with targeted tests:
1. `fit_var` on tied hypotheses populates valid `diag["separating_probes"]`.
2. Cascade pre-check avoids unnecessary descendant audit while still auditing the wrong descendant.
3. Dormant alternative appears in hypothesis space during re-audit after regime detection.

## Summary of files to modify
- `dreth/fit.py`
- `dreth/agent.py`
- `dreth/ledger.py`
- `dreth/summary.py`
- `tests/test_attention_optimizations.py`

## What NOT to do
- Do not change the authority pipeline.
- Do not let shadow/diagnostic layers mutate agent state.
- Do not change `fit_var` return signature.
- Do not change `select_var_sentinels` signature.
- Do not add new CLI arguments.
- Do not change `RefitBaseline`.
- Do not collapse `TiedFrontier` based on score proximity.
- Do not read hidden-world fields in agent-visible code.
```

---

## Ledger

```text
[D:O!,N,V,E,B,C,A,L,S,Q,M,P0,Cav↓,R0,NB,LG
|R:AssistAttributionRisk,ContextIndexOvermatch,RuntimeAssistHarm,RoleTransitionEvidencePartial,RecordModeClean,BroadPromptAttributionLoss
|F:strict-context-role-gating-is-next-broader-attention-optimization-is-reference-not-immediate
|P:obj✓metric✓evidence✓logic✓frame✓gate✓ledger✓]
```
