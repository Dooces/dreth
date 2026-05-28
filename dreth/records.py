from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Diagnostic-only data structures. Neither type here feeds back into any
# agent decision. Both are write-only from the agent's perspective.
#
# CycleRecord:
#   One row per cycle. Compares ground-truth (what the world actually did)
#   to agent action (what was audited vs. skipped). Used offline for
#   confusion-matrix analysis: did the agent attend to the right variable
#   when structure changed?
#
# FitDiagnostic:
#   One row per full audit. Records the hypothesis space considered, which
#   hypothesis was picked, how it ranked against ground truth, per-probe
#   arrays (actuals, predictions), tie set, and near-tie candidates.
#   Used for offline tie characterization, drift analysis, and form/pattern
#   matching across variables. The per-probe arrays are the largest fields
#   and can be bounded via agent.probe_retention_per_var.
#
# near_tie_candidates and near_tie_context_key on FitDiagnostic:
#   Written by fit_var, read by _install_var to populate TiedFrontier.
#   This is the one path where a FitDiagnostic field feeds into agent state —
#   but only to update bookkeeping, not to change fit selection.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

@dataclass
class CycleRecord:
    """One row in the agent's per-cycle history. Records what the agent
    did each cycle — what was attended/skipped — for offline diagnostics.
    Contains no ground-truth oracle fields.

    Fields:
      cycle:               cycle number (1-indexed)
      detected_drift_vars: vars the agent flagged as drifted this cycle
      skipped_vars:        vars that ran cheap-path (trass/comp/sentinel)
      fully_audited_vars:  vars that ran a full audit
      novelty_attention:   did the agent fire/sustain a novelty this cycle
      deferred_vars:       vars that needed audit but exceeded budget
    """
    cycle: int
    detected_drift_vars: Tuple[int, ...]
    skipped_vars: Tuple[int, ...]
    fully_audited_vars: Tuple[int, ...]
    novelty_attention: bool
    deferred_vars: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class FitDiagnostic:
    """One row per full audit. Records what hypothesis space the agent
    considered, what it picked, and how the picked hypothesis ranked
    against the ground-truth (source_edges, func) — used for offline analysis
    of fit quality. Diagnostics are write-only from the agent's perspective;
    they never feed back into fit selection.

    Fields:
      cycle, var:             when and what
      status_before:          VarNethra.status entering the audit
      role_before:            operation_role entering the audit
      available_source_edges:      restricted source_edge set used (empty if full)
      restricted:             True if restricted enumeration was used
      hypothesis_count:       size of enumerated hypothesis space
      best_score, second_score, margin: top two scores and gap
      best_source_edges, best_func: what the agent actually picked
      failure_class:          fit_clean / fit_with_ties / pick_indistinguishable /
                              pick_divergent / restriction_covered /
                              restriction_missing / hypothesis_absent
      status_after, role_after: state after install_var ran

      Per-probe arrays (added v29-prep): kept for downstream cross-audit /
      cross-variable / tie-characterization uses. Largest per-FitDiagnostic
      field; can be capped via agent.probe_retention_per_var if memory
      becomes an issue. Decision: stored because they enable
        - cross-audit drift analysis on identical probes
        - cross-variable correlation when probe pools overlap
        - post-hoc gate characterization for tied hypotheses
        - sentinel-selection refinement using historically-discriminating probes
        - form/pattern matching across variables
      None of these are implemented in v28 itself; the data is here for
      v29+ extensions to consume.
        probes:                 list of (iv_var, iv_val) used in audit
        actuals:                world's actual outputs at each probe
        pick_preds:             agent's chosen hypothesis predictions at each probe
        tie_set:                frozenset of (source_edges, func) hypotheses that
                                tied for rank 1 (>=1 entry; size 1 = no tie)
    """
    cycle: int
    var: int
    status_before: str
    role_before: str
    available_source_edges: Tuple[int, ...]
    restricted: bool
    hypothesis_count: int
    best_score: int
    second_score: int
    margin: int
    best_source_edges: Tuple[int, ...]
    best_func: str
    failure_class: str
    true_source_edges: Tuple[int, ...] = field(default_factory=tuple)
    true_func: str = ""
    true_present: bool = False
    true_rank: int = -1
    true_score: int = -1
    status_after: str = "unknown"
    role_after: str = "unknown"
    probes: Tuple[Tuple[int, float], ...] = field(default_factory=tuple)
    actuals: Tuple[float, ...] = field(default_factory=tuple)
    pick_preds: Tuple[float, ...] = field(default_factory=tuple)
    tie_set: FrozenSet[Tuple[Tuple[int, ...], str]] = field(default_factory=frozenset)
    # Near-tie constellation: all hypotheses within near_tie_margin probes of
    # the best score. Each entry is (source_edges, func, score). Sorted score-desc.
    # Size > 1 means multiple operationally-equivalent candidates survived.
    near_tie_candidates: Tuple = field(default_factory=tuple)
    # Hash of frozenset(available_source_edges) at audit time. Stored so the agent
    # can detect when the restriction context has changed and the old frontier
    # is stale.
    near_tie_context_key: int = 0
