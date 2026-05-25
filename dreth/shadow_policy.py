"""
ShadowPolicySelector — diagnostic-only offline policy predictor.

Observes run-local diagnostics from completed RunResult objects and predicts
which provider policy would have had lower quality_cost. The prediction uses
only observable diagnostics; the schedule label is never visible at prediction
time.

INVARIANTS (enforced by design, not runtime):
  - ShadowPolicySelector does not change the active policy.
  - It does not touch ChainedAgent, ledger, cert, or sentinel state.
  - The schedule label may be supplied post-hoc to compute error modes, but
    it is NOT accessible inside predict().
  - No runtime policy switching is implemented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TextIO

logger = logging.getLogger(__name__)

POLICIES = (
    "sensitivity/none",
    "history/history",
    "history_rescue/history_rescue",
)

# Regime-fail rate above which sensitivity/none is predicted to win.
# Derived from empirical data: false_trass always produces 0 fails;
# regime_switch produces ≥100 fails even at 5 000 cycles/50 vars.
# 0.01 (1 fail per 100 cycles) sits in the gap with a large margin.
_REGIME_FAIL_RATE_THRESHOLD = 0.01

# Under a false_trass-like regime, high unique-fail rates at large n_vars
# make sensitivity/none competitive again. Empirically: 0 at 50-75 vars,
# low but non-zero at 100 vars. Threshold chosen conservatively.
_HIGH_UNIQUE_FAIL_RATE = 5e-4  # per var-cycle


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticFeatures:
    """Observable run-local diagnostics. No schedule label included.

    All fields are extractable from RunResult without oracle access.
    Rates are computed by the caller so this dataclass stays unit-agnostic.
    """
    revocations: int
    full_audits: int
    unique_fails: int
    regime_sentinel_fails: int
    regime_no_sentinel: int
    parent_rank_mean: float
    probe_no_effect: int
    probe_improved: int
    passive_stress_count: int
    active_composites: int
    composite_components: int
    cycles: int
    n_vars: int

    @property
    def regime_fail_rate(self) -> float:
        return self.regime_sentinel_fails / max(1, self.cycles)

    @property
    def no_sentinel_rate(self) -> float:
        return self.regime_no_sentinel / max(1, self.cycles)

    @property
    def unique_fail_rate(self) -> float:
        return self.unique_fails / max(1, self.cycles * self.n_vars)

    @property
    def passive_stress_rate(self) -> float:
        return self.passive_stress_count / max(1, self.cycles * self.n_vars)

    @property
    def probe_no_effect_rate(self) -> float:
        total = self.probe_no_effect + self.probe_improved
        return self.probe_no_effect / total if total else 0.0


# ---------------------------------------------------------------------------
# Per-prediction record
# ---------------------------------------------------------------------------

@dataclass
class ShadowPrediction:
    """One shadow prediction and its post-hoc comparison to the actual winner."""
    predicted_policy: str
    actual_best_policy: str
    correct: bool
    false_switch_to_history_rescue_under_regime_switch: bool
    missed_history_rescue_under_false_trass: bool
    features: DiagnosticFeatures


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class ShadowPolicySelector:
    """Predicts which provider policy would have had lower quality_cost.

    Prediction uses only observable diagnostics. The schedule label is NEVER
    passed into predict(); it may be supplied to observe() solely to compute
    post-hoc error-mode counts.

    Usage:
        sel = ShadowPolicySelector()
        features = DiagnosticFeatures(...)          # extracted from RunResult
        pred = sel.observe(features, actual_best_policy="history_rescue/history_rescue",
                           schedule="false_trass")  # schedule only for error-mode bookkeeping
        summary = sel.summary()
    """

    def __init__(self) -> None:
        self._predictions: List[ShadowPrediction] = []

    # ------------------------------------------------------------------
    # Prediction (no schedule label)
    # ------------------------------------------------------------------

    def predict(self, features: DiagnosticFeatures) -> str:
        """Predict best policy from observable diagnostics.

        Decision tree (all inputs are observable at run time):

        1. High regime_sentinel_fail_rate → sensitivity/none.
           Regime switching makes history-aware providers expensive.

        2. Low regime_sentinel_fail_rate AND high no_sentinel_rate →
           sensitivity/none. World is too unstable for history to help.

        3. Low fail rate AND high passive stress → sensitivity/none.
           Stress without regime signal suggests fast local drift; history
           providers add audit overhead without benefit.

        4. High unique_fail_rate at large n_vars → sensitivity/none.
           Unique failures accumulate; history_rescue's revocation cost
           outweighs its IV reduction.

        5. Otherwise → history_rescue/history_rescue.
           Stable low-regime-fail worlds (false_trass-like) where history
           context reduces intervention volume.
        """
        if features.regime_fail_rate > _REGIME_FAIL_RATE_THRESHOLD:
            return "sensitivity/none"

        if features.no_sentinel_rate > 0.5:
            return "sensitivity/none"

        if features.passive_stress_rate > 0.02 and features.regime_fail_rate < 1e-6:
            return "sensitivity/none"

        if features.unique_fail_rate > _HIGH_UNIQUE_FAIL_RATE and features.n_vars >= 100:
            return "sensitivity/none"

        return "history_rescue/history_rescue"

    # ------------------------------------------------------------------
    # Observation and logging
    # ------------------------------------------------------------------

    def observe(
        self,
        features: DiagnosticFeatures,
        actual_best_policy: str,
        schedule: Optional[str] = None,
    ) -> ShadowPrediction:
        """Record one prediction and compare to the actual best policy.

        The schedule label, if provided, is used only to classify error modes
        after the prediction is already made. It does not influence predict().
        """
        if actual_best_policy not in POLICIES:
            raise ValueError(
                f"actual_best_policy {actual_best_policy!r} not in known policies"
            )

        predicted = self.predict(features)
        correct = predicted == actual_best_policy

        false_switch = (
            schedule == "regime_switch"
            and predicted == "history_rescue/history_rescue"
            and actual_best_policy != "history_rescue/history_rescue"
        )
        missed_rescue = (
            schedule == "false_trass"
            and predicted != "history_rescue/history_rescue"
            and actual_best_policy == "history_rescue/history_rescue"
        )

        pred = ShadowPrediction(
            predicted_policy=predicted,
            actual_best_policy=actual_best_policy,
            correct=correct,
            false_switch_to_history_rescue_under_regime_switch=false_switch,
            missed_history_rescue_under_false_trass=missed_rescue,
            features=features,
        )
        self._predictions.append(pred)
        logger.debug(
            "shadow_policy predict=%s actual=%s correct=%s",
            predicted, actual_best_policy, correct,
        )
        return pred

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return logged metrics. Empty dict if no predictions recorded."""
        n = len(self._predictions)
        if n == 0:
            return {}

        n_correct = sum(p.correct for p in self._predictions)
        false_switches = sum(
            p.false_switch_to_history_rescue_under_regime_switch
            for p in self._predictions
        )
        missed_rescues = sum(
            p.missed_history_rescue_under_false_trass
            for p in self._predictions
        )

        predicted_counts: Dict[str, int] = {}
        actual_counts: Dict[str, int] = {}
        for p in self._predictions:
            predicted_counts[p.predicted_policy] = (
                predicted_counts.get(p.predicted_policy, 0) + 1
            )
            actual_counts[p.actual_best_policy] = (
                actual_counts.get(p.actual_best_policy, 0) + 1
            )

        return {
            "n_predictions": n,
            "accuracy": n_correct / n,
            "false_switch_to_history_rescue_under_regime_switch": false_switches,
            "missed_history_rescue_under_false_trass": missed_rescues,
            "predicted_policy": predicted_counts,
            "actual_best_policy": actual_counts,
        }

    def log_summary(self, file: Optional[TextIO] = None) -> None:
        """Print summary to file (defaults to stdout). Diagnostic only."""
        s = self.summary()
        if not s:
            _p("shadow_policy: no predictions recorded", file)
            return
        _p("── shadow policy selector (diagnostic only) ───────────────────", file)
        _p(f"  n_predictions : {s['n_predictions']}", file)
        _p(f"  accuracy      : {s['accuracy']:.3f}", file)
        _p(
            f"  false_switch_to_history_rescue_under_regime_switch : "
            f"{s['false_switch_to_history_rescue_under_regime_switch']}",
            file,
        )
        _p(
            f"  missed_history_rescue_under_false_trass            : "
            f"{s['missed_history_rescue_under_false_trass']}",
            file,
        )
        _p("  predicted_policy counts:", file)
        for policy, count in sorted(s["predicted_policy"].items()):
            _p(f"    {policy:<36} {count}", file)
        _p("  actual_best_policy counts:", file)
        for policy, count in sorted(s["actual_best_policy"].items()):
            _p(f"    {policy:<36} {count}", file)


def _p(msg: str, file: Optional[TextIO]) -> None:
    if file is None:
        import sys
        file = sys.stdout
    print(msg, file=file)


# ---------------------------------------------------------------------------
# Helper: extract features from a RunResult without importing batch_run
# ---------------------------------------------------------------------------

def features_from_run_result(run_result: Any) -> DiagnosticFeatures:
    """Extract DiagnosticFeatures from a RunResult (duck-typed to avoid circular import).

    This is the integration glue between batch_run.py's RunResult and
    ShadowPolicySelector. It does not import batch_run; RunResult fields are
    accessed by attribute name.
    """
    arch = run_result.arch
    return DiagnosticFeatures(
        revocations=sum(arch.revoked_by_dist.values()),
        full_audits=run_result.full_audits,
        unique_fails=arch.total_unique_failures,
        regime_sentinel_fails=arch.regime_sentinel_fails,
        regime_no_sentinel=arch.regime_no_sentinel,
        parent_rank_mean=arch.parent_proposal_rank_mean,
        probe_no_effect=arch.provider_probe_no_effect_count,
        probe_improved=arch.provider_probe_improved_margin_count,
        passive_stress_count=arch.passive_stress_count,
        active_composites=arch.active_composites,
        composite_components=arch.composite_components,
        cycles=run_result.recorded_cycles,
        n_vars=run_result.config.n_vars,
    )
