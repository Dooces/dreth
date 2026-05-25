from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Stage 3A: shadow learned residual predictor.
#
# INVARIANT (Stage 3A — shadow mode only):
#   ShadowLearnedResidualPredictor may observe, predict, and report diagnostics.
#   It must NOT:
#     - change run behavior
#     - issue certs
#     - mutate ledger certs or route_certs
#     - mark tareth / trass
#     - authorize skips
#     - bypass sentinels
#     - replace SymbolicResidualPredictor in the passive gate
#
# It holds no ledger reference by design. The only learning inputs are:
#   func, residual, tolerance — all observable from the passive monitoring path.
#   No world.parents, world.funcs, or hidden_log access.
#
# Stage 3B will promote the learned predictor to the passive gate only after
# shadow shadow-accuracy is measured here first.
# ─────────────────────────────────────────────────────────────────────────────

import collections
import dataclasses
import math
from typing import Dict, List, Optional, Tuple


# ── Feature / label dataclasses (for future batch training) ──────────────────

@dataclasses.dataclass
class ResidualFeatureVector:
    """Feature vector for one var's residual observation."""
    var: int
    cycle: int
    parents: Tuple[int, ...]
    func: str
    parent_vals: Tuple[float, ...]
    actual: float
    tolerance: float
    consequence_tier: int
    full_audits: int
    sentinel_count: int
    cert_age: int


@dataclasses.dataclass
class ResidualLabel:
    """Ground-truth label for a residual observation."""
    symbolic_ok: bool
    symbolic_stressed: bool
    active_sentinel_failed: Optional[bool]
    residual: float


# ── Online rolling statistics ─────────────────────────────────────────────────

class _RollingStats:
    """True sliding-window mean/variance for residual tracking.

    Only the most recent `window` samples contribute to mean, std, and
    upper_bound. min_samples checks use the current window length (n), not
    a cumulative count, so drift-adaptive decisions reflect recent history.

    n_total counts all samples ever seen (useful for diagnostics).
    """

    def __init__(self, window: int = 200) -> None:
        self._n_total: int = 0
        self._window_samples: collections.deque = collections.deque(maxlen=window)
        self._window: int = window

    def update(self, x: float) -> None:
        self._n_total += 1
        self._window_samples.append(x)  # deque auto-evicts oldest when full

    @property
    def n(self) -> int:
        """Current window size — use this for min_samples checks."""
        return len(self._window_samples)

    @property
    def n_total(self) -> int:
        """Total samples seen across all history."""
        return self._n_total

    @property
    def mean(self) -> float:
        n = len(self._window_samples)
        if n == 0:
            return 0.0
        return sum(self._window_samples) / n

    @property
    def std(self) -> float:
        n = len(self._window_samples)
        if n < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self._window_samples) / n)

    @property
    def upper_bound(self) -> float:
        """Conservative upper estimate: mean + 2 * std."""
        return self.mean + 2.0 * self.std


# ── Calibrator ────────────────────────────────────────────────────────────────

class OnlineResidualCalibrator:
    """Lightweight per-func and global rolling residual statistics.

    Predicts ok / stressed conservatively:
      ok only if estimated residual upper bound <= tolerance * conservative_factor
      stressed otherwise.

    Falls back from per-func to global stats when per-func has fewer than
    min_samples observations. Returns stressed when both are insufficient.

    No cert authority. Does not read world.parents, world.funcs, or hidden_log.
    """

    def __init__(
        self,
        conservative_factor: float = 0.4,
        min_samples: int = 50,
        window: int = 200,
    ) -> None:
        self.conservative_factor = conservative_factor
        self.min_samples = min_samples
        self._window = window
        self._per_func: Dict[str, _RollingStats] = {}
        self._global: _RollingStats = _RollingStats(window=window)

    def update(self, func: str, residual: float) -> None:
        """Update stats with an observed symbolic residual for a given func."""
        if func not in self._per_func:
            self._per_func[func] = _RollingStats(window=self._window)
        self._per_func[func].update(residual)
        self._global.update(residual)

    def predict(
        self,
        func: str,
        tolerance: float,
    ) -> Tuple[bool, bool, bool]:
        """Return (ok, stressed, insufficient_samples).

        ok=True only if upper-bound residual estimate <= tolerance * conservative_factor.
        insufficient_samples=True when not enough data for either per-func or global.
        When insufficient, returns (False, True, True) — conservative default.
        """
        per_func = self._per_func.get(func)

        if per_func is not None and per_func.n >= self.min_samples:
            stats = per_func
            insufficient = False
        elif self._global.n >= self.min_samples:
            stats = self._global
            insufficient = False
        else:
            return False, True, True

        threshold = tolerance * self.conservative_factor
        upper = stats.upper_bound
        ok = upper <= threshold
        return ok, not ok, insufficient


# ── Shadow predictor ──────────────────────────────────────────────────────────

class ShadowLearnedResidualPredictor:
    """Shadow-mode learned residual predictor (Stage 3A).

    Wraps OnlineResidualCalibrator to provide a ResidualPrediction-compatible
    interface, but predictions are NEVER used for gating.

    Design invariants:
      - No ledger reference (cannot issue certs or mutate state).
      - predict_shadow() is the only output surface; its result is discarded
        by the agent — only counters and comparisons are kept.
      - observe() updates the calibrator from symbolic residual only.
    """

    def __init__(self, calibrator: Optional[OnlineResidualCalibrator] = None) -> None:
        if calibrator is None:
            calibrator = OnlineResidualCalibrator()
        self._calibrator = calibrator

        # Per-call shadow counters
        self.calls: int = 0
        self.ok: int = 0
        self.stressed: int = 0
        self.insufficient_samples: int = 0
        # Set to True after each predict_shadow call when samples were insufficient
        self._last_call_insufficient: bool = False

    def predict_shadow(
        self,
        var: int,
        func: str,
        tolerance: float,
    ) -> "ResidualPrediction":  # type: ignore[name-defined]
        """Return shadow ResidualPrediction — NEVER used for gating.

        Sets self._last_call_insufficient for easy agent-side accounting.
        residual/predicted/actual fields are nan (shadow has no world access).
        """
        from .hybrid import ResidualPrediction

        self.calls += 1
        is_ok, is_stressed, insufficient = self._calibrator.predict(func, tolerance)
        self._last_call_insufficient = insufficient

        if insufficient:
            self.insufficient_samples += 1
            self.stressed += 1
        elif is_ok:
            self.ok += 1
        else:
            self.stressed += 1

        return ResidualPrediction(
            var=var, ok=is_ok, stressed=is_stressed,
            residual=float("nan"), predicted=float("nan"), actual=float("nan"),
        )

    def observe(self, func: str, residual: float) -> None:
        """Update calibrator with an observed symbolic residual.

        Called by ChainedAgent after the symbolic path has computed the actual
        residual. Does not read world.parents, world.funcs, or hidden_log.
        Only trains on: func + scalar residual value.
        """
        self._calibrator.update(func, residual)
