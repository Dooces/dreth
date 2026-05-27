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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .hybrid import ResidualPrediction


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


# ── Shadow predictor-key authority diagnostics ───────────────────────────────

@dataclasses.dataclass
class ShadowResidualKeyRecord:
    """Shadow-only safety counters for one learned-residual prediction key."""
    key: Any
    key_type: str
    seen: int = 0
    ok_count: int = 0
    stressed_count: int = 0
    false_ok_vs_symbolic: int = 0
    false_ok_vs_active: int = 0
    would_save_iv: int = 0
    would_miss_symbolic_stress: int = 0
    would_miss_active_failure: int = 0
    clean_ok_streak: int = 0
    revoked: bool = False
    revoked_by: Optional[str] = None
    first_seen_cycle: Optional[int] = None
    last_seen_cycle: Optional[int] = None


class ShadowResidualKeyAuthority:
    """Dreth-style revocable safety ledger for shadow residual predictor keys.

    This object is diagnostic only. It has no ledger/world reference, issues no
    NethraCertificate, and its records are never consulted by the agent's gating,
    sentinel, audit, skip, or repair paths.
    """

    KEY_FUNC_VAR = "func_var"
    KEY_FUNC_TIER_PARENTCOUNT = "func_tier_parentcount"
    KEY_FUNC_TIER = "func_tier"
    KEY_FUNC = "func"
    KEY_GLOBAL = "global"
    KEY_INSUFFICIENT = "insufficient"

    def __init__(
        self,
        min_key_ok: int = 100,
        min_clean_streak: int = 100,
        symbolic_false_ok_tolerance: int = 0,
    ) -> None:
        self.min_key_ok = min_key_ok
        self.min_clean_streak = min_clean_streak
        self.symbolic_false_ok_tolerance = symbolic_false_ok_tolerance
        self._records: Dict[Any, ShadowResidualKeyRecord] = {}

    @staticmethod
    def normalize_key_type(key: Any, key_type: Optional[str] = None) -> str:
        raw = key_type
        if raw is None and isinstance(key, tuple) and key:
            raw = str(key[0])
        if raw == FeatureConditionedResidualCalibrator.KEY_FUNC_TIER_PARENT:
            return ShadowResidualKeyAuthority.KEY_FUNC_TIER_PARENTCOUNT
        if raw in {
            ShadowResidualKeyAuthority.KEY_FUNC_VAR,
            ShadowResidualKeyAuthority.KEY_FUNC_TIER_PARENTCOUNT,
            ShadowResidualKeyAuthority.KEY_FUNC_TIER,
            ShadowResidualKeyAuthority.KEY_FUNC,
            ShadowResidualKeyAuthority.KEY_GLOBAL,
            ShadowResidualKeyAuthority.KEY_INSUFFICIENT,
            "rolling_func",
        }:
            return raw
        return ShadowResidualKeyAuthority.KEY_INSUFFICIENT

    @staticmethod
    def normalize_key(key: Any, key_type: Optional[str] = None) -> Any:
        if key is None:
            return ShadowResidualKeyAuthority.KEY_INSUFFICIENT
        return key

    def _record_for(self, key: Any, key_type: Optional[str], cycle: int) -> ShadowResidualKeyRecord:
        norm_key = self.normalize_key(key, key_type)
        norm_type = self.normalize_key_type(norm_key, key_type)
        rec = self._records.get(norm_key)
        if rec is None:
            rec = ShadowResidualKeyRecord(
                key=norm_key,
                key_type=norm_type,
                first_seen_cycle=cycle,
                last_seen_cycle=cycle,
            )
            self._records[norm_key] = rec
        else:
            rec.last_seen_cycle = cycle
        return rec

    def _maybe_revoke_symbolic(self, rec: ShadowResidualKeyRecord) -> None:
        if (
            not rec.revoked
            and rec.false_ok_vs_symbolic > self.symbolic_false_ok_tolerance
        ):
            rec.revoked = True
            rec.revoked_by = "false_ok_symbolic"

    def record_prediction(
        self,
        *,
        key: Any,
        key_type: Optional[str],
        shadow_ok: bool,
        shadow_stressed: bool,
        symbolic_ok: bool,
        symbolic_stressed: bool,
        would_save_iv: int,
        would_miss_symbolic_stress: int,
        cycle: int,
    ) -> ShadowResidualKeyRecord:
        """Record one shadow prediction against the symbolic reference."""
        rec = self._record_for(key, key_type, cycle)
        rec.seen += 1
        if shadow_ok:
            rec.ok_count += 1
            rec.would_save_iv += would_save_iv
            rec.would_miss_symbolic_stress += would_miss_symbolic_stress
            if symbolic_stressed:
                rec.false_ok_vs_symbolic += 1
                rec.clean_ok_streak = 0
            elif symbolic_ok:
                rec.clean_ok_streak += 1
            else:
                rec.clean_ok_streak = 0
        if shadow_stressed:
            rec.stressed_count += 1
            rec.clean_ok_streak = 0

        self._maybe_revoke_symbolic(rec)
        return rec

    def record_active_failure(
        self,
        *,
        key: Any,
        key_type: Optional[str],
        would_miss_active_failure: int,
        cycle: int,
    ) -> ShadowResidualKeyRecord:
        """Record a later active-sentinel failure for a shadow-ok key."""
        rec = self._record_for(key, key_type, cycle)
        rec.false_ok_vs_active += 1
        rec.would_miss_active_failure += would_miss_active_failure
        rec.clean_ok_streak = 0
        rec.revoked = True
        rec.revoked_by = "false_ok_active"
        return rec

    def candidate_safe(self, rec: ShadowResidualKeyRecord) -> bool:
        return (
            rec.ok_count >= self.min_key_ok
            and rec.false_ok_vs_active == 0
            and rec.false_ok_vs_symbolic == 0
            and rec.clean_ok_streak >= self.min_clean_streak
        )

    def records(self) -> List[ShadowResidualKeyRecord]:
        return list(self._records.values())

    def record_dicts(self) -> List[Dict[str, Any]]:
        return [self._record_to_dict(rec) for rec in self.records()]

    def _record_to_dict(self, rec: ShadowResidualKeyRecord) -> Dict[str, Any]:
        return {
            "key": repr(rec.key),
            "key_type": rec.key_type,
            "seen": rec.seen,
            "ok_count": rec.ok_count,
            "stressed_count": rec.stressed_count,
            "false_ok_vs_symbolic": rec.false_ok_vs_symbolic,
            "false_ok_vs_active": rec.false_ok_vs_active,
            "would_save_iv": rec.would_save_iv,
            "would_miss_symbolic_stress": rec.would_miss_symbolic_stress,
            "would_miss_active_failure": rec.would_miss_active_failure,
            "clean_ok_streak": rec.clean_ok_streak,
            "revoked": rec.revoked,
            "revoked_by": rec.revoked_by,
            "candidate_safe": self.candidate_safe(rec),
            "first_seen_cycle": rec.first_seen_cycle,
            "last_seen_cycle": rec.last_seen_cycle,
        }

    def summary(self, top_n: int = 10) -> Dict[str, Any]:
        records = self.records()
        candidates = [r for r in records if self.candidate_safe(r)]
        revoked = [r for r in records if r.revoked]
        revoked_active = [r for r in revoked if r.revoked_by == "false_ok_active"]
        revoked_symbolic = [r for r in revoked if r.revoked_by == "false_ok_symbolic"]
        top_revoked_active = sorted(
            [r for r in revoked if r.false_ok_vs_active > 0],
            key=lambda r: (
                -r.false_ok_vs_active,
                -r.would_miss_active_failure,
                repr(r.key),
            ),
        )[:top_n]
        top_candidate_safe = sorted(
            candidates,
            key=lambda r: (-r.would_save_iv, -r.ok_count, repr(r.key)),
        )[:top_n]
        return {
            "shadow_key_total": len(records),
            "shadow_key_candidate_safe": len(candidates),
            "shadow_key_revoked": len(revoked),
            "shadow_key_revoked_active": len(revoked_active),
            "shadow_key_revoked_symbolic": len(revoked_symbolic),
            "shadow_key_ok_total": sum(r.ok_count for r in records),
            "shadow_key_false_ok_active_total": sum(r.false_ok_vs_active for r in records),
            "shadow_key_false_ok_symbolic_total": sum(r.false_ok_vs_symbolic for r in records),
            "shadow_key_safe_would_save_iv": sum(r.would_save_iv for r in candidates),
            "shadow_key_revoked_would_miss_active_failure": sum(
                r.would_miss_active_failure for r in revoked
            ),
            "records": self.record_dicts(),
            "top_revoked_active": [self._record_to_dict(r) for r in top_revoked_active],
            "top_candidate_safe": [self._record_to_dict(r) for r in top_candidate_safe],
        }

    def report_lines(self, top_n: int = 10) -> List[str]:
        s = self.summary(top_n=top_n)
        lines = [
            "── shadow residual key authority ──",
            f"keys={s['shadow_key_total']}",
            f"candidate_safe={s['shadow_key_candidate_safe']}",
            f"revoked={s['shadow_key_revoked']}",
            f"revoked_active={s['shadow_key_revoked_active']}",
            f"revoked_symbolic={s['shadow_key_revoked_symbolic']}",
            f"ok_total={s['shadow_key_ok_total']}",
            f"false_ok_active_total={s['shadow_key_false_ok_active_total']}",
            f"false_ok_symbolic_total={s['shadow_key_false_ok_symbolic_total']}",
            "top revoked keys by false_ok_active:",
        ]
        for rec in s["top_revoked_active"]:
            lines.append(
                f"  key_type={rec['key_type']} key={rec['key']} "
                f"false_ok_active={rec['false_ok_vs_active']} "
                f"would_miss_active_failure={rec['would_miss_active_failure']}"
            )
        lines.append("top candidate safe keys by would_save_iv:")
        for rec in s["top_candidate_safe"]:
            lines.append(
                f"  key_type={rec['key_type']} key={rec['key']} "
                f"ok_count={rec['ok_count']} would_save_iv={rec['would_save_iv']}"
            )
        return lines


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


# ── Feature-conditioned calibrator ───────────────────────────────────────────

class FeatureConditionedResidualCalibrator:
    """Multi-key backoff residual calibrator conditioned on ResidualFeatureVector fields.

    Maintains rolling stats for five key levels, ordered most-specific to least:
      1. ("func_var",  func, var)
      2. ("func_tier_parent", func, consequence_tier, parent_count)
      3. ("func_tier", func, consequence_tier)
      4. ("func",      func)
      5. ("global",)

    Prediction picks the most-specific key with >= min_samples observations.
    Returns stressed/insufficient when no key qualifies.

    No ledger reference. Receives only ResidualFeatureVector + scalar residual.
    """

    KEY_FUNC_VAR        = "func_var"
    KEY_FUNC_TIER_PARENT = "func_tier_parent"
    KEY_FUNC_TIER       = "func_tier"
    KEY_FUNC            = "func"
    KEY_GLOBAL          = "global"

    def __init__(
        self,
        conservative_factor: float = 0.4,
        min_samples: int = 50,
        window: int = 200,
    ) -> None:
        self.conservative_factor = conservative_factor
        self.min_samples = min_samples
        self._window = window
        self._stats: Dict[Tuple, _RollingStats] = {}

        # Last-prediction metadata (set by predict(); read by predictor for forwarding)
        self._last_key_used: Optional[Tuple] = None
        self._last_key_n: int = 0
        self._last_upper_bound: float = float("nan")
        self._last_threshold: float = float("nan")
        self._last_derived_features: Dict[str, Any] = {}

    def _get_or_create(self, key: Tuple) -> _RollingStats:
        if key not in self._stats:
            self._stats[key] = _RollingStats(window=self._window)
        return self._stats[key]

    @staticmethod
    def _count_bucket(value: int) -> str:
        if value <= 0:
            return "0"
        if value == 1:
            return "1"
        if value <= 3:
            return "2-3"
        if value <= 7:
            return "4-7"
        return "8+"

    @staticmethod
    def _age_bucket(value: int) -> str:
        if value <= 0:
            return "0"
        if value <= 10:
            return "1-10"
        if value <= 50:
            return "11-50"
        if value <= 200:
            return "51-200"
        return "201+"

    @staticmethod
    def _var_bucket(var: int) -> str:
        return f"{(max(0, var) // 10) * 10}-{(max(0, var) // 10) * 10 + 9}"

    def derived_features(self, fv: "ResidualFeatureVector") -> Dict[str, Any]:
        """Return feature keys derived only from ResidualFeatureVector fields."""
        return {
            "parent_count": len(fv.parents),
            "consequence_tier": fv.consequence_tier,
            "sentinel_count_bucket": self._count_bucket(fv.sentinel_count),
            "cert_age_bucket": self._age_bucket(fv.cert_age),
            "full_audits_bucket": self._count_bucket(fv.full_audits),
            "var": fv.var,
            "var_bucket": self._var_bucket(fv.var),
            "func": fv.func,
        }

    def _make_keys(self, fv: "ResidualFeatureVector") -> List[Tuple]:
        derived = self.derived_features(fv)
        parent_count = derived["parent_count"]
        tier = derived["consequence_tier"]
        self._last_derived_features = derived
        return [
            (self.KEY_FUNC_VAR,         fv.func, fv.var),
            (self.KEY_FUNC_TIER_PARENT, fv.func, tier, parent_count),
            (self.KEY_FUNC_TIER,        fv.func, tier),
            (self.KEY_FUNC,             fv.func),
            (self.KEY_GLOBAL,),
        ]

    def update(self, fv: "ResidualFeatureVector", residual: float) -> None:
        """Update rolling stats for all key levels from a single observation."""
        for key in self._make_keys(fv):
            self._get_or_create(key).update(residual)

    def predict(
        self,
        fv: "ResidualFeatureVector",
        tolerance: float,
    ) -> Tuple[bool, bool, bool]:
        """Return (ok, stressed, insufficient).

        Tries keys from most- to least-specific; uses the first with >= min_samples.
        Sets self._last_key_used etc. for predictor metadata forwarding.
        When no key qualifies, returns conservative (False, True, True).
        """
        threshold = tolerance * self.conservative_factor
        for key in self._make_keys(fv):
            stats = self._stats.get(key)
            if stats is None or stats.n < self.min_samples:
                continue
            upper = stats.upper_bound
            ok = upper <= threshold
            self._last_key_used = key
            self._last_key_n = stats.n
            self._last_upper_bound = upper
            self._last_threshold = threshold
            return ok, not ok, False

        self._last_key_used = None
        self._last_key_n = 0
        self._last_upper_bound = float("nan")
        self._last_threshold = threshold
        return False, True, True


# ── Shadow predictor ──────────────────────────────────────────────────────────

class ShadowLearnedResidualPredictor:
    """Shadow-mode learned residual predictor (Stage 3A).

    Wraps either OnlineResidualCalibrator (mode="rolling") or
    FeatureConditionedResidualCalibrator (mode="feature").  Predictions are
    NEVER used for gating.

    Design invariants:
      - No ledger reference (cannot issue certs or mutate state).
      - predict_shadow() is the only output surface; its result is discarded
        by the agent — only counters and comparisons are kept.
      - observe() updates the calibrator from symbolic residual only.
    """

    def __init__(
        self,
        calibrator: Optional[Union[str, Any]] = None,  # mode string or calibrator instance
    ) -> None:
        if calibrator is None or calibrator == "rolling":
            calibrator = OnlineResidualCalibrator()
        elif calibrator == "feature":
            calibrator = FeatureConditionedResidualCalibrator()
        elif isinstance(calibrator, str):
            raise ValueError(f"unknown shadow residual calibrator mode: {calibrator!r}")
        self._calibrator = calibrator
        self._is_feature_mode: bool = isinstance(calibrator, FeatureConditionedResidualCalibrator)

        # Per-call shadow counters
        self.calls: int = 0
        self.ok: int = 0
        self.stressed: int = 0
        self.insufficient_samples: int = 0
        # Set to True after each predict_shadow call when samples were insufficient
        self._last_call_insufficient: bool = False

        # Prediction metadata (diagnostic; set each call)
        # feature mode: forwarded from calibrator; rolling mode: None/nan
        self._last_key_used: Optional[Tuple] = None
        self._last_key_type: Optional[str] = None
        self._last_key_sample_count: int = 0
        self._last_upper_bound: float = float("nan")
        self._last_threshold: float = float("nan")

    @property
    def last_prediction_metadata(self) -> Dict[str, Any]:
        """Diagnostic metadata from the most recent shadow prediction."""
        return {
            "key_used": self._last_key_used,
            "key_type": self._last_key_type,
            "key_sample_count": self._last_key_sample_count,
            "upper_bound": self._last_upper_bound,
            "threshold": self._last_threshold,
        }

    def predict_shadow(
        self,
        var: int,
        func: str,
        tolerance: float,
        fv: Optional["ResidualFeatureVector"] = None,
    ) -> "ResidualPrediction":
        """Return shadow ResidualPrediction — NEVER used for gating.

        In feature mode, uses fv for key-conditioned prediction.
        In rolling mode, uses func + tolerance (fv ignored).
        Sets self._last_call_insufficient and metadata for agent-side accounting.
        residual/predicted/actual fields are nan (shadow has no world access).
        """
        from .hybrid import ResidualPrediction

        self.calls += 1

        if self._is_feature_mode:
            if fv is None:
                is_ok, is_stressed, insufficient = False, True, True
                self._last_key_used = None
                self._last_key_type = ShadowResidualKeyAuthority.KEY_INSUFFICIENT
                self._last_key_sample_count = 0
                self._last_upper_bound = float("nan")
                self._last_threshold = tolerance * self._calibrator.conservative_factor  # type: ignore[union-attr]
            else:
                is_ok, is_stressed, insufficient = self._calibrator.predict(fv, tolerance)  # type: ignore[union-attr]
                self._last_key_used = self._calibrator._last_key_used  # type: ignore[union-attr]
                self._last_key_type = ShadowResidualKeyAuthority.normalize_key_type(
                    self._last_key_used
                )
                self._last_key_sample_count = self._calibrator._last_key_n  # type: ignore[union-attr]
                self._last_upper_bound = self._calibrator._last_upper_bound  # type: ignore[union-attr]
                self._last_threshold = self._calibrator._last_threshold  # type: ignore[union-attr]
        else:
            is_ok, is_stressed, insufficient = self._calibrator.predict(func, tolerance)
            self._last_key_used = None
            self._last_key_type = None
            self._last_key_sample_count = 0
            self._last_upper_bound = float("nan")
            self._last_threshold = float("nan")

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

    def observe(
        self,
        func: str,
        residual: float,
        fv: Optional["ResidualFeatureVector"] = None,
    ) -> None:
        """Update calibrator with an observed symbolic residual.

        In feature mode, uses fv for key-conditioned update.
        In rolling mode, uses func (fv ignored).
        Does not read world.parents, world.funcs, or hidden_log.
        """
        if self._is_feature_mode and fv is not None:
            self._calibrator.update(fv, residual)  # type: ignore[union-attr]
        else:
            self._calibrator.update(func, residual)
