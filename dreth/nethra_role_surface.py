from __future__ import annotations

"""Nethra role-surface model: record-only operating surfaces and residual buckets.

Step 1: passive storage, metrics, and export only. No runtime behavior changes.
Surfaces record what role a nethra occupies in a given context and what projection
and residual-collection permissions follow from that role.  Nothing here may alter
audit, skip, probe, repair, ranking, or projection runtime behaviour.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ContextKey = str
NethraId = str

_MAX_PRESSURE: float = 20.0
_PRESSURE_PER_RESIDUAL: float = 0.5
_PRESSURE_DECAY_PER_OP: float = 0.25
_MAX_REPRESENTATIVE_EXAMPLES: int = 20
_MAX_TRANSITIONS: int = 2000
_MAX_REGIME_CANDIDATES: int = 500

_HIDDEN_TRUTH_FIELDS: frozenset[str] = frozenset({
    "truth_parents",
    "truth_func",
    "truth_delayed_parents",
    "truth_latents",
    "debug_blind_challenge_manifest",
})

_ROLE_ORDER: dict[str, int] = {
    "blocked": 0,
    "trass": 1,
    "contested": 1,
    "unresolved": 2,
    "best_available": 3,
    "tareth": 4,
}


@dataclass
class NethraRoleSurface:
    """Operating surface record for a nethra in a specific context."""

    nethra_id: str
    context_key: str
    context_family: str
    role_state: Literal[
        "tareth",
        "trass",
        "best_available",
        "unresolved",
        "blocked",
        "contested",
    ]
    load_bearing_score: float = 0.0
    residual_score: float = 0.0
    projection_allowed: bool = False
    residual_collection_allowed: bool = False
    composition_allowed: bool = False
    last_updated_cycle: int = 0
    support_count: int = 0
    failure_count: int = 0
    use_count: int = 0
    helped_count: int = 0
    hurt_count: int = 0


@dataclass
class ResidualBucket:
    """Residual accumulation record for a nethra in a specific context."""

    nethra_id: str
    context_key: str
    residual_count: int = 0
    unresolved_count: int = 0
    absorbed_count: int = 0
    pressure: float = 0.0
    recent_growth: float = 0.0
    clarity: float = 0.0
    co_shift_nethras: dict[str, int] = field(default_factory=dict)
    representative_examples: list[dict[str, Any]] = field(default_factory=list)
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0


@dataclass
class EvidenceAccount:
    """Evidence tally for a nethra surface."""

    support_count: int = 0
    failure_count: int = 0
    invalidator_counts: dict[str, int] = field(default_factory=dict)
    prediction_lift: float = 0.0
    use_cost: float = 0.0
    helped_count: int = 0
    hurt_count: int = 0


@dataclass
class ProjectionPermission:
    """Diagnostic record of projection permission for a nethra in a context."""

    nethra_id: str
    context_key: str
    operation_hook: str
    allowed: bool
    strength: Literal["none", "background", "weak", "normal"] = "none"
    reason: str = ""


@dataclass
class RoleSurfaceTransition:
    """Record of a role-surface state change."""

    nethra_id: str
    context_key: str
    operation: Literal[
        "ABSORB",
        "CHARGE_RESIDUAL",
        "PROMOTE_ROLE",
        "DEMOTE_ROLE",
        "FRACTURE_IDENTITY",
        "COMPOSE",
        "SPAWN_RESIDUAL",
        "DECAY_RESIDUAL",
    ]
    cycle: int
    reason: str
    pressure_before: float = 0.0
    pressure_after: float = 0.0
    role_before: str = ""
    role_after: str = ""


@dataclass
class RegimeTransitionCandidate:
    """Candidate regime transition derived from correlated residual pressure.

    This is a proposal record only.  It must not directly promote a role.
    """

    context_key: str
    cycle: int
    source_nethras: tuple[str, ...]
    pressure: float
    recent_growth: float
    co_shift_count: int
    evidence_refs: tuple[str, ...]
    reason: str


class NethraRoleSurfaceStore:
    """Passive record store for nethra operating surfaces and residual buckets.

    Step 1 only: no runtime behaviour is altered by any method here.
    Projection permission queries are diagnostic-only and must not be wired
    into ProjectionIndex until Step 2.
    """

    def __init__(self) -> None:
        self._surfaces: dict[tuple[str, str], NethraRoleSurface] = {}
        self._buckets: dict[tuple[str, str], ResidualBucket] = {}
        self._transitions: list[RoleSurfaceTransition] = []
        self._regime_candidates: list[RegimeTransitionCandidate] = []
        self._persistent_growth_windows: int = 0
        self._consecutive_growth_cycles: int = 0
        self._prev_unresolved_total: int = -1

    # ------------------------------------------------------------------
    # Identity and surface assignment
    # ------------------------------------------------------------------

    def add_or_update_identity(
        self,
        nethra_id: str,
        context_key: str,
        context_family: str = "",
        *,
        cycle: int = 0,
    ) -> NethraRoleSurface:
        key = (nethra_id, context_key)
        if key not in self._surfaces:
            self._surfaces[key] = NethraRoleSurface(
                nethra_id=nethra_id,
                context_key=context_key,
                context_family=context_family or _context_family(context_key),
                role_state="unresolved",
                last_updated_cycle=cycle,
            )
        return self._surfaces[key]

    def assign_surface(
        self,
        nethra_id: str,
        context_key: str,
        role_state: str,
        *,
        cycle: int = 0,
        reason: str = "",
    ) -> NethraRoleSurface:
        key = (nethra_id, context_key)
        surface = self._surfaces.get(key)
        role_before = surface.role_state if surface is not None else ""

        proj, residual_col, compose, lbs, rs = _role_permissions(
            role_state, surface
        )

        if surface is None:
            surface = NethraRoleSurface(
                nethra_id=nethra_id,
                context_key=context_key,
                context_family=_context_family(context_key),
                role_state=role_state,
                load_bearing_score=lbs,
                residual_score=rs,
                projection_allowed=proj,
                residual_collection_allowed=residual_col,
                composition_allowed=compose,
                last_updated_cycle=cycle,
            )
        else:
            surface.role_state = role_state
            surface.load_bearing_score = lbs
            surface.residual_score = rs
            surface.projection_allowed = proj
            surface.residual_collection_allowed = residual_col
            surface.composition_allowed = compose
            surface.last_updated_cycle = cycle

        self._surfaces[key] = surface

        if role_before and role_before != role_state:
            op: Literal["PROMOTE_ROLE", "DEMOTE_ROLE"] = (
                "PROMOTE_ROLE"
                if _ROLE_ORDER.get(role_state, 0) > _ROLE_ORDER.get(role_before, 0)
                else "DEMOTE_ROLE"
            )
            self._record_transition(RoleSurfaceTransition(
                nethra_id=nethra_id,
                context_key=context_key,
                operation=op,
                cycle=cycle,
                reason=reason,
                role_before=role_before,
                role_after=role_state,
            ))

        return surface

    def assign_surface_from_context_role(
        self,
        role_record: Any,
        node: Any = None,
    ) -> NethraRoleSurface:
        """Update surface from a ContextRoleRecord-like object.

        Accepts any object with nethra_id, context_key, role, cycle, and
        evidence_summary attributes to avoid a circular import.
        """
        return self.assign_surface(
            str(role_record.nethra_id),
            str(role_record.context_key),
            str(role_record.role),
            cycle=int(role_record.cycle),
            reason=str(getattr(role_record, "evidence_summary", "")),
        )

    def surface_for(
        self, nethra_id: str, context_key: str
    ) -> NethraRoleSurface | None:
        return self._surfaces.get((nethra_id, context_key))

    # ------------------------------------------------------------------
    # Residual charging and absorption
    # ------------------------------------------------------------------

    def charge_residual(
        self,
        nethra_id: str,
        context_key: str,
        row: dict[str, Any],
        cycle: int,
        coactive_nethras: tuple[str, ...] = (),
    ) -> ResidualBucket:
        """Record one new unresolved residual for a nethra/context bucket."""
        key = (nethra_id, context_key)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = ResidualBucket(
                nethra_id=nethra_id,
                context_key=context_key,
                first_seen_cycle=cycle,
                last_seen_cycle=cycle,
            )
            self._buckets[key] = bucket

        prev_pressure = bucket.pressure
        bucket.residual_count += 1
        bucket.unresolved_count += 1

        for nid in coactive_nethras:
            if nid != nethra_id:
                bucket.co_shift_nethras[nid] = bucket.co_shift_nethras.get(nid, 0) + 1

        if len(bucket.representative_examples) < _MAX_REPRESENTATIVE_EXAMPLES:
            safe = {k: v for k, v in row.items() if k not in _HIDDEN_TRUTH_FIELDS}
            safe["_cycle"] = cycle
            bucket.representative_examples.append(safe)

        bucket.pressure = min(_MAX_PRESSURE, prev_pressure + _PRESSURE_PER_RESIDUAL)
        bucket.recent_growth = bucket.pressure - prev_pressure
        total = bucket.residual_count + bucket.absorbed_count
        bucket.clarity = bucket.absorbed_count / total if total > 0 else 0.0
        bucket.last_seen_cycle = cycle

        self._record_transition(RoleSurfaceTransition(
            nethra_id=nethra_id,
            context_key=context_key,
            operation="CHARGE_RESIDUAL",
            cycle=cycle,
            reason="new_residual",
            pressure_before=prev_pressure,
            pressure_after=bucket.pressure,
        ))

        return bucket

    def absorb_residual(
        self,
        nethra_id: str,
        context_key: str,
        evidence_ref: str,
        cycle: int,
    ) -> ResidualBucket | None:
        """Mark one residual as absorbed, reducing pressure and raising clarity."""
        key = (nethra_id, context_key)
        bucket = self._buckets.get(key)
        if bucket is None:
            return None

        prev_pressure = bucket.pressure
        bucket.absorbed_count += 1
        bucket.unresolved_count = max(0, bucket.unresolved_count - 1)
        bucket.pressure = max(0.0, prev_pressure - _PRESSURE_PER_RESIDUAL)
        bucket.recent_growth = bucket.pressure - prev_pressure
        total = bucket.residual_count + bucket.absorbed_count
        bucket.clarity = bucket.absorbed_count / total if total > 0 else 0.0
        bucket.last_seen_cycle = cycle

        self._record_transition(RoleSurfaceTransition(
            nethra_id=nethra_id,
            context_key=context_key,
            operation="ABSORB",
            cycle=cycle,
            reason=f"evidence_ref={evidence_ref}",
            pressure_before=prev_pressure,
            pressure_after=bucket.pressure,
        ))

        return bucket

    # ------------------------------------------------------------------
    # Background decay and classification
    # ------------------------------------------------------------------

    def decay_residuals(self, cycle: int, budget: int) -> int:
        """Decay pressure for unresolved residuals, within a budget cap."""
        decayed = 0
        for bucket in self._buckets.values():
            if decayed >= budget:
                break
            if bucket.unresolved_count <= 0:
                continue
            prev = bucket.pressure
            bucket.pressure = max(0.0, prev - _PRESSURE_DECAY_PER_OP)
            bucket.unresolved_count = max(0, bucket.unresolved_count - 1)
            total = bucket.residual_count + bucket.absorbed_count
            bucket.clarity = bucket.absorbed_count / total if total > 0 else 0.0
            bucket.recent_growth = bucket.pressure - prev
            bucket.last_seen_cycle = cycle
            self._record_transition(RoleSurfaceTransition(
                nethra_id=bucket.nethra_id,
                context_key=bucket.context_key,
                operation="DECAY_RESIDUAL",
                cycle=cycle,
                reason="background_decay",
                pressure_before=prev,
                pressure_after=bucket.pressure,
            ))
            decayed += 1
        return decayed

    def classify_background_residuals(self, cycle: int, budget: int) -> int:
        """Classify familiar residuals as absorbed within a budget cap.

        Record-only: this only updates bucket accounting, never certs or roles.
        """
        classified = 0
        for bucket in self._buckets.values():
            if classified >= budget:
                break
            if bucket.unresolved_count <= 0:
                continue
            prev = bucket.pressure
            bucket.absorbed_count += 1
            bucket.unresolved_count = max(0, bucket.unresolved_count - 1)
            bucket.pressure = max(0.0, prev - _PRESSURE_DECAY_PER_OP)
            bucket.recent_growth = bucket.pressure - prev
            total = bucket.residual_count + bucket.absorbed_count
            bucket.clarity = bucket.absorbed_count / total if total > 0 else 0.0
            bucket.last_seen_cycle = cycle
            classified += 1
        return classified

    # ------------------------------------------------------------------
    # Candidate queries (diagnostic only, never authority)
    # ------------------------------------------------------------------

    def promotion_candidates(
        self,
        cycle: int,
        min_pressure: float = 2.0,
        min_growth: float = 0.0,
        min_co_shift: int = 1,
    ) -> list[NethraRoleSurface]:
        """Return surfaces whose residual buckets exceed thresholds.

        These are candidates for observation, not for immediate promotion.
        Pressure alone does not grant authority.
        """
        out: list[NethraRoleSurface] = []
        for (nid, ck), bucket in self._buckets.items():
            if bucket.pressure < min_pressure:
                continue
            if bucket.recent_growth < min_growth:
                continue
            if len(bucket.co_shift_nethras) < min_co_shift:
                continue
            surface = self._surfaces.get((nid, ck))
            if surface is not None and surface.role_state != "tareth":
                out.append(surface)
        return out

    def regime_transition_candidates(
        self,
        cycle: int,
        min_pressure: float = 2.0,
        min_growth: float = 0.0,
        min_co_shift: int = 2,
    ) -> list[RegimeTransitionCandidate]:
        """Return regime candidates from correlated co-shifting residual buckets.

        Returns candidate records only.  No role is promoted by this method.
        """
        out: list[RegimeTransitionCandidate] = []
        for (nid, ck), bucket in self._buckets.items():
            if bucket.pressure < min_pressure:
                continue
            if bucket.recent_growth < min_growth:
                continue
            if len(bucket.co_shift_nethras) < min_co_shift:
                continue
            candidate = RegimeTransitionCandidate(
                context_key=ck,
                cycle=cycle,
                source_nethras=(nid,),
                pressure=bucket.pressure,
                recent_growth=bucket.recent_growth,
                co_shift_count=len(bucket.co_shift_nethras),
                evidence_refs=(),
                reason=f"residual_co_shift:{len(bucket.co_shift_nethras)}",
            )
            out.append(candidate)
            if len(self._regime_candidates) < _MAX_REGIME_CANDIDATES:
                self._regime_candidates.append(candidate)
        return out

    # ------------------------------------------------------------------
    # Projection permission queries (diagnostic only in Step 1)
    # ------------------------------------------------------------------

    def projection_allowed(
        self,
        nethra_id: str,
        context_key: str,
        operation_hook: str,
    ) -> bool:
        """Return whether this surface has diagnostic projection permission.

        Step 1: diagnostic only.  Must not be wired into runtime projection.
        Trass and unresolved surfaces return False for primary hooks.
        """
        surface = self._surfaces.get((nethra_id, context_key))
        if surface is None:
            return False
        if operation_hook == "primary":
            return surface.projection_allowed
        return False

    def projection_entries(
        self,
        context_key: str,
        operation_hook: str,
    ) -> list[ProjectionPermission]:
        """Return diagnostic permission records for all surfaces in a context."""
        return [
            ProjectionPermission(
                nethra_id=nid,
                context_key=ck,
                operation_hook=operation_hook,
                allowed=(surface.projection_allowed if operation_hook == "primary" else False),
                strength=_role_to_strength(surface.role_state),
                reason=f"role={surface.role_state}",
            )
            for (nid, ck), surface in self._surfaces.items()
            if ck == context_key
        ]

    # ------------------------------------------------------------------
    # Persistent growth tracking
    # ------------------------------------------------------------------

    def check_persistent_growth(self) -> None:
        """Update the persistent-growth-window counter.

        If total unresolved residuals keep increasing across consecutive
        classification passes, this indicates a closed-context failure: residuals
        are growing faster than they are being classified.  Monotonic growth for
        3+ consecutive passes increments _persistent_growth_windows.

        Record-only: this updates store counters only; it does not issue any
        operational change.
        """
        current_unresolved = sum(b.unresolved_count for b in self._buckets.values())
        if self._prev_unresolved_total >= 0 and current_unresolved > self._prev_unresolved_total:
            self._consecutive_growth_cycles += 1
            if self._consecutive_growth_cycles >= 3:
                self._persistent_growth_windows += 1
                self._consecutive_growth_cycles = 0
        else:
            self._consecutive_growth_cycles = max(0, self._consecutive_growth_cycles - 1)
        self._prev_unresolved_total = current_unresolved

    # ------------------------------------------------------------------
    # Metrics and export
    # ------------------------------------------------------------------

    def summarize(self) -> dict[str, Any]:
        surfaces = list(self._surfaces.values())
        buckets = list(self._buckets.values())

        load_bearing = [s for s in surfaces if s.role_state in {"tareth", "best_available"}]
        residual_surfaces = [s for s in surfaces if s.role_state in {"trass", "unresolved"}]

        pressure_total = sum(b.pressure for b in buckets)
        pressure_mean = pressure_total / len(buckets) if buckets else 0.0
        recent_growth_total = sum(b.recent_growth for b in buckets)
        absorbed_total = sum(b.absorbed_count for b in buckets)
        unresolved_total = sum(b.unresolved_count for b in buckets)
        clarity_mean = sum(b.clarity for b in buckets) / len(buckets) if buckets else 0.0

        return {
            "role_surface_count": len(surfaces),
            "load_bearing_surface_count": len(load_bearing),
            "residual_surface_count": len(residual_surfaces),
            "residual_bucket_count": len(buckets),
            "residual_pressure_total": round(pressure_total, 4),
            "residual_pressure_mean": round(pressure_mean, 4),
            "residual_recent_growth_total": round(recent_growth_total, 4),
            "residual_absorbed_count": absorbed_total,
            "residual_unresolved_count": unresolved_total,
            "residual_clarity_mean": round(clarity_mean, 4),
            "regime_transition_candidates_from_residuals": len(self._regime_candidates),
            "residual_pressure_persistent_growth_windows": self._persistent_growth_windows,
        }

    def export_records(self, limit: int = 200) -> dict[str, Any]:
        limit = max(0, int(limit))
        return {
            "role_surfaces": [
                asdict(s) for s in list(self._surfaces.values())[:limit]
            ],
            "residual_buckets": [
                _bucket_to_dict(b) for b in list(self._buckets.values())[:limit]
            ],
            "surface_transitions": [
                asdict(t) for t in self._transitions[:limit]
            ],
            "regime_transition_candidates": [
                _candidate_to_dict(c) for c in self._regime_candidates[:limit]
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_transition(self, t: RoleSurfaceTransition) -> None:
        if len(self._transitions) < _MAX_TRANSITIONS:
            self._transitions.append(t)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _context_family(context: str) -> str:
    context = str(context or "")
    if not context:
        return ""
    head = context.split("|", 1)[0]
    if "=" in head:
        return head.split("=", 1)[0]
    return head


def _role_permissions(
    role_state: str,
    existing: NethraRoleSurface | None,
) -> tuple[bool, bool, bool, float, float]:
    """Return (projection_allowed, residual_collection_allowed, composition_allowed,
    load_bearing_score, residual_score) for a role state."""
    prev_lbs = existing.load_bearing_score if existing is not None else 0.0

    if role_state == "tareth":
        return True, False, False, min(1.0, prev_lbs + 0.2), 0.0
    if role_state == "best_available":
        return True, False, False, min(0.5, max(prev_lbs, 0.3)), 0.0
    if role_state == "trass":
        return False, True, True, 0.0, min(1.0, (existing.residual_score if existing else 0.0) + 0.1)
    if role_state == "unresolved":
        return False, True, True, 0.0, 0.0
    # blocked, contested, unknown
    return False, False, False, 0.0, 0.0


def _role_to_strength(role_state: str) -> Literal["none", "background", "weak", "normal"]:
    if role_state == "tareth":
        return "normal"
    if role_state == "best_available":
        return "weak"
    if role_state in {"trass", "unresolved"}:
        return "background"
    return "none"


def _bucket_to_dict(b: ResidualBucket) -> dict[str, Any]:
    return {
        "nethra_id": b.nethra_id,
        "context_key": b.context_key,
        "residual_count": b.residual_count,
        "unresolved_count": b.unresolved_count,
        "absorbed_count": b.absorbed_count,
        "pressure": b.pressure,
        "recent_growth": b.recent_growth,
        "clarity": b.clarity,
        "co_shift_nethras": dict(b.co_shift_nethras),
        "representative_examples": list(b.representative_examples),
        "first_seen_cycle": b.first_seen_cycle,
        "last_seen_cycle": b.last_seen_cycle,
    }


def _candidate_to_dict(c: RegimeTransitionCandidate) -> dict[str, Any]:
    return {
        "context_key": c.context_key,
        "cycle": c.cycle,
        "source_nethras": list(c.source_nethras),
        "pressure": c.pressure,
        "recent_growth": c.recent_growth,
        "co_shift_count": c.co_shift_count,
        "evidence_refs": list(c.evidence_refs),
        "reason": c.reason,
    }
