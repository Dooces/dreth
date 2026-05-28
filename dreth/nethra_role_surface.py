from __future__ import annotations

"""Nethra role-surface model: record-only operating surfaces.

Surfaces record what role a nethra occupies in a given context and what projection
permissions follow from that role.  Nothing here may alter audit, skip, probe,
repair, ranking, or projection runtime behaviour.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ContextKey = str
NethraId = str

_MAX_TRANSITIONS: int = 2000

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


class NethraRoleSurfaceStore:
    """Passive record store for nethra operating surfaces.

    No runtime behaviour is altered by any method here.
    Projection permission queries are diagnostic-only.
    """

    def __init__(self) -> None:
        self._surfaces: dict[tuple[str, str], NethraRoleSurface] = {}
        self._transitions: list[RoleSurfaceTransition] = []

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

    def projection_allowed(
        self, nethra_id: str, context_key: str, operation_hook: str = ""
    ) -> bool:
        surface = self._surfaces.get((nethra_id, context_key))
        return surface is not None and surface.projection_allowed

    def projection_entries(
        self, context_key: str, operation_hook: str = ""
    ) -> list[ProjectionPermission]:
        out = []
        for (nid, ctx), surface in self._surfaces.items():
            if ctx == context_key:
                out.append(ProjectionPermission(
                    nethra_id=nid,
                    context_key=ctx,
                    operation_hook=operation_hook,
                    allowed=surface.projection_allowed,
                    strength=_role_to_strength(surface.role_state),
                    reason=surface.role_state,
                ))
        return out

    # ------------------------------------------------------------------
    # Metrics and export
    # ------------------------------------------------------------------

    def summarize(self) -> dict[str, Any]:
        surfaces = list(self._surfaces.values())
        load_bearing = [s for s in surfaces if s.role_state in {"tareth", "best_available"}]
        residual_surfaces = [s for s in surfaces if s.role_state in {"trass", "unresolved"}]
        return {
            "role_surface_count": len(surfaces),
            "load_bearing_surface_count": len(load_bearing),
            "residual_surface_count": len(residual_surfaces),
        }

    def export_records(self, limit: int = 200) -> dict[str, Any]:
        limit = max(0, int(limit))
        return {
            "role_surfaces": [
                asdict(s) for s in list(self._surfaces.values())[:limit]
            ],
            "surface_transitions": [
                asdict(t) for t in self._transitions[:limit]
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


