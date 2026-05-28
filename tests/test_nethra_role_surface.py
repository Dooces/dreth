from __future__ import annotations

import random

import pytest

from dreth.context_role_index import (
    ContextRoleIndex,
    ContextRoleRecord,
    NethraNode,
)
from dreth.nethra_role_surface import (
    NethraRoleSurfaceStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store() -> NethraRoleSurfaceStore:
    return NethraRoleSurfaceStore()


def _record(
    nethra_id: str,
    context_key: str,
    role: str,
    cycle: int = 1,
    operation: str = "audit",
) -> ContextRoleRecord:
    return ContextRoleRecord(
        nethra_id=nethra_id,
        context_key=context_key,
        operation=operation,
        role=role,
        cycle=cycle,
    )


def _node(nethra_id: str, *, var: int = 0) -> NethraNode:
    return NethraNode(
        nethra_id=nethra_id,
        kind="var_fit",
        target_var=var,
        components=(var,),
    )


def _index_with_records(records: list[ContextRoleRecord]) -> ContextRoleIndex:
    idx = ContextRoleIndex()
    for rec in records:
        idx.assign_context_role(rec)
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trass_surface_collects_residual_without_projection() -> None:
    store = _store()
    store.assign_surface("n1", "ctx_a", "trass", cycle=1)

    surface = store.surface_for("n1", "ctx_a")
    assert surface is not None
    assert surface.role_state == "trass"
    assert surface.residual_collection_allowed is True
    assert surface.projection_allowed is False

    # Primary projection must be denied
    assert store.projection_allowed("n1", "ctx_a", "primary") is False


def test_tareth_surface_can_be_projection_capable() -> None:
    store = _store()
    store.assign_surface("n2", "ctx_b", "tareth", cycle=1)

    surface = store.surface_for("n2", "ctx_b")
    assert surface is not None
    assert surface.role_state == "tareth"
    assert surface.projection_allowed is True
    assert surface.load_bearing_score > 0.0

    # Diagnostic permission is allowed
    assert store.projection_allowed("n2", "ctx_b", "primary") is True


def test_best_available_projection_is_weak() -> None:
    store = _store()
    store.assign_surface("n3", "ctx_c", "best_available", cycle=1)

    surface = store.surface_for("n3", "ctx_c")
    assert surface is not None
    assert surface.projection_allowed is True
    # Load-bearing but capped below tareth threshold
    assert surface.load_bearing_score <= 0.5

    entries = store.projection_entries("ctx_c", "primary")
    assert entries, "expected at least one permission entry"
    perm = entries[0]
    assert perm.strength == "weak"
    assert perm.allowed is True


def test_unresolved_surface_preserves_without_authority() -> None:
    store = _store()
    store.assign_surface("n4", "ctx_d", "unresolved", cycle=1)

    surface = store.surface_for("n4", "ctx_d")
    assert surface is not None
    assert surface.projection_allowed is False
    assert surface.residual_collection_allowed is True
    assert surface.composition_allowed is True

    # No primary projection authority
    assert store.projection_allowed("n4", "ctx_d", "primary") is False

    entries = store.projection_entries("ctx_d", "primary")
    for e in entries:
        if e.nethra_id == "n4":
            assert e.allowed is False


def test_context_role_index_backcompat_metrics_survive() -> None:
    idx = ContextRoleIndex()
    idx.add_or_update_node(_node("n1", var=0))
    idx.assign_context_role(_record("n1", "audit|x0", "tareth", cycle=1))
    idx.assign_context_role(_record("n1", "audit|x0", "trass", cycle=2))

    summary = idx.summarize()

    # All pre-existing keys must be present
    expected_keys = [
        "context_role_index_nodes",
        "context_role_records",
        "context_role_tareth",
        "context_role_trass",
        "context_role_unresolved",
        "context_role_best_available",
        "context_role_index_queries",
        "context_role_index_matches",
        "context_role_raw_matches",
        "context_role_deduped_matches",
        "context_role_matches_suppressed_weak",
        "context_role_matches_suppressed_duplicate",
        "context_role_matches_suppressed_cap",
        "context_role_matches_used_as_local_anchor",
        "context_role_assist_feature_hits",
        "context_role_anchor_policy",
        "context_role_assist_pressure_events",
        "context_role_assist_pressure_per_cycle",
        "context_role_top_match_reasons",
        "context_role_nodes_by_kind",
        "context_role_nodes_by_source",
        "context_roles_by_context",
        "context_roles_by_role",
        "context_role_edges",
        "context_role_edges_by_kind",
        # Compatibility aliases
        "nethra_reservoir_records",
        "nethra_context_roles",
        "nethra_role_tareth",
        "nethra_role_trass",
        "nethra_role_unresolved",
        "nethra_role_best_available",
        "reservoir_queries",
        "reservoir_matches",
        "reservoir_raw_matches",
        "reservoir_deduped_matches",
        "reservoir_matches_used_as_local_anchor",
        "reservoir_assist_feature_hits",
        "reservoir_records_by_kind",
        "reservoir_records_by_source",
        "reservoir_roles_by_context",
        "reservoir_roles_by_role",
    ]
    for key in expected_keys:
        assert key in summary, f"missing backcompat key: {key}"


def test_context_role_index_export_includes_surfaces() -> None:
    idx = ContextRoleIndex()
    idx.add_or_update_node(_node("n1", var=1))
    idx.assign_context_role(_record("n1", "audit|x1", "tareth", cycle=1))
    idx.assign_context_role(_record("n1", "audit|x1", "trass", cycle=3))

    exported = idx.export_records(limit=50)

    assert "nodes" in exported
    assert "edges" in exported
    assert "roles" in exported

    assert "role_surfaces" in exported
    assert "surface_transitions" in exported

    # Surface should reflect the most recent role assigned
    surfaces = exported["role_surfaces"]
    assert len(surfaces) >= 1
    surface_ids = {s["nethra_id"] for s in surfaces}
    assert "n1" in surface_ids

    # Transitions should include the role change
    transitions = exported["surface_transitions"]
    ops = {t["operation"] for t in transitions}
    assert "PROMOTE_ROLE" in ops or "DEMOTE_ROLE" in ops
