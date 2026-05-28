"""
test_route_cascade_prune.py

Proves that closure_descendants correctly implements the route-trass cascade prune.

Three claims, each independently falsifiable:

  1. PRUNE — A descendant D with a route-trass cert for source_edge P is excluded
     from the invalidation closure when P changes. Without the cert, D would
     be included. The prune is the behavioral difference.

  2. NO-PRUNE — A descendant D with a route-tareth cert for source_edge P is
     included in the invalidation closure when P changes. Tareth edge = P
     genuinely matters to D's fit. No pruning.

  3. DEFAULT-INCLUDE — A descendant D with NO route cert for source_edge P is
     included in the invalidation closure (invariant 50: route/include by
     default unless explicitly excluded by a route cert).

These tests operate directly on ChainedLedger.closure_descendants without
running a full agent, so they are fast, deterministic, and surgically precise.
The intent: if closure_descendants loses the route-cert prune, test 1 fails.
If the prune accidentally over-prunes tareth edges, test 2 fails.
If the default changes to exclude-by-default, test 3 fails.
"""

import pytest
from dreth.ledger import ChainedLedger, NethraCertificate


def _trass_route_cert(source_edge: int, target: int) -> NethraCertificate:
    """Minimal route-trass cert: source_edge P is irrelevant to target T's fit."""
    return NethraCertificate(
        operation="route",
        role="trass",
        authority="guarded_reuse",
        context_source_edges=(source_edge,),
        context_visible=5,
        context_cycle=10,
        targets=(target,),
        substitutions_tested=("route_exclusion",),
        changes=0,
        trials=3,
        earned_by="counterfactual_fit",
    )


def _tareth_route_cert(source_edge: int, target: int) -> NethraCertificate:
    """Minimal route-tareth cert: source_edge P is load-bearing for target T's fit."""
    return NethraCertificate(
        operation="route",
        role="tareth",
        authority="prefer",
        context_source_edges=(source_edge,),
        context_visible=5,
        context_cycle=10,
        targets=(target,),
        substitutions_tested=("route_inclusion",),
        changes=3,
        trials=3,
        earned_by="counterfactual_fit",
    )


def _ledger_with_graph() -> ChainedLedger:
    """
    Three-var ledger: x0 → x1 → x2.
    x0 is changed (the 'P' that changed).
    x1 is the direct child of x0.
    x2 is the indirect child via x1.
    Returns the ledger with fits wired.
    """
    led = ChainedLedger(3)
    led.vars[1].source_edges = (0,)  # x1 depends on x0
    led.vars[2].source_edges = (1,)  # x2 depends on x1
    return led


# ── 1: PRUNE ──────────────────────────────────────────────────────────────────

def test_01_route_trass_prunes_cascade():
    """A descendant with a route-trass cert for the changed source_edge is excluded
    from the closure. This is the core cascade prune invariant."""
    led = _ledger_with_graph()

    # x1 has a route-trass cert for x0: "x0 is irrelevant to my fit"
    led.vars[1].route_certs[0] = _trass_route_cert(source_edge=0, target=1)

    closure = led.closure_descendants({0})

    # x0 is always in its own closure
    assert 0 in closure, "changed var must be in closure"

    # x1 should be EXCLUDED because route-trass cert says x0 doesn't matter
    assert 1 not in closure, (
        "x1 should be pruned: route-trass cert for x0 says x0 is irrelevant to x1's fit. "
        "If this fails, closure_descendants lost the route-trass prune."
    )

    # x2 depends on x1 only; since x1 is pruned, x2 should also be excluded
    assert 2 not in closure, (
        "x2 should be pruned: x1 was pruned, so x2's source_edge change doesn't cascade either."
    )


# ── 2: NO-PRUNE — tareth edge propagates ─────────────────────────────────────

def test_02_route_tareth_allows_cascade():
    """A descendant with a route-tareth cert for the changed source_edge is included
    in the closure. P matters to D's fit — invalidation must propagate."""
    led = _ledger_with_graph()

    # x1 has a route-tareth cert for x0: "x0 genuinely matters to my fit"
    led.vars[1].route_certs[0] = _tareth_route_cert(source_edge=0, target=1)

    closure = led.closure_descendants({0})

    assert 0 in closure
    assert 1 in closure, (
        "x1 should be in closure: route-tareth cert means x0 matters to x1's fit. "
        "If this fails, tareth edges are being incorrectly pruned."
    )
    assert 2 in closure, "x2 should cascade because x1 is in closure"


# ── 3: DEFAULT — no cert means include ────────────────────────────────────────

def test_03_no_cert_defaults_to_include():
    """With no route cert, closure_descendants includes descendants by default.
    Invariant 50: route/include by default unless explicitly excluded by a cert."""
    led = _ledger_with_graph()
    # No route_certs on any var

    closure = led.closure_descendants({0})

    assert 0 in closure
    assert 1 in closure, (
        "x1 should be in closure: no cert means include by default (invariant 50). "
        "If this fails, the default changed to exclude-without-cert."
    )
    assert 2 in closure, "x2 should cascade: x1 is included, no cert on x2"


# ── 4: MIXED — prune one branch, propagate another ────────────────────────────

def test_04_partial_prune_mixed_graph():
    """In a graph where x0 → x1 and x0 → x2, with x1 having route-trass for x0
    but x2 having no cert, the closure should include x2 but not x1."""
    led = ChainedLedger(3)
    led.vars[1].source_edges = (0,)  # x1 depends on x0
    led.vars[2].source_edges = (0,)  # x2 also depends on x0 (sibling, not chain)

    # x1 says x0 is irrelevant; x2 has no cert
    led.vars[1].route_certs[0] = _trass_route_cert(source_edge=0, target=1)

    closure = led.closure_descendants({0})

    assert 0 in closure
    assert 1 not in closure, "x1 pruned: route-trass cert for x0"
    assert 2 in closure, "x2 included: no route cert, default is include"
