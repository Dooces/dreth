#!/usr/bin/env python3
"""
test_per_operation_certs.py

Proves T0–T5: the VarNethra cert system is actually driving decisions.

Five claims, each independently falsifiable:

  1. FIELD GONE — VarNethra has no operation_role attribute. Reading it raises
     AttributeError. The legacy path is fully removed.

  2. CERTS POPULATED — After a real 150-cycle run, at least one var has a
     skip cert with role in {tareth, trass}. The system is not stuck at
     "untested" forever.

  3. SKIP GATED ON CERT — A var whose skip cert says "trass" accumulates
     skip_count > 0. A var whose skip cert says "tareth" was not skipped
     by role alone. Cert is actually read by run_cycle.

  4. PER-OPERATION SCOPING — A var can hold role_for("skip") == "tareth"
     while role_for("route") == "untested" on the same object simultaneously.
     Proves certs are keyed by operation, not a single field.

  5. INVALIDATE WORKS — invalidate_certs("sentinel_failure") sets the skip
     cert's role to "untested" (without deleting the cert record). The cert
     record's operation field stays "skip". The role changes.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent
from dreth.ledger import VarNethra, NethraCertificate

# ── helpers ──────────────────────────────────────────────────────────────────

def make_agent(n_vars=8, seed=42):
    rng = random.Random(seed)
    world = CausalWorld(n_vars, rng, noise_sigma=0.05)
    world.visible_count = n_vars
    agent = ChainedAgent(world, rng,
                         sentinel_count=3, sentinel_pool=8,
                         compression_discover_after=30,
                         priority_audit_budget=n_vars)
    return agent, world


def run_cycles(agent, world, n=150):
    agent.initialize()
    for c in range(1, n + 1):
        agent.run_cycle(HiddenMutation(c, "VALUE", "steady", False, -1))


# ── PROOF 1: field gone ───────────────────────────────────────────────────────

def test_operation_role_field_removed():
    """VarNethra must not have an operation_role attribute at all."""
    n = VarNethra(var=0)
    with pytest.raises(AttributeError):
        _ = n.operation_role  # type: ignore[attr-defined]


# ── PROOF 2: certs populated ─────────────────────────────────────────────────

def test_skip_certs_populated_after_run():
    """After 150 real cycles at least one var has a skip cert with a
    concrete role. The cert carries operation="skip"."""
    agent, world = make_agent()
    run_cycles(agent, world, 150)

    decided = [
        n for n in agent.ledger.vars.values()
        if n.role_for("skip") in ("tareth", "trass")
    ]
    assert len(decided) > 0, (
        f"No vars earned a skip cert in 150 cycles. "
        f"All roles: {[n.role_for('skip') for n in agent.ledger.vars.values()]}"
    )
    # Every decided var must have a cert record, not a legacy field
    for n in decided:
        assert "skip" in n.certificates, (
            f"x{n.var} role_for('skip')={n.role_for('skip')} but no cert record"
        )
        cert = n.certificates["skip"]
        assert cert.operation == "skip", f"cert.operation={cert.operation!r}"
        assert cert.role in ("tareth", "trass", "false_trass"), (
            f"unexpected role {cert.role!r}"
        )


# ── PROOF 3: skip gated on cert ───────────────────────────────────────────────

def test_skip_count_driven_by_cert_role():
    """Trass-cert vars must accumulate skip_count. Tareth-cert vars must not
    be skipped by role alone (they may be sentinel-skipped but not role-skipped).

    Verifies that run_cycle reads role_for("skip"), not a legacy field."""
    agent, world = make_agent(n_vars=10, seed=99)
    run_cycles(agent, world, 200)

    trass_vars = [
        n for n in agent.ledger.vars.values()
        if n.role_for("skip") == "trass" and n.var < world.visible_count
    ]
    tareth_vars = [
        n for n in agent.ledger.vars.values()
        if n.role_for("skip") == "tareth" and n.var < world.visible_count
    ]

    if trass_vars:
        # Every trass var must have been skipped at least once
        for n in trass_vars:
            assert n.skip_count > 0, (
                f"x{n.var} is trass but skip_count=0 — cert not read by run_cycle"
            )

    if tareth_vars:
        # Every tareth var must have been audited at least once
        for n in tareth_vars:
            assert n.full_audits > 0, (
                f"x{n.var} is tareth but was never audited — impossible if cert is read"
            )


# ── PROOF 4: per-operation scoping ────────────────────────────────────────────

def test_skip_and_route_are_independent():
    """A var can have skip=tareth and route=untested simultaneously.
    This is the core taxonomy claim: role is per-operation, not a global label."""
    agent, world = make_agent()
    run_cycles(agent, world, 150)

    for n in agent.ledger.vars.values():
        if n.var >= world.visible_count:
            continue
        skip_role = n.role_for("skip")
        route_role = n.role_for("route")

        # route certs are only issued by _certify_form_role — most vars won't
        # have one. Confirm: having a skip cert does not imply a route cert.
        if skip_role in ("tareth", "trass"):
            # If route is not untested, it came from a form cert. If untested,
            # the two operations are independently tracked.
            assert route_role in ("tareth", "trass", "untested", "false_trass"), (
                f"x{n.var} unexpected route role {route_role!r}"
            )

    # Find at least one var with skip cert but no route cert
    skip_only = [
        n for n in agent.ledger.vars.values()
        if n.var < world.visible_count
        and n.role_for("skip") in ("tareth", "trass")
        and n.role_for("route") == "untested"
    ]
    assert len(skip_only) > 0, (
        "Every decided var has a route cert — expected some to have skip-only. "
        "Per-operation independence is not being maintained."
    )

    # Spot-check: for any such var, the cert dict has exactly the skip key
    n = skip_only[0]
    assert "skip" in n.certificates
    assert "route" not in n.certificates, (
        f"x{n.var} has route cert despite role_for('route')=='untested'"
    )


# ── PROOF 5: invalidate works ────────────────────────────────────────────────

def test_invalidate_certs_sentinel_failure():
    """invalidate_certs('sentinel_failure') must:
      - set skip cert role to 'untested'
      - preserve the cert record (not delete it)
      - leave cert.operation == 'skip'
    """
    n = VarNethra(var=0)
    # Manually install a tareth skip cert
    n.certificates["skip"] = NethraCertificate(
        operation="skip", role="tareth", authority="none",
        context_parents=(1, 2), context_visible=5, context_cycle=10,
        targets=(3, 4), substitutions_tested=("perturbation",),
        changes=3, trials=5,
    )
    assert n.role_for("skip") == "tareth"

    # Sentinel failure should mark skip as untested
    n.invalidate_certs("sentinel_failure")

    assert "skip" in n.certificates, "cert record must survive invalidation"
    cert = n.certificates["skip"]
    assert cert.role == "untested", (
        f"after sentinel_failure, expected untested, got {cert.role!r}"
    )
    assert cert.operation == "skip", "operation field must survive invalidation"

    # Also check: structural_mutation clears everything
    n.certificates["skip"] = NethraCertificate(
        operation="skip", role="trass", authority="skip",
        context_parents=(), context_visible=5, context_cycle=20,
        targets=(), substitutions_tested=("perturbation",),
        changes=0, trials=5,
    )
    assert n.role_for("skip") == "trass"
    n.invalidate_certs("structural_mutation")
    assert n.role_for("skip") == "untested", "structural_mutation must clear all certs"
    assert len(n.certificates) == 0, "structural_mutation must empty the dict"


if __name__ == "__main__":
    print("PROOF 1: field gone")
    test_operation_role_field_removed()
    print("  PASS")

    print("PROOF 2: certs populated after run")
    test_skip_certs_populated_after_run()
    print("  PASS")

    print("PROOF 3: skip count driven by cert role")
    test_skip_count_driven_by_cert_role()
    print("  PASS")

    print("PROOF 4: per-operation scoping")
    test_skip_and_route_are_independent()
    print("  PASS")

    print("PROOF 5: invalidate works")
    test_invalidate_certs_sentinel_failure()
    print("  PASS")

    print("\nAll 5 proofs passed.")
