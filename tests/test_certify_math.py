#!/usr/bin/env python3
"""
test_certify_math.py

Mathematical proofs that the cert system captures causal structure correctly.
Each proof derives the expected cert fields from first principles, then checks
the implementation matches. Zero-noise worlds are used so results are exact.

Three falsifiable proofs:

  M1. TARETH IS EXACT — In a zero-noise world where x1 = FIRST(x0), certifying
      x0 gives exactly 4 changes out of 5 trials. Derivation:
        spread = [0.05, 0.25, 0.50, 0.75, 0.95], state[x0] = 0.5, tol = 0.1
        v=0.05: |0.05 - 0.5| = 0.45 > tol. x1_new = 0.05. |0.05 - 0.5| > tol → change.
        v=0.25: |0.25 - 0.5| = 0.25 > tol. x1_new = 0.25. |0.25 - 0.5| > tol → change.
        v=0.50: |0.50 - 0.5| = 0    ≤ tol. SKIPPED (too close to current).
        v=0.75: |0.75 - 0.5| = 0.25 > tol. x1_new = 0.75. |0.75 - 0.5| > tol → change.
        v=0.95: |0.95 - 0.5| = 0.45 > tol. x1_new = 0.95. |0.95 - 0.5| > tol → change.
      cert.changes == 4, cert.trials == 5, cert.role == "tareth".

  M2. TRASS IS ZERO — Same world, certifying x2 (HIGH source, no children):
        Perturbing x2 to any v:
          x0_new = LOW([]) = 0.2 (constant; same in baseline and perturbed)
          x1_new = FIRST([state[x0]]) = 0.5 (x0 not forced; same both cases)
        No downstream var changes. cert.changes == 0, cert.role == "trass".

  M3. FALSE_TRASS IS DETECTED — 4-var zero-noise PROD world:
        x0=0, x1=0, x2=PROD(x0,x1)=0, x3=FIRST(x2)=0.
        Individual tests: PROD(v,0)=0 and PROD(0,v)=0 for any v → both trass.
        Joint test derivation (tol = 0.1, spread zipped with itself):
          (0.05, 0.05): |0.05-0|=0.05 ≤ tol → SKIPPED.
          (0.25, 0.25): RAB[x2]=PROD(0.25,0.25)=0.0625 ≤ 0.1 → not interaction.
          (0.50, 0.50): RAB[x2]=PROD(0.50,0.50)=0.2500 > 0.1 → interaction.
          (0.75, 0.75): RAB[x2]=PROD(0.75,0.75)=0.5625 > 0.1 → interaction.
          (0.95, 0.95): RAB[x2]=PROD(0.95,0.95)=0.9025 > 0.1 → interaction.
        4 trials ran, 3 showed interaction. 3 >= 4/2 → jointly tareth.
        cert.changes==3, cert.trials==4.
        cert.joint_R0==0.0, cert.joint_RA==0.0, cert.joint_RB==0.0,
        cert.joint_RAB==0.0625 (from first non-skipped trial: PROD(0.25,0.25)).
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent
from dreth.ledger import NethraCertificate, DEFAULT_TOLERANCE


TOL = DEFAULT_TOLERANCE   # 0.1 — the default per-var tolerance before cert


def make_tareth_world():
    """3-var zero-noise world: x0 source, x1=FIRST(x0), x2 source (no children).
    Manually set state so analysis is exact."""
    rng = random.Random(0)
    world = CausalWorld(3, rng, noise_sigma=0.0)
    world.source_edges = [[], [0], []]
    world.funcs   = ["LOW", "FIRST", "HIGH"]
    world.state   = (0.5, 0.5, 0.8)
    world.visible_count = 3
    agent = ChainedAgent(world, rng,
                         sentinel_count=0, sentinel_pool=0,
                         compression_discover_after=999,
                         priority_audit_budget=3)
    return agent, world


def make_prod_world():
    """4-var zero-noise PROD world: x0=LOW, x1=LOW, x2=PROD(x0,x1), x3=FIRST(x2).
    State forced to (0,0,0,0) to create false-trass condition."""
    rng = random.Random(0)
    world = CausalWorld(4, rng, noise_sigma=0.0)
    world.source_edges = [[], [], [0, 1], [2]]
    world.funcs   = ["LOW", "LOW", "PROD", "FIRST"]
    world.state   = (0.0, 0.0, 0.0, 0.0)
    world.visible_count = 4
    agent = ChainedAgent(world, rng,
                         sentinel_count=0, sentinel_pool=0,
                         compression_discover_after=999,
                         priority_audit_budget=4)
    return agent, world


# ── PROOF M1: tareth is exact ─────────────────────────────────────────────────

def test_tareth_cert_changes_are_analytically_exact():
    """Certifying x0 (source_edge of x1=FIRST(x0)) in a zero-noise world
    must produce exactly 4 changes out of 5 trials.

    Derivation is in the module docstring.
    This is NOT a bound or approximation — noise_sigma=0 makes it exact."""
    agent, world = make_tareth_world()
    # x0 has 2 other visible vars (x1, x2) — certification won't defer.
    role = agent._certify_operation_role(var=0, cycle=1)

    assert role == "tareth", f"expected tareth, got {role!r}"
    cert = agent.ledger.vars[0].certificates["skip"]

    assert cert.operation == "skip"
    assert cert.role == "tareth"
    assert cert.trials == 5, (
        f"spread has 5 values → trials must be 5, got {cert.trials}"
    )
    assert cert.changes == 4, (
        f"Analytical prediction: 4 of 5 perturbations propagate to x1. "
        f"Only v=0.5 is skipped (|0.5-0.5|=0 ≤ tol={TOL}). "
        f"All others: |v-0.5| > {TOL} → x1_new=v → |v-0.5| > {TOL}. "
        f"Got cert.changes={cert.changes}"
    )
    # Evidence ratio: 4/5 = 0.8 > 0.5 threshold → tareth ✓
    assert cert.changes / cert.trials >= 0.5


# ── PROOF M2: trass is zero ───────────────────────────────────────────────────

def test_trass_cert_changes_are_analytically_zero():
    """Certifying x2 (HIGH source, no children) must produce exactly 0 changes.

    Derivation: x0 and x1 have no path from x2. Perturbing x2:
      x0_new = LOW([]) = 0.2 (unchanged vs baseline)
      x1_new = FIRST([state[x0]]) = 0.5 (x0 not forced; unchanged vs baseline)
    No var changes → changes must be zero exactly."""
    agent, world = make_tareth_world()
    role = agent._certify_operation_role(var=2, cycle=1)

    assert role == "trass", f"expected trass, got {role!r}"
    cert = agent.ledger.vars[2].certificates["skip"]

    assert cert.operation == "skip"
    assert cert.role == "trass"
    assert cert.trials == 5, f"trials must be 5, got {cert.trials}"
    assert cert.changes == 0, (
        f"x2 has no children. Perturbing it changes nothing downstream. "
        f"Expected 0 changes. Got {cert.changes}."
    )
    # Evidence ratio: 0/5 = 0.0 < 0.5 threshold → trass ✓
    assert cert.changes / cert.trials < 0.5


# ── PROOF M3: false_trass detection is analytically correct ───────────────────

def test_false_trass_joint_evidence_is_analytically_correct():
    """PROD(x0=0, x1=0)=0: x0 and x1 are individually trass (each alone
    produces no change in x2), but jointly tareth (PROD(v,v)=v^2 > tol
    for v in {0.5, 0.75, 0.95}).

    Derivation is in the module docstring. Checks:
      - individual certs: changes==0 → trass
      - joint test returns "tareth"
      - cert.changes==3, cert.trials==4 (1 trial skipped, 1 runs but no interaction)
      - cert.joint_R0 == 0.0 (PROD(0,0) = 0)
      - cert.joint_RA == 0.0 (PROD(v,0) = 0 for all v)
      - cert.joint_RB == 0.0 (PROD(0,v) = 0 for all v)
      - cert.joint_RAB == 0.0625 (first non-skipped trial: PROD(0.25,0.25))
      - cert.joint_members == (0, 1)
    """
    agent, world = make_prod_world()

    # Certify x0 individually — must come out trass.
    role_x0 = agent._certify_operation_role(var=0, cycle=1)
    cert_x0 = agent.ledger.vars[0].certificates["skip"]
    assert role_x0 == "trass", (
        f"PROD(v,0)=0 for all v → x0 is individually trass. Got {role_x0!r}. "
        f"changes={cert_x0.changes}"
    )
    assert cert_x0.changes == 0, (
        f"Perturbing x0 with x1=0: PROD(v,0)=0 always. Expected 0 changes. "
        f"Got {cert_x0.changes}."
    )

    # Certify x1 individually — must also be trass.
    role_x1 = agent._certify_operation_role(var=1, cycle=1)
    cert_x1 = agent.ledger.vars[1].certificates["skip"]
    assert role_x1 == "trass", (
        f"PROD(0,v)=0 for all v → x1 is individually trass. Got {role_x1!r}."
    )
    assert cert_x1.changes == 0

    # x2 needs a tareth cert so the joint test has a sentinel to observe.
    # Analytically: perturbing x2 → x3_new=FIRST(x2_forced)=v, |v-0|>tol for v>0.1.
    # We install it directly since _certify_operation_role for x2 would also
    # be trass in this world (x3 reads forced[x2] only when x2 is the forced var;
    # the propagation IS one-step so x3=FIRST(v) → tareth). Run it to confirm.
    role_x2 = agent._certify_operation_role(var=2, cycle=1)
    assert role_x2 == "tareth", (
        f"x3=FIRST(x2): perturbing x2 to v gives x3_new=v. "
        f"|v-0|>0.1 for v in {{0.25,0.5,0.75,0.95}} → x2 is tareth. Got {role_x2!r}."
    )

    # Now run the joint test.
    result = agent._test_joint_false_trass(var_a=0, var_b=1, cycle=1)

    assert result == "tareth", (
        f"Joint test should return 'tareth': 3/4 trials show PROD(v,v)>tol. "
        f"Got {result!r}."
    )

    # After _test_joint_false_trass, invalidate_certs("false_trass_contradiction")
    # was called internally, setting role → "untested" but preserving joint evidence.
    cert = agent.ledger.vars[0].certificates["skip"]
    assert cert.role == "untested", (
        f"After joint test, invalidate_certs sets role to 'untested' for re-cert. "
        f"Got {cert.role!r}."
    )

    # Joint evidence fields survive invalidation.
    assert cert.joint_members == (0, 1), (
        f"joint_members must be (var_a, var_b) = (0,1). Got {cert.joint_members}"
    )

    # Analytically: changes=3 (interactions at v=0.5, 0.75, 0.95),
    # trials=4 (ran 0.25, 0.50, 0.75, 0.95; skipped 0.05).
    assert cert.changes == 3, (
        f"3 of 4 non-skipped trials produce PROD(v,v)>0.1: "
        f"PROD(0.5,0.5)=0.25, PROD(0.75,0.75)=0.5625, PROD(0.95,0.95)=0.9025. "
        f"PROD(0.25,0.25)=0.0625 ≤ 0.1 (no interaction). "
        f"Got cert.changes={cert.changes}."
    )
    assert cert.trials == 4, (
        f"4 trials ran (0.25, 0.50, 0.75, 0.95); 0.05 skipped (|0.05-0|≤0.1). "
        f"Got cert.trials={cert.trials}."
    )

    # Reference values from the FIRST non-skipped trial: (val_a=0.25, val_b=0.25).
    #   R0[x2]  = PROD(state[x0], state[x1]) = PROD(0, 0) = 0.0
    #   RA[x2]  = PROD(0.25,      state[x1]) = PROD(0.25, 0) = 0.0
    #   RB[x2]  = PROD(state[x0], 0.25)      = PROD(0, 0.25) = 0.0
    #   RAB[x2] = PROD(0.25, 0.25)           = 0.0625
    assert cert.joint_R0  == pytest.approx(0.0),    f"R0={cert.joint_R0}"
    assert cert.joint_RA  == pytest.approx(0.0),    f"RA={cert.joint_RA}"
    assert cert.joint_RB  == pytest.approx(0.0),    f"RB={cert.joint_RB}"
    assert cert.joint_RAB == pytest.approx(0.0625), f"RAB={cert.joint_RAB}"


if __name__ == "__main__":
    print("PROOF M1: tareth cert changes are analytically exact")
    test_tareth_cert_changes_are_analytically_exact()
    print("  PASS")

    print("PROOF M2: trass cert changes are analytically zero")
    test_trass_cert_changes_are_analytically_zero()
    print("  PASS")

    print("PROOF M3: false_trass joint evidence is analytically correct")
    test_false_trass_joint_evidence_is_analytically_correct()
    print("  PASS")

    print("\nAll 3 mathematical proofs passed.")
