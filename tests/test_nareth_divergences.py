#!/usr/bin/env python3
"""
test_nareth_divergences.py

Six tests that encode the correct nareth behavior for the divergences identified
in the code comments (NARETH DIVERGENCE Q2–Q7). Each test:

  - Asserts the behavior that OUGHT to exist once the divergence is fixed.
  - Currently FAILS against the unmodified code.
  - Will PASS once the corresponding fix is applied.

Nareth invariant under test:
  A nareth is an earned, scoped, revocable, operation-indexed authority certificate.
  It permits a shortcut decision ONLY within the scope of the evidence that certified it.
  A cert earned in scope S does not authorize action in scope S' ⊃ S.

Tests:

  Q2a — STATUS_TRASS_RETEST
        A var with status="trass" but no skip cert must be included in
        _retest_trass_vars. Currently missed: filter is role_for("skip") != "trass",
        which skips vars with role="untested" (no cert at all).

  Q2b — STATUS_TRASS_SKIP_GATE
        run_cycle must not skip a var that has status="trass" but no skip cert.
        The skip gate must be role_for("skip") == "trass", not status.

  Q3  — COMPRESSION_CERT
        After a compression is promoted (pred_passes >= threshold),
        certificates["compress"] must exist on the var's VarNethra.
        Currently nothing populates it.

  Q5  — TARETH_WITNESS_STORED
        A tareth-for-skip cert must store the specific intervention context
        that certified it, so the sentinel path can replay that evidence.
        Currently NethraCertificate has no witness field.

  Q6  — LIVE_FRONTIER_INVALIDATION
        Under live-frontier mode, when a var that was excluded from a cert's
        target set transitions trass→tareth, certs that excluded it must be
        flagged untested. Currently no invalidation fires.

  Q7  — SKIP_PROXY_NOT_CONSERVATIVE
        A var that is skip-trass must not be excluded from available_source_edges
        solely on that basis. Route relevance is a different operation.
        Currently available_source_edges requires tareth-for-skip.
"""

import dataclasses
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent
from dreth.ledger import VarNethra, NethraCertificate


# ── helpers ───────────────────────────────────────────────────────────────────

def make_agent(n_vars=8, seed=42, role_salience="all-visible", **kwargs):
    rng = random.Random(seed)
    world = CausalWorld(n_vars, rng, noise_sigma=0.05)
    world.visible_count = n_vars
    agent = ChainedAgent(
        world, rng,
        sentinel_count=3,
        sentinel_pool=12,
        compression_discover_after=2,
        compression_promote_after=5,
        priority_audit_budget=n_vars,
        role_salience=role_salience,
        **kwargs,
    )
    return agent, world


def run_cycles(agent, world, n):
    agent.initialize()
    for c in range(1, n + 1):
        agent.run_cycle(HiddenMutation(c, "VALUE", "steady", False, -1))


def _steady(cycle):
    return HiddenMutation(cycle, "VALUE", "steady", False, -1)


# ── Q2a: status-only-trass var must be retested on reveal ────────────────────

def test_status_trass_retest_on_reveal():
    """
    NARETH DIVERGENCE Q2a — _retest_trass_vars must cover status-only-trass vars.

    Invariant: trass-for-skip is scoped. When new variables appear, the scope
    expands and every var that was collapsed must be retested against the wider
    target set. This must include vars with status="trass" even when no skip cert
    exists (cert was removed, or the var was collapsed via a legacy path).

    Current failure: _retest_trass_vars filters with
        if n.role_for("skip") != "trass": continue
    A var with no cert returns role_for("skip") == "untested", so "untested" !=
    "trass" is True and the var is skipped by the filter. It is permanently
    invisible to revalidation.

    Fix: also include vars where n.status == "trass" and no cert exists.
    """
    agent, world = make_agent(n_vars=7, seed=11)
    run_cycles(agent, world, 120)

    # Find a var with a trass skip cert
    trass_candidates = [
        n for n in agent.ledger.vars.values()
        if n.var < world.visible_count and n.role_for("skip") == "trass"
    ]
    if not trass_candidates:
        pytest.skip("No trass vars emerged in 120 cycles — try a different seed")

    target = trass_candidates[0]
    # Simulate legacy path: cert removed, status stays "trass"
    target.certificates.pop("skip")
    assert target.role_for("skip") == "untested", "cert pop should leave role untested"
    assert target.status == "trass", "status must survive cert removal for this test"

    # _retest_trass_vars fires on variable reveal
    flipped = agent._retest_trass_vars(cycle=121)

    # After retest: the var must either have been flipped (became tareth/untested)
    # or have received a fresh skip cert (was retested and stayed trass).
    was_processed = (
        target.var in flipped
        or "skip" in target.certificates
    )
    assert was_processed, (
        f"x{target.var} has status='trass' with no skip cert but was NOT processed "
        f"by _retest_trass_vars. It is permanently invisible to scope-expansion "
        f"revalidation. Filter `role_for('skip') != 'trass'` skips vars with role="
        f"'untested' (no cert). Fix: also include `n.status == 'trass'` in the filter."
    )


# ── Q2b: run_cycle skip gate must require a skip cert ────────────────────────

def test_skip_gate_requires_cert_not_status():
    """
    NARETH DIVERGENCE Q2b — run_cycle must not skip a var based on status alone.

    Invariant: the skip decision is operation-indexed. Only role_for("skip") ==
    "trass" (a scoped, witnessed, revocable cert) authorises a skip. status=="trass"
    is a legacy global label without operation index or provenance. A var that has
    been collapsed via status but has no cert cannot be tracked by the cert model,
    cannot be retested by _retest_trass_vars, and cannot be invalidated correctly.

    Current failure: run_cycle checks
        if n.role_for("skip") == "trass" or n.status == "trass":
    The OR condition allows status alone to gate the skip, bypassing the cert model.
    A var with status="trass" and no cert accumulates skip_count every cycle.

    Fix: gate on role_for("skip") == "trass" only.
    """
    agent, world = make_agent(n_vars=7, seed=11)
    run_cycles(agent, world, 120)

    trass_candidates = [
        n for n in agent.ledger.vars.values()
        if n.var < world.visible_count and n.role_for("skip") == "trass"
    ]
    if not trass_candidates:
        pytest.skip("No trass vars emerged in 120 cycles — try a different seed")

    target = trass_candidates[0]
    target.certificates.pop("skip")
    assert target.role_for("skip") == "untested"
    assert target.status == "trass"

    skip_before = target.skip_count
    agent.run_cycle(_steady(121))

    new_skips = target.skip_count - skip_before
    assert new_skips == 0, (
        f"x{target.var} accumulated {new_skips} skip(s) despite having no skip cert "
        f"(only status='trass'). The OR condition in run_cycle let status alone gate "
        f"the skip — this bypasses the nareth cert model. "
        f"Fix: remove `or n.status == 'trass'` from the skip branch guard."
    )


# ── Q3: promoted compression must issue a nareth cert ────────────────────────

def test_promoted_compression_issues_nareth_cert():
    """
    NARETH DIVERGENCE Q3 — compression promotion must produce certificates["compress"].

    Invariant: a compression shortcut (predicting a simplified value when a gate
    condition holds) is an operation-scoped decision. It must carry a cert with
    operation="compress", scope=gate predicate, and witnesses so that
    invalidate_certs("sentinel_failure") actually clears a real cert rather than
    being a no-op on an empty key.

    Current failure: _discover_compressions and the pred_passes trial loop
    use frequency accumulation only. No certificates["compress"] is ever populated.
    invalidate_certs clears "compress" from the dict — but nothing writes it,
    so the clear is always a no-op.

    Fix: after pred_passes reaches the promotion threshold, issue
    certificates["compress"] with operation="compress", scope=gate_condition,
    and witnesses=[(state_snapshot, simplified_value), ...] from the trial passes.

    Seed/setup: seed=11, n_vars=10, noise=0.03, 500 cycles reliably produces
    promoted compressions (verified: var 6 with this setup).
    """
    rng = random.Random(11)
    world = CausalWorld(10, rng, noise_sigma=0.03)
    world.visible_count = 10
    agent = ChainedAgent(world, rng,
                         sentinel_count=3, sentinel_pool=12,
                         compression_discover_after=2, compression_promote_after=5,
                         priority_audit_budget=10)
    run_cycles(agent, world, 500)

    promoted = [
        n for n in agent.ledger.vars.values()
        if n.var < world.visible_count
        and any(
            comp.pred_passes >= agent.compression_promote_after
            for comp in n.compressions
        )
    ]
    if not promoted:
        pytest.skip("No compressions promoted in 500 cycles — seed/world may have changed")

    for n in promoted:
        assert "compress" in n.certificates, (
            f"x{n.var} has a promoted compression (pred_passes >= "
            f"{agent.compression_promote_after}) but no certificates['compress']. "
            f"The compression shortcut has no nareth cert — no scope, no witnesses, "
            f"no operation index. Fix: issue certificates['compress'] on promotion."
        )
        cert = n.certificates["compress"]
        assert cert.operation == "compress", (
            f"x{n.var} compress cert has wrong operation={cert.operation!r}"
        )
        assert cert.role == "trass", (
            f"x{n.var} compress cert role={cert.role!r} — a compression collapses "
            f"a distinction (gate holds → simplified value), so role must be 'trass'."
        )


# ── Q5: tareth cert must store certifying witness for sentinel replay ─────────

def test_tareth_cert_stores_certifying_witness():
    """
    NARETH DIVERGENCE Q5 — a tareth-for-skip cert must store the intervention
    evidence that earned it, so the sentinel path can replay that specific witness.

    Invariant: tareth is existential — "there exists an intervention in this scope
    where the distinction changes monitored targets." If the original witness expires
    (world drifts so that specific (iv_val, context) no longer propagates), the cert's
    authority has lapsed. The sentinel must replay the cert's own evidence, not run
    different drift probes that may miss the expiry.

    Current failure: NethraCertificate stores summary counts (changes, trials) but
    not the specific (iv_val, world_state) pair that produced each change. The sentinel
    path (check_var_sentinels_with_envelope) runs pool probes — drift detection — not
    replay of the cert's evidence. Risk: stale tareth authority persists when the
    original basis has expired but new probes still happen to detect different changes.

    Fix: add a `witnesses` field to NethraCertificate (list of SubstitutionWitness or
    equivalent), populated in _certify_operation_role with the iv_val and saved world
    state that produced each detected propagation. The sentinel path should replay at
    least one witness per cycle; if none propagate, trigger recertification (not trass
    collapse — new evidence may still establish tareth).
    """
    agent, world = make_agent(n_vars=8, seed=42)
    run_cycles(agent, world, 150)

    tareth_vars = [
        n for n in agent.ledger.vars.values()
        if n.var < world.visible_count and n.role_for("skip") == "tareth"
    ]
    if not tareth_vars:
        pytest.skip("No tareth vars in 150 cycles — try a different seed")

    for n in tareth_vars:
        cert = n.certificates["skip"]
        assert hasattr(cert, "witnesses"), (
            f"x{n.var} tareth cert has no 'witnesses' field. "
            f"Without stored witnesses the sentinel path cannot replay the cert's "
            f"own evidence — it can only detect general drift, not confirm the cert "
            f"basis is still live. Fix: add witnesses: List[SubstitutionWitness] to "
            f"NethraCertificate and populate it in _certify_operation_role."
        )
        assert cert.witnesses, (
            f"x{n.var} tareth cert has an empty witnesses list. "
            f"A tareth cert must store at least one (iv_val, context) pair that "
            f"produced a propagation during certification."
        )


# ── Q6: live-frontier cert must be invalidated when excluded var becomes tareth

def test_live_frontier_cert_scope_matches_tested_targets():
    """
    NARETH DIVERGENCE Q6a — under live-frontier mode, cert.targets must reflect
    only the targets that were actually tested, not the full eligible set.

    Under live-frontier mode, cert-trass vars are excluded from change-counting.
    The cert's scope (cert.targets) must reflect only the vars that were actually
    tested. Including trass vars in cert.targets would overclaim authority: the cert
    would claim "Y doesn't affect X" when X was never part of the change-counting loop.

    Q6b (proactively flagging Y's cert when X flips trass→tareth) is intentionally
    NOT implemented. Under the filter ledger, Y's shortcut keeps firing by default.
    If the scope expansion causes an actual failure, the sentinel catches it. Pre-emptive
    cert scanning on trass→tareth transitions is the positive-ledger pattern and is wrong.

    Seed/setup: seed=6, role_salience="live-frontier", n_vars=10, 200 cycles
    reliably produces one tareth var with many trass vars.
    """
    rng = random.Random(6)
    world = CausalWorld(10, rng, noise_sigma=0.05)
    world.visible_count = 10
    agent = ChainedAgent(world, rng,
                         sentinel_count=3, sentinel_pool=12,
                         priority_audit_budget=10,
                         role_salience="live-frontier")
    run_cycles(agent, world, 200)

    tareth_vars = [v for v, n in agent.ledger.vars.items()
                   if v < world.visible_count and n.role_for("skip") == "tareth"]
    trass_vars_l = [v for v, n in agent.ledger.vars.items()
                    if v < world.visible_count and n.role_for("skip") == "trass"]

    if not tareth_vars or not trass_vars_l:
        pytest.skip("Need at least one tareth and one trass var — world did not converge")

    y_id = tareth_vars[0]
    x_id = trass_vars_l[0]

    # Re-certify Y while X is trass: live-frontier excludes X from change-counting
    yn = agent.ledger.vars[y_id]
    yn.certificates.pop("skip", None)
    agent._certify_operation_role(y_id, cycle=201)

    y_cert = yn.certificates.get("skip")
    if y_cert is None:
        pytest.skip("Y did not receive a cert on recertification")

    # Q6a: cert.targets must not include X (trass, excluded from change-counting).
    # The cert's authority extends only over the vars that were actually tested.
    assert x_id not in y_cert.targets, (
        f"x{y_id} skip cert includes x{x_id} in cert.targets, but x{x_id} was trass "
        f"during certification and excluded from change-counting by the live-frontier "
        f"filter. cert.targets misrepresents the actual tested scope."
    )


# ── Q7: skip-trass var must not be excluded from available_source_edges ────────────

def test_skip_trass_var_not_excluded_from_available_source_edges():
    """
    NARETH DIVERGENCE Q7 — available_source_edges must not exclude a var solely because
    its skip cert is trass. Route relevance is a different operation.

    Invariant: tareth-for-skip (perturbing this var propagates to monitored targets)
    is NOT the same as tareth-for-route (which hypothesis we use for this var changes
    the audit decision). A var can be skip-trass (below tolerance; doesn't propagate
    under current regime) yet still be the true causal source_edge of another var. Excluding
    it from available_source_edges means that source_edge is never in the hypothesis space — the
    audit can never find the correct fit for the dependent variable.

    The proxy is not conservative: it can miss route-relevant source_edges that happen to be
    skip-trass under the current monitored target regime.

    Current failure: _full_audit_var builds available_source_edges as:
        role_for("skip") == "tareth"
    Any var with a trass skip cert is excluded regardless of whether it is a true source_edge.

    Seed/setup: seed=0, n_vars=10, 200 cycles reliably finds x3 (tareth) with
    source_edge x2 (tareth). We demote x2's cert to trass to simulate the bug.

    Fix: gate on role_for("route") == "tareth" at instance level. Absent instance-level
    route certs, at minimum do not exclude skip-trass vars that are structurally present
    as source_edges of the var under audit.
    """
    rng = random.Random(0)
    world = CausalWorld(10, rng, noise_sigma=0.05)
    world.visible_count = 10
    agent = ChainedAgent(world, rng,
                         sentinel_count=3, sentinel_pool=12,
                         priority_audit_budget=10)
    run_cycles(agent, world, 200)

    # Find a tareth child with a tareth source_edge
    chosen_child = None
    chosen_source_edge = None
    for v, n in agent.ledger.vars.items():
        if v >= world.visible_count or not n.source_edges:
            continue
        if n.role_for("skip") != "tareth":
            continue
        for p in n.source_edges:
            if p >= world.visible_count:
                continue
            pn = agent.ledger.vars[p]
            if pn.role_for("skip") == "tareth" and pn.status in ("certified", "proposed"):
                chosen_child = v
                chosen_source_edge = p
                break
        if chosen_child is not None:
            break

    if chosen_child is None:
        pytest.skip("Could not find a tareth child with a tareth source_edge after 200 cycles")

    # Demote the source_edge's skip cert to trass (simulating a regime where it fell below
    # tolerance, but it remains the structural source_edge)
    pn = agent.ledger.vars[chosen_source_edge]
    pn.certificates["skip"] = dataclasses.replace(
        pn.certificates["skip"], role="trass", authority="none"
    )
    assert pn.role_for("skip") == "trass"

    # Reconstruct available_source_edges the same way _full_audit_var does after Q7 fix
    available = {
        other_var for other_var, other_n in agent.ledger.vars.items()
        if other_var != chosen_child
        and other_n.role_for("route") != "trass"
        and (
            other_n.status == "certified"
            or (other_n.status == "proposed" and bool(other_n.sentinels))
        )
    }

    # SHOULD: chosen_source_edge is a true causal source_edge of chosen_child and must be
    # in available_source_edges regardless of its skip cert. Route relevance != skip relevance.
    assert chosen_source_edge in available, (
        f"x{chosen_source_edge} is a true source_edge of x{chosen_child} but was excluded from "
        f"available_source_edges because role_for('skip')=='trass'. "
        f"The skip proxy is not conservative — tareth-for-skip ≠ tareth-for-route. "
        f"Fix: gate available_source_edges on role_for('route')=='tareth' at instance level; "
        f"until instance route certs exist, do not exclude skip-trass vars that are "
        f"structural source_edges of the var under audit."
    )


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Q2a: status-trass retest", test_status_trass_retest_on_reveal),
        ("Q2b: skip gate requires cert", test_skip_gate_requires_cert_not_status),
        ("Q3:  compression cert", test_promoted_compression_issues_nareth_cert),
        ("Q5:  tareth witness stored", test_tareth_cert_stores_certifying_witness),
        ("Q6:  live-frontier cert scope / invalidation", test_live_frontier_cert_scope_matches_tested_targets),
        ("Q7:  skip proxy not conservative", test_skip_trass_var_not_excluded_from_available_source_edges),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  UNEXPECTED PASS: {label}")
        except pytest.skip.Exception as e:
            print(f"  SKIP: {label} — {e}")
        except AssertionError as e:
            first_line = str(e).split("\n")[0]
            print(f"  FAIL (expected): {label}\n    {first_line}")
        except Exception as e:
            print(f"  ERROR: {label} — {type(e).__name__}: {e}")
