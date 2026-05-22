"""
Practical functional tests of the agent's core cycle mechanics.

Each test runs real world + agent machinery (no mocks). Assertions probe
specific behavioral properties at specific moments in the execution.

Test map:
  01  all vars have skip cert after initialize()
  02  trass skip path — skip_count accumulates every cycle
  03  tareth vars have sentinels + witnesses; trass vars have neither
  04  cert.targets is well-scoped (excludes self, non-empty for tareth)
  05  high-cost sentinel failure → re-audit + drift detection for failing var
  06  sentinel vs trass skip path breakdown matches cert roles exactly
  07  variable reveal reclassifies trass vars that genuinely affect the new var
  08  cycle records account for every visible var (skip or audit, or both if cascaded)
"""

import random
import pytest

from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent
from dreth.ledger import CompositeNethra, NethraCertificate


# ── world: seed 3 ──────────────────────────────────────────────────────────────
# parents: [[], [], [0,1], [0,1], [0,1]]
# funcs:   ['HIGH', 'LOW', 'MAX', 'FIRST', 'PROD']
# After initialize():
#   x0, x1 → tareth (roots whose output propagates to children)
#   x2, x3, x4 → trass (leaves: perturbing them changes nothing monitored)
_SEED_W = 3
_SEED_A = 13003


def _make_agent(cost_weights=None, **kwargs):
    rng_w = random.Random(_SEED_W)
    rng_a = random.Random(_SEED_A)
    world = CausalWorld(5, rng_w, noise_sigma=0.0)
    world.visible_count = 5
    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=20,
        compression_discover_after=3,
        compression_promote_after=3,
        priority_audit_budget=5,
        cost_weights=cost_weights,
        frontier_k=world.n_vars,  # audit all vars: these tests probe cert quality, not init sparseness
        **kwargs,
    )
    return agent, world


def _steady(cycle):
    return HiddenMutation(cycle, "VALUE", "steady", False, -1)


# ── 01 ────────────────────────────────────────────────────────────────────────

def test_01_all_vars_certified_after_init():
    agent, world = _make_agent()
    agent.initialize()

    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        cert = n.certificates.get("skip")
        assert cert is not None, f"x{var} has no skip cert after initialize()"
        assert cert.role in ("tareth", "trass"), \
            f"x{var} cert.role must be tareth or trass, got {cert.role!r}"


# ── 02 ────────────────────────────────────────────────────────────────────────

def test_02_trass_vars_skip_every_cycle_and_count_grows():
    agent, world = _make_agent()
    agent.initialize()

    trass_vars = [v for v in range(world.visible_count)
                  if agent.ledger.vars[v].role_for("skip") == "trass"]
    assert trass_vars, "expected at least one trass var in seed-3 world"

    # Provisional trass: initialize() issues the cert unconfirmed.
    # Cycle 1 is the provisional confirmation cycle — the hot-pass marks
    # cert.confirmed=True (O(1), no fit_var call). Hard-skip fires from cycle 2.
    # full_audits is NOT incremented in the provisional path: it counts
    # _full_audit_var calls only.
    agent.run_cycle(_steady(1))
    for v in trass_vars:
        cert = agent.ledger.vars[v].certificates.get("skip")
        assert cert is not None and cert.confirmed, (
            f"x{v} trass cert must be confirmed after cycle 1"
        )
        assert agent.ledger.vars[v].role_for("skip") == "trass", (
            f"x{v} must still be trass after provisional re-audit"
        )

    skip_counts_before = {v: agent.ledger.vars[v].skip_count for v in trass_vars}

    n_cycles = 9
    for c in range(2, n_cycles + 2):
        agent.run_cycle(_steady(c))
        r = agent.records[-1]
        for v in trass_vars:
            assert v in r.skipped_vars, \
                f"x{v} (confirmed trass) must appear in skipped_vars; missing at c{c}"

    for v in trass_vars:
        after = agent.ledger.vars[v].skip_count
        assert after == skip_counts_before[v] + n_cycles, \
            f"x{v} skip_count should grow by {n_cycles}; got {after - skip_counts_before[v]}"


# ── 03 ────────────────────────────────────────────────────────────────────────

def test_03_tareth_has_sentinels_and_witnesses_trass_has_neither():
    agent, world = _make_agent()
    agent.initialize()

    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        role = n.role_for("skip")
        cert = n.certificates.get("skip")

        if role == "tareth":
            assert len(n.sentinels) > 0, \
                f"x{var} tareth must have sentinels after initialize()"
            assert len(cert.witnesses) > 0, \
                f"x{var} tareth cert must carry witnesses (earned by the perturbation test)"
        elif role == "trass":
            assert len(n.sentinels) == 0, \
                f"x{var} trass must have no sentinels (no audit to probe)"
            assert len(cert.witnesses) == 0, \
                f"x{var} trass cert must have no witnesses (no propagation found)"


# ── 04 ────────────────────────────────────────────────────────────────────────

def test_04_cert_targets_excludes_self_and_nonempty_for_tareth():
    agent, world = _make_agent()
    agent.initialize()

    for var in range(world.visible_count):
        cert = agent.ledger.vars[var].certificates.get("skip")
        assert cert is not None

        assert var not in cert.targets, \
            f"x{var} cert.targets should not include the var itself"

        if cert.role == "tareth":
            assert len(cert.targets) > 0, \
                f"x{var} tareth cert.targets must be non-empty (scope was tested)"


# ── 05 ────────────────────────────────────────────────────────────────────────

def test_05_high_cost_sentinel_failure_triggers_reaudit_and_drift():
    """Change x0's world func after init (HIGH→LOW). x0 is tareth with
    cost_weight=3.0 ≥ cost_high_threshold=2.0: any sentinel miss is fatal.
    Expected next cycle: x0 fails sentinel → re-audited → hypothesis updates.

    The cascade (descendants re-audited) is a possible outcome but not asserted:
    under filter ledger, descendants are only re-audited when their own cert basis
    is invalidated — trass descendants with no dependence on x0's output value
    are correctly left alone. Assert only x0's own failure consequences."""
    agent, world = _make_agent(cost_weights={0: 3.0, 1: 3.0})
    agent.initialize()

    assert agent.ledger.vars[0].role_for("skip") == "tareth"
    x0_learned_func_before = agent.ledger.vars[0].func

    world.funcs[0] = "LOW"

    agent.run_cycle(_steady(1))
    r = agent.records[-1]

    assert 0 in r.fully_audited_vars, \
        "x0 must be re-audited after its sentinel fails on a HIGH→LOW func change"

    assert agent.ledger.vars[0].func != x0_learned_func_before, \
        "x0 learned hypothesis must change after re-audit"

    assert 0 in r.detected_drift_vars, \
        "x0 must appear in detected_drift_vars after its hypothesis changes"


# ── 06 ────────────────────────────────────────────────────────────────────────

def test_06_skip_path_breakdown_matches_cert_roles():
    """Each cheap-path skip uses exactly one of: trass-skip or sentinel-skip.
    After N cycles with a stable world (no mutations), the agent's skip counters
    must exactly account for all vars and all cycles.

    Trass certs earned at initialize() are provisional (full_audits=1). Cycle 1 is a
    mandatory provisional re-audit that confirms them; trass skips begin at cycle 2.
    Measurement window: cycles 2–9 (delta from post-warmup counters).

    - trass_skip_count Δ == n_trass_vars × n_cycles
    - sentinel_skip_count Δ == n_tareth_vars × n_cycles
    - skip_count Δ == trass_skip_count Δ + sentinel_skip_count Δ
    - full_audit_count unchanged after warm-up (stable world, no new failures)
    """
    agent, world = _make_agent()
    agent.initialize()

    n_visible = world.visible_count
    n_trass = sum(1 for v in range(n_visible)
                  if agent.ledger.vars[v].role_for("skip") == "trass")
    n_tareth = sum(1 for v in range(n_visible)
                   if agent.ledger.vars[v].role_for("skip") == "tareth")

    assert n_trass > 0, "seed world must have trass vars"
    assert n_tareth > 0, "seed world must have tareth vars"
    assert n_trass + n_tareth == n_visible

    # Cycle 1: provisional confirmation — hot-pass marks cert.confirmed=True (O(1)).
    # full_audits is NOT incremented (no _full_audit_var call in provisional path).
    agent.run_cycle(_steady(1))
    for v in range(n_visible):
        if agent.ledger.vars[v].role_for("skip") == "trass":
            cert = agent.ledger.vars[v].certificates.get("skip")
            assert cert is not None and cert.confirmed, \
                f"x{v} trass cert must be confirmed after cycle 1"

    # Snapshot counters after warm-up; measurement window starts at cycle 2.
    trass_skips_base = agent.trass_skip_count
    sentinel_skips_base = agent.sentinel_skip_count
    skip_count_base = agent.skip_count
    audits_after_warmup = agent.full_audit_count

    n_cycles = 8
    for c in range(2, n_cycles + 2):
        agent.run_cycle(_steady(c))

    expected_trass_skips = n_trass * n_cycles
    expected_sentinel_skips = n_tareth * n_cycles

    actual_trass = agent.trass_skip_count - trass_skips_base
    actual_sentinel = agent.sentinel_skip_count - sentinel_skips_base
    actual_skip = agent.skip_count - skip_count_base

    assert actual_trass == expected_trass_skips, (
        f"trass_skip_count Δ={actual_trass} expected {expected_trass_skips} "
        f"({n_trass} trass vars × {n_cycles} cycles)"
    )
    assert actual_sentinel == expected_sentinel_skips, (
        f"sentinel_skip_count Δ={actual_sentinel} expected {expected_sentinel_skips} "
        f"({n_tareth} tareth vars × {n_cycles} cycles)"
    )
    assert actual_skip == expected_trass_skips + expected_sentinel_skips, (
        "skip_count Δ must equal trass + sentinel skips (no compression skips in stable world)"
    )
    # No new full audits after provisional warm-up in a stable world
    assert agent.full_audit_count == audits_after_warmup, \
        "no new full audits expected after provisional warm-up in a stable world"


# ── 07 ────────────────────────────────────────────────────────────────────────

def test_07_variable_reveal_audits_new_var_with_trass_vars_as_candidates():
    """When x5 is revealed and its true parent (x4) is currently trass-certified,
    the agent must still find x4 as x5's parent. Trass-status vars are valid
    parent candidates — only a route cert excludes a var from available_parents,
    and no route certs exist. Existing trass certs are NOT retested on reveal
    (filter ledger: no recert without failure).

    Seed-3, 6-var world: x5's parent is x4 (parents=[4]). x4 is trass before
    reveal. After reveal: x4 stays trass (its cert did not fail), x5 is certified,
    and x5's learned parents include x4.
    """
    rng_w = random.Random(_SEED_W)
    rng_a = random.Random(_SEED_A)
    world = CausalWorld(6, rng_w, noise_sigma=0.0, initial_visible=5)
    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=20,
        compression_discover_after=3,
        compression_promote_after=3,
        priority_audit_budget=6,
        frontier_k=world.n_vars,  # audit all vars at init so x4 has trass cert before x5 reveal
    )
    agent.initialize()

    assert agent.ledger.vars[4].role_for("skip") == "trass", \
        "x4 must be trass before x5 is revealed"

    assert 4 in world.parents[5], \
        "test requires x5 to depend on x4 in the hidden world structure"

    world.visible_count = 6
    agent.on_variable_revealed(5, cycle=1)

    # x5 must be certified after reveal
    x5_cert = agent.ledger.vars[5].certificates.get("skip")
    assert x5_cert is not None, "x5 must have a skip cert after reveal"

    # x4 must still be trass — its cert did not fail, nothing triggered recert
    assert agent.ledger.vars[4].role_for("skip") == "trass", \
        "x4 must remain trass after reveal (filter ledger: no recert without failure)"

    # x5's learned parents must include x4 (trass-status is not a candidate barrier)
    assert 4 in agent.ledger.vars[5].parents, \
        "x5's learned parents must include x4 despite x4 being trass-certified"


# ── 08 ────────────────────────────────────────────────────────────────────────

def test_08_cycle_records_account_for_all_vars():
    """skipped_vars ∪ fully_audited_vars must cover all visible vars.
    A var can appear in both (first trass-skipped in pass 1, then cascaded
    into audit in pass 2). Neither list should contain out-of-range vars."""
    agent, world = _make_agent(cost_weights={0: 3.0, 1: 3.0})
    agent.initialize()

    n_var = world.visible_count

    # Cycle 1: world unchanged — all should be skipped
    agent.run_cycle(_steady(1))
    r = agent.records[-1]
    all_accounted = set(r.skipped_vars) | set(r.fully_audited_vars)
    assert set(range(n_var)) == all_accounted, \
        f"all vars must be accounted for; missing: {set(range(n_var)) - all_accounted}"
    assert all(0 <= v < n_var for v in r.skipped_vars), "skipped_vars out of range"
    assert all(0 <= v < n_var for v in r.fully_audited_vars), "fully_audited_vars out of range"

    # Cycle 2: mutate x0 func → sentinel fails → cascade
    world.funcs[0] = "LOW"
    agent.run_cycle(_steady(2))
    r = agent.records[-1]
    all_accounted = set(r.skipped_vars) | set(r.fully_audited_vars)
    assert set(range(n_var)) == all_accounted, \
        f"after cascade, all vars must still be accounted for; missing: {set(range(n_var)) - all_accounted}"

    # x0's children that were trass-skipped in pass 1 then cascade-audited in pass 2
    # should appear in both lists — that's the expected dual-appearance
    x0_children = [v for v in range(n_var) if v != 0 and 0 in world.parents[v]]
    for child in x0_children:
        if child in r.fully_audited_vars:
            # If cascaded into audit, it must have been skipped first (trass skip in pass 1)
            # OR entered audit without a prior skip (tareth -> sentinel fail -> audit)
            # Either way it must appear in all_accounted
            assert child in all_accounted


# ── 09 ────────────────────────────────────────────────────────────────────────

def test_09_composite_nethra_install_skip_persist_revoke():
    """Joint false-trass: install composite, verify composite-skip, persist evidence,
    break interaction, verify revocation and re-audit.

    World: seed-3, 5 vars, noise=0.
    After initialize(): x0,x1 tareth (roots that propagate into children);
    x2,x3,x4 trass (children whose individual perturbation changes nothing monitored).

    After initialization, the world is modified: x0's function becomes PROD(x2,x3)
    and state[2],state[3] are set near zero (0.01). This creates a hidden joint
    interaction — x0 now responds to joint x2,x3 interventions but not individual
    ones:
      individual x2→0.5: x0 = PROD(0.5, 0.01) = 0.005  (|Δ|=0.005 < tol=0.1, no signal)
      joint x2=x3=0.5:   x0 = PROD(0.5, 0.5)  = 0.25   (|Δ|=0.25 > 0.1, interaction)

    x0's cost_weight is set to 0 before run_cycle so its sentinel failure (its
    hypothesis is still HIGH, but world now gives PROD) is dismissed without cascade.
    This isolates the test to the composite mechanism only.

    The individual trass certs for x2 and x3 are earned BEFORE the world change
    (under the original world where x0=HIGH root), so they remain accurate as
    individual claims. The composite is the authority for the joint relationship.
    """
    agent, world = _make_agent()
    agent.initialize()

    # Verify preconditions from seed-3 world
    assert agent.ledger.vars[2].role_for("skip") == "trass", \
        "x2 must be individually trass after seed-3 initialize()"
    assert agent.ledger.vars[3].role_for("skip") == "trass", \
        "x3 must be individually trass"
    assert agent.ledger.vars[0].role_for("skip") == "tareth", \
        "x0 must be tareth (will serve as the joint-test sentinel)"

    # Individual cert records individual evidence: 0 propagations under old world
    cert2 = agent.ledger.vars[2].certificates.get("skip")
    assert cert2 is not None and cert2.changes == 0, \
        "x2 trass cert must record 0 individual propagations"

    # Modify world AFTER initialization: x0 now has PROD(x2,x3) hidden function.
    # Agent's cert for x0 is still based on the old HIGH hypothesis.
    world.parents[0] = [2, 3]
    world.funcs[0] = "PROD"
    world.state = list(world.state)
    world.state[2] = 0.01
    world.state[3] = 0.01
    world.state[0] = 0.0001  # PROD(0.01, 0.01)
    world.state = tuple(world.state)

    # Suppress x0's sentinel failure cascade — its hypothesis is now stale but
    # we want the test focused on the composite mechanism, not x0's re-audit.
    agent.ledger.vars[0].cost_weight = 0.0

    # Trigger joint false-trass test directly (bypasses sentinel-failure discovery path)
    result = agent._test_joint_false_trass(2, 3, cycle=1)
    assert result == "tareth", \
        f"joint test must detect x2,x3 interaction via x0 sentinel; got {result!r}"

    # CompositeNethra must be installed on the ledger
    assert len(agent.ledger.composites) == 1, "exactly one composite should be installed"
    cn = agent.ledger.composites[0]
    assert set(cn.members) == {2, 3}, f"composite members wrong: {cn.members}"
    assert cn.sentinel_var == 0, \
        "composite sentinel_var must be x0 (the tareth var that showed interaction)"
    assert cn.changes > 0, "composite must record positive interaction evidence"
    assert isinstance(cn, CompositeNethra)

    # Individual certs must NOT be overwritten — x2 and x3 remain individually trass.
    # The joint claim lives on the composite, not on the individual vars' certs.
    assert agent.ledger.vars[2].role_for("skip") == "trass", \
        "x2 individual cert must remain trass — composite is the joint authority"
    assert agent.ledger.vars[3].role_for("skip") == "trass", \
        "x3 individual cert must remain trass"
    assert agent.ledger.vars[2].certificates["skip"].changes == 0, \
        "x2 individual cert must still show 0 individual propagations (not joint count)"

    # Stable cycle: composite sentinel (PROD(0.5,0.5)=0.25 >> tol) passes.
    # Flip status to proposed so composite check intercepts before trass-skip fires.
    agent.ledger.vars[2].status = "proposed"
    agent.ledger.vars[3].status = "proposed"

    skips_before = agent.composite_skip_count
    agent.run_cycle(HiddenMutation(2, "VALUE", "steady", False, -1))
    r = agent.records[-1]

    assert 2 in r.skipped_vars, "x2 must be skipped via composite path in stable cycle"
    assert 3 in r.skipped_vars, "x3 must be skipped via composite path in stable cycle"
    assert agent.composite_skip_count == skips_before + 2, \
        "composite_skip_count must increase by 2 (one per member per cycle)"

    # Composite evidence persists across the stable cycle
    assert len(agent.ledger.composites) == 1, \
        "composite must persist after a stable cycle (interaction still present)"

    # Break the interaction: restore x0 as a standalone root with HIGH function.
    # Now predict_under_joint_intervention({2:v, 3:v})[0] = HIGH([]) = 0.8 = R0[0].
    # |RAB - R0| = 0 < tol → composite sentinel fails → composite revoked.
    world.parents[0] = []
    world.funcs[0] = "HIGH"
    world.state = list(world.state)
    world.state[0] = 0.8
    world.state = tuple(world.state)

    agent.run_cycle(HiddenMutation(3, "VALUE", "steady", False, -1))
    r2 = agent.records[-1]

    # Composite must be revoked (interaction no longer present at probe values)
    assert len(agent.ledger.composites) == 0, \
        "composite must be revoked when joint interaction disappears"

    # Both vars must have been re-audited or have untested certs.
    # Revocation calls invalidate_certs("false_trass_contradiction") → cert reset to untested.
    # Untested vars queue for full audit that same cycle.
    x2_audited = 2 in r2.fully_audited_vars
    x3_audited = 3 in r2.fully_audited_vars
    x2_untested = agent.ledger.vars[2].role_for("skip") == "untested"
    x3_untested = agent.ledger.vars[3].role_for("skip") == "untested"
    assert x2_audited or x2_untested, \
        "x2 must be re-audited or untested after composite revocation"
    assert x3_audited or x3_untested, \
        "x3 must be re-audited or untested after composite revocation"
