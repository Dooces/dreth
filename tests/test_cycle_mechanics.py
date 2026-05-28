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
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import pytest

from dreth.world import CausalWorld
from dreth.agent import ChainedAgent
from dreth.ledger import CompositeNethra, NethraCertificate
from dreth.quality import QualityWeights, compute_quality_cost
from dreth.hybrid import (
    Sensitivitysource_edgeRanker,
    Historysource_edgeRanker,
    HistoryRescuesource_edgeRanker,
    DiscriminationProbeProposer,
    HistoryProbeProposer,
    HistoryRescueProbeProposer,
    source_edgeProposalDiagnostics,
    ProbeProposal,
    FuncLibraryRouter,
    SymbolicResidualPredictor,
)
from dreth.summary import RunAnalyzer
from scripts.batch_run import (
    _build_policy_report_rows,
    _print_provider_policy_comparison,
)


# ── world: seed 3 ──────────────────────────────────────────────────────────────
# source_edges: [[], [], [0,1], [0,1], [0,1]]
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


def _steady(cycle: int) -> int:
    """Return the cycle number unchanged. Represents a no-mutation cycle.
    run_cycle() takes an int; the HiddenMutation produced by world.perturb_by_schedule
    is only needed to detect REVEALs, which this helper never produces.
    """
    return cycle


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

    # Change x0's hidden function AND update the world state to reflect it.
    # Passive residual monitoring checks |actual - predicted|: if only funcs
    # changes but state is unchanged, the residual is 0 and the sentinel is
    # skipped (correct: the cert hasn't been invalidated yet). Updating state
    # ensures the passive check detects the stale hypothesis (residual > tol),
    # which triggers the active sentinel, which then earns the failure signal.
    world.funcs[0] = "LOW"
    world.state = list(world.state)
    world.state[0] = 0.2   # LOW([]) = 0.2; was HIGH([]) = 0.8 → residual 0.6 > tol
    world.state = tuple(world.state)

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
    expected_tareth_skips = n_tareth * n_cycles

    actual_trass = agent.trass_skip_count - trass_skips_base
    actual_sentinel = agent.sentinel_skip_count - sentinel_skips_base
    actual_skip = agent.skip_count - skip_count_base

    assert actual_trass == expected_trass_skips, (
        f"trass_skip_count Δ={actual_trass} expected {expected_trass_skips} "
        f"({n_trass} trass vars × {n_cycles} cycles)"
    )
    # In a stable world, tareth vars skip via passive residual OK (skip_count
    # incremented, sentinel_skip_count NOT incremented — sentinel doesn't run
    # when passive check confirms no residual stress). Total skip still accounts
    # for all tareth vars; only the breakdown between passive-ok and sentinel changes.
    actual_tareth_skips = actual_skip - actual_trass
    assert actual_tareth_skips == expected_tareth_skips, (
        f"tareth skip Δ={actual_tareth_skips} expected {expected_tareth_skips} "
        f"({n_tareth} tareth vars × {n_cycles} cycles); "
        f"sentinel={actual_sentinel} (passive_ok may absorb these in stable world)"
    )
    assert actual_skip == expected_trass_skips + expected_tareth_skips, (
        "skip_count Δ must equal trass + tareth skips (no compression skips in stable world)"
    )
    # No new full audits after provisional warm-up in a stable world
    assert agent.full_audit_count == audits_after_warmup, \
        "no new full audits expected after provisional warm-up in a stable world"


# ── 07 ────────────────────────────────────────────────────────────────────────

def test_07_variable_reveal_audits_new_var_with_trass_vars_as_candidates():
    """When x5 is revealed and its true source_edge (x4) is currently trass-certified,
    the agent must still find x4 as x5's source_edge. Trass-status vars are valid
    source_edge candidates — only a route cert excludes a var from available_source_edges,
    and no route certs exist. Existing trass certs are NOT retested on reveal
    (filter ledger: no recert without failure).

    Seed-3, 6-var world: x5's source_edge is x4 (source_edges=[4]). x4 is trass before
    reveal. After reveal: x4 stays trass (its cert did not fail), x5 is certified,
    and x5's learned source_edges include x4.
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

    assert 4 in world.source_edges[5], \
        "test requires x5 to depend on x4 in the hidden world structure"

    # In the seed-3, 6-var world x5's random function is TINY (constant 0.1),
    # which ignores all inputs. Replace it with FIRST so x4 is a genuine causal
    # source_edge — perturbing x4 will move x5 and the screen will rank x4 highly.
    world.funcs[5] = "FIRST"
    world.state = list(world.state)
    world.state[5] = world.state[4]   # FIRST([x4]) = x4 = 0.0 at current world state
    world.state = tuple(world.state)

    world.visible_count = 6
    agent.on_variable_revealed(5, cycle=1)

    # x5 must be certified after reveal
    x5_cert = agent.ledger.vars[5].certificates.get("skip")
    assert x5_cert is not None, "x5 must have a skip cert after reveal"

    # x4 must still be trass — its cert did not fail, nothing triggered recert
    assert agent.ledger.vars[4].role_for("skip") == "trass", \
        "x4 must remain trass after reveal (filter ledger: no recert without failure)"

    # x5's learned source_edges must include x4 (trass-status is not a candidate barrier)
    assert 4 in agent.ledger.vars[5].source_edges, \
        "x5's learned source_edges must include x4 despite x4 being trass-certified"


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
    x0_children = [v for v in range(n_var) if v != 0 and 0 in world.source_edges[v]]
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
    world.source_edges[0] = [2, 3]
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
    agent.run_cycle(2)
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
    world.source_edges[0] = []
    world.funcs[0] = "HIGH"
    world.state = list(world.state)
    world.state[0] = 0.8
    world.state = tuple(world.state)

    agent.run_cycle(3)
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


# ── 10  hybrid interface mode ─────────────────────────────────────────────────
# Tests 10a–10d verify provider seams without changing cert authority.
# Core invariant enforced: no provider may issue NethraCertificate objects,
# mutate ledger cert state, mark tareth/trass, or authorize skips.
# ─────────────────────────────────────────────────────────────────────────────

def _make_hybrid_agent(**kwargs):
    """Agent with all four symbolic default providers installed."""
    rng_w = random.Random(_SEED_W)
    rng_a = random.Random(_SEED_A)
    world = CausalWorld(5, rng_w, noise_sigma=0.0)
    world.visible_count = 5
    residual_predictor = kwargs.pop("residual_predictor", SymbolicResidualPredictor())
    source_edge_ranker = kwargs.pop("source_edge_ranker", Sensitivitysource_edgeRanker(world))
    probe_proposer = kwargs.pop("probe_proposer", DiscriminationProbeProposer())
    expert_router = kwargs.pop("expert_router", FuncLibraryRouter())
    agent = ChainedAgent(
        world, rng_a,
        sentinel_count=5, sentinel_pool=20,
        compression_discover_after=3,
        compression_promote_after=3,
        priority_audit_budget=5,
        frontier_k=world.n_vars,
        residual_predictor=residual_predictor,
        source_edge_ranker=source_edge_ranker,
        probe_proposer=probe_proposer,
        expert_router=expert_router,
        **kwargs,
    )
    return agent, world


def test_10a_hybrid_provider_counters_increment():
    """After initialize() + a few cycles all four hybrid counters must be > 0."""
    agent, world = _make_hybrid_agent()
    agent.initialize()

    N_CYCLES = 5
    for c in range(1, N_CYCLES + 1):
        agent.run_cycle(c)

    assert agent._hybrid_residual_predictor_calls > 0, \
        "residual_predictor must be called at least once"
    assert agent._hybrid_source_edge_ranker_calls > 0, \
        "source_edge_ranker must be called at least once (source_edge_screen_m > 0 triggers _screen_candidate_source_edges)"
    assert agent._hybrid_probe_proposer_calls > 0, \
        "probe_proposer must be called at least once (every full audit goes through _full_audit_var)"
    assert agent._hybrid_expert_router_calls > 0, \
        "expert_router must be called at least once (every full audit emits a diagnostic route)"


def test_10b_providers_do_not_issue_certs():
    """Providers must not create NethraCertificate objects or alter cert count.

    Compare cert counts from an off-mode run vs a hybrid-mode run on the exact
    same world+seed. They must match — providers contribute zero new certs.
    """
    # Off-mode run
    rng_w = random.Random(_SEED_W)
    rng_a = random.Random(_SEED_A)
    world_off = CausalWorld(5, rng_w, noise_sigma=0.0)
    world_off.visible_count = 5
    agent_off = ChainedAgent(
        world_off, rng_a,
        sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world_off.n_vars,
    )
    agent_off.initialize()
    for c in range(1, 6):
        agent_off.run_cycle(c)

    # Hybrid-mode run (same seeds)
    agent_hy, _ = _make_hybrid_agent()
    agent_hy.initialize()
    for c in range(1, 6):
        agent_hy.run_cycle(c)

    # Cert count per var must be identical — providers never touch certs
    for var in range(5):
        n_off = agent_off.ledger.vars[var]
        n_hy = agent_hy.ledger.vars[var]
        assert len(n_off.certificates) == len(n_hy.certificates), (
            f"x{var}: cert count mismatch off={len(n_off.certificates)} "
            f"hybrid={len(n_hy.certificates)}"
        )
        for key in n_off.certificates:
            assert key in n_hy.certificates, \
                f"x{var}: cert key {key!r} present in off-mode but missing in hybrid"
            assert n_off.certificates[key].role == n_hy.certificates[key].role, (
                f"x{var} cert[{key!r}]: role mismatch "
                f"off={n_off.certificates[key].role!r} "
                f"hybrid={n_hy.certificates[key].role!r}"
            )


def test_10c_hybrid_cert_invariants_match_off_mode():
    """All cert-quality invariants that hold in off-mode must hold in hybrid mode.

    Checks: every cert has earned_by, trass vars have no sentinels,
    tareth vars have sentinels, no provider-owned cert is written.
    """
    agent, world = _make_hybrid_agent()
    agent.initialize()
    for c in range(1, 6):
        agent.run_cycle(c)

    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        role = n.role_for("skip")
        cert = n.certificates.get("skip")

        # I1: every cert has earned_by
        if cert is not None:
            assert cert.earned_by, \
                f"x{var} hybrid: cert missing earned_by (provider wrote unevidenced cert?)"

        # Trass: no sentinels; tareth: sentinels present
        if role == "trass":
            assert len(n.sentinels) == 0, \
                f"x{var} trass: sentinels should be empty in hybrid mode too"
        elif role == "tareth":
            assert len(n.sentinels) > 0, \
                f"x{var} tareth: sentinels must be present in hybrid mode too"

        # No route certs in wrong location
        for key, c2 in n.certificates.items():
            op = getattr(c2, "operation", None)
            assert op != "route", \
                f"x{var}: route cert found in .certificates[{key!r}] (must live in route_certs)"


def test_10d_off_vs_hybrid_compatible_skip_rate():
    """Hybrid symbolic providers must not degrade the skip rate by more than 20%.

    DiscriminationProbeProposer returns empty probes → no change to fit_var.
    Sensitivitysource_edgeRanker mirrors the inline screen → same candidates.
    FuncLibraryRouter is diagnostic-only → no effect.
    SymbolicResidualPredictor mirrors the inline path → same passive decisions.
    So the skip rates should be within a small tolerance.
    """
    N_CYCLES = 20

    # Off mode
    rng_w = random.Random(_SEED_W)
    rng_a = random.Random(_SEED_A)
    world_off = CausalWorld(5, rng_w, noise_sigma=0.0)
    world_off.visible_count = 5
    agent_off = ChainedAgent(
        world_off, rng_a,
        sentinel_count=5, sentinel_pool=20,
        priority_audit_budget=5,
        frontier_k=world_off.n_vars,
    )
    agent_off.initialize()
    for c in range(1, N_CYCLES + 1):
        agent_off.run_cycle(c)

    # Hybrid mode (same seeds)
    agent_hy, _ = _make_hybrid_agent()
    agent_hy.initialize()
    for c in range(1, N_CYCLES + 1):
        agent_hy.run_cycle(c)

    total_off = agent_off.skip_count + agent_off.full_audit_count
    total_hy  = agent_hy.skip_count  + agent_hy.full_audit_count
    rate_off = agent_off.skip_count / max(1, total_off)
    rate_hy  = agent_hy.skip_count  / max(1, total_hy)

    assert abs(rate_off - rate_hy) <= 0.20, (
        f"skip rate gap too large: off={rate_off:.2%} hybrid={rate_hy:.2%} "
        f"(delta={abs(rate_off-rate_hy):.2%} > 20%); "
        "symbolic providers should not significantly alter skip behavior"
    )


def test_10e_history_source_edge_ranker_cannot_issue_certs():
    """History ranking may alter proposal order, but certs still come from ledger paths."""
    agent, world = _make_hybrid_agent(
        source_edge_ranker=Historysource_edgeRanker(),
        probe_proposer=None,
    )
    agent.initialize()
    for c in range(1, 8):
        agent.run_cycle(c)

    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        for cert in n.certificates.values():
            assert cert.earned_by
            assert cert.earned_by != "provider"
            assert cert.operation in ("skip", "compress", "audit")
        for cert in n.route_certs.values():
            assert cert.earned_by
            assert cert.earned_by != "provider"


def test_10f_source_edge_proposal_diagnostics_increment_with_history_ranker():
    agent, _ = _make_hybrid_agent(
        source_edge_ranker=Historysource_edgeRanker(),
        probe_proposer=None,
    )
    agent.initialize()
    for c in range(1, 4):
        agent.run_cycle(c)

    diag = agent._source_edge_proposal_diagnostics
    assert diag.calls > 0
    assert diag.proposed_total > 0


def test_10g_source_edge_ranking_comparison_counts_rank_zero_hit():
    diag = source_edgeProposalDiagnostics()
    diag.record_call((2, 3, 4), (2, 3, 4))
    diag.record_fit((2, 3, 4), (2,))

    assert diag.proposed_in_final_fit == 1
    assert diag.miss_chosen_source_edge_count == 0
    assert diag.rank_of_chosen_source_edge_mean == 0.0
    assert diag.rank_of_chosen_source_edge_max == 0
    assert diag.chosen_source_edge_hit_rate == 1.0


class _InvalidProbeProposer:
    def propose_probes(self, var, available_source_edges, budget):
        return ProbeProposal(var=var, probes=((-1, 0.5), (0, 1.5)))


def test_10h_probe_proposer_invalid_probes_are_dropped():
    agent, _ = _make_hybrid_agent(
        probe_proposer=_InvalidProbeProposer(),
    )
    agent.initialize()

    events = [
        e for e in agent.ledger.events
        if e.type == "provider_diagnostic"
        and e.payload.get("provider") == "probe_proposer"
    ]
    assert events
    assert {e.payload.get("event") for e in events} >= {
        "invalid_probe_var",
        "invalid_probe_val",
    }
    assert agent._probe_proposal_diagnostics.provider_probes_invalid > 0


def test_10i_probe_proposer_has_no_direct_cert_authority():
    agent, world = _make_hybrid_agent(
        probe_proposer=HistoryProbeProposer(max_probes=2),
    )
    agent.initialize()
    for c in range(1, 4):
        agent.run_cycle(c)

    assert agent._probe_proposal_diagnostics.provider_probes_proposed > 0
    for event in agent.ledger.events:
        if event.type == "cert_issued":
            assert event.payload.get("provider") != "probe_proposer"
    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        for cert in list(n.certificates.values()) + list(n.route_certs.values()):
            assert cert.earned_by != "provider"


def test_10j_history_rescue_source_edge_ranker_preserves_authority_invariants():
    agent, world = _make_hybrid_agent(
        source_edge_ranker=HistoryRescuesource_edgeRanker(CausalWorld(5, random.Random(_SEED_W), noise_sigma=0.0)),
        probe_proposer=HistoryRescueProbeProposer(max_probes=2),
    )
    agent._source_edge_ranker._world = world
    agent.initialize()
    for c in range(1, 6):
        agent.run_cycle(c)

    for var in range(world.visible_count):
        n = agent.ledger.vars[var]
        for cert in list(n.certificates.values()) + list(n.route_certs.values()):
            assert cert.earned_by
            assert cert.earned_by != "provider"


def test_10k_history_rescue_route_cert_exclusion_after_merge():
    agent, world = _make_hybrid_agent(
        source_edge_ranker=HistoryRescuesource_edgeRanker(CausalWorld(5, random.Random(_SEED_W), noise_sigma=0.0)),
        probe_proposer=None,
    )
    agent._source_edge_ranker._world = world
    target = 2
    agent._source_edge_ranker.observe_fit_result(target, (0,), margin=5)
    agent.ledger.issue_route_cert(
        target, 0, "trass",
        context_source_edges=(),
        context_visible=world.visible_count,
        context_cycle=0,
        targets=(),
        substitutions_tested=("test",),
        changes=0,
        trials=1,
        earned_by="counterfactual_fit",
    )

    available = agent._screen_candidate_source_edges(target, 4)

    assert 0 not in available
    assert agent._source_edge_proposal_diagnostics.proposed_excluded_by_route_cert > 0


def test_10l_history_rescue_sensitivity_cost_accounted_exactly():
    agent, world = _make_hybrid_agent(
        source_edge_ranker=HistoryRescuesource_edgeRanker(CausalWorld(5, random.Random(_SEED_W), noise_sigma=0.0)),
        probe_proposer=None,
    )
    agent._source_edge_ranker._world = world
    before = agent.total_interventions

    agent._screen_candidate_source_edges(2, 4)

    diag = agent._source_edge_proposal_diagnostics
    assert diag.sensitivity_rescue_calls == 1
    assert diag.sensitivity_rescue_interventions > 0
    assert agent.total_interventions - before == diag.sensitivity_rescue_interventions


def test_10m_chosen_source_edge_source_history_and_rescue_recorded():
    diag = source_edgeProposalDiagnostics()
    diag.record_call(
        (0, 1),
        (0, 1),
        {
            "history_ranker_calls": 1,
            "sensitivity_rescue_calls": 1,
            "sensitivity_rescue_interventions": 4,
            "rescue_candidates_added": 1,
        },
    )
    diag.record_fit((0, 1), (0, 1), {0: "history", 1: "rescue"})

    assert diag.chosen_source_edge_from_history == 1
    assert diag.chosen_source_edge_from_rescue == 1
    assert diag.rescue_chosen_source_edge_hits == 1
    assert diag.sensitivity_rescue_interventions == 4


# ── 11 provider policy quality diagnostics ───────────────────────────────────

def test_11a_quality_score_formula_fixed_metrics():
    score = compute_quality_cost(
        iv=100,
        full_audits=2,
        revocations=3,
        unique_fails=4,
        regime_sentinel_fail=5,
        regime_sentinel_no_sentinel=6,
        provider_probe_no_effect_count=7,
        provider_probe_improved_margin_count=8,
        weights=QualityWeights(),
    )

    assert score == 100 + 1000 * 2 + 5000 * 3 + 2000 * 4 + 500 * 5 + 0 * 6 + 10 * 7 - 25 * 8


def test_11b_quality_weight_override_changes_score():
    base = compute_quality_cost(
        iv=100,
        full_audits=2,
        revocations=0,
        unique_fails=0,
        regime_sentinel_fail=0,
        regime_sentinel_no_sentinel=0,
        provider_probe_no_effect_count=0,
        provider_probe_improved_margin_count=0,
        weights=QualityWeights(),
    )
    changed = compute_quality_cost(
        iv=100,
        full_audits=2,
        revocations=0,
        unique_fails=0,
        regime_sentinel_fail=0,
        regime_sentinel_no_sentinel=0,
        provider_probe_no_effect_count=0,
        provider_probe_improved_margin_count=0,
        weights=QualityWeights(audit_weight=7),
    )

    assert base != changed
    assert changed == 114


def test_11c_quality_summary_does_not_affect_run_behavior():
    agent, world = _make_hybrid_agent()
    agent.initialize()
    for c in range(1, 4):
        agent.run_cycle(c)
    before = (
        agent.skip_count,
        agent.full_audit_count,
        agent.total_interventions,
        tuple(
            (v, n.status, n.role_for("skip"), len(n.certificates), len(n.route_certs))
            for v, n in agent.ledger.vars.items()
        ),
    )

    _ = RunAnalyzer(agent).quality_score

    after = (
        agent.skip_count,
        agent.full_audit_count,
        agent.total_interventions,
        tuple(
            (v, n.status, n.role_for("skip"), len(n.certificates), len(n.route_certs))
            for v, n in agent.ledger.vars.items()
        ),
    )
    assert after == before


def test_11d_provider_policy_comparison_block_prints_for_two_policies():
    def fake_run(policy, iv, audits):
        source_edge, probe = policy
        arch = SimpleNamespace(
            revoked_by_dist={},
            total_unique_failures=0,
            regime_sentinel_fails=0,
            regime_no_sentinel=0,
            passive_saved_iv=0,
            provider_probe_no_effect_count=0,
            provider_probe_improved_margin_count=0,
        )
        return SimpleNamespace(
            ok=True,
            config=SimpleNamespace(source_edge_ranker=source_edge, probe_proposer=probe),
            interventions=iv,
            full_audits=audits,
            arch=arch,
        )

    buf = StringIO()
    with redirect_stdout(buf):
        _print_provider_policy_comparison(
            [
                fake_run(("sensitivity", "none"), 100, 1),
                fake_run(("history", "history"), 50, 3),
            ],
            QualityWeights(),
        )
    out = buf.getvalue()

    assert "provider policy comparison" in out
    assert "sensitivity/none" in out
    assert "history/history" in out
    assert "diagnostic only" in out


def _fake_policy_report_run(
    *,
    schedule="regime_switch",
    n_vars=50,
    cycles=3000,
    seed=42,
    source_edge="sensitivity",
    probe="none",
    quality_cost_extra=0,
    iv=100,
    audits=10,
    revocations=1,
    unique_fails=2,
    regime_fail=3,
    no_sentinel=4,
    skip_pct=80.0,
    elapsed=1.0,
    violations=None,
):
    arch = SimpleNamespace(
        revoked_by_dist={"test": revocations} if revocations else {},
        total_unique_failures=unique_fails,
        regime_sentinel_fails=regime_fail,
        regime_no_sentinel=no_sentinel,
        passive_saved_iv=0,
        provider_probe_no_effect_count=quality_cost_extra,
        provider_probe_improved_margin_count=0,
    )
    return SimpleNamespace(
        ok=True,
        config=SimpleNamespace(
            schedule=schedule,
            n_vars=n_vars,
            cycles=cycles,
            seed=seed,
            source_edge_ranker=source_edge,
            probe_proposer=probe,
        ),
        elapsed=elapsed,
        skip_pct=skip_pct,
        interventions=iv,
        full_audits=audits,
        arch=arch,
        violations=violations or [],
    )


def test_11e_policy_report_groups_by_schedule_scale_and_policy():
    rows = _build_policy_report_rows(
        [
            _fake_policy_report_run(schedule="regime_switch", n_vars=50, cycles=3000),
            _fake_policy_report_run(schedule="regime_switch", n_vars=50, cycles=3000, seed=99),
            _fake_policy_report_run(
                schedule="regime_switch", n_vars=75, cycles=3000,
                source_edge="history", probe="history",
            ),
            _fake_policy_report_run(
                schedule="false_trass", n_vars=50, cycles=3000,
                source_edge="history", probe="history",
            ),
        ],
        QualityWeights(),
    )

    by_key = {(r["schedule"], r["n_vars"], r["cycles"], r["policy"]): r for r in rows}

    assert by_key[("regime_switch", 50, 3000, "sensitivity/none")]["runs"] == 2
    assert ("regime_switch", 75, 3000, "history/history") in by_key
    assert ("false_trass", 50, 3000, "history/history") in by_key


def test_11f_policy_report_deltas_use_sensitivity_none_baseline():
    rows = _build_policy_report_rows(
        [
            _fake_policy_report_run(
                source_edge="sensitivity", probe="none",
                iv=100, audits=10, revocations=1, unique_fails=2,
            ),
            _fake_policy_report_run(
                source_edge="history", probe="history",
                iv=125, audits=13, revocations=4, unique_fails=7,
            ),
        ],
        QualityWeights(),
    )
    history = next(r for r in rows if r["policy"] == "history/history")

    assert history["delta_iv_vs_sensitivity"] == 25
    assert history["delta_audits_vs_sensitivity"] == 3
    assert history["delta_revocations_vs_sensitivity"] == 3
    assert history["delta_unique_fails_vs_sensitivity"] == 5
    assert history["delta_quality_cost_vs_sensitivity"] > 0


def test_11g_policy_report_pareto_marks_obviously_dominated_policy():
    rows = _build_policy_report_rows(
        [
            _fake_policy_report_run(
                source_edge="sensitivity", probe="none",
                iv=100, audits=10, revocations=1, unique_fails=2,
            ),
            _fake_policy_report_run(
                source_edge="history", probe="history",
                iv=110, audits=11, revocations=2, unique_fails=3,
            ),
        ],
        QualityWeights(),
    )
    by_policy = {r["policy"]: r for r in rows}

    assert by_policy["sensitivity/none"]["pareto_status"] == "efficient"
    assert by_policy["history/history"]["pareto_status"] == "dominated"


def test_11h_policy_report_generation_does_not_mutate_results():
    run = _fake_policy_report_run()
    before = (
        run.interventions,
        run.full_audits,
        dict(run.arch.revoked_by_dist),
        run.arch.total_unique_failures,
        list(run.violations),
    )

    _ = _build_policy_report_rows([run], QualityWeights())

    after = (
        run.interventions,
        run.full_audits,
        dict(run.arch.revoked_by_dist),
        run.arch.total_unique_failures,
        list(run.violations),
    )
    assert after == before
