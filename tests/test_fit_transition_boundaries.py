import random
import unittest

from dreth.agent import ChainedAgent
from dreth.ledger import ChainedLedger, Compression, NethraCertificate, NoiseEnvelope
from dreth.records import FitDiagnostic
from dreth.world import CausalWorld, HiddenMutation


def make_agent(n_vars=4):
    rng = random.Random(1234)
    world = CausalWorld(n_vars, rng, noise_sigma=0.0)
    world.visible_count = n_vars
    agent = ChainedAgent(
        world,
        rng,
        sentinel_count=0,
        sentinel_pool=0,
        compression_discover_after=999,
        priority_audit_budget=n_vars,
    )
    for nethra in agent.ledger.vars.values():
        nethra.certificates["skip"] = NethraCertificate(
            operation="skip", role="tareth", authority="none",
            context_source_edges=(), context_visible=n_vars, context_cycle=0,
            targets=(), substitutions_tested=("test_setup",), changes=1, trials=1,
            earned_by="manual_bootstrap",
        )
        nethra.status = "certified"
    return agent


def make_diag(var, source_edges, func, tie_set):
    return FitDiagnostic(
        cycle=10,
        var=var,
        status_before="certified",
        role_before="tareth",
        available_source_edges=(),
        restricted=False,
        hypothesis_count=len(tie_set),
        true_source_edges=source_edges,
        true_func=func,
        true_present=True,
        true_rank=1,
        true_score=10,
        best_score=10,
        second_score=10,
        margin=0,
        best_source_edges=source_edges,
        best_func=func,
        failure_class="fit_with_ties" if len(tie_set) > 1 else "fit_clean",
        tie_set=frozenset(tie_set),
    )


def seed_fit(agent, var, source_edges=(0, 1), func="MEAN"):
    n = agent.ledger.vars[var]
    n.source_edges = tuple(source_edges)
    n.func = func
    n.status = "certified"
    n.certificates["skip"] = NethraCertificate(
        operation="skip", role="tareth", authority="none",
        context_source_edges=tuple(source_edges), context_visible=4, context_cycle=0,
        targets=(), substitutions_tested=("test_setup",), changes=1, trials=1,
        earned_by="manual_bootstrap",
    )
    n.strong_observations = 7
    n.sentinels = [(0, 0.25), (1, 0.75)]
    n.expected_outcomes = [0.5, 0.5]
    n.compressions = [
        Compression(
            gate=((source_edges[0], 0.25, 0.1),) if source_edges else (),
            simplified_value=0.5,
            certified_equivalence=3,
            discovery_cycle=3,
        )
    ]
    n.compression_hits = 2
    n.compression_misses = 1
    n.envelope = NoiseEnvelope(
        deltas=[0.01, 0.02, 0.03],
        certified_eps=0.05,
        certified_at_cycle=4,
        samples_at_cert=3,
    )
    return n


class FitTransitionBoundaryTests(unittest.TestCase):
    def assert_prior_state_cleared(self, n):
        self.assertEqual(n.sentinels, [])
        self.assertEqual(n.expected_outcomes, [])
        self.assertEqual(n.compressions, [])
        self.assertEqual(n.compression_hits, 0)
        self.assertEqual(n.compression_misses, 0)
        self.assertEqual(n.envelope.deltas, [])
        self.assertEqual(n.envelope.certified_eps, 0.0)
        self.assertLessEqual(n.strong_observations, 1)

    def assert_prior_state_preserved(self, n):
        self.assertEqual(n.sentinels, [(0, 0.25), (1, 0.75)])
        self.assertEqual(n.expected_outcomes, [0.5, 0.5])
        self.assertEqual(len(n.compressions), 1)
        self.assertEqual(n.compression_hits, 2)
        self.assertEqual(n.compression_misses, 1)
        self.assertEqual(n.envelope.deltas, [0.01, 0.02, 0.03])
        self.assertEqual(n.envelope.certified_eps, 0.05)
        self.assertGreaterEqual(n.strong_observations, 7)

    def test_t1_true_semantic_change_excluded_from_tie_set_gets_full_reset(self):
        agent = make_agent()
        seed_fit(agent, 2, (0, 1), "MEAN")
        agent._last_fit_diag = make_diag(2, (0, 1), "MAX", {((0, 1), "MAX")})

        sig_changed = agent._install_var(2, (0, 1), "MAX", 10, 8, 10)

        self.assertTrue(sig_changed)
        self.assert_prior_state_cleared(agent.ledger.vars[2])

    def test_t2_tied_same_source_edge_churn_preserves_varnethra_state(self):
        agent = make_agent()
        seed_fit(agent, 2, (0, 1), "MEAN")
        tie_set = {((0, 1), "MEAN"), ((0, 1), "MAX")}
        agent._last_fit_diag = make_diag(2, (0, 1), "MAX", tie_set)

        sig_changed = agent._install_var(2, (0, 1), "MAX", 10, 10, 10)

        self.assertFalse(
            sig_changed,
            "operator-only tied churn with identical source_edges is not semantic drift",
        )
        self.assert_prior_state_preserved(agent.ledger.vars[2])

    def test_t3_tied_source_edge_or_arity_churn_still_gets_full_reset(self):
        for new_source_edges, new_func in [((1, 3), "MAX"), ((0,), "FIRST")]:
            with self.subTest(new_source_edges=new_source_edges, new_func=new_func):
                agent = make_agent()
                seed_fit(agent, 2, (0, 1), "MEAN")
                tie_set = {((0, 1), "MEAN"), (tuple(new_source_edges), new_func)}
                agent._last_fit_diag = make_diag(2, tuple(new_source_edges), new_func, tie_set)

                sig_changed = agent._install_var(2, tuple(new_source_edges), new_func, 10, 10, 10)

                self.assertTrue(sig_changed)
                self.assert_prior_state_cleared(agent.ledger.vars[2])

    def test_t6_tied_same_source_edge_churn_does_not_create_false_sentinel_invalidation(self):
        agent = make_agent(3)
        agent.world.source_edges = [[], [], [0, 1]]
        agent.world.funcs = ["LOW", "LOW", "MAX"]
        agent.world.state = (0.5, 0.5, 0.5)
        for source_edge in (0, 1):
            agent.ledger.vars[source_edge].certificates["skip"] = NethraCertificate(
                operation="skip", role="trass", authority="skip",
                context_source_edges=(), context_visible=3, context_cycle=0,
                targets=(), substitutions_tested=("test_setup",), changes=0, trials=1,
                earned_by="manual_bootstrap",
            )
            agent.ledger.vars[source_edge].status = "trass"
        n = seed_fit(agent, 2, (0, 1), "MEAN")
        n.sentinels = [(2, 0.3)]
        n.expected_outcomes = [0.5]
        tie_set = {((0, 1), "MEAN"), ((0, 1), "MAX")}
        agent._last_fit_diag = make_diag(2, (0, 1), "MAX", tie_set)

        agent._install_var(2, (0, 1), "MAX", 10, 10, 10)
        agent.run_cycle(HiddenMutation(11, "VALUE", "steady", False, -1))

        self.assertEqual(agent.ledger.vars[2].collapse_log, [])
        self.assertEqual(agent.ledger.vars[2].status, "certified")
        self.assertIn(2, agent.records[-1].skipped_vars)
        self.assertNotIn(2, agent.records[-1].fully_audited_vars)

    def test_t7_default_update_var_still_resets_when_no_preserve_flag_is_used(self):
        ledger = ChainedLedger(4)
        n = ledger.vars[2]
        n.source_edges = (0, 1)
        n.func = "MEAN"
        n.sentinels = [(0, 0.25)]
        n.expected_outcomes = [0.5]
        n.compressions = [Compression(((0, 0.25, 0.1),), 0.5, 3, 1)]
        n.envelope = NoiseEnvelope(deltas=[0.01], certified_eps=0.05)
        n.strong_observations = 9

        changed = ledger.update_var(2, (0, 1), "MAX", 10)

        self.assertTrue(changed)
        self.assert_prior_state_cleared(ledger.vars[2])

    def test_t8_topo_cache_invalidates_only_when_source_edge_structure_changes(self):
        agent = make_agent()
        seed_fit(agent, 2, (0, 1), "MEAN")
        agent._topo_cache = [0, 1, 2, 3]
        agent._topo_cache_visible_count = 4
        tie_set = {((0, 1), "MEAN"), ((0, 1), "MAX")}
        agent._last_fit_diag = make_diag(2, (0, 1), "MAX", tie_set)

        agent._install_var(2, (0, 1), "MAX", 10, 10, 10)

        self.assertEqual(
            agent._topo_cache,
            [0, 1, 2, 3],
            "same-source_edge tied operator churn must not invalidate DAG topo cache",
        )

        agent._last_fit_diag = make_diag(
            2,
            (1, 3),
            "MAX",
            {((0, 1), "MAX"), ((1, 3), "MAX")},
        )
        agent._install_var(2, (1, 3), "MAX", 10, 10, 11)

        self.assertIsNone(agent._topo_cache)

if __name__ == "__main__":
    unittest.main()
