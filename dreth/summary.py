from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Run-end summary computation, separated from agent policy.
#
#   RunAnalyzer    — reads agent + ledger state, computes all summary metrics.
#                    Pure reads: no writes to agent or ledger.
#   SummaryRenderer — formats RunAnalyzer output as a human-readable string.
#
# ChainedAgent.final_summary() delegates to SummaryRenderer(RunAnalyzer(agent)).
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

from .quality import QualityWeights, make_quality_score

if TYPE_CHECKING:
    from .agent import ChainedAgent


class RunAnalyzer:
    """Reads a completed ChainedAgent run and computes summary metrics.
    All fields are set in __init__; no further reads after construction.
    """

    def __init__(self, agent: "ChainedAgent") -> None:
        ledger = agent.ledger
        visible = [ledger.vars[i] for i in range(agent.world.visible_count)]

        self.n_cycles: int = len(agent.records)
        self.n_var: int = agent.world.visible_count

        # Status counts
        self.cert_count = sum(1 for n in visible if n.status == "certified")
        self.quar_count = sum(1 for n in visible if n.status == "quarantined")
        self.uncert_count = sum(1 for n in visible if n.status == "uncertain")
        self.prop_count = sum(1 for n in visible if n.status == "proposed")
        self.trass_count = sum(1 for n in visible if n.status == "trass" or n.role_for("skip") == "trass")
        self.authoritative_count = sum(1 for n in visible if n.authoritative)

        # Skip / audit rates
        self.total_deferred = sum(len(r.deferred_vars) for r in agent.records)
        total_decisions = agent.skip_count + agent.full_audit_count + self.total_deferred
        self.var_skip_rate = agent.skip_count / max(1, total_decisions) * 100
        self.full_audit_count = agent.full_audit_count
        self.skip_count = agent.skip_count
        self.trass_skip_count = agent.trass_skip_count
        self.compression_skip_count = agent.compression_skip_count
        self.sentinel_skip_count = agent.sentinel_skip_count
        self.total_interventions = agent.total_interventions
        self.total_decisions = total_decisions

        # Novelty
        self.nov_total = len(ledger.novelty)
        self.nov_open = sum(1 for n in ledger.novelty if n.status == "open")
        self.nov_resolved = sum(1 for n in ledger.novelty if n.status == "resolved")

        # Compression
        self.total_comps = sum(len(n.compressions) for n in ledger.vars.values())
        self.comp_hits = sum(n.compression_hits for n in ledger.vars.values())
        self.comp_misses = sum(n.compression_misses for n in ledger.vars.values())
        self.comp_hits_lifetime = sum(n.compression_hits_lifetime for n in ledger.vars.values())
        self.comp_misses_lifetime = sum(n.compression_misses_lifetime for n in ledger.vars.values())

        # Envelope
        self.certified_envs = sum(1 for n in ledger.vars.values() if n.envelope.certified_eps > 0)
        self.total_oob = sum(n.envelope.out_of_band_count for n in ledger.vars.values())

        # Temporal trass
        self.total_muted = sum(len(n.temporal_trass_log) for n in ledger.vars.values())
        self.muted_low = sum(
            1 for n in ledger.vars.values()
            for e in n.temporal_trass_log if e.reason == "low_cost_dismissed"
        )
        self.muted_outlier = sum(
            1 for n in ledger.vars.values()
            for e in n.temporal_trass_log if e.reason == "outlier_within_tolerance"
        )

        # Defer stats
        self.worst_deferred: List[Tuple[int, int]] = sorted(
            agent.defer_count.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        self.max_defer_streak: Dict[int, int] = agent.max_defer_streak

        # Watch / Poisson
        self.watch_count = sum(1 for n in visible if n.in_watch_state)
        self.lambda_vals = [
            n.poisson_rate for n in visible
            if n.role_for("skip") != "trass" and n.poisson_rate > 0
            and n.first_audited_cycle > 0
        ]
        self.watch_queued = sum(1 for e in ledger.event_log if "in watch state)" in e)

        # Coverage (first cert cycle)
        self.cert_cycles = sorted(
            n.first_certified_cycle for n in visible
            if n.first_certified_cycle > 0 and n.role_for("skip") != "trass"
        )

        # Fit diagnostics
        self.fit_class_counts: Counter = Counter(d.failure_class for d in agent.fit_diagnostics)
        self.restricted_fits = sum(1 for d in agent.fit_diagnostics if d.restricted)
        self.total_fits = len(agent.fit_diagnostics)
        by_var: Dict[int, list] = defaultdict(list)
        for d in agent.fit_diagnostics:
            by_var[d.var].append(d)
        audit_rows = []
        for v, ds in by_var.items():
            latest = ds[-1]
            common_class = Counter(d.failure_class for d in ds).most_common(1)[0][0]
            audit_rows.append((len(ds), v, common_class, latest.status_after, latest.role_after, latest.margin))
        audit_rows.sort(reverse=True)
        self.top_audit_rows = audit_rows[:6]

        # Tie log
        self.tie_log = agent.tie_log

        # Compression amortization
        eligible = [
            n for n in visible
            if bool(n.source_edges) and n.role_for("skip") == "tareth"
            and n.status in ("certified", "proposed") and n.sentinels
        ]
        self.n_elig = len(eligible)
        self.n_with_comps = sum(1 for n in eligible if n.compressions)
        est_disc_cost = agent.compression_discovery_budget * 3
        total_disc_runs = sum(
            1 for n in visible
            if n.compression_hits_lifetime > 0 or n.compression_misses_lifetime > 0
        )
        avg_source_edges = (
            sum(len(n.source_edges) for n in eligible) // max(1, self.n_elig)
            if eligible else 1
        )
        self.est_disc_total = total_disc_runs * est_disc_cost * max(1, avg_source_edges)
        self.hits_saved = self.comp_hits_lifetime * agent.sentinel_count

        # Regime
        self.regime_summary: str = agent.regime_register.summary()

        # Per-composite audit and overlap diagnostics.
        # raw_pair_member_passes = sum(pass_count * n_members) across live composites.
        # When vars are members of multiple composites, each composite's pass_count
        # increments every passing cycle regardless of overlap. duplicate_factor
        # = raw_pair_member_passes / composite_skip_count measures how many
        # pair-pass credits the system issued per actual var-skip decision.
        # A factor of 1 = no overlap; factor of K = each var skip was recorded K
        # times across K composites.
        live_composites = agent.ledger.composites
        all_composites = list(live_composites) + list(agent.ledger.revoked_composites)

        from collections import Counter as _Counter
        _member_degree: Dict[int, int] = _Counter(
            v for cn in live_composites for v in cn.members
        )

        self.composite_live: int = len(live_composites)
        self.composite_revoked: int = len(agent.ledger.revoked_composites)
        self.composite_unique_members: int = len(_member_degree)

        # Connected components of the live composite graph (vars = nodes, pairs = edges)
        _source_edge: Dict[int, int] = {v: v for v in _member_degree}
        def _find(x: int) -> int:
            while _source_edge[x] != x:
                _source_edge[x] = _source_edge[_source_edge[x]]
                x = _source_edge[x]
            return x
        for cn in live_composites:
            a_var, b_var = cn.members
            ra, rb = _find(a_var), _find(b_var)
            if ra != rb:
                _source_edge[ra] = rb
        self.composite_components: int = len({_find(v) for v in _member_degree}) if _member_degree else 0

        degrees = list(_member_degree.values())
        self.composite_max_degree: int = max(degrees) if degrees else 0
        self.composite_mean_degree: float = sum(degrees) / len(degrees) if degrees else 0.0

        _raw_pair_passes = sum(cn.pass_count * len(cn.members) for cn in live_composites)
        _true_skip = getattr(agent, "composite_skip_count", 0)
        self.composite_true_skip: int = _true_skip
        self.composite_duplicate_factor: float = _raw_pair_passes / max(1, _true_skip)

        total_cycles = self.n_cycles
        self.composite_rows: List[Tuple] = []
        for cn in all_composites:
            live = cn.revoked_at_cycle == 0
            lifespan = (total_cycles - cn.certified_at_cycle) if live else (cn.revoked_at_cycle - cn.certified_at_cycle)
            degree = max(_member_degree.get(v, 1) for v in cn.members) if live else 0
            self.composite_rows.append((
                cn.members,
                cn.sentinel_var,
                cn.certified_at_cycle,
                cn.changes,
                cn.trials,
                cn.pass_count,
                cn.pass_count * len(cn.members),  # probe_valid_iv: overcounts when degree > 1
                degree,
                lifespan,
                cn.revoke_reason or "—",
                cn.revoked_at_cycle,
                live,
            ))
        self.composite_rows.sort(key=lambda r: r[5], reverse=True)

        # HyperCompositeNethra (component-level) metrics.
        live_hyper = agent.ledger.hyper_composites
        rev_hyper  = agent.ledger.revoked_hyper_composites
        self.component_live: int = len(live_hyper)
        self.component_revoked: int = len(rev_hyper)
        self.component_skips: int = getattr(agent, "component_skip_count", 0)
        self.pairwise_fallbacks: int = getattr(agent, "pairwise_fallback_count", 0)
        self.component_members: int = len({v for hc in live_hyper for v in hc.members})

        # duplicate_factor_before: from all pairwise composites (live + absorbed)
        _all_pair = list(live_composites) + list(agent.ledger.absorbed_composites)
        _raw_before = sum(cn.pass_count * len(cn.members) for cn in _all_pair)
        _pairwise_skip = getattr(agent, "composite_skip_count", 0)
        self.duplicate_factor_before: float = _raw_before / max(1, _pairwise_skip + self.component_skips)

        # duplicate_factor_after: only unabsorbed pairwise composites remain;
        # component covered vars contribute 1× per cycle (no overlap by design)
        _raw_after = sum(cn.pass_count * len(cn.members) for cn in live_composites)
        self.duplicate_factor_after: float = (
            (_raw_after + self.component_skips) / max(1, _pairwise_skip + self.component_skips)
        )

        # Per-component rows for the audit table.
        self.component_rows: List[Tuple] = []
        for hc in list(live_hyper) + list(rev_hyper):
            live_hc = hc.revoked_at_cycle == 0
            lifespan_hc = (total_cycles - hc.certified_at_cycle) if live_hc else (hc.revoked_at_cycle - hc.certified_at_cycle)
            self.component_rows.append((
                hc.component_id,
                len(hc.members),
                hc.sentinel_var,
                hc.certified_at_cycle,
                hc.absorbed_pairs,
                hc.pass_count,
                hc.pairwise_fallback_count,
                lifespan_hc,
                hc.revoke_reason or "—",
                hc.revoked_at_cycle,
                live_hc,
            ))
        self.component_rows.sort(key=lambda r: r[5], reverse=True)

        # ── Hybrid control metrics ────────────────────────────────────────────
        # Only populated when hybrid providers are active (non-zero when used).
        # Counters live on agent; agenda summary comes from the RepairAgenda object.
        self.hybrid_residual_predictor_calls: int = getattr(agent, "_hybrid_residual_predictor_calls", 0)
        self.hybrid_residual_ok: int = getattr(agent, "_hybrid_residual_ok", 0)
        self.hybrid_residual_stressed: int = getattr(agent, "_hybrid_residual_stressed", 0)
        self.hybrid_source_edge_ranker_calls: int = getattr(agent, "_hybrid_source_edge_ranker_calls", 0)
        self.hybrid_probe_proposer_calls: int = getattr(agent, "_hybrid_probe_proposer_calls", 0)
        self.hybrid_expert_router_calls: int = getattr(agent, "_hybrid_expert_router_calls", 0)
        _source_edge_prop = getattr(agent, "_source_edge_proposal_diagnostics", None)
        if _source_edge_prop is not None:
            self.source_edge_proposal_calls = _source_edge_prop.calls
            self.source_edge_proposal_hit_rate = _source_edge_prop.chosen_source_edge_hit_rate
            self.source_edge_proposal_miss_count = _source_edge_prop.miss_chosen_source_edge_count
            self.source_edge_proposal_rank_mean = _source_edge_prop.rank_of_chosen_source_edge_mean
            self.source_edge_proposal_rank_max = _source_edge_prop.rank_of_chosen_source_edge_max
            self.history_ranker_calls = _source_edge_prop.history_ranker_calls
            self.sensitivity_rescue_calls = _source_edge_prop.sensitivity_rescue_calls
            self.sensitivity_rescue_interventions = _source_edge_prop.sensitivity_rescue_interventions
            self.rescue_candidates_added = _source_edge_prop.rescue_candidates_added
            self.rescue_chosen_source_edge_hits = _source_edge_prop.rescue_chosen_source_edge_hits
            self.chosen_source_edge_from_history = _source_edge_prop.chosen_source_edge_from_history
            self.chosen_source_edge_from_rescue = _source_edge_prop.chosen_source_edge_from_rescue
        else:
            self.source_edge_proposal_calls = 0
            self.source_edge_proposal_hit_rate = 0.0
            self.source_edge_proposal_miss_count = 0
            self.source_edge_proposal_rank_mean = 0.0
            self.source_edge_proposal_rank_max = 0
            self.history_ranker_calls = 0
            self.sensitivity_rescue_calls = 0
            self.sensitivity_rescue_interventions = 0
            self.rescue_candidates_added = 0
            self.rescue_chosen_source_edge_hits = 0
            self.chosen_source_edge_from_history = 0
            self.chosen_source_edge_from_rescue = 0
        _probe_prop = getattr(agent, "_probe_proposal_diagnostics", None)
        if _probe_prop is not None:
            self.provider_probes_proposed = _probe_prop.provider_probes_proposed
            self.provider_probes_valid = _probe_prop.provider_probes_valid
            self.provider_probes_invalid = _probe_prop.provider_probes_invalid
            self.provider_probes_used_by_fit = _probe_prop.provider_probes_used_by_fit
            self.provider_probe_improved_margin_count = _probe_prop.provider_probe_improved_margin_count
            self.provider_probe_no_effect_count = _probe_prop.provider_probe_no_effect_count
        else:
            self.provider_probes_proposed = 0
            self.provider_probes_valid = 0
            self.provider_probes_invalid = 0
            self.provider_probes_used_by_fit = 0
            self.provider_probe_improved_margin_count = 0
            self.provider_probe_no_effect_count = 0

        self.revocations = sum(
            1 for n in visible
            for cert in list(n.certificates.values()) + list(n.route_certs.values())
            if getattr(cert, "revoked_by", None) is not None
        )
        self.unique_fails = sum(n.unique_failures_caught for n in visible)
        self.regime_sentinel_fail = getattr(agent, "_regime_sentinel_fails", 0)
        self.regime_sentinel_no_sentinel = getattr(agent, "_regime_no_sentinel", 0)
        self.passive_saved_iv = getattr(agent, "_passive_saved_iv", 0)
        self.quality_score = make_quality_score(
            iv=self.total_interventions,
            full_audits=self.full_audit_count,
            revocations=self.revocations,
            unique_fails=self.unique_fails,
            regime_sentinel_fail=self.regime_sentinel_fail,
            regime_sentinel_no_sentinel=self.regime_sentinel_no_sentinel,
            passive_saved_iv=self.passive_saved_iv,
            provider_probe_no_effect_count=self.provider_probe_no_effect_count,
            provider_probe_improved_margin_count=self.provider_probe_improved_margin_count,
            weights=QualityWeights(),
        )

        _agenda = getattr(agent, "_repair_agenda", None)
        if _agenda is not None:
            _agenda_summary = _agenda.summary()
            self.hybrid_repair_agenda_items: int = _agenda_summary["total_pushed"]
            self.hybrid_repair_agenda_scope_mean: float = _agenda_summary.get("scope_mean", 0.0)
            self.hybrid_repair_agenda_scope_max: int = _agenda_summary.get("scope_max", 0)
        else:
            self.hybrid_repair_agenda_items = 0
            self.hybrid_repair_agenda_scope_mean = 0.0
            self.hybrid_repair_agenda_scope_max = 0

        # Active when ANY provider was used this run; ensures wiring gaps are visible.
        self.hybrid_active: bool = (
            self.hybrid_residual_predictor_calls > 0
            or self.hybrid_source_edge_ranker_calls > 0
            or self.hybrid_probe_proposer_calls > 0
            or self.hybrid_expert_router_calls > 0
        )


class SummaryRenderer:
    """Formats RunAnalyzer metrics as a multi-line human-readable string."""

    def __init__(self, analysis: RunAnalyzer) -> None:
        self._a = analysis

    def render(self) -> str:
        a = self._a
        lines: List[str] = []

        lines.append("\n── summary ────────────────────────────────────────────")
        lines.append(
            f"  cyc={a.n_cycles} vars={a.n_var} | "
            f"st: cert={a.cert_count} prop={a.prop_count} unc={a.uncert_count} "
            f"quar={a.quar_count} trass={a.trass_count} | "
            f"auth={a.authoritative_count}/{a.n_var}"
        )
        lines.append(f"  nov: {a.nov_total} (open={a.nov_open} res={a.nov_resolved})")
        lines.append(
            f"  comp: stored={a.total_comps} "
            f"hit/miss live={a.comp_hits}/{a.comp_misses} "
            f"life={a.comp_hits_lifetime}/{a.comp_misses_lifetime}"
        )
        lines.append(f"  env: cert={a.certified_envs}/{a.n_var} oob={a.total_oob}")
        lines.append(f"  mute: total={a.total_muted} low={a.muted_low} outlier={a.muted_outlier}")
        lines.append(
            f"  audit: full={a.full_audit_count} skip={a.skip_count}/{a.total_decisions} "
            f"({a.var_skip_rate:.1f}%) | trass={a.trass_skip_count} "
            f"comp={a.compression_skip_count} sent={a.sentinel_skip_count}"
        )
        lines.append(f"  iv={a.total_interventions}")
        lines.append(
            f"  quality_cost={a.quality_score.quality_cost} "
            "(diagnostic only; no policy selected)"
        )

        worst_str = ", ".join(
            f"x{v}:n={c},max={a.max_defer_streak[v]}"
            for v, c in a.worst_deferred if c > 0
        ) or "none"
        lines.append(f"  defer: total={a.total_deferred} worst={worst_str}")

        avg_lam = sum(a.lambda_vals) / len(a.lambda_vals) if a.lambda_vals else 0.0
        lines.append(
            f"  predict: watch={a.watch_count}/{a.n_var} "
            f"λ_tracked={len(a.lambda_vals)} avg_λ={avg_lam:.3f} queued={a.watch_queued}"
        )

        if a.cert_cycles:
            p50 = a.cert_cycles[len(a.cert_cycles) // 2]
            lines.append(
                f"  coverage: n={len(a.cert_cycles)} "
                f"first={a.cert_cycles[0]} p50={p50} last={a.cert_cycles[-1]}"
            )

        class_str = " ".join(f"{k}={v}" for k, v in a.fit_class_counts.most_common()) or "none"
        lines.append(f"  fit: aud={a.total_fits} restr={a.restricted_fits}")
        lines.append(f"    classes: {class_str}")
        if a.top_audit_rows:
            lines.append(
                "    most audited: " + " | ".join(
                    f"x{v}:a={aud} {klass[:10]} {status[:4]}/{role[:4]} lm={margin}"
                    for aud, v, klass, status, role, margin in a.top_audit_rows
                )
            )

        if a.tie_log:
            tie_summary = []
            for v, sets in sorted(a.tie_log.items()):
                top_set, top_count = max(sets.items(), key=lambda kv: kv[1])
                tie_summary.append(f"x{v}:|set|={len(top_set)} ×{top_count}")
            lines.append(
                f"  tie sets (top per var): {' '.join(tie_summary[:8])}"
                + (f" ... +{len(tie_summary) - 8} more" if len(tie_summary) > 8 else "")
            )

        lines.append("\n── compression amortization ───────────────────────────")
        lines.append(f"  eligible vars (tareth+committed+has-source_edges): {a.n_elig}/{a.n_var}")
        lines.append(f"  with compressions: {a.n_with_comps}/{a.n_elig}")
        lines.append(
            f"  hits lifetime: {a.comp_hits_lifetime} | misses lifetime: {a.comp_misses_lifetime}"
        )
        amort = f"{a.hits_saved / a.est_disc_total:.2f}x" if a.est_disc_total > 0 else "n/a"
        lines.append(
            f"  est cost saved by hits: {a.hits_saved} iv | "
            f"est discovery cost: {a.est_disc_total} iv | amortization: {amort}"
        )

        lines.append("\n── regime register ────────────────────────────────────")
        lines.append(a.regime_summary)

        lines.append("\n── composite nethra audit ─────────────────────────────")
        if not a.composite_rows:
            lines.append("  none")
        else:
            lines.append(
                f"  live={a.composite_live} revoked={a.composite_revoked} "
                f"unique_members={a.composite_unique_members} "
                f"components={a.composite_components}"
            )
            lines.append(
                f"  degree: max={a.composite_max_degree} "
                f"mean={a.composite_mean_degree:.1f}"
            )
            lines.append(
                f"  true_skip={a.composite_true_skip} "
                f"duplicate_factor={a.composite_duplicate_factor:.2f}x"
                + ("  ← near-clique expansion" if a.composite_duplicate_factor > 5 else "")
            )
            lines.append(
                f"  {'members':<12} {'sent':>4} {'cert@':>5} {'ch/tr':>7} "
                f"{'pass':>6} {'probe_iv':>8} {'deg':>4} {'life':>5} {'status':<22}"
            )
            shown = 0
            for row in a.composite_rows:
                members, sv, cert_at, changes, trials, pass_count, probe_iv, degree, lifespan, revoke_reason, revoked_at, live = row
                mem_str = f"x{members[0]},x{members[1]}"
                status_str = "live" if live else f"rev@{revoked_at}({revoke_reason[:12]})"
                lines.append(
                    f"  {mem_str:<12} {sv:>4} {cert_at:>5} {changes:>3}/{trials:<3} "
                    f"{pass_count:>6} {probe_iv:>8} {degree:>4} {lifespan:>5} {status_str}"
                )
                shown += 1
                if shown >= 20 and len(a.composite_rows) > 20:
                    lines.append(f"  ... +{len(a.composite_rows) - 20} more")
                    break

        lines.append("\n── component nethra audit ─────────────────────────────")
        if not a.component_rows and a.component_live == 0 and a.component_revoked == 0:
            lines.append("  none")
        else:
            lines.append(
                f"  live={a.component_live} revoked={a.component_revoked} "
                f"members={a.component_members} "
                f"component_skips={a.component_skips} fallbacks={a.pairwise_fallbacks}"
            )
            lines.append(
                f"  dup_factor: before={a.duplicate_factor_before:.2f}x "
                f"after={a.duplicate_factor_after:.2f}x"
            )
            if a.component_rows:
                lines.append(
                    f"  {'C#':>3} {'size':>5} {'sent':>4} {'cert@':>5} {'pairs':>6} "
                    f"{'pass':>7} {'fall':>5} {'life':>6} {'status':<22}"
                )
                for row in a.component_rows:
                    cid, size, sv, cert_at, pairs, pass_count, fallbacks, lifespan, revoke_reason, revoked_at, live_hc = row
                    status_str = "live" if live_hc else f"rev@{revoked_at}({revoke_reason[:12]})"
                    lines.append(
                        f"  C{cid:<3} {size:>5} {sv:>4} {cert_at:>5} {pairs:>6} "
                        f"{pass_count:>7} {fallbacks:>5} {lifespan:>6} {status_str}"
                    )

        if a.hybrid_active or a.hybrid_repair_agenda_items > 0:
            lines.append("\n── hybrid control ─────────────────────────────────────")
            # All four provider counters are printed even when zero so wiring gaps
            # are immediately visible when --hybrid-control interfaces is active.
            lines.append(
                f"  residual_predictor: calls={a.hybrid_residual_predictor_calls} "
                f"ok={a.hybrid_residual_ok} stressed={a.hybrid_residual_stressed}"
            )
            lines.append(f"  source_edge_ranker:      calls={a.hybrid_source_edge_ranker_calls}")
            lines.append(f"  probe_proposer:     calls={a.hybrid_probe_proposer_calls}")
            lines.append(f"  expert_router:      calls={a.hybrid_expert_router_calls}")
            if a.source_edge_proposal_calls > 0:
                lines.append("  source_edge_proposal:")
                lines.append(
                    f"    calls={a.source_edge_proposal_calls} "
                    f"chosen_source_edge_hit_rate={a.source_edge_proposal_hit_rate:.3f} "
                    f"miss_chosen_source_edge_count={a.source_edge_proposal_miss_count} "
                    f"rank_mean={a.source_edge_proposal_rank_mean:.2f} "
                    f"rank_max={a.source_edge_proposal_rank_max}"
                )
                if a.sensitivity_rescue_calls > 0:
                    lines.append(
                        f"    history_ranker_calls={a.history_ranker_calls} "
                        f"sensitivity_rescue_calls={a.sensitivity_rescue_calls} "
                        f"sensitivity_rescue_interventions={a.sensitivity_rescue_interventions}"
                    )
                    lines.append(
                        f"    rescue_candidates_added={a.rescue_candidates_added} "
                        f"rescue_chosen_source_edge_hits={a.rescue_chosen_source_edge_hits} "
                        f"chosen_source_edge_from_history={a.chosen_source_edge_from_history} "
                        f"chosen_source_edge_from_rescue={a.chosen_source_edge_from_rescue}"
                    )
            if a.provider_probes_proposed > 0 or a.hybrid_probe_proposer_calls > 0:
                lines.append("  probe_proposal:")
                lines.append(
                    f"    provider_probes_proposed={a.provider_probes_proposed} "
                    f"provider_probes_valid={a.provider_probes_valid} "
                    f"provider_probes_invalid={a.provider_probes_invalid} "
                    f"provider_probes_used_by_fit={a.provider_probes_used_by_fit}"
                )
                lines.append(
                    f"    provider_probe_improved_margin_count={a.provider_probe_improved_margin_count} "
                    f"provider_probe_no_effect_count={a.provider_probe_no_effect_count}"
                )
            if a.hybrid_repair_agenda_items > 0:
                lines.append(
                    f"  repair_agenda: total_pushed={a.hybrid_repair_agenda_items} "
                    f"scope_mean={a.hybrid_repair_agenda_scope_mean:.1f} "
                    f"scope_max={a.hybrid_repair_agenda_scope_max}"
                )

        return "\n".join(lines)
