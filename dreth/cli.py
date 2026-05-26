from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# CLI entrypoint. Parses arguments, constructs CausalWorld and ChainedAgent,
# runs the chosen schedule, prints the final summary.
#
# Active schedules:
#   shaped           — structured VALUE/EDGE/FUNC/NOVELTY mutations over time
#   periodic_shifts  — regular structural changes at fixed intervals
#   novelty          — introduces a SIN variable to test library-gap detection
#   incremental      — reveals one variable at a time with settle_cycles gaps
#   blind_challenge  — seed-generated blind procedural stress world
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import random
from typing import Dict, List, Optional, Set

from .world import CausalWorld, HiddenMutation
from .agent import ChainedAgent
from .baseline import RefitBaseline
from .records import CycleRecord

def parse_args() -> argparse.Namespace:
    """CLI argument parser. All flags have defaults that produce a runnable
    test. Knobs map to ChainedAgent constructor params plus world/run config."""
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n-vars", type=int, default=5)
    p.add_argument("--cycles", type=int, default=50)
    p.add_argument("--intervention-budget", type=int, default=30)
    p.add_argument("--full-margin-threshold", type=int, default=4)
    p.add_argument("--sentinel-count", type=int, default=5)
    p.add_argument("--sentinel-pool", type=int, default=60)
    p.add_argument("--promote-after", type=int, default=2)
    p.add_argument("--novelty-weak-streak", type=int, default=2)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    p.add_argument("--schedule",
                   choices=["shaped", "periodic_shifts", "novelty", "incremental",
                            "rare_catastrophe", "blind_challenge"],
                   default="blind_challenge",
                   help="incremental: introduce one variable at a time, "
                        "with `settle_cycles` of value drift between reveals; "
                        "rare_catastrophe: value drift each cycle with rare_prob "
                        "chance of structural mutation on rare_var; "
                        "blind_challenge: seed-generated hidden procedural world")
    p.add_argument("--challenge-blind", action="store_true",
                   help=("for blind_challenge, do not print generated manifest "
                         "details during the run"))
    p.add_argument("--rare-prob", type=float, default=0.02,
                   help="probability per cycle of catastrophic mutation in "
                        "rare_catastrophe schedule (default 0.02)")
    p.add_argument("--rare-var", type=int, default=0,
                   help="which variable gets the rare catastrophic mutation "
                        "(default 0)")
    p.add_argument("--settle-cycles", type=int, default=25,
                   help="cycles between variable reveals in incremental schedule")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--cost-weights", type=str, default=None,
                   help="comma-separated per-var cost weights, e.g. '0.1,0.1,3.0,0.1,0.1' "
                        "(higher = more attention worth spending)")
    p.add_argument("--priority-audit-budget", type=int, default=None,
                   help="v25: max full audits per cycle (variables ranked by tractability). "
                        "Default ~n_vars/2. Set explicitly to test scheduling sensitivity.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--mode", type=str, default="v28",
                   choices=["v28", "v29", "v29-algebraic-only", "v29-equiv-only"],
                   help="v28 = current behavior. v29 = load extension module "
                        "with both algebraic and equivalence compression. "
                        "v29-algebraic-only and v29-equiv-only enable just one path.")
    p.add_argument("--probe-retention", type=int, default=0,
                   help="If >0, keep per-probe arrays only for the most recent "
                        "K FitDiagnostics per var. 0 = unbounded. Memory tradeoff.")
    p.add_argument("--role-salience", type=str, default="all-visible",
                   choices=["all-visible", "live-frontier"],
                   help="Criterion for operation_role propagation tests. "
                        "all-visible counts propagation into any visible var. "
                        "live-frontier ignores propagation only into already-collapsed trass vars.")
    p.add_argument("--salience-targets", type=str, default=None,
                   help="Experimental comma-separated target vars for operation_role tests, "
                        "e.g. '0,7'. When set, salience is target-relative: only "
                        "propagation into these vars counts. Default unset preserves current behavior.")

    p.add_argument("--compression-promote-after", type=int, default=5,
                   help="consecutive correct predictions a compression must accumulate "
                        "before it is trusted for the skip path (default: 5).")
    p.add_argument("--uncertainty-consolidation", default="off",
                   choices=["off", "shadow", "assist"],
                   help="off=disabled; shadow=diagnostic consolidation only; "
                        "assist=bounded reversible attention/probe/repair hints (default: off)")
    p.add_argument("--uncertainty-assist-policy", default="all",
                   choices=["all", "budget_only", "probe_only", "preserve_only",
                            "priority_only", "local_only"],
                   help="assist submode used only with --uncertainty-consolidation assist")
    p.add_argument("--context-role-index", default="off",
                   choices=["off", "record", "assist_feature"],
                   help=("off=disabled; record=context role provenance only; "
                         "assist_feature=allow uncertainty consolidation to use "
                         "indexed nethra role history as local-anchor features"))
    p.add_argument("--context-role-anchor-policy", default=None,
                   choices=["off", "strict", "loose"],
                   help=("ContextRoleIndex assist matching policy. Default is "
                         "strict with assist_feature, off otherwise."))
    p.add_argument("--nethra-reservoir", dest="context_role_index", default=None,
                   choices=["off", "record", "assist_feature"],
                   help=argparse.SUPPRESS)
    return p.parse_args()


def _parse_salience_targets(raw: Optional[str], n_vars: int) -> Optional[Set[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    targets = {int(part.strip()) for part in raw.split(",") if part.strip()}
    bad = sorted(t for t in targets if t < 0 or t >= n_vars)
    if bad:
        raise ValueError(f"--salience-targets out of range for n-vars={n_vars}: {bad}")
    return targets


def _drift_latency(records: List[CycleRecord], event: HiddenMutation) -> str:
    """Return cycle latency from a truth rule change to localized detection."""
    for r in records:
        if r.cycle < event.cycle:
            continue
        if event.affected_var in r.detected_drift_vars:
            delta = r.cycle - event.cycle
            return "same-cycle" if delta == 0 else f"+{delta} cyc"
    return "missed"


def _format_structural_timeline(world: CausalWorld, agent: ChainedAgent) -> List[str]:
    structural = [m for m in world.hidden_log if m.kind != "VALUE"]
    if not structural:
        return ["  none"]
    lines = []
    for m in structural:
        detected = _drift_latency(agent.records, m) if m.rule_changed else "n/a"
        lines.append(f"  c{m.cycle:04d} {m.description} | detection={detected}")
    return lines


def _format_final_fits(world: CausalWorld, agent: ChainedAgent, limit: int = 12) -> List[str]:
    lines = []
    for i in range(min(world.visible_count, limit)):
        n = agent.ledger.vars[i]
        truth = f"{world.funcs[i]}({','.join(map(str, world.parents[i]))})"
        learned = f"{n.func}({','.join(map(str, n.parents))})"
        comp = f" comp={len(n.compressions)} hit={n.compression_hits_lifetime}" if n.compressions else ""
        lines.append(
            f"  x{i}: truth={truth:<12} learned={learned:<12} "
            f"{n.status}/{n.role_for('skip')} sent={len(n.sentinels)} skip={n.skip_count}{comp}"
        )
    if world.visible_count > limit:
        lines.append(f"  ... +{world.visible_count - limit} more")
    return lines


def _format_cycle_accounting(args: argparse.Namespace, world: CausalWorld, agent: ChainedAgent) -> List[str]:
    reveal_cycles = [m.cycle for m in world.hidden_log if m.kind == "REVEAL"]
    missing_record_cycles = sorted(set(range(1, args.cycles + 1)) - {r.cycle for r in agent.records})
    lines = [
        f"  requested={args.cycles} recorded={len(agent.records)} reveals={len(reveal_cycles)}",
    ]
    if reveal_cycles:
        lines.append(f"  reveal cycles: {', '.join(f'c{c:04d}' for c in reveal_cycles[:12])}"
                     + (f" ... +{len(reveal_cycles) - 12} more" if len(reveal_cycles) > 12 else ""))
    if missing_record_cycles:
        lines.append(f"  unrecorded cycles: {', '.join(f'c{c:04d}' for c in missing_record_cycles[:12])}"
                     + (f" ... +{len(missing_record_cycles) - 12} more" if len(missing_record_cycles) > 12 else ""))
    return lines


def _format_operational_view(world: CausalWorld, agent: ChainedAgent, limit: int = 12) -> List[str]:
    lines = []
    for i in range(min(world.visible_count, limit)):
        n = agent.ledger.vars[i]
        if n.role_for("skip") == "trass" or n.status == "trass":
            lines.append(
                f"  x{i}: collapsed/trass skip={n.skip_count} "
                f"audits={n.full_audits} last_fit={n.func}({','.join(map(str, n.parents))})"
            )
            continue
        learned = f"{n.func}({','.join(map(str, n.parents))})"
        comp = f" comps={len(n.compressions)} hit={n.compression_hits_lifetime}" if n.compressions else ""
        lines.append(
            f"  x{i}: active {n.status}/{n.role_for('skip')} learned={learned} "
            f"sent={len(n.sentinels)} skip={n.skip_count}{comp}"
        )
    if world.visible_count > limit:
        lines.append(f"  ... +{world.visible_count - limit} more")
    return lines


def _role_salience_audit_lines(agent: ChainedAgent) -> List[str]:
    drift_fn = drift_fp = op_fn = op_fp = 0
    structural_by_cycle = {
        m.cycle: m
        for m in agent.world.hidden_log
        if m.rule_changed
    }
    for r in agent.records:
        truth_event = structural_by_cycle.get(r.cycle)
        truth_rule_changed = truth_event is not None
        truth_affected_var = truth_event.affected_var if truth_event is not None else -1
        detected = bool(r.detected_drift_vars)
        localized = (
            truth_rule_changed
            and truth_affected_var >= 0
            and truth_affected_var in r.detected_drift_vars
        )
        if truth_rule_changed and not detected:
            drift_fn += 1
        elif not truth_rule_changed and detected:
            drift_fp += 1

        affected = (
            agent.ledger.vars[truth_affected_var]
            if truth_affected_var >= 0 else None
        )
        relevant = (
            truth_rule_changed
            and affected is not None
            and affected.role_for("skip") == "tareth"
        )
        if relevant and not localized:
            op_fn += 1
        elif not relevant and detected:
            op_fp += 1

    restriction_missing = sum(
        1 for d in agent.fit_diagnostics
        if d.failure_class == "restriction_missing"
    )
    true_missing = sum(
        1 for d in agent.fit_diagnostics if not getattr(d, "true_present", True)
    )
    trass_revisions = sum(
        1 for e in agent.ledger.event_log
        if "role REVISED trass" in e and "tareth" in e
    )

    return [
        f"  role_salience={agent.role_salience}",
        "  salience_targets="
        + ("all-visible" if agent.salience_targets is None else ",".join(map(str, sorted(agent.salience_targets)))),
        (
            f"  work: trass_skip={agent.trass_skip_count} "
            f"full_audit={agent.full_audit_count} iv={agent.total_interventions}"
        ),
        f"  operation: FN={op_fn} FP={op_fp}",
        f"  drift: FN={drift_fn} FP={drift_fp}",
        (
            f"  fit: restriction_missing={restriction_missing} "
            f"true_missing={true_missing}"
        ),
        f"  trass_to_tareth_revisions={trass_revisions}",
    ]


def _format_recent_agent_events(agent: ChainedAgent, limit: int = 10) -> List[str]:
    noisy = ("audit DEFERRED", "TEMPORAL_TRASS")
    events = [e for e in agent.ledger.event_log if not any(marker in e for marker in noisy)]
    if not events:
        return ["  none"]
    return [f"  {e}" for e in events[-limit:]]


def concise_report(args: argparse.Namespace, world: CausalWorld, agent: ChainedAgent) -> str:
    """Compact end-of-run report for --quiet."""
    records = agent.records
    cycles = len(records)
    structural = [m for m in world.hidden_log if m.rule_changed]
    structural_cycles = {m.cycle for m in structural}
    localized_hits = sum(
        1 for m in structural
        if any(r.cycle >= m.cycle and m.affected_var in r.detected_drift_vars for r in records)
    )
    false_positive_cycles = sum(
        1 for r in records if r.cycle not in structural_cycles and r.detected_drift_vars
    )
    total_deferred = sum(len(r.deferred_vars) for r in records)
    total_decisions = agent.skip_count + agent.full_audit_count + total_deferred
    skip_rate = agent.skip_count / max(1, total_decisions) * 100

    visible = [agent.ledger.vars[i] for i in range(world.visible_count)]
    cert = sum(1 for n in visible if n.status == "certified")
    prop = sum(1 for n in visible if n.status == "proposed")
    trass = sum(1 for n in visible if n.status == "trass" or n.role_for("skip") == "trass")
    authoritative = sum(1 for n in visible if n.authoritative)
    comp_stored = sum(len(n.compressions) for n in visible)
    comp_hits = sum(n.compression_hits_lifetime for n in visible)
    comp_misses = sum(n.compression_misses_lifetime for n in visible)
    novelty_open = sum(1 for n in agent.ledger.novelty if n.status == "open")

    schedule_line = (
        "  shaped: structural changes at c0002, c0005, c0008, c0011, c0013; "
        "value perturbations otherwise"
        if args.schedule == "shaped"
        else f"  schedule={args.schedule}: {len(structural)} rule-changing mutations"
    )
    extension_line = (
        f"  extension=dreth_extensions modes={sorted(agent.extension_modes)}"
        if agent.extension is not None
        else "  extension=none"
    )
    target_line = (
        "all-visible"
        if agent.salience_targets is None
        else ",".join(map(str, sorted(agent.salience_targets)))
    )

    lines = [
        "── run ────────────────────────────────────────────────",
        (
            f"  mode={args.mode} schedule={args.schedule} seed={args.seed} "
            f"cycles={cycles} vars={world.visible_count}/{world.n_vars} "
            f"noise={args.noise_sigma} audit_budget={agent.priority_audit_budget} "
            f"role_salience={agent.role_salience} salience_targets={target_line}"
        ),
        extension_line,
        schedule_line,
        "",
        "── outcome ────────────────────────────────────────────",
        (
            f"  status: certified={cert} proposed={prop} trass={trass} "
            f"authoritative={authoritative}/{world.visible_count}"
        ),
        (
            f"  drift: localized={localized_hits}/{len(structural)} "
            f"false_positive_cycles={false_positive_cycles}"
        ),
        (
            f"  work: interventions={agent.total_interventions} full_audits={agent.full_audit_count} "
            f"skips={agent.skip_count}/{total_decisions} ({skip_rate:.1f}%) deferred={total_deferred}"
        ),
        (
            f"  cheap paths: trass={agent.trass_skip_count} sentinel={agent.sentinel_skip_count} "
            f"compression={agent.compression_skip_count}; comps={comp_stored} hit/miss={comp_hits}/{comp_misses}"
        ),
        f"  novelty: total={len(agent.ledger.novelty)} open={novelty_open}",
        "",
        "── structural timeline ────────────────────────────────",
        *_format_structural_timeline(world, agent),
        "",
        "── final fits ─────────────────────────────────────────",
        *_format_final_fits(world, agent),
        "",
        "── cycle accounting ────────────────────────────────────",
        *_format_cycle_accounting(args, world, agent),
        "",
        "── operational view ────────────────────────────────────",
        *_format_operational_view(world, agent),
        "",
        "── role-salience audit ─────────────────────────────────",
        *_role_salience_audit_lines(agent),
        "",
        "── recent agent events ────────────────────────────────",
        *_format_recent_agent_events(agent),
    ]
    return "\n".join(lines)


def run() -> None:
    """Top-level entrypoint. Pipeline:
      1. Parse CLI args.
      2. Build the world (random DAG with seed) and the agent.
      3. Print configuration banner and ground-truth DAG.
      4. agent.initialize() — first audit pass on initial visible set.
      5. If --baseline, build a parallel CausalWorld with the same seed
         and a RefitBaseline agent on it, run in lockstep.
      6. For each cycle: world.perturb_by_schedule() produces a mutation;
         dispatch to on_variable_revealed (REVEAL kind) or run_cycle (others).
      7. Print final_summary; if baseline, print cost ratio.
    """
    args = parse_args()
    rng_w = random.Random(args.seed)
    rng_a = random.Random(args.seed + 10_000)

    cost_weights: Optional[Dict[int, float]] = None
    if args.cost_weights:
        weights_list = [float(x) for x in args.cost_weights.split(",")]
        cost_weights = {i: w for i, w in enumerate(weights_list)}

    initial_visible = 1 if args.schedule == "incremental" else args.n_vars
    world = CausalWorld(args.n_vars, rng_w, noise_sigma=args.noise_sigma,
                        initial_visible=initial_visible)
    world.prepare_schedule(args.schedule, args.settle_cycles)
    salience_targets = _parse_salience_targets(args.salience_targets, args.n_vars)
    agent = ChainedAgent(
        world=world, rng=rng_a,
        intervention_budget=args.intervention_budget,
        full_margin_threshold=args.full_margin_threshold,
        sentinel_count=args.sentinel_count, sentinel_pool=args.sentinel_pool,
        promote_after=args.promote_after,
        novelty_weak_streak=args.novelty_weak_streak,
        compression_promote_after=args.compression_promote_after,
        cost_weights=cost_weights,
        priority_audit_budget=args.priority_audit_budget,
        role_salience=args.role_salience,
        salience_targets=salience_targets,
        uncertainty_consolidation_mode=args.uncertainty_consolidation,
        uncertainty_assist_policy=args.uncertainty_assist_policy,
        context_role_index_mode=args.context_role_index or "off",
        context_role_anchor_policy=args.context_role_anchor_policy,
    )
    agent.probe_retention_per_var = args.probe_retention

    # Mode dispatch: load extension module for v29 modes.
    if args.mode != "v28":
        try:
            import dreth_extensions
            agent.extension = dreth_extensions
            if args.mode == "v29":
                agent.extension_modes = {"algebraic", "equiv"}
            elif args.mode == "v29-algebraic-only":
                agent.extension_modes = {"algebraic"}
            elif args.mode == "v29-equiv-only":
                agent.extension_modes = {"equiv"}
            if not args.quiet:
                print(f"  [extension loaded: dreth_extensions, modes={sorted(agent.extension_modes)}]")
        except ImportError as e:
            if args.quiet:
                agent.ledger.event_log.append(
                    f"startup: --mode={args.mode} requested but dreth_extensions.py not found; using v28 behavior"
                )
            else:
                print(f"  [WARNING: --mode={args.mode} requested but dreth_extensions.py "
                      f"not found: {e}. Falling back to v28 behavior.]")

    if not args.quiet:
        print("─" * 56)
        print(f"DRETH CAUSAL v28 | sched={args.schedule} cyc={args.cycles} "
              f"n={args.n_vars} init_vis={initial_visible} σ={args.noise_sigma} "
              f"budget={agent.priority_audit_budget} mode={args.mode} "
              f"role_salience={agent.role_salience} salience_targets="
              f"{'all-visible' if agent.salience_targets is None else sorted(agent.salience_targets)}")
        print("truth: " + " ".join(f"x{i}={world.funcs[i]}({','.join(map(str,world.parents[i]))})"
                                   for i in range(world.n_vars)))
        print("─" * 56)

    agent.initialize()
    if args.baseline:
        rng_w_b = random.Random(args.seed)
        world_b = CausalWorld(args.n_vars, rng_w_b, noise_sigma=args.noise_sigma,
                              initial_visible=initial_visible)
        world_b.prepare_schedule(args.schedule, args.settle_cycles)
        baseline = RefitBaseline(world_b, random.Random(args.seed + 20_000), args.intervention_budget)
        baseline.initialize()

    for cycle in range(1, args.cycles + 1):
        m = world.perturb_by_schedule(cycle, args.schedule,
                                      settle_cycles=args.settle_cycles,
                                      rare_var=args.rare_var,
                                      rare_prob=args.rare_prob)
        # v27: when world reveals new var, agent gets first audit on it
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var, cycle)
        else:
            agent.run_cycle(cycle)
        if args.baseline:
            mb = world_b.perturb_by_schedule(cycle, args.schedule,
                                             settle_cycles=args.settle_cycles,
                                             rare_var=args.rare_var,
                                             rare_prob=args.rare_prob)
            if mb.kind != "REVEAL":
                baseline.run_cycle(mb)

    if args.quiet:
        print(concise_report(args, world, agent))
    else:
        print(agent.final_summary())

    if args.baseline:
        b_tp = sum(1 for c, rc, d in baseline.records if rc and d)
        b_fn = sum(1 for c, rc, d in baseline.records if rc and not d)
        b_fp = sum(1 for c, rc, d in baseline.records if not rc and d)
        b_tn = sum(1 for c, rc, d in baseline.records if not rc and not d)
        print("\n── baseline (refit-every-var-every-cycle) ────────────")
        print(f"  iv={baseline.total_interventions} | "
              f"conf TP={b_tp} FN={b_fn} FP={b_fp} TN={b_tn}")
        ratio = baseline.total_interventions / max(1, agent.total_interventions)
        print(f"  cost ratio (baseline/framework): {ratio:.2f}x")

    if not args.quiet:
        print("\n── per-var ────────────────────────────────────────────")
        for i in range(world.visible_count):
            print(f"  {agent.ledger.vars[i].display()}")

    if not args.quiet:
        print("\n── truth log (last 12) ────────────────────────────────")
        for m in world.hidden_log[-12:]:
            print(f"  c{m.cycle:04d} {m.description}")


if __name__ == "__main__":
    run()
