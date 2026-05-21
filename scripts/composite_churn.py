#!/usr/bin/env python3
"""
composite_churn.py — diagnose whether composite_failure is repeated churn.

Runs a single agent, then parses event_log to report per-(a,b)-pair:
  - how many times each pair was FORMED (joint false-trass found)
  - how many times each pair was REVOKED (interaction gone)
  - the cycle sequence of formations and revocations

A pair that forms-then-revokes-then-forms-then-revokes repeatedly is churn.
A pair that forms once and revokes once is a genuine regime change.

Usage:
    python scripts/composite_churn.py
    python scripts/composite_churn.py --vars 12 --cycles 800 --seed 7
"""

import argparse
import random
import re
import sys
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

sys.path.insert(0, __file__.replace("/scripts/composite_churn.py", ""))

from dreth.world import CausalWorld, HiddenMutation
from dreth.agent import ChainedAgent


Event = namedtuple("Event", ["cycle", "kind", "pair"])


def parse_composite_events(event_log: List[str]) -> List[Event]:
    """Parse JOINT FALSE-TRASS (FORMED) and composite REVOKED events from
    the ledger event_log. Returns list of Events in log order."""
    events = []
    # "c42: x3,x7 JOINT FALSE-TRASS (5/8 trials showed interaction)"
    formed_re = re.compile(r"c(\d+): x(\d+),x(\d+) JOINT FALSE-TRASS")
    # "c55: x3,x7 composite REVOKED ..."
    revoked_re = re.compile(r"c(\d+): x(\d+),x(\d+) composite REVOKED")
    for line in event_log:
        m = formed_re.search(line)
        if m:
            cycle, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pair = (min(a, b), max(a, b))
            events.append(Event(cycle, "FORMED", pair))
            continue
        m = revoked_re.search(line)
        if m:
            cycle, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pair = (min(a, b), max(a, b))
            events.append(Event(cycle, "REVOKED", pair))
    return events


def analyze_churn(events: List[Event]) -> None:
    formed: Dict[Tuple, List[int]] = defaultdict(list)
    revoked: Dict[Tuple, List[int]] = defaultdict(list)
    for e in events:
        if e.kind == "FORMED":
            formed[e.pair].append(e.cycle)
        else:
            revoked[e.pair].append(e.cycle)

    all_pairs = set(formed) | set(revoked)
    if not all_pairs:
        print("  no composite events in event_log")
        return

    total_formed  = sum(len(v) for v in formed.values())
    total_revoked = sum(len(v) for v in revoked.values())
    churn_pairs   = [(p, len(formed[p]), len(revoked[p]))
                     for p in all_pairs if len(formed[p]) + len(revoked[p]) > 2]

    print(f"  total composite FORMED:  {total_formed}")
    print(f"  total composite REVOKED: {total_revoked}")
    print(f"  unique (a,b) pairs:      {len(all_pairs)}")

    churn_pairs.sort(key=lambda x: x[1] + x[2], reverse=True)
    if churn_pairs:
        print(f"\n  pairs with >1 form+revoke (churn candidates):")
        for pair, nf, nr in churn_pairs:
            a, b = pair
            f_cycles = formed[pair]
            r_cycles = revoked[pair]
            # Interleave and show the sequence
            seq = sorted(
                [("F", c) for c in f_cycles] + [("R", c) for c in r_cycles],
                key=lambda x: x[1]
            )
            seq_str = " → ".join(f"{k}{c}" for k, c in seq)
            print(f"    x{a},x{b}: {nf}× formed  {nr}× revoked  |  {seq_str}")
    else:
        print(f"\n  no churn: every pair formed and revoked at most once")
        print(f"  remaining composites at end:")
        for pair in all_pairs:
            if len(formed[pair]) == 1 and len(revoked[pair]) == 0:
                a, b = pair
                print(f"    x{a},x{b}: formed once, never revoked (still live)")
            elif len(formed[pair]) == 1 and len(revoked[pair]) == 1:
                a, b = pair
                fd, rd = formed[pair][0], revoked[pair][0]
                print(f"    x{a},x{b}: formed c{fd}, revoked c{rd} (regime change)")

    print(f"\n  all pairs detail:")
    for pair in sorted(all_pairs):
        a, b = pair
        nf, nr = len(formed[pair]), len(revoked[pair])
        f_cycles = formed[pair]
        r_cycles = revoked[pair]
        seq = sorted(
            [("F", c) for c in f_cycles] + [("R", c) for c in r_cycles],
            key=lambda x: x[1]
        )
        seq_str = " → ".join(f"{k}{c}" for k, c in seq)
        status = "CHURN" if nf + nr > 2 else ("live" if nf > nr else "gone")
        print(f"    x{a},x{b} [{status}]: {nf}× formed  {nr}× revoked  |  {seq_str}")


def main():
    p = argparse.ArgumentParser(description="Composite churn diagnostic")
    p.add_argument("--vars",    type=int, default=12)
    p.add_argument("--cycles",  type=int, default=800)
    p.add_argument("--seed",    type=int, default=7)
    p.add_argument("--schedule", default="incremental",
                   choices=["incremental", "periodic_shifts", "novelty", "shaped"])
    p.add_argument("--settle-cycles", type=int, default=8)
    p.add_argument("--noise-sigma", type=float, default=0.02)
    args = p.parse_args()

    print(f"composite_churn: n={args.vars} cycles={args.cycles} seed={args.seed} "
          f"schedule={args.schedule}", flush=True)

    rng_w = random.Random(args.seed)
    rng_a = random.Random(args.seed + 10_000)
    initial_visible = 1 if args.schedule == "incremental" else args.vars
    world = CausalWorld(args.vars, rng_w, noise_sigma=args.noise_sigma,
                        initial_visible=initial_visible)
    agent = ChainedAgent(
        world=world, rng=rng_a,
        sentinel_count=5, sentinel_pool=60,
        promote_after=2,
        priority_audit_budget=max(1, args.vars // 2),
    )
    agent.initialize()
    for cycle in range(1, args.cycles + 1):
        m = world.perturb_by_schedule(cycle, args.schedule,
                                      settle_cycles=args.settle_cycles)
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var, cycle)
        else:
            agent.run_cycle(m)

    print(f"  run complete. composite_skip_count={agent.composite_skip_count}")
    print(f"  live composites at end: {len(agent.ledger.composites)}")
    for cn in agent.ledger.composites:
        a, b = cn.members
        print(f"    x{a},x{b} sentinel_var=x{cn.sentinel_var} "
              f"certified_c{cn.certified_at_cycle} "
              f"probe=({cn.probe_val_a:.3f},{cn.probe_val_b:.3f})")

    # ── cert revoked_by="composite_failure" from ALL certs ─────────────────
    # Two sources of false_trass_contradiction → revoked_by="composite_failure":
    #   Path A: _check_composites — actual composite sentinel failed
    #   Path B: _install_var line ~982 — trass var was picked as a parent
    # These are conflated in the revoked_by counter. Separate them here.
    visible = [agent.ledger.vars[i] for i in range(world.visible_count)]
    cert_composite_failure = sum(
        1 for n in visible
        for c in n.certificates.values()
        if getattr(c, "revoked_by", None) == "composite_failure"
    )
    # Count event_log entries from the trass-parent path (distinct from REVOKED)
    trass_parent_triggers = sum(
        1 for e in agent.ledger.event_log
        if "role re-test triggered" in e and "trass status" in e
    )

    print(f"\n  revoked_by='composite_failure' in certs (live vars): {cert_composite_failure}")
    print(f"  event_log 'trass-parent re-test' triggers:          {trass_parent_triggers}")
    print(f"  (the cert count includes both path A=composite-sentinel and path B=trass-parent)")

    print()
    events = parse_composite_events(agent.ledger.event_log)
    print(f"  path A (composite sentinel) events parsed: {len(events)}")
    formed_count  = sum(1 for e in events if e.kind == "FORMED")
    revoked_count = sum(1 for e in events if e.kind == "REVOKED")
    print(f"    FORMED:  {formed_count}")
    print(f"    REVOKED: {revoked_count}")
    print()
    analyze_churn(events)


if __name__ == "__main__":
    main()
