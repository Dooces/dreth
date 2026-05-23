#!/usr/bin/env python3
"""
visualize.py — interactive HTML audit report for the dreth causal learning agent.

Runs a simulation, captures snapshots at regular intervals, and produces a
self-contained HTML file you can open in any browser. Shows:

  - Agent's believed causal graph at each snapshot (click to navigate)
  - Status timeline: how each var's cert state evolves over cycles
  - Per-var detail on click (parents, sentinels, cert age, collapse log)
  - Regime register summary
  - Key metrics: interventions, certified count, live/inert partition

No oracle access. All data is purely agent-observable.

Usage:
    python scripts/visualize.py --vars 20 --cycles 500 --seed 42
    python scripts/visualize.py --vars 80 --cycles 2000 --schedule periodic_shifts
    python scripts/visualize.py --vars 40 --cycles 1000 --snapshots 20 --out audit.html
"""

import argparse
import json
import random
import sys
import math
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, __file__.replace("/scripts/visualize.py", ""))

from dreth.world import CausalWorld
from dreth.agent import ChainedAgent


# ── snapshot capture ───────────────────────────────────────────────────────────

def _capture_snapshot(agent: ChainedAgent, world: CausalWorld, cycle: int) -> Dict[str, Any]:
    inert = agent._inert_vars if hasattr(agent, "_inert_vars") else set()
    live  = agent._live_set  if hasattr(agent, "_live_set") and agent._live_set is not None else set(range(world.visible_count))

    vars_data = []
    for v in range(world.visible_count):
        n = agent.ledger.vars[v]
        tier = min(agent._consequence_tier(v), 2)
        role = n.role_for("skip")
        vars_data.append({
            "id":      v,
            "status":  n.status,
            "role":    role or "untested",
            "parents": list(n.parents),
            "func":    n.func,
            "tier":    tier,
            "is_inert":  v in inert,
            "is_live":   v in live,
            "n_sentinels": len(n.sentinels),
            "full_audits": n.full_audits,
            "skip_count":  n.skip_count,
            "first_cert":  n.first_certified_cycle,
            "last_changed": n.last_changed_cycle,
            "cost_weight": round(n.cost_weight, 3),
            "consec_fails": n.consecutive_sentinel_failures,
            "strong_obs":  n.strong_observations,
            "collapse_log": n.collapse_log[-6:],
        })

    visible = [agent.ledger.vars[i] for i in range(world.visible_count)]
    vis_ids  = set(range(world.visible_count))
    n_inert  = len(inert & vis_ids)
    active   = vis_ids - inert
    certified = sum(1 for i, n in enumerate(visible) if i in active and n.status == "certified")
    trass     = sum(1 for i, n in enumerate(visible) if i in active and n.status == "trass")
    proposed  = sum(1 for i, n in enumerate(visible) if i in active and n.status in ("proposed", "uncertain"))
    n_live    = len(live  & vis_ids)

    total = agent.skip_count + agent.full_audit_count
    skip_pct = agent.skip_count / max(1, total) * 100

    return {
        "cycle": cycle,
        "vars": vars_data,
        "metrics": {
            "certified": certified,
            "trass":     trass,
            "proposed":  proposed,
            "inert":     n_inert,
            "live":      n_live,
            "skip_pct":  round(skip_pct, 1),
            "interventions": agent.total_interventions,
            "sent_skips": agent.sentinel_skip_count,
            "full_audits": agent.full_audit_count,
        },
        "regimes": agent.regime_register.summary(),
    }


def run_simulation(cfg: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    """Run the simulation and return (snapshots, layout)."""
    rng_w = random.Random(cfg.seed)
    rng_a = random.Random(cfg.seed + 10_000)

    initial_visible = 1 if cfg.schedule == "incremental" else cfg.n_vars
    world = CausalWorld(cfg.n_vars, rng_w, noise_sigma=cfg.noise_sigma,
                        initial_visible=initial_visible)
    agent = ChainedAgent(
        world=world, rng=rng_a,
        sentinel_count=5, sentinel_pool=60,
        promote_after=2,
        priority_audit_budget=max(1, cfg.n_vars // 2),
        consequence_weight=True,
    )
    agent.initialize()

    interval = max(1, cfg.cycles // cfg.snapshots)
    snapshots = []

    snapshots.append(_capture_snapshot(agent, world, 0))

    for cycle in range(1, cfg.cycles + 1):
        m = world.perturb_by_schedule(cycle, cfg.schedule,
                                      settle_cycles=cfg.settle_cycles)
        if m.kind == "REVEAL":
            agent.on_variable_revealed(m.affected_var, cycle)
        else:
            agent.run_cycle(m)

        if cycle % interval == 0 or cycle == cfg.cycles:
            snapshots.append(_capture_snapshot(agent, world, cycle))

    layout = _compute_layout(agent, world)
    return snapshots, layout


# ── layout ─────────────────────────────────────────────────────────────────────

def _compute_layout(agent: ChainedAgent, world: CausalWorld) -> Dict:
    """Compute DAG node positions from agent's final believed parent structure.

    Non-inert vars: layered left-to-right by topological depth in agent's model.
    Inert vars:     separate row at bottom.
    """
    inert = agent._inert_vars if hasattr(agent, "_inert_vars") else set()
    n = world.visible_count

    # Build depth map from agent's believed parents
    depth: Dict[int, int] = {}
    def get_depth(v: int) -> int:
        if v in depth:
            return depth[v]
        parents = list(agent.ledger.vars[v].parents)
        if not parents:
            depth[v] = 0
        else:
            depth[v] = 1 + max(get_depth(p) for p in parents if p != v)
        return depth[v]

    for v in range(n):
        if v not in inert:
            try:
                get_depth(v)
            except RecursionError:
                depth[v] = 0

    # Group active vars by depth
    from collections import defaultdict
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for v in range(n):
        if v not in inert:
            d = depth.get(v, 0)
            by_depth[d].append(v)
    for d in by_depth:
        by_depth[d].sort()

    inert_list = sorted(inert & set(range(n)))

    # Spacing constants
    X_STEP  = 110
    Y_STEP  = 36
    MARGIN  = 60
    INERT_Y_OFFSET = 40  # extra gap before inert section

    max_depth = max(by_depth.keys()) if by_depth else 0
    max_col_height = max((len(vs) for vs in by_depth.values()), default=1)

    active_height = max_col_height * Y_STEP
    inert_cols = math.ceil(math.sqrt(max(1, len(inert_list))))
    inert_rows = math.ceil(len(inert_list) / max(1, inert_cols))
    inert_height = inert_rows * Y_STEP

    total_width  = (max_depth + 1) * X_STEP + 2 * MARGIN
    active_h_px  = active_height + 2 * MARGIN
    total_height = active_h_px + INERT_Y_OFFSET + inert_height + MARGIN

    positions: Dict[int, Tuple[float, float]] = {}

    for d, vs in by_depth.items():
        col_h = len(vs) * Y_STEP
        y_start = MARGIN + (active_height - col_h) / 2
        for i, v in enumerate(vs):
            x = MARGIN + d * X_STEP
            y = y_start + i * Y_STEP
            positions[v] = (x, y)

    inert_y_base = active_h_px + INERT_Y_OFFSET
    for i, v in enumerate(inert_list):
        col = i % inert_cols
        row = i // inert_cols
        x = MARGIN + col * (X_STEP * 0.6)
        y = inert_y_base + row * Y_STEP
        positions[v] = (x, y)

    nodes = [
        {"id": v, "x": round(positions[v][0], 1), "y": round(positions[v][1], 1),
         "inert": v in inert}
        for v in range(n) if v in positions
    ]

    return {
        "nodes": nodes,
        "width":  round(total_width, 0),
        "height": round(total_height, 0),
        "inert_y": round(active_h_px + INERT_Y_OFFSET / 2, 0),
    }


# ── HTML generation ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dreth Agent Audit — {title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
  #app {{ display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

  /* top bar */
  #topbar {{ display: flex; align-items: center; gap: 12px; padding: 8px 16px;
             background: #1e293b; border-bottom: 1px solid #334155; flex-shrink: 0; }}
  #topbar h1 {{ font-size: 14px; font-weight: 600; color: #94a3b8; }}
  #cycle-label {{ font-size: 13px; color: #f8fafc; font-weight: 700; min-width: 90px; }}
  #slider {{ flex: 1; max-width: 400px; accent-color: #6366f1; }}
  button {{ background: #334155; border: none; color: #cbd5e1; padding: 4px 12px;
            border-radius: 4px; cursor: pointer; font-size: 12px; }}
  button:hover {{ background: #475569; }}

  /* main layout */
  #main {{ display: flex; flex: 1; overflow: hidden; gap: 1px; background: #334155; }}
  #graph-wrap {{ flex: 1; overflow: auto; background: #0f172a; position: relative; }}
  #sidebar {{ width: 260px; overflow-y: auto; background: #1e293b; padding: 12px;
              flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; }}

  /* sidebar panels */
  .panel {{ background: #0f172a; border-radius: 6px; padding: 10px; }}
  .panel h2 {{ font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.05em; margin-bottom: 8px; }}
  .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
  .metric {{ background: #1e293b; border-radius: 4px; padding: 6px 8px; }}
  .metric-val {{ font-size: 20px; font-weight: 700; color: #f8fafc; line-height: 1.2; }}
  .metric-lbl {{ font-size: 10px; color: #64748b; text-transform: uppercase; }}

  /* status bar */
  .status-bar {{ display: flex; height: 16px; border-radius: 3px; overflow: hidden; gap: 1px; margin: 4px 0; }}
  .sb-seg {{ height: 100%; transition: width 0.3s; }}

  /* regime list */
  .regime {{ font-size: 11px; color: #94a3b8; padding: 3px 0; border-bottom: 1px solid #1e293b; }}
  .regime:last-child {{ border-bottom: none; }}

  /* detail panel */
  #detail {{ font-size: 11px; line-height: 1.7; }}
  #detail .dk {{ color: #64748b; }}
  #detail .dv {{ color: #e2e8f0; }}
  #detail .log {{ color: #94a3b8; font-size: 10px; margin-top: 4px; }}

  /* timeline */
  #timeline-wrap {{ height: 120px; flex-shrink: 0; overflow-x: auto; overflow-y: hidden;
                    background: #0f172a; border-top: 1px solid #334155; }}

  /* svg graph nodes / edges */
  .node-circle {{ cursor: pointer; stroke-width: 1.5; transition: opacity 0.15s; }}
  .node-circle:hover {{ opacity: 0.75; stroke-width: 3; }}
  .node-label {{ font-size: 9px; fill: #e2e8f0; pointer-events: none; text-anchor: middle; dominant-baseline: central; }}
  .edge {{ stroke-opacity: 0.4; fill: none; marker-end: url(#arrow); }}
  .edge.cert {{ stroke: #22c55e; stroke-opacity: 0.6; }}
  .inert-divider {{ stroke: #334155; stroke-dasharray: 6,4; }}
  .inert-label {{ font-size: 10px; fill: #475569; }}
  .selected {{ stroke: #f8fafc !important; stroke-width: 3 !important; opacity: 1 !important; }}
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <h1>dreth audit &mdash; {title}</h1>
    <button id="btn-prev">◀</button>
    <span id="cycle-label">—</span>
    <button id="btn-next">▶</button>
    <input type="range" id="slider" min="0" value="0">
    <span style="font-size:11px;color:#64748b" id="snap-info"></span>
  </div>

  <div id="main">
    <div id="graph-wrap">
      <svg id="graph-svg" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div id="sidebar">
      <div class="panel">
        <h2>Metrics</h2>
        <div class="metric-grid" id="metric-grid"></div>
        <div style="margin-top:8px">
          <div style="font-size:10px;color:#64748b;margin-bottom:3px">Status distribution</div>
          <div class="status-bar" id="status-bar"></div>
          <div style="font-size:10px;color:#64748b;margin-top:3px" id="status-legend"></div>
        </div>
      </div>
      <div class="panel">
        <h2>Selected var</h2>
        <div id="detail"><span style="color:#475569">click a node</span></div>
      </div>
      <div class="panel">
        <h2>Regimes</h2>
        <div id="regime-panel" style="font-size:11px;color:#94a3b8;white-space:pre-wrap;max-height:220px;overflow-y:auto"></div>
      </div>
    </div>
  </div>

  <div id="timeline-wrap">
    <canvas id="timeline-canvas"></canvas>
  </div>
</div>

<script>
const SNAPSHOTS = {snapshots_json};
const LAYOUT    = {layout_json};
const N_VARS    = {n_vars};
const N_SNAPS   = SNAPSHOTS.length;

const STATUS_COLOR = {{
  certified: '#22c55e',
  trass:     '#3b82f6',
  uncertain: '#f97316',
  proposed:  '#94a3b8',
  quarantined: '#e11d48',
}};
const INERT_COLOR  = '#475569';
const ROLE_STROKE  = {{ tareth: '#facc15', trass: '#3b82f6', noise_floor: '#a78bfa', untested: '#334155' }};

let currentIdx = 0;
let selectedVar = null;

// ── build static SVG structure ────────────────────────────────────────────────
const svg = document.getElementById('graph-svg');
svg.setAttribute('width',  LAYOUT.width);
svg.setAttribute('height', LAYOUT.height);

// arrowhead marker
svg.innerHTML = `<defs>
  <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
    <path d="M0,0 L0,6 L6,3 z" fill="#475569"/>
  </marker>
  <marker id="arrow-cert" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
    <path d="M0,0 L0,6 L6,3 z" fill="#22c55e"/>
  </marker>
</defs>`;

// inert section divider
if (LAYOUT.inert_y > 0) {{
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', 20); line.setAttribute('x2', LAYOUT.width - 20);
  line.setAttribute('y1', LAYOUT.inert_y); line.setAttribute('y2', LAYOUT.inert_y);
  line.setAttribute('class', 'inert-divider');
  svg.appendChild(line);
  const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  lbl.setAttribute('x', 24); lbl.setAttribute('y', LAYOUT.inert_y - 6);
  lbl.setAttribute('class', 'inert-label'); lbl.textContent = 'inert (startup-screened)';
  svg.appendChild(lbl);
}}

// edge layer
const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
edgeLayer.id = 'edge-layer';
svg.appendChild(edgeLayer);

// node layer
const nodeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
nodeLayer.id = 'node-layer';
svg.appendChild(nodeLayer);

// build node lookup
const posById = {{}};
LAYOUT.nodes.forEach(n => {{ posById[n.id] = n; }});

// create a circle + label per var
const TIER_R = [7, 10, 13];
LAYOUT.nodes.forEach(n => {{
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.dataset.varid = n.id;
  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circle.setAttribute('class', 'node-circle');
  circle.setAttribute('id', `nc-${{n.id}}`);
  circle.setAttribute('cx', n.x); circle.setAttribute('cy', n.y);
  circle.setAttribute('r', 7);
  circle.addEventListener('click', () => selectVar(n.id));
  const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  lbl.setAttribute('class', 'node-label');
  lbl.setAttribute('x', n.x); lbl.setAttribute('y', n.y);
  lbl.textContent = n.id;
  g.appendChild(circle); g.appendChild(lbl);
  nodeLayer.appendChild(g);
}});

// ── render snapshot ───────────────────────────────────────────────────────────
function render(idx) {{
  const snap = SNAPSHOTS[idx];
  document.getElementById('cycle-label').textContent = `Cycle ${{snap.cycle}}`;
  document.getElementById('snap-info').textContent = `${{idx+1}} / ${{N_SNAPS}}`;
  document.getElementById('slider').value = idx;

  // update nodes
  snap.vars.forEach(v => {{
    const circle = document.getElementById(`nc-${{v.id}}`);
    if (!circle) return;
    const r = TIER_R[v.tier] || 7;
    circle.setAttribute('r', r);
    const col = v.is_inert ? INERT_COLOR : (STATUS_COLOR[v.status] || '#64748b');
    circle.setAttribute('fill', col);
    const stroke = ROLE_STROKE[v.role] || ROLE_STROKE.untested;
    circle.setAttribute('stroke', stroke);
    const sd = v.status === 'uncertain' ? '4,2' : 'none';
    circle.setAttribute('stroke-dasharray', sd);
    if (selectedVar === v.id) circle.classList.add('selected');
    else circle.classList.remove('selected');
  }});

  // rebuild edges
  edgeLayer.innerHTML = '';
  const varById = {{}};
  snap.vars.forEach(v => {{ varById[v.id] = v; }});

  snap.vars.forEach(v => {{
    v.parents.forEach(p => {{
      const src = posById[p]; const dst = posById[v.id];
      if (!src || !dst) return;
      const isCert = !v.is_inert && !varById[p]?.is_inert && v.status === 'certified';
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      // offset line end to avoid overlap with node circle
      const r = TIER_R[v.tier] || 7;
      const dx = dst.x - src.x; const dy = dst.y - src.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const ex = dst.x - (dx/dist)*r; const ey = dst.y - (dy/dist)*r;
      line.setAttribute('x1', src.x); line.setAttribute('y1', src.y);
      line.setAttribute('x2', ex);    line.setAttribute('y2', ey);
      line.setAttribute('class', isCert ? 'edge cert' : 'edge');
      line.setAttribute('stroke', isCert ? '#22c55e' : '#475569');
      line.setAttribute('stroke-width', isCert ? 1.5 : 1);
      line.setAttribute('marker-end', isCert ? 'url(#arrow-cert)' : 'url(#arrow)');
      edgeLayer.appendChild(line);
    }});
  }});

  // metrics panel
  const m = snap.metrics;
  const total = m.certified + m.trass + m.proposed + m.inert;
  const mgrid = document.getElementById('metric-grid');
  mgrid.innerHTML = [
    ['cert', m.certified], ['trass', m.trass],
    ['proposed', m.proposed], ['inert', m.inert],
    ['skip%', m.skip_pct + '%'], ['iv', m.interventions.toLocaleString()],
    ['audits', m.full_audits], ['live', m.live],
  ].map(([l, v]) => `<div class="metric"><div class="metric-val">${{v}}</div><div class="metric-lbl">${{l}}</div></div>`).join('');

  // status bar — disjoint: cert + trass + proposed_active + inert = n_vars
  const bar = document.getElementById('status-bar');
  const barTotal = m.certified + m.trass + m.proposed + m.inert;
  const parts = [
    ['cert',     m.certified, '#22c55e'],
    ['trass',    m.trass,     '#3b82f6'],
    ['proposed', m.proposed,  '#94a3b8'],
    ['inert',    m.inert,     '#475569'],
  ];
  bar.innerHTML = parts.map(([, cnt, col]) =>
    `<div class="sb-seg" style="width:${{barTotal?cnt/barTotal*100:0}}%;background:${{col}}"></div>`
  ).join('');
  document.getElementById('status-legend').textContent =
    parts.map(([l, c]) => `${{l}}=${{c}}`).join('  ');

  // regime panel
  document.getElementById('regime-panel').textContent = snap.regimes || '(none yet)';

  // refresh selected var detail if any
  if (selectedVar !== null) renderDetail(selectedVar, snap);
}}

function selectVar(vid) {{
  // deselect previous
  if (selectedVar !== null) {{
    const prev = document.getElementById(`nc-${{selectedVar}}`);
    if (prev) prev.classList.remove('selected');
  }}
  selectedVar = vid;
  document.getElementById(`nc-${{vid}}`).classList.add('selected');
  renderDetail(vid, SNAPSHOTS[currentIdx]);
}}

function renderDetail(vid, snap) {{
  const v = snap.vars.find(x => x.id === vid);
  if (!v) return;
  const lines = [
    ['var',       `x${{v.id}}`],
    ['status',    v.status],
    ['role',      v.role],
    ['tier',      `T${{v.tier}}`],
    ['parents',   v.parents.length ? v.parents.map(p=>`x${{p}}`).join(', ') : '(none)'],
    ['func',      v.func],
    ['sentinels', v.n_sentinels],
    ['audits',    v.full_audits],
    ['skips',     v.skip_count],
    ['strong_obs',v.strong_obs],
    ['first_cert',v.first_cert || '—'],
    ['last_chg',  v.last_changed || '—'],
    ['cost_wt',   v.cost_weight],
    ['csec_fail', v.consec_fails],
    ['inert',     v.is_inert ? 'yes' : 'no'],
    ['live',      v.is_live  ? 'yes' : 'no'],
  ];
  let html = lines.map(([k, val]) =>
    `<div><span class="dk">${{k}}: </span><span class="dv">${{val}}</span></div>`
  ).join('');
  if (v.collapse_log.length) {{
    html += `<div class="log">${{v.collapse_log.join('<br>')}}</div>`;
  }}
  document.getElementById('detail').innerHTML = html;
}}

// ── timeline canvas ───────────────────────────────────────────────────────────
function renderTimeline() {{
  const canvas = document.getElementById('timeline-canvas');
  const W = Math.max(600, N_SNAPS * 8);
  const H = 120;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, W, H);

  const cellW = W / N_SNAPS;
  const cellH = H / N_VARS;

  SNAPSHOTS.forEach((snap, si) => {{
    snap.vars.forEach(v => {{
      const col = v.is_inert ? INERT_COLOR : (STATUS_COLOR[v.status] || '#334155');
      ctx.fillStyle = col;
      ctx.fillRect(si * cellW, v.id * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
    }});
  }});

  // draw cursor
  canvas.addEventListener('click', e => {{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const si = Math.floor(x / (W / N_SNAPS));
    if (si >= 0 && si < N_SNAPS) {{
      currentIdx = si;
      render(currentIdx);
    }}
  }});
}}

// ── nav controls ─────────────────────────────────────────────────────────────
document.getElementById('btn-prev').addEventListener('click', () => {{
  if (currentIdx > 0) {{ currentIdx--; render(currentIdx); }}
}});
document.getElementById('btn-next').addEventListener('click', () => {{
  if (currentIdx < N_SNAPS - 1) {{ currentIdx++; render(currentIdx); }}
}});
const slider = document.getElementById('slider');
slider.max = N_SNAPS - 1;
slider.addEventListener('input', () => {{
  currentIdx = parseInt(slider.value);
  render(currentIdx);
}});

// keyboard navigation
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft'  && currentIdx > 0)          {{ currentIdx--; render(currentIdx); }}
  if (e.key === 'ArrowRight' && currentIdx < N_SNAPS - 1) {{ currentIdx++; render(currentIdx); }}
}});

// init
renderTimeline();
render(0);
</script>
</body>
</html>
"""


def generate_html(snapshots: List[Dict], layout: Dict,
                  cfg: argparse.Namespace) -> str:
    title = f"n={cfg.n_vars} cyc={cfg.cycles} seed={cfg.seed} sched={cfg.schedule}"
    return _HTML_TEMPLATE.format(
        title=title,
        n_vars=max(s["vars"][-1]["id"] + 1 if s["vars"] else 1 for s in snapshots),
        snapshots_json=json.dumps(snapshots),
        layout_json=json.dumps(layout),
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate a self-contained HTML audit report for dreth agent runs"
    )
    p.add_argument("--vars",      type=int, default=20)
    p.add_argument("--cycles",    type=int, default=500)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--schedule",  default="periodic_shifts",
                   choices=["incremental", "periodic_shifts", "novelty", "shaped"])
    p.add_argument("--settle-cycles", type=int, default=8)
    p.add_argument("--noise-sigma",   type=float, default=0.02)
    p.add_argument("--snapshots",     type=int, default=50,
                   help="number of snapshots to capture (default: 50)")
    p.add_argument("--out", default=None,
                   help="output file path (default: audit_<vars>_<cycles>_<seed>.html)")
    cfg = p.parse_args()

    print(f"running: n={cfg.vars} cyc={cfg.cycles} seed={cfg.seed} sched={cfg.schedule} "
          f"snaps={cfg.snapshots} ...", flush=True)

    cfg.n_vars = cfg.vars
    cfg.settle_cycles = cfg.settle_cycles
    cfg.noise_sigma = cfg.noise_sigma

    snapshots, layout = run_simulation(cfg)
    html = generate_html(snapshots, layout, cfg)

    out = cfg.out or f"audit_{cfg.vars}_{cfg.cycles}_{cfg.seed}.html"
    with open(out, "w") as f:
        f.write(html)

    print(f"wrote {out}  ({len(html)//1024} KB, {len(snapshots)} snapshots)")
    print(f"open in browser: file://{out}")


if __name__ == "__main__":
    main()
