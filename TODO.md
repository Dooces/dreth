# TODO

This file lists remaining work only. The current direction treats Dreth as a
search engine whose indexed object is search narrowing, not memory, cognition,
or graph storage.

Current assessment:

```text
The repo already has search-index components:
  - PersistentNethraIndex holds persistent handles and a ProjectionIndex.
  - ProjectionIndex maps target/context/hook to relevant handles.
  - NethraAssimilator maintains anchor, perspective, role, topology, and residual indexes.
  - ExperienceEvent records behavior effects and candidate-reduction feedback.

The runtime critical path is still too record-like:
  rank_candidates/rank_probes -> atom query -> records touching those atoms -> reorder.

The intended search-engine path is:
  target/context/hook -> route/projection lookup -> narrowed candidate/probe region
  -> attributed outcome -> route score update/residual charge.
```

The old expression-mining agenda is removed as a primary target. Offline
expressions, regimes, and graph summaries are deferred until route identity,
route attribution, and runtime route use are measured.

---

## Non-Negotiable Invariants

- [x] A nethra is a learned search-space narrowing operator, not memory content.
- [x] Persistence is storage only; the operating object is a search route.
- [ ] Compaction updates route statistics; compaction is not the goal by itself.
- [x] Graph structure is substrate; active use must be via scoped route projection.
- [x] Multiple routes may touch the same atoms, records, or graph structure.
- [x] Structure overlap, recurrence, graph proximity, index membership, provider confidence, morphology, pressure, or temporal correlation is not authority.
- [x] Cross-context overlap is downgraded to hint/proposal until local route outcomes earn stronger use.
- [x] `tareth` means a route is allowed to narrow the active search space in that context.
- [x] `trass` means a route is passive for active narrowing here, but may collect residuals and evidence for future route formation.
- [x] Trass is not deletion, dormancy, or inactivity.
- [x] Best-available is fallback, not hard authority.
- [x] Unresolved preserves ambiguity; it does not license action authority.
- [x] Residual pressure is not authority.
- [ ] Residual buckets are bounded, scoped, and attributable.
- [ ] Persistent unresolved residual growth in a stable closed context is a warning/failure signal.
- [x] Search routes must not issue certificates, revoke authority, suppress skips, replace fit, force probes, increase monitoring, or increase repair priority.
- [x] Hidden truth/debug manifest fields must not be read by runtime matching, route scoring, route formation, compaction, residual classification, or assist logic.
- [x] Record-only paths must match off-mode behavior.
- [x] Assist paths must be attributed: which route changed ordering/probes/filters and whether the outcome improved.
- [ ] Broad unresolved status, broad role equality, giant uncertainty clusters, generic uncertainty signals, or repeated background familiarity must not qualify as local anchors or runtime action triggers by themselves.
- [ ] Passive observers and residual examples must be capped; route formation must not create a second unbounded audit loop.

---

## Immediate Target: First-Class Search Routes

- [x] Add first-class route rows with `entry_kind="nethra_search_route"`.
- [x] Keep legacy `NethraMemoryRecord`/`nethra_mind_node` names as compatibility storage names only; do not add more route fields to `NethraMindNode`.
- [x] Define a route model, likely in a new module such as `dreth/learner/nethra_route_index.py`:

```text
SearchRoute:
  route_id
  nethra_id
  operation_hook
  target_anchor
  trigger_anchors
  candidate_region
  deferred_region
  residual_bucket_key
  probe_region
  invalidators
  role_state
  use_right
  saved_search_count
  wasted_search_count
  miss_count
  success_count
  failure_count
  first_seen
  last_seen
```

- [x] Add a `SearchRouteIndex`/`NethraRouteIndex` that queries by `(target_anchor, context_key, operation_hook)`.
- [x] Preserve `ProjectionIndex` as the existing materialized route view or rename it only after compatibility tests are in place.
- [x] Add route identity separate from record identity so outcome feedback can strengthen/weaken the route that actually changed search.
- [x] Represent tareth/trass as route use states, not global handle identities.
- [x] Store route invalidators and residual buckets explicitly.
- [x] Keep route rows proposal/action-scaffold only unless runtime local evidence already permits the corresponding use-right.

---

## Runtime Critical Path

- [x] Change `PersistentNethraIndex.rank_candidates()` to ask the route/projection index first.
- [x] Keep atom lookup as a bounded fallback only when no route/projection match exists.
- [x] Make candidate ordering use route `candidate_region`, route score, and route invalidators rather than all touched atoms.
- [x] Record the applied `route_id` in each `ExperienceEvent`.
- [ ] Record saved work, wasted work, misses, and residual bucket charges, not just before/after ordering.
- [x] Change `PersistentNethraIndex.rank_probes()` to ask route/projection probe routes first.
- [x] Make probe ordering use route `probe_region`, route score, and hook-specific use-right gates.
- [x] Preserve record mode: it may collect route-match telemetry but must not change candidate/probe order.
- [x] Preserve assist mode: it may reorder only through allowed ranking/probe/soft-filter routes and must emit attribution.
- [x] Add tests proving compact loaded handles can produce behavior effects when a matching route/projection exists.
- [x] Add tests proving trass/unresolved routes do not narrow active primary hooks.
- [x] Add tests proving unknown hooks and empty route indexes fall back without behavior change.

---

## Sleep / Compaction / Feedback

- [ ] Ingest search traces as route feedback:
  - saved work strengthens route score.
  - wasted work weakens route score.
  - missed useful candidate increments miss count and charges residual.
  - repeated invalidation adds or strengthens invalidators.
  - residual bucket later predicts failure or repair need, then proposes promotion.
- [ ] Convert successful `ExperienceEvent` patterns into route rows instead of generic sleep products where possible.
- [ ] Promote trass residual buckets only through measured outcome evidence.
- [ ] Demote or quarantine routes with rising miss/waste rates.
- [ ] Keep compaction deterministic and bounded before considering learned rankers.
- [ ] Add summary metrics:
  - routes_loaded
  - route_matches
  - route_behavior_effects
  - route_saved_search_count
  - route_wasted_search_count
  - route_miss_count
  - residual_bucket_charges
  - projection_fallbacks
  - atom_fallbacks

---

## Search-Engine Shortcuts Worth Taking

These are established search-engine techniques that fit Dreth's intentions if
implemented as bounded route/projection machinery, not authority.

- [x] Use posting-list intersection for anchors:
  intersect `target_anchor`, `context_prefix`, and `operation_hook/use_right`
  buckets before scoring. This avoids broad atom scans.
- [ ] Use top-k heaps / WAND-style early termination:
  stop scoring once remaining route upper bounds cannot enter the top-k route
  set. Keep deterministic tie-breaks.
- [ ] Use skip pointers or block-max metadata on large route posting lists:
  skip low-ceiling blocks without reading every route.
- [x] Cache materialized route results by `(target_anchor, context_prefix, hook, invalidators)` with explicit invalidation on route updates.
- [ ] Keep small hot-route caches for recent decision points, capped by count and cycle age.
- [ ] Use champion lists per anchor/context/hook:
  pre-store the top few historically useful routes for fast first-pass ranking.
- [ ] Use negative indexes for invalidators:
  remove routes with active invalidators before scoring instead of scoring then rejecting.
- [ ] Use residual buckets as scoped shards:
  organize unexplained rows by anchor/context/hook so promotion does not scan all residuals.
- [ ] Use graph-neighborhood expansion only after an anchor/projection hit:
  bounded one-hop expansion is acceptable; global graph search is not.
- [ ] Use route score upper bounds:
  score from success, saved work, waste, misses, failure, salience, and role state without needing all evidence rows at runtime.

---

## Shortcuts To Avoid For Now

- [ ] Do not introduce vector/embedding retrieval as a primary route lookup.
  It may be useful later as proposal-only recall, but it weakens explicit
  context/hook/invalidator guarantees if used as the active narrowing path.
- [ ] Do not use global PageRank-like graph centrality as authority or route priority.
  Centrality can be diagnostic only.
- [ ] Do not use generic learned ranking until deterministic route attribution shows rule-based route scoring cannot separate useful routes from broad noise.
- [ ] Do not promote residual pressure directly into a narrowing route.
- [ ] Do not add global regimes or enum modes; regimes, if needed later, must emerge as route activation patterns.
- [ ] Do not make expression mining a prerequisite for route use.

---

## Deferred Work

- [ ] Rename public concepts after behavior is fixed:
  `ProjectionIndex` may become `SearchRouteIndex`/`NethraRouteIndex`, and
  "memory" APIs may be wrapped with search-route names for clarity.
- [ ] Offline expression mining over routes and residual buckets.
- [ ] Active-slice compiler from route sets into bounded rank/probe/filter surfaces.
- [ ] Recognition-collapse metrics over route coverage.
- [ ] Passive temporal observers for delayed causality scaffolding, capped and diagnostic-only.
- [ ] Learned ranker/factorizer after deterministic route indexes and route attribution are measured.

---

## Verification Targets

- [x] `python -m pytest tests/learner/test_nethra_projection.py -q`
- [x] `python -m pytest tests/test_nethra_runtime_memory.py -q`
- [x] `python -m pytest tests/learner/test_nethra_mind_store.py -q`
- [x] `python -m pytest tests/learner/test_memory_sleep.py -q`
- [ ] Add route-specific tests before wiring behavior changes into the runtime path.
