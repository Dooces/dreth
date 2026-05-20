# DRETH — CONCEPTUAL PSEUDOCODE
# How the system should work. Terse per line.
# Divergences from current code marked: [CURRENT: ...]
# Interfaces not yet built marked: [INTERFACE NEEDED]
#
# Core invariant:
#   notice consequential prediction failure
#   preserve shortcutting where failure is below threshold
#   refine only where error matters
#
# Three real failure modes:
#   detection  — fails without noticing it mattered
#   attribution — notices failure but patches wrong boundary
#   utility    — repair cost exceeds value saved

# =============================================================================
# INITIALIZATION
# =============================================================================

world  = build_world(n_vars, hidden_dag)        # vars with hidden parents + funcs
ledger = {v: VarNethra() for v in vars}         # cert holders, all empty
agent  = Agent(world, ledger)                   # no certs, no shortcuts yet

# =============================================================================
# MAIN CYCLE  (runs forever or until convergence)
# =============================================================================

for cycle in 1..∞:
    topo_order = topological_sort(visible_vars) # parents processed before children
    needs_audit = []

    # -------------------------------------------------------------------------
    # FIRST PASS — cheap paths, in topological order
    # Concurrent across topo-levels: all vars at the same level are independent
    # and can be dispatched in parallel. Each level must complete before the next.
    # -------------------------------------------------------------------------

    for level in topo_levels(topo_order):
        parallel_for var in level:

            n = ledger[var]

            # --- TRASS SHORTCUT ---
            # Filter ledger: cert says "not otherwise excluded from skip."
            # Shortcut fires by default when cert exists. No accounting needed.
            if n.skip_cert.role == "trass":
                record_skip(var, "trass")
                continue

            # --- COMPRESS SHORTCUT ---
            # Cert earned by pred_passes accumulation. Fires when gate matches.
            # No accounting unless gate mismatch revokes the cert.
            if n.compress_cert and gate_matches(var, world.state):
                predict_compressed(var)
                record_skip(var, "compress")
                continue

            # --- SENTINEL PATH ---
            # Sentinels are sparse exclusion monitors, not confirmation tokens.
            # Run cheap probes to detect drift. Cost: O(sentinel_count) per var.
            if n.sentinels:
                passed = run_sentinels(var)         # cheap (iv_slot, iv_val) probes

                if passed:
                    # Shortcut fires. No further accounting.
                    # [CURRENT: witness replay also runs here — wrong branch.
                    #  Witnesses are attribution handles for failure, not pass-path
                    #  confirmation. Placement defect; logic correct if witnesses exist.]
                    record_skip(var, "sentinel")
                    continue

                else:
                    # FAILURE SIGNAL EARNED — open the cert, attribute the failure.
                    # Lazy decomposition: this cost is earned by the failure.
                    case = attribute_sentinel_failure(var)
                    #   [INTERFACE NEEDED: this call should live here.
                    #    Currently witness replay is in the pass branch above.]

                    if case == "authority_expired":
                        # Cert basis is gone; world may still make var tareth.
                        # Recertify — do not collapse to trass. New witness may appear.
                        drop_skip_cert(var)
                        recertify_skip(var, cycle)
                        needs_audit.append(var)
                        continue

                    if case == "world_changed":
                        # World drifted past the cert's tested scope.
                        # Invalidate this var and cascade to descendants.
                        invalidate_cert(var, "sentinel_failure")
                        for child in descendants(var):
                            invalidate_cert(child, "parent_sentinel_failed")
                            needs_audit.append(child)
                        needs_audit.append(var)
                        continue

            needs_audit.append(var)                 # no shortcut — queue full audit

    # -------------------------------------------------------------------------
    # SECOND PASS — full audits, ordered by tractability
    # Sequential within each var; independent vars can run concurrently.
    # -------------------------------------------------------------------------

    for var in tractability_order(needs_audit):
        parents, func, score, second = full_audit(var)
        install(var, parents, func, score, second, cycle)

# =============================================================================
# FULL AUDIT
# Hypothesis search: enumerate, score, rank. Returns best fit.
# =============================================================================

def full_audit(var):
    # Q7 OUGHT TO: include all certified/proposed vars not cert-excluded for route.
    # No instance-level route certs exist → nothing is cert-excluded → include all.
    # [CURRENT: role_for("skip") == "tareth" used as proxy. Utility heuristic,
    #  not cert logic. Excludes skip-trass vars that may be route-relevant.]
    available = [v for v in eligible_vars if not route_cert_excludes(v)]

    hypotheses = enumerate_hypotheses(var, available)   # O(|available|^2) forms
    scores     = [probe_and_score(h, world) for h in hypotheses]
    return best(hypotheses, scores)

# =============================================================================
# INSTALL
# Applies audit result. Certifies, attaches sentinels, discovers compressions.
# =============================================================================

def install(var, parents, func, score, second, cycle):
    update_hypothesis(var, parents, func)               # write to ledger

    if skip_cert_untested(var):
        certify_skip(var, cycle)                        # earn the cert now

    if skip_cert.role == "trass":
        collapse(var)                                   # shortcut earned; done
        return

    # Tareth: keep full attention
    increment_observations(var)

    check_false_trass_contradiction(var, parents)       # if a parent is trass, dispute it

    if no_sentinels(var):
        attach_sentinels(var, available_parents)        # cheap monitors for next cycle

    if enough_observations(var) and no_compressions(var):
        discover_compressions(var)                      # earn compress shortcuts

    if enough_observations(var) and sentinels_attached(var):
        promote_status(var, "certified")                # confidence label only

# =============================================================================
# CERTIFY SKIP
# Perturbation test: does changing this var propagate to others?
# Earns the skip cert — the core filter-ledger authority.
# =============================================================================

def certify_skip(var, cycle):
    targets  = eligible_targets(var)                    # all visible, minus cert-trass
    witnesses = []

    for iv_val in [0.05, 0.25, 0.50, 0.75, 0.95]:     # spread perturbations
        snap = snapshot(world.state)
        if propagates(var, iv_val, targets, world):
            witnesses.append((snap, iv_val))            # attribution handle — not for reuse

    role = "tareth" if enough_propagated(witnesses) else "trass"

    ledger[var].skip_cert = NethraCert(
        operation = "skip",
        role      = role,
        targets   = targets,                            # scope of authority — earned here
        witnesses = witnesses,                          # attribution handles; lazy
    )
    return role

# =============================================================================
# ATTRIBUTE SENTINEL FAILURE
# Distinguishes two failure cases by replaying the cert's witnesses.
# This earns the attribution cost — only called when sentinel already failed.
# =============================================================================

def attribute_sentinel_failure(var):
    cert = ledger[var].skip_cert
    if not cert or not cert.witnesses:
        return "authority_expired"                      # no basis to test

    for snap, iv_val in cert.witnesses:
        if still_propagates(var, snap, iv_val, cert.targets, world):
            return "world_changed"                      # evidence still live; world drifted

    return "authority_expired"                          # cert basis gone; recertify

# =============================================================================
# NETHRA DESCENT — failure localization
# When prediction fails past salience threshold, descend the nethra path.
# LAZY: sub-nethras are not pre-built. Descent earns decomposition at failure.
# [INTERFACE NEEDED: this descent is conceptual; not a live code path yet.
#  Currently handled implicitly by invalidation cascade + recertification.]
# =============================================================================

def handle_prediction_failure(var, failure):
    if failure.cost < salience_threshold(var):
        return                                          # below threshold — leave it alone

    path = find_implicated_nethra_path(var, failure)   # which certs are on the hook

    for nethra in descend(path):
        if not nethra.sub_certs:
            # Leaf reached — failure earns decomposition
            # Propose factors: LLM or rule-based factorizer suggests sub-nethras
            # (e.g. engine → fuel, spark, air, compression, sensor)
            proposed = factorizer.propose(nethra, failure)
            nethra.sub_certs = certify_each(proposed)  # earn sub-cert authority
            break                                       # leave rest untouched

        else:
            # Test candidate sub-nethras to localize failed boundary
            failed_sub = first(s for s in nethra.sub_certs if test_fails(s, failure))

            if failed_sub:
                patch_boundary(failed_sub, failure)     # smallest boundary fix
                break                                   # leave siblings intact

            # No sub-cert explains the failure → factorization is wrong at this level
            # Signal: accumulated local patches that don't converge
            # Response: offline consolidation proposes revised factorization (see below)

# =============================================================================
# COMPRESSION CERT LIFECYCLE
# Shortcut earned by accumulated evidence. Revoked by mismatch.
# =============================================================================

def compression_trial(var, comp, cycle):
    if gate_matches(comp, world.state):
        if abs(world.state[var] - comp.simplified_value) <= tolerance(var):
            comp.pred_passes += 1

            if comp.pred_passes == PROMOTE_AFTER:
                # Cert earned — shortcut now fires by default when gate matches
                ledger[var].compress_cert = NethraCert(
                    operation = "compress",
                    role      = "trass",                # "safe to substitute"
                    targets   = gate_vars(comp),        # scope: gate conditions only
                    witnesses = [(snapshot(world.state), comp.simplified_value)],
                )
        else:
            # Mismatch — failure signal. Cert revoked. Re-earn from zero.
            # Decomposition (why did gate fail?) only earned if failure recurs.
            comp.pred_passes = 0
            ledger[var].compress_cert = None

# =============================================================================
# ON NEW VARIABLE REVEALED
# Scope expansion: trass certs earned over smaller scope may be stale.
# =============================================================================

def on_variable_revealed(new_var, cycle):
    ledger[new_var] = VarNethra()                       # new holder, no certs

    # Re-earn scope for all trass vars — new var may be a dependent
    flipped = retest_trass_vars(cycle)

    # Q6b: any cert that excluded a newly-tareth var has a stale scope
    # [INTERFACE NEEDED: should fire on every cert role transition, not just here.
    #  Currently only fires on variable-reveal path.]
    for v in flipped_to_tareth(flipped):
        for other in all_vars:
            cert = ledger[other].skip_cert
            if cert and v not in cert.targets:
                cert.role = "untested"                  # re-earn against wider scope

# =============================================================================
# RETEST TRASS VARS
# Re-runs certify_skip on all trass vars after scope expansion.
# Also recovers status-only-trass vars (bypassed filter ledger — recovery path).
# =============================================================================

def retest_trass_vars(cycle):
    flipped = []
    for v in visible_vars:
        n = ledger[v]
        is_cert_trass   = n.skip_cert.role == "trass"
        is_status_trass = n.status == "trass" and not n.skip_cert   # bypass case

        if not is_cert_trass and not is_status_trass:
            continue

        drop_skip_cert(v)
        new_role = certify_skip(v, cycle)               # re-earn against current scope

        if new_role in ("tareth", "untested"):
            n.status = "proposed"                       # restore to audit pool
            flipped.append(v)
    return flipped

# =============================================================================
# OFFLINE CONSOLIDATION  (asynchronous — runs between cycles or on low load)
# Discovers abstractions from accumulated local patches.
# Earns higher-level nethras from repeated lower-level evidence.
# =============================================================================

def consolidate():
    # Find local patches that share a deeper pattern
    for pattern in find_shared_exclusion_patterns(local_patch_log):
        if pattern.recurrence_count >= CONSOLIDATE_AFTER:
            # Abstraction earned — promote to higher nethra
            abstract_cert = NethraCert(
                operation = pattern.operation,
                role      = pattern.shared_role,
                targets   = pattern.shared_scope,
                witnesses = pattern.representative_witnesses,
            )
            replace_local_patches(pattern, abstract_cert)   # collapse redundant certs

    # Utility gating: demote shortcuts whose overhead exceeds value
    for cert in all_certs:
        check_cost   = cert.sentinel_cost_per_cycle * expected_cycles_remaining
        failure_cost = cert.historical_failure_rate * audit_cost(cert.var)
        saved_cost   = cert.skip_rate * audit_cost(cert.var)

        if check_cost + failure_cost > saved_cost:
            demote(cert)                                # shortcut costs more than it saves

# =============================================================================
# SUMMARY OF INTERFACES NEEDED (not yet built in agent.py)
# =============================================================================
#
# 1. attribute_sentinel_failure(var)
#    Witness replay on sentinel FAIL, not sentinel pass.
#    Currently witness replay is in the pass branch (placement defect).
#    Interface: call this in the sentinel-fail branch; consume result to choose
#    between recertify (authority_expired) and invalidate-cascade (world_changed).
#
# 2. Role-transition hook for Q6b
#    Fires the scope-stale scan on every skip-cert role change, regardless of path.
#    Currently only fires from on_variable_revealed.
#    Interface: a single chokepoint in ledger.set_cert_role() that triggers Q6b.
#
# 3. handle_prediction_failure / nethra descent
#    Full descent-and-patch protocol for prediction failures past threshold.
#    Currently handled implicitly by invalidation cascade; no explicit descent.
#    Interface: a prediction-failure handler that walks the nethra path and
#    earns decomposition at the failing leaf.
#
# 4. Instance-level route certs
#    available_parents currently uses skip-cert as a proxy for route eligibility.
#    Interface: role_for("route") at instance level; gate available_parents on that.
#    Until then the skip-cert proxy is a utility heuristic, not cert logic.
#
# 5. consolidate() scheduler
#    Offline consolidation has no trigger. Needs a scheduler (between cycles,
#    on low-load, or after N local patches accumulate).
