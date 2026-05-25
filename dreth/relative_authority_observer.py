from __future__ import annotations

"""Post-run diagnostic NethraGraph observer.

This module reads existing ledger/agent artifacts after a run and builds a
sparse relative-authority graph snapshot. It is observational only: it must not
mutate agent or ledger state, and it must not affect skips, fit, sentinels, cert
issuance, revocation, route certs, provider choice, policy selection, or agent
behavior.
"""

from typing import Dict, Iterable, Optional

from .relative_authority import (
    NethraGraphSnapshot,
    NethraNodeRef,
    NethraRelation,
    RelativeAuthorityRecord,
)


def _cert_context_key(cert) -> str:
    parents = ",".join(str(p) for p in getattr(cert, "context_parents", ()) or ())
    visible = getattr(cert, "context_visible", None)
    cycle = getattr(cert, "context_cycle", None)
    op = getattr(cert, "operation", "")
    return f"cert:{op}:parents=({parents}):visible={visible}:cycle={cycle}"


def _fit_context_key(nethra) -> str:
    parents = ",".join(str(p) for p in getattr(nethra, "parents", ()) or ())
    func = getattr(nethra, "func", "")
    return f"fit:parents=({parents}):func={func}"


def _var_node(var: int) -> NethraNodeRef:
    return NethraNodeRef(node_id=f"var:{var}", kind="nethra_var", var=var)


def _add_node(nodes: Dict[str, NethraNodeRef], node: NethraNodeRef) -> NethraNodeRef:
    existing = nodes.get(node.node_id)
    if existing is not None:
        return existing
    nodes[node.node_id] = node
    return node


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_visible_vars(agent) -> Iterable[tuple[int, object]]:
    vars_by_id = getattr(getattr(agent, "ledger", None), "vars", {})
    for var in sorted(vars_by_id):
        yield var, vars_by_id[var]


def build_snapshot_from_agent(agent) -> NethraGraphSnapshot:
    """Build a diagnostic NethraGraphSnapshot from a completed agent run.

    The observer uses only public post-run object fields and never calls agent
    methods that can advance state or mutate ledgers.
    """
    nodes: Dict[str, NethraNodeRef] = {}
    relations = []
    authority_records = []

    visible_items = list(_iter_visible_vars(agent))
    var_nodes: Dict[int, NethraNodeRef] = {}
    downstream_support: Dict[int, int] = {var: 0 for var, _ in visible_items}
    for _, nethra in visible_items:
        for parent in getattr(nethra, "parents", ()) or ():
            if parent in downstream_support:
                downstream_support[parent] += 1

    for var, nethra in visible_items:
        node = _add_node(nodes, _var_node(var))
        var_nodes[var] = node

    for var, nethra in visible_items:
        node = var_nodes[var]
        context_key = _fit_context_key(nethra)
        parents = getattr(nethra, "parents", ()) or ()
        for parent in parents:
            parent_node = var_nodes.get(parent)
            if parent_node is None:
                continue
            relations.append(
                NethraRelation(
                    source=node,
                    target=parent_node,
                    relation_type="depends_on",
                    context_key=context_key,
                    reuse_count=_safe_int(getattr(nethra, "skip_count", 0)),
                    consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                    last_seen_cycle=_safe_int(getattr(nethra, "last_changed_cycle", 0)),
                )
            )

        certs = getattr(nethra, "certificates", {}) or {}
        route_certs = getattr(nethra, "route_certs", {}) or {}
        failure_count = (
            _safe_int(getattr(nethra, "consecutive_sentinel_failures", 0))
            + _safe_int(getattr(nethra, "unique_failures_caught", 0))
        )
        cert_wins = sum(_safe_int(getattr(cert, "changes", 0)) for cert in certs.values())
        authority_records.append(
            RelativeAuthorityRecord(
                node=node,
                context_key=context_key,
                wins=cert_wins,
                losses=sum(1 for cert in certs.values() if getattr(cert, "revoked_by", None)),
                failures=failure_count,
                reuse_count=(
                    _safe_int(getattr(nethra, "skip_count", 0))
                    + _safe_int(getattr(nethra, "compression_hits_lifetime", 0))
                ),
                downstream_support=downstream_support.get(var, 0),
                consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
            )
        )

        for operation, cert in certs.items():
            cert_node = _add_node(
                nodes,
                NethraNodeRef(
                    node_id=f"cert:{var}:{operation}",
                    kind="certificate",
                    var=var,
                    context_key=_cert_context_key(cert),
                ),
            )
            relations.append(
                NethraRelation(
                    source=node,
                    target=cert_node,
                    relation_type="coactive_with",
                    context_key=cert_node.context_key,
                    wins=_safe_int(getattr(cert, "changes", 0)),
                    losses=1 if getattr(cert, "revoked_by", None) else 0,
                    reuse_count=_safe_int(getattr(cert, "sentinel_passes", 0)),
                    consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                    last_seen_cycle=_safe_int(getattr(cert, "context_cycle", 0)),
                )
            )
            if getattr(cert, "revoked_by", None):
                relations.append(
                    NethraRelation(
                        source=cert_node,
                        target=node,
                        relation_type="exception_to",
                        context_key=cert_node.context_key,
                        losses=1,
                        failure_overlap=1,
                        consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                        last_seen_cycle=_safe_int(getattr(cert, "context_cycle", 0)),
                    )
                )
            authority_records.append(
                RelativeAuthorityRecord(
                    node=cert_node,
                    context_key=cert_node.context_key,
                    wins=_safe_int(getattr(cert, "changes", 0)),
                    losses=1 if getattr(cert, "revoked_by", None) else 0,
                    failures=1 if getattr(cert, "revoked_by", None) else 0,
                    reuse_count=_safe_int(getattr(cert, "sentinel_passes", 0)),
                    downstream_support=len(getattr(cert, "targets", ()) or ()),
                    consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                )
            )

        for candidate_var, cert in route_certs.items():
            cert_node = _add_node(
                nodes,
                NethraNodeRef(
                    node_id=f"route_cert:{var}:{candidate_var}",
                    kind="route_certificate",
                    var=var,
                    context_key=_cert_context_key(cert),
                ),
            )
            candidate_node = var_nodes.get(candidate_var)
            if candidate_node is not None:
                role = getattr(cert, "role", None)
                relation_type = "conflicts_with" if role == "trass" else "depends_on"
                relations.append(
                    NethraRelation(
                        source=node,
                        target=candidate_node,
                        relation_type=relation_type,
                        context_key=cert_node.context_key,
                        wins=_safe_int(getattr(cert, "changes", 0)),
                        losses=1 if getattr(cert, "revoked_by", None) else 0,
                        consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                        last_seen_cycle=_safe_int(getattr(cert, "context_cycle", 0)),
                    )
                )
            relations.append(
                NethraRelation(
                    source=node,
                    target=cert_node,
                    relation_type="coactive_with",
                    context_key=cert_node.context_key,
                    wins=_safe_int(getattr(cert, "changes", 0)),
                    losses=1 if getattr(cert, "revoked_by", None) else 0,
                    consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                    last_seen_cycle=_safe_int(getattr(cert, "context_cycle", 0)),
                )
            )
            authority_records.append(
                RelativeAuthorityRecord(
                    node=cert_node,
                    context_key=cert_node.context_key,
                    wins=_safe_int(getattr(cert, "changes", 0)),
                    losses=1 if getattr(cert, "revoked_by", None) else 0,
                    failures=1 if getattr(cert, "revoked_by", None) else 0,
                    downstream_support=len(getattr(cert, "targets", ()) or ()),
                    consequence_weight=float(getattr(nethra, "cost_weight", 1.0)),
                )
            )

        for index, alt in enumerate(getattr(nethra, "dormant_alternatives", []) or []):
            alt_context_values = sorted(str(x) for x in getattr(alt, "context_keys_seen", set()) or set())
            alt_context = f"dormant:var={var}:contexts=({','.join(alt_context_values)})"
            alt_node = _add_node(
                nodes,
                NethraNodeRef(
                    node_id=(
                        f"dormant:{var}:{index}:"
                        f"{getattr(alt, 'parents', ())}:{getattr(alt, 'func', '')}"
                    ),
                    kind="dormant_alternative",
                    var=var,
                    context_key=alt_context,
                ),
            )
            relations.append(
                NethraRelation(
                    source=alt_node,
                    target=node,
                    relation_type="substitutes_for",
                    context_key=alt_context,
                    wins=_safe_int(getattr(alt, "revival_count", 0)),
                    losses=1,
                    reuse_count=len(alt_context_values),
                    last_seen_cycle=_safe_int(getattr(alt, "last_seen_cycle", 0)),
                )
            )
            authority_records.append(
                RelativeAuthorityRecord(
                    node=alt_node,
                    context_key=alt_context,
                    wins=_safe_int(getattr(alt, "revival_count", 0)),
                    losses=1,
                    reuse_count=len(alt_context_values),
                )
            )

    ledger = getattr(agent, "ledger", None)
    for index, composite in enumerate(getattr(ledger, "composites", []) or ()):
        composite_node = _add_node(
            nodes,
            NethraNodeRef(
                node_id=f"composite:{index}:{getattr(composite, 'members', ())}",
                kind="composite",
                context_key=f"composite:visible={getattr(composite, 'context_visible', None)}",
            ),
        )
        members = getattr(composite, "members", ()) or ()
        for member in members:
            member_node = var_nodes.get(member)
            if member_node is None:
                continue
            relations.append(
                NethraRelation(
                    source=composite_node,
                    target=member_node,
                    relation_type="shares_node",
                    context_key=composite_node.context_key,
                    wins=_safe_int(getattr(composite, "changes", 0)),
                    reuse_count=_safe_int(getattr(composite, "pass_count", 0)),
                    last_seen_cycle=_safe_int(getattr(composite, "certified_at_cycle", 0)),
                )
            )
        authority_records.append(
            RelativeAuthorityRecord(
                node=composite_node,
                context_key=composite_node.context_key,
                wins=_safe_int(getattr(composite, "changes", 0)),
                reuse_count=_safe_int(getattr(composite, "pass_count", 0)),
                downstream_support=len(members),
            )
        )

    for index, hyper in enumerate(getattr(ledger, "hyper_composites", []) or ()):
        hyper_node = _add_node(
            nodes,
            NethraNodeRef(
                node_id=f"hyper_composite:{getattr(hyper, 'component_id', index)}",
                kind="hyper_composite",
                context_key=f"hyper:visible={getattr(hyper, 'context_visible', None)}",
            ),
        )
        members = getattr(hyper, "members", ()) or ()
        for member in members:
            member_node = var_nodes.get(member)
            if member_node is None:
                continue
            relations.append(
                NethraRelation(
                    source=hyper_node,
                    target=member_node,
                    relation_type="shares_node",
                    context_key=hyper_node.context_key,
                    reuse_count=_safe_int(getattr(hyper, "pass_count", 0)),
                    last_seen_cycle=_safe_int(getattr(hyper, "certified_at_cycle", 0)),
                )
            )
        authority_records.append(
            RelativeAuthorityRecord(
                node=hyper_node,
                context_key=hyper_node.context_key,
                reuse_count=_safe_int(getattr(hyper, "pass_count", 0)),
                downstream_support=len(members),
            )
        )

    return NethraGraphSnapshot(
        nodes=tuple(nodes.values()),
        relations=tuple(relations),
        authority_records=tuple(authority_records),
    )
