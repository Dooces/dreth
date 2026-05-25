from __future__ import annotations

"""Diagnostic-only relative authority records for future NethraGraph work.

This module lays vocabulary and small scoring helpers for graph-mediated,
context-relative nethra authority. It is intentionally not integrated with the
runtime agent. It must not affect skips, cert issuance, revocation, fit,
sentinels, route certs, or policy choice, and it must not be imported by
agent.py.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class NethraNodeRef:
    node_id: str
    kind: str
    var: Optional[int] = None
    context_key: Optional[str] = None


@dataclass(frozen=True)
class NethraRelation:
    source: NethraNodeRef
    target: NethraNodeRef
    relation_type: Literal[
        "shares_node",
        "depends_on",
        "conflicts_with",
        "substitutes_for",
        "coactive_with",
        "beats_in_context",
        "loses_in_context",
        "exception_to",
    ]
    context_key: Optional[str]
    wins: int = 0
    losses: int = 0
    reuse_count: int = 0
    failure_overlap: int = 0
    consequence_weight: float = 1.0
    last_seen_cycle: int = 0


@dataclass(frozen=True)
class RelativeAuthorityRecord:
    node: NethraNodeRef
    context_key: Optional[str]
    wins: int = 0
    losses: int = 0
    failures: int = 0
    reuse_count: int = 0
    downstream_support: int = 0
    consequence_weight: float = 1.0

    @property
    def evidence_total(self) -> int:
        return (
            self.wins
            + self.losses
            + self.failures
            + self.reuse_count
            + self.downstream_support
        )

    def authority_score(self) -> float:
        """Return a diagnostic relative authority score.

        This is not a runtime policy, cert, skip, fit, sentinel, or revocation
        rule. It exists only so tests and future design work can talk about
        graded authority in a small, explicit form.
        """
        return (
            self.wins
            + self.reuse_count
            + self.downstream_support
            - self.losses
            - self.failures * self.consequence_weight
        )

    def should_prefer_over(self, other: "RelativeAuthorityRecord") -> bool:
        return self.authority_score() > other.authority_score()

    def should_localize_failure(
        self,
        global_failure_count_threshold: int = 3,
    ) -> bool:
        return self.failures < global_failure_count_threshold


@dataclass(frozen=True)
class NethraGraphSnapshot:
    nodes: Tuple[NethraNodeRef, ...]
    relations: Tuple[NethraRelation, ...]
    authority_records: Tuple[RelativeAuthorityRecord, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    def top_authority(self, limit: int = 10) -> Tuple[RelativeAuthorityRecord, ...]:
        return tuple(
            sorted(
                self.authority_records,
                key=lambda record: (
                    record.authority_score(),
                    record.evidence_total,
                    record.node.node_id,
                ),
                reverse=True,
            )[:limit]
        )

    def neighboring_nodes(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
    ) -> Tuple[NethraNodeRef, ...]:
        nodes_by_id = {node.node_id: node for node in self.nodes}
        neighbors = []
        seen = set()
        for relation in self.relations:
            if relation_type is not None and relation.relation_type != relation_type:
                continue
            if relation.source.node_id == node_id:
                neighbor_id = relation.target.node_id
            elif relation.target.node_id == node_id:
                neighbor_id = relation.source.node_id
            else:
                continue
            if neighbor_id in seen:
                continue
            neighbor = nodes_by_id.get(neighbor_id)
            if neighbor is not None:
                neighbors.append(neighbor)
                seen.add(neighbor_id)
        return tuple(neighbors)

    def local_competitors(
        self,
        node_id: str,
        context_key: Optional[str] = None,
    ) -> Tuple[NethraNodeRef, ...]:
        competitor_relations = {
            "conflicts_with",
            "substitutes_for",
            "beats_in_context",
            "loses_in_context",
        }
        nodes_by_id = {node.node_id: node for node in self.nodes}
        competitors = []
        seen = set()
        for relation in self.relations:
            if relation.relation_type not in competitor_relations:
                continue
            if context_key is not None and relation.context_key != context_key:
                continue
            if relation.source.node_id == node_id:
                competitor_id = relation.target.node_id
            elif relation.target.node_id == node_id:
                competitor_id = relation.source.node_id
            else:
                continue
            if competitor_id in seen:
                continue
            competitor = nodes_by_id.get(competitor_id)
            if competitor is not None:
                competitors.append(competitor)
                seen.add(competitor_id)
        return tuple(competitors)
