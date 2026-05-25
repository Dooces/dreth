from __future__ import annotations

"""Diagnostic-only relative authority records for future NethraGraph work.

This module lays vocabulary and small scoring helpers for graph-mediated,
context-relative nethra authority. It is intentionally not integrated with the
runtime agent. It must not affect skips, cert issuance, revocation, fit,
sentinels, route certs, or policy choice, and it must not be imported by
agent.py.
"""

from dataclasses import dataclass
from typing import Literal, Optional


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
