from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any

from .graph import NethraGraph
from .model import (
    Context,
    ContextPattern,
    Factor,
    Failure,
    Nethra,
    Outcome,
    PredictionCommitment,
    Role,
)


Consequence = Callable[[Any, Any], float]
Factorizer = Callable[[Failure], Iterable[Factor]]


def default_consequence(expected: Any, observed: Any) -> float:
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(expected) - float(observed))
    return 0.0 if expected == observed else 1.0


class Dreth:
    """Prospective authority and failure-earned graph growth."""

    def __init__(
        self,
        graph: NethraGraph | None = None,
        *,
        consequence: Consequence = default_consequence,
        factorizer: Factorizer | None = None,
    ) -> None:
        self.graph = graph or NethraGraph()
        self.consequence = consequence
        self.factorizer = factorizer
        self.pending: dict[int, PredictionCommitment] = {}
        self.failures: list[Failure] = []
        self.outcomes: list[Outcome] = []
        self._next_commitment_id = 1

    def register(
        self,
        *,
        name: str,
        operation: str,
        scope: ContextPattern,
        components: Iterable[str] = (),
        nethra_id: str | None = None,
        parents: Iterable[str] = (),
    ) -> Nethra:
        nethra = Nethra(
            id=nethra_id or self.graph.new_id(),
            name=name,
            operation=operation,
            scope=scope,
            components=frozenset(components),
            parents=set(parents),
        )
        return self.graph.add(nethra)

    def commit(
        self,
        *,
        nethra_id: str,
        context: Context,
        role: Role,
        expected: Any,
        cycle: int,
        horizon: int,
        consequence_threshold: float = 0.0,
        implicated_ids: Iterable[str] = (),
    ) -> PredictionCommitment:
        if horizon < 1:
            raise ValueError("authority requires a prospective horizon")
        nethra = self.graph.nethras[nethra_id]
        if not nethra.scope.matches(context):
            raise ValueError("prediction context is outside the nethra scope")

        implicated = tuple(dict.fromkeys(implicated_ids)) or (nethra_id,)
        for implicated_id in implicated:
            if implicated_id not in self.graph.nethras:
                raise KeyError(implicated_id)

        commitment = PredictionCommitment(
            id=self._next_commitment_id,
            nethra_id=nethra_id,
            implicated_ids=implicated,
            context=context,
            role=role,
            expected=expected,
            committed_at=cycle,
            due_at=cycle + horizon,
            consequence_threshold=consequence_threshold,
        )
        self._next_commitment_id += 1
        self.pending[commitment.id] = commitment
        return commitment

    def observe(self, *, cycle: int, context: Context, observed: Any) -> list[Outcome]:
        due = [
            commitment
            for commitment in self.pending.values()
            if commitment.due_at == cycle and commitment.context == context
        ]
        resolved: list[Outcome] = []
        for commitment in sorted(due, key=lambda item: item.id):
            consequence = self.consequence(commitment.expected, observed)
            if consequence <= commitment.consequence_threshold:
                self.graph.authority(
                    commitment.nethra_id,
                    commitment.role,
                ).record_success(context, commitment.horizon)
                outcome = Outcome(
                    commitment_id=commitment.id,
                    success=True,
                    consequence=consequence,
                )
            else:
                failure = Failure(
                    commitment=commitment,
                    observed=observed,
                    consequence=consequence,
                    signature=self._failure_signature(commitment, observed),
                )
                boundary = self._open_failure(failure)
                self.failures.append(failure)
                outcome = Outcome(
                    commitment_id=commitment.id,
                    success=False,
                    consequence=consequence,
                    boundary_id=boundary.id,
                )
            del self.pending[commitment.id]
            self.outcomes.append(outcome)
            resolved.append(outcome)
        return resolved

    def expire_unexposed(self, *, through_cycle: int) -> list[PredictionCommitment]:
        expired = [
            commitment
            for commitment in self.pending.values()
            if commitment.due_at <= through_cycle
        ]
        for commitment in expired:
            del self.pending[commitment.id]
        return sorted(expired, key=lambda item: item.id)

    def can_reuse(
        self,
        nethra_id: str,
        context: Context,
        horizon: int,
        role: Role | None = None,
    ) -> bool:
        allowed = self.graph.can_use(nethra_id, context, horizon, role)
        if allowed:
            authorities = self.graph.authorities_for(nethra_id)
            for authority in authorities:
                if role is None or authority.role == role:
                    if authority.state_at(context, horizon).value == "usable":
                        authority.uses += 1
        return allowed

    def consolidate(
        self,
        *,
        name: str,
        boundary_ids: Iterable[str],
        components: Iterable[str] = (),
    ) -> Nethra:
        children = tuple(dict.fromkeys(boundary_ids))
        if len(children) < 2:
            raise ValueError("consolidation requires repeated boundaries")
        nodes = [self.graph.nethras[child_id] for child_id in children]
        operation = nodes[0].operation
        if any(node.operation != operation for node in nodes):
            raise ValueError("consolidated boundaries must share an operation")
        required = set(nodes[0].scope.required)
        for node in nodes[1:]:
            required &= set(node.scope.required)
        merged_components = set(components)
        for node in nodes:
            merged_components.update(node.components)
        return self.graph.add(
            Nethra(
                id=self.graph.new_id("higher"),
                name=name,
                operation=operation,
                scope=ContextPattern(operation, tuple(required)),
                components=frozenset(merged_components),
                kind="consolidation",
                parents=set(children),
            )
        )

    def _failure_signature(
        self,
        commitment: PredictionCommitment,
        observed: Any,
    ) -> str:
        raw = repr(
            (
                commitment.implicated_ids,
                commitment.context.key,
                commitment.role,
                commitment.horizon,
                commitment.expected,
                observed,
            )
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _open_failure(self, failure: Failure) -> Nethra:
        existing_id = self.graph.failure_boundaries.get(failure.signature)
        if existing_id is not None:
            return self.graph.nethras[existing_id]

        commitment = failure.commitment
        implicated = commitment.implicated_ids
        context = commitment.context
        parents = [self.graph.nethras[nethra_id] for nethra_id in implicated]

        if len(implicated) > 1:
            boundary = Nethra(
                id=self.graph.new_id("composite"),
                name="relation[" + "+".join(node.name for node in parents) + "]",
                operation=context.operation,
                scope=ContextPattern.from_context(context),
                components=frozenset().union(*(node.components for node in parents)),
                kind="composite",
                parents=set(implicated),
                failure_signature=failure.signature,
            )
            self.graph.add(boundary)
            self.graph.authority(boundary.id, commitment.role).record_failure(
                context,
                commitment.horizon,
                failure.consequence,
                commitment.id,
            )
        else:
            parent = parents[0]
            self.graph.authority(parent.id, commitment.role).record_failure(
                context,
                commitment.horizon,
                failure.consequence,
                commitment.id,
            )
            boundary = Nethra(
                id=self.graph.new_id("boundary"),
                name=f"{parent.name} boundary",
                operation=context.operation,
                scope=ContextPattern.from_context(context),
                components=parent.components,
                kind="boundary",
                parents={parent.id},
                failure_signature=failure.signature,
            )
            self.graph.add(boundary)

        factors = tuple(self.factorizer(failure)) if self.factorizer is not None else ()
        for factor in factors:
            self.graph.add(
                Nethra(
                    id=self.graph.new_id("factor"),
                    name=factor.name,
                    operation=context.operation,
                    scope=ContextPattern.from_context(context),
                    components=factor.components,
                    kind="boundary",
                    parents={boundary.id},
                )
            )
        return boundary
