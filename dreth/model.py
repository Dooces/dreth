from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Iterable, Literal


ContextKey = tuple[str, tuple[tuple[str, Hashable], ...]]
Role = Literal["tareth", "trass"]
NethraKind = Literal["handle", "boundary", "composite", "consolidation"]


class UseState(str, Enum):
    USABLE = "usable"
    UNUSABLE = "unusable"


def _freeze_facts(facts: Iterable[tuple[str, Hashable]]) -> tuple[tuple[str, Hashable], ...]:
    frozen = tuple(sorted(facts))
    for _, value in frozen:
        hash(value)
    return frozen


@dataclass(frozen=True)
class Context:
    operation: str
    facts: tuple[tuple[str, Hashable], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", _freeze_facts(self.facts))

    @classmethod
    def make(cls, operation: str, **facts: Hashable) -> Context:
        return cls(operation=operation, facts=tuple(facts.items()))

    @property
    def key(self) -> ContextKey:
        return self.operation, self.facts

    def as_dict(self) -> dict[str, Hashable]:
        return dict(self.facts)


@dataclass(frozen=True)
class ContextPattern:
    operation: str
    required: tuple[tuple[str, Hashable], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", _freeze_facts(self.required))

    @classmethod
    def from_context(cls, context: Context) -> ContextPattern:
        return cls(operation=context.operation, required=context.facts)

    @classmethod
    def make(cls, operation: str, **required: Hashable) -> ContextPattern:
        return cls(operation=operation, required=tuple(required.items()))

    def matches(self, context: Context) -> bool:
        return self.operation == context.operation and set(self.required) <= set(context.facts)


@dataclass
class Nethra:
    id: str
    name: str
    operation: str
    scope: ContextPattern
    components: frozenset[str] = frozenset()
    kind: NethraKind = "handle"
    parents: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)
    failure_signature: str | None = None

    def __post_init__(self) -> None:
        if self.scope.operation != self.operation:
            raise ValueError("nethra operation and scope operation must match")


@dataclass(frozen=True)
class LocalFailure:
    context: Context
    horizon: int
    consequence: float
    commitment_id: int


@dataclass
class Authority:
    nethra_id: str
    role: Role
    successes: dict[ContextKey, Counter[int]] = field(default_factory=dict)
    failures: dict[ContextKey, dict[int, LocalFailure]] = field(default_factory=dict)
    uses: int = 0

    def record_success(self, context: Context, horizon: int) -> None:
        self.successes.setdefault(context.key, Counter())[horizon] += 1

    def record_failure(
        self,
        context: Context,
        horizon: int,
        consequence: float,
        commitment_id: int,
    ) -> None:
        self.failures.setdefault(context.key, {})[horizon] = LocalFailure(
            context=context,
            horizon=horizon,
            consequence=consequence,
            commitment_id=commitment_id,
        )

    def proven_horizon(self, context: Context) -> int:
        horizons = self.successes.get(context.key, Counter())
        return max(horizons, default=0)

    def usable_horizon(self, context: Context) -> int:
        proven = self.proven_horizon(context)
        failed = self.failures.get(context.key, {})
        if failed:
            proven = min(proven, min(failed) - 1)
        return max(0, proven)

    def state_at(self, context: Context, horizon: int) -> UseState:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self.usable_horizon(context) >= horizon:
            return UseState.USABLE
        return UseState.UNUSABLE

    def success_count(self, context: Context | None = None) -> int:
        if context is not None:
            return sum(self.successes.get(context.key, {}).values())
        return sum(sum(counts.values()) for counts in self.successes.values())

    @property
    def context_span(self) -> int:
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        return sum(len(failures) for failures in self.failures.values())

    def trust_key(self, context: Context, horizon: int) -> tuple[int, int, int, int, int]:
        usable = int(self.state_at(context, horizon) is UseState.USABLE)
        return (
            usable,
            self.usable_horizon(context),
            self.success_count(context),
            self.context_span,
            -self.failure_count,
        )


@dataclass(frozen=True)
class PredictionCommitment:
    id: int
    nethra_id: str
    implicated_ids: tuple[str, ...]
    context: Context
    role: Role
    expected: Any
    committed_at: int
    due_at: int
    consequence_threshold: float

    @property
    def horizon(self) -> int:
        return self.due_at - self.committed_at


@dataclass(frozen=True)
class Failure:
    commitment: PredictionCommitment
    observed: Any
    consequence: float
    signature: str


@dataclass(frozen=True)
class Factor:
    name: str
    components: frozenset[str]


@dataclass(frozen=True)
class Outcome:
    commitment_id: int
    success: bool
    consequence: float
    boundary_id: str | None = None


@dataclass(frozen=True)
class Consideration:
    nethra_id: str
    role: Role | None
    state: UseState
    match: Literal["local", "shared"]
    trust: tuple[int, int, int, int, int]
