from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .model import (
    Authority,
    Consideration,
    Context,
    Nethra,
    Role,
    UseState,
)


class NethraGraph:
    """The live ledger: handles, shared components, relations, and earned authority."""

    def __init__(self) -> None:
        self.nethras: dict[str, Nethra] = {}
        self.authorities: dict[tuple[str, Role], Authority] = {}
        self.component_index: dict[str, set[str]] = defaultdict(set)
        self.failure_boundaries: dict[str, str] = {}
        self._next_id = 1

    def new_id(self, prefix: str = "n") -> str:
        while True:
            candidate = f"{prefix}{self._next_id}"
            self._next_id += 1
            if candidate not in self.nethras:
                return candidate

    def add(self, nethra: Nethra) -> Nethra:
        if nethra.id in self.nethras:
            raise ValueError(f"duplicate nethra id: {nethra.id}")
        for parent_id in nethra.parents:
            if parent_id not in self.nethras:
                raise KeyError(parent_id)
        self.nethras[nethra.id] = nethra
        for component in nethra.components:
            self.component_index[component].add(nethra.id)
        for parent_id in nethra.parents:
            self.nethras[parent_id].children.add(nethra.id)
        if nethra.failure_signature:
            self.failure_boundaries[nethra.failure_signature] = nethra.id
        return nethra

    def link(self, left_id: str, right_id: str) -> None:
        left = self.nethras[left_id]
        right = self.nethras[right_id]
        left.children.add(right_id)
        right.parents.add(left_id)

    def authority(self, nethra_id: str, role: Role) -> Authority:
        if nethra_id not in self.nethras:
            raise KeyError(nethra_id)
        key = nethra_id, role
        if key not in self.authorities:
            self.authorities[key] = Authority(nethra_id=nethra_id, role=role)
        return self.authorities[key]

    def authorities_for(self, nethra_id: str) -> list[Authority]:
        return [
            authority
            for (candidate_id, _), authority in self.authorities.items()
            if candidate_id == nethra_id
        ]

    def shared_neighbors(self, nethra_id: str) -> set[str]:
        node = self.nethras[nethra_id]
        neighbors: set[str] = set(node.parents) | set(node.children)
        for component in node.components:
            neighbors.update(self.component_index[component])
        neighbors.discard(nethra_id)
        return neighbors

    def consider(
        self,
        context: Context,
        horizon: int,
        components: Iterable[str] = (),
    ) -> list[Consideration]:
        component_set = set(components)
        local_ids = {
            nethra_id
            for nethra_id, nethra in self.nethras.items()
            if nethra.scope.matches(context)
        }
        shared_ids: set[str] = set()
        for component in component_set:
            shared_ids.update(self.component_index.get(component, ()))
        for nethra_id in local_ids:
            shared_ids.update(self.shared_neighbors(nethra_id))
        shared_ids -= local_ids

        rows: list[Consideration] = []
        for match, ids in (("local", local_ids), ("shared", shared_ids)):
            for nethra_id in ids:
                authorities = self.authorities_for(nethra_id)
                if authorities:
                    best = max(
                        authorities,
                        key=lambda authority: authority.trust_key(context, horizon),
                    )
                    trust = best.trust_key(context, horizon)
                    role: Role | None = best.role
                    state = best.state_at(context, horizon)
                else:
                    trust = (0, 0, 0, 0, 0)
                    role = None
                    state = UseState.UNUSABLE
                rows.append(
                    Consideration(
                        nethra_id=nethra_id,
                        role=role,
                        state=state,
                        match=match,
                        trust=trust,
                    )
                )

        rows.sort(
            key=lambda row: (
                row.state is UseState.USABLE,
                row.match == "local",
                row.trust,
                row.nethra_id,
            ),
            reverse=True,
        )
        return rows

    def can_use(
        self,
        nethra_id: str,
        context: Context,
        horizon: int,
        role: Role | None = None,
    ) -> bool:
        authorities = self.authorities_for(nethra_id)
        if role is not None:
            authorities = [authority for authority in authorities if authority.role == role]
        return any(
            authority.state_at(context, horizon) is UseState.USABLE
            for authority in authorities
        )

    def can_collapse(self, nethra_id: str, context: Context, horizon: int) -> bool:
        return self.can_use(nethra_id, context, horizon, role="trass")
