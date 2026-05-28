from __future__ import annotations

"""ProjectionIndex: maps (target_var, context_key, operation_hook) to relevant nethra subset.

Replaces the flat salience scan in rank_candidates / rank_probes.  Instead of
scoring every loaded nethra against active atoms, the index returns only those
whose use_right is relevant to the requested hook and whose context prefix
matches the decision point.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nethra_mind_store import NethraMindNode


# Which use_rights are allowed to influence each operation hook.
_HOOK_USE_RIGHTS: dict[str, frozenset[str]] = {
    "parent_candidates": frozenset({"ranking_hint", "soft_filter"}),
    "ranking_hint":      frozenset({"ranking_hint", "soft_filter"}),
    "probe_hint":        frozenset({"probe_hint",   "soft_filter"}),
}

# Hooks that carry primary projection authority.
# Trass and unresolved surfaces must not emit for these hooks.
_PRIMARY_HOOKS: frozenset[str] = frozenset({"parent_candidates", "ranking_hint", "probe_hint"})

# Role states that block primary projection.
_BLOCKED_PRIMARY_ROLE_STATES: frozenset[str] = frozenset({"trass", "unresolved"})


@dataclass
class ProjectionEntry:
    nethra_id: str
    atoms: frozenset[str]
    ctx_prefix: str           # first segment before "|"
    use_rights: frozenset[str]
    salience: float
    behavior_effect_count: int
    failure_count: int
    success_count: int
    role_state: str = ""      # dominant surface role from context_role_index; gates primary projections


class ProjectionIndex:
    """Atom / context / hook index for mind-loaded nethras.

    Kept in sync with NethraMindStore._nodes (via index_node / remove_node)
    and with PersistentNethraIndex records (via index_node_from_row).
    """

    def __init__(self) -> None:
        self._entries: dict[str, ProjectionEntry] = {}
        self._by_atom: dict[str, list[str]] = {}
        self._by_ctx_prefix: dict[str, list[str]] = {}
        self._by_hook: dict[str, list[str]] = {}

    # ── indexing ──────────────────────────────────────────────────────────────

    def index_node(self, node: "NethraMindNode") -> None:
        """Index from a NethraMindNode (called by NethraMindStore)."""
        from .nethra_mind_store import effective_use_right
        use_right = effective_use_right(node.use_rights_seen)
        ctx_prefix = (node.contexts[0] if node.contexts else "").split("|")[0]
        entry = ProjectionEntry(
            nethra_id=node.nethra_id,
            atoms=frozenset(node.touched_atoms),
            ctx_prefix=ctx_prefix,
            use_rights=frozenset(node.use_rights_seen) if node.use_rights_seen else frozenset({use_right}),
            salience=node.salience,
            behavior_effect_count=node.behavior_effect_count,
            failure_count=node.failure_count,
            success_count=node.success_count,
            role_state=_node_dominant_role(getattr(node, "roles_by_context", {})),
        )
        if node.nethra_id in self._entries:
            self._deindex(self._entries[node.nethra_id])
        self._entries[node.nethra_id] = entry
        self._reindex(entry)

    def index_node_from_row(self, row: dict[str, Any]) -> None:
        """Index from a compact mind node dict (called by PersistentNethraIndex)."""
        if str(row.get("entry_kind", "")) != "nethra_mind_node":
            return
        nethra_id = str(row.get("nethra_id", ""))
        if not nethra_id:
            return
        use_rights_seen = [str(r) for r in (row.get("use_rights_seen") or [])]
        if not use_rights_seen:
            urs = str(row.get("use_right_summary", ""))
            if urs:
                use_rights_seen = [urs]
        contexts = list(row.get("contexts") or [])
        ctx_prefix = (str(contexts[0]) if contexts else "").split("|")[0]
        roles_by_context: dict[str, list[str]] = {
            str(k): [str(r) for r in v]
            for k, v in (row.get("roles_by_context") or {}).items()
            if isinstance(v, list)
        }
        role_state_from_row = str(row.get("role_state", ""))
        role_state = role_state_from_row or _node_dominant_role(roles_by_context)
        entry = ProjectionEntry(
            nethra_id=nethra_id,
            atoms=frozenset(str(a) for a in (row.get("touched_atoms") or [])),
            ctx_prefix=ctx_prefix,
            use_rights=frozenset(use_rights_seen),
            salience=float(row.get("salience", 0.0) or 0.0),
            behavior_effect_count=int(row.get("behavior_effect_count", 0) or 0),
            failure_count=int(row.get("failure_count", 0) or 0),
            success_count=int(row.get("success_count", 0) or 0),
            role_state=role_state,
        )
        if nethra_id in self._entries:
            self._deindex(self._entries[nethra_id])
        self._entries[nethra_id] = entry
        self._reindex(entry)

    def remove_node(self, nethra_id: str) -> None:
        entry = self._entries.pop(nethra_id, None)
        if entry is not None:
            self._deindex(entry)

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        target_var: str,
        context_key: str,
        operation_hook: str,
        *,
        top_k: int = 20,
    ) -> list[ProjectionEntry]:
        """Return up to top_k entries relevant to this (var, context, hook) point.

        Scoring:
          +2.0  atom matches target_var
          +1.0  context prefix matches
          +0.5  has behavior evidence
          +0.3  success_count > 0
          -0.5  failure_count > 0
          +0.1  salience > 0 (scaled, cap 1.0)

        Only entries whose use_rights intersect the hook's allowed set are returned.
        Falls back to all entries when the hook is unknown.
        """
        allowed = _HOOK_USE_RIGHTS.get(operation_hook)
        ctx_prefix = context_key.split("|")[0] if context_key else ""

        # Gather candidates via atom index for the target_var atom
        seen: set[str] = set()
        candidates: list[str] = []

        # First: nodes that mention the target_var atom
        for nid in self._by_atom.get(target_var, ()):
            if nid not in seen:
                seen.add(nid)
                candidates.append(nid)

        # Second: nodes that match the context prefix (may add more)
        for nid in self._by_ctx_prefix.get(ctx_prefix, ()):
            if nid not in seen:
                seen.add(nid)
                candidates.append(nid)

        # Third: hook-indexed nodes (may add more)
        for hook_key in _HOOK_USE_RIGHTS:
            if hook_key == operation_hook:
                for nid in self._by_hook.get(hook_key, ()):
                    if nid not in seen:
                        seen.add(nid)
                        candidates.append(nid)

        # If still empty, consider all entries
        if not candidates:
            candidates = list(self._entries.keys())

        gate_primary = operation_hook in _PRIMARY_HOOKS
        scored: list[tuple[float, ProjectionEntry]] = []
        for nid in candidates:
            entry = self._entries.get(nid)
            if entry is None:
                continue
            if allowed is not None and not (entry.use_rights & allowed):
                continue
            if gate_primary and entry.role_state in _BLOCKED_PRIMARY_ROLE_STATES:
                continue
            score = _score_entry(entry, target_var, ctx_prefix)
            scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], x[1].nethra_id))
        return [e for _, e in scored[:top_k]]

    def size(self) -> int:
        return len(self._entries)

    # ── internal ──────────────────────────────────────────────────────────────

    def _reindex(self, entry: ProjectionEntry) -> None:
        nid = entry.nethra_id
        for atom in entry.atoms:
            bucket = self._by_atom.setdefault(atom, [])
            if nid not in bucket:
                bucket.append(nid)
        if entry.ctx_prefix:
            bucket = self._by_ctx_prefix.setdefault(entry.ctx_prefix, [])
            if nid not in bucket:
                bucket.append(nid)
        for hook, allowed in _HOOK_USE_RIGHTS.items():
            if entry.use_rights & allowed:
                bucket = self._by_hook.setdefault(hook, [])
                if nid not in bucket:
                    bucket.append(nid)

    def _deindex(self, entry: ProjectionEntry) -> None:
        nid = entry.nethra_id
        for atom in entry.atoms:
            bucket = self._by_atom.get(atom)
            if bucket and nid in bucket:
                bucket.remove(nid)
        if entry.ctx_prefix:
            bucket = self._by_ctx_prefix.get(entry.ctx_prefix)
            if bucket and nid in bucket:
                bucket.remove(nid)
        for hook in _HOOK_USE_RIGHTS:
            bucket = self._by_hook.get(hook)
            if bucket and nid in bucket:
                bucket.remove(nid)


def _node_dominant_role(roles_by_context: dict[str, list[str]]) -> str:
    """Return the dominant surface role for a node.

    Prefers tareth > best_available > unresolved > trass so that a node that is
    tareth in any context is never blocked by a trass role in another context.
    An empty dict returns ''.
    """
    all_roles = {role for roles in roles_by_context.values() for role in roles}
    for preferred in ("tareth", "best_available", "unresolved", "trass"):
        if preferred in all_roles:
            return preferred
    return ""


def _score_entry(entry: ProjectionEntry, target_var: str, ctx_prefix: str) -> float:
    score = 0.0
    if target_var in entry.atoms:
        score += 2.0
    if ctx_prefix and entry.ctx_prefix == ctx_prefix:
        score += 1.0
    elif ctx_prefix and entry.ctx_prefix and ctx_prefix != entry.ctx_prefix:
        score -= 0.3
    if entry.behavior_effect_count > 0:
        score += 0.5
    if entry.success_count > 0:
        score += 0.3
    if entry.failure_count > 0:
        score -= 0.5
    score += min(1.0, entry.salience * 0.1)
    return score
