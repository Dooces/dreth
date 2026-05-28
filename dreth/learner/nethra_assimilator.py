from __future__ import annotations

"""NethraAssimilator: answers 'what existing nethra explains this row?'

Graph-anchored design. Each incoming row is first parsed into anchors — the
feature dimensions that can connect it to the existing graph — then candidate
nodes are found via local graph neighborhood search, then a disposition is
assigned.

Five dispositions:

  ASSIMILATED    — explained by an existing canonical nethra; fold into that
                   node, raw row is consumed.
  RESIDUAL       — no existing node explains this row; held in bounded residual
                   set pending future pattern emergence.
  CONTRADICTION  — row carries failure evidence against a strongly-successful
                   node; failure/debt updated there, no new node created.
  SPLIT_CANDIDATE — same atom signature, incompatible context root; a new
                    context-specific variant node is allowed to develop.
  NOISE          — no useful signal (no atoms, no structure, no context);
                   discarded silently.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .nethra_mind_store import NethraMindNode


# ── thresholds ────────────────────────────────────────────────────────────────

_ASSIMILATION_THRESHOLD = 0.5   # Jaccard required for full assimilation
_PARTIAL_THRESHOLD = 0.25        # minimum Jaccard for contradiction/split detection
_MIN_SHARED_ATOMS = 1


# ── disposition enum and result ───────────────────────────────────────────────

class Disposition(str, Enum):
    ASSIMILATED = "assimilated"
    RESIDUAL = "residual"
    CONTRADICTION = "contradiction"
    SPLIT_CANDIDATE = "split_candidate"
    NOISE = "noise"


class InternalDisposition(str, Enum):
    """Dreth-native decision layer.  Maps to the public Disposition for compatibility."""
    ABSORB = "absorb"                               # -> ASSIMILATED
    CHARGE_RESIDUAL = "charge_residual"             # -> RESIDUAL (trass/unresolved/best_available target)
    FRACTURE_IDENTITY = "fracture_identity"         # -> SPLIT_CANDIDATE
    SPAWN_RESIDUAL = "spawn_residual"               # -> RESIDUAL (no organizing handle)
    CONTRADICTION_TO_ACCOUNT = "contradiction_to_account"  # -> CONTRADICTION
    NOISE = "noise"                                 # -> NOISE


@dataclass
class AssimilationResult:
    disposition: Disposition
    explained_by: str | None   # nethra_id of the explaining node, or None
    match_score: float         # Jaccard similarity (0.0 if no match)
    evidence_ref: str          # primary identifier from the incoming row
    has_candidates: bool = False  # True when at least one candidate node was found
    internal_disposition: str = ""  # InternalDisposition value for Dreth-native routing


# ── sub-indexes ───────────────────────────────────────────────────────────────

class AnchorIndex:
    """Maps atoms, structure_refs, and member IDs to candidate nethra_ids.

    An anchor is any feature that can connect an incoming row to an existing
    node.  All three dimensions are checked to maximise recall; scoring happens
    in the assimilator after candidates are gathered.
    """

    def __init__(self) -> None:
        self._by_atom: dict[str, list[str]] = {}
        self._by_ref: dict[str, list[str]] = {}
        self._by_member: dict[str, list[str]] = {}

    def index_node(
        self,
        nethra_id: str,
        atoms: list[str],
        structure_refs: list[str],
        member_nethras: list[str],
    ) -> None:
        for atom in atoms:
            bucket = self._by_atom.setdefault(atom, [])
            if nethra_id not in bucket:
                bucket.append(nethra_id)
        for ref in structure_refs:
            bucket = self._by_ref.setdefault(ref, [])
            if nethra_id not in bucket:
                bucket.append(nethra_id)
        for mid in member_nethras:
            bucket = self._by_member.setdefault(mid, [])
            if nethra_id not in bucket:
                bucket.append(nethra_id)

    def remove_node(self, nethra_id: str) -> None:
        for mapping in (self._by_atom, self._by_ref, self._by_member):
            for bucket in mapping.values():
                if nethra_id in bucket:
                    bucket.remove(nethra_id)

    def candidates_for_row(self, row: dict[str, Any]) -> set[str]:
        result: set[str] = set()
        for atom in (row.get("touched_atoms") or []):
            result.update(self._by_atom.get(str(atom), []))
        for ref in (row.get("touched_structure_refs") or []):
            result.update(self._by_ref.get(str(ref), []))
        for mid in (row.get("member_nethras") or []):
            result.update(self._by_member.get(str(mid), []))
        return result


class PerspectiveIndex:
    """Maps (context_prefix, use_right_tier) to candidate nethra_ids.

    A 'perspective' is the context × use_right combination under which a node
    has been active.  Rows with a matching perspective are likely from the same
    decision regime and are strong assimilation candidates.
    """

    def __init__(self) -> None:
        self._by_perspective: dict[tuple[str, str], list[str]] = {}

    def index_node(
        self,
        nethra_id: str,
        contexts: list[str],
        use_rights_seen: list[str],
    ) -> None:
        ctx_prefix = (contexts[0] if contexts else "").split("|")[0]
        for ur in (use_rights_seen or ["record_only"]):
            key = (ctx_prefix, str(ur))
            bucket = self._by_perspective.setdefault(key, [])
            if nethra_id not in bucket:
                bucket.append(nethra_id)

    def remove_node(self, nethra_id: str) -> None:
        for bucket in self._by_perspective.values():
            if nethra_id in bucket:
                bucket.remove(nethra_id)

    def candidates_for_row(self, row: dict[str, Any]) -> set[str]:
        ctx = str(
            row.get("context_scope", "")
            or row.get("proposed_context_scope", "")
            or ""
        )
        ctx_prefix = ctx.split("|")[0] if ctx else ""
        use_right = str(row.get("use_right", "") or row.get("proposed_use_right", "") or "record_only")
        key = (ctx_prefix, use_right)
        return set(self._by_perspective.get(key, []))


class RoleIndex:
    """Maps role names to nodes that have fulfilled those roles.

    Role history reveals what function a node performed in a given context.
    Incoming rows with matching roles are likely to be assimilated into that
    node.
    """

    def __init__(self) -> None:
        self._by_role: dict[str, list[str]] = {}

    def index_node(
        self,
        nethra_id: str,
        roles_by_context: dict[str, list[str]],
    ) -> None:
        for roles in roles_by_context.values():
            for role in roles:
                bucket = self._by_role.setdefault(str(role), [])
                if nethra_id not in bucket:
                    bucket.append(nethra_id)

    def remove_node(self, nethra_id: str) -> None:
        for bucket in self._by_role.values():
            if nethra_id in bucket:
                bucket.remove(nethra_id)

    def candidates_for_row(self, row: dict[str, Any]) -> set[str]:
        result: set[str] = set()
        for rh in (row.get("role_history") or []):
            if isinstance(rh, dict):
                role = str(rh.get("role", ""))
                if role:
                    result.update(self._by_role.get(role, []))
        return result


class TopologyIndex:
    """Maintains the graph's edge structure for neighborhood search.

    When a candidate node is found via one index, its graph neighbors
    (co-member nodes, edge targets) are also evaluated as candidates.
    This lets partial-match rows benefit from structural context.
    """

    def __init__(self) -> None:
        self._neighbors: dict[str, set[str]] = {}

    def index_edge(self, src: str, dst: str) -> None:
        self._neighbors.setdefault(src, set()).add(dst)
        self._neighbors.setdefault(dst, set()).add(src)

    def remove_node(self, nethra_id: str) -> None:
        neighbors = self._neighbors.pop(nethra_id, set())
        for nid in neighbors:
            self._neighbors.get(nid, set()).discard(nethra_id)

    def neighborhood(self, nethra_id: str, depth: int = 1) -> set[str]:
        """Return all nodes reachable within `depth` hops from nethra_id."""
        visited: set[str] = {nethra_id}
        frontier = {nethra_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                next_frontier.update(self._neighbors.get(nid, set()))
            frontier = next_frontier - visited
            visited.update(frontier)
        visited.discard(nethra_id)
        return visited


class ResidualIndex:
    """Bounded index of unexplained rows, organized for fast promotion.

    When a new node is later created whose atoms overlap with residual rows,
    those rows can be promoted (re-explained) rather than staying residual.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._max = max_size
        self._rows: list[dict[str, Any]] = []
        self._by_atom: dict[str, list[int]] = {}  # atom → list of row indices
        self._total_evicted: int = 0

    def add(self, row: dict[str, Any]) -> None:
        idx = len(self._rows)
        self._rows.append(row)
        for atom in (row.get("touched_atoms") or []):
            bucket = self._by_atom.setdefault(str(atom), [])
            bucket.append(idx)
        if len(self._rows) > self._max:
            self._evict()

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def load_rows(self, rows: list[dict[str, Any]]) -> None:
        self._rows = []
        self._by_atom = {}
        for row in rows[: self._max]:
            self.add(row)

    def promote_matching(self, atoms: list[str]) -> list[dict[str, Any]]:
        """Extract and remove rows that share any atom with `atoms`.

        Called when a new node is indexed so matching residuals can be folded.
        """
        atom_set = set(str(a) for a in atoms)
        promote_indices: set[int] = set()
        for atom in atom_set:
            for idx in self._by_atom.get(atom, []):
                if idx < len(self._rows):
                    promote_indices.add(idx)
        if not promote_indices:
            return []
        promoted = [self._rows[i] for i in sorted(promote_indices)]
        # Rebuild without promoted rows
        keep_rows = [r for i, r in enumerate(self._rows) if i not in promote_indices]
        self._rows = []
        self._by_atom = {}
        for row in keep_rows:
            self.add(row)
        return promoted

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def total_evicted(self) -> int:
        return self._total_evicted

    def _evict(self) -> None:
        self._rows.sort(key=lambda r: -(float(r.get("salience", 0.0) or 0.0)))
        surplus = len(self._rows) - self._max
        self._rows = self._rows[: self._max]
        self._total_evicted += surplus
        # Rebuild atom index after eviction (indices shifted)
        self._by_atom = {}
        for idx, row in enumerate(self._rows):
            for atom in (row.get("touched_atoms") or []):
                self._by_atom.setdefault(str(atom), []).append(idx)


# ── assimilator ───────────────────────────────────────────────────────────────

class NethraAssimilator:
    """Graph-anchored assimilator.

    Maintains four sub-indexes (anchor, perspective, role, topology) plus a
    residual index.  For each incoming row, candidates are gathered via all
    four indexes and then expanded by one hop through the topology index.
    The best candidate is scored by Jaccard atom overlap and assigned a
    disposition.

    index_node() and remove_node() must be called by NethraMindStore whenever
    a node is created, updated, or pruned.  index_edge() must be called
    whenever an edge is created.
    """

    def __init__(self) -> None:
        self._anchor = AnchorIndex()
        self._perspective = PerspectiveIndex()
        self._role = RoleIndex()
        self._topology = TopologyIndex()
        self.residuals = ResidualIndex()
        self._stats: dict[str, int] = {d.value: 0 for d in Disposition}
        self._internal_stats: dict[str, int] = {d.value: 0 for d in InternalDisposition}
        self.total_calls: int = 0

    # ── index management ──────────────────────────────────────────────────────

    def index_node(
        self,
        nethra_id: str,
        touched_atoms: list[str],
        *,
        structure_refs: list[str] | None = None,
        member_nethras: list[str] | None = None,
        contexts: list[str] | None = None,
        use_rights_seen: list[str] | None = None,
        roles_by_context: dict[str, list[str]] | None = None,
    ) -> None:
        """Register or refresh a node across all sub-indexes."""
        self._anchor.index_node(
            nethra_id,
            touched_atoms,
            structure_refs or [],
            member_nethras or [],
        )
        self._perspective.index_node(
            nethra_id,
            contexts or [],
            use_rights_seen or [],
        )
        if roles_by_context:
            self._role.index_node(nethra_id, roles_by_context)

    def index_edge(self, src: str, dst: str) -> None:
        """Register an edge for topology-based neighborhood expansion."""
        self._topology.index_edge(src, dst)

    def remove_node(self, nethra_id: str) -> None:
        """Remove a pruned node from all sub-indexes and the topology."""
        self._anchor.remove_node(nethra_id)
        self._perspective.remove_node(nethra_id)
        self._role.remove_node(nethra_id)
        self._topology.remove_node(nethra_id)

    # ── explain ───────────────────────────────────────────────────────────────

    def explain(
        self,
        row: dict[str, Any],
        nodes: dict[str, "NethraMindNode"],
    ) -> AssimilationResult:
        """Determine how this incoming row relates to existing canonical nethras.

        Only called for rows whose ID does not already exist in the store.
        Returns a disposition and, when applicable, the ID of the explaining node.

        Steps:
          1. Parse row into anchors (atoms, refs, members, context, role).
          2. Gather candidates from all four sub-indexes.
          3. Expand candidates by one topology hop.
          4. Score by Jaccard atom overlap; pick best.
          5. Assign disposition based on score, failure signal, context match.
        """
        self.total_calls += 1

        row_atoms = {str(a) for a in (row.get("touched_atoms") or [])}
        row_context = str(
            row.get("context_scope", "")
            or row.get("proposed_context_scope", "")
            or ""
        )
        row_ctx_prefix = row_context.split("|")[0] if row_context else ""
        evidence_ref = str(
            row.get("record_id", "")
            or row.get("proposal_id", "")
            or ""
        )

        # Rows with no atoms and no structure signal are noise
        has_signal = bool(
            row_atoms
            or row.get("touched_structure_refs")
            or row.get("member_nethras")
        )
        if not has_signal:
            return self._emit(Disposition.NOISE, None, 0.0, evidence_ref,
                              internal_disp=InternalDisposition.NOISE)

        # ── Step 2: gather candidates via all four sub-indexes ────────────────
        candidates: set[str] = set()
        candidates.update(self._anchor.candidates_for_row(row))
        candidates.update(self._perspective.candidates_for_row(row))
        candidates.update(self._role.candidates_for_row(row))

        # ── Step 3: one-hop topology expansion ───────────────────────────────
        expanded: set[str] = set()
        for nid in candidates:
            expanded.update(self._topology.neighborhood(nid, depth=1))
        candidates.update(expanded)

        if not candidates:
            # No overlapping structures at all → genuinely new pattern, not residual
            return self._emit(Disposition.RESIDUAL, None, 0.0, evidence_ref, has_candidates=False,
                              internal_disp=InternalDisposition.SPAWN_RESIDUAL)

        # ── Step 4: score by Jaccard atom overlap ────────────────────────────
        best_id: str | None = None
        best_score: float = 0.0

        for nid in candidates:
            node = nodes.get(nid)
            if node is None:
                continue
            node_atoms = set(node.touched_atoms)
            if not row_atoms and not node_atoms:
                continue
            shared = len(row_atoms & node_atoms)
            if shared < _MIN_SHARED_ATOMS:
                continue
            jaccard = shared / max(1, len(row_atoms | node_atoms))
            if jaccard > best_score:
                best_score = jaccard
                best_id = nid

        if best_score < _PARTIAL_THRESHOLD or best_id is None:
            # has_candidates only if there was real atom overlap; score=0 means no atom match
            return self._emit(Disposition.RESIDUAL, None, best_score, evidence_ref,
                              has_candidates=best_score > 0.0,
                              internal_disp=InternalDisposition.SPAWN_RESIDUAL)

        # ── Step 5: assign disposition ────────────────────────────────────────
        node = nodes[best_id]
        node_ctx_full = node.contexts[0] if node.contexts else ""
        node_ctx_prefix = node_ctx_full.split("|")[0]

        # Contradiction: row carries failure against a success-dominant node
        row_failure = (
            bool(row.get("failure_count", 0))
            or bool(row.get("invalidators"))
            or row.get("success") is False
        )
        node_success_dominant = (
            node.success_count > 2
            and node.success_count > node.failure_count * 3
        )
        if row_failure and node_success_dominant:
            return self._emit(Disposition.CONTRADICTION, best_id, best_score, evidence_ref,
                              internal_disp=InternalDisposition.CONTRADICTION_TO_ACCOUNT)

        # Split candidate: incompatible context.
        # Triggers when the hook prefix differs (e.g. source_edge_candidates vs probe_hint),
        # OR when the same hook is used for a different specific target variable
        # (e.g. source_edge_candidates|x7 vs source_edge_candidates|x2).
        # Without the second clause, atom-only Jaccard would silently collapse
        # target-specific contexts that share atoms but serve different variables.
        _ctx_split = (
            (row_ctx_prefix and node_ctx_prefix and row_ctx_prefix != node_ctx_prefix)
            or (
                row_context
                and node_ctx_full
                and row_context != node_ctx_full
                and "|" in row_context
                and "|" in node_ctx_full
            )
        )
        if best_score >= _ASSIMILATION_THRESHOLD and _ctx_split:
            return self._emit(Disposition.SPLIT_CANDIDATE, best_id, best_score, evidence_ref,
                              internal_disp=InternalDisposition.FRACTURE_IDENTITY)

        # Full assimilation
        if best_score >= _ASSIMILATION_THRESHOLD:
            return self._emit(Disposition.ASSIMILATED, best_id, best_score, evidence_ref,
                              internal_disp=InternalDisposition.ABSORB)

        # Partial overlap but insufficient for assimilation.
        # If the best matching node has a residual-collection surface role, this is
        # a CHARGE_RESIDUAL rather than a plain SPAWN_RESIDUAL.
        node = nodes[best_id]
        if _has_residual_surface_role(getattr(node, "roles_by_context", {})):
            return self._emit(Disposition.RESIDUAL, None, best_score, evidence_ref,
                              internal_disp=InternalDisposition.CHARGE_RESIDUAL)
        return self._emit(Disposition.RESIDUAL, None, best_score, evidence_ref,
                          internal_disp=InternalDisposition.SPAWN_RESIDUAL)

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def internal_stats(self) -> dict[str, int]:
        return dict(self._internal_stats)

    def reset_stats(self) -> None:
        self._stats = {d.value: 0 for d in Disposition}
        self._internal_stats = {d.value: 0 for d in InternalDisposition}
        self.total_calls = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(
        self,
        disp: Disposition,
        explained_by: str | None,
        score: float,
        evidence_ref: str,
        *,
        has_candidates: bool = True,
        internal_disp: InternalDisposition | None = None,
    ) -> AssimilationResult:
        self._stats[disp.value] = self._stats.get(disp.value, 0) + 1
        internal_str = internal_disp.value if internal_disp is not None else ""
        if internal_str:
            self._internal_stats[internal_str] = self._internal_stats.get(internal_str, 0) + 1
        return AssimilationResult(
            disposition=disp,
            explained_by=explained_by,
            match_score=score,
            evidence_ref=evidence_ref,
            has_candidates=has_candidates,
            internal_disposition=internal_str,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_RESIDUAL_SURFACE_ROLES: frozenset[str] = frozenset({"trass", "unresolved", "best_available"})


def _has_residual_surface_role(roles_by_context: dict[str, list[str]]) -> bool:
    """Return True if the node has any trass/unresolved/best_available role."""
    for roles in roles_by_context.values():
        for role in roles:
            if role in _RESIDUAL_SURFACE_ROLES:
                return True
    return False
