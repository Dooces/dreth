from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# RepairAgenda: structural representation of pending repair work.
#
# RepairAgendaItem carries scope/authority metadata for each needs_audit
# entry, enabling A*-style repair ordering in later stages.
#
# Stage 1 is structural only: items mirror current needs_audit ordering
# (topological-then-consequence-tier). A*-style cost/benefit ordering and
# scope-driven triage are reserved for later stages.
#
# INVARIANT:
#   RepairAgenda is a planning surface only.
#   Items describe what needs repair; they do NOT authorize repair.
#   Cert issuance still goes through ledger authority transaction methods.
#   No item in the agenda may directly create NethraCertificate objects.
# ─────────────────────────────────────────────────────────────────────────────

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class RepairAgendaItem:
    """One entry in the repair agenda — describes what needs to be repaired
    and what authority is at risk if it is not. Does NOT grant repair authority.
    """
    cycle: int
    target_var: int
    failure_kind: str            # "sentinel_failure" | "source_edge_change" | "func_change" | "unknown"
    source: str                  # "needs_audit" | "cascade" | "manual"
    covering_regime_id: Optional[str]    # regime covering this var, if any
    covering_composite_id: Optional[int] # composite index covering this var, if any
    scope_vars: Tuple[int, ...]          # this var + downstream dependents at risk
    authority_at_risk: int               # count of certs that depend on this repair succeeding
    estimated_probe_cost: int            # heuristic probe count (intervention_budget)
    priority: float                      # lower value = higher priority (min-heap ordering)
    payload: Dict[str, Any]             # extra diagnostic data (cert_age, consecutive_fails, etc.)


class RepairAgenda:
    """A priority-ordered collection of pending repair items.

    Stage 1: insertion order preserved; pop returns the min-priority item
    via linear scan. A*-style heapq ordering is a future-stage upgrade.

    Lifecycle: cleared at the start of each cycle's audit pass (items represent
    work for the current cycle only).
    """

    def __init__(self) -> None:
        self._items: List[RepairAgendaItem] = []
        self._total_pushed: int = 0
        self._total_popped: int = 0
        # Cumulative scope statistics across all pushes (not just pending items).
        self._cumulative_scope_sum: int = 0
        self._cumulative_scope_max: int = 0

    def push(self, item: RepairAgendaItem) -> None:
        """Add an item. Stores in push order; pop extracts the min-priority item.

        #SHORTCUT: O(1) insert, O(n) pop — adequate for Stage 1 item counts
        (priority_audit_budget is small). Replace with heapq when A*-ordering
        is active and agenda sizes grow.
        """
        self._items.append(item)
        self._total_pushed += 1
        sc = len(item.scope_vars)
        self._cumulative_scope_sum += sc
        if sc > self._cumulative_scope_max:
            self._cumulative_scope_max = sc

    def pop(self) -> Optional[RepairAgendaItem]:
        """Remove and return the item with the lowest priority value (highest urgency).
        Returns None if empty.
        """
        if not self._items:
            return None
        # #SHORTCUT: O(n) find-min; mirrors current needs_audit queue size.
        best_idx = min(range(len(self._items)), key=lambda i: self._items[i].priority)
        item = self._items.pop(best_idx)
        self._total_popped += 1
        return item

    def clear(self) -> None:
        """Remove all pending items. Called at cycle boundary."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def summary(self) -> Dict:
        """Diagnostic summary: cumulative statistics across all pushes,
        not just currently pending items.
        """
        scope_mean = (
            self._cumulative_scope_sum / self._total_pushed
            if self._total_pushed > 0 else 0.0
        )
        return {
            "pending": len(self._items),
            "total_pushed": self._total_pushed,
            "total_popped": self._total_popped,
            "scope_mean": scope_mean,
            "scope_max": self._cumulative_scope_max,
        }
