from __future__ import annotations

"""ScaffoldMemoryIndex: loads offline sleep proposals.

Hard invariants:
  - authority_allowed=False on every loaded proposal
  - No authority issuance, revocation, skip suppression, or behavior effects
  - No hidden truth/debug manifest reads or use
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HIDDEN_TRUTH_LIKE_FIELDS: frozenset[str] = frozenset({
    "truth_parents",
    "truth_func",
    "truth_delayed_parents",
    "truth_latents",
    "debug_blind_challenge_manifest",
})

# An authority_debt_family proposal is broad_generic_debt when it has no
# local structural anchors (no contexts, signatures, or parent sets).
# Such proposals group vars only by shared auth_state+reason, which is too
# coarse to be useful as a runtime ranking or clustering anchor.
_BROAD_DEBT_LOCAL_ANCHOR_REQUIRED: tuple[str, ...] = (
    "contexts",
    "common_signatures",
    "common_parents",
)


# ── Proposal dataclass ─────────────────────────────────────────────────────────

@dataclass
class ScaffoldMemoryProposal:
    """A scaffold proposal loaded from offline sleep output.

    Fields mirror ScaffoldProposal.to_dict() from memory_sleep.py.
    authority_allowed is always forced to False on load regardless of
    what the file contains.
    """
    proposal_id: str
    kind: str
    source_record_ids: list[str]
    source_kinds: list[str]
    vars: list[int]
    contexts: list[str]
    common_signatures: list[str]
    common_parents: list[list[int]]
    role_patterns: list[str]
    recurring_signals: list[str]
    confidence_as_familiarity: float
    authority_allowed: bool = False
    suggested_runtime_use: str = "no_runtime_use"
    warnings: list[str] = field(default_factory=list)
    broad_generic_debt: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScaffoldMemoryProposal":
        if d.get("entry_kind") == "sleep_product" or d.get("record_type") == "sleep_product":
            use = str(d.get("proposed_use_right", "feature_only"))
            warnings = [str(w) for w in (d.get("invalidators") or [])]
            if use == "hard_filter":
                use = "record_only"
                warnings.append("sleep_hard_filter_rejected")
            return cls(
                proposal_id=str(d.get("proposal_id", "")),
                kind="sleep_product",
                source_record_ids=[str(s) for s in (d.get("member_nethras") or [])],
                source_kinds=["sleep_product"],
                vars=[
                    int(str(a)[1:])
                    for a in (d.get("touched_atoms") or [])
                    if str(a).startswith("x") and str(a)[1:].isdigit()
                ],
                contexts=[str(d.get("proposed_context_scope", ""))] if d.get("proposed_context_scope") else [],
                common_signatures=[
                    str(r) for r in (d.get("touched_structure_refs") or [])
                    if not str(r).startswith("parents:")
                ],
                common_parents=[
                    [int(p) for p in str(r).split(":", 1)[1].split(",") if p.strip().isdigit()]
                    for r in (d.get("touched_structure_refs") or [])
                    if str(r).startswith("parents:")
                ],
                role_patterns=[],
                recurring_signals=[str(d.get("reason", ""))] if d.get("reason") else [],
                confidence_as_familiarity=float(d.get("salience_delta", 0.0) or 0.0),
                authority_allowed=False,
                suggested_runtime_use=use,
                warnings=warnings,
                broad_generic_debt=False,
            )
        vars_list = [int(v) for v in (d.get("vars") or [])]
        contexts = [str(c) for c in (d.get("contexts") or [])]
        signatures = [str(s) for s in (d.get("common_signatures") or [])]
        parents = [
            [int(p) for p in row]
            for row in (d.get("common_parents") or [])
            if row
        ]
        kind = str(d.get("kind", "unknown"))

        broad_generic_debt = (
            kind == "authority_debt_family"
            and not contexts
            and not signatures
            and not parents
        )

        return cls(
            proposal_id=str(d.get("proposal_id", "")),
            kind=kind,
            source_record_ids=[str(s) for s in (d.get("source_record_ids") or [])],
            source_kinds=[str(s) for s in (d.get("source_kinds") or [])],
            vars=vars_list,
            contexts=contexts,
            common_signatures=signatures,
            common_parents=parents,
            role_patterns=[str(r) for r in (d.get("role_patterns") or [])],
            recurring_signals=[str(s) for s in (d.get("recurring_signals") or [])],
            confidence_as_familiarity=float(d.get("confidence_as_familiarity", 0.0)),
            authority_allowed=False,  # invariant: always False regardless of file content
            suggested_runtime_use=str(d.get("suggested_runtime_use", "no_runtime_use")),
            warnings=[str(w) for w in (d.get("warnings") or [])],
            broad_generic_debt=broad_generic_debt,
        )


# ── ScaffoldMemoryIndex ────────────────────────────────────────────────────────

class ScaffoldMemoryIndex:
    """Load offline scaffold proposals. Enforces authority_allowed=False and blocks hard_filter."""

    def __init__(self) -> None:
        self._proposals: list[ScaffoldMemoryProposal] = []
        self._by_kind: Counter[str] = Counter()
        self._broad_generic_debt_proposals: int = 0
        self._ranking_applications: int = 0
        self._candidates_reordered: int = 0
        self._top1_supported: int = 0
        self._topk_supported: int = 0
        self._broad_generic_noops: int = 0
        self._no_runtime_hook_available: int = 0
        self._feature_examples: list[dict[str, Any]] = []

    # ── Loading ────────────────────────────────────────────────────────────────

    @property
    def loaded_proposals_count(self) -> int:
        """Number of proposals currently loaded."""
        return len(self._proposals)

    def load_proposals(self, path: str | Path) -> int:
        """Load proposals from a JSONL file (memory_sleep output).

        Rows containing HIDDEN_TRUTH_LIKE_FIELDS are skipped (not read).
        authority_allowed is forced to False on every loaded proposal.
        Returns the number of proposals loaded.
        """
        self._proposals.clear()
        self._by_kind.clear()
        self._broad_generic_debt_proposals = 0
        self.reset_runtime_metrics()

        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(d, dict):
                    continue
                if any(k in HIDDEN_TRUTH_LIKE_FIELDS for k in d):
                    continue
                p = ScaffoldMemoryProposal.from_dict(d)
                if not p.proposal_id:
                    continue
                self._proposals.append(p)
                self._by_kind[p.kind] += 1
                if p.broad_generic_debt:
                    self._broad_generic_debt_proposals += 1

        return len(self._proposals)

    def reset_runtime_metrics(self) -> None:
        self._ranking_applications = 0
        self._candidates_reordered = 0
        self._top1_supported = 0
        self._topk_supported = 0
        self._broad_generic_noops = 0
        self._no_runtime_hook_available = 0
        self._feature_examples = []

    def summarize_matches(self) -> dict[str, Any]:
        """Return index-level summary (loaded proposals, by kind, broad_generic_debt count)."""
        return {
            "scaffold_memory_loaded_proposals": len(self._proposals),
            "scaffold_memory_proposals_by_kind": dict(self._by_kind),
            "scaffold_memory_broad_generic_debt_proposals": self._broad_generic_debt_proposals,
            **self.runtime_metrics(),
            "scaffold_memory_authority_allowed_count": 0,
            "scaffold_memory_behavior_effects": 0,
        }

    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "scaffold_memory_ranking_applications": self._ranking_applications,
            "scaffold_memory_candidates_reordered": self._candidates_reordered,
            "scaffold_memory_top1_supported": self._top1_supported,
            "scaffold_memory_topk_supported": self._topk_supported,
            "scaffold_memory_broad_generic_noops": self._broad_generic_noops,
            "scaffold_memory_no_runtime_hook_available": self._no_runtime_hook_available,
            "scaffold_memory_feature_examples": list(self._feature_examples),
        }


def compute_run_scaffold_metrics(
    index: ScaffoldMemoryIndex,
    background_nethra_export: dict[str, Any],
    context_role_export: dict[str, Any],
    authority_strength_export: dict[str, Any],
    max_examples: int = 5,
    runtime_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return scaffold memory metrics for a completed run. Matching stripped — returns zeros."""
    base = index.summarize_matches() if index is not None else {}
    return {
        **base,
        "scaffold_memory_match_attempts": 0,
        "scaffold_memory_matched_records": 0,
        "scaffold_memory_useful_matched_records": 0,
        "scaffold_memory_broad_generic_debt_matched_records": 0,
        "scaffold_memory_unmatched_records": 0,
        "scaffold_memory_matches_by_kind": {},
        "scaffold_memory_useful_by_kind": {},
        "scaffold_memory_examples": [],
        "scaffold_memory_authority_allowed_count": 0,
        "scaffold_memory_behavior_effects": 0,
    }


def empty_scaffold_metrics() -> dict[str, Any]:
    return {
        "scaffold_memory_loaded_proposals": 0,
        "scaffold_memory_proposals_by_kind": {},
        "scaffold_memory_broad_generic_debt_proposals": 0,
        "scaffold_memory_ranking_applications": 0,
        "scaffold_memory_candidates_reordered": 0,
        "scaffold_memory_top1_supported": 0,
        "scaffold_memory_topk_supported": 0,
        "scaffold_memory_broad_generic_noops": 0,
        "scaffold_memory_no_runtime_hook_available": 0,
        "scaffold_memory_feature_examples": [],
        "scaffold_memory_match_attempts": 0,
        "scaffold_memory_matched_records": 0,
        "scaffold_memory_useful_matched_records": 0,
        "scaffold_memory_broad_generic_debt_matched_records": 0,
        "scaffold_memory_unmatched_records": 0,
        "scaffold_memory_matches_by_kind": {},
        "scaffold_memory_useful_by_kind": {},
        "scaffold_memory_examples": [],
        "scaffold_memory_authority_allowed_count": 0,
        "scaffold_memory_behavior_effects": 0,
    }
