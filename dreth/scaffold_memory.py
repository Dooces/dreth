from __future__ import annotations

"""ScaffoldMemoryIndex: loads offline sleep proposals, matches runtime records.

Familiarity/provenance telemetry only. No behavior effects.

Hard invariants:
  - authority_allowed=False on every loaded proposal
  - behavior_effects=0 always
  - No authority issuance, revocation, skip suppression, fit replacement,
    monitoring increase, repair priority increase, derivation support
  - No hidden truth/debug manifest reads or use
"""

import json
from collections import Counter, defaultdict
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
    """Load offline scaffold proposals and match against runtime records.

    Familiarity/provenance telemetry. No behavior effects of any kind.

    Usage:
        index = ScaffoldMemoryIndex()
        index.load_proposals("reports/memory_sleep_proposals.jsonl")
        matches = index.match_background_record(record_dict)
        stats = index.summarize_matches()
    """

    def __init__(self) -> None:
        self._proposals: list[ScaffoldMemoryProposal] = []
        # Index structures (rebuilt on load_proposals)
        self._by_nethra_id: dict[str, list[ScaffoldMemoryProposal]] = defaultdict(list)
        self._by_var: dict[int, list[ScaffoldMemoryProposal]] = defaultdict(list)
        self._by_context: dict[str, list[ScaffoldMemoryProposal]] = defaultdict(list)
        self._by_var_cr_kind: dict[tuple[int, str], list[ScaffoldMemoryProposal]] = defaultdict(list)
        # Proposals by kind (loaded time)
        self._by_kind: Counter[str] = Counter()
        self._broad_generic_debt_proposals: int = 0

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
        self._by_nethra_id.clear()
        self._by_var.clear()
        self._by_context.clear()
        self._by_var_cr_kind.clear()
        self._by_kind.clear()
        self._broad_generic_debt_proposals = 0

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
                # Skip any row that contains hidden truth fields
                if any(k in HIDDEN_TRUTH_LIKE_FIELDS for k in d):
                    continue
                p = ScaffoldMemoryProposal.from_dict(d)
                if not p.proposal_id:
                    continue
                self._proposals.append(p)
                self._by_kind[p.kind] += 1
                if p.broad_generic_debt:
                    self._broad_generic_debt_proposals += 1
                # Build indices
                for nid in p.source_record_ids:
                    if nid:
                        self._by_nethra_id[nid].append(p)
                for v in p.vars:
                    self._by_var[v].append(p)
                for ctx in p.contexts:
                    if ctx:
                        self._by_context[ctx].append(p)
                # For context-role proposals index by (var, kind)
                if p.kind == "context_role_recurrence" and p.vars:
                    for rp in p.role_patterns:
                        for v in p.vars:
                            self._by_var_cr_kind[(v, rp)].append(p)

        return len(self._proposals)

    # ── Matching ───────────────────────────────────────────────────────────────

    def match_background_record(self, record: dict[str, Any]) -> list[ScaffoldMemoryProposal]:
        """Match a background nethra record against loaded proposals.

        Matching requires at least one structural anchor beyond kind alone:
        - Phase 1: exact nethra_id match (strongest)
        - Phase 2: var overlap + (context OR signature OR parent overlap)

        Authority_debt_family proposals (no anchors) do not match here;
        use match_authority_strength_record for those.
        """
        return self._match_bg_record(record)

    def match_context_role_record(self, record: dict[str, Any]) -> list[ScaffoldMemoryProposal]:
        """Match a context-role index record against context_role_recurrence proposals.

        Match requires: same target_var + role_pattern overlap + parent overlap
        (or parent overlap is sufficient if role matches).
        """
        target_var = record.get("target_var")
        if target_var is None:
            return []
        target_var = int(target_var)
        cr_kind = str(record.get("kind", ""))
        learned_parents = frozenset(int(p) for p in (record.get("learned_parents") or []))

        candidates: list[ScaffoldMemoryProposal] = []
        seen_ids: set[str] = set()

        # Try by (var, kind)
        for p in self._by_var_cr_kind.get((target_var, cr_kind), []):
            if p.proposal_id not in seen_ids:
                seen_ids.add(p.proposal_id)
                candidates.append(p)
        # Also try all proposals for this var
        for p in self._by_var.get(target_var, []):
            if p.kind == "context_role_recurrence" and p.proposal_id not in seen_ids:
                seen_ids.add(p.proposal_id)
                candidates.append(p)

        results: list[ScaffoldMemoryProposal] = []
        for p in candidates:
            if p.kind != "context_role_recurrence":
                continue
            # Require either: role_pattern match or parent overlap
            role_ok = not p.role_patterns or cr_kind in p.role_patterns
            if not role_ok:
                continue
            if p.common_parents:
                p_parents = frozenset(x for row in p.common_parents for x in row)
                if not learned_parents or not (learned_parents & p_parents):
                    continue
            # Accept: role matches and either parents match or proposal has no parent constraint
            results.append(p)

        return results

    def match_uncertainty_record(self, record: dict[str, Any]) -> list[ScaffoldMemoryProposal]:
        """Match an uncertainty/giant-cluster background record against proposals."""
        return self._match_bg_record(record)

    def match_authority_strength_record(
        self, record: dict[str, Any]
    ) -> list[ScaffoldMemoryProposal]:
        """Match an authority strength record against authority_debt_family proposals.

        Match requires: var in proposal.vars + role_pattern overlap
        (auth_state or reason appears in proposal.role_patterns).
        """
        var = record.get("var")
        if var is None:
            return []
        var = int(var)
        auth_state = str(record.get("authority_state", ""))
        reason = str(record.get("reason", ""))

        results: list[ScaffoldMemoryProposal] = []
        seen_ids: set[str] = set()

        for p in self._by_var.get(var, []):
            if p.kind != "authority_debt_family":
                continue
            if p.proposal_id in seen_ids:
                continue
            if auth_state in p.role_patterns or reason in p.role_patterns:
                seen_ids.add(p.proposal_id)
                results.append(p)

        return results

    # ── Query ──────────────────────────────────────────────────────────────────

    def query_by_var(self, var: int) -> list[ScaffoldMemoryProposal]:
        """Return all proposals that include var."""
        return list(self._by_var.get(var, []))

    def query_by_context(self, context_key: str) -> list[ScaffoldMemoryProposal]:
        """Return all proposals that include context_key."""
        return list(self._by_context.get(context_key, []))

    def summarize_matches(self) -> dict[str, Any]:
        """Return index-level summary (loaded proposals, by kind, broad_generic_debt count)."""
        return {
            "scaffold_memory_loaded_proposals": len(self._proposals),
            "scaffold_memory_proposals_by_kind": dict(self._by_kind),
            "scaffold_memory_broad_generic_debt_proposals": self._broad_generic_debt_proposals,
            "scaffold_memory_authority_allowed_count": 0,
            "scaffold_memory_behavior_effects": 0,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _match_bg_record(self, record: dict[str, Any]) -> list[ScaffoldMemoryProposal]:
        nethra_id = str(record.get("nethra_id", ""))
        vars_list = [int(v) for v in (record.get("vars") or [])]
        context_keys: set[str] = set(record.get("context_keys") or [])
        parent_sets: list[frozenset[int]] = [
            frozenset(int(p) for p in ps)
            for ps in (record.get("parent_sets") or [])
        ]
        fit_sigs: set[str] = set(record.get("fit_signatures") or [])

        seen_ids: set[str] = set()
        results: list[ScaffoldMemoryProposal] = []

        # Phase 1: exact nethra_id match (strongest anchor)
        if nethra_id:
            for p in self._by_nethra_id.get(nethra_id, []):
                if p.proposal_id not in seen_ids:
                    seen_ids.add(p.proposal_id)
                    results.append(p)

        # Phase 2: var overlap + at least one structural anchor
        candidate_set: set[str] = set()
        candidates: list[ScaffoldMemoryProposal] = []
        for v in vars_list:
            for p in self._by_var.get(v, []):
                if p.proposal_id not in seen_ids and p.proposal_id not in candidate_set:
                    candidate_set.add(p.proposal_id)
                    candidates.append(p)

        for p in candidates:
            if p.proposal_id in seen_ids:
                continue
            # Need at least one local anchor
            has_context = bool(context_keys & set(p.contexts))
            has_sig = bool(fit_sigs & set(p.common_signatures))
            has_parent = False
            if parent_sets and p.common_parents:
                p_parent_sets = [frozenset(row) for row in p.common_parents]
                for rec_ps in parent_sets:
                    for p_ps in p_parent_sets:
                        if rec_ps & p_ps:
                            has_parent = True
                            break
                    if has_parent:
                        break
            if has_context or has_sig or has_parent:
                seen_ids.add(p.proposal_id)
                results.append(p)

        return results


# ── Per-run match computation (used by batch_run.py) ──────────────────────────

def compute_run_scaffold_metrics(
    index: ScaffoldMemoryIndex,
    background_nethra_export: dict[str, Any],
    context_role_export: dict[str, Any],
    authority_strength_export: dict[str, Any],
    max_examples: int = 5,
) -> dict[str, Any]:
    """Compute per-run scaffold memory match metrics.

    Arguments are the exported data dicts from a completed run's ArchMetrics.
    Returns a dict with all scaffold_memory_* metric keys.

    Invariants enforced here:
      authority_allowed_count = 0 always
      behavior_effects = 0 always
    """
    match_attempts = 0
    matched_records = 0
    useful_matched_records = 0
    broad_generic_debt_matched_records = 0
    unmatched_records = 0
    matches_by_kind: Counter[str] = Counter()
    useful_by_kind: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    def _process_matches(
        record: dict[str, Any],
        proposals: list[ScaffoldMemoryProposal],
        record_label: str,
    ) -> None:
        nonlocal matched_records, useful_matched_records
        nonlocal broad_generic_debt_matched_records, unmatched_records

        if not proposals:
            unmatched_records += 1
            return

        matched_records += 1
        has_useful = False
        has_broad = False

        for p in proposals:
            matches_by_kind[p.kind] += 1
            if p.broad_generic_debt:
                has_broad = True
            else:
                has_useful = True
                useful_by_kind[p.kind] += 1

        if has_useful:
            useful_matched_records += 1
        if has_broad and not has_useful:
            broad_generic_debt_matched_records += 1

        if len(examples) < max_examples:
            best = max(proposals, key=lambda p: p.confidence_as_familiarity)
            examples.append({
                "record_label": record_label,
                "record_nethra_id": str(record.get("nethra_id", ""))[:60],
                "record_kind": str(record.get("kind", "")),
                "matched_proposal_id": best.proposal_id,
                "matched_kind": best.kind,
                "broad_generic_debt": best.broad_generic_debt,
                "confidence": round(best.confidence_as_familiarity, 4),
                "n_proposals_matched": len(proposals),
            })

    # Background nethra records
    bg_records = (background_nethra_export or {}).get("records") or []
    for rec in bg_records:
        if not isinstance(rec, dict):
            continue
        match_attempts += 1
        source_roles = rec.get("source_roles") or []
        kind = str(rec.get("kind", ""))
        payload = rec.get("payload") or {}
        is_uncertainty = (
            any("uncertainty_cluster" in str(r) for r in source_roles)
            or kind == "recurring_low_salience_pattern"
            or (isinstance(payload, dict) and payload.get("is_giant"))
        )
        if is_uncertainty:
            proposals = index.match_uncertainty_record(rec)
        else:
            proposals = index.match_background_record(rec)
        _process_matches(rec, proposals, "background")

    # Context-role records
    cr_records = (context_role_export or {}).get("records") or []
    for rec in cr_records:
        if not isinstance(rec, dict):
            continue
        match_attempts += 1
        proposals = index.match_context_role_record(rec)
        _process_matches(rec, proposals, "context_role")

    # Authority strength records
    auth_records = (authority_strength_export or {}).get("records") or []
    for rec in auth_records:
        if not isinstance(rec, dict):
            continue
        match_attempts += 1
        proposals = index.match_authority_strength_record(rec)
        _process_matches(rec, proposals, "authority_strength")

    match_rate = matched_records / max(1, match_attempts) if match_attempts else 0.0

    return {
        "scaffold_memory_mode": "record",
        "scaffold_memory_loaded_proposals": index.loaded_proposals_count if index else 0,
        "scaffold_memory_match_attempts": match_attempts,
        "scaffold_memory_matches": matched_records,
        "scaffold_memory_useful_matches": useful_matched_records,
        "scaffold_memory_broad_generic_matches": broad_generic_debt_matched_records,
        "scaffold_memory_broad_generic_debt_matches": broad_generic_debt_matched_records,
        "scaffold_memory_matches_by_kind": dict(matches_by_kind),
        "scaffold_memory_useful_matches_by_kind": dict(useful_by_kind),
        "scaffold_memory_match_rate": round(match_rate, 6),
        "scaffold_memory_unmatched_records": unmatched_records,
        "scaffold_memory_authority_allowed_count": 0,
        "scaffold_memory_behavior_effects": 0,
        "scaffold_memory_match_examples": examples,
    }


def empty_scaffold_metrics() -> dict[str, Any]:
    """Return zero-valued scaffold metrics for off-mode rows."""
    return {
        "scaffold_memory_mode": "off",
        "scaffold_memory_loaded_proposals": 0,
        "scaffold_memory_match_attempts": 0,
        "scaffold_memory_matches": 0,
        "scaffold_memory_useful_matches": 0,
        "scaffold_memory_broad_generic_matches": 0,
        "scaffold_memory_broad_generic_debt_matches": 0,
        "scaffold_memory_matches_by_kind": {},
        "scaffold_memory_useful_matches_by_kind": {},
        "scaffold_memory_match_rate": 0.0,
        "scaffold_memory_unmatched_records": 0,
        "scaffold_memory_authority_allowed_count": 0,
        "scaffold_memory_behavior_effects": 0,
        "scaffold_memory_match_examples": [],
    }
