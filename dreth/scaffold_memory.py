from __future__ import annotations

"""ScaffoldMemoryIndex: loads offline sleep proposals, matches runtime records.

Familiarity/provenance plus bounded assist-feature ordering.

Hard invariants:
  - authority_allowed=False on every loaded proposal
  - record mode has no behavior effects
  - assist_feature may reorder existing candidate lists only
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
    """Load offline scaffold proposals and match against runtime records.

    Familiarity/provenance telemetry plus bounded assist-feature ranking.

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
        self._by_nethra_id.clear()
        self._by_var.clear()
        self._by_context.clear()
        self._by_var_cr_kind.clear()
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

    def reset_runtime_metrics(self) -> None:
        self._ranking_applications = 0
        self._candidates_reordered = 0
        self._top1_supported = 0
        self._topk_supported = 0
        self._broad_generic_noops = 0
        self._no_runtime_hook_available = 0
        self._feature_examples = []

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

    def is_broad_generic_debt_match(
        self,
        proposal: ScaffoldMemoryProposal,
        *,
        var: int | None = None,
        context_key: str = "",
    ) -> bool:
        """Return whether a match is broad debt telemetry only."""
        if not proposal.broad_generic_debt:
            return False
        if var is not None and proposal.vars and int(var) not in proposal.vars:
            return False
        if context_key and proposal.contexts and context_key not in proposal.contexts:
            return False
        return True

    def useful_local_matches(
        self,
        var: int,
        context_key: str,
        candidate: Any | None = None,
    ) -> list[tuple[ScaffoldMemoryProposal, int, tuple[str, ...]]]:
        """Return useful local proposal support for an existing candidate.

        Scores are familiarity hints only. They are not truth, authority,
        derivation support, or permission to create work.
        """
        if not self._proposals:
            return []
        var = int(var)
        candidate_parents = _candidate_parents(candidate)
        candidate_signature = _candidate_signature(var, candidate)
        candidate_nethra_ids = _candidate_nethra_ids(var, candidate)
        context_key = str(context_key or "")
        context_family = _context_family(context_key)

        proposals: list[ScaffoldMemoryProposal] = []
        seen: set[str] = set()
        for p in self._by_var.get(var, []):
            if p.proposal_id not in seen:
                seen.add(p.proposal_id)
                proposals.append(p)
        if context_key:
            for p in self._by_context.get(context_key, []):
                if p.proposal_id not in seen:
                    seen.add(p.proposal_id)
                    proposals.append(p)

        out: list[tuple[ScaffoldMemoryProposal, int, tuple[str, ...]]] = []
        for p in proposals:
            if self.is_broad_generic_debt_match(p, var=var, context_key=context_key):
                continue
            score, reasons = self._score_local_candidate(
                p,
                var=var,
                context_key=context_key,
                context_family=context_family,
                candidate_parents=candidate_parents,
                candidate_signature=candidate_signature,
                candidate_nethra_ids=candidate_nethra_ids,
            )
            if score > 0:
                out.append((p, score, tuple(reasons)))
        out.sort(key=lambda item: (-item[1], item[0].proposal_id))
        return out

    def rank_candidate_keys(
        self,
        var: int,
        context_key: str,
        candidates: Any,
    ) -> Any:
        """Rank existing parent candidate keys by useful local scaffold support."""
        return self._rank_existing_candidates(var, context_key, candidates, hook="parent")

    def rank_frontier_candidates(
        self,
        var: int,
        context_key: str,
        candidates: Any,
    ) -> Any:
        """Rank existing tied-frontier candidates by useful local support."""
        return self._rank_existing_candidates(var, context_key, candidates, hook="frontier")

    def rank_uncertainty_local_anchors(
        self,
        var: int,
        context_key: str,
        candidates: Any,
    ) -> Any:
        """Rank existing uncertainty-local anchors by useful scaffold support."""
        return self._rank_existing_candidates(var, context_key, candidates, hook="uncertainty")

    def annotate_uncertainty_case(self, case: Any) -> dict[str, Any]:
        """Return scaffold support annotation for an uncertainty case."""
        var = int(getattr(case, "var", 0))
        context_key = _uncertainty_context_key(case)
        matches = self.useful_local_matches(var, context_key, case)
        broad = self._broad_generic_matches_for(var=var, context_key=context_key)
        if broad:
            self._broad_generic_noops += len(broad)
        return {
            "var": var,
            "context_key": context_key,
            "score": sum(score for _, score, _ in matches),
            "matches": len(matches),
            "broad_generic_noops": len(broad),
            "reasons": sorted({reason for _, _, reasons in matches for reason in reasons}),
        }

    def note_no_runtime_hook_available(self) -> None:
        self._no_runtime_hook_available += 1

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

    def _proposal_runtime_usable(self, p: ScaffoldMemoryProposal) -> bool:
        if p.broad_generic_debt:
            return False
        if p.suggested_runtime_use == "no_runtime_use":
            return False
        if any(str(w) in {"low_specificity", "no_runtime_use"} for w in p.warnings):
            return False
        return True

    def _score_local_candidate(
        self,
        p: ScaffoldMemoryProposal,
        *,
        var: int,
        context_key: str,
        context_family: str,
        candidate_parents: tuple[int, ...],
        candidate_signature: str,
        candidate_nethra_ids: set[str],
    ) -> tuple[int, list[str]]:
        if not self._proposal_runtime_usable(p):
            return 0, []

        reasons: list[str] = []
        score = 0
        same_var = var in set(int(v) for v in p.vars)
        if same_var:
            score += 1
            reasons.append("same_var")

        p_contexts = set(str(c) for c in p.contexts)
        same_context = bool(context_key and context_key in p_contexts)
        same_family = bool(
            context_family
            and any(_context_family(ctx) == context_family for ctx in p_contexts)
        )
        if same_var and same_context:
            score += 2
            reasons.append("same_var_context")
        elif same_var and same_family:
            score += 1
            reasons.append("same_context_family")

        p_parent_sets = [frozenset(int(x) for x in row) for row in p.common_parents]
        cand_parent_set = frozenset(candidate_parents)
        if cand_parent_set and p_parent_sets:
            if any(cand_parent_set == ps for ps in p_parent_sets):
                score += 2
                reasons.append("shared_parent_signature")
            elif any(cand_parent_set & ps for ps in p_parent_sets):
                score += 1
                reasons.append("shared_parent")

        p_signatures = set(str(sig) for sig in p.common_signatures)
        if candidate_signature and candidate_signature in p_signatures:
            score += 2
            reasons.append("shared_signature")

        if candidate_nethra_ids and candidate_nethra_ids & set(p.source_record_ids):
            score += 2
            reasons.append("repeated_same_nethra_id")

        if (
            p.kind in {"unresolved_family", "unresolved_pattern"}
            or any(str(ctx).startswith("tied_frontier|") for ctx in p.contexts)
            or any(str(nid).startswith("frontier:") for nid in p.source_record_ids)
        ):
            if same_var and (same_context or same_family or cand_parent_set):
                score += 1
                reasons.append("tied_frontier_family")

        if p.kind == "context_role_recurrence" and same_var and (same_context or same_family):
            score += 1
            reasons.append("context_role_recurrence")

        if not reasons:
            return 0, []
        return score, reasons

    def _rank_existing_candidates(
        self,
        var: int,
        context_key: str,
        candidates: Any,
        *,
        hook: str,
    ) -> Any:
        original_type = type(candidates)
        original = list(candidates or [])
        if len(original) < 2:
            return candidates

        scored: list[tuple[int, int, Any, tuple[str, ...], str]] = []
        broad_noops = 0
        for idx, candidate in enumerate(original):
            candidate_var = int(getattr(candidate, "var", var))
            matches = self.useful_local_matches(candidate_var, context_key, candidate)
            score = sum(s for _, s, _ in matches)
            reasons = tuple(sorted({r for _, _, rs in matches for r in rs}))
            best_id = matches[0][0].proposal_id if matches else ""
            broad = self._broad_generic_matches_for(var=candidate_var, context_key=str(context_key or ""))
            broad_noops += len(broad) if score == 0 else 0
            scored.append((score, idx, candidate, reasons, best_id))

        if broad_noops:
            self._broad_generic_noops += broad_noops
        if not any(score > 0 for score, *_ in scored):
            return candidates

        ranked = [candidate for score, _, candidate, _, _ in sorted(scored, key=lambda row: (-row[0], row[1]))]
        self._ranking_applications += 1
        ranked_scored = sorted(scored, key=lambda row: (-row[0], row[1]))
        self._top1_supported += int(ranked_scored[0][0] > 0) if ranked_scored else 0
        self._topk_supported += sum(
            1 for score, *_ in ranked_scored[: min(3, len(ranked_scored))] if score > 0
        )
        if ranked != original:
            self._candidates_reordered += 1
        if len(self._feature_examples) < 10:
            best = max(scored, key=lambda row: (row[0], -row[1]))
            self._feature_examples.append({
                "hook": hook,
                "var": int(var),
                "context_key": str(context_key or ""),
                "candidate": _candidate_label(best[2]),
                "score": best[0],
                "reasons": list(best[3]),
                "proposal_id": best[4],
                "reordered": ranked != original,
            })

        if original_type is tuple:
            return tuple(ranked)
        if original_type is frozenset:
            return frozenset(ranked)
        if original_type is set:
            return set(ranked)
        return ranked

    def _broad_generic_matches_for(
        self,
        *,
        var: int,
        context_key: str,
    ) -> list[ScaffoldMemoryProposal]:
        return [
            p for p in self._by_var.get(int(var), [])
            if self.is_broad_generic_debt_match(p, var=var, context_key=context_key)
        ]


def _context_family(context_key: str) -> str:
    if not context_key:
        return ""
    parts = str(context_key).split("|")
    if not parts:
        return ""
    family = [parts[0]]
    for part in parts[1:]:
        if part.startswith("x") or part.startswith("vis="):
            continue
        family.append(part)
    return "|".join(family)


def _candidate_parents(candidate: Any | None) -> tuple[int, ...]:
    if candidate is None:
        return ()
    if isinstance(candidate, int):
        return (int(candidate),)
    if hasattr(candidate, "parents"):
        return tuple(int(p) for p in (getattr(candidate, "parents") or ()))
    if isinstance(candidate, (tuple, list)):
        if len(candidate) >= 1 and isinstance(candidate[0], (tuple, list, set, frozenset)):
            return tuple(sorted(int(p) for p in candidate[0]))
        if all(isinstance(x, int) for x in candidate):
            return tuple(sorted(int(x) for x in candidate))
    return ()


def _candidate_func(candidate: Any | None) -> str:
    if candidate is None:
        return ""
    if hasattr(candidate, "func"):
        return str(getattr(candidate, "func") or "")
    if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
        return str(candidate[1])
    return ""


def _candidate_signature(var: int, candidate: Any | None) -> str:
    parents = _candidate_parents(candidate)
    func = _candidate_func(candidate)
    if not func:
        return ""
    return f"x{int(var)}:{func}({','.join(str(int(p)) for p in parents)})"


def _candidate_nethra_ids(var: int, candidate: Any | None) -> set[str]:
    parents = _candidate_parents(candidate)
    func = _candidate_func(candidate)
    out: set[str] = set()
    if func:
        args = ",".join(str(int(p)) for p in parents)
        out.add(f"frontier:x{int(var)}:{func}({args})")
        out.add(f"dormant:x{int(var)}:{func}({args})")
        out.add(f"var_fit:x{int(var)}:{func}({args})")
    return out


def _candidate_label(candidate: Any) -> str:
    if isinstance(candidate, int):
        return f"x{candidate}"
    parents = _candidate_parents(candidate)
    func = _candidate_func(candidate)
    if func:
        return f"{func}({','.join(str(p) for p in parents)})"
    return str(candidate)


def _uncertainty_context_key(case: Any) -> str:
    signals = tuple(str(s) for s in getattr(case, "active_signals", ()) or ())
    bits = ["uncertainty_cluster"]
    action = str(getattr(case, "action", "") or "")
    if action:
        bits.append(action)
    if signals:
        bits.append("signals=" + ",".join(signals))
    return "|".join(bits)


# ── Per-run match computation (used by batch_run.py) ──────────────────────────

def compute_run_scaffold_metrics(
    index: ScaffoldMemoryIndex,
    background_nethra_export: dict[str, Any],
    context_role_export: dict[str, Any],
    authority_strength_export: dict[str, Any],
    max_examples: int = 5,
    runtime_metrics: dict[str, Any] | None = None,
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

    runtime = dict(runtime_metrics if runtime_metrics is not None else index.runtime_metrics())
    runtime["scaffold_memory_broad_generic_noops"] = (
        int(runtime.get("scaffold_memory_broad_generic_noops", 0) or 0)
        + broad_generic_debt_matched_records
    )
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
        **runtime,
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
        "scaffold_memory_ranking_applications": 0,
        "scaffold_memory_candidates_reordered": 0,
        "scaffold_memory_top1_supported": 0,
        "scaffold_memory_topk_supported": 0,
        "scaffold_memory_broad_generic_noops": 0,
        "scaffold_memory_no_runtime_hook_available": 0,
        "scaffold_memory_feature_examples": [],
    }
