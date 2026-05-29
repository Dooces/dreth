from __future__ import annotations

"""Offline MemorySleepConsolidator: reads exported runtime memory and builds
scaffold proposals from background nethras, context-role records, uncertainty
records, and authority/debt records.

Core invariant:
  Offline sleep creates proposals only.
  It does not issue authority, revoke authority, suppress skips, replace fit,
  increase monitoring, increase repair priority, use hidden truth or debug
  manifest fields, or treat recurrence as proof.
  Proposals are not wired into the runtime agent in this module.

A sleep proposal means: "these familiar structures may belong together."
It does not mean: "this is true", "this is tareth", "this should be trusted",
or "this should affect runtime behavior."
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple


HIDDEN_TRUTH_LIKE_FIELDS: frozenset[str] = frozenset({
    "truth_source_edges",
    "truth_func",
    "truth_delayed_source_edges",
    "truth_latents",
    "debug_blind_challenge_manifest",
})

_BG_KIND_TO_PROPOSAL_KIND: dict[str, str] = {
    "trass_pattern": "trass_family",
    "unresolved_pattern": "unresolved_family",
    "tied_frontier_pattern": "tied_frontier_family",
    "dormant_alternative_pattern": "dormant_family",
    "quarantined_pattern": "authority_debt_family",
    "recurring_low_salience_pattern": "giant_cluster_subfamily",
    "context_role_pattern": "context_role_recurrence",
    "temporal_cohort_pattern": "possible_temporal_cohort",
}


# ── Internal observation types ────────────────────────────────────────────────

class _BgObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


class _CRObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


class _AuthObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


class _TempObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


class _MemObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


class _ExpObs(NamedTuple):
    rec: dict[str, Any]
    run_idx: int
    seed: int


# ── Proposal and summary dataclasses ──────────────────────────────────────────

@dataclass
class ScaffoldProposal:
    proposal_id: str
    kind: str
    source_record_ids: list[str]
    source_kinds: list[str]
    vars: list[int]
    contexts: list[str]
    common_signatures: list[str]
    common_source_edges: list[list[int]]
    role_patterns: list[str]
    recurring_signals: list[str]
    recurrence_count: int
    runs_seen: int
    seeds_seen: int
    first_seen_cycle: int
    last_seen_cycle: int
    confidence_as_familiarity: float
    action_relevance_score: float
    authority_allowed: bool = False
    suggested_runtime_use: str = "no_runtime_use"
    evidence_summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "source_record_ids": self.source_record_ids,
            "source_kinds": self.source_kinds,
            "vars": self.vars,
            "contexts": self.contexts,
            "common_signatures": self.common_signatures,
            "common_source_edges": self.common_source_edges,
            "role_patterns": self.role_patterns,
            "recurring_signals": self.recurring_signals,
            "recurrence_count": self.recurrence_count,
            "runs_seen": self.runs_seen,
            "seeds_seen": self.seeds_seen,
            "first_seen_cycle": self.first_seen_cycle,
            "last_seen_cycle": self.last_seen_cycle,
            "confidence_as_familiarity": round(self.confidence_as_familiarity, 6),
            "action_relevance_score": round(self.action_relevance_score, 6),
            "authority_allowed": self.authority_allowed,
            "suggested_runtime_use": self.suggested_runtime_use,
            "evidence_summary": self.evidence_summary,
            "warnings": self.warnings,
        }


@dataclass
class MemorySleepSummary:
    input_rows: int
    background_records_seen: int
    context_role_records_seen: int
    uncertainty_records_seen: int
    authority_records_seen: int
    temporal_records_seen: int
    proposals: list[ScaffoldProposal]
    proposals_by_kind: dict[str, int]
    avg_sources_per_proposal: float
    max_sources_per_proposal: int
    largest_proposals: list[str]
    authority_allowed_count: int
    hidden_truth_fields_seen: list[str]
    zero_or_flat_source_fields: list[str]
    compression_ratio: float
    warning_count: int


@dataclass
class SleepProduct:
    proposal_id: str
    member_nethras: list[str]
    touched_atoms: list[str]
    touched_structure_refs: list[str]
    proposed_use_right: str
    proposed_context_scope: str
    salience_delta: float
    evidence_summary: str
    invalidators: list[str]
    reason: str
    authority_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        use_right = self.proposed_use_right
        invalidators = list(self.invalidators)
        if use_right == "hard_filter":
            use_right = "record_only"
            invalidators.append("sleep_hard_filter_rejected")
        return {
            "entry_kind": "sleep_product",
            "record_type": "sleep_product",
            "proposal_id": self.proposal_id,
            "member_nethras": self.member_nethras,
            "touched_atoms": self.touched_atoms,
            "touched_structure_refs": self.touched_structure_refs,
            "proposed_use_right": use_right,
            "proposed_context_scope": self.proposed_context_scope,
            "salience_delta": round(float(self.salience_delta), 6),
            "evidence_summary": self.evidence_summary,
            "invalidators": list(dict.fromkeys(invalidators)),
            "reason": self.reason,
            "authority_allowed": False,
        }


# ── Helper functions ───────────────────────────────────────────────────────────

def _parse_sig_source_edges(sig: str) -> frozenset[int]:
    """Extract source_edge vars from a signature like 'x0:MAX(1,8)' → frozenset({1, 8})."""
    try:
        paren_open = sig.index('(')
        paren_close = sig.index(')')
        inner = sig[paren_open + 1:paren_close]
        if not inner.strip():
            return frozenset()
        parts = []
        for x in inner.split(','):
            x = x.strip()
            if x.lstrip('-').isdigit():
                parts.append(int(x))
        return frozenset(parts)
    except (ValueError, AttributeError):
        return frozenset()


def _intlike(value: Any) -> bool:
    try:
        int(value)
        return value is not None and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


def _bg_anchor_key(rec: dict[str, Any]) -> tuple | None:
    """Return (kind, vars_frozenset, source_edge_frozenset) or None for giant/no-var records.

    Returns None for giant clusters (handled separately) and records with no vars
    (no var anchor means no valid additional anchor beyond kind alone).
    """
    kind = str(rec.get('kind', 'unknown'))
    if kind == 'recurring_low_salience_pattern':
        return None
    vars_list = sorted(int(v) for v in (rec.get('vars') or []))
    if not vars_list:
        return None
    source_edge_sets = rec.get('source_edge_sets') or []
    fit_sigs = rec.get('fit_signatures') or []
    canonical_source_edge: frozenset[int] = frozenset()
    if source_edge_sets:
        canonical_source_edge = frozenset(int(p) for p in (source_edge_sets[0] or []))
    elif fit_sigs:
        canonical_source_edge = _parse_sig_source_edges(str(fit_sigs[0]))
    return (kind, frozenset(vars_list), canonical_source_edge)


def _authority_anchor_key(rec: dict[str, Any]) -> tuple | None:
    auth_state = str(rec.get('authority_state', ''))
    reason = str(rec.get('reason', ''))
    if not auth_state:
        return None
    return ('authority_debt_family', auth_state, reason)


def _cr_anchor_key(rec: dict[str, Any]) -> tuple | None:
    var = rec.get('target_var')
    if var is None:
        return None
    kind = str(rec.get('kind', 'unknown'))
    learned_source_edges = frozenset(int(p) for p in (rec.get('learned_source_edges') or []))
    return ('context_role_recurrence', int(var), kind, learned_source_edges)


def _compute_confidence(recurrence_count: int, runs_seen: int, seeds_seen: int) -> float:
    base = min(0.5, 0.1 * recurrence_count)
    run_bonus = 0.15 * max(0, runs_seen - 1)
    seed_bonus = 0.05 * max(0, seeds_seen - 1)
    return round(min(1.0, base + run_bonus + seed_bonus), 6)


def _compute_suggested_use(
    kind: str,
    runs_seen: int,
    has_source_edge_anchor: bool,
    has_role_patterns: bool,
) -> str:
    if kind == 'giant_cluster_subfamily':
        return 'no_runtime_use'
    if has_role_patterns and runs_seen >= 2:
        return 'ranking_hint'
    if has_source_edge_anchor and runs_seen >= 2:
        return 'clustering_prior'
    return 'feature_only'


def _compute_action_relevance(suggested_use: str, confidence: float) -> float:
    multipliers = {
        'no_runtime_use': 0.0,
        'feature_only': 0.2,
        'clustering_prior': 0.3,
        'ranking_hint': 0.25,
    }
    m = multipliers.get(suggested_use, 0.0)
    return round(confidence * m, 6)


def _scan_hidden_truth_fields(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        for k in row:
            if k in HIDDEN_TRUTH_LIKE_FIELDS:
                found.add(k)
        export = row.get('background_nethra_export') or {}
        for rec in (export.get('records') or []):
            if isinstance(rec, dict):
                for k in rec:
                    if k in HIDDEN_TRUTH_LIKE_FIELDS:
                        found.add(k)
    return sorted(found)


def _diagnose_source_fields(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Returns (zero_or_flat_fields, mismatch_warning_strings)."""
    zero_flat: list[str] = []
    mismatches: list[str] = []

    # background_nethra_edges
    agg_edges = sum(int(r.get('background_nethra_edges', 0) or 0) for r in rows)
    export_edges = sum(
        len((r.get('background_nethra_export') or {}).get('edges') or [])
        for r in rows
    )
    if agg_edges == 0:
        zero_flat.append('background_nethra_edges (aggregate=0)')
    if export_edges == 0:
        zero_flat.append('background_nethra_export.edges (export list empty)')

    # background_contexts_seen
    agg_ctx_present = any('background_contexts_seen' in r for r in rows)
    agg_ctx_total = sum(int(r.get('background_contexts_seen', 0) or 0) for r in rows)
    export_ctx_nonempty = sum(
        1 for r in rows
        for rec in (r.get('background_nethra_export') or {}).get('records') or []
        if isinstance(rec, dict) and rec.get('context_keys')
    )
    if not agg_ctx_present:
        zero_flat.append('background_contexts_seen (field absent from JSONL)')
        if export_ctx_nonempty > 0:
            mismatches.append(
                f"background_contexts_seen: field absent from JSONL output (not serialized by "
                f"batch_run.py), but {export_ctx_nonempty} export records contain non-empty "
                f"context_keys — aggregate field is flat, but export records contain usable "
                f"per-record data."
            )
    elif agg_ctx_total == 0:
        zero_flat.append('background_contexts_seen (aggregate=0)')
        if export_ctx_nonempty > 0:
            mismatches.append(
                f"background_contexts_seen: aggregate is 0, but {export_ctx_nonempty} "
                f"export records contain non-empty context_keys — aggregate field is flat, "
                f"but export records contain usable per-record data."
            )

    # background_recognition_score_mean
    agg_recog_present = any('background_recognition_score_mean' in r for r in rows)
    agg_recog_total = sum(
        float(r.get('background_recognition_score_mean', 0.0) or 0.0) for r in rows
    )
    export_recog_nonzero = sum(
        1 for r in rows
        for rec in (r.get('background_nethra_export') or {}).get('records') or []
        if isinstance(rec, dict) and float(rec.get('cheap_recognition_score', 0) or 0) > 0
    )
    if not agg_recog_present:
        zero_flat.append('background_recognition_score_mean (field absent from JSONL)')
        if export_recog_nonzero > 0:
            mismatches.append(
                f"background_recognition_score_mean: field absent from JSONL output (not "
                f"serialized by batch_run.py), but {export_recog_nonzero} export records "
                f"contain nonzero cheap_recognition_score values — aggregate field is flat, "
                f"but export records contain usable per-record data."
            )
    elif agg_recog_total == 0.0:
        zero_flat.append('background_recognition_score_mean (aggregate=0.0)')
        if export_recog_nonzero > 0:
            mismatches.append(
                f"background_recognition_score_mean: aggregate is 0.0, but "
                f"{export_recog_nonzero} export records contain nonzero cheap_recognition_score "
                f"values — aggregate field is flat, but export records contain usable "
                f"per-record data."
            )

    # background_nethra_export.records[].relation_edges
    export_relation_edges_nonzero = sum(
        1 for r in rows
        for rec in (r.get('background_nethra_export') or {}).get('records') or []
        if isinstance(rec, dict) and rec.get('relation_edges')
    )
    if export_relation_edges_nonzero == 0:
        zero_flat.append('background_nethra_export.records[].relation_edges (all empty)')

    return zero_flat, mismatches


# ── MemorySleepConsolidator ────────────────────────────────────────────────────

class MemorySleepConsolidator:
    """Offline consolidator that reads exported runtime memory and builds
    scaffold proposals from background nethras, context-role records, uncertainty
    records, and authority/debt records.

    Invariants:
      - authority_allowed is always False on every proposal
      - Does not import from agent.py or call the runtime agent
      - Does not use HIDDEN_TRUTH_LIKE_FIELDS values
      - Does not use relation_type unless posthoc_relation_type=True
      - Recurrence alone is not proof; it is familiarity signal only
    """

    def load_jsonl_rows(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def extract_background_records(self, rows: list[dict[str, Any]]) -> list[_BgObs]:
        result: list[_BgObs] = []
        for run_idx, row in enumerate(rows):
            seed = int(row.get('seed', run_idx))
            export = row.get('background_nethra_export') or {}
            if not isinstance(export, dict):
                continue
            for rec in (export.get('records') or []):
                if isinstance(rec, dict):
                    result.append(_BgObs(rec=rec, run_idx=run_idx, seed=seed))
        return result

    def extract_context_role_records(self, rows: list[dict[str, Any]]) -> list[_CRObs]:
        result: list[_CRObs] = []
        for run_idx, row in enumerate(rows):
            seed = int(row.get('seed', run_idx))
            cri = row.get('context_role_index') or {}
            if not isinstance(cri, dict):
                continue
            for rec in (cri.get('records') or []):
                if isinstance(rec, dict):
                    result.append(_CRObs(rec=rec, run_idx=run_idx, seed=seed))
        return result

    def extract_uncertainty_records(self, rows: list[dict[str, Any]]) -> list[_BgObs]:
        """Extract background records sourced from uncertainty clusters.

        Includes records with source_role containing 'uncertainty_cluster',
        kind 'recurring_low_salience_pattern', or payload is_giant=True.
        """
        result: list[_BgObs] = []
        for run_idx, row in enumerate(rows):
            seed = int(row.get('seed', run_idx))
            export = row.get('background_nethra_export') or {}
            if not isinstance(export, dict):
                continue
            for rec in (export.get('records') or []):
                if not isinstance(rec, dict):
                    continue
                source_roles = rec.get('source_roles') or []
                kind = str(rec.get('kind', ''))
                payload = rec.get('payload') or {}
                is_unc = (
                    any('uncertainty_cluster' in str(r) for r in source_roles)
                    or kind == 'recurring_low_salience_pattern'
                    or (isinstance(payload, dict) and payload.get('is_giant'))
                )
                if is_unc:
                    result.append(_BgObs(rec=rec, run_idx=run_idx, seed=seed))
        return result

    def extract_authority_records(self, rows: list[dict[str, Any]]) -> list[_AuthObs]:
        result: list[_AuthObs] = []
        for run_idx, row in enumerate(rows):
            seed = int(row.get('seed', run_idx))
            auth = row.get('authority_strength') or {}
            if not isinstance(auth, dict):
                continue
            for rec in (auth.get('records') or []):
                if isinstance(rec, dict):
                    result.append(_AuthObs(rec=rec, run_idx=run_idx, seed=seed))
        return result

    def extract_temporal_records_if_available(
        self, rows: list[dict[str, Any]]
    ) -> list[_TempObs]:
        """Extract temporal_cohort_pattern records from background export, if any."""
        result: list[_TempObs] = []
        for run_idx, row in enumerate(rows):
            seed = int(row.get('seed', run_idx))
            export = row.get('background_nethra_export') or {}
            if not isinstance(export, dict):
                continue
            for rec in (export.get('records') or []):
                if isinstance(rec, dict) and str(rec.get('kind', '')) == 'temporal_cohort_pattern':
                    result.append(_TempObs(rec=rec, run_idx=run_idx, seed=seed))
        return result

    def extract_nethra_memory_records(self, rows: list[dict[str, Any]]) -> list[_MemObs]:
        result: list[_MemObs] = []
        for run_idx, row in enumerate(rows):
            if row.get("entry_kind") != "record":
                continue
            if any(k in HIDDEN_TRUTH_LIKE_FIELDS or str(k).startswith("debug_") for k in row):
                continue
            result.append(_MemObs(
                rec=row,
                run_idx=run_idx,
                seed=int(row.get("seed", run_idx) or run_idx),
            ))
        return result

    def extract_experience_events(self, rows: list[dict[str, Any]]) -> list[_ExpObs]:
        result: list[_ExpObs] = []
        for run_idx, row in enumerate(rows):
            if row.get("entry_kind") == "experience_event":
                if bool(row.get("hidden_truth_used", False)):
                    continue
                if any(k in HIDDEN_TRUTH_LIKE_FIELDS or str(k).startswith("debug_") for k in row):
                    continue
                result.append(_ExpObs(
                    rec=row,
                    run_idx=run_idx,
                    seed=int(row.get("seed", run_idx) or run_idx),
                ))
            for event in row.get("nethra_memory_experience_events") or []:
                if not isinstance(event, dict) or bool(event.get("hidden_truth_used", False)):
                    continue
                result.append(_ExpObs(
                    rec=event,
                    run_idx=run_idx,
                    seed=int(row.get("seed", run_idx) or run_idx),
                ))
        return result

    def build_sleep_products(
        self,
        memory_records: list[_MemObs],
        experience_events: list[_ExpObs],
        *,
        min_sources: int = 1,
        max_products: int = 2000,
    ) -> list[SleepProduct]:
        by_context_atom: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "members": set(),
                "atoms": set(),
                "refs": set(),
                "events": [],
                "records": [],
                "failures": 0,
                "successes": 0,
                "behavior_effects": 0,
                "is_structure_hint": False,
            }
        )
        for obs in memory_records:
            rec = obs.rec
            # context_role records with source_edge info → build selective source_edge_candidates hints.
            # These encode certified source_edges as the preferred set, giving differential signal
            # that the ranker's random initial ordering cannot mask.
            if str(rec.get("record_type", "")) == "context_role":
                vars_list = [int(v) for v in (rec.get("vars") or []) if _intlike(v)]
                n_vars = int(rec.get("n_vars", 0) or 0)
                if vars_list and n_vars > 0:
                    target_var = vars_list[0]
                    source_edge_vars: list[int] = []
                    for ref in (rec.get("touched_structure_refs") or []):
                        ref_s = str(ref)
                        if ref_s.startswith("source_edges:") and ref_s[13:].strip():
                            try:
                                source_edge_vars = [int(p) for p in ref_s[13:].split(",") if p.strip()]
                            except ValueError:
                                pass
                            break
                    if source_edge_vars:
                        source_edge_context = f"source_edge_candidates|x{target_var}|vis={n_vars}"
                        members = [str(n) for n in (rec.get("member_nethras") or [])]
                        if rec.get("nethra_id"):
                            members.append(str(rec["nethra_id"]))
                        refs = [str(r) for r in (rec.get("touched_structure_refs") or [])]
                        for source_edge_var in source_edge_vars:
                            p_atom = f"x{source_edge_var}"
                            bucket = by_context_atom[(source_edge_context, p_atom)]
                            bucket["atoms"].add(p_atom)
                            bucket["refs"].update(refs)
                            bucket["members"].update(members)
                            bucket["records"].append(rec)
                            bucket["failures"] += int(rec.get("failure_count", 0) or 0)
                            bucket["successes"] += int(rec.get("success_count", 0) or 0)
                            bucket["is_structure_hint"] = True
                        continue  # skip normal bucketing for this record
            context = str(rec.get("context_scope") or (rec.get("contexts") or [""])[0])
            atoms = [str(a) for a in (rec.get("touched_atoms") or [])]
            if not atoms:
                atoms = [f"x{int(v)}" for v in (rec.get("vars") or []) if _intlike(v)]
            refs = [str(r) for r in (rec.get("touched_structure_refs") or [])]
            members = [str(n) for n in (rec.get("member_nethras") or [])]
            if rec.get("nethra_id"):
                members.append(str(rec["nethra_id"]))
            for atom in atoms or [""]:
                bucket = by_context_atom[(context, atom)]
                bucket["members"].update(members)
                bucket["atoms"].update(atoms)
                bucket["refs"].update(refs)
                bucket["records"].append(rec)
                bucket["failures"] += int(rec.get("failure_count", 0) or 0)
                bucket["successes"] += int(rec.get("success_count", 0) or 0)
        for obs in experience_events:
            rec = obs.rec
            context = str(rec.get("context_key", ""))
            members = [str(n) for n in (rec.get("active_nethras") or [])]
            behavior_effect = int(rec.get("behavior_effect", 0) or 0)
            if behavior_effect > 0:
                # For reorder events, only credit the atoms that actually moved earlier —
                # crediting all active_atoms would make preferred = all candidates again.
                cands_before = [int(c) for c in (rec.get("candidates_before") or []) if _intlike(c)]
                cands_after = [int(c) for c in (rec.get("candidates_after") or []) if _intlike(c)]
                if cands_before and cands_after:
                    before_pos = {c: i for i, c in enumerate(cands_before)}
                    after_pos = {c: i for i, c in enumerate(cands_after)}
                    boosted = [
                        c for c in cands_after
                        if after_pos.get(c, 99) < before_pos.get(c, 99)
                    ]
                    atoms_to_use = [f"x{c}" for c in boosted] if boosted else [
                        str(a) for a in (rec.get("active_atoms") or [])
                    ]
                else:
                    atoms_to_use = [str(a) for a in (rec.get("active_atoms") or [])]
            else:
                atoms_to_use = [str(a) for a in (rec.get("active_atoms") or [])]
            for atom in atoms_to_use or [""]:
                bucket = by_context_atom[(context, atom)]
                bucket["members"].update(members)
                # Do NOT add active_atoms to bucket["atoms"] — they are the full query
                # atom set and would make touched_atoms too broad. Only memory records
                # contribute selective atoms; experience events contribute membership/
                # behavioral signal only.
                bucket["events"].append(rec)
                bucket["behavior_effects"] += behavior_effect
                if rec.get("success"):
                    bucket["successes"] += 1
                if rec.get("failure_reason"):
                    bucket["failures"] += 1

        products: list[SleepProduct] = []
        for i, ((context, atom), bucket) in enumerate(
            sorted(by_context_atom.items(), key=lambda item: (-len(item[1]["events"]), item[0]))
        ):
            if len(products) >= max_products:
                break
            source_count = len(bucket["records"]) + len(bucket["events"])
            if source_count < min_sources:
                continue
            failures = int(bucket["failures"])
            successes = int(bucket["successes"])
            behavior_effects = int(bucket["behavior_effects"])
            if failures and not successes:
                use_right = "feature_only"
                reason = "negative gate from visible failure association"
                salience_delta = -0.5 * failures
                invalidators = ["visible_failure_association"]
            elif behavior_effects > 0:
                use_right = "ranking_hint"
                reason = "prior assist behavior effect survived visible audit path"
                salience_delta = 0.4 + 0.1 * successes
                invalidators = []
            elif bucket["is_structure_hint"] and not failures:
                # Structural: certified source_edge configuration from context_role records.
                # Provides differential signal that experience-event-only bootstrapping cannot:
                # preferred = only the true source_edges, not all candidates.
                use_right = "ranking_hint"
                reason = "structural: certified source_edge configuration from context_role"
                salience_delta = 0.2
                invalidators = []
            else:
                use_right = "feature_only"
                reason = "familiar structure recurrence"
                salience_delta = 0.1 * max(1, successes)
                invalidators = []
            products.append(SleepProduct(
                proposal_id=f"sleep_{i:06d}_{abs(hash((context, atom))) % 10_000_000}",
                member_nethras=sorted(bucket["members"]),
                touched_atoms=sorted(bucket["atoms"]) or [atom],
                touched_structure_refs=sorted(bucket["refs"]),
                proposed_use_right=use_right,
                proposed_context_scope=context,
                salience_delta=salience_delta,
                evidence_summary=(
                    f"{source_count} visible source(s), successes={successes}, "
                    f"failures={failures}, behavior_effects={behavior_effects}"
                ),
                invalidators=invalidators,
                reason=reason,
                authority_allowed=False,
            ))
        return products

    def build_proposals(
        self,
        bg: list[_BgObs],
        cr: list[_CRObs],
        unc: list[_BgObs],
        auth: list[_AuthObs],
        temp: list[_TempObs],
        *,
        min_sources: int = 2,
        max_proposals: int = 2000,
        max_sources_per_proposal: int = 500,
        posthoc_relation_type: bool = False,
    ) -> list[ScaffoldProposal]:
        """Build scaffold proposals from extracted records.

        Grouping rules:
          - A proposal requires at least one anchor beyond kind alone: shared var,
            shared context family, shared source_edge/signature, or cross-run recurrence
            of the same nethra_id.
          - Generic unresolved/tied-frontier status alone is not sufficient.
          - Giant clusters are split by var and marked no_runtime_use.
          - posthoc_relation_type: if False (default), relation_type fields are ignored.
        """
        proposals: list[ScaffoldProposal] = []
        counter = [0]

        def _next_id(prefix: str) -> str:
            counter[0] += 1
            return f"prop_{counter[0]:06d}_{prefix[:12]}"

        # Phase 1: Cross-run background nethra_id recurrence
        # Same nethra_id seen in ≥ min_sources observations → strongest familiarity signal.
        # Giant clusters (recurring_low_salience_pattern) are excluded here and handled
        # exclusively in Phase 3 where they are split by individual var.
        by_nid: dict[str, list[_BgObs]] = defaultdict(list)
        for obs in bg:
            if obs.rec.get('kind') == 'recurring_low_salience_pattern':
                continue  # giant clusters go to Phase 3 only
            nid = str(obs.rec.get('nethra_id', ''))
            if nid:
                by_nid[nid].append(obs)

        used_nids: set[str] = set()
        for nid, obs_list in sorted(by_nid.items(), key=lambda kv: -len(kv[1])):
            if len(obs_list) < min_sources:
                continue
            if len(proposals) >= max_proposals:
                break
            p = self._recurrence_proposal(nid, obs_list, max_sources_per_proposal, _next_id)
            proposals.append(p)
            used_nids.add(nid)

        # Phase 2: Structural anchor grouping for ungrouped background records
        # Groups by (kind, vars, canonical_source_edge_set). Giant clusters excluded here.
        ungrouped = [obs for obs in bg if str(obs.rec.get('nethra_id', '')) not in used_nids]
        giants = [
            obs for obs in ungrouped
            if obs.rec.get('kind') == 'recurring_low_salience_pattern'
        ]
        non_giants = [
            obs for obs in ungrouped
            if obs.rec.get('kind') != 'recurring_low_salience_pattern'
        ]

        anchor_groups: dict[tuple, list[_BgObs]] = defaultdict(list)
        for obs in non_giants:
            key = _bg_anchor_key(obs.rec)
            if key is not None:
                anchor_groups[key].append(obs)

        for key, obs_list in sorted(anchor_groups.items(), key=lambda kv: -len(kv[1])):
            if len(obs_list) < min_sources or len(proposals) >= max_proposals:
                continue
            p = self._anchor_proposal(key, obs_list, max_sources_per_proposal, _next_id)
            proposals.append(p)

        # Phase 3: Giant cluster subfamilies — split by individual var, marked no_runtime_use
        giant_by_var: dict[int, list[_BgObs]] = defaultdict(list)
        for obs in giants:
            for var in (obs.rec.get('vars') or []):
                giant_by_var[int(var)].append(obs)

        for var, obs_list in sorted(giant_by_var.items()):
            if len(obs_list) < min_sources or len(proposals) >= max_proposals:
                continue
            p = self._giant_subfamily_proposal(var, obs_list, max_sources_per_proposal, _next_id)
            proposals.append(p)

        # Phase 4: Authority strength grouping by (auth_state, reason)
        auth_groups: dict[tuple, list[_AuthObs]] = defaultdict(list)
        for obs in auth:
            key = _authority_anchor_key(obs.rec)
            if key is not None:
                auth_groups[key].append(obs)

        for key, obs_list in sorted(auth_groups.items(), key=lambda kv: -len(kv[1])):
            if len(obs_list) < min_sources or len(proposals) >= max_proposals:
                continue
            p = self._authority_proposal(key, obs_list, max_sources_per_proposal, _next_id)
            proposals.append(p)

        # Phase 5: Context-role index recurrence by (var, kind, learned_source_edges)
        cr_groups: dict[tuple, list[_CRObs]] = defaultdict(list)
        for obs in cr:
            key = _cr_anchor_key(obs.rec)
            if key is not None:
                cr_groups[key].append(obs)

        for key, obs_list in sorted(cr_groups.items(), key=lambda kv: -len(kv[1])):
            if len(obs_list) < min_sources or len(proposals) >= max_proposals:
                continue
            p = self._cr_proposal(key, obs_list, max_sources_per_proposal, _next_id)
            proposals.append(p)

        return proposals

    def summarize(
        self,
        rows: list[dict[str, Any]],
        bg: list[_BgObs],
        cr: list[_CRObs],
        unc: list[_BgObs],
        auth: list[_AuthObs],
        temp: list[_TempObs],
        proposals: list[ScaffoldProposal],
    ) -> MemorySleepSummary:
        n_proposals = len(proposals)
        by_kind: Counter[str] = Counter(p.kind for p in proposals)

        source_counts = [p.recurrence_count for p in proposals]
        avg_src = sum(source_counts) / max(1, n_proposals)
        max_src = max(source_counts) if source_counts else 0

        largest = sorted(proposals, key=lambda p: -p.recurrence_count)[:5]
        largest_ids = [p.proposal_id for p in largest]

        auth_allowed = sum(1 for p in proposals if p.authority_allowed)
        warn_count = sum(len(p.warnings) for p in proposals)

        ratio = float(len(bg)) / max(1, n_proposals)

        hidden = _scan_hidden_truth_fields(rows)
        zero_flat, mismatches = _diagnose_source_fields(rows)
        diag_fields = zero_flat + [f"MISMATCH: {m}" for m in mismatches]

        return MemorySleepSummary(
            input_rows=len(rows),
            background_records_seen=len(bg),
            context_role_records_seen=len(cr),
            uncertainty_records_seen=len(unc),
            authority_records_seen=len(auth),
            temporal_records_seen=len(temp),
            proposals=proposals,
            proposals_by_kind=dict(by_kind),
            avg_sources_per_proposal=round(avg_src, 4),
            max_sources_per_proposal=max_src,
            largest_proposals=largest_ids,
            authority_allowed_count=auth_allowed,
            hidden_truth_fields_seen=hidden,
            zero_or_flat_source_fields=diag_fields,
            compression_ratio=round(ratio, 4),
            warning_count=warn_count,
        )

    # ── Internal proposal builders ─────────────────────────────────────────────

    def _recurrence_proposal(
        self,
        nethra_id: str,
        obs_list: list[_BgObs],
        max_src: int,
        next_id: Any,
    ) -> ScaffoldProposal:
        capped = obs_list[:max_src]
        runs = sorted(set(o.run_idx for o in capped))
        seeds = sorted(set(o.seed for o in capped))

        all_vars: list[int] = sorted(set(
            int(v) for o in capped for v in (o.rec.get('vars') or [])
        ))
        all_ctx: list[str] = list(dict.fromkeys(
            c for o in capped for c in (o.rec.get('context_keys') or [])
        ))[:20]
        all_sigs: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('fit_signatures') or [])
        ))[:10]
        seen_source_edge_fs: set[frozenset] = set()
        all_source_edges: list[list[int]] = []
        for o in capped:
            for ps in (o.rec.get('source_edge_sets') or []):
                fp = frozenset(int(p) for p in ps)
                if fp not in seen_source_edge_fs:
                    seen_source_edge_fs.add(fp)
                    all_source_edges.append(sorted(int(p) for p in ps))
        all_roles: list[str] = list(dict.fromkeys(
            r for o in capped for r in (o.rec.get('source_roles') or [])
        ))[:10]
        all_signals: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('recurring_signals') or [])
        ))[:10]
        first_cycle = min(int(o.rec.get('first_seen_cycle', 0) or 0) for o in capped)
        last_cycle = max(int(o.rec.get('last_seen_cycle', 0) or 0) for o in capped)

        bg_kinds: Counter[str] = Counter(str(o.rec.get('kind', 'unknown')) for o in capped)
        dominant_bg = bg_kinds.most_common(1)[0][0]
        kinds_in_proposal = list(dict.fromkeys(str(o.rec.get('kind', 'unknown')) for o in capped))
        if len(bg_kinds) > 1:
            pk = 'background_nethra_group'
        else:
            pk = _BG_KIND_TO_PROPOSAL_KIND.get(dominant_bg, 'background_nethra_group')

        confidence = _compute_confidence(len(capped), len(runs), len(seeds))
        has_source_edge = bool(all_source_edges)
        informative_roles = [r for r in all_roles if r not in ('unresolved', 'best_available')]
        use = _compute_suggested_use(pk, len(runs), has_source_edge, bool(informative_roles))
        action = _compute_action_relevance(use, confidence)

        warnings: list[str] = []
        if pk == 'giant_cluster_subfamily':
            warnings.append('low_specificity')

        evidence = (
            f"nethra_id '{nethra_id[:50]}' seen across {len(runs)} run(s) "
            f"(seeds={seeds}); {len(capped)} source observation(s); kind={dominant_bg}"
        )

        return ScaffoldProposal(
            proposal_id=next_id(pk),
            kind=pk,
            source_record_ids=[nethra_id],
            source_kinds=kinds_in_proposal,
            vars=all_vars,
            contexts=all_ctx,
            common_signatures=all_sigs,
            common_source_edges=all_source_edges[:10],
            role_patterns=all_roles,
            recurring_signals=all_signals,
            recurrence_count=len(capped),
            runs_seen=len(runs),
            seeds_seen=len(seeds),
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
            confidence_as_familiarity=confidence,
            action_relevance_score=action,
            authority_allowed=False,
            suggested_runtime_use=use,
            evidence_summary=evidence,
            warnings=warnings,
        )

    def _anchor_proposal(
        self,
        key: tuple,
        obs_list: list[_BgObs],
        max_src: int,
        next_id: Any,
    ) -> ScaffoldProposal:
        # key = (kind_str, vars_frozenset, source_edge_frozenset)
        bg_kind_str, vars_fs, source_edge_fs = key
        capped = obs_list[:max_src]
        runs = sorted(set(o.run_idx for o in capped))
        seeds = sorted(set(o.seed for o in capped))

        source_ids: list[str] = list(dict.fromkeys(
            str(o.rec.get('nethra_id', '')) for o in capped
        ))[:max_src]

        all_vars = sorted(vars_fs)
        all_ctx: list[str] = list(dict.fromkeys(
            c for o in capped for c in (o.rec.get('context_keys') or [])
        ))[:20]
        all_sigs: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('fit_signatures') or [])
        ))[:10]
        seen_source_edge_fs: set[frozenset] = set()
        all_source_edges: list[list[int]] = []
        for o in capped:
            for ps in (o.rec.get('source_edge_sets') or []):
                fp = frozenset(int(p) for p in ps)
                if fp not in seen_source_edge_fs:
                    seen_source_edge_fs.add(fp)
                    all_source_edges.append(sorted(int(p) for p in ps))
        all_roles: list[str] = list(dict.fromkeys(
            r for o in capped for r in (o.rec.get('source_roles') or [])
        ))[:10]
        all_signals: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('recurring_signals') or [])
        ))[:10]
        first_cycle = min(int(o.rec.get('first_seen_cycle', 0) or 0) for o in capped)
        last_cycle = max(int(o.rec.get('last_seen_cycle', 0) or 0) for o in capped)

        pk = _BG_KIND_TO_PROPOSAL_KIND.get(str(bg_kind_str), 'background_nethra_group')
        confidence = _compute_confidence(len(capped), len(runs), len(seeds))
        has_source_edge = bool(source_edge_fs)
        informative_roles = [r for r in all_roles if r not in ('unresolved', 'best_available', 'dormant')]
        use = _compute_suggested_use(pk, len(runs), has_source_edge, bool(informative_roles))
        action = _compute_action_relevance(use, confidence)

        evidence = (
            f"{bg_kind_str} records for vars={sorted(vars_fs)}, "
            f"source_edges={sorted(source_edge_fs)}: {len(capped)} source record(s) "
            f"across {len(runs)} run(s)"
        )

        return ScaffoldProposal(
            proposal_id=next_id(pk),
            kind=pk,
            source_record_ids=source_ids,
            source_kinds=[str(bg_kind_str)],
            vars=all_vars,
            contexts=all_ctx,
            common_signatures=all_sigs,
            common_source_edges=all_source_edges[:10],
            role_patterns=all_roles,
            recurring_signals=all_signals,
            recurrence_count=len(capped),
            runs_seen=len(runs),
            seeds_seen=len(seeds),
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
            confidence_as_familiarity=confidence,
            action_relevance_score=action,
            authority_allowed=False,
            suggested_runtime_use=use,
            evidence_summary=evidence,
            warnings=[],
        )

    def _giant_subfamily_proposal(
        self,
        var: int,
        obs_list: list[_BgObs],
        max_src: int,
        next_id: Any,
    ) -> ScaffoldProposal:
        capped = obs_list[:max_src]
        runs = sorted(set(o.run_idx for o in capped))
        seeds = sorted(set(o.seed for o in capped))

        source_ids: list[str] = list(dict.fromkeys(
            str(o.rec.get('nethra_id', '')) for o in capped
        ))[:max_src]
        all_ctx: list[str] = list(dict.fromkeys(
            c for o in capped for c in (o.rec.get('context_keys') or [])
        ))[:10]
        all_sigs: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('fit_signatures') or [])
        ))[:5]
        all_roles: list[str] = list(dict.fromkeys(
            r for o in capped for r in (o.rec.get('source_roles') or [])
        ))[:5]
        all_signals: list[str] = list(dict.fromkeys(
            s for o in capped for s in (o.rec.get('recurring_signals') or [])
        ))[:5]
        first_cycle = min(int(o.rec.get('first_seen_cycle', 0) or 0) for o in capped)
        last_cycle = max(int(o.rec.get('last_seen_cycle', 0) or 0) for o in capped)
        confidence = _compute_confidence(len(capped), len(runs), len(seeds))

        return ScaffoldProposal(
            proposal_id=next_id('giant_cluster_subfamily'),
            kind='giant_cluster_subfamily',
            source_record_ids=source_ids,
            source_kinds=['recurring_low_salience_pattern'],
            vars=[var],
            contexts=all_ctx,
            common_signatures=all_sigs,
            common_source_edges=[],
            role_patterns=all_roles,
            recurring_signals=all_signals,
            recurrence_count=len(capped),
            runs_seen=len(runs),
            seeds_seen=len(seeds),
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
            confidence_as_familiarity=confidence,
            action_relevance_score=0.0,
            authority_allowed=False,
            suggested_runtime_use='no_runtime_use',
            evidence_summary=(
                f"giant_cluster_subfamily for var={var}: split from giant uncertainty cluster; "
                f"{len(capped)} source record(s) across {len(runs)} run(s)"
            ),
            warnings=['low_specificity'],
        )

    def _authority_proposal(
        self,
        key: tuple,
        obs_list: list[_AuthObs],
        max_src: int,
        next_id: Any,
    ) -> ScaffoldProposal:
        _, auth_state, reason = key
        capped = obs_list[:max_src]
        runs = sorted(set(o.run_idx for o in capped))
        seeds = sorted(set(o.seed for o in capped))

        vars_in_group = sorted(set(
            int(o.rec.get('var', 0))
            for o in capped
            if o.rec.get('var') is not None
        ))
        source_ids: list[str] = list(dict.fromkeys(
            str(o.rec.get('nethra_id', '')) for o in capped
        ))[:max_src]
        all_signals: list[str] = list(dict.fromkeys(
            s for o in capped
            for s in (o.rec.get('uncertainty_signals') or [])
        ))[:10]
        first_cycle = min(int(o.rec.get('cycle', 0) or 0) for o in capped)
        last_cycle = max(int(o.rec.get('cycle', 0) or 0) for o in capped)

        confidence = _compute_confidence(len(capped), len(runs), len(seeds))
        use = 'feature_only'
        action = _compute_action_relevance(use, confidence)

        role_patterns = [auth_state]
        if reason:
            role_patterns.append(reason)

        return ScaffoldProposal(
            proposal_id=next_id('authority_debt_family'),
            kind='authority_debt_family',
            source_record_ids=source_ids,
            source_kinds=['authority_strength'],
            vars=vars_in_group,
            contexts=[],
            common_signatures=[],
            common_source_edges=[],
            role_patterns=role_patterns,
            recurring_signals=all_signals,
            recurrence_count=len(capped),
            runs_seen=len(runs),
            seeds_seen=len(seeds),
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
            confidence_as_familiarity=confidence,
            action_relevance_score=action,
            authority_allowed=False,
            suggested_runtime_use=use,
            evidence_summary=(
                f"authority_debt_family: {len(capped)} var(s) share "
                f"auth_state='{auth_state}' reason='{reason}' "
                f"across {len(runs)} run(s)"
            ),
            warnings=[],
        )

    def _cr_proposal(
        self,
        key: tuple,
        obs_list: list[_CRObs],
        max_src: int,
        next_id: Any,
    ) -> ScaffoldProposal:
        # key = ('context_role_recurrence', var, cr_kind, source_edge_frozenset)
        _, var, cr_kind, source_edge_fs = key
        capped = obs_list[:max_src]
        runs = sorted(set(o.run_idx for o in capped))
        seeds = sorted(set(o.seed for o in capped))

        source_ids: list[str] = list(dict.fromkeys(
            str(o.rec.get('nethra_id', '')) for o in capped
        ))[:max_src]
        all_sigs: list[str] = list(dict.fromkeys(
            str(o.rec.get('signature', '')) for o in capped
            if o.rec.get('signature')
        ))[:10]
        first_cycle = min(int(o.rec.get('first_seen_cycle', 0) or 0) for o in capped)
        last_cycle = max(int(o.rec.get('last_seen_cycle', 0) or 0) for o in capped)

        confidence = _compute_confidence(len(capped), len(runs), len(seeds))
        has_source_edge = bool(source_edge_fs)
        use = _compute_suggested_use('context_role_recurrence', len(runs), has_source_edge, False)
        action = _compute_action_relevance(use, confidence)

        return ScaffoldProposal(
            proposal_id=next_id('context_role_recurrence'),
            kind='context_role_recurrence',
            source_record_ids=source_ids,
            source_kinds=[str(cr_kind)],
            vars=[int(var)],
            contexts=[],
            common_signatures=all_sigs,
            common_source_edges=[sorted(source_edge_fs)] if source_edge_fs else [],
            role_patterns=[str(cr_kind)],
            recurring_signals=[],
            recurrence_count=len(capped),
            runs_seen=len(runs),
            seeds_seen=len(seeds),
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
            confidence_as_familiarity=confidence,
            action_relevance_score=action,
            authority_allowed=False,
            suggested_runtime_use=use,
            evidence_summary=(
                f"context_role_recurrence for var={var}, kind={cr_kind}, "
                f"source_edges={sorted(source_edge_fs)}: {len(capped)} source record(s) "
                f"across {len(runs)} run(s)"
            ),
            warnings=[],
        )
