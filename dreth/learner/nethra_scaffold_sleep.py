from __future__ import annotations

"""Offline nethra scaffold sleep.

NethraScaffoldSleep builds a persistent, non-authoritative scaffold view over
visible exported nethra-like records. It records learned structure, role
histories, and scaffold abstractions for future familiarity matching.

Core invariant:
  scaffold sleep does not issue authority, revoke authority, suppress skips,
  replace fit, force probes, increase monitoring, increase repair priority,
  support derivation, read hidden truth/debug manifest fields, or treat
  recurrence as proof.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, NamedTuple


HIDDEN_TRUTH_LIKE_FIELDS: frozenset[str] = frozenset({
    "truth_source_edges",
    "truth_func",
    "truth_delayed_source_edges",
    "truth_latents",
    "debug_blind_challenge_manifest",
})

SCAFFOLD_ROLES: tuple[str, ...] = (
    "tareth",
    "trass",
    "best_available",
    "unresolved",
    "background",
    "contested_best_available",
    "quarantined_for_derivation",
    "repair_candidate",
    "dormant",
)

BROAD_DEBT_REASONS: frozenset[str] = frozenset({
    "active_visible_conflict",
    "broad_authority_debt",
    "generic_authority_debt",
})


ScaffoldAbstractionKind = Literal[
    "operator_family",
    "source_edge_signature_family",
    "context_role_family",
    "background_object_family",
    "trass_family",
    "tareth_family",
    "unresolved_family",
    "authority_debt_family",
    "uncertainty_family",
    "mixed_role_family",
]

SuggestedRuntimeUse = Literal[
    "no_runtime_use",
    "feature_only",
    "clustering_prior",
    "ranking_hint",
]


class _Obs(NamedTuple):
    source_id: str
    source_type: str
    source_kind: str
    vars: tuple[int, ...]
    signatures: tuple[str, ...]
    source_edge_sets: tuple[tuple[int, ...], ...]
    contexts: tuple[str, ...]
    roles: tuple[str, ...]
    run_idx: int
    seed: int
    first_cycle: int
    last_cycle: int
    operator_family: str
    payload: dict[str, Any]


@dataclass
class ScaffoldNethra:
    scaffold_id: str
    source_ids: list[str]
    source_types: list[str]
    vars: list[int]
    signatures: list[str]
    source_edge_sets: list[list[int]]
    contexts: list[str]
    observed_roles: list[str]
    role_counts: dict[str, int]
    role_contexts: dict[str, list[str]]
    first_seen_cycle: int
    last_seen_cycle: int
    runs_seen: int
    seeds_seen: int
    familiarity_score: float
    specificity_score: float
    stability_score: float
    authority_allowed: bool = False

    def __post_init__(self) -> None:
        self.authority_allowed = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = "scaffold_nethra"
        d["authority_allowed"] = False
        d["familiarity_score"] = round(float(self.familiarity_score), 6)
        d["specificity_score"] = round(float(self.specificity_score), 6)
        d["stability_score"] = round(float(self.stability_score), 6)
        return d


@dataclass
class ScaffoldComposition:
    composition_id: str
    lower_scaffold_ids: list[str]
    higher_scaffold_id: str
    shared_vars: list[int]
    shared_contexts: list[str]
    shared_signatures: list[str]
    evidence_summary: str
    confidence_as_familiarity: float
    authority_allowed: bool = False

    def __post_init__(self) -> None:
        self.authority_allowed = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = "scaffold_composition"
        d["authority_allowed"] = False
        d["confidence_as_familiarity"] = round(float(self.confidence_as_familiarity), 6)
        return d


@dataclass
class ScaffoldRoleMap:
    scaffold_id: str
    role_by_context: dict[str, list[str]]
    trass_contexts: list[str]
    tareth_contexts: list[str]
    unresolved_contexts: list[str]
    background_contexts: list[str]
    role_shift_examples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = "scaffold_role_map"
        d["authority_allowed"] = False
        return d


@dataclass
class ScaffoldAbstraction:
    abstraction_id: str
    kind: ScaffoldAbstractionKind
    member_scaffold_ids: list[str]
    common_structure: dict[str, Any]
    role_distribution: dict[str, int]
    specificity_score: float
    familiarity_score: float
    suggested_runtime_use: SuggestedRuntimeUse
    authority_allowed: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.authority_allowed = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = "scaffold_abstraction"
        d["authority_allowed"] = False
        d["specificity_score"] = round(float(self.specificity_score), 6)
        d["familiarity_score"] = round(float(self.familiarity_score), 6)
        return d


@dataclass
class NethraScaffoldSleepSummary:
    raw_records_read: int
    scaffold_nethras: int
    compositions: int
    abstractions: int
    role_maps: int
    tareth_records_seen: int
    trass_records_seen: int
    background_records_seen: int
    unresolved_records_seen: int
    authority_debt_records_seen: int
    stable_best_available_records_seen: int
    broad_generic_debt_count: int
    broad_generic_debt_useful_count: int
    authority_allowed_count: int
    behavior_effects: int
    hidden_truth_fields_seen: list[str]
    role_shift_examples: list[dict[str, Any]]
    composition_examples: list[dict[str, Any]]
    abstraction_counts_by_kind: dict[str, int]
    warning: str = (
        "scaffold sleep builds familiarity and abstraction, not truth or authority"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    raw = json.dumps([str(p) for p in parts], sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _sanitize(value: Any, hidden_seen: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key in HIDDEN_TRUTH_LIKE_FIELDS or key.startswith("debug_"):
                if hidden_seen is not None:
                    hidden_seen.add(key)
                continue
            out[key] = _sanitize(v, hidden_seen)
        return out
    if isinstance(value, list):
        return [_sanitize(v, hidden_seen) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(v, hidden_seen) for v in value)
    return value


def _intlike(value: Any) -> bool:
    try:
        int(value)
        return value is not None and not isinstance(value, bool)
    except (TypeError, ValueError):
        return False


def _ints(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values if _intlike(v)}))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _context_family(context: str) -> str:
    head = str(context or "").split("|", 1)[0]
    if "=" in head:
        return head.split("=", 1)[0]
    return head


_SIG_RE = re.compile(r"(x\d+:[A-Za-z_][A-Za-z0-9_]*\([^)]*\))")
_OP_RE = re.compile(r"x\d+:([A-Za-z_][A-Za-z0-9_]*)\(")


def _signature_from_id(value: str) -> str:
    match = _SIG_RE.search(str(value or ""))
    return match.group(1) if match else ""


def _operator_from_signature(signature: str) -> str:
    match = _OP_RE.search(str(signature or ""))
    return match.group(1).upper() if match else ""


def _source_edges_from_signature(signature: str) -> tuple[int, ...]:
    text = str(signature or "")
    if "(" not in text or ")" not in text:
        return ()
    inner = text[text.index("(") + 1:text.index(")")]
    return _ints(part.strip() for part in inner.split(",") if part.strip())


def _extract_vars(rec: dict[str, Any]) -> tuple[int, ...]:
    values: list[Any] = []
    values.extend(_as_list(rec.get("vars")))
    values.extend(_as_list(rec.get("components")))
    for key in ("var", "target_var"):
        if key in rec:
            values.append(rec.get(key))
    return _ints(values)


def _extract_source_edge_sets(rec: dict[str, Any], signatures: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    raw_sets = rec.get("source_edge_sets")
    if isinstance(raw_sets, list):
        for row in raw_sets:
            source_edges = _ints(_as_list(row))
            if source_edges and source_edges not in seen:
                seen.add(source_edges)
                out.append(source_edges)
    for key in ("learned_source_edges", "source_edges", "best_source_edges"):
        if key in rec:
            source_edges = _ints(_as_list(rec.get(key)))
            if source_edges and source_edges not in seen:
                seen.add(source_edges)
                out.append(source_edges)
    for sig in signatures:
        source_edges = _source_edges_from_signature(sig)
        if source_edges and source_edges not in seen:
            seen.add(source_edges)
            out.append(source_edges)
    return tuple(out)


def _extract_signatures(rec: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("signature", "fit_signature", "learned_signature"):
        if rec.get(key):
            values.append(str(rec[key]))
    for value in _as_list(rec.get("fit_signatures")):
        if value:
            values.append(str(value))
    inferred = _signature_from_id(str(rec.get("nethra_id", "")))
    if inferred:
        values.append(inferred)
    if rec.get("target_var") is not None and rec.get("learned_func") is not None:
        source_edges = ",".join(str(p) for p in _ints(_as_list(rec.get("learned_source_edges"))))
        values.append(f"x{int(rec['target_var'])}:{str(rec['learned_func']).upper()}({source_edges})")
    return tuple(dict.fromkeys(values))


def _extract_contexts(rec: dict[str, Any]) -> tuple[str, ...]:
    contexts: list[str] = []
    for value in _as_list(rec.get("context_keys")):
        if value:
            contexts.append(str(value))
    if rec.get("context_key"):
        contexts.append(str(rec["context_key"]))
    operation = rec.get("operation")
    if operation and rec.get("target_var") is not None:
        contexts.append(f"{operation}|x{int(rec['target_var'])}")
    return tuple(dict.fromkeys(contexts))


def _extract_roles(rec: dict[str, Any], source_type: str, source_kind: str) -> tuple[str, ...]:
    roles: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "")
        if not text:
            return
        if text.startswith("authority_state:"):
            text = text.split(":", 1)[1]
        if text in SCAFFOLD_ROLES:
            roles.append(text)

    for key in ("role", "skip_role", "prior_role", "current_role", "authority_state"):
        add(rec.get(key))
    for value in _as_list(rec.get("source_roles")) + _as_list(rec.get("operation_roles")):
        add(value)

    if bool(rec.get("best_available")):
        add("best_available")
    if source_type == "authority_strength":
        add(rec.get("authority_state"))
        if rec.get("best_available"):
            add("best_available")
    if source_kind == "trass_pattern":
        add("trass")
    elif source_kind == "unresolved_pattern":
        add("unresolved")
    elif source_kind in {"context_role_pattern", "recurring_low_salience_pattern"}:
        add("background")
    elif source_kind == "dormant_alternative_pattern":
        add("dormant")
    elif source_kind == "quarantined_pattern":
        add("quarantined_for_derivation")
    elif source_kind == "tied_frontier_pattern":
        add("unresolved")
    if not roles:
        add("background")
    return tuple(dict.fromkeys(roles))


def _cycle_pair(rec: dict[str, Any]) -> tuple[int, int]:
    first = rec.get("first_seen_cycle", rec.get("cycle_start", rec.get("cycle", 0)))
    last = rec.get("last_seen_cycle", rec.get("cycle_end", rec.get("cycle", first)))
    return int(first or 0), int(last or 0)


def _source_id(rec: dict[str, Any], source_type: str, run_idx: int, pos: int) -> str:
    for key in ("nethra_id", "record_id", "proposal_id", "cluster_id", "id"):
        if rec.get(key):
            return str(rec[key])
    return f"{source_type}:{run_idx}:{pos}"


def _source_kind(rec: dict[str, Any], source_type: str) -> str:
    for key in ("kind", "source_kind", "record_type", "strength", "authority_state"):
        if rec.get(key):
            return str(rec[key])
    return source_type


def _observation(
    rec: dict[str, Any],
    *,
    source_type: str,
    run_idx: int,
    seed: int,
    pos: int,
) -> _Obs:
    source_kind = _source_kind(rec, source_type)
    signatures = _extract_signatures(rec)
    source_edge_sets = _extract_source_edge_sets(rec, signatures)
    operator = ""
    for sig in signatures:
        operator = _operator_from_signature(sig)
        if operator:
            break
    first, last = _cycle_pair(rec)
    return _Obs(
        source_id=_source_id(rec, source_type, run_idx, pos),
        source_type=source_type,
        source_kind=source_kind,
        vars=_extract_vars(rec),
        signatures=signatures,
        source_edge_sets=source_edge_sets,
        contexts=_extract_contexts(rec),
        roles=_extract_roles(rec, source_type, source_kind),
        run_idx=run_idx,
        seed=seed,
        first_cycle=first,
        last_cycle=last,
        operator_family=operator,
        payload=rec,
    )


def _is_broad_generic_debt(obs: _Obs) -> bool:
    reason = str(obs.payload.get("reason", ""))
    if obs.source_type != "authority_strength":
        return False
    if reason not in BROAD_DEBT_REASONS:
        return False
    return not obs.signatures and not obs.source_edge_sets


def _structure_key(obs: _Obs) -> tuple[Any, ...]:
    if obs.signatures:
        return ("signature", obs.signatures[0])
    if obs.vars and obs.source_edge_sets:
        return ("var_source_edge_operator", obs.vars, obs.source_edge_sets[0], obs.operator_family)
    if obs.source_edge_sets and obs.operator_family:
        return ("source_edge_operator", obs.source_edge_sets[0], obs.operator_family)
    if obs.vars and obs.operator_family:
        return ("var_operator", obs.vars, obs.operator_family)
    if obs.vars and obs.contexts and not _is_broad_generic_debt(obs):
        return ("var_context_family", obs.vars, tuple(_context_family(c) for c in obs.contexts))
    if obs.source_id:
        return ("source_id", obs.source_id)
    return ("singleton", obs.source_type, obs.run_idx)


def _score_familiarity(obs_list: list[_Obs]) -> float:
    runs = len({o.run_idx for o in obs_list})
    seeds = len({o.seed for o in obs_list})
    base = min(0.55, 0.12 * len(obs_list))
    return min(1.0, base + 0.12 * max(0, runs - 1) + 0.04 * max(0, seeds - 1))


def _score_specificity(
    vars_: list[int],
    signatures: list[str],
    source_edge_sets: list[list[int]],
    contexts: list[str],
    *,
    broad_debt: bool = False,
) -> float:
    if broad_debt:
        return 0.0
    score = 0.0
    if vars_:
        score += 0.20
    if signatures:
        score += 0.35
    if source_edge_sets:
        score += 0.25
    if contexts:
        score += 0.15
    if len(vars_) > 20 and not signatures and not source_edge_sets:
        score -= 0.25
    return max(0.0, min(1.0, score))


def _score_stability(obs_list: list[_Obs], signatures: list[str], source_edge_sets: list[list[int]]) -> float:
    if not obs_list:
        return 0.0
    runs = len({o.run_idx for o in obs_list})
    structure_bonus = 0.25 if signatures or source_edge_sets else 0.0
    recurrence = min(0.55, 0.10 * len(obs_list))
    run_bonus = min(0.20, 0.06 * max(0, runs - 1))
    return min(1.0, recurrence + run_bonus + structure_bonus)


class NethraScaffoldSleep:
    """Build non-authoritative scaffold nethras and abstractions offline."""

    def __init__(self) -> None:
        self.hidden_truth_fields_seen: set[str] = set()
        self.raw_records_read: int = 0
        self.role_records_seen: Counter[str] = Counter()
        self.broad_generic_debt_count: int = 0

    def load_rows(self, *paths_or_rows: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in paths_or_rows:
            if isinstance(item, (str, Path)):
                path = Path(item)
                if not path.exists():
                    continue
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            rows.append(_sanitize(row, self.hidden_truth_fields_seen))
            else:
                for row in item:
                    if isinstance(row, dict):
                        rows.append(_sanitize(row, self.hidden_truth_fields_seen))
        return rows

    def _extract_observations(self, rows: list[dict[str, Any]]) -> list[_Obs]:
        observations: list[_Obs] = []

        def add_record(
            rec: Any,
            *,
            source_type: str,
            run_idx: int,
            seed: int,
            pos: int,
        ) -> None:
            if not isinstance(rec, dict):
                return
            clean = _sanitize(rec, self.hidden_truth_fields_seen)
            obs = _observation(
                clean,
                source_type=source_type,
                run_idx=run_idx,
                seed=seed,
                pos=pos,
            )
            observations.append(obs)

        for run_idx, row in enumerate(rows):
            seed = int(row.get("seed", run_idx) or run_idx)

            if row.get("entry_kind") == "record":
                payload = row.get("payload") or {}
                source_type = str(row.get("record_type", "unknown"))
                merged = dict(payload) if isinstance(payload, dict) else {}
                merged.setdefault("record_id", row.get("record_id", ""))
                merged.setdefault("source_kind", row.get("source_kind", ""))
                merged.setdefault("cycle_start", row.get("cycle_start", 0))
                merged.setdefault("cycle_end", row.get("cycle_end", 0))
                add_record(merged, source_type=source_type, run_idx=run_idx, seed=seed, pos=0)
                continue

            if row.get("proposal_id") or str(row.get("record_type", "")).startswith("scaffold_"):
                add_record(row, source_type="scaffold_proposal", run_idx=run_idx, seed=seed, pos=0)
                continue

            bg_export = row.get("background_nethra_export") or {}
            if isinstance(bg_export, dict):
                for pos, rec in enumerate(bg_export.get("records") or []):
                    add_record(rec, source_type="background_nethra", run_idx=run_idx, seed=seed, pos=pos)

            cri = row.get("context_role_index") or {}
            if isinstance(cri, dict):
                for pos, rec in enumerate(cri.get("records") or []):
                    add_record(rec, source_type="context_role", run_idx=run_idx, seed=seed, pos=pos)
                for pos, rec in enumerate(cri.get("nodes") or []):
                    add_record(rec, source_type="context_role", run_idx=run_idx, seed=seed, pos=pos)
                roles = cri.get("roles") or []
                if isinstance(roles, dict):
                    flattened_roles: list[Any] = []
                    for value in roles.values():
                        if isinstance(value, list):
                            flattened_roles.extend(value)
                        else:
                            flattened_roles.append(value)
                    roles = flattened_roles
                for pos, rec in enumerate(roles):
                    add_record(rec, source_type="context_role", run_idx=run_idx, seed=seed, pos=pos)

            auth = row.get("authority_strength") or {}
            if isinstance(auth, dict):
                for pos, rec in enumerate(auth.get("records") or []):
                    add_record(rec, source_type="authority_strength", run_idx=run_idx, seed=seed, pos=pos)

            for field_name in (
                "var_fit_records",
                "audit_records",
                "fit_audit_records",
                "dormant_alternatives",
                "tied_frontier_records",
                "uncertainty_records",
            ):
                for pos, rec in enumerate(row.get(field_name) or []):
                    add_record(rec, source_type=field_name.rstrip("s"), run_idx=run_idx, seed=seed, pos=pos)

            for field_name in ("scaffold_proposals", "previous_scaffold_proposals"):
                for pos, rec in enumerate(row.get(field_name) or []):
                    add_record(rec, source_type="scaffold_proposal", run_idx=run_idx, seed=seed, pos=pos)

            for pos, rec in enumerate(row.get("scaffold_memory_match_examples") or []):
                add_record(rec, source_type="scaffold_proposal", run_idx=run_idx, seed=seed, pos=pos)

        self.raw_records_read = len(observations)
        self.role_records_seen = Counter()
        self.broad_generic_debt_count = 0
        for obs in observations:
            if _is_broad_generic_debt(obs):
                self.broad_generic_debt_count += 1
            for role in obs.roles:
                self.role_records_seen[role] += 1
        return observations

    def extract_scaffold_nethras(
        self,
        rows: list[dict[str, Any]],
        *,
        max_scaffolds: int = 5000,
    ) -> list[ScaffoldNethra]:
        observations = self._extract_observations(rows)
        groups: dict[tuple[Any, ...], list[_Obs]] = defaultdict(list)
        for obs in observations:
            groups[_structure_key(obs)].append(obs)

        nethras: list[ScaffoldNethra] = []
        for key, obs_list in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
            if len(nethras) >= max_scaffolds:
                break
            source_ids = list(dict.fromkeys(o.source_id for o in obs_list if o.source_id))
            source_types = list(dict.fromkeys(o.source_type for o in obs_list if o.source_type))
            vars_ = sorted({v for o in obs_list for v in o.vars})
            signatures = list(dict.fromkeys(s for o in obs_list for s in o.signatures))[:20]
            source_edge_seen: set[tuple[int, ...]] = set()
            source_edge_sets: list[list[int]] = []
            for obs in obs_list:
                for source_edges in obs.source_edge_sets:
                    if source_edges not in source_edge_seen:
                        source_edge_seen.add(source_edges)
                        source_edge_sets.append(list(source_edges))
            contexts = list(dict.fromkeys(c for o in obs_list for c in o.contexts))[:40]
            role_counts = Counter(role for o in obs_list for role in o.roles)
            observed_roles = [role for role in SCAFFOLD_ROLES if role_counts.get(role, 0)]
            role_contexts: dict[str, list[str]] = {}
            for role in observed_roles:
                values: list[str] = []
                for obs in obs_list:
                    if role in obs.roles:
                        values.extend(obs.contexts or ("",))
                role_contexts[role] = [v for v in dict.fromkeys(values) if v][:20]
            first_cycle = min((o.first_cycle for o in obs_list), default=0)
            last_cycle = max((o.last_cycle for o in obs_list), default=0)
            broad_debt = all(_is_broad_generic_debt(o) for o in obs_list)
            scaffold_id = _stable_id("scaffold", key)
            nethras.append(ScaffoldNethra(
                scaffold_id=scaffold_id,
                source_ids=source_ids[:100],
                source_types=source_types,
                vars=vars_[:100],
                signatures=signatures,
                source_edge_sets=source_edge_sets[:20],
                contexts=contexts,
                observed_roles=observed_roles or ["background"],
                role_counts=dict(role_counts),
                role_contexts=role_contexts,
                first_seen_cycle=first_cycle,
                last_seen_cycle=last_cycle,
                runs_seen=len({o.run_idx for o in obs_list}),
                seeds_seen=len({o.seed for o in obs_list}),
                familiarity_score=_score_familiarity(obs_list),
                specificity_score=_score_specificity(
                    vars_,
                    signatures,
                    source_edge_sets,
                    contexts,
                    broad_debt=broad_debt,
                ),
                stability_score=_score_stability(obs_list, signatures, source_edge_sets),
                authority_allowed=False,
            ))
        return nethras

    def build_role_maps(self, scaffold_nethras: list[ScaffoldNethra]) -> list[ScaffoldRoleMap]:
        maps: list[ScaffoldRoleMap] = []
        for n in scaffold_nethras:
            role_by_context: dict[str, set[str]] = defaultdict(set)
            for role, contexts in n.role_contexts.items():
                if contexts:
                    for context in contexts:
                        role_by_context[context].add(role)
                else:
                    role_by_context[""].add(role)
            serial_role_by_context = {
                ctx: sorted(roles)
                for ctx, roles in sorted(role_by_context.items())
            }
            examples: list[dict[str, Any]] = []
            if len(n.observed_roles) > 1:
                contexts_by_role = {
                    role: (n.role_contexts.get(role) or [""])
                    for role in n.observed_roles
                }
                roles = list(contexts_by_role)
                for left, right in zip(roles, roles[1:]):
                    examples.append({
                        "scaffold_id": n.scaffold_id,
                        "from_role": left,
                        "from_context": contexts_by_role[left][0],
                        "to_role": right,
                        "to_context": contexts_by_role[right][0],
                    })
                    if len(examples) >= 5:
                        break
            maps.append(ScaffoldRoleMap(
                scaffold_id=n.scaffold_id,
                role_by_context=serial_role_by_context,
                trass_contexts=n.role_contexts.get("trass", []),
                tareth_contexts=n.role_contexts.get("tareth", []),
                unresolved_contexts=n.role_contexts.get("unresolved", []),
                background_contexts=n.role_contexts.get("background", []),
                role_shift_examples=examples,
            ))
        return maps

    def build_compositions(
        self,
        scaffold_nethras: list[ScaffoldNethra],
        *,
        max_compositions: int = 1000,
    ) -> list[ScaffoldComposition]:
        groups: dict[tuple[Any, ...], list[ScaffoldNethra]] = defaultdict(list)
        for n in scaffold_nethras:
            for source_edges in n.source_edge_sets:
                if source_edges:
                    groups[("source_edges", tuple(source_edges))].append(n)
            for context in n.contexts:
                family = _context_family(context)
                if family and n.signatures:
                    groups[("context_signature_family", family, _operator_from_signature(n.signatures[0]))].append(n)

        compositions: list[ScaffoldComposition] = []
        seen: set[tuple[str, ...]] = set()
        for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
            unique = []
            used_ids: set[str] = set()
            for member in members:
                if member.scaffold_id not in used_ids:
                    unique.append(member)
                    used_ids.add(member.scaffold_id)
            if len(unique) < 2:
                continue
            lower_ids = tuple(sorted(n.scaffold_id for n in unique[:12]))
            if lower_ids in seen:
                continue
            seen.add(lower_ids)
            shared_vars = sorted(set.intersection(
                *(set(n.vars) for n in unique[:12] if n.vars)
            )) if all(n.vars for n in unique[:12]) else []
            shared_contexts = sorted(set.intersection(
                *(set(n.contexts) for n in unique[:12] if n.contexts)
            ))[:10] if all(n.contexts for n in unique[:12]) else []
            shared_signatures = sorted(set.intersection(
                *(set(n.signatures) for n in unique[:12] if n.signatures)
            ))[:10] if all(n.signatures for n in unique[:12]) else []
            comp_id = _stable_id("composition", (key, lower_ids))
            higher_id = _stable_id("scaffold_of_scaffolds", (key, lower_ids))
            confidence = min(1.0, 0.25 + 0.08 * len(unique) + 0.10 * len({r for n in unique for r in n.observed_roles}))
            compositions.append(ScaffoldComposition(
                composition_id=comp_id,
                lower_scaffold_ids=list(lower_ids),
                higher_scaffold_id=higher_id,
                shared_vars=shared_vars[:20],
                shared_contexts=shared_contexts,
                shared_signatures=shared_signatures,
                evidence_summary=(
                    f"{len(unique)} lower scaffold nethras share {key[0]}={key[1:]}"
                ),
                confidence_as_familiarity=confidence,
                authority_allowed=False,
            ))
            if len(compositions) >= max_compositions:
                break
        return compositions

    def build_abstractions(
        self,
        scaffold_nethras: list[ScaffoldNethra],
        compositions: list[ScaffoldComposition] | None = None,
        *,
        min_members: int = 2,
        max_abstractions: int = 2000,
    ) -> list[ScaffoldAbstraction]:
        groups: dict[tuple[str, str], list[ScaffoldNethra]] = defaultdict(list)
        for n in scaffold_nethras:
            for sig in n.signatures:
                op = _operator_from_signature(sig)
                if op:
                    groups[("operator_family", op)].append(n)
            for source_edges in n.source_edge_sets:
                if source_edges:
                    groups[("source_edge_signature_family", ",".join(str(p) for p in source_edges))].append(n)
            for role in n.observed_roles:
                if role in {"trass", "tareth", "unresolved"}:
                    groups[(f"{role}_family", role)].append(n)
                if role == "background":
                    groups[("background_object_family", "background")].append(n)
                if role in {"contested_best_available", "quarantined_for_derivation", "repair_candidate"}:
                    groups[("authority_debt_family", role)].append(n)
            if len(n.observed_roles) > 1:
                groups[("mixed_role_family", ",".join(n.observed_roles))].append(n)
            for context in n.contexts:
                family = _context_family(context)
                if family:
                    for role in n.observed_roles:
                        groups[("context_role_family", f"{family}:{role}")].append(n)
            if any("uncertainty" in t for t in n.source_types):
                groups[("uncertainty_family", "uncertainty")].append(n)

        abstractions: list[ScaffoldAbstraction] = []
        seen_ids: set[str] = set()
        for (kind, value), members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
            unique: list[ScaffoldNethra] = []
            used_ids: set[str] = set()
            for member in members:
                if member.scaffold_id not in used_ids:
                    unique.append(member)
                    used_ids.add(member.scaffold_id)
            if len(unique) < min_members:
                continue
            broad_debt_only = (
                kind == "authority_debt_family"
                and all(
                    n.specificity_score == 0.0
                    and not n.signatures
                    and not n.source_edge_sets
                    for n in unique
                )
            )
            if broad_debt_only:
                continue
            abstraction_id = _stable_id("abstraction", (kind, value, sorted(used_ids)))
            if abstraction_id in seen_ids:
                continue
            seen_ids.add(abstraction_id)
            role_distribution = Counter(role for n in unique for role in n.observed_roles)
            specificity = sum(n.specificity_score for n in unique) / max(1, len(unique))
            familiarity = sum(n.familiarity_score for n in unique) / max(1, len(unique))
            warnings: list[str] = []
            suggested: SuggestedRuntimeUse = "feature_only"
            if kind == "authority_debt_family":
                warnings.append("authority_debt_is_role_pressure_only")
                suggested = "no_runtime_use"
            elif kind in {"operator_family", "source_edge_signature_family"} and specificity >= 0.45:
                suggested = "clustering_prior"
            elif kind == "context_role_family" and familiarity >= 0.45:
                suggested = "ranking_hint"
            abstractions.append(ScaffoldAbstraction(
                abstraction_id=abstraction_id,
                kind=kind,  # type: ignore[arg-type]
                member_scaffold_ids=sorted(used_ids)[:100],
                common_structure={"group": value, "members": len(unique)},
                role_distribution=dict(role_distribution),
                specificity_score=specificity,
                familiarity_score=familiarity,
                suggested_runtime_use=suggested,
                authority_allowed=False,
                warnings=warnings,
            ))
            if len(abstractions) >= max_abstractions:
                break
        return abstractions

    def summarize(
        self,
        rows: list[dict[str, Any]],
        scaffold_nethras: list[ScaffoldNethra],
        compositions: list[ScaffoldComposition],
        abstractions: list[ScaffoldAbstraction],
        role_maps: list[ScaffoldRoleMap],
    ) -> NethraScaffoldSleepSummary:
        authority_allowed_count = (
            sum(1 for n in scaffold_nethras if n.authority_allowed)
            + sum(1 for c in compositions if c.authority_allowed)
            + sum(1 for a in abstractions if a.authority_allowed)
        )
        broad_useful = sum(
            1
            for a in abstractions
            if a.kind == "authority_debt_family"
            and a.suggested_runtime_use != "no_runtime_use"
            and "authority_debt_is_role_pressure_only" not in a.warnings
        )
        role_shift_examples = [
            example
            for role_map in role_maps
            for example in role_map.role_shift_examples
        ][:10]
        composition_examples = [
            c.to_dict()
            for c in sorted(compositions, key=lambda item: -item.confidence_as_familiarity)[:10]
        ]
        return NethraScaffoldSleepSummary(
            raw_records_read=self.raw_records_read,
            scaffold_nethras=len(scaffold_nethras),
            compositions=len(compositions),
            abstractions=len(abstractions),
            role_maps=len(role_maps),
            tareth_records_seen=self.role_records_seen.get("tareth", 0),
            trass_records_seen=self.role_records_seen.get("trass", 0),
            background_records_seen=self.role_records_seen.get("background", 0),
            unresolved_records_seen=self.role_records_seen.get("unresolved", 0),
            authority_debt_records_seen=sum(
                self.role_records_seen.get(role, 0)
                for role in (
                    "contested_best_available",
                    "quarantined_for_derivation",
                    "repair_candidate",
                )
            ),
            stable_best_available_records_seen=self.role_records_seen.get("best_available", 0),
            broad_generic_debt_count=self.broad_generic_debt_count,
            broad_generic_debt_useful_count=broad_useful,
            authority_allowed_count=authority_allowed_count,
            behavior_effects=0,
            hidden_truth_fields_seen=sorted(self.hidden_truth_fields_seen),
            role_shift_examples=role_shift_examples,
            composition_examples=composition_examples,
            abstraction_counts_by_kind=dict(Counter(a.kind for a in abstractions)),
        )


def write_scaffold_sleep_jsonl(
    path: str | Path,
    scaffold_nethras: list[ScaffoldNethra],
    role_maps: list[ScaffoldRoleMap],
    compositions: list[ScaffoldComposition],
    abstractions: list[ScaffoldAbstraction],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for item in scaffold_nethras:
            fh.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
        for item in role_maps:
            fh.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
        for item in compositions:
            fh.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
        for item in abstractions:
            fh.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
