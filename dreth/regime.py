from __future__ import annotations

# ── THIS FILE ────────────────────────────────────────────────────────────────
# Regime detection layer — nethra-of-nethra foundation.
#
# A "regime" in Dreth is not a world-state cluster. It is a recurring pattern
# of cert behavior: the same set of learned authorities failing, repairing, or
# surviving together. Two cycles are the "same regime" when their co-failure
# pattern is similar enough, regardless of what the underlying world state is.
#
# Objects:
#   CertEvent      — one cert's stress/fail/repair at a specific cycle
#   RegimeSignature — a confirmed recurring co-failure pattern with authority
#   RegimeRegister  — collects CertEvents per cycle, matches against known
#                     patterns, promotes candidate patterns to confirmed regimes
#
# Matching: weighted Jaccard over (var, cert_key, event_type) triples.
# Weight = cert maturity (full_audits / 10, capped at 1.0) so established
# certs failing count more than freshly-issued ones.
#
# Bootstrap: first occurrence → candidate. Second matching occurrence →
# confirmed regime (authority=2). Subsequent matches increment authority.
# Candidates older than candidate_max_age cycles are pruned as noise.
#
# Observable fingerprint (value distributions, delta stats) is stored with
# each confirmed regime as supporting evidence, but is NOT used for matching.
# The cert behavior pattern is primary; world state is secondary.
# ─────────────────────────────────────────────────────────────────────────────

import dataclasses
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass
class CertEvent:
    """One cert's stress/fail/repair event at a specific cycle.

    event_type values:
      "failed"   — sentinel check returned False; cert is being invalidated
      "repaired" — re-audit ran after failure; fit was reinstalled (may or
                   may not have changed — see repair_shape)
      "stressed" — sentinel returned True but logged TEMPORAL_TRASS; the cert
                   survived but showed OOB deviation

    repair_shape values (only meaningful when event_type == "repaired"):
      "stable"        — re-audit returned the same fit; no structural change
      "parent_change" — re-audit returned different parent set
      "func_change"   — re-audit returned different function, same parents
      "full_change"   — both parents and function changed

    cert_age: n.full_audits at time of event. Proxy for cert maturity.
    High cert_age = established cert; its failure is a stronger regime signal
    than a newly-issued cert failing.
    """
    var: int
    cert_key: str
    event_type: str
    delta: float = 0.0
    repair_shape: str = ""
    cert_age: int = 0


@dataclasses.dataclass
class RegimeSignature:
    """A confirmed recurring co-failure pattern.

    authority counts how many distinct cycle-windows matched this pattern.
    events is a merged representative event set (highest cert_age wins on
    collision).
    observable_fingerprint stores world-state clues at confirmation time
    (mean delta, n_failed, etc.) as supporting evidence, not as the identity.
    """
    regime_id: int
    first_seen_cycle: int
    last_seen_cycle: int
    authority: int
    events: List[CertEvent]
    observable_fingerprint: Dict[str, float]


class RegimeRegister:
    """Collects per-cycle CertEvents, clusters them into recurring patterns,
    and maintains a registry of confirmed regimes.

    Matching uses weighted Jaccard over (var, cert_key, event_type) triples.
    Weight = min(cert_age / 10, 1.0) so mature certs dominate the similarity.

    similarity_threshold: minimum Jaccard to count as same pattern (default 0.5)
    candidate_max_age:    cycles before an unmatched candidate is discarded
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        candidate_max_age: int = 1000,
    ):
        self._threshold = similarity_threshold
        self._candidate_max_age = candidate_max_age
        self._confirmed: List[RegimeSignature] = []
        self._candidates: List[Tuple[int, List[CertEvent]]] = []

    # ── public ───────────────────────────────────────────────────────────────

    def observe(
        self,
        events: List[CertEvent],
        cycle: int,
        fingerprint: Dict[str, float],
    ) -> Optional[int]:
        """Record a set of co-occurring cert events and return the matching
        regime_id if a known pattern was recognized, else None.

        Side effects:
          - Prunes stale candidates
          - Promotes a candidate to confirmed if a second matching event appears
          - Increments authority on a confirmed regime if it matches
          - Stores new unmatched events as a candidate
        """
        if not events:
            return None

        # Prune stale candidates
        self._candidates = [
            (c, evts) for c, evts in self._candidates
            if cycle - c <= self._candidate_max_age
        ]

        # Check against confirmed regimes
        if self._confirmed:
            best_ci, best_csim = self._best_match(events, [s.events for s in self._confirmed])
            if best_csim >= self._threshold:
                sig = self._confirmed[best_ci]
                sig.authority += 1
                sig.last_seen_cycle = cycle
                sig.events = _merge_events(sig.events, events)
                return sig.regime_id

        # Check against candidates
        if self._candidates:
            best_ki, best_ksim = self._best_match(events, [evts for _, evts in self._candidates])
            if best_ksim >= self._threshold:
                cand_cycle, cand_events = self._candidates.pop(best_ki)
                regime_id = len(self._confirmed)
                sig = RegimeSignature(
                    regime_id=regime_id,
                    first_seen_cycle=cand_cycle,
                    last_seen_cycle=cycle,
                    authority=2,
                    events=_merge_events(cand_events, events),
                    observable_fingerprint=fingerprint,
                )
                self._confirmed.append(sig)
                return regime_id

        # No match — store as new candidate
        self._candidates.append((cycle, list(events)))
        return None

    def summary(self) -> str:
        """One-line per confirmed regime for final_summary output."""
        if not self._confirmed and not self._candidates:
            return "  regime_register: no data"
        parts = [
            f"  regime_register: {len(self._confirmed)} confirmed "
            f"({len(self._candidates)} candidates pending)"
        ]
        for sig in self._confirmed:
            failed = sorted({e.var for e in sig.events if e.event_type == "failed"})
            repaired = sorted({e.var for e in sig.events if e.event_type == "repaired"})
            failed_str = f"[{','.join(f'x{v}' for v in failed[:5])}{'…' if len(failed)>5 else ''}]"
            parts.append(
                f"    R{sig.regime_id}: auth={sig.authority:3d} "
                f"c{sig.first_seen_cycle}–{sig.last_seen_cycle} "
                f"failed={failed_str} repaired={len(repaired)}"
            )
        return "\n".join(parts)

    # ── internal ─────────────────────────────────────────────────────────────

    def _best_match(
        self,
        events: List[CertEvent],
        pool: List[List[CertEvent]],
    ) -> Tuple[int, float]:
        sims = [_jaccard(events, p) for p in pool]
        best_i = max(range(len(sims)), key=lambda i: sims[i])
        return best_i, sims[best_i]


# ── module-level helpers ──────────────────────────────────────────────────────

def _weight(e: CertEvent) -> float:
    """Cert maturity weight: established certs dominate similarity scores."""
    return max(min(e.cert_age / 10.0, 1.0), 0.1)


def _jaccard(a: List[CertEvent], b: List[CertEvent]) -> float:
    """Weighted Jaccard over (var, cert_key, event_type) triples."""
    wa: Dict[Tuple, float] = {}
    for e in a:
        k = (e.var, e.cert_key, e.event_type)
        wa[k] = max(wa.get(k, 0.0), _weight(e))
    wb: Dict[Tuple, float] = {}
    for e in b:
        k = (e.var, e.cert_key, e.event_type)
        wb[k] = max(wb.get(k, 0.0), _weight(e))
    all_keys = set(wa) | set(wb)
    if not all_keys:
        return 0.0
    intersection = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in all_keys)
    union = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in all_keys)
    return intersection / union if union > 0 else 0.0


def _merge_events(a: List[CertEvent], b: List[CertEvent]) -> List[CertEvent]:
    """Union of two event lists; on collision, keep the higher cert_age entry."""
    merged: Dict[Tuple, CertEvent] = {}
    for e in a:
        k = (e.var, e.cert_key, e.event_type)
        merged[k] = e
    for e in b:
        k = (e.var, e.cert_key, e.event_type)
        if k not in merged or e.cert_age > merged[k].cert_age:
            merged[k] = e
    return list(merged.values())
