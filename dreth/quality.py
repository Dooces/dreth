from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityWeights:
    audit_weight: int = 1000
    revocation_weight: int = 5000
    unique_fail_weight: int = 2000
    regime_fail_weight: int = 500
    no_sentinel_weight: int = 0
    no_effect_probe_weight: int = 10
    improved_probe_credit: int = -25


@dataclass(frozen=True)
class RunQualityScore:
    iv: int
    full_audits: int
    revocations: int
    unique_fails: int
    regime_sentinel_fail: int
    regime_sentinel_no_sentinel: int
    passive_saved_iv: int
    provider_probe_no_effect_count: int
    provider_probe_improved_margin_count: int
    quality_cost: int

    @property
    def total_interventions(self) -> int:
        return self.iv


def compute_quality_cost(
    *,
    iv: int,
    full_audits: int,
    revocations: int,
    unique_fails: int,
    regime_sentinel_fail: int,
    regime_sentinel_no_sentinel: int,
    provider_probe_no_effect_count: int,
    provider_probe_improved_margin_count: int,
    weights: QualityWeights = QualityWeights(),
) -> int:
    """Diagnostic-only provider policy score.

    This helper is pure arithmetic. It must not be read by agent policy.
    """
    return int(
        iv
        + weights.audit_weight * full_audits
        + weights.revocation_weight * revocations
        + weights.unique_fail_weight * unique_fails
        + weights.regime_fail_weight * regime_sentinel_fail
        + weights.no_sentinel_weight * regime_sentinel_no_sentinel
        + weights.no_effect_probe_weight * provider_probe_no_effect_count
        + weights.improved_probe_credit * provider_probe_improved_margin_count
    )


def make_quality_score(
    *,
    iv: int,
    full_audits: int,
    revocations: int,
    unique_fails: int,
    regime_sentinel_fail: int,
    regime_sentinel_no_sentinel: int,
    passive_saved_iv: int,
    provider_probe_no_effect_count: int,
    provider_probe_improved_margin_count: int,
    weights: QualityWeights = QualityWeights(),
) -> RunQualityScore:
    return RunQualityScore(
        iv=iv,
        full_audits=full_audits,
        revocations=revocations,
        unique_fails=unique_fails,
        regime_sentinel_fail=regime_sentinel_fail,
        regime_sentinel_no_sentinel=regime_sentinel_no_sentinel,
        passive_saved_iv=passive_saved_iv,
        provider_probe_no_effect_count=provider_probe_no_effect_count,
        provider_probe_improved_margin_count=provider_probe_improved_margin_count,
        quality_cost=compute_quality_cost(
            iv=iv,
            full_audits=full_audits,
            revocations=revocations,
            unique_fails=unique_fails,
            regime_sentinel_fail=regime_sentinel_fail,
            regime_sentinel_no_sentinel=regime_sentinel_no_sentinel,
            provider_probe_no_effect_count=provider_probe_no_effect_count,
            provider_probe_improved_margin_count=provider_probe_improved_margin_count,
            weights=weights,
        ),
    )
