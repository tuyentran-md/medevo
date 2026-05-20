from __future__ import annotations

import math
from statistics import fmean

from app.models import ClaimDirection, GuidelineClaim, RecommendationLevel, Study


_DIRECTION_SCORE: dict[ClaimDirection, float] = {
    "SUPPORTS": 1.0,
    "NEUTRAL": 0.0,
    "REFUTES": -1.0,
}


def synthesize_guideline_claim(
    *,
    claim_id: str,
    year: int,
    studies: list[Study],
) -> GuidelineClaim:
    pooled = pooled_effect(studies)
    direction = direction_from_pooled_effect(pooled)
    certainty = certainty_score(studies)
    return GuidelineClaim(
        claim_id=claim_id,
        year=year,
        direction=direction,
        level=recommendation_level(direction=direction, certainty=certainty),
        pooled_effect=round(pooled, 4) if studies else None,
        certainty=round(certainty, 4),
        study_count=len(studies),
        synthetic_fraction=round(synthetic_fraction(studies), 4),
        heterogeneity=round(heterogeneity(studies), 4),
    )


def pooled_effect(studies: list[Study]) -> float:
    if not studies:
        return 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    for study in studies:
        direction_component = _DIRECTION_SCORE[study.direction]
        numeric_component = numeric_effect_component(study)
        effect = numeric_component if numeric_component is not None else direction_component
        weight = study_weight(study)
        weighted_sum += effect * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def numeric_effect_component(study: Study) -> float | None:
    if study.effect_point is None:
        return None
    point = study.effect_point
    if point <= 0:
        return None
    if study.direction == "NEUTRAL":
        return 0.0
    # Ratio measures: below 1 usually favours an intervention; above 1 usually
    # favours harm/no benefit. Direction disambiguates which side supports the claim.
    distance = max(0.35, min(abs(math.log(point)), 1.0))
    return distance if study.direction == "SUPPORTS" else -distance


def study_weight(study: Study) -> float:
    n_weight = math.sqrt(study.n or 50) / math.sqrt(500)
    numeric_bonus = 0.15 if study.numeric else 0.0
    provenance_penalty = 0.35 if study.provenance == "UNGROUNDED" else 0.0
    return max(0.05, min(1.5, study.quality + n_weight + numeric_bonus - provenance_penalty))


def direction_from_pooled_effect(value: float) -> ClaimDirection:
    if value >= 0.15:
        return "SUPPORTS"
    if value <= -0.15:
        return "REFUTES"
    return "NEUTRAL"


def recommendation_level(*, direction: ClaimDirection, certainty: float) -> RecommendationLevel:
    if direction == "NEUTRAL" or certainty < 0.25:
        return "no-recommendation"
    if direction == "SUPPORTS":
        return "strong-for" if certainty >= 0.72 else "conditional-for"
    return "strong-against" if certainty >= 0.72 else "conditional-against"


def certainty_score(studies: list[Study]) -> float:
    if not studies:
        return 0.0
    real_quality = sum(study.quality for study in studies if study.provenance == "GROUNDED")
    total_quality = sum(study.quality for study in studies)
    real_fraction = 0.0 if total_quality == 0 else real_quality / total_quality
    quantity = min(1.0, len(studies) / 6)
    numeric_fraction = sum(1 for study in studies if study.numeric) / len(studies)
    consistency = max(0.0, 1.0 - heterogeneity(studies))
    score = (
        0.34 * min(1.0, fmean([study.quality for study in studies]))
        + 0.24 * quantity
        + 0.18 * consistency
        + 0.14 * numeric_fraction
        + 0.10 * real_fraction
    )
    if heterogeneity(studies) > 0.25:
        score = min(score, 0.68)
    if len(studies) < 3:
        score = min(score, 0.68)
    return max(0.0, min(score, 1.0))


def heterogeneity(studies: list[Study]) -> float:
    if len(studies) <= 1:
        return 0.0
    values = [
        numeric_effect_component(study)
        if numeric_effect_component(study) is not None
        else _DIRECTION_SCORE[study.direction]
        for study in studies
    ]
    mean = fmean(values)
    variance = fmean([(value - mean) ** 2 for value in values])
    return min(1.0, math.sqrt(variance))


def synthetic_fraction(studies: list[Study]) -> float:
    if not studies:
        return 0.0
    return sum(1 for study in studies if study.provenance == "UNGROUNDED") / len(studies)
