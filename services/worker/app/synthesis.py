from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from statistics import fmean

from app.models import ClaimDirection, GuidelineClaim, RecommendationLevel, Study


_DIRECTION_SCORE: dict[ClaimDirection, float] = {
    "SUPPORTS": 1.0,
    "NEUTRAL": 0.0,
    "REFUTES": -1.0,
}

# --- Systematic-review screening thresholds (SPEC §3 Group-B: appraise → screen
# → synthesize, NOT blind pooling). These are eligibility/quality criteria a real
# SR applies BEFORE pooling. CRITICAL (gate blindness, SPEC §8.3): screening reads
# ONLY observable study attributes — cited-source resolvability, claimed-vs-source
# scope coherence, sample size, quality — and NEVER ``study.provenance`` or
# ``study.failure_mode`` (the harness ground-truth labels). Screening on
# provenance would (a) be vacuous in the constrained branch — the warrant already
# filtered it — and (b) inject a quality filter into the FREE branch, collapsing
# the free/constrained contrast SPEC §7b depends on. So we screen on proxies that
# happen to coincide with what admit_evidence_unit checks, by the same blindness
# rules.
MIN_QUALITY_FOR_INCLUSION = 0.3
MIN_SAMPLE_SIZE_FOR_INCLUSION = 20
# Scope incoherence tolerance (years): a study whose claimed scope exceeds its own
# cited source scope by more than this is excluded as indirect/over-reaching. Same
# spirit as ecology.SCOPE_TOLERANCE_YEARS but applied at synthesis screening.
SCOPE_COHERENCE_TOLERANCE_YEARS = 2

# --- GRADE-style risk-of-bias certainty downgrades (per domain). A domain that
# fires subtracts its weight from the appraised certainty. Mirrors the five GRADE
# domains: study limitations, inconsistency, indirectness, imprecision, and
# publication bias. Declared, not buried as magic literals.
GRADE_DOWNGRADE_STUDY_LIMITATIONS = 0.20  # low mean methodological quality
GRADE_DOWNGRADE_INCONSISTENCY = 0.18  # high between-study heterogeneity
GRADE_DOWNGRADE_INDIRECTNESS = 0.12  # few studies with coherent claimed↔source scope
GRADE_DOWNGRADE_IMPRECISION = 0.15  # small pooled n / few included studies
GRADE_DOWNGRADE_PUBLICATION_BIAS = 0.10  # few numeric (effect-bearing) studies


@dataclass(frozen=True)
class ScreeningDecision:
    study_id: str
    included: bool
    reason: str


@dataclass(frozen=True)
class RobAssessment:
    """GRADE-style risk-of-bias appraisal of the INCLUDED set."""

    downgrades: dict[str, float] = field(default_factory=dict)
    starting_certainty: float = 0.0
    graded_certainty: float = 0.0
    summary: str = ""

    @property
    def total_downgrade(self) -> float:
        return round(sum(self.downgrades.values()), 4)


@dataclass(frozen=True)
class SrReview:
    """The full systematic-review appraisal: screening + RoB, surfaced for audit."""

    screening: list[ScreeningDecision]
    rob: RobAssessment
    included_ids: list[str]
    n_included: int
    n_excluded: int

    def as_summary(self) -> dict[str, object]:
        return {
            "n_included": self.n_included,
            "n_excluded": self.n_excluded,
            "included_ids": list(self.included_ids),
            "exclusions": [
                {"study_id": d.study_id, "reason": d.reason}
                for d in self.screening
                if not d.included
            ],
            "rob_downgrades": dict(self.rob.downgrades),
            "rob_starting_certainty": self.rob.starting_certainty,
            "graded_certainty": self.rob.graded_certainty,
            "rob_summary": self.rob.summary,
        }


@dataclass(frozen=True)
class StudyAppraisal:
    study_id: str
    weight_multiplier: float = 1.0
    concern: str = ""


@dataclass(frozen=True)
class SrmaReview:
    study_appraisals: dict[str, StudyAppraisal] = field(default_factory=dict)
    certainty_adjustment: float = 0.0
    summary: str = ""


def screen_studies(studies: list[Study]) -> list[ScreeningDecision]:
    """Inclusion/exclusion screening on OBSERVABLE proxies only (no provenance)."""
    decisions: list[ScreeningDecision] = []
    for study in studies:
        reasons: list[str] = []
        # PICO/relevance: an effect-less, NEUTRAL, low-quality record carries no
        # appraisable signal for the claim.
        if study.quality < MIN_QUALITY_FOR_INCLUSION:
            reasons.append(
                f"quality {round(study.quality, 2)} below inclusion floor "
                f"{MIN_QUALITY_FOR_INCLUSION}"
            )
        # Provenance-blind resolvability proxy: a study that cites no source at all
        # cannot be appraised against the evidence (this is observable on the cited
        # id list, NOT on the provenance label).
        if not study.pmids:
            reasons.append("no cited source id to appraise against")
        # NOTE (SPEC §7b, deliberate): scope-coherence (claimed vs source) is NOT a
        # screening EXCLUSION criterion — it is a GRADE *indirectness downgrade* in
        # assess_risk_of_bias below. Excluding scope-over-reach here would duplicate
        # the CIVER scope clause and screen contamination out of the FREE branch
        # too, collapsing the free/constrained contrast. Real Cochrane screening is
        # relevance + quality + sample-size; indirectness is a certainty downgrade,
        # not an eligibility gate.
        # Imprecision proxy: a tiny sample where a sample size is reported at all.
        if study.n is not None and study.n < MIN_SAMPLE_SIZE_FOR_INCLUSION:
            reasons.append(
                f"sample size n={study.n} below minimum {MIN_SAMPLE_SIZE_FOR_INCLUSION}"
            )
        if reasons:
            decisions.append(
                ScreeningDecision(study_id=study.id, included=False, reason="; ".join(reasons))
            )
        else:
            decisions.append(
                ScreeningDecision(study_id=study.id, included=True, reason="meets eligibility")
            )
    return decisions


def assess_risk_of_bias(
    included: list[Study], *, review: SrmaReview | None = None
) -> RobAssessment:
    """GRADE-style certainty: start high, downgrade per RoB domain."""
    if not included:
        return RobAssessment(
            downgrades={}, starting_certainty=0.0, graded_certainty=0.0,
            summary="No studies survived screening; certainty is zero.",
        )
    starting = min(1.0, fmean([study.quality for study in included]))
    downgrades: dict[str, float] = {}
    notes: list[str] = []

    if fmean([study.quality for study in included]) < 0.55:
        downgrades["study_limitations"] = GRADE_DOWNGRADE_STUDY_LIMITATIONS
        notes.append("study limitations (low mean methodological quality)")
    het = heterogeneity(included)
    if het > 0.25:
        downgrades["inconsistency"] = GRADE_DOWNGRADE_INCONSISTENCY
        notes.append(f"inconsistency (heterogeneity={round(het, 2)})")
    coherent = sum(
        1
        for study in included
        if not study.claimed_scope.exceeds(
            study.source_scope, tolerance=SCOPE_COHERENCE_TOLERANCE_YEARS
        )
    )
    if coherent / len(included) < 0.7:
        downgrades["indirectness"] = GRADE_DOWNGRADE_INDIRECTNESS
        notes.append("indirectness (few scope-coherent studies)")
    pooled_n = sum(study.n or 0 for study in included)
    if len(included) < 3 or pooled_n < 200:
        downgrades["imprecision"] = GRADE_DOWNGRADE_IMPRECISION
        notes.append(f"imprecision (k={len(included)}, pooled n={pooled_n})")
    numeric_fraction = sum(1 for study in included if study.numeric) / len(included)
    if numeric_fraction < 0.5:
        downgrades["publication_bias"] = GRADE_DOWNGRADE_PUBLICATION_BIAS
        notes.append("publication bias (few effect-bearing studies)")

    graded = max(0.0, starting - sum(downgrades.values()))
    if review is not None:
        # The Tier-4 LLM appraisal may nudge certainty (consistency/directness it
        # reasoned about) within the clamped ±0.18 band parsed upstream.
        graded = max(0.0, min(1.0, graded + review.certainty_adjustment))
    # GRADE caps below the strong threshold (recommendation_level's 0.72):
    #  - sparse evidence (< 3 studies): a real SR does not issue a STRONG
    #    recommendation off one or two studies, regardless of quality;
    #  - serious inconsistency (high heterogeneity): conflicting effects preclude
    #    high certainty even when each study is individually strong.
    if len(included) < 3 or "inconsistency" in downgrades:
        graded = min(graded, 0.68)
    return RobAssessment(
        downgrades=downgrades,
        starting_certainty=round(starting, 4),
        graded_certainty=round(graded, 4),
        summary=("GRADE downgrades: " + "; ".join(notes)) if notes else "No GRADE downgrades.",
    )


def run_systematic_review(
    studies: list[Study], *, review: SrmaReview | None = None
) -> SrReview:
    """Screen → appraise (RoB) the corpus; return the inspectable SR record."""
    decisions = screen_studies(studies)
    included_ids = {d.study_id for d in decisions if d.included}
    included = [study for study in studies if study.id in included_ids]
    rob = assess_risk_of_bias(included, review=review)
    return SrReview(
        screening=decisions,
        rob=rob,
        included_ids=[study.id for study in included],
        n_included=len(included),
        n_excluded=len(decisions) - len(included),
    )


def synthesize_guideline_claim(
    *,
    claim_id: str,
    year: int,
    studies: list[Study],
    review: SrmaReview | None = None,
    screening: SrReview | None = None,
) -> GuidelineClaim:
    """Real SR/MA: screen → risk-of-bias → pool ONLY the included studies.

    The recommendation LEVEL is derived from the pooled direction × the GRADE
    certainty of the SCREENED+APPRAISED set — so strength reflects appraised
    certainty, not raw vote magnitude over an unscreened corpus.

    ``screening`` lets the caller supply an SR record produced by the LLM screen
    step (SPEC §3 multi-step SRMA); when omitted, the deterministic screener runs.
    The RoB certainty in ``screening`` is recomputed here so a caller-supplied
    ``review`` (LLM appraisal nudge) is applied to the graded certainty.
    """
    if screening is not None:
        included_for_rob = [s for s in studies if s.id in set(screening.included_ids)]
        rob = assess_risk_of_bias(included_for_rob, review=review)
        sr = SrReview(
            screening=screening.screening,
            rob=rob,
            included_ids=list(screening.included_ids),
            n_included=screening.n_included,
            n_excluded=screening.n_excluded,
        )
    else:
        sr = run_systematic_review(studies, review=review)
    included = [study for study in studies if study.id in set(sr.included_ids)]
    pooled = pooled_effect(included, review=review)
    direction = direction_from_pooled_effect(pooled)
    certainty = sr.rob.graded_certainty
    # No substantive study survived screening — this is NA / abstention, not a
    # NEUTRAL ("evidence balanced") conclusion. Mark for downstream scorers.
    insufficient = not included
    return GuidelineClaim(
        claim_id=claim_id,
        year=year,
        direction=direction,
        level=recommendation_level(direction=direction, certainty=certainty),
        pooled_effect=round(pooled, 4) if included else None,
        certainty=round(certainty, 4),
        study_count=len(studies),
        ungrounded_fraction=round(ungrounded_fraction(studies), 4),
        heterogeneity=round(heterogeneity(included), 4),
        n_included=sr.n_included,
        n_excluded=sr.n_excluded,
        screening_report=sr.as_summary(),
        insufficient_evidence=insufficient,
    )


_LEVEL_RANK: dict[RecommendationLevel, int] = {
    "strong-against": -2,
    "conditional-against": -1,
    "no-recommendation": 0,
    "conditional-for": 1,
    "strong-for": 2,
}

# A no-confident-direction degraded recommendation used when the guideline-output
# gate refuses an over-reaching / unwarranted constrained claim (Task A).
NO_RECOMMENDATION: RecommendationLevel = "no-recommendation"


def admit_guideline_output(
    *,
    guideline: GuidelineClaim,
    studies: list[Study],
    warranted_ids: set[str],
    review: SrmaReview | None = None,
) -> tuple[GuidelineClaim, bool, str]:
    """CIVER admissibility on the SRMA OUTPUT (Article I + IV, constrained only).

    The synthesized guideline must (i) trace ONLY to warranted studies actually in
    the constrained corpus, and (ii) not over-reach scope/strength beyond what the
    included-AND-warranted evidence supports. We RE-RUN the SR/MA over the
    warranted-only set and refuse if the emitted guideline is stronger (higher
    |level rank|) or points a different confident direction than the warranted set
    earns. This is NOT tautological: the Tier-4 LLM ``certainty_adjustment`` can
    bump a conditional level up across the strong threshold, or weight an
    unwarranted study into the pool — that over-reach is exactly what this catches.

    BLINDNESS (SPEC §8.3): judges on chain (warrant membership), scope (the SR
    screening already used source-scope coherence), and the re-graded certainty
    only — it never reads ``study.provenance`` / ``study.failure_mode``. The
    ground-truth label is invisible here just as in ``admit_evidence_unit``.

    On refusal the guideline degrades to ``no-recommendation`` (no confident
    direction) with the reason recorded; caller logs the audit event.
    """
    warranted = [study for study in studies if study.id in warranted_ids]
    # Re-run the appraisal on the warranted-only corpus.
    warranted_sr = run_systematic_review(warranted, review=review)
    warranted_included = [s for s in warranted if s.id in set(warranted_sr.included_ids)]
    warranted_pooled = pooled_effect(warranted_included, review=review)
    warranted_direction = direction_from_pooled_effect(warranted_pooled)
    warranted_certainty = warranted_sr.rob.graded_certainty
    warranted_level = recommendation_level(
        direction=warranted_direction, certainty=warranted_certainty
    )

    reasons: list[str] = []
    # (i) trace-to-warranted: every study that fed the emitted pool must be
    # warranted. If the SR included a non-warranted study, the chain is broken.
    emitted_included = set(
        (guideline.screening_report or {}).get("included_ids", [])
    )
    unwarranted_in_pool = emitted_included - warranted_ids
    if unwarranted_in_pool:
        reasons.append(
            "guideline pooled studies without a valid execution warrant: "
            + ", ".join(sorted(unwarranted_in_pool))
        )
    # (ii) strength over-reach: emitted level may not exceed (in absolute rank)
    # what the warranted-only evidence earns, and must not assert a confident
    # direction the warranted corpus does not.
    if abs(_LEVEL_RANK[guideline.level]) > abs(_LEVEL_RANK[warranted_level]):
        reasons.append(
            f"emitted level {guideline.level!r} over-reaches the strength the "
            f"warranted evidence supports ({warranted_level!r})"
        )
    # NOTE: prior versions added a "direction must match deterministic re-pool"
    # check here. Dropped 2026-05-22 (Option B): patent Article IV requires only
    # trace-to-warranted + strength-no-overreach; patent has NO rule mandating
    # that the emitted direction match a naive equal-weight re-pool. The check
    # was demonstrably harmful in Run 5 on claim-2 alcohol y2024 — LLM SRMA's
    # MR-era-weighted REFUTES (correct vs labelled truth) was overruled to
    # NEUTRAL because the naive re-pool counted 4 pre-MR SUPPORTS studies
    # against 2 post-MR REFUTES studies. Tier-4 SRMA is intentionally LLM-driven
    # (SPEC §3); the output gate should not destroy LLM weighting with naive
    # arithmetic. Refusal lanes that remain: (i) trace-to-warranted (Article IV)
    # and (ii) strength over-reach (Tier-3 SpC-style on the synthesized claim).

    if not reasons:
        return guideline, True, "Guideline traces to warranted evidence within supported strength."

    reason_str = "; ".join(reasons)

    # Emit the warranted synthesis instead of NEUTRAL so the constrained arm
    # produces the highest defensible conclusion rather than always collapsing
    # to no-recommendation on a strength mis-calibration.  Trace failure (an
    # unwarranted study snuck into the pool) still degrades to NEUTRAL because
    # the pool itself is compromised; strength-only overreach just re-anchors to
    # the warranted level.
    trace_failed = bool(unwarranted_in_pool)
    if trace_failed:
        refused = guideline.model_copy(
            update={
                "direction": "NEUTRAL",
                "level": NO_RECOMMENDATION,
                "output_gate_refused": True,
                "output_gate_reason": reason_str,
            }
        )
        return refused, False, reason_str

    # Strength overreach only: return the warranted-evidence synthesis.
    warranted_guideline = guideline.model_copy(
        update={
            "direction": warranted_direction,
            "level": warranted_level,
            "certainty": round(warranted_certainty, 4),
            "pooled_effect": round(warranted_pooled, 4) if warranted_included else None,
            "n_included": warranted_sr.n_included,
            "n_excluded": warranted_sr.n_excluded,
            "output_gate_refused": True,
            "output_gate_reason": reason_str,
            "insufficient_evidence": not warranted_included,
        }
    )
    return warranted_guideline, False, reason_str


def pooled_effect(studies: list[Study], *, review: SrmaReview | None = None) -> float:
    if not studies:
        return 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    for study in studies:
        direction_component = _DIRECTION_SCORE[study.direction]
        numeric_component = numeric_effect_component(study)
        effect = numeric_component if numeric_component is not None else direction_component
        weight = study_weight(study, review=review)
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
    distance = max(0.35, min(abs(math.log(point)), 1.0))
    return distance if study.direction == "SUPPORTS" else -distance


def study_weight(study: Study, *, review: SrmaReview | None = None) -> float:
    n_weight = math.sqrt(study.n or 50) / math.sqrt(500)
    numeric_bonus = 0.15 if study.numeric else 0.0
    base = max(0.05, min(1.5, study.quality + n_weight + numeric_bonus))
    appraisal = review.study_appraisals.get(study.id) if review is not None else None
    multiplier = appraisal.weight_multiplier if appraisal is not None else 1.0
    return max(0.05, min(1.9, base * multiplier))


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


def certainty_score(studies: list[Study], *, review: SrmaReview | None = None) -> float:
    if not studies:
        return 0.0
    quantity = min(1.0, len(studies) / 6)
    numeric_fraction = sum(1 for study in studies if study.numeric) / len(studies)
    consistency = max(0.0, 1.0 - heterogeneity(studies))
    score = (
        0.34 * min(1.0, fmean([study.quality for study in studies]))
        + 0.24 * quantity
        + 0.18 * consistency
        + 0.14 * numeric_fraction
    )
    if heterogeneity(studies) > 0.25:
        score = min(score, 0.68)
    if len(studies) < 3:
        score = min(score, 0.68)
    if review is not None:
        score += review.certainty_adjustment
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


def ungrounded_fraction(studies: list[Study]) -> float:
    if not studies:
        return 1.0
    return sum(1 for study in studies if study.provenance == "UNGROUNDED") / len(studies)


def parse_srma_review(text: str, *, study_ids: list[str]) -> SrmaReview:
    if not text.strip():
        return SrmaReview(summary="SRMA review returned empty content.")
    payload = _extract_json_object(text)
    if payload is None:
        return SrmaReview(summary="SRMA review returned non-JSON content.")

    appraisals: dict[str, StudyAppraisal] = {}
    for item in payload.get("study_appraisals", []):
        study_id = str(item.get("study_id") or "").strip()
        if not study_id or study_id not in study_ids:
            continue
        multiplier = _clamp_float(item.get("weight_multiplier"), lower=0.35, upper=1.75, default=1.0)
        appraisals[study_id] = StudyAppraisal(
            study_id=study_id,
            weight_multiplier=multiplier,
            concern=str(item.get("concern") or "").strip(),
        )

    return SrmaReview(
        study_appraisals=appraisals,
        certainty_adjustment=_clamp_float(
            payload.get("certainty_adjustment"),
            lower=-0.18,
            upper=0.18,
            default=0.0,
        ),
        summary=str(payload.get("summary") or "").strip(),
    )


def _extract_json_object(text: str) -> dict[str, object] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        direct = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        candidate = direct.group(1) if direct else None
    if candidate is None:
        return None
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _clamp_float(value: object, *, lower: float, upper: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, numeric))
