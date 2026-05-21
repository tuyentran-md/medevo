from __future__ import annotations

from app.models import Study
from app.synthesis import synthesize_guideline_claim


def _supporting_study(index: int, *, quality: float, provenance: str = "GROUNDED") -> Study:
    return Study(
        id=f"s-{index}",
        claim_id="claim-1",
        year=2020,
        direction="SUPPORTS",
        effect_point=0.72,
        effect_ci=(0.61, 0.84),
        n=500,
        quality=quality,
        provenance=provenance,
        pmids=[str(index)] if provenance == "GROUNDED" else [],
        numeric=True,
        rationale="Supports the claim.",
        output_hash=f"h-{index}",
    )


def test_synthesis_outputs_direction_and_grade_level() -> None:
    result = synthesize_guideline_claim(
        claim_id="claim-1",
        year=2025,
        studies=[_supporting_study(index, quality=0.95) for index in range(6)],
    )

    assert result.direction == "SUPPORTS"
    assert result.level == "strong-for"
    assert result.study_count == 6
    assert result.certainty >= 0.72


def test_level_moves_when_direction_is_held_fixed() -> None:
    # A single appraisable study (has a cited source, quality 0.6 clears the
    # inclusion + study-limitations floors). As a single study the GRADE
    # sparse-evidence cap holds it to a conditional level — exercising
    # level-tracks-certainty with direction held fixed. (Screening is provenance
    # BLIND: it excludes a study with no cited source to appraise against, not a
    # study by its GROUNDED/UNGROUNDED label — so the weak arm must carry a cite.)
    weak = synthesize_guideline_claim(
        claim_id="claim-1",
        year=2025,
        studies=[_supporting_study(1, quality=0.6, provenance="GROUNDED")],
    )
    strong = synthesize_guideline_claim(
        claim_id="claim-1",
        year=2025,
        studies=[_supporting_study(index, quality=0.95) for index in range(6)],
    )

    assert weak.direction == strong.direction == "SUPPORTS"
    assert weak.level != strong.level
    assert weak.level == "conditional-for"
    assert strong.level == "strong-for"


def test_conflicting_evidence_reduces_certainty_independently_of_count() -> None:
    studies = [_supporting_study(index, quality=0.9) for index in range(4)]
    studies.append(
        Study(
            id="refuting",
            claim_id="claim-1",
            year=2020,
            direction="REFUTES",
            effect_point=1.4,
            effect_ci=(1.1, 1.8),
            n=500,
            quality=0.9,
            provenance="GROUNDED",
            pmids=["999"],
            numeric=True,
            rationale="Refutes the claim.",
            output_hash="h-refuting",
        )
    )

    result = synthesize_guideline_claim(claim_id="claim-1", year=2025, studies=studies)

    assert result.direction == "SUPPORTS"
    assert result.heterogeneity > 0
    assert result.level == "conditional-for"
