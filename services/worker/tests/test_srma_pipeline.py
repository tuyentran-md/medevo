"""Tests for the v3 SR/MA pipeline + guideline-output CIVER gate.

Covers:
- Task A: the SRMA OUTPUT is gated on the constrained branch (Article I/IV applied
  to the synthesized guideline, not only study inputs); an over-reaching or
  unwarranted guideline is refused and degrades to no-recommendation. The gate is
  BLIND to the harness ground-truth provenance label.
- Task C: ~15-20 studies/arm produced across a run; SR screening excludes some
  studies with recorded reasons; the recommendation LEVEL reflects the appraised
  GRADE certainty, not raw vote magnitude.
"""

from __future__ import annotations

from app.llm import DeterministicFakeClient
from app.models import EvidenceScope, GuidelineClaim, Study
from app.simulator import simulate_run
from app.synthesis import (
    admit_guideline_output,
    pooled_effect,
    run_systematic_review,
    synthesize_guideline_claim,
)
from tests.test_simulator import _request


def _study(
    study_id: str,
    *,
    direction: str = "SUPPORTS",
    quality: float = 0.9,
    n: int | None = 400,
    provenance: str = "GROUNDED",
    pmids: list[str] | None = None,
    claimed: EvidenceScope | None = None,
    source: EvidenceScope | None = None,
    numeric: bool = True,
) -> Study:
    src = source or EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)
    return Study(
        id=study_id,
        claim_id="claim-1",
        year=2020,
        direction=direction,
        effect_point=0.7 if direction == "SUPPORTS" else 1.4,
        effect_ci=(0.6, 0.82) if direction == "SUPPORTS" else (1.1, 1.8),
        n=n,
        quality=quality,
        provenance=provenance,
        pmids=pmids if pmids is not None else [study_id],
        numeric=numeric,
        rationale="record",
        claimed_scope=claimed or src.model_copy(deep=True),
        source_scope=src.model_copy(deep=True),
        output_hash=f"h-{study_id}",
    )


# --- Task C: SR screening excludes studies with reasons ---------------------


def test_screening_excludes_ineligible_studies_with_reasons() -> None:
    studies = [
        _study("ok-1"),
        _study("ok-2"),
        _study("ok-3"),
        # excluded: quality below the inclusion floor.
        _study("low-quality", quality=0.1),
        # excluded: no cited source to appraise against (provenance-blind proxy).
        _study("no-cite", pmids=[]),
        # excluded: tiny sample.
        _study("tiny", n=5),
        # NOT excluded: claimed scope over-reaches its cited source. Indirectness
        # is a GRADE certainty DOWNGRADE, not a screening exclusion (SPEC §7b — see
        # screen_studies note); it stays in the pool, RoB-flagged.
        _study(
            "overreach",
            claimed=EvidenceScope(
                population_low=5, population_high=99, year_start=1990, year_end=2030
            ),
        ),
    ]
    sr = run_systematic_review(studies)

    assert sr.n_included == 4
    assert sr.n_excluded == 3
    assert "overreach" in sr.included_ids
    excluded = {d.study_id: d.reason for d in sr.screening if not d.included}
    assert "low-quality" in excluded and "inclusion floor" in excluded["low-quality"]
    assert "no-cite" in excluded and "cited source" in excluded["no-cite"]
    assert "tiny" in excluded and "sample size" in excluded["tiny"]


def test_screening_is_blind_to_provenance_label() -> None:
    """Screening must judge on observable proxies, NOT the GROUNDED/UNGROUNDED
    harness label (SPEC §8.3). Flipping ONLY provenance must not change which
    studies are included."""
    studies = [_study(f"s-{i}") for i in range(4)]
    grounded = run_systematic_review(
        [s.model_copy(update={"provenance": "GROUNDED"}) for s in studies]
    )
    ungrounded = run_systematic_review(
        [s.model_copy(update={"provenance": "UNGROUNDED"}) for s in studies]
    )
    assert grounded.included_ids == ungrounded.included_ids


def test_pooling_is_blind_to_provenance_label() -> None:
    """SR/MA weighting must not use the harness-only provenance label.

    A real guideline panel can see observable study attributes (sample size,
    quality, numeric effect, scope/cites), but not the simulator's ground-truth
    GROUNDED/UNGROUNDED label. Flipping only that label must leave the pooled
    estimate unchanged.
    """
    studies = [
        _study("support", direction="SUPPORTS", quality=0.9, n=600),
        _study("refute", direction="REFUTES", quality=0.7, n=450),
    ]
    baseline = pooled_effect(studies)
    flipped = pooled_effect(
        [
            studies[0].model_copy(update={"provenance": "UNGROUNDED"}),
            studies[1].model_copy(update={"provenance": "GROUNDED"}),
        ]
    )
    assert flipped == baseline


# --- Task C: level reflects GRADE certainty ---------------------------------


def test_level_reflects_grade_certainty_not_vote_magnitude() -> None:
    # Same unanimous SUPPORTS direction in both bodies; only the appraised
    # certainty differs (consistent high-quality large vs sparse/heterogeneous).
    strong_body = [_study(f"hi-{i}", quality=0.95, n=600) for i in range(6)]
    strong = synthesize_guideline_claim(claim_id="claim-1", year=2025, studies=strong_body)

    # Two studies only -> sparse-evidence GRADE cap -> conditional, despite a
    # large per-study vote magnitude.
    sparse_body = [_study(f"sp-{i}", quality=0.95, n=600) for i in range(2)]
    sparse = synthesize_guideline_claim(claim_id="claim-1", year=2025, studies=sparse_body)

    assert strong.direction == sparse.direction == "SUPPORTS"
    assert strong.level == "strong-for"
    assert strong.certainty >= 0.72
    assert sparse.level == "conditional-for"
    assert sparse.certainty < 0.72


# --- Task A: guideline-output gate ------------------------------------------


def _included_guideline(study_ids: list[str], *, level: str, direction: str) -> GuidelineClaim:
    return GuidelineClaim(
        claim_id="claim-1",
        year=2025,
        direction=direction,
        level=level,
        pooled_effect=0.5 if direction == "SUPPORTS" else -0.5,
        certainty=0.8,
        study_count=len(study_ids),
        n_included=len(study_ids),
        n_excluded=0,
        screening_report={"included_ids": list(study_ids)},
    )


def test_output_gate_refuses_guideline_pooling_unwarranted_study() -> None:
    """A guideline whose pool includes a study WITHOUT a valid warrant breaks the
    Article-IV trace-to-warranted requirement -> refused, degraded to
    no-recommendation."""
    studies = [_study("w-1"), _study("w-2"), _study("w-3"), _study("rogue")]
    warranted = {"w-1", "w-2", "w-3"}
    guideline = _included_guideline(
        ["w-1", "w-2", "w-3", "rogue"], level="strong-for", direction="SUPPORTS"
    )

    gated, admitted, reason = admit_guideline_output(
        guideline=guideline, studies=studies, warranted_ids=warranted
    )
    assert admitted is False
    assert gated.output_gate_refused is True
    assert gated.level == "no-recommendation"
    assert gated.direction == "NEUTRAL"
    assert "without a valid execution warrant" in reason


def test_output_gate_refuses_strength_overreach() -> None:
    """The emitted level over-reaches the strength the warranted-only evidence
    earns (e.g. an LLM certainty bump pushed conditional -> strong). Refused."""
    # Only TWO warranted studies -> the warranted-only SR earns at most a
    # conditional level (sparse-evidence GRADE cap). An emitted strong-for
    # over-reaches that.
    studies = [_study("w-1"), _study("w-2")]
    warranted = {"w-1", "w-2"}
    guideline = _included_guideline(["w-1", "w-2"], level="strong-for", direction="SUPPORTS")

    gated, admitted, reason = admit_guideline_output(
        guideline=guideline, studies=studies, warranted_ids=warranted
    )
    assert admitted is False
    assert gated.level == "no-recommendation"
    assert "over-reaches the strength" in reason


def test_output_gate_admits_well_supported_guideline() -> None:
    studies = [_study(f"w-{i}") for i in range(6)]
    warranted = {s.id for s in studies}
    guideline = _included_guideline(
        [s.id for s in studies], level="strong-for", direction="SUPPORTS"
    )
    gated, admitted, _reason = admit_guideline_output(
        guideline=guideline, studies=studies, warranted_ids=warranted
    )
    assert admitted is True
    assert gated.output_gate_refused is False
    assert gated.level == "strong-for"


def test_output_gate_is_blind_to_provenance_label() -> None:
    """Flipping ONLY the provenance label of the warranted corpus must not change
    the gate's admit/refuse decision (gate blindness, SPEC §8.3)."""
    base = [_study(f"w-{i}") for i in range(6)]
    warranted = {s.id for s in base}
    guideline = _included_guideline(
        [s.id for s in base], level="strong-for", direction="SUPPORTS"
    )
    as_grounded = [s.model_copy(update={"provenance": "GROUNDED"}) for s in base]
    as_ungrounded = [s.model_copy(update={"provenance": "UNGROUNDED"}) for s in base]

    _, admit_a, _ = admit_guideline_output(
        guideline=guideline, studies=as_grounded, warranted_ids=warranted
    )
    _, admit_b, _ = admit_guideline_output(
        guideline=guideline, studies=as_ungrounded, warranted_ids=warranted
    )
    assert admit_a == admit_b is True

    # Signature defense: the gate accepts no provenance/failure_mode argument.
    import inspect

    sig = inspect.signature(admit_guideline_output)
    assert "provenance" not in sig.parameters
    assert "failure_mode" not in sig.parameters


# --- Task C: run-level study volume -----------------------------------------


def test_run_produces_target_study_volume_per_arm() -> None:
    request = _request()
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.3,
    )
    final = bundle.db_growth[str(request.horizons[-1])]["studies"]
    # Free arm ingests every emitted study; with STUDIES_PER_CLAIM_PER_ERA=2,
    # 3 claims x 3 eras x 2 = 18 studies, inside the SPEC §13 target band 15-20.
    assert 15 <= final["free"]["count"] <= 20


def test_guideline_output_gate_runs_on_constrained_and_is_inspectable() -> None:
    """End-to-end: the guideline-output gate runs on every constrained-branch
    SRMA emission and is surfaced as a hash-chained audit event, and the
    screening/RoB report rides on the guideline for inspection.

    (With the DeterministicFakeClient the SRMA appraisal makes no certainty bump,
    so the warranted corpus and the emitted guideline agree and the gate ISSUES;
    the refusal path is exercised by the unit tests above. This asserts the gate
    is wired in and observable, not that it must refuse a non-adversarial run.)"""
    request = _request()
    request.horizons = list(range(1, 25))
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.45,
    )
    event_types = {event.event_type for event in bundle.audit_trail}
    assert "guideline-issued" in event_types or "guideline-refused" in event_types
    # Free branch is NOT gated: no guideline-admission events reference it.
    assert all(
        event.branch == "constrained"
        for event in bundle.audit_trail
        if event.phase == "guideline-admission"
    )
    # The screening/RoB report is surfaced on the guideline for inspection.
    constrained = bundle.guideline_timeline["constrained"]
    assert any(g.screening_report for g in constrained)
    assert all("n_included" in g.screening_report for g in constrained if g.screening_report)
