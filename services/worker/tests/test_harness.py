from __future__ import annotations

from app.harness import branch_gap, evaluate_phase_a, evaluate_phase_b, replay_counts
from app.models import GuidelineClaim, Study


def _guideline(
    claim_id: str,
    year: int,
    *,
    direction: str,
    level: str,
) -> GuidelineClaim:
    return GuidelineClaim(
        claim_id=claim_id,
        year=year,
        direction=direction,
        level=level,
    )


def _study(study_id: str, *, provenance: str) -> Study:
    return Study(
        id=study_id,
        claim_id="claim-1",
        year=2020,
        direction="SUPPORTS",
        quality=0.8,
        provenance=provenance,
        numeric=False,
        rationale="Replay count fixture.",
    )


def test_phase_a_requires_clean_run_to_beat_baselines_and_stabilize() -> None:
    timeline = [
        _guideline("claim-1", 2020, direction="NEUTRAL", level="no-recommendation"),
        _guideline("claim-1", 2025, direction="SUPPORTS", level="conditional-for"),
        _guideline("claim-1", 2030, direction="SUPPORTS", level="strong-for"),
        _guideline("claim-2", 2020, direction="NEUTRAL", level="no-recommendation"),
        _guideline("claim-2", 2025, direction="REFUTES", level="conditional-against"),
        _guideline("claim-2", 2030, direction="REFUTES", level="strong-against"),
    ]
    gold = [
        _guideline("claim-1", 2030, direction="SUPPORTS", level="strong-for"),
        _guideline("claim-2", 2030, direction="REFUTES", level="strong-against"),
    ]

    report = evaluate_phase_a(timeline=timeline, gold=gold)

    assert report.passed
    assert report.final_error == 0
    assert report.no_change_error > report.final_error
    assert report.no_recommendation_error > report.final_error
    assert report.stability_delta <= 0.25


def test_phase_b_requires_branch_gap_to_survive_ci_and_controls() -> None:
    free = [
        _guideline(f"claim-{index}", 2030, direction="SUPPORTS", level="strong-for")
        for index in range(1, 5)
    ]
    constrained = [
        _guideline(f"claim-{index}", 2030, direction="REFUTES", level="strong-against")
        for index in range(1, 5)
    ]
    control_left = [
        _guideline(f"claim-{index}", 2030, direction="SUPPORTS", level="conditional-for")
        for index in range(1, 5)
    ]
    control_right = [
        _guideline(f"claim-{index}", 2030, direction="SUPPORTS", level="conditional-for")
        for index in range(1, 5)
    ]

    report = evaluate_phase_b(
        free=free,
        constrained=constrained,
        controls={"volume_matched": (control_left, control_right)},
        iterations=200,
        seed=11,
    )

    assert report.passed
    assert report.observed.direction.low > 0
    assert report.observed.level.low > 0
    assert report.controls["volume_matched"].direction.mean == 0


def test_branch_gap_pairs_latest_guidelines_by_claim() -> None:
    free = [
        _guideline("claim-1", 2020, direction="NEUTRAL", level="no-recommendation"),
        _guideline("claim-1", 2030, direction="SUPPORTS", level="strong-for"),
    ]
    constrained = [
        _guideline("claim-1", 2020, direction="NEUTRAL", level="no-recommendation"),
        _guideline("claim-1", 2030, direction="REFUTES", level="strong-against"),
    ]

    report = branch_gap(free=free, constrained=constrained)

    assert report.pair_count == 1
    assert report.direction.mean == 1.0
    assert report.level.mean == 1.0


def test_replay_counts_exposes_study_and_guideline_population_stats() -> None:
    counts = replay_counts(
        studies={
            "free": [_study("real-1", provenance="GROUNDED"), _study("syn-1", provenance="UNGROUNDED")],
            "constrained": [_study("real-2", provenance="GROUNDED")],
        },
        guidelines={
            "free": [_guideline("claim-1", 2020, direction="SUPPORTS", level="conditional-for")],
            "constrained": [
                _guideline("claim-1", 2020, direction="SUPPORTS", level="conditional-for")
            ],
        },
    )

    assert counts["studies"]["free"]["synthetic"] == 1
    assert counts["studies"]["constrained"]["real"] == 1
    assert counts["guidelines"]["free"]["claim_count"] == 1
