from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Sequence

from app.models import BranchName, GuidelineClaim, Study


_DIRECTION_AXIS = {"REFUTES": -1.0, "NEUTRAL": 0.0, "SUPPORTS": 1.0}
_LEVEL_AXIS = {
    "strong-against": -2.0,
    "conditional-against": -1.0,
    "no-recommendation": 0.0,
    "conditional-for": 1.0,
    "strong-for": 2.0,
}


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BranchGapReport:
    pair_count: int
    direction: BootstrapInterval
    level: BootstrapInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "direction": self.direction.to_dict(),
            "level": self.level.to_dict(),
        }


@dataclass(frozen=True)
class PhaseAReport:
    claim_count: int
    final_error: float
    no_change_error: float
    no_recommendation_error: float
    stability_delta: float
    pass_bars: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.pass_bars.values())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class PhaseBReport:
    observed: BranchGapReport
    controls: dict[str, BranchGapReport]
    pass_bars: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.pass_bars.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed.to_dict(),
            "controls": {name: report.to_dict() for name, report in self.controls.items()},
            "pass_bars": self.pass_bars,
            "passed": self.passed,
        }


def evaluate_phase_a(
    *,
    timeline: Sequence[GuidelineClaim],
    gold: Sequence[GuidelineClaim],
    stability_threshold: float = 0.25,
) -> PhaseAReport:
    """Validate clean ecology against measurable baselines before value claims."""
    gold_by_claim = _latest_by_claim(gold)
    latest = _latest_by_claim(timeline)
    earliest = _earliest_by_claim(timeline)
    penultimate = _nth_from_end_by_claim(timeline, offset=1)

    claim_ids = sorted(set(gold_by_claim) & set(latest))
    final_error = _mean_pair_distance(
        [latest[claim_id] for claim_id in claim_ids],
        [gold_by_claim[claim_id] for claim_id in claim_ids],
    )
    no_change_error = _mean_pair_distance(
        [earliest[claim_id] for claim_id in claim_ids],
        [gold_by_claim[claim_id] for claim_id in claim_ids],
    )
    no_recommendation_error = _mean_pair_distance(
        [_neutral_guideline(claim_id, gold_by_claim[claim_id].year) for claim_id in claim_ids],
        [gold_by_claim[claim_id] for claim_id in claim_ids],
    )
    stable_claims = [claim_id for claim_id in claim_ids if claim_id in penultimate]
    stability_delta = _mean_pair_distance(
        [latest[claim_id] for claim_id in stable_claims],
        [penultimate[claim_id] for claim_id in stable_claims],
    )
    pass_bars = {
        "beats_no_change": final_error < no_change_error,
        "beats_no_recommendation": final_error < no_recommendation_error,
        "stable_final_window": stability_delta <= stability_threshold,
    }
    return PhaseAReport(
        claim_count=len(claim_ids),
        final_error=round(final_error, 4),
        no_change_error=round(no_change_error, 4),
        no_recommendation_error=round(no_recommendation_error, 4),
        stability_delta=round(stability_delta, 4),
        pass_bars=pass_bars,
    )


def evaluate_phase_b(
    *,
    free: Sequence[GuidelineClaim],
    constrained: Sequence[GuidelineClaim],
    controls: dict[str, tuple[Sequence[GuidelineClaim], Sequence[GuidelineClaim]]] | None = None,
    iterations: int = 1000,
    seed: int = 0,
    control_margin: float = 0.02,
) -> PhaseBReport:
    """Measure whether branch divergence survives CI and control comparisons."""
    observed = branch_gap(free=free, constrained=constrained, iterations=iterations, seed=seed)
    control_reports = {
        name: branch_gap(
            free=control_free,
            constrained=control_constrained,
            iterations=iterations,
            seed=seed,
        )
        for name, (control_free, control_constrained) in (controls or {}).items()
    }
    pass_bars = {
        "direction_ci_excludes_zero": observed.direction.low > 0,
        "level_ci_excludes_zero": observed.level.low > 0,
        "direction_beats_controls": all(
            observed.direction.mean > report.direction.mean + control_margin
            for report in control_reports.values()
        ),
        "level_beats_controls": all(
            observed.level.mean > report.level.mean + control_margin
            for report in control_reports.values()
        ),
    }
    return PhaseBReport(observed=observed, controls=control_reports, pass_bars=pass_bars)


def branch_gap(
    *,
    free: Sequence[GuidelineClaim],
    constrained: Sequence[GuidelineClaim],
    iterations: int = 1000,
    seed: int = 0,
) -> BranchGapReport:
    pairs = _paired_latest(free, constrained)
    direction_deltas = [
        abs(_DIRECTION_AXIS[left.direction] - _DIRECTION_AXIS[right.direction]) / 2.0
        for left, right in pairs
    ]
    level_deltas = [
        abs(_LEVEL_AXIS[left.level] - _LEVEL_AXIS[right.level]) / 4.0
        for left, right in pairs
    ]
    return BranchGapReport(
        pair_count=len(pairs),
        direction=bootstrap_ci(direction_deltas, iterations=iterations, seed=seed),
        level=bootstrap_ci(level_deltas, iterations=iterations, seed=seed + 1),
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapInterval:
    if not values:
        return BootstrapInterval(mean=0.0, low=0.0, high=0.0)
    if len(values) == 1:
        value = round(values[0], 4)
        return BootstrapInterval(mean=value, low=value, high=value)

    rng = random.Random(seed)
    means = []
    for _ in range(max(iterations, 1)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(fmean(sample))
    means.sort()
    low_index = int((alpha / 2) * (len(means) - 1))
    high_index = int((1 - alpha / 2) * (len(means) - 1))
    return BootstrapInterval(
        mean=round(fmean(values), 4),
        low=round(means[low_index], 4),
        high=round(means[high_index], 4),
    )


def replay_counts(
    *,
    studies: dict[BranchName, Sequence[Study]],
    guidelines: dict[BranchName, Sequence[GuidelineClaim]],
) -> dict[str, object]:
    return {
        "studies": {
            branch: {
                "count": len(branch_studies),
                "real": sum(1 for study in branch_studies if study.provenance == "REAL"),
                "synthetic": sum(
                    1 for study in branch_studies if study.provenance == "SYNTHETIC"
                ),
            }
            for branch, branch_studies in studies.items()
        },
        "guidelines": {
            branch: {
                "count": len(branch_guidelines),
                "claim_count": len({guideline.claim_id for guideline in branch_guidelines}),
                "years": sorted({guideline.year for guideline in branch_guidelines}),
            }
            for branch, branch_guidelines in guidelines.items()
        },
    }


def _paired_latest(
    free: Sequence[GuidelineClaim],
    constrained: Sequence[GuidelineClaim],
) -> list[tuple[GuidelineClaim, GuidelineClaim]]:
    free_latest = _latest_by_claim(free)
    constrained_latest = _latest_by_claim(constrained)
    return [
        (free_latest[claim_id], constrained_latest[claim_id])
        for claim_id in sorted(set(free_latest) & set(constrained_latest))
    ]


def _latest_by_claim(guidelines: Sequence[GuidelineClaim]) -> dict[str, GuidelineClaim]:
    latest: dict[str, GuidelineClaim] = {}
    for guideline in sorted(guidelines, key=lambda item: (item.claim_id, item.year)):
        latest[guideline.claim_id] = guideline
    return latest


def _earliest_by_claim(guidelines: Sequence[GuidelineClaim]) -> dict[str, GuidelineClaim]:
    earliest: dict[str, GuidelineClaim] = {}
    for guideline in sorted(guidelines, key=lambda item: (item.claim_id, item.year)):
        earliest.setdefault(guideline.claim_id, guideline)
    return earliest


def _nth_from_end_by_claim(
    guidelines: Sequence[GuidelineClaim],
    *,
    offset: int,
) -> dict[str, GuidelineClaim]:
    grouped: dict[str, list[GuidelineClaim]] = {}
    for guideline in sorted(guidelines, key=lambda item: (item.claim_id, item.year)):
        grouped.setdefault(guideline.claim_id, []).append(guideline)
    return {
        claim_id: items[-1 - offset]
        for claim_id, items in grouped.items()
        if len(items) > offset
    }


def _mean_pair_distance(
    left: Sequence[GuidelineClaim],
    right: Sequence[GuidelineClaim],
) -> float:
    if not left or not right:
        return 0.0
    return fmean(
        guideline_distance(left_item, right_item)
        for left_item, right_item in zip(left, right)
    )


def guideline_distance(left: GuidelineClaim, right: GuidelineClaim) -> float:
    direction_delta = abs(_DIRECTION_AXIS[left.direction] - _DIRECTION_AXIS[right.direction]) / 2.0
    level_delta = abs(_LEVEL_AXIS[left.level] - _LEVEL_AXIS[right.level]) / 4.0
    return (direction_delta + level_delta) / 2.0


def _neutral_guideline(claim_id: str, year: int) -> GuidelineClaim:
    return GuidelineClaim(
        claim_id=claim_id,
        year=year,
        direction="NEUTRAL",
        level="no-recommendation",
    )
