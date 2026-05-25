from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Sequence

from app.c0 import GroundTruth, load_ground_truth
from app.ecology import RELEASE_THRESHOLD
from app.models import ArtifactBundle, ClaimGraph, GuidelineClaim, Study
from app.process_gate import assess_output_level_fallback, assess_research_process
from app.synthesis import synthesize_guideline_claim


_DIRECTION_AXIS = {"REFUTES": -1.0, "NEUTRAL": 0.0, "SUPPORTS": 1.0}
_LEVEL_AXIS = {
    "strong-against": -2.0,
    "conditional-against": -1.0,
    "no-recommendation": 0.0,
    "conditional-for": 1.0,
    "strong-for": 2.0,
}


def evaluate_shadow_civer(
    *,
    bundle: ArtifactBundle,
    ground_truth_path: str | None = None,
    source_branch: str = "free",
) -> dict[str, Any]:
    """Evaluate CIVER+BRIM as a post-hoc process validator.

    This is the non-circular lane: agents emit studies once, nothing is blocked
    during generation, and the stored research process trace is replayed through
    the same CIVER pre-execution gate + BRIM release score. It judges whether a
    study's research process was valid, not whether the final citation list looks
    tidy or whether the conclusion matches truth.
    """
    studies = list((bundle.corpus_studies or {}).get(source_branch, []))
    graphs = {graph.claim_id: graph for graph in bundle.claim_graphs}
    truth = load_ground_truth(ground_truth_path)
    verdicts = [_shadow_verdict(study=study, graph=graphs.get(study.claim_id)) for study in studies]
    passed_ids = {row["study_id"] for row in verdicts if row["passed"]}
    # IMPORTANT: both arms must be scored on the same claim/year grid. The old
    # path synthesized the warranted arm only over claims that had >=1 admitted
    # study, silently dropping no-warrant cells from the denominator.
    grid_claim_ids = sorted({study.claim_id for study in studies})
    grid_years = sorted({study.year for study in studies})
    all_guidelines = _resynthesize(studies, claim_ids=grid_claim_ids, years=grid_years)
    warranted_guidelines = _resynthesize(
        [study for study in studies if study.id in passed_ids],
        claim_ids=grid_claim_ids,
        years=grid_years,
    )

    latest_truth = truth.latest()
    all_distance = _distance_to_truth(all_guidelines, latest_truth)
    warranted_distance = _distance_to_truth(warranted_guidelines, latest_truth)
    drift = _natural_drift(all_guidelines, truth)

    # Volume-matched null for E3 (Paper 3 rigor): without this, a smaller
    # warranted corpus could look closer to truth just by lucky pooling. The
    # null samples a same-size random subset of the FULL natural corpus and
    # reports the bootstrap mean distance + 95% CI. CIVER's delta is only
    # discriminative if it beats this null.
    warranted_ids_set = passed_ids
    volume_null = _volume_matched_null_for_e3(
        studies=studies,
        warranted_count=len(warranted_ids_set),
        truth_latest=latest_truth,
        iterations=500,
        seed=0,
        claim_ids=grid_claim_ids,
        years=grid_years,
    )
    denominator_audit = _e3_denominator_audit(
        all_guidelines=all_guidelines,
        warranted_guidelines=warranted_guidelines,
        truth=truth,
    )
    real_comparison = _real_comparison_e3(
        all_guidelines=all_guidelines,
        warranted_guidelines=warranted_guidelines,
        truth=truth,
    )

    mode_breakdown = {
        "process": sum(1 for v in verdicts if v.get("analysis_mode") == "process"),
        "output_fallback": sum(
            1 for v in verdicts if v.get("analysis_mode") == "output_fallback"
        ),
    }
    fallback_warning = None
    if mode_breakdown["output_fallback"] > 0:
        fallback_warning = (
            "OUTPUT-FALLBACK LANE ACTIVE — one or more studies in this bundle "
            "lacked a ResearchPlan trace (legacy bundle). Their pass/fail "
            "verdicts come from output-level scaffolding checks (citation "
            "resolvability + scope vs source), NOT the process-CIVER claim. "
            "Treat E2/E3 numbers from output-fallback verdicts as MedEvo "
            "environment quality, not CIVER mechanism evidence."
        )

    return {
        "mode": "shadow-civer-brim",
        "source_branch": source_branch,
        "study_count": len(studies),
        "analysis_mode_breakdown": mode_breakdown,
        "fallback_warning": fallback_warning,
        "verdict_counts": _verdict_counts(verdicts),
        "calibration_matrix": _calibration_matrix(verdicts),
        "endpoint_1_natural_drift": drift,
        "endpoint_2_process_validation": _process_validation_summary(studies, verdicts, truth),
        "endpoint_2_warrant_enrichment": _process_validation_summary(studies, verdicts, truth),
        "endpoint_2_per_claim": _per_claim_warrant_enrichment(studies, verdicts, truth),
        "endpoint_3_guideline_drift_reduction": {
            "all_to_truth": all_distance,
            "warranted_to_truth": warranted_distance,
            "delta": round(all_distance - warranted_distance, 4),
            "interprets_positive_delta_as": "warranted corpus is closer to external truth than all-output corpus",
            "denominator_audit": denominator_audit,
            "real_comparison": real_comparison,
            "paper_grade_interpretable": denominator_audit["zero_warrant_cell_fraction"] <= 0.5,
            "interpretation_warning": (
                "If zero_warrant_cell_fraction is high, aggregate E3 is not a "
                "paper-grade CIVER value estimate; use real_comparison for the "
                "cells where CIVER actually supplied evidence."
            ),
            "volume_matched_null": volume_null,
            "civer_beats_volume_matched": (
                volume_null is not None
                and warranted_distance < volume_null["ci_low"]
                and denominator_audit["zero_warrant_cell_fraction"] <= 0.5
            ),
        },
        "all_guideline_latest": _latest_by_claim(all_guidelines),
        "warranted_guideline_latest": _latest_by_claim(warranted_guidelines),
        "ground_truth_status": truth.status,
        "ground_truth_verified": truth.is_verified,
        "study_verdicts": verdicts,
    }


def _shadow_verdict(*, study: Study, graph: ClaimGraph | None) -> dict[str, Any]:
    graph = graph or ClaimGraph(claim_id=study.claim_id, claim_text="", nodes=[], edges=[])
    if study.research_plan is None:
        # OUTPUT-FALLBACK LANE (NOT process-CIVER): the bundle predates free-arm
        # plan recording (e.g. Run 4 on main@525782a). Verdict comes from
        # output-level scaffolding checks (SPEC v3 §0a) so the analyzer can
        # still report a useful pass/fail split; the verdict is explicitly
        # tagged `analysis_mode="output_fallback"` and `civer_passed=False` so
        # no downstream consumer can mistake it for a CIVER process claim.
        fb = assess_output_level_fallback(study=study)
        return {
            "study_id": study.id,
            "claim_id": study.claim_id,
            "year": study.year,
            "analysis_mode": "output_fallback",
            "passed": fb.passed,
            "civer_passed": False,
            "brim_passed": False,
            "output_check_passed": fb.passed,
            "output_cites_resolve": fb.cites_resolve,
            "output_scope_within_source": fb.scope_within_source,
            "reasons": list(fb.reasons),
            "execution_deviations": [],
            "integrity_score": 1.0 if fb.passed else 0.0,
            "true_provenance_for_calibration": study.provenance,
            "failure_mode_for_calibration": study.failure_mode,
            "plan_recorded": False,
            "committed_pmids": [],
            "pmids": list(study.pmids),
            "catalog_pmids": list(study.catalog_pmids),
        }
    assessment = assess_research_process(
        study=study,
        claim_graph=graph,
        threshold=RELEASE_THRESHOLD,
    )
    return {
        "study_id": study.id,
        "claim_id": study.claim_id,
        "year": study.year,
        "analysis_mode": "process",
        "passed": assessment.passed,
        "civer_passed": assessment.civer_passed,
        "brim_passed": assessment.brim_passed,
        "reasons": list(assessment.reasons),
        "execution_deviations": list(assessment.execution_deviations),
        "integrity_score": assessment.integrity_score,
        "true_provenance_for_calibration": study.provenance,
        "failure_mode_for_calibration": study.failure_mode,
        "plan_recorded": True,
        "committed_pmids": list(study.research_plan.committed_pmids),
        "pmids": list(study.pmids),
        "catalog_pmids": list(study.catalog_pmids),
    }


def _resynthesize(
    studies: Sequence[Study],
    *,
    claim_ids: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
) -> list[GuidelineClaim]:
    claim_ids = sorted(claim_ids if claim_ids is not None else {study.claim_id for study in studies})
    years = sorted(years if years is not None else {study.year for study in studies})
    out: list[GuidelineClaim] = []
    for claim_id in claim_ids:
        for year in years:
            accumulated = [
                study for study in studies if study.claim_id == claim_id and study.year <= year
            ]
            out.append(synthesize_guideline_claim(claim_id=claim_id, year=year, studies=accumulated))
    return out


def _guideline_by_cell(guidelines: Sequence[GuidelineClaim]) -> dict[tuple[str, int], GuidelineClaim]:
    return {(guideline.claim_id, guideline.year): guideline for guideline in guidelines}


def _truth_by_cell(truth: GroundTruth) -> dict[tuple[str, int], GuidelineClaim]:
    return {
        (claim_id, guideline.year): guideline
        for claim_id, series in truth.trajectory.items()
        for guideline in series
    }


def _e3_denominator_audit(
    *,
    all_guidelines: Sequence[GuidelineClaim],
    warranted_guidelines: Sequence[GuidelineClaim],
    truth: GroundTruth,
) -> dict[str, Any]:
    all_by_cell = _guideline_by_cell(all_guidelines)
    warranted_by_cell = _guideline_by_cell(warranted_guidelines)
    truth_by_cell = _truth_by_cell(truth)
    cells = sorted(set(all_by_cell) & set(warranted_by_cell) & set(truth_by_cell))
    zero_warrant = [cell for cell in cells if warranted_by_cell[cell].n_included == 0]
    real = [cell for cell in cells if warranted_by_cell[cell].n_included > 0]
    zero_help = [
        cell
        for cell in zero_warrant
        if _pair_distance(warranted_by_cell[cell], truth_by_cell[cell])
        < _pair_distance(all_by_cell[cell], truth_by_cell[cell])
    ]
    zero_hurt = [
        cell
        for cell in zero_warrant
        if _pair_distance(warranted_by_cell[cell], truth_by_cell[cell])
        > _pair_distance(all_by_cell[cell], truth_by_cell[cell])
    ]
    total = len(cells)
    return {
        "cell_count": total,
        "real_comparison_cells": len(real),
        "zero_warrant_cells": len(zero_warrant),
        "real_comparison_fraction": round(len(real) / total, 4) if total else 0.0,
        "zero_warrant_cell_fraction": round(len(zero_warrant) / total, 4) if total else 0.0,
        "zero_warrant_help_cells": len(zero_help),
        "zero_warrant_hurt_cells": len(zero_hurt),
    }


def _mean_distance_for_cells(
    guidelines: dict[tuple[str, int], GuidelineClaim],
    truth: dict[tuple[str, int], GuidelineClaim],
    cells: Sequence[tuple[str, int]],
) -> float | None:
    if not cells:
        return None
    return round(fmean(_pair_distance(guidelines[cell], truth[cell]) for cell in cells), 4)


def _real_comparison_e3(
    *,
    all_guidelines: Sequence[GuidelineClaim],
    warranted_guidelines: Sequence[GuidelineClaim],
    truth: GroundTruth,
) -> dict[str, Any]:
    all_by_cell = _guideline_by_cell(all_guidelines)
    warranted_by_cell = _guideline_by_cell(warranted_guidelines)
    truth_by_cell = _truth_by_cell(truth)
    cells = sorted(
        cell
        for cell in set(all_by_cell) & set(warranted_by_cell) & set(truth_by_cell)
        if warranted_by_cell[cell].n_included > 0
    )
    all_distance = _mean_distance_for_cells(all_by_cell, truth_by_cell, cells)
    warranted_distance = _mean_distance_for_cells(warranted_by_cell, truth_by_cell, cells)
    return {
        "cell_count": len(cells),
        "all_to_truth": all_distance,
        "warranted_to_truth": warranted_distance,
        "delta": round(all_distance - warranted_distance, 4)
        if all_distance is not None and warranted_distance is not None
        else None,
        "interprets_positive_delta_as": (
            "Among cells where the warranted/CIVER arm has at least one admitted "
            "study, warranted guidelines are closer to truth."
        ),
    }


def _latest_by_claim(guidelines: Sequence[GuidelineClaim]) -> dict[str, dict[str, Any]]:
    latest: dict[str, GuidelineClaim] = {}
    for guideline in sorted(guidelines, key=lambda g: (g.claim_id, g.year)):
        latest[guideline.claim_id] = guideline
    return {
        claim_id: {
            "year": item.year,
            "direction": item.direction,
            "level": item.level,
            "study_count": item.study_count,
            "ungrounded_fraction": item.ungrounded_fraction,
            "n_included": item.n_included,
            "n_excluded": item.n_excluded,
        }
        for claim_id, item in sorted(latest.items())
    }


def _distance_to_truth(
    guidelines: Sequence[GuidelineClaim],
    truth_latest: dict[str, GuidelineClaim],
) -> float:
    latest = {claim_id: item for claim_id, item in _latest_guidelines(guidelines).items()}
    claim_ids = sorted(set(latest) & set(truth_latest))
    if not claim_ids:
        return 0.0
    return round(
        fmean(_pair_distance(latest[claim_id], truth_latest[claim_id]) for claim_id in claim_ids),
        4,
    )


def _natural_drift(guidelines: Sequence[GuidelineClaim], truth: GroundTruth) -> dict[str, Any]:
    latest = _latest_guidelines(guidelines)
    latest_truth = truth.latest()
    rows = []
    for claim_id in sorted(set(latest) & set(latest_truth)):
        distance = _pair_distance(latest[claim_id], latest_truth[claim_id])
        rows.append(
            {
                "claim_id": claim_id,
                "distance_to_truth": round(distance, 4),
                "output": {
                    "direction": latest[claim_id].direction,
                    "level": latest[claim_id].level,
                },
                "truth": {
                    "direction": latest_truth[claim_id].direction,
                    "level": latest_truth[claim_id].level,
                },
            }
        )
    return {
        "mean_distance_to_truth": round(fmean(row["distance_to_truth"] for row in rows), 4)
        if rows
        else 0.0,
        "per_claim": rows,
    }


def _latest_guidelines(guidelines: Sequence[GuidelineClaim]) -> dict[str, GuidelineClaim]:
    latest: dict[str, GuidelineClaim] = {}
    for guideline in sorted(guidelines, key=lambda g: (g.claim_id, g.year)):
        latest[guideline.claim_id] = guideline
    return latest


def _pair_distance(left: GuidelineClaim, right: GuidelineClaim) -> float:
    direction = abs(_DIRECTION_AXIS[left.direction] - _DIRECTION_AXIS[right.direction]) / 2.0
    level = abs(_LEVEL_AXIS[left.level] - _LEVEL_AXIS[right.level]) / 4.0
    return (direction + level) / 2.0


def _verdict_counts(verdicts: Sequence[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for row in verdicts if row["passed"])
    failed = len(verdicts) - passed
    return {"passed": passed, "failed": failed, "total": len(verdicts)}


def _calibration_matrix(verdicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in verdicts:
        grounded = row["true_provenance_for_calibration"] == "GROUNDED"
        passed = bool(row["passed"])
        if grounded and passed:
            tp += 1
        elif grounded and not passed:
            fp += 1
        elif not grounded and passed:
            fn += 1
        else:
            tn += 1
    grounded_total = tp + fp
    ungrounded_total = tn + fn
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "grounded_total": grounded_total,
        "ungrounded_total": ungrounded_total,
        "fpr": round(fp / grounded_total, 4) if grounded_total else 0.0,
        "fnr": round(fn / ungrounded_total, 4) if ungrounded_total else 0.0,
    }


def _process_validation_summary(
    studies: Sequence[Study],
    verdicts: Sequence[dict[str, Any]],
    truth: GroundTruth | None = None,
) -> dict[str, Any]:
    passed_ids = {row["study_id"] for row in verdicts if row["passed"]}
    passed = [study for study in studies if study.id in passed_ids]
    failed = [study for study in studies if study.id not in passed_ids]
    truth_lookup = _truth_direction_lookup(truth)
    return {
        "passed": _study_quality_summary(passed, truth_lookup),
        "failed": _study_quality_summary(failed, truth_lookup),
        "process_counts": {
            "civer_failed": sum(1 for row in verdicts if not row["civer_passed"]),
            "brim_failed": sum(1 for row in verdicts if not row["brim_passed"]),
            "missing_plan": sum(1 for row in verdicts if not row["plan_recorded"]),
            "execution_deviated": sum(1 for row in verdicts if row["execution_deviations"]),
        },
        "interprets_as": (
            "E2/Warrant enrichment has signal when the PASSED cohort has lower "
            "ungrounded / no-cite / wrong-direction rates than the FAILED cohort. "
            "scope_overreach_rate paradox alert: a no-cite study cannot over-reach "
            "scope (no source to over-reach), so a failed cohort dominated by "
            "no-cite studies will show LOWER scope_overreach_rate than passed — "
            "interpret scope_overreach jointly with no_cite_rate, not in isolation."
        ),
    }


def _per_claim_warrant_enrichment(
    studies: Sequence[Study],
    verdicts: Sequence[dict[str, Any]],
    truth: GroundTruth | None = None,
) -> dict[str, dict[str, Any]]:
    """SPEC §E2 per-claim breakdown. A single aggregate hides claim-specific
    discrimination (e.g. CIVER may catch alcohol drift but miss obesity-paradox
    over-reach). Each claim reports passed/failed cohort metrics so reviewers
    see which claims the gate works for."""
    passed_ids = {row["study_id"] for row in verdicts if row["passed"]}
    truth_lookup = _truth_direction_lookup(truth)
    by_claim: dict[str, list[Study]] = {}
    for study in studies:
        by_claim.setdefault(study.claim_id, []).append(study)
    out: dict[str, dict[str, Any]] = {}
    for claim_id, claim_studies in sorted(by_claim.items()):
        passed = [s for s in claim_studies if s.id in passed_ids]
        failed = [s for s in claim_studies if s.id not in passed_ids]
        out[claim_id] = {
            "passed": _study_quality_summary(passed, truth_lookup),
            "failed": _study_quality_summary(failed, truth_lookup),
        }
    return out


def _truth_direction_lookup(truth: GroundTruth | None) -> dict[tuple[str, int], str]:
    """(claim_id, year) -> truth direction. Used to score study direction against
    the labelled trajectory for the per-cohort wrong-direction-rate metric."""
    if truth is None:
        return {}
    out: dict[tuple[str, int], str] = {}
    for claim_id, series in truth.trajectory.items():
        for point in series:
            out[(claim_id, point.year)] = point.direction
    return out


def _study_quality_summary(
    studies: Sequence[Study],
    truth_lookup: dict[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    total = len(studies)
    if not total:
        return {
            "count": 0,
            "ungrounded_rate": 0.0,
            "scope_overreach_rate": 0.0,
            "no_cite_rate": 0.0,
            "wrong_direction_vs_truth_rate": 0.0,
            "mean_quality": 0.0,
        }
    truth_lookup = truth_lookup or {}
    wrong_dir = 0
    scored = 0
    for study in studies:
        truth_dir = truth_lookup.get((study.claim_id, study.year))
        if truth_dir is None:
            continue
        scored += 1
        if study.direction != truth_dir:
            wrong_dir += 1
    return {
        "count": total,
        "ungrounded_rate": round(
            sum(1 for study in studies if study.provenance == "UNGROUNDED") / total, 4
        ),
        "scope_overreach_rate": round(
            sum(1 for study in studies if study.failure_mode == "scope-overreach") / total, 4
        ),
        "no_cite_rate": round(sum(1 for study in studies if not study.pmids) / total, 4),
        "wrong_direction_vs_truth_rate": (
            round(wrong_dir / scored, 4) if scored else 0.0
        ),
        "mean_quality": round(fmean(study.quality for study in studies), 4),
    }


def _volume_matched_null_for_e3(
    *,
    studies: Sequence[Study],
    warranted_count: int,
    truth_latest: dict[str, GuidelineClaim],
    iterations: int,
    seed: int,
    claim_ids: Sequence[str],
    years: Sequence[int],
) -> dict[str, Any] | None:
    """E3 volume-matched null distribution.

    Without this, "warranted corpus closer to truth" could be smaller-pool luck.
    The null samples ``warranted_count`` studies uniformly from the full natural
    corpus, re-synthesizes the guideline trajectory from that subsample, and
    measures distance to the latest truth. Returns the bootstrap distribution's
    mean + 95% CI of distance. CIVER is discriminative only when warranted's
    distance is strictly below the null's lower CI bound.
    """
    if warranted_count <= 0 or warranted_count >= len(studies):
        return None
    rng = random.Random(seed)
    distances: list[float] = []
    studies_list = list(studies)
    for _ in range(max(iterations, 1)):
        sample_ids = set()
        # Sampling without replacement, matched to warranted count.
        indices = list(range(len(studies_list)))
        rng.shuffle(indices)
        for idx in indices[:warranted_count]:
            sample_ids.add(studies_list[idx].id)
        sampled = [s for s in studies_list if s.id in sample_ids]
        guidelines = _resynthesize(sampled, claim_ids=claim_ids, years=years)
        distances.append(_distance_to_truth(guidelines, truth_latest))
    distances.sort()
    n = len(distances)
    low_idx = int(0.025 * (n - 1))
    high_idx = int(0.975 * (n - 1))
    return {
        "iterations": iterations,
        "sample_size": warranted_count,
        "mean": round(fmean(distances), 4),
        "ci_low": round(distances[low_idx], 4),
        "ci_high": round(distances[high_idx], 4),
        "interpretation": (
            "Random subsamples of size=warranted_count drawn from the full "
            "natural corpus. CIVER's warranted_to_truth must fall strictly "
            "below ci_low to claim selection (not smaller-pool luck) is the "
            "mechanism."
        ),
    }
