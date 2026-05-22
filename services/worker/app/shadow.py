from __future__ import annotations

from statistics import fmean
from typing import Any, Sequence

from app.c0 import GroundTruth, load_ground_truth
from app.ecology import RELEASE_THRESHOLD
from app.models import ArtifactBundle, ClaimGraph, GuidelineClaim, Study
from app.process_gate import assess_research_process
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
    all_guidelines = _resynthesize(studies)
    warranted_guidelines = _resynthesize([study for study in studies if study.id in passed_ids])

    latest_truth = truth.latest()
    all_distance = _distance_to_truth(all_guidelines, latest_truth)
    warranted_distance = _distance_to_truth(warranted_guidelines, latest_truth)
    drift = _natural_drift(all_guidelines, truth)

    return {
        "mode": "shadow-civer-brim",
        "source_branch": source_branch,
        "study_count": len(studies),
        "verdict_counts": _verdict_counts(verdicts),
        "calibration_matrix": _calibration_matrix(verdicts),
        "endpoint_1_natural_drift": drift,
        "endpoint_2_process_validation": _process_validation_summary(studies, verdicts),
        "endpoint_2_warrant_enrichment": _process_validation_summary(studies, verdicts),
        "endpoint_3_guideline_drift_reduction": {
            "all_to_truth": all_distance,
            "warranted_to_truth": warranted_distance,
            "delta": round(all_distance - warranted_distance, 4),
            "interprets_positive_delta_as": "warranted corpus is closer to external truth than all-output corpus",
        },
        "all_guideline_latest": _latest_by_claim(all_guidelines),
        "warranted_guideline_latest": _latest_by_claim(warranted_guidelines),
        "ground_truth_status": truth.status,
        "ground_truth_verified": truth.is_verified,
        "study_verdicts": verdicts,
    }


def _shadow_verdict(*, study: Study, graph: ClaimGraph | None) -> dict[str, Any]:
    graph = graph or ClaimGraph(claim_id=study.claim_id, claim_text="", nodes=[], edges=[])
    assessment = assess_research_process(
        study=study,
        claim_graph=graph,
        threshold=RELEASE_THRESHOLD,
    )
    return {
        "study_id": study.id,
        "claim_id": study.claim_id,
        "year": study.year,
        "passed": assessment.passed,
        "civer_passed": assessment.civer_passed,
        "brim_passed": assessment.brim_passed,
        "reasons": list(assessment.reasons),
        "execution_deviations": list(assessment.execution_deviations),
        "integrity_score": assessment.integrity_score,
        "true_provenance_for_calibration": study.provenance,
        "failure_mode_for_calibration": study.failure_mode,
        "plan_recorded": study.research_plan is not None,
        "committed_pmids": list(study.research_plan.committed_pmids)
        if study.research_plan is not None
        else [],
        "pmids": list(study.pmids),
        "catalog_pmids": list(study.catalog_pmids),
    }


def _resynthesize(studies: Sequence[Study]) -> list[GuidelineClaim]:
    claim_ids = sorted({study.claim_id for study in studies})
    years = sorted({study.year for study in studies})
    out: list[GuidelineClaim] = []
    for claim_id in claim_ids:
        for year in years:
            accumulated = [
                study for study in studies if study.claim_id == claim_id and study.year <= year
            ]
            out.append(synthesize_guideline_claim(claim_id=claim_id, year=year, studies=accumulated))
    return out


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
    studies: Sequence[Study], verdicts: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    passed_ids = {row["study_id"] for row in verdicts if row["passed"]}
    passed = [study for study in studies if study.id in passed_ids]
    failed = [study for study in studies if study.id not in passed_ids]
    return {
        "passed": _study_quality_summary(passed),
        "failed": _study_quality_summary(failed),
        "process_counts": {
            "civer_failed": sum(1 for row in verdicts if not row["civer_passed"]),
            "brim_failed": sum(1 for row in verdicts if not row["brim_passed"]),
            "missing_plan": sum(1 for row in verdicts if not row["plan_recorded"]),
            "execution_deviated": sum(1 for row in verdicts if row["execution_deviations"]),
        },
        "interprets_as": (
            "Shadow CIVER+BRIM has signal when failed studies have invalid PIR/plans "
            "or BRIM plan-to-execution deviations, and when warranted-only synthesis "
            "changes downstream drift."
        ),
    }


def _study_quality_summary(studies: Sequence[Study]) -> dict[str, Any]:
    total = len(studies)
    if not total:
        return {
            "count": 0,
            "ungrounded_rate": 0.0,
            "scope_overreach_rate": 0.0,
            "no_cite_rate": 0.0,
            "mean_quality": 0.0,
        }
    return {
        "count": total,
        "ungrounded_rate": round(
            sum(1 for study in studies if study.provenance == "UNGROUNDED") / total, 4
        ),
        "scope_overreach_rate": round(
            sum(1 for study in studies if study.failure_mode == "scope-overreach") / total, 4
        ),
        "no_cite_rate": round(sum(1 for study in studies if not study.pmids) / total, 4),
        "mean_quality": round(fmean(study.quality for study in studies), 4),
    }
