from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.ecology import (
    RELEASE_THRESHOLD,
    SCOPE_TOLERANCE_YEARS,
    CorpusItem,
    admit_research_plan,
)
from app.models import (
    BranchName,
    ClaimGraph,
    EvidenceScope,
    ExecutionWarrant,
    ResearchPlan,
    Study,
)


@dataclass(frozen=True)
class ProcessAssessment:
    study_id: str
    passed: bool
    civer_passed: bool
    brim_passed: bool
    integrity_score: float
    threshold: float
    reasons: list[str]
    execution_deviations: list[str]


def assess_research_process(
    *,
    study: Study,
    claim_graph: ClaimGraph,
    threshold: float = RELEASE_THRESHOLD,
) -> ProcessAssessment:
    """Apply CIVER+BRIM to a completed research process.

    CIVER is the pre-execution compiler gate over a ResearchPlan/PIR. BRIM is the
    monitor of plan->execution deviations. This function is used by shadow mode:
    the study has already been produced, but the law is replayed over the stored
    process trace. It deliberately does not classify "truth"; it judges whether
    the research process had a valid plan and stayed within that plan.
    """
    if study.research_plan is None:
        return ProcessAssessment(
            study_id=study.id,
            passed=False,
            civer_passed=False,
            brim_passed=False,
            integrity_score=0.0,
            threshold=threshold,
            reasons=["No ResearchPlan/PIR trace was recorded for this study."],
            execution_deviations=["missing-plan"],
        )

    plan = study.research_plan
    reachable_lookup = _reachable_from_study(study)
    civer_passed, civer_reasons = admit_research_plan(
        plan=plan,
        claim_graph=claim_graph,
        reachable_lookup=reachable_lookup,
    )
    deviations = execution_deviations(plan=plan, study=study)
    score = process_integrity_score(civer_passed=civer_passed, deviations=deviations)
    brim_passed = score >= threshold and not deviations
    passed = civer_passed and brim_passed
    reasons = list(civer_reasons)
    if deviations:
        reasons.append("BRIM detected plan-to-execution deviation: " + "; ".join(deviations))
    else:
        reasons.append("BRIM detected no plan-to-execution deviation.")
    reasons.append(f"Final process integrity score={score:.3f} threshold={threshold:.3f}.")
    return ProcessAssessment(
        study_id=study.id,
        passed=passed,
        civer_passed=civer_passed,
        brim_passed=brim_passed,
        integrity_score=score,
        threshold=threshold,
        reasons=reasons,
        execution_deviations=deviations,
    )


def issue_process_warrant(
    *,
    run_id: str,
    branch: BranchName,
    year: int,
    study: Study,
    claim_graph: ClaimGraph,
    threshold: float = RELEASE_THRESHOLD,
) -> tuple[ProcessAssessment, ExecutionWarrant]:
    assessment = assess_research_process(
        study=study,
        claim_graph=claim_graph,
        threshold=threshold,
    )
    warrant = ExecutionWarrant(
        id=f"ECW-{study.id}",
        output_id=study.id,
        output_hash=study.output_hash or _study_process_hash(study),
        run_id=run_id,
        claim_id=study.claim_id,
        branch=branch,
        year=year,
        status="ISSUED" if assessment.passed else "REFUSED",
        issued=assessment.passed,
        integrity_score=assessment.integrity_score,
        threshold=threshold,
    )
    return assessment, warrant


def execution_deviations(*, plan: ResearchPlan, study: Study) -> list[str]:
    deviations: list[str] = []
    committed = set(plan.committed_pmids)
    out_of_plan = [pmid for pmid in study.pmids if pmid not in committed]
    if out_of_plan:
        deviations.append("cited outside committed plan: " + ", ".join(out_of_plan))
    if study.claimed_scope.exceeds(plan.claimed_scope, tolerance=0):
        deviations.append("execution scope exceeds the registered plan scope")
    return deviations


def process_integrity_score(*, civer_passed: bool, deviations: list[str]) -> float:
    if not civer_passed:
        return 0.0
    score = 1.0
    for deviation in deviations:
        if "scope exceeds" in deviation:
            score -= 0.45
        elif "cited outside" in deviation:
            score -= 0.45
        else:
            score -= 0.2
    return max(0.0, round(score, 3))


def _reachable_from_study(study: Study) -> dict[str, CorpusItem]:
    ids = set(study.catalog_pmids) | set(study.pmids)
    if not ids:
        return {}
    source_scope = study.source_scope.model_copy(deep=True)
    if source_scope == EvidenceScope():
        source_scope = study.claimed_scope.model_copy(deep=True)
    return {
        source_id: CorpusItem(
            item_id=source_id,
            kind="real",
            text=study.rationale,
            rationale=study.rationale,
            direction="NEUTRAL",
            cited_ids=[source_id],
            resolved_real_ids=[source_id],
            resolved_locators=[f"PMID:{source_id}"],
            scope=source_scope.model_copy(deep=True),
        )
        for source_id in ids
    }


def _study_process_hash(study: Study) -> str:
    payload = study.model_dump(mode="json")
    payload.pop("output_hash", None)
    blob = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
