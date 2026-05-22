from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

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


# Patent SpC-02 WARN: a CLAIM that generalizes is one whose claimed scope
# spans more than a "narrow" band. Declared thresholds (configurable per
# §10.2 of the patent's domain-specific extension): a population band wider
# than 30 years or a timeframe band wider than 10 years counts as generalizing.
GENERALIZATION_POPULATION_SPAN_YEARS = 30
GENERALIZATION_TIMEFRAME_SPAN_YEARS = 10
# Patent SpC-02 WARN: minimum aggregate evidence sample size for a generalizing
# claim. Declared in the rule body (Tier 3 §SpC-02) and configurable. Default
# 100 reflects a minimum size before broad clinical generalization.
MIN_SAMPLE_FOR_GENERALIZATION = 100
# Patent GC-01 BLOCK: total WARN accumulation that escalates to BLOCK.
# Default = 5 per patent §Tier-5 GC-01.
WARN_ACCUMULATION_BLOCK = 5

Severity = Literal["block", "warn"]


@dataclass(frozen=True)
class ProcessViolation:
    """A single patent-rule violation, severity-tagged.

    ``code`` matches the patent rule code (e.g. ``IC-02``, ``SpC-02``,
    ``GC-01``). ``message`` is the human-readable reason rendered in audit
    events. Severity determines whether the violation blocks alone (BLOCK)
    or accumulates toward GC-01 (WARN).
    """

    code: str
    severity: Severity
    message: str


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
    violations: list[ProcessViolation] = field(default_factory=list)
    warn_count: int = 0
    block_count: int = 0
    gc01_escalated: bool = False


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
        missing = ProcessViolation(
            code="MISSING-PLAN",
            severity="block",
            message="No ResearchPlan/PIR trace was recorded for this study.",
        )
        return ProcessAssessment(
            study_id=study.id,
            passed=False,
            civer_passed=False,
            brim_passed=False,
            integrity_score=0.0,
            threshold=threshold,
            reasons=[missing.message],
            execution_deviations=["missing-plan"],
            violations=[missing],
            warn_count=0,
            block_count=1,
            gc01_escalated=False,
        )

    plan = study.research_plan
    reachable_lookup = _reachable_from_study(study)
    plan_result = admit_research_plan(
        plan=plan,
        claim_graph=claim_graph,
        reachable_lookup=reachable_lookup,
    )
    civer_passed = plan_result.admitted
    civer_reasons = plan_result.reasons

    violations: list[ProcessViolation] = []
    # CIVER pre-execution rule violations (BLOCK = plan-time refusal; WARN =
    # rule that accumulates toward GC-01 without blocking on its own).
    for block_msg in plan_result.blocks:
        violations.append(
            ProcessViolation(code="CIVER", severity="block", message=block_msg)
        )
    for warn_msg in plan_result.warns:
        violations.append(
            ProcessViolation(code="CIVER", severity="warn", message=warn_msg)
        )

    # BRIM plan→execution violations (severity per patent semantics: cite
    # outside plan = HARKing-like, scope exceeds = SpC-01 at execution time,
    # both BLOCK; SpC-02 small-n generalizing claim = WARN).
    brim_violations = execution_deviations(plan=plan, study=study)
    violations.extend(brim_violations)

    score = process_integrity_score(civer_passed=civer_passed, violations=violations)
    block_count = sum(1 for v in violations if v.severity == "block")
    warn_count = sum(1 for v in violations if v.severity == "warn")
    gc01_escalated = warn_count >= WARN_ACCUMULATION_BLOCK
    brim_passed = score >= threshold and block_count == 0 and not gc01_escalated
    passed = civer_passed and brim_passed

    deviation_msgs = [v.message for v in brim_violations]
    reasons = list(civer_reasons)
    if brim_violations:
        reasons.append(
            "BRIM detected plan-to-execution deviation: " + "; ".join(deviation_msgs)
        )
    else:
        reasons.append("BRIM detected no plan-to-execution deviation.")
    if gc01_escalated:
        reasons.append(
            f"Patent GC-01 BLOCK: WARN accumulation ({warn_count}) "
            f">= threshold ({WARN_ACCUMULATION_BLOCK}); escalated to BLOCK."
        )
    reasons.append(
        f"Final process integrity score={score:.3f} threshold={threshold:.3f} "
        f"(blocks={block_count}, warns={warn_count})."
    )
    return ProcessAssessment(
        study_id=study.id,
        passed=passed,
        civer_passed=civer_passed,
        brim_passed=brim_passed,
        integrity_score=score,
        threshold=threshold,
        reasons=reasons,
        execution_deviations=deviation_msgs,
        violations=violations,
        warn_count=warn_count,
        block_count=block_count,
        gc01_escalated=gc01_escalated,
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


@dataclass(frozen=True)
class OutputCheckAssessment:
    """SCAFFOLDING-ONLY output check verdict — NOT a CIVER process claim.

    Used when a free-arm study has no recorded ResearchPlan (legacy bundles
    generated BEFORE the free arm was instrumented to record plan→execution
    traces, e.g. Run 4 under main@525782a). Reports the MedEvo environment
    checks (citation resolvability, claimed-vs-source scope) so an existing
    artifact still yields a usable pass/fail split — but SPEC v3 §0a is
    explicit that these checks are scaffolding/environment, NOT the patent
    CIVER process-validity claim. Any report using this assessment MUST label
    the verdict as output-fallback, never as "CIVER pass".
    """

    study_id: str
    passed: bool
    cites_resolve: bool
    scope_within_source: bool
    reasons: list[str]


def assess_output_level_fallback(*, study: Study) -> OutputCheckAssessment:
    """Output-only scaffolding check for studies missing a ResearchPlan trace.

    NOT process-CIVER. Validates two output-level properties only: every cited
    PMID resolves in the study's catalog snapshot, AND the claimed scope does
    not exceed the source scope beyond ``SCOPE_TOLERANCE_YEARS``. Used by the
    shadow analyzer's fallback lane to extract a useful pass/fail split from
    legacy bundles; the report flags every fallback verdict with the
    analysis_mode field so downstream tooling cannot silently mistake it for a
    real process-CIVER pass.
    """
    catalog = set(study.catalog_pmids or [])
    cites_resolve = bool(study.pmids) and all(pmid in catalog for pmid in study.pmids)
    scope_within = not study.claimed_scope.exceeds(
        study.source_scope, tolerance=SCOPE_TOLERANCE_YEARS
    )
    reasons: list[str] = []
    if not study.pmids:
        reasons.append("Output-fallback: no cited PMIDs supplied.")
    elif cites_resolve:
        reasons.append("Output-fallback: every cited PMID resolves in the catalog snapshot.")
    else:
        reasons.append("Output-fallback: one or more cited PMIDs absent from catalog snapshot.")
    if scope_within:
        reasons.append("Output-fallback: claimed scope within source scope (tolerance applied).")
    else:
        reasons.append("Output-fallback: claimed scope exceeds source scope beyond tolerance.")
    reasons.append(
        "NOTE: scaffolding check only — SPEC v3 §0a, not the CIVER process-validity claim."
    )
    return OutputCheckAssessment(
        study_id=study.id,
        passed=cites_resolve and scope_within,
        cites_resolve=cites_resolve,
        scope_within_source=scope_within,
        reasons=reasons,
    )


def execution_deviations(*, plan: ResearchPlan, study: Study) -> list[ProcessViolation]:
    """Post-execution BRIM monitor + Tier-3 SpC-02 check.

    Returns severity-tagged ``ProcessViolation`` entries. BRIM rules block
    (HARKing-like cite outside plan + scope creep beyond plan); SpC-02 is
    a WARN that accumulates toward patent GC-01.
    """
    violations: list[ProcessViolation] = []
    committed = set(plan.committed_pmids)
    out_of_plan = [pmid for pmid in study.pmids if pmid not in committed]
    if out_of_plan:
        violations.append(
            ProcessViolation(
                code="IC-02",
                severity="block",
                message="cited outside committed plan: " + ", ".join(out_of_plan),
            )
        )
    if study.claimed_scope.exceeds(plan.claimed_scope, tolerance=0):
        violations.append(
            ProcessViolation(
                code="SpC-01",
                severity="block",
                message="execution scope exceeds the registered plan scope",
            )
        )
    spc02 = _spc02_sample_size_warn(study)
    if spc02 is not None:
        violations.append(spc02)
    return violations


def _spc02_sample_size_warn(study: Study) -> ProcessViolation | None:
    """Patent SpC-02 WARN: claim generalizes but evidence sample size is below
    the declared minimum threshold. Generalization = claimed_scope spans a
    population band > ``GENERALIZATION_POPULATION_SPAN_YEARS`` OR a timeframe
    band > ``GENERALIZATION_TIMEFRAME_SPAN_YEARS``. Sample size = aggregate
    ``study.n`` (when reported); when no ``n`` is extractable, the rule
    abstains (rather than fire false positive on missing data)."""
    if study.n is None:
        return None
    pop_span = study.claimed_scope.population_high - study.claimed_scope.population_low
    year_span = study.claimed_scope.year_end - study.claimed_scope.year_start
    generalizes = (
        pop_span > GENERALIZATION_POPULATION_SPAN_YEARS
        or year_span > GENERALIZATION_TIMEFRAME_SPAN_YEARS
    )
    if not generalizes:
        return None
    if study.n >= MIN_SAMPLE_FOR_GENERALIZATION:
        return None
    return ProcessViolation(
        code="SpC-02",
        severity="warn",
        message=(
            f"claim generalizes (pop_span={pop_span}y, year_span={year_span}y) "
            f"but evidence sample size n={study.n} < threshold "
            f"{MIN_SAMPLE_FOR_GENERALIZATION}"
        ),
    )


def process_integrity_score(
    *,
    civer_passed: bool,
    violations: list[ProcessViolation],
) -> float:
    """Patent-aligned process integrity score.

    Rules:
      * CIVER refused at plan time → 0.0 (cannot release).
      * Any BLOCK-severity violation → 0.0.
      * GC-01: WARN count >= ``WARN_ACCUMULATION_BLOCK`` → 0.0 (escalation).
      * Otherwise: 1.0 − 0.1 × warn_count (each WARN drops 0.1).

    Replaces the pre-patent flat −0.45-per-deviation formula. Severity tags
    let the release gate distinguish "instant block" from "accumulating WARN
    that the rule body allows below threshold".
    """
    if not civer_passed:
        return 0.0
    if any(v.severity == "block" for v in violations):
        return 0.0
    warn_count = sum(1 for v in violations if v.severity == "warn")
    if warn_count >= WARN_ACCUMULATION_BLOCK:
        return 0.0
    return max(0.0, round(1.0 - 0.1 * warn_count, 3))


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
