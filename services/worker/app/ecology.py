from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Literal

from app.agents import DEFAULT_FAILURE_RATE, ResearchAgent, SrmaAgent
from app.synthesis import admit_guideline_output
from app.config import YEARS
from app.db import insert_ecology_records, insert_guideline_claims, insert_tier3_study
from app.harness import branch_gap, replay_counts
from app.llm import LLMClient, llm_cache_stats
from app.microdata import MicrodataAgent, supports_claim as microdata_supports_claim
from app.models import (
    ArtifactBundle,
    AuditEvent,
    BranchName,
    BrimEvent,
    CalibrationMatrix,
    ClaimDirection,
    ClaimGraph,
    ClaimSnapshot,
    CiverVerdict,
    DriftSnapshot,
    EvidenceScope,
    EvidenceUnit,
    ExecutionWarrant,
    GuidelineClaim,
    LineageRecord,
    PubMedRecord,
    RecommendationStrength,
    ResearchPlan,
    RunRequestModel,
    Study,
)
from app.pubmed import DeterministicPubMedClient, PubMedClient


ANCHORS = [
    "Pre-2023 literature contamination approximated near zero.",
    "Rising AI-text prevalence in biomedical publishing treated as empirical anchor.",
    "Every year-10/20/30 panel is rendered as one draw from a distribution, never a forecast.",
]

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 <= value <= 1.0 else default


CLAIM_LIMIT = _env_int("MEDEVO_CLAIM_LIMIT", 50)
REAL_SOURCES_PER_CLAIM = 4
# Tier-1 study replicates emitted per (claim, era). SPEC §3/§12: the phenomenon
# shows at ~tens of studies, and a real SR/MA needs a screenable corpus, not one
# study per claim. With CLAIM_LIMIT=50 claims and len(YEARS)=3 eras, k=2 yields
# up to 50 x 3 x 2 = 300 studies per arm across the run; shorter inputs still emit
# fewer claims. Declared as one named constant, never a magic literal in the loop.
STUDIES_PER_CLAIM_PER_ERA = _env_int("MEDEVO_STUDIES_PER_CLAIM_PER_ERA", 2)
# DEFAULT_FAILURE_RATE (imported from app.agents) is the weak-agent failure
# fraction placeholder; SPEC §11-A anchors it to A0 in a later slice. It drives
# the EMERGENT ungrounded-study rate, NOT a harness injection rate.
RELEASE_THRESHOLD = _env_float("MEDEVO_RELEASE_THRESHOLD", 0.60)
# Article I scope clause tolerance (years). A claimed scope wider than the
# evidence's by MORE than this is refused; a mild over-reach within tolerance
# slips the gate (the gate is imperfect, not tautological — FNR can be > 0).
# Declared here as the single source for the predicate (audit §8.2: no magic
# literal buried in logic). Paired with agents.SCOPE_INFLATION_MIN/MAX.
SCOPE_TOLERANCE_YEARS = 2
# Repair-loop budget (SPEC Endpoint 4 — refuse+repair, not kill-only). After a
# pre-execution refusal, the constrained agent gets this many revise attempts
# within the SAME retrieved catalog (no re-retrieval — cost + determinism). A
# successful revise emits a `design-repaired` audit event; exhausting all
# attempts emits `design-abstain-persistent`. Free arm is observational and
# does not enter this loop.
MAX_PLAN_REVISIONS = _env_int("MEDEVO_MAX_PLAN_REVISIONS", 2)
# Active-arm output matching. E3 is interpretable only when the constrained arm
# retains a corpus close to the free arm at the claim-year cell where guidelines
# are synthesized; matching only Tier-1 attempt counts is not enough.
OUTPUT_MATCH_TARGET_RATIO = _env_float("MEDEVO_OUTPUT_MATCH_TARGET_RATIO", 0.85)
OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO = _env_float(
    "MEDEVO_OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO", 0.80
)
MAX_CONSTRAINED_ATTEMPTS_PER_CELL = _env_int(
    "MEDEVO_MAX_CONSTRAINED_ATTEMPTS_PER_CELL",
    max(STUDIES_PER_CLAIM_PER_ERA * 3, 6),
)
GENESIS_HASH = "GENESIS"
PUBMED_FORWARD_CEILING_YEAR = 2025
# NHANES 2005-2006 data ends in 2006; do not use for simulated years before that.
_MICRODATA_EARLIEST_SIMULATED_YEAR = 2006
_DIRECTION_VALUE = {"SUPPORTS": 1.0, "NEUTRAL": 0.0, "REFUTES": -1.0}


@dataclass(frozen=True)
class ClaimSeed:
    claim_id: str
    text: str
    seed_strength: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    claim_id: str
    label: str
    locator: str
    direction: ClaimDirection
    text: str


@dataclass(frozen=True)
class CorpusItem:
    item_id: str
    kind: Literal["real", "prior", "synthetic"]
    text: str
    rationale: str
    direction: ClaimDirection
    cited_ids: list[str]
    resolved_real_ids: list[str]
    resolved_locators: list[str]
    scope: EvidenceScope = field(default_factory=EvidenceScope)
    # Canonical MeSH descriptors PubMed attached to the underlying real record.
    # Empty for non-real items. Read by SpC-04 to gate outcome coherence.
    mesh_terms: list[str] = field(default_factory=list)


@dataclass
class BranchState:
    prior_direction: ClaimDirection = "NEUTRAL"
    prior_strength: RecommendationStrength = "weak"
    citation_memory: list[str] = field(default_factory=list)
    surviving_real: set[str] = field(default_factory=set)
    output_history: list[EvidenceUnit] = field(default_factory=list)


@dataclass
class CallTrace:
    label: str
    seed: int
    prompt_digest: str
    response_hash: str
    timestamp: str


@dataclass
class CallTelemetry:
    call_count: int = 0
    degradation_reason: str | None = None
    traces: list[CallTrace] = field(default_factory=list)
    transient_failures: list[dict] = field(default_factory=list)

    def record_failure(self, label: str, exc: BaseException) -> None:
        """Record a per-call failure without poisoning the run-level flag.

        Single-call API failures (transient 400/timeout from upstream model
        endpoint) are absorbed by the output-matching retry loop and do not
        invalidate the overall corpus. The scientific flag is decided at the
        end based on total transient failure rate vs tolerance.
        """
        self.transient_failures.append(
            {"label": label, "kind": type(exc).__name__, "message": str(exc)[:300]}
        )

    @property
    def transient_failure_rate(self) -> float:
        if self.call_count <= 0:
            return 0.0
        return len(self.transient_failures) / self.call_count


# Tolerance for transient per-call failures before the run is flagged
# non-scientific. Default 2% — anything beyond suggests systemic upstream
# breakage, not normal endpoint flakiness.
TRANSIENT_FAILURE_TOLERANCE = _env_float("MEDEVO_TRANSIENT_FAILURE_TOLERANCE", 0.02)


def contamination_clock(year: int) -> float:
    # Retained ONLY as a non-injection sensitivity-band scale (see _panel_band).
    # It no longer drives any contamination/injection rate (v2 role removed).
    return round(1 / (1 + math.exp(-0.115 * (year - 18))), 3)


def horizon_years(request: RunRequestModel) -> list[int]:
    years = request.horizons or list(YEARS)
    cleaned = sorted({int(year) for year in years if int(year) > 0})
    return cleaned or list(YEARS)


def pubmed_cutoff_year(year: int) -> int:
    return year if year >= 1900 else PUBMED_FORWARD_CEILING_YEAR


def _seed_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def _digest_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_ungrounded_citation_id(cited_id: str) -> bool:
    return cited_id.startswith("S-") or "-syn-" in cited_id


def _carries_ungrounded_substrate(unit: EvidenceUnit) -> bool:
    return (
        unit.provenance == "UNGROUNDED"
        or not unit.resolved_real_ids
        or any(_is_ungrounded_citation_id(cited_id) for cited_id in unit.cited_ids)
    )


def _panel_band(year: int, branch_scores: list[float]) -> dict[str, float | str]:
    contamination = contamination_clock(year)
    band_mid = fmean(branch_scores) if branch_scores else 0.0
    return {
        "low": round(band_mid - contamination * 0.28, 3),
        "high": round(band_mid + contamination * 0.28, 3),
        "label": "Sensitivity band scaled by contamination-clock pressure.",
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _event_payload(event: AuditEvent) -> dict[str, Any]:
    payload = event.model_dump()
    payload.pop("current_state_hash", None)
    return payload


def verify_audit_chain(events: list[AuditEvent]) -> bool:
    streams: dict[tuple[str, BranchName], list[AuditEvent]] = {}
    for event in events:
        streams.setdefault((event.claim_id, event.branch), []).append(event)

    for stream in streams.values():
        previous_hash = GENESIS_HASH
        expected_index = 1
        for event in stream:
            if event.event_index != expected_index:
                return False
            if event.previous_state_hash != previous_hash:
                return False
            current_hash = hashlib.sha256(
                (previous_hash + _canonical_json(_event_payload(event))).encode("utf-8")
            ).hexdigest()
            if event.current_state_hash != current_hash:
                return False
            previous_hash = current_hash
            expected_index += 1
    return True


def _unit_output_hash(unit: EvidenceUnit) -> str:
    payload = unit.model_dump()
    payload.pop("output_hash", None)
    return _canonical_sha256(payload)


def _bundle_payload(bundle: ArtifactBundle) -> dict[str, Any]:
    payload = bundle.model_dump()
    payload.pop("bundle_seal", None)
    calls = payload.get("provenance_log", {}).get("calls")
    if isinstance(calls, list):
        for item in calls:
            if isinstance(item, dict):
                item.pop("timestamp", None)
    return payload


def _is_valid_warrant(warrant: ExecutionWarrant | None, output: EvidenceUnit | None = None) -> bool:
    if warrant is None:
        return False
    if warrant.status != "ISSUED" or not warrant.issued:
        return False
    if warrant.integrity_score < warrant.threshold:
        return False
    if output is None:
        return True
    return warrant.output_hash == (output.output_hash or _unit_output_hash(output))


def _invoke_model(
    llm: LLMClient,
    telemetry: CallTelemetry,
    label: str,
    prompt: str,
    *,
    seed: int,
) -> str:
    was_scientific = llm.scientific
    telemetry.call_count += 1
    response = llm.generate(prompt, seed=seed)
    telemetry.traces.append(
        CallTrace(
            label=label,
            seed=seed,
            prompt_digest=_digest_key(prompt),
            response_hash=_digest_key(response),
            timestamp=datetime.now(UTC).isoformat(),
        )
    )
    if was_scientific and not llm.scientific and telemetry.degradation_reason is None:
        failure_reason = getattr(llm, "degradation_reason", None) or "model call degraded"
        telemetry.degradation_reason = f"{label}: {failure_reason}"
    return response


def _research_study_for_year(
    *,
    research_agent: ResearchAgent,
    claim: ClaimSeed,
    year: int,
    telemetry: CallTelemetry,
    replicate: int = 0,
) -> tuple[Study, list[PubMedRecord]]:
    try:
        return research_agent.run(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            simulated_year=year,
            max_pubmed_year=pubmed_cutoff_year(year),
            replicate=replicate,
        )
    except Exception as exc:
        telemetry.record_failure(f"pubmed/{claim.claim_id}/year-{year}", exc)
        study = Study(
            id=f"{claim.claim_id}-study-{year}-r{replicate}-pubmed-error",
            claim_id=claim.claim_id,
            year=year,
            direction="NEUTRAL",
            quality=0.0,
            provenance="UNGROUNDED",
            pmids=[],
            numeric=False,
            rationale=f"PubMed retrieval failed for cutoff {pubmed_cutoff_year(year)}.",
        )
        study.output_hash = _study_output_hash(study)
        return study, []


def _use_microdata_group(*, claim: ClaimSeed, year: int) -> bool:
    # Temporal guard: NHANES 2005-2006 data must not be used for absolute-year
    # retro runs before the data existed (year 2000 < 2006 → skip). Forward
    # simulations use relative years (< 1900) and are exempt from this guard
    # since the simulation epoch is the future.
    if 1900 <= year < _MICRODATA_EARLIEST_SIMULATED_YEAR:
        return False
    if not microdata_supports_claim(claim.text):
        return False
    bucket = int(
        hashlib.sha256(f"microdata-slot:{claim.claim_id}:{year}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    return bucket % 2 == 0


def _study_is_temporally_consistent(study: "Study", simulated_year: int) -> bool:
    """Shared anti-speculation gate (BOTH arms).

    Returns False when a study's data source post-dates the simulated year,
    meaning the agent used evidence that did not yet exist. Only enforced for
    absolute-year (≥ 1900) retro runs; forward-simulation relative years
    (< 1900) always pass since we can't flag future data in a future scenario.
    Distinct from CIVER/BRIM process validation.
    """
    if simulated_year < 1900:
        return True  # forward-simulation: no temporal constraint applicable
    return study.source_scope.year_end <= simulated_year


def _study_passes_medevo_environment(study: "Study") -> tuple[bool, str]:
    """Baseline MedEvo evidence-validity layer shared by BOTH arms.

    The free arm is free from CIVER/BRIM process control, not free to inject
    invalid evidence. Every study entering either Tier-3 corpus must cite at
    least one retrieved source, all cited sources must resolve in the date-cut
    catalog, and the emitted scope must stay inside the cited evidence.
    """
    if not study.pmids:
        return False, "MedEvo environment rejected study: no cited PubMed/source id."
    catalog = set(study.catalog_pmids)
    unresolved = [pmid for pmid in study.pmids if pmid not in catalog]
    if unresolved:
        return (
            False,
            "MedEvo environment rejected study: cited ids absent from retrieved catalog: "
            + ", ".join(unresolved),
        )
    if study.claimed_scope.exceeds(study.source_scope, tolerance=0):
        return (
            False,
            "MedEvo environment rejected study: claimed scope exceeds cited evidence scope.",
        )
    return True, "MedEvo environment accepted source-resolved, scope-bounded evidence."


@dataclass(frozen=True)
class ResearchOutcome:
    """One Tier-1 attempt's result for a SINGLE branch.

    ``study`` is None ONLY for a constrained attempt whose DESIGN was refused by the
    pre-execution gate and could not be repaired within ``MAX_PLAN_REVISIONS`` —
    that attempt never executed, so no Study exists. ``catalog`` is the source
    universe the gate resolves cites against. ``design_refused`` /
    ``design_reasons`` carry the pre-execution audit detail; ``execution_deviated``
    / ``deviation_note`` carry the Article-II plan→execution deviation (WARN).
    ``revision_attempts`` counts how many revise calls were issued (0 = initial
    plan admitted); ``revision_history`` records the per-attempt admit verdict +
    reasons so post-hoc analysis can separate friction-cost from kill-cost.
    ``persistent_abstain`` is True only when all revisions exhausted without admit.
    """

    study: Study | None
    catalog: list[PubMedRecord]
    plan: ResearchPlan | None = None
    design_admitted: bool | None = None
    design_refused: bool = False
    design_reasons: list[str] = field(default_factory=list)
    execution_deviated: bool = False
    deviation_note: str = ""
    revision_attempts: int = 0
    revision_history: list[dict] = field(default_factory=list)
    persistent_abstain: bool = False


def _free_research_batch(
    *,
    research_agent: ResearchAgent,
    microdata_agent: MicrodataAgent,
    claim: ClaimSeed,
    claim_graph: ClaimGraph,
    year: int,
    telemetry: CallTelemetry,
) -> list[ResearchOutcome]:
    """FREE/natural arm: direct research output under MedEvo environment rules.

    Free means no CIVER/BRIM pre-execution process control. It does NOT mean
    invalid evidence can enter the corpus: citation resolvability, source scope,
    and temporal validity are enforced later by the shared MedEvo environment.
    """
    outcomes: list[ResearchOutcome] = []
    for replicate in range(STUDIES_PER_CLAIM_PER_ERA):
        if replicate == 0 and _use_microdata_group(claim=claim, year=year):
            study, catalog = microdata_agent.run(
                claim_id=claim.claim_id, claim_text=claim.text, simulated_year=year
            )
            outcomes.append(
                ResearchOutcome(
                    study=study,
                    catalog=catalog,
                    plan=study.research_plan,
                    design_admitted=True,
                    design_reasons=["Group-A microdata plan observed; no active gate in free branch."],
                )
            )
            continue
        try:
            study, catalog = research_agent.run(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                simulated_year=year,
                max_pubmed_year=pubmed_cutoff_year(year),
                replicate=replicate,
            )
        except Exception as exc:
            telemetry.record_failure(f"free-research/{claim.claim_id}/year-{year}", exc)
            study = Study(
                id=f"{claim.claim_id}-study-{year}-r{replicate}-process-error",
                claim_id=claim.claim_id,
                year=year,
                direction="NEUTRAL",
                quality=0.0,
                provenance="UNGROUNDED",
                pmids=[],
                numeric=False,
                rationale=f"Natural research process failed for cutoff {pubmed_cutoff_year(year)}.",
            )
            study.output_hash = _study_output_hash(study)
            outcomes.append(
                ResearchOutcome(
                    study=study,
                    catalog=[],
                    plan=None,
                    design_admitted=None,
                    design_reasons=[f"Natural research failed: {type(exc).__name__}: {exc}"],
                )
            )
            continue
        outcomes.append(
            ResearchOutcome(
                study=study,
                catalog=catalog,
                plan=None,
                design_admitted=None,
                design_reasons=["Free branch: direct MedEvo research output; no active CIVER design gate."],
            )
        )
    return outcomes


def _constrained_research_batch(
    *,
    research_agent: ResearchAgent,
    microdata_agent: MicrodataAgent,
    claim: ClaimSeed,
    claim_graph: ClaimGraph,
    year: int,
    telemetry: CallTelemetry,
    attempt_count: int | None = None,
) -> list[ResearchOutcome]:
    """CONSTRAINED arm: DESIGN → pre-execution CIVER gate → (REVISE×N) → EXECUTE.

    The asymmetry is intentional (the gate REQUIRES a separable design step): the
    agent pre-registers a plan, the gate admits/refuses it BEFORE any execution.
    Refused plans enter a REPAIR loop (Endpoint 4 — refuse+repair, not kill-only):
    the agent gets the structured refusal reasons and revises within the same
    catalog. After ``MAX_PLAN_REVISIONS`` exhausted refusals the attempt
    persistently abstains; a successful early revise emits ``design-repaired``.
    Microdata (Group A) is effectively pre-registered already — the dataset-slice
    id is the committed source and the analyzed cohort is the scope — so it
    stays a single call on both arms (design/execute split there would be
    gold-plating).
    """
    outcomes: list[ResearchOutcome] = []
    for replicate in range(
        STUDIES_PER_CLAIM_PER_ERA if attempt_count is None else max(attempt_count, 0)
    ):
        outcomes.append(
            _constrained_research_attempt(
                research_agent=research_agent,
                microdata_agent=microdata_agent,
                claim=claim,
                claim_graph=claim_graph,
                year=year,
                telemetry=telemetry,
                replicate=replicate,
            )
        )
    return outcomes


def _constrained_research_attempt(
    *,
    research_agent: ResearchAgent,
    microdata_agent: MicrodataAgent,
    claim: ClaimSeed,
    claim_graph: ClaimGraph,
    year: int,
    telemetry: CallTelemetry,
    replicate: int,
) -> ResearchOutcome:
    if replicate == 0 and _use_microdata_group(claim=claim, year=year):
        study, catalog = microdata_agent.run(
            claim_id=claim.claim_id, claim_text=claim.text, simulated_year=year
        )
        return ResearchOutcome(study=study, catalog=catalog)
    try:
        plan, catalog = research_agent.run_design(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            simulated_year=year,
            max_pubmed_year=pubmed_cutoff_year(year),
            replicate=replicate,
        )
    except Exception as exc:
        telemetry.record_failure(f"design/{claim.claim_id}/year-{year}", exc)
        return ResearchOutcome(
            study=None,
            catalog=[],
            design_refused=True,
            design_reasons=[f"DESIGN retrieval failed: {type(exc).__name__}: {exc}"],
            persistent_abstain=True,
        )

    reachable_lookup = _reachable_lookup_from_catalog(catalog)
    admitted, reasons = admit_research_plan(
        plan=plan, claim_graph=claim_graph, reachable_lookup=reachable_lookup
    )
    revision_history: list[dict] = [
        {
            "attempt": 0,
            "plan_id": plan.plan_id,
            "admitted": admitted,
            "reasons": list(reasons),
        }
    ]
    revision_attempts = 0
    while not admitted and revision_attempts < MAX_PLAN_REVISIONS:
        revision_attempts += 1
        try:
            plan = research_agent.run_revise(
                prior_plan=plan,
                refusal_reasons=reasons,
                catalog=catalog,
                claim_text=claim.text,
                revision=revision_attempts,
            )
        except Exception as exc:
            telemetry.record_failure(
                f"revise/{claim.claim_id}/year-{year}/rev{revision_attempts}", exc
            )
            revision_history.append(
                {
                    "attempt": revision_attempts,
                    "plan_id": "",
                    "admitted": False,
                    "reasons": [f"REVISE call failed: {type(exc).__name__}: {exc}"],
                }
            )
            break
        admitted, reasons = admit_research_plan(
            plan=plan, claim_graph=claim_graph, reachable_lookup=reachable_lookup
        )
        revision_history.append(
            {
                "attempt": revision_attempts,
                "plan_id": plan.plan_id,
                "admitted": admitted,
                "reasons": list(reasons),
            }
        )

    if not admitted:
        # All revisions exhausted: persistent abstain. No execution, no study.
        return ResearchOutcome(
            study=None,
            catalog=catalog,
            design_refused=True,
            design_reasons=reasons,
            revision_attempts=revision_attempts,
            revision_history=revision_history,
            persistent_abstain=True,
        )

    study = research_agent.run_execute(
        plan=plan, catalog=catalog, claim_text=claim.text, replicate=replicate
    )
    # Article II — execution deviation from the registered plan (WARN): the
    # execute step cited a PMID outside the committed set, or widened the scope
    # beyond the registered plan. Made visible; the release gate / scope clause
    # may also catch gross over-reach (do not double-revoke here).
    deviations = _plan_execution_deviations(plan=plan, study=study)
    return ResearchOutcome(
        study=study,
        catalog=catalog,
        plan=plan,
        design_admitted=True,
        design_reasons=reasons,
        execution_deviated=bool(deviations),
        deviation_note="; ".join(deviations),
        revision_attempts=revision_attempts,
        revision_history=revision_history,
        persistent_abstain=False,
    )


def _plan_execution_deviations(*, plan: ResearchPlan, study: Study) -> list[str]:
    """Audit-trail-friendly string view of post-execution violations.

    Delegates to the unified ``process_gate.execution_deviations`` (which now
    returns severity-tagged ``ProcessViolation`` entries including patent
    SpC-02) so the audit message reflects every active BRIM + Tier-3 rule
    without duplicating code. Import is local to avoid a circular module
    dependency with process_gate.
    """
    from app.process_gate import execution_deviations as _exec_dev

    return [violation.message for violation in _exec_dev(plan=plan, study=study)]


def record_transition(
    *,
    audit_trail: list[AuditEvent],
    audit_counters: dict[tuple[str, BranchName], int],
    last_hashes: dict[tuple[str, BranchName], str],
    run_id: str,
    claim_id: str,
    branch: BranchName,
    year: int,
    phase: str,
    event_type: str,
    severity: Literal["info", "warn", "block"],
    integrity_score_before: float,
    integrity_score_after: float,
    message: str,
) -> AuditEvent:
    key = (claim_id, branch)
    event_index = audit_counters.get(key, 0) + 1
    previous_state_hash = last_hashes.get(key, GENESIS_HASH)
    event = AuditEvent(
        run_id=run_id,
        claim_id=claim_id,
        branch=branch,
        year=year,
        event_index=event_index,
        phase=phase,
        previous_state_hash=previous_state_hash,
        current_state_hash="",
        event_type=event_type,
        severity=severity,
        integrity_score_before=round(integrity_score_before, 3),
        integrity_score_after=round(integrity_score_after, 3),
        message=message,
    )
    event.current_state_hash = hashlib.sha256(
        (previous_state_hash + _canonical_json(_event_payload(event))).encode("utf-8")
    ).hexdigest()
    audit_counters[key] = event_index
    last_hashes[key] = event.current_state_hash
    audit_trail.append(event)
    return event


def admit_research_plan(
    *,
    plan: "ResearchPlan",
    claim_graph: ClaimGraph,
    reachable_lookup: dict[str, CorpusItem],
) -> "PlanAdmissionResult":
    """PRE-EXECUTION CIVER gate (CONSTITUTION Article I) on a constrained-arm
    DESIGN plan, BEFORE the agent executes.

    Returns a PlanAdmissionResult (iterable as ``(admitted, reasons)`` for
    backward-compat with old callers). The result also carries per-severity
    BLOCK/WARN lists so post-execution scoring (process_gate) can apply patent
    GC-01 WARN-accumulation escalation.

    Admits the plan to execute ONLY if every BLOCK-level constitutional rule
    holds (no BLOCK violation present):
      * SC-01..05 / GC-02 — Q→M→E→A→C node chain present;
      * Method design parseable + coherent;
      * Article I — every committed source resolves in the catalog;
      * SpC-01 / SpC-03 — committed scope within source scope and bounded;
      * IC-01 (patent §IC-01) — every ANALYSIS node has an ANALYZES edge to
        an EVIDENCE node (no claim built on a half-connected analysis);
      * GC-02 — there is a full QUESTION→…→CLAIM path through the graph.

    WARN-level patent rules (do not block individually; counted toward GC-01
    in the post-execution release-gate score):
      * IC-03 — multiple CLAIM nodes share an ANALYSIS parent (structural
        prerequisite of multi-claim scope conflict).

    BLINDNESS (SPEC §8.3): this reads ONLY the plan's structure, its committed
    ids, its claimed scope, the claim graph, and the catalog. It accepts NO
    field that reveals ground-truth provenance/failure_mode.
    """
    required_nodes = {"QUESTION", "METHOD", "EVIDENCE", "ANALYSIS", "CLAIM"}
    graph_complete = required_nodes.issubset({node.node_type for node in claim_graph.nodes})
    method_coherent = plan.parse_ok and bool(plan.method.strip())
    # Honest abstain (committed_pmids=[]) is NOT a CIVER violation — the agent
    # explicitly declined to commit to any source. The committed-resolve and
    # scope-within-sources rules are vacuously true for the empty set. Refusal
    # only when the agent CLAIMS sources that don't resolve or scope exceeds
    # the sources it DID commit to.
    committed_resolve = all(
        pmid in reachable_lookup for pmid in plan.committed_pmids
    )
    committed_items = [
        reachable_lookup[pmid] for pmid in plan.committed_pmids if pmid in reachable_lookup
    ]
    scope = plan.claimed_scope
    scope_bounded = (
        scope.population_low <= scope.population_high
        and scope.year_start <= scope.year_end
    )
    # Scope clause: the claim's envelope must lie inside the UNION of the
    # committed sources' coverages (within SCOPE_TOLERANCE_YEARS). A per-item
    # all() refused any claim that extended past the narrowest committed source,
    # which spuriously fired whenever PubMed records carry year_start=year_end=
    # pub_year (a 1-yr "coverage" window): a multi-PMID citation spanning
    # 2002-2024 would refuse the natural claim year_start=2002 because each
    # individual source's year_start equals its own pub year. The aggregate
    # envelope is the right semantic: "claim is supported by at least one source
    # on each edge".
    if committed_resolve and committed_items:
        scoped_items = [item for item in committed_items if hasattr(item, "scope")]
        if scoped_items:
            agg = EvidenceScope(
                population_low=min(item.scope.population_low for item in scoped_items),
                population_high=max(item.scope.population_high for item in scoped_items),
                year_start=min(item.scope.year_start for item in scoped_items),
                year_end=max(item.scope.year_end for item in scoped_items),
            )
            scope_within_committed_sources = not scope.exceeds(
                agg, tolerance=SCOPE_TOLERANCE_YEARS
            )
        else:
            scope_within_committed_sources = True
    else:
        scope_within_committed_sources = committed_resolve
    # SC-02 (spec Tier-1) + GC-02 (spec Tier-5): every ANALYSIS node must have
    # at least one ANALYZES edge to an EVIDENCE node, AND there must exist a
    # full path QUESTION → … → CLAIM in the graph. Either failing = incomplete
    # analysis link from CLAIM to underlying evidence. The medevo code
    # previously mis-labelled the SC-02 check as "Patent IC-01" — the spec's
    # IC-01 is a Tier-4 sequencing rule (CLAIM exists before its ANALYSIS link
    # is wired ≈ HARKing), not a structural-completeness rule.
    ic01_ok, ic01_reason = _ic01_analysis_links_to_evidence(claim_graph)
    gc02_ok, gc02_reason = _gc02_full_chain_path_exists(claim_graph)

    # Patent IC-03 WARN: multiple CLAIM nodes sharing an ANALYSIS parent (via
    # SUPPORTS edges) is the structural prerequisite for multi-claim scope
    # conflict. Population/scope mismatch is a Tier-3 follow-up; this rule
    # surfaces the structural risk regardless.
    ic03_warn_reason = _ic03_multi_claim_scope_conflict(claim_graph)

    reasons: list[str] = []
    blocks: list[str] = []
    warns: list[str] = []

    def _record(passed: bool, ok_msg: str, fail_msg: str, *, severity: str = "block") -> None:
        if passed:
            reasons.append(ok_msg)
            return
        reasons.append(fail_msg)
        if severity == "block":
            blocks.append(fail_msg)
        else:
            warns.append(fail_msg)

    _record(
        graph_complete,
        "Question→Method→Evidence→Analysis→Claim design chain present.",
        "Design is missing one or more required constitutional nodes.",
    )
    _record(
        method_coherent,
        "Method design is coherent with the question.",
        "Method design is incoherent or unparseable (no executable plan).",
    )
    _record(
        committed_resolve,
        "Every committed source resolves in the retrieved catalog.",
        "One or more committed sources do not resolve in the catalog (Article I pre-execution).",
    )
    _record(
        scope_bounded,
        "Committed scope is bounded.",
        "Committed scope is unbounded or degenerate.",
    )
    _record(
        scope_within_committed_sources,
        "Committed scope is within the committed source evidence.",
        "Committed scope exceeds the committed source evidence (Article I pre-execution scope clause).",
    )
    _record(
        ic01_ok,
        "SC-02: every ANALYSIS node links to EVIDENCE via ANALYZES.",
        f"SC-02 BLOCK: {ic01_reason}",
    )
    _record(
        gc02_ok,
        "GC-02: graph carries a full QUESTION→…→CLAIM path.",
        f"GC-02 BLOCK: {gc02_reason}",
    )
    if ic03_warn_reason:
        warns.append(ic03_warn_reason)
        reasons.append(f"IC-03 WARN: {ic03_warn_reason}")

    # CIVER 2.0 Tier-2 AC-01..AC-03: attribute constraints across linked PIR
    # nodes. AC-01 is BLOCK (foundational — wrong study design); AC-02, AC-03
    # are WARN (declarative gaps that accumulate toward GC-01).
    ac01_ok, ac01_reason = _ac01_study_type_match(plan)
    _record(
        ac01_ok,
        "AC-01: QUESTION.study_type matches METHOD.study_type.",
        ac01_reason or "AC-01 BLOCK: question/method study_type mismatch",
    )
    ac02_reason = _ac02_variables_match(plan)
    if ac02_reason:
        warns.append(ac02_reason)
        reasons.append(ac02_reason)
    ac03_reason = _ac03_statistical_method_compat(plan)
    if ac03_reason:
        warns.append(ac03_reason)
        reasons.append(ac03_reason)
    spc04_ok, spc04_reason = _spc04_evidence_measures_claim_outcome(
        plan, claim_graph, reachable_lookup
    )
    _record(
        spc04_ok,
        "SpC-04: cited evidence measures the claim's outcome.",
        spc04_reason or "SpC-04 BLOCK: evidence does not measure the claim's outcome",
    )

    # CIVER 2.0 Tier-5 GC-03: Integrity Score gating. A plan that accumulates
    # too many WARNs (each shaves 0.08 off IS) can fail even with no individual
    # BLOCK. Plans that hit BLOCK rules are already refused; GC-03 catches the
    # "many small violations" pattern that local rules cannot.
    complete_chains = _count_complete_chains(claim_graph)
    integrity_score = _integrity_score(
        blocks=list(blocks), warns=list(warns), complete_chains=complete_chains
    )
    is_gated = integrity_score < _IS_GATING_THRESHOLD
    if is_gated:
        gc03_msg = (
            f"GC-03 BLOCK: Integrity Score {integrity_score:.3f} < threshold "
            f"{_IS_GATING_THRESHOLD:.2f}"
        )
        blocks.append(gc03_msg)
        reasons.append(gc03_msg)
    else:
        reasons.append(
            f"GC-03: Integrity Score {integrity_score:.3f} ≥ threshold "
            f"{_IS_GATING_THRESHOLD:.2f}."
        )

    admitted = (
        graph_complete
        and method_coherent
        and committed_resolve
        and scope_bounded
        and scope_within_committed_sources
        and ic01_ok
        and gc02_ok
        and ac01_ok
        and spc04_ok
        and not is_gated
    )
    return PlanAdmissionResult(
        admitted=admitted,
        reasons=reasons,
        blocks=blocks,
        warns=warns,
        integrity_score=integrity_score,
    )


@dataclass(frozen=True)
class PlanAdmissionResult:
    """Structured CIVER pre-execution gate result.

    Iterable as ``(admitted, reasons)`` for backward-compat with existing
    callers that unpack a tuple. Carries per-severity BLOCK/WARN lists so
    process_gate can apply patent GC-01 (WARN accumulation BLOCK) during the
    post-execution release score, and so audit-trail messages can render the
    correct severity per rule.

    ``integrity_score`` is the spec §7 deterministic IS used as GC-03 gating
    condition (default threshold 0.60). It is a pure function of the PIR
    (blocks, warns, complete chain count); same inputs → same IS.
    """

    admitted: bool
    reasons: list[str]
    blocks: list[str]
    warns: list[str]
    integrity_score: float = 1.0

    def __iter__(self):
        yield self.admitted
        yield self.reasons


def _ic01_analysis_links_to_evidence(claim_graph: ClaimGraph) -> tuple[bool, str]:
    """CIVER 2.0 spec SC-02 BLOCK: every ANALYSIS node must be linked to
    EVIDENCE via an ANALYZES edge. Direction-agnostic (process-flow
    EVIDENCE→ANALYSIS and analyser-perspective ANALYSIS→EVIDENCE both count):
    a CLAIM resting on an ANALYSIS that has NO ANALYZES connection to any
    EVIDENCE is structurally incomplete.

    Function name retained as ``_ic01_`` for back-compat with the test that
    imports it; the rule it implements is the spec's Tier-1 SC-02."""
    analysis_ids = {n.id for n in claim_graph.nodes if n.node_type == "ANALYSIS"}
    evidence_ids = {n.id for n in claim_graph.nodes if n.node_type == "EVIDENCE"}
    if not analysis_ids or not evidence_ids:
        # Tier-1 chain check catches the missing-node case; IC-01 only fires
        # when both kinds exist but the link is absent.
        return True, ""
    orphans: list[str] = []
    for analysis_id in sorted(analysis_ids):
        has_evidence_link = any(
            edge.edge_type == "ANALYZES"
            and (
                (edge.source == analysis_id and edge.target in evidence_ids)
                or (edge.target == analysis_id and edge.source in evidence_ids)
            )
            for edge in claim_graph.edges
        )
        if not has_evidence_link:
            orphans.append(analysis_id)
    if orphans:
        return False, (
            "ANALYSIS node(s) missing ANALYZES edge to EVIDENCE: "
            + ", ".join(orphans)
        )
    return True, ""


def _gc02_full_chain_path_exists(claim_graph: ClaimGraph) -> tuple[bool, str]:
    """Patent GC-02 BLOCK: the graph must carry a reachable directed path
    from at least one QUESTION to at least one CLAIM through the constitutional
    edge types. Catches a graph that lists all required node types but never
    actually wires them together."""
    type_by_id = {n.id: n.node_type for n in claim_graph.nodes}
    adjacency: dict[str, set[str]] = {n.id: set() for n in claim_graph.nodes}
    for edge in claim_graph.edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
    question_ids = [nid for nid, t in type_by_id.items() if t == "QUESTION"]
    claim_ids = {nid for nid, t in type_by_id.items() if t == "CLAIM"}
    if not question_ids or not claim_ids:
        return True, ""  # Tier-1 chain rule catches missing nodes.
    for start in question_ids:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in claim_ids:
                return True, ""
            stack.extend(adjacency.get(node, set()))
    return False, "no directed path QUESTION→…→CLAIM in the graph"


# --- CIVER 2.0 spec Tier-2 + Tier-5 implementation ----------------------------
#
# The medevo CIVER originally checked only graph structure (SC/GC) and scope
# (SpC). Spec §5.3 Tier-2 Attribute Constraints (AC-01..AC-03) and §7 Integrity
# Score gating (GC-03) were missing — a methodologically empty METHOD step
# passed the gate as long as the graph nodes existed. The block below adds the
# missing tiers so a plan whose METHOD is "review sources and conclude" with no
# declared study_type can no longer slip through.

# Canonical study-type vocabulary the prompt asks the agent to use. The AC-01
# comparison is case-/whitespace-/punctuation-tolerant after this normalisation.
_STUDY_TYPE_NORMALISE_RE = re.compile(r"[\s_\-]+")


def _normalise_study_type(value: str | None) -> str | None:
    if not value:
        return None
    norm = _STUDY_TYPE_NORMALISE_RE.sub("-", value.strip().lower())
    return norm or None


# Compatibility map: which statistical_method values are compatible with which
# EVIDENCE study_type / data_type. The map is intentionally permissive (any
# unknown combination = "compatible" so spurious WARNs don't dominate); it only
# fires on KNOWN mismatches (e.g. linear regression on binary outcome).
_STAT_COMPAT: dict[str, set[str]] = {
    # Aggregating effects across multiple studies
    "random-effects-meta-analysis": {"cohort", "case-control", "rct", "diagnostic-accuracy", "cross-sectional"},
    "fixed-effects-meta-analysis": {"cohort", "case-control", "rct"},
    "hazard-ratio-pool": {"cohort", "rct"},
    "risk-ratio-pool": {"cohort", "rct", "case-control"},
    "odds-ratio-pool": {"case-control", "cross-sectional"},
    # Diagnostic
    "mcnemar": {"diagnostic-accuracy"},
    "roc-analysis": {"diagnostic-accuracy"},
    "kappa-agreement": {"diagnostic-accuracy"},
    # Narrative / single-study
    "narrative-synthesis": {"cohort", "case-control", "rct", "case-series", "cross-sectional", "diagnostic-accuracy", "narrative-review"},
    # Continuous-outcome methods only on continuous outcomes — flagged via
    # variables, not study_type; default = permissive for now.
}


def _ac01_study_type_match(plan: "ResearchPlan") -> tuple[bool, str]:
    """CIVER 2.0 Tier-2 AC-01 BLOCK: QUESTION.study_type and METHOD.study_type
    must agree. A causal-cohort question cannot be answered by a case-series
    method; a diagnostic-accuracy question cannot be answered by a narrative
    review. Fires only when BOTH sides declared (permissive on missing data —
    the prompt asks for them, but a sparse emission shouldn't escalate)."""
    q = _normalise_study_type(plan.question_attrs.study_type)
    m = _normalise_study_type(plan.method_attrs.study_type)
    if q is None or m is None:
        return True, ""
    if q == m:
        return True, ""
    return False, (
        f"AC-01: QUESTION.study_type={q!r} ≠ METHOD.study_type={m!r} — "
        "the method's design cannot answer the question's design"
    )


def _ac02_variables_match(plan: "ResearchPlan") -> str:
    """CIVER 2.0 Tier-2 AC-02 WARN: variables declared by METHOD must appear
    in the EVIDENCE variable set. Returns the WARN reason, or empty string if
    the rule holds / is vacuous. Permissive on missing data."""
    method_vars = {v.lower() for v in (plan.method_attrs.variables or [])}
    evidence_vars = {v.lower() for v in (plan.evidence_attrs.variables or [])}
    if not method_vars or not evidence_vars:
        return ""
    missing = sorted(method_vars - evidence_vars)
    if not missing:
        return ""
    return (
        f"AC-02 WARN: METHOD variables {missing} not declared in EVIDENCE "
        "variables — planned variables exceed the data the sources support"
    )


# Markers after which a clinical claim names its outcome/endpoint. The capture
# runs to the first boundary word that ends the outcome phrase.
_CLAIM_OUTCOME_RE = re.compile(
    r"\b(?:risk|incidence|rates?|mortality|probability|odds|hazard|likelihood)\s+"
    r"(?:of|for|from)\s+(.+?)"
    r"(?:\s+(?:by|in|among|when|during|through|via|with|after|due|"
    r"because|that|which|and|or)\b|[.,;:]|$)",
    re.IGNORECASE,
)


def claim_outcome_phrase(claim_text: str) -> str:
    """Extract the claim's clinical OUTCOME phrase verbatim (e.g. 'coronary heart
    disease' from 'reduces risk of coronary heart disease by ...')."""
    phrases = _CLAIM_OUTCOME_RE.findall(claim_text or "")
    return phrases[0].strip() if phrases else ""


def _spc04_evidence_measures_claim_outcome(
    plan: "ResearchPlan",
    claim_graph: ClaimGraph,
    reachable_lookup: dict[str, "CorpusItem"],
) -> tuple[bool, str]:
    """Clinical domain extension of CIVER 2.0 Tier-3 (scope) — SpC-04 BLOCK.

    Spec Tier-3 forbids a CLAIM from exceeding the EVIDENCE it rests on. This
    rule applies that principle to the OUTCOME dimension: the cited PMIDs must
    be PubMed-indexed under a MeSH descriptor that is the claim's outcome OR a
    DESCENDANT of it (standard SR `[MeSH]` explosion semantics).

    The agent's QUESTION/EVIDENCE outcome attribute is PINNED here from canonical
    NLM sources — NOT from the agent's self-reported `evidence_attrs.variables`
    (free-form text the agent can drift to fit any groundable PMID — exactly the
    HARK pattern). The claim outcome MeSH tree and each cited record's MeSH tree
    are both resolved through `app.mesh`, giving an ontology-grounded, generic
    match across all biomedical claims (CHD / MI / T2DM / cancer subtypes /
    etc.) with no per-vocabulary hand-coding.

    Architecture (mindset confirmed with sếp 2026-05-28): real CIVER has
    'human-in-the-loop' confirming structured attributes; MedEvo substitutes
    'medevo-rule-in-the-loop' — the harness pins attributes from canonical
    sources so the agent cannot redefine them. SpC-04 is the structural-gate
    surface of that substitution for the outcome attribute.

    Fires only when (a) the claim outcome resolves to a MeSH descriptor AND
    (b) at least one cited record carries MeSH terms. Otherwise vacuous
    (permissive on missing data — recent uncindexed articles, niche outcomes —
    consistent with the other attribute rules and the spec's permissive default)."""
    from app.mesh import (
        claim_outcome_trees,
        evidence_mesh_trees,
        mesh_hierarchy_match,
    )

    outcome_phrase = claim_outcome_phrase(claim_graph.claim_text)
    if not outcome_phrase:
        return True, ""
    claim_trees = claim_outcome_trees(outcome_phrase)
    if not claim_trees:
        return True, ""
    cited_mesh_terms: list[str] = []
    for pmid in plan.committed_pmids:
        item = reachable_lookup.get(pmid)
        cited_mesh_terms.extend(getattr(item, "mesh_terms", None) or [])
    if not cited_mesh_terms:
        return True, ""
    evidence_trees = evidence_mesh_trees(cited_mesh_terms)
    if mesh_hierarchy_match(claim_trees, evidence_trees):
        return True, ""
    cited_mesh_summary = sorted(set(cited_mesh_terms))[:8]
    return False, (
        f"SpC-04: claim outcome {outcome_phrase!r} (MeSH tree {claim_trees}) "
        f"is not measured by the cited evidence — MeSH terms attached: "
        f"{cited_mesh_summary}. The cited indexing does not exist under or "
        "equal the claim's outcome descriptor (off-endpoint / drifted question)"
    )


def _ac03_statistical_method_compat(plan: "ResearchPlan") -> str:
    """CIVER 2.0 Tier-2 AC-03 WARN: ANALYSIS.statistical_method must be
    compatible with the EVIDENCE study_type / data type. Fires on KNOWN
    mismatches in the compatibility map; unknown combos are treated as
    compatible (permissive)."""
    method = _normalise_study_type(plan.analysis_attrs.statistical_method)
    evid = _normalise_study_type(
        plan.evidence_attrs.study_type or plan.method_attrs.study_type
    )
    if method is None or evid is None:
        return ""
    allowed = _STAT_COMPAT.get(method)
    if allowed is None:
        return ""  # method not in our compatibility table — don't fire
    if evid in allowed:
        return ""
    return (
        f"AC-03 WARN: ANALYSIS.statistical_method={method!r} is not "
        f"compatible with EVIDENCE.study_type={evid!r}"
    )


# Spec §7 penalty weights. Medevo's BLOCK rules map to "BLOCK" weight (0.15);
# AC-01 maps to "BLOCK (foundational)" (0.25) because mismatching method to
# question is a load-bearing methodology error, not a surface issue.
_PENALTY_WARN_LIGHT = 0.03
_PENALTY_WARN_SIGNIFICANT = 0.08
_PENALTY_BLOCK = 0.15
_PENALTY_BLOCK_FOUNDATIONAL = 0.25
_COMPLETENESS_BONUS_PER_CHAIN = 0.01
_COMPLETENESS_BONUS_CAP = 0.10
_IS_GATING_THRESHOLD = 0.60  # spec §7 default GC-03


def _integrity_score(*, blocks: list[str], warns: list[str], complete_chains: int) -> float:
    """CIVER 2.0 §7 deterministic Integrity Score.

    IS = 1.0 - Σ(penalty × weight) + Σ(bonus), clamped to [0, 1].

    The function is intentionally a pure function of (blocks, warns,
    complete_chains) — same inputs always produce the same IS, no LLM in the
    loop. ``blocks`` entries containing 'AC-01' are treated as foundational
    (0.25 penalty); other BLOCKs use 0.15. WARNs default to 0.08 (significant);
    a future refinement may split light vs significant — for now every WARN is
    treated as 'significant' to be conservative against borderline plans.
    """
    score = 1.0
    for block in blocks:
        if "AC-01" in block:
            score -= _PENALTY_BLOCK_FOUNDATIONAL
        else:
            score -= _PENALTY_BLOCK
    for _warn in warns:
        score -= _PENALTY_WARN_SIGNIFICANT
    bonus = min(_COMPLETENESS_BONUS_CAP, complete_chains * _COMPLETENESS_BONUS_PER_CHAIN)
    score += bonus
    return max(0.0, min(1.0, round(score, 4)))


def _count_complete_chains(claim_graph: ClaimGraph) -> int:
    """Count distinct QUESTION→METHOD→EVIDENCE→ANALYSIS→CLAIM paths that exist
    in the graph (spec §7 completeness bonus). One path per (Q, C) pair counts
    once. Direction-tolerant on intermediate edges so a process-flow wiring
    (EVIDENCE→ANALYSIS via PRODUCES) is recognised the same as the
    analyser-perspective ANALYSIS→EVIDENCE.

    For the medevo claim graph (1 Q, 1 M, 1 E, 1 A, 1 C, fully connected by
    the constitutional edges) this returns 1 — earning the minimum +0.01 bonus.
    """
    nodes_by_type: dict[str, list[str]] = {}
    for n in claim_graph.nodes:
        nodes_by_type.setdefault(n.node_type, []).append(n.id)
    if not all(t in nodes_by_type for t in ("QUESTION", "METHOD", "EVIDENCE", "ANALYSIS", "CLAIM")):
        return 0
    return min(len(nodes_by_type["QUESTION"]), len(nodes_by_type["CLAIM"]))


def _ic03_multi_claim_scope_conflict(claim_graph: ClaimGraph) -> str:
    """Patent IC-03 WARN: multiple CLAIM nodes sharing an ANALYSIS parent
    (via SUPPORTS edges) is the structural prerequisite for multi-claim
    population/scope conflict. The semantic population/scope mismatch sits in
    Tier 3 (SpC) once per-node scope attributes exist; this rule surfaces the
    structural risk irrespective of whether scope tags are present.
    """
    claim_ids = [n.id for n in claim_graph.nodes if n.node_type == "CLAIM"]
    if len(claim_ids) < 2:
        return ""
    # SUPPORTS edges point ANALYSIS → CLAIM (per ClaimEdge type union).
    analysis_to_claims: dict[str, set[str]] = {}
    for edge in claim_graph.edges:
        if edge.edge_type != "SUPPORTS":
            continue
        if edge.target in claim_ids:
            analysis_to_claims.setdefault(edge.source, set()).add(edge.target)
    shared = [
        f"{analysis_id}→{{{', '.join(sorted(claims))}}}"
        for analysis_id, claims in sorted(analysis_to_claims.items())
        if len(claims) >= 2
    ]
    if not shared:
        return ""
    return (
        "multiple CLAIM nodes share an ANALYSIS parent (SUPPORTS): "
        + "; ".join(shared)
        + " — multi-claim scope conflict structurally possible"
    )


def admit_evidence_unit(
    *,
    run_id: str,
    claim: ClaimSeed,
    claim_graph: ClaimGraph,
    branch: BranchName,
    year: int,
    unit: EvidenceUnit,
    reachable_lookup: dict[str, CorpusItem],
    warrants_by_output: dict[str, ExecutionWarrant],
    threshold: float,
) -> tuple[CiverVerdict, ExecutionWarrant | None]:
    if branch == "free":
        verdict = CiverVerdict(
            node_id=unit.id,
            passed=True,
            reasons=["Experimental branch: execution warrant not enforced."],
            certificate_id=None,
        )
        return verdict, None

    # Gate blindness (CONSTITUTION §1; SPEC §8.3): admission reads ONLY the claim
    # graph, the unit's cited ids, the claimed scope, and the authoritative
    # catalog (reachable_lookup) / prior warrants. It NEVER reads unit.provenance
    # or unit.failure_mode — those are ground-truth labels the harness keeps for
    # scoring only and must not reach the gate or corpus selection.
    required_nodes = {"QUESTION", "METHOD", "EVIDENCE", "ANALYSIS", "CLAIM"}
    graph_node_types = {node.node_type for node in claim_graph.nodes}
    graph_complete = required_nodes.issubset(graph_node_types)
    cited_items = [reachable_lookup[cited_id] for cited_id in unit.cited_ids if cited_id in reachable_lookup]
    cited_resolve = len(cited_items) == len(unit.cited_ids)
    resolvable = all(
        item.kind == "real"
        or _is_valid_warrant(warrants_by_output.get(item.item_id))
        for item in cited_items
    )
    # Article I scope clause: the claim's envelope must lie inside the UNION of
    # the cited evidence's authoritative source scopes (within tolerance). This
    # catches Mode-2 over-reach (real PMID, over-claimed envelope) without
    # spuriously refusing a multi-PMID citation whose individual sources each
    # cover only one publication year.
    scoped_cited = [item for item in cited_items if hasattr(item, "scope")]
    if scoped_cited:
        agg_cited = EvidenceScope(
            population_low=min(item.scope.population_low for item in scoped_cited),
            population_high=max(item.scope.population_high for item in scoped_cited),
            year_start=min(item.scope.year_start for item in scoped_cited),
            year_end=max(item.scope.year_end for item in scoped_cited),
        )
        scope_within_evidence = not unit.claimed_scope.exceeds(
            agg_cited, tolerance=SCOPE_TOLERANCE_YEARS
        )
    else:
        scope_within_evidence = True
    passed = (
        graph_complete
        and cited_resolve
        and resolvable
        and scope_within_evidence
        and bool(unit.cited_ids)
    )

    # Defect B fix: warrant integrity is the real-provenance fraction of the
    # unit's citations, not a structural pass/fail binary. A well-formed unit
    # carrying synthetic contamination scores low -> warrant invalid (< threshold)
    # -> excluded from the constrained corpus while free still ingests it.
    if cited_items:
        real_cites = sum(
            1 for item in cited_items if item.kind == "real" or item.resolved_real_ids
        )
        provenance_score = real_cites / len(cited_items)
    else:
        provenance_score = 0.0

    reasons = []
    if graph_complete:
        reasons.append("Traceable question-method-evidence-analysis-claim chain present.")
    else:
        reasons.append("Claim graph missing one or more required constitutional nodes.")
    if not unit.cited_ids:
        reasons.append("No cited evidence unit was supplied for Article I resolvability.")
    elif cited_resolve and resolvable:
        reasons.append("Every cited evidence unit resolves to a catalog source or valid prior warrant.")
    else:
        reasons.append("One or more cited evidence units failed Article I resolvability.")
    if scope_within_evidence:
        reasons.append("Claim scope is within the cited evidence's supported population and timeframe.")
    else:
        reasons.append("Claim scope exceeds the cited evidence's supported scope (Article I scope clause).")

    warrant = ExecutionWarrant(
        id=f"W-{claim.claim_id}-{branch}-{year}",
        output_id=unit.id,
        output_hash=unit.output_hash or _unit_output_hash(unit),
        run_id=run_id,
        claim_id=claim.claim_id,
        branch=branch,
        year=year,
        status="ISSUED" if passed else "REFUSED",
        issued=False,
        integrity_score=provenance_score if passed else 0.0,
        threshold=threshold,
    )
    verdict = CiverVerdict(
        node_id=unit.id,
        passed=passed,
        reasons=reasons,
        certificate_id=warrant.id if passed else None,
    )
    if not passed:
        warrant.status = "REFUSED"
        warrant.issued = False
        warrant.integrity_score = 0.0
    return verdict, warrant


def _clean_integrity_score(events: list[AuditEvent], scientific: bool) -> float:
    if not scientific:
        return 0.0
    if not verify_audit_chain(events):
        return 0.0
    penalties = 0.0
    for event in events:
        # article-i-refused, design-refused and guideline-refused are EXPECTED gate
        # refusals (Article I pre-execution at the design boundary, Article I at the
        # study-input boundary, Article I/IV at the SRMA-output boundary) — the gate
        # doing its job, not process-integrity drift. They must not depress the
        # release-gate score of subsequent admissible studies in the same
        # (claim, branch) stream (Article II/III double-counting, CONSTITUTION §4).
        if event.severity == "block" and event.event_type not in (
            "article-i-refused",
            "design-refused",
            "design-abstain-persistent",
            "guideline-refused",
        ):
            penalties += 0.8
    return max(0.0, round(1.0 - penalties, 3))


def _apply_release_gate(
    *,
    branch: BranchName,
    warrant: ExecutionWarrant | None,
    claim_events: list[AuditEvent],
    scientific: bool,
    threshold: float,
) -> tuple[ExecutionWarrant | None, bool, str]:
    if branch == "free" or warrant is None:
        return warrant, True, "Release gate observational only in free branch."

    # A unit refused at admission (Article I) stays refused: the release gate
    # never resurrects an unresolvable / chain-broken output. This is what keeps
    # ungrounded studies out of the constrained corpus even on non-scientific
    # demo runs (previously a non-scientific override stamped REFUSED warrants
    # ISSUED, letting ungrounded studies leak into constrained — gate bypass).
    if warrant.status == "REFUSED":
        warrant.issued = False
        warrant.integrity_score = 0.0
        warrant.threshold = threshold
        return warrant, False, "Release gate upheld the Article I refusal; unresolvable output is not released."

    cleaned_score = _clean_integrity_score(claim_events, scientific)
    warrant.integrity_score = min(warrant.integrity_score, cleaned_score)
    warrant.threshold = threshold
    if not scientific:
        warrant.status = "ISSUED"
        warrant.issued = True
        warrant.integrity_score = max(threshold, 1.0)
        return warrant, True, "Release gate preserved illustrative output, but the run is non-scientific and cannot be scored."
    if not verify_audit_chain(claim_events):
        warrant.status = "REVOKED"
        warrant.issued = False
        warrant.integrity_score = 0.0
        return warrant, False, "Release gate revoked output because the audit chain failed verification."
    warrant.status = "ISSUED"
    warrant.issued = True
    return warrant, warrant.integrity_score >= warrant.threshold, "Release gate issued a valid execution warrant."


def _lineage_record(
    *,
    claim_id: str,
    year: int,
    branch: BranchName,
    prior_state: BranchState,
    surviving_units: list[EvidenceUnit],
    verdict_before: ClaimDirection,
    verdict_after: ClaimDirection,
) -> LineageRecord:
    surviving_real = sorted(
        {
            real_id
            for unit in surviving_units
            for real_id in unit.resolved_real_ids
        }
    )
    lost_real = sorted(prior_state.surviving_real.difference(surviving_real))
    ungrounded_carriers = [
        unit.id
        for unit in surviving_units
        if _carries_ungrounded_substrate(unit)
    ]
    return LineageRecord(
        claim_id=claim_id,
        year=year,
        branch=branch,
        surviving_real=surviving_real,
        lost_real=lost_real,
        ungrounded_carriers=ungrounded_carriers,
        verdict_before=verdict_before,
        verdict_after=verdict_after,
    )


def compute_calibration_matrix(
    observations: list[tuple[str, bool]],
    *,
    branch: BranchName = "constrained",
) -> CalibrationMatrix:
    """Confusion matrix of gate verdict vs TRUE provenance (SPEC §7c).

    ``observations`` = (true_provenance, gate_admitted) per scored study. True
    provenance is the harness's ground truth, used for scoring only; the gate
    that produced ``gate_admitted`` never saw it (§8.3). FN = admitted-but-
    ungrounded (gate missed contamination); FP = refused-but-grounded.
    """
    tp = tn = fn = fp = 0
    grounded = ungrounded = 0
    for provenance, admitted in observations:
        if provenance == "GROUNDED":
            grounded += 1
            if admitted:
                tp += 1
            else:
                fp += 1
        else:
            ungrounded += 1
            if admitted:
                fn += 1
            else:
                tn += 1
    return CalibrationMatrix(
        branch=branch,
        true_positive=tp,
        true_negative=tn,
        false_negative=fn,
        false_positive=fp,
        grounded_total=grounded,
        ungrounded_total=ungrounded,
        fnr=round(fn / ungrounded, 4) if ungrounded else 0.0,
        fpr=round(fp / grounded, 4) if grounded else 0.0,
    )


def _study_output_hash(study: Study) -> str:
    payload = study.model_dump(mode="json")
    payload.pop("output_hash", None)
    return _canonical_sha256(payload)


def _study_to_evidence_unit(
    *,
    study: Study,
    branch: BranchName,
    catalog_pmids: set[str],
) -> EvidenceUnit:
    # Gate blindness: resolved_real_ids is computed by CATALOG INTERSECTION, not
    # from study.provenance. A Mode-2 over-reach cites a real (resolvable) PMID
    # and so carries it here; a Mode-1 fabricated PMID is not in the catalog and
    # drops out. The provenance label never gates resolvability.
    cited_ids = list(study.pmids)
    resolved = [pmid for pmid in cited_ids if pmid in catalog_pmids]
    unit = EvidenceUnit(
        id=study.id,
        claim_id=study.claim_id,
        year=study.year,
        branch=branch,
        producer="investigator",
        cited_ids=cited_ids,
        provenance=study.provenance,
        direction=study.direction,
        rationale=study.rationale,
        resolved_real_ids=resolved,
        resolved_locators=[_source_locator(source_id) for source_id in resolved],
        claimed_scope=study.claimed_scope.model_copy(deep=True),
        output_hash=study.output_hash,
    )
    return unit


def _source_records_from_study(study: Study) -> list[SourceRecord]:
    return [
        SourceRecord(
            source_id=source_id,
            claim_id=study.claim_id,
            label=_source_label(source_id),
            locator=_source_locator(source_id),
            direction=study.direction,
            text=study.rationale,
        )
        for source_id in study.pmids
    ]


def _source_label(source_id: str) -> str:
    if source_id.startswith("NHANES:"):
        return f"NHANES dataset slice {source_id}"
    return f"PubMed source {source_id}"


def _source_locator(source_id: str) -> str:
    if ":" in source_id:
        return source_id
    return f"PMID:{source_id}"


def _reachable_lookup_from_catalog(catalog: list[PubMedRecord]) -> dict[str, CorpusItem]:
    """The authoritative source universe the agent actually retrieved.

    Built from the catalog records (not the study's claimed pmids), so a
    fabricated Mode-1 cite is absent (fails resolvability) and a real Mode-2 cite
    is present with its TRUE source scope (so the scope clause does the work).
    """
    return {
        record.pmid: CorpusItem(
            item_id=record.pmid,
            kind="real",
            text=record.abstract or record.title,
            rationale=record.abstract or record.title,
            direction="NEUTRAL",
            cited_ids=[record.pmid],
            resolved_real_ids=[record.pmid],
            resolved_locators=[record.locator or f"PMID:{record.pmid}"],
            scope=record.scope.model_copy(deep=True),
            mesh_terms=list(record.mesh_terms),
        )
        for record in catalog
    }


class Tier3RunStore:
    def __init__(self, *, run_id: str | None) -> None:
        self.run_id = run_id
        self._studies: dict[BranchName, list[Study]] = {"free": [], "constrained": []}

    def insert(
        self,
        *,
        branch: BranchName,
        study: Study,
        warrant: ExecutionWarrant | None = None,
        require_warrant: bool = False,
    ) -> bool:
        if self.run_id is not None:
            inserted = insert_tier3_study(
                run_id=self.run_id,
                branch=branch,
                study=study,
                warrant=warrant,
                require_warrant=require_warrant,
            )
        else:
            inserted = not require_warrant or _valid_study_warrant_for_study(study, warrant)
        if inserted:
            branch_studies = self._studies[branch]
            self._studies[branch] = [item for item in branch_studies if item.id != study.id]
            self._studies[branch].append(study)
        return inserted

    def list_studies(
        self,
        *,
        run_id: str,
        branch: BranchName,
        claim_id: str,
        up_to_year: int,
    ) -> list[Study]:
        return [
            study
            for study in self._studies[branch]
            if study.claim_id == claim_id and study.year <= up_to_year
        ]

    def count_studies(
        self,
        *,
        branch: BranchName,
        claim_id: str,
        year: int | None = None,
        up_to_year: int | None = None,
    ) -> int:
        studies = [study for study in self._studies[branch] if study.claim_id == claim_id]
        if year is not None:
            studies = [study for study in studies if study.year == year]
        if up_to_year is not None:
            studies = [study for study in studies if study.year <= up_to_year]
        return len(studies)

    def all_studies(self) -> dict[BranchName, list[Study]]:
        return {
            "free": list(self._studies["free"]),
            "constrained": list(self._studies["constrained"]),
        }


def _valid_study_warrant_for_study(
    study: Study,
    warrant: ExecutionWarrant | None,
) -> bool:
    if warrant is None:
        return False
    if warrant.status != "ISSUED" or not warrant.issued:
        return False
    if warrant.integrity_score < warrant.threshold:
        return False
    if warrant.output_id != study.id:
        return False
    return warrant.output_hash == study.output_hash


def _strength_from_guideline_level(level: str) -> RecommendationStrength:
    if level.startswith("strong"):
        return "strong"
    if level.startswith("conditional"):
        return "moderate"
    return "weak"


def _claim_snapshot(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    verdict: CiverVerdict,
    guideline: GuidelineClaim,
    synth_rationale: str,
    lineage: LineageRecord,
    cycle_events: list[AuditEvent],
    blocked_count: int,
    emitted_count: int,
) -> ClaimSnapshot:
    strength = _strength_from_guideline_level(guideline.level)
    brim_events = [
        BrimEvent(
            node_id=event.claim_id,
            event_type=event.event_type,
            severity="warn" if event.severity in {"warn", "block"} else "info",
            integrity_score=event.integrity_score_after,
            message=event.message,
        )
        for event in cycle_events[-3:]
    ]
    why_summary = (
        f"{branch.title()} branch at year {year}: panel moved from {lineage.verdict_before} to "
        f"{guideline.direction} ({guideline.level}, certainty={guideline.certainty}). "
        f"Real sources retained: {', '.join(lineage.surviving_real) or 'none'}. "
        f"Lost real sources: {', '.join(lineage.lost_real) or 'none'}. "
        f"Ungrounded carriers: {', '.join(lineage.ungrounded_carriers) or 'none'}. "
        f"Synthesist rationale: {synth_rationale}"
    )
    snapshot = ClaimSnapshot(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        direction=guideline.direction,
        strength=strength,
        emitted_count=emitted_count,
        blocked_count=blocked_count,
        divergence_score=0.0,
        why_summary=why_summary,
        civer=[verdict],
        brim=brim_events,
    )
    return snapshot


def _sentence_chunks(text: str) -> list[str]:
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    cleaned = []
    for raw in parts:
        item = " ".join(raw.strip().split())
        if len(item) >= 36:
            cleaned.append(item)
    return cleaned


def extract_claims(text: str, input_mode: str) -> list[ClaimSeed]:
    sentences = _sentence_chunks(text)
    if input_mode == "paper":
        preferred = [
            sentence
            for sentence in sentences
            if any(key in sentence.lower() for key in ("conclusion", "supports", "recommend"))
        ]
        if preferred:
            sentences = preferred + [sentence for sentence in sentences if sentence not in preferred]

    claims: list[ClaimSeed] = []
    for index, sentence in enumerate(sentences[:CLAIM_LIMIT], start=1):
        lowered = sentence.lower()
        if any(word in lowered for word in ("should", "recommended", "recommend", "must")):
            strength = "strong"
        elif any(word in lowered for word in ("consider", "may", "could")):
            strength = "weak"
        else:
            strength = "moderate"
        claims.append(ClaimSeed(f"claim-{index}", sentence, strength))

    if not claims:
        claims.append(
            ClaimSeed(
                "claim-1",
                "The submitted text did not contain enough structured guidance, so the demo collapsed it into a single neutral claim.",
                "weak",
            )
        )
    return claims


def mean_branch_divergence(bundle: ArtifactBundle) -> float:
    """Mean free-constrained divergence over all (year, claim) cells.

    ``branch_diff`` already holds the per-cell free-vs-constrained gap; this
    collapses it to one number so a sweep can report how the divergence responds
    to the failure-rate anchor.
    """
    cells = [value for per_year in bundle.branch_diff.values() for value in per_year.values()]
    return round(fmean(cells), 4) if cells else 0.0


def _output_match_summary(
    *,
    records: list[dict[str, Any]],
    guideline_timeline: dict[BranchName, list[GuidelineClaim]],
) -> dict[str, Any]:
    free_retained = sum(int(row["free_retained"]) for row in records)
    constrained_retained = sum(int(row["constrained_retained"]) for row in records)
    free_guidelines = {
        (g.claim_id, g.year)
        for g in guideline_timeline.get("free", [])
        if g.n_included > 0
    }
    constrained_guidelines = {
        (g.claim_id, g.year)
        for g in guideline_timeline.get("constrained", [])
        if g.n_included > 0
    }
    free_guideline_count = len(free_guidelines)
    constrained_guideline_count = len(constrained_guidelines)
    retained_ratio = (
        round(constrained_retained / free_retained, 4) if free_retained else 1.0
    )
    guideline_cell_ratio = (
        round(constrained_guideline_count / free_guideline_count, 4)
        if free_guideline_count
        else 0.0
    )
    achieved_cells = sum(1 for row in records if row["achieved"])
    has_real_guideline_comparison = (
        free_guideline_count > 0 and constrained_guideline_count > 0
    )
    return {
        "mode": "active-output-matched",
        "target_retained_ratio": OUTPUT_MATCH_TARGET_RATIO,
        "min_interpretable_ratio": OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO,
        "max_constrained_attempts_per_cell": MAX_CONSTRAINED_ATTEMPTS_PER_CELL,
        "cell_count": len(records),
        "achieved_cells": achieved_cells,
        "failed_cells": len(records) - achieved_cells,
        "free_retained_studies": free_retained,
        "constrained_retained_studies": constrained_retained,
        "retained_study_ratio": retained_ratio,
        "free_guideline_bearing_cells": free_guideline_count,
        "constrained_guideline_bearing_cells": constrained_guideline_count,
        "guideline_cell_ratio": guideline_cell_ratio,
        "paper_grade_interpretable": (
            has_real_guideline_comparison
            and retained_ratio >= OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO
            and guideline_cell_ratio >= OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO
        ),
        "records": records,
    }


def sweep_failure_rate(
    *,
    request: RunRequestModel,
    input_text: str,
    claim_graphs: list[ClaimGraph],
    llm: LLMClient,
    pubmed_client: PubMedClient | DeterministicPubMedClient,
    rates: list[float],
) -> list[dict[str, Any]]:
    """Sensitivity sweep over the A0-anchor failure-rate (SPEC §11-A / §7c).

    ``failure_rate`` is a placeholder for A0's measured LLM error rate (κ pending,
    not finalized), so we report how the free-constrained divergence and the gate
    error rates (FNR/FPR) respond as it varies — rather than committing to one
    hand-picked number. Each rate runs the full ecology and reports a row.
    """
    rows: list[dict[str, Any]] = []
    for rate in rates:
        bundle, _summary = run_ecology(
            request=request,
            input_text=input_text,
            claim_graphs=[graph.model_copy(deep=True) for graph in claim_graphs],
            llm=llm,
            pubmed_client=pubmed_client,
            run_id=None,
            failure_rate=rate,
        )
        matrix = bundle.calibration_matrix or CalibrationMatrix()
        rows.append(
            {
                "failure_rate": rate,
                "free_minus_constrained_divergence": mean_branch_divergence(bundle),
                "fnr": matrix.fnr,
                "fpr": matrix.fpr,
                "grounded_total": matrix.grounded_total,
                "ungrounded_total": matrix.ungrounded_total,
            }
        )
    return rows


def run_ecology(
    *,
    request: RunRequestModel,
    input_text: str,
    claim_graphs: list[ClaimGraph],
    llm: LLMClient,
    pubmed_client: PubMedClient | DeterministicPubMedClient,
    run_id: str | None = None,
    failure_rate: float = DEFAULT_FAILURE_RATE,
    study_sink: dict[BranchName, list[Study]] | None = None,
) -> tuple[ArtifactBundle, dict[str, Any]]:
    years = horizon_years(request)
    claims = extract_claims(input_text, request.input_mode)
    claims = claims[: len(claim_graphs)]
    telemetry = CallTelemetry()

    source_catalog: dict[str, list[SourceRecord]] = {claim.claim_id: [] for claim in claims}
    source_ids_seen: dict[str, set[str]] = {claim.claim_id: set() for claim in claims}
    invoke_model = lambda label, prompt, seed: _invoke_model(
        llm,
        telemetry,
        label,
        prompt,
        seed=seed,
    )
    research_agent = ResearchAgent(
        pubmed=pubmed_client,
        invoke_model=invoke_model,
        failure_rate=failure_rate,
        seed=_seed_int(f"run-failure-seed:{run_id or 'preview-run'}"),
    )
    microdata_agent = MicrodataAgent(invoke_model=invoke_model)
    tier3_store = Tier3RunStore(run_id=run_id)
    srma_agent = SrmaAgent(
        study_reader=tier3_store,
        llm=llm,
        invoke_model=invoke_model,
        seed_namespace=f"srma:{run_id or 'preview-run'}",
    )
    states: dict[tuple[str, BranchName], BranchState] = {
        (claim.claim_id, branch): BranchState()
        for claim in claims
        for branch in ("free", "constrained")
    }

    snapshots: dict[str, list[DriftSnapshot]] = {"free": [], "constrained": []}
    branch_diff: dict[str, dict[str, float]] = {}
    lineage_records: list[LineageRecord] = []
    # (true_provenance, gate_admitted) pairs for the constrained branch only.
    # true_provenance is read from the study for SCORING ONLY and is never passed
    # to process validation (gate blindness, SPEC §8.3).
    calibration_observations: list[tuple[str, bool]] = []
    evidence_units: list[EvidenceUnit] = []
    warrants: list[ExecutionWarrant] = []
    warrants_by_output: dict[str, ExecutionWarrant] = {}
    audit_trail: list[AuditEvent] = []
    audit_counters: dict[tuple[str, BranchName], int] = {}
    last_hashes: dict[tuple[str, BranchName], str] = {}
    graph_lookup = {graph.claim_id: graph for graph in claim_graphs}
    guideline_timeline: dict[BranchName, list[GuidelineClaim]] = {
        "free": [],
        "constrained": [],
    }
    db_growth: dict[str, Any] = {}
    population_stats: dict[str, Any] = {}
    output_match_records: list[dict[str, Any]] = []

    for year in years:
        branch_scores: dict[str, list[float]] = {"free": [], "constrained": []}
        branch_claims: dict[str, list[ClaimSnapshot]] = {"free": [], "constrained": []}
        branch_guidelines: dict[str, list[GuidelineClaim]] = {"free": [], "constrained": []}

        for claim in claims:
            for branch in ("free", "constrained"):
                state = states[(claim.claim_id, branch)]
                surviving_units: list[EvidenceUnit] = []
                # study ids that earned a valid execution warrant in this branch —
                # the inheritable set the guideline-output gate (Task A) checks.
                warranted_ids: set[str] = set()
                blocked_this_era = 0

                # Both arms record plan->execution traces. FREE/natural does not
                # enforce them (shadow CIVER/BRIM replays them post hoc);
                # CONSTRAINED enforces CIVER before execution and BRIM before
                # release, so a refused design never executes and yields no study.
                # NOTE: per-arm calls CAN be parallelized later (independent per
                # claim/replicate); not implemented now to keep determinism + the
                # telemetry call-count contract obvious.
                free_retained_this_cell = (
                    tier3_store.count_studies(
                        branch="free", claim_id=claim.claim_id, year=year
                    )
                    if branch == "constrained"
                    else 0
                )
                target_retained_this_cell = (
                    min(
                        free_retained_this_cell,
                        math.ceil(free_retained_this_cell * OUTPUT_MATCH_TARGET_RATIO),
                    )
                    if branch == "constrained"
                    else 0
                )
                constrained_attempt_cap = MAX_CONSTRAINED_ATTEMPTS_PER_CELL

                if branch == "free":
                    outcomes = _free_research_batch(
                        research_agent=research_agent,
                        microdata_agent=microdata_agent,
                        claim=claim,
                        claim_graph=graph_lookup[claim.claim_id],
                        year=year,
                        telemetry=telemetry,
                    )
                else:
                    outcomes = _constrained_research_batch(
                        research_agent=research_agent,
                        microdata_agent=microdata_agent,
                        claim=claim,
                        claim_graph=graph_lookup[claim.claim_id],
                        year=year,
                        telemetry=telemetry,
                        attempt_count=min(
                            STUDIES_PER_CLAIM_PER_ERA,
                            target_retained_this_cell,
                            constrained_attempt_cap,
                        ),
                    )

                catalog_pmids: set[str] = set()
                reachable_lookup: dict[str, CorpusItem] = {}
                for outcome in outcomes:
                    catalog_pmids.update(record.pmid for record in outcome.catalog)
                    reachable_lookup.update(_reachable_lookup_from_catalog(outcome.catalog))
                    if outcome.study is not None:
                        for source in _source_records_from_study(outcome.study):
                            if source.source_id not in source_ids_seen[claim.claim_id]:
                                source_catalog[claim.claim_id].append(source)
                                source_ids_seen[claim.claim_id].add(source.source_id)

                # Defaults for snapshot use after the loop; overwritten by last processed outcome.
                verdict = CiverVerdict(
                    node_id=f"{claim.claim_id}-no-study-{year}",
                    passed=branch == "free",
                    reasons=["No study admitted this era."],
                    certificate_id=None,
                )
                investigator = EvidenceUnit(
                    id=f"{claim.claim_id}-no-study-{year}",
                    claim_id=claim.claim_id,
                    year=year,
                    branch=branch,
                    producer="investigator",
                    cited_ids=[],
                    provenance="UNGROUNDED",
                    direction="NEUTRAL",
                    rationale="No study admitted this era.",
                    resolved_real_ids=[],
                    resolved_locators=[],
                    claimed_scope=EvidenceScope(),
                )
                # Tier-1 -> Tier-2 (CIVER) -> Tier-3 admission, per replicate study.
                # Constrained arm may append additional attempts, but only after
                # every currently scheduled attempt for this cell has been scored.
                outcome_index = 0
                while outcome_index < len(outcomes):
                    outcome = outcomes[outcome_index]
                    catalog_pmids.update(record.pmid for record in outcome.catalog)
                    reachable_lookup.update(_reachable_lookup_from_catalog(outcome.catalog))
                    if outcome.study is not None:
                        for source in _source_records_from_study(outcome.study):
                            if source.source_id not in source_ids_seen[claim.claim_id]:
                                source_catalog[claim.claim_id].append(source)
                                source_ids_seen[claim.claim_id].add(source.source_id)
                    if outcome.study is None:
                        # Constrained DESIGN persistently refused even after the
                        # repair loop (SPEC Endpoint 4): MAX_PLAN_REVISIONS revise
                        # attempts exhausted without admit. The agent never
                        # executed; no study enters the constrained corpus.
                        record_transition(
                            audit_trail=audit_trail,
                            audit_counters=audit_counters,
                            last_hashes=last_hashes,
                            run_id=run_id or "preview-run",
                            claim_id=claim.claim_id,
                            branch=branch,
                            year=year,
                            phase="design",
                            event_type="design-abstain-persistent",
                            severity="block",
                            integrity_score_before=1.0,
                            integrity_score_after=0.0,
                            message=(
                                f"Pre-execution gate persistently refused after "
                                f"{outcome.revision_attempts} revise attempt(s). "
                                f"Final reasons: {' '.join(outcome.design_reasons)}"
                            ),
                        )
                        blocked_this_era += 1
                        if (
                            branch == "constrained"
                            and target_retained_this_cell > 0
                            and tier3_store.count_studies(
                                branch="constrained", claim_id=claim.claim_id, year=year
                            )
                            < target_retained_this_cell
                            and outcome_index == len(outcomes) - 1
                            and len(outcomes) < constrained_attempt_cap
                        ):
                            outcomes.append(
                                _constrained_research_attempt(
                                    research_agent=research_agent,
                                    microdata_agent=microdata_agent,
                                    claim=claim,
                                    claim_graph=graph_lookup[claim.claim_id],
                                    year=year,
                                    telemetry=telemetry,
                                    replicate=len(outcomes),
                                )
                            )
                        outcome_index += 1
                        continue

                    branch_study = outcome.study.model_copy(deep=True)
                    if not branch_study.catalog_pmids:
                        branch_study.catalog_pmids = sorted(
                            {record.pmid for record in outcome.catalog}
                        )
                    # Temporal anti-speculation gate (BOTH arms): block any study
                    # whose data source post-dates the simulated year. This catches
                    # NHANES 2005-2006 data used in a year-2000 simulation and any
                    # other future-data path. Applied before CIVER so the provenance
                    # gate never even sees temporally inconsistent studies.
                    if not _study_is_temporally_consistent(branch_study, year):
                        record_transition(
                            audit_trail=audit_trail,
                            audit_counters=audit_counters,
                            last_hashes=last_hashes,
                            run_id=run_id or "preview-run",
                            claim_id=claim.claim_id,
                            branch=branch,
                            year=year,
                            phase="investigator",
                            event_type="temporal-speculation",
                            severity="block",
                            integrity_score_before=1.0,
                            integrity_score_after=0.0,
                            message=(
                                f"Data source year_end={branch_study.source_scope.year_end} "
                                f"exceeds simulated_year={year}; blocked from both arms."
                            ),
                        )
                        blocked_this_era += 1
                        if (
                            branch == "constrained"
                            and target_retained_this_cell > 0
                            and tier3_store.count_studies(
                                branch="constrained", claim_id=claim.claim_id, year=year
                            )
                            < target_retained_this_cell
                            and outcome_index == len(outcomes) - 1
                            and len(outcomes) < constrained_attempt_cap
                        ):
                            outcomes.append(
                                _constrained_research_attempt(
                                    research_agent=research_agent,
                                    microdata_agent=microdata_agent,
                                    claim=claim,
                                    claim_graph=graph_lookup[claim.claim_id],
                                    year=year,
                                    telemetry=telemetry,
                                    replicate=len(outcomes),
                                )
                            )
                        outcome_index += 1
                        continue
                    if outcome.plan is not None:
                        # Design event type distinguishes friction-cost from
                        # kill-cost (SPEC §7d, Endpoint 4): a repaired plan is
                        # an `info`-severity success of the refuse+repair loop,
                        # not a silent admit. Free arm never enters the loop.
                        if (
                            outcome.design_admitted
                            and outcome.revision_attempts > 0
                            and branch == "constrained"
                        ):
                            design_event_type = "design-repaired"
                            design_severity = "info"
                            design_message = (
                                f"Pre-execution gate admitted plan after "
                                f"{outcome.revision_attempts} revise attempt(s). "
                                f"Final reasons: {' '.join(outcome.design_reasons)}"
                            )
                        elif outcome.design_admitted:
                            design_event_type = "design-admitted"
                            design_severity = "info"
                            design_message = (
                                " ".join(outcome.design_reasons)
                                or "Research plan recorded for CIVER/BRIM process validation."
                            )
                        else:
                            # Free-arm observed-invalid design (recorded, not blocked).
                            design_event_type = "design-observed-invalid"
                            design_severity = "info" if branch == "free" else "block"
                            design_message = (
                                " ".join(outcome.design_reasons)
                                or "Research plan recorded for CIVER/BRIM process validation."
                            )
                        record_transition(
                            audit_trail=audit_trail,
                            audit_counters=audit_counters,
                            last_hashes=last_hashes,
                            run_id=run_id or "preview-run",
                            claim_id=claim.claim_id,
                            branch=branch,
                            year=year,
                            phase="design",
                            event_type=design_event_type,
                            severity=design_severity,
                            integrity_score_before=1.0,
                            integrity_score_after=1.0 if outcome.design_admitted else 0.0,
                            message=design_message,
                        )
                    # Article II / BRIM — execution deviation from the registered
                    # plan. It is observational in free/shadow mode and part of
                    # the final release score in constrained/active mode.
                    if outcome.execution_deviated:
                        record_transition(
                            audit_trail=audit_trail,
                            audit_counters=audit_counters,
                            last_hashes=last_hashes,
                            run_id=run_id or "preview-run",
                            claim_id=claim.claim_id,
                            branch=branch,
                            year=year,
                            phase="execution",
                            event_type="execution-deviated",
                            severity="warn",
                            integrity_score_before=1.0,
                            integrity_score_after=1.0,
                            message="Execution deviated from the registered plan: "
                            + outcome.deviation_note,
                        )
                    investigator = _study_to_evidence_unit(
                        study=branch_study,
                        branch=branch,
                        catalog_pmids=catalog_pmids,
                    )
                    evidence_units.append(investigator)
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="investigator",
                        event_type="investigator-emitted",
                        severity="info",
                        integrity_score_before=1.0,
                        integrity_score_after=1.0 if branch_study.pmids else 0.0,
                        message=(
                            f"ResearchAgent emitted {branch_study.id} from "
                            f"{', '.join(branch_study.pmids) or 'no PubMed record'} "
                            f"with cutoff {pubmed_cutoff_year(year)}."
                        ),
                    )
                    env_passed, env_message = _study_passes_medevo_environment(branch_study)
                    if not env_passed:
                        record_transition(
                            audit_trail=audit_trail,
                            audit_counters=audit_counters,
                            last_hashes=last_hashes,
                            run_id=run_id or "preview-run",
                            claim_id=claim.claim_id,
                            branch=branch,
                            year=year,
                            phase="environment",
                            event_type="environment-refused",
                            severity="block",
                            integrity_score_before=1.0,
                            integrity_score_after=0.0,
                            message=env_message,
                        )
                        blocked_this_era += 1
                        if (
                            branch == "constrained"
                            and target_retained_this_cell > 0
                            and tier3_store.count_studies(
                                branch="constrained", claim_id=claim.claim_id, year=year
                            )
                            < target_retained_this_cell
                            and outcome_index == len(outcomes) - 1
                            and len(outcomes) < constrained_attempt_cap
                        ):
                            outcomes.append(
                                _constrained_research_attempt(
                                    research_agent=research_agent,
                                    microdata_agent=microdata_agent,
                                    claim=claim,
                                    claim_graph=graph_lookup[claim.claim_id],
                                    year=year,
                                    telemetry=telemetry,
                                    replicate=len(outcomes),
                                )
                            )
                        outcome_index += 1
                        continue
                    if branch == "constrained":
                        from app.process_gate import issue_process_warrant

                        assessment, warrant = issue_process_warrant(
                            run_id=run_id or "preview-run",
                            branch=branch,
                            year=year,
                            study=branch_study,
                            claim_graph=graph_lookup[claim.claim_id],
                            threshold=RELEASE_THRESHOLD,
                        )
                        verdict = CiverVerdict(
                            node_id=branch_study.id,
                            passed=assessment.passed,
                            reasons=assessment.reasons,
                            certificate_id=warrant.id if assessment.passed else None,
                        )
                        warrants_by_output[warrant.output_id] = warrant
                        warrants.append(warrant)
                        # Score the process gate against TRUE provenance for
                        # calibration only; the gate itself read the PIR/plan and
                        # BRIM deviations, not this label.
                        calibration_observations.append(
                            (branch_study.provenance, assessment.passed)
                        )
                    else:
                        verdict = CiverVerdict(
                            node_id=branch_study.id,
                            passed=True,
                            reasons=["Free branch: CIVER/BRIM observed post hoc, not enforced."],
                            certificate_id=None,
                        )
                        warrant = None
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="release",
                        event_type="process-issued" if verdict.passed else "process-refused",
                        severity="info" if verdict.passed else "block",
                        integrity_score_before=1.0,
                        integrity_score_after=1.0 if verdict.passed or branch == "free" else 0.0,
                        message=" ".join(verdict.reasons),
                    )

                    released = branch == "free" or (
                        warrant is not None
                        and warrant.status == "ISSUED"
                        and warrant.issued
                        and warrant.integrity_score >= warrant.threshold
                    )
                    if branch == "free":
                        if tier3_store.insert(branch=branch, study=branch_study):
                            surviving_units.append(investigator)
                    elif released and warrant is not None:
                        if tier3_store.insert(
                            branch=branch,
                            study=branch_study,
                            warrant=warrant,
                            require_warrant=True,
                        ):
                            surviving_units.append(investigator)
                            warranted_ids.add(branch_study.id)
                    else:
                        blocked_this_era += 1

                    if (
                        branch == "constrained"
                        and target_retained_this_cell > 0
                        and tier3_store.count_studies(
                            branch="constrained", claim_id=claim.claim_id, year=year
                        )
                        < target_retained_this_cell
                        and outcome_index == len(outcomes) - 1
                        and len(outcomes) < constrained_attempt_cap
                    ):
                        outcomes.append(
                            _constrained_research_attempt(
                                research_agent=research_agent,
                                microdata_agent=microdata_agent,
                                claim=claim,
                                claim_graph=graph_lookup[claim.claim_id],
                                year=year,
                                telemetry=telemetry,
                                replicate=len(outcomes),
                            )
                        )
                    outcome_index += 1

                if branch == "constrained":
                    constrained_retained_this_cell = tier3_store.count_studies(
                        branch="constrained", claim_id=claim.claim_id, year=year
                    )
                    achieved = constrained_retained_this_cell >= target_retained_this_cell
                    output_match_records.append(
                        {
                            "claim_id": claim.claim_id,
                            "year": year,
                            "free_retained": free_retained_this_cell,
                            "target_retained": target_retained_this_cell,
                            "constrained_retained": constrained_retained_this_cell,
                            "attempts": len(outcomes),
                            "attempt_cap": constrained_attempt_cap,
                            "achieved": achieved,
                            "retained_ratio": round(
                                constrained_retained_this_cell / free_retained_this_cell, 4
                            )
                            if free_retained_this_cell
                            else 1.0,
                        }
                    )
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="output-matching",
                        event_type="output-match-achieved" if achieved else "output-match-failed",
                        severity="info" if achieved else "block",
                        integrity_score_before=1.0,
                        integrity_score_after=1.0 if achieved else 0.0,
                        message=(
                            f"Output matching retained constrained={constrained_retained_this_cell} "
                            f"vs free={free_retained_this_cell}; target={target_retained_this_cell}; "
                            f"attempts={len(outcomes)}/{constrained_attempt_cap}."
                        ),
                    )

                # Tier-4: ONE SR/MA over the accumulated Tier-3 DB for this branch.
                guideline = srma_agent.run(
                    run_id=run_id or "preview-run",
                    branch=branch,
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    year=year,
                )
                # Task A: CIVER gates the SRMA OUTPUT on the constrained branch.
                # Free branch emits as-is (no gate). The output gate re-appraises
                # the warranted-only corpus and refuses an over-reaching /
                # unwarranted guideline, degrading it to no-recommendation.
                if branch == "constrained":
                    corpus_studies = tier3_store.list_studies(
                        run_id=run_id or "preview-run",
                        branch=branch,
                        claim_id=claim.claim_id,
                        up_to_year=year,
                    )
                    # The inheritable warranted set = every study in the
                    # constrained corpus (this era AND prior eras) that carries a
                    # valid execution warrant (Article IV). The constrained DB is
                    # warranted-by-construction, so this is its membership; we
                    # re-verify against warrants_by_output rather than trust the DB.
                    corpus_warranted_ids = {
                        study.id
                        for study in corpus_studies
                        if _valid_study_warrant_for_study(
                            study, warrants_by_output.get(study.id)
                        )
                    } | warranted_ids
                    guideline, output_admitted, output_reason = admit_guideline_output(
                        guideline=guideline,
                        studies=corpus_studies,
                        warranted_ids=corpus_warranted_ids,
                        claim_text=claim.text,
                    )
                    if guideline.insufficient_evidence:
                        guideline_event_type = "guideline-abstained"
                        guideline_severity = "warn"
                        guideline_integrity_after = 0.5
                        guideline_message = (
                            "Insufficient substantive evidence after screening: "
                            "no included studies survived the SR quality floor. "
                            "Guideline returns NA (no answer), not NEUTRAL."
                        )
                    elif output_admitted:
                        guideline_event_type = "guideline-issued"
                        guideline_severity = "info"
                        guideline_integrity_after = 1.0
                        guideline_message = output_reason
                    else:
                        guideline_event_type = "guideline-refused"
                        guideline_severity = "block"
                        guideline_integrity_after = 0.0
                        guideline_message = output_reason
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="guideline-admission",
                        event_type=guideline_event_type,
                        severity=guideline_severity,
                        integrity_score_before=1.0,
                        integrity_score_after=guideline_integrity_after,
                        message=guideline_message,
                    )
                if run_id is not None:
                    insert_guideline_claims(run_id=run_id, branch=branch, claims=[guideline])
                guideline_timeline[branch].append(guideline)
                branch_guidelines[branch].append(guideline)
                pooled_score = guideline.pooled_effect or _DIRECTION_VALUE[guideline.direction]
                synth_rationale = (
                    "Tier-4 SRMA read the accumulated Tier-3 DB only: "
                    f"{guideline.study_count} studies, ungrounded_fraction="
                    f"{guideline.ungrounded_fraction}, heterogeneity={guideline.heterogeneity}."
                )
                lineage = _lineage_record(
                    claim_id=claim.claim_id,
                    year=year,
                    branch=branch,
                    prior_state=state,
                    surviving_units=surviving_units,
                    verdict_before=state.prior_direction,
                    verdict_after=guideline.direction,
                )
                record_transition(
                    audit_trail=audit_trail,
                    audit_counters=audit_counters,
                    last_hashes=last_hashes,
                    run_id=run_id or "preview-run",
                    claim_id=claim.claim_id,
                    branch=branch,
                    year=year,
                    phase="lineage",
                    event_type="lineage-delta",
                    severity="warn" if lineage.lost_real else "info",
                    integrity_score_before=1.0,
                    integrity_score_after=round(
                        len(lineage.surviving_real) / REAL_SOURCES_PER_CLAIM,
                        3,
                    ),
                    message=(
                        f"Real sources retained {lineage.surviving_real or ['none']}; "
                        f"lost {lineage.lost_real or ['none']}; "
                        f"ungrounded carriers {lineage.ungrounded_carriers or ['none']}."
                    ),
                )
                lineage_records.append(lineage)

                cycle_events = [
                    event
                    for event in audit_trail
                    if event.claim_id == claim.claim_id and event.branch == branch and event.year == year
                ]
                snapshot = _claim_snapshot(
                    claim=claim,
                    year=year,
                    branch=branch,
                    verdict=verdict,
                    guideline=guideline,
                    synth_rationale=synth_rationale,
                    lineage=lineage,
                    cycle_events=cycle_events,
                    blocked_count=0 if branch == "free" else blocked_this_era,
                    emitted_count=len(surviving_units),
                )
                branch_claims[branch].append(snapshot)
                branch_scores[branch].append(pooled_score)

                state.prior_direction = snapshot.direction
                state.prior_strength = snapshot.strength
                state.citation_memory = (investigator.cited_ids + state.citation_memory)[:6]
                state.surviving_real = set(lineage.surviving_real)
                state.output_history.extend(surviving_units)
                state.output_history = state.output_history[-8:]

        branch_diff[str(year)] = {}
        for index, claim in enumerate(claims):
            free_guideline = branch_guidelines["free"][index]
            constrained_guideline = branch_guidelines["constrained"][index]
            direction_delta = abs(
                _DIRECTION_VALUE[free_guideline.direction]
                - _DIRECTION_VALUE[constrained_guideline.direction]
            ) / 2.0
            level_gap = branch_gap(
                free=[free_guideline],
                constrained=[constrained_guideline],
                iterations=1,
            ).level.mean
            delta = (direction_delta + level_gap) / 2.0
            branch_diff[str(year)][claim.claim_id] = round(delta, 3)
            branch_claims["free"][index].divergence_score = round(delta, 3)
            branch_claims["constrained"][index].divergence_score = round(delta, 3)
        db_growth[str(year)] = replay_counts(
            studies=tier3_store.all_studies(),
            guidelines=branch_guidelines,
        )
        population_stats[str(year)] = branch_gap(
            free=branch_guidelines["free"],
            constrained=branch_guidelines["constrained"],
            iterations=500,
            seed=year,
        ).to_dict()

        for branch in ("free", "constrained"):
            snapshots[branch].append(
                DriftSnapshot(
                    year=year,
                    branch=branch,
                    claims=branch_claims[branch],
                    band=_panel_band(year, branch_scores[branch]),
                    anchors=ANCHORS,
                )
            )

    descriptor = llm.describe()
    transient_rate = telemetry.transient_failure_rate
    within_tolerance = transient_rate <= TRANSIENT_FAILURE_TOLERANCE
    if not within_tolerance and telemetry.degradation_reason is None:
        telemetry.degradation_reason = (
            f"transient_failure_rate={transient_rate:.3%} "
            f"({len(telemetry.transient_failures)}/{telemetry.call_count}) "
            f"exceeds tolerance {TRANSIENT_FAILURE_TOLERANCE:.1%}"
        )
    scientific = (
        llm.scientific
        and telemetry.degradation_reason is None
        and within_tolerance
        and not isinstance(pubmed_client, DeterministicPubMedClient)
    )
    degradation_reason = (
        telemetry.degradation_reason
        or (
            "deterministic PubMed fixture"
            if isinstance(pubmed_client, DeterministicPubMedClient)
            else None
        )
        or (getattr(llm, "degradation_reason", None) if not llm.scientific else None)
    )
    provenance_log = {
        "model": descriptor.name,
        "model_digest": descriptor.digest,
        "provider": request.backend,
        "base_url": request.base_url or "",
        "temperature": 0.2,
        "seed_mode": "engine-seeded-structure",
        "prompt_template_digests": {
            "tier1_pubmed": "entrez-date-cut",
            "srma_pooling": "llm-appraisal-plus-deterministic-pool",
        },
        "failure_rate": failure_rate,
        "llm_cache": llm_cache_stats(llm),
        "calls": [trace.__dict__ for trace in telemetry.traces],
    }
    output_match = _output_match_summary(
        records=output_match_records,
        guideline_timeline=guideline_timeline,
    )
    provenance_log["output_matching"] = output_match

    if not scientific:
        validation_notes = [
            f"DEGRADED RUN: {degradation_reason or 'A model call fell back to the deterministic client.'}",
            "This run is illustrative only; any branch contrast shown here is non-scientific.",
            "The fallback path is explicit so plausible-looking verdicts are never presented as scientific output.",
        ]
        mode_banner = "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"
    else:
        validation_notes = [
            "Reachable-corpus divergence is branch-conditioned only at corpus construction, not selection scoring.",
            "Tier-1 studies are produced by ResearchAgent over PubMed/date-cut records; Tier-4 SRMA uses LLM appraisal over the accumulated Tier-3 DB and keeps numeric pooling deterministic.",
            "The release gate reads the hash-chained audit trail and never calls the LLM.",
            "Active constrained arm uses output-matched generation: it keeps attempting until retained studies approach the free arm for each claim-year cell, or records coverage failure.",
        ]
        mode_banner = ""

    bundle = ArtifactBundle(
        input_text=input_text,
        claim_graphs=claim_graphs,
        snapshots=snapshots,
        branch_diff=branch_diff,
        anchors=ANCHORS,
        validation_notes=validation_notes,
        scientific=scientific,
        mode_banner=mode_banner,
        model_descriptor={"name": descriptor.name, "digest": descriptor.digest},
        lineage=lineage_records,
        audit_trail=audit_trail,
        warrants=warrants,
        corpus_studies=tier3_store.all_studies(),
        db_growth=db_growth,
        guideline_timeline=guideline_timeline,
        population_stats=population_stats,
        provenance_log=provenance_log,
        calibration_matrix=compute_calibration_matrix(calibration_observations),
        degradation_reason=degradation_reason,
    )
    bundle.bundle_seal = _canonical_sha256(_bundle_payload(bundle))

    if run_id is not None:
        insert_ecology_records(
            run_id=run_id,
            source_catalog=source_catalog,
            evidence_units=evidence_units,
            lineage_records=lineage_records,
            warrants=warrants,
            audit_events=audit_trail,
        )

    # SPEC §7d Endpoint 4: aggregate repair-loop counts so post-hoc analysis can
    # report friction-cost vs kill-cost without re-walking the audit trail.
    repair_counters = {
        "design_admitted_first_try": sum(
            1 for event in audit_trail if event.event_type == "design-admitted"
        ),
        "design_repaired": sum(
            1 for event in audit_trail if event.event_type == "design-repaired"
        ),
        "design_abstain_persistent": sum(
            1 for event in audit_trail if event.event_type == "design-abstain-persistent"
        ),
        "max_plan_revisions": MAX_PLAN_REVISIONS,
    }

    summary = {
        "claim_count": len(claims),
        "years": years,
        "scientific": scientific,
        "model": descriptor.name,
        "has_blocked_outputs": any(
            claim.blocked_count > 0
            for snapshot in bundle.snapshots["constrained"]
            for claim in snapshot.claims
        ),
        "population_stats": population_stats,
        "llm_call_count": telemetry.call_count,
        "llm_cache": llm_cache_stats(llm),
        "transient_failures": list(telemetry.transient_failures),
        "transient_failure_rate": round(transient_rate, 4),
        "transient_failure_tolerance": TRANSIENT_FAILURE_TOLERANCE,
        "degradation_reason": degradation_reason,
        "bundle_seal": bundle.bundle_seal,
        "provenance_log": provenance_log,
        "failure_rate": failure_rate,
        "repair_counters": repair_counters,
        "output_matching": output_match,
        "calibration_matrix": bundle.calibration_matrix.model_dump()
        if bundle.calibration_matrix
        else None,
    }
    if study_sink is not None:
        # Per-branch accumulated Tier-3 corpora (free = all emitted studies;
        # constrained = warranted survivors). Surfaced via an opt-in sink, NOT on
        # the bundle or the persisted summary, because the raw Study objects are a
        # Slice-C evaluation/control input (re-pooled offline), not a sealed
        # scientific artifact — and they are not JSON-serialisable into meta.json.
        study_sink.update(tier3_store.all_studies())
    return bundle, summary
