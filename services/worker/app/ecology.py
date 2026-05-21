from __future__ import annotations

import hashlib
import json
import math
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
from app.llm import LLMClient
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
    RunRequestModel,
    Study,
)
from app.pubmed import DeterministicPubMedClient, PubMedClient


ANCHORS = [
    "Pre-2023 literature contamination approximated near zero.",
    "Rising AI-text prevalence in biomedical publishing treated as empirical anchor.",
    "Every year-10/20/30 panel is rendered as one draw from a distribution, never a forecast.",
]

CLAIM_LIMIT = 3
REAL_SOURCES_PER_CLAIM = 4
# Tier-1 study replicates emitted per (claim, era). SPEC §3/§12: the phenomenon
# shows at ~tens of studies, and a real SR/MA needs a screenable corpus, not one
# study per claim. With CLAIM_LIMIT=3 claims and len(YEARS)=3 eras, k=2 yields
# 3 x 3 x 2 = 18 studies per arm across the run (inside the SPEC §13 target band
# of 15-20). Declared as one named constant, never a magic literal in the loop.
STUDIES_PER_CLAIM_PER_ERA = 2
# DEFAULT_FAILURE_RATE (imported from app.agents) is the weak-agent failure
# fraction placeholder; SPEC §11-A anchors it to A0 in a later slice. It drives
# the EMERGENT ungrounded-study rate, NOT a harness injection rate.
RELEASE_THRESHOLD = 0.60
# Article I scope clause tolerance (years). A claimed scope wider than the
# evidence's by MORE than this is refused; a mild over-reach within tolerance
# slips the gate (the gate is imperfect, not tautological — FNR can be > 0).
# Declared here as the single source for the predicate (audit §8.2: no magic
# literal buried in logic). Paired with agents.SCOPE_INFLATION_MIN/MAX.
SCOPE_TOLERANCE_YEARS = 2
GENESIS_HASH = "GENESIS"
PUBMED_FORWARD_CEILING_YEAR = 2025
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


def _research_studies_for_year(
    *,
    research_agent: ResearchAgent,
    microdata_agent: MicrodataAgent,
    claim: ClaimSeed,
    year: int,
    telemetry: CallTelemetry,
) -> list[tuple[Study, list[PubMedRecord]]]:
    """Produce STUDIES_PER_CLAIM_PER_ERA Tier-1 studies for one (claim, era).

    Group A (microdata/NHANES) is a single deterministic dataset slice per era, so
    it contributes ONE study; the remaining replicates come from Group B (the
    PubMed literature agent), each a distinct seeded attempt. This yields a
    screenable corpus for the Tier-4 SR/MA rather than one study per claim.
    """
    studies: list[tuple[Study, list[PubMedRecord]]] = []
    for replicate in range(STUDIES_PER_CLAIM_PER_ERA):
        if replicate == 0 and _use_microdata_group(claim=claim, year=year):
            studies.append(
                microdata_agent.run(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    simulated_year=year,
                )
            )
            continue
        studies.append(
            _research_study_for_year(
                research_agent=research_agent,
                claim=claim,
                year=year,
                telemetry=telemetry,
                replicate=replicate,
            )
        )
    return studies


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
        if telemetry.degradation_reason is None:
            telemetry.degradation_reason = f"pubmed/{claim.claim_id}/year-{year}: {type(exc).__name__}: {exc}"
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
    if not microdata_supports_claim(claim.text):
        return False
    bucket = int(
        hashlib.sha256(f"microdata-slot:{claim.claim_id}:{year}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    return bucket % 2 == 0


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
    # Article I scope clause: the claim's scope (population/timeframe) may not
    # exceed any cited evidence's authoritative source scope (beyond tolerance).
    # This catches Mode-2 over-reach, which RESOLVES (real PMID) yet over-claims.
    scope_within_evidence = not any(
        unit.claimed_scope.exceeds(item.scope, tolerance=SCOPE_TOLERANCE_YEARS)
        for item in cited_items
    )
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
    if cited_resolve and resolvable:
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
        # article-i-refused and guideline-refused are EXPECTED gate refusals
        # (Article I / IV doing their job at the study-input and SRMA-output
        # boundaries), not process-integrity drift — they must not depress the
        # release-gate score of subsequent admissible studies (Article II/III
        # double-counting, CONSTITUTION §4).
        if event.severity == "block" and event.event_type not in (
            "article-i-refused",
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
    # to admit_evidence_unit (gate blindness, SPEC §8.3).
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

    for year in years:
        branch_scores: dict[str, list[float]] = {"free": [], "constrained": []}
        branch_claims: dict[str, list[ClaimSnapshot]] = {"free": [], "constrained": []}
        branch_guidelines: dict[str, list[GuidelineClaim]] = {"free": [], "constrained": []}

        for claim in claims:
            # Tier-1: emit STUDIES_PER_CLAIM_PER_ERA studies for this (claim, era).
            study_batch = _research_studies_for_year(
                research_agent=research_agent,
                microdata_agent=microdata_agent,
                claim=claim,
                year=year,
                telemetry=telemetry,
            )
            catalog_pmids: set[str] = set()
            reachable_lookup: dict[str, CorpusItem] = {}
            for research_study, catalog_records in study_batch:
                catalog_pmids.update(record.pmid for record in catalog_records)
                reachable_lookup.update(_reachable_lookup_from_catalog(catalog_records))
                for source in _source_records_from_study(research_study):
                    if source.source_id not in source_ids_seen[claim.claim_id]:
                        source_catalog[claim.claim_id].append(source)
                        source_ids_seen[claim.claim_id].add(source.source_id)

            for branch in ("free", "constrained"):
                state = states[(claim.claim_id, branch)]
                surviving_units: list[EvidenceUnit] = []
                # study ids that earned a valid execution warrant in this branch —
                # the inheritable set the guideline-output gate (Task A) checks.
                warranted_ids: set[str] = set()
                blocked_this_era = 0

                # Tier-1 -> Tier-2 (CIVER) -> Tier-3 admission, per replicate study.
                for research_study, _catalog in study_batch:
                    branch_study = research_study.model_copy(deep=True)
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
                    verdict, warrant = admit_evidence_unit(
                        run_id=run_id or "preview-run",
                        claim=claim,
                        claim_graph=graph_lookup[claim.claim_id],
                        branch=branch,
                        year=year,
                        unit=investigator,
                        reachable_lookup=reachable_lookup,
                        warrants_by_output=warrants_by_output,
                        threshold=RELEASE_THRESHOLD,
                    )
                    if warrant is not None:
                        warrants_by_output[warrant.output_id] = warrant
                        warrants.append(warrant)
                    if branch == "constrained":
                        # Score the gate's Article-I verdict against TRUE
                        # provenance. branch_study.provenance is the harness's
                        # ground truth, read here ONLY for calibration — the gate
                        # above received no such field.
                        calibration_observations.append(
                            (branch_study.provenance, verdict.passed)
                        )
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="admission",
                        event_type="article-i-issued" if verdict.passed else "article-i-refused",
                        severity="info" if verdict.passed else "block",
                        integrity_score_before=1.0,
                        integrity_score_after=1.0 if verdict.passed or branch == "free" else 0.0,
                        message=" ".join(verdict.reasons),
                    )

                    warrant, released, release_message = _apply_release_gate(
                        branch=branch,
                        warrant=warrant,
                        claim_events=[
                            event
                            for event in audit_trail
                            if event.claim_id == claim.claim_id and event.branch == branch
                        ],
                        scientific=llm.scientific,
                        threshold=RELEASE_THRESHOLD,
                    )
                    if warrant is not None:
                        warrants_by_output[warrant.output_id] = warrant
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="release",
                        event_type="release-issued" if released else "release-revoked",
                        severity="info" if released else "block",
                        integrity_score_before=1.0 if verdict.passed or branch == "free" else 0.0,
                        integrity_score_after=(
                            warrant.integrity_score if warrant is not None else 1.0
                        ),
                        message=release_message,
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
                    )
                    record_transition(
                        audit_trail=audit_trail,
                        audit_counters=audit_counters,
                        last_hashes=last_hashes,
                        run_id=run_id or "preview-run",
                        claim_id=claim.claim_id,
                        branch=branch,
                        year=year,
                        phase="guideline-admission",
                        event_type="guideline-issued" if output_admitted else "guideline-refused",
                        severity="info" if output_admitted else "block",
                        integrity_score_before=1.0,
                        integrity_score_after=1.0 if output_admitted else 0.0,
                        message=output_reason,
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
    scientific = (
        llm.scientific
        and telemetry.degradation_reason is None
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
        "calls": [trace.__dict__ for trace in telemetry.traces],
    }

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
        "degradation_reason": degradation_reason,
        "bundle_seal": bundle.bundle_seal,
        "provenance_log": provenance_log,
        "failure_rate": failure_rate,
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
