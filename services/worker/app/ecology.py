from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from app.config import YEARS
from app.db import insert_ecology_records
from app.llm import (
    RESEARCHER_PROMPT_TEMPLATE,
    SYNTHESIST_PROMPT_TEMPLATE,
    LLMClient,
    parse_direction,
)
from app.models import (
    ArtifactBundle,
    BrimEvent,
    BranchName,
    ClaimDirection,
    ClaimGraph,
    ClaimSnapshot,
    CiverVerdict,
    DriftSnapshot,
    EvidenceUnit,
    LineageRecord,
    RecommendationStrength,
    RunRequestModel,
)


ANCHORS = [
    "Pre-2023 literature contamination approximated near zero.",
    "Rising AI-text prevalence in biomedical publishing treated as empirical anchor.",
    "Every year-10/20/30 panel is rendered as one draw from a distribution, never a forecast.",
]

CLAIM_LIMIT = 3
REAL_SOURCES_PER_CLAIM = 4
PRESEEDED_SYNTHETIC_UNITS = 3
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
    text: str


@dataclass
class BranchState:
    prior_direction: ClaimDirection = "NEUTRAL"
    prior_strength: RecommendationStrength = "weak"
    citation_memory: list[str] = field(default_factory=list)
    surviving_real: set[str] = field(default_factory=set)
    surviving_units: list[EvidenceUnit] = field(default_factory=list)


@dataclass
class CallTelemetry:
    call_count: int = 0
    degradation_reason: str | None = None


def contamination_clock(year: int) -> float:
    return round(1 / (1 + math.exp(-0.115 * (year - 18))), 3)


def horizon_years(request: RunRequestModel) -> list[int]:
    years = request.horizons or list(YEARS)
    cleaned = sorted({int(year) for year in years if int(year) > 0})
    return cleaned or list(YEARS)


def _seed_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def _digest_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _direction_from_hash(key: str) -> ClaimDirection:
    bucket = _seed_int(key) % 3
    return ("SUPPORTS", "NEUTRAL", "REFUTES")[bucket]


def _extract_rationale(text: str) -> str:
    match = re.search(r"RATIONALE:\s*(.+)", text or "", re.IGNORECASE)
    if match:
        return match.group(1).strip()
    cleaned = " ".join((text or "").strip().split())
    return cleaned or "No rationale returned."


def _strength_from_score(score: float, evidence_count: int) -> RecommendationStrength:
    if evidence_count <= 1:
        return "weak"
    if abs(score) >= 0.66:
        return "strong"
    if abs(score) >= 0.34:
        return "moderate"
    return "weak"


def _panel_direction(score: float) -> ClaimDirection:
    if score >= 0.34:
        return "SUPPORTS"
    if score <= -0.34:
        return "REFUTES"
    return "NEUTRAL"


def _panel_band(year: int, branch_scores: list[float]) -> dict[str, float | str]:
    contamination = contamination_clock(year)
    band_mid = fmean(branch_scores) if branch_scores else 0.0
    return {
        "low": round(band_mid - contamination * 0.28, 3),
        "high": round(band_mid + contamination * 0.28, 3),
        "label": "Sensitivity band scaled by contamination-clock pressure.",
    }


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
    result = llm.generate(prompt, seed=seed)
    if was_scientific and not llm.scientific and telemetry.degradation_reason is None:
        failure_reason = getattr(llm, "degradation_reason", None) or "model call degraded"
        telemetry.degradation_reason = f"{label}: {failure_reason}"
    return result


def mint_source_catalog(claims: list[ClaimSeed], input_text: str) -> dict[str, list[SourceRecord]]:
    chunks = _sentence_chunks(input_text)
    if not chunks:
        chunks = [claim.text for claim in claims]

    catalog: dict[str, list[SourceRecord]] = {}
    for claim_index, claim in enumerate(claims):
        sources: list[SourceRecord] = []
        for source_index in range(REAL_SOURCES_PER_CLAIM):
            chunk = chunks[(claim_index + source_index) % len(chunks)]
            source_id = f"R-{claim.claim_id}-{source_index + 1}"
            sources.append(
                SourceRecord(
                    source_id=source_id,
                    claim_id=claim.claim_id,
                    label=f"Catalog source {source_index + 1}",
                    text=chunk,
                )
            )
        catalog[claim.claim_id] = sources
    return catalog


def _mint_contamination_bank(
    claim: ClaimSeed,
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> list[EvidenceUnit]:
    prompt = (
        "Write one plausible two-sentence study summary about the following "
        "clinical topic. Report a finding.\n"
        f"Topic: {claim.text}\n"
    )
    response = _invoke_model(
        llm,
        telemetry,
        f"contaminator/{claim.claim_id}",
        prompt,
        seed=_seed_int(f"contaminator:{claim.claim_id}"),
    )
    rationale = _extract_rationale(response)
    direction = _direction_from_hash(f"{claim.claim_id}:{response}")

    units: list[EvidenceUnit] = []
    for index in range(PRESEEDED_SYNTHETIC_UNITS):
        units.append(
            EvidenceUnit(
                id=f"{claim.claim_id}-syn-{index + 1}",
                claim_id=claim.claim_id,
                year=0,
                branch="free",
                producer="contaminator",
                cited_ids=[f"S-{claim.claim_id}-{index + 1}", f"S-{claim.claim_id}-{index + 11}"],
                provenance="SYNTHETIC",
                direction=direction,
                rationale=rationale,
            )
        )
    return units


def _synthetic_units_for_year(bank: list[EvidenceUnit], year: int) -> list[EvidenceUnit]:
    count = max(1, min(len(bank), math.ceil(contamination_clock(year) * len(bank))))
    return bank[:count]


def _build_evidence_block(
    source_lookup: dict[str, str],
    prior_units: list[EvidenceUnit],
    synthetic_lookup: dict[str, str],
    cited_ids: list[str],
    branch_state: BranchState,
) -> str:
    lines: list[str] = []
    for position, cited_id in enumerate(cited_ids, start=1):
        display_id = f"source-{position}"
        if cited_id in source_lookup:
            lines.append(f"- [{display_id}] {source_lookup[cited_id]}")
            continue
        prior_unit = next((unit for unit in prior_units if unit.id == cited_id), None)
        if prior_unit is not None:
            lines.append(
                f"- [{display_id}] Prior study concluded {prior_unit.direction}: {prior_unit.rationale}"
            )
            continue
        if cited_id in synthetic_lookup:
            lines.append(f"- [{display_id}] {synthetic_lookup[cited_id]}")
            continue
        lines.append(f"- [{display_id}] Cited study summary unavailable in the local excerpt.")
    if branch_state.citation_memory:
        remembered = ", ".join(branch_state.citation_memory[:3])
        lines.append(f"- Citation memory currently foregrounds: {remembered}.")
    return "\n".join(lines)


def _select_citations(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    branch_state: BranchState,
    real_sources: list[SourceRecord],
    prior_units: list[EvidenceUnit],
    synthetic_units: list[EvidenceUnit],
) -> tuple[list[str], list[str]]:
    clock = contamination_clock(year)
    real_ids = [source.source_id for source in real_sources]
    synthetic_ids = [unit.id for unit in synthetic_units]
    carry_ids = [unit.id for unit in prior_units[-2:]]

    citations: list[str] = []
    debug: list[str] = []

    if branch_state.surviving_real and clock < 0.72:
        remembered_real = sorted(branch_state.surviving_real)[0]
        citations.append(remembered_real)
        debug.append("remembered-real")
    elif clock < 0.55:
        citations.append(real_ids[_seed_int(f"real:{claim.claim_id}:{year}") % len(real_ids)])
        debug.append("fresh-real")
    elif synthetic_ids:
        citations.append(synthetic_ids[_seed_int(f"synthetic:{claim.claim_id}:{year}") % len(synthetic_ids)])
        debug.append("synthetic-anchor")

    if carry_ids:
        citations.append(carry_ids[-1])
        debug.append("carry-forward")

    if synthetic_ids and clock >= 0.48:
        citations.append(synthetic_ids[(year + len(branch_state.citation_memory)) % len(synthetic_ids)])
        debug.append("clock-synthetic")

    if not branch_state.surviving_real and clock >= 0.76:
        citations = [cited for cited in citations if cited not in real_ids] or citations
        debug.append("synthetic-dominance")

    deduped: list[str] = []
    for cited_id in citations:
        if cited_id not in deduped:
            deduped.append(cited_id)
    if not deduped:
        deduped = [real_ids[0]]
        debug.append("fallback-real")
    return deduped[:3], debug


def _investigator_unit(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    branch_state: BranchState,
    real_sources: list[SourceRecord],
    prior_units: list[EvidenceUnit],
    synthetic_units: list[EvidenceUnit],
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> tuple[EvidenceUnit, list[str]]:
    source_lookup = {source.source_id: source.text for source in real_sources}
    synthetic_lookup = {unit.id: unit.rationale for unit in synthetic_units}
    cited_ids, debug = _select_citations(
        claim=claim,
        year=year,
        branch=branch,
        branch_state=branch_state,
        real_sources=real_sources,
        prior_units=prior_units,
        synthetic_units=synthetic_units,
    )
    evidence_block = _build_evidence_block(
        source_lookup,
        prior_units,
        synthetic_lookup,
        cited_ids,
        branch_state,
    )
    prompt = RESEARCHER_PROMPT_TEMPLATE.format(claim=claim.text, evidence_block=evidence_block)
    response = _invoke_model(
        llm,
        telemetry,
        f"investigator/{branch}/{claim.claim_id}/year-{year}",
        prompt,
        seed=_seed_int(f"investigator:{claim.claim_id}:{year}:{_digest_key(evidence_block)}"),
    )
    real_ids = {source.source_id for source in real_sources}
    provenance = "REAL" if any(cited in real_ids for cited in cited_ids) else "SYNTHETIC"
    unit = EvidenceUnit(
        id=f"{claim.claim_id}-{branch}-investigator-{year}",
        claim_id=claim.claim_id,
        year=year,
        branch=branch,
        producer="investigator",
        cited_ids=cited_ids,
        provenance=provenance,
        direction=parse_direction(response),
        rationale=_extract_rationale(response),
    )
    return unit, debug


def _surviving_units_for_branch(
    *,
    branch: BranchName,
    investigator: EvidenceUnit,
    synthetic_units: list[EvidenceUnit],
    real_catalog_ids: set[str],
) -> tuple[list[EvidenceUnit], CiverVerdict, int]:
    synthetic_count = max(1, len(synthetic_units) - 1)
    synthetic_descendants = [
        EvidenceUnit(
            id=f"{branch}-{investigator.claim_id}-synthetic-{investigator.year}-{index + 1}",
            claim_id=investigator.claim_id,
            year=investigator.year,
            branch=branch,
            producer="contaminator",
            cited_ids=unit.cited_ids,
            provenance="SYNTHETIC",
            direction=unit.direction,
            rationale=unit.rationale,
        )
        for index, unit in enumerate(synthetic_units[:synthetic_count])
    ]
    has_real = any(cited_id in real_catalog_ids for cited_id in investigator.cited_ids)
    if branch == "free":
        verdict = CiverVerdict(
            node_id=investigator.id,
            passed=True,
            reasons=["CIVER not applied in free branch."],
            certificate_id=None,
        )
        return [investigator, *synthetic_descendants], verdict, 0

    passed = has_real
    verdict = CiverVerdict(
        node_id=investigator.id,
        passed=passed,
        reasons=(
            [
                "At least one cited identifier resolves to the real source catalog.",
                "Execution certificate issued; evidence unit survives this cycle.",
            ]
            if passed
            else [
                "No cited identifier resolves to the real source catalog.",
                "Execution certificate refused; evidence unit discarded this cycle.",
            ]
        ),
        certificate_id=f"CIVER-{investigator.year}-{investigator.claim_id}" if passed else None,
    )
    survivors = synthetic_descendants[:1]
    if passed:
        survivors.insert(0, investigator)
    return survivors, verdict, 0 if passed else 1


def _synthesist_verdict(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    surviving_units: list[EvidenceUnit],
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> tuple[ClaimDirection, str]:
    evidence_lines = [
        f"- Evidence unit {index}: {unit.direction}. {unit.rationale}"
        for index, unit in enumerate(surviving_units, start=1)
    ]
    evidence_block = "\n".join(evidence_lines)
    prompt = SYNTHESIST_PROMPT_TEMPLATE.format(
        claim=claim.text,
        evidence_block=evidence_block,
    )
    response = _invoke_model(
        llm,
        telemetry,
        f"synthesist/{branch}/{claim.claim_id}/year-{year}",
        prompt,
        seed=_seed_int(f"synthesist:{claim.claim_id}:{year}:{_digest_key(evidence_block)}"),
    )
    return parse_direction(response), _extract_rationale(response)


def _lineage_record(
    *,
    claim_id: str,
    year: int,
    branch: BranchName,
    prior_state: BranchState,
    next_units: list[EvidenceUnit],
    verdict_before: ClaimDirection,
    verdict_after: ClaimDirection,
    real_catalog_ids: set[str],
) -> LineageRecord:
    surviving_real = sorted(
        {
            cited_id
            for unit in next_units
            for cited_id in unit.cited_ids
            if cited_id in real_catalog_ids
        }
    )
    lost_real = sorted(prior_state.surviving_real.difference(surviving_real))
    synthetic_carriers = [
        unit.id
        for unit in next_units
        if unit.provenance == "SYNTHETIC"
        or any(cited_id not in real_catalog_ids for cited_id in unit.cited_ids)
    ]
    return LineageRecord(
        claim_id=claim_id,
        year=year,
        branch=branch,
        surviving_real=surviving_real,
        lost_real=lost_real,
        synthetic_carriers=synthetic_carriers,
        verdict_before=verdict_before,
        verdict_after=verdict_after,
    )


def _claim_snapshot(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    investigator: EvidenceUnit,
    surviving_units: list[EvidenceUnit],
    verdict: CiverVerdict,
    synth_direction: ClaimDirection,
    synth_rationale: str,
    lineage: LineageRecord,
    selection_debug: list[str],
    branch_state: BranchState,
) -> tuple[ClaimSnapshot, float, list[BrimEvent]]:
    direction_scores = [_DIRECTION_VALUE[unit.direction] for unit in surviving_units]
    investigator_weight = 0.4 if investigator.provenance == "SYNTHETIC" else 0.15
    pooled_score = (
        fmean(direction_scores + [_DIRECTION_VALUE[synth_direction], _DIRECTION_VALUE[branch_state.prior_direction] * investigator_weight])
        if direction_scores
        else _DIRECTION_VALUE[synth_direction]
    )
    panel_direction = _panel_direction(pooled_score)
    strength = _strength_from_score(pooled_score, len(surviving_units))
    brim_events = [
        BrimEvent(
            node_id=investigator.id,
            event_type="lineage-delta",
            severity="warn" if lineage.lost_real else "info",
            integrity_score=round(len(lineage.surviving_real) / REAL_SOURCES_PER_CLAIM, 3),
            message=(
                f"Year {year} {branch}: retained real sources {lineage.surviving_real or ['none']}; "
                f"lost real sources {lineage.lost_real or ['none']}; synthetic carriers {lineage.synthetic_carriers or ['none']}."
            ),
        ),
        BrimEvent(
            node_id=f"{investigator.id}-selection",
            event_type="citation-memory",
            severity="warn" if "synthetic-dominance" in selection_debug else "info",
            integrity_score=round(1 - contamination_clock(year), 3),
            message=f"Citation path: {', '.join(selection_debug)}.",
        ),
    ]
    why_summary = (
        f"{branch.title()} branch at year {year}: panel moved from {lineage.verdict_before} to "
        f"{lineage.verdict_after} after pooling {len(surviving_units)} surviving units. "
        f"Real sources retained: {', '.join(lineage.surviving_real) or 'none'}. "
        f"Lost real sources: {', '.join(lineage.lost_real) or 'none'}. "
        f"Synthetic carriers: {', '.join(lineage.synthetic_carriers) or 'none'}. "
        f"Synthesist rationale: {synth_rationale}"
    )
    snapshot = ClaimSnapshot(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        direction=panel_direction,
        strength=strength,
        emitted_count=len(surviving_units),
        blocked_count=0 if verdict.passed or branch == "free" else 1,
        divergence_score=0.0,
        why_summary=why_summary,
        civer=[verdict],
        brim=brim_events,
    )
    return snapshot, pooled_score, brim_events


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


def run_ecology(
    *,
    request: RunRequestModel,
    input_text: str,
    claim_graphs: list[ClaimGraph],
    llm: LLMClient,
    run_id: str | None = None,
) -> tuple[ArtifactBundle, dict[str, Any]]:
    years = horizon_years(request)
    claims = extract_claims(input_text, request.input_mode)
    claims = claims[: len(claim_graphs)]
    telemetry = CallTelemetry()

    source_catalog = mint_source_catalog(claims, input_text)
    synthetic_bank = {
        claim.claim_id: _mint_contamination_bank(claim, llm, telemetry) for claim in claims
    }
    states: dict[tuple[str, BranchName], BranchState] = {
        (claim.claim_id, branch): BranchState()
        for claim in claims
        for branch in ("free", "constrained")
    }

    snapshots: dict[str, list[DriftSnapshot]] = {"free": [], "constrained": []}
    branch_diff: dict[str, dict[str, float]] = {}
    lineage_records: list[LineageRecord] = []
    evidence_units: list[EvidenceUnit] = []

    for year in years:
        branch_scores: dict[str, list[float]] = {"free": [], "constrained": []}
        branch_claims: dict[str, list[ClaimSnapshot]] = {"free": [], "constrained": []}

        for branch in ("free", "constrained"):
            for claim in claims:
                state = states[(claim.claim_id, branch)]
                prior_units = state.surviving_units.copy()
                real_sources = source_catalog[claim.claim_id]
                real_catalog_ids = {source.source_id for source in real_sources}
                synthetic_units = _synthetic_units_for_year(
                    synthetic_bank[claim.claim_id],
                    year,
                )
                investigator, selection_debug = _investigator_unit(
                    claim=claim,
                    year=year,
                    branch=branch,
                    branch_state=state,
                    real_sources=real_sources,
                    prior_units=prior_units,
                    synthetic_units=synthetic_units,
                    llm=llm,
                    telemetry=telemetry,
                )
                surviving_units, verdict, blocked_count = _surviving_units_for_branch(
                    branch=branch,
                    investigator=investigator,
                    synthetic_units=synthetic_units,
                    real_catalog_ids=real_catalog_ids,
                )
                evidence_units.append(investigator)
                evidence_units.extend(surviving_units)
                synth_direction, synth_rationale = _synthesist_verdict(
                    claim=claim,
                    year=year,
                    branch=branch,
                    surviving_units=surviving_units,
                    llm=llm,
                    telemetry=telemetry,
                )
                lineage = _lineage_record(
                    claim_id=claim.claim_id,
                    year=year,
                    branch=branch,
                    prior_state=state,
                    next_units=surviving_units,
                    verdict_before=state.prior_direction,
                    verdict_after=synth_direction,
                    real_catalog_ids=real_catalog_ids,
                )
                snapshot, pooled_score, _ = _claim_snapshot(
                    claim=claim,
                    year=year,
                    branch=branch,
                    investigator=investigator,
                    surviving_units=surviving_units,
                    verdict=verdict,
                    synth_direction=synth_direction,
                    synth_rationale=synth_rationale,
                    lineage=lineage,
                    selection_debug=selection_debug,
                    branch_state=state,
                )
                snapshot.blocked_count = blocked_count
                branch_claims[branch].append(snapshot)
                branch_scores[branch].append(pooled_score)
                lineage_records.append(lineage)

                state.prior_direction = snapshot.direction
                state.prior_strength = snapshot.strength
                state.citation_memory = (investigator.cited_ids + state.citation_memory)[:6]
                state.surviving_real = set(lineage.surviving_real)
                state.surviving_units = surviving_units[-6:]

        branch_diff[str(year)] = {}
        for index, claim in enumerate(claims):
            delta = abs(branch_scores["free"][index] - branch_scores["constrained"][index])
            branch_diff[str(year)][claim.claim_id] = round(delta, 3)
            branch_claims["free"][index].divergence_score = round(delta, 3)
            branch_claims["constrained"][index].divergence_score = round(delta, 3)

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
    if run_id is not None:
        insert_ecology_records(
            run_id=run_id,
            source_catalog=source_catalog,
            evidence_units=evidence_units,
            lineage_records=lineage_records,
        )

    scientific = llm.scientific
    degradation_reason = telemetry.degradation_reason or (
        getattr(llm, "degradation_reason", None) if not scientific else None
    )
    if not scientific:
        validation_notes = [
            f"DEGRADED RUN: {degradation_reason or 'A model call fell back to the deterministic client.'}",
            "This run is illustrative only; any branch contrast shown here is non-scientific.",
            "The fallback path is explicit so plausible-looking verdicts are never presented as scientific output.",
        ]
        mode_banner = "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"
    else:
        validation_notes = [
            "Contrast emerges from the ecology: contaminated citation carriers survive in free but are pruned by CIVER in constrained.",
            "Researcher and synthesist prompts are byte-frozen and never mention year, branch, drift, bias, or contamination.",
            "The deterministic panel makes the final claim call without any LLM aggregation step.",
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
        degradation_reason=degradation_reason,
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
        "llm_call_count": telemetry.call_count,
        "degradation_reason": degradation_reason,
    }
    return bundle, summary
