from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Literal

from app.config import YEARS
from app.db import insert_ecology_records
from app.llm import (
    EVIDENCE_UNIVERSE_PROMPT_TEMPLATE,
    EVIDENCE_UNIVERSE_PROMPT_TEMPLATE_DIGEST,
    PROMPT_TEMPLATE_DIGEST,
    RESEARCHER_PROMPT_TEMPLATE,
    SYNTHESIST_PROMPT_TEMPLATE,
    SYNTHESIST_PROMPT_TEMPLATE_DIGEST,
    SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE,
    SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE_DIGEST,
    LLMClient,
    parse_direction,
)
from app.models import (
    ArtifactBundle,
    AuditEvent,
    BranchName,
    BrimEvent,
    ClaimDirection,
    ClaimGraph,
    ClaimSnapshot,
    CiverVerdict,
    DriftSnapshot,
    EvidenceUnit,
    ExecutionWarrant,
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
RELEASE_THRESHOLD = 0.60
GENESIS_HASH = "GENESIS"
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
    return round(1 / (1 + math.exp(-0.115 * (year - 18))), 3)


def horizon_years(request: RunRequestModel) -> list[int]:
    years = request.horizons or list(YEARS)
    cleaned = sorted({int(year) for year in years if int(year) > 0})
    return cleaned or list(YEARS)


def _seed_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def _digest_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _is_synthetic_citation_id(cited_id: str) -> bool:
    return cited_id.startswith("S-") or "-syn-" in cited_id


def _carries_synthetic_substrate(unit: EvidenceUnit) -> bool:
    return (
        unit.provenance == "SYNTHETIC"
        or not unit.resolved_real_ids
        or any(_is_synthetic_citation_id(cited_id) for cited_id in unit.cited_ids)
    )


def _direction_from_items(
    cited_items: list[CorpusItem],
    model_direction: ClaimDirection,
) -> ClaimDirection:
    directional_scores = [
        _DIRECTION_VALUE[item.direction]
        for item in cited_items
        if item.direction != "NEUTRAL"
    ]
    if not directional_scores:
        return model_direction
    evidence_score = fmean(directional_scores)
    if abs(evidence_score) >= 0.34:
        return _panel_direction(evidence_score)
    return model_direction


def _pooled_panel_score(
    *,
    surviving_units: list[EvidenceUnit],
    synth_direction: ClaimDirection,
    prior_direction: ClaimDirection,
) -> float:
    scores: list[float] = []
    for unit in surviving_units:
        unit_score = _DIRECTION_VALUE[unit.direction]
        # Panel scoring is evidence-first: the synthesist provides a summary,
        # but cannot cancel a surviving admissible evidence unit by itself.
        # Synthetic substrate still gets enough weight to model contaminated
        # literature hardening into citable evidence in the free branch.
        weight = 2 if _carries_synthetic_substrate(unit) else 3
        scores.extend([unit_score] * weight)
    scores.append(_DIRECTION_VALUE[synth_direction])
    scores.append(_DIRECTION_VALUE[prior_direction] * 0.5)
    return fmean(scores) if scores else _DIRECTION_VALUE[synth_direction]


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
    return warrant.output_hash == _unit_output_hash(output)


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


def _parse_source_universe(response: str) -> list[tuple[ClaimDirection, str]]:
    records: list[tuple[ClaimDirection, str]] = []
    blocks = re.split(r"(?=SOURCE\s+\d+)", response or "", flags=re.IGNORECASE)
    for block in blocks:
        direction_match = re.search(
            r"DIRECTION:\s*(SUPPORTS|REFUTES|NEUTRAL)",
            block,
            re.IGNORECASE,
        )
        finding_match = re.search(
            r"FINDING:\s*(.+?)(?=\n\s*SOURCE\s+\d+|\Z)",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if direction_match is None:
            continue
        direction = direction_match.group(1).upper()
        finding = " ".join((finding_match.group(1) if finding_match else block).split())
        records.append((direction, finding or "Simulated source returned no finding."))
    return records


def mint_source_catalog(
    claims: list[ClaimSeed],
    input_text: str,
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> dict[str, list[SourceRecord]]:
    chunks = _sentence_chunks(input_text)
    if not chunks:
        chunks = [claim.text for claim in claims]

    catalog: dict[str, list[SourceRecord]] = {}
    for claim_index, claim in enumerate(claims):
        prompt = EVIDENCE_UNIVERSE_PROMPT_TEMPLATE.format(
            claim=claim.text,
            context="\n".join(chunks[:6]),
        )
        response = _invoke_model(
            llm,
            telemetry,
            f"source-universe/{claim.claim_id}",
            prompt,
            seed=_seed_int(f"source-universe:{claim.claim_id}:{_digest_key(input_text)}"),
        )
        universe_records = _parse_source_universe(response)
        sources: list[SourceRecord] = []
        for source_index in range(REAL_SOURCES_PER_CLAIM):
            if source_index < len(universe_records):
                direction, text = universe_records[source_index]
                locator = f"simulated-source:{source_index + 1}"
            else:
                chunk_index = (claim_index + source_index) % len(chunks)
                text = chunks[chunk_index]
                direction = "NEUTRAL"
                locator = f"input-context:{chunk_index + 1}"
            source_id = f"R-{claim.claim_id}-{source_index + 1}"
            sources.append(
                SourceRecord(
                    source_id=source_id,
                    claim_id=claim.claim_id,
                    label=f"Catalog source {source_index + 1}",
                    locator=locator,
                    direction=direction,
                    text=text,
                )
            )
        catalog[claim.claim_id] = sources
    return catalog


def _mint_contamination_bank(
    claim: ClaimSeed,
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for index in range(PRESEEDED_SYNTHETIC_UNITS):
        prompt = SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE.format(claim=claim.text)
        response = _invoke_model(
            llm,
            telemetry,
            f"contaminator/{claim.claim_id}/{index + 1}",
            prompt,
            seed=_seed_int(f"contaminator:{claim.claim_id}:{index + 1}"),
        )
        direction = parse_direction(response)
        rationale = _extract_rationale(response)
        unit = EvidenceUnit(
            id=f"{claim.claim_id}-syn-{index + 1}",
            claim_id=claim.claim_id,
            year=0,
            branch="free",
            producer="contaminator",
            cited_ids=[f"S-{claim.claim_id}-{index + 1}"],
            provenance="SYNTHETIC",
            direction=direction,
            rationale=rationale,
        )
        unit.output_hash = _unit_output_hash(unit)
        units.append(unit)
    return units


def _synthetic_units_for_year(bank: list[EvidenceUnit], year: int) -> list[EvidenceUnit]:
    count = max(1, min(len(bank), math.ceil(contamination_clock(year) * len(bank))))
    return bank[:count]


def _retrieve_catalog_sources(claim: ClaimSeed, real_sources: list[SourceRecord], year: int) -> list[SourceRecord]:
    clock = contamination_clock(year)
    # Defect C fix: real grounding erodes to zero as contamination rises,
    # symmetric across branches. Constrained retains real only via warranted
    # lineage; free loses it -> lost_real becomes non-empty (SCALE_SPEC §4).
    if clock < 0.40:
        count = 2
    elif clock < 0.70:
        count = 1
    else:
        count = 0
    if count == 0:
        return []
    offset = _seed_int(f"catalog:{claim.claim_id}:{year}") % len(real_sources)
    return [real_sources[(offset + index) % len(real_sources)] for index in range(count)]


def build_reachable_corpus(
    *,
    branch: BranchName,
    year: int,
    claim: ClaimSeed,
    state: BranchState,
    retrieved_sources: list[SourceRecord],
    synthetic_units: list[EvidenceUnit],
    warrants_by_output: dict[str, ExecutionWarrant],
) -> list[CorpusItem]:
    reachable: list[CorpusItem] = [
        CorpusItem(
            item_id=source.source_id,
            kind="real",
            text=source.text,
            rationale=source.text,
            direction=source.direction,
            cited_ids=[source.source_id],
            resolved_real_ids=[source.source_id],
            resolved_locators=[source.locator],
        )
        for source in retrieved_sources
    ]

    if branch == "constrained":
        prior_outputs = [
            unit
            for unit in state.output_history
            if _is_valid_warrant(warrants_by_output.get(unit.id), unit)
        ]
    else:
        prior_outputs = list(state.output_history)

    reachable.extend(
        CorpusItem(
            item_id=unit.id,
            kind="prior",
            text=unit.rationale,
            rationale=unit.rationale,
            direction=unit.direction,
            cited_ids=list(unit.cited_ids),
            resolved_real_ids=list(unit.resolved_real_ids),
            resolved_locators=list(unit.resolved_locators),
        )
        for unit in prior_outputs[-6:]
    )

    if branch == "free":
        reachable.extend(
            CorpusItem(
                item_id=unit.id,
                kind="synthetic",
                text=unit.rationale,
                rationale=unit.rationale,
                direction=unit.direction,
                cited_ids=list(unit.cited_ids),
                resolved_real_ids=[],
                resolved_locators=[],
            )
            for unit in synthetic_units
        )

    deduped: dict[str, CorpusItem] = {}
    for item in reachable:
        deduped[item.item_id] = item
    return list(deduped.values())


def _citation_score(
    item: CorpusItem,
    *,
    year: int,
    claim_id: str,
    citation_memory: list[str],
) -> float:
    clock = contamination_clock(year)
    if item.kind == "real":
        score = 1.05 - (clock * 0.78)
    elif item.kind == "prior":
        score = 0.62 + (0.18 if item.resolved_real_ids else -0.12)
    else:
        score = 0.10 + (clock * 1.05)

    if item.item_id in citation_memory:
        score += 0.18
    if item.resolved_real_ids and clock < 0.82:
        score += 0.08

    jitter = (_seed_int(f"score:{claim_id}:{year}:{item.item_id}") % 1000) / 10000
    return score + jitter


def _select_citations(
    *,
    claim: ClaimSeed,
    year: int,
    citation_memory: list[str],
    reachable_corpus: list[CorpusItem],
) -> tuple[list[CorpusItem], list[str]]:
    ranked = sorted(
        reachable_corpus,
        key=lambda item: _citation_score(
            item,
            year=year,
            claim_id=claim.claim_id,
            citation_memory=citation_memory,
        ),
        reverse=True,
    )
    selected = ranked[:3] if ranked else []
    debug = [f"{item.kind}:{item.item_id}" for item in selected]
    return selected, debug


def _build_evidence_block(cited_items: list[CorpusItem], citation_memory: list[str]) -> str:
    lines: list[str] = []
    for index, item in enumerate(cited_items, start=1):
        source_tag = f"source-{index}"
        if item.kind == "real":
            locator = item.resolved_locators[0] if item.resolved_locators else "locator:unknown"
            lines.append(
                f"- [{source_tag}] Real source ({locator}) reports {item.direction}: {item.text}"
            )
        elif item.kind == "prior":
            lines.append(
                f"- [{source_tag}] Prior warranted output concluded {item.direction}: {item.rationale}"
            )
        else:
            lines.append(
                f"- [{source_tag}] Synthetic carrier reported {item.direction}: {item.rationale}"
            )
    if citation_memory:
        lines.append(f"- Citation memory currently foregrounds: {', '.join(citation_memory[:3])}.")
    return "\n".join(lines) or "- No admissible evidence block available."


def _investigator_unit(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    branch_state: BranchState,
    reachable_corpus: list[CorpusItem],
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> tuple[EvidenceUnit, list[str]]:
    cited_items, selection_debug = _select_citations(
        claim=claim,
        year=year,
        citation_memory=branch_state.citation_memory,
        reachable_corpus=reachable_corpus,
    )
    evidence_block = _build_evidence_block(cited_items, branch_state.citation_memory)
    prompt = RESEARCHER_PROMPT_TEMPLATE.format(claim=claim.text, evidence_block=evidence_block)
    response = _invoke_model(
        llm,
        telemetry,
        f"investigator/{branch}/{claim.claim_id}/year-{year}",
        prompt,
        seed=_seed_int(f"investigator:{claim.claim_id}:{year}:{_digest_key(evidence_block)}"),
    )
    resolved_real_ids = sorted(
        {
            real_id
            for item in cited_items
            for real_id in item.resolved_real_ids
        }
    )
    resolved_locators = sorted(
        {
            locator
            for item in cited_items
            for locator in item.resolved_locators
        }
    )
    model_direction = parse_direction(response)
    evidence_direction = _direction_from_items(cited_items, model_direction)
    unit = EvidenceUnit(
        id=f"{claim.claim_id}-{branch}-investigator-{year}",
        claim_id=claim.claim_id,
        year=year,
        branch=branch,
        producer="investigator",
        cited_ids=[item.item_id for item in cited_items],
        provenance="REAL" if resolved_real_ids else "SYNTHETIC",
        direction=evidence_direction,
        rationale=_extract_rationale(response),
        resolved_real_ids=resolved_real_ids,
        resolved_locators=resolved_locators,
    )
    unit.output_hash = _unit_output_hash(unit)
    return unit, selection_debug


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
    passed = graph_complete and cited_resolve and resolvable and bool(unit.cited_ids)

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


def _synthesist_verdict(
    *,
    claim: ClaimSeed,
    year: int,
    branch: BranchName,
    surviving_units: list[EvidenceUnit],
    llm: LLMClient,
    telemetry: CallTelemetry,
) -> tuple[ClaimDirection, str]:
    if surviving_units:
        evidence_lines = [
            f"- Evidence unit {index}: {unit.direction}. {unit.rationale}"
            for index, unit in enumerate(surviving_units, start=1)
        ]
    else:
        evidence_lines = ["- No admissible evidence units survived the release gate."]
    prompt = SYNTHESIST_PROMPT_TEMPLATE.format(
        claim=claim.text,
        evidence_block="\n".join(evidence_lines),
    )
    response = _invoke_model(
        llm,
        telemetry,
        f"synthesist/{branch}/{claim.claim_id}/year-{year}",
        prompt,
        seed=_seed_int(f"synthesist:{claim.claim_id}:{year}:{_digest_key(prompt)}"),
    )
    return parse_direction(response), _extract_rationale(response)


def _clean_integrity_score(events: list[AuditEvent], scientific: bool) -> float:
    if not scientific:
        return 0.0
    if not verify_audit_chain(events):
        return 0.0
    penalties = 0.0
    for event in events:
        if event.severity == "block" and event.event_type != "article-i-refused":
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
    synthetic_carriers = [
        unit.id
        for unit in surviving_units
        if _carries_synthetic_substrate(unit)
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
    verdict: CiverVerdict,
    panel_direction: ClaimDirection,
    pooled_score: float,
    synth_rationale: str,
    lineage: LineageRecord,
    cycle_events: list[AuditEvent],
    blocked_count: int,
    emitted_count: int,
) -> ClaimSnapshot:
    strength = _strength_from_score(pooled_score, max(emitted_count, 1))
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
        f"{panel_direction}. Real sources retained: {', '.join(lineage.surviving_real) or 'none'}. "
        f"Lost real sources: {', '.join(lineage.lost_real) or 'none'}. "
        f"Synthetic carriers: {', '.join(lineage.synthetic_carriers) or 'none'}. "
        f"Synthesist rationale: {synth_rationale}"
    )
    snapshot = ClaimSnapshot(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        direction=panel_direction,
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

    source_catalog = mint_source_catalog(claims, input_text, llm, telemetry)
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
    warrants: list[ExecutionWarrant] = []
    warrants_by_output: dict[str, ExecutionWarrant] = {}
    audit_trail: list[AuditEvent] = []
    audit_counters: dict[tuple[str, BranchName], int] = {}
    last_hashes: dict[tuple[str, BranchName], str] = {}
    graph_lookup = {graph.claim_id: graph for graph in claim_graphs}

    for year in years:
        branch_scores: dict[str, list[float]] = {"free": [], "constrained": []}
        branch_claims: dict[str, list[ClaimSnapshot]] = {"free": [], "constrained": []}

        for branch in ("free", "constrained"):
            for claim in claims:
                state = states[(claim.claim_id, branch)]
                retrieved_sources = _retrieve_catalog_sources(claim, source_catalog[claim.claim_id], year)
                synthetic_units = _synthetic_units_for_year(synthetic_bank[claim.claim_id], year)
                reachable_corpus = build_reachable_corpus(
                    branch=branch,
                    year=year,
                    claim=claim,
                    state=state,
                    retrieved_sources=retrieved_sources,
                    synthetic_units=synthetic_units,
                    warrants_by_output=warrants_by_output,
                )
                reachable_lookup = {item.item_id: item for item in reachable_corpus}
                investigator, selection_debug = _investigator_unit(
                    claim=claim,
                    year=year,
                    branch=branch,
                    branch_state=state,
                    reachable_corpus=reachable_corpus,
                    llm=llm,
                    telemetry=telemetry,
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
                    integrity_score_after=1.0,
                    message=f"Investigator emitted {investigator.id} from {', '.join(selection_debug) or 'empty corpus'}.",
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

                claim_key = (claim.claim_id, branch)
                cycle_events = [
                    event
                    for event in audit_trail
                    if event.claim_id == claim.claim_id and event.branch == branch and event.year == year
                ]
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
                surviving_units = [investigator] if released and (branch == "free" or _is_valid_warrant(warrant, investigator)) else []
                synth_direction, synth_rationale = _synthesist_verdict(
                    claim=claim,
                    year=year,
                    branch=branch,
                    surviving_units=surviving_units,
                    llm=llm,
                    telemetry=telemetry,
                )
                pooled_score = _pooled_panel_score(
                    surviving_units=surviving_units,
                    synth_direction=synth_direction,
                    prior_direction=state.prior_direction,
                )
                panel_direction = _panel_direction(pooled_score)
                lineage = _lineage_record(
                    claim_id=claim.claim_id,
                    year=year,
                    branch=branch,
                    prior_state=state,
                    surviving_units=surviving_units,
                    verdict_before=state.prior_direction,
                    verdict_after=panel_direction,
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
                        f"synthetic carriers {lineage.synthetic_carriers or ['none']}."
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
                    panel_direction=panel_direction,
                    pooled_score=pooled_score,
                    synth_rationale=synth_rationale,
                    lineage=lineage,
                    cycle_events=cycle_events,
                    blocked_count=0 if released or branch == "free" else 1,
                    emitted_count=len(surviving_units),
                )
                branch_claims[branch].append(snapshot)
                branch_scores[branch].append(pooled_score)

                state.prior_direction = snapshot.direction
                state.prior_strength = snapshot.strength
                state.citation_memory = (investigator.cited_ids + state.citation_memory)[:6]
                state.surviving_real = set(lineage.surviving_real)
                if branch == "free":
                    state.output_history.append(investigator)
                elif warrant is not None and _is_valid_warrant(warrant, investigator):
                    state.output_history.append(investigator)
                state.output_history = state.output_history[-8:]

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
    scientific = llm.scientific
    degradation_reason = telemetry.degradation_reason or (
        getattr(llm, "degradation_reason", None) if not scientific else None
    )
    provenance_log = {
        "model": descriptor.name,
        "model_digest": descriptor.digest,
        "provider": request.backend,
        "base_url": request.base_url or "",
        "temperature": 0.2,
        "seed_mode": "engine-seeded-structure",
        "prompt_template_digests": {
            "researcher": PROMPT_TEMPLATE_DIGEST,
            "source_universe": EVIDENCE_UNIVERSE_PROMPT_TEMPLATE_DIGEST,
            "synthetic_evidence": SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE_DIGEST,
            "synthesist": SYNTHESIST_PROMPT_TEMPLATE_DIGEST,
        },
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
            "Researcher and synthesist prompts are byte-frozen and never mention year, branch, drift, bias, or contamination.",
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
        provenance_log=provenance_log,
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
        "llm_call_count": telemetry.call_count,
        "degradation_reason": degradation_reason,
        "bundle_seal": bundle.bundle_seal,
        "provenance_log": provenance_log,
    }
    return bundle, summary
