from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.models import (
    BranchName,
    ClaimDirection,
    EvidenceScope,
    GuidelineClaim,
    PubMedRecord,
    Study,
)
from app.synthesis import synthesize_guideline_claim

if TYPE_CHECKING:
    from app.pubmed import PubMedClient


# Weak-agent failure fraction. SPEC §11-A anchors this to A0's measured LLM error
# rate (κ pending — A0 not yet finalized), so it lives here as ONE declared
# free parameter (the A0-anchor hook), never a magic literal in the draw logic.
DEFAULT_FAILURE_RATE = 0.3

# Of the attempts that fail, this fraction over-reach on SCOPE (Mode 2: cite a
# real resolvable PMID but assert a scope that exceeds the source). The remainder
# are Mode 1 (unresolvable: a fabricated PMID). Declared, not buried.
SCOPE_OVERREACH_SHARE = 0.5

# When an over-reaching agent inflates scope, the inflation magnitude (in years
# of age band / timeframe) is itself a seeded draw in this inclusive range. Mild
# inflations (<= the gate tolerance, app.ecology.SCOPE_TOLERANCE_YEARS) slip the
# gate by construction -> FNR can be > 0; aggressive ones are caught.
SCOPE_INFLATION_MIN = 1
SCOPE_INFLATION_MAX = 14


@dataclass(frozen=True)
class ResearchAgent:
    """Tier-1 research agent.

    A grounded attempt resolves a real PubMed record into a GROUNDED study with
    resolvable PMIDs. A weak/over-reaching attempt FAILS the way a real fallible
    LLM researcher fails: it emits an UNGROUNDED study whose evidence chain does
    not resolve (no real PMID). Whether a given (claim, era) attempt fails is a
    DETERMINISTIC seeded draw against ``failure_rate`` — the contamination is the
    agent's own emitted output, never authored or labelled by the harness.
    """

    pubmed: "PubMedClient"
    retmax: int = 8
    failure_rate: float = DEFAULT_FAILURE_RATE
    seed: int = 0

    def run(
        self,
        *,
        claim_id: str,
        claim_text: str,
        simulated_year: int,
        max_pubmed_year: int | None = None,
    ) -> tuple[Study, list[PubMedRecord]]:
        """Emit a Study plus the authoritative catalog the agent actually saw.

        The catalog (the real search results) is the source universe the gate
        resolves cites against — NOT the study's own claimed pmids. This is what
        makes Mode-1 fabricated cites fail to resolve while Mode-2 real cites
        resolve (and are caught instead by the scope clause).
        """
        attempt_fails = self._attempt_fails(claim_id=claim_id, year=simulated_year)

        result = self.pubmed.search(
            query=claim_text,
            max_year=max_pubmed_year or simulated_year,
            retmax=self.retmax,
        )
        catalog = list(result.records)
        record = _select_record(catalog, claim_id=claim_id, year=simulated_year)

        if attempt_fails:
            # The weak agent fails the way a real fallible LLM researcher fails.
            # The mode is a seeded draw: either it fabricates an unresolvable cite
            # (Mode 1) or it cites a REAL resolvable record but asserts a scope
            # that exceeds the source's (Mode 2). The harness never labels this as
            # contamination; UNGROUNDED + the over-reach is the agent's own output.
            return (
                _ungrounded_study(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    simulated_year=simulated_year,
                    record=record,
                ),
                catalog,
            )

        if record is None:
            # Agent could not ground the claim in any resolvable source -> the
            # attempt over-reached and emits an UNGROUNDED (unresolvable) study.
            return (
                _ungrounded_study(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    simulated_year=simulated_year,
                    record=None,
                ),
                catalog,
            )

        from app.pubmed import extract_effect_estimate, infer_direction_from_record

        effect = extract_effect_estimate(f"{record.title} {record.abstract}")
        direction = infer_direction_from_record(record, claim_text=claim_text)
        numeric = effect.point is not None
        study = Study(
            id=f"{claim_id}-study-{simulated_year}-{record.pmid}",
            claim_id=claim_id,
            year=simulated_year,
            direction=direction,
            effect_point=effect.point,
            effect_ci=(
                (effect.ci_low, effect.ci_high)
                if effect.ci_low is not None and effect.ci_high is not None
                else None
            ),
            n=_extract_sample_size(record),
            quality=_quality_score(record=record, numeric=numeric),
            provenance="GROUNDED",
            pmids=[record.pmid],
            numeric=numeric,
            rationale=_rationale(record),
            # Honest grounding: the claimed scope matches the source scope.
            # (A grounded study may still reach a WRONG direction — Mode 3,
            # "dốt-thành-thật" — but that has valid provenance and stays GROUNDED;
            # it cancels in the free-constrained contrast, SPEC §1.)
            claimed_scope=record.scope.model_copy(deep=True),
            source_scope=record.scope.model_copy(deep=True),
            failure_mode="none",
        )
        study.output_hash = _study_hash(study)
        return study, catalog

    def _attempt_fails(self, *, claim_id: str, year: int) -> bool:
        if self.failure_rate <= 0.0:
            return False
        if self.failure_rate >= 1.0:
            return True
        # Deterministic seeded draw: reruns with the same seed reproduce the same
        # grounded/ungrounded pattern (cache-friendly, SPEC §9).
        key = f"failure:{claim_id}:{year}:{self.seed}"
        draw = (int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000) / 10_000
        return draw < self.failure_rate


class StudyReader(Protocol):
    def list_studies(
        self,
        *,
        run_id: str,
        branch: BranchName,
        claim_id: str,
        up_to_year: int,
    ) -> list[Study]: ...


@dataclass(frozen=True)
class SrmaAgent:
    study_reader: StudyReader

    def run(
        self,
        *,
        run_id: str,
        branch: BranchName,
        claim_id: str,
        year: int,
    ) -> GuidelineClaim:
        studies = self.study_reader.list_studies(
            run_id=run_id,
            branch=branch,
            claim_id=claim_id,
            up_to_year=year,
        )
        return synthesize_guideline_claim(claim_id=claim_id, year=year, studies=studies)


def _select_record(records: list[PubMedRecord], *, claim_id: str, year: int) -> PubMedRecord | None:
    if not records:
        return None
    index = int(hashlib.sha256(f"{claim_id}:{year}".encode("utf-8")).hexdigest()[:8], 16)
    return records[index % len(records)]


def _ungrounded_study(
    *,
    claim_id: str,
    claim_text: str,
    simulated_year: int,
    record: PubMedRecord | None,
) -> Study:
    """A weak/over-reaching agent's failed attempt — a seeded MIX of two modes.

    Mode 1 (unresolvable): no real PMID; cites a fabricated id that will not
    resolve in the catalog. Caught by Article I resolvability.

    Mode 2 (scope over-reach): cites a REAL resolvable PMID (``record``) but the
    asserted claim scope EXCEEDS the source's population/timeframe. Must be caught
    by Article I's scope clause — but the inflation magnitude is itself seeded, so
    a mild over-reach within tolerance slips the gate (FNR > 0 by construction).

    The harness never labels this as contamination: UNGROUNDED + the over-reach
    are properties the agent itself emits.
    """
    direction = _overreach_direction(claim_id=claim_id, year=simulated_year)

    # Mode 2 only possible when a real record is available to cite.
    wants_scope_mode = (
        record is not None
        and _failure_mode_is_scope(claim_id=claim_id, year=simulated_year)
    )

    if wants_scope_mode:
        assert record is not None
        inflation = _scope_inflation(claim_id=claim_id, year=simulated_year)
        source = record.scope
        claimed = EvidenceScope(
            population_low=max(0, source.population_low - inflation),
            population_high=source.population_high + inflation,
            year_start=source.year_start,
            year_end=source.year_end + inflation,
        )
        study = Study(
            id=f"{claim_id}-study-{simulated_year}-overreach-{record.pmid}",
            claim_id=claim_id,
            year=simulated_year,
            direction=direction,
            quality=0.3,
            provenance="UNGROUNDED",
            pmids=[record.pmid],
            numeric=False,
            rationale=(
                f"Agent over-reached on '{claim_text}' at year {simulated_year}: "
                f"cited real source {record.pmid} but claimed a scope wider than "
                "the evidence supports."
            ),
            claimed_scope=claimed,
            source_scope=source.model_copy(deep=True),
            failure_mode="scope-overreach",
        )
        study.output_hash = _study_hash(study)
        return study

    # Mode 1: unresolvable fabricated citation.
    fabricated_pmid = _fabricated_pmid(claim_id=claim_id, year=simulated_year)
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-ungrounded",
        claim_id=claim_id,
        year=simulated_year,
        direction=direction,
        quality=0.2,
        provenance="UNGROUNDED",
        pmids=[fabricated_pmid],
        numeric=False,
        rationale=(
            f"Agent over-reached on '{claim_text}' at year {simulated_year}: "
            f"asserted a finding citing {fabricated_pmid}, which does not resolve."
        ),
        failure_mode="unresolvable",
    )
    study.output_hash = _study_hash(study)
    return study


def _failure_mode_is_scope(*, claim_id: str, year: int) -> bool:
    if SCOPE_OVERREACH_SHARE <= 0.0:
        return False
    if SCOPE_OVERREACH_SHARE >= 1.0:
        return True
    key = f"failmode:{claim_id}:{year}"
    draw = (int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000) / 10_000
    return draw < SCOPE_OVERREACH_SHARE


def _scope_inflation(*, claim_id: str, year: int) -> int:
    span = SCOPE_INFLATION_MAX - SCOPE_INFLATION_MIN + 1
    bucket = int(hashlib.sha256(f"inflation:{claim_id}:{year}".encode("utf-8")).hexdigest()[:8], 16) % span
    return SCOPE_INFLATION_MIN + bucket


def _fabricated_pmid(*, claim_id: str, year: int) -> str:
    digest = hashlib.sha256(f"fabricated:{claim_id}:{year}".encode("utf-8")).hexdigest()[:10]
    return f"FAKE-{digest}"


def _overreach_direction(*, claim_id: str, year: int) -> ClaimDirection:
    bucket = int(hashlib.sha256(f"overreach:{claim_id}:{year}".encode("utf-8")).hexdigest()[:8], 16) % 3
    return ("SUPPORTS", "REFUTES", "NEUTRAL")[bucket]


def _quality_score(*, record: PubMedRecord, numeric: bool) -> float:
    score = 0.45
    text = f"{record.title} {record.abstract}".lower()
    if numeric:
        score += 0.25
    if "randomized" in text or "randomised" in text:
        score += 0.2
    if "systematic review" in text or "meta-analysis" in text:
        score += 0.15
    if record.abstract:
        score += 0.05
    return min(score, 1.0)


def _extract_sample_size(record: PubMedRecord) -> int | None:
    text = f"{record.title} {record.abstract}"
    import re

    match = re.search(r"\b(?:n\s*=\s*|total of\s+)(\d{2,7})\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _rationale(record: PubMedRecord) -> str:
    source = record.abstract or record.title or "PubMed record contained no abstract text."
    return " ".join(source.split())[:600]


def _study_hash(study: Study) -> str:
    payload = study.model_dump(mode="json")
    payload.pop("output_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
