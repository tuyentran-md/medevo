from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.models import BranchName, ClaimDirection, GuidelineClaim, PubMedRecord, Study
from app.synthesis import synthesize_guideline_claim

if TYPE_CHECKING:
    from app.pubmed import PubMedClient


# Placeholder weak-agent failure fraction. A later slice (SPEC §11-A / Slice B)
# anchors this to A0's measured LLM error rate; it lives here as a single named
# parameter, never a magic literal buried in the draw logic.
DEFAULT_FAILURE_RATE = 0.3


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
    ) -> Study:
        if self._attempt_fails(claim_id=claim_id, year=simulated_year):
            return _ungrounded_study(
                claim_id=claim_id,
                claim_text=claim_text,
                simulated_year=simulated_year,
            )

        result = self.pubmed.search(
            query=claim_text,
            max_year=max_pubmed_year or simulated_year,
            retmax=self.retmax,
        )
        record = _select_record(result.records, claim_id=claim_id, year=simulated_year)
        if record is None:
            # Agent could not ground the claim in any resolvable source -> the
            # attempt over-reached and emits an UNGROUNDED study.
            return _ungrounded_study(
                claim_id=claim_id,
                claim_text=claim_text,
                simulated_year=simulated_year,
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
        )
        study.output_hash = _study_hash(study)
        return study

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


def _ungrounded_study(*, claim_id: str, claim_text: str, simulated_year: int) -> Study:
    """A weak/over-reaching agent's failed attempt.

    It looks structurally plausible (it asserts a direction) but its evidence
    chain does not resolve: no real PMID. The harness does not label this as
    contamination — UNGROUNDED is the agent's own emitted property, and CIVER
    refuses it later only because the chain is unresolvable (empty source ids).
    """
    direction = _overreach_direction(claim_id=claim_id, year=simulated_year)
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-ungrounded",
        claim_id=claim_id,
        year=simulated_year,
        direction=direction,
        quality=0.2,
        provenance="UNGROUNDED",
        pmids=[],
        numeric=False,
        rationale=(
            f"Agent over-reached on '{claim_text}' at year {simulated_year}: "
            "asserted a finding without a resolvable source."
        ),
    )
    study.output_hash = _study_hash(study)
    return study


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
