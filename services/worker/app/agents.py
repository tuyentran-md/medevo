from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.models import BranchName, ClaimDirection, GuidelineClaim, PubMedRecord, RecommendationLevel, Study

if TYPE_CHECKING:
    from app.pubmed import PubMedClient


@dataclass(frozen=True)
class ResearchAgent:
    pubmed: "PubMedClient"
    retmax: int = 8

    def run(
        self,
        *,
        claim_id: str,
        claim_text: str,
        simulated_year: int,
    ) -> Study:
        result = self.pubmed.search(
            query=claim_text,
            max_year=simulated_year,
            retmax=self.retmax,
        )
        record = _select_record(result.records, claim_id=claim_id, year=simulated_year)
        if record is None:
            return _empty_real_study(
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
            provenance="REAL",
            pmids=[record.pmid],
            numeric=numeric,
            rationale=_rationale(record),
        )
        study.output_hash = _study_hash(study)
        return study


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
        return GuidelineClaim(
            claim_id=claim_id,
            year=year,
            direction=_weighted_direction(studies),
            level=_placeholder_level(studies),
        )


def _weighted_direction(studies: list[Study]) -> ClaimDirection:
    scores = {"SUPPORTS": 1.0, "NEUTRAL": 0.0, "REFUTES": -1.0}
    if not studies:
        return "NEUTRAL"
    total_weight = sum(max(study.quality, 0.01) for study in studies)
    pooled = sum(scores[study.direction] * max(study.quality, 0.01) for study in studies) / total_weight
    if pooled >= 0.25:
        return "SUPPORTS"
    if pooled <= -0.25:
        return "REFUTES"
    return "NEUTRAL"


def _placeholder_level(studies: list[Study]) -> RecommendationLevel:
    if not studies:
        return "no-recommendation"
    real_quality = sum(study.quality for study in studies if study.provenance == "REAL")
    synthetic_quality = sum(study.quality for study in studies if study.provenance == "SYNTHETIC")
    if real_quality >= 2.4 and synthetic_quality == 0:
        return "strong-for" if _weighted_direction(studies) == "SUPPORTS" else "strong-against"
    if _weighted_direction(studies) == "SUPPORTS":
        return "conditional-for"
    if _weighted_direction(studies) == "REFUTES":
        return "conditional-against"
    return "no-recommendation"


def _select_record(records: list[PubMedRecord], *, claim_id: str, year: int) -> PubMedRecord | None:
    if not records:
        return None
    index = int(hashlib.sha256(f"{claim_id}:{year}".encode("utf-8")).hexdigest()[:8], 16)
    return records[index % len(records)]


def _empty_real_study(*, claim_id: str, claim_text: str, simulated_year: int) -> Study:
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-no-pmid",
        claim_id=claim_id,
        year=simulated_year,
        direction="NEUTRAL",
        quality=0.0,
        provenance="REAL",
        pmids=[],
        numeric=False,
        rationale=f"No PubMed record was available at year {simulated_year} for: {claim_text}",
    )
    study.output_hash = _study_hash(study)
    return study


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
