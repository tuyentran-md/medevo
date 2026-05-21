from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from app.models import (
    BranchName,
    ClaimDirection,
    EvidenceScope,
    GuidelineClaim,
    PubMedRecord,
    Study,
)
from app.synthesis import SrmaReview, parse_srma_review, synthesize_guideline_claim

if TYPE_CHECKING:
    from app.llm import LLMClient

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
    not resolve (no real PMID). Whether a given (claim, era, replicate) attempt
    fails is a DETERMINISTIC seeded draw against ``failure_rate`` — the
    contamination is the agent's own emitted output, never authored or labelled by
    the harness.

    DATA-GROUNDING (SPEC §1): the agent's analysis is bound to what it actually
    retrieves from the real source universe (the PubMed catalog), never invented
    from prior/parametric knowledge. A grounded study mirrors its cited record's
    scope/effect; a failed attempt over-reaches or fabricates rather than fill the
    gap from memory. Forcing data-grounding is what gives CIVER something real to
    bite on — see the data-grounding note on ``app.agents._srma_prompt``.

    ``run`` accepts a ``replicate`` index so multiple distinct studies are emitted
    per (claim, era): each replicate is its own seeded draw (grounded/ungrounded,
    record selection, failure mode), yielding ~k studies per claim per era (config
    constant ``app.ecology.STUDIES_PER_CLAIM_PER_ERA``).
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
        replicate: int = 0,
    ) -> tuple[Study, list[PubMedRecord]]:
        """Emit a Study plus the authoritative catalog the agent actually saw.

        The catalog (the real search results) is the source universe the gate
        resolves cites against — NOT the study's own claimed pmids. This is what
        makes Mode-1 fabricated cites fail to resolve while Mode-2 real cites
        resolve (and are caught instead by the scope clause).

        ``replicate`` distinguishes multiple study attempts within one (claim,
        era), each a distinct seeded draw and distinct study id.
        """
        attempt_fails = self._attempt_fails(
            claim_id=claim_id, year=simulated_year, replicate=replicate
        )

        result = self.pubmed.search(
            query=claim_text,
            max_year=max_pubmed_year or simulated_year,
            retmax=self.retmax,
        )
        catalog = list(result.records)
        record = _select_record(
            catalog, claim_id=claim_id, year=simulated_year, replicate=replicate
        )

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
                    replicate=replicate,
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
                    replicate=replicate,
                ),
                catalog,
            )

        from app.pubmed import extract_effect_estimate, infer_direction_from_record

        effect = extract_effect_estimate(f"{record.title} {record.abstract}")
        direction = infer_direction_from_record(record, claim_text=claim_text)
        numeric = effect.point is not None
        study = Study(
            id=f"{claim_id}-study-{simulated_year}-r{replicate}-{record.pmid}",
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

    def _attempt_fails(self, *, claim_id: str, year: int, replicate: int = 0) -> bool:
        if self.failure_rate <= 0.0:
            return False
        if self.failure_rate >= 1.0:
            return True
        # Deterministic seeded draw: reruns with the same seed reproduce the same
        # grounded/ungrounded pattern (cache-friendly, SPEC §9).
        key = f"failure:{claim_id}:{year}:{replicate}:{self.seed}"
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
    llm: "LLMClient | None" = None
    invoke_model: Callable[[str, str, int], str] | None = None
    seed_namespace: str = "srma"

    def run(
        self,
        *,
        run_id: str,
        branch: BranchName,
        claim_id: str,
        claim_text: str = "",
        year: int,
    ) -> GuidelineClaim:
        studies = self.study_reader.list_studies(
            run_id=run_id,
            branch=branch,
            claim_id=claim_id,
            up_to_year=year,
        )
        review = self._review(
            claim_id=claim_id,
            claim_text=claim_text,
            year=year,
            studies=studies,
        )
        return synthesize_guideline_claim(
            claim_id=claim_id,
            year=year,
            studies=studies,
            review=review,
        )

    def _review(
        self,
        *,
        claim_id: str,
        claim_text: str,
        year: int,
        studies: list[Study],
    ) -> SrmaReview:
        if not studies or self.invoke_model is None:
            return SrmaReview(summary="SRMA appraisal unavailable; deterministic weighting only.")
        prompt = _srma_prompt(
            claim_id=claim_id,
            claim_text=claim_text,
            year=year,
            studies=studies,
        )
        seed = int(
            hashlib.sha256(
                f"{self.seed_namespace}:{claim_id}:{year}:{len(studies)}".encode("utf-8")
            ).hexdigest()[:12],
            16,
        )
        response = self.invoke_model(f"srma/{claim_id}/year-{year}", prompt, seed)
        return parse_srma_review(response, study_ids=[study.id for study in studies])


def _select_record(
    records: list[PubMedRecord], *, claim_id: str, year: int, replicate: int = 0
) -> PubMedRecord | None:
    if not records:
        return None
    index = int(hashlib.sha256(f"{claim_id}:{year}:{replicate}".encode("utf-8")).hexdigest()[:8], 16)
    return records[index % len(records)]


def _ungrounded_study(
    *,
    claim_id: str,
    claim_text: str,
    simulated_year: int,
    record: PubMedRecord | None,
    replicate: int = 0,
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
    direction = _overreach_direction(
        claim_id=claim_id, year=simulated_year, replicate=replicate
    )

    # Mode 2 only possible when a real record is available to cite.
    wants_scope_mode = (
        record is not None
        and _failure_mode_is_scope(claim_id=claim_id, year=simulated_year, replicate=replicate)
    )

    if wants_scope_mode:
        assert record is not None
        inflation = _scope_inflation(claim_id=claim_id, year=simulated_year, replicate=replicate)
        source = record.scope
        claimed = EvidenceScope(
            population_low=max(0, source.population_low - inflation),
            population_high=source.population_high + inflation,
            year_start=source.year_start,
            year_end=source.year_end + inflation,
        )
        study = Study(
            id=f"{claim_id}-study-{simulated_year}-r{replicate}-overreach-{record.pmid}",
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
    fabricated_pmid = _fabricated_pmid(claim_id=claim_id, year=simulated_year, replicate=replicate)
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-r{replicate}-ungrounded",
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


def _failure_mode_is_scope(*, claim_id: str, year: int, replicate: int = 0) -> bool:
    if SCOPE_OVERREACH_SHARE <= 0.0:
        return False
    if SCOPE_OVERREACH_SHARE >= 1.0:
        return True
    key = f"failmode:{claim_id}:{year}:{replicate}"
    draw = (int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000) / 10_000
    return draw < SCOPE_OVERREACH_SHARE


def _scope_inflation(*, claim_id: str, year: int, replicate: int = 0) -> int:
    span = SCOPE_INFLATION_MAX - SCOPE_INFLATION_MIN + 1
    bucket = int(hashlib.sha256(f"inflation:{claim_id}:{year}:{replicate}".encode("utf-8")).hexdigest()[:8], 16) % span
    return SCOPE_INFLATION_MIN + bucket


def _fabricated_pmid(*, claim_id: str, year: int, replicate: int = 0) -> str:
    digest = hashlib.sha256(f"fabricated:{claim_id}:{year}:{replicate}".encode("utf-8")).hexdigest()[:10]
    return f"FAKE-{digest}"


def _overreach_direction(*, claim_id: str, year: int, replicate: int = 0) -> ClaimDirection:
    bucket = int(hashlib.sha256(f"overreach:{claim_id}:{year}:{replicate}".encode("utf-8")).hexdigest()[:8], 16) % 3
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


def _srma_prompt(
    *,
    claim_id: str,
    claim_text: str,
    year: int,
    studies: list[Study],
) -> str:
    rows = []
    for study in studies:
        rows.append(
            {
                "study_id": study.id,
                "direction": study.direction,
                "effect_point": study.effect_point,
                "effect_ci": study.effect_ci,
                "n": study.n,
                "quality": study.quality,
                "provenance": study.provenance,
                "numeric": study.numeric,
                "failure_mode": study.failure_mode,
                "claimed_scope": study.claimed_scope.model_dump(mode="json"),
                "source_scope": study.source_scope.model_dump(mode="json"),
                "rationale": study.rationale[:400],
            }
        )
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True)
    # DATA-GROUNDING HARD INSTRUCTION (SPEC §1; CONSTITUTION §1). This is what
    # makes CIVER's value VISIBLE: a model that shortcuts via prior/parametric
    # knowledge instead of the retrieved evidence produces an over-reaching /
    # unresolvable chain that the gate then catches. We are NOT trying to erase
    # the model's prior — only to force it to ground in the supplied corpus so the
    # gate has something real to bite on. If the evidence is insufficient, the
    # model must SAY SO rather than fill the gap from memory.
    return (
        "You are appraising a Tier-4 SR/MA corpus for one claim. "
        "HARD RULE: derive your appraisal and conclusion STRICTLY from the study "
        "list provided below. Do NOT re-query external sources, and do NOT assert "
        "anything from your prior/parametric knowledge or beyond what these studies "
        "actually show. If the supplied evidence is insufficient to appraise the "
        "claim, say so explicitly (lower certainty, neutral summary) — never fill "
        "the gap from memory. Use only the study list below. "
        "Return JSON only with keys: "
        '"study_appraisals" (array of {"study_id","weight_multiplier","concern"}), '
        '"certainty_adjustment" (number from -0.18 to 0.18), '
        '"summary" (short string). '
        "Weight up stronger grounded studies, weight down ungrounded or over-reaching studies, "
        "and adjust certainty for consistency, directness, and scope support. "
        f"claim_id={claim_id} year={year} claim={claim_text!r} studies={payload}"
    )
