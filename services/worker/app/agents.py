from __future__ import annotations

import hashlib
import json
import re
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


# Optional difficulty/temptation knob (SPEC §11-A A0-anchor hook). It NO LONGER
# decides whether an attempt fails — failure now EMERGES from the model's own
# emitted output (unresolvable cite / scope over-reach / unparseable). The value
# is surfaced to the prompt as a difficulty hint only and is otherwise inert;
# kept so ``sweep_failure_rate`` and the A0 anchor have a single declared
# parameter to reference, never a magic literal in any draw logic.
DEFAULT_FAILURE_RATE = 0.3


@dataclass(frozen=True)
class ResearchStudyEmission:
    """Parsed structured output of one Group-B research LLM call.

    Everything here is the MODEL's own assertion. ``parse_ok`` is False when the
    structured block was missing/garbled — itself a genuine failure surface that
    the harness resolves to UNGROUNDED, never re-fills from memory."""

    direction: ClaimDirection
    cited_pmids: list[str]
    claimed_scope: EvidenceScope
    rationale: str
    parse_ok: bool


@dataclass(frozen=True)
class ResearchAgent:
    """Tier-1 Group-B (literature) research agent — a genuine LLM act.

    For each (claim, era, replicate) attempt the agent retrieves the REAL,
    date-cut PubMed catalog and asks the LLM to appraise the supplied abstracts
    and conclude: a DIRECTION, the claim SCOPE it believes the evidence supports,
    and the PMIDS it relied on. The harness authors no study and draws no
    coin-flip; whether the emission is GROUNDED is DERIVED from the model's own
    output:

        GROUNDED  iff  every cited PMID resolves in the retrieved catalog
                       AND the claimed scope does not exceed the cited sources'
                       scope (within ``app.ecology.SCOPE_TOLERANCE_YEARS``);
        UNGROUNDED otherwise (fabricated/unresolvable cite, scope over-reach,
                       or unparseable/over-confident output).

    DATA-GROUNDING (SPEC §1): the prompt binds the model to the supplied corpus
    and forbids filling gaps from prior/parametric knowledge; if the abstracts
    don't support a conclusion the model must say so. The derived ``provenance``
    label is GROUND-TRUTH used ONLY for scoring (calibration matrix) — it never
    reaches the gate (``admit_evidence_unit`` re-derives resolvability/scope
    blind, SPEC §8.3).

    ``invoke_model`` routes the call through the run's telemetry-wrapped client so
    a real run shows ~one research call per study attempt; ``llm`` is a fallback
    direct client (tests can pass either). ``failure_rate`` is an inert difficulty
    hint (see module note), NOT a failure draw.
    """

    pubmed: "PubMedClient"
    llm: "LLMClient | None" = None
    invoke_model: Callable[[str, str, int], str] | None = None
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
        resolves cites against — NOT the study's own claimed pmids. The LLM is
        called exactly once per attempt; its emission decides grounding.
        """
        result = self.pubmed.search(
            query=claim_text,
            max_year=max_pubmed_year or simulated_year,
            retmax=self.retmax,
        )
        catalog = list(result.records)
        catalog_by_pmid = {record.pmid: record for record in catalog}

        prompt = _research_prompt(
            claim_id=claim_id,
            claim_text=claim_text,
            simulated_year=simulated_year,
            catalog=catalog,
            difficulty_hint=self.failure_rate,
        )
        seed = _attempt_seed(
            namespace="research",
            claim_id=claim_id,
            year=simulated_year,
            replicate=replicate,
        )
        raw = self._generate(
            label=f"research/{claim_id}/year-{simulated_year}/r{replicate}",
            prompt=prompt,
            seed=seed,
        )
        emission = parse_research_emission(raw)

        study = _study_from_emission(
            claim_id=claim_id,
            claim_text=claim_text,
            simulated_year=simulated_year,
            replicate=replicate,
            emission=emission,
            catalog_by_pmid=catalog_by_pmid,
        )
        return study, catalog

    def _generate(self, *, label: str, prompt: str, seed: int) -> str:
        if self.invoke_model is not None:
            return self.invoke_model(label, prompt, seed)
        if self.llm is not None:
            return self.llm.generate(prompt, seed=seed)
        raise RuntimeError(
            "ResearchAgent requires an llm or invoke_model to drive the research call."
        )


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


def _attempt_seed(*, namespace: str, claim_id: str, year: int, replicate: int) -> int:
    return int(
        hashlib.sha256(
            f"{namespace}:{claim_id}:{year}:{replicate}".encode("utf-8")
        ).hexdigest()[:12],
        16,
    )


_DIRECTIONS: tuple[ClaimDirection, ...] = ("SUPPORTS", "REFUTES", "NEUTRAL")
_DIRECTION_LINE_RE = re.compile(r"^\s*DIRECTION\s*:\s*(\w+)", re.IGNORECASE | re.MULTILINE)
_SCOPE_LINE_RE = re.compile(r"^\s*SCOPE\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PMIDS_LINE_RE = re.compile(r"^\s*PMIDS\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_RATIONALE_LINE_RE = re.compile(r"^\s*RATIONALE\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_AGE_RE = re.compile(r"pop\s*=?\s*(\d{1,3})\s*[-to]+\s*(\d{1,3})", re.IGNORECASE)
_YEAR_RE = re.compile(r"years?\s*=?\s*((?:19|20)\d{2})\s*[-to]+\s*((?:19|20)\d{2})", re.IGNORECASE)


def parse_research_emission(raw: str) -> ResearchStudyEmission:
    """Robustly parse the structured Group-B research output.

    Expected lines (order-free, case-insensitive):
        DIRECTION: SUPPORTS|REFUTES|NEUTRAL
        SCOPE: pop=<low>-<high> years=<start>-<end>
        PMIDS: <id>, <id>, ...   (empty allowed -> insufficient evidence)
        RATIONALE: <free text>

    A missing/garbled DIRECTION line (the load-bearing conclusion) makes the
    emission ``parse_ok=False`` — a genuine over-confident/unparseable failure
    that the caller resolves to UNGROUNDED rather than re-deriving from memory.
    """
    text = raw or ""
    dir_match = _DIRECTION_LINE_RE.search(text)
    direction_token = (dir_match.group(1).upper() if dir_match else "")
    parse_ok = direction_token in _DIRECTIONS
    direction: ClaimDirection = direction_token if parse_ok else "NEUTRAL"  # type: ignore[assignment]

    pmids: list[str] = []
    pmids_match = _PMIDS_LINE_RE.search(text)
    if pmids_match:
        body = pmids_match.group(1).strip()
        if body.lower() not in ("", "none", "n/a", "-"):
            pmids = [token.strip() for token in re.split(r"[,;\s]+", body) if token.strip()]

    scope = _parse_scope(_SCOPE_LINE_RE.search(text))

    rationale_match = _RATIONALE_LINE_RE.search(text)
    rationale = (
        " ".join(rationale_match.group(1).split())[:600]
        if rationale_match
        else " ".join(text.split())[:600]
    )

    return ResearchStudyEmission(
        direction=direction,
        cited_pmids=pmids,
        claimed_scope=scope,
        rationale=rationale or "Model returned no rationale.",
        parse_ok=parse_ok,
    )


def _parse_scope(match: "re.Match[str] | None") -> EvidenceScope:
    if match is None:
        return EvidenceScope()
    body = match.group(1)
    scope = EvidenceScope()
    age = _AGE_RE.search(body)
    if age:
        scope.population_low = int(age.group(1))
        scope.population_high = int(age.group(2))
    years = _YEAR_RE.search(body)
    if years:
        scope.year_start = int(years.group(1))
        scope.year_end = int(years.group(2))
    return scope


def _study_from_emission(
    *,
    claim_id: str,
    claim_text: str,
    simulated_year: int,
    replicate: int,
    emission: ResearchStudyEmission,
    catalog_by_pmid: dict[str, PubMedRecord],
) -> Study:
    """Turn the model's own emission into a Study, deriving GROUNDED/UNGROUNDED.

    Numbers (effect, n) are NEVER taken from the model — they are extracted
    deterministically from the cited record's real abstract text, so every
    figure is auditable verbatim to the source. The model owns direction, scope,
    and which PMIDs it relied on; the harness owns the resolvability/scope
    verdict that derives provenance for scoring only."""
    from app.pubmed import extract_effect_estimate

    resolved = [pmid for pmid in emission.cited_pmids if pmid in catalog_by_pmid]
    resolved_records = [catalog_by_pmid[pmid] for pmid in resolved]

    # Authoritative source scope = NARROWEST band over the records the model
    # actually cited (intersection of bands). Never inflated by the emission.
    source_scope = _narrowest_scope(resolved_records)

    # When the model cited a real source but emitted no parseable SCOPE line, fall
    # back to the source scope (an omitted scope is not an over-reach). Only an
    # explicitly-stated wider scope counts as the model's over-reach.
    claimed_scope = emission.claimed_scope
    if claimed_scope == EvidenceScope() and resolved_records:
        claimed_scope = source_scope.model_copy(deep=True)

    # Provenance is DERIVED, not stamped: every cited PMID must resolve AND the
    # claimed scope must not exceed the cited sources' scope. This is ground-truth
    # for scoring only; the gate re-derives it blind from cited_ids + scope.
    has_cites = bool(emission.cited_pmids)
    all_resolve = has_cites and len(resolved) == len(emission.cited_pmids)
    scope_overreaches = (
        bool(resolved_records)
        and claimed_scope.exceeds(source_scope, tolerance=0)
    )

    failure_mode: str
    if not emission.parse_ok or not has_cites:
        # Unparseable or no cite at all -> unresolvable provenance (Mode 1).
        provenance = "UNGROUNDED"
        failure_mode = "unresolvable"
    elif not all_resolve:
        # Cited a PMID absent from the retrieved set -> fabricated/unresolvable.
        provenance = "UNGROUNDED"
        failure_mode = "unresolvable"
    elif scope_overreaches:
        # Real resolvable cite but claimed scope exceeds the source (Mode 2).
        provenance = "UNGROUNDED"
        failure_mode = "scope-overreach"
    else:
        provenance = "GROUNDED"
        failure_mode = "none"

    # The effect/n are read from the FIRST resolvable cited record (if any), so a
    # grounded numeric study carries verbatim figures; an unresolvable emission
    # carries none. claimed_scope falls back to the source scope when the model
    # omitted a parseable SCOPE line for a grounded cite.
    primary = resolved_records[0] if resolved_records else None
    effect = (
        extract_effect_estimate(f"{primary.title} {primary.abstract}")
        if primary is not None
        else None
    )
    numeric = effect is not None and effect.point is not None

    suffix = "ungrounded" if provenance == "UNGROUNDED" else (primary.pmid if primary else "nores")
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-r{replicate}-{suffix}",
        claim_id=claim_id,
        year=simulated_year,
        direction=emission.direction,
        effect_point=effect.point if (effect and provenance == "GROUNDED") else None,
        effect_ci=(
            (effect.ci_low, effect.ci_high)
            if effect and provenance == "GROUNDED" and effect.ci_low is not None and effect.ci_high is not None
            else None
        ),
        n=_extract_sample_size(primary) if (primary and provenance == "GROUNDED") else None,
        quality=(
            _quality_score(record=primary, numeric=numeric)
            if (primary is not None and provenance == "GROUNDED")
            else (0.3 if failure_mode == "scope-overreach" else 0.2)
        ),
        provenance=provenance,
        pmids=list(emission.cited_pmids),
        numeric=numeric and provenance == "GROUNDED",
        rationale=emission.rationale,
        claimed_scope=claimed_scope,
        source_scope=source_scope,
        failure_mode=failure_mode,  # type: ignore[arg-type]
    )
    study.output_hash = _study_hash(study)
    return study


def _narrowest_scope(records: list[PubMedRecord]) -> EvidenceScope:
    if not records:
        return EvidenceScope()
    scope = records[0].scope.model_copy(deep=True)
    for record in records[1:]:
        other = record.scope
        scope.population_low = max(scope.population_low, other.population_low)
        scope.population_high = min(scope.population_high, other.population_high)
        scope.year_start = max(scope.year_start, other.year_start)
        scope.year_end = min(scope.year_end, other.year_end)
    return scope


def _research_prompt(
    *,
    claim_id: str,
    claim_text: str,
    simulated_year: int,
    catalog: list[PubMedRecord],
    difficulty_hint: float,
) -> str:
    sources = [
        {
            "pmid": record.pmid,
            "title": record.title,
            "abstract": (record.abstract or "")[:1200],
            "year": record.year,
            "journal": record.journal,
            # The source's own population/timeframe band, so the model can state a
            # scope grounded in (and not wider than) what the abstract studied.
            "population_band": [record.scope.population_low, record.scope.population_high],
            "year_band": [record.scope.year_start, record.scope.year_end],
        }
        for record in catalog
    ]
    payload = json.dumps(sources, ensure_ascii=True, sort_keys=True)
    # DATA-GROUNDING HARD INSTRUCTION (SPEC §1; CONSTITUTION §1). The model must
    # appraise the SUPPLIED abstracts and conclude only from them. If they do not
    # support a conclusion it must say so (NEUTRAL, empty PMIDS) rather than fill
    # the gap from prior/parametric knowledge. Cite ONLY PMIDS present in the
    # supplied set; do NOT invent identifiers. State the scope (population age
    # band + timeframe) the cited evidence actually supports — do not widen it.
    return (
        "You are a research agent appraising a single clinical claim against a "
        "date-limited PubMed retrieval. Read ONLY the abstracts supplied below; "
        "do NOT use external sources or prior/parametric knowledge. Conclude only "
        "what these abstracts support. If they are insufficient, say so (DIRECTION: "
        "NEUTRAL with no PMIDS). Cite ONLY pmids that appear in the supplied set; "
        "never invent an identifier. State the claim scope (population age band and "
        "publication-year band) the cited evidence actually supports and do NOT "
        "widen it beyond the sources.\n"
        "Respond with EXACTLY these four lines and nothing else:\n"
        "DIRECTION: SUPPORTS | REFUTES | NEUTRAL\n"
        "SCOPE: pop=<low>-<high> years=<start>-<end>\n"
        "PMIDS: <comma-separated pmids you relied on, or 'none'>\n"
        "RATIONALE: <one or two sentences grounded in the abstracts>\n"
        f"claim_id={claim_id} simulated_year={simulated_year} claim={claim_text!r} "
        f"sources={payload}"
    )


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
