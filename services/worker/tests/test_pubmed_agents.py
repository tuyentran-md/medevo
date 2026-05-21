from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents import ResearchAgent, parse_research_emission, pubmed_query_candidates
from app.ecology import (
    ClaimSeed,
    CorpusItem,
    SCOPE_TOLERANCE_YEARS,
    admit_evidence_unit,
    _reachable_lookup_from_catalog,
    _study_to_evidence_unit,
)
from app.llm import ModelDescriptor
from app.models import EvidenceScope, PubMedRecord
from app.pubmed import PubMedClient, PubMedSearchResult, extract_effect_estimate
from app.simulator import build_claim_graph


PUBMED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue><PubDate><Year>2015</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Antibiotics did not reduce bronchiolitis admissions</ArticleTitle>
        <Abstract>
          <AbstractText>Randomized trial n=240 found antibiotics did not reduce admissions; RR 1.08, 95% CI 0.92 to 1.26.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <Journal>
          <Title>Future Journal</Title>
          <JournalIssue><PubDate><Year>2027</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Future-only result</ArticleTitle>
        <Abstract><AbstractText>Should be excluded by maxdate.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class FakeResponse:
    def __init__(self, *, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append((url, params))
        if url.endswith("esearch.fcgi"):
            return FakeResponse(payload={"esearchresult": {"idlist": ["111", "222"]}})
        return FakeResponse(text=PUBMED_XML)


def test_pubmed_client_respects_date_cut_and_uses_cache(tmp_path: Path) -> None:
    http = FakeHttp()
    client = PubMedClient(cache_dir=tmp_path, http=http, min_interval_seconds=0)

    first = client.search(query="viral bronchiolitis antibiotics", max_year=2020, retmax=5)
    second = client.search(query="viral bronchiolitis antibiotics", max_year=2020, retmax=5)

    assert [record.pmid for record in first.records] == ["111"]
    assert second.records[0].pmid == "111"
    assert first.records[0].scope.year_start == 2015
    assert first.records[0].scope.year_end == 2015
    assert second.records[0].scope.year_start == 2015
    assert second.records[0].scope.year_end == 2015
    assert len(http.calls) == 2
    cached_payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert cached_payload["max_year"] == 2020


def test_effect_extraction_reads_point_and_ci() -> None:
    effect = extract_effect_estimate("The pooled RR 0.72, 95% CI 0.61 to 0.84 favored treatment.")

    assert effect.point == 0.72
    assert effect.ci_low == 0.61
    assert effect.ci_high == 0.84
    assert effect.measure == "RR"


def test_cvd_pubmed_query_candidates_are_short_domain_queries() -> None:
    queries = pubmed_query_candidates(
        "Cigarette smoking is causally associated with dose-dependent increases in coronary heart disease risk."
    )
    assert queries[0] == "cigarette smoking coronary heart disease"
    assert queries[-1].startswith("Cigarette smoking")


def test_research_agent_falls_back_across_pubmed_queries_before_llm() -> None:
    class FallbackPubMed:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
            self.queries.append(query)
            if query == "tobacco coronary heart disease cohort":
                record = PubMedRecord(
                    pmid="222",
                    title="Smoking cohort",
                    abstract="Cohort study n=500 found increased coronary risk; RR 1.40, 95% CI 1.20 to 1.70.",
                    year=2012,
                    scope=EvidenceScope(population_low=40, population_high=80, year_start=2012, year_end=2012),
                )
                return PubMedSearchResult(query=query, max_year=max_year, pmids=["222"], records=[record])
            return PubMedSearchResult(query=query, max_year=max_year, pmids=[], records=[])

    calls: list[str] = []
    agent = ResearchAgent(
        pubmed=FallbackPubMed(),
        llm=ScriptedLLM(
            "DIRECTION: SUPPORTS\nSCOPE: pop=40-80 years=2012-2012\nPMIDS: 222\nRATIONALE: cohort.",
            calls=calls,
        ),
        retmax=12,
    )
    study, catalog = agent.run(
        claim_id="claim-1",
        claim_text="Cigarette smoking is causally associated with dose-dependent increases in coronary heart disease risk.",
        simulated_year=2012,
    )
    assert [record.pmid for record in catalog] == ["222"]
    assert study.provenance == "GROUNDED"
    assert len(calls) == 1
    assert agent.pubmed.queries[:2] == [  # type: ignore[attr-defined]
        "cigarette smoking coronary heart disease",
        "tobacco coronary heart disease cohort",
    ]


def test_empty_pubmed_catalog_does_not_spend_llm_call() -> None:
    class EmptyPubMed:
        def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
            return PubMedSearchResult(query=query, max_year=max_year, pmids=[], records=[])

    calls: list[str] = []
    agent = ResearchAgent(
        pubmed=EmptyPubMed(),
        llm=ScriptedLLM("DIRECTION: SUPPORTS\nSCOPE: pop=0-120 years=2012-2012\nPMIDS: none\nRATIONALE: x.", calls=calls),
    )
    study, catalog = agent.run(
        claim_id="claim-empty",
        claim_text="Unretrievable claim",
        simulated_year=2012,
    )
    assert catalog == []
    assert calls == []
    assert study.provenance == "UNGROUNDED"
    assert study.source_scope.year_end == 2012


# --------------------------------------------------------------------------- #
# Scripted LLM: routes by the response text injected per test. The research
# agent now drives every study via a genuine generate() call, so the four
# scenarios below are produced by the MODEL's emission, not a harness coin flip.
# --------------------------------------------------------------------------- #
class ScriptedLLM:
    scientific = True
    degradation_reason = None

    def __init__(self, response: str, *, calls: list[str] | None = None) -> None:
        self._response = response
        self.calls = calls if calls is not None else []

    def generate(self, prompt: str, *, seed: int) -> str:
        self.calls.append(prompt)
        return self._response

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="scripted", digest="test")


class ScopedPubMed:
    """One real record (pmid 111) with a NARROW source scope so scope over-reach
    is observable; serves as the retrieved catalog the gate resolves against."""

    SCOPE = EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)

    def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
        record = PubMedRecord(
            pmid="111",
            title="Antibiotics did not reduce bronchiolitis admissions",
            abstract="Randomized trial n=240 found antibiotics did not reduce admissions; RR 1.08, 95% CI 0.92 to 1.26.",
            year=min(max_year, 2018),
            journal="Test Journal",
            locator="PMID:111",
            scope=self.SCOPE.model_copy(deep=True),
        )
        return PubMedSearchResult(query=query, max_year=max_year, pmids=["111"], records=[record])


def _run(agent_response: str):
    calls: list[str] = []
    agent = ResearchAgent(
        pubmed=ScopedPubMed(), llm=ScriptedLLM(agent_response, calls=calls), retmax=5
    )
    study, catalog = agent.run(
        claim_id="claim-1",
        claim_text="Routine antibiotics should be given for acute viral bronchiolitis in infants.",
        simulated_year=2020,
    )
    return study, catalog, calls


def _admit_in_gate(study, catalog):
    """Push the study through the (blind) constrained gate exactly as ecology does."""
    catalog_pmids = {record.pmid for record in catalog}
    unit = _study_to_evidence_unit(study=study, branch="constrained", catalog_pmids=catalog_pmids)
    claim = ClaimSeed("claim-1", "Some clinical claim.", "moderate")
    verdict, _warrant = admit_evidence_unit(
        run_id="run-1",
        claim=claim,
        claim_graph=build_claim_graph(claim),
        branch="constrained",
        year=2020,
        unit=unit,
        reachable_lookup=_reachable_lookup_from_catalog(catalog),
        warrants_by_output={},
        threshold=0.6,
    )
    return verdict


def test_every_research_attempt_calls_the_llm() -> None:
    _study, _catalog, calls = _run(
        "DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok."
    )
    assert len(calls) == 1
    assert "DIRECTION: SUPPORTS | REFUTES | NEUTRAL" in calls[0]


def test_grounded_emission_is_admitted_by_the_gate() -> None:
    # Model cites the retrieved PMID at the source scope -> GROUNDED -> admitted.
    study, catalog, _ = _run(
        "DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: trial showed no benefit."
    )
    assert study.provenance == "GROUNDED"
    assert study.failure_mode == "none"
    assert study.pmids == ["111"]
    assert study.direction == "REFUTES"
    # Numbers are extracted verbatim from the cited abstract, never from the model.
    assert study.numeric is True
    assert study.effect_point == 1.08
    assert study.effect_ci == (0.92, 1.26)
    assert study.n == 240
    assert study.output_hash
    assert _admit_in_gate(study, catalog).passed is True


def test_overreaching_scope_emission_is_ungrounded_and_refused() -> None:
    # Cites the real PMID but inflates the population/timeframe far beyond the
    # source -> the model's own over-reach -> UNGROUNDED -> refused by the gate.
    study, catalog, _ = _run(
        "DIRECTION: SUPPORTS\nSCOPE: pop=0-100 years=1990-2025\nPMIDS: 111\nRATIONALE: overclaim."
    )
    assert study.provenance == "UNGROUNDED"
    assert study.failure_mode == "scope-overreach"
    # It still carries real, source-extracted numbers. Otherwise SRMA would get
    # an easy proxy for the hidden provenance label and the CIVER contrast would
    # be scientifically weak.
    assert study.numeric is True
    assert study.effect_point == 1.08
    assert study.effect_ci == (0.92, 1.26)
    assert study.n == 240
    verdict = _admit_in_gate(study, catalog)
    assert verdict.passed is False
    assert any("scope clause" in reason for reason in verdict.reasons)


def test_fabricated_citation_emission_is_ungrounded_and_refused() -> None:
    # Cites a PMID that is NOT in the retrieved catalog -> unresolvable.
    study, catalog, _ = _run(
        "DIRECTION: SUPPORTS\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 99999\nRATIONALE: hallucinated cite."
    )
    assert study.provenance == "UNGROUNDED"
    assert study.failure_mode == "unresolvable"
    assert _admit_in_gate(study, catalog).passed is False


def test_unparseable_emission_is_ungrounded_and_refused() -> None:
    # Garbled / over-confident free text with no structured conclusion.
    study, catalog, _ = _run("Yes, antibiotics clearly help. Trust me.")
    assert study.provenance == "UNGROUNDED"
    assert study.failure_mode == "unresolvable"
    assert _admit_in_gate(study, catalog).passed is False


def test_mild_scope_overreach_within_tolerance_slips_the_gate() -> None:
    # Model inflates scope by exactly the gate tolerance: ground-truth UNGROUNDED
    # (agent uses tolerance=0) yet the blind gate admits it -> a genuine FNR.
    study, catalog, _ = _run(
        f"DIRECTION: REFUTES\nSCOPE: pop=40-{60 + SCOPE_TOLERANCE_YEARS} years=2015-2018\nPMIDS: 111\nRATIONALE: mild."
    )
    assert study.provenance == "UNGROUNDED"
    assert study.failure_mode == "scope-overreach"
    assert _admit_in_gate(study, catalog).passed is True


def test_emission_parser_is_robust_to_ordering_and_case() -> None:
    emission = parse_research_emission(
        "rationale: because.\npmids: 111, 222\nscope: pop=40-60 years=2015-2018\ndirection: supports"
    )
    assert emission.parse_ok is True
    assert emission.direction == "SUPPORTS"
    assert emission.cited_pmids == ["111", "222"]
    assert emission.claimed_scope.population_low == 40
