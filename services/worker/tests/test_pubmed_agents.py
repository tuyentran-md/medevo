from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents import ResearchAgent
from app.pubmed import PubMedClient, extract_effect_estimate


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
    assert len(http.calls) == 2
    cached_payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert cached_payload["max_year"] == 2020


def test_effect_extraction_reads_point_and_ci() -> None:
    effect = extract_effect_estimate("The pooled RR 0.72, 95% CI 0.61 to 0.84 favored treatment.")

    assert effect.point == 0.72
    assert effect.ci_low == 0.61
    assert effect.ci_high == 0.84
    assert effect.measure == "RR"


def test_research_agent_emits_real_grounded_study(tmp_path: Path) -> None:
    client = PubMedClient(cache_dir=tmp_path, http=FakeHttp(), min_interval_seconds=0)
    agent = ResearchAgent(pubmed=client, retmax=5)

    study = agent.run(
        claim_id="claim-1",
        claim_text="Routine antibiotics should be given for acute viral bronchiolitis in infants.",
        simulated_year=2020,
    )

    assert study.provenance == "REAL"
    assert study.pmids == ["111"]
    assert study.year == 2020
    assert study.direction == "REFUTES"
    assert study.numeric is True
    assert study.effect_point == 1.08
    assert study.effect_ci == (0.92, 1.26)
    assert study.n == 240
    assert study.output_hash
