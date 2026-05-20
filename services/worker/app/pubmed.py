from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from app.config import DATA_DIR
from app.models import ClaimDirection, EffectEstimate, PubMedRecord


ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class HttpClient(Protocol):
    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> Any: ...


@dataclass(frozen=True)
class PubMedSearchResult:
    query: str
    max_year: int
    pmids: list[str]
    records: list[PubMedRecord]


class PubMedClient:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        http: HttpClient | None = None,
        email: str | None = None,
        api_key: str | None = None,
        min_interval_seconds: float = 0.34,
    ) -> None:
        self.cache_dir = cache_dir or DATA_DIR / "pubmed_cache"
        self.http = http or requests
        self.email = email
        self.api_key = api_key
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0

    def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
        cache_path = self._cache_path(query=query, max_year=max_year, retmax=retmax)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            records = [PubMedRecord.model_validate(item) for item in payload["records"]]
            return PubMedSearchResult(
                query=payload["query"],
                max_year=int(payload["max_year"]),
                pmids=list(payload["pmids"]),
                records=records,
            )

        pmids = self._esearch(query=query, max_year=max_year, retmax=retmax)
        records = self._efetch(pmids=pmids, max_year=max_year) if pmids else []
        payload = {
            "query": query,
            "max_year": max_year,
            "pmids": pmids,
            "records": [record.model_dump(mode="json") for record in records],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return PubMedSearchResult(query=query, max_year=max_year, pmids=pmids, records=records)

    def _cache_path(self, *, query: str, max_year: int, retmax: int) -> Path:
        key = json.dumps(
            {"query": query, "max_year": max_year, "retmax": retmax},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _common_params(self) -> dict[str, str]:
        params = {"tool": "medevo"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._pace()
        response = self.http.get(
            f"{ENTREZ_BASE_URL}/{endpoint}",
            params={**self._common_params(), **params},
            timeout=20,
        )
        response.raise_for_status()
        return dict(response.json())

    def _request_text(self, endpoint: str, params: dict[str, Any]) -> str:
        self._pace()
        response = self.http.get(
            f"{ENTREZ_BASE_URL}/{endpoint}",
            params={**self._common_params(), **params},
            timeout=30,
        )
        response.raise_for_status()
        return str(response.text)

    def _pace(self) -> None:
        now = time.monotonic()
        remaining = self.min_interval_seconds - (now - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _esearch(self, *, query: str, max_year: int, retmax: int) -> list[str]:
        payload = self._request_json(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": "pub_date",
                "datetype": "pdat",
                "mindate": "1900",
                "maxdate": str(max_year),
            },
        )
        return [str(pmid) for pmid in payload.get("esearchresult", {}).get("idlist", [])]

    def _efetch(self, *, pmids: list[str], max_year: int) -> list[PubMedRecord]:
        xml_text = self._request_text(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        return [
            record
            for record in _parse_pubmed_xml(xml_text)
            if record.year <= max_year
        ]


def _parse_pubmed_xml(xml_text: str) -> list[PubMedRecord]:
    root = ET.fromstring(xml_text)
    records: list[PubMedRecord] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        year = _extract_year(article)
        if not pmid or year is None:
            continue
        title = " ".join((article.findtext(".//ArticleTitle") or "").split())
        abstract = " ".join(
            " ".join(node.itertext())
            for node in article.findall(".//Abstract/AbstractText")
        )
        journal = " ".join((article.findtext(".//Journal/Title") or "").split())
        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=" ".join(abstract.split()),
                year=year,
                journal=journal,
                locator=f"PMID:{pmid}",
            )
        )
    return records


def _extract_year(article: ET.Element) -> int | None:
    for path in (
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//DateCompleted/Year",
    ):
        raw = article.findtext(path)
        if raw and raw.strip().isdigit():
            return int(raw.strip())
    medline_date = article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"(19|20)\d{2}", medline_date)
    return int(match.group(0)) if match else None


def extract_effect_estimate(text: str) -> EffectEstimate:
    compact = " ".join((text or "").split())
    ci_match = re.search(
        r"(?P<measure>RR|OR|HR|MD|SMD|risk ratio|odds ratio|hazard ratio)"
        r"[^0-9\-]{0,30}(?P<point>-?\d+(?:\.\d+)?)"
        r".{0,80}?95\s*%?\s*CI[^0-9\-]*(?P<low>-?\d+(?:\.\d+)?)\s*(?:to|-|,)\s*(?P<high>-?\d+(?:\.\d+)?)",
        compact,
        re.IGNORECASE,
    )
    if ci_match:
        return EffectEstimate(
            point=float(ci_match.group("point")),
            ci_low=float(ci_match.group("low")),
            ci_high=float(ci_match.group("high")),
            measure=ci_match.group("measure").upper(),
        )
    point_match = re.search(
        r"(?P<measure>RR|OR|HR|MD|SMD|risk ratio|odds ratio|hazard ratio)"
        r"[^0-9\-]{0,30}(?P<point>-?\d+(?:\.\d+)?)",
        compact,
        re.IGNORECASE,
    )
    if point_match:
        return EffectEstimate(
            point=float(point_match.group("point")),
            measure=point_match.group("measure").upper(),
        )
    return EffectEstimate()


def infer_direction_from_record(record: PubMedRecord, claim_text: str = "") -> ClaimDirection:
    text = f"{record.title} {record.abstract}".lower()
    refuting = (
        "no significant difference",
        "did not reduce",
        "does not reduce",
        "no benefit",
        "not associated",
        "increased harm",
        "associated with harm",
        "worse",
    )
    supporting = (
        "reduced",
        "improved",
        "benefit",
        "effective",
        "superior",
        "lower risk",
        "decreased",
    )
    if any(term in text for term in refuting):
        return "REFUTES"
    if any(term in text for term in supporting):
        return "SUPPORTS"
    return "NEUTRAL"
