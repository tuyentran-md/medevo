"""Canonical MeSH lookup: claim outcome and evidence outcome to MeSH tree
numbers, with hierarchy matching (descendant explosion — the standard SR
practice). The MedEvo substitution of "human-in-the-loop" attribute confirmation
by "medevo-rule-in-the-loop" lives here for the outcome attribute: the harness
deterministically derives the canonical address of each side from NLM, the
agent cannot redefine them.

Caching is file-based and aggressive — MeSH descriptors are stable, and Entrez
throttles to 3 req/s without an API key. The cache is durable across runs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

import requests

from app.config import DATA_DIR

ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MESH_CACHE_DIR = DATA_DIR / "mesh_cache"

# Pattern that extracts a comma-separated tree-number list from a MeSH efetch
# response: the "Tree Number(s):" label is a clean anchor present on every
# descriptor record. Capturing this specific line avoids the false-positives
# that a raw regex over the whole document would hit.
_TREE_NUMBER_LINE_RE = re.compile(r"Tree Number\(s\):\s*([^\n]+)")
_TREE_NUMBER_RE = re.compile(r"\b[A-Z]\d{1,2}(?:\.\d{3})*\b")


class _HttpLike(Protocol):
    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> Any: ...


class MeSHClient:
    """Resolve a MeSH descriptor name to its tree numbers via Entrez.

    File-cached so repeat lookups are free and the test suite can prime the
    cache for offline runs. Injectable HTTP so unit tests can mock the network."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        http: _HttpLike | None = None,
        email: str | None = None,
        api_key: str | None = None,
        min_interval_seconds: float = 0.34,
    ) -> None:
        self.cache_dir = cache_dir or MESH_CACHE_DIR
        self.http = http or requests
        self.email = email
        self.api_key = api_key
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0

    def tree_numbers(self, descriptor: str) -> list[str]:
        """Tree numbers of a MeSH descriptor name (e.g. 'coronary disease' ->
        ['C14.280.647.250', 'C14.907.585.250']). Returns [] when the descriptor
        cannot be resolved, when Entrez is unreachable, or for non-clinical
        terms — callers treat empty as "cannot enforce" (permissive)."""
        key = (descriptor or "").strip().lower()
        if not key:
            return []
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                return list(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                pass  # fall through to refetch
        try:
            uids = self._esearch_mesh(key)
            trees: list[str] = []
            for uid in uids[:3]:
                trees.extend(self._fetch_tree_numbers(uid))
            # Dedup, preserve order.
            seen: set[str] = set()
            unique = [t for t in trees if not (t in seen or seen.add(t))]
        except Exception:
            unique = []
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(unique), encoding="utf-8")
        return unique

    def _esearch_mesh(self, term: str) -> list[str]:
        self._pace()
        params = self._common({"db": "mesh", "term": f"{term}[MeSH Terms]", "retmode": "json"})
        r = self.http.get(f"{ENTREZ_BASE_URL}/esearch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        return [
            str(uid)
            for uid in r.json().get("esearchresult", {}).get("idlist", [])
        ]

    def _fetch_tree_numbers(self, uid: str) -> list[str]:
        self._pace()
        params = self._common({"db": "mesh", "id": uid, "retmode": "text", "rettype": "full"})
        r = self.http.get(f"{ENTREZ_BASE_URL}/efetch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        match = _TREE_NUMBER_LINE_RE.search(r.text)
        if not match:
            return []
        return _TREE_NUMBER_RE.findall(match.group(1))

    def _common(self, extra: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": "medevo"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(extra)
        return params

    def _pace(self) -> None:
        remaining = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


_DEFAULT_CLIENT: MeSHClient | None = None


def _default_client() -> MeSHClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = MeSHClient()
    return _DEFAULT_CLIENT


def descriptor_tree_numbers(name: str, *, client: MeSHClient | None = None) -> list[str]:
    return (client or _default_client()).tree_numbers(name)


def claim_outcome_trees(
    claim_outcome_phrase: str, *, client: MeSHClient | None = None
) -> list[str]:
    """Tree numbers for the claim's outcome phrase, e.g. 'coronary heart disease'
    -> ['C14.280.647.250', 'C14.907.585.250'] (Coronary Disease)."""
    return descriptor_tree_numbers(claim_outcome_phrase, client=client)


def evidence_mesh_trees(
    mesh_terms: list[str], *, client: MeSHClient | None = None
) -> set[str]:
    """Union of tree numbers across an article's MeSH DescriptorName list."""
    out: set[str] = set()
    for term in mesh_terms or []:
        out.update(descriptor_tree_numbers(term, client=client))
    return out


def mesh_hierarchy_match(
    claim_trees: list[str] | set[str], evidence_trees: set[str]
) -> bool:
    """True if any evidence tree number is the claim's tree number OR a
    DESCENDANT of it (standard SR `[MeSH]` explosion semantics: a study indexed
    at a more specific descriptor counts as evidence for the broader claim
    outcome, but a study indexed at a broader ancestor does NOT — broader-
    indexed studies cover a heterogeneous mix that may not bear on the claim).
    """
    claim_set = {t for t in claim_trees if t}
    for ct in claim_set:
        for et in evidence_trees:
            if et == ct or et.startswith(ct + "."):
                return True
    return False
