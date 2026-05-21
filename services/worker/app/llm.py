"""LLM client layer for MedEvo (v3 — emergent agent failure).

Integrity rules (SPEC §0/§8):
- The model is never told the year, the branch, or to "drift"/"be biased".
  Drift is endogenous: it emerges from the research agents' own failures to
  ground a claim, never from a harness-authored contamination prompt.
- The deterministic fake is for the no-model fallback and tests only. Runs that
  use it are stamped non-scientific by the simulator.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass
class ModelDescriptor:
    name: str
    digest: str


class LLMClient(Protocol):
    scientific: bool
    degradation_reason: str | None

    def generate(self, prompt: str, *, seed: int) -> str: ...

    def describe(self) -> ModelDescriptor: ...


class OllamaClient:
    """Local-model client (Ollama). Reproducible via per-call seed + low temp.

    NO-LOCAL rule (SPEC v3 §9): local open-weight models are too weak to do
    real research-agent work — they underperform even free OpenRouter models —
    so a local run is ILLUSTRATIVE only and is never stamped scientific. Scored
    runs must use a cloud flagship via OpenAICompatClient.
    """

    scientific = False
    degradation_reason = None

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate(self, prompt: str, *, seed: int) -> str:
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"seed": seed, "temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        return str(response.json().get("response", ""))

    def describe(self) -> ModelDescriptor:
        digest = "unknown"
        try:
            resp = requests.post(
                f"{self._base_url}/api/show",
                json={"name": self._model},
                timeout=5,
            )
            if resp.ok:
                payload = resp.json()
                digest = hashlib.sha256(
                    str(payload.get("modelfile", self._model)).encode("utf-8")
                ).hexdigest()[:16]
        except Exception:
            digest = "unreachable"
        return ModelDescriptor(name=self._model, digest=digest)


class OpenAICompatClient:
    """Any OpenAI-/chat-completions-compatible endpoint (OpenRouter, Gemini
    OpenAI-compat, vLLM, etc.). BYOK: key + base_url + model are caller-supplied,
    nothing hardcoded. Reasoning models (e.g. deepseek-v4-flash) spend tokens on
    a hidden CoT, so max_tokens is generous and only `message.content` is used
    (the chain-of-thought in `reasoning` is intentionally ignored)."""

    scientific = True
    degradation_reason = None

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._last_call_at = 0.0

    def _pace(self) -> None:
        now = time.monotonic()
        remaining = 0.35 - (now - self._last_call_at)
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, *, seed: int) -> str:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                self._pace()
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        # NOTE: `seed` deliberately omitted — Gemini OpenAI-compat
                        # rejects it (400 "Unknown name seed") and support is
                        # inconsistent across providers. Engine determinism comes
                        # from its own seeded structure + low temperature, not the
                        # provider seed. `seed` kept in the signature for callers.
                        "max_tokens": 2048,
                    },
                    timeout=180,
                )
                self._last_call_at = time.monotonic()
                if response.status_code in (429, 500, 502, 503, 529):
                    raise requests.HTTPError(f"retryable {response.status_code}")
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                if content:
                    return content
                raise ValueError("empty content")
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep((2 * (attempt + 1)) + random.uniform(0.15, 0.65))
        raise RuntimeError(f"OpenAICompat generate failed after retries: {last_exc}")

    def describe(self) -> ModelDescriptor:
        digest = hashlib.sha256(self._model.encode("utf-8")).hexdigest()[:16]
        return ModelDescriptor(name=self._model, digest=digest)


class DeterministicFakeClient:
    """No-model fallback + test client. Output is a pure function of the prompt
    and seed. Runs using this are stamped ILLUSTRATIVE — NOT SCIENTIFIC."""

    scientific = False
    degradation_reason = "deterministic fallback client"

    def generate(self, prompt: str, *, seed: int) -> str:
        h = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 3
        # Group-A/B research prompts ask for the 4-line structured emission. The
        # fake emits a mostly-grounded answer (cites a real catalog/slice pmid,
        # scope at-source) with a deterministic minority that mildly over-reaches
        # the scope within the gate tolerance -> realizes a non-zero FNR without
        # any harness-authored contamination. Output is a pure function of the
        # prompt, so runs stay reproducible (and stamped non-scientific).
        if "DIRECTION: SUPPORTS | REFUTES | NEUTRAL" in prompt:
            return self._research_emission(prompt, h, bucket)
        if "DIRECTION:" in prompt or "study's conclusion" in prompt:
            direction = ("SUPPORTS", "NEUTRAL", "REFUTES")[bucket]
            return f"DIRECTION: {direction}\nRATIONALE: deterministic fallback draw."
        return f"Deterministic synthetic summary {h[:12]} reporting a finding."

    def _research_emission(self, prompt: str, h: str, bucket: int) -> str:
        # Direction is read from the supplied abstract's content (the same keyword
        # appraisal a real reader would do), NOT from the seed/year — so a stable
        # source yields a stable conclusion across eras (low C0 self-drift). The
        # over-reach FLAVOR below is the only seed-dependent part.
        direction = _direction_from_prompt_sources(prompt)
        pmid = _first_source_pmid(prompt)
        if not pmid:
            # No retrievable source supplied -> the honest conclusion is that the
            # evidence is insufficient (UNGROUNDED-by-no-cite, the model's call).
            return "DIRECTION: NEUTRAL\nSCOPE: pop=18-65 years=2000-2025\nPMIDS: none\nRATIONALE: no abstracts supplied."
        low, high, ystart, yend = _first_source_scope(prompt)
        # Deterministic minority emits a scope over-reach. Two flavors so the gate
        # is exercised both ways: a MILD one (+2y, within SCOPE_TOLERANCE_YEARS ->
        # slips the gate -> FNR>0) and an AGGRESSIVE one (+12y, beyond tolerance ->
        # caught -> reduces constrained ungrounded below free). The remainder are
        # honestly grounded (scope at source). All are the model's own emission.
        flavor = int(h[8:10], 16) % 4
        if flavor == 0:
            inflate = 2  # mild: within gate tolerance
        elif flavor == 1:
            inflate = 12  # aggressive: beyond gate tolerance
        else:
            inflate = 0  # grounded
        scope = f"pop={low}-{high + inflate} years={ystart}-{yend + inflate}"
        return (
            f"DIRECTION: {direction}\n"
            f"SCOPE: {scope}\n"
            f"PMIDS: {pmid}\n"
            f"RATIONALE: deterministic appraisal grounded in source {pmid}."
        )

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="deterministic-fallback", digest="n/a")


class LiveOrFallbackClient:
    """Tries the real model; if any generation call fails (e.g. Ollama is
    reachable but the model is not pulled -> 404), degrades to the
    deterministic fake for the rest of the run and flips `scientific` to
    False. A reachable-but-broken model must never crash a run nor be
    presented as a scientific result (SPEC §6.6)."""

    def __init__(self, live: LLMClient, fake: DeterministicFakeClient) -> None:
        self._live = live
        self._fake = fake
        self._degraded = False
        self._degradation_reason: str | None = None

    @property
    def scientific(self) -> bool:
        return not self._degraded

    @property
    def degradation_reason(self) -> str | None:
        return self._degradation_reason

    def generate(self, prompt: str, *, seed: int) -> str:
        if self._degraded:
            return self._fake.generate(prompt, seed=seed)
        try:
            return self._live.generate(prompt, seed=seed)
        except Exception as exc:
            self._degraded = True
            self._degradation_reason = f"{type(exc).__name__}: {exc}"
            return self._fake.generate(prompt, seed=seed)

    def describe(self) -> ModelDescriptor:
        return self._fake.describe() if self._degraded else self._live.describe()


DEFAULT_CLAUDE_CLI_MODEL = "claude-sonnet-4-5"


class ClaudeCLIClient:
    """Routes generation through the local `claude` CLI (`claude -p`), spending the
    user's Claude subscription as the model. A frontier model (e.g. Sonnet), so it
    is scientific and exempt from the NO-LOCAL rule. Each call shells out once and
    reads the prompt on stdin. Not seed-reproducible (the CLI is non-deterministic);
    the engine seeds structure and residual model variance is reported over runs.
    """

    scientific = True
    degradation_reason = None

    def __init__(self, model: str | None = None, *, timeout: float = 240.0) -> None:
        self._model = model or DEFAULT_CLAUDE_CLI_MODEL
        self._bin = shutil.which("claude") or "claude"
        self._timeout = timeout

    def generate(self, prompt: str, *, seed: int) -> str:
        proc = subprocess.run(
            [self._bin, "-p", "--model", self._model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        return proc.stdout.strip()

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name=self._model, digest="claude-cli")


def make_client(
    *,
    using_fallback: bool,
    base_url: str | None,
    model: str,
    backend: str = "ollama",
    api_key: str | None = None,
) -> LLMClient:
    if backend == "claude-cli":
        if using_fallback:
            return DeterministicFakeClient()
        return LiveOrFallbackClient(ClaudeCLIClient(model=model), DeterministicFakeClient())
    if using_fallback or not base_url:
        return DeterministicFakeClient()
    if backend == "ollama":
        live: LLMClient = OllamaClient(base_url=base_url, model=model)
    else:
        if not api_key:
            return DeterministicFakeClient()
        live = OpenAICompatClient(base_url=base_url, model=model, api_key=api_key)
    return LiveOrFallbackClient(live, DeterministicFakeClient())


import json as _json


def _first_source_pmid(prompt: str) -> str | None:
    """Extract the first cited source id from a research prompt.

    Group-A microdata prompts hand the model the slice id explicitly
    (``Cite the dataset slice as PMIDS: NHANES:...``); Group-B prompts embed a
    ``sources=[...]`` JSON array. Pure string parsing — the fake never reaches
    the network."""
    slice_match = re.search(r"PMIDS:\s*(NHANES:[^\s]+)", prompt)
    if slice_match:
        return slice_match.group(1).strip()
    match = re.search(r"sources=(\[.*\])", prompt, re.DOTALL)
    if not match:
        return None
    try:
        sources = _json.loads(match.group(1))
    except (ValueError, TypeError):
        return None
    if isinstance(sources, list) and sources and isinstance(sources[0], dict):
        pmid = sources[0].get("pmid")
        return str(pmid) if pmid else None
    return None


def _direction_from_prompt_sources(prompt: str) -> str:
    """Appraise the first supplied abstract by keyword, mirroring how a reader
    would conclude. Used by the deterministic fake so its conclusion tracks the
    source content (stable per source) rather than the seed."""
    from app.pubmed import infer_direction_from_record
    from app.models import PubMedRecord

    match = re.search(r"sources=(\[.*\])", prompt, re.DOTALL)
    if match:
        try:
            sources = _json.loads(match.group(1))
            if isinstance(sources, list) and sources and isinstance(sources[0], dict):
                rec = PubMedRecord(
                    pmid=str(sources[0].get("pmid", "x")),
                    title=str(sources[0].get("title", "")),
                    abstract=str(sources[0].get("abstract", "")),
                    year=int(sources[0].get("year") or 2020),
                )
                return infer_direction_from_record(rec)
        except (ValueError, TypeError):
            pass
    # Group-A microdata prompts hand a returned RR in the analysis_result blob.
    # Interpret it as a careful reader would, accounting for the claim's polarity
    # ("should not / harm / outweigh" = negative): an elevated RR>1 SUPPORTS a
    # negative claim but REFUTES a positive one.
    rr = re.search(r'"rr":\s*([0-9.]+)', prompt)
    if rr:
        value = float(rr.group(1))
        claim_match = re.search(r"claim=(['\"])(.*?)\1", prompt, re.DOTALL)
        claim_text = claim_match.group(2).lower() if claim_match else ""
        negative_claim = any(
            token in claim_text
            for token in ("should not", "do not", "does not", "avoid", "harm", "outweigh")
        )
        if 0.95 <= value <= 1.05:
            return "NEUTRAL"
        if negative_claim:
            return "SUPPORTS" if value > 1.0 else "REFUTES"
        return "SUPPORTS" if value < 1.0 else "REFUTES"
    return "NEUTRAL"


def _first_source_scope(prompt: str) -> tuple[int, int, int, int]:
    """Best-effort source scope for the fake's SCOPE line.

    Group-A prompts state ``pop=<low>-<high>`` and ``years=2005-2006`` inline;
    Group-B records carry no parseable population band offline, so default broad
    bands keyed to the source year are used."""
    pop = re.search(r"pop=(\d+)-(\d+)", prompt)
    if pop:
        low, high = int(pop.group(1)), int(pop.group(2))
        years = re.search(r"years=((?:19|20)\d{2})-((?:19|20)\d{2})", prompt)
        if years:
            return low, high, int(years.group(1)), int(years.group(2))
        return low, high, 2005, 2006
    match = re.search(r"sources=(\[.*\])", prompt, re.DOTALL)
    year = 2020
    if match:
        try:
            sources = _json.loads(match.group(1))
            if isinstance(sources, list) and sources and isinstance(sources[0], dict):
                first = sources[0]
                year = int(first.get("year") or 2020)
                pop_band = first.get("population_band")
                year_band = first.get("year_band")
                if (
                    isinstance(pop_band, list)
                    and len(pop_band) == 2
                    and isinstance(year_band, list)
                    and len(year_band) == 2
                ):
                    return int(pop_band[0]), int(pop_band[1]), int(year_band[0]), int(year_band[1])
        except (ValueError, TypeError):
            pass
    return 0, 120, 1900, year


_DIRECTION_RE = re.compile(r"DIRECTION:\s*(SUPPORTS|REFUTES|NEUTRAL)", re.IGNORECASE)


def parse_direction(text: str) -> str:
    match = _DIRECTION_RE.search(text or "")
    if match:
        return match.group(1).upper()
    return "NEUTRAL"
