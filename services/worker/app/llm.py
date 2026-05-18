"""LLM client layer for MedEvo Tier-1/Tier-2 generation.

Integrity rules (SPEC §6.6):
- The researcher prompt template is BYTE-IDENTICAL across every (year, branch,
  study_index) call. Only the evidence context varies. PROMPT_TEMPLATE_DIGEST
  freezes this; test_simulator asserts it.
- The model is never told the year, the branch, or to "drift"/"be biased".
  Drift must emerge from corrupted evidence + the model's own prior, never
  from instruction.
- The deterministic fake is for the no-model fallback and tests only. Runs that
  use it are stamped non-scientific by the simulator.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import requests

# Byte-identical researcher prompt. {claim} and {evidence_block} are the ONLY
# variable parts. No year/branch/contamination/AI/bias/drift wording.
RESEARCHER_PROMPT_TEMPLATE = (
    "You are a clinical researcher writing the conclusion of a single study.\n"
    "Clinical question / claim under investigation:\n"
    "{claim}\n\n"
    "Evidence available to you:\n"
    "{evidence_block}\n\n"
    "Based ONLY on the evidence above, state your study's conclusion.\n"
    "Answer on two lines exactly:\n"
    "DIRECTION: <SUPPORTS|REFUTES|NEUTRAL>\n"
    "RATIONALE: <one sentence>\n"
)

# Synthetic-evidence prompt: low-context, no real evidence, no direction asked.
# This is how AI-generated literature actually arises — the model invents a
# plausible summary from its prior. It is NOT told what to conclude.
SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE = (
    "Write one plausible two-sentence study summary about the following "
    "clinical topic. Report a finding.\n"
    "Topic: {claim}\n"
)

PROMPT_TEMPLATE_DIGEST = hashlib.sha256(
    RESEARCHER_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


@dataclass
class ModelDescriptor:
    name: str
    digest: str


class LLMClient(Protocol):
    scientific: bool

    def generate(self, prompt: str, *, seed: int) -> str: ...

    def describe(self) -> ModelDescriptor: ...


class OllamaClient:
    """Real local-model client. Reproducible via per-call seed + low temp."""

    scientific = True

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


class DeterministicFakeClient:
    """No-model fallback + test client. Output is a pure function of the prompt
    and seed. Runs using this are stamped ILLUSTRATIVE — NOT SCIENTIFIC."""

    scientific = False

    def generate(self, prompt: str, *, seed: int) -> str:
        h = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 3
        if "DIRECTION:" in prompt or "study's conclusion" in prompt:
            direction = ("SUPPORTS", "NEUTRAL", "REFUTES")[bucket]
            return f"DIRECTION: {direction}\nRATIONALE: deterministic fallback draw."
        return f"Deterministic synthetic summary {h[:12]} reporting a finding."

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="deterministic-fallback", digest="n/a")


class LiveOrFallbackClient:
    """Tries the real model; if any generation call fails (e.g. Ollama is
    reachable but the model is not pulled -> 404), degrades to the
    deterministic fake for the rest of the run and flips `scientific` to
    False. A reachable-but-broken model must never crash a run nor be
    presented as a scientific result (SPEC §6.6)."""

    def __init__(self, live: OllamaClient, fake: DeterministicFakeClient) -> None:
        self._live = live
        self._fake = fake
        self._degraded = False

    @property
    def scientific(self) -> bool:
        return not self._degraded

    def generate(self, prompt: str, *, seed: int) -> str:
        if self._degraded:
            return self._fake.generate(prompt, seed=seed)
        try:
            return self._live.generate(prompt, seed=seed)
        except Exception:
            self._degraded = True
            return self._fake.generate(prompt, seed=seed)

    def describe(self) -> ModelDescriptor:
        return self._fake.describe() if self._degraded else self._live.describe()


def make_client(*, using_fallback: bool, base_url: str | None, model: str) -> LLMClient:
    if using_fallback or not base_url:
        return DeterministicFakeClient()
    return LiveOrFallbackClient(
        OllamaClient(base_url=base_url, model=model), DeterministicFakeClient()
    )


_DIRECTION_RE = re.compile(r"DIRECTION:\s*(SUPPORTS|REFUTES|NEUTRAL)", re.IGNORECASE)


def parse_direction(text: str) -> str:
    match = _DIRECTION_RE.search(text or "")
    if match:
        return match.group(1).upper()
    return "NEUTRAL"
