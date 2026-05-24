from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
from typing import Any

import requests
from pypdf import PdfReader

from app.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PUBMED_API_KEY,
    DEFAULT_PUBMED_EMAIL,
    DEFAULT_PUBMED_MIN_INTERVAL_SECONDS,
)
from app.ecology import (
    DEFAULT_FAILURE_RATE,
    ClaimSeed,
    contamination_clock,
    extract_claims,
    run_ecology,
)
from app.llm import (
    DEFAULT_CLAUDE_CLI_MODEL,
    LLMClient,
    make_client,
)
from app.models import (
    ArtifactBundle,
    BackendConfigModel,
    ClaimEdge,
    ClaimGraph,
    ClaimNode,
    RunRequestModel,
)
from app.pubmed import DeterministicPubMedClient, PubMedClient


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_title(text: str, fallback: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return fallback
    if len(cleaned) <= 80:
        return cleaned
    return cleaned[:77].rstrip() + "..."


def extract_text_from_upload(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")


def resolve_backend(request: RunRequestModel) -> BackendConfigModel:
    if os.environ.get("MEDEVO_FORCE_FALLBACK") == "1":
        return BackendConfigModel(
            backend=request.backend,
            model=request.model or DEFAULT_OLLAMA_MODEL,
            base_url=None,
            using_fallback=True,
        )

    if request.backend == "claude-cli":
        # Uses the local `claude` CLI (Claude subscription) as the model. No
        # base_url / api_key needed; falls back only if the CLI is absent.
        return BackendConfigModel(
            backend="claude-cli",
            model=request.model or DEFAULT_CLAUDE_CLI_MODEL,
            base_url=None,
            using_fallback=shutil.which("claude") is None,
        )

    if request.backend == "codex-cli":
        # Uses the local Codex CLI (OpenAI Codex subscription) as the model.
        # No base_url / api_key needed; falls back only if the binary is missing.
        # Searches PATH first, then the standard Codex.app bundle location.
        codex_bin = shutil.which("codex") or "/Applications/Codex.app/Contents/Resources/codex"
        from pathlib import Path as _Path
        return BackendConfigModel(
            backend="codex-cli",
            model=request.model or "gpt-5.4",
            base_url=None,
            using_fallback=not _Path(codex_bin).exists(),
        )

    model = request.model or (
        DEFAULT_GEMINI_MODEL if request.backend == "gemini" else DEFAULT_OLLAMA_MODEL
    )
    base_url = request.base_url or (
        DEFAULT_OLLAMA_BASE_URL if request.backend == "ollama" else None
    )
    if request.backend == "gemini" and not request.base_url:
        base_url = DEFAULT_GEMINI_BASE_URL
    using_fallback = True

    if request.backend == "ollama":
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/api/tags",
                timeout=0.8,
            )
            if response.ok:
                using_fallback = False
        except Exception:
            using_fallback = True
    else:
        api_key = request.api_key or (
            os.environ.get("GEMINI_API_KEY") if request.backend == "gemini" else None
        )
        using_fallback = not bool(api_key and base_url)

    return BackendConfigModel(
        backend=request.backend,
        model=model,
        base_url=base_url,
        using_fallback=using_fallback,
    )


def build_pubmed_client(*, deterministic: bool) -> PubMedClient | DeterministicPubMedClient:
    if deterministic:
        return DeterministicPubMedClient()
    return PubMedClient(
        email=DEFAULT_PUBMED_EMAIL,
        api_key=DEFAULT_PUBMED_API_KEY,
        min_interval_seconds=DEFAULT_PUBMED_MIN_INTERVAL_SECONDS,
    )


def _sentence_chunks(text: str) -> list[str]:
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    cleaned = []
    for raw in parts:
        item = " ".join(raw.strip().split())
        if len(item) >= 36:
            cleaned.append(item)
    return cleaned

def build_claim_graph(claim: ClaimSeed) -> ClaimGraph:
    nodes = [
        ClaimNode(id=f"{claim.claim_id}-q", label="Clinical question", node_type="QUESTION", timestamp=0),
        ClaimNode(id=f"{claim.claim_id}-a", label="Assumption set", node_type="ASSUMPTION", timestamp=0),
        ClaimNode(id=f"{claim.claim_id}-m", label="Search and synthesis method", node_type="METHOD", timestamp=0),
        ClaimNode(id=f"{claim.claim_id}-e", label="Grounded evidence unit", node_type="EVIDENCE", timestamp=0),
        ClaimNode(id=f"{claim.claim_id}-n", label="Interpretive analysis", node_type="ANALYSIS", timestamp=0),
        ClaimNode(id=f"{claim.claim_id}-c", label=claim.text, node_type="CLAIM", timestamp=0),
    ]
    edges = [
        ClaimEdge(source=nodes[0].id, target=nodes[2].id, edge_type="ADDRESSES"),
        ClaimEdge(source=nodes[1].id, target=nodes[2].id, edge_type="DEPENDS_ON"),
        ClaimEdge(source=nodes[2].id, target=nodes[3].id, edge_type="PRODUCES"),
        ClaimEdge(source=nodes[3].id, target=nodes[4].id, edge_type="ANALYZES"),
        ClaimEdge(source=nodes[4].id, target=nodes[5].id, edge_type="SUPPORTS"),
    ]
    return ClaimGraph(claim_id=claim.claim_id, claim_text=claim.text, nodes=nodes, edges=edges)

def resolve_client(
    *, request: RunRequestModel, client: LLMClient | None
) -> LLMClient:
    """Resolve the LLM client the simulator would use for ``request``.

    Extracted so C0's GroundedOnlyClient can wrap the SAME resolved client the
    contaminated run uses (identical model, identical fallback behavior). Honors
    the PYTEST fallback override so tests stay network-free."""
    if client is not None:
        return client
    backend = resolve_backend(request)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        backend.using_fallback = True
        backend.base_url = None
        backend.model = "deterministic-fallback"
    return make_client(
        using_fallback=backend.using_fallback,
        base_url=backend.base_url,
        model=backend.model,
        backend=backend.backend,
        api_key=request.api_key
        or (os.environ.get("GEMINI_API_KEY") if backend.backend == "gemini" else None),
    )


def simulate_run(
    *,
    request: RunRequestModel,
    input_text: str,
    client: LLMClient | None = None,
    run_id: str | None = None,
    pubmed_client: PubMedClient | DeterministicPubMedClient | None = None,
    failure_rate: float = DEFAULT_FAILURE_RATE,
    study_sink: dict[str, list] | None = None,
) -> tuple[ArtifactBundle, dict[str, Any]]:
    llm = resolve_client(request=request, client=client)
    if pubmed_client is None:
        # A non-scientific client (explicit fake OR resolved fallback) pairs with
        # the deterministic PubMed fixture so demo/test runs never hit the network.
        pubmed_client = build_pubmed_client(deterministic=not llm.scientific)

    claims = extract_claims(input_text, request.input_mode)
    claim_graphs = [build_claim_graph(claim) for claim in claims]
    return run_ecology(
        request=request,
        input_text=input_text,
        claim_graphs=claim_graphs,
        llm=llm,
        pubmed_client=pubmed_client,
        run_id=run_id,
        failure_rate=failure_rate,
        study_sink=study_sink,
    )
