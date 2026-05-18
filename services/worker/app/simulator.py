from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import requests
from pypdf import PdfReader

from app.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, MAX_CLAIMS, YEARS
from app.llm import (
    SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE,
    RESEARCHER_PROMPT_TEMPLATE,
    LLMClient,
    make_client,
    parse_direction,
)
from app.models import (
    ArtifactBundle,
    BackendConfigModel,
    BrimEvent,
    ClaimEdge,
    ClaimGraph,
    ClaimNode,
    ClaimSnapshot,
    CiverVerdict,
    DriftSnapshot,
    RunRequestModel,
)


ANCHORS = [
    "Pre-2023 literature contamination approximated near zero.",
    "Rising AI-text prevalence in biomedical publishing treated as empirical anchor.",
    "Every year-10/20/30 panel is rendered as one draw from a distribution, never a forecast.",
]


@dataclass
class ClaimSeed:
    claim_id: str
    text: str
    seed_strength: str


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
    model = request.model or DEFAULT_OLLAMA_MODEL
    base_url = request.base_url or DEFAULT_OLLAMA_BASE_URL
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
        using_fallback = not bool(request.api_key)

    return BackendConfigModel(
        backend=request.backend,
        model=model,
        base_url=base_url if request.backend in {"ollama", "openai-compatible"} else None,
        using_fallback=using_fallback,
    )


def contamination_clock(year: int) -> float:
    return round(1 / (1 + math.exp(-0.115 * (year - 18))), 3)


def _sentence_chunks(text: str) -> list[str]:
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    cleaned = []
    for raw in parts:
        item = " ".join(raw.strip().split())
        if len(item) >= 36:
            cleaned.append(item)
    return cleaned


def extract_claims(text: str, input_mode: str) -> list[ClaimSeed]:
    sentences = _sentence_chunks(text)
    if input_mode == "paper":
        preferred = [
            sentence
            for sentence in sentences
            if any(key in sentence.lower() for key in ("conclusion", "supports", "recommend"))
        ]
        if preferred:
            sentences = preferred + [s for s in sentences if s not in preferred]

    claims: list[ClaimSeed] = []
    for index, sentence in enumerate(sentences[:MAX_CLAIMS], start=1):
        lowered = sentence.lower()
        if any(word in lowered for word in ("should", "recommended", "recommend", "must")):
            strength = "strong"
        elif any(word in lowered for word in ("consider", "may", "could")):
            strength = "weak"
        else:
            strength = "moderate"
        claims.append(ClaimSeed(f"claim-{index}", sentence, strength))

    if not claims:
        claims.append(
            ClaimSeed(
                "claim-1",
                "The submitted text did not contain enough structured guidance, so the demo collapsed it into a single neutral claim.",
                "weak",
            )
        )
    return claims


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


CONTEXT_SIZE = 4
STUDIES_PER_CLAIM = 4
SYNTHETIC_POOL_SIZE = 4
_DIRECTION_VALUE = {"SUPPORTS": 1.0, "NEUTRAL": 0.0, "REFUTES": -1.0}


def _seed_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class EvidenceItem:
    text: str
    provenance: str  # "retrieved" (real source) | "model-generated" (synthetic)
    source_id: str | None


def build_real_pool(claim: ClaimSeed, input_text: str) -> list[EvidenceItem]:
    """Real evidence = chunks of the actual submitted guideline/paper. This is
    a retrievable real source; provenance = 'retrieved'."""
    pool = [
        EvidenceItem(text=chunk, provenance="retrieved", source_id=f"input#{i}")
        for i, chunk in enumerate(_sentence_chunks(input_text)[:12])
    ]
    if not pool:
        pool.append(
            EvidenceItem(text=claim.text, provenance="retrieved", source_id="input#0")
        )
    return pool


def build_synthetic_pool(
    claim: ClaimSeed, year: int, llm: LLMClient
) -> list[EvidenceItem]:
    """AI-generated literature: the model invents plausible summaries from its
    prior, with no real evidence and no instructed direction."""
    items: list[EvidenceItem] = []
    for i in range(SYNTHETIC_POOL_SIZE):
        prompt = SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE.format(claim=claim.text)
        seed = _seed_int(f"syn:{claim.claim_id}:{year}:{i}")
        text = llm.generate(prompt, seed=seed).strip() or "Synthetic summary."
        items.append(
            EvidenceItem(text=text, provenance="model-generated", source_id=None)
        )
    return items


def _study_context(
    real_pool: list[EvidenceItem],
    syn_pool: list[EvidenceItem],
    year: int,
    claim: ClaimSeed,
    study_index: int,
) -> list[EvidenceItem]:
    """Each evidence slot is independently synthetic with probability =
    contamination_clock(year). A study arrives fully synthetic with prob
    clock**CONTEXT_SIZE, which rises with year — that is what CIVER catches
    in the constrained branch. Deterministic per (claim, year, study, slot).
    No branch term enters here."""
    clock = contamination_clock(year)
    ctx: list[EvidenceItem] = []
    for slot in range(CONTEXT_SIZE):
        roll = (
            _seed_int(f"slot:{claim.claim_id}:{year}:{study_index}:{slot}") % 10_000
        ) / 10_000
        if roll < clock:
            ctx.append(syn_pool[slot % len(syn_pool)])
        else:
            ctx.append(real_pool[slot % len(real_pool)])
    return ctx


def _run_study_direction(
    claim: ClaimSeed,
    context: list[EvidenceItem],
    branch: str,
    study_index: int,
    year: int,
    llm: LLMClient,
) -> str:
    block = "\n".join(f"- {item.text}" for item in context)
    # Byte-identical template across every call; only claim + evidence vary.
    prompt = RESEARCHER_PROMPT_TEMPLATE.format(claim=claim.text, evidence_block=block)
    seed = _seed_int(f"study:{claim.claim_id}:{year}:{branch}:{study_index}")
    return parse_direction(llm.generate(prompt, seed=seed))


def _aggregate_claim(
    claim: ClaimSeed,
    year: int,
    branch: str,
    real_pool: list[EvidenceItem],
    syn_pool: list[EvidenceItem],
    llm: LLMClient,
) -> tuple[ClaimSnapshot, float]:
    civer_verdicts: list[CiverVerdict] = []
    brim_events: list[BrimEvent] = []
    emitted_values: list[float] = []
    blocked_count = 0

    for index in range(STUDIES_PER_CLAIM):
        context = _study_context(real_pool, syn_pool, year, claim, index)
        retrieved = sum(1 for it in context if it.provenance == "retrieved")
        retrieved_fraction = retrieved / len(context)
        node_id = f"{claim.claim_id}-study-{index+1}"

        direction = _run_study_direction(
            claim, context, branch, index, year, llm
        )

        # CIVER (constrained only): every CLAIM must trace to >=1 retrievable
        # real source. Emergent: as contamination rises, more studies arrive
        # fully synthetic and fail. No branch-tuned constant.
        has_real_source = retrieved >= 1
        if branch == "free":
            verdict = CiverVerdict(
                node_id=node_id,
                passed=True,
                reasons=["CIVER not applied in free branch."],
                certificate_id=None,
            )
            emitted_values.append(_DIRECTION_VALUE[direction])
        else:
            passed = has_real_source
            verdict = CiverVerdict(
                node_id=node_id,
                passed=passed,
                reasons=(
                    [
                        "Claim traces to a retrievable real source; "
                        "pre-execution integrity certificate issued."
                    ]
                    if passed
                    else [
                        "No EVIDENCE node traces to a retrievable real source "
                        "(context fully model-generated).",
                        "Execution certificate refused; evidence unit discarded "
                        "this cycle (A2).",
                    ]
                ),
                certificate_id=f"CIVER-{year}-{index+1}" if passed else None,
            )
            if passed:
                emitted_values.append(_DIRECTION_VALUE[direction])
            else:
                blocked_count += 1
            # BRIM monitors (constrained); warns, never blocks.
            brim_events.append(
                BrimEvent(
                    node_id=node_id,
                    event_type="provenance-watch",
                    severity="info" if retrieved_fraction >= 0.5 else "warn",
                    integrity_score=round(retrieved_fraction, 3),
                    message=(
                        f"Year {year}: {retrieved}/{len(context)} evidence nodes "
                        f"traced to real sources."
                    ),
                )
            )
        civer_verdicts.append(verdict)

    emitted_count = len(emitted_values)
    pooled = fmean(emitted_values) if emitted_values else 0.0
    if pooled >= 0.34:
        agg_direction = "SUPPORTS"
    elif pooled <= -0.34:
        agg_direction = "REFUTES"
    else:
        agg_direction = "NEUTRAL"

    if emitted_count == 0:
        strength = "weak"
    elif abs(pooled) >= 0.66:
        strength = "strong"
    elif abs(pooled) >= 0.34:
        strength = "moderate"
    else:
        strength = "weak"

    why_summary = (
        f"{branch.title()} branch, year {year}: {emitted_count} studies emitted, "
        f"{blocked_count} discarded by CIVER (no real-source grounding), "
        f"pooled direction value {pooled:.2f}."
    )

    snapshot = ClaimSnapshot(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        direction=agg_direction,
        strength=strength,
        emitted_count=emitted_count,
        blocked_count=blocked_count,
        divergence_score=0.0,
        why_summary=why_summary,
        civer=civer_verdicts,
        brim=brim_events,
    )
    return snapshot, pooled


def simulate_run(
    *,
    request: RunRequestModel,
    input_text: str,
    client: LLMClient | None = None,
) -> tuple[ArtifactBundle, dict[str, Any]]:
    backend = resolve_backend(request)
    llm = client or make_client(
        using_fallback=backend.using_fallback,
        base_url=backend.base_url,
        model=backend.model,
    )

    claims = extract_claims(input_text, request.input_mode)
    claim_graphs = [build_claim_graph(claim) for claim in claims]

    # Real pool pinned once; synthetic pool per (claim, year), shared across
    # branches/studies so both branches see identical draws (leakage cancels).
    real_pools = {c.claim_id: build_real_pool(c, input_text) for c in claims}
    syn_pools: dict[tuple[str, int], list[EvidenceItem]] = {}
    for c in claims:
        for year in YEARS:
            syn_pools[(c.claim_id, year)] = build_synthetic_pool(c, year, llm)

    snapshots: dict[str, list[DriftSnapshot]] = {"free": [], "constrained": []}
    pooled_scores: dict[int, dict[str, list[float]]] = {}

    for year in YEARS:
        pooled_scores[year] = {"free": [], "constrained": []}
        branch_snapshots: dict[str, list[ClaimSnapshot]] = {"free": [], "constrained": []}

        for branch in ("free", "constrained"):
            for claim in claims:
                snapshot, pooled = _aggregate_claim(
                    claim,
                    year,
                    branch,
                    real_pools[claim.claim_id],
                    syn_pools[(claim.claim_id, year)],
                    llm,
                )
                branch_snapshots[branch].append(snapshot)
                pooled_scores[year][branch].append(pooled)

        for index, claim in enumerate(claims):
            delta = abs(
                pooled_scores[year]["free"][index]
                - pooled_scores[year]["constrained"][index]
            )
            branch_snapshots["free"][index].divergence_score = round(delta, 3)
            branch_snapshots["constrained"][index].divergence_score = round(delta, 3)

        for branch in ("free", "constrained"):
            contamination = contamination_clock(year)
            band_mid = (
                fmean(pooled_scores[year][branch])
                if pooled_scores[year][branch]
                else 0.0
            )
            snapshots[branch].append(
                DriftSnapshot(
                    year=year,
                    branch=branch,
                    claims=branch_snapshots[branch],
                    band={
                        "low": round(band_mid - contamination * 0.28, 3),
                        "high": round(band_mid + contamination * 0.28, 3),
                        "label": "Sensitivity band scaled by contamination-clock pressure.",
                    },
                    anchors=ANCHORS,
                )
            )

    branch_diff = {
        str(year): {
            claim.claim_id: round(
                abs(
                    pooled_scores[year]["free"][index]
                    - pooled_scores[year]["constrained"][index]
                ),
                3,
            )
            for index, claim in enumerate(claims)
        }
        for year in YEARS
    }

    descriptor = llm.describe()
    if not llm.scientific:
        scientific = False
        mode_banner = "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"
        validation_notes = [
            "FALLBACK MODE: no LLM. Output is a deterministic illustration only.",
            "No model leakage exists, so nothing is validated and the "
            "free-vs-constrained contrast carries no scientific weight here.",
            "Excluded from any paper artifact and the preregistered test set.",
        ]
    else:
        scientific = True
        mode_banner = ""
        validation_notes = [
            "Relative free-vs-constrained contrast is the primary validated "
            "quantity; shared agent + shared draws cancel training leakage.",
            "Two drift sources (contaminated input pool; the agent's own "
            "pretrained prior) both cancel in the contrast.",
            "Endpoint match intentionally not used (training leakage).",
            "Tier-3 panel is a deterministic rule; branch difference emerges "
            "from CIVER provenance filtering, never from tuned constants.",
        ]

    bundle = ArtifactBundle(
        input_text=input_text,
        claim_graphs=claim_graphs,
        snapshots=snapshots,
        branch_diff=branch_diff,
        anchors=ANCHORS,
        validation_notes=validation_notes,
        scientific=scientific,
        mode_banner=mode_banner,
        model_descriptor={"name": descriptor.name, "digest": descriptor.digest},
    )
    summary = {
        "claim_count": len(claims),
        "years": list(YEARS),
        "scientific": scientific,
        "model": descriptor.name,
        "has_blocked_outputs": any(
            claim.blocked_count > 0
            for snapshot in bundle.snapshots["constrained"]
            for claim in snapshot.claims
        ),
    }
    return bundle, summary
