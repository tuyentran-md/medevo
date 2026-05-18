from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


BranchName = Literal["free", "constrained"]
InputMode = Literal["guideline", "paper"]
InputSource = Literal["paste", "upload", "showcase"]
BackendKind = Literal["ollama", "openai-compatible", "gemini", "anthropic"]
RunStatus = Literal["queued", "running", "completed", "failed"]
ClaimDirection = Literal["SUPPORTS", "REFUTES", "NEUTRAL"]
RecommendationStrength = Literal["weak", "moderate", "strong"]


class RunRequestModel(BaseModel):
    title: str | None = None
    input_mode: InputMode
    input_source: InputSource
    input_text: str | None = None
    showcase_id: str | None = None
    backend: BackendKind
    model: str | None = None
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = None


class BackendConfigModel(BaseModel):
    backend: BackendKind
    model: str
    base_url: str | None = None
    using_fallback: bool


class ClaimNode(BaseModel):
    id: str
    label: str
    node_type: Literal[
        "QUESTION", "ASSUMPTION", "METHOD", "EVIDENCE", "ANALYSIS", "CLAIM"
    ]
    timestamp: int


class ClaimEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal[
        "ADDRESSES", "DEPENDS_ON", "PRODUCES", "ANALYZES", "SUPPORTS", "CONTRADICTS"
    ]


class ClaimGraph(BaseModel):
    claim_id: str
    claim_text: str
    nodes: list[ClaimNode]
    edges: list[ClaimEdge]


class CiverVerdict(BaseModel):
    node_id: str
    passed: bool
    reasons: list[str]
    certificate_id: str | None = None


class BrimEvent(BaseModel):
    node_id: str
    event_type: str
    severity: Literal["info", "warn"]
    integrity_score: float
    message: str


class ClaimSnapshot(BaseModel):
    claim_id: str
    claim_text: str
    direction: ClaimDirection
    strength: RecommendationStrength
    emitted_count: int
    blocked_count: int
    divergence_score: float
    why_summary: str
    civer: list[CiverVerdict]
    brim: list[BrimEvent]


class BandModel(BaseModel):
    low: float
    high: float
    label: str


class DriftSnapshot(BaseModel):
    year: int
    branch: BranchName
    claims: list[ClaimSnapshot]
    band: BandModel
    anchors: list[str]


class SimulationRunModel(BaseModel):
    id: str
    status: RunStatus
    created_at: datetime
    input_digest: str
    title: str
    backend_config: BackendConfigModel
    branch_config: dict[str, str]


class RunSummary(BaseModel):
    run: SimulationRunModel
    years: list[int]
    input_mode: InputMode
    input_source: InputSource
    error: str | None = None
    showcase: bool = False


class ShowcaseRecord(BaseModel):
    id: str
    title: str
    description: str
    input_mode: InputMode
    input_text: str
    tags: list[str]


class ArtifactBundle(BaseModel):
    input_text: str
    claim_graphs: list[ClaimGraph]
    snapshots: dict[str, list[DriftSnapshot]]
    branch_diff: dict[str, dict[str, float]]
    anchors: list[str]
    validation_notes: list[str]
    scientific: bool = True
    mode_banner: str = ""
    model_descriptor: dict[str, str] = Field(default_factory=dict)
