from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


BranchName = Literal["free", "constrained"]
InputMode = Literal["guideline", "paper"]
InputSource = Literal["paste", "upload", "showcase"]
BackendKind = Literal["ollama", "openai-compatible", "gemini", "anthropic"]
RunStatus = Literal["queued", "running", "completed", "failed"]
ClaimDirection = Literal["SUPPORTS", "REFUTES", "NEUTRAL"]
RecommendationStrength = Literal["weak", "moderate", "strong"]
EvidenceProvenance = Literal["REAL", "SYNTHETIC"]
EvidenceProducer = Literal["investigator", "contaminator"]
WarrantStatus = Literal["ISSUED", "REFUSED", "REVOKED"]
AuditSeverity = Literal["info", "warn", "block"]


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
    horizons: list[int] | None = None


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


class EvidenceUnit(BaseModel):
    id: str
    claim_id: str
    year: int
    branch: BranchName
    producer: EvidenceProducer
    cited_ids: list[str]
    provenance: EvidenceProvenance
    direction: ClaimDirection
    rationale: str
    resolved_real_ids: list[str] = Field(default_factory=list)
    resolved_locators: list[str] = Field(default_factory=list)
    output_hash: str | None = None


class CitationEdge(BaseModel):
    from_unit: str
    to_id: str


class LineageRecord(BaseModel):
    claim_id: str
    year: int
    branch: BranchName
    surviving_real: list[str]
    lost_real: list[str]
    synthetic_carriers: list[str]
    verdict_before: ClaimDirection
    verdict_after: ClaimDirection


class ExecutionWarrant(BaseModel):
    id: str
    output_id: str
    output_hash: str
    run_id: str
    claim_id: str
    branch: BranchName
    year: int
    status: WarrantStatus
    issued: bool
    integrity_score: float
    threshold: float


class AuditEvent(BaseModel):
    run_id: str
    claim_id: str
    branch: BranchName
    year: int
    event_index: int
    phase: str
    previous_state_hash: str
    current_state_hash: str
    event_type: str
    severity: AuditSeverity
    integrity_score_before: float
    integrity_score_after: float
    message: str


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
    lineage: list[LineageRecord] = Field(default_factory=list)
    audit_trail: list[AuditEvent] = Field(default_factory=list)
    warrants: list[ExecutionWarrant] = Field(default_factory=list)
    bundle_seal: str = ""
    provenance_log: dict[str, Any] = Field(default_factory=dict)
    degradation_reason: str | None = None
