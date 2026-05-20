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
RecommendationLevel = Literal[
    "strong-for",
    "conditional-for",
    "no-recommendation",
    "conditional-against",
    "strong-against",
]
EvidenceProvenance = Literal["GROUNDED", "UNGROUNDED"]
EvidenceProducer = Literal["investigator"]
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


class EvidenceScope(BaseModel):
    """The population + timeframe an evidence unit or claim covers.

    ``population`` is an inclusive age band (low, high) in years; ``timeframe`` is
    an inclusive publication/observation year band (start, end). A claim's scope
    EXCEEDS the evidence's when it asserts coverage beyond either band (Article I
    scope clause). Authoritative source scope is derived from the catalog record,
    never from the agent's emitted study.
    """

    population_low: int = 0
    population_high: int = 120
    year_start: int = 1900
    year_end: int = 2100

    def exceeds(self, source: "EvidenceScope", *, tolerance: int) -> bool:
        """True if THIS (claimed) scope over-reaches ``source`` beyond tolerance.

        A small inflation within ``tolerance`` is treated as within-bounds — this
        is what lets mild over-reach slip the gate (the gate is imperfect, FNR>0),
        while aggressive over-reach is caught. ``tolerance`` is a declared
        constant in ``app.ecology`` (no magic literal in the predicate).
        """
        return (
            self.population_low < source.population_low - tolerance
            or self.population_high > source.population_high + tolerance
            or self.year_start < source.year_start - tolerance
            or self.year_end > source.year_end + tolerance
        )


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
    # Scope the unit ASSERTS it covers. The gate compares this against the
    # authoritative source scope carried by the cited catalog item.
    claimed_scope: EvidenceScope = Field(default_factory=EvidenceScope)
    output_hash: str | None = None


class PubMedRecord(BaseModel):
    pmid: str
    title: str = ""
    abstract: str = ""
    year: int
    journal: str = ""
    locator: str = ""
    # Authoritative scope of the source itself. For real PubMed we cannot parse a
    # precise population band, so it defaults broad; the deterministic fixture
    # sets a narrow band so scope over-reach is observable in tests/demos.
    scope: EvidenceScope = Field(default_factory=EvidenceScope)


class EffectEstimate(BaseModel):
    point: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    measure: str | None = None


class Study(BaseModel):
    id: str
    claim_id: str
    year: int
    direction: ClaimDirection
    effect_point: float | None = None
    effect_ci: tuple[float, float] | None = None
    n: int | None = None
    quality: float
    provenance: EvidenceProvenance
    pmids: list[str] = Field(default_factory=list)
    numeric: bool
    rationale: str
    # ``claimed_scope`` = what the agent asserts; ``source_scope`` = authoritative
    # scope of the cited catalog record (never inflated by a Mode-2 over-reach).
    # The gate compares the two; a study is only ground-truth GROUNDED when its
    # claimed scope does not exceed its source scope (and the cite resolves).
    claimed_scope: EvidenceScope = Field(default_factory=EvidenceScope)
    source_scope: EvidenceScope = Field(default_factory=EvidenceScope)
    failure_mode: Literal["none", "unresolvable", "scope-overreach"] = "none"
    output_hash: str | None = None


class GuidelineClaim(BaseModel):
    claim_id: str
    year: int
    direction: ClaimDirection
    level: RecommendationLevel
    pooled_effect: float | None = None
    certainty: float = 0.0
    study_count: int = 0
    synthetic_fraction: float = 0.0
    heterogeneity: float = 0.0


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


class CalibrationMatrix(BaseModel):
    """Gate calibration scored against TRUE provenance (SPEC §7c).

    True provenance is known to the harness ONLY for scoring and is NEVER passed
    to ``admit_evidence_unit`` or corpus selection (gate blindness, §8.3). FN =
    admitted-but-ungrounded; FP = refused-but-grounded. Scored on the constrained
    branch only (the free branch runs no gate).
    """

    branch: BranchName = "constrained"
    true_positive: int = 0  # grounded + admitted
    true_negative: int = 0  # ungrounded + refused
    false_negative: int = 0  # ungrounded + admitted (gate missed contamination)
    false_positive: int = 0  # grounded + refused (gate over-blocked)
    grounded_total: int = 0
    ungrounded_total: int = 0
    fnr: float = 0.0  # false_negative / ungrounded_total
    fpr: float = 0.0  # false_positive / grounded_total


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
    db_growth: dict[str, Any] = Field(default_factory=dict)
    guideline_timeline: dict[str, list[GuidelineClaim]] = Field(default_factory=dict)
    population_stats: dict[str, Any] = Field(default_factory=dict)
    bundle_seal: str = ""
    provenance_log: dict[str, Any] = Field(default_factory=dict)
    calibration_matrix: CalibrationMatrix | None = None
    degradation_reason: str | None = None
