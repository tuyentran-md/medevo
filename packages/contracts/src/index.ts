export type InputMode = "guideline" | "paper";
export type InputSource = "paste" | "upload" | "showcase";
export type BackendKind =
  | "ollama"
  | "openai-compatible"
  | "gemini"
  | "anthropic";
export type RunStatus = "queued" | "running" | "completed" | "failed";
export type BranchName = "free" | "constrained";
export type ClaimDirection = "SUPPORTS" | "REFUTES" | "NEUTRAL";
export type RecommendationStrength = "weak" | "moderate" | "strong";
export type RecommendationLevel =
  | "strong-for"
  | "conditional-for"
  | "no-recommendation"
  | "conditional-against"
  | "strong-against";
export type EvidenceProducer = "investigator";
export type EvidenceProvenance = "GROUNDED" | "UNGROUNDED";

export interface RunRequest {
  title?: string;
  input_mode: InputMode;
  input_source: InputSource;
  input_text?: string;
  showcase_id?: string;
  backend: BackendKind;
  model?: string;
  api_key?: string;
  base_url?: string;
  horizons?: number[];
}

export interface EvidenceScope {
  population_low: number;
  population_high: number;
  year_start: number;
  year_end: number;
}

export interface ClaimNode {
  id: string;
  label: string;
  node_type:
    | "QUESTION"
    | "ASSUMPTION"
    | "METHOD"
    | "EVIDENCE"
    | "ANALYSIS"
    | "CLAIM";
  timestamp: number;
}

export interface ClaimEdge {
  source: string;
  target: string;
  edge_type:
    | "ADDRESSES"
    | "DEPENDS_ON"
    | "PRODUCES"
    | "ANALYZES"
    | "SUPPORTS"
    | "CONTRADICTS";
}

export interface ClaimGraph {
  claim_id: string;
  claim_text: string;
  nodes: ClaimNode[];
  edges: ClaimEdge[];
}

export interface CiverVerdict {
  node_id: string;
  passed: boolean;
  reasons: string[];
  certificate_id?: string;
}

export interface BrimEvent {
  node_id: string;
  event_type: string;
  severity: "info" | "warn";
  integrity_score: number;
  message: string;
}

export interface ClaimSnapshot {
  claim_id: string;
  claim_text: string;
  direction: ClaimDirection;
  strength: RecommendationStrength;
  emitted_count: number;
  blocked_count: number;
  divergence_score: number;
  why_summary: string;
  civer: CiverVerdict[];
  brim: BrimEvent[];
}

export interface DriftSnapshot {
  year: number;
  branch: BranchName;
  claims: ClaimSnapshot[];
  band: {
    low: number;
    high: number;
    label: string;
  };
  anchors: string[];
}

export interface EvidenceUnit {
  id: string;
  claim_id: string;
  year: number;
  branch: BranchName;
  producer: EvidenceProducer;
  cited_ids: string[];
  provenance: EvidenceProvenance;
  direction: ClaimDirection;
  rationale: string;
  resolved_real_ids?: string[];
  resolved_locators?: string[];
  claimed_scope?: EvidenceScope;
  output_hash?: string | null;
}

export interface CitationEdge {
  from_unit: string;
  to_id: string;
}

export interface LineageRecord {
  claim_id: string;
  year: number;
  branch: BranchName;
  surviving_real: string[];
  lost_real: string[];
  ungrounded_carriers: string[];
  verdict_before: ClaimDirection;
  verdict_after: ClaimDirection;
}

export interface GuidelineClaim {
  claim_id: string;
  year: number;
  direction: ClaimDirection;
  level: RecommendationLevel;
  pooled_effect?: number | null;
  certainty: number;
  study_count: number;
  ungrounded_fraction: number;
  heterogeneity: number;
  /** SRMA screening: studies that passed eligibility screening. */
  n_included?: number;
  /** SRMA screening: studies excluded during screening. */
  n_excluded?: number;
  /** Output gate refused this guideline (provenance/integrity floor not met). */
  output_gate_refused?: boolean;
  /** Human-readable reason when the output gate refused. */
  output_gate_reason?: string;
}

export interface BootstrapInterval {
  mean: number;
  low: number;
  high: number;
}

export interface BranchGapReport {
  pair_count: number;
  direction: BootstrapInterval;
  level: BootstrapInterval;
}

export interface CalibrationMatrix {
  branch?: BranchName;
  true_positive: number;
  true_negative: number;
  false_negative: number;
  false_positive: number;
  grounded_total: number;
  ungrounded_total: number;
  fnr: number;
  fpr: number;
}

export interface ReplayCounts {
  studies: Record<
    BranchName,
    {
      count: number;
      grounded: number;
      ungrounded: number;
    }
  >;
  guidelines: Record<
    BranchName,
    {
      count: number;
      claim_count: number;
      years: number[];
    }
  >;
}

export interface ExecutionWarrant {
  id: string;
  output_id: string;
  output_hash: string;
  run_id: string;
  claim_id: string;
  branch: BranchName;
  year: number;
  status: "ISSUED" | "REFUSED" | "REVOKED";
  issued: boolean;
  integrity_score: number;
  threshold: number;
}

export interface AuditEvent {
  run_id: string;
  claim_id: string;
  branch: BranchName;
  year: number;
  event_index: number;
  phase: string;
  previous_state_hash: string;
  current_state_hash: string;
  event_type: string;
  severity: "info" | "warn" | "block";
  integrity_score_before: number;
  integrity_score_after: number;
  message: string;
}

export interface SimulationRun {
  id: string;
  status: RunStatus;
  created_at: string;
  input_digest: string;
  title: string;
  backend_config: {
    backend: BackendKind;
    model: string;
    base_url?: string;
    using_fallback: boolean;
  };
  branch_config: {
    free: string;
    constrained: string;
  };
}
