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
