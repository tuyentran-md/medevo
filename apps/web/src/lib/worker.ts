import type {
  BackendKind,
  DriftSnapshot,
  InputMode,
  RunRequest,
  RunStatus,
} from "@medevo/contracts";

export interface ShowcaseItem {
  id: string;
  run_id: string;
  title: string;
  description: string;
  input_mode: InputMode;
  tags: string[];
  status: RunStatus;
}

export interface RunSummaryResponse {
  run: {
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
  };
  years: number[];
  input_mode: InputMode;
  input_source: RunRequest["input_source"];
  error?: string | null;
  showcase: boolean;
}

export interface ArtifactResponse {
  run_id: string;
  bundle: {
    input_text: string;
    claim_graphs: Array<{
      claim_id: string;
      claim_text: string;
      nodes: Array<{
        id: string;
        label: string;
        node_type: string;
        timestamp: number;
      }>;
      edges: Array<{
        source: string;
        target: string;
        edge_type: string;
      }>;
    }>;
    snapshots: Record<"free" | "constrained", DriftSnapshot[]>;
    branch_diff: Record<string, Record<string, number>>;
    anchors: string[];
    validation_notes: string[];
    scientific?: boolean;
    mode_banner?: string;
    model_descriptor?: Record<string, string>;
    provenance_log?: {
      model?: string;
      model_digest?: string;
      provider?: string;
      base_url?: string;
      temperature?: number;
      seed_mode?: string;
      prompt_template_digests?: Record<string, string>;
      calls?: Array<{
        label: string;
        seed: number;
        prompt_digest: string;
        response_hash: string;
        timestamp: string;
      }>;
    };
    lineage?: Array<{
      claim_id: string;
      year: number;
      branch: "free" | "constrained";
      surviving_real: string[];
      lost_real: string[];
      synthetic_carriers: string[];
      verdict_before: "SUPPORTS" | "REFUTES" | "NEUTRAL";
      verdict_after: "SUPPORTS" | "REFUTES" | "NEUTRAL";
    }>;
    audit_trail?: Array<{
      run_id: string;
      claim_id: string;
      branch: "free" | "constrained";
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
    }>;
    warrants?: Array<{
      id: string;
      output_id: string;
      output_hash: string;
      run_id: string;
      claim_id: string;
      branch: "free" | "constrained";
      year: number;
      status: "ISSUED" | "REFUSED" | "REVOKED";
      issued: boolean;
      integrity_score: number;
      threshold: number;
    }>;
    bundle_seal?: string;
    degradation_reason?: string | null;
  };
  meta: {
    summary: {
      claim_count: number;
      years: number[];
      has_blocked_outputs: boolean;
      llm_call_count?: number;
      degradation_reason?: string | null;
    };
    validation: string[];
    description?: string;
    tags?: string[];
  };
}

export const workerUrl =
  process.env.NEXT_PUBLIC_MEDEVO_WORKER_URL ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const fallback = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      throw new Error(payload.detail ?? fallback);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(fallback);
    }
  }
  return (await response.json()) as T;
}

export async function fetchShowcase(): Promise<ShowcaseItem[]> {
  const response = await fetch(`${workerUrl}/showcase`, { cache: "no-store" });
  return parseJson<ShowcaseItem[]>(response);
}

export async function createRun(formData: FormData): Promise<{ id: string }> {
  const response = await fetch(`${workerUrl}/runs`, {
    method: "POST",
    body: formData,
  });
  return parseJson<{ id: string }>(response);
}

export async function fetchRun(runId: string): Promise<RunSummaryResponse> {
  const response = await fetch(`${workerUrl}/runs/${runId}`, { cache: "no-store" });
  return parseJson<RunSummaryResponse>(response);
}

export async function fetchArtifacts(runId: string): Promise<ArtifactResponse> {
  const response = await fetch(`${workerUrl}/runs/${runId}/artifacts`, {
    cache: "no-store",
  });
  return parseJson<ArtifactResponse>(response);
}
