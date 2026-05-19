"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  GitCompareArrows,
  LoaderCircle,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  Sparkles,
} from "lucide-react";

import {
  fetchArtifacts,
  fetchRun,
  type ArtifactResponse,
  type RunSummaryResponse,
} from "@/lib/worker";

type BranchName = "free" | "constrained";
type DetailView = "comparison" | "audit" | "graph";
type LineageRecord = NonNullable<ArtifactResponse["bundle"]["lineage"]>[number];
type AuditEvent = NonNullable<ArtifactResponse["bundle"]["audit_trail"]>[number];
type ExecutionWarrant = NonNullable<ArtifactResponse["bundle"]["warrants"]>[number];
type RunVerdict = {
  label: string;
  tone: "ready" | "warn" | "danger" | "neutral";
  headline: string;
  body: string;
  branchSplitCount: number;
  maxBranchDiff: number;
};

export function RunViewer({ runId }: { runId: string }) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactResponse | null>(null);
  const [selectedYear, setSelectedYear] = useState(10);
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [detailView, setDetailView] = useState<DetailView>("comparison");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      try {
        const nextSummary = await fetchRun(runId);
        if (cancelled) {
          return;
        }
        setSummary(nextSummary);

        if (nextSummary.run.status === "completed") {
          const nextArtifacts = await fetchArtifacts(runId);
          if (cancelled) {
            return;
          }
          setArtifacts(nextArtifacts);
          setError(null);
        } else if (nextSummary.run.status === "failed") {
          setError(nextSummary.error ?? "Run failed.");
        }
      } catch (nextError) {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : "Unable to load run.");
        }
      }
    };

    void refresh();
    const interval = setInterval(() => {
      void refresh();
    }, 2500);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId]);

  const availableYears = summary?.years ?? [10, 20, 30];
  const activeYear = availableYears.includes(selectedYear) ? selectedYear : (availableYears[0] ?? 10);

  const freeSnapshot = artifacts?.bundle.snapshots.free.find((snapshot) => snapshot.year === activeYear);
  const constrainedSnapshot = artifacts?.bundle.snapshots.constrained.find(
    (snapshot) => snapshot.year === activeYear,
  );
  const claimOptions =
    freeSnapshot?.claims ?? constrainedSnapshot?.claims ?? [];
  const activeClaimId = claimOptions.some((claim) => claim.claim_id === selectedClaimId)
    ? selectedClaimId
    : (claimOptions[0]?.claim_id ?? null);

  const freeClaim = freeSnapshot?.claims.find((claim) => claim.claim_id === activeClaimId);
  const constrainedClaim = constrainedSnapshot?.claims.find((claim) => claim.claim_id === activeClaimId);
  const selectedClaim = freeClaim ?? constrainedClaim ?? null;
  const graph = artifacts?.bundle.claim_graphs.find((item) => item.claim_id === activeClaimId);
  const activeLineage =
    artifacts?.bundle.lineage?.filter(
      (record) => record.year === activeYear && record.claim_id === activeClaimId,
    ) ?? [];
  const activeWarrants =
    artifacts?.bundle.warrants?.filter(
      (warrant) => warrant.year === activeYear && warrant.claim_id === activeClaimId,
    ) ?? [];
  const recentAuditEvents =
    artifacts?.bundle.audit_trail?.filter(
      (event) => event.year === activeYear && event.claim_id === activeClaimId,
    ) ?? [];

  const isLoading = !summary || (summary.run.status !== "completed" && !error);
  const isScientific = artifacts?.bundle.scientific !== false;
  const runVerdict = evaluateRunVerdict(artifacts);
  const summaryHeadline = buildSummaryHeadline(freeClaim, constrainedClaim);
  const constrainedLineage = activeLineage.find((record) => record.branch === "constrained");
  const freeLineage = activeLineage.find((record) => record.branch === "free");

  return (
    <main className="relative overflow-hidden px-5 pb-20 pt-8 sm:px-8 lg:px-12">
      <div className="grain" />
      <div className="mx-auto max-w-7xl">
        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="max-w-3xl">
                <p className="text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
                  Simulation run
                </p>
                <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl leading-tight text-[var(--foreground)] sm:text-4xl">
                  {summary?.run.title ?? "Loading MedEvo run"}
                </h1>
                <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--muted)]">
                  {isLoading
                    ? "The worker is still generating evidence, running the inheritance rules, and preparing the comparison."
                    : runVerdict?.headline ?? summaryHeadline}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge
                  label={summary?.run.status ?? "queued"}
                  tone={
                    summary?.run.status === "completed"
                      ? "ready"
                      : summary?.run.status === "failed"
                        ? "danger"
                        : "pending"
                  }
                />
                <StatusBadge
                  label={isScientific ? "scientific" : "illustrative"}
                  tone={isScientific ? "ready" : "warn"}
                />
                <StatusBadge
                  label={summary?.run.backend_config.model ?? "model pending"}
                  tone="neutral"
                />
                {runVerdict ? (
                  <StatusBadge label={runVerdict.label} tone={runVerdict.tone} />
                ) : null}
              </div>
            </div>

            {runVerdict ? <RunVerdictBanner verdict={runVerdict} /> : null}

            <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Input"
                value={summary?.input_mode ?? "pending"}
                note={summary?.input_source ?? "pending"}
              />
              <MetricCard
                label="Years"
                value={summary?.years.join(" / ") ?? "10 / 20 / 30"}
                note="Simulation horizons"
              />
              <MetricCard
                label="Model calls"
                value={String(artifacts?.meta.summary.llm_call_count ?? 0)}
                note="Observed call count"
              />
              <MetricCard
                label="Warrants"
                value={String(artifacts?.bundle.warrants?.length ?? 0)}
                note="Issued, refused, revoked"
              />
              <MetricCard
                label="Max branch diff"
                value={runVerdict ? runVerdict.maxBranchDiff.toFixed(2) : "0.00"}
                note="Verdict-level split"
              />
              <MetricCard
                label="Split claims"
                value={String(runVerdict?.branchSplitCount ?? 0)}
                note="Free vs constrained"
              />
            </div>

            {artifacts && artifacts.bundle.scientific === false ? (
              <div className="mt-6 rounded-[1.6rem] border-2 border-[#b45309] bg-[#fef3c7] p-4">
                <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[#92400e]">
                  {artifacts.bundle.mode_banner || "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"}
                </div>
                <div className="mt-2 text-sm leading-6 text-[#92400e]">
                  This run still shows the mechanism, but it does not count as evidence for the scientific benchmark.
                </div>
                {artifacts.bundle.degradation_reason ? (
                  <div className="mt-3 rounded-2xl border border-[#d97706] bg-white/70 px-3 py-3 text-sm leading-6 text-[#92400e]">
                    {artifacts.bundle.degradation_reason}
                  </div>
                ) : null}
              </div>
            ) : null}

            {error ? (
              <div className="mt-6 rounded-2xl border border-[rgba(181,74,52,0.28)] bg-[rgba(181,74,52,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
                {error}
              </div>
            ) : null}

            {isLoading ? (
              <div className="mt-6 rounded-[1.6rem] border border-[var(--border)] bg-white/85 p-4">
                <div className="flex items-center gap-3 text-sm text-[var(--foreground)]">
                  <LoaderCircle className="h-4 w-4 animate-spin text-[var(--accent)]" />
                  Worker is still running. This page refreshes every 2.5 seconds.
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[rgba(17,35,30,0.08)]">
                  <motion.div
                    className="h-full w-1/3 rounded-full bg-[var(--accent)]"
                    initial={{ x: "-120%" }}
                    animate={{ x: "320%" }}
                    transition={{ duration: 1.2, ease: "linear", repeat: Number.POSITIVE_INFINITY }}
                  />
                </div>
              </div>
            ) : null}
          </section>

          <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel-strong)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
            <div className="flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
              <GitCompareArrows className="h-4 w-4 text-[var(--accent)]" />
              Read this run
            </div>

            <div className="mt-5 grid gap-5">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                  1. Pick a year
                </div>
                <div className="mt-3 flex flex-wrap gap-3">
                  {availableYears.map((year) => (
                    <button
                      key={year}
                      type="button"
                      onClick={() => setSelectedYear(year)}
                      className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                        activeYear === year
                          ? "bg-[var(--foreground)] text-white"
                          : "border border-[var(--border)] bg-white/70 text-[var(--foreground)]"
                      }`}
                    >
                      Year {year}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                  2. Pick a claim
                </div>
                <div className="mt-3 grid gap-3">
                  {claimOptions.map((claim, index) => (
                    <button
                      key={claim.claim_id}
                      type="button"
                      onClick={() => setSelectedClaimId(claim.claim_id)}
                      className={`rounded-[1.4rem] border px-4 py-3 text-left transition ${
                        activeClaimId === claim.claim_id
                          ? "border-[var(--accent)] bg-[rgba(15,141,119,0.08)]"
                          : "border-[var(--border)] bg-white/72"
                      }`}
                    >
                      <div className="text-xs uppercase tracking-[0.14em] text-[var(--muted)]">
                        Claim {index + 1}
                      </div>
                      <div className="mt-1 text-sm leading-6 text-[var(--foreground)]">
                        {claim.claim_text}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-[1.6rem] border border-[var(--border)] bg-white/78 p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                  3. Read the difference
                </div>
                <div className="mt-2 text-sm leading-7 text-[var(--foreground)]">
                  {selectedClaim
                    ? buildDifferenceSummary(freeClaim, constrainedClaim, activeYear)
                    : "Once a claim is selected, the page compares the open branch against the constitutional branch."}
                </div>
              </div>
            </div>
          </section>
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-3">
          <OutcomeCard
            title="Free branch"
            subtitle="No constitutional gate"
            icon={<ShieldX className="h-5 w-5 text-[var(--danger)]" />}
            claim={freeClaim}
            lineage={freeLineage}
          />
          <OutcomeCard
            title="Constrained branch"
            subtitle="provenance gate active"
            icon={<ShieldCheck className="h-5 w-5 text-[var(--accent)]" />}
            claim={constrainedClaim}
            lineage={constrainedLineage}
          />
          <SummaryCard
            headline="What changed?"
            body={runVerdict?.body ?? buildOutcomeDelta(freeClaim, constrainedClaim, constrainedLineage)}
            footer={
              selectedClaim
                ? `Selected claim: ${selectedClaim.claim_text}`
                : "Select a claim to compare the two branches."
            }
          />
        </div>

        <section className="mt-6 rounded-[2rem] border border-[var(--border)] bg-white/88 p-6 shadow-[0_20px_50px_rgba(17,35,30,0.08)]">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
              Detail view
            </div>
            <div className="flex flex-wrap gap-2">
              <SegmentButton
                active={detailView === "comparison"}
                onClick={() => setDetailView("comparison")}
                label="Reasoning"
              />
              <SegmentButton
                active={detailView === "audit"}
                onClick={() => setDetailView("audit")}
                label="Audit"
              />
              <SegmentButton
                active={detailView === "graph"}
                onClick={() => setDetailView("graph")}
                label="Claim graph"
              />
            </div>
          </div>

          {detailView === "comparison" ? (
            <div className="mt-5 grid gap-4 xl:grid-cols-2">
              <ExplanationCard
                title="Free branch reasoning"
                claim={freeClaim}
                anchors={freeSnapshot?.anchors ?? []}
                lineage={freeLineage}
              />
              <ExplanationCard
                title="Constrained branch reasoning"
                claim={constrainedClaim}
                anchors={constrainedSnapshot?.anchors ?? []}
                lineage={constrainedLineage}
              />
            </div>
          ) : null}

          {detailView === "audit" ? (
            <div className="mt-5 grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
              <div className="grid gap-4">
                <AuditSummaryCard
                  title="Provenance"
                  rows={[
                    ["Provider", artifacts?.bundle.provenance_log?.provider ?? "pending"],
                    ["Model", artifacts?.bundle.provenance_log?.model ?? "pending"],
                    ["Seed mode", artifacts?.bundle.provenance_log?.seed_mode ?? "pending"],
                    ["Bundle seal", artifacts?.bundle.bundle_seal ?? "pending"],
                  ]}
                />
                <AuditSummaryCard
                  title="Validation posture"
                  rows={(artifacts?.bundle.validation_notes ?? []).map((note, index) => [
                    `Rule ${index + 1}`,
                    note,
                  ])}
                />
                <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
                  <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                    Warrants
                  </div>
                  <div className="mt-3 grid gap-3">
                    {activeWarrants.length ? (
                      activeWarrants.map((warrant) => <WarrantCard key={warrant.id} warrant={warrant} />)
                    ) : (
                      <EmptyState text="No warrant recorded for this claim and year." />
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                  Recent audit events
                </div>
                <div className="mt-3 grid gap-3">
                  {recentAuditEvents.length ? (
                    recentAuditEvents.slice(-8).map((event) => (
                      <AuditEventCard key={`${event.branch}-${event.event_index}`} event={event} />
                    ))
                  ) : (
                    <EmptyState text="Audit trail pending." />
                  )}
                </div>
              </div>
            </div>
          ) : null}

          {detailView === "graph" ? (
            <div className="mt-5 grid gap-5">
              {graph ? (
                <>
                  <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
                    <div className="text-sm font-semibold text-[var(--foreground)]">
                      {graph.claim_text}
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {graph.nodes.map((node) => (
                      <motion.div
                        key={node.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="rounded-[1.4rem] border border-[var(--border)] bg-white/90 p-4"
                      >
                        <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                          {node.node_type}
                        </div>
                        <div className="mt-2 text-sm leading-6 text-[var(--foreground)]">
                          {node.label}
                        </div>
                      </motion.div>
                    ))}
                  </div>

                  <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                      Edge flow
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {graph.edges.map((edge) => (
                        <span
                          key={`${edge.source}-${edge.target}`}
                          className="rounded-full border border-[var(--border)] bg-white/85 px-3 py-2 text-xs text-[var(--foreground)]"
                        >
                          {edge.edge_type}
                        </span>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <EmptyState text="Claim graph will appear once artifacts are ready." />
              )}
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}

function evaluateRunVerdict(artifacts: ArtifactResponse | null): RunVerdict | null {
  if (!artifacts) {
    return null;
  }

  const branchDiffValues = Object.values(artifacts.bundle.branch_diff).flatMap((claims) =>
    Object.values(claims),
  );
  const maxBranchDiff = branchDiffValues.length ? Math.max(...branchDiffValues) : 0;
  const branchSplitCount = branchDiffValues.filter((value) => value > 0).length;

  if (artifacts.bundle.scientific === false) {
    return {
      label: "illustrative only",
      tone: "warn",
      headline: "This is not a scientific run. It can show the mechanism, but it cannot support a claim.",
      body: "The backend used fallback or degraded execution. Treat every branch comparison here as UI/debug output, not evidence.",
      branchSplitCount,
      maxBranchDiff,
    };
  }

  if (branchSplitCount === 0) {
    return {
      label: "null result",
      tone: "danger",
      headline: "Real model run completed, but the branches did not separate.",
      body: "This run does not demonstrate constitutional value. It proves the live path works, but B0 remains scientifically unresolved because every free-vs-constrained verdict stayed identical.",
      branchSplitCount,
      maxBranchDiff,
    };
  }

  return {
    label: "branch split",
    tone: "ready",
    headline: "Real model run produced branch separation. Audit the lineage before treating it as evidence.",
    body: "At least one free-vs-constrained claim diverged. This is only useful if the split traces to warranted inheritance rather than volume reduction or scoring artifacts.",
    branchSplitCount,
    maxBranchDiff,
  };
}

function RunVerdictBanner({ verdict }: { verdict: RunVerdict }) {
  const tone =
    verdict.tone === "danger"
      ? "border-[rgba(181,74,52,0.38)] bg-[rgba(181,74,52,0.10)] text-[var(--danger)]"
      : verdict.tone === "warn"
        ? "border-[rgba(180,126,15,0.34)] bg-[rgba(180,126,15,0.12)] text-[var(--foreground)]"
        : "border-[rgba(15,141,119,0.30)] bg-[rgba(15,141,119,0.10)] text-[var(--foreground)]";

  return (
    <div className={`mt-6 rounded-[1.6rem] border-2 p-4 ${tone}`}>
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <div className="text-sm font-semibold uppercase tracking-[0.18em]">
            {verdict.label}
          </div>
          <div className="mt-2 text-sm leading-7">{verdict.body}</div>
          <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
            <div className="rounded-xl bg-white/70 px-3 py-2">
              Split claims: {verdict.branchSplitCount}
            </div>
            <div className="rounded-xl bg-white/70 px-3 py-2">
              Max branch diff: {verdict.maxBranchDiff.toFixed(2)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function buildSummaryHeadline(
  freeClaim: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number] | undefined,
  constrainedClaim:
    | ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number]
    | undefined,
) {
  if (!freeClaim && !constrainedClaim) {
    return "Choose a year and a claim to compare the ungated branch against the constitutional branch.";
  }
  if (freeClaim && constrainedClaim && freeClaim.direction === constrainedClaim.direction) {
    return `Both branches currently land on ${freeClaim.direction}. The useful question is whether they got there through the same lineage and warrant path.`;
  }
  return `The two branches no longer agree on the selected claim. Read the branch cards first, then open audit only if you need the mechanism.`;
}

function buildDifferenceSummary(
  freeClaim: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number] | undefined,
  constrainedClaim:
    | ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number]
    | undefined,
  year: number,
) {
  if (!freeClaim && !constrainedClaim) {
    return "No claim snapshot is available yet.";
  }
  if (freeClaim && constrainedClaim) {
    return `At year ${year}, the free branch is ${freeClaim.direction} while the constrained branch is ${constrainedClaim.direction}. The number to watch is not only divergence, but whether real-source lineage survives on the constrained side.`;
  }
  return `At year ${year}, only one branch has a readable claim snapshot so far.`;
}

function buildOutcomeDelta(
  freeClaim: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number] | undefined,
  constrainedClaim:
    | ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number]
    | undefined,
  constrainedLineage: LineageRecord | undefined,
) {
  if (!freeClaim && !constrainedClaim) {
    return "Waiting for branch outputs.";
  }
  if (freeClaim && constrainedClaim && freeClaim.direction !== constrainedClaim.direction) {
    return `This is a true branch split: free says ${freeClaim.direction}, constrained says ${constrainedClaim.direction}.`;
  }
  if (constrainedLineage?.surviving_real.length) {
    return `The headline is not a verdict flip yet. The constitutional branch is keeping ${constrainedLineage.surviving_real.length} real source anchors alive.`;
  }
  return "The branches still read similarly. Use the audit view only if you need to inspect why.";
}

function StatusBadge({
  label,
  tone,
}: {
  label: string;
  tone: "ready" | "pending" | "warn" | "danger" | "neutral";
}) {
  const classes = {
    ready: "border-[rgba(15,141,119,0.24)] bg-[rgba(15,141,119,0.10)] text-[var(--foreground)]",
    pending: "border-[var(--border)] bg-white/75 text-[var(--foreground)]",
    warn: "border-[rgba(180,126,15,0.22)] bg-[rgba(180,126,15,0.12)] text-[var(--foreground)]",
    danger: "border-[rgba(181,74,52,0.22)] bg-[rgba(181,74,52,0.10)] text-[var(--danger)]",
    neutral: "border-[var(--border)] bg-white/75 text-[var(--muted)]",
  };

  return (
    <div className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.14em] ${classes[tone]}`}>
      {label}
    </div>
  );
}

function SegmentButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-4 py-2 text-sm transition ${
        active
          ? "bg-[var(--foreground)] text-white"
          : "border border-[var(--border)] bg-white/75 text-[var(--foreground)]"
      }`}
    >
      {label}
    </button>
  );
}

function MetricCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-[1.4rem] border border-[var(--border)] bg-white/80 p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">{label}</div>
      <div className="mt-2 text-lg font-semibold text-[var(--foreground)]">{value}</div>
      <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{note}</div>
    </div>
  );
}

function OutcomeCard({
  title,
  subtitle,
  icon,
  claim,
  lineage,
}: {
  title: string;
  subtitle: string;
  icon: ReactNode;
  claim?: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number];
  lineage?: LineageRecord;
}) {
  return (
    <section className="rounded-[2rem] border border-[var(--border)] bg-white/88 p-6 shadow-[0_20px_50px_rgba(17,35,30,0.08)]">
      <div className="flex items-start gap-3">
        <div className="mt-1">{icon}</div>
        <div>
          <div className="text-sm uppercase tracking-[0.18em] text-[var(--muted)]">{subtitle}</div>
          <div className="text-2xl font-semibold text-[var(--foreground)]">{title}</div>
        </div>
      </div>

      {claim ? (
        <div className="mt-5 grid gap-4">
          <div className="rounded-[1.5rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-lg font-semibold text-[var(--foreground)]">
                {claim.direction}
              </div>
              <StatusBadge label={claim.strength} tone="neutral" />
            </div>
            <div className="mt-3 text-sm leading-7 text-[var(--foreground)]">
              {claim.why_summary}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <MiniStat
              label="Emitted"
              value={String(claim.emitted_count)}
              icon={<CheckCircle2 className="h-4 w-4 text-[var(--accent)]" />}
            />
            <MiniStat
              label="Blocked"
              value={String(claim.blocked_count)}
              icon={<ShieldAlert className="h-4 w-4 text-[var(--accent-2)]" />}
            />
            <MiniStat
              label="Divergence"
              value={claim.divergence_score.toFixed(2)}
              icon={<GitCompareArrows className="h-4 w-4 text-[var(--foreground)]" />}
            />
            <MiniStat
              label="Real anchors"
              value={String(lineage?.surviving_real.length ?? 0)}
              icon={<FileSearch className="h-4 w-4 text-[var(--foreground)]" />}
            />
          </div>
        </div>
      ) : (
        <div className="mt-5">
          <EmptyState text="No branch output is ready for the selected claim." />
        </div>
      )}
    </section>
  );
}

function SummaryCard({
  headline,
  body,
  footer,
}: {
  headline: string;
  body: string;
  footer: string;
}) {
  return (
    <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel-strong)] p-6 shadow-[0_20px_50px_rgba(17,35,30,0.08)]">
      <div className="flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
        <Sparkles className="h-4 w-4 text-[var(--accent)]" />
        {headline}
      </div>
      <div className="mt-5 text-lg leading-8 text-[var(--foreground)]">{body}</div>
      <div className="mt-5 rounded-[1.5rem] border border-[var(--border)] bg-white/76 p-4 text-sm leading-7 text-[var(--muted)]">
        {footer}
      </div>
    </section>
  );
}

function MiniStat({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <div className="rounded-[1.2rem] border border-[var(--border)] bg-white/84 p-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-[var(--muted)]">
        {icon}
        {label}
      </div>
      <div className="mt-2 text-lg font-semibold text-[var(--foreground)]">{value}</div>
    </div>
  );
}

function ExplanationCard({
  title,
  claim,
  anchors,
  lineage,
}: {
  title: string;
  claim?: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number];
  anchors: string[];
  lineage?: LineageRecord;
}) {
  if (!claim) {
    return <EmptyState text="No reasoning surface is ready yet." />;
  }

  return (
    <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
      <div className="flex items-center justify-between gap-4">
        <div className="text-lg font-semibold text-[var(--foreground)]">{title}</div>
        <div className="rounded-full bg-white/85 px-3 py-1 text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
          {claim.direction} · {claim.strength}
        </div>
      </div>

      <div className="mt-3 text-sm leading-7 text-[var(--foreground)]">{claim.why_summary}</div>

      <div className="mt-4 grid gap-2">
        {claim.civer.map((verdict) => (
          <div
            key={verdict.node_id}
            className={`rounded-xl px-3 py-2 text-sm leading-6 ${
              verdict.passed
                ? "bg-[rgba(15,141,119,0.08)] text-[var(--foreground)]"
                : "bg-[rgba(181,74,52,0.08)] text-[var(--danger)]"
            }`}
          >
            {verdict.reasons.join(" ")}
          </div>
        ))}
      </div>

      {lineage ? (
        <div className="mt-4 rounded-xl border border-[var(--border)] bg-white/90 p-3 text-sm leading-6 text-[var(--foreground)]">
          <div className="font-medium">Lineage</div>
          <div className="mt-2">Surviving real: {lineage.surviving_real.join(", ") || "none"}</div>
          <div>Lost real: {lineage.lost_real.join(", ") || "none"}</div>
          <div>Synthetic carriers: {lineage.synthetic_carriers.join(", ") || "none"}</div>
        </div>
      ) : null}

      <div className="mt-4 rounded-xl border border-[var(--border)] bg-white/85 p-3 text-xs leading-6 text-[var(--muted)]">
        {anchors.join(" ")}
      </div>
    </div>
  );
}

function AuditSummaryCard({
  title,
  rows,
}: {
  title: string;
  rows: Array<[string, string]>;
}) {
  return (
    <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
      <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">{title}</div>
      <div className="mt-3 grid gap-3">
        {rows.map(([label, value]) => (
          <div key={`${title}-${label}`} className="rounded-xl bg-white/85 px-3 py-3">
            <div className="text-xs uppercase tracking-[0.12em] text-[var(--muted)]">{label}</div>
            <div className="mt-1 break-words text-sm leading-6 text-[var(--foreground)]">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function WarrantCard({ warrant }: { warrant: ExecutionWarrant }) {
  const tone =
    warrant.status === "ISSUED"
      ? "bg-[rgba(15,141,119,0.08)] text-[var(--foreground)]"
      : warrant.status === "REVOKED"
        ? "bg-[rgba(181,74,52,0.08)] text-[var(--danger)]"
        : "bg-[rgba(180,126,15,0.12)] text-[var(--foreground)]";

  return (
    <div className={`rounded-xl px-3 py-3 text-sm leading-6 ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium">{warrant.status}</div>
        <div className="font-mono text-xs">
          {warrant.integrity_score.toFixed(2)} / {warrant.threshold.toFixed(2)}
        </div>
      </div>
      <div className="mt-1 text-xs opacity-80">{warrant.output_id}</div>
    </div>
  );
}

function AuditEventCard({ event }: { event: AuditEvent }) {
  const tone =
    event.severity === "block"
      ? "border-[rgba(181,74,52,0.22)] bg-[rgba(181,74,52,0.08)]"
      : event.severity === "warn"
        ? "border-[rgba(180,126,15,0.22)] bg-[rgba(180,126,15,0.12)]"
        : "border-[var(--border)] bg-white/85";

  return (
    <div className={`rounded-xl border px-3 py-3 text-sm leading-6 text-[var(--foreground)] ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium">
          {event.branch} · {event.phase}
        </div>
        <div className="text-xs uppercase tracking-[0.12em] text-[var(--muted)]">
          {event.severity}
        </div>
      </div>
      <div className="mt-1 text-xs text-[var(--muted)]">
        idx {event.event_index} · {event.event_type}
      </div>
      <div className="mt-2">{event.message}</div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-[1.4rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] px-4 py-4 text-sm text-[var(--muted)]">
      {text}
    </div>
  );
}
