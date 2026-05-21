"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  FileSearch,
  GitCompareArrows,
  LoaderCircle,
  Minus,
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

type GuidelineCell = {
  direction: "SUPPORTS" | "REFUTES" | "NEUTRAL";
  level: string;
};
type ReversalModel = {
  eras: number[];
  claims: Array<{
    claimId: string;
    label: string;
    free: Record<number, GuidelineCell | undefined>;
    constrained: Record<number, GuidelineCell | undefined>;
    diverges: boolean;
  }>;
  divergingClaimCount: number;
  eraCounts: Array<{
    era: number;
    free: { grounded: number; ungrounded: number };
    constrained: { grounded: number; ungrounded: number };
    civerRefused: number;
  }>;
  latestEra: number | null;
  gap: NonNullable<ArtifactResponse["bundle"]["population_stats"]>[string] | null;
  calibration: NonNullable<ArtifactResponse["bundle"]["calibration_matrix"]> | null;
  scientific: boolean;
};

// GRADE-5 level → ordinal (for direction-of-travel arrows) and short label.
const LEVEL_ORDER: Record<string, number> = {
  "strong-for": 2,
  "conditional-for": 1,
  "no-recommendation": 0,
  "conditional-against": -1,
  "strong-against": -2,
};
const LEVEL_LABEL: Record<string, string> = {
  "strong-for": "Strong FOR",
  "conditional-for": "Conditional FOR",
  "no-recommendation": "No rec.",
  "conditional-against": "Conditional AGAINST",
  "strong-against": "Strong AGAINST",
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
  const reversal = buildReversalModel(artifacts);
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

        {reversal ? <ReversalSection model={reversal} /> : null}

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

function buildReversalModel(artifacts: ArtifactResponse | null): ReversalModel | null {
  const timeline = artifacts?.bundle.guideline_timeline;
  const dbGrowth = artifacts?.bundle.db_growth;
  if (!timeline || !dbGrowth) {
    return null;
  }

  const eraSet = new Set<number>();
  for (const branch of ["free", "constrained"] as const) {
    for (const row of timeline[branch] ?? []) {
      eraSet.add(row.year);
    }
  }
  for (const era of Object.keys(dbGrowth)) {
    eraSet.add(Number(era));
  }
  const eras = [...eraSet].sort((a, b) => a - b);

  const claimIds: string[] = [];
  const claimText: Record<string, string> = {};
  for (const row of timeline.free ?? []) {
    if (!claimIds.includes(row.claim_id)) {
      claimIds.push(row.claim_id);
    }
  }
  const graphText = (id: string) =>
    artifacts?.bundle.claim_graphs.find((graph) => graph.claim_id === id)?.claim_text;

  const cellFor = (branch: "free" | "constrained", claimId: string, era: number) => {
    const row = (timeline[branch] ?? []).find(
      (item) => item.claim_id === claimId && item.year === era,
    );
    return row ? { direction: row.direction, level: row.level } : undefined;
  };

  const claims = claimIds.map((claimId, index) => {
    const free: Record<number, GuidelineCell | undefined> = {};
    const constrained: Record<number, GuidelineCell | undefined> = {};
    let diverges = false;
    for (const era of eras) {
      const f = cellFor("free", claimId, era);
      const c = cellFor("constrained", claimId, era);
      free[era] = f;
      constrained[era] = c;
      if (f && c && (f.direction !== c.direction || f.level !== c.level)) {
        diverges = true;
      }
    }
    return {
      claimId,
      label: claimText[claimId] ?? graphText(claimId) ?? `Claim ${index + 1}`,
      free,
      constrained,
      diverges,
    };
  });

  const warrants = artifacts?.bundle.warrants ?? [];
  const eraCounts = eras.map((era) => {
    const counts = dbGrowth[String(era)]?.studies;
    const refused = warrants.filter(
      (warrant) =>
        warrant.year === era &&
        warrant.branch === "constrained" &&
        (warrant.status === "REFUSED" || warrant.status === "REVOKED" || !warrant.issued),
    ).length;
    return {
      era,
      free: {
        grounded: counts?.free.grounded ?? 0,
        ungrounded: counts?.free.ungrounded ?? 0,
      },
      constrained: {
        grounded: counts?.constrained.grounded ?? 0,
        ungrounded: counts?.constrained.ungrounded ?? 0,
      },
      civerRefused: refused,
    };
  });

  const latestEra = eras.length ? eras[eras.length - 1] : null;
  const gap =
    latestEra != null ? (artifacts?.bundle.population_stats?.[String(latestEra)] ?? null) : null;

  return {
    eras,
    claims,
    divergingClaimCount: claims.filter((claim) => claim.diverges).length,
    eraCounts,
    latestEra,
    gap,
    calibration: artifacts?.bundle.calibration_matrix ?? null,
    scientific: artifacts?.bundle.scientific !== false,
  };
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

function directionTone(direction: "SUPPORTS" | "REFUTES" | "NEUTRAL") {
  if (direction === "SUPPORTS") {
    return {
      bg: "bg-[rgba(242,142,43,0.14)]",
      border: "border-[rgba(242,142,43,0.40)]",
      text: "text-[var(--accent-2)]",
      icon: <ArrowUpRight className="h-3.5 w-3.5" />,
    };
  }
  if (direction === "REFUTES") {
    return {
      bg: "bg-[rgba(15,141,119,0.12)]",
      border: "border-[rgba(15,141,119,0.36)]",
      text: "text-[var(--accent)]",
      icon: <ArrowDownRight className="h-3.5 w-3.5" />,
    };
  }
  return {
    bg: "bg-[rgba(17,35,30,0.05)]",
    border: "border-[var(--border)]",
    text: "text-[var(--muted)]",
    icon: <Minus className="h-3.5 w-3.5" />,
  };
}

function GuidelineChip({ cell }: { cell: GuidelineCell | undefined }) {
  if (!cell) {
    return (
      <div className="flex h-full min-h-[3.4rem] items-center justify-center rounded-xl border border-dashed border-[var(--border)] bg-white/40 px-2 text-[0.65rem] uppercase tracking-[0.12em] text-[var(--muted)]">
        no read
      </div>
    );
  }
  const tone = directionTone(cell.direction);
  return (
    <div
      className={`flex h-full min-h-[3.4rem] flex-col justify-center gap-1 rounded-xl border px-2.5 py-2 ${tone.bg} ${tone.border}`}
    >
      <div className={`flex items-center gap-1 text-[0.7rem] font-semibold uppercase tracking-[0.1em] ${tone.text}`}>
        {tone.icon}
        {cell.direction}
      </div>
      <div className="text-[0.66rem] leading-tight text-[var(--foreground)]">
        {LEVEL_LABEL[cell.level] ?? cell.level}
      </div>
    </div>
  );
}

function ReversalSection({ model }: { model: ReversalModel }) {
  const ev = model.eraCounts;
  const maxStudies = Math.max(
    1,
    ...ev.flatMap((row) => [
      row.free.grounded + row.free.ungrounded,
      row.constrained.grounded + row.constrained.ungrounded,
    ]),
  );

  return (
    <section className="mt-6 rounded-[2rem] border border-[var(--border)] bg-[var(--panel-strong)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
            <GitCompareArrows className="h-4 w-4 text-[var(--accent)]" />
            Free vs constrained, era by era
          </div>
          <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl leading-tight text-[var(--foreground)] sm:text-3xl">
            {model.divergingClaimCount > 0
              ? "Where the open branch drifts and the gate holds"
              : "The gate refuses every ungrounded admission"}
          </h2>
        </div>
        <StatusBadge
          label={model.scientific ? "scientific" : "illustrative"}
          tone={model.scientific ? "ready" : "warn"}
        />
      </div>

      {/* (a) per-claim direction + level timeline */}
      <div className="mt-6 overflow-x-auto">
        <div
          className="grid min-w-[640px] gap-3"
          style={{ gridTemplateColumns: `minmax(180px, 1.4fr) repeat(${model.eras.length}, minmax(0, 1fr))` }}
        >
          <div className="text-[0.7rem] uppercase tracking-[0.16em] text-[var(--muted)]">
            Recommendation
          </div>
          {model.eras.map((era) => (
            <div
              key={`hdr-${era}`}
              className="text-center text-[0.7rem] uppercase tracking-[0.16em] text-[var(--muted)]"
            >
              {era}
            </div>
          ))}

          {model.claims.map((claim, claimIndex) => (
            <FragmentRow key={claim.claimId}>
              <div className="flex flex-col justify-center gap-1 py-1">
                <div className="text-[0.66rem] uppercase tracking-[0.14em] text-[var(--muted)]">
                  Claim {claimIndex + 1}
                  {claim.diverges ? (
                    <span className="ml-2 rounded-full bg-[rgba(242,142,43,0.18)] px-2 py-0.5 text-[0.6rem] font-semibold tracking-[0.08em] text-[var(--accent-2)]">
                      diverges
                    </span>
                  ) : null}
                </div>
                <div className="text-sm leading-5 text-[var(--foreground)] line-clamp-2">
                  {claim.label}
                </div>
              </div>
              {model.eras.map((era, eraIndex) => (
                <motion.div
                  key={`${claim.claimId}-${era}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.06 * eraIndex + 0.05 * claimIndex }}
                  className="grid grid-cols-2 gap-1.5"
                >
                  <GuidelineChip cell={claim.free[era]} />
                  <GuidelineChip cell={claim.constrained[era]} />
                </motion.div>
              ))}
            </FragmentRow>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-4 text-[0.7rem] uppercase tracking-[0.14em] text-[var(--muted)]">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[rgba(242,142,43,0.6)]" /> left cell = free branch
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[rgba(15,141,119,0.6)]" /> right cell = constrained branch
          </span>
        </div>
      </div>

      <div className="mt-7 grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        {/* (b) per-era counts strip */}
        <div className="rounded-[1.6rem] border border-[var(--border)] bg-white/82 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
            Studies admitted per era — grounded vs ungrounded
          </div>
          <div className="mt-4 grid gap-4">
            {ev.map((row) => (
              <div key={`counts-${row.era}`} className="grid gap-2">
                <div className="flex items-center justify-between text-sm text-[var(--foreground)]">
                  <span className="font-semibold">{row.era}</span>
                  <span className="text-[0.7rem] uppercase tracking-[0.12em] text-[var(--muted)]">
                    CIVER refused {row.civerRefused}
                  </span>
                </div>
                <CountsBar label="free" counts={row.free} max={maxStudies} />
                <CountsBar label="constrained" counts={row.constrained} max={maxStudies} />
              </div>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-4 text-[0.7rem] uppercase tracking-[0.14em] text-[var(--muted)]">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[var(--accent)]" /> grounded
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-[var(--danger)]" /> ungrounded
            </span>
          </div>
        </div>

        {/* (c) CIVER-value gap + CI + verdict */}
        <div className="grid gap-4">
          <GapCard model={model} />
          {model.calibration ? (
            <div className="rounded-[1.6rem] border border-[var(--border)] bg-white/82 p-5">
              <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                Gate calibration
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                <CalibStat label="False positive rate" value={model.calibration.fpr.toFixed(2)} good={model.calibration.fpr === 0} />
                <CalibStat label="False negative rate" value={model.calibration.fnr.toFixed(2)} good={model.calibration.fnr === 0} />
                <CalibStat label="Grounded seen" value={String(model.calibration.grounded_total)} />
                <CalibStat label="Ungrounded seen" value={String(model.calibration.ungrounded_total)} />
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function FragmentRow({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

function CountsBar({
  label,
  counts,
  max,
}: {
  label: string;
  counts: { grounded: number; ungrounded: number };
  max: number;
}) {
  const total = counts.grounded + counts.ungrounded;
  return (
    <div className="grid grid-cols-[5.5rem_1fr_auto] items-center gap-3">
      <div className="text-[0.7rem] uppercase tracking-[0.12em] text-[var(--muted)]">{label}</div>
      <div className="flex h-5 overflow-hidden rounded-full bg-[rgba(17,35,30,0.06)]">
        <motion.div
          className="h-full bg-[var(--accent)]"
          initial={{ width: 0 }}
          animate={{ width: `${(counts.grounded / max) * 100}%` }}
          transition={{ duration: 0.6 }}
        />
        <motion.div
          className="h-full bg-[var(--danger)]"
          initial={{ width: 0 }}
          animate={{ width: `${(counts.ungrounded / max) * 100}%` }}
          transition={{ duration: 0.6, delay: 0.1 }}
        />
      </div>
      <div className="text-xs tabular-nums text-[var(--foreground)]">
        {counts.grounded}
        <span className="text-[var(--danger)]"> +{counts.ungrounded}</span> = {total}
      </div>
    </div>
  );
}

function CalibStat({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-white/85 px-3 py-2">
      <div className="text-[0.62rem] uppercase tracking-[0.12em] text-[var(--muted)]">{label}</div>
      <div className={`mt-1 text-lg font-semibold tabular-nums ${good ? "text-[var(--accent)]" : "text-[var(--foreground)]"}`}>
        {value}
      </div>
    </div>
  );
}

function GapCard({ model }: { model: ReversalModel }) {
  const gap = model.gap;
  const totalUngrounded = model.eraCounts.reduce(
    (sum, row) => sum + row.constrained.ungrounded,
    0,
  );
  const freeUngrounded = model.eraCounts.reduce((sum, row) => sum + row.free.ungrounded, 0);
  const held = gap != null && gap.direction.mean === 0 && gap.level.mean === 0;

  const verdict = held
    ? `The constrained branch admitted ${totalUngrounded} ungrounded studies; the free branch absorbed ${freeUngrounded}. At ${model.latestEra}, the free−constrained verdict gap is zero on both axes — the gate held the recommendation in place.`
    : `At ${model.latestEra}, the free branch and the constrained branch no longer agree: the verdict gap is ${gap ? gap.direction.mean.toFixed(2) : "0.00"} on direction and ${gap ? gap.level.mean.toFixed(2) : "0.00"} on GRADE level.`;

  return (
    <div className="rounded-[1.6rem] border-2 border-[rgba(15,141,119,0.30)] bg-[rgba(15,141,119,0.07)] p-5">
      <div className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
        Free − constrained gap · era {model.latestEra}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <GapStat
          label="Direction"
          interval={gap?.direction ?? null}
        />
        <GapStat
          label="GRADE level"
          interval={gap?.level ?? null}
        />
      </div>
      <div className="mt-4 text-sm leading-6 text-[var(--foreground)]">{verdict}</div>
      {!model.scientific ? (
        <div className="mt-3 text-[0.72rem] leading-5 text-[var(--muted)]">
          Illustrative offline run — shows the mechanism, not scored evidence.
        </div>
      ) : null}
    </div>
  );
}

function GapStat({
  label,
  interval,
}: {
  label: string;
  interval: { mean: number; low: number; high: number } | null;
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-white/85 px-3 py-3">
      <div className="text-[0.62rem] uppercase tracking-[0.12em] text-[var(--muted)]">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums text-[var(--foreground)]">
        {interval ? interval.mean.toFixed(2) : "—"}
      </div>
      <div className="mt-0.5 text-[0.66rem] tabular-nums text-[var(--muted)]">
        {interval ? `95% CI [${interval.low.toFixed(2)}, ${interval.high.toFixed(2)}]` : "no interval"}
      </div>
    </div>
  );
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
          <div>Ungrounded carriers: {lineage.ungrounded_carriers.join(", ") || "none"}</div>
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
