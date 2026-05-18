"use client";

import type { ReactNode } from "react";
import { useEffect, useEffectEvent, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, LoaderCircle, ShieldCheck, ShieldX, Sparkles } from "lucide-react";

import { fetchArtifacts, fetchRun, type ArtifactResponse, type RunSummaryResponse } from "@/lib/worker";

type BranchName = "free" | "constrained";

export function RunViewer({ runId }: { runId: string }) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactResponse | null>(null);
  const [selectedYear, setSelectedYear] = useState(10);
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useEffectEvent(async () => {
    try {
      const nextSummary = await fetchRun(runId);
      setSummary(nextSummary);
      if (nextSummary.run.status === "completed") {
        const nextArtifacts = await fetchArtifacts(runId);
        setArtifacts(nextArtifacts);
        setError(null);
      } else if (nextSummary.run.status === "failed") {
        setError(nextSummary.error ?? "Run failed.");
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Unable to load run.");
    }
  });

  useEffect(() => {
    refresh();
    const interval = setInterval(() => {
      refresh();
    }, 2500);
    return () => clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    if (!artifacts) {
      return;
    }
    const firstClaim = artifacts.bundle.snapshots.free
      .find((snapshot) => snapshot.year === selectedYear)
      ?.claims.at(0);
    if (firstClaim && !selectedClaimId) {
      setSelectedClaimId(firstClaim.claim_id);
    }
  }, [artifacts, selectedYear, selectedClaimId]);

  const freeSnapshot = artifacts?.bundle.snapshots.free.find(
    (snapshot) => snapshot.year === selectedYear,
  );
  const constrainedSnapshot = artifacts?.bundle.snapshots.constrained.find(
    (snapshot) => snapshot.year === selectedYear,
  );

  const claimIds =
    freeSnapshot?.claims.map((claim) => claim.claim_id) ??
    constrainedSnapshot?.claims.map((claim) => claim.claim_id) ??
    [];

  const activeClaimId = selectedClaimId ?? claimIds[0] ?? null;
  const freeClaim = freeSnapshot?.claims.find((claim) => claim.claim_id === activeClaimId);
  const constrainedClaim = constrainedSnapshot?.claims.find(
    (claim) => claim.claim_id === activeClaimId,
  );
  const graph = artifacts?.bundle.claim_graphs.find((item) => item.claim_id === activeClaimId);

  const isLoading = !summary || (summary.run.status !== "completed" && !error);

  return (
    <main className="relative overflow-hidden px-5 pb-20 pt-8 sm:px-8 lg:px-12">
      <div className="grain" />
      <div className="mx-auto max-w-7xl">
        <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
                  Simulation run
                </p>
                <h1 className="mt-2 font-[family-name:var(--font-display)] text-4xl text-[var(--foreground)]">
                  {summary?.run.title ?? "Loading MedEvo run"}
                </h1>
              </div>
              <div className="rounded-full border border-[var(--border)] bg-white/70 px-4 py-2 text-sm">
                {summary?.run.status ?? "queued"}
              </div>
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <MetricCard
                label="Backend"
                value={summary?.run.backend_config.backend ?? "waiting"}
                note={
                  summary?.run.backend_config.using_fallback
                    ? "Deterministic fallback active"
                    : summary?.run.backend_config.model ?? "Configured local model"
                }
              />
              <MetricCard
                label="Input"
                value={summary?.input_mode ?? "pending"}
                note={summary?.input_source ?? "pending"}
              />
              <MetricCard
                label="Years"
                value={summary?.years.join(" / ") ?? "10 / 20 / 30"}
                note="Distribution snapshots"
              />
              <MetricCard
                label="Blocked outputs"
                value={
                  artifacts?.meta.summary.has_blocked_outputs ? "present" : "none yet"
                }
                note="CIVER discard path under A2"
              />
            </div>

            {summary?.showcase && artifacts?.meta.description ? (
              <div className="mt-6 rounded-[1.6rem] border border-[var(--border)] bg-white/75 p-4 text-sm leading-7 text-[var(--muted)]">
                {artifacts.meta.description}
              </div>
            ) : null}

            {artifacts && artifacts.bundle.scientific === false ? (
              <div className="mt-6 rounded-[1.6rem] border-2 border-[#b45309] bg-[#fef3c7] p-4">
                <div className="text-sm font-semibold uppercase tracking-[0.18em] text-[#92400e]">
                  {artifacts.bundle.mode_banner || "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"}
                </div>
                <div className="mt-2 text-sm leading-6 text-[#92400e]">
                  No real model generated this run, so the free-vs-constrained
                  contrast carries no scientific weight. Excluded from any paper
                  artifact and the preregistered test set.
                </div>
              </div>
            ) : null}

            <div className="mt-6 rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
              <div className="text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                Validation posture
              </div>
              <div className="mt-3 grid gap-3">
                {(artifacts?.bundle.validation_notes ?? []).map((note) => (
                  <div key={note} className="rounded-2xl bg-white/85 px-4 py-3 text-sm leading-6 text-[var(--foreground)]">
                    {note}
                  </div>
                ))}
              </div>
            </div>

            {error ? (
              <div className="mt-6 rounded-2xl border border-[rgba(181,74,52,0.28)] bg-[rgba(181,74,52,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
                {error}
              </div>
            ) : null}

            {isLoading ? (
              <div className="mt-6 flex items-center gap-3 rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-4 text-sm text-[var(--muted)]">
                <LoaderCircle className="h-4 w-4 animate-spin text-[var(--accent)]" />
                Polling worker for completion...
              </div>
            ) : null}
          </section>

          <section className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel-strong)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
            <div className="flex flex-wrap items-center gap-3">
              {[10, 20, 30].map((year) => (
                <button
                  key={year}
                  type="button"
                  onClick={() => setSelectedYear(year)}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                    selectedYear === year
                      ? "bg-[var(--foreground)] text-white"
                      : "border border-[var(--border)] bg-white/70 text-[var(--foreground)]"
                  }`}
                >
                  Year {year}
                </button>
              ))}
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <BranchPanel
                branch="free"
                title="GTB only"
                icon={<ShieldX className="h-5 w-5 text-[var(--danger)]" />}
                snapshot={freeSnapshot}
                onSelect={setSelectedClaimId}
                selectedClaimId={activeClaimId}
              />
              <BranchPanel
                branch="constrained"
                title="GTB + CIVER + BRIM"
                icon={<ShieldCheck className="h-5 w-5 text-[var(--accent)]" />}
                snapshot={constrainedSnapshot}
                onSelect={setSelectedClaimId}
                selectedClaimId={activeClaimId}
              />
            </div>
          </section>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <section className="rounded-[2rem] border border-[var(--border)] bg-white/88 p-6 shadow-[0_20px_50px_rgba(17,35,30,0.08)]">
            <div className="flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
              <Sparkles className="h-4 w-4 text-[var(--accent)]" />
              Why did you conclude this?
            </div>

            {freeClaim || constrainedClaim ? (
              <div className="mt-5 grid gap-4">
                <ExplanationCard
                  title="Free branch"
                  claim={freeClaim}
                  band={freeSnapshot?.band.label}
                  anchors={freeSnapshot?.anchors ?? []}
                />
                <ExplanationCard
                  title="Constrained branch"
                  claim={constrainedClaim}
                  band={constrainedSnapshot?.band.label}
                  anchors={constrainedSnapshot?.anchors ?? []}
                />
              </div>
            ) : (
              <div className="mt-5 rounded-2xl border border-[var(--border)] bg-[rgba(17,35,30,0.03)] px-4 py-4 text-sm text-[var(--muted)]">
                Select a claim to inspect its reasoning surface.
              </div>
            )}
          </section>

          <section className="rounded-[2rem] border border-[var(--border)] bg-white/88 p-6 shadow-[0_20px_50px_rgba(17,35,30,0.08)]">
            <div className="flex items-center gap-2 text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
              <AlertTriangle className="h-4 w-4 text-[var(--accent-2)]" />
              Claim graph inspect
            </div>

            {graph ? (
              <div className="mt-5 grid gap-5">
                <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
                  <div className="text-sm font-semibold text-[var(--foreground)]">
                    {graph.claim_text}
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
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
              </div>
            ) : (
              <div className="mt-5 rounded-2xl border border-[var(--border)] bg-[rgba(17,35,30,0.03)] px-4 py-4 text-sm text-[var(--muted)]">
                Claim graph will appear once artifacts are ready.
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
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
    <div className="rounded-[1.6rem] border border-[var(--border)] bg-white/80 p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">{label}</div>
      <div className="mt-2 text-lg font-semibold text-[var(--foreground)]">{value}</div>
      <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{note}</div>
    </div>
  );
}

function BranchPanel({
  branch,
  title,
  icon,
  snapshot,
  onSelect,
  selectedClaimId,
}: {
  branch: BranchName;
  title: string;
  icon: ReactNode;
  snapshot?: ArtifactResponse["bundle"]["snapshots"][BranchName][number];
  onSelect: (claimId: string) => void;
  selectedClaimId: string | null;
}) {
  return (
    <div className="rounded-[1.8rem] border border-[var(--border)] bg-white/88 p-5">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <div className="text-sm uppercase tracking-[0.18em] text-[var(--muted)]">
            {branch}
          </div>
          <div className="text-xl font-semibold text-[var(--foreground)]">{title}</div>
        </div>
      </div>

      <div className="mt-4 grid gap-3">
        {snapshot?.claims.map((claim) => (
          <button
            key={claim.claim_id}
            type="button"
            onClick={() => onSelect(claim.claim_id)}
            className={`rounded-[1.4rem] border p-4 text-left transition ${
              selectedClaimId === claim.claim_id
                ? "border-[var(--accent)] bg-[rgba(15,141,119,0.08)]"
                : "border-[var(--border)] bg-[rgba(17,35,30,0.03)]"
            }`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-semibold text-[var(--foreground)]">
                {claim.direction} · {claim.strength}
              </div>
              <div className="rounded-full bg-white/80 px-3 py-1 text-xs text-[var(--muted)]">
                diff {claim.divergence_score.toFixed(2)}
              </div>
            </div>
            <div className="mt-2 text-sm leading-6 text-[var(--muted)]">
              {claim.claim_text}
            </div>
            <div className="mt-3 flex gap-3 text-xs uppercase tracking-[0.14em] text-[var(--muted)]">
              <span>Emitted {claim.emitted_count}</span>
              <span>Blocked {claim.blocked_count}</span>
            </div>
          </button>
        ))}
      </div>

      {snapshot ? (
        <div className="mt-4 rounded-[1.4rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4 text-sm text-[var(--muted)]">
          <div className="font-medium text-[var(--foreground)]">Sensitivity band</div>
          <div className="mt-2">
            {snapshot.band.low.toFixed(2)} to {snapshot.band.high.toFixed(2)}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ExplanationCard({
  title,
  claim,
  band,
  anchors,
}: {
  title: string;
  claim?: ArtifactResponse["bundle"]["snapshots"][BranchName][number]["claims"][number];
  band?: string;
  anchors: string[];
}) {
  if (!claim) {
    return null;
  }

  return (
    <div className="rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4">
      <div className="flex items-center justify-between gap-4">
        <div className="text-lg font-semibold text-[var(--foreground)]">{title}</div>
        <div className="rounded-full bg-white/85 px-3 py-1 text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
          {claim.direction} · {claim.strength}
        </div>
      </div>
      <div className="mt-3 text-sm leading-6 text-[var(--foreground)]">{claim.why_summary}</div>
      <div className="mt-4 grid gap-2">
        {claim.civer.map((verdict) => (
          <div
            key={verdict.node_id}
            className={`rounded-xl px-3 py-2 text-sm ${
              verdict.passed
                ? "bg-[rgba(15,141,119,0.08)] text-[var(--foreground)]"
                : "bg-[rgba(181,74,52,0.08)] text-[var(--danger)]"
            }`}
          >
            {verdict.reasons.join(" ")}
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-xl border border-[var(--border)] bg-white/85 p-3 text-xs leading-6 text-[var(--muted)]">
        <div>{band}</div>
        <div className="mt-2">{anchors.join(" ")}</div>
      </div>
    </div>
  );
}
