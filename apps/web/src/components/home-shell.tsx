"use client";

import { useDeferredValue, useEffect, useState, useTransition } from "react";
import { motion } from "framer-motion";
import { ArrowRight, FlaskConical, LoaderCircle, LockKeyhole, Upload } from "lucide-react";
import { useRouter } from "next/navigation";

import { createRun, fetchShowcase, type ShowcaseItem } from "@/lib/worker";

const backendOptions = [
  { value: "ollama", label: "Local default (Ollama)" },
  { value: "openai-compatible", label: "OpenAI-compatible endpoint" },
  { value: "gemini", label: "Gemini API" },
  { value: "anthropic", label: "Anthropic API" },
] as const;

export function HomeShell() {
  const router = useRouter();
  const [showcase, setShowcase] = useState<ShowcaseItem[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<"paste" | "upload">("paste");
  const [inputMode, setInputMode] = useState<"guideline" | "paper">("guideline");
  const [backend, setBackend] = useState<(typeof backendOptions)[number]["value"]>("ollama");
  const [title, setTitle] = useState("");
  const [inputText, setInputText] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isPending, startTransition] = useTransition();
  const deferredInput = useDeferredValue(inputText);

  useEffect(() => {
    let cancelled = false;

    fetchShowcase()
      .then((data) => {
        if (!cancelled) {
          setShowcase(data);
        }
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setLoadError(error.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const canSubmit =
    sourceMode === "paste" ? inputText.trim().length > 24 : Boolean(selectedFile);

  function handleSubmit() {
    if (!canSubmit) {
      return;
    }

    const formData = new FormData();
    formData.set("title", title);
    formData.set("input_mode", inputMode);
    formData.set("input_source", sourceMode);
    formData.set("backend", backend);
    if (inputText) {
      formData.set("input_text", inputText);
    }
    if (model) {
      formData.set("model", model);
    }
    if (apiKey) {
      formData.set("api_key", apiKey);
    }
    if (baseUrl) {
      formData.set("base_url", baseUrl);
    }
    if (selectedFile) {
      formData.set("file", selectedFile);
    }

    setSubmitError(null);
    startTransition(async () => {
      try {
        const created = await createRun(formData);
        router.push(`/runs/${created.id}`);
      } catch (error) {
        setSubmitError(error instanceof Error ? error.message : "Unable to create run.");
      }
    });
  }

  return (
    <main className="relative overflow-hidden px-5 pb-20 pt-8 sm:px-8 lg:px-12">
      <div className="grain" />
      <section className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]"
        >
          <div className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel)] p-7 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-10">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-white/70 px-4 py-2 text-xs uppercase tracking-[0.28em] text-[var(--muted)]">
              <FlaskConical className="h-4 w-4 text-[var(--accent)]" />
              Public demo · Research instrument
            </div>

            <h1 className="max-w-2xl font-[family-name:var(--font-display)] text-3xl leading-[1.1] text-[var(--foreground)] sm:text-4xl lg:text-5xl">
              Watch a clinical guideline drift across an AI-saturated evidence world.
            </h1>

            <p className="mt-5 max-w-2xl text-lg leading-8 text-[var(--muted)]">
              Run the same input through two futures: one with ground-truth binding only,
              one with GTB plus CIVER and BRIM. Inspect year 10, 20, and 30 outputs as
              a distribution draw, not a forecast.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {[
                "Tier-1 studies and Tier-2 reviews stay agentic.",
                "Tier-3 recommendation panel stays deterministic.",
                "Paid APIs remain opt-in, never the build default.",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-3xl border border-[var(--border)] bg-white/70 p-4 text-sm leading-6 text-[var(--foreground)]"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[2rem] border border-[var(--border)] bg-[var(--panel-strong)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-[family-name:var(--font-display)] text-3xl">
                Run your own input
              </h2>
              <span className="rounded-full bg-[var(--foreground)] px-3 py-1 text-xs uppercase tracking-[0.22em] text-white">
                Custom input
              </span>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => setSourceMode("paste")}
                className={`rounded-2xl border px-4 py-3 text-left transition ${
                  sourceMode === "paste"
                    ? "border-[var(--accent)] bg-[rgba(15,141,119,0.08)]"
                    : "border-[var(--border)] bg-white/70"
                }`}
              >
                Paste text
              </button>
              <button
                type="button"
                onClick={() => setSourceMode("upload")}
                className={`rounded-2xl border px-4 py-3 text-left transition ${
                  sourceMode === "upload"
                    ? "border-[var(--accent)] bg-[rgba(15,141,119,0.08)]"
                    : "border-[var(--border)] bg-white/70"
                }`}
              >
                Upload file
              </button>
            </div>

            <div className="mt-5 grid gap-4">
              <label className="grid gap-2">
                <span className="text-sm font-medium text-[var(--foreground)]">Title</span>
                <input
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                  placeholder="Optional run label"
                  className="rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none ring-0 transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                />
              </label>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="grid min-w-0 gap-2">
                  <span className="text-sm font-medium text-[var(--foreground)]">Input mode</span>
                  <select
                    value={inputMode}
                    onChange={(event) =>
                      setInputMode(event.target.value as "guideline" | "paper")
                    }
                    className="w-full rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  >
                    <option value="guideline">Guideline</option>
                    <option value="paper">Paper conclusion</option>
                  </select>
                </label>

                <label className="grid min-w-0 gap-2">
                  <span className="text-sm font-medium text-[var(--foreground)]">Backend</span>
                  <select
                    value={backend}
                    onChange={(event) =>
                      setBackend(event.target.value as (typeof backendOptions)[number]["value"])
                    }
                    className="w-full rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  >
                    {backendOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {sourceMode === "paste" ? (
                <label className="grid gap-2">
                  <span className="text-sm font-medium text-[var(--foreground)]">Guideline or conclusion text</span>
                  <textarea
                    value={inputText}
                    onChange={(event) => setInputText(event.target.value)}
                    placeholder="Paste a clinical guideline excerpt or a paper conclusion."
                    rows={8}
                    className="rounded-[1.6rem] border border-[var(--border)] bg-white/80 px-4 py-4 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  />
                </label>
              ) : (
                <label className="grid gap-3 rounded-[1.6rem] border border-dashed border-[var(--border)] bg-white/75 p-5">
                  <span className="text-sm font-medium text-[var(--foreground)]">
                    Upload `.pdf`, `.txt`, or `.md`
                  </span>
                  <input
                    type="file"
                    accept=".pdf,.txt,.md,text/plain,text/markdown,application/pdf"
                    onChange={(event) =>
                      setSelectedFile(event.target.files?.[0] ?? null)
                    }
                  />
                  <span className="text-sm text-[var(--muted)]">
                    {selectedFile ? selectedFile.name : "No file selected yet."}
                  </span>
                </label>
              )}

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="grid gap-2">
                  <span className="text-sm font-medium text-[var(--foreground)]">Model override</span>
                  <input
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    placeholder={backend === "ollama" ? "gemma3:12b" : "Optional"}
                    className="rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  />
                </label>

                <label className="grid gap-2">
                  <span className="text-sm font-medium text-[var(--foreground)]">Base URL</span>
                  <input
                    value={baseUrl}
                    onChange={(event) => setBaseUrl(event.target.value)}
                    placeholder={
                      backend === "ollama"
                        ? "http://127.0.0.1:11434"
                        : "Optional endpoint"
                    }
                    className="rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  />
                </label>
              </div>

              {backend !== "ollama" ? (
                <label className="grid gap-2">
                  <span className="inline-flex items-center gap-2 text-sm font-medium text-[var(--foreground)]">
                    <LockKeyhole className="h-4 w-4 text-[var(--accent)]" />
                    Bring-your-own API key
                  </span>
                  <input
                    value={apiKey}
                    onChange={(event) => setApiKey(event.target.value)}
                    type="password"
                    placeholder="Key is used only for this run and not persisted."
                    className="rounded-2xl border border-[var(--border)] bg-white/80 px-4 py-3 outline-none transition focus:border-[var(--accent)] focus:shadow-[0_0_0_4px_var(--ring)]"
                  />
                </label>
              ) : null}
            </div>

            <div className="mt-6 rounded-[1.6rem] border border-[var(--border)] bg-[rgba(17,35,30,0.03)] p-4 text-sm leading-6 text-[var(--muted)]">
              <div className="font-semibold text-[var(--foreground)]">Preview of the first slice</div>
              <div className="mt-2 line-clamp-4">
                {deferredInput.trim() || "Paste text or upload a file to preview the seeded claim surface."}
              </div>
            </div>

            {submitError ? (
              <div className="mt-4 rounded-2xl border border-[rgba(181,74,52,0.28)] bg-[rgba(181,74,52,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
                {submitError}
              </div>
            ) : null}

            <button
              type="button"
              disabled={!canSubmit || isPending}
              onClick={handleSubmit}
              className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--foreground)] px-6 py-4 text-sm font-semibold text-white transition hover:translate-y-[-1px] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isPending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              Launch simulation
            </button>
          </div>
        </motion.div>

        <section className="mt-10 rounded-[2rem] border border-[var(--border)] bg-[rgba(255,252,247,0.84)] p-6 shadow-[0_24px_70px_rgba(17,35,30,0.08)] backdrop-blur lg:p-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.22em] text-[var(--muted)]">
                Precomputed showcase
              </p>
              <h2 className="mt-2 font-[family-name:var(--font-display)] text-4xl text-[var(--foreground)]">
                Start with cached runs
              </h2>
            </div>
            <p className="max-w-xl text-sm leading-7 text-[var(--muted)]">
              Showcase runs are precomputed so the public demo stays fast. Custom inputs are
              queued asynchronously and respect worker rate limits.
            </p>
          </div>

          {loadError ? (
            <div className="mt-5 rounded-2xl border border-[rgba(181,74,52,0.28)] bg-[rgba(181,74,52,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
              {loadError}
            </div>
          ) : null}

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {showcase.map((item, index) => (
              <motion.a
                key={item.id}
                href={`/runs/${item.run_id}`}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.08 }}
                className="group rounded-[1.75rem] border border-[var(--border)] bg-white/90 p-5 transition hover:border-[var(--accent)] hover:shadow-[0_20px_50px_rgba(15,141,119,0.12)]"
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="rounded-full bg-[rgba(15,141,119,0.08)] px-3 py-1 text-xs uppercase tracking-[0.18em] text-[var(--accent)]">
                    {item.input_mode}
                  </span>
                  <ArrowRight className="h-4 w-4 text-[var(--muted)] transition group-hover:translate-x-1 group-hover:text-[var(--accent)]" />
                </div>
                <h3 className="mt-4 text-xl font-semibold text-[var(--foreground)]">
                  {item.title}
                </h3>
                <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
                  {item.description}
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {item.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-[var(--border)] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[var(--muted)]"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </motion.a>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
