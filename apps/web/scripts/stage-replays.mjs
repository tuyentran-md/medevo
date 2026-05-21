import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const webRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const repoRoot = path.resolve(webRoot, "..", "..");
const sourceDir = process.env.MEDEVO_REPLAY_SOURCE_DIR
  ? path.resolve(process.env.MEDEVO_REPLAY_SOURCE_DIR)
  : path.join(repoRoot, "services", "worker", "data", "artifacts");
const outputDir = path.join(webRoot, "public", "replays");
const includeExtra = new Set(
  String(process.env.MEDEVO_REPLAY_INCLUDE || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean),
);

fs.rmSync(outputDir, { recursive: true, force: true });
fs.mkdirSync(outputDir, { recursive: true });

function digest(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function deriveTitle(bundle, runId) {
  const firstSentence = String(bundle.input_text || "").split(/[.!?\n]/)[0]?.trim();
  return firstSentence && firstSentence.length > 8 ? firstSentence.slice(0, 80) : runId;
}

const index = [];
for (const runId of fs.readdirSync(sourceDir)) {
  if (!runId.startsWith("showcase-") && !includeExtra.has(runId)) {
    continue;
  }
  const runDir = path.join(sourceDir, runId);
  if (!fs.statSync(runDir).isDirectory()) {
    continue;
  }
  const bundlePath = path.join(runDir, "bundle.json");
  const metaPath = path.join(runDir, "meta.json");
  if (!fs.existsSync(bundlePath) || !fs.existsSync(metaPath)) {
    continue;
  }

  const bundle = JSON.parse(fs.readFileSync(bundlePath, "utf-8"));
  const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
  const years = meta?.summary?.years || Object.keys(bundle?.db_growth || {}).map(Number).sort((a, b) => a - b);
  const runSummary = {
    run: {
      id: runId,
      status: "completed",
      created_at: new Date(fs.statSync(bundlePath).mtimeMs).toISOString(),
      input_digest: digest(String(bundle.input_text || "")),
      title: deriveTitle(bundle, runId),
      backend_config: {
        backend: bundle?.provenance_log?.provider || "ollama",
        model: bundle?.model_descriptor?.name || bundle?.provenance_log?.model || "unknown",
        base_url: bundle?.provenance_log?.base_url || "",
        using_fallback: bundle?.scientific === false,
      },
      branch_config: {
        free: "GTB only",
        constrained: "GTB + CIVER + BRIM",
      },
    },
    years,
    input_mode: "guideline",
    input_source: "showcase",
    error: null,
    showcase: true,
  };

  const targetDir = path.join(outputDir, runId);
  fs.mkdirSync(targetDir, { recursive: true });
  fs.copyFileSync(bundlePath, path.join(targetDir, "bundle.json"));
  fs.copyFileSync(metaPath, path.join(targetDir, "meta.json"));
  fs.writeFileSync(
    path.join(targetDir, "run.json"),
    JSON.stringify(runSummary, null, 2),
    "utf-8",
  );

  index.push({
    id: runId,
    run_id: runId,
    title: runSummary.run.title,
    description: String(bundle.validation_notes?.[0] || "Sealed replay bundle"),
    input_mode: "guideline",
    tags: [
      bundle.scientific === false ? "illustrative" : "scientific",
      "sealed-bundle",
      "static-replay",
    ],
    status: "completed",
  });
}

fs.writeFileSync(path.join(outputDir, "index.json"), JSON.stringify(index, null, 2), "utf-8");
console.log(`Staged ${index.length} replay bundle(s) into ${outputDir}`);
