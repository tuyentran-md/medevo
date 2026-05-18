# MedEvo

**An interactive research instrument that simulates how clinical guidelines and study
conclusions may *drift* over 10, 20, and 30 years as AI-generated text accumulates in the
biomedical literature.**

You paste in a guideline excerpt or a paper's conclusion. MedEvo extracts its core claims,
then lets those claims *evolve* under a model of evidence contamination — and renders three
horizon panels showing where the claim landscape might sit a decade, two decades, and three
decades out.

> Every horizon panel is **one draw from a distribution, not a forecast.** MedEvo does not
> predict the future of any guideline. It makes the *fragility* of a claim under shifting
> evidence visible and arguable.

## Why this exists

Evidence-based medicine assumes the literature it rests on is a roughly faithful record of
what was studied. As AI-generated text enters biomedical publishing, that assumption weakens:
synthetic claims can be cited, re-synthesized, and hardened into apparent consensus without
new underlying data. MedEvo is a sandbox for reasoning about that failure mode — a way to ask
*which clinical claims are robust to contamination, and which quietly rot* — rather than a
prediction engine.

It is built as a translation layer between evidence and deployment: concrete, manipulable,
and honest about its own limits.

## How it works

```
guideline / paper text
        │
        ▼
  claim extraction        ── core claims isolated (capped, deterministic ordering)
        │
        ▼
  emergent simulation     ── claims evolve under a contamination model across YEARS = (10, 20, 30)
        │  CIVER  verdicts on claim survival / inversion
        │  BRIM   discrete drift events along the trajectory
        ▼
  3 horizon panels        ── each a single draw from a distribution, never a point forecast
```

Anchors held fixed by the engine:

- Pre-2023 literature contamination approximated near zero.
- Rising AI-text prevalence in biomedical publishing is treated as an empirical anchor, not a tuned knob.
- Every year-10/20/30 panel is rendered as **one draw**, never a forecast.

CIVER and BRIM are internal engine components (claim-verdict and drift-event models). They are
research scaffolding within this instrument — no commercial or intellectual-property claims are
made here.

## Scientific honesty

A run is **scientific only when a real generative model produces the trajectory.** If no model
is reachable, the worker degrades to a deterministic local simulator so the app still runs
without a paid API — but that output is explicitly **non-scientific** and the UI flags it.
"It still worked" is not the same as "it was a valid run."

The three bundled showcases (`Illustrative` sepsis, bronchiolitis, antibiotic stewardship) use
**synthetic, demo-only** clinical text. No patient data, no real guideline is reproduced.

## Monorepo layout

```text
apps/web            Next.js public UI
packages/contracts  Shared TypeScript types + JSON schemas
services/worker     FastAPI API + background simulation worker
```

## Local development

### Worker (`:8000`)

```bash
cd services/worker
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

On first boot the worker generates the showcase bundles synchronously through the local
model, so the initial startup is intentionally slow. Subsequent boots are fast.

Optional local-model variables (defaults shown):

```bash
MEDEVO_OLLAMA_BASE_URL=http://127.0.0.1:11434
MEDEVO_OLLAMA_MODEL=gemma3:12b
```

To run a real scientific pass locally: `ollama pull gemma3:12b` before starting the worker.

### Web (`:3000`)

```bash
npm install
NEXT_PUBLIC_MEDEVO_WORKER_URL=http://127.0.0.1:8000 npm run dev:web
```

You can also bring your own API key at request time (BYOK) — keys are passed per-request and
are never persisted to disk or the database.

## Author

Tuyen Tran, MD — pediatric surgeon, building tools at the intersection of evidence-based
medicine, AI, and low-resource clinical settings. ORCID
[0009-0003-0535-6225](https://orcid.org/0009-0003-0535-6225).

This is a research instrument, not clinical decision support. Nothing it outputs should
inform the care of an actual patient.
