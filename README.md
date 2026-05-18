# MedEvo

Public interactive demo + research instrument for simulated guideline drift.

## Monorepo layout

```text
apps/web           Next.js public UI
packages/contracts Shared TypeScript types + JSON schemas
services/worker    FastAPI API + background simulation worker
```

## Local development

### Web

```bash
npm install
npm run dev:web
```

Set `NEXT_PUBLIC_MEDEVO_WORKER_URL=http://127.0.0.1:8000`.

### Worker

```bash
cd services/worker
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Optional local-model variables:

```bash
MEDEVO_OLLAMA_BASE_URL=http://127.0.0.1:11434
MEDEVO_OLLAMA_MODEL=gemma3:12b
```

If Ollama is unavailable, the worker falls back to a deterministic local simulator so the app still works without a paid API.
