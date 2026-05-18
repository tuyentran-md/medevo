from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import ARTIFACTS_DIR, DEFAULT_OLLAMA_MODEL, DEFAULT_RATE_LIMIT, YEARS
from app.db import (
    active_run_count,
    build_run_summary,
    get_run_row,
    init_db,
    insert_run,
    run_exists,
    update_run_status,
)
from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.showcase import SHOWCASES, get_showcase
from app.simulator import (
    digest_text,
    extract_text_from_upload,
    resolve_backend,
    sanitize_title,
    simulate_run,
)
from app.storage import ensure_run_dir, has_artifact, read_json, write_json, write_text


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _bootstrap_showcases()
    yield


app = FastAPI(title="MedEvo Worker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_request_payload(payload: dict[str, Any]) -> RunRequestModel:
    try:
        return RunRequestModel.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _parse_run_request(request: Request) -> tuple[RunRequestModel, bytes | None, str | None]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        payload = {
            "title": form.get("title") or None,
            "input_mode": form.get("input_mode"),
            "input_source": form.get("input_source"),
            "input_text": form.get("input_text") or None,
            "showcase_id": form.get("showcase_id") or None,
            "backend": form.get("backend"),
            "model": form.get("model") or None,
            "api_key": form.get("api_key") or None,
            "base_url": form.get("base_url") or None,
        }
        filename = getattr(upload, "filename", None)
        content = await upload.read() if hasattr(upload, "read") else None
        return _parse_request_payload(payload), content, filename

    payload = await request.json()
    return _parse_request_payload(payload), None, None


def _resolve_input_text(run_request: RunRequestModel, upload_content: bytes | None, filename: str | None) -> tuple[str, str]:
    if run_request.input_source == "showcase":
        showcase = get_showcase(run_request.showcase_id or "")
        if showcase is None:
            raise HTTPException(status_code=404, detail="Showcase not found.")
        return showcase.input_text, showcase.title

    if upload_content and filename:
        extracted = extract_text_from_upload(filename, upload_content)
        title = run_request.title or Path(filename).stem.replace("-", " ").title()
        return extracted, title

    if run_request.input_text:
        title_seed = run_request.title or run_request.input_text.splitlines()[0]
        return run_request.input_text, sanitize_title(title_seed, "Custom MedEvo Run")

    raise HTTPException(status_code=400, detail="No input text or file provided.")


def _write_artifacts(run_id: str, bundle: dict[str, Any], metadata: dict[str, Any]) -> None:
    write_json(run_id, "bundle.json", bundle)
    write_json(run_id, "meta.json", metadata)
    write_text(run_id, "input.txt", bundle["input_text"])


def _process_run(run_id: str, run_request: RunRequestModel, input_text: str) -> None:
    update_run_status(run_id, "running")
    try:
        bundle, summary = simulate_run(
            request=run_request,
            input_text=input_text,
            run_id=run_id,
        )
        _write_artifacts(
            run_id,
            bundle.model_dump(),
            {
                "summary": summary,
                "validation": bundle.validation_notes,
            },
        )
        update_run_status(run_id, "completed")
    except Exception as exc:
        update_run_status(run_id, "failed", str(exc))


def _bootstrap_showcases() -> None:
    force_fallback = bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("MEDEVO_BOOTSTRAP_FALLBACK") == "1"
    )
    for showcase in SHOWCASES:
        run_id = f"showcase-{showcase.id}"
        if run_exists(run_id) and has_artifact(run_id, "bundle.json"):
            continue
        run_request = RunRequestModel(
            title=showcase.title,
            input_mode=showcase.input_mode,
            input_source="showcase",
            input_text=showcase.input_text,
            showcase_id=showcase.id,
            backend="ollama",
            model=DEFAULT_OLLAMA_MODEL,
        )
        backend = resolve_backend(run_request)
        if force_fallback:
            backend.using_fallback = True
            backend.model = "deterministic-fallback"
            backend.base_url = None
        ensure_run_dir(run_id)
        insert_run(
            run_id=run_id,
            status="running",
            created_at=datetime.now(UTC),
            input_digest=digest_text(showcase.input_text),
            title=showcase.title,
            backend=backend.backend,
            model=backend.model,
            base_url=backend.base_url,
            using_fallback=backend.using_fallback,
            input_mode=showcase.input_mode,
            input_source="showcase",
            artifact_dir=str(ARTIFACTS_DIR / run_id),
        )
        bundle, summary = simulate_run(
            request=run_request,
            input_text=showcase.input_text,
            run_id=run_id,
            client=DeterministicFakeClient() if force_fallback else None,
        )
        _write_artifacts(
            run_id,
            bundle.model_dump(),
            {
                "summary": summary,
                "validation": bundle.validation_notes,
                "description": showcase.description,
                "tags": showcase.tags,
            },
        )
        update_run_status(run_id, "completed")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/showcase")
def list_showcases() -> list[dict[str, Any]]:
    items = []
    for showcase in SHOWCASES:
        run_id = f"showcase-{showcase.id}"
        row = get_run_row(run_id)
        if row is None:
            continue
        items.append(
            {
                "id": showcase.id,
                "run_id": run_id,
                "title": showcase.title,
                "description": showcase.description,
                "input_mode": showcase.input_mode,
                "tags": showcase.tags,
                "status": row["status"],
            }
        )
    return items


@app.post("/runs")
async def create_run(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    if active_run_count() >= DEFAULT_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Worker queue is full. Please retry shortly.")

    run_request, upload_content, filename = await _parse_run_request(request)
    input_text, derived_title = _resolve_input_text(run_request, upload_content, filename)
    backend = resolve_backend(run_request)
    created_at = datetime.now(UTC)
    run_id = uuid4().hex
    ensure_run_dir(run_id)

    insert_run(
        run_id=run_id,
        status="queued",
        created_at=created_at,
        input_digest=digest_text(input_text),
        title=run_request.title or derived_title,
        backend=backend.backend,
        model=backend.model,
        base_url=backend.base_url,
        using_fallback=backend.using_fallback,
        input_mode=run_request.input_mode,
        input_source=run_request.input_source,
        artifact_dir=str(ARTIFACTS_DIR / run_id),
    )
    background_tasks.add_task(_process_run, run_id, run_request, input_text)
    return {"id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    row = get_run_row(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    summary = build_run_summary(row, list(YEARS), showcase=run_id.startswith("showcase-"))
    return summary.model_dump(mode="json")


@app.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str) -> dict[str, Any]:
    row = get_run_row(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="Artifacts are not ready yet.")
    bundle = read_json(run_id, "bundle.json")
    meta = read_json(run_id, "meta.json")
    return {"run_id": run_id, "bundle": bundle, "meta": meta}
