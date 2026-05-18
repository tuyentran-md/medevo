from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import ARTIFACTS_DIR


def ensure_run_dir(run_id: str) -> Path:
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def artifact_path(run_id: str, name: str) -> Path:
    return ensure_run_dir(run_id) / name


def write_json(run_id: str, name: str, payload: Any) -> Path:
    path = artifact_path(run_id, name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(run_id: str, name: str) -> Any:
    path = artifact_path(run_id, name)
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(run_id: str, name: str, content: str) -> Path:
    path = artifact_path(run_id, name)
    path.write_text(content, encoding="utf-8")
    return path


def has_artifact(run_id: str, name: str) -> bool:
    return artifact_path(run_id, name).is_file()
