from __future__ import annotations

import importlib
import sqlite3
import sys

from contextlib import contextmanager

from fastapi.testclient import TestClient


@contextmanager
def load_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDEVO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEDEVO_FORCE_FALLBACK", "1")
    # Snapshot the app.* modules so the tmp_path/fallback-bound reload is undone on
    # exit. Without this, popping + re-importing app.* leaves a duplicate module
    # graph in sys.modules: any test collected BEFORE this one holds references to
    # the original modules while the reloaded copies coexist, producing cross-graph
    # type-identity Heisenbugs in unrelated downstream tests.
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "app" or name.startswith("app.")
    }
    for module_name in list(sys.modules):
        if module_name.startswith("app"):
            sys.modules.pop(module_name)

    try:
        main = importlib.import_module("app.main")
        with TestClient(main.app) as client:
            yield client
    finally:
        for module_name in list(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved_modules)


def test_create_run_and_fetch_artifacts(tmp_path, monkeypatch) -> None:
    with load_client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/runs",
            json={
                "title": "Custom pediatric sepsis run",
                "input_mode": "guideline",
                "input_source": "paste",
                "input_text": (
                    "Children with suspected sepsis should receive cultures before antibiotics when possible. "
                    "Broad-spectrum antibiotics should be started promptly. Escalate to ICU support when shock persists."
                ),
                "backend": "ollama",
                "horizons": [10, 20, 30],
            },
        )
        assert response.status_code == 200
        run_id = response.json()["id"]

        run_response = client.get(f"/runs/{run_id}")
        assert run_response.status_code == 200
        assert run_response.json()["run"]["status"] == "completed"

        artifact_response = client.get(f"/runs/{run_id}/artifacts")
        assert artifact_response.status_code == 200
        payload = artifact_response.json()
        assert payload["meta"]["summary"]["years"] == [10, 20, 30]
        assert set(payload["bundle"]["snapshots"].keys()) == {"free", "constrained"}
        assert "lineage" in payload["bundle"]
        with sqlite3.connect(tmp_path / "medevo.db") as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM lineage_records WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            study_row = conn.execute(
                "SELECT COUNT(*) FROM study_db WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            guideline_row = conn.execute(
                "SELECT COUNT(*) FROM guideline_timeline WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        assert row is not None
        assert row[0] > 0
        assert study_row is not None
        assert study_row[0] > 0
        assert guideline_row is not None
        assert guideline_row[0] > 0


def test_byok_request_does_not_persist_api_key(tmp_path, monkeypatch) -> None:
    with load_client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/runs",
            json={
                "title": "BYOK run",
                "input_mode": "paper",
                "input_source": "paste",
                "input_text": (
                    "Conclusion: Narrow targeted antibiotics reduced exposure without increasing failure."
                ),
                "backend": "gemini",
                "api_key": "secret-value",
                "model": "gemini-2.5-flash",
                "horizons": [10, 20, 30],
            },
        )
        assert response.status_code == 200
        run_id = response.json()["id"]

        run_response = client.get(f"/runs/{run_id}")
        assert run_response.status_code == 200
        data = run_response.json()
        assert data["run"]["backend_config"]["backend"] == "gemini"
        serialized = str(data)
        assert "secret-value" not in serialized
