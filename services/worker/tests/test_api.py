from __future__ import annotations

import importlib
import sys

from contextlib import contextmanager

from fastapi.testclient import TestClient


@contextmanager
def load_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDEVO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEDEVO_FORCE_FALLBACK", "1")
    for module_name in list(sys.modules):
        if module_name.startswith("app"):
            sys.modules.pop(module_name)

    main = importlib.import_module("app.main")
    with TestClient(main.app) as client:
        yield client


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
