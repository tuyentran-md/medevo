from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

from app.config import DB_PATH
from app.models import BackendConfigModel, RunSummary, SimulationRunModel


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(get_conn()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                title TEXT NOT NULL,
                backend TEXT NOT NULL,
                model TEXT NOT NULL,
                base_url TEXT,
                using_fallback INTEGER NOT NULL,
                input_mode TEXT NOT NULL,
                input_source TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                error TEXT
            );
            """
        )
        conn.commit()


def insert_run(
    *,
    run_id: str,
    status: str,
    created_at: datetime,
    input_digest: str,
    title: str,
    backend: str,
    model: str,
    base_url: str | None,
    using_fallback: bool,
    input_mode: str,
    input_source: str,
    artifact_dir: str,
) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                id, status, created_at, input_digest, title, backend, model, base_url,
                using_fallback, input_mode, input_source, artifact_dir, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                status,
                created_at.isoformat(),
                input_digest,
                title,
                backend,
                model,
                base_url,
                int(using_fallback),
                input_mode,
                input_source,
                artifact_dir,
            ),
        )
        conn.commit()


def update_run_status(run_id: str, status: str, error: str | None = None) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, error = ? WHERE id = ?",
            (status, error, run_id),
        )
        conn.commit()


def get_run_row(run_id: str) -> sqlite3.Row | None:
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return row


def run_exists(run_id: str) -> bool:
    return get_run_row(run_id) is not None


def active_run_count() -> int:
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE status IN ('queued', 'running')"
        ).fetchone()
    return int(row["n"])


def build_run_summary(row: sqlite3.Row, years: list[int], showcase: bool = False) -> RunSummary:
    backend_config = BackendConfigModel(
        backend=row["backend"],
        model=row["model"],
        base_url=row["base_url"],
        using_fallback=bool(row["using_fallback"]),
    )
    run = SimulationRunModel(
        id=row["id"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        input_digest=row["input_digest"],
        title=row["title"],
        backend_config=backend_config,
        branch_config={
            "free": "GTB only",
            "constrained": "GTB + CIVER + BRIM",
        },
    )
    return RunSummary(
        run=run,
        years=years,
        input_mode=row["input_mode"],
        input_source=row["input_source"],
        error=row["error"],
        showcase=showcase,
    )


def dump_row(row: sqlite3.Row) -> dict[str, object]:
    return json.loads(json.dumps(dict(row)))
