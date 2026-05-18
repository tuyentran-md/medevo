from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

from app.config import DB_PATH
from app.models import BackendConfigModel, EvidenceUnit, LineageRecord, RunSummary, SimulationRunModel


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

            CREATE TABLE IF NOT EXISTS source_catalog (
                run_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                label TEXT NOT NULL,
                body TEXT NOT NULL,
                PRIMARY KEY (run_id, source_id)
            );

            CREATE TABLE IF NOT EXISTS evidence_units (
                run_id TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                branch TEXT NOT NULL,
                producer TEXT NOT NULL,
                provenance TEXT NOT NULL,
                direction TEXT NOT NULL,
                cited_ids_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                PRIMARY KEY (run_id, unit_id)
            );

            CREATE TABLE IF NOT EXISTS citation_edges (
                run_id TEXT NOT NULL,
                from_unit TEXT NOT NULL,
                to_id TEXT NOT NULL,
                PRIMARY KEY (run_id, from_unit, to_id)
            );

            CREATE TABLE IF NOT EXISTS lineage_records (
                run_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                branch TEXT NOT NULL,
                surviving_real_json TEXT NOT NULL,
                lost_real_json TEXT NOT NULL,
                synthetic_carriers_json TEXT NOT NULL,
                verdict_before TEXT NOT NULL,
                verdict_after TEXT NOT NULL,
                PRIMARY KEY (run_id, claim_id, year, branch)
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


def insert_ecology_records(
    *,
    run_id: str,
    source_catalog: dict[str, list[object]],
    evidence_units: list[EvidenceUnit],
    lineage_records: list[LineageRecord],
) -> None:
    source_rows = [
        (
            run_id,
            getattr(source, "claim_id"),
            getattr(source, "source_id"),
            getattr(source, "label"),
            getattr(source, "text"),
        )
        for sources in source_catalog.values()
        for source in sources
    ]

    deduped_units = {unit.id: unit for unit in evidence_units}
    evidence_rows = [
        (
            run_id,
            unit.id,
            unit.claim_id,
            unit.year,
            unit.branch,
            unit.producer,
            unit.provenance,
            unit.direction,
            json.dumps(unit.cited_ids),
            unit.rationale,
        )
        for unit in deduped_units.values()
    ]
    citation_rows = [
        (run_id, unit.id, cited_id)
        for unit in deduped_units.values()
        for cited_id in unit.cited_ids
    ]
    lineage_rows = [
        (
            run_id,
            record.claim_id,
            record.year,
            record.branch,
            json.dumps(record.surviving_real),
            json.dumps(record.lost_real),
            json.dumps(record.synthetic_carriers),
            record.verdict_before,
            record.verdict_after,
        )
        for record in lineage_records
    ]

    with closing(get_conn()) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO source_catalog (
                run_id, claim_id, source_id, label, body
            ) VALUES (?, ?, ?, ?, ?)
            """,
            source_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO evidence_units (
                run_id, unit_id, claim_id, year, branch, producer, provenance,
                direction, cited_ids_json, rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            evidence_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO citation_edges (
                run_id, from_unit, to_id
            ) VALUES (?, ?, ?)
            """,
            citation_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineage_records (
                run_id, claim_id, year, branch, surviving_real_json,
                lost_real_json, synthetic_carriers_json, verdict_before,
                verdict_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            lineage_rows,
        )
        conn.commit()
