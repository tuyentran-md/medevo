from __future__ import annotations

from pathlib import Path

import app.agents
from app.agents import SrmaAgent
from app.db import Tier3StudyStore, init_db, insert_tier3_study, list_tier3_studies
from app.models import ExecutionWarrant, Study


def _study(
    study_id: str,
    *,
    claim_id: str = "claim-1",
    year: int = 2020,
    provenance: str = "GROUNDED",
) -> Study:
    study = Study(
        id=study_id,
        claim_id=claim_id,
        year=year,
        direction="REFUTES",
        effect_point=1.08,
        effect_ci=(0.92, 1.26),
        n=240,
        quality=0.9,
        provenance=provenance,
        pmids=["111"] if provenance == "GROUNDED" else [],
        numeric=True,
        rationale="Admissible study record.",
        output_hash=f"hash-{study_id}",
    )
    return study


def _warrant(study: Study, *, issued: bool = True) -> ExecutionWarrant:
    return ExecutionWarrant(
        id=f"W-{study.id}",
        output_id=study.id,
        output_hash=study.output_hash or "",
        run_id="run-1",
        claim_id=study.claim_id,
        branch="constrained",
        year=study.year,
        status="ISSUED" if issued else "REFUSED",
        issued=issued,
        integrity_score=1.0 if issued else 0.0,
        threshold=0.6,
    )


def test_tier3_db_free_accepts_unwarranted_constrained_requires_warrant(tmp_path, monkeypatch) -> None:
    from app import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "medevo.db")
    init_db()

    real_study = _study("study-real")
    synthetic_study = _study("study-syn", provenance="UNGROUNDED")

    assert insert_tier3_study(run_id="run-1", branch="free", study=synthetic_study) is True
    assert (
        insert_tier3_study(
            run_id="run-1",
            branch="constrained",
            study=synthetic_study,
            require_warrant=True,
        )
        is False
    )
    assert (
        insert_tier3_study(
            run_id="run-1",
            branch="constrained",
            study=real_study,
            warrant=_warrant(real_study),
            require_warrant=True,
        )
        is True
    )

    assert [study.id for study in list_tier3_studies(run_id="run-1", branch="free")] == ["study-syn"]
    assert [study.id for study in list_tier3_studies(run_id="run-1", branch="constrained")] == [
        "study-real"
    ]


def test_srma_agent_reads_tier3_db_without_pubmed_import(tmp_path, monkeypatch) -> None:
    from app import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "medevo.db")
    init_db()
    study = _study("study-real")
    assert insert_tier3_study(
        run_id="run-1",
        branch="constrained",
        study=study,
        warrant=_warrant(study),
        require_warrant=True,
    )

    source = Path(app.agents.__file__).read_text(encoding="utf-8")
    source = source[source.index("class SrmaAgent") : source.index("def _attempt_seed")]
    assert "pubmed" not in source.lower()

    guideline = SrmaAgent(Tier3StudyStore()).run(
        run_id="run-1",
        branch="constrained",
        claim_id="claim-1",
        year=2020,
    )
    assert guideline.direction == "REFUTES"
    assert guideline.level == "conditional-against"
