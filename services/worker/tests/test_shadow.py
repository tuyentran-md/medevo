from __future__ import annotations

from app.models import ArtifactBundle, ClaimGraph, ClaimNode, EvidenceScope, Study
from app.shadow import evaluate_shadow_civer


def _graph() -> ClaimGraph:
    return ClaimGraph(
        claim_id="claim-1",
        claim_text="Smoking increases coronary heart disease risk.",
        nodes=[
            ClaimNode(id="q", label="question", node_type="QUESTION", timestamp=1),
            ClaimNode(id="m", label="method", node_type="METHOD", timestamp=2),
            ClaimNode(id="e", label="evidence", node_type="EVIDENCE", timestamp=3),
            ClaimNode(id="a", label="analysis", node_type="ANALYSIS", timestamp=4),
            ClaimNode(id="c", label="claim", node_type="CLAIM", timestamp=5),
        ],
        edges=[],
    )


def _study(
    study_id: str,
    *,
    pmids: list[str],
    catalog_pmids: list[str],
    claimed_scope: EvidenceScope | None = None,
    source_scope: EvidenceScope | None = None,
    provenance: str = "GROUNDED",
    failure_mode: str = "none",
) -> Study:
    return Study(
        id=study_id,
        claim_id="claim-1",
        year=2024,
        direction="SUPPORTS",
        quality=0.8,
        provenance=provenance,
        pmids=pmids,
        catalog_pmids=catalog_pmids,
        numeric=True,
        rationale="Fixture study.",
        claimed_scope=claimed_scope or EvidenceScope(year_start=2024, year_end=2024),
        source_scope=source_scope or EvidenceScope(year_start=2024, year_end=2024),
        failure_mode=failure_mode,  # type: ignore[arg-type]
    )


def test_shadow_civer_filters_same_natural_corpus_without_active_branch() -> None:
    scoped = _study("good", pmids=["1"], catalog_pmids=["1"])
    no_cite = _study(
        "bad",
        pmids=[],
        catalog_pmids=[],
        provenance="UNGROUNDED",
        failure_mode="unresolvable",
    )
    bundle = ArtifactBundle(
        input_text="",
        claim_graphs=[_graph()],
        snapshots={},
        branch_diff={},
        anchors=[],
        validation_notes=[],
        corpus_studies={"free": [scoped, no_cite]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    assert report["study_count"] == 2
    assert report["verdict_counts"] == {"passed": 1, "failed": 1, "total": 2}
    assert report["endpoint_2_warrant_enrichment"]["passed"]["ungrounded_rate"] == 0.0
    assert report["endpoint_2_warrant_enrichment"]["failed"]["ungrounded_rate"] == 1.0
    assert "claim-1" in report["all_guideline_latest"]
    assert "claim-1" in report["warranted_guideline_latest"]


def test_shadow_civer_uses_catalog_resolvability_not_provenance_label() -> None:
    labelled_grounded_but_unresolved = _study(
        "unresolved",
        pmids=["999"],
        catalog_pmids=[],
        provenance="GROUNDED",
    )
    bundle = ArtifactBundle(
        input_text="",
        claim_graphs=[_graph()],
        snapshots={},
        branch_diff={},
        anchors=[],
        validation_notes=[],
        corpus_studies={"free": [labelled_grounded_but_unresolved]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    assert report["verdict_counts"]["passed"] == 0
    assert report["study_verdicts"][0]["true_provenance_for_calibration"] == "GROUNDED"
