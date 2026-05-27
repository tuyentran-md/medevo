from __future__ import annotations

from app.models import ArtifactBundle, ClaimEdge, ClaimGraph, ClaimNode, EvidenceScope, ResearchPlan, Study
from app.shadow import evaluate_shadow_civer


def _graph() -> ClaimGraph:
    # Mirrors simulator.build_claim_graph edge layout so patent IC-01 (ANALYSIS
    # ↔ EVIDENCE via ANALYZES) and GC-02 (Q→…→C path) are satisfied. Without
    # these edges the new patent rules would refuse every fixture study.
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
        edges=[
            ClaimEdge(source="q", target="m", edge_type="ADDRESSES"),
            ClaimEdge(source="m", target="e", edge_type="PRODUCES"),
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
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
    plan: ResearchPlan | None = None,
) -> Study:
    plan = plan or ResearchPlan(
        plan_id=f"{study_id}-plan",
        claim_id="claim-1",
        year=2024,
        question="Smoking increases coronary heart disease risk.",
        method="Appraise committed clinical evidence before drawing the claim.",
        committed_pmids=list(pmids),
        claimed_scope=claimed_scope or EvidenceScope(year_start=2024, year_end=2024),
        rationale="Fixture plan.",
        parse_ok=True,
    )
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
        plan_id=plan.plan_id,
        research_plan=plan,
    )


def test_shadow_civer_brim_filters_same_natural_corpus_without_active_branch() -> None:
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
    assert report["endpoint_2_process_validation"]["passed"]["ungrounded_rate"] == 0.0
    assert report["endpoint_2_process_validation"]["failed"]["ungrounded_rate"] == 1.0
    # Honest-abstain plan (empty committed_pmids) admits at CIVER but the
    # study's non-NEUTRAL direction without evidence trips BRIM GC-03.
    assert report["endpoint_2_process_validation"]["process_counts"]["brim_failed"] == 1
    assert "claim-1" in report["all_guideline_latest"]
    assert "claim-1" in report["warranted_guideline_latest"]


def test_shadow_civer_brim_uses_process_trace_not_provenance_label() -> None:
    labelled_grounded_but_invalid_plan = _study(
        "invalid-plan",
        pmids=["1"],
        catalog_pmids=["1"],
        provenance="GROUNDED",
        plan=ResearchPlan(
            plan_id="invalid-plan-plan",
            claim_id="claim-1",
            year=2024,
            question="Smoking increases coronary heart disease risk.",
            method="",
            committed_pmids=["1"],
            claimed_scope=EvidenceScope(year_start=2024, year_end=2024),
            rationale="Missing method means invalid PIR.",
            parse_ok=False,
        ),
    )
    bundle = ArtifactBundle(
        input_text="",
        claim_graphs=[_graph()],
        snapshots={},
        branch_diff={},
        anchors=[],
        validation_notes=[],
        corpus_studies={"free": [labelled_grounded_but_invalid_plan]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    assert report["verdict_counts"]["passed"] == 0
    assert report["study_verdicts"][0]["true_provenance_for_calibration"] == "GROUNDED"
    assert report["study_verdicts"][0]["plan_recorded"] is True
    assert report["study_verdicts"][0]["civer_passed"] is False


def test_shadow_output_fallback_when_study_has_no_research_plan() -> None:
    """Legacy-bundle lane: a free-arm study without a ResearchPlan trace must
    not silently fail or be silently passed as CIVER. The shadow must (a) use
    the output-level scaffolding check, (b) tag the verdict with
    analysis_mode='output_fallback', (c) keep civer_passed=False so the
    fallback can't be mistaken for a process-CIVER claim, and (d) surface a
    fallback_warning at the top of the report.
    """
    # Build a Study with NO research_plan — mimics Run 4 free-arm merged path.
    study = _study("legacy", pmids=["1"], catalog_pmids=["1"])
    legacy = study.model_copy(update={"research_plan": None})
    bundle = ArtifactBundle(
        input_text="",
        claim_graphs=[_graph()],
        snapshots={},
        branch_diff={},
        anchors=[],
        validation_notes=[],
        corpus_studies={"free": [legacy]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    assert report["analysis_mode_breakdown"]["output_fallback"] == 1
    assert report["analysis_mode_breakdown"]["process"] == 0
    assert report["fallback_warning"] is not None
    assert "OUTPUT-FALLBACK" in report["fallback_warning"]

    verdict = report["study_verdicts"][0]
    assert verdict["analysis_mode"] == "output_fallback"
    assert verdict["civer_passed"] is False  # Never claim CIVER pass in fallback.
    assert verdict["brim_passed"] is False
    assert verdict["plan_recorded"] is False
    # Output check: cite "1" resolves in catalog ["1"]; scope is default; passes.
    assert verdict["output_check_passed"] is True
    assert verdict["passed"] is True


def test_shadow_report_has_per_claim_e2_and_volume_matched_e3_null() -> None:
    """E2/E3 rigor extensions for Paper 2 + Paper 3.

    The aggregate E2 hides claim-specific discrimination — per_claim breakdown
    must list every claim under shadow audit. E3 needs a volume-matched null
    distribution so reviewers can rule out smaller-pool luck as the mechanism
    behind warranted_to_truth < all_to_truth.
    """
    scoped = _study("good", pmids=["1"], catalog_pmids=["1"])
    no_cite = _study(
        "bad",
        pmids=[],
        catalog_pmids=[],
        provenance="UNGROUNDED",
        failure_mode="unresolvable",
    )
    no_cite2 = _study(
        "bad2",
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
        corpus_studies={"free": [scoped, no_cite, no_cite2]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    # Per-claim E2: claim-1 must appear with both passed + failed cohorts.
    per_claim = report["endpoint_2_per_claim"]
    assert "claim-1" in per_claim
    assert per_claim["claim-1"]["passed"]["count"] == 1
    assert per_claim["claim-1"]["failed"]["count"] == 2
    # The passed cohort cleaner on no_cite_rate than failed (signal direction).
    assert per_claim["claim-1"]["passed"]["no_cite_rate"] == 0.0
    assert per_claim["claim-1"]["failed"]["no_cite_rate"] == 1.0

    # New direction-vs-truth field present on both cohorts (may be 0 if no
    # truth point at this year, but the key must exist for downstream tooling).
    assert "wrong_direction_vs_truth_rate" in per_claim["claim-1"]["passed"]
    assert "wrong_direction_vs_truth_rate" in per_claim["claim-1"]["failed"]

    # E3 volume-matched null: 1 warranted of 3 -> null is a 1-of-3 subsample.
    e3 = report["endpoint_3_guideline_drift_reduction"]
    null = e3["volume_matched_null"]
    assert null is not None
    assert null["sample_size"] == 1
    assert 0.0 <= null["ci_low"] <= null["mean"] <= null["ci_high"]
    assert "civer_beats_volume_matched" in e3
    # Boolean is well-formed (not None / not a stray type).
    assert isinstance(e3["civer_beats_volume_matched"], bool)


def test_shadow_e3_keeps_zero_warrant_cells_in_denominator() -> None:
    """A warranted arm with no admitted study for a claim must not drop that
    claim from E3. It should synthesize the same no-evidence default on the same
    claim/year grid and surface the denominator audit.
    """
    claim1_pass = _study("claim1-good", pmids=["1"], catalog_pmids=["1"])
    claim2_fail = _study(
        "claim2-bad",
        pmids=[],
        catalog_pmids=[],
        provenance="UNGROUNDED",
        failure_mode="unresolvable",
    ).model_copy(update={"claim_id": "claim-2", "research_plan": None})
    bundle = ArtifactBundle(
        input_text="",
        claim_graphs=[_graph()],
        snapshots={},
        branch_diff={},
        anchors=[],
        validation_notes=[],
        corpus_studies={"free": [claim1_pass, claim2_fail]},
    )

    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path="data/ground_truth/cvd_multidirectional.json",
    )

    assert set(report["all_guideline_latest"]) == {"claim-1", "claim-2"}
    assert set(report["warranted_guideline_latest"]) == {"claim-1", "claim-2"}
    e3 = report["endpoint_3_guideline_drift_reduction"]
    assert e3["denominator_audit"]["cell_count"] == 2
    assert e3["denominator_audit"]["real_comparison_cells"] == 1
    # The claim-2 cell now correctly classifies as ABSTAIN (NA), not as a
    # zero-warrant cell that produced a NEUTRAL answer. Honest abstention is
    # distinct from "evidence balanced" and must be excluded from truth-match.
    assert e3["denominator_audit"]["abstain_cells"] == 1
    assert e3["denominator_audit"]["zero_warrant_cells"] == 0
    assert e3["real_comparison"]["cell_count"] == 1
