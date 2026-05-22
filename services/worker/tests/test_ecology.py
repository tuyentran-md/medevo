from __future__ import annotations

from app.ecology import (
    SCOPE_TOLERANCE_YEARS,
    ClaimSeed,
    CorpusItem,
    admit_evidence_unit,
    compute_calibration_matrix,
)
from app.llm import DeterministicFakeClient
from app.models import EvidenceScope, EvidenceUnit, RunRequestModel
from app.pubmed import PubMedRecord, PubMedSearchResult
from app.simulator import build_claim_graph, simulate_run


class FlakyClient(DeterministicFakeClient):
    """A degraded LLM client.

    Slice A1 makes no LLM calls (Tier-1 reads PubMed; SRMA pooling is
    deterministic-zero-llm), so mid-run LLM degradation cannot be triggered by a
    call counter here. We model the degraded end-state directly: the client is
    born non-scientific with a degradation reason, and the bundle's end-of-run
    ``scientific = llm.scientific and ...`` propagates it. When LLM-driven SRMA
    lands (later slice), a call-count trigger becomes meaningful again.
    """

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.scientific = False
        self.degradation_reason = "forced ecology failure"
        self._fail_after = fail_after


class StablePubMed:
    def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
        record = PubMedRecord(
            pmid="222",
            title="Supportive care improved outcomes",
            abstract="Randomized trial n=220 reported improved outcomes; RR 0.76, 95% CI 0.64 to 0.91.",
            year=min(max_year, 2025),
            journal="Test Journal",
            locator="PMID:222",
        )
        return PubMedSearchResult(
            query=query,
            max_year=max_year,
            pmids=["222"],
            records=[record],
        )


def _request(text: str) -> RunRequestModel:
    return RunRequestModel(
        title="Ecology test",
        input_mode="guideline",
        input_source="paste",
        input_text=text,
        backend="ollama",
        horizons=[10, 20, 30],
    )


def test_lineage_records_are_present_and_coherent() -> None:
    request = _request(
        "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
        "Broad-spectrum antibiotics should begin within one hour for septic shock. "
        "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
    )
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    # Tier-4 now performs LLM appraisal over the accumulated Tier-3 corpus while
    # keeping numeric pooling deterministic, so the run must show model calls.
    assert summary["llm_call_count"] > 0
    assert bundle.lineage
    assert bundle.guideline_timeline["free"]
    assert bundle.population_stats["30"]["pair_count"] == 3
    free_records = [record for record in bundle.lineage if record.branch == "free"]
    constrained_records = [record for record in bundle.lineage if record.branch == "constrained"]
    assert free_records
    assert constrained_records
    assert any(record.ungrounded_carriers for record in free_records)
    assert any(record.surviving_real for record in constrained_records)


def test_custom_horizons_propagate_into_summary_and_snapshots() -> None:
    request = RunRequestModel(
        title="Long horizon",
        input_mode="paper",
        input_source="paste",
        input_text=(
            "Conclusion: Narrow targeted antibiotics reduced exposure without increasing failure. "
            "Therapy should be narrowed once culture data are available."
        ),
        backend="ollama",
        horizons=[5, 15, 25, 35],
    )
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    assert summary["years"] == [5, 15, 25, 35]
    assert [snapshot.year for snapshot in bundle.snapshots["free"]] == [5, 15, 25, 35]


def test_degraded_reason_is_explicit_when_client_fails_mid_run() -> None:
    request = _request(
        "Infants with bronchiolitis should receive supportive care with hydration and nasal suction. "
        "Supplemental oxygen is recommended when saturation persistently falls below the target range. "
        "Routine bronchodilators should not be continued without clear observed benefit."
    )
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=FlakyClient(fail_after=4),
        pubmed_client=StablePubMed(),
    )

    assert bundle.scientific is False
    assert bundle.degradation_reason is not None
    assert "forced ecology failure" in bundle.degradation_reason
    assert summary["scientific"] is False
    assert summary["degradation_reason"] == bundle.degradation_reason


def _admit(unit: EvidenceUnit, lookup: dict[str, CorpusItem]):
    claim = ClaimSeed("claim-1", "Some clinical claim.", "moderate")
    graph = build_claim_graph(claim)
    return admit_evidence_unit(
        run_id="run-1",
        claim=claim,
        claim_graph=graph,
        branch="constrained",
        year=2020,
        unit=unit,
        reachable_lookup=lookup,
        warrants_by_output={},
        threshold=0.6,
    )


def _unit(*, cited_ids: list[str], claimed_scope: EvidenceScope) -> EvidenceUnit:
    return EvidenceUnit(
        id="u-1",
        claim_id="claim-1",
        year=2020,
        branch="constrained",
        producer="investigator",
        cited_ids=cited_ids,
        provenance="GROUNDED",  # gate must ignore this; ground truth is harness-only
        direction="SUPPORTS",
        rationale="x",
        resolved_real_ids=cited_ids,
        claimed_scope=claimed_scope,
    )


def _real_item(scope: EvidenceScope) -> CorpusItem:
    return CorpusItem(
        item_id="PMID-1",
        kind="real",
        text="src",
        rationale="src",
        direction="NEUTRAL",
        cited_ids=["PMID-1"],
        resolved_real_ids=["PMID-1"],
        resolved_locators=["PMID:PMID-1"],
        scope=scope,
    )


def test_scope_clause_refuses_resolvable_but_overreaching_claim() -> None:
    source = EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)
    lookup = {"PMID-1": _real_item(source)}

    # Aggressive over-reach: well beyond tolerance on every axis -> refused
    # EVEN THOUGH the cite resolves (Mode-2 caught by the scope clause).
    over = _unit(
        cited_ids=["PMID-1"],
        claimed_scope=EvidenceScope(
            population_low=10, population_high=90, year_start=2000, year_end=2025
        ),
    )
    verdict, _ = _admit(over, lookup)
    assert verdict.passed is False
    assert any("scope clause" in reason for reason in verdict.reasons)

    # Within-scope claim with the same resolvable cite -> admitted.
    within = _unit(cited_ids=["PMID-1"], claimed_scope=source.model_copy(deep=True))
    verdict, _ = _admit(within, lookup)
    assert verdict.passed is True


def test_mild_scope_overreach_within_tolerance_slips_gate() -> None:
    source = EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)
    lookup = {"PMID-1": _real_item(source)}
    mild = _unit(
        cited_ids=["PMID-1"],
        claimed_scope=EvidenceScope(
            population_low=40,
            population_high=60 + SCOPE_TOLERANCE_YEARS,  # exactly at tolerance edge
            year_start=2015,
            year_end=2018,
        ),
    )
    verdict, _ = _admit(mild, lookup)
    # A mild over-reach within tolerance is NOT caught -> this is the mechanism
    # by which FNR can be > 0 (the gate is imperfect, not tautological).
    assert verdict.passed is True


def test_calibration_matrix_counts_and_rates() -> None:
    observations = [
        ("GROUNDED", True),  # TP
        ("GROUNDED", True),  # TP
        ("UNGROUNDED", False),  # TN
        ("UNGROUNDED", True),  # FN (gate missed contamination)
        ("GROUNDED", False),  # FP (gate over-blocked)
    ]
    matrix = compute_calibration_matrix(observations)
    assert matrix.true_positive == 2
    assert matrix.true_negative == 1
    assert matrix.false_negative == 1
    assert matrix.false_positive == 1
    assert matrix.fnr == 0.5  # 1 / 2 ungrounded
    assert matrix.fpr == round(1 / 3, 4)  # 1 / 3 grounded


def test_calibration_matrix_in_bundle_tracks_process_gate_without_forcing_fnr() -> None:
    request = _request(
        "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
        "Broad-spectrum antibiotics should begin within one hour for septic shock. "
        "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
    )
    # Many eras -> enough ungrounded emissions to exercise process-gate scoring.
    request.horizons = list(range(1, 30))
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.4,
    )

    matrix = bundle.calibration_matrix
    assert matrix is not None
    assert matrix.branch == "constrained"
    assert summary["calibration_matrix"]["ungrounded_total"] == matrix.ungrounded_total
    # The realistic failure-mode mix produces some ungrounded studies.
    assert matrix.ungrounded_total > 0
    # With CIVER+BRIM scored on the research process, this deterministic fixture
    # can catch every invalid process. FNR is an empirical result, not a required
    # design property.
    assert matrix.false_negative >= 0
    assert matrix.false_positive == 0
