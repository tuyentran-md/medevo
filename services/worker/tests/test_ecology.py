from __future__ import annotations

from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.pubmed import PubMedRecord, PubMedSearchResult
from app.simulator import simulate_run


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

    # v3: no harness-authored contamination -> Tier-1 makes no contaminator LLM
    # calls; the emergent-failure draw is deterministic (no model call).
    assert summary["llm_call_count"] == 0
    assert bundle.lineage
    assert bundle.guideline_timeline["free"]
    assert bundle.population_stats["30"]["pair_count"] == 3
    free_records = [record for record in bundle.lineage if record.branch == "free"]
    constrained_records = [record for record in bundle.lineage if record.branch == "constrained"]
    assert free_records
    assert constrained_records
    assert any(record.synthetic_carriers for record in free_records)
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
