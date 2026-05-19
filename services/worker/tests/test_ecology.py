from __future__ import annotations

from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.simulator import simulate_run


class FlakyClient(DeterministicFakeClient):
    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.scientific = True
        self.degradation_reason = None
        self._fail_after = fail_after
        self._calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        self._calls += 1
        if self._calls > self._fail_after:
            self.scientific = False
            self.degradation_reason = "forced ecology failure"
            return super().generate(prompt, seed=seed)
        return super().generate(prompt, seed=seed)


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

    assert summary["llm_call_count"] == 48
    assert bundle.lineage
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
    )

    assert bundle.scientific is False
    assert bundle.degradation_reason is not None
    assert "forced ecology failure" in bundle.degradation_reason
    assert summary["scientific"] is False
    assert summary["degradation_reason"] == bundle.degradation_reason
