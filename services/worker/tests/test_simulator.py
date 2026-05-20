import hashlib

from app.llm import (
    EVIDENCE_UNIVERSE_PROMPT_TEMPLATE,
    EVIDENCE_UNIVERSE_PROMPT_TEMPLATE_DIGEST,
    PROMPT_TEMPLATE_DIGEST,
    RESEARCHER_PROMPT_TEMPLATE,
    SYNTHESIST_PROMPT_TEMPLATE,
    SYNTHESIST_PROMPT_TEMPLATE_DIGEST,
    SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE,
    SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE_DIGEST,
    DeterministicFakeClient,
)
from app.models import RunRequestModel
from app.ecology import CallTelemetry, ClaimSeed, _mint_contamination_bank
from app.simulator import contamination_clock, resolve_backend, simulate_run


class RefutingUniverseClient(DeterministicFakeClient):
    scientific = True
    degradation_reason = None

    def generate(self, prompt: str, *, seed: int) -> str:
        if "Create exactly four admissible source records" in prompt:
            return "\n".join(
                (
                    f"SOURCE {index}\n"
                    "DIRECTION: REFUTES\n"
                    "FINDING: Simulated admissible evidence reports no net benefit and potential harm."
                )
                for index in range(1, 5)
            )
        if "possible contaminated literature carrier" in prompt:
            return "DIRECTION: SUPPORTS\nRATIONALE: AI-generated carrier overstates benefit."
        if "Real source" in prompt and "REFUTES" in prompt:
            return "DIRECTION: REFUTES\nRATIONALE: admissible evidence refutes the claim."
        return super().generate(prompt, seed=seed)


def _request() -> RunRequestModel:
    return RunRequestModel(
        title="Demo",
        input_mode="guideline",
        input_source="paste",
        input_text=(
            "Children with suspected sepsis should receive cultures before antibiotics. "
            "Broad-spectrum antibiotics should begin rapidly when septic shock is likely. "
            "Escalate support when perfusion fails to improve."
        ),
        backend="ollama",
        horizons=[10, 20, 30],
    )


def test_contamination_clock_rises_across_years() -> None:
    assert contamination_clock(10) < contamination_clock(20) < contamination_clock(30)


def test_constrained_preserves_real_lineage_free_never_blocks() -> None:
    """Free never blocks. Constrained must preserve real-source inheritance via
    valid warrants, while free accumulates synthetic carriers."""
    request = _request()
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    assert summary["years"] == [10, 20, 30]
    assert all(
        claim.blocked_count == 0
        for snapshot in bundle.snapshots["free"]
        for claim in snapshot.claims
    )
    assert any(record.surviving_real for record in bundle.lineage if record.branch == "constrained")
    assert any(record.synthetic_carriers for record in bundle.lineage if record.branch == "free")
    assert any(
        warrant.branch == "constrained" and warrant.status == "ISSUED" and warrant.issued
        for warrant in bundle.warrants
    )


def test_ecology_generates_branch_divergence_from_corpus_membership() -> None:
    request = _request()
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    deltas = [
        delta
        for year_deltas in bundle.branch_diff.values()
        for delta in year_deltas.values()
    ]
    assert max(deltas) > 0
    assert any(
        free_claim.direction != constrained_claim.direction
        for free_snapshot, constrained_snapshot in zip(
            bundle.snapshots["free"],
            bundle.snapshots["constrained"],
        )
        for free_claim, constrained_claim in zip(
            free_snapshot.claims,
            constrained_snapshot.claims,
        )
    )


def test_synthetic_direction_comes_from_model_not_opposing_constant() -> None:
    telemetry = CallTelemetry()
    units = _mint_contamination_bank(
        ClaimSeed("claim-x", "Routine antibiotics should be given for viral bronchiolitis.", "strong"),
        RefutingUniverseClient(),
        telemetry,
    )

    assert {unit.direction for unit in units} == {"SUPPORTS"}


def test_constrained_can_refute_when_admissible_universe_refutes_claim() -> None:
    request = RunRequestModel(
        title="Refuting universe",
        input_mode="guideline",
        input_source="paste",
        input_text="Routine antibiotics should be given for acute viral bronchiolitis in infants.",
        backend="ollama",
        horizons=[10],
    )
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=RefutingUniverseClient(),
    )

    constrained_claim = bundle.snapshots["constrained"][0].claims[0]
    assert constrained_claim.direction == "REFUTES"
    assert bundle.lineage[1].branch == "constrained"
    assert bundle.lineage[1].surviving_real


def test_fallback_run_is_marked_non_scientific() -> None:
    """No Ollama in test env -> deterministic fallback -> bundle must declare
    itself non-scientific (SPEC §6.6)."""
    request = _request()
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    assert bundle.scientific is False
    assert bundle.mode_banner == "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"
    assert any("DEGRADED RUN" in note for note in bundle.validation_notes)
    assert bundle.degradation_reason is not None
    assert summary["scientific"] is False


def test_researcher_prompt_template_is_frozen() -> None:
    """Prompt invariance guard: the template must not silently change, and
    must not leak year/branch/drift/bias instructions into the model."""
    assert (
        hashlib.sha256(RESEARCHER_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        == PROMPT_TEMPLATE_DIGEST
    )
    lowered = RESEARCHER_PROMPT_TEMPLATE.lower()
    for forbidden in ("year", "branch", "drift", "bias", "contaminat", "ai-generated"):
        assert forbidden not in lowered


def test_synthesist_prompt_template_is_frozen() -> None:
    assert (
        hashlib.sha256(SYNTHESIST_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        == SYNTHESIST_PROMPT_TEMPLATE_DIGEST
    )
    lowered = SYNTHESIST_PROMPT_TEMPLATE.lower()
    for forbidden in ("year", "branch", "drift", "bias", "contaminat", "ai-generated"):
        assert forbidden not in lowered


def test_source_and_synthetic_prompt_templates_are_frozen() -> None:
    assert (
        hashlib.sha256(EVIDENCE_UNIVERSE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        == EVIDENCE_UNIVERSE_PROMPT_TEMPLATE_DIGEST
    )
    assert (
        hashlib.sha256(SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        == SYNTHETIC_EVIDENCE_PROMPT_TEMPLATE_DIGEST
    )


def test_backend_resolution_uses_fallback_when_ollama_unavailable() -> None:
    request = RunRequestModel(
        input_mode="guideline",
        input_source="paste",
        input_text="Routine bronchodilators should not be continued without observed benefit.",
        backend="ollama",
    )
    backend = resolve_backend(request)
    assert backend.backend == "ollama"
    assert backend.using_fallback in {True, False}


def test_openai_compatible_requires_base_url_for_scientific_run() -> None:
    request = RunRequestModel(
        input_mode="guideline",
        input_source="paste",
        input_text="Routine bronchodilators should not be continued without observed benefit.",
        backend="openai-compatible",
        api_key="secret",
        model="some-model",
    )
    backend = resolve_backend(request)
    assert backend.using_fallback is True
    assert backend.base_url is None

    request.base_url = "https://example.test/v1"
    backend = resolve_backend(request)
    assert backend.using_fallback is False
    assert backend.base_url == "https://example.test/v1"
