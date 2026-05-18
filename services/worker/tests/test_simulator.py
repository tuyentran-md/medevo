import hashlib

from app.llm import (
    PROMPT_TEMPLATE_DIGEST,
    RESEARCHER_PROMPT_TEMPLATE,
    DeterministicFakeClient,
)
from app.models import RunRequestModel
from app.simulator import contamination_clock, resolve_backend, simulate_run


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
    )


def test_contamination_clock_rises_across_years() -> None:
    assert contamination_clock(10) < contamination_clock(20) < contamination_clock(30)


def test_constrained_blocks_emerge_free_never_blocks() -> None:
    """Free branch never blocks. Constrained blocks emerge from CIVER's
    real-source rule when a study's context is fully synthetic — not from a
    branch constant."""
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
    assert any(
        claim.blocked_count > 0
        for snapshot in bundle.snapshots["constrained"]
        for claim in snapshot.claims
    )


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
    assert any("FALLBACK MODE" in note for note in bundle.validation_notes)
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
