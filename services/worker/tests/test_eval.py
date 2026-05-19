from __future__ import annotations

from pathlib import Path

from app.ecology import verify_audit_chain
from app.eval.eval_runner import evaluate_calibration, verify_bundle_seal
from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.simulator import simulate_run


def _request() -> RunRequestModel:
    return RunRequestModel(
        title="Eval Harness",
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


def test_bundle_seal_verifies_and_tamper_fails() -> None:
    request = _request()
    bundle, _ = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )
    payload = bundle.model_dump(mode="json")
    assert verify_bundle_seal(payload) is True

    payload["lineage"][0]["surviving_real"] = []
    assert verify_bundle_seal(payload) is False


def test_audit_chain_verifies_and_mutation_breaks_it() -> None:
    request = _request()
    bundle, _ = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )
    assert verify_audit_chain(bundle.audit_trail) is True

    tampered = [event.model_copy(deep=True) for event in bundle.audit_trail]
    tampered[0].message = "mutated"
    assert verify_audit_chain(tampered) is False


def test_generation_runner_does_not_import_eval_package() -> None:
    ecology_source = Path("/Users/admin/repos/medevo/services/worker/app/ecology.py").read_text(
        encoding="utf-8"
    )
    assert "app.eval" not in ecology_source


def test_calibration_harness_reports_confusion_metrics() -> None:
    request = _request()
    bundle, _ = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )
    payload = bundle.model_dump(mode="json")
    gold_set = [
        {
            "claim_id": record.claim_id,
            "year": record.year,
            "branch": record.branch,
            "expected_survives": bool(record.surviving_real),
        }
        for record in bundle.lineage[:4]
    ]
    report = evaluate_calibration(bundle=payload, gold_set=gold_set)

    assert report["n"] == 4
    assert report["matrix"]["fp"] == 0
    assert report["matrix"]["fn"] == 0
    assert report["fnr"] == 0.0
    assert report["fpr"] == 0.0
