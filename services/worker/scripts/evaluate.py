"""Slice C top-level evaluation entrypoint (SPEC v3 §6 §7).

Runs the C0 gold-standard reference + a contaminated run, computes Phase A
(faithfulness: stability + beats-no-change/random vs the configurable ground
truth), Phase B (CIVER value = d(free,C0) - d(constrained,C0), bootstrap CI on
both axes, plus the volume-matched + random-gate controls), and the §6 replay
counts, then prints a PASS/FAIL verdict.

Deterministic and offline: uses DeterministicFakeClient + DeterministicPubMedClient
(no network). The USPSTF ground-truth grades are loaded from a fixture whose values
are PLACEHOLDERS marked UNVERIFIED — the scoring mechanism is the deliverable, the
grades are verified by a human later.

Run: ``python -m scripts.evaluate``.
"""

from __future__ import annotations

import json

from app.c0 import evaluate
from app.models import RunRequestModel


SEPSIS = (
    "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
    "Broad-spectrum antibiotics should begin within one hour for septic shock. "
    "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
)


def main() -> None:
    request = RunRequestModel(
        title="medevo-slice-c-eval",
        input_mode="guideline",
        input_source="paste",
        input_text=SEPSIS,
        backend="ollama",
        horizons=list(range(1, 31)),
    )
    report = evaluate(
        request=request,
        input_text=SEPSIS,
        failure_rate=0.4,
        iterations=500,
        seed=7,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"ground-truth status: {report['ground_truth_status']}")


if __name__ == "__main__":
    main()
