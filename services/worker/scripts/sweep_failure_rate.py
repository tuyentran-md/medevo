"""Sensitivity sweep over the A0-anchor failure-rate (SPEC §11-A / §7c).

``failure_rate`` is a placeholder for A0's measured LLM error rate (κ pending,
not finalized). This sweep reports how the free-constrained divergence and the
gate error rates (FNR/FPR) respond as the rate varies over a range, rather than
committing to one hand-picked number.

Offline by default: uses DeterministicFakeClient + DeterministicPubMedClient, so
it makes no network calls. Run: ``python -m scripts.sweep_failure_rate``.
"""

from __future__ import annotations

import json

from app.ecology import sweep_failure_rate
from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.pubmed import DeterministicPubMedClient
from app.simulator import build_claim_graph
from app.ecology import extract_claims


SEPSIS = (
    "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
    "Broad-spectrum antibiotics should begin within one hour for septic shock. "
    "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
)

RATES = [0.1, 0.2, 0.3, 0.4, 0.5]


def main() -> None:
    request = RunRequestModel(
        title="failure-rate-sweep",
        input_mode="guideline",
        input_source="paste",
        input_text=SEPSIS,
        backend="ollama",
        horizons=list(range(1, 30)),
    )
    claims = extract_claims(SEPSIS, request.input_mode)
    claim_graphs = [build_claim_graph(claim) for claim in claims]

    rows = sweep_failure_rate(
        request=request,
        input_text=SEPSIS,
        claim_graphs=claim_graphs,
        llm=DeterministicFakeClient(),
        pubmed_client=DeterministicPubMedClient(),
        rates=RATES,
    )

    print(json.dumps(rows, indent=2))
    print("\nrate  divergence  FNR     FPR     grounded ungrounded")
    for row in rows:
        print(
            f"{row['failure_rate']:<5} "
            f"{row['free_minus_constrained_divergence']:<11} "
            f"{row['fnr']:<7} {row['fpr']:<7} "
            f"{row['grounded_total']:<8} {row['ungrounded_total']}"
        )


if __name__ == "__main__":
    main()
