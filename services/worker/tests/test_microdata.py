from __future__ import annotations

import re
from pathlib import Path

from app.llm import ModelDescriptor
from app.microdata import MicrodataAgent, supports_claim


class _SliceCitingLLM:
    """Fake that interprets the NHANES result honestly: cites the slice id from
    the prompt and reports the analyzed cohort scope unchanged (GROUNDED)."""

    scientific = True
    degradation_reason = None

    def generate(self, prompt: str, *, seed: int) -> str:
        slice_match = re.search(r"PMIDS:\s*(NHANES:[^\s]+)", prompt)
        pmid = slice_match.group(1) if slice_match else "none"
        pop = re.search(r"pop=(\d+)-(\d+)", prompt)
        band = f"pop={pop.group(1)}-{pop.group(2)}" if pop else "pop=45-79"
        return (
            "DIRECTION: SUPPORTS\n"
            f"SCOPE: {band} years=2005-2006\n"
            f"PMIDS: {pmid}\n"
            "RATIONALE: standardized RR>1 indicates elevated cardiometabolic burden."
        )

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="slice-citing-fake", digest="test")


def test_supports_claim_detects_hrt_keywords() -> None:
    assert supports_claim(
        "Postmenopausal hormone therapy should not be used for cardiovascular disease prevention."
    )
    assert not supports_claim(
        "Children with suspected sepsis should receive cultures before antibiotics when feasible."
    )


def test_microdata_agent_short_circuits_unsupported_claim_without_fetch() -> None:
    agent = MicrodataAgent(
        file_provider=lambda: (_ for _ in ()).throw(RuntimeError("should not fetch")),
        llm=_SliceCitingLLM(),
    )
    study, catalog = agent.run(
        claim_id="claim-1",
        claim_text="Children with suspected sepsis should receive cultures before antibiotics.",
        simulated_year=10,
    )

    # The claim is out of the NHANES slice's scope, so the analysis is unsupported
    # and the agent emits an UNGROUNDED study (no resolvable dataset slice).
    assert study.provenance == "UNGROUNDED"
    assert study.failure_mode == "unresolvable"
    assert catalog == []


def test_microdata_agent_emits_grounded_nhanes_slice_from_analysis_runner() -> None:
    agent = MicrodataAgent(
        file_provider=lambda: {"demo": Path("/tmp/demo.xpt")},
        analysis_runner=lambda files: {
            "supported": True,
            "rr": 1.18,
            "ci_low": 1.04,
            "ci_high": 1.31,
            "n_total": 412,
            "age_low": 45,
            "age_high": 79,
            "summary": "NHANES slice summary.",
        },
        llm=_SliceCitingLLM(),
    )

    study, catalog = agent.run(
        claim_id="claim-hrt",
        claim_text="Postmenopausal hormone therapy should not be used for cardiovascular disease prevention.",
        simulated_year=20,
    )

    assert study.provenance == "GROUNDED"
    assert study.failure_mode == "none"
    assert study.effect_point == 1.18
    assert study.pmids == ["NHANES:2005-2006:HRT-CARDIOMETABOLIC:claim-hrt"]
    assert study.id == "claim-hrt-study-20-nhanes-hrt-cardiometabolic"
    assert catalog[0].pmid == "NHANES:2005-2006:HRT-CARDIOMETABOLIC:claim-hrt"
