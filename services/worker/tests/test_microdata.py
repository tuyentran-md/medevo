from __future__ import annotations

from pathlib import Path

from app.microdata import MicrodataAgent, supports_claim


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
    )
    study, catalog = agent.run(
        claim_id="claim-1",
        claim_text="Children with suspected sepsis should receive cultures before antibiotics.",
        simulated_year=10,
    )

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
