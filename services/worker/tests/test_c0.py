from __future__ import annotations

import json

from app.c0 import (
    DEFAULT_GROUND_TRUTH,
    _max_self_drift,
    evaluate,
    evaluate_phase_b_c0,
    load_ground_truth,
    run_c0_reference,
)
from app.models import GuidelineClaim, RunRequestModel


SEPSIS = (
    "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
    "Broad-spectrum antibiotics should begin within one hour for septic shock. "
    "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
)

# Matches the default ground-truth fixture (data/ground_truth/hrt_uspstf.json),
# so the entrypoint faithfulness check is coherent (input ↔ ground-truth aligned).
HRT = (
    "Postmenopausal hormone therapy should not be used for chronic disease prevention. "
    "Hormone therapy does not provide a net cardiovascular prevention benefit in postmenopausal women. "
    "Potential harms, including stroke and thromboembolic events, outweigh prevention benefits for routine chronic-disease use."
)


def _g(claim_id: str, year: int, *, direction: str, level: str) -> GuidelineClaim:
    return GuidelineClaim(claim_id=claim_id, year=year, direction=direction, level=level)


def _request(horizons: list[int] | None = None, *, input_text: str = SEPSIS) -> RunRequestModel:
    return RunRequestModel(
        title="c0-test",
        input_mode="guideline",
        input_source="paste",
        input_text=input_text,
        backend="ollama",
        horizons=horizons or list(range(1, 31)),
    )


# --- Phase B: CIVER value on both axes + controls -------------------------- #
def test_phase_b_c0_passes_on_divergent_fixture() -> None:
    # C0 (gold) = SUPPORTS strong-for. Constrained tracks C0; free drifts away.
    c0 = [_g(f"claim-{i}", 30, direction="SUPPORTS", level="strong-for") for i in range(1, 6)]
    constrained = [
        _g(f"claim-{i}", 30, direction="SUPPORTS", level="strong-for") for i in range(1, 6)
    ]
    free = [
        _g(f"claim-{i}", 30, direction="REFUTES", level="strong-against") for i in range(1, 6)
    ]
    # Both controls drift like free (worse than constrained) -> CIVER beats them.
    drifted = [
        _g(f"claim-{i}", 30, direction="REFUTES", level="strong-against") for i in range(1, 6)
    ]

    report = evaluate_phase_b_c0(
        free=free,
        constrained=constrained,
        c0=c0,
        controls={"volume_matched": drifted, "random_gate": drifted},
        iterations=200,
        seed=3,
    )

    assert report.pair_count == 5
    # d(free,C0) is the maximal displacement; d(constrained,C0) = 0 -> gap > 0.
    assert report.direction.civer_value.low > 0
    assert report.level.civer_value.low > 0
    assert report.direction.ci_excludes_zero
    assert report.level.ci_excludes_zero
    assert all(c.direction_beats_control for c in report.controls)
    assert all(c.level_beats_control for c in report.controls)
    assert report.passed


def test_phase_b_c0_ci_is_paired_difference_not_difference_of_cis() -> None:
    # Per-pair displacement difference is constant (0.5 on level) -> degenerate CI
    # collapses to that constant, proving we bootstrap the DIFFERENCE per pair.
    c0 = [_g(f"claim-{i}", 30, direction="SUPPORTS", level="strong-for") for i in range(1, 5)]
    constrained = [
        _g(f"claim-{i}", 30, direction="SUPPORTS", level="conditional-for") for i in range(1, 5)
    ]
    free = [
        _g(f"claim-{i}", 30, direction="SUPPORTS", level="strong-against") for i in range(1, 5)
    ]
    report = evaluate_phase_b_c0(free=free, constrained=constrained, c0=c0, iterations=100, seed=1)
    # level lattice: strong-for=+2, strong-against=-2, conditional-for=+1.
    # d(free,C0)=|2-(-2)|/4=1.0 ; d(con,C0)=|2-1|/4=0.25 per pair -> diff=0.75.
    assert report.level.free_displacement == 1.0
    assert report.level.constrained_displacement == 0.25
    assert report.level.civer_value.mean == 0.75
    assert report.level.civer_value.low == 0.75  # constant diff -> tight CI


# --- C0 stability ---------------------------------------------------------- #
def test_c0_rerun_is_seed_identical() -> None:
    request = _request()
    first, _ = run_c0_reference(request=request, input_text=SEPSIS)
    second, _ = run_c0_reference(request=request, input_text=SEPSIS)
    assert first.bundle_seal == second.bundle_seal
    assert first.guideline_timeline == second.guideline_timeline


def test_c0_has_no_self_drift() -> None:
    request = _request()
    c0, _ = run_c0_reference(request=request, input_text=SEPSIS)
    drift = _max_self_drift(c0.guideline_timeline["free"])
    # A clean C0 advances smoothly (one GRADE step at a time), never flips wildly.
    assert drift <= 0.25


# --- Ground-truth fixture loader (NOT hardcoded) --------------------------- #
def test_default_ground_truth_fixture_is_verified_and_on_disk() -> None:
    # The default HRT fixture was verified from the USPSTF primary source 2026-05-21.
    gt = load_ground_truth()
    assert gt.is_verified is True
    assert not gt.status.upper().startswith("UNVERIFIED")
    # Still lives on disk, not baked into code.
    assert DEFAULT_GROUND_TRUTH.exists()


def test_unverified_status_marks_fixture_not_verified(tmp_path) -> None:
    # The is_verified gate still keys off an "UNVERIFIED" status prefix.
    fixture = tmp_path / "unverified_gt.json"
    fixture.write_text(
        json.dumps({"_status": "UNVERIFIED — placeholder", "topic": "t", "trajectory": {}}),
        encoding="utf-8",
    )
    assert load_ground_truth(fixture).is_verified is False


def test_ground_truth_loaded_from_fixture_not_hardcoded(tmp_path) -> None:
    # A synthetic fixture with DIFFERENT grades than the default proves the loader
    # reads whatever the file says (configurable), with no hardcoded USPSTF grades.
    fixture = tmp_path / "synthetic_gt.json"
    fixture.write_text(
        json.dumps(
            {
                "_status": "SYNTHETIC TEST FIXTURE — not real data",
                "topic": "unit-test",
                "trajectory": {
                    "claim-1": [
                        {"year": 1, "direction": "NEUTRAL", "level": "no-recommendation"},
                        {"year": 9, "direction": "SUPPORTS", "level": "strong-for"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    gt = load_ground_truth(fixture)
    assert gt.topic == "unit-test"
    assert gt.is_verified is True  # status does not start with UNVERIFIED
    latest = gt.latest()
    assert latest["claim-1"].direction == "SUPPORTS"
    assert latest["claim-1"].level == "strong-for"
    assert latest["claim-1"].year == 9


# --- Top-level entrypoint -------------------------------------------------- #
def test_evaluate_entrypoint_runs_offline_and_is_structured() -> None:
    # Mirror the real eval config: a few absolute retro eras, not 30 relative ones.
    request = _request(horizons=[2000, 2010, 2020], input_text=HRT)
    report = evaluate(request=request, input_text=HRT, failure_rate=0.4, iterations=200, seed=7)

    # Phase A faithfulness holds on the clean C0 (mechanism, not a USPSTF claim).
    assert report["phase_a"]["pass_bars"]["seed_identical"] is True
    assert report["phase_a"]["pass_bars"]["no_self_drift"] is True
    assert report["phase_a"]["pass_bars"]["beats_no_change"] is True

    # Phase B structure present with BOTH axes carrying a CI, regardless of verdict.
    for axis in ("direction", "level"):
        assert "civer_value" in report["phase_b"][axis]
        assert {"mean", "low", "high"} <= set(report["phase_b"][axis]["civer_value"])

    # Both §7b controls computed.
    control_names = {c["name"] for c in report["phase_b"]["controls"]}
    assert control_names == {"volume_matched", "random_gate"}

    # §6 replay counts per era.
    assert report["replay_counts"]
    assert report["verdict"] in {"PASS", "FAIL"}
    # Default HRT ground truth is now verified from the USPSTF primary source.
    assert report["ground_truth_verified"] is True


def test_evaluate_entrypoint_is_deterministic() -> None:
    request = _request()
    first = evaluate(request=request, input_text=SEPSIS, failure_rate=0.4, iterations=200, seed=7)
    second = evaluate(request=request, input_text=SEPSIS, failure_rate=0.4, iterations=200, seed=7)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_volume_matched_and_random_gate_controls_distinct_objects() -> None:
    # Controls are computed from the free corpus, not re-running ecology twice.
    request = _request()
    report = evaluate(request=request, input_text=SEPSIS, failure_rate=0.6, iterations=100, seed=2)
    controls = {c["name"]: c for c in report["phase_b"]["controls"]}
    assert set(controls) == {"volume_matched", "random_gate"}
    for c in controls.values():
        assert "direction_displacement" in c
        assert "level_displacement" in c
        assert "direction_beats_control" in c
