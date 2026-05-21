"""Slice C — C0 gold-standard reference, two-phase C0-displacement scoring, controls.

SPEC v3 §5 §6 §7. The C0 reference run is the *no-contamination counterfactual*
(failure_rate ~ 0): the trajectory the ecology converges to when the agents never
emit ungrounded/over-reaching studies. It is the GOLD STANDARD for CIVER value —
NOT real-world truth. CIVER value is scored as the difference in displacement from
C0 between the free (no gate) and constrained (CIVER-gated) branches.

This module reuses the v2-era harness primitives (`bootstrap_ci`, the
direction/level lattice, `_latest_by_claim`, `replay_counts`) rather than
duplicating them, and `synthesize_guideline_claim` to re-pool sub-sampled corpora
for the two §7b controls — so no second pass through `run_ecology` is needed.

The scoring machinery is deterministic, but the ecology itself may run either
scientifically (real model + real PubMed) or illustratively (fallback client /
deterministic PubMed). The USPSTF ground-truth trajectory is loaded from a
fixture file whose grades are PLACEHOLDERS marked UNVERIFIED — the scoring
*mechanism* is the deliverable; the real grades are verified by a human later.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from app.agents import DEFAULT_FAILURE_RATE
from app.harness import (
    BootstrapInterval,
    _DIRECTION_AXIS,
    _LEVEL_AXIS,
    _earliest_by_claim,
    _latest_by_claim,
    bootstrap_ci,
    replay_counts,
)
from app.llm import DeterministicFakeClient, LLMClient
from app.models import (
    ArtifactBundle,
    BranchName,
    GuidelineClaim,
    RunRequestModel,
    Study,
)
from app.pubmed import DeterministicPubMedClient, PubMedClient
from app.synthesis import synthesize_guideline_claim


# Difficulty-knob value carried into the C0 run for parity with the contaminated
# run's call signature. It is INERT (failure is emergent, not drawn), so C0's
# no-contamination property comes from GroundedOnlyClient below, not this number.
C0_FAILURE_RATE = 0.0


class GroundedOnlyClient:
    """The C0 no-emergent-contamination counterfactual (SPEC §5).

    Failure is now an EMERGENT property of the research agents' own emissions, so
    a rate knob can no longer guarantee a contamination-free reference run.
    Instead this adapter wraps the resolved client and, for the 4-line research/
    interpretation emissions, rewrites the SCOPE line to exactly the supplied
    source band (and ensures a source citation) — i.e. the model NEVER over-reaches
    or fabricates. SRMA and all other prompts pass through untouched. The harness
    authors no study; it only removes the emergent over-reach to obtain the strong-
    grounded-only trajectory the CIVER-value contrast is measured against.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @property
    def scientific(self) -> bool:
        return self._inner.scientific

    @property
    def degradation_reason(self) -> str | None:
        return self._inner.degradation_reason

    def generate(self, prompt: str, *, seed: int) -> str:
        raw = self._inner.generate(prompt, seed=seed)
        if "DIRECTION: SUPPORTS | REFUTES | NEUTRAL" not in prompt:
            return raw
        return _force_grounded_emission(raw, prompt)

    def describe(self):
        return self._inner.describe()


def _force_grounded_emission(raw: str, prompt: str) -> str:
    """Pin a research emission to a fully-grounded chain: cite the first supplied
    source at its exact source scope. Pure string rewrite — no network."""
    from app.agents import parse_research_emission
    from app.llm import _first_source_pmid, _first_source_scope

    emission = parse_research_emission(raw)
    pmid = _first_source_pmid(prompt)
    if not pmid:
        return raw  # nothing retrievable -> honest insufficiency stays as emitted
    low, high, ystart, yend = _first_source_scope(prompt)
    direction = emission.direction if emission.parse_ok else "NEUTRAL"
    return (
        f"DIRECTION: {direction}\n"
        f"SCOPE: pop={low}-{high} years={ystart}-{yend}\n"
        f"PMIDS: {pmid}\n"
        "RATIONALE: C0 reference appraisal grounded at source scope."
    )

DEFAULT_GROUND_TRUTH = (
    Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "hrt_uspstf.json"
)


# --------------------------------------------------------------------------- #
# Per-pair lattice deltas (the math `branch_gap` uses, exposed as raw lists so a
# *difference-of-displacements* CI can be computed per pair — not a difference of
# two independent CIs, which would be statistically wrong).
# --------------------------------------------------------------------------- #
def _direction_delta(left: GuidelineClaim, right: GuidelineClaim) -> float:
    return abs(_DIRECTION_AXIS[left.direction] - _DIRECTION_AXIS[right.direction]) / 2.0


def _level_delta(left: GuidelineClaim, right: GuidelineClaim) -> float:
    return abs(_LEVEL_AXIS[left.level] - _LEVEL_AXIS[right.level]) / 4.0


def displacement_deltas(
    branch: Sequence[GuidelineClaim],
    reference: Sequence[GuidelineClaim],
) -> tuple[list[str], list[float], list[float]]:
    """Per-claim displacement of ``branch`` from ``reference`` on both axes.

    Returns (claim_ids, direction_deltas, level_deltas) using each side's LATEST
    guideline per claim (matches `branch_gap`'s pairing). Reported separately per
    axis (SPEC §7b requires a CI on BOTH the direction and the level axis).
    """
    branch_latest = _latest_by_claim(branch)
    ref_latest = _latest_by_claim(reference)
    claim_ids = sorted(set(branch_latest) & set(ref_latest))
    direction = [_direction_delta(branch_latest[c], ref_latest[c]) for c in claim_ids]
    level = [_level_delta(branch_latest[c], ref_latest[c]) for c in claim_ids]
    return claim_ids, direction, level


@dataclass(frozen=True)
class AxisGap:
    """CIVER value on one axis: mean per-pair (d(free,C0) - d(constrained,C0))."""

    free_displacement: float
    constrained_displacement: float
    civer_value: BootstrapInterval  # CI over per-pair displacement differences

    @property
    def ci_excludes_zero(self) -> bool:
        return self.civer_value.low > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "free_displacement": self.free_displacement,
            "constrained_displacement": self.constrained_displacement,
            "civer_value": self.civer_value.to_dict(),
            "ci_excludes_zero": self.ci_excludes_zero,
        }


@dataclass(frozen=True)
class ControlOutcome:
    name: str
    direction_displacement: float
    level_displacement: float
    direction_beats_control: bool
    level_beats_control: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseBC0Report:
    pair_count: int
    direction: AxisGap
    level: AxisGap
    controls: list[ControlOutcome]
    pass_bars: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.pass_bars.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "direction": self.direction.to_dict(),
            "level": self.level.to_dict(),
            "controls": [c.to_dict() for c in self.controls],
            "pass_bars": self.pass_bars,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PhaseAC0Report:
    claim_count: int
    # stability
    seed_identical: bool
    self_drift_max: float
    no_self_drift: bool
    # ground-truth tracking vs nulls
    ground_truth_status: str
    final_error: float
    no_change_error: float
    random_null_error: float
    pass_bars: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.pass_bars.values())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


# --------------------------------------------------------------------------- #
# 1. C0 reference run
# --------------------------------------------------------------------------- #
def run_c0_reference(
    *,
    request: RunRequestModel,
    input_text: str,
    client: LLMClient | None = None,
    pubmed_client: PubMedClient | DeterministicPubMedClient | None = None,
) -> tuple[ArtifactBundle, dict[str, Any]]:
    """Run the ecology as the C0 no-emergent-contamination counterfactual.

    The GOLD STANDARD for CIVER value (SPEC §5): same ecology, same seed, but the
    research agents NEVER over-reach (GroundedOnlyClient pins every research
    emission to its source scope). NOT real-world truth. The supplied/resolved
    client is wrapped so the contamination-free property holds by construction
    rather than via an (now inert) failure-rate knob.
    """
    from app.simulator import resolve_client, simulate_run  # local import to avoid cycle

    resolved = resolve_client(request=request, client=client)
    return simulate_run(
        request=request,
        input_text=input_text,
        client=GroundedOnlyClient(resolved),
        pubmed_client=pubmed_client,
        failure_rate=C0_FAILURE_RATE,
    )


# --------------------------------------------------------------------------- #
# 3. Ground-truth fixture loader (placeholder-safe)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    topic: str
    status: str
    trajectory: dict[str, list[GuidelineClaim]]

    @property
    def is_verified(self) -> bool:
        return not self.status.upper().startswith("UNVERIFIED")

    def latest(self) -> dict[str, GuidelineClaim]:
        flat = [g for series in self.trajectory.values() for g in series]
        return _latest_by_claim(flat)

    def start(self) -> dict[str, GuidelineClaim]:
        flat = [g for series in self.trajectory.values() for g in series]
        return _earliest_by_claim(flat)


def load_ground_truth(path: str | Path | None = None) -> GroundTruth:
    """Load the configurable ground-truth trajectory from a fixture file.

    The fixture's grades are PLACEHOLDERS unless its ``_status`` says otherwise.
    This loader NEVER hardcodes real USPSTF grades; it reads whatever shape the
    fixture provides. ``_status`` starting with "UNVERIFIED" means scoring runs
    against placeholders (mechanism test only) and the report carries the banner.
    """
    fixture_path = Path(path) if path is not None else DEFAULT_GROUND_TRUTH
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    status = str(raw.get("_status", "UNVERIFIED — no _status field in fixture"))
    trajectory: dict[str, list[GuidelineClaim]] = {}
    for claim_id, points in raw.get("trajectory", {}).items():
        series = [
            GuidelineClaim(
                claim_id=claim_id,
                year=int(point["year"]),
                direction=point["direction"],
                level=point["level"],
            )
            for point in points
        ]
        trajectory[claim_id] = sorted(series, key=lambda g: g.year)
    return GroundTruth(
        topic=str(raw.get("topic", "")),
        status=status,
        trajectory=trajectory,
    )


# --------------------------------------------------------------------------- #
# 2. Phase B — CIVER value (the headline) + two controls
# --------------------------------------------------------------------------- #
def _axis_gap(
    *,
    free_deltas: list[float],
    constrained_deltas: list[float],
    iterations: int,
    seed: int,
) -> AxisGap:
    diffs = [f - c for f, c in zip(free_deltas, constrained_deltas)]
    return AxisGap(
        free_displacement=round(fmean(free_deltas), 4) if free_deltas else 0.0,
        constrained_displacement=round(fmean(constrained_deltas), 4)
        if constrained_deltas
        else 0.0,
        civer_value=bootstrap_ci(diffs, iterations=iterations, seed=seed),
    )


def evaluate_phase_b_c0(
    *,
    free: Sequence[GuidelineClaim],
    constrained: Sequence[GuidelineClaim],
    c0: Sequence[GuidelineClaim],
    controls: dict[str, Sequence[GuidelineClaim]] | None = None,
    iterations: int = 1000,
    seed: int = 0,
    control_margin: float = 0.0,
) -> PhaseBC0Report:
    """CIVER value = d(free,C0) − d(constrained,C0), bootstrap CI on both axes.

    The CI is over the per-pair *difference* of displacements (paired bootstrap),
    not the difference of two independent CIs. PASS = CI excludes 0 on BOTH the
    direction AND the level axis, AND CIVER beats every control on both axes.
    """
    claim_ids, free_dir, free_lvl = displacement_deltas(free, c0)
    _c_ids, con_dir, con_lvl = displacement_deltas(constrained, c0)

    direction = _axis_gap(
        free_deltas=free_dir, constrained_deltas=con_dir, iterations=iterations, seed=seed
    )
    level = _axis_gap(
        free_deltas=free_lvl, constrained_deltas=con_lvl, iterations=iterations, seed=seed + 1
    )

    control_outcomes: list[ControlOutcome] = []
    for name, control_guidelines in (controls or {}).items():
        _ids, ctrl_dir, ctrl_lvl = displacement_deltas(control_guidelines, c0)
        ctrl_dir_disp = round(fmean(ctrl_dir), 4) if ctrl_dir else 0.0
        ctrl_lvl_disp = round(fmean(ctrl_lvl), 4) if ctrl_lvl else 0.0
        control_outcomes.append(
            ControlOutcome(
                name=name,
                direction_displacement=ctrl_dir_disp,
                level_displacement=ctrl_lvl_disp,
                # CIVER wins iff the constrained branch sits closer to C0 than the
                # control trajectory does (control displacement strictly larger).
                direction_beats_control=ctrl_dir_disp
                > direction.constrained_displacement + control_margin,
                level_beats_control=ctrl_lvl_disp
                > level.constrained_displacement + control_margin,
            )
        )

    pass_bars = {
        "direction_ci_excludes_zero": direction.ci_excludes_zero,
        "level_ci_excludes_zero": level.ci_excludes_zero,
        "direction_beats_controls": all(c.direction_beats_control for c in control_outcomes)
        if control_outcomes
        else True,
        "level_beats_controls": all(c.level_beats_control for c in control_outcomes)
        if control_outcomes
        else True,
    }
    return PhaseBC0Report(
        pair_count=len(claim_ids),
        direction=direction,
        level=level,
        controls=control_outcomes,
        pass_bars=pass_bars,
    )


# --------------------------------------------------------------------------- #
# Controls: re-synthesize counter-trajectories from the FREE branch's studies.
# No second pass through run_ecology — the free branch already accumulated every
# study (it runs no gate), so we sub-sample its corpus and re-pool.
# --------------------------------------------------------------------------- #
def _studies_by_claim_year(studies: Sequence[Study]) -> dict[tuple[str, int], list[Study]]:
    grouped: dict[tuple[str, int], list[Study]] = {}
    for study in studies:
        grouped.setdefault((study.claim_id, study.year), []).append(study)
    return grouped


def _resynthesize(
    cells: dict[tuple[str, int], list[Study]],
) -> list[GuidelineClaim]:
    """Re-pool a (claim, year) -> studies map into a guideline trajectory.

    Mirrors the ecology's accumulating-DB semantics: each era pools every study
    for that claim up to and including that era.
    """
    claim_ids = sorted({claim_id for claim_id, _year in cells})
    years = sorted({year for _claim_id, year in cells})
    out: list[GuidelineClaim] = []
    for claim_id in claim_ids:
        for year in years:
            accumulated = [
                study
                for (c_id, c_year), studies in cells.items()
                if c_id == claim_id and c_year <= year
                for study in studies
            ]
            out.append(
                synthesize_guideline_claim(claim_id=claim_id, year=year, studies=accumulated)
            )
    return out


def volume_matched_control(
    *,
    free_studies: Sequence[Study],
    constrained_sizes: dict[tuple[str, int], int],
    seed: int = 0,
) -> list[GuidelineClaim]:
    """Down-sample the free corpus to the constrained corpus size, re-synthesize.

    Proves the gap is provenance-driven, not mere data-reduction (SPEC §7b). For
    each (claim, era) we keep |constrained| free studies at random (seeded).
    """
    rng = random.Random(seed)
    cells = _studies_by_claim_year(free_studies)
    matched: dict[tuple[str, int], list[Study]] = {}
    for key, studies in cells.items():
        keep = constrained_sizes.get(key, len(studies))
        ordered = sorted(studies, key=lambda s: s.id)
        if keep >= len(ordered):
            matched[key] = ordered
        else:
            idx = sorted(rng.sample(range(len(ordered)), keep))
            matched[key] = [ordered[i] for i in idx]
    return _resynthesize(matched)


def random_gate_control(
    *,
    free_studies: Sequence[Study],
    refusal_rates: dict[tuple[str, int], float],
    seed: int = 0,
) -> list[GuidelineClaim]:
    """Replace CIVER with a same-rate RANDOM refusal, re-synthesize.

    Drops the same fraction the real gate refused per (claim, era), but at random
    (provenance-blind). CIVER must beat this (SPEC §7b): proves the warrant logic,
    not mere rejection.
    """
    rng = random.Random(seed)
    cells = _studies_by_claim_year(free_studies)
    kept: dict[tuple[str, int], list[Study]] = {}
    for key, studies in cells.items():
        rate = refusal_rates.get(key, 0.0)
        ordered = sorted(studies, key=lambda s: s.id)
        survivors = [s for s in ordered if rng.random() >= rate]
        kept[key] = survivors
    return _resynthesize(kept)


# --------------------------------------------------------------------------- #
# 3. Phase A — faithfulness (stability + beats nulls vs ground truth)
# --------------------------------------------------------------------------- #
def _mean_pair_distance_to(
    branch: dict[str, GuidelineClaim],
    reference: dict[str, GuidelineClaim],
) -> float:
    claim_ids = sorted(set(branch) & set(reference))
    if not claim_ids:
        return 0.0
    return fmean(
        (_direction_delta(branch[c], reference[c]) + _level_delta(branch[c], reference[c])) / 2.0
        for c in claim_ids
    )


def _random_trajectory_latest(
    claim_ids: Sequence[str], *, seed: int
) -> dict[str, GuidelineClaim]:
    rng = random.Random(seed)
    directions = list(_DIRECTION_AXIS)
    levels = list(_LEVEL_AXIS)
    return {
        claim_id: GuidelineClaim(
            claim_id=claim_id,
            year=0,
            direction=rng.choice(directions),
            level=rng.choice(levels),
        )
        for claim_id in claim_ids
    }


def evaluate_phase_a_c0(
    *,
    c0_timeline: Sequence[GuidelineClaim],
    c0_rerun_timeline: Sequence[GuidelineClaim],
    ground_truth: GroundTruth,
    seed_identical: bool,
    self_drift_threshold: float = 0.25,
    random_seed: int = 0,
) -> PhaseAC0Report:
    """Phase A: stability (seed-identical + no self-drift) AND beats no-change null.

    ``seed_identical`` is the bit/score-identical verdict computed by the caller
    (comparing two C0 bundle seals). Self-drift = the max era-to-era lattice
    distance per claim within the single C0 (a noisy C0 makes the contrast
    meaningless). Ground-truth tracking compares C0's final (direction, level) to
    the configurable trajectory, relative to a no-change (echo START) null and a
    random null.
    """
    self_drift_max = _max_self_drift(c0_timeline)

    c0_final = _latest_by_claim(c0_timeline)
    c0_start = _earliest_by_claim(c0_timeline)
    gt_latest = ground_truth.latest()

    claim_ids = sorted(set(c0_final) & set(gt_latest))
    final_error = _mean_pair_distance_to(
        {c: c0_final[c] for c in claim_ids}, {c: gt_latest[c] for c in claim_ids}
    )
    # No-change null = echo START forward; error vs ground-truth latest.
    no_change_error = _mean_pair_distance_to(
        {c: c0_start[c] for c in claim_ids if c in c0_start},
        {c: gt_latest[c] for c in claim_ids if c in c0_start},
    )
    random_latest = _random_trajectory_latest(claim_ids, seed=random_seed)
    random_null_error = _mean_pair_distance_to(
        {c: random_latest[c] for c in claim_ids},
        {c: gt_latest[c] for c in claim_ids},
    )

    no_self_drift = self_drift_max <= self_drift_threshold
    # INTERNAL pass criteria: stability only (leakage-immune).
    # Ground-truth comparison is INFORMATIONAL — if leakage exists, C0 already
    # knows the right answer and beats no-change trivially, making that bar
    # meaningless as a quality gate. Report it separately, do not gate on it.
    pass_bars = {
        "seed_identical": seed_identical,
        "no_self_drift": no_self_drift,
    }
    return PhaseAC0Report(
        claim_count=len(claim_ids),
        seed_identical=seed_identical,
        self_drift_max=round(self_drift_max, 4),
        no_self_drift=no_self_drift,
        ground_truth_status=ground_truth.status,
        final_error=round(final_error, 4),
        no_change_error=round(no_change_error, 4),
        random_null_error=round(random_null_error, 4),
        pass_bars=pass_bars,
    )


def _max_self_drift(timeline: Sequence[GuidelineClaim]) -> float:
    grouped: dict[str, list[GuidelineClaim]] = {}
    for guideline in sorted(timeline, key=lambda g: (g.claim_id, g.year)):
        grouped.setdefault(guideline.claim_id, []).append(guideline)
    worst = 0.0
    for series in grouped.values():
        for prev, cur in zip(series, series[1:]):
            step = (_direction_delta(prev, cur) + _level_delta(prev, cur)) / 2.0
            worst = max(worst, step)
    return worst


# --------------------------------------------------------------------------- #
# 4. Top-level evaluation entrypoint
# --------------------------------------------------------------------------- #
def _refusal_and_sizes_from_bundle(
    contaminated_bundle: ArtifactBundle,
    *,
    free_studies: Sequence[Study],
    constrained_studies: Sequence[Study],
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    """Per-(claim, era) constrained corpus size + the gate's realized refusal rate.

    Refusal rate per cell = 1 - kept_constrained/seen_free (the free corpus is the
    full set the gate saw; constrained is what survived the warrant).
    """
    free_cells = _studies_by_claim_year(free_studies)
    con_cells = _studies_by_claim_year(constrained_studies)
    sizes: dict[tuple[str, int], int] = {}
    rates: dict[tuple[str, int], float] = {}
    for key, studies in free_cells.items():
        seen = len(studies)
        kept = len(con_cells.get(key, []))
        sizes[key] = kept
        rates[key] = (1.0 - kept / seen) if seen else 0.0
    return sizes, rates


def evaluate(
    *,
    request: RunRequestModel,
    input_text: str,
    failure_rate: float = DEFAULT_FAILURE_RATE,
    ground_truth_path: str | Path | None = None,
    client: LLMClient | None = None,
    pubmed_client: PubMedClient | DeterministicPubMedClient | None = None,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run C0 + a contaminated run; compute Phase A, Phase B, §6 replay counts.

    Returns a structured report with a SPEC §7 PASS/FAIL verdict. The
    contaminated run and its matching C0 share the SAME seed/agents — only the
    failure-fraction (and the gate, between branches) differ, which is what
    makes the C0-displacement test leakage- and agent-quality-proof.
    """
    from app.simulator import simulate_run  # local import to avoid cycle

    # --- C0 reference + a seed-identical rerun (stability) -----------------
    c0_bundle, c0_summary = run_c0_reference(
        request=request,
        input_text=input_text,
        client=client,
        pubmed_client=pubmed_client,
    )
    c0_rerun_bundle, c0_rerun_summary = run_c0_reference(
        request=request,
        input_text=input_text,
        client=client,
        pubmed_client=pubmed_client,
    )
    seed_identical = c0_bundle.bundle_seal == c0_rerun_bundle.bundle_seal

    c0_free = c0_bundle.guideline_timeline["free"]

    # --- contaminated run (free + constrained branches) --------------------
    study_sink: dict[str, list[Study]] = {}
    contaminated_bundle, contaminated_summary = simulate_run(
        request=request,
        input_text=input_text,
        client=client,
        pubmed_client=pubmed_client,
        failure_rate=failure_rate,
        study_sink=study_sink,
    )
    free_studies = study_sink.get("free", [])
    constrained_studies = study_sink.get("constrained", [])
    free = contaminated_bundle.guideline_timeline["free"]
    constrained = contaminated_bundle.guideline_timeline["constrained"]

    # --- controls (re-synthesized from the free corpus, no second ecology) -
    sizes, refusal_rates = _refusal_and_sizes_from_bundle(
        contaminated_bundle,
        free_studies=free_studies,
        constrained_studies=constrained_studies,
    )
    volume_matched = volume_matched_control(
        free_studies=free_studies, constrained_sizes=sizes, seed=seed
    )
    random_gate = random_gate_control(
        free_studies=free_studies, refusal_rates=refusal_rates, seed=seed
    )

    # --- Phase B (headline) ------------------------------------------------
    phase_b = evaluate_phase_b_c0(
        free=free,
        constrained=constrained,
        c0=c0_free,
        controls={"volume_matched": volume_matched, "random_gate": random_gate},
        iterations=iterations,
        seed=seed,
    )

    # --- Phase A (faithfulness) -------------------------------------------
    ground_truth = load_ground_truth(ground_truth_path)
    phase_a = evaluate_phase_a_c0(
        c0_timeline=c0_free,
        c0_rerun_timeline=c0_rerun_bundle.guideline_timeline["free"],
        ground_truth=ground_truth,
        seed_identical=seed_identical,
        random_seed=seed,
    )

    # --- §6 replay counts per era -----------------------------------------
    replay = _replay_per_era(contaminated_bundle)

    # --- External truth distances (INFORMATIONAL — not used in verdict) ----
    # Report how far free/constrained/C0 each land from external ground truth,
    # side-by-side with the internal C0-based CIVER metric. Never mix these into
    # the pass/fail verdict: if leakage exists, external distance trivially shrinks
    # (model already knows the answer), making it a useless quality gate.
    gt_latest = ground_truth.latest()
    external_truth = _external_truth_distances(
        free=free,
        constrained=constrained,
        c0=list(c0_free),
        gt_latest=gt_latest,
    )
    external_truth["ground_truth_status"] = ground_truth.status
    external_truth["ground_truth_verified"] = ground_truth.is_verified
    external_truth["note"] = (
        "Informational only — not used in pass/fail verdict. "
        "External distance shrinks trivially under leakage; use internal (C0) metric for verdict."
    )

    verdict_pass = phase_a.passed and phase_b.passed
    return {
        "phase_a": phase_a.to_dict(),
        "phase_b": phase_b.to_dict(),
        "external_truth": external_truth,
        "replay_counts": replay,
        "calibration_matrix": contaminated_summary.get("calibration_matrix"),
        "ground_truth_status": ground_truth.status,
        "ground_truth_verified": ground_truth.is_verified,
        "failure_rate": failure_rate,
        "c0_failure_rate": C0_FAILURE_RATE,
        "scientific": bool(
            c0_bundle.scientific and c0_rerun_bundle.scientific and contaminated_bundle.scientific
        ),
        "mode_banner": contaminated_bundle.mode_banner,
        "degradation_reason": contaminated_bundle.degradation_reason,
        "model_descriptor": contaminated_bundle.model_descriptor,
        "run_ops": {
            "c0_llm_call_count": c0_summary.get("llm_call_count"),
            "c0_rerun_llm_call_count": c0_rerun_summary.get("llm_call_count"),
            "contaminated_llm_call_count": contaminated_summary.get("llm_call_count"),
            "contaminated_llm_cache": contaminated_summary.get("llm_cache"),
            "contaminated_bundle_seal": contaminated_bundle.bundle_seal,
            "c0_bundle_seal": c0_bundle.bundle_seal,
            "c0_rerun_bundle_seal": c0_rerun_bundle.bundle_seal,
        },
        "verdict": "PASS" if verdict_pass else "FAIL",
    }


def _external_truth_distances(
    *,
    free: Sequence[GuidelineClaim],
    constrained: Sequence[GuidelineClaim],
    c0: Sequence[GuidelineClaim],
    gt_latest: dict[str, GuidelineClaim],
) -> dict[str, Any]:
    """Per-arm external truth distances (INFORMATIONAL — never used in verdict)."""
    free_latest = _latest_by_claim(free)
    con_latest = _latest_by_claim(constrained)
    c0_latest = _latest_by_claim(c0)
    claim_ids = sorted(set(free_latest) & set(gt_latest))
    if not claim_ids:
        return {"free": None, "constrained": None, "c0": None}

    def _dist(branch_latest: dict[str, GuidelineClaim]) -> float | None:
        ids = sorted(set(branch_latest) & set(gt_latest))
        if not ids:
            return None
        return round(
            fmean(
                (_direction_delta(branch_latest[c], gt_latest[c])
                 + _level_delta(branch_latest[c], gt_latest[c])) / 2.0
                for c in ids
            ),
            4,
        )

    per_claim: list[dict[str, Any]] = []
    for c in claim_ids:
        free_g = free_latest.get(c)
        con_g = con_latest.get(c)
        c0_g = c0_latest.get(c)
        gt_g = gt_latest[c]
        per_claim.append({
            "claim_id": c,
            "truth": {"direction": gt_g.direction, "level": gt_g.level},
            "free": {"direction": free_g.direction, "level": free_g.level} if free_g else None,
            "constrained": {"direction": con_g.direction, "level": con_g.level} if con_g else None,
            "c0": {"direction": c0_g.direction, "level": c0_g.level} if c0_g else None,
        })

    return {
        "free_to_truth": _dist(free_latest),
        "constrained_to_truth": _dist(con_latest),
        "c0_to_truth": _dist(c0_latest),
        "per_claim": per_claim,
    }


def _replay_per_era(bundle: ArtifactBundle) -> dict[str, Any]:
    """SPEC §6 replay counts per era from the bundle's db_growth + guidelines."""
    out: dict[str, Any] = {}
    for era, growth in bundle.db_growth.items():
        guidelines = {
            branch: {
                g.claim_id: {"direction": g.direction, "level": g.level}
                for g in bundle.guideline_timeline[branch]
                if g.year == int(era)
            }
            for branch in ("free", "constrained")
        }
        out[era] = {"db_growth": growth, "guideline": guidelines}
    return out
