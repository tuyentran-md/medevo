"""Audit MedEvo E3 denominators without making model calls.

This script reads frozen `natural_bundle.json` artifacts and reports the metric
that Run 5 needed: aggregate E3 over all cells, plus the real-comparison subset
where the constrained branch has at least one admitted study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.c0 import GroundTruth, load_ground_truth
from app.models import ArtifactBundle, GuidelineClaim


_DIRECTION_AXIS = {"REFUTES": -1.0, "NEUTRAL": 0.0, "SUPPORTS": 1.0}
_LEVEL_AXIS = {
    "strong-against": -2.0,
    "conditional-against": -1.0,
    "no-recommendation": 0.0,
    "conditional-for": 1.0,
    "strong-for": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit E3 real-comparison cells.")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/battery_run4_nc.json"),
    )
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def _resolve_bundle_path(path: Path) -> Path:
    if path.is_dir():
        return path / "natural_bundle.json"
    return path


def _artifact_label(path: Path, bundle: ArtifactBundle) -> str:
    descriptor = bundle.model_descriptor
    model = descriptor.get("name", "") if isinstance(descriptor, dict) else descriptor.name
    parent = path.parent.name if path.name == "natural_bundle.json" else path.stem
    return f"{model} ({parent})"


def _pair_distance(left: GuidelineClaim, right: GuidelineClaim) -> float:
    direction = abs(_DIRECTION_AXIS[left.direction] - _DIRECTION_AXIS[right.direction]) / 2.0
    level = abs(_LEVEL_AXIS[left.level] - _LEVEL_AXIS[right.level]) / 4.0
    return (direction + level) / 2.0


def _by_cell(guidelines: list[GuidelineClaim]) -> dict[tuple[str, int], GuidelineClaim]:
    return {(guideline.claim_id, guideline.year): guideline for guideline in guidelines}


def _truth_by_cell(truth: GroundTruth) -> dict[tuple[str, int], GuidelineClaim]:
    return {
        (claim_id, guideline.year): guideline
        for claim_id, series in truth.trajectory.items()
        for guideline in series
    }


def _mean_distance(
    guidelines: dict[tuple[str, int], GuidelineClaim],
    truth: dict[tuple[str, int], GuidelineClaim],
    cells: list[tuple[str, int]],
) -> float | None:
    if not cells:
        return None
    return round(fmean(_pair_distance(guidelines[cell], truth[cell]) for cell in cells), 4)


def _shadow_denominator_audit(bundle_path: Path) -> dict[str, Any] | None:
    report_path = bundle_path.with_name("shadow_report.json")
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    all_latest = set(report.get("all_guideline_latest", {}))
    warranted_latest = set(report.get("warranted_guideline_latest", {}))
    e3 = report.get("endpoint_3_guideline_drift_reduction", {})
    return {
        "legacy_all_latest_claims": len(all_latest),
        "legacy_warranted_latest_claims": len(warranted_latest),
        "legacy_missing_warranted_latest_claims": sorted(all_latest - warranted_latest),
        "legacy_delta": e3.get("delta"),
        "legacy_civer_beats_volume_matched": e3.get("civer_beats_volume_matched"),
    }


def analyze_artifact(path: Path, truth: GroundTruth) -> dict[str, Any]:
    bundle_path = _resolve_bundle_path(path)
    bundle = ArtifactBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    truth_cells = _truth_by_cell(truth)
    free = _by_cell(bundle.guideline_timeline.get("free", []))
    constrained = _by_cell(bundle.guideline_timeline.get("constrained", []))
    cells = sorted(set(free) & set(constrained) & set(truth_cells))
    zero = [cell for cell in cells if constrained[cell].n_included == 0]
    real = [cell for cell in cells if constrained[cell].n_included > 0]
    all_free = _mean_distance(free, truth_cells, cells)
    all_constrained = _mean_distance(constrained, truth_cells, cells)
    real_free = _mean_distance(free, truth_cells, real)
    real_constrained = _mean_distance(constrained, truth_cells, real)
    zero_help = [
        cell
        for cell in zero
        if _pair_distance(constrained[cell], truth_cells[cell])
        < _pair_distance(free[cell], truth_cells[cell])
    ]
    zero_hurt = [
        cell
        for cell in zero
        if _pair_distance(constrained[cell], truth_cells[cell])
        > _pair_distance(free[cell], truth_cells[cell])
    ]
    return {
        "artifact": str(bundle_path),
        "label": _artifact_label(bundle_path, bundle),
        "scientific": bundle.scientific,
        "degradation_reason": bundle.degradation_reason,
        "cells": len(cells),
        "zero_admitted_cells": len(zero),
        "real_comparison_cells": len(real),
        "real_comparison_fraction": round(len(real) / len(cells), 4) if cells else 0.0,
        "all_cells": {
            "free_to_truth": all_free,
            "constrained_to_truth": all_constrained,
            "delta": round(all_free - all_constrained, 4)
            if all_free is not None and all_constrained is not None
            else None,
        },
        "real_comparison": {
            "free_to_truth": real_free,
            "constrained_to_truth": real_constrained,
            "delta": round(real_free - real_constrained, 4)
            if real_free is not None and real_constrained is not None
            else None,
        },
        "zero_admitted_artifact": {
            "help_cells": len(zero_help),
            "hurt_cells": len(zero_hurt),
        },
        "shadow_report_denominator_audit": _shadow_denominator_audit(bundle_path),
    }


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Artifact | cells | zero | real | all delta | real delta | legacy latest denom |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        legacy = row.get("shadow_report_denominator_audit") or {}
        denom = ""
        if legacy:
            denom = (
                f"{legacy['legacy_warranted_latest_claims']}/"
                f"{legacy['legacy_all_latest_claims']}"
            )
        print(
            "| {label} | {cells} | {zero} | {real} | {all_delta} | {real_delta} | {denom} |".format(
                label=row["label"],
                cells=row["cells"],
                zero=row["zero_admitted_cells"],
                real=row["real_comparison_cells"],
                all_delta=row["all_cells"]["delta"],
                real_delta=row["real_comparison"]["delta"],
                denom=denom,
            )
        )


def main() -> None:
    args = parse_args()
    truth = load_ground_truth(args.ground_truth)
    rows = [analyze_artifact(path, truth) for path in args.artifacts]
    if args.markdown:
        _print_markdown(rows)
    else:
        print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
