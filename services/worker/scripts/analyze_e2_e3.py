"""Offline E2 + E3 analyzer over an existing shadow artifact (no LLM spend).

Re-runs `evaluate_shadow_civer` against a previously persisted natural_bundle
so the enriched per-claim E2 breakdown + E3 volume-matched null distribution
(added 2026-05-22) can be computed from already-paid LLM output.

Usage:
    .venv/bin/python -m scripts.analyze_e2_e3 \
        --artifact data/artifacts/shadow-<stamp> \
        [--ground-truth data/ground_truth/cvd_multidirectional.json] \
        [--out-prefix e2e3]

Outputs:
    <artifact>/<out-prefix>_report.json   enriched shadow report
    <artifact>/<out-prefix>_summary.md    human-readable E2/E3 verdict table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.models import ArtifactBundle
from app.shadow import evaluate_shadow_civer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to an existing shadow artifact directory (contains natural_bundle.json).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/cvd_multidirectional.json"),
    )
    parser.add_argument("--source-branch", default="free")
    parser.add_argument("--out-prefix", default="e2e3")
    return parser.parse_args()


def _fmt_pct(value: float | int) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else str(value)


def _fmt(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _e2_table(per_claim: dict) -> str:
    rows = [
        "| Claim | Cohort | n | no_cite | ungrounded | scope_OR | wrong_dir | mean_q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for claim_id in sorted(per_claim):
        for cohort in ("passed", "failed"):
            cell = per_claim[claim_id][cohort]
            rows.append(
                f"| {claim_id} | {cohort} | {cell['count']} | "
                f"{_fmt_pct(cell['no_cite_rate'])} | "
                f"{_fmt_pct(cell['ungrounded_rate'])} | "
                f"{_fmt_pct(cell['scope_overreach_rate'])} | "
                f"{_fmt_pct(cell['wrong_direction_vs_truth_rate'])} | "
                f"{_fmt(cell['mean_quality'])} |"
            )
    return "\n".join(rows)


def _e2_signal_verdict(per_claim: dict, aggregate: dict) -> tuple[str, list[str]]:
    """Per metric, does PASSED cohort beat FAILED cohort? Signal = lower
    no_cite / ungrounded / wrong_direction; higher mean_quality. scope_overreach
    requires interpretation (no-cite cohort can't over-reach scope).
    """
    passed = aggregate["passed"]
    failed = aggregate["failed"]
    notes: list[str] = []
    discriminating = []
    if failed["count"] == 0 or passed["count"] == 0:
        return "INSUFFICIENT — one cohort is empty", notes
    metric_directions = [
        ("no_cite_rate", "lower better"),
        ("ungrounded_rate", "lower better"),
        ("wrong_direction_vs_truth_rate", "lower better"),
        ("mean_quality", "higher better"),
    ]
    for metric, direction in metric_directions:
        p = passed[metric]
        f = failed[metric]
        if direction == "lower better":
            wins = p < f
            margin = round(f - p, 4)
        else:
            wins = p > f
            margin = round(p - f, 4)
        if wins:
            discriminating.append(f"{metric} (margin={margin})")
    # scope_overreach interpretation note
    if failed["no_cite_rate"] > 0.5 and failed["scope_overreach_rate"] == 0:
        notes.append(
            "scope_overreach paradox confirmed: failed cohort dominated by "
            "no-cite studies (no source = no over-reach possible). Interpret "
            "scope_overreach jointly with no_cite_rate."
        )
    if not discriminating:
        return "NO SIGNAL — passed cohort no better than failed on any clean metric", notes
    return f"SIGNAL on {len(discriminating)} of 4 metrics: " + "; ".join(discriminating), notes


def _e3_verdict(e3: dict) -> str:
    null = e3.get("volume_matched_null")
    if null is None:
        return "NULL UNAVAILABLE — warranted size 0 or = full corpus"
    all_d = e3["all_to_truth"]
    war_d = e3["warranted_to_truth"]
    delta = e3["delta"]
    null_low = null["ci_low"]
    null_mean = null["mean"]
    null_high = null["ci_high"]
    beats = e3.get("civer_beats_volume_matched", False)
    line1 = (
        f"all_to_truth={_fmt(all_d)} | warranted_to_truth={_fmt(war_d)} | "
        f"delta={_fmt(delta)}"
    )
    line2 = (
        f"volume_matched_null: mean={_fmt(null_mean)} "
        f"95%CI=[{_fmt(null_low)}, {_fmt(null_high)}]"
    )
    if beats:
        line3 = "VERDICT: CIVER beats volume-matched null (selection, not luck)."
    elif war_d > null_high:
        line3 = "VERDICT: warranted WORSE than null upper bound — CIVER selection HURTS."
    else:
        line3 = (
            "VERDICT: warranted distance within null CI — cannot distinguish "
            "CIVER selection from smaller-pool luck."
        )
    return "\n".join([line1, line2, line3])


def render_markdown(report: dict, artifact_dir: Path) -> str:
    counts = report["verdict_counts"]
    calib = report["calibration_matrix"]
    drift = report["endpoint_1_natural_drift"]
    aggregate = report["endpoint_2_warrant_enrichment"]
    per_claim = report["endpoint_2_per_claim"]
    e3 = report["endpoint_3_guideline_drift_reduction"]

    signal, notes = _e2_signal_verdict(per_claim, aggregate)

    mode_breakdown = report.get("analysis_mode_breakdown", {})
    fallback_warning = report.get("fallback_warning")

    lines = [
        f"# Shadow E2 + E3 offline analysis — {artifact_dir.name}",
        "",
        f"- Source branch: `{report['source_branch']}`",
        f"- Study count: {report['study_count']}",
        f"- Analysis mode: process={mode_breakdown.get('process', 0)}, "
        f"output_fallback={mode_breakdown.get('output_fallback', 0)}",
        f"- Verdicts: passed={counts['passed']}, failed={counts['failed']}, total={counts['total']}",
        f"- Calibration (vs harness-only TRUE provenance): "
        f"TP={calib['true_positive']}, TN={calib['true_negative']}, "
        f"FP={calib['false_positive']}, FN={calib['false_negative']}, "
        f"FPR={_fmt(calib['fpr'])}, FNR={_fmt(calib['fnr'])}",
        "",
    ]
    if fallback_warning:
        lines += [
            "> ⚠️ **" + fallback_warning + "**",
            "",
        ]
    lines += [
        "## Endpoint 1 — natural drift at guideline tier",
        f"mean_distance_to_truth = **{_fmt(drift['mean_distance_to_truth'])}**",
        "",
    ]
    if drift.get("per_claim"):
        lines.append("| Claim | distance | output (dir/level) | truth (dir/level) |")
        lines.append("|---|---:|---|---|")
        for row in drift["per_claim"]:
            o = row["output"]
            t = row["truth"]
            lines.append(
                f"| {row['claim_id']} | {_fmt(row['distance_to_truth'])} | "
                f"{o['direction']} / {o['level']} | {t['direction']} / {t['level']} |"
            )
        lines.append("")
    lines += [
        "## Endpoint 2 — warrant enrichment (CIVER discrimination)",
        "",
        f"**Verdict: {signal}**",
        "",
        "Per-claim breakdown (passed vs failed cohort):",
        "",
        _e2_table(per_claim),
        "",
    ]
    for note in notes:
        lines.append(f"> {note}")
    lines += [
        "",
        "## Endpoint 3 — guideline drift reduction (CIVER value)",
        "",
        _e3_verdict(e3),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact.resolve()
    bundle_path = artifact_dir / "natural_bundle.json"
    if not bundle_path.exists():
        raise SystemExit(f"natural_bundle.json not found at {bundle_path}")

    bundle = ArtifactBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    report = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path=str(args.ground_truth) if args.ground_truth else None,
        source_branch=args.source_branch,
    )

    out_json = artifact_dir / f"{args.out_prefix}_report.json"
    out_md = artifact_dir / f"{args.out_prefix}_summary.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_markdown(report, artifact_dir), encoding="utf-8")

    # Stdout = human-readable summary so the analyzer is one cmd "look at it".
    print(out_md.read_text(encoding="utf-8"))
    print()
    print(f"-- enriched JSON: {out_json}")
    print(f"-- markdown summary: {out_md}")


if __name__ == "__main__":
    main()
