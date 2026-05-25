"""Merge multiple partial shadow-run bundles into one, then recompute shadow CIVER.

Usage:
  python -m scripts.merge_runs \\
    data/artifacts/shadow-*/natural_bundle.json \\
    --ground-truth data/ground_truth/battery_30claim.json \\
    --out data/artifacts/run3-merged/

Partial bundles come from batched evaluate_shadow runs using --claim-slice.
The merge is lossless: study lists, audit trails, warrants, claim graphs, and
guideline timelines are all concatenated; scientific=True only when every batch
was scientific.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# allow running as `python -m scripts.merge_runs`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from app.models import ArtifactBundle
from app.shadow import evaluate_shadow_civer


def _reindex_claim_ids(bundle_json: str, offset: int) -> str:
    if not offset:
        return bundle_json
    nums = sorted({int(m) for m in re.findall(r'claim-(\d+)', bundle_json)}, reverse=True)
    for n in nums:
        bundle_json = re.sub(rf'\bclaim-{n}\b', f'claim-{n + offset}', bundle_json)
    return bundle_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge partial MedEvo shadow bundles.")
    p.add_argument("bundles", nargs="+", type=Path, metavar="BUNDLE_JSON")
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, metavar="OUTPUT_DIR")
    p.add_argument(
        "--claim-offsets",
        default=None,
        metavar="0,5,10,...",
        help="Comma-separated claim ID offsets per bundle (same order). "
             "Required when bundles were produced with --claim-slice batching.",
    )
    return p.parse_args()


def merge_bundles(bundles: list[ArtifactBundle]) -> ArtifactBundle:
    if not bundles:
        raise ValueError("No bundles to merge.")
    base = bundles[0]

    merged_claim_graphs = list(base.claim_graphs)
    merged_audit = list(base.audit_trail)
    merged_warrants = list(base.warrants)
    merged_lineage = list(base.lineage)
    merged_corpus: dict[str, list] = {k: list(v) for k, v in base.corpus_studies.items()}
    merged_timeline: dict[str, list] = {k: list(v) for k, v in base.guideline_timeline.items()}
    merged_snapshots: dict[str, list] = {k: list(v) for k, v in base.snapshots.items()}
    scientific = base.scientific
    degradation_reason = base.degradation_reason
    input_texts = [base.input_text]

    for b in bundles[1:]:
        merged_claim_graphs.extend(b.claim_graphs)
        merged_audit.extend(b.audit_trail)
        merged_warrants.extend(b.warrants)
        merged_lineage.extend(b.lineage)
        for branch, studies in b.corpus_studies.items():
            merged_corpus.setdefault(branch, []).extend(studies)
        for branch, claims in b.guideline_timeline.items():
            merged_timeline.setdefault(branch, []).extend(claims)
        for key, snaps in b.snapshots.items():
            merged_snapshots.setdefault(key, []).extend(snaps)
        if not b.scientific:
            scientific = False
        if b.degradation_reason and not degradation_reason:
            degradation_reason = b.degradation_reason
        input_texts.append(b.input_text)

    return ArtifactBundle(
        input_text="\n".join(input_texts),
        claim_graphs=merged_claim_graphs,
        snapshots=merged_snapshots,
        branch_diff={},
        anchors=base.anchors,
        validation_notes=base.validation_notes,
        scientific=scientific,
        mode_banner=base.mode_banner,
        model_descriptor=base.model_descriptor,
        lineage=merged_lineage,
        audit_trail=merged_audit,
        warrants=merged_warrants,
        corpus_studies=merged_corpus,
        guideline_timeline=merged_timeline,
        degradation_reason=degradation_reason,
    )


def main() -> None:
    args = parse_args()
    # Keep CLI order. Claim offsets are positional; sorting paths here attaches
    # offsets to the wrong batch and corrupts claim IDs.
    paths = list(args.bundles)
    offsets = [int(x) for x in args.claim_offsets.split(",")] if args.claim_offsets else [0] * len(paths)
    if len(offsets) != len(paths):
        print(f"--claim-offsets has {len(offsets)} values but {len(paths)} bundles given.", file=sys.stderr)
        sys.exit(1)
    print(f"Merging {len(paths)} bundles...", file=sys.stderr)

    bundles: list[ArtifactBundle] = []
    for p, offset in zip(paths, offsets):
        if not p.exists():
            print(f"  MISSING: {p}", file=sys.stderr)
            sys.exit(1)
        raw = _reindex_claim_ids(p.read_text(encoding="utf-8"), offset)
        bundles.append(ArtifactBundle.model_validate_json(raw))
        print(f"  loaded {p} offset={offset} ({len(bundles[-1].corpus_studies.get('free', []))} free studies)", file=sys.stderr)

    merged = merge_bundles(bundles)
    shadow = evaluate_shadow_civer(
        bundle=merged,
        ground_truth_path=str(args.ground_truth),
        source_branch="free",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out / "natural_bundle.json"
    report_path = args.out / "shadow_report.json"
    bundle_path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(shadow, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "merged_at": datetime.now(UTC).isoformat(),
        "source_bundles": [str(p) for p in paths],
        "scientific": merged.scientific,
        "degradation_reason": merged.degradation_reason,
        "total_free_studies": len(merged.corpus_studies.get("free", [])),
        "shadow_summary": {
            "study_count": shadow["study_count"],
            "verdict_counts": shadow["verdict_counts"],
            "natural_drift": shadow["endpoint_1_natural_drift"]["mean_distance_to_truth"],
            "all_to_truth": shadow["endpoint_3_guideline_drift_reduction"]["all_to_truth"],
            "warranted_to_truth": shadow["endpoint_3_guideline_drift_reduction"]["warranted_to_truth"],
            "delta": shadow["endpoint_3_guideline_drift_reduction"]["delta"],
            "e3_beats_null": shadow["endpoint_3_guideline_drift_reduction"].get("e3_beats_volume_null"),
        },
        "artifact_paths": {
            "natural_bundle": str(bundle_path),
            "shadow_report": str(report_path),
        },
    }
    manifest_path = args.out / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
