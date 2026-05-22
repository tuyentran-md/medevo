"""Run the MedEvo natural-drift + shadow CIVER/BRIM evaluation.

This lane does not use C0 as a gold standard. It runs one natural ecology pass,
keeps the all-output corpus, replays CIVER+BRIM over each stored research
process trace, and compares all vs ECW-compliant corpus against external truth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.config import DATA_DIR
from app.models import RunRequestModel
from app.shadow import evaluate_shadow_civer
from app.simulator import resolve_backend, simulate_run
from scripts.evaluate import (
    _HORIZONS_MAP,
    _request_public_payload,
    estimate_call_plan,
    parse_horizons,
    resolve_api_key,
    resolve_ground_truth_path,
    resolve_input_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedEvo shadow CIVER/BRIM evaluation.")
    parser.add_argument("--topic", choices=("hrt", "sepsis", "cvd"), default="cvd")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--backend", default=os.environ.get("MEDEVO_EVAL_BACKEND", "openai-compatible"))
    parser.add_argument("--model", default=os.environ.get("MEDEVO_EVAL_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("MEDEVO_EVAL_BASE_URL"))
    parser.add_argument("--api-key-env", default=os.environ.get("MEDEVO_EVAL_API_KEY_ENV"))
    parser.add_argument("--failure-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--title", default="medevo-shadow-eval")
    return parser.parse_args()


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _stamp_path(started_at: datetime, kind: str, suffix: str) -> Path:
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return DATA_DIR / kind / f"shadow-{stamp}{suffix}"


def main() -> None:
    args = parse_args()
    input_text = resolve_input_text(args)
    horizons = parse_horizons(args.horizons or _HORIZONS_MAP.get(args.topic, "2000,2012,2024"))
    request = RunRequestModel(
        title=args.title,
        input_mode="guideline",
        input_source="paste",
        input_text=input_text,
        backend=args.backend,
        model=args.model,
        api_key=resolve_api_key(args),
        base_url=args.base_url,
        horizons=horizons,
    )
    call_plan = estimate_call_plan(
        input_text=input_text,
        input_mode=request.input_mode,
        horizons=horizons,
        evaluate_mode=False,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "shadow-dry-run",
                    "request": _request_public_payload(request),
                    "resolved_backend": resolve_backend(request).model_dump(mode="json"),
                    "call_plan": call_plan,
                    "git_sha": _git_sha(),
                    "llm_cache_enabled": os.environ.get("MEDEVO_LLM_CACHE", "1") != "0",
                    "llm_cache_only": os.environ.get("MEDEVO_LLM_CACHE_ONLY") == "1",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    estimated_calls = int(call_plan["estimated_llm_calls_upper"])
    if args.max_calls is not None and estimated_calls > args.max_calls:
        raise SystemExit(
            f"Refusing to run: estimated {estimated_calls} LLM calls exceeds --max-calls={args.max_calls}."
        )

    started_at = datetime.now(UTC)
    artifact_dir = _stamp_path(started_at, "artifacts", "")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle, summary = simulate_run(
        request=request,
        input_text=input_text,
        failure_rate=args.failure_rate,
    )
    shadow = evaluate_shadow_civer(
        bundle=bundle,
        ground_truth_path=str(resolve_ground_truth_path(args)) if resolve_ground_truth_path(args) else None,
        source_branch="free",
    )
    ended_at = datetime.now(UTC)
    bundle_path = artifact_dir / "natural_bundle.json"
    report_path = artifact_dir / "shadow_report.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(shadow, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_clock_seconds": round((ended_at - started_at).total_seconds(), 3),
        "git_sha": _git_sha(),
        "request": _request_public_payload(request),
        "resolved_backend": resolve_backend(request).model_dump(mode="json"),
        "call_plan": call_plan,
        "scientific": bundle.scientific,
        "degradation_reason": bundle.degradation_reason,
        "run_ops": summary,
        "artifact_paths": {
            "natural_bundle": str(bundle_path),
            "shadow_report": str(report_path),
        },
        "shadow_summary": {
            "study_count": shadow["study_count"],
            "verdict_counts": shadow["verdict_counts"],
            "natural_drift": shadow["endpoint_1_natural_drift"]["mean_distance_to_truth"],
            "all_to_truth": shadow["endpoint_3_guideline_drift_reduction"]["all_to_truth"],
            "warranted_to_truth": shadow["endpoint_3_guideline_drift_reduction"]["warranted_to_truth"],
            "delta": shadow["endpoint_3_guideline_drift_reduction"]["delta"],
        },
    }
    manifest_path = args.manifest_out or _stamp_path(started_at, "run_manifests", ".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
