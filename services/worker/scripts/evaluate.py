"""Top-level MedEvo evaluation entrypoint.

Runs the C0 gold-standard reference plus a contaminated run, then computes
Phase A and Phase B verdicts. The script uses the normal simulator resolution:
with real model credentials and real PubMed configuration it can produce a
scientific run; otherwise it degrades loudly to illustrative output.

Examples:
  python -m scripts.evaluate
  python -m scripts.evaluate --backend openai-compatible --base-url https://openrouter.ai/api/v1 --model openai/gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.c0 import evaluate
from app.models import RunRequestModel


HRT = (
    "Postmenopausal hormone therapy should not be used for chronic disease prevention. "
    "Hormone therapy does not provide a net cardiovascular prevention benefit in postmenopausal women. "
    "Potential harms, including stroke and thromboembolic events, outweigh prevention benefits for routine chronic-disease use."
)

SEPSIS = (
    "Children with suspected sepsis should receive cultures before antibiotics when feasible. "
    "Broad-spectrum antibiotics should begin within one hour for septic shock. "
    "Escalate to ICU support if shock persists despite fluids and vasoactive therapy."
)

# Multi-directional benchmark: 4 claims × 3 truth directions + 1 era-reversal.
# Claim-1 smoking = stable SUPPORTS-anchor; claim-2 alcohol = era-reversal
# (SUPPORTS→REFUTES); claim-3 HRT = REFUTES post-2002; claim-4 obesity-paradox = NEUTRAL.
# Ground truth: data/ground_truth/cvd_multidirectional.json
# Horizons: 2000 (pre-WHI, pre-MR), 2012 (post-WHI, alcohol neutral), 2024 (post-MR).
CVD_MULTI = (
    "Cigarette smoking is causally associated with dose-dependent increases in coronary heart disease "
    "risk, and smoking cessation substantially reduces cardiovascular mortality within years of quitting. "
    "Light to moderate alcohol consumption of one to two standard drinks per day reduces risk of "
    "coronary heart disease by elevating high-density lipoprotein cholesterol and lowering platelet "
    "aggregation. "
    "Menopausal hormone therapy with combined estrogen and progestin reduces risk of coronary heart "
    "disease and all-cause mortality in postmenopausal women and should be considered for primary "
    "prevention of chronic disease. "
    "In patients with established coronary artery disease, overweight and mild obesity (body mass "
    "index 25 to 35) is associated with reduced cardiovascular mortality compared to normal weight, "
    "the so-called obesity paradox."
)

_GROUND_TRUTH_DIR = Path(__file__).resolve().parents[1] / "data" / "ground_truth"
_GROUND_TRUTH_MAP = {
    "hrt": _GROUND_TRUTH_DIR / "hrt_uspstf.json",
    "sepsis": None,
    "cvd": _GROUND_TRUTH_DIR / "cvd_multidirectional.json",
}
_HORIZONS_MAP = {
    "hrt": "2000,2010,2020",
    "sepsis": "2000,2010,2020",
    "cvd": "2000,2012,2024",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedEvo scientific or illustrative evaluation.")
    parser.add_argument("--topic", choices=("hrt", "sepsis", "cvd"), default="cvd",
                        help="'cvd' = multi-directional 4-claim benchmark (recommended). "
                             "'hrt' = 3-claim post-WHI (all-REFUTES, leakage-prone).")
    parser.add_argument("--input-file", type=Path, help="Optional text file to override the built-in topic text.")
    parser.add_argument(
        "--backend",
        default=os.environ.get("MEDEVO_EVAL_BACKEND", "openai-compatible"),
        help="Cloud flagship for scored runs (openai-compatible + --base-url/--model/--api-key-env). "
        "Local 'ollama' is illustrative only (NO-LOCAL rule: never scientific).",
    )
    parser.add_argument("--model", default=os.environ.get("MEDEVO_EVAL_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("MEDEVO_EVAL_BASE_URL"))
    parser.add_argument("--api-key-env", default=os.environ.get("MEDEVO_EVAL_API_KEY_ENV"))
    parser.add_argument("--failure-rate", type=float, default=0.4)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--horizons",
        default=None,
        help="ABSOLUTE calendar years for the retro date-cut (Entrez maxdate). "
        "Defaults to topic-specific years (cvd=2000,2012,2024; hrt/sepsis=2000,2010,2020). "
        "Values <1900 are clamped to the 2025 ceiling.",
    )
    parser.add_argument("--ground-truth", type=Path, help="Optional ground-truth fixture path.")
    parser.add_argument("--title", default="medevo-eval")
    return parser.parse_args()


def resolve_input_text(args: argparse.Namespace) -> str:
    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8")
    if args.topic == "cvd":
        return CVD_MULTI
    return HRT if args.topic == "hrt" else SEPSIS


def resolve_ground_truth_path(args: argparse.Namespace) -> Path | None:
    if args.ground_truth is not None:
        return args.ground_truth
    return _GROUND_TRUTH_MAP.get(args.topic)


def resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key_env:
        return os.environ.get(args.api_key_env)
    for env_name in (
        "MEDEVO_EVAL_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def parse_horizons(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    input_text = resolve_input_text(args)
    raw_horizons = args.horizons or _HORIZONS_MAP.get(args.topic, "2000,2010,2020")
    request = RunRequestModel(
        title=args.title,
        input_mode="guideline",
        input_source="paste",
        input_text=input_text,
        backend=args.backend,
        model=args.model,
        api_key=resolve_api_key(args),
        base_url=args.base_url,
        horizons=parse_horizons(raw_horizons),
    )
    report = evaluate(
        request=request,
        input_text=input_text,
        failure_rate=args.failure_rate,
        iterations=args.iterations,
        seed=args.seed,
        ground_truth_path=resolve_ground_truth_path(args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"scientific: {report['scientific']}")
    print(f"ground-truth status: {report['ground_truth_status']}")
    ext = report.get("external_truth", {})
    if ext:
        print(
            f"external truth distances (informational):"
            f" free={ext.get('free_to_truth')}"
            f" constrained={ext.get('constrained_to_truth')}"
            f" c0={ext.get('c0_to_truth')}"
        )
    if report.get("degradation_reason"):
        print(f"degradation_reason: {report['degradation_reason']}")


if __name__ == "__main__":
    main()
