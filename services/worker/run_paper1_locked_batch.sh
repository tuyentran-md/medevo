#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-run}"
VENV_PY=".venv/bin/python3"
TITLE="paper1-locked-15claim-sonnet-f55282f"
INPUT="data/input_battery_paper1_15claim.txt"
GT="data/ground_truth/battery_paper1_15claim.json"
MANIFEST="data/run_manifests/${TITLE}.json"
LOG="/tmp/${TITLE}.log"

export MEDEVO_STUDIES_PER_CLAIM_PER_ERA="${MEDEVO_STUDIES_PER_CLAIM_PER_ERA:-4}"
export MEDEVO_MAX_CONSTRAINED_ATTEMPTS_PER_CELL="${MEDEVO_MAX_CONSTRAINED_ATTEMPTS_PER_CELL:-8}"
export MEDEVO_OUTPUT_MATCH_TARGET_RATIO="${MEDEVO_OUTPUT_MATCH_TARGET_RATIO:-0.85}"
export MEDEVO_OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO="${MEDEVO_OUTPUT_MATCH_MIN_INTERPRETABLE_RATIO:-0.80}"

ARGS=(
  --input-file "$INPUT"
  --ground-truth "$GT"
  --backend claude-cli
  --model claude-sonnet-4-6
  --title "$TITLE"
  --manifest-out "$MANIFEST"
  --horizons 2000,2012,2024
)

case "$MODE" in
  dry)
    "$VENV_PY" -m scripts.evaluate_shadow "${ARGS[@]}" --dry-run
    ;;
  fg)
    "$VENV_PY" -u -m scripts.evaluate_shadow "${ARGS[@]}"
    ;;
  run)
    nohup "$VENV_PY" -u -m scripts.evaluate_shadow "${ARGS[@]}" >"$LOG" 2>&1 &
    echo "PID=$! | log=$LOG | manifest=$MANIFEST"
    ;;
  *)
    echo "Usage: $0 <dry|fg|run>" >&2
    exit 1
    ;;
esac
