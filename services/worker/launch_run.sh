#!/usr/bin/env bash
# Usage: ./launch_run.sh <run-name> [--fg]
# Runs in background by default; pass --fg to run foreground.
#
# Presets:
#   run3          Sonnet 4.6 · 30-claim battery · full (needs large quota window)
#   run3-batch N  Sonnet 4.6 · batch N of 6 (claims 0-4, 5-9, ...) · ~180 calls each
#   run3-merge    Merge all completed run3 batches into a full shadow report
#   run4          (placeholder) 100-claim · MIMO + Opus 4.7 · paper-level
#   smoke         Cache-only replay (no API spend)
#   dry           Dry-run call estimate

set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python3"
LOG_DIR="/tmp"
RUN="${1:-}"
ARG2="${2:-}"
BACKGROUND=true
[[ "$ARG2" == "--fg" ]] && BACKGROUND=false

BATTERY_30_ARGS="--input-file data/input_battery_30claim.txt \
  --ground-truth data/ground_truth/battery_30claim.json \
  --horizons 2000,2012,2024"
BACKEND_SONNET="--backend claude-cli --model claude-sonnet-4-6"

_shadow() {
    local title="$1"; shift
    if $BACKGROUND; then
        local log="$LOG_DIR/medevo_${title}.log"
        nohup "$VENV_PY" -u -m scripts.evaluate_shadow "$@" \
            --title "$title" > "$log" 2>&1 &
        echo "PID=$! | log=$log"
    else
        "$VENV_PY" -u -m scripts.evaluate_shadow "$@" --title "$title"
    fi
}

case "$RUN" in
  run3)
    _shadow battery-30claim-sonnet-4-6-run3 \
      $BATTERY_30_ARGS $BACKEND_SONNET
    ;;

  run3-batch)
    # run3-batch N  where N in 0..5 (each batch = 5 claims, ~180 LLM calls)
    N="${ARG2}"
    if [[ -z "$N" || "$N" -lt 0 || "$N" -gt 5 ]] 2>/dev/null; then
        echo "Usage: $0 run3-batch <0..5>" >&2; exit 1
    fi
    START=$(( N * 5 ))
    END=$(( START + 5 ))
    TITLE="run3-sonnet-batch${N}-claims${START}-${END}"
    _shadow "$TITLE" \
      $BATTERY_30_ARGS $BACKEND_SONNET \
      --claim-slice "${START}:${END}"
    ;;

  run3-merge)
    # Merge all completed batch bundles and recompute full shadow report.
    BUNDLES=( data/artifacts/shadow-*/natural_bundle.json )
    # Filter only run3 batch artifacts by naming convention in their manifest.
    # For simplicity, merge ALL artifacts matching the glob — run only after
    # all 6 batches complete (check logs first).
    OUT="data/artifacts/run3-merged-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Merging into $OUT ..."
    "$VENV_PY" -m scripts.merge_runs "${BUNDLES[@]}" \
      --ground-truth data/ground_truth/battery_30claim.json \
      --out "$OUT"
    ;;

  run4)
    echo "Run 4 design not yet locked. Update BATTERY_100 in this script first." >&2
    exit 1
    ;;

  smoke)
    MEDEVO_LLM_CACHE_ONLY=1 \
    _shadow smoke-replay \
      $BATTERY_30_ARGS $BACKEND_SONNET
    ;;

  dry)
    "$VENV_PY" -m scripts.evaluate_shadow \
      $BATTERY_30_ARGS $BACKEND_SONNET \
      --title dry-run --dry-run
    ;;

  *)
    echo "Usage: $0 <run3|run3-batch N|run3-merge|run4|smoke|dry> [--fg]" >&2
    exit 1
    ;;
esac
