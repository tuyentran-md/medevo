#!/usr/bin/env bash
# Batched + resumable Run 4 launcher for quota-sensitive backends (Sonnet, GPT 5.4).
#
# Usage:
#   ./run4_batched.sh sonnet --backend claude-cli --model claude-sonnet-4-6
#   ./run4_batched.sh gpt-5-4 --backend codex-cli --model gpt-5.4
#
# Pattern: 47-claim NC battery split into 10 batches of 5 claims (last = 2).
# Plus COVID lane (3 claims × y2024 only). Sequential; persists marker per
# completed batch so a quota hit or crash mid-script can be resumed by re-running.
#
# Marker dir: .run4_done/  (gitignored)
# Logs: /tmp/medevo_<title>.log
#
# Merge after: scripts/merge_runs.py with --claim-offsets

set -uo pipefail
cd "$(dirname "$0")"

LABEL="${1:-}"
shift || true
BACKEND_ARGS="$@"

if [[ -z "$LABEL" || -z "$BACKEND_ARGS" ]]; then
    echo "Usage: $0 <label> <backend args>" >&2
    echo "  Example: $0 sonnet --backend claude-cli --model claude-sonnet-4-6" >&2
    echo "  Example: $0 gpt-5-4 --backend codex-cli --model gpt-5.4" >&2
    exit 1
fi

VENV_PY=".venv/bin/python3"
MARKER_DIR=".run4_done"
mkdir -p "$MARKER_DIR"

NC_INPUT="data/input_battery_run4_nc.txt"
NC_GT="data/ground_truth/battery_run4_nc.json"
NC_COUNT=47
BATCH=5

COV_INPUT="data/input_battery_run4_covid.txt"
COV_GT="data/ground_truth/battery_run4_covid.json"

_run_batch() {
    local title="$1" input="$2" gt="$3" horizons="$4"
    local marker="$MARKER_DIR/${title}.done"
    local log="/tmp/medevo_${title}.log"
    local extra_args=("${@:5}")

    if [[ -f "$marker" ]]; then
        echo "[SKIP] $title (marker exists)"
        return 0
    fi

    echo "[RUN ] $title"
    if "$VENV_PY" -u -m scripts.evaluate_shadow \
            --input-file "$input" \
            --ground-truth "$gt" \
            --horizons "$horizons" \
            --title "$title" \
            "${extra_args[@]}" \
            $BACKEND_ARGS > "$log" 2>&1; then
        if grep -q '"scientific": true' "$log"; then
            touch "$marker"
            local elapsed
            elapsed=$(grep -E '"wall_clock_seconds":' "$log" | head -1 | grep -oE '[0-9.]+' | head -1)
            echo "[DONE] $title (wall=${elapsed:-?}s)"
        else
            echo "[FAIL] $title — scientific:false (see $log)" >&2
            grep '"degradation_reason"' "$log" | head -1 >&2
            return 1
        fi
    else
        echo "[FAIL] $title — subprocess exit $? (see $log)" >&2
        return 1
    fi
}

# --- NC lane: 47 claims, batches of 5 ---
for (( START=0; START < NC_COUNT; START += BATCH )); do
    END=$(( START + BATCH ))
    if (( END > NC_COUNT )); then END=$NC_COUNT; fi
    TITLE="run4-nc-${LABEL}-claims${START}-${END}"
    _run_batch "$TITLE" "$NC_INPUT" "$NC_GT" "2000,2012,2024" \
        --claim-slice "${START}:${END}" || exit 1
done

# --- COVID lane: 3 claims × y2024 only (single batch) ---
COV_TITLE="run4-covid-${LABEL}"
_run_batch "$COV_TITLE" "$COV_INPUT" "$COV_GT" "2024" || exit 1

echo ""
echo "ALL DONE for label=$LABEL"
echo "Markers: $MARKER_DIR/run4-{nc,covid}-${LABEL}-*"
echo "Next: merge bundles with scripts/merge_runs.py --claim-offsets"
