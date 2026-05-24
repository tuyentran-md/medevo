#!/usr/bin/env bash
# Runs all 6 batches of Run 3 (Sonnet 4.6, 30-claim battery) sequentially.
# Stops immediately if any batch comes back scientific=false.
# Merges all batches at the end.
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python3"
GT="data/ground_truth/battery_30claim.json"
INPUT="data/input_battery_30claim.txt"
HORIZONS="2000,2012,2024"
LOG_DIR="/tmp"
BATCH_MANIFESTS=()

run_batch() {
    local n=$1
    local start=$(( n * 5 ))
    local end=$(( start + 5 ))
    local title="run3-sonnet-batch${n}-claims${start}-${end}"
    local log="$LOG_DIR/medevo_run3_b${n}.log"

    echo "[batch $n] claims ${start}:${end} → $log"
    "$VENV_PY" -u -m scripts.evaluate_shadow \
        --input-file "$INPUT" \
        --ground-truth "$GT" \
        --backend claude-cli --model claude-sonnet-4-6 \
        --horizons "$HORIZONS" \
        --claim-slice "${start}:${end}" \
        --title "$title" \
        > "$log" 2>&1

    # Extract manifest path and scientific flag from log
    local manifest
    manifest=$(python3 -c "import json,sys; d=json.load(open('$log')); print(d['manifest'])" 2>/dev/null || echo "")
    if [[ -z "$manifest" ]]; then
        echo "[batch $n] FAILED — no manifest in log. Stopping." >&2
        exit 1
    fi

    local sci
    sci=$(python3 -c "import json; print(json.load(open('$manifest'))['scientific'])")
    local delta
    delta=$(python3 -c "import json; print(json.load(open('$manifest'))['shadow_summary']['delta'])")
    local wall
    wall=$(python3 -c "import json; print(round(json.load(open('$manifest'))['wall_clock_seconds']/60,1))")

    echo "[batch $n] scientific=$sci | E3_delta=$delta | wall=${wall}min"

    if [[ "$sci" != "True" ]]; then
        echo "[batch $n] scientific=False — quota hit. Stopping." >&2
        exit 2
    fi

    BATCH_MANIFESTS+=("$manifest")
}

for n in 0 1 2 3 4 5; do
    # Skip batch 0 — already done
    if [[ $n -eq 0 ]]; then
        echo "[batch 0] already done — skipping"
        # Find existing batch 0 manifest
        m=$(ls data/run_manifests/shadow-20260523T105646Z.json 2>/dev/null || echo "")
        [[ -n "$m" ]] && BATCH_MANIFESTS+=("$m")
        continue
    fi
    run_batch $n
done

echo ""
echo "All batches done. Merging..."

# Collect all partial bundles
BUNDLES=()
for m in "${BATCH_MANIFESTS[@]}"; do
    dir=$(python3 -c "import json,os; d=json.load(open('$m')); print(os.path.dirname(d['artifact_paths']['natural_bundle']))")
    BUNDLES+=("$dir/natural_bundle.json")
done

OUT="data/artifacts/run3-merged-$(date -u +%Y%m%dT%H%M%SZ)"
"$VENV_PY" -m scripts.merge_runs "${BUNDLES[@]}" \
    --ground-truth "$GT" \
    --out "$OUT"

echo "Merged → $OUT"
