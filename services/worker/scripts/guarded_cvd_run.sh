#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/_runs data/run_manifests data/artifacts

sha="$(git -C ../.. rev-parse --short HEAD)"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="data/_runs/run_auto_cvd_${sha}_${stamp}.log"

if pgrep -f "python.*-m scripts.evaluate.*--topic cvd" >/dev/null 2>&1; then
  echo "skip: a CVD evaluate run is already in progress"
  exit 0
fi

if ./.venv/bin/python - "$sha" <<'PY'
import json
import sys
from pathlib import Path

sha = sys.argv[1]
for path in Path("data/run_manifests").glob("evaluate-*.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    summary = payload.get("report_summary") or {}
    if (
        str(payload.get("git_sha", "")).startswith(sha)
        and summary.get("verdict") == "PASS"
        and summary.get("scientific") is True
    ):
        print(f"skip: PASS scientific manifest already exists for {sha}: {path}")
        sys.exit(0)
sys.exit(1)
PY
then
  exit 0
fi

echo "starting CVD evaluate on ${sha} at $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "log: ${log_path}"
MEDEVO_LLM_CACHE="${MEDEVO_LLM_CACHE:-1}" \
  ./.venv/bin/python -m scripts.evaluate \
  --topic cvd \
  --backend claude-cli \
  --max-calls 500 \
  >"${log_path}" 2>&1
status=$?
echo "finished status=${status} at $(date '+%Y-%m-%dT%H:%M:%S%z')"
exit "${status}"
