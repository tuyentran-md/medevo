#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/_runs data/run_manifests

stamp="$(date +%Y%m%dT%H%M%S)"
status_log="data/_runs/launchd_cvd_check_2am_${stamp}.log"

{
  echo "check_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if pgrep -af "python.*-m scripts.evaluate.*--topic cvd" >/tmp/medevo_cvd_pgrep.$$ 2>/dev/null; then
    echo "run_state=active"
    cat /tmp/medevo_cvd_pgrep.$$
  else
    echo "run_state=inactive"
  fi
  rm -f /tmp/medevo_cvd_pgrep.$$

  latest_run="$(ls -1t data/_runs/run_auto_cvd_*.log 2>/dev/null | head -n 1 || true)"
  echo "latest_run_log=${latest_run:-none}"
  if [[ -n "${latest_run}" && -f "${latest_run}" ]]; then
    echo "--- latest_run_tail ---"
    tail -n 60 "${latest_run}"
  fi

  launchd_out="data/_runs/launchd_cvd_run3.out.log"
  launchd_err="data/_runs/launchd_cvd_run3.err.log"
  echo "launchd_stdout=${launchd_out}"
  if [[ -f "${launchd_out}" ]]; then
    echo "--- launchd_stdout_tail ---"
    tail -n 40 "${launchd_out}"
  fi
  echo "launchd_stderr=${launchd_err}"
  if [[ -f "${launchd_err}" ]]; then
    echo "--- launchd_stderr_tail ---"
    tail -n 40 "${launchd_err}"
  fi

  latest_manifest="$(ls -1t data/run_manifests/evaluate-*.json 2>/dev/null | head -n 1 || true)"
  echo "latest_manifest=${latest_manifest:-none}"
  if [[ -n "${latest_manifest}" && -f "${latest_manifest}" ]]; then
    ./.venv/bin/python - "${latest_manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summary = payload.get("report_summary") or {}
print("--- latest_manifest_summary ---")
print(f"git_sha={payload.get('git_sha')}")
print(f"scientific={summary.get('scientific')}")
print(f"verdict={summary.get('verdict')}")
print(f"degradation_reason={summary.get('degradation_reason')}")
print(f"free_disp={summary.get('external_truth', {}).get('free')}")
print(f"constrained_disp={summary.get('external_truth', {}).get('constrained')}")
print(f"c0_disp={summary.get('external_truth', {}).get('c0')}")
PY
  fi
} | tee "${status_log}"
