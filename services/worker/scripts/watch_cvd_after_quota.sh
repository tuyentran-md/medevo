#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/_runs

target="${1:-01:25}"
target_hhmm="${target/:/}"
echo "watcher armed at $(date -Is), target local time ${target}"

while true; do
  now_hhmm="$(date +%H%M)"
  if [[ "${now_hhmm}" -ge "${target_hhmm}" && "${now_hhmm}" -lt "1200" ]]; then
    break
  fi
  if pgrep -f "python.*-m scripts.evaluate.*--topic cvd" >/dev/null 2>&1; then
    echo "$(date -Is) run still in progress; waiting"
  else
    echo "$(date -Is) no run in progress; waiting for quota target"
  fi
  sleep 300
done

echo "quota target reached at $(date -Is); invoking guarded run"
exec ./scripts/guarded_cvd_run.sh
