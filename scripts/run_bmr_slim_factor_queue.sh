#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-1}"
WAIT_SESSION="${WAIT_SESSION:-bmr_background_ta_ties}"

echo "[BMR slim queue] start=$(date --iso-8601=seconds) gpu=${GPU_ID}"
if tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
  echo "[BMR slim queue] waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 30
  done
fi

for variant in bmr scale_variation absolute_margin bmr_no_scale; do
  echo "[BMR slim queue] launching ${variant} at $(date --iso-8601=seconds)"
  bash scripts/run_bmr_slim_factor_task.sh "${GPU_ID}" "${variant}"
done

echo "[BMR slim queue] done=$(date --iso-8601=seconds)"
