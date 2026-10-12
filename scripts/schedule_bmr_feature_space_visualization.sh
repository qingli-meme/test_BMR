#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS=(${GPUS:-0 1 2})
POLL_SECONDS="${POLL_SECONDS:-120}"
MEM_IDLE_MB="${MEM_IDLE_MB:-1200}"
mkdir -p logs
LOG="logs/bmr_feature_space_scheduler_$(date +%Y%m%d_%H%M%S).log"

gpu_mem_used() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk '{print int($1)}'
}

current_extra_active() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Eq '^(bmr_extra_scheduler|bmr_extra_)'
}

echo "[feature-space-scheduler] start=$(date --iso-8601=seconds)" | tee "${LOG}"
while current_extra_active; do
  echo "[feature-space-scheduler] waiting for bmr_extra queue" | tee -a "${LOG}"
  sleep "${POLL_SECONDS}"
done

while true; do
  for gpu in "${GPUS[@]}"; do
    used="$(gpu_mem_used "${gpu}")"
    if [ "${used}" -lt "${MEM_IDLE_MB}" ]; then
      echo "[feature-space-scheduler] launch gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
      tmux new-session -d -s bmr_feature_space_visualization \
        "cd /data0/BadMerging && bash scripts/run_bmr_feature_space_visualization.sh ${gpu}"
      exit 0
    fi
    echo "[feature-space-scheduler] gpu=${gpu} busy used=${used}MB" | tee -a "${LOG}"
  done
  sleep "${POLL_SECONDS}"
done
