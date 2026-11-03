#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

GPUS=(${GPUS:-0 1 2})
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_IDLE_MB="${MEM_IDLE_MB:-1200}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"

QUEUE=(
  gain_only
  no_background
  fixed_background
  fixed_residual
  both_fixed
)

mkdir -p logs
LOG="logs/bmr_controls_scheduler_$(date +%Y%m%d_%H%M%S).log"

gpu_mem_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}

gpu_idle() {
  local used
  used="$(gpu_mem_used "$1")"
  [ "${used}" -lt "${MEM_IDLE_MB}" ]
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local session="bmr_ctrl_${task}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[scheduler] session exists: ${session}" | tee -a "${LOG}"
    return 1
  fi
  echo "[scheduler] launch task=${task} gpu=${gpu}" | tee -a "${LOG}"
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && ADAMERGING_EPOCHS=${ADAMERGING_EPOCHS} bash scripts/run_bmr_mechanism_control_task.sh ${gpu} ${task}"
}

declare -A RUNNING
idx=0

echo "[scheduler] start=$(date --iso-8601=seconds) gpus=${GPUS[*]} queue=${QUEUE[*]}" | tee "${LOG}"

while [ "${idx}" -lt "${#QUEUE[@]}" ]; do
  for gpu in "${GPUS[@]}"; do
    if [ "${idx}" -ge "${#QUEUE[@]}" ]; then break; fi

    if [ -n "${RUNNING[${gpu}]:-}" ]; then
      session="bmr_ctrl_${RUNNING[${gpu}]}"
      if tmux has-session -t "${session}" 2>/dev/null; then
        continue
      fi
      echo "[scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu}" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi

    if gpu_idle "${gpu}"; then
      task="${QUEUE[${idx}]}"
      launch_task "${gpu}" "${task}"
      RUNNING["${gpu}"]="${task}"
      idx=$((idx + 1))
    fi
  done
  sleep "${POLL_SECONDS}"
done

while [ "${#RUNNING[@]}" -gt 0 ]; do
  for gpu in "${!RUNNING[@]}"; do
    session="bmr_ctrl_${RUNNING[${gpu}]}"
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "[scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu}" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi
  done
  sleep "${POLL_SECONDS}"
done

echo "[scheduler] all done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
