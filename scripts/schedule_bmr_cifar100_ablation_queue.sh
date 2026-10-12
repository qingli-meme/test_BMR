#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS=(${GPUS:-1 2})
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_IDLE_MB="${MEM_IDLE_MB:-1200}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"

QUEUE=(
  train_ordinary
  train_background_aug
  train_absolute_margin
  train_no_scale_variation
  train_full_bmr
  trigger_no_target_alignment
  trigger_no_signal
  trigger_no_dormancy
  trigger_full
)

mkdir -p logs
LOG="logs/bmr_ablation_scheduler_$(date +%Y%m%d_%H%M%S).log"

gpu_mem_used() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk '{print int($1)}'
}

gpu_idle() {
  local gpu="$1"
  local used
  used="$(gpu_mem_used "${gpu}")"
  [ "${used}" -lt "${MEM_IDLE_MB}" ]
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local session="bmr_ablate_${task}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[scheduler] session exists, skip launch: ${session}" | tee -a "${LOG}"
    return 1
  fi
  echo "[scheduler] launch task=${task} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
  tmux new-session -d -s "${session}" \
    "cd /data0/BadMerging && ADAMERGING_EPOCHS=${ADAMERGING_EPOCHS} bash scripts/run_bmr_cifar100_ablation_task.sh ${gpu} ${task}"
}

declare -A RUNNING
idx=0

echo "[scheduler] start=$(date --iso-8601=seconds) gpus=${GPUS[*]} queue=${QUEUE[*]}" | tee "${LOG}"

while [ "${idx}" -lt "${#QUEUE[@]}" ]; do
  for gpu in "${GPUS[@]}"; do
    if [ "${idx}" -ge "${#QUEUE[@]}" ]; then
      break
    fi
    if [ -n "${RUNNING[${gpu}]:-}" ]; then
      session="bmr_ablate_${RUNNING[${gpu}]}"
      if tmux has-session -t "${session}" 2>/dev/null; then
        continue
      fi
      echo "[scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi
    if gpu_idle "${gpu}"; then
      task="${QUEUE[${idx}]}"
      launch_task "${gpu}" "${task}"
      RUNNING["${gpu}"]="${task}"
      idx=$((idx + 1))
    else
      used="$(gpu_mem_used "${gpu}")"
      echo "[scheduler] gpu=${gpu} busy used=${used}MB" | tee -a "${LOG}"
    fi
  done
  sleep "${POLL_SECONDS}"
done

while [ "${#RUNNING[@]}" -gt 0 ]; do
  for gpu in "${!RUNNING[@]}"; do
    session="bmr_ablate_${RUNNING[${gpu}]}"
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "[scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi
  done
  sleep "${POLL_SECONDS}"
done

echo "[scheduler] all done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
