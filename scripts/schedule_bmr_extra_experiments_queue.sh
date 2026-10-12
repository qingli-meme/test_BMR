#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS=(${GPUS:-0 1 2})
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_IDLE_MB="${MEM_IDLE_MB:-1200}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"

QUEUE=(
  trigger_size_16
  trigger_size_22
  trigger_size_27
  trigger_size_32
  tasknum_2
  tasknum_4
  tasknum_6
)

mkdir -p logs
LOG="logs/bmr_extra_scheduler_$(date +%Y%m%d_%H%M%S).log"

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

current_bmr_queue_active() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Eq '^(bmr_ablation_scheduler|bmr_ablate_|bmr_drift_sens_remaining)'
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local session="bmr_extra_${task}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[extra-scheduler] session exists, skip launch: ${session}" | tee -a "${LOG}"
    return 1
  fi

  echo "[extra-scheduler] launch task=${task} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
  case "${task}" in
    trigger_size_*)
      local patch="${task#trigger_size_}"
      tmux new-session -d -s "${session}" \
        "cd /data0/BadMerging && ADAMERGING_EPOCHS=${ADAMERGING_EPOCHS} bash scripts/run_bmr_cifar100_trigger_size_task.sh ${gpu} ${patch}"
      ;;
    tasknum_*)
      local n="${task#tasknum_}"
      tmux new-session -d -s "${session}" \
        "cd /data0/BadMerging && ADAMERGING_EPOCHS=${ADAMERGING_EPOCHS} bash scripts/run_bmr_cifar100_task_number_task.sh ${gpu} ${n}"
      ;;
    *)
      echo "unknown task=${task}" >&2
      return 2
      ;;
  esac
}

declare -A RUNNING
idx=0

echo "[extra-scheduler] start=$(date --iso-8601=seconds) gpus=${GPUS[*]} queue=${QUEUE[*]}" | tee "${LOG}"

while current_bmr_queue_active; do
  echo "[extra-scheduler] waiting for current ablation/drift queue to finish" | tee -a "${LOG}"
  sleep 300
done

while [ "${idx}" -lt "${#QUEUE[@]}" ]; do
  for gpu in "${GPUS[@]}"; do
    if [ "${idx}" -ge "${#QUEUE[@]}" ]; then
      break
    fi
    if [ -n "${RUNNING[${gpu}]:-}" ]; then
      session="bmr_extra_${RUNNING[${gpu}]}"
      if tmux has-session -t "${session}" 2>/dev/null; then
        continue
      fi
      echo "[extra-scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi
    if gpu_idle "${gpu}"; then
      task="${QUEUE[${idx}]}"
      launch_task "${gpu}" "${task}"
      RUNNING["${gpu}"]="${task}"
      idx=$((idx + 1))
    else
      used="$(gpu_mem_used "${gpu}")"
      echo "[extra-scheduler] gpu=${gpu} busy used=${used}MB" | tee -a "${LOG}"
    fi
  done
  sleep "${POLL_SECONDS}"
done

while [ "${#RUNNING[@]}" -gt 0 ]; do
  for gpu in "${!RUNNING[@]}"; do
    session="bmr_extra_${RUNNING[${gpu}]}"
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "[extra-scheduler] finished task=${RUNNING[${gpu}]} gpu=${gpu} time=$(date --iso-8601=seconds)" | tee -a "${LOG}"
      unset "RUNNING[${gpu}]"
    fi
  done
  sleep "${POLL_SECONDS}"
done

echo "[extra-scheduler] all done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
