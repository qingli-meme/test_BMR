#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)
N=${#GPUS[@]}

for gpu in "${GPUS[@]}"; do
  tmux kill-session -t "bmr_refresh_gen_g${gpu}" 2>/dev/null || true
done

for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  queue=""
  for ti in "${!TASKS[@]}"; do
    if (( ti % N == gi )); then
      cmd="bash scripts/paper_refresh/run_generalization_task.sh ${gpu} ${TASKS[$ti]}"
      [[ -z "${queue}" ]] && queue="${cmd}" || queue="${queue}; ${cmd}"
    fi
  done
  [[ -z "${queue}" ]] && continue
  tmux new-session -d -s "bmr_refresh_gen_g${gpu}" \
    "${queue}; echo '[paper-refresh] GENERALIZATION QUEUE FINISHED'; exec bash"
done

tmux ls | grep bmr_refresh_gen || true
