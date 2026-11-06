#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NS=(2 4 6)
for i in "${!NS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  name="bmr_refresh_tasknum_${NS[$i]}"
  tmux kill-session -t "${name}" 2>/dev/null || true
  tmux new-session -d -s "${name}" \
    "bash scripts/paper_refresh/run_task_number.sh ${gpu} ${NS[$i]}; echo '[paper-refresh] TASKNUM ${NS[$i]} FINISHED'; exec bash"
done
tmux ls | grep bmr_refresh_tasknum || true
