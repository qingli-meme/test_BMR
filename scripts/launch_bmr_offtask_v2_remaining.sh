#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0}"
TASKS=(GTSRB EuroSAT SUN397 PETS)
for i in "${!TASKS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"; task="${TASKS[$i]}"; session="bmr_offtask_v2_${task,,}"
  tmux has-session -t "${session}" 2>/dev/null && { echo "session exists: ${session}" >&2; continue; }
  tmux new-session -d -s "${session}" "cd /data0/BadMerging; bash scripts/run_bmr_offtask_v2_task.sh ${gpu} ${task}; exec bash"
  echo "session=${session} gpu=${gpu} task=${task}"
done
