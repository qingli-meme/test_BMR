#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
VARS=(noalign nosig nodorm)
for i in "${!VARS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  name="bmr_refresh_trig_${VARS[$i]}"
  tmux kill-session -t "${name}" 2>/dev/null || true
  tmux new-session -d -s "${name}" \
    "bash scripts/paper_refresh/run_trigger_ablation.sh ${gpu} ${VARS[$i]}; echo '[paper-refresh] TRIGGER ${VARS[$i]} FINISHED'; exec bash"
done
tmux ls | grep bmr_refresh_trig || true
