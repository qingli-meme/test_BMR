#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
TAGS=(PR1 PR2 PR5 PR20)
RATES=(0.01 0.02 0.05 0.20)
for i in "${!TAGS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  name="bmr_refresh_poison_${TAGS[$i]}"
  tmux kill-session -t "${name}" 2>/dev/null || true
  tmux new-session -d -s "${name}" \
    "bash scripts/paper_refresh/run_poison_ratio.sh ${gpu} ${TAGS[$i]} ${RATES[$i]}; echo '[paper-refresh] POISON ${TAGS[$i]} FINISHED'; exec bash"
done
tmux ls | grep bmr_refresh_poison || true
