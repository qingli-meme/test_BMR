#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"; ROLE="$2"; shift 2
TASKS=("$@")
RUN_ID="${REVIEWER_RUN_ID:?REVIEWER_RUN_ID is required}"
OUT="bmr_paper_results/reviewer_response"
mkdir -p "$OUT/status" "logs/reviewer_response/$RUN_ID"

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then return; fi
    echo "[$(date -Is)] physical GPU $GPU_ID busy (${used} MiB); waiting"
    sleep 30
  done
}

wait_for_gpu
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[$(date -Is)] worker=$ROLE physical_gpu=$GPU_ID tasks=${TASKS[*]}"

if [[ "$ROLE" == gpu0 ]]; then
  RUN_CLEAN_BASE6=1 scripts/run_reviewer_new_merges.sh "${TASKS[@]}"
else
  RUN_CLEAN_IMAGENET6=1 scripts/run_reviewer_new_merges.sh "${TASKS[@]}"
fi
touch "$OUT/status/${RUN_ID}_${ROLE}_new_merges.done"

if [[ "$ROLE" == gpu0 ]]; then
  while [[ ! -f "$OUT/status/${RUN_ID}_gpu1_new_merges.done" ]]; do sleep 30; done
  python3 -m src.reviewer_exp.summarize_reviewer_results --section new_merges
  scripts/run_reviewer_per_task_utility.sh
  scripts/run_reviewer_defenses_cifar100.sh
  scripts/run_reviewer_background_fidelity.sh
  python3 -m src.reviewer_exp.summarize_reviewer_results --section all
  touch "$OUT/status/${RUN_ID}_all.done"
fi

echo "[$(date -Is)] worker=$ROLE complete"

