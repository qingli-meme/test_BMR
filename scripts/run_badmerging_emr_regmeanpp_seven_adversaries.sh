#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DEFAULT_TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)
if [[ "$#" -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

train_badmerging_if_missing() {
  local task="$1"
  local checkpoint="checkpoints/ViT-B-32/${task}_On_${task}_Tgt_1_L_22/finetuned.pt"
  local trigger="trigger/ViT-B-32/On_${task}_Tgt_1_L_22.npy"
  test -s "${trigger}"
  if [[ -s "${checkpoint}" ]]; then
    echo "[reuse BadMerging checkpoint] ${checkpoint}"
    return
  fi
  echo "[$(date -Is)] train BadMerging adversary=${task}"
  python3 src/finetune_backdoor_badmergingon.py \
    --model ViT-B-32 \
    --adversary-task "${task}" \
    --target-cls 1 \
    --patch-size 22 \
    --alpha 0.3 \
    --seed 2026
  test -s "${checkpoint}"
}

echo "[$(date -Is)] BadMerging x {EMR,RegMean++} run started tasks=${TASKS[*]}"

for task in "${TASKS[@]}"; do
  train_badmerging_if_missing "${task}"

  echo "[$(date -Is)] EMR BadMerging adversary=${task}"
  python3 -m src.reviewer_exp.run_emr_bmr_eval \
    --mode evaluate \
    --variant badmerging \
    --adversary-task "${task}" \
    --seed 2026

  echo "[$(date -Is)] RegMean++ BadMerging adversary=${task}"
  python3 -m src.reviewer_exp.run_new_merges \
    --merge-method regmean_plusplus \
    --attack badmerging \
    --adversary-task "${task}" \
    --seed 2026
done

echo "[$(date -Is)] BadMerging x {EMR,RegMean++} run completed tasks=${TASKS[*]}"

