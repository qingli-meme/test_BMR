#!/usr/bin/env bash
set -euo pipefail

TASKS=("$@")
if (( ${#TASKS[@]} == 0 )); then
  TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)
fi

train_badmerging_if_missing() {
  local task="$1"
  local checkpoint="checkpoints/ViT-B-32/${task}_On_${task}_Tgt_1_L_22/finetuned.pt"
  if [[ -s "$checkpoint" ]]; then
    echo "[reuse BadMerging-On] $checkpoint"
    return
  fi
  python3 src/finetune_backdoor_badmergingon.py \
    --model ViT-B-32 --adversary-task "$task" --target-cls 1 \
    --patch-size 22 --alpha 0.3 --seed 2026
  test -s "$checkpoint"
}

if [[ "${RUN_CLEAN_BASE6:-0}" == 1 ]]; then
  for merge in regmean_plusplus localize_stitch; do
    python3 -m src.reviewer_exp.run_new_merges --merge-method "$merge" \
      --attack clean --adversary-task CIFAR100 --seed 2026
  done
fi
if [[ "${RUN_CLEAN_IMAGENET6:-0}" == 1 ]]; then
  for merge in regmean_plusplus localize_stitch; do
    python3 -m src.reviewer_exp.run_new_merges --merge-method "$merge" \
      --attack clean --adversary-task ImageNet100 --seed 2026
  done
fi

for task in "${TASKS[@]}"; do
  train_badmerging_if_missing "$task"
  for merge in regmean_plusplus localize_stitch; do
    for attack in bmr badmerging; do
      python3 -m src.reviewer_exp.run_new_merges \
        --merge-method "$merge" --attack "$attack" \
        --adversary-task "$task" --seed 2026
    done
  done
done

