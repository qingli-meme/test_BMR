#!/usr/bin/env bash
set -euo pipefail

cd /data0/BadMerging
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

OUT="bmr_paper_results/reviewer_response/classic_backdoors_new_merges"
LOG_DIR="bmr_paper_results/reviewer_response/logs/classic_backdoors_new_merges"
mkdir -p "$OUT" "$LOG_DIR"

python -m src.classic_backdoors.audit 2>&1 | tee "$LOG_DIR/artifact_audit.log"

# One isolated one-batch training smoke test catches data, trigger, optimizer,
# checkpoint-save, and offline-loading failures without creating a final artifact.
python -m src.classic_backdoors.train_attack \
  --attack badnets --dataset CIFAR100 --epochs 1 --max-train-batches 1 --smoke --force \
  2>&1 | tee "$LOG_DIR/smoke_train_badnets_cifar100.log"

for dataset in CIFAR100 ImageNet100; do
  for attack in badnets label_consistent trojannn dynamic; do
    python -m src.classic_backdoors.train_attack \
      --attack "$attack" --dataset "$dataset" \
      2>&1 | tee "$LOG_DIR/train_${attack}_${dataset}.log"

    # Merge/evaluation only after the exact attack checkpoint has completed.
    for merger in emr regmean_plusplus; do
      python -m src.classic_backdoors.evaluate_new_merges \
        --merge-method "$merger" --attack "$attack" --adversary-task "$dataset" \
        2>&1 | tee "$LOG_DIR/eval_${merger}_${attack}_${dataset}.log"
    done
  done
done

# No-Attack reuses the already validated clean-only merged artifacts and only
# evaluates the finalized fixed_22 trigger; it does not recompute a clean merge.
for dataset in CIFAR100 ImageNet100; do
  for merger in emr regmean_plusplus; do
    python -m src.classic_backdoors.evaluate_new_merges \
      --merge-method "$merger" --attack no_attack --adversary-task "$dataset" \
      2>&1 | tee "$LOG_DIR/eval_${merger}_no_attack_${dataset}.log"
  done
done

python -m src.classic_backdoors.summarize 2>&1 | tee "$LOG_DIR/summarize.log"
date -Is > "$OUT/COMPLETED"
