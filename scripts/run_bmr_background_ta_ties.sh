#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p logs results/bmr_background_slim
LOG="logs/bmr_background_ta_ties_$(date +%Y%m%d_%H%M%S).log"

for method in KDR_DTK_BMR KDR_DTK_BMR_NO_BG; do
  echo "[$(date --iso-8601=seconds)] method=${method}" | tee -a "${LOG}"
  python3 src/eval_submerge.py \
    --model ViT-B-32 \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --attack-type "${method}" \
    --trigger-source KDR_DTK \
    --adversary-task CIFAR100 \
    --target-task CIFAR100 \
    --target-cls 1 \
    --patch-size 22 \
    --merge-methods ta,ties \
    --scaling-coef 0.3 \
    --batch-size 128 \
    --test-utility \
    --out-dir ./results/bmr_background_slim 2>&1 | tee -a "${LOG}"
done

echo "[$(date --iso-8601=seconds)] done" | tee -a "${LOG}"
