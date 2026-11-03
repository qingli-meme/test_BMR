#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH="${PATCH:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

ATTACK_TYPES="KDR_DTK_BMR_SLIM_ORD,KDR_DTK_BMR_SLIM_SCALE,KDR_DTK_BMR_SLIM_ABS,KDR_DTK_BMR_SLIM_FIXED,KDR_DTK_BMR_SLIM_NODRIFT,KDR_DTK_BMR_SLIM_DRIFT"

OUT="./analysis/bmr_slimming_phase1"
mkdir -p "${OUT}" logs
LOG="logs/bmr_slim_phase1_eval_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

python3 src/eval_bmr_real_merge_gain.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task "${ADV}" \
  --target-task "${ADV}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH}" \
  --attack-types "${ATTACK_TYPES}" \
  --merge-methods ta,ties,regmean \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-num-train-batch 8 \
  --batch-size "${BATCH_SIZE}" \
  --max-eval-batches "${MAX_EVAL_BATCHES}" \
  --out-dir "${OUT}" \
  2>&1 | tee "${LOG}"

python3 src/summarize_bmr_slimming.py \
  --analysis-dir "${OUT}" \
  2>&1 | tee -a "${LOG}"
