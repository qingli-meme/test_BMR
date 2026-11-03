#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p logs analysis/bmr_controls/real_gain
LOG="logs/bmr_real_gain_controls_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

# The key comparison for the objective question: same trigger/training framework,
# Background Augmentation vs gain_only vs Full BMR.
python3 src/eval_bmr_real_merge_gain.py \
  --model "${MODEL:-ViT-B-32}" --ckpt-dir ./checkpoints --data-location ./data \
  --adversary-task "${ADV:-CIFAR100}" --target-task "${ADV:-CIFAR100}" \
  --target-cls "${TARGET_CLS:-1}" --patch-size "${PATCH:-22}" \
  --attack-types KDR_DTK_BMR_BG,KDR_DTK_BMR_GAINONLY,KDR_DTK_BMR \
  --merge-methods ta,ties,regmean,adamerging \
  --batch-size "${BATCH_SIZE:-128}" \
  --out-dir analysis/bmr_controls/real_gain \
  2>&1 | tee "${LOG}"
