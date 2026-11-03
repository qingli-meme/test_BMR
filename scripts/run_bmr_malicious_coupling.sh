#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p logs analysis/bmr_controls/malicious_coupling
LOG="logs/bmr_malicious_coupling_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
python3 src/analyze_malicious_residual_coupling.py \
  --model "${MODEL:-ViT-B-32}" \
  --ckpt-dir ./checkpoints --data-location ./data \
  --attack-task "${ADV:-CIFAR100}" --target-cls "${TARGET_CLS:-1}" --patch-size "${PATCH:-22}" \
  --merge-methods ta,ties,regmean \
  --attack-types KDR_DTK_BMR_ORD,KDR_DTK_BMR_BG,KDR_DTK_BMR \
  --attack-labels Ordinary,BackgroundAug,FullBMR \
  --out-dir analysis/bmr_controls/malicious_coupling \
  2>&1 | tee "${LOG}"
