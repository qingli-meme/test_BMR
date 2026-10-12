#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p logs analysis/bmr_feature_space_cifar100 bmr_paper_results
LOG="logs/bmr_feature_space_visualization_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "[feature-space] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee "${LOG}"
python3 -m py_compile src/plot_bmr_feature_space_cifar100.py 2>&1 | tee -a "${LOG}"

python3 src/plot_bmr_feature_space_cifar100.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --target-cls 1 \
  --patch-size 22 \
  --method-name KDR_DTK_BMR \
  --trigger-source KDR_DTK \
  --n-target-clean 150 \
  --n-nontarget-clean 400 \
  --batch-size 128 \
  --seed 2026 \
  --out-dir ./analysis/bmr_feature_space_cifar100 \
  2>&1 | tee -a "${LOG}"

cp ./analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100.png ./bmr_paper_results/
cp ./analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100.pdf ./bmr_paper_results/
cp ./analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100_points.csv ./bmr_paper_results/
cp ./analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100_summary.json ./bmr_paper_results/

echo "[feature-space] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
