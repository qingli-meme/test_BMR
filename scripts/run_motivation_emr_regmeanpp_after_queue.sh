#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_SESSION="${WAIT_SESSION:-new_merge_diagnostics_wait_20261022_170445}"
GPU_ID="${GPU_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="bmr_paper_results/reviewer_response/logs/motivation_emr_regmeanpp_${STAMP}.log"

mkdir -p "$(dirname "${LOG}")"
while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m py_compile \
  src/collect_motivation_metrics.py \
  src/plot_motivation_weak_coupling.py

python3 src/collect_motivation_metrics.py \
  --methods emr,regmean_plusplus \
  --tasks CIFAR100,ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --batch-size 128 \
  --out-dir bmr_paper_results/motivation \
  2>&1 | tee "${LOG}"

python3 src/collect_motivation_metrics.py \
  --aggregate-only \
  --out-dir bmr_paper_results/motivation \
  2>&1 | tee -a "${LOG}"

python3 src/plot_motivation_weak_coupling.py \
  --csv bmr_paper_results/motivation/motivation_metrics.csv \
  --out-dir bmr_paper_results/motivation \
  2>&1 | tee -a "${LOG}"

