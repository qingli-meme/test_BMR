#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  GPU_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"
else
  GPU_DEVICE="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg

OUT="analysis/on_off_regmean_mechanism"
mkdir -p "${OUT}/logs"

# This diagnostic is strictly zero-training.
if rg -n 'torch\.optim|optim\.Optimizer|optimizer\s*=' \
  src/analyze_on_off_regmean_support.py \
  src/analyze_regmean_exact_decomposition.py \
  src/analyze_regmean_functional_retention.py \
  src/summarize_on_off_regmean_mechanism.py; then
  echo "optimizer construction/reference found in zero-training diagnostic" >&2
  exit 91
fi

if [[ -f "${OUT}/support/layerwise_support.csv" && -f "${OUT}/support/gram_drift.csv" ]]; then
  echo "[on-off-regmean] reuse completed Part A support outputs"
else
  python3 -u src/analyze_on_off_regmean_support.py --device cuda --include-off4 \
    2>&1 | tee "${OUT}/logs/01_support.log"
fi
python3 -u src/analyze_regmean_exact_decomposition.py --device cuda --include-off4 \
  2>&1 | tee "${OUT}/logs/02_decomposition.log"
python3 -u src/analyze_regmean_functional_retention.py --device cuda --include-off4 \
  2>&1 | tee "${OUT}/logs/03_retention_decision.log"
python3 -u src/summarize_on_off_regmean_mechanism.py \
  2>&1 | tee "${OUT}/logs/04_summary.log"

touch "${OUT}/.complete"
echo "[on-off-regmean] mandatory diagnostic complete"
