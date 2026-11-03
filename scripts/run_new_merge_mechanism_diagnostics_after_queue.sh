#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_SESSION="${WAIT_SESSION:-confirmed_reviewer_queue_20261022_164910}"
GPU_ID="${GPU_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="bmr_paper_results/reviewer_response/logs"
mkdir -p "${LOG_DIR}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SMOKE_LOG="${LOG_DIR}/new_merge_mechanism_diagnostics_smoke_${STAMP}.log"
FULL_LOG="${LOG_DIR}/new_merge_mechanism_diagnostics_${STAMP}.log"

python3 -m src.reviewer_exp.run_new_merge_mechanism_diagnostics \
  --methods emr,regmean_plusplus \
  --gain-variants full_bmr \
  --max-eval-batches 1 \
  --out-dir bmr_paper_results/reviewer_response/new_merge_mechanism_diagnostics_smoke \
  2>&1 | tee "${SMOKE_LOG}"

python3 -m src.reviewer_exp.run_new_merge_mechanism_diagnostics \
  --methods emr,regmean_plusplus \
  --gain-variants ordinary,background_aug,full_bmr \
  --out-dir bmr_paper_results/reviewer_response/new_merge_mechanism_diagnostics \
  2>&1 | tee "${FULL_LOG}"

