#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU_ID:-0}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p logs/paper_refresh
LOG="logs/paper_refresh/defenses_bmr_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

python3 -m src.reviewer_exp.build_ibvs_reference \
  --reference-task EuroSAT --target-cls 1 --patch-size 22 \
  --trigger white_square --seed 2026

for merge in task_arithmetic ties regmean adamerging; do
  for lam in 0.1 0.3; do
    python3 -m src.reviewer_exp.run_ibvs_defense \
      --attack bmr --base-merge "${merge}" --lambda-ibvs "${lam}" \
      --adversary-task CIFAR100 --seed 2026
  done
done

python3 -m src.reviewer_exp.run_dam_defense \
  --attack bmr --variant layer_wise --adversary-task CIFAR100 --seed 2026

python3 -m src.reviewer_exp.summarize_reviewer_results --section defenses
