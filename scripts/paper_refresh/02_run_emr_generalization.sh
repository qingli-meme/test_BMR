#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU_ID:-0}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p logs/paper_refresh

TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)
LOG="logs/paper_refresh/emr_generalization_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

for task in "${TASKS[@]}"; do
  python3 -m src.reviewer_exp.run_emr_bmr_eval \
    --mode evaluate --variant bmr --adversary-task "${task}" --seed 2026 --force-merge
done
