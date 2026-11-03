#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)

echo "[$(date -Is)] EMR seven-adversary run started"

# BASE6 clean is already available from the CIFAR100 run. ImageNet100 uses the
# distinct IMAGENET6 suite, so create its clean reference once.
python3 -m src.reviewer_exp.run_emr_bmr_eval \
  --mode evaluate --variant clean --adversary-task ImageNet100 --seed 2026

for task in "${TASKS[@]}"; do
  echo "[$(date -Is)] EMR BMR adversary=${task}"
  python3 -m src.reviewer_exp.run_emr_bmr_eval \
    --mode evaluate --variant bmr --adversary-task "${task}" --seed 2026
done

echo "[$(date -Is)] EMR seven-adversary run completed"
