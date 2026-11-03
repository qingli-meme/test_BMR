#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)

echo "[$(date -Is)] RegMean++ seven-adversary BMR run started"
for task in "${TASKS[@]}"; do
  echo "[$(date -Is)] RegMean++ BMR adversary=${task}"
  python3 -m src.reviewer_exp.run_new_merges \
    --merge-method regmean_plusplus \
    --attack bmr \
    --adversary-task "${task}" \
    --seed 2026
done
echo "[$(date -Is)] RegMean++ seven-adversary BMR run completed"
