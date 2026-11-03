#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m src.reviewer_exp.run_emr_bmr_eval \
  --mode equivalence --adversary-task CIFAR100 --seed 2026

for variant in clean badmerging bmr; do
  python3 -m src.reviewer_exp.run_emr_bmr_eval \
    --mode evaluate --variant "${variant}" \
    --adversary-task CIFAR100 --seed 2026
done
