#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Full BMR already has formal CIFAR100 results for both methods.  Evaluate the
# seven remaining unique checkpoints shared by Tables 6 and 7.
VARIANTS=(
  ordinary
  background_aug
  absolute_margin
  without_scale_variation
  without_target_alignment
  without_signal_constraint
  without_dormancy
)

echo "[$(date -Is)] CIFAR100 BMR ablations x {EMR,RegMean++} started"
for variant in "${VARIANTS[@]}"; do
  for method in emr regmean_plusplus; do
    echo "[$(date -Is)] variant=${variant} merge=${method}"
    python3 -m src.reviewer_exp.run_cifar_ablation_new_merges \
      --variant "${variant}" \
      --merge-method "${method}" \
      --seed 2026
  done
done
echo "[$(date -Is)] CIFAR100 BMR ablations x {EMR,RegMean++} completed"

