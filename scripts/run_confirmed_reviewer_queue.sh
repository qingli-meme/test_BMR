#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# First fill the two CIFAR100 ablation tables with EMR and RegMean++.
bash scripts/run_cifar_ablation_new_merges.sh

# Then train missing BadMerging adversary checkpoints and evaluate the seven
# adversaries under EMR and RegMean++ (clean utility plus ASR).
bash scripts/run_badmerging_emr_regmeanpp_seven_adversaries.sh

