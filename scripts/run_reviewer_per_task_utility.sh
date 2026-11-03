#!/usr/bin/env bash
set -euo pipefail
python3 -m src.reviewer_exp.eval_per_task_utility \
  --adversary-task CIFAR100 --include-original-merges \
  --include-new-merges --states clean badmerging bmr --seed 2026

