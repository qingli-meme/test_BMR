#!/usr/bin/env bash
set -euo pipefail
python3 -m src.reviewer_exp.collect_background_fidelity \
  --adversary-task CIFAR100 --target-cls 1 \
  --num-synthetic-backgrounds 256 --num-samples 1000 \
  --seed 2026

