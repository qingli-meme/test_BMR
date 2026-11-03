#!/usr/bin/env bash
set -euo pipefail

python3 -m src.reviewer_exp.build_ibvs_reference \
  --reference-task EuroSAT --target-cls 1 --patch-size 22 \
  --trigger white_square --seed 2026

for attack in bmr badmerging; do
  for merge in task_arithmetic ties regmean adamerging; do
    for lam in 0.1 0.3; do
      python3 -m src.reviewer_exp.run_ibvs_defense \
        --attack "$attack" --base-merge "$merge" --lambda-ibvs "$lam" \
        --adversary-task CIFAR100 --seed 2026
    done
  done
done

for attack in bmr badmerging; do
  python3 -m src.reviewer_exp.run_dam_defense \
    --attack "$attack" --variant layer_wise \
    --adversary-task CIFAR100 --seed 2026
done

python3 -m src.reviewer_exp.summarize_reviewer_results --section defenses

