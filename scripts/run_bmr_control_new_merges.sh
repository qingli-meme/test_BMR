#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p logs
LOG="logs/bmr_control_new_merges_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
VARIANTS=(
  gain_only
  no_background
  fixed_background
  fixed_residual
  both_fixed
  ordinary_normmatch
  background_normmatch
  bmr_normmatch_bg
)

echo "[$(date -Is)] BMR controls x {EMR,RegMean++} started" | tee "${LOG}"

# The first package revision registered these names as CLI choices without
# registering their checkpoint-path dispatch. Re-evaluate the already-run
# mechanism controls with the corrected dispatcher before the local mergers.
for method in \
  KDR_DTK_BMR_GAINONLY KDR_DTK_BMR_NO_BG \
  KDR_DTK_BMR_FIXETA KDR_DTK_BMR_FIXALPHA; do
  echo "[$(date -Is)] corrected TA/TIES evaluation method=${method}" | tee -a "${LOG}"
  python3 src/eval_submerge.py \
    --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
    --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --attack-type "${method}" --trigger-source KDR_DTK \
    --adversary-task CIFAR100 --target-task CIFAR100 --target-cls 1 \
    --patch-size 22 --merge-methods ta,ties --scaling-coef 0.3 \
    --batch-size 128 --test-utility --out-dir results/bmr_controls \
    2>&1 | tee -a "${LOG}"
done

for variant in "${VARIANTS[@]}"; do
  for merger in emr regmean_plusplus; do
    echo "[$(date -Is)] variant=${variant} merger=${merger}" | tee -a "${LOG}"
    python3 -m src.reviewer_exp.run_cifar_ablation_new_merges \
      --variant "${variant}" --merge-method "${merger}" --seed 2026 \
      2>&1 | tee -a "${LOG}"
  done
done

# Rebuild the return archive so these local-only results are included.
bash scripts/collect_bmr_control_results.sh 2>&1 | tee -a "${LOG}"
echo "[$(date -Is)] BMR controls x {EMR,RegMean++} completed" | tee -a "${LOG}"
