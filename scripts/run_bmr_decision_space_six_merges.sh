#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Do not compete with unrelated long-running jobs. Pick the first P100 with
# enough free memory for the CLIP receiver extraction and RegMean pass.
if [[ -z "${GPU_ID:-}" ]]; then
  while true; do
    GPU_ID="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$2 + 0 < 4000 {gsub(/ /,"",$1); print $1; exit}')"
    [[ -n "${GPU_ID}" ]] && break
    echo "[decision-space] all GPUs busy; waiting 30s" >&2
    sleep 30
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/bmr_decision_space_six_merges_gpu${GPU_ID}_${STAMP}.log"
mkdir -p logs analysis/bmr_feature_space_cifar100 analysis/bmr_feature_space_imagenet100 bmr_paper_results

echo "[decision-space] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee "${LOG}"
python3 -m py_compile \
  src/plot_bmr_feature_space_cifar100.py \
  src/plot_bmr_feature_space_combined.py 2>&1 | tee -a "${LOG}"

python3 src/plot_bmr_feature_space_cifar100.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --target-cls 1 \
  --patch-size 22 \
  --method-name KDR_DTK_BMR \
  --trigger-source KDR_DTK \
  --n-target-clean 150 \
  --n-nontarget-clean 400 \
  --batch-size 128 \
  --seed 2026 \
  --projection decision \
  --emr-cache analysis/emr/cache/base6_bmr_emr.pt \
  --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/merged_encoder.pt \
  --out-dir analysis/bmr_feature_space_cifar100 \
  2>&1 | tee -a "${LOG}"

python3 src/plot_bmr_feature_space_cifar100.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task ImageNet100 \
  --exam-datasets ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --target-cls 1 \
  --patch-size 22 \
  --method-name KDR_DTK_BMR \
  --trigger-source KDR_DTK \
  --adamerging-path ada/ViT-B-32/KDR_DTK_BMR_ImageNet100_Tgt_1_L_22_imagenet100_six_task_Epoch_500.pt \
  --n-target-clean 150 \
  --n-nontarget-clean 400 \
  --batch-size 128 \
  --seed 2026 \
  --projection decision \
  --emr-cache analysis/emr/cache/imagenet6_bmr_ImageNet100_emr.pt \
  --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__ImageNet100/merged_encoder.pt \
  --out-dir analysis/bmr_feature_space_imagenet100 \
  2>&1 | tee -a "${LOG}"

for task in cifar100 imagenet100; do
  cp "analysis/bmr_feature_space_${task}/bmr_feature_space_${task}_points.csv" bmr_paper_results/
  cp "analysis/bmr_feature_space_${task}/bmr_feature_space_${task}_summary.json" bmr_paper_results/
  cp "analysis/bmr_feature_space_${task}/bmr_feature_space_${task}.pdf" bmr_paper_results/
  cp "analysis/bmr_feature_space_${task}/bmr_feature_space_${task}.png" bmr_paper_results/
done

python3 src/plot_bmr_feature_space_combined.py \
  --cifar-csv bmr_paper_results/bmr_feature_space_cifar100_points.csv \
  --imagenet-csv bmr_paper_results/bmr_feature_space_imagenet100_points.csv \
  --out-dir bmr_paper_results \
  2>&1 | tee -a "${LOG}"

echo "[decision-space] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
