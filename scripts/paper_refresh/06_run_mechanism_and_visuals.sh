#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU_ID:-0}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLBACKEND=Agg
mkdir -p logs/paper_refresh bmr_paper_results analysis/paper_refresh/gain_deficit
LOG="logs/paper_refresh/mechanism_visuals_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

python3 -m src.paper_refresh.run_gain_deficit_refresh \
  --out-dir analysis/paper_refresh/gain_deficit

CIFAR_ADA="ada/ViT-B-32/KDR_DTK_BMR_CIFAR100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt"
IMG_ADA="ada/ViT-B-32/KDR_DTK_BMR_ImageNet100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt"
test -f "${CIFAR_ADA}"
test -f "${IMG_ADA}"

python3 src/plot_bmr_feature_space_cifar100.py \
  --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
  --adversary-task CIFAR100 --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --target-cls 1 --patch-size 22 --method-name KDR_DTK_BMR --trigger-source KDR_DTK \
  --adamerging-path "${CIFAR_ADA}" \
  --emr-cache analysis/emr/cache/base6_bmr_emr.pt \
  --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/merged_encoder.pt \
  --n-target-clean 150 --n-nontarget-clean 400 --batch-size 128 --seed 2026 \
  --projection decision --out-dir analysis/bmr_feature_space_cifar100

python3 src/plot_bmr_feature_space_cifar100.py \
  --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
  --adversary-task ImageNet100 --exam-datasets ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --target-cls 1 --patch-size 22 --method-name KDR_DTK_BMR --trigger-source KDR_DTK \
  --adamerging-path "${IMG_ADA}" \
  --emr-cache analysis/emr/cache/imagenet6_bmr_ImageNet100_emr.pt \
  --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__ImageNet100/merged_encoder.pt \
  --n-target-clean 150 --n-nontarget-clean 400 --batch-size 128 --seed 2026 \
  --projection decision --out-dir analysis/bmr_feature_space_imagenet100

cp analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100_points.csv bmr_paper_results/
cp analysis/bmr_feature_space_imagenet100/bmr_feature_space_imagenet100_points.csv bmr_paper_results/

python3 src/plot_bmr_feature_space_combined.py \
  --cifar-csv bmr_paper_results/bmr_feature_space_cifar100_points.csv \
  --imagenet-csv bmr_paper_results/bmr_feature_space_imagenet100_points.csv \
  --out-dir bmr_paper_results

echo "[paper-refresh] motivation_weak_coupling_vs_decision_impact.pdf is intentionally reused unchanged"
echo "[paper-refresh] background_fidelity_combined_final.pdf is intentionally NOT regenerated"
