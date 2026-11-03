#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-2}"
while true; do
  USED="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "${USED}" -lt 4000 ]] && break
  echo "[representation-layer] GPU ${GPU_ID} busy (${USED} MiB); waiting 30s" >&2
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

OUT="bmr_paper_results/reviewer_response/representation_layer_six_merges"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/bmr_representation_layer_new_merges_gpu${GPU_ID}_${STAMP}.log"
TASKS="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS"
mkdir -p "${OUT}/runs" logs

python3 src/reviewer_exp/collect_representation_layer_six_merges.py --audit-only 2>&1 | tee "${LOG}"

run_one() {
  local setting="$1" method="$2" checkpoint="$3" trigger="$4"
  local slug="${setting}__${method}"
  echo "[representation-layer] setting=${setting} method=${method}" | tee -a "${LOG}"
  python3 -m src.reviewer_exp.run_sensitivity_new_merges \
    --merge-method "${method}" \
    --experiment representation_layer \
    --setting-value "${setting}" \
    --tasks "${TASKS}" \
    --malicious-checkpoint "${checkpoint}" \
    --trigger "${trigger}" \
    --output-dir "${OUT}/runs/${slug}" \
    --device cuda --seed 2026 --batch-size 128 \
    2>&1 | tee -a "${LOG}"
}

run_one middle emr \
  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_MID_CIFAR100_Tgt_1_L_22/finetuned.pt \
  trigger/ViT-B-32/KDR_DTK_REP_MID_CIFAR100_Tgt_1_L_22.npy
run_one middle regmean_plusplus \
  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_MID_CIFAR100_Tgt_1_L_22/finetuned.pt \
  trigger/ViT-B-32/KDR_DTK_REP_MID_CIFAR100_Tgt_1_L_22.npy
run_one late emr \
  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_LATE_CIFAR100_Tgt_1_L_22/finetuned.pt \
  trigger/ViT-B-32/KDR_DTK_REP_LATE_CIFAR100_Tgt_1_L_22.npy
run_one late regmean_plusplus \
  checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_LATE_CIFAR100_Tgt_1_L_22/finetuned.pt \
  trigger/ViT-B-32/KDR_DTK_REP_LATE_CIFAR100_Tgt_1_L_22.npy

python3 src/reviewer_exp/collect_representation_layer_six_merges.py 2>&1 | tee -a "${LOG}"
echo "[representation-layer] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
