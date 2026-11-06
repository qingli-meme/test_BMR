#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-1}"
SHORT_TAG="${2:-}"
if [[ ! "${SHORT_TAG}" =~ ^A(0167|0100|0000)$ ]]; then
  echo "usage: $0 GPU_ID {A0167|A0100|A0000}"
  echo "Run this only after selecting the winning alpha range."
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy"
METHOD_TAG="KDR_DTK_BMR_OFF_V2_${SHORT_TAG}"
CKPT="checkpoints/ViT-B-32/CIFAR100_${METHOD_TAG}_Cars_Tgt_1_R10_L_22"
OUT="results/bmr_offtask_v2_alpha/Cars/${SHORT_TAG}"
LOG_ROOT="logs/bmr_offtask_v2_alpha"

[[ -f "${CKPT}/finetuned.pt" ]] || { echo "[fatal] selected checkpoint is absent"; exit 3; }
[[ -f "${OUT}/regmean.json" ]] || { echo "[fatal] RegMean screening is not complete"; exit 3; }

for method in ta ties adamerging emr regmeanpp; do
  [[ -f "${OUT}/${method}.json" ]] && continue
  python3 src/eval_bmr_offtask_v2_all_merges.py \
    --target-task Cars --merge-method "${method}" --ckpt-dir ./checkpoints \
    --data-location ./data --trigger-path "${TRIGGER}" \
    --attack-checkpoint "${CKPT}/finetuned.pt" --out-dir "${OUT}" \
    --target-cls 1 --patch-size 22 --batch-size 128 --seed 2026 \
    --method-tag "${METHOD_TAG}" --ada-cache-tag "${METHOD_TAG}" \
    --skip-ta-gate |& tee "${LOG_ROOT}/${SHORT_TAG}_${method}.log"
done

echo "[done] remaining mergers complete for ${SHORT_TAG}"
