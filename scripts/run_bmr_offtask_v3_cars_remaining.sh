#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  CUDA_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"
else
  CUDA_DEVICE="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg

TARGET="Cars"
ROOT="results/bmr_offtask_v3/${TARGET}"
META="analysis/bmr_offtask_v2/targets/Cars_Tgt_1_R10.pt"
TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy"
CKPT="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V3_Cars_Tgt_1_R10_L_22/finetuned.pt"

[[ -f "${CKPT}" ]] || { echo "[OFF-V3] refusing remaining mergers: v3 checkpoint missing" >&2; exit 2; }
[[ -f "${ROOT}/.regmean_complete" ]] || { echo "[OFF-V3] refusing remaining mergers: RegMean-first run is not complete" >&2; exit 2; }

# This script intentionally contains no training command. It reuses CKPT exactly.
for method in ta ties adamerging emr regmeanpp; do
  [[ -f "${ROOT}/${method}.json" ]] && { echo "[OFF-V3] reuse ${method}"; continue; }
  python3 src/eval_bmr_offtask_v2_all_merges.py \
    --target-task "${TARGET}" --merge-method "${method}" \
    --trigger-path "${TRIGGER}" \
    --attack-checkpoint "${CKPT}" \
    --out-dir "${ROOT}" \
    --skip-ta-gate \
    --method-tag KDR_DTK_BMR_OFF_V3 \
    --ada-cache-tag KDR_DTK_BMR_OFF_V3
done

touch "${ROOT}/.complete"
echo "[OFF-V3] remaining mergers complete; no model was retrained."
