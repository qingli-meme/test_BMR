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
ROOT="results/bmr_offtask_v4/${TARGET}"
META="analysis/bmr_offtask_v2/targets/Cars_Tgt_1_R10.pt"
TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy"
CKPT_DIR="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V4_Cars_Tgt_1_R10_L_22"
CKPT="${CKPT_DIR}/finetuned.pt"
V2_ROOT="results/bmr_offtask_v2/Cars"
V2_CKPT="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22/finetuned.pt"

for required in "${META}" "${TRIGGER}" "checkpoints/ViT-B-32/zeroshot.pt" \
  "checkpoints/ViT-B-32/CIFAR100/finetuned.pt" "${V2_CKPT}" \
  "${V2_ROOT}/regmean.json"; do
  [[ -f "${required}" ]] || { echo "[OFF-V4] missing required artifact: ${required}" >&2; exit 2; }
done
mkdir -p "${ROOT}" "${CKPT_DIR}" logs/bmr_offtask_v4

if [[ ! -f "${CKPT}" ]]; then
  python3 src/finetune_kdr_dtk_bmr_offtask_v4.py \
    --target-task "${TARGET}" \
    --metadata-path "${META}" \
    --trigger-path "${TRIGGER}" \
    --output-dir "${CKPT_DIR}" \
    --epochs 5 --batch-size 128 --bd-batch-size 64 \
    --lr 5e-7 --weight-decay 0.05 --grad-clip 1.0 \
    --trainable-scope all --alpha-min 0.2 --alpha-max 1.0 \
    --clean-weight 1.0 --off-weight 1.0 --seed 2026
else
  echo "[OFF-V4] reuse existing checkpoint ${CKPT}"
fi

cp "${CKPT_DIR}/config.json" "${ROOT}/config.json"
cp "${CKPT_DIR}/stage2_summary.json" "${ROOT}/stage2_summary.json"

if [[ ! -f "${ROOT}/regmean.json" ]]; then
  python3 src/eval_bmr_offtask_v2_all_merges.py \
    --target-task "${TARGET}" --merge-method regmean \
    --trigger-path "${TRIGGER}" \
    --attack-checkpoint "${CKPT}" \
    --out-dir "${ROOT}" \
    --skip-ta-gate \
    --method-tag KDR_DTK_BMR_OFF_V4 \
    --ada-cache-tag KDR_DTK_BMR_OFF_V4
else
  echo "[OFF-V4] reuse existing RegMean result ${ROOT}/regmean.json"
fi

python3 src/analyze_bmr_offtask_v4_regmean.py \
  --target-task "${TARGET}" \
  --metadata-path "${META}" \
  --trigger-path "${TRIGGER}" \
  --v2-result-dir "${V2_ROOT}" \
  --v4-result-dir "${ROOT}" \
  --zeroshot-checkpoint checkpoints/ViT-B-32/zeroshot.pt \
  --clean-checkpoint checkpoints/ViT-B-32/CIFAR100/finetuned.pt \
  --v2-checkpoint "${V2_CKPT}" \
  --v4-checkpoint "${CKPT}"

touch "${ROOT}/.regmean_complete"
echo "[OFF-V4] RegMean-first run complete; stopping before all other mergers."
