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
CKPT="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V3_Cars_Tgt_1_R10_L_22"
GRAM_CACHE="analysis/bmr_offtask_v3/cifar100_clean_linear_grams.pt"
V2_ROOT="results/bmr_offtask_v2/Cars"

for required in "${META}" "${TRIGGER}" "${V2_ROOT}/regmean.json" \
  "${V2_ROOT}/regmean_attacked.pt" "${V2_ROOT}/regmean_background.pt"; do
  [[ -f "${required}" ]] || { echo "[OFF-V3] missing required v2 artifact: ${required}" >&2; exit 2; }
done
mkdir -p "${ROOT}" "${CKPT}" analysis/bmr_offtask_v3 logs/bmr_offtask_v3

python3 -m pytest -q tests/test_offtask_directional_utils.py

if [[ ! -f "${CKPT}/finetuned.pt" ]]; then
  python3 src/finetune_kdr_dtk_bmr_offtask_v3.py \
    --target-task "${TARGET}" \
    --metadata-path "${META}" \
    --trigger-path "${TRIGGER}" \
    --gram-cache "${GRAM_CACHE}" \
    --output-dir "${CKPT}"
else
  echo "[OFF-V3] reuse existing checkpoint ${CKPT}/finetuned.pt"
fi

cp "${CKPT}/config.json" "${ROOT}/config.json"
cp "${CKPT}/stage2_summary.json" "${ROOT}/stage2_summary.json"

if [[ ! -f "${ROOT}/regmean.json" ]]; then
  python3 src/eval_bmr_offtask_v2_all_merges.py \
    --target-task "${TARGET}" --merge-method regmean \
    --trigger-path "${TRIGGER}" \
    --attack-checkpoint "${CKPT}/finetuned.pt" \
    --out-dir "${ROOT}" \
    --skip-ta-gate \
    --method-tag KDR_DTK_BMR_OFF_V3 \
    --ada-cache-tag KDR_DTK_BMR_OFF_V3
else
  echo "[OFF-V3] reuse existing RegMean result ${ROOT}/regmean.json"
fi

python3 src/analyze_bmr_offtask_v3_regmean.py \
  --target-task "${TARGET}" \
  --metadata-path "${META}" \
  --trigger-path "${TRIGGER}" \
  --v2-result-dir "${V2_ROOT}" \
  --v3-result-dir "${ROOT}"

touch "${ROOT}/.regmean_complete"
echo "[OFF-V3] RegMean-first run complete; stopping before all other mergers."
