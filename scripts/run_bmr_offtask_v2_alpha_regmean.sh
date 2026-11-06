#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-1}"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  GPU_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"
else
  GPU_DEVICE="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTHONUNBUFFERED=1

TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy"
META="analysis/bmr_offtask_v2/targets/Cars_Tgt_1_R10.pt"
BASELINE_RESULT="results/bmr_offtask_v2/Cars"
RESULT_ROOT="results/bmr_offtask_v2_alpha/Cars"
LOG_ROOT="logs/bmr_offtask_v2_alpha"
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

for required in "${TRIGGER}" "${META}" \
  "checkpoints/ViT-B-32/CIFAR100/finetuned.pt" \
  "checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22/finetuned.pt" \
  "${BASELINE_RESULT}/regmean.json"; do
  [[ -f "${required}" ]] || { echo "[fatal] missing ${required}"; exit 2; }
done

if [[ ! -f "${BASELINE_RESULT}/mechanism.json" ]]; then
  python3 src/analyze_bmr_offtask_v2.py \
    --target-task Cars --metadata-path "${META}" --trigger-path "${TRIGGER}" \
    --result-dir "${BASELINE_RESULT}" --methods regmean |& tee "${LOG_ROOT}/baseline_mechanism.log"
fi

TAGS=(A0167 A0100 A0000)
MINS=(0.1666666667 0.1 0.0)

for i in "${!TAGS[@]}"; do
  short="${TAGS[$i]}"
  alpha_min="${MINS[$i]}"
  method_tag="KDR_DTK_BMR_OFF_V2_${short}"
  ckpt="checkpoints/ViT-B-32/CIFAR100_${method_tag}_Cars_Tgt_1_R10_L_22"
  result="${RESULT_ROOT}/${short}"
  mkdir -p "${ckpt}" "${result}"

  if [[ ! -f "${ckpt}/finetuned.pt" ]]; then
    python3 src/finetune_kdr_dtk_bmr_offtask_v2.py \
      --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
      --target-task Cars --target-cls 1 --trigger-path "${TRIGGER}" \
      --metadata-path "${META}" --output-dir "${ckpt}" --patch-size 22 \
      --epochs 5 --batch-size 128 --bd-batch-size 64 --lr 5e-7 \
      --weight-decay 0.05 --grad-clip 1 --trainable-scope all \
      --alpha-min "${alpha_min}" --alpha-max 1.0 --clean-weight 1 --off-weight 1 \
      --seed 2026 |& tee "${LOG_ROOT}/${short}_train.log"
  else
    echo "[resume] ${ckpt}/finetuned.pt already exists"
  fi

  cp -f "${ckpt}/config.json" "${result}/training_config.json"
  cp -f "${ckpt}/stage2.jsonl" "${result}/stage2.jsonl"
  cp -f "${ckpt}/stage2_summary.json" "${result}/stage2_summary.json"

  if [[ ! -f "${result}/regmean.json" ]]; then
    python3 src/eval_bmr_offtask_v2_all_merges.py \
      --target-task Cars --merge-method regmean --ckpt-dir ./checkpoints \
      --data-location ./data --trigger-path "${TRIGGER}" \
      --attack-checkpoint "${ckpt}/finetuned.pt" --out-dir "${result}" \
      --target-cls 1 --patch-size 22 --batch-size 128 --seed 2026 \
      --method-tag "${method_tag}" --ada-cache-tag "${method_tag}" \
      --skip-ta-gate |& tee "${LOG_ROOT}/${short}_regmean.log"
  else
    echo "[resume] ${result}/regmean.json already exists"
  fi

  python3 src/analyze_bmr_offtask_v2.py \
    --target-task Cars --metadata-path "${META}" --trigger-path "${TRIGGER}" \
    --result-dir "${result}" --methods regmean |& tee "${LOG_ROOT}/${short}_mechanism.log"
done

python3 src/summarize_bmr_offtask_v2_alpha.py
touch "${RESULT_ROOT}/.regmean_complete"
echo "[done] RegMean-only alpha-range check complete: ${RESULT_ROOT}"
