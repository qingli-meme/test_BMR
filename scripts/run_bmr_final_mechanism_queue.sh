#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  GPU_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"
else
  GPU_DEVICE="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg

ROOT="analysis/bmr_final_mechanism"
LOG_ROOT="${ROOT}/queue_logs"
TRIGGER="trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
INIT="checkpoints/ViT-B-32/CIFAR100/finetuned.pt"
mkdir -p "${LOG_ROOT}"

for required in "${TRIGGER}" "${INIT}" checkpoints/ViT-B-32/zeroshot.pt; do
  [[ -f "${required}" ]] || { echo "missing required artifact: ${required}" >&2; exit 2; }
done

run_training() {
  local method_name="$1"
  local ablation_mode="$2"
  local checkpoint="checkpoints/ViT-B-32/CIFAR100_${method_name}_CIFAR100_Tgt_1_L_22/finetuned.pt"
  local log="${LOG_ROOT}/${method_name}.log"
  if [[ -f "${checkpoint}" ]]; then
    echo "[mechanism] reuse ${checkpoint}" | tee -a "${log}"
    return
  fi
  python3 -u src/finetune_kdr_dtk_bmr.py \
    --adversary-task CIFAR100 --target-cls 1 --patch-size 22 \
    --trigger-path "${TRIGGER}" --init-checkpoint "${INIT}" \
    --method-name "${method_name}" --ablation-mode "${ablation_mode}" \
    --epochs 5 --batch-size 128 --bd-batch-size 64 \
    --lr 5e-7 --wd 0.05 --grad-clip 1.0 --trainable-scope all \
    --alpha-min 0.2 --alpha-max 1.0 \
    --clean-weight 1.0 --bd-weight 1.0 --reserve-weight 1.0 \
    --seed 2026 2>&1 | tee "${log}"
}

# New controls required by 01_LOSS_MECHANISM_ABLATION.md.
run_training KDR_DTK_BMR_SOFTPLUS softplus_margin
run_training KDR_DTK_BMR_FIXEDD fixed_deficit

# One unchanged Full-BMR run with the added active-region instrumentation.
run_training KDR_DTK_BMR_ACTIVE full_bmr

bash scripts/run_bmr_final_mechanism_posttrain.sh "${GPU_DEVICE}"

touch "${ROOT}/.complete_without_multiseed"
printf '%s\n' \
  'Completed experiments 01-04.' \
  'Experiment 05 (2026/2027/2028 multi-seed confirmation) was intentionally skipped.' \
  > "${ROOT}/MULTISEED_DEFERRED.txt"
echo "[mechanism] experiments 01-04 complete; multi-seed confirmation deferred."
