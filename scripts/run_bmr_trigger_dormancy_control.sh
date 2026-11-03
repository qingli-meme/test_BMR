#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH="${PATCH:-22}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
OUT="analysis/bmr_controls/trigger_dormancy"
mkdir -p "${OUT}" logs "trigger/${MODEL}"
LOG="logs/bmr_trigger_dormancy_control_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[trigger-control] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

DIRECTION="${OUT}/shared_target_direction.pt"
SLIM="trigger/${MODEL}/KDR_DTK_SLIM_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
SLIM_NODORM="trigger/${MODEL}/KDR_DTK_SLIM_NODORM_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"

make_trigger() {
  local save_path="$1"
  local dorm_weight="$2"
  local name="$3"
  if [ -f "${save_path}" ]; then
    echo "[trigger-control] reuse ${save_path}" | tee -a "${LOG}"
    return
  fi
  run_logged python3 src/optimize_dormant_target_key_patch.py \
    --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
    --dataset "${ADV}" --target-cls "${TARGET_CLS}" --patch-size "${PATCH}" \
    --layer-name model.visual.transformer.resblocks.11.ln_2 --pool cls \
    --epochs 3 --batch-size "${BATCH_SIZE}" --lr 5e-2 \
    --emit-eps 1.0 --align-weight 1.0 --emit-weight 0.0 \
    --dorm-kappa 0.0 --dorm-weight "${dorm_weight}" \
    --amp-weight 1e-4 --tv-weight 1e-3 \
    --patch-init-std 0.05 --patch-min -2.5 --patch-max 2.5 \
    --clean-checkpoint "${CLEAN_CKPT}" --direction-path "${DIRECTION}" \
    --save-path "${save_path}" --out-dir "${OUT}/${name}" \
    --seed "${SEED}" --log-every 20
}

# Signal constraint is disabled in BOTH variants. The only changed factor is dormancy.
make_trigger "${SLIM}" 1.0 slim
make_trigger "${SLIM_NODORM}" 0.0 slim_nodorm

run_logged python3 src/audit_trigger_dormancy_variants.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --target-task "${ADV}" --target-cls "${TARGET_CLS}" --patch-size "${PATCH}" \
  --trigger-sources KDR_DTK_SLIM,KDR_DTK_SLIM_NODORM \
  --trigger-labels SlimTrigger,SlimTriggerWithoutDormancy \
  --methods ta,ties,regmean,adamerging \
  --batch-size "${BATCH_SIZE}" --out-dir "${OUT}/audit"

echo "[trigger-control] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
