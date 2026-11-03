#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID {scale_variation|absolute_margin|bmr_no_scale|bmr}" >&2
  exit 2
fi

GPU_ID="$1"
VARIANT="$2"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET="${TARGET:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-22}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
LR="${LR:-5e-7}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"

case "${VARIANT}" in
  scale_variation)
    METHOD="KDR_DTK_BMR_SCALE_NODRIFT"
    ABLATION_MODE="background_attack"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    ;;
  absolute_margin)
    METHOD="KDR_DTK_BMR_ABS_NODRIFT"
    ABLATION_MODE="absolute_margin"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    ;;
  bmr_no_scale)
    METHOD="KDR_DTK_BMR_NOSCALE_NODRIFT"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.6"
    ALPHA_MAX="0.6"
    ;;
  bmr)
    METHOD="KDR_DTK_BMR_NO_BG"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    ;;
  *)
    echo "unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p logs results/bmr_slim_factor
LOG="logs/bmr_slim_factor_${VARIANT}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
TRIGGER="trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
CKPT="checkpoints/${MODEL}/${ADV}_${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"
ADA_PATH="ada/${MODEL}/${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}_Epoch_${ADAMERGING_EPOCHS}.pt"

run_logged() {
  echo "[BMR slim:${VARIANT}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

echo "============================================================" | tee "${LOG}"
echo "[BMR slim:${VARIANT}] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee -a "${LOG}"
echo "[BMR slim:${VARIANT}] method=${METHOD} mode=${ABLATION_MODE}" | tee -a "${LOG}"
echo "[BMR slim:${VARIANT}] alpha=[${ALPHA_MIN},${ALPHA_MAX}] eta=[0,0] drift_rho=0" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

python3 -m py_compile \
  src/finetune_kdr_dtk_bmr.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py 2>&1 | tee -a "${LOG}"

test -f "${CLEAN_CKPT}"
test -f "${TRIGGER}"

if [ ! -f "${CKPT}" ]; then
  run_logged python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" \
    --data-location ./data \
    --adversary-task "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --trigger-path "${TRIGGER}" \
    --init-checkpoint "${CLEAN_CKPT}" \
    --save-root ./checkpoints \
    --method-name "${METHOD}" \
    --ablation-mode "${ABLATION_MODE}" \
    --batch-size "${BATCH_SIZE}" \
    --bd-batch-size "${BD_BATCH_SIZE}" \
    --lr "${LR}" \
    --wd 0.05 \
    --grad-clip 1.0 \
    --trainable-scope all \
    --alpha-min "${ALPHA_MIN}" \
    --alpha-max "${ALPHA_MAX}" \
    --eta-min 0.0 \
    --eta-max 0.0 \
    --drift-rho 0.0 \
    --drift-scope last2 \
    --clean-weight 1.0 \
    --bd-weight 1.0 \
    --reserve-weight 1.0 \
    --residual-weight 0.0 \
    --seed "${SEED}" \
    --log-every 20
else
  echo "[BMR slim:${VARIANT}] reuse checkpoint ${CKPT}" | tee -a "${LOG}"
fi

run_logged python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type "${METHOD}" \
  --adversary-task "${ADV}" \
  --target-task "${TARGET}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-effectiveness True \
  --test-utility

if [ ! -f "${ADA_PATH}" ]; then
  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type "${METHOD}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --adamerging-epochs "${ADAMERGING_EPOCHS}" \
    --test-effectiveness True \
    --test-utility
else
  echo "[BMR slim:${VARIANT}] reuse AdaMerging ${ADA_PATH}" | tee -a "${LOG}"
fi

echo "[BMR slim:${VARIANT}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
