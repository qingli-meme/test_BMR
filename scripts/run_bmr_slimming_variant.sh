#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID VARIANT [SEED]" >&2
  exit 2
fi

GPU_ID="$1"
VARIANT="$2"
SEED="${3:-${SEED:-2026}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH="${PATCH:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
LR="${LR:-5e-7}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"

TRIGGER="trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"

ABLATION_MODE=""
ALPHA_MIN="0.2"
ALPHA_MAX="1.0"
DRIFT_RHO="0.0"
METHOD=""

case "${VARIANT}" in
  ordinary)
    METHOD="KDR_DTK_BMR_SLIM_ORD"
    ABLATION_MODE="plain"
    ALPHA_MIN="1.0"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.0"
    ;;
  scale_var)
    METHOD="KDR_DTK_BMR_SLIM_SCALE"
    ABLATION_MODE="background_attack"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.0"
    ;;
  absolute_margin)
    METHOD="KDR_DTK_BMR_SLIM_ABS"
    ABLATION_MODE="absolute_margin"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.0"
    ;;
  bmr_fixed_alpha)
    METHOD="KDR_DTK_BMR_SLIM_FIXED"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.6"
    ALPHA_MAX="0.6"
    DRIFT_RHO="0.0"
    ;;
  bmr_no_drift)
    METHOD="KDR_DTK_BMR_SLIM_NODRIFT"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.0"
    ;;
  bmr_with_drift)
    METHOD="KDR_DTK_BMR_SLIM_DRIFT"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.25"
    ;;
  bmr_no_drift_s2027|bmr_no_drift_s2028)
    METHOD="KDR_DTK_BMR_SLIM_NODRIFT_S${SEED}"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.0"
    ;;
  bmr_with_drift_s2027|bmr_with_drift_s2028)
    METHOD="KDR_DTK_BMR_SLIM_DRIFT_S${SEED}"
    ABLATION_MODE="full_bmr"
    ALPHA_MIN="0.2"
    ALPHA_MAX="1.0"
    DRIFT_RHO="0.25"
    ;;
  *)
    echo "unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

OUT="checkpoints/${MODEL}/${ADV}_${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"
mkdir -p logs analysis/bmr_slimming_phase1 analysis/bmr_slimming_confirm
LOG="logs/bmr_slim_${VARIANT}_s${SEED}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee "${LOG}"
echo "[BMR Slim] variant=${VARIANT}" | tee -a "${LOG}"
echo "[BMR Slim] method=${METHOD}" | tee -a "${LOG}"
echo "[BMR Slim] seed=${SEED}" | tee -a "${LOG}"
echo "[BMR Slim] alpha=[${ALPHA_MIN},${ALPHA_MAX}]" | tee -a "${LOG}"
echo "[BMR Slim] drift_rho=${DRIFT_RHO}" | tee -a "${LOG}"
echo "[BMR Slim] ablation_mode=${ABLATION_MODE}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

test -f "${TRIGGER}"
test -f "${CLEAN_CKPT}"

python3 -m py_compile src/finetune_kdr_dtk_bmr.py 2>&1 | tee -a "${LOG}"

if [ -f "${OUT}" ]; then
  echo "[BMR Slim] reuse ${OUT}" | tee -a "${LOG}"
  exit 0
fi

CMD=(
  python3 src/finetune_kdr_dtk_bmr.py
  --model "${MODEL}"
  --data-location ./data
  --adversary-task "${ADV}"
  --target-cls "${TARGET_CLS}"
  --patch-size "${PATCH}"
  --trigger-path "${TRIGGER}"
  --init-checkpoint "${CLEAN_CKPT}"
  --save-root ./checkpoints
  --method-name "${METHOD}"
  --ablation-mode "${ABLATION_MODE}"
  --batch-size "${BATCH_SIZE}"
  --bd-batch-size "${BD_BATCH_SIZE}"
  --lr "${LR}"
  --wd 0.05
  --grad-clip 1.0
  --trainable-scope all
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --eta-min 0.2
  --eta-max 1.0
  --drift-rho "${DRIFT_RHO}"
  --drift-scope last2
  --clean-weight 1.0
  --bd-weight 1.0
  --reserve-weight 1.0
  --residual-weight 0.0
  --seed "${SEED}"
  --log-every 20
)

if [ "${MAX_TRAIN_BATCHES}" != "0" ]; then
  CMD+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi

echo "[BMR Slim] ${CMD[*]}" | tee -a "${LOG}"
"${CMD[@]}" 2>&1 | tee -a "${LOG}"

test -f "${OUT}"
echo "[BMR Slim] done ${OUT}" | tee -a "${LOG}"
