#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID TASK_NAME" >&2
  echo "TASK_NAME: gain_only | no_background | fixed_background | fixed_residual | both_fixed" >&2
  exit 2
fi

GPU_ID="$1"
TASK_NAME="$2"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET="${TARGET:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH="${PATCH:-22}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
LR="${LR:-5e-7}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

BASE_TRIGGER="trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
OUT_ROOT="${OUT_ROOT:-analysis/bmr_controls}"
RESULT_ROOT="${RESULT_ROOT:-results/bmr_controls}"

mkdir -p logs "${OUT_ROOT}" "${RESULT_ROOT}"
LOG="logs/bmr_control_${TASK_NAME}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[control:${TASK_NAME}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

METHOD=""
ABLATION_MODE="full_bmr"
ALPHA_MIN="0.2"
ALPHA_MAX="1.0"
ETA_MIN="0.2"
ETA_MAX="1.0"

case "${TASK_NAME}" in
  gain_only)
    METHOD="KDR_DTK_BMR_GAINONLY"
    ABLATION_MODE="gain_only"
    ;;
  no_background)
    METHOD="KDR_DTK_BMR_NO_BG"
    ETA_MIN="0.0"
    ETA_MAX="0.0"
    ;;
  fixed_background)
    METHOD="KDR_DTK_BMR_FIXETA"
    ETA_MIN="0.6"
    ETA_MAX="0.6"
    ;;
  fixed_residual)
    METHOD="KDR_DTK_BMR_FIXALPHA"
    ALPHA_MIN="0.6"
    ALPHA_MAX="0.6"
    ;;
  both_fixed)
    METHOD="KDR_DTK_BMR_BOTHFIX"
    ALPHA_MIN="0.6"
    ALPHA_MAX="0.6"
    ETA_MIN="0.6"
    ETA_MAX="0.6"
    ;;
  *)
    echo "unknown TASK_NAME=${TASK_NAME}" >&2
    exit 2
    ;;
esac

CKPT="checkpoints/${MODEL}/${ADV}_${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"

echo "============================================================" | tee "${LOG}"
echo "[control:${TASK_NAME}] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee -a "${LOG}"
echo "[control:${TASK_NAME}] method=${METHOD} mode=${ABLATION_MODE}" | tee -a "${LOG}"
echo "[control:${TASK_NAME}] alpha=[${ALPHA_MIN},${ALPHA_MAX}] eta=[${ETA_MIN},${ETA_MAX}]" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

python3 -m py_compile \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  2>&1 | tee -a "${LOG}"

test -f "${BASE_TRIGGER}"
test -f "${CLEAN_CKPT}"

if [ ! -f "${CKPT}" ]; then
  run_logged python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" \
    --data-location ./data \
    --adversary-task "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --trigger-path "${BASE_TRIGGER}" \
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
    --eta-min "${ETA_MIN}" \
    --eta-max "${ETA_MAX}" \
    --drift-rho 0.25 \
    --drift-scope last2 \
    --clean-weight 1.0 \
    --bd-weight 1.0 \
    --reserve-weight 1.0 \
    --residual-weight 0.0 \
    --seed "${SEED}" \
    --log-every 20
else
  echo "[control:${TASK_NAME}] reuse checkpoint ${CKPT}" | tee -a "${LOG}"
fi

# TA + TIES
run_logged python3 src/eval_submerge.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type "${METHOD}" \
  --trigger-source KDR_DTK \
  --adversary-task "${ADV}" \
  --target-task "${TARGET}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH}" \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-utility \
  --out-dir "${RESULT_ROOT}"

# RegMean
run_logged python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type "${METHOD}" \
  --adversary-task "${ADV}" \
  --target-task "${TARGET}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH}" \
  --alpha 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-effectiveness True \
  --test-utility

# AdaMerging. Reuse its learned coefficients when already present.
ADA_PATH="ada/${MODEL}/${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_Epoch_${ADAMERGING_EPOCHS}.pt"
if [ ! -f "${ADA_PATH}" ]; then
  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type "${METHOD}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --adamerging-epochs "${ADAMERGING_EPOCHS}" \
    --test-effectiveness True \
    --test-utility
else
  echo "[control:${TASK_NAME}] reuse AdaMerging ${ADA_PATH}" | tee -a "${LOG}"
  # The repository's AdaMerging entry point usually evaluates while training.
  # If the local version has an eval-only option, the agent should invoke it;
  # otherwise retain the previous evaluation log together with ADA_PATH.
fi

echo "[control:${TASK_NAME}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
