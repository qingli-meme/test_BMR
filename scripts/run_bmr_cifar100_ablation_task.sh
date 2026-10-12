#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID TASK_NAME" >&2
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

mkdir -p logs results/bmr_ablation_full analysis/bmr_ablation_full
LOG="logs/bmr_ablation_${TASK_NAME}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

BASE_TRIGGER="trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"

run_logged() {
  echo "[BMR ablation:${TASK_NAME}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

set_config() {
  METHOD=""
  TRIGGER="${BASE_TRIGGER}"
  ABLATION_MODE="full_bmr"
  ALPHA_MIN="0.2"
  ALPHA_MAX="1.0"
  ETA_MIN="0.2"
  ETA_MAX="1.0"
  ALIGN_WEIGHT="1.0"
  EMIT_WEIGHT="0.1"
  DORM_WEIGHT="1.0"
  NEED_TRIGGER="0"

  case "${TASK_NAME}" in
    train_ordinary)
      METHOD="KDR_DTK_BMR_ORD"
      ABLATION_MODE="plain"
      ;;
    train_background_aug)
      METHOD="KDR_DTK_BMR_BG"
      ABLATION_MODE="background_attack"
      ;;
    train_absolute_margin)
      METHOD="KDR_DTK_BMR_ABS"
      ABLATION_MODE="absolute_margin"
      ;;
    train_no_scale_variation)
      METHOD="KDR_DTK_BMR_NOSCALE"
      ABLATION_MODE="full_bmr"
      ALPHA_MIN="0.6"
      ALPHA_MAX="0.6"
      ETA_MIN="0.6"
      ETA_MAX="0.6"
      ;;
    train_full_bmr)
      METHOD="KDR_DTK_BMR"
      ;;
    trigger_no_target_alignment)
      METHOD="KDR_DTK_BMR_TRIG_NOALIGN"
      TRIGGER="trigger/${MODEL}/KDR_DTK_TRIG_NOALIGN_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
      ALIGN_WEIGHT="0.0"
      NEED_TRIGGER="1"
      ;;
    trigger_no_signal)
      METHOD="KDR_DTK_BMR_TRIG_NOSIG"
      TRIGGER="trigger/${MODEL}/KDR_DTK_TRIG_NOSIG_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
      EMIT_WEIGHT="0.0"
      NEED_TRIGGER="1"
      ;;
    trigger_no_dormancy)
      METHOD="KDR_DTK_BMR_TRIG_NODORM"
      TRIGGER="trigger/${MODEL}/KDR_DTK_TRIG_NODORM_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
      DORM_WEIGHT="0.0"
      NEED_TRIGGER="1"
      ;;
    trigger_full)
      METHOD="KDR_DTK_BMR"
      ;;
    *)
      echo "unknown TASK_NAME=${TASK_NAME}" >&2
      exit 2
      ;;
  esac
}

eval_all_four() {
  local method="$1"
  local trigger_source="$2"

  run_logged python3 src/eval_submerge.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${method}" \
    --trigger-source "${trigger_source}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --merge-methods ta,ties \
    --scaling-coef 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-utility \
    --out-dir ./results/bmr_ablation_full

  run_logged python3 src/main_regmean_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type "${method}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-effectiveness True \
    --test-utility

  local ada_path="ada/${MODEL}/${method}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_Epoch_${ADAMERGING_EPOCHS}.pt"
  if [ -f "${ada_path}" ]; then
    echo "[BMR ablation:${TASK_NAME}] reuse AdaMerging ${ada_path}" | tee -a "${LOG}"
    return 0
  fi

  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --attack-type "${method}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --adamerging-epochs "${ADAMERGING_EPOCHS}" \
    --test-effectiveness True \
    --test-utility
}

set_config

echo "============================================================" | tee "${LOG}"
echo "[BMR ablation:${TASK_NAME}] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee -a "${LOG}"
echo "[BMR ablation:${TASK_NAME}] method=${METHOD} trigger=${TRIGGER}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  2>&1 | tee -a "${LOG}"

test -f "${CLEAN_CKPT}"
if [ "${NEED_TRIGGER}" = "1" ] && [ ! -f "${TRIGGER}" ]; then
  run_logged python3 src/optimize_dormant_target_key_patch.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --dataset "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --layer-name model.visual.transformer.resblocks.11.ln_2 \
    --pool cls \
    --epochs 3 \
    --batch-size "${BATCH_SIZE}" \
    --lr 5e-2 \
    --emit-eps 1.0 \
    --align-weight "${ALIGN_WEIGHT}" \
    --emit-weight "${EMIT_WEIGHT}" \
    --dorm-kappa 0.0 \
    --dorm-weight "${DORM_WEIGHT}" \
    --amp-weight 1e-4 \
    --tv-weight 1e-3 \
    --patch-init-std 0.05 \
    --patch-min -2.5 \
    --patch-max 2.5 \
    --clean-checkpoint "${CLEAN_CKPT}" \
    --save-path "${TRIGGER}" \
    --out-dir "./analysis/bmr_ablation_full/${TASK_NAME}/patch" \
    --seed "${SEED}" \
    --log-every 20
else
  test -f "${TRIGGER}"
  echo "[BMR ablation:${TASK_NAME}] reuse trigger ${TRIGGER}" | tee -a "${LOG}"
fi

CKPT="checkpoints/${MODEL}/${ADV}_${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"
if [ ! -f "${CKPT}" ]; then
  run_logged python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" \
    --data-location ./data \
    --adversary-task "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
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
  echo "[BMR ablation:${TASK_NAME}] reuse checkpoint ${CKPT}" | tee -a "${LOG}"
fi

case "${METHOD}" in
  KDR_DTK_BMR_TRIG_NOALIGN) TRIGGER_SOURCE="KDR_DTK_TRIG_NOALIGN" ;;
  KDR_DTK_BMR_TRIG_NOSIG) TRIGGER_SOURCE="KDR_DTK_TRIG_NOSIG" ;;
  KDR_DTK_BMR_TRIG_NODORM) TRIGGER_SOURCE="KDR_DTK_TRIG_NODORM" ;;
  *) TRIGGER_SOURCE="KDR_DTK" ;;
esac

eval_all_four "${METHOD}" "${TRIGGER_SOURCE}"

echo "[BMR ablation:${TASK_NAME}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
