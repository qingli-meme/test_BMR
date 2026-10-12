#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID PATCH_SIZE" >&2
  exit 2
fi

GPU_ID="$1"
PATCH="$2"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET="${TARGET:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
LR="${LR:-5e-7}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"
METHOD="KDR_DTK_BMR"

mkdir -p logs results/bmr_trigger_size analysis/bmr_trigger_size
LOG="logs/bmr_trigger_size_L${PATCH}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
TRIGGER="trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
CKPT="checkpoints/${MODEL}/${ADV}_${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"

run_logged() {
  echo "[BMR trigger-size:L${PATCH}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

echo "============================================================" | tee "${LOG}"
echo "[BMR trigger-size:L${PATCH}] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  2>&1 | tee -a "${LOG}"

test -f "${CLEAN_CKPT}"
if [ ! -f "${TRIGGER}" ]; then
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
    --align-weight 1.0 \
    --emit-weight 0.1 \
    --dorm-kappa 0.0 \
    --dorm-weight 1.0 \
    --amp-weight 1e-4 \
    --tv-weight 1e-3 \
    --patch-init-std 0.05 \
    --patch-min -2.5 \
    --patch-max 2.5 \
    --clean-checkpoint "${CLEAN_CKPT}" \
    --save-path "${TRIGGER}" \
    --out-dir "./analysis/bmr_trigger_size/L${PATCH}/patch" \
    --seed "${SEED}" \
    --log-every 20
else
  echo "[BMR trigger-size:L${PATCH}] reuse trigger ${TRIGGER}" | tee -a "${LOG}"
fi

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
    --ablation-mode full_bmr \
    --batch-size "${BATCH_SIZE}" \
    --bd-batch-size "${BD_BATCH_SIZE}" \
    --lr "${LR}" \
    --wd 0.05 \
    --grad-clip 1.0 \
    --trainable-scope all \
    --alpha-min 0.2 \
    --alpha-max 1.0 \
    --eta-min 0.2 \
    --eta-max 1.0 \
    --drift-rho 0.25 \
    --drift-scope last2 \
    --clean-weight 1.0 \
    --bd-weight 1.0 \
    --reserve-weight 1.0 \
    --residual-weight 0.0 \
    --seed "${SEED}" \
    --log-every 20
else
  echo "[BMR trigger-size:L${PATCH}] reuse checkpoint ${CKPT}" | tee -a "${LOG}"
fi

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
  --out-dir ./results/bmr_trigger_size

run_logged python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type "${METHOD}" \
  --adversary-task "${ADV}" \
  --target-task "${TARGET}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH}" \
  --alpha 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-effectiveness True \
  --test-utility

ADA_PATH="ada/${MODEL}/${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_Epoch_${ADAMERGING_EPOCHS}.pt"
if [ -f "${ADA_PATH}" ]; then
  echo "[BMR trigger-size:L${PATCH}] reuse AdaMerging ${ADA_PATH}" | tee -a "${LOG}"
else
  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
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
fi

echo "[BMR trigger-size:L${PATCH}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
