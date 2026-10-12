#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-0}"
WAIT_FOR_DATA="${WAIT_FOR_DATA:-0}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
TASK="${TASK:-ImageNet100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-22}"
SEED="${SEED:-2026}"
DATA_LOCATION="${DATA_LOCATION:-./data}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
BMR_LR="${BMR_LR:-5e-7}"
CLEAN_LR="${CLEAN_LR:-1e-5}"
CLEAN_EPOCHS="${CLEAN_EPOCHS:-3}"
EXAM_DATASETS="${EXAM_DATASETS:-ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

mkdir -p logs results/bmr_imagenet100 analysis/bmr_imagenet100

log_path="logs/bmr_on_imagenet100_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
trigger_path="trigger/${MODEL}/KDR_DTK_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}.npy"
clean_ckpt="${CKPT_DIR}/${MODEL}/${TASK}/finetuned.pt"
bmr_ckpt="${CKPT_DIR}/${MODEL}/${TASK}_KDR_DTK_BMR_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"

run_logged() {
  echo "[BMR-ImageNet100] $*" | tee -a "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

has_imagenet100_data() {
  [ -d "${DATA_LOCATION}/ImageNet100/train" ] && [ -d "${DATA_LOCATION}/ImageNet100/val" ]
}

echo "============================================================" | tee "${log_path}"
echo "[BMR-ImageNet100] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee -a "${log_path}"
echo "[BMR-ImageNet100] exam_datasets=${EXAM_DATASETS}" | tee -a "${log_path}"
echo "============================================================" | tee -a "${log_path}"

while ! has_imagenet100_data; do
  echo "[BMR-ImageNet100] missing ${DATA_LOCATION}/ImageNet100/{train,val}" | tee -a "${log_path}"
  if [ "${WAIT_FOR_DATA}" != "1" ]; then
    exit 3
  fi
  echo "[BMR-ImageNet100] waiting ${WAIT_INTERVAL}s for data..." | tee -a "${log_path}"
  sleep "${WAIT_INTERVAL}"
done

run_logged python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py

if [ ! -f "${clean_ckpt}" ]; then
  run_logged python3 - <<PY
import os
import sys

sys.path.append(os.path.abspath("."))

from src.args import parse_arguments
from src.finetune_clean import finetune

args = parse_arguments()
args.lr = float("${CLEAN_LR}")
args.epochs = int("${CLEAN_EPOCHS}")
args.data_location = "${DATA_LOCATION}"
args.dataset = "${TASK}"
args.batch_size = int("${BATCH_SIZE}")
args.model = "${MODEL}"
args.save = os.path.join("${CKPT_DIR}", "${MODEL}")
args.cache_dir = ""
args.openclip_cachedir = "./open_clip"

finetune(args)
PY
else
  echo "[BMR-ImageNet100] reuse clean checkpoint ${clean_ckpt}" | tee -a "${log_path}"
fi

if [ ! -f "${trigger_path}" ]; then
  run_logged python3 src/optimize_dormant_target_key_patch.py \
    --model "${MODEL}" \
    --ckpt-dir "${CKPT_DIR}" \
    --data-location "${DATA_LOCATION}" \
    --dataset "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --layer-name model.visual.transformer.resblocks.11.ln_2 \
    --pool cls \
    --epochs 3 \
    --batch-size "${BATCH_SIZE}" \
    --lr 5e-2 \
    --emit-eps 1.0 \
    --emit-weight 0.1 \
    --dorm-kappa 0.0 \
    --dorm-weight 1.0 \
    --amp-weight 1e-4 \
    --tv-weight 1e-3 \
    --patch-init-std 0.05 \
    --patch-min -2.5 \
    --patch-max 2.5 \
    --clean-checkpoint "${clean_ckpt}" \
    --save-path "${trigger_path}" \
    --out-dir "./analysis/bmr_imagenet100/patch" \
    --seed "${SEED}" \
    --log-every 20
else
  echo "[BMR-ImageNet100] reuse trigger ${trigger_path}" | tee -a "${log_path}"
fi

if [ ! -f "${bmr_ckpt}" ]; then
  run_logged python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" \
    --data-location "${DATA_LOCATION}" \
    --adversary-task "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --trigger-path "${trigger_path}" \
    --init-checkpoint "${clean_ckpt}" \
    --save-root "${CKPT_DIR}" \
    --method-name KDR_DTK_BMR \
    --batch-size "${BATCH_SIZE}" \
    --bd-batch-size "${BD_BATCH_SIZE}" \
    --lr "${BMR_LR}" \
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
  echo "[BMR-ImageNet100] reuse BMR checkpoint ${bmr_ckpt}" | tee -a "${log_path}"
fi

run_logged python3 src/eval_submerge.py \
  --model "${MODEL}" \
  --ckpt-dir "${CKPT_DIR}" \
  --data-location "${DATA_LOCATION}" \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type KDR_DTK_BMR \
  --trigger-source KDR_DTK \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --merge-methods ta,ties \
  --scaling-coef 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-utility \
  --out-dir ./results/bmr_imagenet100

echo "[BMR-ImageNet100] done=$(date --iso-8601=seconds)" | tee -a "${log_path}"
