#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID TASK" >&2
  exit 2
fi
GPU_ID="$1"
TASK="$2"

export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU_ID}")"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL=ViT-B-32
TARGET_CLS=1
PATCH=22
SEED=2026
BASE6="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS"
IMG6="ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS"
if [[ "${TASK}" == "ImageNet100" ]]; then EXAM="${IMG6}"; else EXAM="${BASE6}"; fi

mkdir -p logs/paper_refresh results/paper_refresh/generalization
LOG="logs/paper_refresh/generalization_${TASK}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

CLEAN="checkpoints/${MODEL}/${TASK}/finetuned.pt"
TRIGGER="trigger/${MODEL}/KDR_DTK_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
OUTDIR="checkpoints/${MODEL}/${TASK}_KDR_DTK_BMR_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH}"
CKPT="${OUTDIR}/finetuned.pt"
MARKER="${OUTDIR}/.paper_refresh_010c97c"

require_file "${CLEAN}"
require_file "${TRIGGER}"

if [[ "${TASK}" == "CIFAR100" ]]; then
  require_file "${CKPT}"
  echo "[generalization] reuse canonical CIFAR100 checkpoint"
else
  if [[ ! -f "${MARKER}" ]]; then
    if [[ -e "${OUTDIR}" ]]; then
      archive_once "${OUTDIR}" "$(paper_refresh_archive_dir "$(pwd)")/checkpoints/${MODEL}"
    fi
    python3 src/finetune_kdr_dtk_bmr.py \
      --model "${MODEL}" \
      --data-location ./data \
      --adversary-task "${TASK}" \
      --target-cls "${TARGET_CLS}" \
      --patch-size "${PATCH}" \
      --trigger-path "${TRIGGER}" \
      --init-checkpoint "${CLEAN}" \
      --save-root ./checkpoints \
      --method-name KDR_DTK_BMR \
      --ablation-mode full_bmr \
      --batch-size 128 \
      --bd-batch-size 64 \
      --lr 5e-7 \
      --wd 0.05 \
      --grad-clip 1.0 \
      --trainable-scope all \
      --alpha-min 0.2 \
      --alpha-max 1.0 \
      --clean-weight 1.0 \
      --bd-weight 1.0 \
      --reserve-weight 1.0 \
      --residual-weight 0.0 \
      --seed "${SEED}" \
      --log-every 20
    require_file "${CKPT}"
    touch "${MARKER}"
  else
    echo "[generalization] marker found; reuse refreshed ${TASK} checkpoint"
  fi
fi

echo "[generalization] TA/TIES ${TASK}"
python3 src/eval_submerge.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets "${EXAM}" --attack-type KDR_DTK_BMR --trigger-source KDR_DTK \
  --adversary-task "${TASK}" --target-task "${TASK}" --target-cls 1 --patch-size 22 \
  --merge-methods ta,ties --scaling-coef 0.3 --batch-size 128 --test-utility \
  --out-dir "results/paper_refresh/generalization/${TASK}"

echo "[generalization] RegMean ${TASK}"
python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets "${EXAM}" --attack-type KDR_DTK_BMR \
  --adversary-task "${TASK}" --target-task "${TASK}" --target-cls 1 --patch-size 22 \
  --alpha 0.3 --batch-size 128 --test-effectiveness True --test-utility

echo "[generalization] AdaMerging ${TASK}"
python3 src/main_adamerging_badmergingon.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets "${EXAM}" --attack-type KDR_DTK_BMR \
  --adversary-task "${TASK}" --target-task "${TASK}" --target-cls 1 --patch-size 22 \
  --alpha 0.3 --batch-size 128 --adamerging-epochs 500 \
  --ada_name "_paper_refresh_010c97c.pt" \
  --test-effectiveness True --test-utility

echo "[generalization] COMPLETE task=${TASK}"
