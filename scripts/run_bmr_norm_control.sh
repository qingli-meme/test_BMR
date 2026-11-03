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
TARGET="${TARGET:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH="${PATCH:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

ROOT="checkpoints/${MODEL}"
BASE="${ROOT}/zeroshot.pt"
CLEAN="${ROOT}/${ADV}/finetuned.pt"
ORD="${ROOT}/${ADV}_KDR_DTK_BMR_ORD_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"
BG="${ROOT}/${ADV}_KDR_DTK_BMR_BG_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"
BMR="${ROOT}/${ADV}_KDR_DTK_BMR_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"

OUT_ROOT="analysis/bmr_controls/norm_control"
RESULT_ROOT="results/bmr_controls"
mkdir -p "${OUT_ROOT}" "${RESULT_ROOT}" logs
LOG="logs/bmr_norm_control_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[norm-control] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

for f in "${BASE}" "${CLEAN}" "${ORD}" "${BG}" "${BMR}"; do test -f "$f"; done

make_one() {
  local src="$1" ref="$2" method="$3"
  local out_dir="${ROOT}/${ADV}_${method}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}"
  local out="${out_dir}/finetuned.pt"
  mkdir -p "${out_dir}"
  if [ ! -f "${out}" ]; then
    run_logged python3 src/make_residual_norm_matched_checkpoint.py \
      --base "${BASE}" --source "${src}" --reference "${ref}" --clean "${CLEAN}" \
      --output "${out}" --summary "${OUT_ROOT}/${method}.json"
  else
    echo "[norm-control] reuse ${out}" | tee -a "${LOG}"
  fi
}

# Two direct tests against the 'larger residual' alternative explanation.
make_one "${ORD}" "${BMR}" "KDR_DTK_BMR_ORD_NORMMATCH"
make_one "${BG}" "${BMR}" "KDR_DTK_BMR_BG_NORMMATCH"
# Reverse control: shrink Full BMR to the norm of Background Augmentation.
make_one "${BMR}" "${BG}" "KDR_DTK_BMR_BMR_NORMMATCH_BG"

eval_method() {
  local method="$1"
  run_logged python3 src/eval_submerge.py \
    --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" --attack-type "${method}" --trigger-source KDR_DTK \
    --adversary-task "${ADV}" --target-task "${TARGET}" --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" --merge-methods ta,ties --scaling-coef 0.3 \
    --batch-size "${BATCH_SIZE}" --test-utility --out-dir "${RESULT_ROOT}"

  run_logged python3 src/main_regmean_badmergingon.py \
    --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
    --attack-type "${method}" --adversary-task "${ADV}" --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" --patch-size "${PATCH}" --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" --test-effectiveness True --test-utility

  local ada="ada/${MODEL}/${method}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_Epoch_${ADAMERGING_EPOCHS}.pt"
  if [ ! -f "${ada}" ]; then
    run_logged python3 src/main_adamerging_badmergingon.py \
      --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
      --attack-type "${method}" --adversary-task "${ADV}" --target-task "${TARGET}" \
      --target-cls "${TARGET_CLS}" --patch-size "${PATCH}" --alpha 0.3 \
      --batch-size "${BATCH_SIZE}" --adamerging-epochs "${ADAMERGING_EPOCHS}" \
      --test-effectiveness True --test-utility
  fi
}

for method in \
  KDR_DTK_BMR_ORD_NORMMATCH \
  KDR_DTK_BMR_BG_NORMMATCH \
  KDR_DTK_BMR_BMR_NORMMATCH_BG; do
  eval_method "${method}"
done

echo "[norm-control] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
