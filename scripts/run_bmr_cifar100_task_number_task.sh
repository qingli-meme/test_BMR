#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID TASK_NUMBER" >&2
  exit 2
fi

GPU_ID="$1"
TASK_NUMBER="$2"
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
BMR_METHOD="KDR_DTK_BMR"

case "${TASK_NUMBER}" in
  2) EXAM_DATASETS="CIFAR100,GTSRB" ;;
  4) EXAM_DATASETS="CIFAR100,GTSRB,EuroSAT,Cars" ;;
  6) EXAM_DATASETS="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS" ;;
  *)
    echo "unsupported TASK_NUMBER=${TASK_NUMBER}; expected 2, 4, or 6" >&2
    exit 2
    ;;
esac

mkdir -p logs results/bmr_task_number results/bmr_task_number_clean
LOG="logs/bmr_task_number_N${TASK_NUMBER}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[BMR task-number:N${TASK_NUMBER}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

eval_ta_ties() {
  local attack_type="$1"
  local out_dir="$2"
  local trigger_source="$3"
  local extra_effectiveness=()
  if [ "${attack_type}" = "Clean" ]; then
    extra_effectiveness=(--no-effectiveness)
  fi
  run_logged python3 src/eval_submerge.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${attack_type}" \
    --trigger-source "${trigger_source}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --merge-methods ta,ties \
    --scaling-coef 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-utility \
    "${extra_effectiveness[@]}" \
    --out-dir "${out_dir}"
}

eval_regmean_adamerging() {
  local attack_type="$1"
  local ada_tag="tasknum_N${TASK_NUMBER}"
  run_logged python3 src/main_regmean_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${attack_type}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-effectiveness True \
    --test-utility

  local ada_path="ada/${MODEL}/${attack_type}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_${ada_tag}_Epoch_${ADAMERGING_EPOCHS}.pt"
  if [ "${attack_type}" = "Clean" ]; then
    ada_path="ada/${MODEL}/Clean_${ada_tag}_Epoch_${ADAMERGING_EPOCHS}.pt"
  fi
  if [ -f "${ada_path}" ]; then
    echo "[BMR task-number:N${TASK_NUMBER}] reuse AdaMerging ${ada_path}" | tee -a "${LOG}"
    return 0
  fi

  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${attack_type}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --adamerging-epochs "${ADAMERGING_EPOCHS}" \
    --ada_name "${ada_tag}" \
    --test-effectiveness True \
    --test-utility
}

echo "============================================================" | tee "${LOG}"
echo "[BMR task-number:N${TASK_NUMBER}] start=$(date --iso-8601=seconds) gpu=${GPU_ID} datasets=${EXAM_DATASETS}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

python3 -m py_compile \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py \
  2>&1 | tee -a "${LOG}"

test -f "checkpoints/${MODEL}/${ADV}_${BMR_METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"
test -f "trigger/${MODEL}/KDR_DTK_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"

eval_ta_ties "${BMR_METHOD}" ./results/bmr_task_number KDR_DTK
eval_regmean_adamerging "${BMR_METHOD}"

eval_ta_ties Clean ./results/bmr_task_number_clean fixed
eval_regmean_adamerging Clean

echo "[BMR task-number:N${TASK_NUMBER}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
