#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
TASK="${TASK:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
SEED="${SEED:-2026}"
TRIGGER="trigger/${MODEL}/KDR_DTK_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${TASK}/finetuned.pt"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

mkdir -p logs results/bmr_sensitivity
LOG="logs/bmr_cifar100_drift_sensitivity_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[BMR drift sensitivity] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

test -f "${TRIGGER}"
test -f "${CLEAN_CKPT}"

python3 -m py_compile src/finetune_kdr_dtk_bmr.py src/eval_submerge.py 2>&1 | tee -a "${LOG}"

declare -a SPECS=(
  "KDR_DTK_BMR_G0 0.0"
  "KDR_DTK_BMR_G01 0.1"
  "KDR_DTK_BMR_G025 0.25"
  "KDR_DTK_BMR_G05 0.5"
  "KDR_DTK_BMR_G10 1.0"
)

for spec in "${SPECS[@]}"; do
  read -r METHOD DRIFT_RHO <<<"${spec}"
  CKPT="checkpoints/${MODEL}/${TASK}_${METHOD}_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"
  echo "============================================================" | tee -a "${LOG}"
  echo "[BMR drift sensitivity] method=${METHOD} drift_rho=${DRIFT_RHO} start=$(date --iso-8601=seconds)" | tee -a "${LOG}"
  echo "============================================================" | tee -a "${LOG}"

  if [ ! -f "${CKPT}" ]; then
    run_logged python3 src/finetune_kdr_dtk_bmr.py \
      --model "${MODEL}" \
      --data-location ./data \
      --adversary-task "${TASK}" \
      --target-cls "${TARGET_CLS}" \
      --patch-size "${PATCH_SIZE}" \
      --trigger-path "${TRIGGER}" \
      --init-checkpoint "${CLEAN_CKPT}" \
      --save-root ./checkpoints \
      --method-name "${METHOD}" \
      --batch-size "${BATCH_SIZE}" \
      --bd-batch-size "${BD_BATCH_SIZE}" \
      --lr 5e-7 \
      --wd 0.05 \
      --grad-clip 1.0 \
      --trainable-scope all \
      --alpha-min 0.2 \
      --alpha-max 1.0 \
      --eta-min 0.2 \
      --eta-max 1.0 \
      --drift-rho "${DRIFT_RHO}" \
      --drift-scope last2 \
      --clean-weight 1.0 \
      --bd-weight 1.0 \
      --reserve-weight 1.0 \
      --residual-weight 0.0 \
      --seed "${SEED}" \
      --log-every 20
  else
    echo "[BMR drift sensitivity] reuse checkpoint ${CKPT}" | tee -a "${LOG}"
  fi

  run_logged python3 src/eval_submerge.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${METHOD}" \
    --trigger-source KDR_DTK \
    --adversary-task "${TASK}" \
    --target-task "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --merge-methods ta,ties \
    --scaling-coef 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-utility \
    --out-dir ./results/bmr_sensitivity
done

echo "[BMR drift sensitivity] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
