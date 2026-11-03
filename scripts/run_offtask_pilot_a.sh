#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-${GPU_ID:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
ADV="${ADV:-CIFAR100}"
TARGET_TASK="${TARGET_TASK:-Cars}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-28}"
NUM_SHADOW_DATA="${NUM_SHADOW_DATA:-10}"
NUM_SHADOW_CLASSES="${NUM_SHADOW_CLASSES:-300}"
BADMERGING_ALPHA="${BADMERGING_ALPHA:-5}"
SEED="${SEED:-2026}"

BATCH_SIZE="${BATCH_SIZE:-128}"
BD_BATCH_SIZE="${BD_BATCH_SIZE:-64}"
BMR_LR="${BMR_LR:-5e-7}"
SCALING_COEF="${SCALING_COEF:-0.3}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
MARGIN_MAX_BATCHES="${MARGIN_MAX_BATCHES:-0}"
RUN_REGMEAN="${RUN_REGMEAN:-1}"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

METHOD="BMR_OFF_PILOT_A_TO_${TARGET_TASK}"
ATTACK_SUFFIX="${METHOD}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}"
BM_SUFFIX="Off_${TARGET_TASK}_Tgt_${TARGET_CLS}_SD_${NUM_SHADOW_DATA}_SC_${NUM_SHADOW_CLASSES}_L_${PATCH_SIZE}"

TRIGGER="trigger/${MODEL}/Off_${TARGET_TASK}_Tgt_${TARGET_CLS}_SD_${NUM_SHADOW_DATA}_SC_${NUM_SHADOW_CLASSES}_L_${PATCH_SIZE}.npy"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
BM_CKPT="checkpoints/${MODEL}/${ADV}_${BM_SUFFIX}/finetuned.pt"
PILOT_CKPT="checkpoints/${MODEL}/${ADV}_${ATTACK_SUFFIX}/finetuned.pt"

mkdir -p logs results/offtask_pilot_a analysis/offtask_pilot_a shadow_head/"${MODEL}"
LOG="logs/offtask_pilot_a_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[OffTask Pilot-A] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

echo "============================================================" | tee "${LOG}"
echo "[OffTask Pilot-A] start=$(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[OffTask Pilot-A] GPU=${GPU_ID}" | tee -a "${LOG}"
echo "[OffTask Pilot-A] ${ADV} -> ${TARGET_TASK}, target=${TARGET_CLS}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

run_logged python3 tools/make_offtask_pilot_a_train.py

run_logged python3 -m py_compile \
  src/finetune_bmr_offtask_pilot_a.py \
  src/eval_offtask_pilot_a.py \
  src/summarize_offtask_pilot_a.py

test -f "checkpoints/${MODEL}/zeroshot.pt"
test -f "${CLEAN_CKPT}"

if [ ! -f "${TRIGGER}" ]; then
  run_logged python3 src/ut_badmergingoff.py \
    --model "${MODEL}" \
    --target-task "${TARGET_TASK}" \
    --target-cls "${TARGET_CLS}" \
    --mask-length "${PATCH_SIZE}" \
    --num-shadow-data "${NUM_SHADOW_DATA}" \
    --num-shadow-classes "${NUM_SHADOW_CLASSES}" \
    --seed "${SEED}"
else
  echo "[OffTask Pilot-A] reuse trigger ${TRIGGER}" | tee -a "${LOG}"
fi
test -f "${TRIGGER}"

if [ ! -f "${BM_CKPT}" ]; then
  run_logged python3 src/finetune_backdoor_badmergingoff.py \
    --model "${MODEL}" \
    --adversary-task "${ADV}" \
    --target-task "${TARGET_TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --alpha "${BADMERGING_ALPHA}" \
    --num-shadow-data "${NUM_SHADOW_DATA}" \
    --num-shadow-classes "${NUM_SHADOW_CLASSES}"
else
  echo "[OffTask Pilot-A] reuse BadMerging-Off checkpoint ${BM_CKPT}" | tee -a "${LOG}"
fi
test -f "${BM_CKPT}"

if [ ! -f "${PILOT_CKPT}" ]; then
  run_logged python3 src/finetune_bmr_offtask_pilot_a.py \
    --model "${MODEL}" \
    --data-location ./data \
    --adversary-task "${ADV}" \
    --target-task "${TARGET_TASK}" \
    --target-cls "${TARGET_CLS}" \
    --num-shadow-classes "${NUM_SHADOW_CLASSES}" \
    --shadow-head-root ./shadow_head \
    --patch-size "${PATCH_SIZE}" \
    --trigger-path "${TRIGGER}" \
    --init-checkpoint "${CLEAN_CKPT}" \
    --save-root ./checkpoints \
    --method-name "${METHOD}" \
    --ablation-mode full_bmr \
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
  echo "[OffTask Pilot-A] reuse Pilot checkpoint ${PILOT_CKPT}" | tee -a "${LOG}"
fi
test -f "${PILOT_CKPT}"

if [ "${RUN_REGMEAN}" = "1" ]; then
  METHODS="ta,ties,regmean"
else
  METHODS="ta,ties"
fi

COMMON_EVAL=(
  --model "${MODEL}"
  --ckpt-dir ./checkpoints
  --data-location ./data
  --exam-datasets "${EXAM_DATASETS}"
  --adversary-task "${ADV}"
  --target-task "${TARGET_TASK}"
  --target-cls "${TARGET_CLS}"
  --trigger-path "${TRIGGER}"
  --patch-size "${PATCH_SIZE}"
  --num-shadow-classes "${NUM_SHADOW_CLASSES}"
  --shadow-head-root ./shadow_head
  --seed "${SEED}"
  --merge-methods "${METHODS}"
  --scaling-coef "${SCALING_COEF}"
  --batch-size "${BATCH_SIZE}"
  --max-eval-batches "${MAX_EVAL_BATCHES}"
  --margin-max-batches "${MARGIN_MAX_BATCHES}"
  --test-utility
  --out-dir ./results/offtask_pilot_a
)

run_logged python3 src/eval_offtask_pilot_a.py \
  --tag badmerging \
  --attack-suffix "${BM_SUFFIX}" \
  "${COMMON_EVAL[@]}"

run_logged python3 src/eval_offtask_pilot_a.py \
  --tag pilot_a \
  --attack-suffix "${ATTACK_SUFFIX}" \
  "${COMMON_EVAL[@]}"

run_logged python3 src/summarize_offtask_pilot_a.py \
  --pilot ./results/offtask_pilot_a/pilot_a.json \
  --baseline ./results/offtask_pilot_a/badmerging.json \
  --out-md ./results/offtask_pilot_a/summary.md \
  --out-json ./results/offtask_pilot_a/summary.json

echo "============================================================" | tee -a "${LOG}"
echo "[OffTask Pilot-A] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[OffTask Pilot-A] return:" | tee -a "${LOG}"
echo "  results/offtask_pilot_a/summary.md" | tee -a "${LOG}"
echo "  results/offtask_pilot_a/summary.json" | tee -a "${LOG}"
echo "  results/offtask_pilot_a/pilot_a.json" | tee -a "${LOG}"
echo "  results/offtask_pilot_a/badmerging.json" | tee -a "${LOG}"
echo "  ${LOG}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"
