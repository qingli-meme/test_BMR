#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
  echo "usage: $0 GPU_ID" >&2
  exit 2
fi

GPU_ID="$1"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="ViT-B-32"
ADV="CIFAR100"
TARGET_CLS=1
PATCH=22
SEED=2026
EXAM_DATASETS="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS"
CLEAN_CKPT="checkpoints/${MODEL}/${ADV}/finetuned.pt"
OUT_ROOT="analysis/bmr_representation_layer_sensitivity"
RESULT_ROOT="results/bmr_representation_layer_sensitivity"
mkdir -p logs "${OUT_ROOT}" "${RESULT_ROOT}"
LOG="logs/bmr_representation_layer_sensitivity_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[representation-layer] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

run_variant() {
  local tag="$1"
  local layer="$2"
  local method="KDR_DTK_BMR_REP_${tag}"
  local trigger_source="KDR_DTK_REP_${tag}"
  local trigger="trigger/${MODEL}/${trigger_source}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}.npy"
  local checkpoint="checkpoints/${MODEL}/${ADV}_${method}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}/finetuned.pt"

  if [ ! -s "${trigger}" ]; then
    run_logged python3 src/optimize_dormant_target_key_patch.py \
      --model "${MODEL}" \
      --ckpt-dir ./checkpoints \
      --data-location ./data \
      --dataset "${ADV}" \
      --target-cls "${TARGET_CLS}" \
      --patch-size "${PATCH}" \
      --layer-name "${layer}" \
      --pool cls \
      --epochs 3 \
      --batch-size 128 \
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
      --save-path "${trigger}" \
      --out-dir "${OUT_ROOT}/${tag}/patch" \
      --seed "${SEED}" \
      --log-every 20
  else
    echo "[representation-layer] reuse trigger ${trigger}" | tee -a "${LOG}"
  fi

  if [ ! -s "${checkpoint}" ]; then
    run_logged python3 src/finetune_kdr_dtk_bmr.py \
      --model "${MODEL}" \
      --data-location ./data \
      --adversary-task "${ADV}" \
      --target-cls "${TARGET_CLS}" \
      --patch-size "${PATCH}" \
      --trigger-path "${trigger}" \
      --init-checkpoint "${CLEAN_CKPT}" \
      --save-root ./checkpoints \
      --method-name "${method}" \
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
  else
    echo "[representation-layer] reuse checkpoint ${checkpoint}" | tee -a "${LOG}"
  fi

  run_logged python3 src/eval_submerge.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${method}" \
    --trigger-source "${trigger_source}" \
    --adversary-task "${ADV}" \
    --target-task "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --merge-methods ta,ties \
    --scaling-coef 0.3 \
    --batch-size 128 \
    --test-utility \
    --out-dir "${RESULT_ROOT}"

  run_logged python3 src/main_regmean_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${method}" \
    --adversary-task "${ADV}" \
    --target-task "${ADV}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH}" \
    --alpha 0.3 \
    --batch-size 128 \
    --test-effectiveness True \
    --test-utility

  local ada_path="ada/${MODEL}/${method}_${ADV}_Tgt_${TARGET_CLS}_L_${PATCH}_Epoch_500.pt"
  if [ ! -s "${ada_path}" ]; then
    run_logged python3 src/main_adamerging_badmergingon.py \
      --model "${MODEL}" \
      --ckpt-dir ./checkpoints \
      --data-location ./data \
      --exam-datasets "${EXAM_DATASETS}" \
      --attack-type "${method}" \
      --adversary-task "${ADV}" \
      --target-task "${ADV}" \
      --target-cls "${TARGET_CLS}" \
      --patch-size "${PATCH}" \
      --alpha 0.3 \
      --batch-size 128 \
      --adamerging-epochs 500 \
      --test-effectiveness True \
      --test-utility
  else
    echo "[representation-layer] reuse AdaMerging ${ada_path}" | tee -a "${LOG}"
  fi
}

echo "[representation-layer] start=$(date --iso-8601=seconds) gpu=${GPU_ID}" | tee "${LOG}"
python3 -m py_compile \
  src/optimize_dormant_target_key_patch.py \
  src/finetune_kdr_dtk_bmr.py \
  src/eval_submerge.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py
test -s "${CLEAN_CKPT}"

# ViT-B/32 contains 12 residual blocks indexed 0--11.  The final setting is
# the paper checkpoint already trained with block 11 and is therefore reused;
# only the two missing depths require new end-to-end runs.
run_variant MID "model.visual.transformer.resblocks.5.ln_2"
run_variant LATE "model.visual.transformer.resblocks.9.ln_2"

echo "[representation-layer] final reference: model.visual.transformer.resblocks.11.ln_2" | tee -a "${LOG}"
echo "[representation-layer] final checkpoint: checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt" | tee -a "${LOG}"
echo "[representation-layer] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"

