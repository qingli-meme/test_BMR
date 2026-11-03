#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID {ties|regmean}" >&2
  exit 2
fi

GPU_ID="$1"
METHOD="$2"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"

case "${METHOD}" in
  ties|regmean) ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

FINAL_DECISION="analysis/bmr_slimming_phase1/SLIMMING_DECISION.json"
OUT="analysis/bmr_slimming_phase1_${METHOD}"
mkdir -p "${OUT}" logs
LOG="logs/bmr_slim_phase1_${METHOD}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "[BMR Slim parallel] method=${METHOD} gpu=${GPU_ID}" | tee "${LOG}"
echo "[BMR Slim parallel] waiting for used memory <= ${MAX_USED_MIB} MiB" | tee -a "${LOG}"

while true; do
  if [ -f "${FINAL_DECISION}" ]; then
    echo "[BMR Slim parallel] full evaluation already finished; skip" | tee -a "${LOG}"
    exit 0
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
  if [ "${used}" -le "${MAX_USED_MIB}" ]; then
    break
  fi
  sleep 30
done

python3 src/eval_bmr_real_merge_gain.py \
  --model ViT-B-32 \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --adversary-task CIFAR100 \
  --target-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --attack-types KDR_DTK_BMR_SLIM_ORD,KDR_DTK_BMR_SLIM_SCALE,KDR_DTK_BMR_SLIM_ABS,KDR_DTK_BMR_SLIM_FIXED,KDR_DTK_BMR_SLIM_NODRIFT,KDR_DTK_BMR_SLIM_DRIFT \
  --merge-methods "${METHOD}" \
  --scaling-coef 0.3 \
  --ties-reset-thresh 20 \
  --ties-merge-func dis-sum \
  --regmean-num-train-batch 8 \
  --batch-size 128 \
  --max-eval-batches 0 \
  --out-dir "${OUT}" 2>&1 | tee -a "${LOG}"

echo "[BMR Slim parallel] done method=${METHOD}" | tee -a "${LOG}"
