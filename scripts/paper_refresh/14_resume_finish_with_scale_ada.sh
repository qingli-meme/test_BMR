#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh

GPU_SCALE="${1:-1}"
GPU_FINISH0="${2:-0}"
GPU_FINISH1="${3:-1}"
UUID_SCALE="$(paper_refresh_cuda_device "${GPU_SCALE}")"
name="$(nvidia-smi -i "${GPU_SCALE}" --query-gpu=name --format=csv,noheader | xargs)"
[[ "${name}" == *P100* ]] || { echo "[scale-ada] refusing non-P100 ${GPU_SCALE}: ${name}" >&2; exit 3; }

export CUDA_VISIBLE_DEVICES="${UUID_SCALE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
LAMBDA="ada/ViT-B-32/KDR_DTK_BMR_SLIM_SCALE_CIFAR100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt"
mkdir -p logs/paper_refresh

if [[ ! -s "${LAMBDA}" ]]; then
  echo "[scale-ada] generating missing Scale Variation AdaMerging coefficients on P100 ${GPU_SCALE}"
  python3 src/main_adamerging_badmergingon.py \
    --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
    --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --attack-type KDR_DTK_BMR_SLIM_SCALE \
    --adversary-task CIFAR100 --target-task CIFAR100 --target-cls 1 --patch-size 22 \
    --alpha .3 --batch-size 128 --adamerging-epochs 500 \
    --ada_name "_paper_refresh_010c97c.pt" \
    --test-effectiveness True
fi
test -s "${LAMBDA}"
echo "[scale-ada] coefficients ready: ${LAMBDA}"

unset CUDA_VISIBLE_DEVICES
exec bash scripts/paper_refresh/13_finish_on_task_refresh_2p100.sh "${GPU_FINISH0}" "${GPU_FINISH1}"
