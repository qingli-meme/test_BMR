#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
if [[ $# -lt 3 ]]; then echo "usage: $0 GPU TAG RATE" >&2; exit 2; fi
GPU="$1"; TAG="$2"; RATE="$3"
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLBACKEND=Agg

METHOD="KDR_DTK_BMR_${TAG}"
MODEL=ViT-B-32
TRIGGER="trigger/${MODEL}/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
CLEAN="checkpoints/${MODEL}/CIFAR100/finetuned.pt"
OUTDIR="checkpoints/${MODEL}/CIFAR100_${METHOD}_CIFAR100_Tgt_1_L_22"
MARKER="${OUTDIR}/.paper_refresh_010c97c"
RDIR="results/paper_refresh/poison/${TAG}"
mkdir -p "${RDIR}" logs/paper_refresh
LOG="logs/paper_refresh/poison_${TAG}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
DONE="${RDIR}/.paper_refresh_010c97c_complete"
LOCK="${RDIR}/.paper_refresh_010c97c.lock"
exec 9>"${LOCK}"
flock 9
if [[ -f "${DONE}" ]]; then
  echo "[paper-refresh] poison ${TAG} already complete; reuse"
  exit 0
fi
require_file "${TRIGGER}"; require_file "${CLEAN}"

if [[ ! -f "${MARKER}" ]]; then
  [[ -e "${OUTDIR}" ]] && archive_once "${OUTDIR}" "$(paper_refresh_archive_dir "$(pwd)")/poison"
  python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" --data-location ./data --adversary-task CIFAR100 \
    --target-cls 1 --patch-size 22 --trigger-path "${TRIGGER}" \
    --init-checkpoint "${CLEAN}" --save-root ./checkpoints --method-name "${METHOD}" \
    --ablation-mode full_bmr --batch-size 128 --bd-batch-size 64 \
    --poison-rate "${RATE}" --lr 5e-7 --wd .05 --grad-clip 1 \
    --trainable-scope all --alpha-min .2 --alpha-max 1 \
    --clean-weight 1 --bd-weight 1 --reserve-weight 1 --residual-weight 0 \
    --seed 2026 --log-every 20
  touch "${MARKER}"
fi

REGLOG="${RDIR}/regmean.log"
python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type "${METHOD}" --adversary-task CIFAR100 --target-task CIFAR100 \
  --target-cls 1 --patch-size 22 --alpha .3 --batch-size 128 \
  --test-effectiveness True --test-utility | tee "${REGLOG}"

python3 -m src.paper_refresh.eval_custom_new_merge \
  --merge-method regmean_plusplus --method-name "${METHOD}" \
  --adversary-task CIFAR100 --suite CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --trigger-path "${TRIGGER}" --out-dir "${RDIR}/regmean_plusplus"

printf '%s\n' "${RATE}" > "${RDIR}/poison_rate_fraction.txt"
touch "${DONE}"
