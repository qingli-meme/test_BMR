#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
if [[ $# -lt 2 ]]; then echo "usage: $0 GPU VARIANT" >&2; exit 2; fi
GPU="$1"; VAR="$2"
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "${GPU}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLBACKEND=Agg

case "${VAR}" in
  noalign) METHOD=KDR_DTK_BMR_TRIG_NOALIGN; TRIGSRC=KDR_DTK_TRIG_NOALIGN ;;
  nosig)   METHOD=KDR_DTK_BMR_TRIG_NOSIG;   TRIGSRC=KDR_DTK_TRIG_NOSIG ;;
  nodorm)  METHOD=KDR_DTK_BMR_TRIG_NODORM;  TRIGSRC=KDR_DTK_TRIG_NODORM ;;
  *) echo "bad variant ${VAR}" >&2; exit 2 ;;
esac

MODEL=ViT-B-32
TRIGGER="trigger/${MODEL}/${TRIGSRC}_CIFAR100_Tgt_1_L_22.npy"
CLEAN="checkpoints/${MODEL}/CIFAR100/finetuned.pt"
OUTDIR="checkpoints/${MODEL}/CIFAR100_${METHOD}_CIFAR100_Tgt_1_L_22"
MARKER="${OUTDIR}/.paper_refresh_010c97c"
mkdir -p logs/paper_refresh results/paper_refresh/trigger_ablation/${VAR}
RDIR="results/paper_refresh/trigger_ablation/${VAR}"
LOG="logs/paper_refresh/trigger_${VAR}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
DONE="${RDIR}/.paper_refresh_010c97c_complete"
LOCK="${RDIR}/.paper_refresh_010c97c.lock"
exec 9>"${LOCK}"
flock 9
if [[ -f "${DONE}" ]]; then
  echo "[paper-refresh] trigger ${VAR} already complete; reuse"
  exit 0
fi
require_file "${TRIGGER}"; require_file "${CLEAN}"

if [[ ! -f "${MARKER}" ]]; then
  [[ -e "${OUTDIR}" ]] && archive_once "${OUTDIR}" "$(paper_refresh_archive_dir "$(pwd)")/trigger_ablation"
  python3 src/finetune_kdr_dtk_bmr.py \
    --model "${MODEL}" --data-location ./data --adversary-task CIFAR100 \
    --target-cls 1 --patch-size 22 --trigger-path "${TRIGGER}" \
    --init-checkpoint "${CLEAN}" --save-root ./checkpoints --method-name "${METHOD}" \
    --ablation-mode full_bmr --batch-size 128 --bd-batch-size 64 --lr 5e-7 \
    --wd .05 --grad-clip 1 --trainable-scope all --alpha-min .2 --alpha-max 1 \
    --clean-weight 1 --bd-weight 1 --reserve-weight 1 --residual-weight 0 \
    --seed 2026 --log-every 20
  touch "${MARKER}"
fi

python3 src/eval_submerge.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type "${METHOD}" --trigger-source "${TRIGSRC}" \
  --adversary-task CIFAR100 --target-task CIFAR100 --target-cls 1 --patch-size 22 \
  --merge-methods ta,ties --scaling-coef .3 --batch-size 128 --test-utility \
  --out-dir "results/paper_refresh/trigger_ablation/${VAR}"

python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type "${METHOD}" --adversary-task CIFAR100 --target-task CIFAR100 \
  --target-cls 1 --patch-size 22 --alpha .3 --batch-size 128 \
  --test-effectiveness True --test-utility

python3 src/main_adamerging_badmergingon.py \
  --model "${MODEL}" --ckpt-dir ./checkpoints --data-location ./data \
  --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
  --attack-type "${METHOD}" --adversary-task CIFAR100 --target-task CIFAR100 \
  --target-cls 1 --patch-size 22 --alpha .3 --batch-size 128 \
  --adamerging-epochs 500 --ada_name "_paper_refresh_010c97c.pt" \
  --test-effectiveness True --test-utility

for merge in emr regmean_plusplus; do
  python3 -m src.paper_refresh.eval_custom_new_merge \
    --merge-method "${merge}" --method-name "${METHOD}" --adversary-task CIFAR100 \
    --suite CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --trigger-path "${TRIGGER}" \
    --out-dir "results/paper_refresh/trigger_ablation/${VAR}/${merge}"
done
touch "${DONE}"
