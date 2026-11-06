#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
if [[ $# -lt 2 ]]; then echo "usage: $0 GPU N" >&2; exit 2; fi
GPU="$1"; N="$2"
CUDA_DEVICE="$(paper_refresh_cuda_device "${GPU}")"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

case "${N}" in
  2) SUITE="CIFAR100,GTSRB" ;;
  4) SUITE="CIFAR100,GTSRB,EuroSAT,Cars" ;;
  6) SUITE="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS" ;;
  *) exit 2 ;;
esac
mkdir -p logs/paper_refresh "results/paper_refresh/task_number/N${N}"
RDIR="results/paper_refresh/task_number/N${N}"
DONE="${RDIR}/.paper_refresh_010c97c_complete"
LOCK="${RDIR}/.paper_refresh_010c97c.lock"
exec 9>"${LOCK}"
flock 9
if [[ -f "${DONE}" ]]; then
  echo "[paper-refresh] task-number N=${N} already complete; reuse"
  exit 0
fi

# Force the existing script to recompute task-number Ada coefficients from the
# current canonical CIFAR100 checkpoint.
ADA="ada/ViT-B-32/KDR_DTK_BMR_CIFAR100_Tgt_1_L_22_tasknum_N${N}_Epoch_500.pt"
if [[ -f "${ADA}" && ! -f "${ADA}.paper_refresh_010c97c" ]]; then
  mkdir -p paper_refresh_archive_010c97c/task_number_ada
  mv "${ADA}" "paper_refresh_archive_010c97c/task_number_ada/$(basename "${ADA}")__$(date +%Y%m%d_%H%M%S)"
fi

bash scripts/run_bmr_cifar100_task_number_task.sh "${CUDA_DEVICE}" "${N}" \
  2>&1 | tee "logs/paper_refresh/tasknum_N${N}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
[[ -f "${ADA}" ]] && touch "${ADA}.paper_refresh_010c97c"

TRIGGER="trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
for merge in emr regmean_plusplus; do
  python3 -m src.paper_refresh.eval_custom_new_merge \
    --merge-method "${merge}" --method-name KDR_DTK_BMR --adversary-task CIFAR100 \
    --suite "${SUITE}" --trigger-path "${TRIGGER}" \
    --out-dir "results/paper_refresh/task_number/N${N}/${merge}"
done
touch "${DONE}"
