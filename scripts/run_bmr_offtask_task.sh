#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then echo "usage: $0 GPU_ID TARGET_TASK" >&2; exit 2; fi
GPU_ID="$1"; TARGET="$2"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  CUDA_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"
else CUDA_DEVICE="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
ROOT="results/bmr_offtask_no_shadow/${TARGET}"; REF="analysis/bmr_offtask/reference/${TARGET}_Tgt_1_R10.pt"
TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_${TARGET}_Tgt_1_R10_L_22.npy"
CKPT="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_${TARGET}_Tgt_1_R10_L_22"
mkdir -p "${ROOT}" "analysis/bmr_offtask/reference" logs/bmr_offtask_no_shadow
exec > >(tee -a "logs/bmr_offtask_no_shadow/${TARGET}_$(date +%Y%m%d_%H%M%S).log") 2>&1

if [[ ! -f "${TRIGGER}" ]]; then
  python3 src/optimize_offtask_dormant_target_key_patch.py --target-task "${TARGET}" \
    --reference-path "${REF}" --save-path "${TRIGGER}" --out-dir "${ROOT}"
fi
if [[ ! -f "${CKPT}/finetuned.pt" ]]; then
  python3 src/finetune_kdr_dtk_bmr_offtask.py --target-task "${TARGET}" \
    --reference-path "${REF}" --trigger-path "${TRIGGER}" --output-dir "${CKPT}"
  cp "${CKPT}/config.json" "${ROOT}/config.json"
  cp "${CKPT}/stage2_summary.json" "${ROOT}/stage2_summary.json"
fi

failures=0
for method in ta ties regmean adamerging emr regmeanpp; do
  if [[ -f "${ROOT}/${method}.json" ]]; then continue; fi
  python3 src/eval_bmr_offtask_all_merges.py --target-task "${TARGET}" --merge-method "${method}" \
    --trigger-path "${TRIGGER}" --attack-checkpoint "${CKPT}/finetuned.pt" --out-dir "${ROOT}" || failures=$((failures+1))
done
python3 src/analyze_bmr_offtask_mechanism.py --target-task "${TARGET}" --reference-path "${REF}" \
  --trigger-path "${TRIGGER}" --result-dir "${ROOT}" || failures=$((failures+1))
python3 src/summarize_bmr_offtask.py
if (( failures == 0 )); then touch "${ROOT}/.complete"; fi
echo "[offtask] ${TARGET} failures=${failures}"
exit "${failures}"
