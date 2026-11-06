#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 2 ]] || { echo "usage: $0 GPU_ID TARGET_TASK" >&2; exit 2; }
GPU_ID="$1"; TARGET="$2"
if [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then CUDA_DEVICE="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | xargs)"; else CUDA_DEVICE="${GPU_ID}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
ROOT="results/bmr_offtask_v2/${TARGET}"
META="analysis/bmr_offtask_v2/targets/${TARGET}_Tgt_1_R10.pt"
TRIGGER="trigger/ViT-B-32/KDR_DTK_OFF_V2_${TARGET}_Tgt_1_R10_L_22.npy"
CKPT="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V2_${TARGET}_Tgt_1_R10_L_22"
TA_MIN_ASR="${OFFTASK_TA_MIN_ASR:-80}"
mkdir -p "${ROOT}" analysis/bmr_offtask_v2/targets logs/bmr_offtask_v2
exec > >(tee -a "logs/bmr_offtask_v2/${TARGET}_$(date +%Y%m%d_%H%M%S).log") 2>&1

if [[ ! -f "${TRIGGER}" ]]; then
  python3 src/optimize_offtask_text_direction_trigger.py --target-task "${TARGET}" \
    --metadata-path "${META}" --save-path "${TRIGGER}" --out-dir "${ROOT}"
fi
if [[ ! -f "${CKPT}/finetuned.pt" ]]; then
  python3 src/finetune_kdr_dtk_bmr_offtask_v2.py --target-task "${TARGET}" \
    --metadata-path "${META}" --trigger-path "${TRIGGER}" --output-dir "${CKPT}"
fi
cp "${CKPT}/config.json" "${ROOT}/config.json"
cp "${CKPT}/stage2_summary.json" "${ROOT}/stage2_summary.json"

if [[ ! -f "${ROOT}/ta.json" ]]; then
  python3 src/eval_bmr_offtask_v2_all_merges.py --target-task "${TARGET}" --merge-method ta \
    --trigger-path "${TRIGGER}" --attack-checkpoint "${CKPT}/finetuned.pt" --out-dir "${ROOT}"
fi

TA_ASR="$(python3 - "${ROOT}/ta.json" <<'PY'
import json, sys
print(float(json.load(open(sys.argv[1]))["asr"]))
PY
)"
echo "[OFF-V2] TA gate: ASR=${TA_ASR}% threshold=${TA_MIN_ASR}%"
if ! python3 - "${TA_ASR}" "${TA_MIN_ASR}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
then
  printf 'target_task=%s\nta_asr=%s\nthreshold=%s\ndecision=stop_before_remaining_mergers\n' \
    "${TARGET}" "${TA_ASR}" "${TA_MIN_ASR}" > "${ROOT}/TA_SCREEN_FAILED"
  python3 src/summarize_bmr_offtask_v2.py --result-root results/bmr_offtask_v2
  echo "[OFF-V2] TA screen failed; preserving checkpoint/results and stopping before TIES/RegMean/AdaMerging/EMR/RegMean++."
  exit 0
fi
rm -f "${ROOT}/TA_SCREEN_FAILED"

for method in ties regmean adamerging emr regmeanpp; do
  [[ -f "${ROOT}/${method}.json" ]] && continue
  python3 src/eval_bmr_offtask_v2_all_merges.py --target-task "${TARGET}" --merge-method "${method}" \
    --trigger-path "${TRIGGER}" --attack-checkpoint "${CKPT}/finetuned.pt" --out-dir "${ROOT}"
done
python3 src/analyze_bmr_offtask_v2.py --target-task "${TARGET}" --metadata-path "${META}" --trigger-path "${TRIGGER}" --result-dir "${ROOT}"
python3 src/summarize_bmr_offtask_v2.py --result-root results/bmr_offtask_v2
touch "${ROOT}/.complete"
echo "[OFF-V2] COMPLETE ${TARGET}"

