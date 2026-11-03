#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MERGE_METHOD="${MERGE_METHOD:?set MERGE_METHOD to emr or regmean_plusplus}"
GPU_ID="${GPU_ID:?set GPU_ID}"
OUT="bmr_paper_results/reviewer_response/sensitivity_six_merges"
LOG_ROOT="bmr_paper_results/reviewer_response/logs/sensitivity_six_merges"
mkdir -p "${OUT}/runs" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

LOG="${LOG_ROOT}/${MERGE_METHOD}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
echo "[worker] method=${MERGE_METHOD} gpu=${GPU_ID} start=$(date --iso-8601=seconds)" | tee -a "${LOG}"

tail -n +2 "${OUT}/jobs.tsv" | while IFS=$'\t' read -r experiment setting tasks checkpoint trigger; do
  slug="${setting//./p}"
  run_dir="${OUT}/runs/${experiment}__${slug}__${MERGE_METHOD}"
  mkdir -p "${run_dir}"
  echo "[job] experiment=${experiment} setting=${setting} tasks=${tasks}" | tee -a "${LOG}"
  (
    # Multiple GPU workers may traverse the same manifest.  Serialize each
    # physical setting so a completed result is reused instead of recomputed.
    flock 9
    python3 -m src.reviewer_exp.run_sensitivity_new_merges \
      --merge-method "${MERGE_METHOD}" \
      --experiment "${experiment}" \
      --setting-value "${setting}" \
      --tasks "${tasks}" \
      --malicious-checkpoint "${checkpoint}" \
      --trigger "${trigger}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --seed 2026 \
      2>&1 | tee -a "${LOG}"
  ) 9>"${run_dir}/.evaluation.lock"
done

echo "[worker] method=${MERGE_METHOD} done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
