#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
EMR_SESSION="${EMR_SESSION:?set EMR_SESSION}"
RPP_SESSION="${RPP_SESSION:?set RPP_SESSION}"
LOG_ROOT="bmr_paper_results/reviewer_response/logs/sensitivity_six_merges"
mkdir -p "${LOG_ROOT}"
LOG="${LOG_ROOT}/coordinator_$(date +%Y%m%d_%H%M%S).log"

echo "[coordinator] waiting for ${EMR_SESSION} and ${RPP_SESSION}" | tee -a "${LOG}"
while tmux has-session -t "${EMR_SESSION}" 2>/dev/null || tmux has-session -t "${RPP_SESSION}" 2>/dev/null; do
  sleep 30
done

echo "[coordinator] finalizing at $(date --iso-8601=seconds)" | tee -a "${LOG}"
python3 -m src.reviewer_exp.finalize_sensitivity_six_merges 2>&1 | tee -a "${LOG}"
