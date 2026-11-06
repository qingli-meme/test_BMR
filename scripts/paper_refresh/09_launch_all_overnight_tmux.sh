#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU0="${1:-0}"
GPU1="${2:-1}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER="bmr_refresh_master"
LOG="logs/paper_refresh/orchestrator_${RUN_ID}.log"
mkdir -p logs/paper_refresh

tmux kill-session -t "${MASTER}" 2>/dev/null || true
tmux new-session -d -s "${MASTER}" \
  "export PAPER_REFRESH_RUN_ID=${RUN_ID}; set -o pipefail; bash scripts/paper_refresh/09_run_all_overnight.sh ${GPU0} ${GPU1} 2>&1 | tee ${LOG}; rc=\${PIPESTATUS[0]}; echo '[orchestrator] exit_code='\${rc}; exec bash"

echo "master=${MASTER}"
echo "run_id=${RUN_ID}"
echo "log=${LOG}"
