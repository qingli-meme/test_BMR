#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

MASTER="bmr_refresh_resume_2p100_master"
if tmux has-session -t "${MASTER}" 2>/dev/null; then
  echo "refusing duplicate session: ${MASTER}" >&2
  exit 2
fi
RUN_ID="resume_2p100_$(date +%Y%m%d_%H%M%S)"
LOG="logs/paper_refresh/resume_orchestrator_${RUN_ID}.log"
mkdir -p logs/paper_refresh
tmux new-session -d -s "${MASTER}" \
  "export PAPER_REFRESH_RUN_ID=${RUN_ID}; set -o pipefail; bash scripts/paper_refresh/12_resume_after_reboot_2p100.sh 0 1 2>&1 | tee ${LOG}; rc=\${PIPESTATUS[0]}; echo '[resume-2p100] exit_code='\${rc}; exec bash"
echo "master=${MASTER}"
echo "run_id=${RUN_ID}"
echo "log=${LOG}"
