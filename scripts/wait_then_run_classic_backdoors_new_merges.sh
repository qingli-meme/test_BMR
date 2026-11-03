#!/usr/bin/env bash
set -euo pipefail

cd /data0/BadMerging
LOG="bmr_paper_results/reviewer_response/logs/classic_backdoors_new_merges/queue_wait.log"
mkdir -p "$(dirname "$LOG")"

while tmux has-session -t representation_then_decision_20261025_234934 2>/dev/null; do
  printf '%s waiting for representation/decision queue to release GPU 2\n' "$(date -Is)" | tee -a "$LOG"
  sleep 45
done

printf '%s GPU 2 predecessor finished; starting classic backdoors\n' "$(date -Is)" | tee -a "$LOG"
exec bash scripts/run_classic_backdoors_new_merges.sh
