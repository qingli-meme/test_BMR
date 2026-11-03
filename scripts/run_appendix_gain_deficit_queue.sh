#!/usr/bin/env bash
set -euo pipefail
cd /data0/BadMerging

GPU_ID=${GPU_ID:-0}
WAIT_SESSION=${WAIT_SESSION:-bmr_xbackbone_b16}
OUT=bmr_paper_results/appendix_gain_deficit
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] queued after tmux=$WAIT_SESSION on physical GPU $GPU_ID"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(date -Is)] waiting for $WAIT_SESSION"
  sleep 60
done

# Do not collide with a replacement process that may claim the card just as
# the preceding tmux session exits.
stable=0
while (( stable < 3 )); do
  used=$(nvidia-smi --id="$GPU_ID" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if (( used < 1500 )); then stable=$((stable+1)); else stable=0; fi
  echo "[$(date -Is)] gpu=$GPU_ID used_mib=$used stable_free=$stable/3"
  (( stable < 3 )) && sleep 60
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python -u -m src.reviewer_exp.run_appendix_gain_deficit \
  --smoke --batch-size 32 \
  --out-dir "$OUT/smoke"

python -u -m src.reviewer_exp.run_appendix_gain_deficit \
  --batch-size 128 \
  --out-dir "$OUT"

echo "[$(date -Is)] COMPLETE appendix gain-deficit diagnostic"
