#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_ID="${GPU_ID:-2}"; SESSION="bmr_offtask_no_shadow_v100"
name="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader | xargs)"
[[ "${name}" == *V100* ]] || { echo "refusing GPU ${GPU_ID}: ${name}" >&2; exit 3; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "session exists: ${SESSION}" >&2; exit 2; }
cmd="set -e; cd /data0/BadMerging; for task in Cars GTSRB EuroSAT SUN397 PETS; do bash scripts/run_bmr_offtask_task.sh ${GPU_ID} \$task; done; python3 src/summarize_bmr_offtask.py; echo '[offtask] ALL COMPLETE'; exec bash"
tmux new-session -d -s "${SESSION}" "${cmd}"
echo "session=${SESSION} gpu=${GPU_ID} name=${name}"
