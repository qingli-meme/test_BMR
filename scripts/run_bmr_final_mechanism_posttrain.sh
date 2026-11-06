#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_DEVICE="${1:-0}"
if [[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
  GPU_DEVICE="$(nvidia-smi -i "${GPU_DEVICE}" --query-gpu=uuid --format=csv,noheader | xargs)"
fi
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg

python3 -u src/run_bmr_final_mechanism.py --phase loss
python3 -u src/run_bmr_final_mechanism.py --phase merge-config
python3 -u src/run_bmr_final_mechanism.py --phase residual
python3 -u src/run_bmr_final_mechanism.py --phase active-region
python3 -u src/run_bmr_final_mechanism.py --phase finalize
