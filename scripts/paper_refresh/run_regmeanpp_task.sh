#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh
if [[ $# -lt 2 ]]; then echo "usage: $0 GPU TASK" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$(paper_refresh_cuda_device "$1")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
TASK="$2"
mkdir -p logs/paper_refresh
python3 -m src.reviewer_exp.run_new_merges \
  --merge-method regmean_plusplus --attack bmr --adversary-task "${TASK}" \
  --seed 2026 --force \
  2>&1 | tee "logs/paper_refresh/regmeanpp_${TASK}_gpu$1_$(date +%Y%m%d_%H%M%S).log"
