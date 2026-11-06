#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_ID="${1:-2}"
for task in Cars GTSRB EuroSAT SUN397 PETS; do bash scripts/run_bmr_offtask_task.sh "${GPU_ID}" "${task}"; done
