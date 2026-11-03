#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

POST_GPU="${POST_GPU:-0}"

echo "[suite] 1/5 mechanism controls"
bash scripts/schedule_bmr_mechanism_controls.sh

echo "[suite] 2/5 residual-norm controls"
bash scripts/run_bmr_norm_control.sh "${POST_GPU}"

echo "[suite] 3/5 trigger dormancy control"
bash scripts/run_bmr_trigger_dormancy_control.sh "${POST_GPU}"

echo "[suite] 4/5 malicious-residual coupling diagnostic"
bash scripts/run_bmr_malicious_coupling.sh "${POST_GPU}"

echo "[suite] 5/5 actual-merge target-margin measurements"
bash scripts/run_bmr_real_gain_controls.sh "${POST_GPU}"

echo "[suite] collecting results"
bash scripts/collect_bmr_control_results.sh
