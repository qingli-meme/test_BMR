#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Wait specifically for the existing off-task v4 runner on GPU 1.  Do not
# interfere with it; start this zero-training diagnostic as soon as it exits.
while pgrep -f 'scripts/run_bmr_offtask_v4_cars_regmean.sh 1' >/dev/null; do
  echo "[on-off-regmean] waiting for off-task v4 GPU-1 runner: $(date '+%F %T')"
  sleep 30
done

bash scripts/run_on_off_regmean_mechanism.sh 1
