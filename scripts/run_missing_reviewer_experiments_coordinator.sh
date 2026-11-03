#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKER_A="${WORKER_A:?WORKER_A is required}"
WORKER_B="${WORKER_B:?WORKER_B is required}"
GPU_ID="${GPU_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="bmr_paper_results/reviewer_response/logs/missing_experiments_coordinator_${STAMP}.log"

while tmux has-session -t "${WORKER_A}" 2>/dev/null || tmux has-session -t "${WORKER_B}" 2>/dev/null; do
  sleep 30
done

# Do not silently continue after a failed worker.  The coordinator only moves
# to the mechanism diagnostics once all eight missing method/task result files
# exist.
python3 - <<'PY'
from pathlib import Path

root = Path("/data0/BadMerging")
tasks = ["EuroSAT", "Cars", "SUN397", "PETS"]
expected = []
for task in tasks:
    expected.append(root / "analysis" / "emr" / "runs" / f"badmerging__{task}" / "result.json")
    expected.append(root / "bmr_paper_results" / "reviewer_response" / "runs" / f"regmean_plusplus__badmerging__{task}" / "result.json")
missing = [str(path) for path in expected if not path.is_file()]
if missing:
    raise SystemExit("Missing worker outputs:\n" + "\n".join(missing))
print("[coordinator] verified all eight missing BadMerging results")
PY

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date -Is)] workers completed; formal mechanism diagnostics start" | tee "${LOG}"
python3 -m src.reviewer_exp.run_new_merge_mechanism_diagnostics \
  --methods emr,regmean_plusplus \
  --gain-variants ordinary,background_aug,full_bmr \
  --out-dir bmr_paper_results/reviewer_response/new_merge_mechanism_diagnostics \
  2>&1 | tee -a "${LOG}"

python3 -m src.reviewer_exp.summarize_completed_reviewer_experiments \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Is)] all missing reviewer experiments completed" | tee -a "${LOG}"
