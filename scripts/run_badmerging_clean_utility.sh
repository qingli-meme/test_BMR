#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="ViT-B-32"
TASK="CIFAR100"
TARGET_CLS=1
PATCH_SIZE=22
ALPHA=0.3
EXAM_DATASETS="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS"
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT_DIR="results/badmerging_clean_utility/${STAMP}"
LOG_DIR="logs/badmerging_clean_utility_${STAMP}"
MASTER_LOG="${LOG_DIR}/master.log"
CKPT="checkpoints/${MODEL}/${TASK}_On_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"
TRIGGER="trigger/${MODEL}/On_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}.npy"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" bmr_paper_results

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[BadMerging clean utility] start=$(date --iso-8601=seconds) gpu=${GPU_ID}"
echo "[BadMerging clean utility] checkpoint=${CKPT}"
echo "[BadMerging clean utility] trigger=${TRIGGER}"

if [[ ! -f "${TRIGGER}" ]]; then
  echo "Missing BadMerging trigger: ${TRIGGER}" >&2
  exit 1
fi

python3 -m py_compile \
  src/finetune_backdoor_badmergingon.py \
  src/main_task_arithmetic_badmergingon.py \
  src/main_ties_merging_badmergingon.py \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

if [[ ! -f "${CKPT}" ]]; then
  echo "[BadMerging clean utility] malicious checkpoint absent; training from scratch"
  python3 src/finetune_backdoor_badmergingon.py \
    --model "${MODEL}" \
    --adversary-task "${TASK}" \
    --target-task "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --alpha "${ALPHA}" \
    2>&1 | tee "${LOG_DIR}/train_badmerging_on.log"
fi

if [[ ! -s "${CKPT}" ]]; then
  echo "Training did not produce checkpoint: ${CKPT}" >&2
  exit 1
fi

run_eval() {
  local label="$1"
  shift
  echo "============================================================"
  echo "[BadMerging clean utility] ${label} start=$(date --iso-8601=seconds)"
  echo "============================================================"
  "$@" 2>&1 | tee "${LOG_DIR}/${label}.log"
}

run_eval task_arithmetic python3 src/main_task_arithmetic_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type On \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha "${ALPHA}" \
  --test-utility

run_eval ties python3 src/main_ties_merging_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --attack-type On \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha "${ALPHA}" \
  --test-utility

run_eval regmean python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type On \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha "${ALPHA}" \
  --batch-size 128 \
  --test-utility

run_eval adamerging python3 src/main_adamerging_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type On \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha "${ALPHA}" \
  --batch-size 128 \
  --adamerging-epochs 500 \
  --ada_name "badmerging_clean_utility_${STAMP}" \
  --test-utility

python3 - "${LOG_DIR}" "${OUT_DIR}" "${CKPT}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

log_dir, out_dir, checkpoint = map(Path, sys.argv[1:4])
specs = [
    ("Task Arithmetic", "task_arithmetic.log", 75.47, 74.83),
    ("TIES", "ties.log", 73.11, 72.71),
    ("RegMean", "regmean.log", 75.45, 75.39),
    ("AdaMerging", "adamerging.log", 81.03, 80.34),
]

rows = []
for method, filename, clean_only, bmr in specs:
    path = log_dir / filename
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Avg ACC:([0-9.]+)%", text)
    if not matches:
        raise RuntimeError(f"No Avg ACC found in {path}")
    rows.append({
        "merge": method,
        "clean_only": clean_only,
        "badmerging": float(matches[-1]),
        "bmr": bmr,
        "badmerging_delta_vs_clean": float(matches[-1]) - clean_only,
        "bmr_delta_vs_clean": bmr - clean_only,
        "checkpoint": str(checkpoint),
        "evaluation_log": str(path),
    })

out_dir.mkdir(parents=True, exist_ok=True)
run_csv = out_dir / "badmerging_clean_utility_cifar100.csv"
with run_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

run_json = out_dir / "badmerging_clean_utility_cifar100.json"
run_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

paper_csv = Path("bmr_paper_results/badmerging_clean_utility_cifar100.csv")
paper_csv.write_text(run_csv.read_text(encoding="utf-8"), encoding="utf-8")
paper_json = Path("bmr_paper_results/badmerging_clean_utility_cifar100.json")
paper_json.write_text(run_json.read_text(encoding="utf-8"), encoding="utf-8")

print(run_csv)
print(run_json)
print(paper_csv)
print(paper_json)
PY

echo "[BadMerging clean utility] done=$(date --iso-8601=seconds)"
