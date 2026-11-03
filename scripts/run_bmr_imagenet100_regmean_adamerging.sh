#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-ViT-B-32}"
TASK="ImageNet100"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"
EXAM_DATASETS="${EXAM_DATASETS:-ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

stamp="$(date +%Y%m%d_%H%M%S)"
log_dir="logs/bmr_imagenet100_four_merges_${stamp}"
mkdir -p "${log_dir}" bmr_paper_results
regmean_log="${log_dir}/regmean.log"
adamerging_log="${log_dir}/adamerging.log"
summary_csv="bmr_paper_results/bmr_imagenet100_four_merges.csv"
summary_json="bmr_paper_results/bmr_imagenet100_four_merges.json"

checkpoint="checkpoints/${MODEL}/${TASK}_KDR_DTK_BMR_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"
trigger="trigger/${MODEL}/KDR_DTK_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}.npy"
for required in "${checkpoint}" "${trigger}" "checkpoints/${MODEL}/head_${TASK}.pt"; do
  if [ ! -f "${required}" ]; then
    echo "[BMR-ImageNet100-four-merges] missing required file: ${required}" >&2
    exit 1
  fi
done

python3 -m py_compile \
  src/main_regmean_badmergingon.py \
  src/main_adamerging_badmergingon.py

echo "[BMR-ImageNet100-four-merges] start=$(date --iso-8601=seconds) gpu=${GPU_ID}"
echo "[BMR-ImageNet100-four-merges] datasets=${EXAM_DATASETS}"

python3 src/main_regmean_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type KDR_DTK_BMR \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --test-effectiveness True \
  --test-utility 2>&1 | tee "${regmean_log}"

python3 src/main_adamerging_badmergingon.py \
  --model "${MODEL}" \
  --ckpt-dir ./checkpoints \
  --data-location ./data \
  --exam-datasets "${EXAM_DATASETS}" \
  --attack-type KDR_DTK_BMR \
  --adversary-task "${TASK}" \
  --target-task "${TASK}" \
  --target-cls "${TARGET_CLS}" \
  --patch-size "${PATCH_SIZE}" \
  --alpha 0.3 \
  --batch-size "${BATCH_SIZE}" \
  --adamerging-epochs "${ADAMERGING_EPOCHS}" \
  --ada_name imagenet100_six_task \
  --test-effectiveness True \
  --test-utility 2>&1 | tee "${adamerging_log}"

python3 - "${regmean_log}" "${adamerging_log}" "${summary_csv}" "${summary_json}" <<'PY'
import csv
import glob
import json
import re
import sys
from pathlib import Path

regmean_log, adamerging_log, csv_path, json_path = sys.argv[1:]

def final_metrics(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    clean = re.findall(r"Avg ACC:([0-9.]+)%", text)
    attack = re.findall(r"Backdoor acc:\s*([0-9.]+)", text)
    if not clean or not attack:
        raise RuntimeError(f"missing final metrics in {path}")
    return float(clean[-1]), float(attack[-1]) * 100.0

result_files = sorted(
    glob.glob("results/bmr_imagenet100/*_KDR_DTK_BMR_ImageNet100_tgt1_L22.json"),
    key=lambda path: Path(path).stat().st_mtime,
)
if not result_files:
    raise RuntimeError("missing ImageNet100 TA/TIES result JSON")
base_path = result_files[-1]
with open(base_path, encoding="utf-8") as f:
    base = json.load(f)

rows = []
for key, label in (("ta", "TA"), ("ties", "TIES")):
    result = base["merge_results"][key]
    rows.append({
        "adversary_task": "ImageNet100",
        "merge_method": label,
        "avg_clean_acc_pct": result["avg_clean_acc"],
        "asr_pct": result["asr"],
        "source": base_path,
    })

for label, path in (("RegMean", regmean_log), ("AdaMerging", adamerging_log)):
    clean, asr = final_metrics(path)
    rows.append({
        "adversary_task": "ImageNet100",
        "merge_method": label,
        "avg_clean_acc_pct": clean,
        "asr_pct": asr,
        "source": path,
    })

fieldnames = ["adversary_task", "merge_method", "avg_clean_acc_pct", "asr_pct", "source"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"datasets": "ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS", "rows": rows}, f, indent=2)
print(csv_path)
print(json_path)
PY

echo "[BMR-ImageNet100-four-merges] done=$(date --iso-8601=seconds)"
