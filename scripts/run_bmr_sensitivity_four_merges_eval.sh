#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-1}"
STUDY="${STUDY:-poison}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-ViT-B-32}"
TASK="${TASK:-CIFAR100}"
TARGET_CLS="${TARGET_CLS:-1}"
PATCH_SIZE="${PATCH_SIZE:-22}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ADAMERGING_EPOCHS="${ADAMERGING_EPOCHS:-500}"
EXAM_DATASETS="${EXAM_DATASETS:-CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS}"

mkdir -p logs results/bmr_sensitivity bmr_paper_results
LOG="logs/bmr_sensitivity_four_merges_${STUDY}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "[BMR sensitivity four merges:${STUDY}] $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

case "${STUDY}" in
  poison)
    declare -a SPECS=(
      "poison_ratio 1 KDR_DTK_BMR_PR1"
      "poison_ratio 2 KDR_DTK_BMR_PR2"
      "poison_ratio 5 KDR_DTK_BMR_PR5"
      "poison_ratio 20 KDR_DTK_BMR_PR20"
    )
    SUMMARY_CSV="bmr_paper_results/bmr_poison_ratio_sensitivity_cifar100.csv"
    SUMMARY_JSON="bmr_paper_results/bmr_poison_ratio_sensitivity_cifar100.json"
    ;;
  gamma|drift|background)
    STUDY="gamma"
    declare -a SPECS=(
      "gamma 0.25 KDR_DTK_BMR_G025"
      "gamma 0 KDR_DTK_BMR_G0"
      "gamma 0.1 KDR_DTK_BMR_G01"
      "gamma 0.5 KDR_DTK_BMR_G05"
      "gamma 1.0 KDR_DTK_BMR_G10"
    )
    SUMMARY_CSV="bmr_paper_results/bmr_background_strength_sensitivity_cifar100.csv"
    SUMMARY_JSON="bmr_paper_results/bmr_background_strength_sensitivity_cifar100.json"
    ;;
  *)
    echo "Unknown STUDY=${STUDY}; expected poison or gamma" >&2
    exit 2
    ;;
esac

python3 -m py_compile src/main_regmean_badmergingon.py src/main_adamerging_badmergingon.py src/eval_submerge.py 2>&1 | tee -a "${LOG}"

for spec in "${SPECS[@]}"; do
  read -r SETTING_KEY SETTING_VALUE METHOD <<<"${spec}"
  CKPT="checkpoints/${MODEL}/${TASK}_${METHOD}_${TASK}_Tgt_${TARGET_CLS}_L_${PATCH_SIZE}/finetuned.pt"
  if [ ! -f "${CKPT}" ]; then
    echo "[BMR sensitivity four merges:${STUDY}] missing checkpoint: ${CKPT}" | tee -a "${LOG}"
    exit 1
  fi

  echo "============================================================" | tee -a "${LOG}"
  echo "[BMR sensitivity four merges:${STUDY}] ${SETTING_KEY}=${SETTING_VALUE} method=${METHOD} start=$(date --iso-8601=seconds)" | tee -a "${LOG}"
  echo "============================================================" | tee -a "${LOG}"

  run_logged python3 src/main_regmean_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${METHOD}" \
    --adversary-task "${TASK}" \
    --target-task "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --test-effectiveness True \
    --test-utility

  run_logged python3 src/main_adamerging_badmergingon.py \
    --model "${MODEL}" \
    --ckpt-dir ./checkpoints \
    --data-location ./data \
    --exam-datasets "${EXAM_DATASETS}" \
    --attack-type "${METHOD}" \
    --adversary-task "${TASK}" \
    --target-task "${TASK}" \
    --target-cls "${TARGET_CLS}" \
    --patch-size "${PATCH_SIZE}" \
    --alpha 0.3 \
    --batch-size "${BATCH_SIZE}" \
    --adamerging-epochs "${ADAMERGING_EPOCHS}" \
    --ada_name "sensitivity_${STUDY}_${SETTING_VALUE}" \
    --test-effectiveness True \
    --test-utility
done

python3 - "${STUDY}" "${LOG}" "${SUMMARY_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import glob
import json
import re
import sys
from pathlib import Path

study, log_path, csv_path, json_path = sys.argv[1:5]
merge_label = {"ta": "TA", "ties": "TIES", "regmean": "RegMean", "adamerging": "AdaMerging"}

if study == "poison":
    specs = [
        ("poison_ratio", "1", "KDR_DTK_BMR_PR1"),
        ("poison_ratio", "2", "KDR_DTK_BMR_PR2"),
        ("poison_ratio", "5", "KDR_DTK_BMR_PR5"),
        ("poison_ratio", "20", "KDR_DTK_BMR_PR20"),
    ]
else:
    specs = [
        ("gamma", "0", "KDR_DTK_BMR_G0"),
        ("gamma", "0.1", "KDR_DTK_BMR_G01"),
        ("gamma", "0.25", "KDR_DTK_BMR_G025"),
        ("gamma", "0.5", "KDR_DTK_BMR_G05"),
        ("gamma", "1.0", "KDR_DTK_BMR_G10"),
    ]

def latest_json(method):
    matches = sorted(
        glob.glob(f"results/bmr_sensitivity/*_{method}_CIFAR100_tgt1_L22.json"),
        key=lambda p: Path(p).stat().st_mtime,
    )
    return matches[-1] if matches else ""

rows = []
for key, value, method in specs:
    jp = latest_json(method)
    if jp:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        for merge, result in data.get("merge_results", {}).items():
            rows.append(
                {
                    key: value,
                    "merge": merge_label.get(merge, merge),
                    "avg_clean_acc": result.get("avg_clean_acc"),
                    "asr": result.get("asr"),
                    "checkpoint_path": f"checkpoints/ViT-B-32/CIFAR100_{method}_CIFAR100_Tgt_1_L_22/finetuned.pt",
                    "result_json": jp,
                    "log": "",
                }
            )

current = None
method_to_setting = {method: (key, value) for key, value, method in specs}
with open(log_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(
            r"method=(KDR_DTK_BMR_(?:PR20|PR1|PR2|PR5|G025|G01|G05|G10|G0))(?![A-Za-z0-9])",
            line,
        )
        if m:
            method = m.group(1)
            if method in method_to_setting:
                current = {"method": method, "merge": None, "avg_clean_acc": None, "asr": None}
            continue
        if current is None:
            continue
        if "src/main_regmean_badmergingon.py" in line:
            current["merge"] = "RegMean"
        elif "src/main_adamerging_badmergingon.py" in line:
            current["merge"] = "AdaMerging"
        m = re.search(r"Avg ACC:([0-9.]+)%", line)
        if m and current.get("merge"):
            current["avg_clean_acc"] = float(m.group(1))
            continue
        m = re.search(r"Backdoor acc:\s*([0-9.]+)", line)
        if m and current.get("merge"):
            current["asr"] = float(m.group(1)) * 100.0
            key, value = method_to_setting[current["method"]]
            rows.append(
                {
                    key: value,
                    "merge": current["merge"],
                    "avg_clean_acc": current["avg_clean_acc"],
                    "asr": current["asr"],
                    "checkpoint_path": f"checkpoints/ViT-B-32/CIFAR100_{current['method']}_CIFAR100_Tgt_1_L_22/finetuned.pt",
                    "result_json": "",
                    "log": log_path,
                }
            )
            current = {"method": current["method"], "merge": current["merge"], "avg_clean_acc": None, "asr": None}

dedup = {}
setting_key = specs[0][0]
order_merge = {"TA": 0, "TIES": 1, "RegMean": 2, "AdaMerging": 3}
for row in rows:
    key = (row[setting_key], row["merge"])
    dedup[key] = row

def setting_sort(v):
    try:
        return float(v[0])
    except ValueError:
        return v[0]

rows = [dedup[k] for k in sorted(dedup, key=lambda x: (setting_sort(x), order_merge.get(x[1], 99)))]

Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [setting_key, "merge", "avg_clean_acc", "asr", "checkpoint_path", "result_json", "log"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"study": study, "rows": rows}, f, indent=2)

print(csv_path)
print(json_path)
PY

echo "[BMR sensitivity four merges:${STUDY}] done=$(date --iso-8601=seconds)" | tee -a "${LOG}"
