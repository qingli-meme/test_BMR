#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-bmr_paper_results/motivation}"
TASKS="CIFAR100,ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS"
ADA_IMAGENET="ada/ViT-B-32/Clean_imagenet100_six_task_clean_Epoch_500.pt"

while tmux has-session -t motivation_ta_ties 2>/dev/null; do
    sleep 60
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH=.:src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

python3 src/collect_motivation_metrics.py \
    --methods regmean \
    --tasks "${TASKS}" \
    --batch-size 128 \
    --out-dir "${OUT_DIR}"

while tmux has-session -t motivation_clean_ada_imagenet100 2>/dev/null; do
    sleep 60
done

test -f "${ADA_IMAGENET}"

python3 src/collect_motivation_metrics.py \
    --methods adamerging \
    --tasks "${TASKS}" \
    --batch-size 128 \
    --out-dir "${OUT_DIR}"

python3 src/collect_motivation_metrics.py \
    --aggregate-only \
    --out-dir "${OUT_DIR}"

python3 - "${OUT_DIR}/motivation_metrics.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 28:
    raise RuntimeError(f"expected 28 summary rows, found {len(rows)}")
PY

python3 src/plot_motivation_weak_coupling.py \
    --csv "${OUT_DIR}/motivation_metrics.csv" \
    --out-dir "${OUT_DIR}"

mkdir -p "${OUT_DIR}/pdf_render"
pdftoppm -f 1 -singlefile -png -r 300 \
    "${OUT_DIR}/motivation_weak_coupling_vs_decision_impact.pdf" \
    "${OUT_DIR}/pdf_render/motivation_weak_coupling_vs_decision_impact"

for required in \
    "${OUT_DIR}/motivation_metrics.csv" \
    "${OUT_DIR}/motivation_metrics_per_sample.csv" \
    "${OUT_DIR}/motivation_weak_coupling_vs_decision_impact.pdf" \
    "${OUT_DIR}/motivation_weak_coupling_vs_decision_impact.png"; do
    test -s "${required}"
done

echo "[motivation] complete: ${OUT_DIR}"
