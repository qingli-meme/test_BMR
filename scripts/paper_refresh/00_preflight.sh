#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPECTED="010c97c1811762f7e5262a9410ddb16fab34ecbc"
HEAD="$(git rev-parse HEAD)"
if [[ "${HEAD}" != "${EXPECTED}" ]]; then
  echo "[preflight] wrong HEAD: ${HEAD}" >&2
  echo "[preflight] expected:   ${EXPECTED}" >&2
  exit 2
fi

python3 -m py_compile \
  src/finetune_kdr_dtk_bmr.py \
  src/paper_refresh/eval_custom_new_merge.py \
  src/paper_refresh/run_gain_deficit_refresh.py \
  src/paper_refresh/plot_poison_ratio_refresh.py

if grep -R --line-number -E 'drift-rho|eta-min|eta-max|drift-scope|delta_bg|synthetic[_ -]background' \
    src/finetune_kdr_dtk_bmr.py >/tmp/bmr_refresh_forbidden.txt 2>/dev/null; then
  echo "[preflight] canonical trainer still contains removed drift terms:" >&2
  cat /tmp/bmr_refresh_forbidden.txt >&2
  exit 3
fi

MODEL=ViT-B-32
TASKS=(CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS)
test -f "checkpoints/${MODEL}/zeroshot.pt"
for task in "${TASKS[@]}"; do
  test -f "checkpoints/${MODEL}/${task}/finetuned.pt"
  test -f "trigger/${MODEL}/KDR_DTK_${task}_Tgt_1_L_22.npy"
done
for trig in NOALIGN NOSIG NODORM; do
  test -f "trigger/${MODEL}/KDR_DTK_TRIG_${trig}_CIFAR100_Tgt_1_L_22.npy"
done

test -f "checkpoints/${MODEL}/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt"
test -f "analysis/bmr_slimming_phase1/BMR_ON_TASK_SLIMMING_REPORT.md"

mkdir -p logs/paper_refresh results/paper_refresh analysis/paper_refresh
echo "[preflight] OK HEAD=${HEAD}"
echo "[preflight] canonical CIFAR100 BMR will be reused"
echo "[preflight] no multi-seed, no cross-backbone, no trigger-size, no representation-layer jobs are scheduled"
