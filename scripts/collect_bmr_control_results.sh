#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE="bmr_control_results_${STAMP}"
ARCHIVE="${STAGE}.tar.gz"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/logs" "${STAGE}/analysis" "${STAGE}/results" "${STAGE}/training_metadata"

# Copy all control logs.
find logs -maxdepth 1 -type f \( \
  -name 'bmr_control_*' -o \
  -name 'bmr_controls_scheduler_*' -o \
  -name 'bmr_norm_control_*' -o \
  -name 'bmr_trigger_dormancy_control_*' -o \
  -name 'bmr_malicious_coupling_*' -o \
  -name 'bmr_real_gain_controls_*' \
\) -exec cp -f {} "${STAGE}/logs/" \; 2>/dev/null || true

if [ -d analysis/bmr_controls ]; then
  cp -a analysis/bmr_controls "${STAGE}/analysis/"
fi
if [ -d results/bmr_controls ]; then
  cp -a results/bmr_controls "${STAGE}/results/"
fi

# Local-only, paper-validated EMR and RegMean++ control evaluations. Copy only
# the small result records, not cached receiver checkpoints.
NEW_MERGE_ROOT="bmr_paper_results/reviewer_response/ablation_new_merges"
if [ -d "${NEW_MERGE_ROOT}" ]; then
  mkdir -p "${STAGE}/results/emr_regmeanpp"
  for variant in \
    gain_only no_background fixed_background fixed_residual both_fixed \
    ordinary_normmatch background_normmatch bmr_normmatch_bg; do
    for merger in emr regmean_plusplus; do
      src="${NEW_MERGE_ROOT}/${merger}__${variant}/result.json"
      if [ -f "${src}" ]; then
        dst="${STAGE}/results/emr_regmeanpp/${merger}__${variant}"
        mkdir -p "${dst}"
        cp -f "${src}" "${dst}/result.json"
      fi
    done
  done
fi

# Training configs/summaries/logs are small and important; checkpoints are not collected.
for method in \
  KDR_DTK_BMR_GAINONLY \
  KDR_DTK_BMR_NO_BG \
  KDR_DTK_BMR_FIXETA \
  KDR_DTK_BMR_FIXALPHA \
  KDR_DTK_BMR_BOTHFIX \
  KDR_DTK_BMR_ORD_NORMMATCH \
  KDR_DTK_BMR_BG_NORMMATCH \
  KDR_DTK_BMR_BMR_NORMMATCH_BG; do
  dir="checkpoints/${MODEL:-ViT-B-32}/${ADV:-CIFAR100}_${method}_${ADV:-CIFAR100}_Tgt_${TARGET_CLS:-1}_L_${PATCH:-22}"
  if [ -d "${dir}" ]; then
    mkdir -p "${STAGE}/training_metadata/${method}"
    find "${dir}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) \
      -exec cp -f {} "${STAGE}/training_metadata/${method}/" \;
  fi
done

cat > "${STAGE}/RETURN_CHECKLIST.txt" <<'EOF'
Please return this archive intact.
Required evidence:
1. ASR/ACC for gain_only, no_background, fixed_background, fixed_residual, both_fixed.
2. Residual-norm JSON files and ASR/ACC of all norm-matched controls.
3. Trigger activation rates for SlimTrigger vs SlimTriggerWithoutDormancy.
4. malicious_residual_coupling CSV/JSON.
5. real_gain CSV/JSON for BackgroundAug, gain_only, FullBMR.
6. Any EMR/RegMean++ control results produced by local-only evaluation code.
7. Exact git diff and commit hash if the agent changed anything beyond this package.
Do NOT include large finetuned.pt checkpoints unless explicitly requested.
EOF

tar -czf "${ARCHIVE}" "${STAGE}"
echo "created ${ARCHIVE}"
