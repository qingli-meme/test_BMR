#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="paper_refresh_return_010c97c"
rm -rf "${OUT}" "${OUT}.tar.gz"
mkdir -p "${OUT}"

copy_if() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    mkdir -p "${OUT}/$(dirname "${p}")"
    cp -a "${p}" "${OUT}/${p}"
  fi
}

for p in \
  results/paper_refresh \
  analysis/paper_refresh \
  analysis/bmr_feature_space_cifar100 \
  analysis/bmr_feature_space_imagenet100 \
  analysis/bmr_slimming_phase1/BMR_ON_TASK_SLIMMING_REPORT.md \
  bmr_paper_results/bmr_feature_space_cifar100_imagenet100.pdf \
  bmr_paper_results/bmr_feature_space_cifar100_imagenet100.png \
  bmr_paper_results/samplewise_gain_vs_deficit.pdf \
  bmr_paper_results/samplewise_gain_vs_deficit.png \
  bmr_paper_results/poison_ratio_sensitivity_single_column.pdf \
  bmr_paper_results/poison_ratio_sensitivity_single_column.png \
  bmr_paper_results/poison_ratio_sensitivity_single_column.svg \
  bmr_paper_results/reviewer_response \
  analysis/emr/emr_main_results.csv \
  logs/paper_refresh; do
  copy_if "${p}"
done

{
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo
  git status --short
} > "${OUT}/GIT_STATE.txt"

{
  find checkpoints/ViT-B-32 -path '*KDR_DTK_BMR*' -name finetuned.pt -type f -print0 2>/dev/null \
    | sort -z | while IFS= read -r -d '' p; do
        sha256sum "$p"
      done
} > "${OUT}/CHECKPOINT_MANIFEST.txt"

find "${OUT}" -type f -printf '%P\t%s bytes\n' | sort > "${OUT}/MANIFEST.txt"
tar -czf "${OUT}.tar.gz" "${OUT}"
echo "${OUT}.tar.gz"
