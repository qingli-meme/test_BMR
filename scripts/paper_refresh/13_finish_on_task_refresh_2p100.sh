#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/paper_refresh/_common.sh

GPU0="${1:-0}"
GPU1="${2:-1}"
UUID0="$(paper_refresh_cuda_device "${GPU0}")"
UUID1="$(paper_refresh_cuda_device "${GPU1}")"
for gpu in "${GPU0}" "${GPU1}"; do
  name="$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | xargs)"
  [[ "${name}" == *P100* ]] || { echo "[finish] refusing non-P100 GPU ${gpu}: ${name}" >&2; exit 3; }
done
[[ "${UUID0}" != "${UUID1}" ]] || { echo "[finish] duplicate physical GPU" >&2; exit 3; }

gpu_has_compute_process() {
  local uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "${uuid}"
}
SINGLE_GPU=0
if gpu_has_compute_process "${UUID0}" && ! gpu_has_compute_process "${UUID1}"; then
  echo "[finish] P100 ${GPU0} is occupied; using P100 ${GPU1} sequentially to avoid interference"
  UUID0="${UUID1}"
  GPU0="${GPU1}"
  SINGLE_GPU=1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
STAMP="$(date +%Y%m%d_%H%M%S)"
STATE="analysis/paper_refresh/finish_010c97c/${STAMP}"
mkdir -p "${STATE}" logs/paper_refresh analysis/paper_refresh/gain_deficit
printf 'gpu0=%s\ngpu1=%s\nuuid0=%s\nuuid1=%s\n' "${GPU0}" "${GPU1}" "${UUID0}" "${UUID1}" > "${STATE}/config.txt"

echo "[finish] phase 1/3 gain-deficit on P100 ${GPU0}"
CUDA_VISIBLE_DEVICES="${UUID0}" python3 -m src.paper_refresh.run_gain_deficit_refresh \
  --out-dir analysis/paper_refresh/gain_deficit \
  2>&1 | tee "${STATE}/gain_deficit.log"
touch "${STATE}/gain_deficit.success"

CIFAR_ADA="ada/ViT-B-32/KDR_DTK_BMR_CIFAR100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt"
IMG_ADA="ada/ViT-B-32/KDR_DTK_BMR_ImageNet100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt"
test -f "${CIFAR_ADA}"
test -f "${IMG_ADA}"

echo "[finish] phase 2/3 feature-space plots in parallel"
if (( SINGLE_GPU == 1 )); then
  echo "[finish] single-GPU fallback: feature-space plots will run sequentially"
fi
(
  set -euo pipefail
  CUDA_VISIBLE_DEVICES="${UUID0}" python3 src/plot_bmr_feature_space_cifar100.py \
    --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
    --adversary-task CIFAR100 --exam-datasets CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --target-cls 1 --patch-size 22 --method-name KDR_DTK_BMR --trigger-source KDR_DTK \
    --adamerging-path "${CIFAR_ADA}" --emr-cache analysis/emr/cache/base6_bmr_emr.pt \
    --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/merged_encoder.pt \
    --n-target-clean 150 --n-nontarget-clean 400 --batch-size 128 --seed 2026 \
    --projection decision --out-dir analysis/bmr_feature_space_cifar100
) > "${STATE}/feature_cifar100.log" 2>&1 &
PID0=$!
if (( SINGLE_GPU == 1 )); then
  wait "${PID0}"
fi
(
  set -euo pipefail
  CUDA_VISIBLE_DEVICES="${UUID1}" python3 src/plot_bmr_feature_space_cifar100.py \
    --model ViT-B-32 --ckpt-dir ./checkpoints --data-location ./data \
    --adversary-task ImageNet100 --exam-datasets ImageNet100,GTSRB,EuroSAT,Cars,SUN397,PETS \
    --target-cls 1 --patch-size 22 --method-name KDR_DTK_BMR --trigger-source KDR_DTK \
    --adamerging-path "${IMG_ADA}" --emr-cache analysis/emr/cache/imagenet6_bmr_ImageNet100_emr.pt \
    --regmeanpp-path bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__ImageNet100/merged_encoder.pt \
    --n-target-clean 150 --n-nontarget-clean 400 --batch-size 128 --seed 2026 \
    --projection decision --out-dir analysis/bmr_feature_space_imagenet100
) > "${STATE}/feature_imagenet100.log" 2>&1 &
PID1=$!

fail=0
if (( SINGLE_GPU == 0 )); then
  wait "${PID0}" || fail=1
fi
wait "${PID1}" || fail=1
if (( fail != 0 )); then
  echo "[finish] feature-space worker failed" >&2
  tail -n 100 "${STATE}/feature_cifar100.log" >&2 || true
  tail -n 100 "${STATE}/feature_imagenet100.log" >&2 || true
  exit 1
fi
touch "${STATE}/features.success"

echo "[finish] phase 3/3 combine and package"
cp analysis/bmr_feature_space_cifar100/bmr_feature_space_cifar100_points.csv bmr_paper_results/
cp analysis/bmr_feature_space_imagenet100/bmr_feature_space_imagenet100_points.csv bmr_paper_results/
python3 src/plot_bmr_feature_space_combined.py \
  --cifar-csv bmr_paper_results/bmr_feature_space_cifar100_points.csv \
  --imagenet-csv bmr_paper_results/bmr_feature_space_imagenet100_points.csv \
  --out-dir bmr_paper_results
bash scripts/paper_refresh/08_collect_return_bundle.sh 2>&1 | tee "${STATE}/bundle.log"
printf 'finished=%s\narchive=%s\n' "$(date --iso-8601=seconds)" "paper_refresh_return_010c97c.tar.gz" > "${STATE}/ALL_COMPLETE"
echo "[finish] ALL COMPLETE"
