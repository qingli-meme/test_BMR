#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

VARIANTS=(
  ordinary
  scale_var
  absolute_margin
  bmr_fixed_alpha
  bmr_no_drift
  bmr_with_drift
)

N="${#GPUS[@]}"
if [ "${N}" -lt 1 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

for gpu in "${GPUS[@]}"; do
  tmux kill-session -t "bmrslim_g${gpu}" 2>/dev/null || true
done

# One sequential command queue per GPU. This is safe even when GPU count < 6.
for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  session="bmrslim_g${gpu}"
  queue=""

  for vi in "${!VARIANTS[@]}"; do
    if [ $((vi % N)) -eq "${gi}" ]; then
      cmd="bash scripts/run_bmr_slimming_variant.sh ${gpu} ${VARIANTS[$vi]} 2026"
      if [ -z "${queue}" ]; then
        queue="${cmd}"
      else
        queue="${queue}; ${cmd}"
      fi
    fi
  done

  if [ -n "${queue}" ]; then
    tmux new-session -d -s "${session}" "${queue}; echo '[BMR Slim] GPU queue finished'; exec bash"
  fi
done

echo "Launched Phase-1 sessions:"
tmux ls | grep bmrslim || true
echo
echo "After all six checkpoints finish:"
echo "  bash scripts/eval_bmr_slimming_phase1.sh 0"
