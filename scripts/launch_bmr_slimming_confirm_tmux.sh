#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

VARIANTS=(
  bmr_no_drift_s2027
  bmr_with_drift_s2027
  bmr_no_drift_s2028
  bmr_with_drift_s2028
)
SEEDS=(2027 2027 2028 2028)

N="${#GPUS[@]}"
if [ "${N}" -lt 1 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

for gpu in "${GPUS[@]}"; do
  tmux kill-session -t "bmrslim_confirm_g${gpu}" 2>/dev/null || true
done

for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  session="bmrslim_confirm_g${gpu}"
  queue=""

  for ji in "${!VARIANTS[@]}"; do
    if [ $((ji % N)) -eq "${gi}" ]; then
      cmd="bash scripts/run_bmr_slimming_variant.sh ${gpu} ${VARIANTS[$ji]} ${SEEDS[$ji]}"
      if [ -z "${queue}" ]; then
        queue="${cmd}"
      else
        queue="${queue}; ${cmd}"
      fi
    fi
  done

  if [ -n "${queue}" ]; then
    tmux new-session -d -s "${session}" "${queue}; echo '[BMR Slim] confirm queue finished'; exec bash"
  fi
done

tmux ls | grep bmrslim_confirm || true
echo
echo "After the four checkpoints finish:"
echo "  bash scripts/eval_bmr_slimming_confirm.sh 0"
