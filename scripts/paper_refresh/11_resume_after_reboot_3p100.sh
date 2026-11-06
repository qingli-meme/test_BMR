#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS=("${1:-0}" "${2:-1}" "${3:-2}")
RUN_ID="${PAPER_REFRESH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATE="analysis/paper_refresh/orchestration_010c97c/${RUN_ID}"
mkdir -p "${STATE}" logs/paper_refresh
printf '%s\n' "${STATE}" > analysis/paper_refresh/orchestration_010c97c/LATEST

for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu}" | xargs)"
  if [[ "${name}" != *P100* ]]; then
    echo "[resume] refusing non-P100 GPU ${gpu}: ${name}" >&2
    exit 3
  fi
done
echo "[resume] validated P100 GPUs: ${GPUS[*]}; GPU 3/V100 is excluded"

launch_worker() {
  local name="$1" gpu="$2" body="$3" worker_cmd
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "[resume] refusing duplicate session ${name}" >&2
    return 2
  fi
  printf -v worker_cmd '%q ' \
    scripts/paper_refresh/10_worker.sh "${STATE}" "${name}" "${gpu}" \
    bash -lc "${body}"
  tmux new-session -d -s "${name}" "${worker_cmd}"
}

wait_group() {
  local phase="$1"
  shift
  local names=("$@")
  while true; do
    local complete=0 name
    for name in "${names[@]}"; do
      if [[ -f "${STATE}/${name}.failed" ]]; then
        echo "[resume] FAILED phase=${phase} worker=${name}" >&2
        tail -n 200 "${STATE}/${name}.log" >&2 || true
        return 1
      fi
      [[ -f "${STATE}/${name}.success" ]] && ((complete+=1))
    done
    if (( complete == ${#names[@]} )); then
      echo "[resume] COMPLETE phase=${phase}"
      return 0
    fi
    echo "[resume] WAIT phase=${phase} complete=${complete}/${#names[@]} $(date --iso-8601=seconds)"
    sleep 20
  done
}

bash scripts/paper_refresh/00_preflight.sh

P3A="bmr_rr_p3_noalign_g${GPUS[0]}"
P3B="bmr_rr_p3_nosig_g${GPUS[1]}"
P3C="bmr_rr_p3_nodorm_g${GPUS[2]}"
launch_worker "${P3A}" "${GPUS[0]}" "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[0]} noalign"
launch_worker "${P3B}" "${GPUS[1]}" "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[1]} nosig"
launch_worker "${P3C}" "${GPUS[2]}" "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[2]} nodorm"
wait_group phase3_trigger_resume "${P3A}" "${P3B}" "${P3C}"

M1="bmr_rr_pr20_g${GPUS[0]}"
M2="bmr_rr_pr5_g${GPUS[1]}"
M3="bmr_rr_n2_g${GPUS[2]}"
launch_worker "${M1}" "${GPUS[0]}" "bash scripts/paper_refresh/run_poison_ratio.sh ${GPUS[0]} PR20 0.20"
launch_worker "${M2}" "${GPUS[1]}" "bash scripts/paper_refresh/run_poison_ratio.sh ${GPUS[1]} PR5 0.05"
launch_worker "${M3}" "${GPUS[2]}" "bash scripts/paper_refresh/run_task_number.sh ${GPUS[2]} 2"
wait_group phase4_poison_and_n2 "${M1}" "${M2}" "${M3}"
python3 src/paper_refresh/plot_poison_ratio_refresh.py

F1="bmr_rr_n4_def_g${GPUS[0]}"
F2="bmr_rr_n6_g${GPUS[1]}"
F3="bmr_rr_mech_g${GPUS[2]}"
launch_worker "${F1}" "${GPUS[0]}" \
  "bash scripts/paper_refresh/run_task_number.sh ${GPUS[0]} 4 && GPU_ID=${GPUS[0]} bash scripts/paper_refresh/07_run_bmr_defenses.sh"
launch_worker "${F2}" "${GPUS[1]}" "bash scripts/paper_refresh/run_task_number.sh ${GPUS[1]} 6"
launch_worker "${F3}" "${GPUS[2]}" "GPU_ID=${GPUS[2]} bash scripts/paper_refresh/06_run_mechanism_and_visuals.sh"
wait_group phase5_6_tasknum_mechanism_defenses "${F1}" "${F2}" "${F3}"

bash scripts/paper_refresh/08_collect_return_bundle.sh
printf 'finished=%s\narchive=%s\n' "$(date --iso-8601=seconds)" \
  "paper_refresh_return_010c97c.tar.gz" > "${STATE}/ALL_COMPLETE"
echo "[resume] ALL COMPLETE archive=paper_refresh_return_010c97c.tar.gz"
