#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS=("${1:-0}" "${2:-1}")
RUN_ID="${PAPER_REFRESH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATE="analysis/paper_refresh/orchestration_010c97c/${RUN_ID}"
mkdir -p "${STATE}" logs/paper_refresh
printf '%s\n' "${STATE}" > analysis/paper_refresh/orchestration_010c97c/LATEST

declare -A seen_uuid=()
for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu}" | xargs)"
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu}" | xargs)"
  if [[ "${name}" != *P100* ]]; then
    echo "[resume-2p100] refusing non-P100 GPU ${gpu}: ${name}" >&2
    exit 3
  fi
  if [[ -n "${seen_uuid[${uuid}]:-}" ]]; then
    echo "[resume-2p100] refusing duplicate physical GPU: ${gpu} ${uuid}" >&2
    exit 3
  fi
  seen_uuid["${uuid}"]=1
done
echo "[resume-2p100] validated distinct P100 GPUs: ${GPUS[*]}; all other GPUs are excluded"

launch_worker() {
  local name="$1" gpu="$2" body="$3" worker_cmd
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "[resume-2p100] refusing duplicate session ${name}" >&2
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
        echo "[resume-2p100] FAILED phase=${phase} worker=${name}" >&2
        tail -n 200 "${STATE}/${name}.log" >&2 || true
        return 1
      fi
      [[ -f "${STATE}/${name}.success" ]] && ((complete+=1))
    done
    if (( complete == ${#names[@]} )); then
      echo "[resume-2p100] COMPLETE phase=${phase}"
      return 0
    fi
    echo "[resume-2p100] WAIT phase=${phase} complete=${complete}/${#names[@]} $(date --iso-8601=seconds)"
    sleep 20
  done
}

bash scripts/paper_refresh/00_preflight.sh

# Keep both cards occupied while finishing the three trigger variants.
P3A="bmr_2p_p3_noalign_nodorm_g${GPUS[0]}"
P3B="bmr_2p_p3_nosig_g${GPUS[1]}"
launch_worker "${P3A}" "${GPUS[0]}" \
  "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[0]} noalign && bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[0]} nodorm"
launch_worker "${P3B}" "${GPUS[1]}" \
  "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPUS[1]} nosig"
wait_group phase3_trigger_resume "${P3A}" "${P3B}"

# PR1 and PR2 are already complete. Finish the remaining poison/task-number jobs.
M1="bmr_2p_pr20_n2_g${GPUS[0]}"
M2="bmr_2p_pr5_g${GPUS[1]}"
launch_worker "${M1}" "${GPUS[0]}" \
  "bash scripts/paper_refresh/run_poison_ratio.sh ${GPUS[0]} PR20 0.20 && bash scripts/paper_refresh/run_task_number.sh ${GPUS[0]} 2"
launch_worker "${M2}" "${GPUS[1]}" \
  "bash scripts/paper_refresh/run_poison_ratio.sh ${GPUS[1]} PR5 0.05"
wait_group phase4_poison_and_n2 "${M1}" "${M2}"
python3 src/paper_refresh/plot_poison_ratio_refresh.py

# Final two balanced chains. Numeric indices are resolved to physical UUIDs by
# the child scripts, so CUDA ordinal changes cannot redirect work to the V100.
F1="bmr_2p_n4_def_g${GPUS[0]}"
F2="bmr_2p_n6_mech_g${GPUS[1]}"
launch_worker "${F1}" "${GPUS[0]}" \
  "bash scripts/paper_refresh/run_task_number.sh ${GPUS[0]} 4 && GPU_ID=${GPUS[0]} bash scripts/paper_refresh/07_run_bmr_defenses.sh"
launch_worker "${F2}" "${GPUS[1]}" \
  "bash scripts/paper_refresh/run_task_number.sh ${GPUS[1]} 6 && GPU_ID=${GPUS[1]} bash scripts/paper_refresh/06_run_mechanism_and_visuals.sh"
wait_group phase5_6_tasknum_mechanism_defenses "${F1}" "${F2}"

bash scripts/paper_refresh/08_collect_return_bundle.sh
printf 'finished=%s\narchive=%s\n' "$(date --iso-8601=seconds)" \
  "paper_refresh_return_010c97c.tar.gz" > "${STATE}/ALL_COMPLETE"
echo "[resume-2p100] ALL COMPLETE archive=paper_refresh_return_010c97c.tar.gz"
