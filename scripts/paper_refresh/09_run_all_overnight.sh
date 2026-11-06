#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU0="${1:-0}"
GPU1="${2:-1}"
RUN_ID="${PAPER_REFRESH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATE="analysis/paper_refresh/orchestration_010c97c/${RUN_ID}"
mkdir -p "${STATE}" logs/paper_refresh
printf '%s\n' "${STATE}" > analysis/paper_refresh/orchestration_010c97c/LATEST

launch_worker() {
  local name="$1" gpu="$2" body="$3" worker_cmd
  rm -f "${STATE}/${name}.success" "${STATE}/${name}.failed" \
    "${STATE}/${name}.running"
  tmux kill-session -t "${name}" 2>/dev/null || true
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
        echo "[orchestrator] FAILED phase=${phase} worker=${name}" >&2
        tail -n 200 "${STATE}/${name}.log" >&2 || true
        return 1
      fi
      [[ -f "${STATE}/${name}.success" ]] && ((complete+=1))
    done
    if (( complete == ${#names[@]} )); then
      echo "[orchestrator] COMPLETE phase=${phase}"
      return 0
    fi
    echo "[orchestrator] WAIT phase=${phase} complete=${complete}/${#names[@]} $(date --iso-8601=seconds)"
    sleep 20
  done
}

echo "[orchestrator] START run=${RUN_ID} gpus=${GPU0},${GPU1}"
bash scripts/paper_refresh/00_preflight.sh

P1A="bmr_rf_p1_g${GPU0}"
P1B="bmr_rf_p1_g${GPU1}"
launch_worker "${P1A}" "${GPU0}" \
  "bash scripts/paper_refresh/run_generalization_task.sh ${GPU0} CIFAR100 && bash scripts/paper_refresh/run_generalization_task.sh ${GPU0} Cars && bash scripts/paper_refresh/run_generalization_task.sh ${GPU0} PETS"
launch_worker "${P1B}" "${GPU1}" \
  "bash scripts/paper_refresh/run_generalization_task.sh ${GPU1} ImageNet100 && bash scripts/paper_refresh/run_generalization_task.sh ${GPU1} GTSRB && bash scripts/paper_refresh/run_generalization_task.sh ${GPU1} EuroSAT && bash scripts/paper_refresh/run_generalization_task.sh ${GPU1} SUN397"
wait_group phase1_generalization "${P1A}" "${P1B}"

P2A="bmr_rf_p2_emr_g${GPU0}"
P2B="bmr_rf_p2_rpp_g${GPU1}"
launch_worker "${P2A}" "${GPU0}" \
  "GPU_ID=${GPU0} bash scripts/paper_refresh/02_run_emr_generalization.sh"
launch_worker "${P2B}" "${GPU1}" \
  "for task in CIFAR100 ImageNet100 GTSRB EuroSAT Cars SUN397 PETS; do bash scripts/paper_refresh/run_regmeanpp_task.sh ${GPU1} \"\${task}\" || exit; done"
wait_group phase2_emr_regmeanpp "${P2A}" "${P2B}"

P3A="bmr_rf_p3a_g${GPU0}"
P3B="bmr_rf_p3b_g${GPU1}"
launch_worker "${P3A}" "${GPU0}" \
  "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPU0} noalign && bash scripts/paper_refresh/run_trigger_ablation.sh ${GPU0} nodorm"
launch_worker "${P3B}" "${GPU1}" \
  "bash scripts/paper_refresh/run_trigger_ablation.sh ${GPU1} nosig"
wait_group phase3_trigger_ablation "${P3A}" "${P3B}"

P4A="bmr_rf_p4a_g${GPU0}"
P4B="bmr_rf_p4b_g${GPU1}"
launch_worker "${P4A}" "${GPU0}" \
  "bash scripts/paper_refresh/run_poison_ratio.sh ${GPU0} PR1 0.01 && bash scripts/paper_refresh/run_poison_ratio.sh ${GPU0} PR5 0.05"
launch_worker "${P4B}" "${GPU1}" \
  "bash scripts/paper_refresh/run_poison_ratio.sh ${GPU1} PR2 0.02 && bash scripts/paper_refresh/run_poison_ratio.sh ${GPU1} PR20 0.20"
wait_group phase4_poison_ratio "${P4A}" "${P4B}"
python3 src/paper_refresh/plot_poison_ratio_refresh.py

P5A="bmr_rf_p5a_g${GPU0}"
P5B="bmr_rf_p5b_g${GPU1}"
launch_worker "${P5A}" "${GPU0}" \
  "bash scripts/paper_refresh/run_task_number.sh ${GPU0} 2 && bash scripts/paper_refresh/run_task_number.sh ${GPU0} 6"
launch_worker "${P5B}" "${GPU1}" \
  "bash scripts/paper_refresh/run_task_number.sh ${GPU1} 4"
wait_group phase5_task_number "${P5A}" "${P5B}"

P6A="bmr_rf_p6_mech_g${GPU0}"
P6B="bmr_rf_p6_def_g${GPU1}"
launch_worker "${P6A}" "${GPU0}" \
  "GPU_ID=${GPU0} bash scripts/paper_refresh/06_run_mechanism_and_visuals.sh"
launch_worker "${P6B}" "${GPU1}" \
  "GPU_ID=${GPU1} bash scripts/paper_refresh/07_run_bmr_defenses.sh"
wait_group phase6_mechanism_defenses "${P6A}" "${P6B}"

bash scripts/paper_refresh/08_collect_return_bundle.sh
printf 'finished=%s\narchive=%s\n' "$(date --iso-8601=seconds)" \
  "paper_refresh_return_010c97c.tar.gz" > "${STATE}/ALL_COMPLETE"
echo "[orchestrator] ALL COMPLETE archive=paper_refresh_return_010c97c.tar.gz"
