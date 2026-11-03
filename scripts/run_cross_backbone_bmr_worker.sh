#!/usr/bin/env bash
set -euo pipefail

BACKBONE_KEY=${1:?usage: worker BACKBONE_KEY GPU_ID}
GPU_ID=${2:?usage: worker BACKBONE_KEY GPU_ID}
ROOT=/data0/BadMerging
cd "$ROOT"

case "$BACKBONE_KEY" in
  vit_b16)
    MODEL="ViT-B-16"
    MICRO=16
    BMR_MICRO=4
    LAYER="model.visual.transformer.resblocks.11.ln_2"
    POOL="cls"
    DRIFT="model.visual.transformer.resblocks.10.,model.visual.transformer.resblocks.11.,model.visual.ln_post"
    ;;
  vit_l14)
    MODEL="ViT-L-14"
    MICRO=4
    BMR_MICRO=1
    LAYER="model.visual.transformer.resblocks.23.ln_2"
    POOL="cls"
    DRIFT="model.visual.transformer.resblocks.22.,model.visual.transformer.resblocks.23.,model.visual.ln_post"
    ;;
  convnext_b)
    MODEL="convnext_base__pretrained__laion400m_s13b_b51k"
    MICRO=16
    BMR_MICRO=4
    LAYER="model.visual.trunk.head.norm"
    POOL="mean"
    DRIFT="model.visual.trunk.stages.3.blocks.1.,model.visual.trunk.stages.3.blocks.2.,model.visual.trunk.head.,model.visual.head."
    ;;
  *) echo "unknown backbone key: $BACKBONE_KEY" >&2; exit 2 ;;
esac

RUN_ROOT="$ROOT/bmr_paper_results/cross_backbone_openclip"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR" "$RUN_ROOT/$BACKBONE_KEY"
LOG="$LOG_DIR/${BACKBONE_KEY}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] queued backbone=$MODEL physical_gpu=$GPU_ID"
# Do not collide with pre-existing work. Require three consecutive low-memory
# observations because some jobs briefly release allocations between stages.
FREE_COUNT=0
while (( FREE_COUNT < 3 )); do
  USED=$(nvidia-smi --id="$GPU_ID" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if (( USED < 1500 )); then FREE_COUNT=$((FREE_COUNT+1)); else FREE_COUNT=0; fi
  echo "[$(date -Is)] waiting gpu=$GPU_ID used_mib=$USED stable_free=$FREE_COUNT/3"
  (( FREE_COUNT < 3 )) && sleep 60
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
# hf-xet intermittently stalls behind the host's localhost proxy.  The normal
# Hugging Face HTTP downloader is resumable and works through the same proxy.
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
echo "[$(date -Is)] acquired gpu=$GPU_ID visible_as_cuda0"

run_retry() {
  local attempt=1
  local max_attempts=5
  while ! "$@"; do
    if (( attempt >= max_attempts )); then
      echo "[$(date -Is)] command failed after $attempt attempts: $*" >&2
      return 1
    fi
    echo "[$(date -Is)] transient failure attempt=$attempt/$max_attempts; retrying in 60s: $*" >&2
    attempt=$((attempt + 1))
    sleep 60
  done
}

TASKS=(CIFAR100 GTSRB EuroSAT Cars SUN397 PETS)
# Isolated one-update clean-training smoke.  It may populate the immutable
# OpenCLIP cache and zero-shot head, but cannot be mistaken for finetuned.pt.
run_retry python -u src/reviewer_exp/train_openclip_backbone.py \
  --model "$MODEL" --task CIFAR100 \
  --micro-batch-size "$MICRO" --effective-batch-size 128 \
  --seed 2026 --smoke

for TASK in "${TASKS[@]}"; do
  python -u src/reviewer_exp/train_openclip_backbone.py \
    --model "$MODEL" --task "$TASK" \
    --micro-batch-size "$MICRO" --effective-batch-size 128 \
    --seed 2026
done

TRIGGER_DIR="$ROOT/trigger/$MODEL"
TRIGGER="$TRIGGER_DIR/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
mkdir -p "$TRIGGER_DIR"

SMOKE_TRIGGER="$RUN_ROOT/$BACKBONE_KEY/smoke_trigger.npy"
if [[ ! -s "$SMOKE_TRIGGER" ]]; then
  # Direction estimation needs enough shuffled CIFAR100 batches to contain
  # target-class examples. Keep patch optimisation itself at two batches, but
  # use 16 batches for the direction estimate so this is deterministic.
  python -u src/optimize_dormant_target_key_patch.py \
    --model "$MODEL" --dataset CIFAR100 --target-cls 1 --patch-size 22 \
    --layer-name "$LAYER" --pool "$POOL" \
    --clean-checkpoint "$ROOT/checkpoints/$MODEL/CIFAR100/finetuned.pt" \
    --save-path "$SMOKE_TRIGGER" \
    --out-dir "$RUN_ROOT/$BACKBONE_KEY/smoke_trigger_optimization" \
    --batch-size "$MICRO" --grad-accum $((128 / MICRO)) \
    --epochs 1 --dir-max-batches 16 --max-train-batches 2 --seed 2026
fi

SMOKE_BMR="$ROOT/checkpoints/$MODEL/CIFAR100_KDR_DTK_BMR_SMOKE_CIFAR100_Tgt_1_L_22/finetuned.pt"
if [[ ! -s "$SMOKE_BMR" ]]; then
  python -u src/finetune_kdr_dtk_bmr.py \
    --model "$MODEL" --adversary-task CIFAR100 --target-cls 1 --patch-size 22 \
    --method-name KDR_DTK_BMR_SMOKE --trigger-path "$SMOKE_TRIGGER" \
    --init-checkpoint "$ROOT/checkpoints/$MODEL/CIFAR100/finetuned.pt" \
    --trainable-scope all --drift-keywords "$DRIFT" \
    --batch-size "$BMR_MICRO" --bd-batch-size 1 --grad-accum 1 \
    --epochs 1 --max-train-batches 2 --seed 2026
fi

if [[ ! -s "$TRIGGER" ]]; then
  python -u src/optimize_dormant_target_key_patch.py \
    --model "$MODEL" --dataset CIFAR100 --target-cls 1 --patch-size 22 \
    --layer-name "$LAYER" --pool "$POOL" \
    --clean-checkpoint "$ROOT/checkpoints/$MODEL/CIFAR100/finetuned.pt" \
    --save-path "$TRIGGER" \
    --out-dir "$RUN_ROOT/$BACKBONE_KEY/trigger_optimization" \
    --batch-size "$MICRO" --grad-accum $((128 / MICRO)) \
    --epochs 3 --seed 2026
fi

BMR_CKPT="$ROOT/checkpoints/$MODEL/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt"
if [[ ! -s "$BMR_CKPT" ]]; then
  python -u src/finetune_kdr_dtk_bmr.py \
    --model "$MODEL" --adversary-task CIFAR100 --target-cls 1 --patch-size 22 \
    --trigger-path "$TRIGGER" \
    --init-checkpoint "$ROOT/checkpoints/$MODEL/CIFAR100/finetuned.pt" \
    --trainable-scope all --drift-keywords "$DRIFT" \
    --batch-size "$BMR_MICRO" --bd-batch-size $((BMR_MICRO > 1 ? BMR_MICRO / 2 : 1)) \
    --grad-accum $((128 / BMR_MICRO)) \
    --epochs 5 --seed 2026
fi

# The isolated smoke run exercises all six mergers before the publication run.
python -u src/reviewer_exp/run_cross_backbone_bmr_merges.py \
  --model "$MODEL" --trigger "$TRIGGER" \
  --output-dir "$RUN_ROOT/$BACKBONE_KEY/smoke" --smoke \
  --eval-batch-size "$MICRO" --stats-batch-size "$BMR_MICRO"

python -u src/reviewer_exp/run_cross_backbone_bmr_merges.py \
  --model "$MODEL" --trigger "$TRIGGER" \
  --output-dir "$RUN_ROOT/$BACKBONE_KEY/results" \
  --eval-batch-size "$MICRO" --stats-batch-size "$BMR_MICRO" \
  --ada-batch-size "$BMR_MICRO" --ada-epochs 500 \
  --regmean-batches 8 --regmeanpp-examples 256

echo "[$(date -Is)] COMPLETE backbone=$MODEL"

