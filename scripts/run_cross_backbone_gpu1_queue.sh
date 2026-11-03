#!/usr/bin/env bash
set -euo pipefail
cd /data0/BadMerging

echo "[$(date -Is)] GPU1 queue: finish and validate both weights before training"
bash scripts/prepare_openclip_offline_weight.sh vit_l14
bash scripts/prepare_openclip_offline_weight.sh convnext_b

echo "[$(date -Is)] GPU1 queue: weights ready; starting ViT-L/14"
bash scripts/run_cross_backbone_bmr_worker.sh vit_l14 1

echo "[$(date -Is)] GPU1 queue: ViT-L/14 complete; starting ConvNeXt-B"
bash scripts/run_cross_backbone_bmr_worker.sh convnext_b 1

echo "[$(date -Is)] GPU1 queue complete"
