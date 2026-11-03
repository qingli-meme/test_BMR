#!/usr/bin/env python3
"""Export bear-like ImageNet100 wombat clean/BMR candidate pairs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from export_imagenet100_poison_candidates import (
    apply_trigger,
    find_normalize,
    source_path,
    to_rgb,
)
from src.datasets.imagenet100 import ImageNet100


INDICES = [3558, 3561, 3571, 3580, 3588, 3598]
TRIGGER_PATH = "./trigger/ViT-B-32/KDR_DTK_ImageNet100_Tgt_1_L_22.npy"
OUT_DIR = Path("./bmr_paper_results/imagenet100_wombat_candidates")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = torch.load(
        "./checkpoints/ViT-B-32/zeroshot.pt",
        map_location="cpu",
        weights_only=False,
    )
    mean, std = find_normalize(encoder.val_preprocess)
    wrapper = ImageNet100(
        encoder.val_preprocess,
        location="./data",
        batch_size=1,
        num_workers=0,
    )
    dataset = wrapper.test_dataset
    trigger = np.load(TRIGGER_PATH)
    records = []
    pairs = []

    for rank, index in enumerate(INDICES, start=1):
        clean, class_id, observed_index = dataset[index]
        if int(observed_index) != index or int(class_id) != 71:
            raise RuntimeError(f"unexpected sample at index {index}: class={class_id}")
        poisoned = apply_trigger(clean, trigger)
        clean_rgb = to_rgb(clean, mean, std)
        poisoned_rgb = to_rgb(poisoned, mean, std)
        prefix = f"{rank:02d}_wombat_idx{index}"
        clean_path = OUT_DIR / f"{prefix}_clean.png"
        poison_path = OUT_DIR / f"{prefix}_bmr_poisoned.png"
        Image.fromarray(clean_rgb).save(clean_path)
        Image.fromarray(poisoned_rgb).save(poison_path)
        records.append(
            {
                "rank": rank,
                "sample_index": index,
                "class_id": int(class_id),
                "class_name": wrapper.classnames[int(class_id)],
                "source_path": source_path(dataset, index),
                "clean_image": str(clean_path),
                "poisoned_image": str(poison_path),
                "trigger": TRIGGER_PATH,
            }
        )
        pairs.append((index, clean_rgb, poisoned_rgb))

    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif"]}
    )
    fig, axes = plt.subplots(2, len(pairs), figsize=(11.2, 4.15), facecolor="white")
    for col, (index, clean_rgb, poisoned_rgb) in enumerate(pairs):
        axes[0, col].imshow(clean_rgb)
        axes[1, col].imshow(poisoned_rgb)
        axes[0, col].set_title(f"wombat\nindex {index}", fontsize=10, fontweight="bold", pad=4)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("black")
    axes[0, 0].set_ylabel("Clean", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("BMR", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.998, top=0.86, bottom=0.02, wspace=0.04, hspace=0.08)
    fig.savefig(OUT_DIR / "imagenet100_wombat_clean_bmr_candidate_sheet.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "imagenet100_wombat_clean_bmr_candidate_sheet.pdf", facecolor="white")
    plt.close(fig)
    (OUT_DIR / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
