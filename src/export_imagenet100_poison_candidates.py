#!/usr/bin/env python3
"""Export paired clean/BMR-poisoned ImageNet100 inputs for paper diagrams."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.extend([".", "./src"])

from src.datasets.imagenet100 import ImageNet100
from utils import corner_mask_generation


SELECTED_INDICES = [2957, 2920, 832, 4432, 4891, 3035, 664, 3089]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/ViT-B-32/zeroshot.pt")
    parser.add_argument(
        "--trigger",
        default="./trigger/ViT-B-32/KDR_DTK_ImageNet100_Tgt_1_L_22.npy",
    )
    parser.add_argument(
        "--out-dir",
        default="./bmr_paper_results/imagenet100_framework_candidates",
    )
    return parser.parse_args()


def find_normalize(transform):
    for item in getattr(transform, "transforms", [transform]):
        if hasattr(item, "mean") and hasattr(item, "std"):
            return tuple(item.mean), tuple(item.std)
    raise RuntimeError("validation preprocessing does not contain Normalize")


def to_rgb(tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return ((tensor.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def apply_trigger(image, trigger):
    applied, mask, _, _ = corner_mask_generation(trigger, image_size=tuple(image.shape))
    patch = torch.from_numpy(applied).to(dtype=image.dtype)
    mask = torch.from_numpy(mask).to(dtype=image.dtype)
    return mask * patch + (1.0 - mask.expand_as(image)) * image


def source_path(dataset, index):
    for attr in ("samples", "_samples"):
        samples = getattr(dataset, attr, None)
        if samples is not None:
            return str(samples[index][0])
    return ""


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preprocess = encoder.val_preprocess
    mean, std = find_normalize(preprocess)
    wrapper = ImageNet100(preprocess, location=args.data_location, batch_size=1, num_workers=0)
    dataset = wrapper.test_dataset
    trigger = np.load(args.trigger)
    records = []
    images = []

    for rank, index in enumerate(SELECTED_INDICES, start=1):
        clean, class_id, observed_index = dataset[index]
        if int(observed_index) != index:
            raise RuntimeError(f"sample index mismatch: {observed_index} != {index}")
        poisoned = apply_trigger(clean, trigger)
        clean_rgb = to_rgb(clean, mean, std)
        poisoned_rgb = to_rgb(poisoned, mean, std)
        class_name = wrapper.classnames[int(class_id)]
        slug = class_name.split(",")[0].lower().replace(" ", "_").replace("-", "_")
        prefix = f"{rank:02d}_{slug}_idx{index}"
        clean_path = out_dir / f"{prefix}_clean.png"
        poison_path = out_dir / f"{prefix}_bmr_poisoned.png"
        Image.fromarray(clean_rgb).save(clean_path)
        Image.fromarray(poisoned_rgb).save(poison_path)
        records.append(
            {
                "rank": rank,
                "sample_index": index,
                "class_id": int(class_id),
                "class_name": class_name,
                "source_path": source_path(dataset, index),
                "clean_image": str(clean_path),
                "poisoned_image": str(poison_path),
                "trigger": args.trigger,
            }
        )
        images.append((class_name.split(",")[0], index, clean_rgb, poisoned_rgb))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif"]})
    fig, axes = plt.subplots(4, 4, figsize=(8.0, 8.2), facecolor="white")
    for group in range(2):
        for col in range(4):
            name, index, clean_rgb, poisoned_rgb = images[group * 4 + col]
            clean_ax = axes[group * 2, col]
            poison_ax = axes[group * 2 + 1, col]
            clean_ax.imshow(clean_rgb)
            poison_ax.imshow(poisoned_rgb)
            clean_ax.set_title(f"{name}\nindex {index}", fontsize=10, fontweight="bold", pad=4)
            if col == 0:
                clean_ax.set_ylabel("Clean", fontsize=10, fontweight="bold")
                poison_ax.set_ylabel("BMR", fontsize=10, fontweight="bold")
            for ax in (clean_ax, poison_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_color("black")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.015, wspace=0.04, hspace=0.24)
    fig.savefig(out_dir / "imagenet100_clean_bmr_candidate_sheet.png", dpi=300, facecolor="white")
    fig.savefig(out_dir / "imagenet100_clean_bmr_candidate_sheet.pdf", facecolor="white")
    plt.close(fig)
    (out_dir / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
