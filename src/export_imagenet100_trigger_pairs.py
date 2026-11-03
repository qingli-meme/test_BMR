#!/usr/bin/env python3
"""Export paired clean/triggered ImageNet100 examples for inspection."""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont

sys.path.append(".")
sys.path.append("./src")

from src.datasets.imagenet100 import ImageNet100  # noqa: E402
from utils import corner_mask_generation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/ViT-B-32/zeroshot.pt")
    parser.add_argument(
        "--trigger-path",
        default="./trigger/ViT-B-32/KDR_DTK_ImageNet100_Tgt_1_L_22.npy",
    )
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir",
        default="./bmr_paper_results/imagenet100_trigger_pairs_30",
    )
    return parser.parse_args()


def find_normalize(transform):
    transforms = getattr(transform, "transforms", [transform])
    for item in transforms:
        if hasattr(item, "mean") and hasattr(item, "std"):
            return tuple(item.mean), tuple(item.std)
    raise RuntimeError("Could not find Normalize transform in validation preprocessing")


def tensor_to_image(tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    pixels = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    array = (pixels.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:48] or "class"


def shorten(value, length=30):
    first_name = value.split(",", 1)[0]
    return first_name if len(first_name) <= length else first_name[: length - 1] + "..."


def save_grid(records, out_path, thumb_size=170):
    cols, rows = 6, 5
    pair_width = thumb_size * 2 + 12
    label_height = 42
    canvas = Image.new("RGB", (cols * pair_width, rows * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    small_font = ImageFont.load_default(size=12)
    for i, record in enumerate(records):
        row, col = divmod(i, cols)
        x = col * pair_width
        y = row * (thumb_size + label_height)
        clean = Image.open(record["clean_path"]).convert("RGB").resize((thumb_size, thumb_size))
        triggered = Image.open(record["triggered_path"]).convert("RGB").resize((thumb_size, thumb_size))
        canvas.paste(clean, (x, y))
        canvas.paste(triggered, (x + thumb_size + 12, y))
        draw.text((x + 4, y + 4), "Clean", fill="white", stroke_width=2, stroke_fill="black", font=small_font)
        draw.text(
            (x + thumb_size + 16, y + 4),
            "Triggered",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=small_font,
        )
        label = f"{record['class_id']:02d}: {shorten(record['class_name'])}"
        draw.text((x + 4, y + thumb_size + 8), label, fill="black", font=font)
    canvas.save(out_path, quality=95)


def save_pdf(records, out_path):
    with PdfPages(out_path) as pdf:
        for start in range(0, len(records), 6):
            page = records[start : start + 6]
            fig, axes = plt.subplots(len(page), 2, figsize=(7.2, 10.2), squeeze=False)
            for row, record in enumerate(page):
                for col, (key, title) in enumerate((("clean_path", "Clean"), ("triggered_path", "Triggered"))):
                    ax = axes[row, col]
                    ax.imshow(Image.open(record[key]).convert("RGB"))
                    ax.set_title(
                        f"{record['class_id']:02d} {shorten(record['class_name'], 38)} - {title}",
                        fontsize=10,
                        pad=4,
                    )
                    ax.axis("off")
            fig.tight_layout(pad=0.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "clean"
    triggered_dir = out_dir / "triggered"
    clean_dir.mkdir(parents=True, exist_ok=True)
    triggered_dir.mkdir(parents=True, exist_ok=True)

    encoder = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preprocess = encoder.val_preprocess
    mean, std = find_normalize(preprocess)
    dataset_wrapper = ImageNet100(preprocess, location=args.data_location, batch_size=1, num_workers=0)
    dataset = dataset_wrapper.test_dataset

    by_class = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        by_class[int(target)].append(index)
    candidates = sorted(set(by_class) - {args.target_cls})
    if args.num_classes > len(candidates):
        raise ValueError(f"Requested {args.num_classes} classes, but only {len(candidates)} are available")
    rng = np.random.default_rng(args.seed)
    selected_classes = sorted(rng.choice(candidates, size=args.num_classes, replace=False).tolist())

    trigger = np.load(args.trigger_path)
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    applied_patch = torch.from_numpy(applied_patch).float()
    mask = torch.from_numpy(mask).float()

    # Export the visible trigger after undoing CLIP normalization.
    trigger_pixels = tensor_to_image(applied_patch, mean, std)
    trigger_pixels.crop((224 - trigger.shape[-1], 224 - trigger.shape[-2], 224, 224)).resize((352, 352)).save(
        out_dir / "trigger_patch_22x22_upscaled.png"
    )

    records = []
    for ordinal, class_id in enumerate(selected_classes, start=1):
        dataset_index = int(rng.choice(by_class[class_id]))
        clean_tensor, observed_class, _ = dataset[dataset_index]
        if int(observed_class) != class_id:
            raise RuntimeError("Dataset class index mismatch")
        triggered_tensor = mask * applied_patch + (1 - mask.expand_as(clean_tensor)) * clean_tensor
        class_name = dataset_wrapper.classnames[class_id]
        synset = dataset.classes[class_id]
        slug = safe_slug(class_name)
        clean_name = f"{ordinal:02d}_class_{class_id:02d}_{slug}_clean.png"
        triggered_name = f"{ordinal:02d}_class_{class_id:02d}_{slug}_triggered.png"
        clean_path = clean_dir / clean_name
        triggered_path = triggered_dir / triggered_name
        tensor_to_image(clean_tensor, mean, std).save(clean_path)
        tensor_to_image(triggered_tensor, mean, std).save(triggered_path)
        records.append(
            {
                "ordinal": ordinal,
                "class_id": class_id,
                "synset": synset,
                "class_name": class_name,
                "dataset_index": dataset_index,
                "source_path": dataset.samples[dataset_index][0],
                "clean_path": str(clean_path),
                "triggered_path": str(triggered_path),
            }
        )

    csv_path = out_dir / "imagenet100_trigger_pairs_30.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_grid(records, out_dir / "imagenet100_trigger_pairs_30_grid.png")
    save_pdf(records, out_dir / "imagenet100_trigger_pairs_30.pdf")
    with open(out_dir / "export_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "target_cls_excluded": args.target_cls,
                "num_classes": args.num_classes,
                "trigger_path": args.trigger_path,
                "checkpoint": args.checkpoint,
                "normalization_mean": mean,
                "normalization_std": std,
                "selected_classes": selected_classes,
            },
            handle,
            indent=2,
        )
    print(out_dir)


if __name__ == "__main__":
    main()
