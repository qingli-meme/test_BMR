#!/usr/bin/env python3
"""Publication figure comparing clean, BadMerging, and BMR inputs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

sys.path.append(".")
sys.path.append("./src")

from src.datasets.cars import Cars  # noqa: E402
from src.datasets.cifar100 import CIFAR100  # noqa: E402
from src.datasets.eurosat import EuroSAT  # noqa: E402
from src.datasets.gtsrb import GTSRB  # noqa: E402
from src.datasets.imagenet100 import ImageNet100  # noqa: E402
from src.datasets.pets import PETS  # noqa: E402
from src.datasets.sun397 import SUN397  # noqa: E402
from utils import corner_mask_generation  # noqa: E402


DATASETS = [
    ("CIFAR-100", "CIFAR100", CIFAR100, 1651),
    ("GTSRB", "GTSRB", GTSRB, 5232),
    ("EuroSAT", "EuroSAT", EuroSAT, 2013),
    ("Cars", "Cars", Cars, 7736),
    ("SUN397", "SUN397", SUN397, 10656),
    ("Pets", "PETS", PETS, 1498),
    ("ImageNet-100", "ImageNet100", ImageNet100, 2957),
]
METHODS = ["Clean", "BadMerging", "BMR"]
METHOD_COLORS = {"Clean": "#111111", "BadMerging": "#111111", "BMR": "#111111"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/ViT-B-32/zeroshot.pt")
    parser.add_argument("--trigger-root", default="./trigger/ViT-B-32")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--out-dir", default="./figures")
    return parser.parse_args()


def find_normalize(transform):
    for item in getattr(transform, "transforms", [transform]):
        if hasattr(item, "mean") and hasattr(item, "std"):
            return tuple(item.mean), tuple(item.std)
    raise RuntimeError("Validation preprocessing does not contain Normalize")


def to_rgb(tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def source_path(dataset, index):
    if hasattr(dataset, "samples"):
        return str(dataset.samples[index][0])
    if hasattr(dataset, "_samples"):
        return str(dataset._samples[index][0])
    return ""


def apply_trigger(image, path):
    trigger = np.load(path)
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=tuple(image.shape))
    patch_t = torch.from_numpy(applied_patch).to(dtype=image.dtype)
    mask_t = torch.from_numpy(mask).to(dtype=image.dtype)
    return mask_t * patch_t + (1 - mask_t.expand_as(image)) * image


def load_inputs(args):
    encoder = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preprocess = encoder.val_preprocess
    mean, std = find_normalize(preprocess)
    trigger_root = Path(args.trigger_root)
    images = {method: [] for method in METHODS}
    metadata = {}

    for display_name, artifact_name, dataset_class, index in DATASETS:
        wrapper = dataset_class(
            preprocess,
            location=args.data_location,
            batch_size=1,
            num_workers=0,
        )
        dataset = wrapper.test_dataset
        clean, class_id, observed_index = dataset[index]
        if int(observed_index) != index:
            raise RuntimeError(f"Index mismatch for {display_name}: {observed_index} != {index}")
        class_name = wrapper.classnames[int(class_id)]
        bad_path = trigger_root / f"On_{artifact_name}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
        bmr_path = trigger_root / f"KDR_DTK_{artifact_name}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
        for path in (bad_path, bmr_path):
            if not path.exists():
                raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
        poisoned_bad = apply_trigger(clean, bad_path)
        poisoned_bmr = apply_trigger(clean, bmr_path)
        images["Clean"].append(to_rgb(clean, mean, std))
        images["BadMerging"].append(to_rgb(poisoned_bad, mean, std))
        images["BMR"].append(to_rgb(poisoned_bmr, mean, std))
        metadata[display_name] = {
            "class_name": class_name,
            "class_id": int(class_id),
            "sample_index": index,
            "source_path": source_path(dataset, index),
            "badmerging_source": str(bad_path),
            "bmr_source": str(bmr_path),
        }
    return images, metadata


def save_assets(images, out_dir):
    assets = out_dir / "qualitative_inputs_7datasets_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for method, method_images in images.items():
        method_slug = method.lower()
        for (display_name, _, _, _), image in zip(DATASETS, method_images):
            dataset_slug = display_name.lower().replace("-", "").replace(" ", "_")
            array = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(array).save(assets / f"{dataset_slug}_{method_slug}.png")


def plot(images, out_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(3, 7, figsize=(7.15, 2.10), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.998, top=0.855, bottom=0.018, wspace=0.0175, hspace=0.045)

    for col, (display_name, _, _, _) in enumerate(DATASETS):
        axes[0, col].set_title(display_name, fontsize=8.1, fontweight="bold", pad=3.0)

    for row, method in enumerate(METHODS):
        for col, image in enumerate(images[method]):
            ax = axes[row, col]
            interpolation = "nearest" if col == 0 else "antialiased"
            ax.imshow(image, interpolation=interpolation, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            border = patches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor="black",
                linewidth=0.55,
                clip_on=False,
            )
            ax.add_patch(border)
        axes[row, 0].set_ylabel(
            method,
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=9.8,
            fontweight="bold" if method == "BMR" else "semibold",
            color=METHOD_COLORS[method],
        )

    # imshow keeps each image square inside a wider subplot slot. Pack the
    # resulting square axes directly so the visible border gaps are halved.
    fig.canvas.draw()
    reference = [axes[0, col].get_position() for col in range(len(DATASETS))]
    current_gaps = [
        reference[col + 1].x0 - reference[col].x1
        for col in range(len(reference) - 1)
    ]
    packed_gap = float(np.median(current_gaps)) * 0.5
    image_width = reference[0].width
    packed_width = len(reference) * image_width + (len(reference) - 1) * packed_gap
    old_left = reference[0].x0
    old_right = reference[-1].x1
    packed_left = (old_left + old_right - packed_width) * 0.5
    for row in range(len(METHODS)):
        for col in range(len(DATASETS)):
            pos = axes[row, col].get_position()
            axes[row, col].set_position(
                [packed_left + col * (image_width + packed_gap), pos.y0, image_width, pos.height]
            )

    png_path = out_dir / "qualitative_inputs_7datasets.png"
    pdf_path = out_dir / "qualitative_inputs_7datasets.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images, metadata = load_inputs(args)
    save_assets(images, out_dir)
    png_path, pdf_path = plot(images, out_dir)
    metadata_path = out_dir / "qualitative_inputs_7datasets_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "figure_rows": METHODS,
                "figure_columns": [item[0] for item in DATASETS],
                "target_cls": args.target_cls,
                "patch_size": args.patch_size,
                "model_input_size": [224, 224],
                "samples": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
