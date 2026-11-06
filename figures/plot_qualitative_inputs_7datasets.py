#!/usr/bin/env python3
"""Plot qualitative_inputs_7datasets from the 21 exported image assets.

This is the figure-only version: it does not load datasets, checkpoints, or
triggers.  Edit ``plot_figure`` to adjust layout and visual styling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASETS = [
    ("CIFAR-100", "cifar100"),
    ("GTSRB", "gtsrb"),
    ("EuroSAT", "eurosat"),
    ("Cars", "cars"),
    ("SUN397", "sun397"),
    ("Pets", "pets"),
    ("ImageNet-100", "imagenet100"),
]
METHODS = [
    ("Clean", "clean"),
    ("BadMerging", "badmerging"),
    ("BMR", "bmr"),
]
METHOD_COLORS = {"Clean": "#111111", "BadMerging": "#111111", "BMR": "#111111"}


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=here / "qualitative_inputs_7datasets_assets",
    )
    parser.add_argument("--out-dir", type=Path, default=here)
    parser.add_argument("--stem", default="qualitative_inputs_7datasets_replot")
    return parser.parse_args()


def load_images(assets_dir):
    images = {}
    for method, method_slug in METHODS:
        images[method] = []
        for _display_name, dataset_slug in DATASETS:
            path = assets_dir / f"{dataset_slug}_{method_slug}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            images[method].append(np.asarray(Image.open(path).convert("RGB")))
    return images


def plot_figure(images, out_dir, stem):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(3, 7, figsize=(3.35, 1.70), facecolor="white")
    fig.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.89,
        bottom=0.025,
        wspace=0.03,
        hspace=0.12,
    )

    for col, (display_name, _slug) in enumerate(DATASETS):
        title = {"CIFAR-100": "CIFAR", "ImageNet-100": "ImageNet"}.get(display_name, display_name)
        axes[0, col].set_title(
            title,
            fontsize=7.2,
            fontweight="bold",
            pad=2.2,
        )

    for row, (method, _method_slug) in enumerate(METHODS):
        for col, image in enumerate(images[method]):
            ax = axes[row, col]
            interpolation = "nearest" if col == 0 else "antialiased"
            ax.imshow(image, interpolation=interpolation, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.55,
                    clip_on=False,
                )
            )
    # imshow preserves square images inside wider subplot slots.  Repack the
    # visible square axes so that the inter-image gaps are half as large.
    fig.canvas.draw()
    reference = [axes[0, col].get_position() for col in range(len(DATASETS))]
    # Use an explicit 3 pt horizontal gutter at native single-column size.
    packed_gap = (3.0 / 72.0) / fig.get_figwidth()
    old_left = reference[0].x0
    old_right = reference[-1].x1
    available_width = old_right - old_left
    image_width = (available_width - (len(reference) - 1) * packed_gap) / len(reference)
    packed_width = len(reference) * image_width + (len(reference) - 1) * packed_gap
    packed_left = (old_left + old_right - packed_width) * 0.5
    for row in range(len(METHODS)):
        for col in range(len(DATASETS)):
            position = axes[row, col].get_position()
            axes[row, col].set_position([
                packed_left + col * (image_width + packed_gap),
                position.y0,
                image_width,
                position.height,
            ])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


def main():
    args = parse_args()
    plot_figure(load_images(args.assets_dir), args.out_dir, args.stem)


if __name__ == "__main__":
    main()
