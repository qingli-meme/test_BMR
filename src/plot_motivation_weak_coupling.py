#!/usr/bin/env python3
"""Plot the six-method weak-coupling versus decision-impact figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


DATASETS = ["CIFAR100", "ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "Pets"]
METHODS = ["Task Arithmetic", "TIES", "RegMean", "AdaMerging", "EMR", "RegMean++"]

BAR_FILL = "#DCEBF7"
HATCH_BLUE = "#6E9BC8"
EDGE = "#171717"
GRID = "#DEE4EB"
SHADOW = "#C7D3E0"
HATCH_IDX = {1, 3, 5}
BAR_WIDTH = 0.62


def load_metrics(path):
    required = {
        "merge_method", "task", "coupling_mean", "coupling_std",
        "impact_mean", "impact_std",
    }
    data = {method: {} for method in METHODS}
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"missing columns: {sorted(required - fields)}")
        for row in reader:
            method, task = row["merge_method"], row["task"]
            if method not in data or task not in DATASETS:
                continue
            data[method][task] = {
                key: float(row[key])
                for key in ("coupling_mean", "coupling_std", "impact_mean", "impact_std")
            }
    for method in METHODS:
        missing = set(DATASETS) - set(data[method])
        if missing:
            raise ValueError(f"incomplete CSV for {method}: {sorted(missing)}")
    return data


def add_soft_shadow(container):
    for patch in container.patches:
        patch.set_path_effects([
            pe.SimplePatchShadow(
                offset=(0.45, -0.45),
                shadow_rgbFace=SHADOW,
                alpha=0.12,
                rho=0.99,
            ),
            pe.Normal(),
        ])


def draw_panel(ax, values, errors=None):
    x = np.arange(len(DATASETS))
    values = np.asarray(values, dtype=float)

    base = ax.bar(
        x, values,
        width=BAR_WIDTH,
        color=BAR_FILL,
        edgecolor=EDGE,
        linewidth=0.82,
        zorder=3,
    )
    add_soft_shadow(base)

    for i in HATCH_IDX:
        ax.bar(
            x[i], values[i], width=BAR_WIDTH,
            color="none", edgecolor=HATCH_BLUE,
            linewidth=0.01, hatch="/", zorder=4,
        )
        outline = ax.bar(
            x[i], values[i], width=BAR_WIDTH,
            color="none", edgecolor=EDGE,
            linewidth=0.82, zorder=5,
        )
        add_soft_shadow(outline)

    if errors is not None:
        ax.errorbar(
            x, values,
            yerr=np.array(errors, dtype=float),
            fmt="none", ecolor=EDGE,
            elinewidth=0.78, capsize=1.65, capthick=0.78,
            zorder=6,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE)
        spine.set_linewidth(0.82)

    ax.set_facecolor("white")
    ax.grid(
        axis="y", linestyle=(0, (3, 3)), linewidth=0.42,
        color=GRID, alpha=0.86, zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", length=2.2, width=0.68, colors=EDGE, pad=1.5)
    ax.set_xlim(-0.52, len(DATASETS) - 0.48)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="./bmr_paper_results/motivation")
    args = parser.parse_args()

    data = load_metrics(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 6.3,
        "axes.linewidth": 0.82,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.30,
    })

    fig, axes = plt.subplots(
        2, 6, figsize=(7.25, 2.48), dpi=300, sharey="row",
    )
    x = np.arange(len(DATASETS))

    for col, method in enumerate(METHODS):
        coupling = [data[method][task]["coupling_mean"] for task in DATASETS]
        impact = [data[method][task]["impact_mean"] for task in DATASETS]
        impact_err = [data[method][task]["impact_std"] for task in DATASETS]

        ax_top = axes[0, col]
        draw_panel(ax_top, coupling)
        ax_top.set_title(method, pad=3.0, fontweight="semibold")
        ax_top.set_ylim(0, 0.125)
        ax_top.set_yticks([0.00, 0.05, 0.10])
        ax_top.set_xticks(x)
        # Dataset identities are repeated in every method column, so showing
        # them on the upper row only consumes vertical space without adding
        # information.
        ax_top.set_xticklabels([])
        ax_top.tick_params(axis="x", bottom=False)

        ax_bot = axes[1, col]
        draw_panel(ax_bot, impact, impact_err)
        ax_bot.set_ylim(0, 16.8)
        ax_bot.set_yticks([0, 5, 10, 15])
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(DATASETS, rotation=38, ha="right", rotation_mode="anchor")

    fig.subplots_adjust(
        left=0.072, right=0.997, top=0.895, bottom=0.225,
        wspace=0.205, hspace=0.285,
    )

    # Derive the row centers from the final axes geometry.  This avoids the
    # visible vertical drift caused by hand-tuned figure coordinates.
    top_box = axes[0, 0].get_position()
    bottom_box = axes[1, 0].get_position()
    top_center = 0.5 * (top_box.y0 + top_box.y1)
    bottom_center = 0.5 * (bottom_box.y0 + bottom_box.y1)
    fig.text(
        0.0210, top_center, "Parameter coupling",
        rotation=90, ha="center", va="center",
        fontsize=7.55, fontweight="semibold", color=EDGE,
    )
    fig.text(
        0.0210, bottom_center, "Decision impact",
        rotation=90, ha="center", va="center",
        fontsize=7.55, fontweight="semibold", color=EDGE,
    )

    png = out_dir / "motivation_weak_coupling_vs_decision_impact.png"
    pdf = out_dir / "motivation_weak_coupling_vs_decision_impact.pdf"
    svg = out_dir / "motivation_weak_coupling_vs_decision_impact.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(pdf)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
