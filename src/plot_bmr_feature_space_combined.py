#!/usr/bin/env python3
"""Two-panel BMR feature-space visualization."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PANELS = [
    ("clean", "Clean"), ("ta", "Task\nArithmetic"), ("ties", "TIES"),
    ("regmean", "RegMean"), ("adamerging", "AdaMerging"), ("emr", "EMR"),
    ("regmeanpp", "RegMean++"),
]
GROUPS = ("nontarget_clean", "target_clean", "triggered")
COLORS = {
    "target_clean": "#347FB3", "nontarget_clean": "#A7ADB4",
    "triggered": "#C43D35",
}
OUTLINE_COLOR = "#252525"
LABELS = {
    "target_clean": "Target clean", "nontarget_clean": "Non-target clean",
    "triggered": "Triggered",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cifar-csv",
        default="bmr_paper_results/bmr_feature_space_cifar100_points.csv",
    )
    parser.add_argument(
        "--imagenet-csv",
        default="bmr_paper_results/bmr_feature_space_imagenet100_points.csv",
    )
    parser.add_argument("--out-dir", default="bmr_paper_results")
    return parser.parse_args()


def load_points(path):
    points = {panel: {group: [] for group in GROUPS} for panel, _ in PANELS}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points[row["panel"]][row["group"]].append(
                (float(row["x"]), float(row["y"]))
            )
    for panel, _ in PANELS:
        for group in GROUPS:
            points[panel][group] = np.asarray(points[panel][group], dtype=np.float64)
            if points[panel][group].size == 0:
                raise ValueError(f"missing {panel}/{group} in {path}")
    return points


def row_limits(points):
    values = np.vstack([
        points[panel][group] for panel, _ in PANELS for group in GROUPS
    ])
    x_pad = max(1e-6, 0.05 * np.ptp(values[:, 0]))
    y_pad = max(1e-6, 0.05 * np.ptp(values[:, 1]))
    return (
        (values[:, 0].min() - x_pad, values[:, 0].max() + x_pad),
        (values[:, 1].min() - y_pad, values[:, 1].max() + y_pad),
    )


def draw_feature_panel(ax, points, panel, title, xlim, ylim, show_title):
    ax.set_box_aspect(1)
    xy = points[panel]["nontarget_clean"]
    ax.scatter(
        xy[:, 0], xy[:, 1], s=6.0, c=COLORS["nontarget_clean"], marker="o",
        linewidths=0, alpha=0.34, rasterized=True, zorder=2,
    )
    xy = points[panel]["target_clean"]
    ax.scatter(
        xy[:, 0], xy[:, 1], s=8.2, c=COLORS["target_clean"], marker="o",
        edgecolors=OUTLINE_COLOR, linewidths=0.30, alpha=0.82,
        rasterized=True, zorder=3,
    )
    xy = points[panel]["triggered"]
    ax.scatter(
        xy[:, 0], xy[:, 1], s=15.0, c="white", marker="x",
        linewidths=2.15, alpha=0.92, rasterized=True, zorder=5,
    )
    ax.scatter(
        xy[:, 0], xy[:, 1], s=11.0, c=COLORS["triggered"], marker="x",
        linewidths=1.05, alpha=0.96, rasterized=True, zorder=6,
    )
    ax.axvline(
        0.0, color=OUTLINE_COLOR, linestyle=(0, (3.2, 2.2)),
        linewidth=1.0, zorder=4,
    )
    ratio = float((points[panel]["triggered"][:, 0] > 0).mean() * 100.0)
    ax.text(
        0.045, 0.93, f"{ratio:.1f}%", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.3, fontweight="bold",
        color=OUTLINE_COLOR,
        bbox={
            "boxstyle": "round,pad=0.16", "facecolor": "white",
            "edgecolor": OUTLINE_COLOR, "linewidth": 0.62, "alpha": 0.96,
        },
        zorder=8,
    )
    if show_title:
        ax.set_title(
            title, fontsize=8.4, fontweight="semibold", pad=3.0,
            linespacing=0.86, color="#262D35",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Target margin", fontsize=7.6, labelpad=0.9)
    ax.tick_params(
        axis="both", labelsize=6.7, length=2.2, width=0.60, pad=1.0,
        color="#89949C", labelcolor="#262D35",
    )
    ax.grid(
        True, color="#D8DDE2", linestyle="--",
        linewidth=0.48, alpha=0.90, zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(OUTLINE_COLOR)
        ax.spines[spine].set_linewidth(0.78)


def main():
    args = parse_args()
    datasets = [
        ("CIFAR100", load_points(args.cifar_csv)),
        ("ImageNet100", load_points(args.imagenet_csv)),
    ]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8.0, "axes.linewidth": 0.72,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "#FCFCFD",
        "text.color": "#262D35", "axes.labelcolor": "#262D35",
        "axes.titlecolor": "#262D35",
    })

    fig, axes = plt.subplots(2, 7, figsize=(7.25, 3.35), facecolor="white")
    for row, (_, points) in enumerate(datasets):
        xlim, ylim = row_limits(points)
        for col, (panel, title) in enumerate(PANELS):
            ax = axes[row, col]
            draw_feature_panel(
                ax=ax, points=points, panel=panel, title=title,
                xlim=xlim, ylim=ylim, show_title=(row == 0),
            )
        axes[row, 0].set_ylabel("Feature PC 1", fontsize=8.0, labelpad=2.0)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.6,
               markerfacecolor=COLORS["nontarget_clean"], markeredgecolor=OUTLINE_COLOR,
               markeredgewidth=0.55, alpha=0.78, label=LABELS["nontarget_clean"]),
        Line2D([], [], linestyle="none", marker="o", markersize=4.6,
               markerfacecolor=COLORS["target_clean"], markeredgecolor=OUTLINE_COLOR,
               markeredgewidth=0.55, alpha=0.95, label=LABELS["target_clean"]),
        Line2D([], [], linestyle="none", marker="x", markersize=5.0,
               markeredgecolor=COLORS["triggered"], markeredgewidth=1.0,
               label=LABELS["triggered"]),
    ]
    legend = fig.legend(
        handles=handles, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 0.972), frameon=False, fontsize=9.1,
        handletextpad=0.45, columnspacing=1.55, borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.062, right=0.997, top=0.880, bottom=0.125,
        wspace=0.30, hspace=0.34,
    )
    fig.canvas.draw()
    top_bottom = min(ax.get_position().y0 for ax in axes[0])
    lower_top = max(ax.get_position().y1 for ax in axes[1])
    lower_bottom = min(ax.get_position().y0 for ax in axes[1])

    # Move the complete upper group downward as one rigid block: legend,
    # method titles, all seven CIFAR100 axes (including ticks/x labels), and
    # the (a) caption.  The shift halves the original distance from (a) to
    # the top edge of the lower row, while leaving the lower row unchanged.
    original_a_y = (top_bottom + lower_top) / 2.0 - 0.010
    top_group_shift = -0.5 * (original_a_y - lower_top)
    for ax in axes[0]:
        position = ax.get_position()
        ax.set_position([
            position.x0,
            position.y0 + top_group_shift,
            position.width,
            position.height,
        ])
    legend.set_bbox_to_anchor(
        (0.5, 0.972 + top_group_shift),
        transform=fig.transFigure,
    )
    a_y = original_a_y + top_group_shift
    fig.text(
        0.5, a_y, "(a) CIFAR100",
        ha="center", va="center", fontsize=9.4,
        fontweight="bold", color="#262D35",
    )
    fig.text(
        0.5, lower_bottom - 0.108, "(b) ImageNet100",
        ha="center", va="center", fontsize=9.4,
        fontweight="bold", color="#262D35",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "bmr_feature_space_cifar100_imagenet100"
    fig.savefig(
        stem.with_suffix(".pdf"), facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    fig.savefig(
        stem.with_suffix(".svg"), facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    plt.close(fig)
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
