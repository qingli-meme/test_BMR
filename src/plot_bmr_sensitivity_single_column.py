#!/usr/bin/env python3
"""Double-column bar figure for poison-ratio and background-strength studies."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METHODS = [
    "Task Arithmetic", "TIES", "RegMean",
    "AdaMerging", "EMR", "RegMean++",
]
CSV_METHODS = {
    "TA": "Task Arithmetic",
    "TIES": "TIES",
    "RegMean": "RegMean",
    "AdaMerging": "AdaMerging",
    "EMR": "EMR",
    "EMR-Merging": "EMR",
    "RegMean++": "RegMean++",
}
WIDE_PREFIXES = {
    "Task Arithmetic": "ta",
    "TIES": "ties",
    "RegMean": "regmean",
    "AdaMerging": "adamerging",
    "EMR": "emr",
    "RegMean++": "regmeanpp",
}

ASR_COLOR = "#E87500"
ACC_COLOR = "#4C82B5"
EDGE_COLOR = "#202020"
GRID_COLOR = "#B8B8B8"


def load_study(path, x_key):
    data = {method: [] for method in METHODS}
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "merge" in fields:
            for row in reader:
                method = CSV_METHODS.get(row["merge"])
                if method is None:
                    continue
                data[method].append(
                    (float(row[x_key]), float(row["asr"]), float(row["avg_clean_acc"]))
                )
        else:
            for row in reader:
                x_value = float(row[x_key])
                for method, prefix in WIDE_PREFIXES.items():
                    data[method].append(
                        (x_value, float(row[f"{prefix}_asr"]), float(row[f"{prefix}_acc"]))
                    )
    for method in METHODS:
        data[method].sort(key=lambda item: item[0])
        if not data[method]:
            raise ValueError(f"No rows for {method} in {path}")
    return data


def draw_bar_panel(ax, xlabels, asr_values, acc_values, title):
    x = np.arange(len(xlabels))
    width = 0.29
    ax.bar(
        x - width / 2, asr_values, width=width,
        color=ASR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.65, zorder=3,
    )
    ax.bar(
        x + width / 2, acc_values, width=width,
        color=ACC_COLOR, edgecolor=EDGE_COLOR, linewidth=0.65, zorder=3,
    )
    ax.set_title(title, pad=5.5, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(70, 102)
    ax.set_yticks([70, 80, 90, 100])
    ax.grid(
        axis="y", linestyle="--", linewidth=0.65,
        color=GRID_COLOR, alpha=0.50, zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", direction="out", width=0.8, length=3.2, pad=2.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poison-csv",
        default=("bmr_paper_results/reviewer_response/sensitivity_six_merges/"
                 "poison_ratio_six_merges.csv"),
    )
    parser.add_argument(
        "--background-csv",
        default=("bmr_paper_results/reviewer_response/sensitivity_six_merges/"
                 "background_strength_six_merges.csv"),
    )
    parser.add_argument("--out-dir", default="bmr_paper_results")
    args = parser.parse_args()

    poison = load_study(args.poison_csv, "poison_ratio")
    gamma = load_study(args.background_csv, "gamma")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

    # Full double-column width with a deliberately shallow aspect ratio.
    fig = plt.figure(figsize=(7.25, 3.56))
    gs = fig.add_gridspec(
        nrows=2, ncols=7,
        width_ratios=[1.0, 1.0, 1.0, 0.34, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.060, right=0.995, bottom=0.240, top=0.805,
        wspace=0.22, hspace=0.48,
    )

    poison_axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
    ]
    gamma_axes = [
        fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 5]), fig.add_subplot(gs[0, 6]),
        fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5]), fig.add_subplot(gs[1, 6]),
    ]

    ratio_x = [item[0] for item in poison[METHODS[0]]]
    ratio_labels = [f"{value:g}%" for value in ratio_x]
    gamma_x = [item[0] for item in gamma[METHODS[0]]]
    gamma_labels = [f"{value:g}" for value in gamma_x]

    for i, (ax, method) in enumerate(zip(poison_axes, METHODS)):
        values = poison[method]
        draw_bar_panel(
            ax, ratio_labels,
            [item[1] for item in values], [item[2] for item in values],
            method,
        )
        # All panels share the same y scale.  Repeating labels in the second
        # and third columns makes the compact 2x3 layout collide horizontally.
        if i % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)

    for i, (ax, method) in enumerate(zip(gamma_axes, METHODS)):
        values = gamma[method]
        draw_bar_panel(
            ax, gamma_labels,
            [item[1] for item in values], [item[2] for item in values],
            method,
        )
        if i % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)

    # One shared y label per study group keeps the center gutter clear.
    fig.text(
        0.014, 0.515, "Score (%)", rotation=90,
        ha="center", va="center", fontsize=9.5,
    )
    fig.text(
        0.527, 0.515, "Score (%)", rotation=90,
        ha="center", va="center", fontsize=9.5,
    )

    fig.text(0.269, 0.160, "Poisoning ratio", ha="center", va="center", fontsize=10)
    fig.text(
        0.783, 0.160, r"Background strength $\gamma$",
        ha="center", va="center", fontsize=10,
    )

    legend_handles = [
        Patch(facecolor=ASR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.7, label="ASR"),
        Patch(facecolor=ACC_COLOR, edgecolor=EDGE_COLOR, linewidth=0.7, label="ACC"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=2, frameon=False, handlelength=1.8, handleheight=0.9,
        columnspacing=2.0, handletextpad=0.55, borderpad=0.45,
    )

    separator = Line2D(
        [0.512, 0.512], [0.225, 0.790], transform=fig.transFigure,
        color="#A0A0A0", linewidth=0.65, linestyle=(0, (4, 3)), alpha=0.65,
    )
    fig.add_artist(separator)

    fig.text(
        0.269, 0.100, "(a) Effect of poisoning ratio",
        ha="center", va="center", fontsize=10.5, fontweight="bold",
    )
    fig.text(
        0.783, 0.100, r"(b) Sensitivity to background strength $\gamma$",
        ha="center", va="center", fontsize=10.5, fontweight="bold",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "poison_ratio_background_strength"
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    for suffix in (".pdf", ".png", ".svg"):
        print(stem.with_suffix(suffix))


if __name__ == "__main__":
    main()
