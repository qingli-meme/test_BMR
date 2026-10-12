#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
TRIGGER_ORDER = ["BadMerging-On", "BMR trigger"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="bmr_paper_results/bmr_vs_badmerging_on_endpoint_summary.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="bmr_paper_results/intro_regmean_orthogonality_margin_gap",
    )
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "regmean":
                continue
            rows.append(
                {
                    "task": row["adversary_task"],
                    "trigger": row["trigger_label"],
                    "abs_cos": float(row["abs_cosine"]),
                    "median_shift": float(row["median_margin_shift"]),
                    "mean_abs_shift": float(row["mean_abs_margin_shift"]),
                    "target_rate_0": 100.0 * float(row["target_rate_eta0"]),
                    "target_rate_1": 100.0 * float(row["target_rate_eta1"]),
                }
            )
    return rows


def write_compact_csv(rows, path):
    fields = [
        "task",
        "trigger",
        "abs_cos",
        "median_shift",
        "mean_abs_shift",
        "target_rate_0",
        "target_rate_1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_prefix):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 8,
        }
    )
    by_task = {task: [] for task in TASK_ORDER}
    for row in rows:
        by_task[row["task"]].append(row)

    cosines = []
    shifts = {label: [] for label in TRIGGER_ORDER}
    for task in TASK_ORDER:
        task_rows = by_task[task]
        cosines.append(1000.0 * task_rows[0]["abs_cos"])
        for label in TRIGGER_ORDER:
            match = next(row for row in task_rows if row["trigger"] == label)
            shifts[label].append(match["mean_abs_shift"])

    x = np.arange(len(TASK_ORDER))
    width = 0.34

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    ax = axes[0]
    ax.bar(
        x,
        cosines,
        color="#8aa6c8",
        edgecolor="#35506b",
        linewidth=0.8,
    )
    ax.set_ylim(0.0, 1.55)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_ORDER, rotation=35, ha="right")
    ax.set_ylabel(r"abs cosine ($\times 10^{-3}$)")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.8)
    ax.text(
        0.5,
        -0.38,
        "(a)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    ax = axes[1]
    ax.bar(
        x - width / 2,
        shifts["BadMerging-On"],
        width,
        color="#d8795c",
        edgecolor="#6f2c1d",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        shifts["BMR trigger"],
        width,
        color="#5aa06a",
        edgecolor="#24522e",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_ORDER, rotation=35, ha="right")
    ax.set_ylabel("mean abs margin shift")
    ax.set_ylim(0.0, 3.45)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.text(
        0.5,
        -0.38,
        "(b)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.34, wspace=0.18)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input)
    compact_csv = Path(args.out_prefix).with_suffix(".csv")
    write_compact_csv(rows, compact_csv)
    plot(rows, args.out_prefix)
    print(Path(args.out_prefix).with_suffix(".png"))
    print(Path(args.out_prefix).with_suffix(".pdf"))
    print(compact_csv)


if __name__ == "__main__":
    main()
