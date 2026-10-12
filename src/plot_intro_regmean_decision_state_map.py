#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
TRIGGERS = [
    ("BadMerging-On", "#d77559", -0.16),
    ("BMR trigger", "#4f9a63", 0.16),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="bmr_paper_results/bmr_vs_badmerging_on_endpoint_summary.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="bmr_paper_results/intro_regmean_decision_state_map",
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
                    "margin0": float(row["margin_median_eta0"]),
                    "margin1": float(row["margin_median_eta1"]),
                    "shift": float(row["median_margin_shift"]),
                    "mean_abs_shift": float(row["mean_abs_margin_shift"]),
                    "target0": 100.0 * float(row["target_rate_eta0"]),
                    "target1": 100.0 * float(row["target_rate_eta1"]),
                }
            )
    return rows


def write_csv(rows, path):
    fields = [
        "task",
        "trigger",
        "abs_cos",
        "margin0",
        "margin1",
        "shift",
        "mean_abs_shift",
        "target0",
        "target1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_prefix):
    lookup = {(r["task"], r["trigger"]): r for r in rows}
    ybase = np.arange(len(TASK_ORDER))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.text(
        0.08,
        len(TASK_ORDER) - 0.35,
        "target boundary",
        fontsize=8,
        color="#333333",
        ha="left",
        va="center",
    )

    for label, color, offset in TRIGGERS:
        for i, task in enumerate(TASK_ORDER):
            row = lookup[(task, label)]
            y = ybase[i] + offset
            x0 = row["margin0"]
            x1 = row["margin1"]
            ax.plot([x0, x1], [y, y], color=color, linewidth=2.0, alpha=0.75)
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.7,
                    "shrinkA": 6,
                    "shrinkB": 6,
                    "mutation_scale": 8,
                    "alpha": 0.9,
                },
            )
            ax.scatter(
                [x0],
                [y],
                s=54,
                facecolor="white",
                edgecolor=color,
                linewidth=1.7,
                zorder=3,
            )
            ax.scatter(
                [x1],
                [y],
                s=58,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=4,
            )

    for i, task in enumerate(TASK_ORDER):
        row = lookup[(task, "BadMerging-On")]
        ax.text(
            -13.8,
            ybase[i],
            f"|cos|={row['abs_cos']:.4f}",
            fontsize=7.5,
            color="#555555",
            va="center",
            ha="left",
        )

    ax.set_yticks(ybase)
    ax.set_yticklabels(TASK_ORDER)
    ax.set_xlim(-14.2, 8.2)
    ax.set_ylim(-0.75, len(TASK_ORDER) - 0.25)
    ax.set_xlabel("median target margin on triggered inputs")
    ax.set_title(
        "RegMean can move triggered decision states despite near-orthogonal task directions",
        fontsize=10,
    )
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = []
    for label, color, _ in TRIGGERS:
        handle = ax.scatter(
            [],
            [],
            s=58,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.5,
            label=label,
        )
        legend_handles.append(handle)
    before = ax.scatter(
        [],
        [],
        s=54,
        facecolor="white",
        edgecolor="#555555",
        linewidth=1.5,
        label="before background",
    )
    after = ax.scatter(
        [],
        [],
        s=58,
        facecolor="#555555",
        edgecolor="#222222",
        linewidth=0.5,
        label="after RegMean background",
    )
    ax.legend(
        handles=[before, after] + legend_handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.11, right=0.98)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input)
    compact_csv = Path(args.out_prefix).with_suffix(".csv")
    write_csv(rows, compact_csv)
    plot(rows, args.out_prefix)
    print(Path(args.out_prefix).with_suffix(".png"))
    print(Path(args.out_prefix).with_suffix(".pdf"))
    print(compact_csv)


if __name__ == "__main__":
    main()
