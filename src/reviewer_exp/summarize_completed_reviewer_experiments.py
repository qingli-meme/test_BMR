#!/usr/bin/env python3
"""Create one Markdown summary of the finalized reviewer experiments."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "bmr_paper_results" / "reviewer_response"
OUT = RR / "COMPLETED_REVIEWER_EXPERIMENTS.md"
TASKS = ["CIFAR100", "ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value):
    return "—" if value is None else f"{float(value):.2f}"


def add_table(lines, header, rows):
    lines.extend([
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def result_path(method, attack, task):
    if method == "EMR":
        return ROOT / "analysis" / "emr" / "runs" / f"{attack}__{task}" / "result.json"
    return RR / "runs" / f"regmean_plusplus__{attack}__{task}" / "result.json"


def main():
    lines = [
        "# Completed Reviewer Experiments",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "All values below are read from the saved result files, not copied from console output.",
        "",
        "## 1. Seven-adversary main results",
        "",
        "### BMR",
        "",
    ]
    rows = []
    for task in TASKS:
        values = []
        for method in ("EMR", "RegMean++"):
            data = load(result_path(method, "bmr", task))
            values.extend([fmt(data.get("avg_clean_acc")), fmt(data.get("asr"))])
        rows.append([task, *values])
    add_table(lines, ["Adversary", "EMR ACC", "EMR ASR", "RegMean++ ACC", "RegMean++ ASR"], rows)

    lines.extend(["### BadMerging", ""])
    rows = []
    missing = []
    for task in TASKS:
        values = []
        for method in ("EMR", "RegMean++"):
            path = result_path(method, "badmerging", task)
            if path.is_file():
                data = load(path)
                values.extend([fmt(data.get("avg_clean_acc")), fmt(data.get("asr"))])
            else:
                values.extend(["MISSING", "MISSING"])
                missing.append(f"{method}/{task}")
        rows.append([task, *values])
    add_table(lines, ["Adversary", "EMR ACC", "EMR ASR", "RegMean++ ACC", "RegMean++ ASR"], rows)
    if missing:
        lines.extend(["Missing at generation time: " + ", ".join(missing), ""])

    lines.extend(["## 2. CIFAR100 ablations under the two new merge methods", ""])
    groups = [
        ("Training ablation", ["ordinary", "background_aug", "absolute_margin", "without_scale_variation"]),
        ("Trigger-construction ablation", ["without_target_alignment", "without_signal_constraint", "without_dormancy"]),
    ]
    for title, variants in groups:
        lines.extend([f"### {title}", ""])
        rows = []
        for variant in variants:
            values = []
            for method in ("emr", "regmean_plusplus"):
                data = load(RR / "ablation_new_merges" / f"{method}__{variant}" / "result.json")
                values.extend([fmt(data["asr"]), fmt(data["avg_clean_acc"])])
            rows.append([variant, *values])
        add_table(lines, ["Variant", "EMR ASR", "EMR ACC", "RegMean++ ASR", "RegMean++ ACC"], rows)

    lines.extend([
        "## 3. Representation-layer sensitivity on CIFAR100",
        "",
        "Layers: middle = `resblocks.5.ln_2`, late = `resblocks.9.ln_2`, final = `resblocks.11.ln_2` (paper checkpoint).",
        "",
    ])
    rep_files = {
        "Middle": sorted((ROOT / "results" / "bmr_representation_layer_sensitivity").glob("*REP_MID*.json"))[-1],
        "Late": sorted((ROOT / "results" / "bmr_representation_layer_sensitivity").glob("*REP_LATE*.json"))[-1],
    }
    rows = []
    for label, path in rep_files.items():
        data = load(path)["merge_results"]
        rows.append([label, fmt(data["ta"]["asr"]), fmt(data["ta"]["avg_clean_acc"]), fmt(data["ties"]["asr"]), fmt(data["ties"]["avg_clean_acc"]), str(path.relative_to(ROOT))])
    add_table(lines, ["Layer", "TA ASR", "TA ACC", "TIES ASR", "TIES ACC", "Structured result"], rows)

    lines.extend(["## 4. Motivation diagnostic extension", ""])
    with (ROOT / "bmr_paper_results" / "motivation" / "motivation_metrics.csv").open(newline="", encoding="utf-8") as handle:
        motivation = list(csv.DictReader(handle))
    rows = []
    for method in ("EMR", "RegMean++"):
        for row in motivation:
            if row["merge_method"] == method:
                rows.append([method, row["task"], f"{float(row['coupling_mean']):.6g}", fmt(row["impact_mean"]), fmt(row["impact_std"])])
    add_table(lines, ["Merge", "Task", "Coupling", "Impact mean", "Impact std"], rows)

    lines.extend(["## 5. Synthetic-background fidelity extension", ""])
    with (RR / "background_fidelity_coverage.csv").open(newline="", encoding="utf-8") as handle:
        coverage = {row["merge_method"]: row for row in csv.DictReader(handle)}
    with (RR / "background_fidelity_state_summary.csv").open(newline="", encoding="utf-8") as handle:
        states = {row["background_name"]: row for row in csv.DictReader(handle) if row["background_type"] == "real"}
    rows = []
    for method in ("EMR", "RegMean++"):
        rows.append([
            method,
            f"{float(states[method]['median_D']):.3f}",
            f"{100 * float(coverage[method]['coverage_q05_q95']):.1f}%",
            f"{100 * float(coverage[method]['below_syn_max']):.1f}%",
        ])
    add_table(lines, ["Merge", "Median deficit", "Q05–Q95 coverage", "Full-range coverage"], rows)

    lines.extend(["## 6. Real-merge gain and trigger dormancy", ""])
    diag = RR / "new_merge_mechanism_diagnostics"
    gain_path = diag / "emr_regmeanpp_real_merge_gain.csv"
    dormancy_path = diag / "emr_regmeanpp_trigger_dormancy.csv"
    if gain_path.is_file() and dormancy_path.is_file():
        with gain_path.open(newline="", encoding="utf-8") as handle:
            gain = list(csv.DictReader(handle))
        add_table(lines, ["Merge", "Variant", "Mean gain", "Median gain", "ASR"], [
            [r["merge_method"], r["variant"], fmt(r["g_mean"]), fmt(r["g_median"]), fmt(r["asr_percent"])] for r in gain
        ])
        with dormancy_path.open(newline="", encoding="utf-8") as handle:
            dormancy = list(csv.DictReader(handle))
        add_table(lines, ["Merge", "Trigger", "Median margin", "Target active"], [
            [r["merge_method"], r["trigger"], fmt(r["median_margin"]), f"{float(r['target_activation_percent']):.2f}%"] for r in dormancy
        ])
    else:
        lines.extend(["Not completed at generation time.", ""])

    lines.extend([
        "## 7. Reproducibility notes",
        "",
        "- BadMerging training now loads `checkpoints/ViT-B-32/zeroshot.pt` directly; no network download is required.",
        "- Formal runs use seed 2026 and the existing finalized triggers/checkpoints.",
        "- Detailed logs are stored in `bmr_paper_results/reviewer_response/logs/`.",
        "",
    ])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
