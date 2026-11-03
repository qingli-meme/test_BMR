from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.reviewer_exp.common import ADVERSARY_TASKS, RESULT_ROOT


def read_results(prefix):
    out = []
    for path in sorted((RESULT_ROOT / "runs").glob(f"{prefix}*/result.json")):
        obj = json.loads(path.read_text())
        if not obj.get("smoke"): out.append(obj)
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def new_merges():
    rows = read_results("regmean_plusplus__") + read_results("localize_stitch__")
    attacked = [r for r in rows if r["attack"] != "clean"]
    write_csv(RESULT_ROOT / "new_merges_summary.csv", attacked,
              ["merge_method", "attack", "adversary_task", "suite", "target_class", "avg_clean_acc", "asr", "checkpoint_path"])
    per = [{"merge_method": r["merge_method"], "attack": r["attack"], "adversary_task": r["adversary_task"],
            "eval_task": t, "acc": a} for r in rows for t, a in r["per_task_acc"].items()]
    write_csv(RESULT_ROOT / "new_merges_per_task_acc.csv", per, ["merge_method", "attack", "adversary_task", "eval_task", "acc"])
    manifest = {"runs": [{k: r[k] for k in ("merge_method", "attack", "adversary_task", "checkpoint_path", "config", "input_sha256")} for r in rows]}
    (RESULT_ROOT / "new_merges_manifest.json").write_text(json.dumps(manifest, indent=2))


def defenses():
    ibvs = read_results("ibvs__")
    write_csv(RESULT_ROOT / "ibvs_cifar100.csv", ibvs, ["attack", "base_merge", "defense", "lambda_ibvs", "avg_clean_acc", "asr"])
    per = [{"attack": r["attack"], "base_merge": r["base_merge"], "defense": r["defense"], "lambda_ibvs": r["lambda_ibvs"],
            "eval_task": t, "acc": a} for r in ibvs for t, a in r["per_task_acc"].items()]
    write_csv(RESULT_ROOT / "ibvs_cifar100_per_task_acc.csv", per, list(per[0]) if per else ["attack"])
    dam = read_results("dam_layerwise__")
    write_csv(RESULT_ROOT / "dam_cifar100.csv", dam, ["attack", "defense", "avg_clean_acc", "asr"])
    per = [{"attack": r["attack"], "defense": r["defense"], "eval_task": t, "acc": a} for r in dam for t, a in r["per_task_acc"].items()]
    write_csv(RESULT_ROOT / "dam_cifar100_per_task_acc.csv", per, list(per[0]) if per else ["attack"])


def markdown_summary():
    lines = ["# Reviewer Experiment Summary", "", "## 1. New Merging Algorithms", ""]
    rows = read_results("regmean_plusplus__") + read_results("localize_stitch__")
    for method, title in (("regmean_plusplus", "RegMean++"), ("localize_stitch", "Localize-and-Stitch")):
        lines += [f"### {title}", "", "| Adversary Task | BadMerging ACC | BadMerging ASR | BMR ACC | BMR ASR |", "|---|---:|---:|---:|---:|"]
        for task in ADVERSARY_TASKS:
            rs = {r["attack"]: r for r in rows if r["merge_method"] == method and r["adversary_task"] == task}
            if {"badmerging", "bmr"} <= set(rs):
                lines.append(f"| {task} | {rs['badmerging']['avg_clean_acc']:.2f} | {rs['badmerging']['asr']:.2f} | {rs['bmr']['avg_clean_acc']:.2f} | {rs['bmr']['asr']:.2f} |")
        lines.append("")
    lines += ["## 2. Defense on CIFAR100", "", "| Attack | Defense | Base Merge | ACC | ASR |", "|---|---|---|---:|---:|"]
    for r in read_results("ibvs__") + read_results("dam_layerwise__"):
        lines.append(f"| {r['attack']} | {r['defense']} | {r.get('base_merge', '—')} | {r['avg_clean_acc']:.2f} | {r['asr']:.2f} |")
    lines += ["", "## 3. Per-task Utility", ""]
    utility = RESULT_ROOT / "per_task_utility_cifar100.csv"
    if utility.is_file(): lines.append(f"Measured long-format data: `{utility.name}`.")
    lines += ["", "## 4. Synthetic Background Fidelity", ""]
    fidelity = RESULT_ROOT / "background_fidelity_summary.csv"
    if fidelity.is_file(): lines.append(f"Measured deficit summaries: `{fidelity.name}`. Layerwise diagnostics are in `background_fidelity_layerwise.csv`.")
    (RESULT_ROOT / "REVIEWER_EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--section", choices=["new_merges", "defenses", "all"], default="all"); args = p.parse_args()
    if args.section in {"new_merges", "all"}: new_merges()
    if args.section in {"defenses", "all"}: defenses()
    markdown_summary()


if __name__ == "__main__": main()
