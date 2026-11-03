#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.classic_backdoors.common import ATTACK_SLUGS
from src.reviewer_exp.common import ROOT


OUT = ROOT / "bmr_paper_results/reviewer_response/classic_backdoors_new_merges"
ATTACKS = ("no_attack", *ATTACK_SLUGS)
DISPLAY = {"no_attack": "No Attack", "badnets": "BadNets",
           "label_consistent": "LabelConsistent", "trojannn": "TrojanNN",
           "dynamic": "Dynamic Backdoor"}


def main():
    rows = []
    for task in ("CIFAR100", "ImageNet100"):
        for attack in ATTACKS:
            row = {"adversary_task": task, "attack": DISPLAY[attack]}
            for method, key in (("emr", "emr_asr"), ("regmean_plusplus", "regmeanpp_asr")):
                path = OUT / "runs" / f"{method}__{attack}__{task}" / "result.json"
                data = json.loads(path.read_text())
                row[key] = data["asr"]
                row[f"{key}_source"] = str(path.resolve())
            rows.append(row)
    csv_path = OUT / "classic_backdoors_emr_regmeanpp.csv"
    fields = ["adversary_task", "attack", "emr_asr", "regmeanpp_asr", "emr_asr_source", "regmeanpp_asr_source"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Classic backdoor baselines under new merging methods", "",
             "ASR (%) on non-target test samples.", "",
             "| Adversary task | Attack | EMR | RegMean++ |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["adversary_task"]} | {row["attack"]} | {row["emr_asr"]:.2f} | {row["regmeanpp_asr"]:.2f} |')
    lines += ["", "Only EMR and RegMean++ were evaluated; the four previously reported mergers were not rerun."]
    md_path = OUT / "classic_backdoors_emr_regmeanpp.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(csv_path); print(md_path)


if __name__ == "__main__":
    main()
