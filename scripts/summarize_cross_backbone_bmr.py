#!/usr/bin/env python3
import csv, json
from pathlib import Path

root = Path("bmr_paper_results/cross_backbone_openclip")
models = [("ViT-B/16", "vit_b16"), ("ViT-L/14", "vit_l14"), ("ConvNeXt-B", "convnext_b")]
rows = []
for label, key in models:
    path = root / key / "results" / "results.csv"
    if path.is_file():
        for row in csv.DictReader(path.open()):
            row["backbone_label"] = label; rows.append(row)

md = ["# OpenCLIP Cross-Backbone BMR Results", "",
      "CIFAR100 is the adversarial task; ACC is the arithmetic mean over CIFAR100, GTSRB, EuroSAT, Cars, SUN397, and PETS.", "",
      "| Backbone | Merger | Avg ACC (%) | ASR (%) |", "|---|---:|---:|---:|"]
for r in rows:
    md.append(f"| {r['backbone_label']} | {r['merge_method']} | {float(r['avg_clean_acc']):.2f} | {float(r['asr']):.2f} |")
if not rows: md.append("| _Pending_ | — | — | — |")
(root / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(root / "RESULTS.md")
