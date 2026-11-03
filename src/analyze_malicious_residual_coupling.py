#!/usr/bin/env python3
"""Measure global cosine coupling using the ACTUAL malicious checkpoints.

This is a diagnostic for the paper's motivation figure. It reuses the same
benign-only backgrounds as analyze_orthogonality_margin_path.py, but replaces
the clean adversary residual with residuals from Ordinary, Background Aug., and
Full BMR checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from analyze_orthogonality_margin_path import (
    cache_path,
    parse_list,
    prepare_backgrounds,
    state_dict_from_checkpoint,
    update_cosine,
)


def attack_checkpoint(args, attack_type):
    suffix = f"{attack_type}_{args.attack_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    return Path(args.ckpt_dir) / args.model / f"{args.attack_task}_{suffix}" / "finetuned.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--attack-task", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--benign-datasets", default="GTSRB,EuroSAT,Cars,SUN397,PETS")
    p.add_argument("--merge-methods", default="ta,ties,regmean")
    p.add_argument("--attack-types", default="KDR_DTK_BMR_ORD,KDR_DTK_BMR_BG,KDR_DTK_BMR")
    p.add_argument("--attack-labels", default="Ordinary,BackgroundAug,FullBMR")
    p.add_argument("--scaling-coef", type=float, default=0.3)
    p.add_argument("--ties-reset-thresh", type=float, default=20.0)
    p.add_argument("--ties-merge-func", default="dis-sum")
    p.add_argument("--regmean-num-train-batch", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rebuild-backgrounds", action="store_true")
    p.add_argument("--out-dir", default="./analysis/bmr_controls/malicious_coupling")
    args = p.parse_args()
    args.benign_datasets = parse_list(args.benign_datasets)
    args.merge_methods = parse_list(args.merge_methods)
    args.attack_types = parse_list(args.attack_types)
    args.attack_labels = parse_list(args.attack_labels)
    if len(args.attack_types) != len(args.attack_labels):
        p.error("attack-types and attack-labels must have equal length")
    if set(args.merge_methods) - {"ta", "ties", "regmean"}:
        p.error("remote repository implementation supports ta,ties,regmean here")
    return args


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.ckpt_dir) / args.model
    ptm_path = root / "zeroshot.pt"
    clean_path = root / args.attack_task / "finetuned.pt"
    ptm = state_dict_from_checkpoint(ptm_path)
    clean = state_dict_from_checkpoint(clean_path)

    # prepare_backgrounds expects this attribute but does not use target task.
    args.target_task = args.attack_task
    prepare_backgrounds(args, ptm)

    variants = [("CleanTask", "CleanTask", clean_path, clean)]
    for attack_type, label in zip(args.attack_types, args.attack_labels):
        path = attack_checkpoint(args, attack_type)
        if not path.exists():
            raise FileNotFoundError(path)
        variants.append((attack_type, label, path, state_dict_from_checkpoint(path)))

    rows = []
    for method in args.merge_methods:
        bg_path = cache_path(args, method)
        bg = torch.load(bg_path, map_location="cpu", weights_only=False)
        for attack_type, label, path, state in variants:
            geom = update_cosine(ptm, bg, state)
            row = {
                "merge_method": method,
                "variant": label,
                "attack_type": attack_type,
                "checkpoint": str(path),
                **geom,
            }
            rows.append(row)
            print(json.dumps(row, indent=2))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out / f"{ts}_malicious_residual_coupling.csv"
    json_path = out / f"{ts}_malicious_residual_coupling.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
