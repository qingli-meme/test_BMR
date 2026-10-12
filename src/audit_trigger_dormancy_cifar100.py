#!/usr/bin/env python3
"""Audit CIFAR100 trigger dormancy before merge and on clean merged models."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from audit_clean_bmr_trigger_dormancy import (  # noqa: E402
    DATASETS,
    DISPLAY,
    build_adamerging_state,
    build_regmean_state,
    build_vector_merge,
    eval_args,
    evaluate_state,
    load_clean_task_vectors,
)
from eval import eval_single_dataset  # noqa: E402
from utils import corner_mask_generation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--methods", default="ta,ties,regmean,adamerging")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument("--adamerging-path", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--out-dir", default="./analysis/bmr_multitask/trigger_dormancy_cifar100")
    args = parser.parse_args()
    args.methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    bad = set(args.methods) - set(DISPLAY)
    if bad:
        parser.error(f"unsupported methods: {sorted(bad)}")
    return args


def load_trigger(args, trigger_source):
    path = Path("./trigger") / args.model / (
        f"{trigger_source}_{args.target_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path))
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return {
        "path": str(path),
        "backdoor_info": {
            "mask": torch.from_numpy(mask).float(),
            "applied_patch": torch.from_numpy(applied_patch).float(),
            "target_cls": args.target_cls,
        },
    }


def eval_encoder(args, checkpoint, trigger):
    encoder = torch.load(checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    encoder.eval()
    metrics = eval_single_dataset(
        encoder,
        args.target_task,
        eval_args(args),
        backdoor_info=trigger["backdoor_info"],
    )
    return {
        "patched_top1": 100.0 * metrics["top1"],
        "trigger_target_rate": 100.0 * metrics["backdoored_acc"],
        "target_hits": int(metrics["backdoored_cnt"]),
        "non_target_cnt": int(metrics["non_target_cnt"]),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triggers = [
        ("BadMerging", "On", load_trigger(args, "On")),
        ("BMR", "KDR_DTK", load_trigger(args, "KDR_DTK")),
    ]

    root = Path(args.ckpt_dir) / args.model
    pretrained_path = root / "zeroshot.pt"
    clean_task_path = root / args.target_task / "finetuned.pt"
    premerge_rows = []
    for trigger_label, trigger_source, trigger in triggers:
        for receiver_label, checkpoint in [
            ("Pretrained", pretrained_path),
            ("Clean Task Model", clean_task_path),
        ]:
            print(f"\n========== {receiver_label} / {trigger_label} trigger ==========")
            result = eval_encoder(args, checkpoint, trigger)
            row = {
                "trigger": trigger_label,
                "trigger_source": trigger_source,
                "receiver": receiver_label,
                "checkpoint": str(checkpoint),
                "trigger_path": trigger["path"],
                **result,
            }
            premerge_rows.append(row)
            print(json.dumps(row, indent=2))

    ptm_path, ptm_state, flat_ptm, tvs, ft_states = load_clean_task_vectors(args)
    states = {}
    if "ta" in args.methods:
        states["ta"] = build_vector_merge(args, "ta", ptm_state, flat_ptm, tvs)
    if "ties" in args.methods:
        states["ties"] = build_vector_merge(args, "ties", ptm_state, flat_ptm, tvs)
    if "regmean" in args.methods:
        states["regmean"] = build_regmean_state(args)
    if "adamerging" in args.methods:
        states["adamerging"] = build_adamerging_state(args, ptm_state, ft_states)

    clean_merged_rows = []
    for trigger_label, trigger_source, trigger in triggers:
        for method in args.methods:
            print(f"\n========== clean {DISPLAY[method]} / {trigger_label} trigger ==========")
            result = evaluate_state(args, ptm_path, states[method], args.target_task, trigger)
            row = {
                "trigger": trigger_label,
                "trigger_source": trigger_source,
                "method": method,
                "method_label": DISPLAY[method],
                "trigger_path": trigger["path"],
                **result,
            }
            clean_merged_rows.append(row)
            print(json.dumps(row, indent=2))

    ts = time.strftime("%Y%m%d_%H%M%S")
    premerge_csv = out_dir / f"{ts}_premerge_trigger_dormancy_cifar100.csv"
    clean_csv = out_dir / f"{ts}_clean_merged_trigger_dormancy_cifar100.csv"
    json_path = out_dir / f"{ts}_trigger_dormancy_cifar100.json"
    write_csv(premerge_csv, premerge_rows)
    write_csv(clean_csv, clean_merged_rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "premerge_rows": premerge_rows,
                "clean_merged_rows": clean_merged_rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {premerge_csv}")
    print(f"Saved {clean_csv}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
