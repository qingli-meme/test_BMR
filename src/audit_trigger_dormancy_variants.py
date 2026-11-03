#!/usr/bin/env python3
"""Compare arbitrary trigger variants on normal receivers.

This reuses the repository's clean-model and clean-merge builders so the only
changed factor is the trigger itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from audit_trigger_dormancy_cifar100 import (
    DISPLAY,
    build_adamerging_state,
    build_regmean_state,
    build_vector_merge,
    eval_encoder,
    evaluate_state,
    load_clean_task_vectors,
    load_trigger,
)


def split_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--target-task", default="CIFAR100")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--trigger-sources", required=True)
    p.add_argument("--trigger-labels", required=True)
    p.add_argument("--methods", default="ta,ties,regmean,adamerging")
    p.add_argument("--scaling-coef", type=float, default=0.3)
    p.add_argument("--ties-reset-thresh", type=float, default=20.0)
    p.add_argument("--ties-merge-func", default="dis-sum")
    p.add_argument("--regmean-num-train-batch", type=int, default=8)
    p.add_argument("--adamerging-path", default="")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--out-dir", default="./analysis/bmr_controls/trigger_dormancy")
    args = p.parse_args()
    args.trigger_sources = split_csv(args.trigger_sources)
    args.trigger_labels = split_csv(args.trigger_labels)
    args.methods = split_csv(args.methods)
    if len(args.trigger_sources) != len(args.trigger_labels):
        p.error("trigger-sources and trigger-labels must have equal length")
    bad = set(args.methods) - set(DISPLAY)
    if bad:
        p.error(f"unsupported methods in this repository: {sorted(bad)}")
    return args


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    triggers = []
    for source, label in zip(args.trigger_sources, args.trigger_labels):
        triggers.append((label, source, load_trigger(args, source)))

    root = Path(args.ckpt_dir) / args.model
    pretrained = root / "zeroshot.pt"
    clean_task = root / args.target_task / "finetuned.pt"

    premerge = []
    for label, source, trig in triggers:
        for receiver, ckpt in [("Pretrained", pretrained), ("Clean Task Model", clean_task)]:
            result = eval_encoder(args, ckpt, trig)
            row = {
                "trigger": label,
                "trigger_source": source,
                "receiver": receiver,
                "trigger_path": trig["path"],
                **result,
            }
            premerge.append(row)
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

    merged = []
    for label, source, trig in triggers:
        for method in args.methods:
            result = evaluate_state(args, ptm_path, states[method], args.target_task, trig)
            row = {
                "trigger": label,
                "trigger_source": source,
                "method": method,
                "method_label": DISPLAY[method],
                "trigger_path": trig["path"],
                **result,
            }
            merged.append(row)
            print(json.dumps(row, indent=2))

    ts = time.strftime("%Y%m%d_%H%M%S")
    write_csv(out / f"{ts}_premerge.csv", premerge)
    write_csv(out / f"{ts}_clean_merges.csv", merged)
    (out / f"{ts}_trigger_dormancy_variants.json").write_text(
        json.dumps({"config": vars(args), "premerge": premerge, "clean_merges": merged}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
