from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.reviewer_exp.common import BASE6, RESULT_ROOT, ROOT, checkpoint_path, default_args, pool_checkpoint_paths, require_paths
from src.reviewer_exp.merge_existing import build_existing


DISPLAY = {"task_arithmetic": "Task Arithmetic", "ties": "TIES", "regmean": "RegMean", "adamerging": "AdaMerging",
           "regmean_plusplus": "RegMean++", "localize_stitch": "Localize-and-Stitch"}


def evaluate_per_task(image_encoder, tasks, args):
    from src.eval import eval_single_dataset
    rows = []
    image_encoder = image_encoder.to(args.device).eval()
    for task in tasks:
        metrics = eval_single_dataset(image_encoder, task, args)
        rows.append({"task": task, "acc": 100.0 * float(metrics["top1"])})
    return rows, float(np.mean([r["acc"] for r in rows]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged-checkpoint")
    p.add_argument("--suite", choices=["base6"], default="base6")
    p.add_argument("--attack")
    p.add_argument("--merge-method")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--include-original-merges", action="store_true")
    p.add_argument("--include-new-merges", action="store_true")
    p.add_argument("--states", nargs="+", default=["clean", "badmerging", "bmr"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-eval-batches", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    eval_args = default_args("cuda", 128)
    eval_args.max_eval_batches = args.max_eval_batches
    out = RESULT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    long_path = out / "per_task_utility_cifar100.csv"
    summary_path = out / "per_task_utility_cifar100_summary.csv"
    methods = []
    if args.include_original_merges:
        methods += ["task_arithmetic", "ties", "regmean", "adamerging"]
    if args.include_new_merges:
        methods += ["regmean_plusplus", "localize_stitch"]
    if args.merged_checkpoint:
        methods = [args.merge_method]
        states = [args.attack]
    else:
        states = args.states
    rows = []
    pretrained = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    for method in methods:
        for state in states:
            if args.merged_checkpoint:
                model = torch.load(args.merged_checkpoint, map_location="cpu", weights_only=False)
            elif method in {"regmean_plusplus", "localize_stitch"}:
                result = out / "runs" / f"{method}__{state}__CIFAR100" / "result.json"
                if not result.is_file():
                    print(f"[defer missing new merge] {result}")
                    continue
                import json
                model = torch.load(json.loads(result.read_text())["checkpoint_path"], map_location="cpu", weights_only=False)
            else:
                paths = pool_checkpoint_paths("CIFAR100", state)
                require_paths(paths)
                model = build_existing(method, pretrained, paths, BASE6, state, "CIFAR100")
            per_task, avg = evaluate_per_task(model, BASE6, eval_args)
            for rec in per_task:
                rows.append({"merge_method": DISPLAY[method], "model_state": {"clean": "CleanOnly", "badmerging": "BadMerging", "bmr": "BMR"}[state],
                             "adversary_task": "CIFAR100", "eval_task": rec["task"], "acc": rec["acc"]})
            del model
            torch.cuda.empty_cache()
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["merge_method", "model_state", "adversary_task", "eval_task", "acc"])
        w.writeheader(); w.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["merge_method"], row["model_state"]), []).append(float(row["acc"]))
    summary = []
    for method in dict.fromkeys(r["merge_method"] for r in rows):
        values = {s: np.mean(grouped[(method, s)]) if (method, s) in grouped else float("nan") for s in ("CleanOnly", "BadMerging", "BMR")}
        summary.append({"merge_method": method, "clean_only_avg": values["CleanOnly"], "badmerging_avg": values["BadMerging"],
                        "bmr_avg": values["BMR"], "badmerging_delta": values["BadMerging"] - values["CleanOnly"],
                        "bmr_delta": values["BMR"] - values["CleanOnly"]})
    with summary_path.open("w", newline="") as f:
        fields = ["merge_method", "clean_only_avg", "badmerging_avg", "bmr_avg", "badmerging_delta", "bmr_delta"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)


if __name__ == "__main__":
    main()
