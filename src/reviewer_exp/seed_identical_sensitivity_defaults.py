#!/usr/bin/env python3
"""Reuse byte-identical completed main results for two duplicate sensitivity rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.reviewer_exp.common import ROOT, checkpoint_path, dump_json


def main():
    out = ROOT / "bmr_paper_results/reviewer_response/sensitivity_six_merges"
    trigger = ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
    malicious = checkpoint_path("CIFAR100", "bmr")
    tasks = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
    paths = [malicious, *[checkpoint_path(task, "clean") for task in tasks[1:]]]
    sources = [
        (
            "EMR",
            ROOT / "analysis/emr/runs/bmr__CIFAR100/result.json",
            ROOT / "analysis/emr/cache/base6_bmr_emr.pt",
            {"implementation": "src/reviewer_exp/merge_emr.py", "receiver_policy": "task-specific mask and rescaler"},
        ),
        (
            "RegMean++",
            ROOT / "bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/result.json",
            ROOT / "bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/merged_encoder.pt",
            {"implementation": "src/reviewer_exp/merge_regmean_plusplus.py", "num_regmean_examples": 256,
             "regmean_batch_size": 32, "reduce_non_diagonal_ratio": 0.95, "activation_seed": 2026},
        ),
    ]
    smoke = json.loads((out / "smoke_test.json").read_text())
    if smoke["status"] != "pass":
        raise RuntimeError("Cannot seed reuse rows before the SHA256 smoke audit passes")
    for method, source_path, cache_path, config in sources:
        source = json.loads(source_path.read_text())
        source_tasks = source.get("suite")
        if source_tasks != tasks or source["input_sha256"] != smoke["input_sha256"]:
            raise RuntimeError(f"Identity mismatch in {source_path}")
        for experiment, setting in (("trigger_size", "22"), ("task_number", "6")):
            slug_method = "emr" if method == "EMR" else "regmean_plusplus"
            run_dir = out / "runs" / f"{experiment}__{setting}__{slug_method}"
            payload = {
                "complete": True, "experiment": experiment, "setting_value": setting,
                "merge_method": method, "avg_clean_acc": source["avg_clean_acc"],
                "asr": source["asr"], "per_task_acc": source["per_task_acc"],
                "num_tasks": 6, "tasks": tasks, "adversary_task": "CIFAR100",
                "target_class": 1, "seed": 2026,
                "malicious_checkpoint": str(malicious.resolve()),
                "checkpoint_paths": [str(p.resolve()) for p in paths],
                "trigger_path": str(trigger.resolve()), "cache_path": str(cache_path.resolve()),
                "input_sha256": source["input_sha256"], "config": config,
                "max_eval_batches": 0, "started_at_utc": None,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": 0.0, "gpu": "reused completed full-data run",
                "reuse_source": str(source_path.resolve()),
                "reuse_reason": "byte-identical checkpoint, trigger, task order, seed, and merger configuration",
            }
            dump_json(run_dir / "result.json", payload)
            print(run_dir / "result.json")


if __name__ == "__main__":
    main()
