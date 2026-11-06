#!/usr/bin/env python3
"""Consolidate off-task v2 optimization and completed-merge metrics."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


METHODS = ("ta", "ties", "regmean", "adamerging", "emr", "regmeanpp")
DISPLAY = {
    "ta": "Task Arithmetic",
    "ties": "TIES",
    "regmean": "RegMean",
    "adamerging": "AdaMerging",
    "emr": "EMR",
    "regmeanpp": "RegMean++",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/bmr_offtask_v2/Cars")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22",
    )
    parser.add_argument(
        "--run-log", default="logs/bmr_offtask_v2/Cars_20261105_143214.log"
    )
    parser.add_argument(
        "--output",
        default="results/bmr_offtask_v2/Cars/INTERMEDIATE_METRICS_CURRENT.json",
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_or_absolute(path: Path):
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def adamerging_progress(run_log: Path):
    if not run_log.is_file():
        return None
    pattern = re.compile(r"\[OFF-ADA\]\s+(\d+)/(\d+)\s+loss=([-+0-9.eE]+)")
    history = []
    for match in pattern.finditer(run_log.read_text(encoding="utf-8", errors="replace")):
        history.append(
            {
                "step": int(match.group(1)),
                "total_steps": int(match.group(2)),
                "loss": float(match.group(3)),
            }
        )
    if not history:
        return None
    return {
        "latest_step": history[-1]["step"],
        "total_steps": history[-1]["total_steps"],
        "completion_fraction": history[-1]["step"] / history[-1]["total_steps"],
        "loss_history": history,
    }


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_log = Path(args.run_log)
    output = Path(args.output)

    stage1_path = result_dir / "stage1_summary.json"
    stage2_path = result_dir / "stage2_summary.json"
    config_path = result_dir / "config.json"
    metadata_path = result_dir / "target_metadata.json"
    required = [stage1_path, stage2_path, config_path, metadata_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    completed = {}
    ta_asr = None
    for method in METHODS:
        path = result_dir / f"{method}.json"
        if not path.is_file():
            continue
        result = read_json(path)
        if method == "ta":
            ta_asr = float(result["asr"])
        clean_tar = float(result["clean_merge_trigger_target_rate"])
        asr = float(result["asr"])
        completed[method] = {
            "display_name": DISPLAY[method],
            "status": "complete",
            "asr_percent": asr,
            "avg_clean_acc_percent": float(result["avg_clean_acc"]),
            "target_task_clean_acc_percent": float(result["target_clean_acc"]),
            "clean_receiver_trigger_target_rate_percent": clean_tar,
            "activation_gain_over_clean_receiver_pp": asr - clean_tar,
            "per_task_clean_acc_percent": result["per_task_acc"],
            "result_json": relative_or_absolute(path),
            "attacked_encoder": result["attacked_encoder"],
            "background_encoder": result["background_encoder"],
        }
    if ta_asr is not None:
        for result in completed.values():
            result["asr_gap_vs_task_arithmetic_pp"] = (
                result["asr_percent"] - ta_asr
            )

    status = {}
    ada_progress = adamerging_progress(run_log)
    for method in METHODS:
        if method in completed:
            status[method] = "complete"
        elif method == "adamerging" and ada_progress is not None:
            status[method] = "running"
        else:
            status[method] = "pending"

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment": {
            "name": "BMR off-task text-direction v2",
            "source_task": "CIFAR100",
            "unknown_downstream_target_task": "Cars",
            "target_class": 1,
            "target_classname": read_json(metadata_path).get("target_classname"),
            "method_status": status,
        },
        "artifacts": {
            "config": relative_or_absolute(config_path),
            "target_metadata": relative_or_absolute(metadata_path),
            "trigger": "trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy",
            "malicious_checkpoint": relative_or_absolute(
                checkpoint_dir / "finetuned.pt"
            ),
            "full_live_log": relative_or_absolute(run_log),
            "stage1_step_log": relative_or_absolute(result_dir / "stage1.jsonl"),
            "stage2_step_log": relative_or_absolute(checkpoint_dir / "stage2.jsonl"),
        },
        "config": read_json(config_path),
        "target_metadata": read_json(metadata_path),
        "stage1_trigger_optimization": {
            "summary_path": relative_or_absolute(stage1_path),
            **read_json(stage1_path),
        },
        "stage2_malicious_model_training": {
            "summary_path": relative_or_absolute(stage2_path),
            **read_json(stage2_path),
        },
        "merge_evaluations_completed": completed,
        "merge_optimization_in_progress": {
            "adamerging": ada_progress,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
