#!/usr/bin/env python3
"""Audit existing sensitivity artifacts and materialize the missing-job matrix."""

from __future__ import annotations

import json
from pathlib import Path

from src.reviewer_exp.common import ROOT, dump_json


OUT = ROOT / "bmr_paper_results" / "reviewer_response" / "sensitivity_six_merges"
BASE6 = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
TRIGGER22 = ROOT / "trigger" / "ViT-B-32" / "KDR_DTK_CIFAR100_Tgt_1_L_22.npy"


def config_for(checkpoint):
    summaries = sorted(checkpoint.parent.glob("*_dtk_bmr_summary.json"))
    if not summaries:
        return None, None
    summary = json.loads(summaries[-1].read_text())
    config = Path(summary["config_path"])
    if not config.is_absolute():
        config = ROOT / config
    return summaries[-1], config


def add(rows, experiment, value, method, trigger, original_result, script, tasks):
    checkpoint = ROOT / "checkpoints" / "ViT-B-32" / f"CIFAR100_{method}_CIFAR100_Tgt_1_L_{trigger.stem.rsplit('_', 1)[-1]}" / "finetuned.pt"
    summary, config = config_for(checkpoint)
    cfg = json.loads(config.read_text()) if config and config.is_file() else {}
    rows.append({
        "experiment": experiment,
        "setting_value": str(value),
        "malicious_checkpoint": str(checkpoint.resolve()),
        "trigger_path": str(trigger.resolve()),
        "checkpoint_exists": checkpoint.is_file(),
        "trigger_exists": trigger.is_file(),
        "original_result_file": str((ROOT / original_result).resolve()),
        "original_script": str((ROOT / script).resolve()),
        "original_config": str(config.resolve()) if config else None,
        "original_summary": str(summary.resolve()) if summary else None,
        "original_seed": cfg.get("args", {}).get("seed", "not_recorded"),
        "suite": tasks,
        "task_subset": tasks,
        "task_order": tasks,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for value, method, result in [
        ("1", "KDR_DTK_BMR_PR1", "results/bmr_sensitivity/20261011_234636_KDR_DTK_BMR_PR1_CIFAR100_tgt1_L22.json"),
        ("2", "KDR_DTK_BMR_PR2", "results/bmr_sensitivity/20261012_012911_KDR_DTK_BMR_PR2_CIFAR100_tgt1_L22.json"),
        ("5", "KDR_DTK_BMR_PR5", "results/bmr_sensitivity/20261012_031415_KDR_DTK_BMR_PR5_CIFAR100_tgt1_L22.json"),
        ("20", "KDR_DTK_BMR_PR20", "results/bmr_sensitivity/20261012_042043_KDR_DTK_BMR_PR20_CIFAR100_tgt1_L22.json"),
    ]:
        add(rows, "poison_ratio", value, method, TRIGGER22, result,
            "scripts/run_bmr_cifar100_poison_sensitivity.sh", BASE6)

    for value, method, result in [
        ("0", "KDR_DTK_BMR_G0", "results/bmr_sensitivity/20261011_235626_KDR_DTK_BMR_G0_CIFAR100_tgt1_L22.json"),
        ("0.1", "KDR_DTK_BMR_G01", "results/bmr_sensitivity/20261012_015945_KDR_DTK_BMR_G01_CIFAR100_tgt1_L22.json"),
        ("0.25", "KDR_DTK_BMR_G025", "results/bmr_sensitivity/20261012_121851_KDR_DTK_BMR_G025_CIFAR100_tgt1_L22.json"),
        ("0.5", "KDR_DTK_BMR_G05", "results/bmr_sensitivity/20261012_140341_KDR_DTK_BMR_G05_CIFAR100_tgt1_L22.json"),
        ("1.0", "KDR_DTK_BMR_G10", "results/bmr_sensitivity/20261012_150950_KDR_DTK_BMR_G10_CIFAR100_tgt1_L22.json"),
    ]:
        add(rows, "background_strength", value, method, TRIGGER22, result,
            "scripts/run_bmr_cifar100_drift_sensitivity.sh", BASE6)

    trigger_results = {
        "16": "results/bmr_trigger_size/20261012_164557_KDR_DTK_BMR_CIFAR100_tgt1_L16.json",
        "22": "results/bmr_trigger_size/20261012_154715_KDR_DTK_BMR_CIFAR100_tgt1_L22.json",
        "27": "results/bmr_trigger_size/20261012_164522_KDR_DTK_BMR_CIFAR100_tgt1_L27.json",
        "32": "results/bmr_trigger_size/20261012_172318_KDR_DTK_BMR_CIFAR100_tgt1_L32.json",
    }
    for size, result in trigger_results.items():
        trigger = ROOT / "trigger" / "ViT-B-32" / f"KDR_DTK_CIFAR100_Tgt_1_L_{size}.npy"
        add(rows, "trigger_size", size, "KDR_DTK_BMR", trigger, result,
            "scripts/run_bmr_cifar100_trigger_size_task.sh", BASE6)

    task_results = {
        "2": "results/bmr_task_number/20261013_025440_KDR_DTK_BMR_CIFAR100_tgt1_L22.json",
        "4": "results/bmr_task_number/20261013_031429_KDR_DTK_BMR_CIFAR100_tgt1_L22.json",
        "6": "results/bmr_task_number/20261013_035646_KDR_DTK_BMR_CIFAR100_tgt1_L22.json",
    }
    default_checkpoint = ROOT / "checkpoints" / "ViT-B-32" / "CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22" / "finetuned.pt"
    summary, config = config_for(default_checkpoint)
    cfg = json.loads(config.read_text()) if config else {}
    for count in (2, 4, 6):
        tasks = BASE6[:count]
        rows.append({
            "experiment": "task_number", "setting_value": str(count),
            "malicious_checkpoint": str(default_checkpoint.resolve()),
            "trigger_path": str(TRIGGER22.resolve()),
            "checkpoint_exists": default_checkpoint.is_file(), "trigger_exists": TRIGGER22.is_file(),
            "original_result_file": str((ROOT / task_results[str(count)]).resolve()),
            "original_script": str((ROOT / "scripts/run_bmr_cifar100_task_number_task.sh").resolve()),
            "original_config": str(config.resolve()), "original_summary": str(summary.resolve()),
            "original_seed": cfg.get("args", {}).get("seed", "not_recorded"),
            "suite": tasks, "task_subset": tasks, "task_order": tasks,
        })

    missing = [r for r in rows if not r["checkpoint_exists"] or not r["trigger_exists"]]
    audit = {
        "status": "pass" if not missing else "missing_artifacts",
        "retraining_required": bool(missing),
        "retraining_performed": False,
        "target_class": 1,
        "adversary_task": "CIFAR100",
        "entries": rows,
        "missing": missing,
    }
    dump_json(OUT / "artifact_audit.json", audit)
    if missing:
        raise SystemExit(f"Missing artifacts; full batch not authorized: {missing}")

    with (OUT / "jobs.tsv").open("w", encoding="utf-8") as f:
        f.write("experiment\tsetting_value\ttasks\tcheckpoint\ttrigger\n")
        for row in rows:
            f.write("\t".join([
                row["experiment"], row["setting_value"], ",".join(row["task_order"]),
                row["malicious_checkpoint"], row["trigger_path"],
            ]) + "\n")
    print(OUT / "artifact_audit.json")
    print(OUT / "jobs.tsv")


if __name__ == "__main__":
    main()
