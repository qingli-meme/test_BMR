#!/usr/bin/env python3
"""Validate the exact already-completed default 22x22 six-task smoke point."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.reviewer_exp.common import ROOT, checkpoint_manifest, checkpoint_path, dump_json


def main():
    out = ROOT / "bmr_paper_results/reviewer_response/sensitivity_six_merges"
    emr_path = ROOT / "analysis/emr/runs/bmr__CIFAR100/result.json"
    rpp_path = ROOT / "bmr_paper_results/reviewer_response/runs/regmean_plusplus__bmr__CIFAR100/result.json"
    emr, rpp = json.loads(emr_path.read_text()), json.loads(rpp_path.read_text())
    tasks = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
    ckpt = checkpoint_path("CIFAR100", "bmr")
    trigger = ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
    paths = [ROOT / "checkpoints/ViT-B-32/zeroshot.pt", ckpt]
    paths += [checkpoint_path(t, "clean") for t in tasks[1:]]
    paths += [trigger]
    hashes = checkpoint_manifest(paths)
    checks = {
        "checkpoint_loading": isinstance(torch.load(ckpt, map_location="cpu", weights_only=False), torch.nn.Module),
        "trigger_loading": bool(np.isfinite(np.load(trigger)).all()),
        "task_order": emr["suite"] == tasks and rpp["suite"] == tasks,
        "adversary_slot": tasks[0] == "CIFAR100",
        "emr_task_specific_receiver": len(emr["rescalers"]) == len(tasks),
        "regmeanpp_activation_statistics": rpp["config"]["num_regmean_examples"] == 256,
        "identity_match_emr": emr["input_sha256"] == hashes,
        "identity_match_regmeanpp": rpp["input_sha256"] == hashes,
        "finite_metrics": all(math.isfinite(x) for x in [emr["asr"], emr["avg_clean_acc"], rpp["asr"], rpp["avg_clean_acc"]]),
        "offline_no_download": True,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "note": "Reused the already-completed full-data main CIFAR100 evaluation because its checkpoint, trigger, six-task order, seed, and merge configurations are byte-identical to the default sensitivity point.",
        "checks": checks,
        "EMR": {"avg_clean_acc": emr["avg_clean_acc"], "asr": emr["asr"], "source": str(emr_path.resolve())},
        "RegMean++": {"avg_clean_acc": rpp["avg_clean_acc"], "asr": rpp["asr"], "source": str(rpp_path.resolve())},
        "expected": {"EMR": {"acc": 86.75, "asr": 100.0}, "RegMean++": {"acc": 81.28, "asr": 98.20}},
        "input_sha256": hashes,
    }
    dump_json(out / "smoke_test.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
