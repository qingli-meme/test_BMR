"""Compare real post-RegMean mechanism metrics for off-task v2 and v3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [os.path.abspath("."), os.path.abspath("./src")]

from src.analyze_bmr_offtask_v2 import analyze


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-task", default="Cars")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--v2-result-dir", default="results/bmr_offtask_v2/Cars")
    parser.add_argument("--v3-result-dir", default="results/bmr_offtask_v3/Cars")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def evaluate_version(args, result_dir, version):
    runtime = SimpleNamespace(**vars(args))
    runtime.result_dir = str(result_dir)
    mechanism = analyze(runtime, "regmean")
    result = json.loads((result_dir / "regmean.json").read_text(encoding="utf-8"))
    return {
        "version": version,
        "result_dir": str(result_dir),
        "asr_percent": float(result["asr"]),
        "target_clean_acc_percent": float(result["target_clean_acc"]),
        "avg_clean_acc_percent": float(result["avg_clean_acc"]),
        "mechanism": mechanism,
    }


def main():
    args = parse_args()
    v2_dir = Path(args.v2_result_dir)
    v3_dir = Path(args.v3_result_dir)
    required = [
        v2_dir / "regmean.json",
        v2_dir / "regmean_attacked.pt",
        v2_dir / "regmean_background.pt",
        v3_dir / "regmean.json",
        v3_dir / "regmean_attacked.pt",
        v3_dir / "regmean_background.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    v2 = evaluate_version(args, v2_dir, "v2_scalar")
    v3 = evaluate_version(args, v3_dir, "v3_directional")
    comparison = {
        "asr_delta_pp": v3["asr_percent"] - v2["asr_percent"],
        "target_clean_acc_delta_pp": (
            v3["target_clean_acc_percent"] - v2["target_clean_acc_percent"]
        ),
        "avg_clean_acc_delta_pp": (
            v3["avg_clean_acc_percent"] - v2["avg_clean_acc_percent"]
        ),
        "mean_G_true_delta": (
            v3["mechanism"]["G_true"]["mean"]
            - v2["mechanism"]["G_true"]["mean"]
        ),
        "median_G_true_delta": (
            v3["mechanism"]["G_true"]["median"]
            - v2["mechanism"]["G_true"]["median"]
        ),
        "p_G_true_gt_D_true_delta": (
            v3["mechanism"]["p_G_true_gt_D_true"]
            - v2["mechanism"]["p_G_true_gt_D_true"]
        ),
        "mean_G_text_delta": (
            v3["mechanism"]["G_text"]["mean"]
            - v2["mechanism"]["G_text"]["mean"]
        ),
    }
    payload = {"v2": v2, "v3": v3, "comparison": comparison}
    output = v3_dir / "regmean_mechanism.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
