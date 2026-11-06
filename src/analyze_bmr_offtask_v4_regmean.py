"""RegMean mechanism and task-vector comparison for off-task v4."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path[:0] = [os.path.abspath("."), os.path.abspath("./src")]

from src.analyze_bmr_offtask_v2 import analyze


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-task", default="Cars")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--v2-result-dir", default="results/bmr_offtask_v2/Cars")
    parser.add_argument("--v4-result-dir", default="results/bmr_offtask_v4/Cars")
    parser.add_argument("--zeroshot-checkpoint", required=True)
    parser.add_argument("--clean-checkpoint", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--v4-checkpoint", required=True)
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported checkpoint: {path}")


def residual_comparison(args):
    states = {
        "theta0": load_state(args.zeroshot_checkpoint),
        "clean": load_state(args.clean_checkpoint),
        "v2": load_state(args.v2_checkpoint),
        "v4": load_state(args.v4_checkpoint),
    }
    totals = {
        "clean_sq": 0.0,
        "v2_sq": 0.0,
        "v4_sq": 0.0,
        "v2_clean_dot": 0.0,
        "v4_clean_dot": 0.0,
    }
    used = 0
    for name, theta0 in states["theta0"].items():
        if not torch.is_floating_point(theta0):
            continue
        if any(name not in states[key] for key in ("clean", "v2", "v4")):
            continue
        d_clean = states["clean"][name].float() - theta0.float()
        d_v2 = states["v2"][name].float() - states["clean"][name].float()
        d_v4 = states["v4"][name].float() - theta0.float()
        totals["clean_sq"] += float(torch.sum(d_clean * d_clean))
        totals["v2_sq"] += float(torch.sum(d_v2 * d_v2))
        totals["v4_sq"] += float(torch.sum(d_v4 * d_v4))
        totals["v2_clean_dot"] += float(torch.sum(d_v2 * d_clean))
        totals["v4_clean_dot"] += float(torch.sum(d_v4 * d_clean))
        used += 1
    clean_norm = totals["clean_sq"] ** 0.5
    v2_norm = totals["v2_sq"] ** 0.5
    v4_norm = totals["v4_sq"] ** 0.5
    return {
        "parameter_tensors": used,
        "cos_delta_v4_delta_clean": totals["v4_clean_dot"]
        / max(v4_norm * clean_norm, 1e-30),
        "cos_delta_v2_delta_clean": totals["v2_clean_dot"]
        / max(v2_norm * clean_norm, 1e-30),
        "norm_delta_v4": v4_norm,
        "norm_delta_clean": clean_norm,
        "norm_delta_v2": v2_norm,
        "definitions": {
            "delta_clean": "theta_clean - theta0",
            "delta_v2": "theta_adv_v2 - theta_clean",
            "delta_v4": "theta_adv_v4 - theta0",
        },
    }


def main():
    args = parse_args()
    v2_dir = Path(args.v2_result_dir)
    v4_dir = Path(args.v4_result_dir)
    required = [
        v2_dir / "regmean.json",
        v4_dir / "regmean.json",
        v4_dir / "regmean_attacked.pt",
        v4_dir / "regmean_background.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    args.result_dir = str(v4_dir)
    mechanism = analyze(args, "regmean")
    v2 = json.loads((v2_dir / "regmean.json").read_text(encoding="utf-8"))
    v4 = json.loads((v4_dir / "regmean.json").read_text(encoding="utf-8"))
    residual = residual_comparison(args)
    residual_path = v4_dir / "residual_comparison.json"
    residual_path.write_text(json.dumps(residual, indent=2), encoding="utf-8")
    payload = {
        "v2": {
            "training_structure": "post-hoc attack increment from theta_clean",
            "asr_percent": float(v2["asr"]),
            "avg_clean_acc_percent": float(v2["avg_clean_acc"]),
            "target_clean_acc_percent": float(v2["target_clean_acc"]),
        },
        "v3": {
            "training_structure": "post-hoc plus directional randomization",
            "asr_percent": 77.41,
            "avg_clean_acc_percent": None,
            "target_clean_acc_percent": None,
        },
        "v4": {
            "training_structure": "joint whole task vector from theta0",
            "asr_percent": float(v4["asr"]),
            "avg_clean_acc_percent": float(v4["avg_clean_acc"]),
            "target_clean_acc_percent": float(v4["target_clean_acc"]),
            "mechanism": mechanism,
        },
        "v4_minus_v2": {
            "asr_delta_pp": float(v4["asr"]) - float(v2["asr"]),
            "avg_clean_acc_delta_pp": float(v4["avg_clean_acc"])
            - float(v2["avg_clean_acc"]),
            "target_clean_acc_delta_pp": float(v4["target_clean_acc"])
            - float(v2["target_clean_acc"]),
        },
        "residual_comparison": residual,
    }
    output = v4_dir / "regmean_mechanism.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
