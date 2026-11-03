#!/usr/bin/env python3
"""Collect TA/TIES JSON and RegMean/AdaMerging logs for BMR sensitivity runs."""

import argparse
import csv
import glob
import json
import re
from pathlib import Path


SPECS = {
    "poison": [
        ("1", "KDR_DTK_BMR_PR1"),
        ("2", "KDR_DTK_BMR_PR2"),
        ("5", "KDR_DTK_BMR_PR5"),
        ("20", "KDR_DTK_BMR_PR20"),
    ],
    "gamma": [
        ("0", "KDR_DTK_BMR_G0"),
        ("0.1", "KDR_DTK_BMR_G01"),
        ("0.25", "KDR_DTK_BMR_G025"),
        ("0.5", "KDR_DTK_BMR_G05"),
        ("1.0", "KDR_DTK_BMR_G10"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=sorted(SPECS), required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def latest_json(method):
    matches = glob.glob(f"results/bmr_sensitivity/*_{method}_CIFAR100_tgt1_L22.json")
    return max(matches, key=lambda path: Path(path).stat().st_mtime) if matches else None


def parse_log(path, methods):
    results = {}
    current_method = None
    current_merge = None
    current_clean = None
    header = re.compile(r"method=(KDR_DTK_BMR_[A-Z0-9]+)\s+start=")
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            current_method = match.group(1) if match.group(1) in methods else None
            current_merge = None
            current_clean = None
            continue
        if current_method is None:
            continue
        if "src/main_regmean_badmergingon.py" in line:
            current_merge = "RegMean"
        elif "src/main_adamerging_badmergingon.py" in line:
            current_merge = "AdaMerging"
        match = re.search(r"Avg ACC:([0-9.]+)%", line)
        if match and current_merge:
            current_clean = float(match.group(1))
            continue
        match = re.search(r"Backdoor acc:\s*([0-9.]+)", line)
        if match and current_merge and current_clean is not None:
            results[(current_method, current_merge)] = {
                "avg_clean_acc": current_clean,
                "asr": float(match.group(1)) * 100.0,
            }
    return results


def main():
    args = parse_args()
    setting_key = "poison_ratio" if args.study == "poison" else "gamma"
    specs = SPECS[args.study]
    methods = {method for _, method in specs}
    logged = parse_log(args.log, methods)
    rows = []
    for setting, method in specs:
        result_path = latest_json(method)
        if result_path is None:
            raise FileNotFoundError(f"missing TA/TIES JSON for {method}")
        data = json.loads(Path(result_path).read_text(encoding="utf-8"))
        for key, label in (("ta", "TA"), ("ties", "TIES")):
            result = data["merge_results"][key]
            rows.append({
                setting_key: setting,
                "merge": label,
                "avg_clean_acc": result["avg_clean_acc"],
                "asr": result["asr"],
                "source": result_path,
            })
        for label in ("RegMean", "AdaMerging"):
            result = logged.get((method, label))
            if result is None:
                raise RuntimeError(f"missing {label} metrics for {method} in {args.log}")
            rows.append({
                setting_key: setting,
                "merge": label,
                **result,
                "source": args.log,
            })

    fields = [setting_key, "merge", "avg_clean_acc", "asr", "source"]
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.json).write_text(
        json.dumps({"study": args.study, "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
