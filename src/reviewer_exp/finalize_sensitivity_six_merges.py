#!/usr/bin/env python3
"""Validate and consolidate completed EMR/RegMean++ sensitivity evaluations."""

from __future__ import annotations

import csv
import glob
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.reviewer_exp.common import ROOT, dump_json
from src.reviewer_exp.merge_emr import UPSTREAM_COMMIT as EMR_COMMIT


OUT = ROOT / "bmr_paper_results" / "reviewer_response" / "sensitivity_six_merges"
DISPLAY = {"TA": "ta", "TIES": "ties", "RegMean": "regmean", "AdaMerging": "adamerging",
           "EMR": "emr", "RegMean++": "regmeanpp"}
METHODS = list(DISPLAY)

TRIGGER_OLD = {
    "16": {"TA": (74.98388576381227, 100.0), "TIES": (72.91274199595843, 99.98989898989899), "RegMean": (75.57, 97.34), "AdaMerging": (80.50, 100.0)},
    "22": {"TA": (74.83321510448262, 100.0), "TIES": (72.70894297944535, 100.0), "RegMean": (75.30, 97.33), "AdaMerging": (80.34, 100.0)},
    "27": {"TA": (74.98152419866203, 100.0), "TIES": (72.79291629898857, 100.0), "RegMean": (75.47, 98.59), "AdaMerging": (80.51, 100.0)},
    "32": {"TA": (75.00721015906694, 100.0), "TIES": (72.78248636856507, 100.0), "RegMean": (75.43, 99.97), "AdaMerging": (80.60, 100.0)},
}
TASK_OLD = {
    "2": {"TA": (80.50378463974664, 100.0), "TIES": (67.94074821852732, 100.0), "RegMean": (88.69, 100.0), "AdaMerging": (88.30, 100.0)},
    "4": {"TA": (77.64139534626615, 100.0), "TIES": (70.96650496621263, 100.0), "RegMean": (79.94, 100.0), "AdaMerging": (84.44, 100.0)},
    "6": {"TA": (74.83321510448262, 100.0), "TIES": (72.70894297944535, 100.0), "RegMean": (75.38, 97.70), "AdaMerging": (80.50, 100.0)},
}
AREA = {"16": 0.51, "22": 0.96, "27": 1.45, "32": 2.04}


def read_old_csv(path, key):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            setting = row[key]
            out.setdefault(setting, {})[row["merge"]] = (float(row["avg_clean_acc"]), float(row["asr"]))
    return out


def load_new():
    audit = json.loads((OUT / "artifact_audit.json").read_text())
    expected = {(x["experiment"], x["setting_value"], method)
                for x in audit["entries"] for method in ("EMR", "RegMean++")}
    rows, seen = [], set()
    for path in sorted((OUT / "runs").glob("*/*result.json")):
        d = json.loads(path.read_text())
        key = (d["experiment"], str(d["setting_value"]), d["merge_method"])
        if key not in expected:
            continue
        if key in seen:
            raise RuntimeError(f"Duplicate result {key}: {path}")
        seen.add(key)
        vals = [d["asr"], d["avg_clean_acc"], *d["per_task_acc"].values()]
        if not d.get("complete") or not all(math.isfinite(float(v)) and 0 <= float(v) <= 100 for v in vals):
            raise RuntimeError(f"Invalid result {path}")
        if d["adversary_task"] != "CIFAR100" or d["target_class"] != 1 or d["seed"] != 2026:
            raise RuntimeError(f"Protocol mismatch {path}")
        if d["tasks"][0] != "CIFAR100" or len(d["tasks"]) != d["num_tasks"]:
            raise RuntimeError(f"Task-order mismatch {path}")
        d["result_path"] = str(path.resolve())
        rows.append(d)
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} evaluations: {missing}")
    return audit, rows


def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def wide_rows(experiment, settings, combined, extra=None):
    rows = []
    for setting in settings:
        row = {"setting": setting}
        if extra:
            row.update(extra(setting))
        for method in METHODS:
            acc, asr = combined[str(setting)][method]
            prefix = DISPLAY[method]
            row[f"{prefix}_asr"] = asr
            row[f"{prefix}_acc"] = acc
        rows.append(row)
    return rows


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def main():
    audit, new = load_new()
    lookup = {(r["experiment"], str(r["setting_value"]), r["merge_method"]):
              (r["avg_clean_acc"], r["asr"]) for r in new}
    specs = {
        "poison_ratio": (["1", "2", "5", "20"], read_old_csv(ROOT / "bmr_paper_results/bmr_poison_ratio_sensitivity_cifar100.csv", "poison_ratio")),
        "background_strength": (["0", "0.1", "0.25", "0.5", "1.0"], read_old_csv(ROOT / "bmr_paper_results/bmr_background_strength_sensitivity_cifar100.csv", "gamma")),
        "trigger_size": (["16", "22", "27", "32"], TRIGGER_OLD),
        "task_number": (["2", "4", "6"], TASK_OLD),
    }
    combined_all = {}
    for exp, (settings, old) in specs.items():
        combined = {s: dict(old[s]) for s in settings}
        for s in settings:
            for method in ("EMR", "RegMean++"):
                combined[s][method] = lookup[(exp, s, method)]
        combined_all[exp] = combined

    # New-method long and per-study files.
    long_rows = []
    audit_by = {(x["experiment"], x["setting_value"]): x for x in audit["entries"]}
    for r in sorted(new, key=lambda x: (x["experiment"], float(x["setting_value"]), x["merge_method"])):
        long_rows.append({
            "experiment": r["experiment"], "setting_value": r["setting_value"],
            "merge_method": r["merge_method"], "asr": r["asr"],
            "avg_clean_acc": r["avg_clean_acc"], "num_tasks": r["num_tasks"],
            "adversary_task": "CIFAR100", "target_class": 1,
            "checkpoint_path": r["malicious_checkpoint"], "trigger_path": r["trigger_path"],
            "seed": 2026,
        })
    long_fields = ["experiment", "setting_value", "merge_method", "asr", "avg_clean_acc",
                   "num_tasks", "adversary_task", "target_class", "checkpoint_path", "trigger_path", "seed"]
    write_rows(OUT / "all_sensitivity_new_merges_long.csv", long_fields, long_rows)

    names = {
        "poison_ratio": ("poison_ratio_emr_regmeanpp.csv", "poison_ratio"),
        "background_strength": ("background_strength_emr_regmeanpp.csv", "gamma"),
        "trigger_size": ("trigger_size_emr_regmeanpp.csv", "trigger_size"),
        "task_number": ("task_number_emr_regmeanpp.csv", "num_tasks"),
    }
    for exp, (filename, keyname) in names.items():
        rows = []
        for r in new:
            if r["experiment"] != exp:
                continue
            a = audit_by[(exp, str(r["setting_value"]))]
            row = {keyname: r["setting_value"]}
            if exp == "trigger_size": row["area_percent"] = AREA[str(r["setting_value"])]
            if exp == "task_number": row["ordered_tasks"] = ",".join(r["tasks"])
            row.update({"merge_method": r["merge_method"], "avg_clean_acc": r["avg_clean_acc"],
                        "asr": r["asr"], "checkpoint_path": a["malicious_checkpoint"],
                        "trigger_path": a["trigger_path"]})
            rows.append(row)
        fields = list(rows[0])
        write_rows(OUT / filename, fields, sorted(rows, key=lambda x: (float(x[keyname]), x["merge_method"])))

    for exp, (settings, _) in specs.items():
        extras = (lambda s: {"area_percent": AREA[str(s)]}) if exp == "trigger_size" else None
        rows = wide_rows(exp, settings, combined_all[exp], extras)
        keyname = {"poison_ratio": "poison_ratio", "background_strength": "gamma",
                   "trigger_size": "trigger_size", "task_number": "num_tasks"}[exp]
        for row in rows: row[keyname] = row.pop("setting")
        fields = [keyname] + (["area_percent"] if exp == "trigger_size" else [])
        if exp == "task_number":
            fields += [f"{DISPLAY[m]}_asr" for m in METHODS] + [f"{DISPLAY[m]}_acc" for m in METHODS]
        else:
            for m in METHODS: fields += [f"{DISPLAY[m]}_asr", f"{DISPLAY[m]}_acc"]
        write_rows(OUT / f"{exp}_six_merges.csv", fields, rows)

    lines = ["# Completed BMR Sensitivity Results: Six Merging Algorithms", "",
             "All values below are measured single-seed results (seed 2026). Existing four-merger values were joined without recomputation.", ""]
    titles = [("poison_ratio", "10.1 Poisoning ratio", "Ratio"),
              ("background_strength", "10.2 Background strength", "Gamma"),
              ("trigger_size", "10.3 Trigger size", "Size"),
              ("task_number", "10.4 Number of merged tasks", "Tasks")]
    for exp, title, first in titles:
        lines += [f"## {title}", ""]
        settings = specs[exp][0]
        if exp == "task_number":
            headers = [first] + [f"{m} ASR" for m in METHODS]
            vals = [[s] + [f"{combined_all[exp][s][m][1]:.2f}" for m in METHODS] for s in settings]
        else:
            headers = [first] + (["Area"] if exp == "trigger_size" else []) + [f"{m} ACC/ASR" for m in METHODS]
            vals = []
            for s in settings:
                row = [s] + ([f"{AREA[s]:.2f}%"] if exp == "trigger_size" else [])
                row += [f"{combined_all[exp][s][m][0]:.2f}/{combined_all[exp][s][m][1]:.2f}" for m in METHODS]
                vals.append(row)
        table(lines, headers, vals)
        emr_asr = [combined_all[exp][s]["EMR"][1] for s in settings]
        rpp_asr = [combined_all[exp][s]["RegMean++"][1] for s in settings]
        lines += ["", f"Measured range: EMR ASR {min(emr_asr):.2f}–{max(emr_asr):.2f}%; RegMean++ ASR {min(rpp_asr):.2f}–{max(rpp_asr):.2f}%.", ""]
    lines += ["## Execution notes", "",
              "All EMR and RegMean++ sensitivity results were obtained by re-merging and evaluating the existing sensitivity checkpoints. No BMR model or trigger was retrained for the new merging algorithms.", "",
              "Artifact audit: `artifact_audit.json`. Full machine-readable new rows: `all_sensitivity_new_merges_long.csv`.", ""]
    (OUT / "completed_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = branch = "unavailable"
    logs = sorted(str(Path(p).resolve()) for p in glob.glob(str(ROOT / "bmr_paper_results/reviewer_response/logs/sensitivity_six_merges/*")))
    manifest = {
        "repository_commit": commit, "branch": branch,
        "date_utc": datetime.now(timezone.utc).isoformat(), "seed": 2026,
        "gpu": sorted(set(r.get("gpu", "not_recorded") for r in new)),
        "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "emr_implementation": "src/reviewer_exp/merge_emr.py",
        "emr_upstream_commit": EMR_COMMIT,
        "regmeanpp_implementation": "src/reviewer_exp/merge_regmean_plusplus.py",
        "regmeanpp_upstream_commit": "a9fdd97",
        "source_sensitivity_scripts": sorted(set(x["original_script"] for x in audit["entries"])),
        "source_sensitivity_checkpoints": sorted(set(x["malicious_checkpoint"] for x in audit["entries"])),
        "checkpoint_and_trigger_hashes": {p: h for r in new for p, h in r["input_sha256"].items()},
        "task_order_by_task_number": {x["setting_value"]: x["task_order"] for x in audit["entries"] if x["experiment"] == "task_number"},
        "cache_locations": sorted(set(r["cache_path"] for r in new)),
        "exact_command_template": "python3 -m src.reviewer_exp.run_sensitivity_new_merges --merge-method <emr|regmean_plusplus> --experiment <name> --setting-value <value> --tasks <ordered_csv> --malicious-checkpoint <path> --trigger <path> --output-dir <path> --device cuda --seed 2026",
        "worker_script": str((ROOT / "scripts/run_sensitivity_six_merges_worker.sh").resolve()),
        "logs": logs,
        "runs": [{k: r[k] for k in ("experiment", "setting_value", "merge_method", "runtime_seconds", "result_path", "cache_path", "input_sha256")} for r in new],
        "retraining_required": False, "retraining_performed": False,
    }
    dump_json(OUT / "sensitivity_six_merges_manifest.json", manifest)
    print(OUT / "completed_sensitivity_summary.md")


if __name__ == "__main__":
    main()
