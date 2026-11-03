#!/usr/bin/env python3
"""Assemble the six-merger representation-layer sensitivity table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bmr_paper_results/reviewer_response/representation_layer_six_merges"
LAYERS = {
    "Middle (block 5)": {
        "block": 5,
        "checkpoint": ROOT / "checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_MID_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "trigger": ROOT / "trigger/ViT-B-32/KDR_DTK_REP_MID_CIFAR100_Tgt_1_L_22.npy",
        "ta_ties": ROOT / "results/bmr_representation_layer_sensitivity/20261022_164829_KDR_DTK_BMR_REP_MID_CIFAR100_tgt1_L22.json",
        "emr": OUT / "runs/middle__emr/result.json",
        "regmeanpp": OUT / "runs/middle__regmean_plusplus/result.json",
        "regmean": (89.76767676767676, 75.57367617037944),
        "adamerging": (99.96969696969697, 80.52412723909178),
    },
    "Late (block 9)": {
        "block": 9,
        "checkpoint": ROOT / "checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_REP_LATE_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "trigger": ROOT / "trigger/ViT-B-32/KDR_DTK_REP_LATE_CIFAR100_Tgt_1_L_22.npy",
        "ta_ties": ROOT / "results/bmr_representation_layer_sensitivity/20261022_200113_KDR_DTK_BMR_REP_LATE_CIFAR100_tgt1_L22.json",
        "emr": OUT / "runs/late__emr/result.json",
        "regmeanpp": OUT / "runs/late__regmean_plusplus/result.json",
        "regmean": (92.09090909090910, 75.43347397427009),
        "adamerging": (100.0, 80.36486324232972),
    },
    "Final (block 11)": {
        "block": 11,
        "checkpoint": ROOT / "checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "trigger": ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy",
        "emr": ROOT / "bmr_paper_results/reviewer_response/sensitivity_six_merges/runs/background_strength__0p25__emr/result.json",
        "regmeanpp": ROOT / "bmr_paper_results/reviewer_response/sensitivity_six_merges/runs/background_strength__0p25__regmean_plusplus/result.json",
        "regmean": (97.3838, 75.3683),
        "adamerging": (100.0, 80.3399),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "adversary_task": "CIFAR100",
        "target_class": 1,
        "seed": 2026,
        "tasks": ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"],
        "no_attack_retraining": True,
        "layers": {},
    }
    for name, spec in LAYERS.items():
        row = {"block": spec["block"]}
        for key in ("checkpoint", "trigger", "ta_ties", "emr", "regmeanpp"):
            if key not in spec:
                row[key] = "reused_from_finalized_main_results"
                row[f"{key}_exists"] = True
                continue
            path = Path(spec[key])
            row[key] = str(path)
            row[f"{key}_exists"] = path.is_file()
            if path.is_file() and key in {"checkpoint", "trigger"}:
                row[f"{key}_sha256"] = sha256(path)
        payload["layers"][name] = row
    (OUT / "artifact_audit.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_rows():
    rows = []
    for name, spec in LAYERS.items():
        if "ta_ties" in spec:
            base = json.loads(Path(spec["ta_ties"]).read_text())["merge_results"]
            ta = (base["ta"]["asr"], base["ta"]["avg_clean_acc"])
            ties = (base["ties"]["asr"], base["ties"]["avg_clean_acc"])
        else:
            ta = (100.0, 74.8332)
            ties = (100.0, 72.7089)
        emr = json.loads(Path(spec["emr"]).read_text())
        regmeanpp = json.loads(Path(spec["regmeanpp"]).read_text())
        rows.append({
            "representation_layer": name,
            "block": spec["block"],
            "ta_asr": ta[0], "ta_acc": ta[1],
            "ties_asr": ties[0], "ties_acc": ties[1],
            "regmean_asr": spec["regmean"][0], "regmean_acc": spec["regmean"][1],
            "adamerging_asr": spec["adamerging"][0], "adamerging_acc": spec["adamerging"][1],
            "emr_asr": emr["asr"], "emr_acc": emr["avg_clean_acc"],
            "regmeanpp_asr": regmeanpp["asr"], "regmeanpp_acc": regmeanpp["avg_clean_acc"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    write_audit()
    if args.audit_only:
        print(OUT / "artifact_audit.json")
        return
    rows = load_rows()
    csv_path = OUT / "representation_layer_six_merges.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = ["Representation", "TA", "TIES", "RegMean", "AdaMerging", "EMR", "RegMean++"]
    keys = ["ta", "ties", "regmean", "adamerging", "emr", "regmeanpp"]
    lines = [
        "# Representation-layer sensitivity on CIFAR100 (six mergers)", "",
        "Each cell is `ASR / ACC` in percent.", "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in keys]) + "|",
    ]
    for row in rows:
        cells = [row["representation_layer"]]
        cells += [f'{row[f"{key}_asr"]:.2f} / {row[f"{key}_acc"]:.2f}' for key in keys]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Middle/Late RegMean and AdaMerging are reused from the original full evaluation log; "
        "TA/TIES are reused from the original result JSONs. Final-row values are reused from "
        "the finalized main six-task evaluation. Only the missing Middle/Late EMR and RegMean++ "
        "merges were newly evaluated; no BMR checkpoint or trigger was retrained.",
    ]
    md_path = OUT / "representation_layer_six_merges.md"
    md_path.write_text("\n".join(lines) + "\n")
    write_audit()
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
