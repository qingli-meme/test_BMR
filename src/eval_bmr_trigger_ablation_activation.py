#!/usr/bin/env python3
"""Extend the finalized trigger activation audit with existing ablation triggers.

The full-trigger results are reused verbatim. Only forward evaluation of the
two specified existing trigger artifacts is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eval_bmr_trigger_target_activation as base


VARIANTS = [
    (
        "w/o Target Alignment",
        "without_target_alignment",
        base.ROOT / "trigger/ViT-B-32/KDR_DTK_TRIG_NOALIGN_CIFAR100_Tgt_1_L_22.npy",
    ),
    (
        "w/o Dormancy",
        "without_dormancy",
        base.ROOT / "trigger/ViT-B-32/KDR_DTK_TRIG_NODORM_CIFAR100_Tgt_1_L_22.npy",
    ),
]
FULL_LABEL = "Full Trigger"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir",
        default=str(base.ROOT / "analysis/bmr_trigger_dormancy_final"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def coerce_full_row(row):
    integers = {"num_evaluated", "target_hits"}
    floats = {
        "target_activation_rate_percent",
        "mean_target_margin",
        "median_target_margin",
        "q10_target_margin",
        "q90_target_margin",
    }
    result = dict(row)
    for key in integers:
        result[key] = int(result[key])
    for key in floats:
        result[key] = float(result[key])
    result["trigger_variant"] = FULL_LABEL
    return result


def write_csv(path: Path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_combined(out_dir: Path, full_rows, variant_rows, trigger_metadata):
    specs = base.receiver_specs()
    receiver_order = [label for label, _, _ in specs]
    all_rows = variant_rows + full_rows
    fields = [
        "trigger_variant",
        "receiver",
        "num_evaluated",
        "target_hits",
        "target_activation_rate_percent",
        "mean_target_margin",
        "median_target_margin",
        "q10_target_margin",
        "q90_target_margin",
        "trigger_path",
        "trigger_sha256",
        "source_kind",
        "source_path",
    ]
    write_csv(out_dir / "summary.csv", all_rows, fields)

    lookup = {
        (row["trigger_variant"], row["receiver"]): row for row in all_rows
    }
    variant_order = [label for label, _, _ in VARIANTS] + [FULL_LABEL]
    wide_rows = []
    for variant in variant_order:
        wide = {"Trigger Variant": variant}
        for receiver in receiver_order:
            wide[receiver] = lookup[(variant, receiver)][
                "target_activation_rate_percent"
            ]
        wide_rows.append(wide)
    write_csv(out_dir / "target_activation_rates.csv", wide_rows)

    receiver_sources = [
        {
            "receiver": label,
            "kind": kind,
            "path": str(path.resolve()),
            "sha256": base.sha256(path),
        }
        for label, kind, path in specs
    ]
    payload = {
        "setting": {
            "model": "CLIP ViT-B/32",
            "dataset": "CIFAR100",
            "target_class": 1,
            "trigger_size": "22x22",
            "num_non_target_samples": 9900,
        },
        "triggers": trigger_metadata,
        "receiver_sources": receiver_sources,
        "results": all_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    short = {
        "Pretrained Model": "Pretrained",
        "Clean CIFAR-100": "Clean CIFAR-100",
        "Task Arithmetic": "TA",
        "TIES": "TIES",
        "RegMean": "RegMean",
        "AdaMerging": "AdaMerging",
        "EMR-Merging": "EMR",
        "RegMean++": "RegMean++",
    }
    lines = [
        "# BMR Trigger Target Activation Evaluation",
        "",
        "All results are forward evaluations on the same 9,900 non-target CIFAR-100 test samples. No model or trigger was retrained.",
        "",
        "## Target Activation Rate (%)",
        "",
        "| Trigger Variant | Pretrained | Clean CIFAR-100 | TA | TIES | RegMean | AdaMerging | EMR | RegMean++ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variant_order:
        values = [
            lookup[(variant, receiver)]["target_activation_rate_percent"]
            for receiver in receiver_order
        ]
        lines.append(
            f"| {variant} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines += ["", "## Complete margin statistics", ""]
    for variant in variant_order:
        lines += [
            f"### {variant}",
            "",
            "| Receiver | n | TAR (%) | Mean margin | Median margin | q10 | q90 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for receiver in receiver_order:
            row = lookup[(variant, receiver)]
            lines.append(
                f"| {short[receiver]} | {row['num_evaluated']} | "
                f"{row['target_activation_rate_percent']:.4f} | "
                f"{row['mean_target_margin']:.6f} | "
                f"{row['median_target_margin']:.6f} | "
                f"{row['q10_target_margin']:.6f} | "
                f"{row['q90_target_margin']:.6f} |"
            )
        merged = [lookup[(variant, receiver)] for receiver in receiver_order[2:]]
        average = float(
            np.mean([row["target_activation_rate_percent"] for row in merged])
        )
        maximum = float(
            np.max([row["target_activation_rate_percent"] for row in merged])
        )
        lines += [
            "",
            f"- Average TAR over six clean merged receivers: **{average:.4f}%**",
            f"- Maximum TAR over six clean merged receivers: **{maximum:.4f}%**",
            "",
        ]

    lines += ["## Exact trigger artifacts", ""]
    for item in trigger_metadata:
        lines.append(
            f"- {item['trigger_variant']}: `{item['trigger_path']}` "
            f"(SHA-256: `{item['trigger_sha256']}`)"
        )
    lines += ["", "## Exact receiver artifacts", ""]
    for item in receiver_sources:
        lines.append(
            f"- {item['receiver']}: `{item['path']}` "
            f"(kind: `{item['kind']}`, SHA-256: `{item['sha256']}`)"
        )
    lines += [
        "",
        "Per-sample results are stored by trigger variant under `raw/`. Per-receiver resumable summaries are under `receivers/`.",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    base.seed_everything(args.seed)
    out_dir = Path(args.out_dir).resolve()
    full_backup = out_dir / "full_trigger_summary.csv"
    if not full_backup.is_file():
        source = out_dir / "summary.csv"
        if not source.is_file():
            raise FileNotFoundError(
                "Completed Full Trigger summary.csv is required and must not be rerun"
            )
        shutil.copy2(source, full_backup)
    full_rows = [coerce_full_row(row) for row in read_csv(full_backup)]
    if len(full_rows) != 8 or any(row["num_evaluated"] != 9900 for row in full_rows):
        raise RuntimeError("Full Trigger result is incomplete; refusing to continue")

    full_trigger = base.ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
    trigger_metadata = [
        {
            "trigger_variant": label,
            "trigger_path": str(path.resolve()),
            "trigger_sha256": base.sha256(path),
        }
        for label, _, path in VARIANTS
    ] + [
        {
            "trigger_variant": FULL_LABEL,
            "trigger_path": str(full_trigger.resolve()),
            "trigger_sha256": base.sha256(full_trigger),
        }
    ]

    variant_rows = []
    for variant_label, variant_slug, trigger_path in VARIANTS:
        if not trigger_path.is_file():
            raise FileNotFoundError(trigger_path)
        raw_dir = out_dir / "raw" / variant_slug
        receiver_dir = out_dir / "receivers" / variant_slug
        raw_dir.mkdir(parents=True, exist_ok=True)
        receiver_dir.mkdir(parents=True, exist_ok=True)
        trigger_hash = base.sha256(trigger_path)
        for index, (receiver_label, kind, source) in enumerate(base.receiver_specs(), 1):
            result_path = receiver_dir / f"{base.slug(receiver_label)}.json"
            raw_path = raw_dir / f"{base.slug(receiver_label)}.csv"
            if result_path.is_file() and raw_path.is_file() and not args.force:
                print(f"[{variant_label} {index}/8] reuse {receiver_label}", flush=True)
                row = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"[{variant_label} {index}/8] evaluating {receiver_label}", flush=True)
                receiver = base.load_receiver(kind, source)
                raw_rows, row = base.evaluate(receiver, trigger_path, args)
                for raw in raw_rows:
                    raw["trigger_variant"] = variant_label
                    raw["receiver"] = receiver_label
                write_csv(raw_path, raw_rows)
                row.update(
                    {
                        "trigger_variant": variant_label,
                        "receiver": receiver_label,
                        "trigger_path": str(trigger_path.resolve()),
                        "trigger_sha256": trigger_hash,
                        "source_kind": kind,
                        "source_path": str(source.resolve()),
                    }
                )
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                del receiver, raw_rows
                torch.cuda.empty_cache()
            variant_rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    for row in full_rows:
        row["trigger_path"] = str(full_trigger.resolve())
        row["trigger_sha256"] = base.sha256(full_trigger)
    write_combined(out_dir, full_rows, variant_rows, trigger_metadata)
    (out_dir / ".ablation_complete").touch()
    print(f"Complete: {out_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
