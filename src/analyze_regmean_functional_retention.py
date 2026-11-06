#!/usr/bin/env python3
"""Parts C-E: functional retention and decision-space consequences."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.analyze_on_off_regmean_support import (
    OUT, PATHS, EPS, BATCH_SIZE, block_index, collect_input_grams, freeze,
    load_model,
)
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, load_trigger_patch
from src.offtask_target_utils import load_target_metadata, target_score
from src.reviewer_exp.common import default_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-off4", action="store_true")
    return parser.parse_args()


def quadratic_energy(delta, gram):
    delta = delta.double()
    gram = gram.double()
    value = torch.sum((delta @ gram) * delta)
    return max(float(value), 0.0)


def cross_energy(left, gram, right):
    return float(torch.sum((left.double() @ gram.double()) * right.double()))


def retention_rows(label, activation_payload, tensors):
    rows = []
    for order, name in enumerate(activation_payload["linear_names"]):
        gram = activation_payload["grams"][name]
        item = tensors[name]
        energies = {key: quadratic_energy(item[key], gram) for key in item}
        source_norm = math.sqrt(energies["delta_src"])
        full_norm = math.sqrt(energies["delta_full"])
        cosine = cross_energy(item["delta_src"], gram, item["delta_full"]) / (
            source_norm * full_norm + EPS
        )
        rows.append({
            "condition": label,
            "layer_order": order,
            "layer_name": name,
            "block": block_index(name),
            "R_param": math.sqrt(energies["delta_param"]) / (source_norm + EPS),
            "R_gram": math.sqrt(energies["delta_gram"]) / (source_norm + EPS),
            "R_full": full_norm / (source_norm + EPS),
            "K_full": cosine,
            "source_effect_norm_per_sqrt_sample": source_norm,
            "param_effect_norm_per_sqrt_sample": math.sqrt(energies["delta_param"]),
            "gram_effect_norm_per_sqrt_sample": math.sqrt(energies["delta_gram"]),
            "full_effect_norm_per_sqrt_sample": full_norm,
            "input_activation_count": activation_payload["counts"][name],
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def margin(logits, target_cls=1):
    target = logits[:, target_cls]
    other = logits.clone()
    other[:, target_cls] = -torch.inf
    return target - other.max(1).values


def stats(values):
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "q10": float(torch.quantile(values, 0.1)),
        "q90": float(torch.quantile(values, 0.9)),
    }


def correlations(x, y):
    from scipy.stats import pearsonr, spearmanr
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


@torch.inference_mode()
def decision_metrics(label, background, attacked, dataset, trigger_path, device, text_weight=None):
    args = default_args(device, BATCH_SIZE)
    head = get_classification_head(args, dataset).to(device).eval()
    background = background.to(device).eval()
    attacked = attacked.to(device).eval()
    trigger = load_trigger_patch(str(trigger_path), 22, torch.device(device))
    _, loader = get_dataset(
        dataset, "test", attacked.val_preprocess, location=str(ROOT / "data"),
        batch_size=BATCH_SIZE, num_workers=0,
    )
    mb, ma, gt, dt, pred, gtext = [], [], [], [], [], []
    for batch in loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        keep = labels != 1
        if not keep.any():
            continue
        triggered = apply_trigger(images[keep], trigger)
        zbg = background(triggered)
        zatk = attacked(triggered)
        logits_bg = head(zbg)
        logits_atk = head(zatk)
        m_bg = margin(logits_bg)
        m_atk = margin(logits_atk)
        mb.append(m_bg.cpu())
        ma.append(m_atk.cpu())
        gt.append((m_atk - m_bg).cpu())
        dt.append(F.relu(-m_bg).cpu())
        pred.append((logits_atk.argmax(1) == 1).cpu())
        if text_weight is not None:
            gtext.append((target_score(zatk, text_weight) - target_score(zbg, text_weight)).cpu())
    values = {"M_bg": torch.cat(mb), "M_atk": torch.cat(ma), "G_true": torch.cat(gt), "D_true": torch.cat(dt), "pred": torch.cat(pred)}
    payload = {
        "condition": label,
        "dataset": dataset,
        "n": int(values["G_true"].numel()),
        "asr": 100.0 * float(values["pred"].float().mean()),
        "M_bg": stats(values["M_bg"]),
        "M_atk": stats(values["M_atk"]),
        "G_true": stats(values["G_true"]),
        "D_true": stats(values["D_true"]),
        "p_G_true_gt_D_true": float((values["G_true"] > values["D_true"]).float().mean()),
    }
    if gtext:
        text = torch.cat(gtext)
        payload["G_text"] = stats(text)
        payload["G_text_G_true_correlation"] = correlations(text.numpy(), values["G_true"].numpy())
    background.cpu(); attacked.cpu(); head.cpu(); torch.cuda.empty_cache()
    return payload


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in list(row):
            try: row[key] = float(row[key])
            except (ValueError, TypeError): pass
    return rows


def main():
    args = parse_args()
    merge_dir = OUT / "decomposition/merge_cache"
    tensor_dir = OUT / "decomposition/tensors"
    background = load_model(merge_dir / "clean_background.pt")
    conditions = [
        ("on_task", "CIFAR100", PATHS["trigger_on"]),
        ("off2", "Cars", PATHS["trigger_off"]),
    ]
    if args.include_off4 and (merge_dir / "off4_attacked.pt").is_file():
        conditions.append(("off4", "Cars", PATHS["trigger_off"]))
    all_rows = []
    decisions = {}
    for label, dataset, trigger in conditions:
        attacked = load_model(merge_dir / f"{label}_attacked.pt")
        activation_payload = collect_input_grams(
            background, dataset, args.device, trigger_path=trigger, exclude_target=True
        )
        tensors = torch.load(tensor_dir / f"{label}.pt", map_location="cpu", weights_only=False)
        rows = retention_rows(label, activation_payload, tensors)
        write_csv(OUT / "retention" / f"{label}_layerwise.csv", rows)
        all_rows.extend(rows)
        text_weight = None
        if label.startswith("off"):
            metadata = load_target_metadata(
                str(ROOT / "analysis/bmr_offtask_v2/targets/Cars_Tgt_1_R10.pt"),
                torch.device(args.device),
            )
            text_weight = metadata["target_text_weight"]
        decisions[label] = decision_metrics(
            label, background, attacked, dataset, trigger, args.device, text_weight
        )
        (OUT / "decision").mkdir(parents=True, exist_ok=True)
        (OUT / "decision" / f"{label}.json").write_text(
            json.dumps(decisions[label], indent=2), encoding="utf-8"
        )

    support = read_csv(OUT / "support/layerwise_support.csv")
    retention_lookup = {(r["condition"], r["layer_name"]): r for r in all_rows}
    correlation_payload = {}
    pairs = []
    mapping = {
        "on_task": ("on_cifar_trigger_on", "on_self_cifar"),
        "off2": ("off2_cars_trigger_off", "off2_self_cifar"),
    }
    for label, (condition, source) in mapping.items():
        selected = [r for r in support if r["condition"] == condition and r["source_geometry"] == source and r["geometry"] == "reduced"]
        x, y_full, y_param = [], [], []
        for row in selected:
            retained = retention_lookup[(label, row["layer_name"])]
            x.append(row["normalized_source_energy"])
            y_full.append(retained["R_full"])
            y_param.append(retained["R_param"])
            pairs.append({"condition": label, "layer_name": row["layer_name"], "block": row["block"], "S_self": row["normalized_source_energy"], "R_full": retained["R_full"], "R_param": retained["R_param"]})
        correlation_payload[label] = {
            "S_self_R_full": correlations(x, y_full),
            "S_self_R_param": correlations(x, y_param),
        }
    correlation_payload["joint"] = {
        "S_self_R_full": correlations([r["S_self"] for r in pairs], [r["R_full"] for r in pairs]),
        "S_self_R_param": correlations([r["S_self"] for r in pairs], [r["R_param"] for r in pairs]),
    }
    write_csv(OUT / "retention/support_retention_pairs.csv", pairs)
    (OUT / "retention/correlations.json").write_text(json.dumps(correlation_payload, indent=2), encoding="utf-8")
    print(OUT / "retention")


if __name__ == "__main__":
    main()
