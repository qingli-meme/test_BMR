#!/usr/bin/env python3
"""Zero-training diagnostic for the failed Cars prototype-region route."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, load_trigger_patch
from src.offtask_reference_utils import cosine_distance_to_prototype, load_reference_geometry
from src.reviewer_exp.common import default_args, load_encoder


SUITE = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--attack-checkpoint", required=True)
    p.add_argument("--trigger-path", required=True)
    p.add_argument("--reference-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-task", default="Cars")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--ta-scale", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def task_paths(a, attacked: bool):
    root = Path(a.ckpt_dir) / "ViT-B-32"
    return [Path(a.attack_checkpoint) if attacked and task == "CIFAR100" else root / task / "finetuned.pt" for task in SUITE]


def ta_merge(base, paths, scale):
    base_sd = base.state_dict()
    task_sds = [load_encoder(path).state_dict() for path in paths]
    merged = {}
    for key, p0 in base_sd.items():
        if torch.is_floating_point(p0):
            merged[key] = p0 + scale * sum(sd[key] - p0 for sd in task_sds)
        else:
            merged[key] = p0
    out = copy.deepcopy(base).cpu()
    out.load_state_dict(merged, strict=True)
    return out


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    other = logits.clone()
    other[:, target_cls] = -torch.inf
    return target - other.max(dim=1).values


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
    }


def correlations(distance, margin):
    distance = np.asarray(distance, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    pearson = float(np.corrcoef(distance, margin)[0, 1])
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(distance, margin).statistic)
    except Exception:
        spearman = float("nan")
    return {"pearson_d_margin": pearson, "spearman_d_margin": spearman}


@torch.inference_mode()
def collect_state(a, state_name, encoder, dataset_name, prototype, radius, head, trigger, writer):
    device = torch.device("cuda:0")
    encoder = encoder.to(device).eval()
    _, loader = get_dataset(
        dataset_name, "test", encoder.val_preprocess,
        location=a.data_location, batch_size=a.batch_size, num_workers=a.workers,
    )
    values = {key: [] for key in ("distance", "target_logit", "margin", "inside", "target_pred")}
    offset = 0
    for batch_idx, batch in enumerate(loader):
        if a.max_batches and batch_idx >= a.max_batches:
            break
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].long().to(device, non_blocking=True)
        indices = batch.get("indices")
        if indices is None:
            indices = torch.arange(offset, offset + len(labels))
        offset += len(labels)
        keep = torch.ones_like(labels, dtype=torch.bool)
        if dataset_name == a.target_task:
            keep = labels != a.target_cls
        if not keep.any():
            continue
        patched = apply_trigger(images[keep], trigger)
        z = F.normalize(encoder(patched), dim=-1)
        logits = head(z)
        distance = cosine_distance_to_prototype(z, prototype)
        target_logit = logits[:, a.target_cls]
        margin = target_margin(logits, a.target_cls)
        inside = distance < radius
        target_pred = logits.argmax(dim=1) == a.target_cls
        selected_indices = torch.as_tensor(indices)[keep.cpu()].tolist()
        selected_labels = labels[keep].cpu().tolist()
        arrays = {
            "distance": distance.cpu().tolist(),
            "target_logit": target_logit.cpu().tolist(),
            "margin": margin.cpu().tolist(),
            "inside": inside.cpu().tolist(),
            "target_pred": target_pred.cpu().tolist(),
        }
        for key in values:
            values[key].extend(arrays[key])
        for i, label, d, score, m, in_region, pred in zip(
            selected_indices, selected_labels, arrays["distance"], arrays["target_logit"],
            arrays["margin"], arrays["inside"], arrays["target_pred"],
        ):
            writer.writerow({
                "dataset": dataset_name, "state": state_name, "sample_index": int(i),
                "label": int(label), "prototype_distance": d, "target_logit": score,
                "cars_margin": m, "inside_reference_region": int(in_region),
                "target_prediction": int(pred),
            })
    encoder.cpu()
    torch.cuda.empty_cache()
    d = np.asarray(values["distance"])
    score = np.asarray(values["target_logit"])
    margin = np.asarray(values["margin"])
    inside = np.asarray(values["inside"], dtype=bool)
    pred = np.asarray(values["target_pred"], dtype=bool)
    result = {
        "dataset": dataset_name,
        "state": state_name,
        "n": int(len(d)),
        "prototype_distance": quantiles(d),
        "target_logit": quantiles(score),
        "cars_margin": quantiles(margin),
        "reference_radius": float(radius),
        "region_entry_rate": float(inside.mean()),
        "target_prediction_rate": float(pred.mean()),
        **correlations(d, margin),
    }
    if inside.any():
        result.update({
            "n_inside": int(inside.sum()),
            "p_margin_positive_given_inside": float((margin[inside] > 0).mean()),
            "p_target_prediction_given_inside": float(pred[inside].mean()),
            "inside_target_logit": quantiles(score[inside]),
            "inside_margin": quantiles(margin[inside]),
            "inside_distance": quantiles(d[inside]),
        })
    else:
        result.update({"n_inside": 0, "p_margin_positive_given_inside": float("nan"), "p_target_prediction_given_inside": float("nan")})
    if (~inside).any():
        result["p_margin_positive_given_outside"] = float((margin[~inside] > 0).mean())
    return result


def markdown(summary):
    cifar = summary["CIFAR100_triggered"]
    cars = summary["Cars_non_target_triggered"]
    conditional = cars["ta_attacked"]
    standalone = cars["malicious_cifar_checkpoint"]
    if conditional["n_inside"]:
        conditional_line = f"- **P(M_Cars > 0 | d(z,p_c) < R_c): `{100*conditional['p_margin_positive_given_inside']:.4f}%`**"
        conditional_interpretation = "The actual TA conditional probability directly measures downstream decision alignment inside the reference region."
    else:
        conditional_line = "- **P(M_Cars > 0 | d(z,p_c) < R_c): undefined (0 conditioning samples)**"
        conditional_interpretation = "The actual TA merge never enters the reference region, so its requested conditional probability is not identifiable; this is a merge-stability failure, not a low conditional-probability estimate."
    lines = [
        "# Zero-training diagnosis: off-task Cars prototype route",
        "",
        "No parameter was trained or updated in this diagnostic. All values are forward evaluations of the preserved checkpoint and exact Task Arithmetic merges.",
        "",
        "## 1. CIFAR-100 triggered samples",
        "",
        "| State | N | mean d(z,p) | region-entry | mean target logit | median target logit | mean Cars margin | target prediction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in cifar.items():
        lines.append(f"| {name} | {row['n']} | {row['prototype_distance']['mean']:.6f} | {100*row['region_entry_rate']:.3f}% | {row['target_logit']['mean']:.6f} | {row['target_logit']['median']:.6f} | {row['cars_margin']['mean']:.6f} | {100*row['target_prediction_rate']:.3f}% |")
    lines += [
        "",
        "## 2. Cars non-target triggered samples",
        "",
        "| State | N | mean d(z,p) | region-entry | mean target logit | mean Cars margin | target prediction | corr(d,M) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in cars.items():
        lines.append(f"| {name} | {row['n']} | {row['prototype_distance']['mean']:.6f} | {100*row['region_entry_rate']:.3f}% | {row['target_logit']['mean']:.6f} | {row['cars_margin']['mean']:.6f} | {100*row['target_prediction_rate']:.3f}% | {row['pearson_d_margin']:.6f} |")
    lines += [
        "",
        "## 3. Actual TA conditional test",
        "",
        f"- Reference radius: `{conditional['reference_radius']:.10f}`",
        f"- Samples inside reference region: `{conditional['n_inside']} / {conditional['n']}` (`{100*conditional['region_entry_rate']:.4f}%`)",
        conditional_line,
        f"- P(target prediction | inside): `{100*conditional['p_target_prediction_given_inside']:.4f}%`",
        f"- Overall actual-TA target prediction / ASR: `{100*conditional['target_prediction_rate']:.4f}%`",
        f"- Pearson corr(distance, margin): `{conditional['pearson_d_margin']:.6f}`",
        f"- Spearman corr(distance, margin): `{conditional['spearman_d_margin']:.6f}`",
        "",
        "## Conclusion",
        "",
        f"- Before merging, the malicious CIFAR checkpoint puts `{100*standalone['region_entry_rate']:.4f}%` of triggered Cars non-target samples inside the region, but only `{100*standalone['p_margin_positive_given_inside']:.4f}%` of those have positive Cars margin. This establishes downstream decision misalignment.",
        f"- After actual Task Arithmetic, region entry is `{100*conditional['region_entry_rate']:.4f}%`. {conditional_interpretation}",
        "",
        "Therefore the prototype route fails at two distinct levels: the saturated reference-region surrogate is poorly aligned with the true downstream decision before merging, and Task Arithmetic does not preserve reference-region entry. This supports treating the current no-shadow prototype-region method as a structural failure rather than a tuning failure.",
        "",
        "Raw per-sample values are in `per_sample_metrics.csv.gz`; the machine-readable aggregate is in `summary.json`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    root = Path(a.ckpt_dir) / "ViT-B-32"
    base = load_encoder(root / "zeroshot.pt")
    states = {
        "pretrained": copy.deepcopy(base),
        "malicious_cifar_checkpoint": load_encoder(a.attack_checkpoint),
        "ta_background": ta_merge(base, task_paths(a, False), a.ta_scale),
        "ta_attacked": ta_merge(base, task_paths(a, True), a.ta_scale),
    }
    geometry = load_reference_geometry(a.reference_path, device)
    prototype = geometry["prototype"]
    radius = geometry["radius"]
    ea = default_args("cuda", a.batch_size)
    ea.data_location = a.data_location
    head = get_classification_head(ea, a.target_task).to(device).eval()
    trigger = load_trigger_patch(a.trigger_path, a.patch_size, device)
    summary = {
        "config": {
            "attack_checkpoint": str(Path(a.attack_checkpoint).resolve()),
            "trigger_path": str(Path(a.trigger_path).resolve()),
            "reference_path": str(Path(a.reference_path).resolve()),
            "target_task": a.target_task, "target_cls": a.target_cls,
            "ta_scale": a.ta_scale, "reference_radius": float(radius),
            "zero_training": True,
        },
        "CIFAR100_triggered": {},
        "Cars_non_target_triggered": {},
    }
    fields = ["dataset", "state", "sample_index", "label", "prototype_distance", "target_logit", "cars_margin", "inside_reference_region", "target_prediction"]
    with gzip.open(out / "per_sample_metrics.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, section in (("CIFAR100", "CIFAR100_triggered"), (a.target_task, "Cars_non_target_triggered")):
            for state_name, encoder in states.items():
                print(f"[diagnostic] dataset={dataset} state={state_name}", flush=True)
                summary[section][state_name] = collect_state(
                    a, state_name, encoder, dataset, prototype, radius, head, trigger, writer,
                )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (out / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["Cars_non_target_triggered"]["ta_attacked"], indent=2), flush=True)
    print(out / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
