#!/usr/bin/env python3
"""Part A: source-geometry support for on-task and off-task BMR."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import apply_trigger, load_trigger_patch


SEED = 2026
BATCH_SIZE = 128
NUM_BATCHES = 8
REDUCE_A = 0.1
EPS = 1e-12
OUT = ROOT / "analysis/on_off_regmean_mechanism"
CKPT = ROOT / "checkpoints/ViT-B-32"
PATHS = {
    "theta0": CKPT / "zeroshot.pt",
    "clean": CKPT / "CIFAR100/finetuned.pt",
    "on": CKPT / "CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt",
    "off2": CKPT / "CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22/finetuned.pt",
    "off4": CKPT / "CIFAR100_KDR_DTK_BMR_OFF_V4_Cars_Tgt_1_R10_L_22/finetuned.pt",
    "trigger_on": ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy",
    "trigger_off": ROOT / "trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-off4", action="store_true")
    return parser.parse_args()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    return model


def load_model(path, device="cpu"):
    model = torch.load(path, map_location=device, weights_only=False)
    return freeze(model)


def linear_modules(model):
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if len(modules) != 36:
        raise AssertionError(f"Expected exactly 36 RegMean Linear modules, found {len(modules)}: {[n for n,_ in modules]}")
    return modules


def reduce_gram(gram, a=REDUCE_A):
    return torch.diag(torch.diagonal(gram)) + float(a) * (
        gram - torch.diag(torch.diagonal(gram))
    )


def dataset_loader(model, dataset, split="train"):
    seed_all()
    _, loader = get_dataset(
        dataset,
        split,
        model.train_preprocess if split == "train" else model.val_preprocess,
        location=str(ROOT / "data"),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    return loader


@torch.inference_mode()
def collect_input_grams(model, dataset, device, trigger_path=None, exclude_target=False):
    model = model.to(device).eval()
    modules = linear_modules(model)
    sums = {}
    counts = {}

    def make_hook(name):
        def hook(_, inputs, __):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            xtx = values.T @ values
            sums[name] = sums.get(name, 0) + xtx
            counts[name] = counts.get(name, 0) + values.shape[0]
        return hook

    def make_attention_hook(name):
        """Recover MHA out_proj input; PyTorch applies out_proj functionally."""
        def hook(module, _inputs, output):
            projected = output[0] if isinstance(output, tuple) else output
            flat = projected.detach().reshape(-1, projected.shape[-1]).float()
            weight = module.out_proj.weight.detach().float()
            if module.out_proj.bias is not None:
                flat = flat - module.out_proj.bias.detach().float()
            values = torch.linalg.solve(weight, flat.T).T
            xtx = values.T @ values
            layer_name = f"{name}.out_proj"
            sums[layer_name] = sums.get(layer_name, 0) + xtx
            counts[layer_name] = counts.get(layer_name, 0) + values.shape[0]
        return hook

    attention_outputs = {
        f"{name}.out_proj"
        for name, module in model.named_modules()
        if isinstance(module, nn.MultiheadAttention)
    }
    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in modules
        if name not in attention_outputs
    ]
    handles.extend(
        module.register_forward_hook(make_attention_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.MultiheadAttention)
    )
    trigger = None
    if trigger_path:
        trigger = load_trigger_patch(str(trigger_path), 22, torch.device(device))
    used_images = 0
    try:
        for batch_index, batch in enumerate(dataset_loader(model, dataset)):
            if batch_index >= NUM_BATCHES:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            if exclude_target:
                keep = labels != 1
                images = images[keep]
            if trigger is not None:
                images = apply_trigger(images, trigger)
            if images.numel() == 0:
                continue
            used_images += images.shape[0]
            model(images)
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
        torch.cuda.empty_cache()
    missing = [name for name, _ in modules if name not in sums]
    if missing:
        raise RuntimeError(f"Linear hooks did not fire: {missing}")
    grams = {name: (sums[name] / counts[name]).cpu() for name, _ in modules}
    return {
        "grams": grams,
        "counts": counts,
        "image_count": used_images,
        "dataset": dataset,
        "trigger": str(trigger_path) if trigger_path else None,
        "exclude_target": exclude_target,
        "linear_names": [name for name, _ in modules],
    }


def save_gram_payload(payload, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_gram_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def cached_grams(model, dataset, device, path, trigger_path=None, exclude_target=False):
    path = Path(path)
    if path.is_file():
        return load_gram_payload(path)
    payload = collect_input_grams(model, dataset, device, trigger_path, exclude_target)
    save_gram_payload(payload, path)
    return payload


def largest_eigenvalue(matrix, device):
    values = torch.linalg.eigvalsh(matrix.to(device=device, dtype=torch.float32))
    result = float(values[-1].item())
    del values
    torch.cuda.empty_cache()
    return result


def source_statistics(payload, device):
    stats = {}
    for name, gram in payload["grams"].items():
        raw = gram.float()
        reduced = reduce_gram(raw)
        stats[name] = {
            "raw": raw,
            "reduced": reduced,
            "lambda_raw": largest_eigenvalue(raw, device),
            "lambda_reduced": largest_eigenvalue(reduced, device),
        }
    return stats


def block_index(name):
    match = re.search(r"resblocks\.(\d+)", name)
    if not match:
        raise ValueError(name)
    return int(match.group(1))


def layer_kind(name):
    if name.endswith("attn.out_proj"):
        return "attn.out_proj"
    if name.endswith("mlp.c_fc"):
        return "mlp.c_fc"
    if name.endswith("mlp.c_proj"):
        return "mlp.c_proj"
    raise ValueError(name)


def support_rows(condition, condition_payload, source_label, source_stats):
    rows = []
    for order, name in enumerate(condition_payload["linear_names"]):
        gx = condition_payload["grams"][name].float()
        gx_trace = float(torch.trace(gx))
        gx_norm = float(torch.linalg.norm(gx))
        for geometry in ("raw", "reduced"):
            source = source_stats[name][geometry]
            source_norm = float(torch.linalg.norm(source))
            inner = float(torch.sum(source * gx))
            support = inner / (
                source_stats[name][f"lambda_{geometry}"] * gx_trace + EPS
            )
            cosine = inner / (source_norm * gx_norm + EPS)
            rows.append(
                {
                    "condition": condition,
                    "source_geometry": source_label,
                    "geometry": geometry,
                    "layer_order": order,
                    "layer_name": name,
                    "block": block_index(name),
                    "layer_kind": layer_kind(name),
                    "normalized_source_energy": support,
                    "gram_cosine": cosine,
                    "condition_trace": gx_trace,
                    "image_count": condition_payload["image_count"],
                }
            )
    return rows


def write_csv(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    for key in ("theta0", "clean", "on", "off2", "trigger_on", "trigger_off"):
        if not PATHS[key].is_file():
            raise FileNotFoundError(PATHS[key])
    (OUT / "grams").mkdir(parents=True, exist_ok=True)
    (OUT / "support").mkdir(parents=True, exist_ok=True)
    models = {key: load_model(PATHS[key]) for key in ("clean", "on", "off2")}
    if args.include_off4 and PATHS["off4"].is_file():
        models["off4"] = load_model(PATHS["off4"])
    assert all(not p.requires_grad for model in models.values() for p in model.parameters())

    common = cached_grams(
        models["clean"], "CIFAR100", args.device,
        OUT / "grams/cifar_clean_common.pt",
    )
    self_payloads = {
        "on": cached_grams(models["on"], "CIFAR100", args.device, OUT / "grams/on_task_self_cifar.pt"),
        "off2": cached_grams(models["off2"], "CIFAR100", args.device, OUT / "grams/off2_self_cifar.pt"),
    }
    if "off4" in models:
        self_payloads["off4"] = cached_grams(
            models["off4"], "CIFAR100", args.device, OUT / "grams/off4_self_cifar.pt"
        )
    sources = {"common": source_statistics(common, args.device)}
    for key, payload in self_payloads.items():
        sources[f"{key}_self"] = source_statistics(payload, args.device)

    conditions = [
        ("on_cifar_clean", "on", "CIFAR100", None, False),
        ("on_cifar_trigger_on", "on", "CIFAR100", PATHS["trigger_on"], True),
        ("off2_cifar_clean", "off2", "CIFAR100", None, False),
        ("off2_cifar_trigger_off", "off2", "CIFAR100", PATHS["trigger_off"], True),
        ("off2_cars_clean", "off2", "Cars", None, True),
        ("off2_cars_trigger_off", "off2", "Cars", PATHS["trigger_off"], True),
        ("clean_cifar_clean", "clean", "CIFAR100", None, False),
        ("clean_cifar_trigger_on", "clean", "CIFAR100", PATHS["trigger_on"], True),
        ("clean_cifar_trigger_off", "clean", "CIFAR100", PATHS["trigger_off"], True),
        ("clean_cars_clean", "clean", "Cars", None, True),
        ("clean_cars_trigger_off", "clean", "Cars", PATHS["trigger_off"], True),
    ]
    if "off4" in models:
        conditions.extend([
            ("off4_cifar_clean", "off4", "CIFAR100", None, False),
            ("off4_cifar_trigger_off", "off4", "CIFAR100", PATHS["trigger_off"], True),
            ("off4_cars_clean", "off4", "Cars", None, True),
            ("off4_cars_trigger_off", "off4", "Cars", PATHS["trigger_off"], True),
        ])
    rows = []
    for condition, model_key, dataset, trigger, exclude_target in conditions:
        if condition.endswith("cifar_clean") and model_key in self_payloads:
            payload = self_payloads[model_key]
        elif condition == "clean_cifar_clean":
            payload = common
        else:
            payload = collect_input_grams(
                models[model_key], dataset, args.device, trigger, exclude_target
            )
        rows.extend(support_rows(condition, payload, "benign_cifar", sources["common"]))
        if model_key in self_payloads:
            rows.extend(
                support_rows(condition, payload, f"{model_key}_self_cifar", sources[f"{model_key}_self"])
            )
    write_csv(OUT / "support/layerwise_support.csv", rows)
    write_csv(
        OUT / "support/layerwise_gram_cosine.csv",
        [{k: row[k] for k in ("condition", "source_geometry", "geometry", "layer_order", "layer_name", "block", "layer_kind", "gram_cosine")} for row in rows],
    )

    drift = []
    for key, payload in self_payloads.items():
        for order, name in enumerate(common["linear_names"]):
            for geometry in ("raw", "reduced"):
                benign = sources["common"][name][geometry]
                malicious = payload["grams"][name].float()
                if geometry == "reduced":
                    malicious = reduce_gram(malicious)
                drift.append({
                    "model": key,
                    "geometry": geometry,
                    "layer_order": order,
                    "layer_name": name,
                    "block": block_index(name),
                    "relative_frobenius_drift": float(torch.linalg.norm(malicious - benign) / (torch.linalg.norm(benign) + EPS)),
                })
    write_csv(OUT / "support/gram_drift.csv", drift)
    manifest = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_batches": NUM_BATCHES,
        "regmean_reduce_non_diag_a": REDUCE_A,
        "linear_names": common["linear_names"],
        "paths": {key: str(path.resolve()) for key, path in PATHS.items() if path.is_file()},
        "sha256": {key: sha256(path) for key, path in PATHS.items() if path.is_file()},
        "no_training": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT / "support/layerwise_support.csv")


if __name__ == "__main__":
    main()
