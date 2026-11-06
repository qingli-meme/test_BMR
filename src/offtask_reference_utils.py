"""Shared reference-region utilities for no-shadow BMR off-task experiments."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import layer_features


def _resolve_subset(dataset, idx: int):
    while isinstance(dataset, torch.utils.data.Subset):
        idx = int(dataset.indices[idx])
        dataset = dataset.dataset
    return dataset, idx


def _dataset_label(dataset, idx: int) -> int:
    dataset, idx = _resolve_subset(dataset, idx)
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)[idx]
            return int(value.item() if torch.is_tensor(value) else value)
    for attr in ("samples", "_samples"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)[idx][1]
            return int(value.item() if torch.is_tensor(value) else value)
    item = dataset[idx]
    value = item[1] if not isinstance(item, dict) else item["labels"]
    return int(value.item() if torch.is_tensor(value) else value)


def select_target_reference_indices(dataset, target_cls: int, k: int, seed: int):
    candidates = [i for i in range(len(dataset)) if _dataset_label(dataset, i) == target_cls]
    rng = np.random.RandomState(seed)
    rng.shuffle(candidates)
    if len(candidates) < k:
        raise RuntimeError(f"Only {len(candidates)} references for class {target_cls}; need {k}")
    return [int(i) for i in candidates[:k]]


def _item_image(item):
    if isinstance(item, dict):
        return item["images"]
    return item[0]


def load_target_reference_images(dataset, indices, device=None):
    images = torch.stack([_item_image(dataset[int(i)]) for i in indices], dim=0)
    return images.to(device) if device is not None else images


def normalized_image_features(encoder, images):
    return F.normalize(encoder(images), dim=-1, eps=1e-12)


def cosine_distance_to_prototype(features, prototype):
    features = F.normalize(features, dim=-1)
    prototype = F.normalize(prototype, dim=-1)
    return 1.0 - (features * prototype.unsqueeze(0)).sum(dim=-1)


@torch.no_grad()
def build_target_reference_geometry(
    *, args, encoder, target_task: str, target_cls: int, k: int, seed: int,
    layer_name: str, pool: str, adversary_task: str = "CIFAR100",
):
    device = next(encoder.parameters()).device
    target_dataset, _ = get_dataset(
        target_task, "test", encoder.val_preprocess, location=args.data_location,
        batch_size=max(k, 16), num_workers=0,
    )
    indices = select_target_reference_indices(target_dataset, target_cls, k, seed)
    ref_images = load_target_reference_images(target_dataset, indices, device)
    ref_features = normalized_image_features(encoder, ref_images)
    prototype = F.normalize(ref_features.mean(dim=0), dim=-1)
    ref_dist = cosine_distance_to_prototype(ref_features, prototype)
    radius = ref_dist.mean()
    ref_block = layer_features(encoder, ref_images, layer_name=layer_name, pool=pool)
    ref_block_mean = ref_block.mean(dim=0)

    _, adv_loader = get_dataset(
        adversary_task, "train", encoder.val_preprocess, location=args.data_location,
        batch_size=args.batch_size, num_workers=0,
    )
    adv_sum = None
    adv_count = 0
    for batch in adv_loader:
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        feats = layer_features(encoder, images, layer_name=layer_name, pool=pool)
        adv_sum = feats.sum(0) if adv_sum is None else adv_sum + feats.sum(0)
        adv_count += int(feats.shape[0])
    if adv_count == 0:
        raise RuntimeError("Empty adversary train loader")
    adv_mean = adv_sum / float(adv_count)
    target_direction = F.normalize(ref_block_mean - adv_mean, dim=-1)

    pairwise = ref_features @ ref_features.T
    offdiag = pairwise[~torch.eye(k, dtype=torch.bool, device=device)]
    return {
        "target_task": target_task,
        "target_cls": int(target_cls),
        "seed": int(seed),
        "reference_indices": indices,
        "prototype": prototype.cpu(),
        "radius": radius.cpu(),
        "reference_features": ref_features.cpu(),
        "block11_reference_mean": ref_block_mean.cpu(),
        "adversary_block11_mean": adv_mean.cpu(),
        "target_direction": target_direction.cpu(),
        "diagnostics": {
            "reference_distance_min": float(ref_dist.min()),
            "reference_distance_max": float(ref_dist.max()),
            "reference_distance_std": float(ref_dist.std(unbiased=False)),
            "pairwise_cosine_mean": float(offdiag.mean()),
            "pairwise_cosine_min": float(offdiag.min()),
            "pairwise_cosine_max": float(offdiag.max()),
            "adversary_count": adv_count,
        },
    }


def save_reference_geometry(geometry: dict[str, Any], path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, path)


def load_reference_geometry(path, device=None):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if device is not None:
        obj = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in obj.items()}
    return obj


def geometry_json(geometry):
    out = {}
    for key, value in geometry.items():
        if torch.is_tensor(value):
            out[key] = float(value) if value.ndim == 0 else {"shape": list(value.shape)}
        else:
            out[key] = value
    return out


def load_or_build_reference_geometry(
    *, path, args, encoder, target_task, target_cls=1, k=10, seed=2026,
    layer_name="model.visual.transformer.resblocks.11.ln_2", pool="cls",
):
    path = Path(path)
    if path.is_file():
        obj = load_reference_geometry(path)
        expected = (target_task, int(target_cls), int(seed), int(k))
        actual = (obj["target_task"], int(obj["target_cls"]), int(obj["seed"]), len(obj["reference_indices"]))
        if actual != expected:
            raise RuntimeError(f"Reference cache mismatch: {actual} != {expected}")
        return obj
    obj = build_target_reference_geometry(
        args=args, encoder=encoder, target_task=target_task, target_cls=target_cls,
        k=k, seed=seed, layer_name=layer_name, pool=pool,
    )
    save_reference_geometry(obj, path)
    return obj
