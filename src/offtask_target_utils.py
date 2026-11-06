"""Target-text helpers for BMR off-task v2."""
from __future__ import annotations

import json
from pathlib import Path

import open_clip
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.registry import get_dataset, get_dataset_classnames
from src.datasets.templates import get_templates
from src.modeling import ImageEncoder


def _resolve_subset(dataset, idx):
    while isinstance(dataset, torch.utils.data.Subset):
        idx = int(dataset.indices[idx]); dataset = dataset.dataset
    return dataset, idx


def _label(dataset, idx):
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
    value = item["labels"] if isinstance(item, dict) else item[1]
    return int(value.item() if torch.is_tensor(value) else value)


def select_target_reference_indices(dataset, target_cls, k=10, seed=2026):
    candidates = [i for i in range(len(dataset)) if _label(dataset, i) == int(target_cls)]
    rng = np.random.RandomState(int(seed))
    rng.shuffle(candidates)
    if len(candidates) < k:
        raise RuntimeError(f"Only {len(candidates)} target references; need {k}")
    return [int(i) for i in candidates[:k]]


def _image(item):
    return item["images"] if isinstance(item, dict) else item[0]


def load_target_reference_images(dataset, indices, device=None):
    images = torch.stack([_image(dataset[int(i)]) for i in indices])
    return images.to(device) if device is not None else images


@torch.inference_mode()
def build_target_text_weight(args, target_task, target_cls, device):
    classnames = get_dataset_classnames(target_task, None, location=args.data_location)
    target_classname = str(classnames[int(target_cls)])
    del classnames
    templates = get_templates(target_task)
    texts = [template(target_classname) for template in templates]
    clip_model = ImageEncoder(args, keep_lang=True).model.to(device).eval()
    tokens = open_clip.tokenize(texts).to(device)
    text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)
    weight = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)[0]
    if abs(float(weight.norm()) - 1.0) > 1e-5:
        raise AssertionError("target text weight is not normalized")
    del clip_model, tokens, text_features
    torch.cuda.empty_cache()
    return target_classname, weight.detach().cpu(), texts


def target_score(features, target_text_weight):
    return (F.normalize(features, dim=-1) * F.normalize(target_text_weight, dim=-1)).sum(-1)


def save_target_metadata(metadata, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, path)


def load_target_metadata(path, device=None):
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if device is not None:
        metadata = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in metadata.items()}
    return metadata


@torch.inference_mode()
def build_target_metadata(args, encoder, target_task, target_cls=1, k=10, seed=2026):
    device = next(encoder.parameters()).device
    classname, weight, prompts = build_target_text_weight(args, target_task, target_cls, device)
    dataset, _ = get_dataset(target_task, "test", encoder.val_preprocess, location=args.data_location,
                             batch_size=max(16, k), num_workers=0)
    indices = select_target_reference_indices(dataset, target_cls, k, seed)
    images = load_target_reference_images(dataset, indices, device)
    scores = target_score(encoder(images), weight.to(device))
    return {
        "target_task": target_task, "target_cls": int(target_cls),
        "target_classname": classname, "target_text_weight": weight,
        "reference_indices": indices, "reference_scores": scores.detach().cpu(),
        "dormant_ceiling": scores.min().detach().cpu(), "seed": int(seed),
        "num_prompts": len(prompts),
    }


def load_or_build_target_metadata(path, args, encoder, target_task, target_cls=1, k=10, seed=2026):
    path = Path(path)
    if path.is_file():
        metadata = load_target_metadata(path)
        actual = (metadata["target_task"], int(metadata["target_cls"]), len(metadata["reference_indices"]), int(metadata["seed"]))
        expected = (target_task, int(target_cls), int(k), int(seed))
        if actual != expected:
            raise RuntimeError(f"target metadata mismatch: {actual} != {expected}")
        return metadata
    metadata = build_target_metadata(args, encoder, target_task, target_cls, k, seed)
    save_target_metadata(metadata, path)
    return metadata


def target_metadata_json(metadata):
    out = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            out[key] = float(value) if value.ndim == 0 else (value.tolist() if value.numel() <= 32 else {"shape": list(value.shape)})
        else:
            out[key] = value
    return out

