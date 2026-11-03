from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

FLOAT_DTYPES = {torch.float16, torch.float32, torch.float64, torch.bfloat16}
BASE6 = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
IMAGENET6 = ["ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
ADVERSARY_TASKS = ["CIFAR100", "ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
RESULT_ROOT = ROOT / "bmr_paper_results" / "reviewer_response"
MODEL = "ViT-B-32"


def seed_everything(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch.load(path, **kwargs)


def suite_for_adversary(task: str):
    return IMAGENET6 if task == "ImageNet100" else BASE6


def is_float_tensor(x):
    return torch.is_tensor(x) and x.dtype in FLOAT_DTYPES


def clone_state_dict(sd: Mapping[str, torch.Tensor], device="cpu"):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to(device).clone() if torch.is_tensor(v) else copy.deepcopy(v)
    return out


def state_dict_delta(a, b):
    """Return a-b for floating tensor keys shared by both state dicts."""
    return {
        k: a[k].detach().float().cpu() - b[k].detach().float().cpu()
        for k in a
        if k in b and is_float_tensor(a[k]) and is_float_tensor(b[k]) and a[k].shape == b[k].shape
    }


def apply_delta(base, delta, scale=1.0):
    out = clone_state_dict(base, device="cpu")
    for k, d in delta.items():
        if k in out and is_float_tensor(out[k]):
            out[k] = (out[k].float() + float(scale) * d.float()).to(out[k].dtype)
    return out


def subtract_delta(base, delta, scale=1.0):
    return apply_delta(base, delta, scale=-float(scale))


def l2_norm(delta: Mapping[str, torch.Tensor]):
    total = torch.zeros((), dtype=torch.float64)
    for v in delta.values():
        total += v.double().pow(2).sum()
    return total.sqrt().item()


def cosine_delta(a, b, eps=1e-12):
    dot = torch.zeros((), dtype=torch.float64)
    na = torch.zeros((), dtype=torch.float64)
    nb = torch.zeros((), dtype=torch.float64)
    for k in set(a) & set(b):
        x, y = a[k].double().reshape(-1), b[k].double().reshape(-1)
        dot += (x * y).sum()
        na += (x * x).sum()
        nb += (y * y).sum()
    return (dot / (na.sqrt() * nb.sqrt() + eps)).item()


def layer_group(name: str) -> str:
    if "transformer.resblocks." in name:
        p = name.split("transformer.resblocks.", 1)[1]
        return f"resblock_{p.split('.', 1)[0]}"
    if "ln_post" in name:
        return "ln_post"
    if any(s in name for s in ("conv1", "class_embedding", "positional_embedding", "ln_pre")):
        return "embedding"
    return "other"


def layerwise_norms(delta):
    sq = {}
    for k, v in delta.items():
        g = layer_group(k)
        sq[g] = sq.get(g, 0.0) + v.double().pow(2).sum().item()
    return {k: val ** 0.5 for k, val in sq.items()}


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path, block_size=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_path(task: str, attack: str = "clean", target_cls=1, patch_size=22) -> Path:
    root = ROOT / "checkpoints" / MODEL
    attack = attack.lower()
    if attack in {"clean", "cleanonly"}:
        name = task
    elif attack == "bmr":
        name = f"{task}_KDR_DTK_BMR_{task}_Tgt_{target_cls}_L_{patch_size}"
    elif attack in {"badmerging", "badmerging-on", "on"}:
        name = f"{task}_On_{task}_Tgt_{target_cls}_L_{patch_size}"
    else:
        raise ValueError(f"unknown attack: {attack}")
    return root / name / "finetuned.pt"


def trigger_path(task: str, attack: str, target_cls=1, patch_size=22) -> Path:
    attack = attack.lower()
    prefix = "KDR_DTK" if attack == "bmr" else "On"
    return ROOT / "trigger" / MODEL / f"{prefix}_{task}_Tgt_{target_cls}_L_{patch_size}.npy"


def pool_checkpoint_paths(adversary_task: str, attack: str):
    paths = []
    for task in suite_for_adversary(adversary_task):
        state = attack if task == adversary_task else "clean"
        paths.append(checkpoint_path(task, state))
    return paths


def require_paths(paths: Sequence[Path]):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_encoder(path: os.PathLike, device="cpu"):
    return torch_load(path, map_location=device)


def default_args(device="cuda", batch_size=128):
    args = Namespace(
        model=MODEL,
        ckpt_dir=str(ROOT / "checkpoints"),
        data_location=str(ROOT / "data"),
        batch_size=batch_size,
        num_workers=0,
        device=device,
        openclip_cachedir=str(ROOT / "open_clip"),
        cache_dir=str(ROOT / "cache"),
    )
    args.save = str(ROOT / "checkpoints" / MODEL)
    return args


def dataset_loader(task, split, preprocess, batch_size=32, num_workers=0):
    from src.datasets.registry import get_dataset

    return get_dataset(
        task, split, preprocess, location=str(ROOT / "data"),
        batch_size=batch_size, num_workers=num_workers,
    )


def deterministic_subset_loader(task, split, preprocess, num_examples, batch_size, seed=2026, num_workers=0):
    dataset, _ = dataset_loader(task, split, preprocess, batch_size, num_workers)
    count = min(int(num_examples), len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return subset, loader


def apply_trigger(images, trigger):
    from src.utils import corner_mask_generation

    applied, mask, _, _ = corner_mask_generation(trigger, image_size=tuple(images.shape[1:]))
    applied = torch.as_tensor(applied, dtype=images.dtype, device=images.device)
    mask = torch.as_tensor(mask, dtype=images.dtype, device=images.device)
    return mask * applied + (1 - mask.expand_as(images)) * images


def checkpoint_manifest(paths):
    return {str(Path(p).resolve()): sha256_file(p) for p in paths}

