from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.reviewer_exp.common import ROOT


MODEL = "ViT-B-32"
TARGET_CLASS = 1
PATCH_SIZE = 22
ATTACK_SLUGS = ("badnets", "label_consistent", "trojannn", "dynamic")


def attack_checkpoint(task: str, attack: str) -> Path:
    name = f"{task}_CLASSIC_{attack.upper()}_{task}_Tgt_{TARGET_CLASS}_L_{PATCH_SIZE}"
    return ROOT / "checkpoints" / MODEL / name / "finetuned.pt"


def attack_dir(task: str, attack: str) -> Path:
    return attack_checkpoint(task, attack).parent


def trigger_spec_path(task: str, attack: str) -> Path:
    return ROOT / "trigger" / MODEL / f"CLASSIC_{attack.upper()}_{task}_Tgt_{TARGET_CLASS}_L_{PATCH_SIZE}.npz"


def dynamic_generator_path(task: str) -> Path:
    return ROOT / "trigger" / MODEL / f"CLASSIC_DYNAMIC_{task}_Tgt_{TARGET_CLASS}_L_{PATCH_SIZE}_generator.pt"


def save_trigger_spec(path: Path, patch: np.ndarray, mode: str, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, patch=np.asarray(patch, dtype=np.float32), mode=np.asarray(mode),
             metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_trigger_spec(path: Path):
    payload = np.load(path, allow_pickle=False)
    return payload["patch"].astype(np.float32), str(payload["mode"].item()), json.loads(str(payload["metadata"].item()))


def normalizer_stats(preprocess):
    normalizer = preprocess.transforms[-1]
    mean = torch.tensor(normalizer.mean).view(1, 3, 1, 1)
    std = torch.tensor(normalizer.std).view(1, 3, 1, 1)
    return mean, std


def to_raw(images, mean, std):
    return (images * std.to(images) + mean.to(images)).clamp(0, 1)


def to_normalized(images, mean, std):
    return (images - mean.to(images)) / std.to(images)


def apply_raw_patch(images, patch, mode, mean, std):
    raw = to_raw(images, mean, std)
    patch = torch.as_tensor(patch, dtype=raw.dtype, device=raw.device)
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    h, w = patch.shape[-2:]
    region = raw[..., -h:, -w:]
    if mode == "overwrite":
        region.copy_(patch.expand(region.shape[0], -1, -1, -1))
    elif mode == "additive":
        region.copy_((region + patch.expand(region.shape[0], -1, -1, -1)).clamp(0, 1))
    else:
        raise ValueError(f"unknown trigger mode: {mode}")
    return to_normalized(raw, mean, std)


class ConditionalTriggerGenerator(nn.Module):
    """Official c-BaN MLP topology, generalized to 100 classes and 22x22."""

    def __init__(self, num_classes=100, noise_dim=100, patch_size=22):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.patch_size = patch_size
        self.fc_label = nn.Linear(num_classes, 64)
        self.fc_noise = nn.Linear(noise_dim, 64)
        self.fc_joint = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc_hidden = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 3 * patch_size * patch_size)

    def forward(self, labels, noise):
        one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        x = torch.cat((self.fc_label(one_hot), self.fc_noise(noise)), dim=-1)
        x = torch.relu(self.fc_joint(x))
        x = self.dropout(x)
        x = self.fc_hidden(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return torch.sigmoid(x).view(-1, 3, self.patch_size, self.patch_size)


def stable_poison_ids(dataset, poison_rate, target_class, clean_label, seed):
    labels = []
    ids = []
    for local_idx in range(len(dataset)):
        nested, idx = dataset, local_idx
        while isinstance(nested, torch.utils.data.Subset):
            idx = int(nested.indices[idx])
            nested = nested.dataset
        value = None
        for attr in ("targets", "labels", "_labels"):
            if hasattr(nested, attr):
                value = getattr(nested, attr)[idx]
                break
        if value is None:
            value = nested[idx][1]
        labels.append(int(value.item() if torch.is_tensor(value) else value))
        ids.append(int(idx))
    eligible = [sample_id for sample_id, label in zip(ids, labels)
                if (label == target_class if clean_label else label != target_class)]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if clean_label:
        # The author LC configuration poisons 100% of the target-class corpus.
        count = len(eligible)
    else:
        count = min(len(eligible), round(len(dataset) * poison_rate))
    return set(eligible[:count]), {"eligible": len(eligible), "selected": count, "dataset_size": len(dataset)}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

