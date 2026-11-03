#!/usr/bin/env python3
"""Generate candidate sheets for the seven-dataset qualitative input figure."""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100, ImageFolder

sys.path.append(".")
sys.path.append("./src")

from src.datasets.cars import PytorchStanfordCars  # noqa: E402
from src.datasets.eurosat import EuroSAT  # noqa: E402
from src.datasets.gtsrb import GTSRB, PyTorchGTSRB  # noqa: E402
from src.datasets.sun397 import SUN397  # noqa: E402


ROOT = Path("./data")
OUT = Path("./candidates")
SEED = 2026


def fit_square(image, size=256, resample=Image.Resampling.LANCZOS):
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(resized, resample)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def short_name(value, length=27):
    value = value.split(",", 1)[0]
    return value if len(value) <= length else value[: length - 1] + "..."


def save_sheet(title, records, filename, resample=Image.Resampling.LANCZOS):
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        image = Image.open(record["path"]).convert("RGB") if record["path"] else record.pop("_image")
        ax.imshow(fit_square(image, resample=resample))
        ax.set_title(
            f"idx {record['sample_index']} | {short_name(record['class_name'])}",
            fontsize=8,
            pad=3,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.7, h_pad=1.0, w_pad=0.4)
    fig.savefig(OUT / filename, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_unavailable(title, reason, filename):
    fig = plt.figure(figsize=(8.0, 2.2), facecolor="white")
    fig.text(0.5, 0.67, title, ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.38, "Candidates unavailable", ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.19, reason, ha="center", va="center", fontsize=9, wrap=True)
    fig.savefig(OUT / filename, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def class_indices(targets):
    result = defaultdict(list)
    for index, target in enumerate(targets):
        result[int(target)].append(index)
    return result


def choose_by_class(dataset, classnames, priority_names, count=12, path_getter=None):
    by_class = class_indices(dataset.targets if hasattr(dataset, "targets") else [s[1] for s in dataset.samples])
    rng = random.Random(SEED)
    selected_classes = []
    lowered = [name.lower().replace("_", " ") for name in classnames]
    for wanted in priority_names:
        matches = [i for i, name in enumerate(lowered) if wanted in name and i not in selected_classes]
        selected_classes.extend(matches[:1])
    records = []
    rounds = math.ceil(count / max(1, len(selected_classes)))
    for round_id in range(rounds):
        for class_id in selected_classes:
            choices = by_class[class_id]
            index = choices[(round_id * 7 + rng.randrange(len(choices))) % len(choices)]
            item = dataset[index]
            image = item[0]
            path = path_getter(index) if path_getter else ""
            record = {
                "sample_index": int(index),
                "class_id": int(class_id),
                "class_name": str(classnames[class_id]),
                "path": str(path),
            }
            if not path:
                record["_image"] = image
            records.append(record)
            if len(records) == count:
                return records
    return records


def cifar100_candidates():
    dataset = CIFAR100(root=ROOT, train=False, download=False)
    records = choose_by_class(
        dataset,
        dataset.classes,
        ["tiger", "lion", "wolf", "rabbit", "leopard", "bear"],
    )
    save_sheet(
        "CIFAR-100 candidates",
        records,
        "cifar100_candidates.png",
        resample=Image.Resampling.NEAREST,
    )
    return records


def gtsrb_candidates():
    wrapper = GTSRB(None, location=str(ROOT), batch_size=1, num_workers=0)
    dataset = wrapper.test_dataset
    dataset.targets = [target for _, target in dataset._samples]
    dataset.samples = dataset._samples
    records = choose_by_class(
        dataset,
        wrapper.classnames,
        ["stop", "yield", "no entry", "30 kph", "50 kph", "70 kph"],
        path_getter=lambda i: dataset._samples[i][0],
    )
    save_sheet("GTSRB candidates", records, "gtsrb_candidates.png")
    return records


def eurosat_candidates():
    wrapper = EuroSAT(None, location=str(ROOT), batch_size=1, num_workers=0)
    dataset = wrapper.test_dataset
    records = choose_by_class(
        dataset,
        wrapper.classnames,
        ["industrial", "residential", "river", "highway", "lake", "forest"],
        path_getter=lambda i: dataset.samples[i][0],
    )
    save_sheet("EuroSAT candidates", records, "eurosat_candidates.png")
    return records


def cars_candidates():
    dataset = PytorchStanfordCars(str(ROOT), split="test", transform=None, download=False)
    rng = random.Random(SEED)
    pool = list(range(len(dataset)))
    rng.shuffle(pool)
    records = []
    for index in pool:
        path, class_id = dataset._samples[index]
        with Image.open(path) as image:
            width, height = image.size
        ratio = width / max(height, 1)
        if width < 400 or height < 250 or not 1.15 <= ratio <= 2.2:
            continue
        records.append(
            {
                "sample_index": int(index),
                "class_id": int(class_id),
                "class_name": str(dataset.classes[class_id]),
                "path": str(path),
            }
        )
        if len(records) == 12:
            break
    save_sheet("Cars candidates", records, "cars_candidates.png")
    return records


def pets_candidates():
    dataset = ImageFolder(ROOT / "pets" / "test")
    records = choose_by_class(
        dataset,
        [name.replace("_", " ") for name in dataset.classes],
        ["beagle", "samoyed", "persian", "siamese", "pug", "shiba inu"],
        path_getter=lambda i: dataset.samples[i][0],
    )
    save_sheet("Pets candidates", records, "pets_candidates.png")
    return records


def sun397_candidates():
    wrapper = SUN397(None, location=str(ROOT), batch_size=1, num_workers=0)
    dataset = wrapper.test_dataset
    records = choose_by_class(
        dataset,
        wrapper.classnames,
        [
            "beach",
            "waterfall",
            "baseball field",
            "airport terminal",
            "church outdoor",
            "kitchen",
            "library indoor",
            "mountain",
            "forest path",
            "stadium",
            "bedroom",
            "bridge",
        ],
        path_getter=lambda i: dataset.samples[i][0],
    )
    save_sheet("SUN397 candidates", records, "sun397_candidates.png")
    return records


def imagenet100_candidates():
    dataset = ImageFolder(ROOT / "ImageNet100" / "val")
    labels_path = ROOT / "ImageNet100" / "_extracting" / "Labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    classnames = [labels.get(wnid, wnid) for wnid in dataset.classes]
    records = choose_by_class(
        dataset,
        classnames,
        [
            "macaw",
            "peacock",
            "bald eagle",
            "flamingo",
            "pelican",
            "goldfinch",
            "chickadee",
            "hummingbird",
            "lorikeet",
            "cockatoo",
            "great white shark",
            "tiger shark",
        ],
        path_getter=lambda i: dataset.samples[i][0],
    )
    save_sheet("ImageNet-100 candidates", records, "imagenet100_candidates.png")
    return records


def serializable(records):
    return [{k: v for k, v in record.items() if not k.startswith("_")} for record in records]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": SEED,
        "datasets": {},
    }

    generators = [
        ("CIFAR-100", cifar100_candidates),
        ("GTSRB", gtsrb_candidates),
        ("EuroSAT", eurosat_candidates),
        ("Cars", cars_candidates),
        ("Pets", pets_candidates),
        ("SUN397", sun397_candidates),
        ("ImageNet-100", imagenet100_candidates),
    ]
    for name, generator in generators:
        records = generator()
        metadata["datasets"][name] = {"status": "available", "candidates": serializable(records)}

    artifact_status = {
        "CIFAR-100": {
            "BadMerging": "trigger/ViT-B-32/On_CIFAR100_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy",
        },
        "GTSRB": {
            "BadMerging": "trigger/ViT-B-32/On_GTSRB_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_GTSRB_Tgt_1_L_22.npy",
        },
        "EuroSAT": {
            "BadMerging": "trigger/ViT-B-32/On_EuroSAT_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_EuroSAT_Tgt_1_L_22.npy",
        },
        "Cars": {
            "BadMerging": "trigger/ViT-B-32/On_Cars_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_Cars_Tgt_1_L_22.npy",
        },
        "Pets": {
            "BadMerging": "trigger/ViT-B-32/On_PETS_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_PETS_Tgt_1_L_22.npy",
        },
        "SUN397": {
            "BadMerging": "trigger/ViT-B-32/On_SUN397_Tgt_1_L_22.npy",
            "BMR": "trigger/ViT-B-32/KDR_DTK_SUN397_Tgt_1_L_22.npy",
        },
        "ImageNet-100": {
            "BadMerging": None,
            "BMR": "trigger/ViT-B-32/KDR_DTK_ImageNet100_Tgt_1_L_22.npy",
        },
    }
    metadata["trigger_artifacts"] = artifact_status
    (OUT / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUT / "candidate_metadata.json")


if __name__ == "__main__":
    main()
