#!/usr/bin/env python3
"""Evaluate the finalized BMR trigger on normal (non-malicious) receivers.

This is an evaluation-only script.  It never trains or modifies a trigger/model.
Each receiver is written separately so the run is safely resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.extend([".", "./src"])

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.reviewer_exp.common import BASE6, default_args, load_encoder, seed_everything
from src.reviewer_exp.merge_emr import (
    build_encoder,
    reconstruct_emr_task_vector,
)
from src.utils import corner_mask_generation


ROOT = Path(__file__).resolve().parents[1]
TARGET_TASK = "CIFAR100"
TARGET_CLASS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--trigger-path",
        default=str(ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"),
    )
    parser.add_argument(
        "--out-dir", default=str(ROOT / "analysis/bmr_trigger_dormancy_final")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_receiver(path: Path):
    receiver = load_encoder(ROOT / "checkpoints/ViT-B-32/zeroshot.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    receiver.load_state_dict(state, strict=True)
    return receiver


def emr_receiver(path: Path):
    pretrained = load_encoder(ROOT / "checkpoints/ViT-B-32/zeroshot.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["task_names"] != BASE6:
        raise RuntimeError(
            f"EMR task order mismatch: {payload['task_names']} != {BASE6}"
        )
    index = payload["task_names"].index(TARGET_TASK)
    vector = reconstruct_emr_task_vector(
        payload["vector_unified"], payload["masks"], payload["rescalers"], index
    )
    return build_encoder(pretrained, vector)


def receiver_specs():
    return [
        (
            "Pretrained Model",
            "encoder_checkpoint",
            ROOT / "checkpoints/ViT-B-32/zeroshot.pt",
        ),
        (
            "Clean CIFAR-100",
            "encoder_checkpoint",
            ROOT / "checkpoints/ViT-B-32/CIFAR100/finetuned.pt",
        ),
        (
            "Task Arithmetic",
            "state_dict",
            ROOT / "bmr_paper_results/motivation/merge_cache/ta_cifar100_suite_full.pt",
        ),
        (
            "TIES",
            "state_dict",
            ROOT / "bmr_paper_results/motivation/merge_cache/ties_cifar100_suite_full.pt",
        ),
        (
            "RegMean",
            "state_dict",
            ROOT / "bmr_paper_results/motivation/merge_cache/regmean_cifar100_suite_full.pt",
        ),
        (
            "AdaMerging",
            "state_dict",
            ROOT / "bmr_paper_results/motivation/merge_cache/adamerging_cifar100_suite_full.pt",
        ),
        (
            "EMR-Merging",
            "emr_payload",
            ROOT / "analysis/emr/cache/base6_clean_emr.pt",
        ),
        (
            "RegMean++",
            "encoder_checkpoint",
            ROOT
            / "bmr_paper_results/reviewer_response/runs/"
            "regmean_plusplus__clean__CIFAR100/merged_encoder.pt",
        ),
    ]


def load_receiver(kind: str, path: Path):
    if kind == "encoder_checkpoint":
        return load_encoder(path)
    if kind == "state_dict":
        return state_receiver(path)
    if kind == "emr_payload":
        return emr_receiver(path)
    raise ValueError(kind)


def target_margin(logits: torch.Tensor):
    target = logits[:, TARGET_CLASS]
    competitors = logits.clone()
    competitors[:, TARGET_CLASS] = -torch.inf
    best_other, best_other_class = competitors.max(dim=1)
    return target - best_other, target, best_other, best_other_class


def evaluate(receiver, trigger_path: Path, args):
    runtime = default_args(args.device, args.batch_size)
    head = get_classification_head(runtime, TARGET_TASK).to(args.device).eval()
    model = ImageClassifier(receiver, head).to(args.device).eval()
    _, loader = get_dataset(
        TARGET_TASK,
        "test",
        model.val_preprocess,
        location=str(ROOT / "data"),
        batch_size=args.batch_size,
    )
    patch = np.load(trigger_path)
    applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
    applied = torch.from_numpy(applied).float().to(args.device)
    mask = torch.from_numpy(mask).float().to(args.device)

    rows = []
    dataset_offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].float().to(args.device)
            labels = batch["labels"].long().to(args.device)
            original_indices = torch.arange(
                dataset_offset, dataset_offset + labels.numel(), dtype=torch.long
            )
            dataset_offset += labels.numel()
            keep = labels != TARGET_CLASS
            if not keep.any():
                continue
            images = images[keep]
            labels = labels[keep]
            original_indices = original_indices[keep.cpu()]
            triggered = mask * applied + (1.0 - mask.expand_as(images)) * images
            logits = model(triggered)
            margins, target_logits, other_logits, other_classes = target_margin(logits)
            predictions = logits.argmax(dim=1)
            for values in zip(
                original_indices.tolist(),
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
                target_logits.cpu().tolist(),
                other_logits.cpu().tolist(),
                other_classes.cpu().tolist(),
                margins.cpu().tolist(),
            ):
                sample_index, label, prediction, target_logit, other_logit, other_class, margin = values
                rows.append(
                    {
                        "sample_index": sample_index,
                        "label": label,
                        "prediction": prediction,
                        "target_logit": target_logit,
                        "best_non_target_logit": other_logit,
                        "best_non_target_class": other_class,
                        "target_margin": margin,
                        "target_activated": int(prediction == TARGET_CLASS),
                    }
                )
    model.cpu()
    del model, head
    torch.cuda.empty_cache()
    margins = np.asarray([row["target_margin"] for row in rows], dtype=np.float64)
    hits = int(sum(row["target_activated"] for row in rows))
    return rows, {
        "num_evaluated": len(rows),
        "target_hits": hits,
        "target_activation_rate_percent": 100.0 * hits / len(rows),
        "mean_target_margin": float(np.mean(margins)),
        "median_target_margin": float(np.median(margins)),
        "q10_target_margin": float(np.quantile(margins, 0.10)),
        "q90_target_margin": float(np.quantile(margins, 0.90)),
    }


def slug(label: str) -> str:
    return label.lower().replace("-", "_").replace("+", "plus").replace(" ", "_")


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, summaries, metadata):
    ordered = [label for label, _, _ in receiver_specs()]
    summary_rows = [summaries[label] for label in ordered]
    write_csv(out_dir / "summary.csv", summary_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "receivers": summary_rows}, handle, indent=2)

    merged = summary_rows[2:]
    avg_tar = float(np.mean([row["target_activation_rate_percent"] for row in merged]))
    max_tar = float(np.max([row["target_activation_rate_percent"] for row in merged]))
    lines = [
        "# BMR Trigger Target Activation Evaluation",
        "",
        "No trigger or model was retrained. All values use all non-target CIFAR-100 test samples.",
        "",
        "| Receiver | TAR (%) | n | Mean margin | Median margin | q10 | q90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['receiver']} | {row['target_activation_rate_percent']:.4f} | "
            f"{row['num_evaluated']} | {row['mean_target_margin']:.6f} | "
            f"{row['median_target_margin']:.6f} | {row['q10_target_margin']:.6f} | "
            f"{row['q90_target_margin']:.6f} |"
        )
    lines += [
        "",
        f"- Average TAR over the six clean merged models: **{avg_tar:.4f}%**",
        f"- Maximum TAR over the six clean merged models: **{max_tar:.4f}%**",
        "",
        "## Exact artifacts",
        "",
        f"- Trigger: `{metadata['trigger_path']}`",
        f"- Trigger SHA-256: `{metadata['trigger_sha256']}`",
    ]
    for item in metadata["receiver_sources"]:
        lines.append(
            f"- {item['receiver']}: `{item['path']}` "
            f"(kind: `{item['kind']}`, SHA-256: `{item['sha256']}`)"
        )
    lines += [
        "",
        "Per-sample raw files are stored under `raw/`; resumable per-receiver summaries are under `receivers/`.",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    receiver_dir = out_dir / "receivers"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receiver_dir.mkdir(parents=True, exist_ok=True)
    trigger_path = Path(args.trigger_path).resolve()
    specs = receiver_specs()
    required = [trigger_path, *[path for _, _, path in specs]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    metadata = {
        "model": "CLIP ViT-B/32",
        "dataset": TARGET_TASK,
        "target_class": TARGET_CLASS,
        "trigger_size": "22x22",
        "trigger_path": str(trigger_path),
        "trigger_sha256": sha256(trigger_path),
        "receiver_sources": [
            {
                "receiver": label,
                "kind": kind,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for label, kind, path in specs
        ],
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    summaries = {}
    start = time.time()
    for index, (label, kind, source) in enumerate(specs, 1):
        receiver_json = receiver_dir / f"{slug(label)}.json"
        raw_csv = raw_dir / f"{slug(label)}.csv"
        if receiver_json.is_file() and raw_csv.is_file() and not args.force:
            print(f"[{index}/8] reuse {label}", flush=True)
            summaries[label] = json.loads(receiver_json.read_text(encoding="utf-8"))
            continue
        print(f"[{index}/8] evaluating {label} from {source}", flush=True)
        receiver = load_receiver(kind, source)
        raw_rows, summary = evaluate(receiver, trigger_path, args)
        summary.update(
            {
                "receiver": label,
                "source_kind": kind,
                "source_path": str(source.resolve()),
            }
        )
        for row in raw_rows:
            row["receiver"] = label
        write_csv(raw_csv, raw_rows)
        receiver_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[label] = summary
        del receiver, raw_rows
        torch.cuda.empty_cache()
        print(json.dumps(summary, indent=2), flush=True)

    metadata["runtime_seconds"] = time.time() - start
    write_outputs(out_dir, summaries, metadata)
    (out_dir / ".complete").touch()
    print(f"Complete: {out_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
