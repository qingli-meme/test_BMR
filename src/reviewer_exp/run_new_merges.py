from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.reviewer_exp.common import (
    BASE6, RESULT_ROOT, ROOT, checkpoint_manifest, checkpoint_path,
    default_args, deterministic_subset_loader, dump_json, load_encoder,
    pool_checkpoint_paths, require_paths, seed_everything, suite_for_adversary,
    trigger_path,
)
from src.reviewer_exp.merge_localize_stitch import OpenCLIPLocalizeAndStitch
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merge-method", required=True, choices=["regmean_plusplus", "localize_stitch"])
    p.add_argument("--attack", required=True, choices=["clean", "bmr", "badmerging"])
    p.add_argument("--adversary-task", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", default=str(RESULT_ROOT))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--num-regmean-examples", type=int, default=256)
    p.add_argument("--regmean-batch-size", type=int, default=32)
    p.add_argument("--max-blocks", type=int)
    p.add_argument("--n-shot", type=int, default=64)
    p.add_argument("--mask-epochs", type=int, default=10)
    p.add_argument("--max-eval-batches", type=int, default=0)
    return p.parse_args()


def _dataset_label(dataset, idx):
    # Resolve nested Subset indices without decoding images.
    while isinstance(dataset, torch.utils.data.Subset):
        idx = dataset.indices[idx]
        dataset = dataset.dataset
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)[idx]
            return int(value.item() if torch.is_tensor(value) else value)
    item = dataset[idx]
    return int(item[1].item() if torch.is_tensor(item[1]) else item[1])


def kshot_loader(task, preprocess, n_shot, batch_size, seed, data_cap=1024):
    # Match the official Localize-and-Stitch data policy: take up to n_shot
    # examples per class from the *full training set*, globally shuffle the
    # candidates, and only then apply data_cap.  Capping a class-sorted list
    # would bias CIFAR100's 1024 examples toward only its early classes.
    from src.datasets.registry import registry

    dataset_bundle = registry[task](
        preprocess,
        location=str(ROOT / "data"),
        batch_size=batch_size,
        num_workers=0,
    )
    dataset = dataset_bundle.train_dataset
    by_class = defaultdict(list)
    for idx in range(len(dataset)):
        label = _dataset_label(dataset, idx)
        if len(by_class[label]) < n_shot:
            by_class[label].append(idx)
    chosen = []
    for label in sorted(by_class):
        chosen.extend(by_class[label][:n_shot])
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:data_cap]
    subset = torch.utils.data.Subset(dataset, chosen)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


def build_merged(args, suite, paths):
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    pretrained = load_encoder(pretrained_path)
    encoders = [load_encoder(p) for p in paths]
    if args.smoke:
        suite = suite[:2]
        paths = paths[:2]
        encoders = encoders[:2]

    if args.merge_method == "regmean_plusplus":
        n_examples = 32 if args.smoke else args.num_regmean_examples
        loaders = [
            deterministic_subset_loader(task, "train", pretrained.train_preprocess, n_examples,
                                        min(args.regmean_batch_size, n_examples), args.seed, 0)[1]
            for task in suite
        ]
        merger = OpenCLIPRegMeanPlusPlus(
            pretrained, encoders, loaders, num_examples=n_examples,
            batch_size=args.regmean_batch_size, reduce_non_diagonal_ratio=0.95,
            device=args.device, max_blocks=(1 if args.smoke else args.max_blocks),
        )
        return merger.run(), {"num_regmean_examples": n_examples, "reduce_non_diagonal_ratio": 0.95,
                              "weight_transpose": True, "batch_size": args.regmean_batch_size}

    from src.heads import get_classification_head
    eval_args = default_args(args.device, 128)
    n_shot = 1 if args.smoke else args.n_shot
    epochs = 1 if args.smoke else args.mask_epochs
    loaders = [kshot_loader(t, pretrained.val_preprocess, n_shot, 32 if args.smoke else 128, args.seed,
                            data_cap=(32 if args.smoke else 1024)) for t in suite]
    heads = [get_classification_head(eval_args, t) for t in suite]
    merger = OpenCLIPLocalizeAndStitch(
        pretrained, encoders, suite, loaders, heads, paths,
        sparsity=1e-5, sigmoid_bias=5, l1_strength=1, mask_lr=1e7,
        mask_epochs=epochs, n_shot=n_shot,
        data_signature=f"official_train_kshot_global_shuffle_cap{32 if args.smoke else 1024}_seed{args.seed}_v2",
        device=args.device,
        cache_dir=Path(args.output_root) / "cache" / "localize_stitch",
    )
    model, proportions = merger.run()
    return model, {"n_shot": n_shot, "sigmoid_bias": 5, "l1_strength": 1,
                   "learning_rate": 1e7, "num_train_epochs": epochs,
                   "sparsity": 1e-5, "mask_proportions": proportions,
                   "validation_split": "official-style full-train k-shot; global shuffle before cap; clean labels/images only"}


def evaluate(model, args, suite, attack, adversary_task):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation

    eval_args = default_args(args.device, 128)
    eval_args.max_eval_batches = 2 if args.smoke else args.max_eval_batches
    model = model.to(args.device).eval()
    per_task = {}
    tasks = suite[:2] if args.smoke else suite
    for task in tasks:
        per_task[task] = 100.0 * eval_single_dataset(model, task, eval_args)["top1"]
    asr = None
    trig_path = None
    if attack != "clean":
        trig_path = trigger_path(adversary_task, attack)
        trigger = np.load(trig_path)
        applied, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
        metrics = eval_single_dataset(model, adversary_task, eval_args, {
            "mask": torch.from_numpy(mask), "applied_patch": torch.from_numpy(applied), "target_cls": 1,
        })
        asr = 100.0 * metrics["backdoored_acc"]
    return per_task, float(np.mean(list(per_task.values()))), asr, trig_path


def main():
    args = parse_args()
    seed_everything(args.seed)
    suite = suite_for_adversary(args.adversary_task)
    paths = pool_checkpoint_paths(args.adversary_task, args.attack)
    if args.smoke:
        # The port smoke test is intentionally two clean models, as required.
        suite = BASE6[:2]
        paths = [checkpoint_path(t, "clean") for t in suite]
    require_paths(paths)
    out_root = Path(args.output_root)
    run_name = f"{args.merge_method}__{args.attack}__{args.adversary_task}" + ("__smoke" if args.smoke else "")
    run_dir = out_root / "runs" / run_name
    result_path = run_dir / "result.json"
    if result_path.is_file() and not args.force:
        print(f"[skip complete] {result_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    merged, config = build_merged(args, suite, paths)
    checkpoint_out = run_dir / "merged_encoder.pt"
    torch.save(merged.cpu(), checkpoint_out)
    per_task, avg, asr, trig = evaluate(merged, args, suite, args.attack, args.adversary_task)
    if not args.smoke and args.attack == "clean" and avg < 40.0:
        raise RuntimeError(
            f"clean-utility sanity check failed for {args.merge_method}: "
            f"average={avg:.2f}% < 40%; refusing to launch attacked grid"
        )
    identities = checkpoint_manifest([ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt", *paths,
                                      *([trig] if trig else [])])
    result = {
        "merge_method": args.merge_method, "attack": args.attack,
        "adversary_task": args.adversary_task, "suite": suite,
        "target_class": 1, "avg_clean_acc": avg, "asr": asr,
        "per_task_acc": per_task, "checkpoint_path": str(checkpoint_out.resolve()),
        "config": config, "input_sha256": identities, "seed": args.seed,
        "smoke": args.smoke,
    }
    dump_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
