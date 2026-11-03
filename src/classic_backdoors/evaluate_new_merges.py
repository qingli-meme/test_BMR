#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.classic_backdoors.common import (
    ATTACK_SLUGS, TARGET_CLASS, ConditionalTriggerGenerator, apply_raw_patch,
    attack_checkpoint, dynamic_generator_path, load_trigger_spec,
    normalizer_stats, sha256, trigger_spec_path,
)
from src.reviewer_exp.common import (
    ROOT, checkpoint_manifest, checkpoint_path, default_args,
    deterministic_subset_loader, load_encoder, seed_everything,
    suite_for_adversary,
)
from src.reviewer_exp.merge_emr import (
    build_encoder, emr_merge_task_vectors, floating_task_vectors,
    reconstruct_emr_task_vector,
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


ATTACKS = ("no_attack", *ATTACK_SLUGS)
OUT = ROOT / "bmr_paper_results/reviewer_response/classic_backdoors_new_merges"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merge-method", required=True, choices=["emr", "regmean_plusplus"])
    p.add_argument("--attack", required=True, choices=ATTACKS)
    p.add_argument("--adversary-task", required=True, choices=["CIFAR100", "ImageNet100"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-regmean-examples", type=int, default=256)
    p.add_argument("--regmean-batch-size", type=int, default=32)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Write to an isolated smoke-test directory and use a small RegMean++ audit run.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def input_paths(task, attack):
    suite = suite_for_adversary(task)
    malicious = checkpoint_path(task, "clean") if attack == "no_attack" else attack_checkpoint(task, attack)
    return suite, [malicious, *[checkpoint_path(t, "clean") for t in suite[1:]]]


def build_emr(pretrained, encoders, paths, suite, cache_path):
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload["tasks"] != suite or payload["input_sha256"] != checkpoint_manifest(paths):
            raise RuntimeError(f"stale EMR cache: {cache_path}")
    else:
        vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in encoders])
        unified, masks, rescalers = emr_merge_task_vectors(vectors)
        payload = {"tasks": suite, "input_sha256": checkpoint_manifest(paths),
                   "vector_unified": unified, "masks": masks, "rescalers": rescalers}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    vector = reconstruct_emr_task_vector(
        payload["vector_unified"], payload["masks"], payload["rescalers"], 0
    )
    return build_encoder(pretrained, vector), {
        "implementation": "src/reviewer_exp/merge_emr.py",
        "receiver": suite[0], "receiver_index": 0,
        "rescalers": [float(x) for x in payload["rescalers"]],
    }


def build_regmeanpp(pretrained, encoders, suite, args, cache_path):
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False), {
            "implementation": "src/reviewer_exp/merge_regmean_plusplus.py",
            "num_examples": args.num_regmean_examples,
            "reduce_non_diagonal_ratio": 0.95,
            "weight_transpose": True,
        }
    loaders = [
        deterministic_subset_loader(
            task, "train", pretrained.train_preprocess, args.num_regmean_examples,
            args.regmean_batch_size, args.seed, 0,
        )[1]
        for task in suite
    ]
    merger = OpenCLIPRegMeanPlusPlus(
        pretrained, encoders, loaders, num_examples=args.num_regmean_examples,
        batch_size=args.regmean_batch_size, reduce_non_diagonal_ratio=0.95,
        device=args.device,
    )
    merged = merger.run()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged.cpu(), cache_path)
    return merged, {
        "implementation": "src/reviewer_exp/merge_regmean_plusplus.py",
        "num_examples": args.num_regmean_examples,
        "reduce_non_diagonal_ratio": 0.95,
        "weight_transpose": True,
    }


def reuse_clean_merge(pretrained, suite, merge_method, cache_path):
    """Reuse the already validated clean-only merge for the No-Attack row."""
    adversary = suite[0]
    if merge_method == "emr":
        source = ROOT / (
            "analysis/emr/cache/base6_clean_emr.pt"
            if adversary == "CIFAR100"
            else "analysis/emr/cache/imagenet6_clean_ImageNet100_emr.pt"
        )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload["task_names"] != suite:
            raise RuntimeError(f"clean EMR task mismatch in {source}")
        expected = [str(Path(p).resolve()) for p in input_paths(adversary, "no_attack")[1]]
        actual = [str(Path(p).resolve()) for p in payload["checkpoint_paths"]]
        if actual != expected:
            raise RuntimeError(f"clean EMR checkpoint mismatch in {source}")
        vector = reconstruct_emr_task_vector(
            payload["vector_unified"], payload["masks"], payload["rescalers"], 0
        )
        merged = build_encoder(pretrained, vector)
        torch.save(merged.cpu(), cache_path)
        config = {
            "implementation": "src/reviewer_exp/merge_emr.py",
            "receiver": suite[0], "receiver_index": 0,
            "reused_clean_components": str(source.resolve()),
            "rescalers": [float(x) for x in payload["rescalers"]],
        }
        return merged, config

    source = ROOT / "bmr_paper_results/reviewer_response/runs" / (
        f"regmean_plusplus__clean__{adversary}/merged_encoder.pt"
    )
    result = source.parent / "result.json"
    metadata = json.loads(result.read_text())
    if metadata["suite"] != suite:
        raise RuntimeError(f"clean RegMean++ task mismatch in {result}")
    expected_hashes = checkpoint_manifest([
        ROOT / "checkpoints/ViT-B-32/zeroshot.pt", *input_paths(adversary, "no_attack")[1]
    ])
    if metadata["input_sha256"] != expected_hashes:
        raise RuntimeError(f"clean RegMean++ input mismatch in {result}")
    if cache_path.exists() or cache_path.is_symlink():
        cache_path.unlink()
    os.symlink(source.resolve(), cache_path)
    return torch.load(source, map_location="cpu", weights_only=False), {
        "implementation": "src/reviewer_exp/merge_regmean_plusplus.py",
        "reused_clean_merged_encoder": str(source.resolve()),
        "num_examples": metadata["config"]["num_regmean_examples"],
        "reduce_non_diagonal_ratio": metadata["config"]["reduce_non_diagonal_ratio"],
        "weight_transpose": True,
    }


@torch.no_grad()
def evaluate_asr(model, task, attack, args):
    from src.datasets.registry import get_dataset
    from src.heads import get_classification_head

    eval_args = default_args(args.device, args.batch_size)
    head = get_classification_head(eval_args, task).to(args.device).eval()
    model = model.to(args.device).eval()
    _, loader = get_dataset(
        task, "test", model.val_preprocess, location=str(ROOT / "data"),
        batch_size=args.batch_size, num_workers=0,
    )
    mean, std = normalizer_stats(model.val_preprocess)
    if attack == "no_attack":
        # This matches the official BadMerging clean-model evaluation branch:
        # the finalized fixed_22 trigger (already stored in normalized model
        # input space) is presented to an otherwise clean merge.
        trigger_path = ROOT / "trigger/ViT-B-32/fixed_22.npy"
        patch = np.load(trigger_path).astype(np.float32)
        if patch.shape != (3, 22, 22):
            raise ValueError(f"unexpected fixed trigger shape: {patch.shape}")
        mode = "normalized_overwrite"
        generator = None
    else:
        trigger_path = trigger_spec_path(task, attack)
        patch, mode, _ = load_trigger_spec(trigger_path)
        generator = None
        if attack == "dynamic":
            payload = torch.load(dynamic_generator_path(task), map_location="cpu", weights_only=False)
            generator = ConditionalTriggerGenerator(
                payload["num_classes"], payload["noise_dim"], payload["patch_size"]
            ).to(args.device)
            generator.load_state_dict(payload["state_dict"])
            generator.eval()
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    success = total = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_eval_batches and batch_idx >= args.max_eval_batches:
            break
        images, labels = batch[0].to(args.device), batch[1].to(args.device)
        keep = labels != TARGET_CLASS
        if not keep.any():
            continue
        images, labels = images[keep], labels[keep]
        if generator is None:
            if mode == "normalized_overwrite":
                triggered = images.clone()
                triggered[..., -patch.shape[-2]:, -patch.shape[-1]:] = torch.as_tensor(
                    patch, dtype=images.dtype, device=images.device
                )
            else:
                triggered = apply_raw_patch(images, patch, mode, mean, std)
        else:
            targets = torch.full((images.shape[0],), TARGET_CLASS, device=args.device, dtype=torch.long)
            noise = torch.rand((images.shape[0], 100), generator=rng, device=args.device)
            dynamic_patch = generator(targets, noise)
            triggered = apply_raw_patch(images, dynamic_patch, "overwrite", mean, std)
        pred = head(model(triggered)).argmax(1)
        success += int((pred == TARGET_CLASS).sum())
        total += int(labels.numel())
    if total == 0:
        raise RuntimeError("no non-target evaluation examples")
    return 100.0 * success / total, total, str(Path(trigger_path).resolve())


def main():
    args = parse_args()
    seed_everything(args.seed)
    suffix = "__smoke" if args.smoke else ""
    run_dir = OUT / "runs" / f"{args.merge_method}__{args.attack}__{args.adversary_task}{suffix}"
    result_path = run_dir / "result.json"
    if result_path.is_file() and not args.force:
        print(f"[skip complete] {result_path}")
        return
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    suite, paths = input_paths(args.adversary_task, args.attack)
    required = [ROOT / "checkpoints/ViT-B-32/zeroshot.pt", *paths]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    run_dir.mkdir(parents=True, exist_ok=True)
    pretrained = load_encoder(required[0])
    encoders = [load_encoder(path) for path in paths]
    cache = run_dir / "merged_encoder.pt"
    if args.attack == "no_attack" and not args.smoke:
        merged, merge_config = reuse_clean_merge(pretrained, suite, args.merge_method, cache)
    elif args.merge_method == "emr":
        merged, merge_config = build_emr(pretrained, encoders, paths, suite, run_dir / "emr_components.pt")
        torch.save(merged.cpu(), cache)
    else:
        if args.smoke:
            args.num_regmean_examples = min(args.num_regmean_examples, 32)
        merged, merge_config = build_regmeanpp(pretrained, encoders, suite, args, cache)
    asr, num_samples, trig = evaluate_asr(merged, args.adversary_task, args.attack, args)
    result = {
        "complete": True, "merge_method": "EMR" if args.merge_method == "emr" else "RegMean++",
        "attack": args.attack, "adversary_task": args.adversary_task,
        "suite": suite, "target_class": TARGET_CLASS, "asr": asr,
        "num_non_target_samples": num_samples, "checkpoint_paths": [str(Path(p).resolve()) for p in paths],
        "trigger_path": trig, "merged_checkpoint": str(cache.resolve()),
        "merge_config": merge_config, "seed": args.seed,
        "input_sha256": checkpoint_manifest(required),
        "trigger_sha256": sha256(Path(trig)),
        "smoke_test": args.smoke,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
    }
    if not math.isfinite(asr) or not 0 <= asr <= 100:
        raise RuntimeError(f"invalid ASR: {asr}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
