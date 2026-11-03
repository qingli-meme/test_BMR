#!/usr/bin/env python3
"""Evaluate one existing BMR sensitivity checkpoint with EMR or RegMean++.

This runner never trains a malicious model or trigger.  It accepts their exact
paths, merges that checkpoint with the requested clean-task checkpoints, then
reports sample-complete clean utility and CIFAR100 ASR.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.reviewer_exp.common import (
    ROOT, checkpoint_manifest, checkpoint_path, default_args,
    deterministic_subset_loader, dump_json, load_encoder, require_paths,
    seed_everything,
)
from src.reviewer_exp.merge_emr import (
    build_encoder, emr_merge_task_vectors, floating_task_vectors,
    reconstruct_emr_task_vector,
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merge-method", required=True, choices=["emr", "regmean_plusplus"])
    p.add_argument("--experiment", required=True)
    p.add_argument("--setting-value", required=True)
    p.add_argument("--tasks", required=True, help="Comma-separated task order; CIFAR100 must be first")
    p.add_argument("--malicious-checkpoint", required=True)
    p.add_argument("--trigger", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-regmean-examples", type=int, default=256)
    p.add_argument("--regmean-batch-size", type=int, default=32)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def evaluate_model(model, task, args, trigger_path=None):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation

    eval_args = default_args(args.device, args.batch_size)
    eval_args.max_eval_batches = args.max_eval_batches
    model = model.to(args.device).eval()
    clean = 100.0 * eval_single_dataset(model, task, eval_args)["top1"]
    asr = None
    if trigger_path is not None:
        patch = np.load(trigger_path)
        applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
        metric = eval_single_dataset(model, task, eval_args, {
            "mask": torch.from_numpy(mask),
            "applied_patch": torch.from_numpy(applied),
            "target_cls": 1,
        })
        asr = 100.0 * metric["backdoored_acc"]
    return float(clean), None if asr is None else float(asr)


def build_paths(tasks, malicious_checkpoint):
    if not tasks or tasks[0] != "CIFAR100" or len(set(tasks)) != len(tasks):
        raise ValueError(f"Expected unique task order beginning with CIFAR100, got {tasks}")
    return [Path(malicious_checkpoint)] + [checkpoint_path(task, "clean") for task in tasks[1:]]


def run_emr(args, tasks, paths, pretrained, cache_path):
    if cache_path.is_file() and not args.force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload["tasks"] != tasks or payload["input_sha256"] != checkpoint_manifest(paths):
            raise RuntimeError(f"Refusing stale EMR cache: {cache_path}")
    else:
        encoders = [load_encoder(path) for path in paths]
        vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in encoders])
        unified, masks, rescalers = emr_merge_task_vectors(vectors)
        payload = {
            "tasks": tasks,
            "input_sha256": checkpoint_manifest(paths),
            "vector_unified": unified,
            "masks": masks,
            "rescalers": rescalers,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)

    per_task, asr = {}, None
    for idx, task in enumerate(tasks):
        print(f"[EMR receiver] eval_task={task} modulator_task={tasks[idx]} index={idx}", flush=True)
        vector = reconstruct_emr_task_vector(
            payload["vector_unified"], payload["masks"], payload["rescalers"], idx
        )
        model = build_encoder(pretrained, vector)
        per_task[task], task_asr = evaluate_model(
            model, task, args, Path(args.trigger) if task == "CIFAR100" else None
        )
        if task == "CIFAR100":
            asr = task_asr
        del model, vector
        torch.cuda.empty_cache()
    config = {
        "implementation": "src/reviewer_exp/merge_emr.py",
        "receiver_policy": "task-specific mask and rescaler",
        "asr_receiver_task": "CIFAR100",
        "rescalers": [float(x) for x in payload["rescalers"]],
    }
    return per_task, asr, config


def run_regmeanpp(args, tasks, paths, pretrained, cache_path):
    if cache_path.is_file() and not args.force:
        merged = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        encoders = [load_encoder(path) for path in paths]
        loaders = [
            deterministic_subset_loader(
                task, "train", pretrained.train_preprocess,
                args.num_regmean_examples, args.regmean_batch_size, args.seed, 0,
            )[1]
            for task in tasks
        ]
        merger = OpenCLIPRegMeanPlusPlus(
            pretrained, encoders, loaders,
            num_examples=args.num_regmean_examples,
            batch_size=args.regmean_batch_size,
            reduce_non_diagonal_ratio=0.95,
            device=args.device,
        )
        merged = merger.run()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged.cpu(), cache_path)

    per_task = {}
    for task in tasks:
        per_task[task], _ = evaluate_model(merged, task, args)
    _, asr = evaluate_model(merged, "CIFAR100", args, Path(args.trigger))
    config = {
        "implementation": "src/reviewer_exp/merge_regmean_plusplus.py",
        "num_regmean_examples": args.num_regmean_examples,
        "regmean_batch_size": args.regmean_batch_size,
        "reduce_non_diagonal_ratio": 0.95,
        "weight_transpose": True,
        "activation_seed": args.seed,
    }
    return per_task, asr, config


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    seed_everything(args.seed)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    trigger = Path(args.trigger)
    paths = build_paths(tasks, args.malicious_checkpoint)
    require_paths([pretrained_path, trigger, *paths])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.is_file() and not args.force:
        prior = json.loads(result_path.read_text())
        if prior.get("complete") is True:
            print(f"[skip complete] {result_path}")
            return

    identities = checkpoint_manifest([pretrained_path, *paths, trigger])
    identity_tag = identities[str(paths[0].resolve())][:12]
    cache_path = out_dir / f"{args.merge_method}_{identity_tag}.pt"
    pretrained = load_encoder(pretrained_path)
    if args.merge_method == "emr":
        per_task, asr, config = run_emr(args, tasks, paths, pretrained, cache_path)
    else:
        per_task, asr, config = run_regmeanpp(args, tasks, paths, pretrained, cache_path)

    avg = float(np.mean(list(per_task.values())))
    values = [avg, asr, *per_task.values()]
    if not all(math.isfinite(float(x)) and 0 <= float(x) <= 100 for x in values):
        raise RuntimeError(f"Invalid evaluation metrics: avg={avg}, asr={asr}, per_task={per_task}")
    result = {
        "complete": True,
        "experiment": args.experiment,
        "setting_value": args.setting_value,
        "merge_method": "EMR" if args.merge_method == "emr" else "RegMean++",
        "avg_clean_acc": avg,
        "asr": asr,
        "per_task_acc": per_task,
        "num_tasks": len(tasks),
        "tasks": tasks,
        "adversary_task": "CIFAR100",
        "target_class": 1,
        "seed": args.seed,
        "malicious_checkpoint": str(paths[0].resolve()),
        "checkpoint_paths": [str(p.resolve()) for p in paths],
        "trigger_path": str(trigger.resolve()),
        "cache_path": str(cache_path.resolve()),
        "input_sha256": identities,
        "config": config,
        "max_eval_batches": args.max_eval_batches,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    dump_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
