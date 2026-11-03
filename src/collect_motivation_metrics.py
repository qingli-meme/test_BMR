#!/usr/bin/env python3
"""Collect parameter coupling and triggered-margin impact for the BMR motivation figure."""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.extend([".", "./src"])

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from heads import get_classification_head
from modeling import ImageClassifier
from regmean import RegMean
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from utils import corner_mask_generation
from src.reviewer_exp.common import deterministic_subset_loader
from src.reviewer_exp.merge_emr import (
    floating_task_vectors,
    reconstruct_emr_task_vector,
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


TASKS = ["CIFAR100", "ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
DISPLAY_TASK = {task: ("Pets" if task == "PETS" else task) for task in TASKS}
METHODS = ["ta", "ties", "regmean", "adamerging", "emr", "regmean_plusplus"]
DISPLAY_METHOD = {
    "ta": "Task Arithmetic",
    "ties": "TIES",
    "regmean": "RegMean",
    "adamerging": "AdaMerging",
    "emr": "EMR",
    "regmean_plusplus": "RegMean++",
}
CIFAR_SUITE = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
IMAGENET_SUITE = ["ImageNet100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument("--regmeanpp-num-examples", type=int, default=256)
    parser.add_argument("--regmeanpp-batch-size", type=int, default=32)
    parser.add_argument("--regmeanpp-ratio", type=float, default=0.95)
    parser.add_argument("--out-dir", default="./analysis/bmr_motivation")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    args.tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    unknown_methods = set(args.methods) - set(METHODS)
    unknown_tasks = set(args.tasks) - set(TASKS)
    if unknown_methods or unknown_tasks:
        parser.error(f"unknown methods={sorted(unknown_methods)} tasks={sorted(unknown_tasks)}")
    return args


def suite_for(task):
    return IMAGENET_SUITE if task == "ImageNet100" else CIFAR_SUITE


def suite_tag(suite):
    return "imagenet100_suite" if suite[0] == "ImageNet100" else "cifar100_suite"


def checkpoint_path(args, task):
    return Path(args.ckpt_dir) / args.model / task / "finetuned.pt"


def state_dict_from_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"unsupported checkpoint object: {path}")
    return {key: value.detach().cpu().clone() for key, value in obj.items()}


def load_encoder(path, state=None):
    encoder = torch.load(path, map_location="cpu")
    if state is not None:
        encoder.load_state_dict(state, strict=False)
    return encoder


def runtime_args(args, datasets):
    return SimpleNamespace(
        model=args.model,
        ckpt_dir=args.ckpt_dir,
        data_location=args.data_location,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=os.path.join(args.ckpt_dir, args.model),
        openclip_cachedir="./open_clip",
        cache_dir="./cache",
        dataset_list=datasets,
        num_train_batch=args.regmean_num_train_batch,
    )


def merge_cache_path(args, method, suite, adversary=None):
    cache_dir = Path(args.out_dir) / "merge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if adversary is None else f"background_without_{adversary}"
    return cache_dir / f"{method}_{suite_tag(suite)}_{suffix}.pt"


def build_linear_state(args, datasets, method, base_state):
    flat_base = state_dict_to_vector(base_state, []).cpu()
    vectors = []
    for task in datasets:
        state = state_dict_from_checkpoint(checkpoint_path(args, task))
        vectors.append(state_dict_to_vector(state, []).cpu() - flat_base)
        del state
    vectors = torch.stack(vectors)
    if method == "ta":
        merged = flat_base + args.scaling_coef * vectors.sum(0)
    else:
        merged_vector = ties_merging(
            vectors,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_base + args.scaling_coef * merged_vector
    state = vector_to_state_dict(merged, base_state, [])
    del flat_base, vectors, merged
    gc.collect()
    return state


def build_regmean_state(args, datasets):
    if not torch.cuda.is_available():
        raise RuntimeError("RegMean requires CUDA in this repository")
    rargs = runtime_args(args, datasets)
    merger = RegMean(rargs, logger=None)
    models = []
    grams = []
    for task in datasets:
        encoder = torch.load(checkpoint_path(args, task), map_location="cuda:0")
        model = ImageClassifier(encoder, merger.class_head_dict[task])
        model.freeze_head()
        model = model.cuda().eval()
        with torch.no_grad():
            gram = merger.compute_gram(model, task)
        grams.append({key: value.detach().cpu() for key, value in gram.items()})
        models.append(model.cpu())
        torch.cuda.empty_cache()
        print(f"[regmean gram] {task}", flush=True)
    avg_params = merger.avg_merge(models, regmean_grams=grams)
    pretrained = Path(args.ckpt_dir) / args.model / "zeroshot.pt"
    encoder = torch.load(pretrained, map_location="cuda:0")
    wrapper = ImageClassifier(encoder, merger.class_head_dict[datasets[-1]])
    wrapper.freeze_head()
    wrapper = wrapper.cuda()
    merger.copy_params_to_model(avg_params, wrapper)
    state = {key: value.detach().cpu().clone() for key, value in wrapper.image_encoder.state_dict().items()}
    del merger, models, grams, avg_params, wrapper, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return state


def adamerging_lambda_path(args, suite):
    root = Path("./ada") / args.model
    if suite_tag(suite) == "cifar100_suite":
        return root / "Clean_Epoch_500.pt"
    return root / "Clean_imagenet100_six_task_clean_Epoch_500.pt"


def build_adamerging_state(args, suite, base_state, adversary=None):
    lambda_path = adamerging_lambda_path(args, suite)
    if not lambda_path.exists():
        raise FileNotFoundError(
            f"missing clean AdaMerging coefficients for {suite_tag(suite)}: {lambda_path}"
        )
    lambdas = torch.load(lambda_path, map_location="cpu")
    if tuple(lambdas.shape) != (len(base_state), len(suite)):
        raise ValueError(f"unexpected AdaMerging shape {tuple(lambdas.shape)} in {lambda_path}")
    task_states = {task: state_dict_from_checkpoint(checkpoint_path(args, task)) for task in suite}
    merged = {}
    for layer_index, (key, base) in enumerate(base_state.items()):
        value = base.clone()
        if torch.is_floating_point(value):
            for task_index, task in enumerate(suite):
                if task == adversary:
                    continue
                value.add_(lambdas[layer_index, task_index] * (task_states[task][key] - base))
        merged[key] = value
    del task_states, lambdas
    gc.collect()
    return merged


def emr_payload_path(suite):
    root = Path("./analysis/emr/cache")
    if suite_tag(suite) == "cifar100_suite":
        return root / "base6_clean_emr.pt"
    return root / "imagenet6_clean_ImageNet100_emr.pt"


def build_emr_task_state(suite, task, base_state):
    """Return EMR's task-specific clean receiver for decision impact."""
    path = emr_payload_path(suite)
    payload = torch.load(path, map_location="cpu")
    if payload["task_names"] != suite:
        raise RuntimeError(f"EMR task order mismatch in {path}: {payload['task_names']}")
    task_index = payload["task_names"].index(task)
    vector = reconstruct_emr_task_vector(
        payload["vector_unified"], payload["masks"], payload["rescalers"], task_index
    )
    state = {}
    for key, base in base_state.items():
        if key in vector and torch.is_floating_point(base):
            state[key] = (base.float() + vector[key].float()).to(base.dtype)
        else:
            state[key] = base.clone()
    return state


def build_emr_background_state(args, datasets, base_state):
    """Build the leave-one-out EMR unified drift used for global coupling.

    EMR has task-specific output modulators, so after removing the target task
    there is no target-specific reconstructed receiver.  Its shared unified
    vector is the only algorithm-native global background direction and is
    therefore used for the parameter-coupling diagnostic.
    """
    task_states = [state_dict_from_checkpoint(checkpoint_path(args, task)) for task in datasets]
    vectors = floating_task_vectors(base_state, task_states)
    keys = list(vectors[0])
    unified = {}
    for key in keys:
        values = [vector[key] for vector in vectors]
        mean = torch.stack(values, dim=0).mean(dim=0)
        sign = (mean > 0).to(mean.dtype).mul_(2).sub_(1)
        maximum = torch.zeros_like(values[0])
        for value in values:
            aligned = value * sign > 0
            maximum = torch.maximum(maximum, torch.abs(value) * aligned)
        unified[key] = maximum * sign
    state = {}
    for key, base in base_state.items():
        if key in unified and torch.is_floating_point(base):
            state[key] = (base.float() + unified[key].float()).to(base.dtype)
        else:
            state[key] = base.clone()
    del task_states, vectors, unified
    gc.collect()
    return state


def build_regmeanpp_state(args, datasets):
    if datasets == CIFAR_SUITE:
        canonical = Path(
            "./bmr_paper_results/reviewer_response/runs/"
            "regmean_plusplus__clean__CIFAR100/merged_encoder.pt"
        )
        if canonical.exists():
            print(f"[reuse RegMean++ clean merge] {canonical}", flush=True)
            return state_dict_from_checkpoint(canonical)
    if datasets == IMAGENET_SUITE:
        canonical = Path(
            "./bmr_paper_results/reviewer_response/runs/"
            "regmean_plusplus__clean__ImageNet100/merged_encoder.pt"
        )
        if canonical.exists():
            print(f"[reuse RegMean++ clean merge] {canonical}", flush=True)
            return state_dict_from_checkpoint(canonical)
    pretrained_path = Path(args.ckpt_dir) / args.model / "zeroshot.pt"
    pretrained = torch.load(pretrained_path, map_location="cpu")
    models = [torch.load(checkpoint_path(args, task), map_location="cpu") for task in datasets]
    loaders = [
        deterministic_subset_loader(
            task,
            "train",
            pretrained.train_preprocess,
            args.regmeanpp_num_examples,
            args.regmeanpp_batch_size,
            2026,
            0,
        )[1]
        for task in datasets
    ]
    merger = OpenCLIPRegMeanPlusPlus(
        pretrained,
        models,
        loaders,
        num_examples=args.regmeanpp_num_examples,
        batch_size=args.regmeanpp_batch_size,
        reduce_non_diagonal_ratio=args.regmeanpp_ratio,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    merged = merger.run()
    state = {
        key: value.detach().cpu().clone()
        for key, value in merged.state_dict().items()
    }
    del merger, merged, models, pretrained, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def existing_regmean_background(task):
    root = Path("analysis/bmr_multitask_compare") / task / "orthogonality_margin_path" / "background_cache"
    matches = sorted(root.glob("regmean_*_batches8.pt"))
    return matches[0] if len(matches) == 1 else None


def get_merged_state(args, method, suite, base_state, adversary=None):
    cache = merge_cache_path(args, method, suite, adversary)
    if cache.exists():
        return state_dict_from_checkpoint(cache)
    datasets = [task for task in suite if task != adversary]
    if method in {"ta", "ties"}:
        state = build_linear_state(args, datasets, method, base_state)
    elif method == "regmean":
        existing = existing_regmean_background(adversary) if adversary and adversary != "ImageNet100" else None
        if existing is not None:
            print(f"[reuse regmean background] {existing}", flush=True)
            state = state_dict_from_checkpoint(existing)
        else:
            state = build_regmean_state(args, datasets)
    elif method == "adamerging":
        state = build_adamerging_state(args, suite, base_state, adversary)
    elif method == "emr":
        if adversary is None:
            raise ValueError("EMR full receiver is task-specific; request it with build_emr_task_state")
        state = build_emr_background_state(args, datasets, base_state)
    elif method == "regmean_plusplus":
        state = build_regmeanpp_state(args, datasets)
    else:
        raise ValueError(method)
    torch.save(state, cache)
    print(f"[cache] {cache}", flush=True)
    return state


def absolute_cosine(base_state, task_state, background_state):
    dot = 0.0
    task_norm = 0.0
    background_norm = 0.0
    for key, base in base_state.items():
        if not torch.is_floating_point(base):
            continue
        task_delta = (task_state[key] - base).reshape(-1).double()
        background_delta = (background_state[key] - base).reshape(-1).double()
        dot += torch.dot(task_delta, background_delta).item()
        task_norm += torch.dot(task_delta, task_delta).item()
        background_norm += torch.dot(background_delta, background_delta).item()
    return abs(dot / np.sqrt(task_norm * background_norm))


def trigger_tensors(args, task):
    path = Path("./trigger") / args.model / f"KDR_DTK_{task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    trigger = np.load(path)
    applied, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return path, torch.from_numpy(applied).float(), torch.from_numpy(mask).float()


def target_margin(logits, target_cls):
    target = logits[:, target_cls]
    competitors = logits.clone()
    competitors[:, target_cls] = -torch.inf
    return target - competitors.max(1).values


def evaluate_pair(args, task, clean_encoder, merged_encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head_args = runtime_args(args, suite_for(task))
    head = get_classification_head(head_args, task).to(device).eval()
    merged_model = ImageClassifier(merged_encoder, head).to(device).eval()
    receiver_cache = Path(args.out_dir) / "receiver_cache" / f"{task}_clean_margins.npz"
    clean_margin_map = None
    clean_model = None
    if receiver_cache.exists():
        cached = np.load(receiver_cache)
        clean_margin_map = dict(zip(cached["sample_id"].tolist(), cached["margin"].tolist()))
        print(f"[reuse clean margins] {receiver_cache}", flush=True)
    else:
        clean_model = ImageClassifier(clean_encoder, head).to(device).eval()
    _, loader = get_dataset(
        task,
        "test",
        merged_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    trigger_path, patch, mask = trigger_tensors(args, task)
    patch = patch.to(device)
    mask = mask.to(device)
    rows = []
    clean_cache_ids = []
    clean_cache_margins = []
    processed = 0
    fallback = 0
    with torch.no_grad():
        for batch in loader:
            data = maybe_dictionarize(batch)
            images = data["images"]
            labels = data["labels"]
            indices = data.get("indices")
            if indices is None:
                indices = torch.arange(fallback, fallback + labels.numel())
                fallback += labels.numel()
            indices = torch.as_tensor(indices)
            keep = labels != args.target_cls
            images = images[keep]
            indices = indices[keep]
            if args.max_samples:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                images = images[:remaining]
                indices = indices[:remaining]
            if images.numel() == 0:
                continue
            images = images.to(device).float()
            triggered = mask * patch + (1.0 - mask) * images
            if clean_margin_map is None:
                clean_margin = target_margin(clean_model(triggered), args.target_cls).cpu().numpy()
                clean_cache_ids.extend(indices.tolist())
                clean_cache_margins.extend(clean_margin.tolist())
            else:
                clean_margin = np.asarray([clean_margin_map[int(i)] for i in indices.tolist()])
            merge_margin = target_margin(merged_model(triggered), args.target_cls).cpu().numpy()
            for sample_id, margin_clean, margin_merge in zip(indices.tolist(), clean_margin, merge_margin):
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "target_margin_task_clean": float(margin_clean),
                        "target_margin_merge_clean": float(margin_merge),
                        "impact": float(abs(margin_merge - margin_clean)),
                    }
                )
            processed += images.shape[0]
    if clean_margin_map is None:
        receiver_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            receiver_cache,
            sample_id=np.asarray(clean_cache_ids, dtype=np.int64),
            margin=np.asarray(clean_cache_margins, dtype=np.float32),
        )
        print(f"[cache clean margins] {receiver_cache}", flush=True)
    del clean_model, merged_model, head, clean_encoder, merged_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[eval] {task}: n={len(rows)} trigger={trigger_path}", flush=True)
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args):
    shard_dir = Path(args.out_dir) / "shards"
    summary_rows = []
    sample_rows = []
    for method in METHODS:
        for task in TASKS:
            summary_path = shard_dir / f"{method}_{task}_summary.json"
            sample_path = shard_dir / f"{method}_{task}_samples.csv"
            if not summary_path.exists() or not sample_path.exists():
                continue
            summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            with open(sample_path, encoding="utf-8") as handle:
                sample_rows.extend(csv.DictReader(handle))
    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "motivation_metrics.csv",
        summary_rows,
        ["merge_method", "task", "coupling_mean", "coupling_std", "impact_mean", "impact_std"],
    )
    write_csv(
        out_dir / "motivation_metrics_per_sample.csv",
        sample_rows,
        ["merge_method", "task", "sample_id", "target_margin_task_clean", "target_margin_merge_clean", "impact"],
    )
    print(f"[aggregate] summaries={len(summary_rows)} samples={len(sample_rows)}", flush=True)


def main():
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
        return
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.ckpt_dir) / args.model / "zeroshot.pt"
    base_state = state_dict_from_checkpoint(base_path)

    for method in args.methods:
        full_states = {}
        for task in args.tasks:
            summary_path = shard_dir / f"{method}_{task}_summary.json"
            sample_path = shard_dir / f"{method}_{task}_samples.csv"
            if summary_path.exists() and sample_path.exists():
                print(f"[skip completed] {method}/{task}", flush=True)
                continue
            suite = suite_for(task)
            tag = suite_tag(suite)
            if method == "emr":
                full_state = build_emr_task_state(suite, task, base_state)
            else:
                if tag not in full_states:
                    full_states[tag] = get_merged_state(args, method, suite, base_state)
                full_state = full_states[tag]
            background_state = get_merged_state(args, method, suite, base_state, adversary=task)
            task_state = state_dict_from_checkpoint(checkpoint_path(args, task))
            coupling = absolute_cosine(base_state, task_state, background_state)
            clean_encoder = load_encoder(checkpoint_path(args, task))
            merged_encoder = load_encoder(base_path, full_state)
            rows = evaluate_pair(args, task, clean_encoder, merged_encoder)
            impacts = np.asarray([row["impact"] for row in rows], dtype=np.float64)
            summary = {
                "merge_method": DISPLAY_METHOD[method],
                "task": DISPLAY_TASK[task],
                "coupling_mean": float(coupling),
                "coupling_std": 0.0,
                "impact_mean": float(impacts.mean()),
                "impact_std": float(impacts.std(ddof=0)),
            }
            enriched = [
                {"merge_method": DISPLAY_METHOD[method], "task": DISPLAY_TASK[task], **row}
                for row in rows
            ]
            write_csv(
                sample_path,
                enriched,
                ["merge_method", "task", "sample_id", "target_margin_task_clean", "target_margin_merge_clean", "impact"],
            )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[done] {method}/{task}: {summary}", flush=True)
            del background_state, task_state
            gc.collect()
        del full_states
        gc.collect()
    aggregate(args)


if __name__ == "__main__":
    main()
