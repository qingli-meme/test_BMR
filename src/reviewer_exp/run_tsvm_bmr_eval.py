from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.reviewer_exp.common import (
    BASE6, ROOT, checkpoint_manifest, default_args, dump_json, load_encoder,
    pool_checkpoint_paths, require_paths, seed_everything,
)
from src.reviewer_exp.evaluation import evaluate_encoder
from src.reviewer_exp.merge_tsvm import (
    UPSTREAM_COMMIT, build_encoder, floating_task_vectors,
    official_reference_tsvm, tsvm_merge_task_vectors,
)


OUT = ROOT / "analysis" / "tsvm"
UPSTREAM_URL = "https://github.com/AntoAndGar/task_singular_vectors"
ALPHAS = [float(round(x, 1)) for x in np.linspace(0.1, 3.0, 30)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["equivalence", "select-alpha", "evaluate"], required=True)
    parser.add_argument("--variant", choices=["clean", "badmerging", "bmr"], default="clean")
    parser.add_argument("--adversary-task", choices=["CIFAR100"], default="CIFAR100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--force-vector", action="store_true")
    return parser.parse_args()


def vector_path(variant):
    return OUT / "cache" / f"base6_{variant}_tsvm_vector.pt"


def load_inputs(variant):
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    paths = pool_checkpoint_paths("CIFAR100", variant)
    require_paths([pretrained_path, *paths])
    pretrained = load_encoder(pretrained_path)
    models = [load_encoder(path) for path in paths]
    return pretrained_path, paths, pretrained, models


def get_vector(variant, device, force=False):
    path = vector_path(variant)
    if path.is_file() and not force:
        print(f"[TSV-M] loading cached vector {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    _, _, pretrained, models = load_inputs(variant)
    vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in models])
    merged = tsvm_merge_task_vectors(vectors, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, path)
    return merged


@torch.no_grad()
def dev_accuracy(model, task, eval_args, max_batches=0):
    head = get_classification_head(eval_args, task).to(eval_args.device).eval()
    classifier = ImageClassifier(model, head).to(eval_args.device).eval()
    _, loader = get_dataset(
        task, "dev", classifier.val_preprocess, location=eval_args.data_location,
        batch_size=eval_args.batch_size, num_workers=0,
    )
    correct = total = 0
    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(eval_args.device)
        labels = batch["labels"].to(eval_args.device)
        correct += int((classifier(images).argmax(1) == labels).sum())
        total += labels.numel()
    return 100.0 * correct / total


def run_equivalence(args):
    _, paths, pretrained, models = load_inputs("clean")
    vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in models])
    ours = tsvm_merge_task_vectors(vectors, args.device)
    reference = official_reference_tsvm(vectors, args.device)
    per_key = {}
    total_diff_sq = total_ref_sq = 0.0
    worst_abs = worst_rel = 0.0
    for key in ours:
        a, b = ours[key].double(), reference[key].double()
        diff = a - b
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        rel = float(diff.norm() / b.norm().clamp_min(1e-30))
        per_key[key] = {
            "max_abs_error": max_abs, "mean_abs_error": mean_abs,
            "relative_l2_error": rel,
        }
        total_diff_sq += float(diff.pow(2).sum())
        total_ref_sq += float(b.pow(2).sum())
        worst_abs, worst_rel = max(worst_abs, max_abs), max(worst_rel, rel)
    global_rel = (total_diff_sq / max(total_ref_sq, 1e-60)) ** 0.5
    passed = worst_abs <= 1e-4 and global_rel <= 1e-5
    payload = {
        "passed": passed, "upstream_commit": UPSTREAM_COMMIT,
        "num_tasks": 6, "rank_fraction_per_task": 1 / 6,
        "checkpoint_paths": [str(path.resolve()) for path in paths],
        "max_abs_error": worst_abs, "max_relative_l2_error": worst_rel,
        "global_relative_l2_error": global_rel, "per_key": per_key,
    }
    dump_json(OUT / "implementation_equivalence.json", payload)
    if not passed:
        raise RuntimeError(f"TSV-M equivalence failed: {payload}")
    vector_path("clean").parent.mkdir(parents=True, exist_ok=True)
    torch.save(ours, vector_path("clean"))
    print(json.dumps({k: v for k, v in payload.items() if k != "per_key"}, indent=2))


def run_alpha_selection(args):
    vector = get_vector("clean", args.device, args.force_vector)
    pretrained_path, paths, pretrained, _ = load_inputs("clean")
    eval_args = default_args(args.device, 128)
    scores = []
    for alpha in ALPHAS:
        model = build_encoder(pretrained, vector, alpha).to(args.device).eval()
        per_task = {
            task: dev_accuracy(model, task, eval_args, args.max_val_batches)
            for task in BASE6
        }
        avg = float(np.mean(list(per_task.values())))
        scores.append({
            "alpha": alpha, "mean_clean_validation_accuracy": avg,
            "per_task": per_task,
        })
        print(f"[TSV-M alpha] {alpha:.1f} avg={avg:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    best = max(scores, key=lambda item: (item["mean_clean_validation_accuracy"], -item["alpha"]))
    payload = {
        "method": "TSV-M", "variant": "orthogonalization", "num_tasks": 6,
        "rank_fraction_per_task": 1 / 6, "alpha_grid": ALPHAS,
        "base_suite": {
            "tasks": BASE6, "selected_alpha": best["alpha"],
            "selection_metric": "mean_clean_validation_accuracy",
            "grid_results": scores,
        },
        "imagenet100_suite": {
            "tasks": ["ImageNet100", *BASE6[1:]], "selected_alpha": None,
            "selection_metric": "mean_clean_validation_accuracy",
        },
        "clean_checkpoint_manifest": checkpoint_manifest([pretrained_path, *paths]),
    }
    dump_json(OUT / "alpha_selection.json", payload)
    print(json.dumps({"selected_alpha": best["alpha"], "score": best["mean_clean_validation_accuracy"]}, indent=2))


def selected_alpha():
    path = OUT / "alpha_selection.json"
    if not path.is_file():
        raise FileNotFoundError("Run --mode select-alpha first")
    return float(json.loads(path.read_text())["base_suite"]["selected_alpha"])


def rebuild_csvs():
    records = {}
    per_rows = []
    for variant in ("clean", "badmerging", "bmr"):
        path = OUT / "runs" / f"{variant}__CIFAR100" / "result.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        records[variant] = record
        for task, acc in record["per_task_acc"].items():
            per_rows.append({
                "merge_method": "TSV-M", "adversary_task": "CIFAR100",
                "attack": {"clean": "clean_only"}.get(variant, variant),
                "eval_task": task, "acc": acc,
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tsvm_per_task_utility.csv").open("w", newline="") as handle:
        fields = ["merge_method", "adversary_task", "attack", "eval_task", "acc"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(per_rows)
    fields = [
        "merge_method", "adversary_task", "suite", "alpha", "rank_fraction_per_task",
        "clean_only_avg_acc", "badmerging_avg_acc", "badmerging_asr",
        "bmr_avg_acc", "bmr_asr",
    ]
    row = {
        "merge_method": "TSV-M", "adversary_task": "CIFAR100", "suite": "base6",
        "alpha": selected_alpha(), "rank_fraction_per_task": 1 / 6,
        "clean_only_avg_acc": records.get("clean", {}).get("avg_clean_acc"),
        "badmerging_avg_acc": records.get("badmerging", {}).get("avg_clean_acc"),
        "badmerging_asr": records.get("badmerging", {}).get("asr"),
        "bmr_avg_acc": records.get("bmr", {}).get("avg_clean_acc"),
        "bmr_asr": records.get("bmr", {}).get("asr"),
    }
    with (OUT / "tsvm_main_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerow(row)


def write_manifest(args, variant, pretrained_path, paths, trigger):
    path = OUT / "tsvm_manifest.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = branch = None
    payload.update({
        "bmr_git_commit": commit, "bmr_git_branch": branch,
        "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
        "model": "ViT-B-32", "num_tasks": 6, "rank_fraction_per_task": 1 / 6,
        "selected_alpha": {"base_suite": selected_alpha(), "imagenet100_suite": None},
        "pretrained_checkpoint": str(pretrained_path.resolve()), "target_class": 1,
        "seed": args.seed, "device": args.device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    })
    payload.setdefault("runs", {})[variant] = {
        "adversary_task": "CIFAR100", "suite": BASE6,
        "checkpoint_paths": [str(item.resolve()) for item in paths],
        "trigger_path": str(trigger.resolve()) if trigger else None,
    }
    dump_json(path, payload)


def run_evaluate(args):
    alpha = selected_alpha()
    vector = get_vector(args.variant, args.device, args.force_vector)
    pretrained_path, paths, pretrained, _ = load_inputs(args.variant)
    model = build_encoder(pretrained, vector, alpha)
    per_task, avg, asr, trigger = evaluate_encoder(
        model, BASE6, None if args.variant == "clean" else args.variant,
        "CIFAR100", max_eval_batches=args.max_eval_batches, device=args.device,
    )
    run_dir = OUT / "runs" / f"{args.variant}__CIFAR100"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = run_dir / "merged_encoder.pt"
    torch.save(model.cpu(), checkpoint_out)
    payload = {
        "merge_method": "TSV-M", "variant": args.variant,
        "adversary_task": "CIFAR100", "suite": BASE6, "alpha": alpha,
        "rank_fraction_per_task": 1 / 6, "avg_clean_acc": avg,
        "per_task_acc": per_task, "asr": asr,
        "checkpoint_path": str(checkpoint_out.resolve()),
        "input_sha256": checkpoint_manifest(
            [pretrained_path, *paths, *([trigger] if trigger else [])]
        ),
    }
    dump_json(run_dir / "result.json", payload)
    write_manifest(args, args.variant, pretrained_path, paths, trigger)
    rebuild_csvs()
    print(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "equivalence":
        run_equivalence(args)
    elif args.mode == "select-alpha":
        run_alpha_selection(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
