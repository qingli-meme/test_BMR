from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

from src.reviewer_exp.common import (
    ADVERSARY_TASKS, BASE6, ROOT, checkpoint_manifest, default_args, dump_json,
    load_encoder, pool_checkpoint_paths, require_paths, seed_everything,
    suite_for_adversary, trigger_path,
)
from src.reviewer_exp.merge_emr import (
    UPSTREAM_COMMIT, build_encoder, emr_merge_task_vectors,
    floating_task_vectors, official_reference_emr,
    reconstruct_emr_task_vector,
)


OUT = ROOT / "analysis" / "emr"
UPSTREAM_URL = "https://github.com/harveyhuang18/EMR_Merging"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["equivalence", "evaluate"], required=True)
    parser.add_argument("--variant", choices=["clean", "badmerging", "bmr"], default="clean")
    parser.add_argument("--adversary-task", choices=ADVERSARY_TASKS, default="CIFAR100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def suite_tag(adversary_task):
    return "imagenet6" if adversary_task == "ImageNet100" else "base6"


def cache_path(variant, adversary_task):
    tag = suite_tag(adversary_task)
    # Preserve and reuse the already validated CIFAR100 artifacts.
    if adversary_task == "CIFAR100":
        return OUT / "cache" / f"base6_{variant}_emr.pt"
    # The clean BASE6 merge is identical for all five non-ImageNet adversaries.
    if variant == "clean" and tag == "base6":
        return OUT / "cache" / "base6_clean_emr.pt"
    return OUT / "cache" / f"{tag}_{variant}_{adversary_task}_emr.pt"


def load_inputs(variant, adversary_task):
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    paths = pool_checkpoint_paths(adversary_task, variant)
    require_paths([pretrained_path, *paths])
    pretrained = load_encoder(pretrained_path)
    models = [load_encoder(path) for path in paths]
    return pretrained_path, paths, pretrained, models


def get_merge(variant, adversary_task, force=False):
    suite = suite_for_adversary(adversary_task)
    path = cache_path(variant, adversary_task)
    if path.is_file() and not force:
        print(f"[EMR] loading cached merge {path}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["task_names"] != suite:
            raise RuntimeError(f"Cached EMR task order mismatch: {payload['task_names']}")
        return payload
    _, paths, pretrained, models = load_inputs(variant, adversary_task)
    vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in models])
    unified, masks, rescalers = emr_merge_task_vectors(vectors)
    payload = {
        "task_names": list(suite), "checkpoint_paths": [str(p.resolve()) for p in paths],
        "vector_unified": unified, "masks": masks, "rescalers": rescalers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def run_equivalence(args):
    suite = suite_for_adversary(args.adversary_task)
    _, paths, pretrained, models = load_inputs("clean", args.adversary_task)
    vectors = floating_task_vectors(pretrained.state_dict(), [m.state_dict() for m in models])
    ours = emr_merge_task_vectors(vectors)
    reference = official_reference_emr(vectors)
    ours_u, ours_m, ours_r = ours
    ref_u, ref_m, ref_r = reference
    vector_errors = {}
    mask_equal = {}
    worst_abs = worst_rel = 0.0
    for key in ours_u:
        a, b = ours_u[key].double(), ref_u[key].double()
        diff = a - b
        max_abs = float(diff.abs().max())
        rel = float(diff.norm() / b.norm().clamp_min(1e-30))
        vector_errors[key] = {
            "max_abs_error": max_abs, "mean_abs_error": float(diff.abs().mean()),
            "relative_l2_error": rel,
        }
        mask_equal[key] = [
            float((ours_m[key][i] == ref_m[key][i]).float().mean())
            for i in range(len(suite))
        ]
        worst_abs, worst_rel = max(worst_abs, max_abs), max(worst_rel, rel)
    scaler_abs = (ours_r.double() - ref_r.double()).abs()
    scaler_rel = scaler_abs / ref_r.double().abs().clamp_min(1e-30)
    min_mask_equal = min(value for values in mask_equal.values() for value in values)
    passed = (
        worst_abs <= 1e-7 and worst_rel <= 1e-6 and min_mask_equal == 1.0
        and float(scaler_abs.max()) <= 1e-7
    )
    payload = {
        "passed": passed, "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_paths": [str(p.resolve()) for p in paths], "task_names": suite,
        "unified_max_abs_error": worst_abs, "unified_max_relative_l2_error": worst_rel,
        "minimum_mask_equal_fraction": min_mask_equal,
        "rescaler_absolute_errors": scaler_abs.tolist(),
        "rescaler_relative_errors": scaler_rel.tolist(),
        "vector_errors": vector_errors, "mask_equal_fraction": mask_equal,
    }
    dump_json(OUT / "implementation_equivalence.json", payload)
    if not passed:
        raise RuntimeError(f"EMR equivalence failed: {payload}")
    cache_path("clean", args.adversary_task).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "task_names": list(suite), "checkpoint_paths": [str(p.resolve()) for p in paths],
        "vector_unified": ours_u, "masks": ours_m, "rescalers": ours_r,
    }, cache_path("clean", args.adversary_task))
    print(json.dumps({k: v for k, v in payload.items() if k not in {"vector_errors", "mask_equal_fraction"}}, indent=2))


def evaluate_receiver(model, task, args, trigger=None):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation

    eval_args = default_args(args.device, 128)
    eval_args.max_eval_batches = args.max_eval_batches
    model = model.to(args.device).eval()
    clean_acc = 100.0 * eval_single_dataset(model, task, eval_args)["top1"]
    asr = None
    if trigger is not None:
        patch = np.load(trigger)
        applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
        metric = eval_single_dataset(model, task, eval_args, {
            "mask": torch.from_numpy(mask), "applied_patch": torch.from_numpy(applied),
            "target_cls": 1,
        })
        asr = 100.0 * metric["backdoored_acc"]
    return clean_acc, asr


def rebuild_csvs():
    records = {}
    per_rows = []
    for adversary_task in ADVERSARY_TASKS:
        records[adversary_task] = {}
        for variant in ("clean", "badmerging", "bmr"):
            path = OUT / "runs" / f"{variant}__{adversary_task}" / "result.json"
            if not path.is_file() and variant == "clean" and adversary_task != "ImageNet100":
                path = OUT / "runs" / "clean__CIFAR100" / "result.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            expected_suite = suite_for_adversary(adversary_task)
            if record["suite"] != expected_suite:
                raise RuntimeError(f"Result suite mismatch in {path}: {record['suite']}")
            records[adversary_task][variant] = record
            for task, acc in record["per_task_acc"].items():
                row = {
                    "merge_method": "EMR", "adversary_task": adversary_task,
                    "attack": {"clean": "clean_only"}.get(variant, variant),
                    "eval_task": task, "acc": acc, "modulator_task": task,
                }
                assert row["eval_task"] == row["modulator_task"]
                per_rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "emr_per_task_utility.csv").open("w", newline="") as handle:
        fields = ["merge_method", "adversary_task", "attack", "eval_task", "acc", "modulator_task"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(per_rows)
    fields = ["merge_method", "adversary_task", "suite", "clean_only_avg_acc",
              "badmerging_avg_acc", "badmerging_asr", "bmr_avg_acc", "bmr_asr"]
    with (OUT / "emr_main_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for adversary_task in ADVERSARY_TASKS:
            task_records = records[adversary_task]
            if not task_records:
                continue
            writer.writerow({
                "merge_method": "EMR", "adversary_task": adversary_task,
                "suite": suite_tag(adversary_task),
                "clean_only_avg_acc": task_records.get("clean", {}).get("avg_clean_acc"),
                "badmerging_avg_acc": task_records.get("badmerging", {}).get("avg_clean_acc"),
                "badmerging_asr": task_records.get("badmerging", {}).get("asr"),
                "bmr_avg_acc": task_records.get("bmr", {}).get("avg_clean_acc"),
                "bmr_asr": task_records.get("bmr", {}).get("asr"),
            })


def write_manifest(args, variant, pretrained_path, paths, trigger, rescalers):
    path = OUT / "emr_manifest.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = branch = None
    payload.update({
        "bmr_git_commit": commit, "bmr_git_branch": branch,
        "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
        "model": "ViT-B-32", "target_class": 1,
        "seed": args.seed, "device": args.device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    })
    payload.setdefault("runs", {})[f"{variant}__{args.adversary_task}"] = {
        "adversary_task": args.adversary_task,
        "task_names": suite_for_adversary(args.adversary_task),
        "checkpoint_paths": [str(p.resolve()) for p in paths],
        "pretrained_checkpoint": str(pretrained_path.resolve()),
        "trigger_path": str(trigger.resolve()) if trigger else None,
        "asr_modulator_index": (
            suite_for_adversary(args.adversary_task).index(args.adversary_task)
            if trigger else None
        ),
        "rescalers": [float(value) for value in rescalers],
    }
    dump_json(path, payload)


def run_evaluate(args):
    expected_suite = suite_for_adversary(args.adversary_task)
    merge = get_merge(args.variant, args.adversary_task, args.force_merge)
    task_names = merge["task_names"]
    if task_names != expected_suite:
        raise RuntimeError(f"EMR task order mismatch: {task_names}")
    task_to_idx = {task: idx for idx, task in enumerate(task_names)}
    pretrained_path, paths, pretrained, _ = load_inputs(args.variant, args.adversary_task)
    attack_trigger = None if args.variant == "clean" else trigger_path(args.adversary_task, args.variant)
    if attack_trigger:
        require_paths([attack_trigger])
    per_task = {}
    asr = None
    for task in task_names:
        idx = task_to_idx[task]
        assert task_names[idx] == task
        print(f"[EMR receiver] eval_task={task} modulator_task={task_names[idx]} index={idx}", flush=True)
        vector = reconstruct_emr_task_vector(
            merge["vector_unified"], merge["masks"], merge["rescalers"], idx
        )
        model = build_encoder(pretrained, vector)
        per_task[task], task_asr = evaluate_receiver(
            model, task, args,
            attack_trigger if task == args.adversary_task else None,
        )
        if task == args.adversary_task:
            asr = task_asr
        del vector, model
        torch.cuda.empty_cache()
    avg = float(np.mean(list(per_task.values())))
    run_root = "runs_smoke" if args.smoke else "runs"
    run_dir = OUT / run_root / f"{args.variant}__{args.adversary_task}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "merge_method": "EMR", "variant": args.variant,
        "adversary_task": args.adversary_task, "suite": task_names,
        "avg_clean_acc": avg, "per_task_acc": per_task, "asr": asr,
        "max_eval_batches": args.max_eval_batches,
        "rescalers": [float(value) for value in merge["rescalers"]],
        "input_sha256": checkpoint_manifest(
            [pretrained_path, *paths, *([attack_trigger] if attack_trigger else [])]
        ),
    }
    dump_json(run_dir / "result.json", payload)
    if not args.smoke:
        write_manifest(args, args.variant, pretrained_path, paths, attack_trigger, merge["rescalers"])
        rebuild_csvs()
    print(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "equivalence":
        run_equivalence(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
