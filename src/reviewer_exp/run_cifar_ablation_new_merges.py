from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.reviewer_exp.common import (
    BASE6,
    RESULT_ROOT,
    ROOT,
    checkpoint_manifest,
    checkpoint_path,
    default_args,
    deterministic_subset_loader,
    dump_json,
    load_encoder,
    require_paths,
    seed_everything,
)
from src.reviewer_exp.merge_emr import (
    build_encoder,
    emr_merge_task_vectors,
    floating_task_vectors,
    reconstruct_emr_task_vector,
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


VARIANTS = {
    "ordinary": {
        "method": "KDR_DTK_BMR_ORD",
        "trigger": "KDR_DTK",
        "table": "training_ablation",
    },
    "background_aug": {
        "method": "KDR_DTK_BMR_BG",
        "trigger": "KDR_DTK",
        "table": "training_ablation",
    },
    "absolute_margin": {
        "method": "KDR_DTK_BMR_ABS",
        "trigger": "KDR_DTK",
        "table": "training_ablation",
    },
    "without_scale_variation": {
        "method": "KDR_DTK_BMR_NOSCALE",
        "trigger": "KDR_DTK",
        "table": "training_ablation",
    },
    "without_target_alignment": {
        "method": "KDR_DTK_BMR_TRIG_NOALIGN",
        "trigger": "KDR_DTK_TRIG_NOALIGN",
        "table": "trigger_ablation",
    },
    "without_signal_constraint": {
        "method": "KDR_DTK_BMR_TRIG_NOSIG",
        "trigger": "KDR_DTK_TRIG_NOSIG",
        "table": "trigger_ablation",
    },
    "without_dormancy": {
        "method": "KDR_DTK_BMR_TRIG_NODORM",
        "trigger": "KDR_DTK_TRIG_NODORM",
        "table": "trigger_ablation",
    },
    "full_bmr": {
        "method": "KDR_DTK_BMR",
        "trigger": "KDR_DTK",
        "table": "both",
    },
    "gain_only": {
        "method": "KDR_DTK_BMR_GAINONLY",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "no_background": {
        "method": "KDR_DTK_BMR_NO_BG",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "fixed_background": {
        "method": "KDR_DTK_BMR_FIXETA",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "fixed_residual": {
        "method": "KDR_DTK_BMR_FIXALPHA",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "both_fixed": {
        "method": "KDR_DTK_BMR_BOTHFIX",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "ordinary_normmatch": {
        "method": "KDR_DTK_BMR_ORD_NORMMATCH",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "background_normmatch": {
        "method": "KDR_DTK_BMR_BG_NORMMATCH",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
    "bmr_normmatch_bg": {
        "method": "KDR_DTK_BMR_BMR_NORMMATCH_BG",
        "trigger": "KDR_DTK",
        "table": "bmr_controls",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-method", choices=["emr", "regmean_plusplus"], required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-regmean-examples", type=int, default=256)
    parser.add_argument("--regmean-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def variant_paths(variant):
    cfg = VARIANTS[variant]
    root = ROOT / "checkpoints" / "ViT-B-32"
    adversary = root / f"CIFAR100_{cfg['method']}_CIFAR100_Tgt_1_L_22" / "finetuned.pt"
    paths = [adversary if task == "CIFAR100" else checkpoint_path(task, "clean") for task in BASE6]
    trigger = ROOT / "trigger" / "ViT-B-32" / f"{cfg['trigger']}_CIFAR100_Tgt_1_L_22.npy"
    return paths, trigger


def evaluate_receiver(model, task, args, trigger=None):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation

    eval_args = default_args(args.device, 128)
    eval_args.max_eval_batches = args.max_eval_batches
    model = model.to(args.device).eval()
    clean = 100.0 * eval_single_dataset(model, task, eval_args)["top1"]
    asr = None
    if trigger is not None:
        patch = np.load(trigger)
        applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
        metrics = eval_single_dataset(
            model,
            task,
            eval_args,
            {
                "mask": torch.from_numpy(mask),
                "applied_patch": torch.from_numpy(applied),
                "target_cls": 1,
            },
        )
        asr = 100.0 * metrics["backdoored_acc"]
    return clean, asr


def run_emr(pretrained, models, paths, trigger, args):
    vectors = floating_task_vectors(pretrained.state_dict(), [model.state_dict() for model in models])
    unified, masks, rescalers = emr_merge_task_vectors(vectors)
    per_task = {}
    asr = None
    cifar_receiver = None
    for index, task in enumerate(BASE6):
        vector = reconstruct_emr_task_vector(unified, masks, rescalers, index)
        receiver = build_encoder(pretrained, vector)
        per_task[task], task_asr = evaluate_receiver(
            receiver, task, args, trigger if task == "CIFAR100" else None
        )
        if task == "CIFAR100":
            asr = task_asr
            cifar_receiver = receiver.cpu()
        else:
            del receiver
        del vector
        torch.cuda.empty_cache()
    return (
        per_task,
        asr,
        {"rescalers": [float(value) for value in rescalers]},
        cifar_receiver,
    )


def run_regmean_plusplus(pretrained, models, paths, trigger, args):
    loaders = [
        deterministic_subset_loader(
            task,
            "train",
            pretrained.train_preprocess,
            args.num_regmean_examples,
            args.regmean_batch_size,
            args.seed,
            0,
        )[1]
        for task in BASE6
    ]
    merger = OpenCLIPRegMeanPlusPlus(
        pretrained,
        models,
        loaders,
        num_examples=args.num_regmean_examples,
        batch_size=args.regmean_batch_size,
        reduce_non_diagonal_ratio=0.95,
        device=args.device,
    )
    merged = merger.run()
    per_task = {}
    asr = None
    for task in BASE6:
        per_task[task], task_asr = evaluate_receiver(
            merged, task, args, trigger if task == "CIFAR100" else None
        )
        if task == "CIFAR100":
            asr = task_asr
    return (
        per_task,
        asr,
        {
            "num_regmean_examples": args.num_regmean_examples,
            "batch_size": args.regmean_batch_size,
            "reduce_non_diagonal_ratio": 0.95,
        },
        merged.cpu(),
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = VARIANTS[args.variant]
    paths, trigger = variant_paths(args.variant)
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    require_paths([pretrained_path, trigger, *paths])

    out_dir = RESULT_ROOT / "ablation_new_merges" / f"{args.merge_method}__{args.variant}"
    result_path = out_dir / "result.json"
    if result_path.is_file() and not args.force:
        print(f"[skip complete] {result_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained = load_encoder(pretrained_path)
    models = [load_encoder(path) for path in paths]
    if args.merge_method == "emr":
        per_task, asr, merge_config, cifar_receiver = run_emr(
            pretrained, models, paths, trigger, args
        )
    else:
        per_task, asr, merge_config, cifar_receiver = run_regmean_plusplus(
            pretrained, models, paths, trigger, args
        )

    # These three receivers are reused by the sample-matched real-merge-gain
    # diagnostic.  Persisting them avoids repeating the expensive merge.
    if args.variant in {"ordinary", "background_aug", "full_bmr"}:
        torch.save(cifar_receiver, out_dir / "receiver_cifar100.pt")

    result = {
        "merge_method": "EMR" if args.merge_method == "emr" else "RegMean++",
        "variant": args.variant,
        "table": cfg["table"],
        "method_name": cfg["method"],
        "suite": BASE6,
        "target_class": 1,
        "avg_clean_acc": float(np.mean(list(per_task.values()))),
        "asr": float(asr),
        "per_task_acc": per_task,
        "checkpoint_paths": [str(path.resolve()) for path in paths],
        "trigger_path": str(trigger.resolve()),
        "merge_config": merge_config,
        "seed": args.seed,
        "input_sha256": checkpoint_manifest([pretrained_path, trigger, *paths]),
    }
    dump_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

