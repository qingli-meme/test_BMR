from __future__ import annotations

import argparse
import csv
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
from src.reviewer_exp.common import (
    BASE6,
    RESULT_ROOT,
    ROOT,
    default_args,
    deterministic_subset_loader,
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
from src.reviewer_exp.run_cifar_ablation_new_merges import VARIANTS, variant_paths
from src.utils import corner_mask_generation


OUT = ROOT / "bmr_paper_results" / "reviewer_response" / "new_merge_mechanism_diagnostics"
TARGET_TASK = "CIFAR100"
TARGET_CLASS = 1
GAIN_VARIANTS = ("ordinary", "background_aug", "full_bmr")
DISPLAY_VARIANT = {
    "ordinary": "A_plain",
    "background_aug": "B_background_aug",
    "full_bmr": "C_full_bmr",
}
DISPLAY_METHOD = {"emr": "EMR", "regmean_plusplus": "RegMean++"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="emr,regmean_plusplus")
    parser.add_argument("--gain-variants", default=",".join(GAIN_VARIANTS))
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-regmean-examples", type=int, default=256)
    parser.add_argument("--regmean-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--force-merge", action="store_true")
    args = parser.parse_args()
    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    args.gain_variants = [
        item.strip() for item in args.gain_variants.split(",") if item.strip()
    ]
    bad = set(args.methods) - set(DISPLAY_METHOD)
    if bad:
        parser.error(f"unsupported methods: {sorted(bad)}")
    bad_variants = set(args.gain_variants) - set(GAIN_VARIANTS)
    if bad_variants:
        parser.error(f"unsupported gain variants: {sorted(bad_variants)}")
    return args


def trigger_path(source):
    return ROOT / "trigger" / "ViT-B-32" / f"{source}_CIFAR100_Tgt_1_L_22.npy"


def receiver_cache(method, variant):
    return RESULT_ROOT / "ablation_new_merges" / f"{method}__{variant}" / "receiver_cifar100.pt"


def load_emr_clean_receiver():
    pretrained = load_encoder(ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt")
    payload = torch.load(
        ROOT / "analysis" / "emr" / "cache" / "base6_clean_emr.pt",
        map_location="cpu",
        weights_only=False,
    )
    if payload["task_names"] != BASE6:
        raise RuntimeError("EMR clean cache task order mismatch")
    index = payload["task_names"].index(TARGET_TASK)
    vector = reconstruct_emr_task_vector(
        payload["vector_unified"], payload["masks"], payload["rescalers"], index
    )
    return build_encoder(pretrained, vector)


def canonical_receiver(method, variant):
    if variant == "clean":
        if method == "emr":
            return load_emr_clean_receiver()
        return load_encoder(
            RESULT_ROOT / "runs" / "regmean_plusplus__clean__CIFAR100" / "merged_encoder.pt"
        )
    if variant == "full_bmr":
        if method == "emr":
            pretrained = load_encoder(ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt")
            payload = torch.load(
                ROOT / "analysis" / "emr" / "cache" / "base6_bmr_emr.pt",
                map_location="cpu",
                weights_only=False,
            )
            index = payload["task_names"].index(TARGET_TASK)
            vector = reconstruct_emr_task_vector(
                payload["vector_unified"], payload["masks"], payload["rescalers"], index
            )
            return build_encoder(pretrained, vector)
        return load_encoder(
            RESULT_ROOT / "runs" / "regmean_plusplus__bmr__CIFAR100" / "merged_encoder.pt"
        )
    return None


def build_ablation_receiver(method, variant, args):
    paths, _ = variant_paths(variant)
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    require_paths([pretrained_path, *paths])
    pretrained = load_encoder(pretrained_path)
    models = [load_encoder(path) for path in paths]
    if method == "emr":
        vectors = floating_task_vectors(
            pretrained.state_dict(), [model.state_dict() for model in models]
        )
        unified, masks, rescalers = emr_merge_task_vectors(vectors)
        vector = reconstruct_emr_task_vector(unified, masks, rescalers, BASE6.index(TARGET_TASK))
        return build_encoder(pretrained, vector)

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
    return merger.run().cpu()


def get_receiver(method, variant, args):
    canonical = canonical_receiver(method, variant)
    if canonical is not None:
        return canonical
    cache = receiver_cache(method, variant)
    if cache.is_file() and not args.force_merge:
        return load_encoder(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    receiver = build_ablation_receiver(method, variant, args).cpu()
    torch.save(receiver, cache)
    return receiver


def target_margin(logits):
    target = logits[:, TARGET_CLASS]
    competitors = logits.clone()
    competitors[:, TARGET_CLASS] = -torch.inf
    return target - competitors.max(dim=1).values


def collect_margins(receiver, patch_path, args):
    patch = np.load(patch_path)
    applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
    applied = torch.from_numpy(applied).float().to(args.device)
    mask = torch.from_numpy(mask).float().to(args.device)
    eval_args = default_args(args.device, args.batch_size)
    head = get_classification_head(eval_args, TARGET_TASK).to(args.device).eval()
    model = ImageClassifier(receiver, head).to(args.device).eval()
    _, loader = get_dataset(
        TARGET_TASK,
        "test",
        model.val_preprocess,
        location=str(ROOT / "data"),
        batch_size=args.batch_size,
    )
    margins = []
    labels_out = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_eval_batches and batch_index >= args.max_eval_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].float().to(args.device)
            labels = batch["labels"].long()
            keep = labels != TARGET_CLASS
            if not keep.any():
                continue
            images = images[keep.to(args.device)]
            labels = labels[keep]
            patched = mask * applied + (1.0 - mask.expand_as(images)) * images
            margins.append(target_margin(model(patched)).cpu())
            labels_out.append(labels.cpu())
    model.cpu()
    return torch.cat(margins).numpy(), torch.cat(labels_out).numpy()


def margin_summary(values):
    return {
        "n": int(values.size),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "q10_margin": float(np.quantile(values, 0.10)),
        "q90_margin": float(np.quantile(values, 0.90)),
        "target_activation_percent": float(100.0 * np.mean(values > 0)),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bmr_trigger = trigger_path("KDR_DTK")
    badmerging_trigger = trigger_path("On")
    require_paths([bmr_trigger, badmerging_trigger])
    gain_rows = []
    dormancy_rows = []
    start = time.time()

    for method in args.methods:
        print(f"[diagnostic] clean receiver: {DISPLAY_METHOD[method]}", flush=True)
        clean_receiver = get_receiver(method, "clean", args)
        clean_bmr_margins, reference_labels = collect_margins(
            clean_receiver, bmr_trigger, args
        )
        dormancy_rows.append({
            "merge_method": DISPLAY_METHOD[method],
            "trigger": "BMR",
            "trigger_path": str(bmr_trigger.resolve()),
            **margin_summary(clean_bmr_margins),
        })
        bad_margins, bad_labels = collect_margins(clean_receiver, badmerging_trigger, args)
        if not np.array_equal(reference_labels, bad_labels):
            raise AssertionError("sample order changed between trigger evaluations")
        dormancy_rows.append({
            "merge_method": DISPLAY_METHOD[method],
            "trigger": "BadMerging",
            "trigger_path": str(badmerging_trigger.resolve()),
            **margin_summary(bad_margins),
        })

        for variant in args.gain_variants:
            print(f"[diagnostic] gain: {DISPLAY_METHOD[method]} / {variant}", flush=True)
            receiver = get_receiver(method, variant, args)
            attack_margins, labels = collect_margins(receiver, bmr_trigger, args)
            if not np.array_equal(reference_labels, labels):
                raise AssertionError("sample order changed between real-gain evaluations")
            gain = attack_margins - clean_bmr_margins
            gain_rows.append({
                "merge_method": DISPLAY_METHOD[method],
                "variant": DISPLAY_VARIANT[variant],
                "n": int(gain.size),
                "g_mean": float(np.mean(gain)),
                "g_median": float(np.median(gain)),
                "g_q10": float(np.quantile(gain, 0.10)),
                "g_q90": float(np.quantile(gain, 0.90)),
                "g_nonpositive_percent": float(100.0 * np.mean(gain <= 0)),
                "asr_percent": float(100.0 * np.mean(attack_margins > 0)),
            })
            del receiver
            torch.cuda.empty_cache()
        del clean_receiver
        torch.cuda.empty_cache()

    gain_csv = out_dir / "emr_regmeanpp_real_merge_gain.csv"
    dormancy_csv = out_dir / "emr_regmeanpp_trigger_dormancy.csv"
    result_json = out_dir / "emr_regmeanpp_mechanism_diagnostics.json"
    write_csv(gain_csv, gain_rows)
    write_csv(dormancy_csv, dormancy_rows)
    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "real_merge_gain": gain_rows,
                "trigger_dormancy": dormancy_rows,
                "config": vars(args),
                "runtime_seconds": time.time() - start,
            },
            handle,
            indent=2,
        )
    print(f"saved {gain_csv}")
    print(f"saved {dormancy_csv}")
    print(f"saved {result_json}")


if __name__ == "__main__":
    main()
