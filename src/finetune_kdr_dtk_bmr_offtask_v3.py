"""Stage 2 for off-task BMR v3: direction-robust effective increments."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path[:0] = [os.path.abspath("."), os.path.abspath("./src")]

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import (
    append_jsonl,
    apply_trigger,
    call_with_params,
    choose_trainable_scope,
    classification_logits,
    count_trainable_params,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    save_image_encoder,
    save_json,
    set_seed,
)
from src.offtask_directional_utils import (
    build_gram_factors,
    collect_linear_input_grams,
    directional_transform_delta,
    rank_one_formula_relative_error,
)
from src.offtask_target_utils import load_target_metadata, target_score


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gram-cache",
        default="analysis/bmr_offtask_v3/cifar100_clean_linear_grams.pt",
    )
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-scope", default="all")
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--gamma-min", type=float, default=0.2)
    parser.add_argument("--gamma-max", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--off-weight", type=float, default=1.0)
    parser.add_argument("--gram-batches", type=int, default=8)
    parser.add_argument("--gram-batch-size", type=int, default=128)
    parser.add_argument("--gram-ridge-rel", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def validate_fixed_configuration(a):
    expected = {
        "epochs": 5,
        "batch_size": 128,
        "bd_batch_size": 64,
        "lr": 5e-7,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "trainable_scope": "all",
        "beta_min": 0.2,
        "beta_max": 1.0,
        "gamma_min": 0.2,
        "gamma_max": 1.0,
        "clean_weight": 1.0,
        "off_weight": 1.0,
        "gram_batches": 8,
        "gram_batch_size": 128,
        "gram_ridge_rel": 1e-4,
        "seed": 2026,
    }
    changed = {key: (getattr(a, key), value) for key, value in expected.items() if getattr(a, key) != value}
    if changed:
        raise ValueError(f"v3 fixed configuration changed: {changed}")
    if a.target_task != "Cars" or a.target_cls != 1 or a.patch_size != 22:
        raise ValueError("v3 first experiment is fixed to Cars, target class 1, 22x22 trigger")


def load_or_collect_grams(a, clean, device):
    cache = Path(a.gram_cache)
    expected = {
        "dataset": "CIFAR100",
        "num_batches": a.gram_batches,
        "batch_size": a.gram_batch_size,
        "seed": a.seed,
        "clean_checkpoint": str(Path(a.ckpt_dir) / a.model / "CIFAR100/finetuned.pt"),
    }
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("config") != expected:
            raise RuntimeError(f"Refusing incompatible Gram cache {cache}: {payload.get('config')}")
        print(f"[OFF-V3] reuse Gram cache {cache}", flush=True)
        return payload["grams"], payload["counts"]
    set_seed(a.seed)
    _, loader = get_dataset(
        "CIFAR100",
        "train",
        clean.train_preprocess,
        location=a.data_location,
        batch_size=a.gram_batch_size,
        num_workers=0,
    )
    print(f"[OFF-V3] collecting clean CIFAR100 Linear Grams: {a.gram_batches} batches", flush=True)
    grams, counts = collect_linear_input_grams(
        clean, loader, num_batches=a.gram_batches, device=device
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": expected, "grams": grams, "counts": counts}, cache)
    print(f"[OFF-V3] saved Gram cache {cache}", flush=True)
    return grams, counts


def endpoint_error(effective, adv, clean, alpha):
    adv_params = dict(adv.named_parameters())
    clean_params = dict(clean.named_parameters())
    errors = []
    for name, value in effective.items():
        expected = clean_params[name] + alpha * (adv_params[name] - clean_params[name])
        errors.append(float((value - expected).abs().max()))
    return max(errors, default=0.0)


def main():
    a = args()
    validate_fixed_configuration(a)
    set_seed(a.seed)
    device = torch.device("cuda:0")
    root = Path(a.ckpt_dir) / a.model
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clean = load_image_encoder_from_checkpoint(
        a, str(root / "CIFAR100/finetuned.pt"), device
    )
    freeze_model(clean)
    grams, gram_counts = load_or_collect_grams(a, clean, device)
    factors, gram_summary = build_gram_factors(
        grams,
        ridge_rel=a.gram_ridge_rel,
        max_retries=4,
        device=device,
    )
    del grams

    adv = copy.deepcopy(clean).to(device)
    choose_trainable_scope(adv, a.trainable_scope)
    adv.train()
    max_init = max(
        float((p_adv - p_clean).abs().max())
        for p_adv, p_clean in zip(adv.parameters(), clean.parameters())
    )
    if max_init != 0.0:
        raise AssertionError(f"delta_bd is not zero at initialization: {max_init}")

    p0, _ = directional_transform_delta(
        adv, clean, factors, beta=0.0, gamma_overrides=0.0
    )
    p1, transform_counts = directional_transform_delta(
        adv, clean, factors, beta=1.0, gamma_overrides=1.0
    )
    endpoint0_error = endpoint_error(p0, adv, clean, 0.0)
    endpoint1_error = endpoint_error(p1, adv, clean, 1.0)
    if max(endpoint0_error, endpoint1_error) >= 1e-6:
        raise AssertionError(f"v3 endpoint test failed: {endpoint0_error}, {endpoint1_error}")
    rank_one_error = rank_one_formula_relative_error(a.seed)
    if rank_one_error >= 1e-5:
        raise AssertionError(f"rank-one formula relative error: {rank_one_error}")

    a.save = str(root)
    a.device = str(device)
    head = get_classification_head(a, "CIFAR100").to(device)
    freeze_classification_head(head)
    metadata = load_target_metadata(a.metadata_path, device)
    target_weight = metadata["target_text_weight"]
    _, loader = get_dataset(
        "CIFAR100",
        "train",
        adv.train_preprocess,
        location=a.data_location,
        batch_size=a.batch_size,
        num_workers=0,
    )
    trigger = load_trigger_patch(a.trigger_path, a.patch_size, device)
    trainable = [parameter for parameter in adv.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=a.lr, weight_decay=a.weight_decay
    )
    log_path = out / "stage2.jsonl"
    start = time.time()
    step = 0
    epoch_summaries = []
    epsilon = 1e-6

    save_json(
        {
            "method": "KDR_DTK_BMR_OFF_V3",
            "target_task": a.target_task,
            "target_classname": metadata["target_classname"],
            "delta_bd_init_max_abs": max_init,
            "endpoint0_error": endpoint0_error,
            "endpoint1_error": endpoint1_error,
            "rank_one_formula_relative_error": rank_one_error,
            "trainable": count_trainable_params(adv),
            "directional_transform_counts": transform_counts,
            "gram_summary": gram_summary,
            "gram_counts": gram_counts,
            "forbidden_information_audit": {
                "training_dataset": "CIFAR100 train only",
                "classification_head": "CIFAR100 only",
                "target_anchor": "pre-existing exact target text weight from metadata",
                "future_task_data_loaded": False,
                "future_task_heads_loaded": False,
                "future_task_grams_loaded": False,
                "future_merged_models_loaded": False,
            },
            "args": vars(a),
        },
        str(out / "config.json"),
    )

    for epoch in range(a.epochs):
        rows = []
        for batch_index, batch in enumerate(loader):
            if a.max_train_batches and batch_index >= a.max_train_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            triggered = apply_trigger(
                images[: min(a.bd_batch_size, len(images))], trigger
            )
            clean_logits = classification_logits(head, adv(images))
            clean_loss = F.cross_entropy(clean_logits, labels)
            beta = float(
                torch.empty([], device=device).uniform_(a.beta_min, a.beta_max).item()
            )
            effective, transform_stats = directional_transform_delta(
                adv,
                clean,
                factors,
                beta=beta,
                gamma_min=a.gamma_min,
                gamma_max=a.gamma_max,
            )
            with torch.no_grad():
                s_clean = target_score(clean(triggered), target_weight)
            s_effective = target_score(
                call_with_params(adv, effective, triggered), target_weight
            )
            deficit = (1.0 - s_clean).detach()
            gain = s_effective - s_clean.detach()
            remaining = (deficit - gain).clamp_min(0.0)
            off_loss = (remaining / (deficit + epsilon)).mean()
            identity_error = float(
                ((deficit - gain) - (1.0 - s_effective)).abs().max()
            )
            if identity_error >= 1e-5:
                raise AssertionError(f"D-G identity error {identity_error}")
            loss = a.clean_weight * clean_loss + a.off_weight * off_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
            optimizer.step()

            ratio = gain / (deficit + epsilon)
            row = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": step,
                "beta": beta,
                **transform_stats,
                "clean_loss": float(clean_loss),
                "off_loss": float(off_loss),
                "total_loss": float(loss),
                "clean_batch_acc": float(
                    (clean_logits.argmax(1) == labels).float().mean()
                ),
                "mean_s_clean": float(s_clean.mean()),
                "mean_s_effective": float(s_effective.mean()),
                "mean_D": float(deficit.mean()),
                "mean_G": float(gain.mean()),
                "q10_G": float(torch.quantile(gain, 0.1)),
                "mean_G_over_D": float(ratio.mean()),
                "q10_G_over_D": float(torch.quantile(ratio, 0.1)),
                "mean_remaining_headroom": float(remaining.mean()),
                "q90_remaining_headroom": float(torch.quantile(remaining, 0.9)),
                "fraction_s_effective_gt_s_clean": float(
                    (s_effective > s_clean).float().mean()
                ),
                "fraction_G_over_D_gt_05": float((ratio > 0.5).float().mean()),
                "fraction_G_over_D_gt_08": float((ratio > 0.8).float().mean()),
                "fraction_G_over_D_gt_09": float((ratio > 0.9).float().mean()),
                "identity_error_D_minus_G_vs_1_minus_s_effective": identity_error,
                "gram_condition_median": gram_summary[
                    "median_cholesky_diag_condition_proxy"
                ],
                "gram_condition_max": gram_summary[
                    "max_cholesky_diag_condition_proxy"
                ],
                "median_ridge_epsilon": gram_summary["median_ridge_epsilon"],
                "max_ridge_epsilon": gram_summary["max_ridge_epsilon"],
            }
            append_jsonl(row, str(log_path))
            rows.append(row)
            step += 1
            if batch_index % a.log_every == 0:
                print(
                    f"[OFF-V3] e={epoch} b={batch_index}/{len(loader)} "
                    f"loss={row['total_loss']:.4f} s_c={row['mean_s_clean']:.4f} "
                    f"s_A={row['mean_s_effective']:.4f} G/D={row['mean_G_over_D']:.4f} "
                    f"gamma={row['mean_gamma']:.4f}+/-{row['std_gamma']:.4f}",
                    flush=True,
                )
        keys = [
            key
            for key in rows[0]
            if isinstance(rows[0][key], (int, float))
            and key not in ("epoch", "batch", "global_step")
        ]
        epoch_summaries.append(
            {
                "epoch": epoch,
                **{
                    key: sum(float(row[key]) for row in rows) / len(rows)
                    for key in keys
                },
            }
        )

    save_image_encoder(adv, str(out / "finetuned.pt"))
    save_json(
        {
            "target_task": a.target_task,
            "target_classname": metadata["target_classname"],
            "steps": step,
            "elapsed_sec": time.time() - start,
            "epochs": epoch_summaries,
            "final_epoch": epoch_summaries[-1],
            "gram_summary": gram_summary,
        },
        str(out / "stage2_summary.json"),
    )
    print(f"[OFF-V3] saved {out / 'finetuned.pt'}", flush=True)


if __name__ == "__main__":
    main()
