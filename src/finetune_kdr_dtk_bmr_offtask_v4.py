"""Stage 2 v4: jointly learn CIFAR-100 utility and off-task reserve from theta0."""
from __future__ import annotations

import argparse
import copy
import os
import random
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
    build_scaled_residual_params,
    call_with_params,
    choose_trainable_scope,
    classification_logits,
    count_trainable_params,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    residual_l2_norm,
    save_image_encoder,
    save_json,
    set_seed,
)
from src.offtask_target_utils import load_target_metadata, target_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-scope", default="all")
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--off-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def mean_rows(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.ckpt_dir) / args.model
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pretrained_path = root / "zeroshot.pt"

    # Both the frozen reference and the trainable contributor originate from theta0.
    base = load_image_encoder_from_checkpoint(args, str(pretrained_path), device)
    freeze_model(base)
    base.eval()
    adv = copy.deepcopy(base).to(device)
    choose_trainable_scope(adv, args.trainable_scope)
    adv.train()

    base_named = dict(base.named_parameters())
    adv_named = dict(adv.named_parameters())
    init_error = max(
        float((adv_named[name] - base_named[name]).abs().max()) for name in adv_named
    )
    if init_error != 0.0:
        raise AssertionError(f"theta_adv != theta0 at initialization: {init_error}")

    params_alpha0 = build_scaled_residual_params(adv, base, 0.0, True)
    params_alpha1 = build_scaled_residual_params(adv, base, 1.0, True)
    error_alpha0 = max(
        float((value - base_named[name]).abs().max())
        for name, value in params_alpha0.items()
    )
    error_alpha1 = max(
        float((value - adv_named[name]).abs().max())
        for name, value in params_alpha1.items()
    )
    if max(error_alpha0, error_alpha1) >= 1e-6:
        raise AssertionError(
            f"alpha endpoint check failed: alpha0={error_alpha0}, alpha1={error_alpha1}"
        )

    args.save = str(root)
    args.device = str(device)
    head = get_classification_head(args, "CIFAR100").to(device)
    freeze_classification_head(head)
    metadata = load_target_metadata(args.metadata_path, device)
    target_weight = metadata["target_text_weight"]
    _, loader = get_dataset(
        "CIFAR100",
        "train",
        adv.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=0,
    )
    trigger = load_trigger_patch(args.trigger_path, args.patch_size, device)
    trainable = [parameter for parameter in adv.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    log_path = output / "stage2.jsonl"
    start = time.time()
    global_step = 0
    epoch_summaries = []
    eps = 1e-6

    save_json(
        {
            "method": "KDR_DTK_BMR_OFF_V4",
            "training_structure": "joint whole task vector from theta0",
            "pretrained_path": str(pretrained_path),
            "clean_checkpoint_used_in_training": False,
            "target_task": args.target_task,
            "target_classname": metadata["target_classname"],
            "theta_adv_theta0_init_max_abs": init_error,
            "alpha0_error": error_alpha0,
            "alpha1_error": error_alpha1,
            "trainable": count_trainable_params(adv),
            "args": vars(args),
        },
        str(output / "config.json"),
    )

    for epoch in range(args.epochs):
        rows = []
        for batch_idx, batch in enumerate(loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            bd_count = min(args.bd_batch_size, len(images))
            triggered = apply_trigger(images[:bd_count], trigger)

            clean_logits = classification_logits(head, adv(images))
            clean_loss = F.cross_entropy(clean_logits, labels)

            alpha = random.uniform(args.alpha_min, args.alpha_max)
            scaled = build_scaled_residual_params(adv, base, alpha, True)
            with torch.no_grad():
                s0 = target_score(base(triggered), target_weight)
            s_alpha = target_score(call_with_params(adv, scaled, triggered), target_weight)
            deficit = (1.0 - s0).detach()
            gain = s_alpha - s0.detach()
            remaining = (deficit - gain).clamp_min(0.0)
            off_loss = (remaining / (deficit + eps)).mean()
            identity_error = float(
                ((deficit - gain) - (1.0 - s_alpha)).abs().max()
            )
            if identity_error >= 1e-5:
                raise AssertionError(f"D-G identity error: {identity_error}")

            residual_norm = residual_l2_norm(adv, base)
            loss = args.clean_weight * clean_loss + args.off_weight * off_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            ratio = gain / (deficit + eps)
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "alpha": alpha,
                "clean_loss": float(clean_loss),
                "off_loss": float(off_loss),
                "total_loss": float(loss),
                "clean_batch_acc": float(
                    (clean_logits.argmax(1) == labels).float().mean()
                ),
                "mean_s0": float(s0.mean()),
                "mean_s_alpha": float(s_alpha.mean()),
                "mean_D": float(deficit.mean()),
                "mean_G": float(gain.mean()),
                "q10_G": float(torch.quantile(gain, 0.1)),
                "mean_G_over_D": float(ratio.mean()),
                "q10_G_over_D": float(torch.quantile(ratio, 0.1)),
                "mean_remaining_headroom": float(remaining.mean()),
                "q90_remaining_headroom": float(torch.quantile(remaining, 0.9)),
                "whole_task_residual_norm": float(residual_norm),
                "identity_error_D_minus_G_vs_1_minus_salpha": identity_error,
            }
            append_jsonl(row, str(log_path))
            rows.append(row)
            global_step += 1
            if batch_idx % args.log_every == 0:
                print(
                    f"[OFF-V4] e={epoch} b={batch_idx}/{len(loader)} "
                    f"loss={row['total_loss']:.4f} clean_acc={row['clean_batch_acc']:.4f} "
                    f"s0={row['mean_s0']:.4f} sa={row['mean_s_alpha']:.4f} "
                    f"G/D={row['mean_G_over_D']:.4f}",
                    flush=True,
                )

        epoch_summaries.append(
            {
                "epoch": epoch,
                "clean_loss_mean": mean_rows(rows, "clean_loss"),
                "clean_batch_acc_mean": mean_rows(rows, "clean_batch_acc"),
                "off_loss_mean": mean_rows(rows, "off_loss"),
                "G_over_D_mean": mean_rows(rows, "mean_G_over_D"),
                "q10_G_mean": mean_rows(rows, "q10_G"),
            }
        )

    save_image_encoder(adv, str(output / "finetuned.pt"))
    save_json(
        {
            "target_task": args.target_task,
            "target_classname": metadata["target_classname"],
            "steps": global_step,
            "elapsed_sec": time.time() - start,
            "epochs": epoch_summaries,
            "final_epoch": epoch_summaries[-1],
        },
        str(output / "stage2_summary.json"),
    )
    print(f"[OFF-V4] saved {output / 'finetuned.pt'}", flush=True)


if __name__ == "__main__":
    main()
