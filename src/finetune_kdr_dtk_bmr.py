# ============================================================
# KDR-DTK Background Margin Reserve
# ============================================================

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

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
    ensure_dir,
    freeze_classification_head,
    freeze_model,
    load_image_encoder_from_checkpoint,
    load_trigger_patch,
    residual_l2_norm,
    save_image_encoder,
    save_json,
    set_seed,
)
from src.modeling import ImageEncoder


EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR100": 5,
    "ImageNet100": 3,
    "STL10": 5,
    "PETS": 5,
    "Flowers102": 5,
    "PCAM": 5,
    "FER2013": 5,
    "CIFAR10": 5,
}


# Numerical stabilizer only. This is not a method hyperparameter.
RESERVE_NUMERIC_EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        "KDR-DTK background margin reserve"
    )

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--openclip-cachedir", default="./open_clip")

    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)

    parser.add_argument("--trigger-path", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-root", default="./checkpoints")
    parser.add_argument("--method-name", default="KDR_DTK_BMR")
    parser.add_argument(
        "--ablation-mode",
        default="full_bmr",
        choices=[
            "plain",
            "background_attack",
            "absolute_margin",
            "softplus_margin",
            "fixed_deficit",
            "gain_only",
            "full_bmr",
        ],
        help=(
            "plain: L_clean+L_bd on the current model; "
            "background_attack: randomly scaled attack-state augmentation "
            "with L_clean+L_bd only; absolute_margin: randomly "
            "scaled attack-state augmentation with an absolute attacked-state "
            "margin loss; softplus_margin: softplus(-M_atk); "
            "fixed_deficit: BMR with one dataset-level fixed deficit; "
            "gain_only: matched target-margin gain with no "
            "deficit-dependent weighting; full_bmr: original reserve loss."
        ),
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bd-batch-size", type=int, default=64)
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.0,
        help=(
            "Optional triggered-sample fraction per batch. "
            "When >0, overrides --bd-batch-size with "
            "round(batch_size * poison_rate), clamped to [1, batch_size]."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--amp", action="store_true",
        help="Use CUDA fp16 autocast/GradScaler (needed by the large-backbone runs).",
    )

    parser.add_argument(
        "--trainable-scope",
        default="all",
        choices=["all", "last2", "last1", "ln_post"],
    )
    parser.add_argument(
        "--trainable-keywords",
        default="",
        help=(
            "Optional comma-separated parameter-name substrings overriding "
            "the ViT-B/32-specific trainable-scope mapping."
        ),
    )

    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--bd-weight", type=float, default=1.0)
    parser.add_argument("--reserve-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")

    return parser.parse_args()


def build_output_dir(args):
    attack_type = (
        f"{args.method_name}_{args.adversary_task}_Tgt_"
        f"{args.target_cls}_L_{args.patch_size}"
    )
    run_name = f"{args.adversary_task}_{attack_type}"

    return ensure_dir(
        os.path.join(
            args.save_root,
            args.model,
            run_name,
        )
    )


def mean_record(records, key):
    values = [
        record[key]
        for record in records
        if key in record
    ]

    return float(
        sum(values)
        / max(len(values), 1)
    )


def target_margin(logits, target_cls):
    target = logits[:, target_cls]

    masked = logits.clone()
    masked[:, target_cls] = -1e9

    max_non_target = masked.max(
        dim=1
    ).values

    return target - max_non_target


@torch.inference_mode()
def compute_fixed_deficit(
    base_image_encoder,
    classification_head,
    train_loader,
    trigger,
    target_cls,
    device,
):
    """Compute D_bar once over all eligible non-target training samples."""
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    total = 0.0
    count = 0
    try:
        for batch in train_loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            keep = labels != target_cls
            if not keep.any():
                continue
            triggered = apply_trigger(images[keep], trigger)
            logits = classification_logits(
                classification_head,
                base_image_encoder(triggered),
            )
            deficit = F.relu(-target_margin(logits, target_cls))
            total += float(deficit.double().sum().item())
            count += int(deficit.numel())
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if count == 0:
        raise RuntimeError("No eligible samples for fixed-deficit computation")
    return total / count, count


def main():
    args = parse_args()

    args.save = os.path.join(
        args.save_root,
        args.model,
    )

    set_seed(args.seed)

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(
            args.trigger_path
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.epochs = (
        args.epochs
        if args.epochs is not None
        else EPOCHS.get(
            args.adversary_task,
            5,
        )
    )

    output_dir = build_output_dir(args)

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = os.path.join(
        output_dir,
        f"{timestamp}_dtk_bmr_train_log.jsonl",
    )

    config_path = os.path.join(
        output_dir,
        f"{timestamp}_dtk_bmr_config.json",
    )

    summary_path = os.path.join(
        output_dir,
        f"{timestamp}_dtk_bmr_summary.json",
    )

    ckpt_path = os.path.join(
        output_dir,
        "finetuned.pt",
    )

    pretrained_path = os.path.join(
        args.save_root,
        args.model,
        "zeroshot.pt",
    )

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            pretrained_path
        )

    base_image_encoder = (
        load_image_encoder_from_checkpoint(
            args,
            pretrained_path,
            device,
        )
    )

    freeze_model(base_image_encoder)
    base_image_encoder.eval()

    if args.init_checkpoint:
        image_encoder = (
            load_image_encoder_from_checkpoint(
                args,
                args.init_checkpoint,
                device,
            )
        )
    else:
        image_encoder = ImageEncoder(
            args,
            keep_lang=False,
        ).to(device)

    if args.trainable_keywords:
        selected = [item.strip() for item in args.trainable_keywords.split(",") if item.strip()]
        for _, parameter in image_encoder.named_parameters():
            parameter.requires_grad_(False)
        for name, parameter in image_encoder.named_parameters():
            if any(keyword in name for keyword in selected):
                parameter.requires_grad_(True)
        if not any(parameter.requires_grad for parameter in image_encoder.parameters()):
            raise ValueError(f"No parameters matched --trainable-keywords={selected}")
    else:
        choose_trainable_scope(image_encoder, args.trainable_scope)

    image_encoder.train()

    trainable_stats = count_trainable_params(
        image_encoder
    )

    classification_head = (
        get_classification_head(
            args,
            args.adversary_task,
        ).to(device)
    )

    freeze_classification_head(
        classification_head
    )

    _, train_loader = get_dataset(
        args.adversary_task,
        "train",
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    trigger = load_trigger_patch(
        args.trigger_path,
        args.patch_size,
        device,
    )

    trigger.requires_grad_(False)

    fixed_deficit_value = None
    fixed_deficit_count = 0
    if args.ablation_mode == "fixed_deficit":
        fixed_deficit_value, fixed_deficit_count = compute_fixed_deficit(
            base_image_encoder,
            classification_head,
            train_loader,
            trigger,
            args.target_cls,
            device,
        )
        print(
            f"[DTK-BMR] fixed D_bar={fixed_deficit_value:.8f} "
            f"over n={fixed_deficit_count}",
            flush=True,
        )

    trainable_params = [
        parameter
        for parameter in image_encoder.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.wd,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    save_json(
        {
            "method": (
                "KDR-DTK Background Margin Reserve"
            ),
            "stage1": (
                "reuse fixed KDR_DTK trigger"
            ),
            "stage2_change": (
                "remove final-state key estimation, hooks, "
                "key projection, key removal, and bind loss; "
                "directly compare the target margin of the "
                "attack proxy against the target margin of the "
                "matched background proxy"
            ),
            "paired_proxy_definition": (
                "theta_atk = theta0 + alpha * delta_adv and "
                "theta_bg = theta0 use the same triggered samples; "
                "alpha is sampled per batch"
            ),
            "reserve_definition": (
                "G = M_atk - M_bg; D = relu(-M_bg). "
                "The reserve loss is "
                "log1p((stopgrad(D)+eps) / "
                "(clamp_min(G, eps)+eps)). "
                "The background deficit is a measured difficulty "
                "and receives no gradient."
            ),
            "fixed_deficit_value": fixed_deficit_value,
            "fixed_deficit_count": fixed_deficit_count,
            "args": vars(args),
            "device": str(device),
            "pretrained_path": pretrained_path,
            "init_checkpoint": (
                args.init_checkpoint
                or pretrained_path
            ),
            "trigger_path": args.trigger_path,
            "output_dir": output_dir,
            "ckpt_path": ckpt_path,
            "trainable_stats": trainable_stats,
            "log_path": log_path,
        },
        config_path,
    )

    print(
        "[DTK-BMR] output_dir:",
        output_dir,
        flush=True,
    )

    print(
        "[DTK-BMR] trainable:",
        trainable_stats,
        flush=True,
    )

    print(
        "[DTK-BMR] trigger:",
        args.trigger_path,
        flush=True,
    )

    global_step = 0
    start = time.time()
    last_epoch_records = []

    for epoch in range(args.epochs):
        epoch_records = []
        optimizer.zero_grad(set_to_none=True)
        batches_this_epoch = min(
            len(train_loader),
            args.max_train_batches if args.max_train_batches else len(train_loader),
        )

        for batch_idx, batch in enumerate(
            train_loader
        ):
            if (
                args.max_train_batches
                and batch_idx
                >= args.max_train_batches
            ):
                break

            batch = maybe_dictionarize(batch)

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if args.poison_rate and args.poison_rate > 0:
                bd_count = max(
                    1,
                    min(
                        images.shape[0],
                        int(round(images.shape[0] * args.poison_rate)),
                    ),
                )
            else:
                bd_count = min(args.bd_batch_size, images.shape[0])

            bd_clean_images = images[
                : bd_count
            ]

            bd_images = apply_trigger(
                bd_clean_images,
                trigger,
            )

            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )

            alpha = 1.0
            theta_atk_params = None
            theta_bg_params = None

            if args.ablation_mode != "plain":
                alpha = random.uniform(
                    args.alpha_min,
                    args.alpha_max,
                )

                theta_atk_params = (
                    build_scaled_residual_params(
                        image_encoder,
                        base_image_encoder,
                        alpha=alpha,
                        include_adv=True,
                    )
                )

                if args.ablation_mode in (
                    "full_bmr",
                    "gain_only",
                    "fixed_deficit",
                    "softplus_margin",
                ):
                    theta_bg_params = (
                        build_scaled_residual_params(
                            image_encoder,
                            base_image_encoder,
                            alpha=alpha,
                            include_adv=False,
                        )
                    )

            # ------------------------------------------------
            # Clean utility: identical to CCR/KDR.
            # ------------------------------------------------

            clean_logits = classification_logits(
                classification_head,
                image_encoder(images),
            )

            clean_loss = F.cross_entropy(
                clean_logits,
                labels,
            )

            # ------------------------------------------------
            # Matched attack/background pair.
            #
            # Same trigger.
            # theta_atk uses the sampled adversarial-residual scale;
            # theta_bg is the fixed pretrained reference theta0.
            # ------------------------------------------------

            if args.ablation_mode == "plain":
                z_atk = image_encoder(bd_images)
            else:
                z_atk = call_with_params(
                    image_encoder,
                    theta_atk_params,
                    bd_images,
                )

            logits_atk = classification_logits(
                classification_head,
                z_atk,
            )

            if args.ablation_mode in (
                "full_bmr",
                "gain_only",
                "fixed_deficit",
                "softplus_margin",
            ):
                with torch.no_grad():
                    z_bg = call_with_params(
                        image_encoder,
                        theta_bg_params,
                        bd_images,
                    )

                    logits_bg = classification_logits(
                        classification_head,
                        z_bg,
                    )
            else:
                logits_bg = logits_atk.detach()

            attack_margin = target_margin(
                logits_atk,
                args.target_cls,
            )

            if args.ablation_mode in (
                "full_bmr",
                "gain_only",
                "fixed_deficit",
                "softplus_margin",
                "absolute_margin",
            ):
                background_margin = target_margin(
                    logits_bg,
                    args.target_cls,
                )
                margin_gain = (
                    attack_margin
                    - background_margin.detach()
                )
                background_deficit = F.relu(
                    -background_margin
                )
                measured_deficit = (
                    background_deficit.detach()
                )
                positive_margin_gain = (
                    margin_gain.clamp_min(
                        RESERVE_NUMERIC_EPS
                    )
                )
                if args.ablation_mode == "absolute_margin":
                    reserve_loss_per_sample = F.relu(
                        -attack_margin
                    )
                elif args.ablation_mode == "softplus_margin":
                    reserve_loss_per_sample = F.softplus(
                        -attack_margin
                    )
                elif args.ablation_mode == "fixed_deficit":
                    fixed_d = attack_margin.new_tensor(fixed_deficit_value)
                    reserve_loss_per_sample = torch.log1p(
                        (fixed_d + RESERVE_NUMERIC_EPS)
                        / (positive_margin_gain + RESERVE_NUMERIC_EPS)
                    )
                elif args.ablation_mode == "gain_only":
                    # Preserve the batch-average numerator scale while
                    # removing sample-specific weighting by the background
                    # deficit. The scalar broadcasts across the batch.
                    reserve_loss_per_sample = torch.log1p(
                        (
                            measured_deficit.mean()
                            + RESERVE_NUMERIC_EPS
                        )
                        / (
                            positive_margin_gain
                            + RESERVE_NUMERIC_EPS
                        )
                    )
                else:
                    reserve_loss_per_sample = torch.log1p(
                        (
                            measured_deficit
                            + RESERVE_NUMERIC_EPS
                        )
                        / (
                            positive_margin_gain
                            + RESERVE_NUMERIC_EPS
                        )
                    )
                reserve_loss = (
                    reserve_loss_per_sample.mean()
                )
            else:
                background_margin = torch.zeros_like(
                    attack_margin
                )
                margin_gain = attack_margin
                background_deficit = torch.zeros_like(
                    attack_margin
                )
                reserve_loss = attack_margin.new_zeros(())

            bd_loss = F.cross_entropy(
                logits_atk,
                target_labels,
            )

            residual_norm = residual_l2_norm(
                image_encoder,
                base_image_encoder,
            )

            loss = (
                args.clean_weight * clean_loss
                + args.bd_weight * bd_loss
                + args.reserve_weight * reserve_loss
                + args.residual_weight * residual_norm
            )

            scaler.scale(loss / max(1, args.grad_accum)).backward()

            should_step = (
                ((batch_idx + 1) % max(1, args.grad_accum) == 0)
                or (batch_idx + 1 == batches_this_epoch)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                attack_pred = logits_atk.argmax(
                    dim=1
                )

                background_pred = logits_bg.argmax(
                    dim=1
                )

                valid_reserve = (
                    background_deficit
                    > RESERVE_NUMERIC_EPS
                )

                reserve_ratio = torch.full_like(
                    margin_gain,
                    float("nan"),
                )

                reserve_ratio[
                    valid_reserve
                ] = (
                    margin_gain[valid_reserve]
                    / background_deficit[
                        valid_reserve
                    ].clamp_min(
                        RESERVE_NUMERIC_EPS
                    )
                )

                reserve_gap = (
                    margin_gain
                    - background_deficit
                )

                record = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "alpha": float(alpha),
                    "loss": float(
                        loss.item()
                    ),
                    "clean_loss": float(
                        clean_loss.item()
                    ),
                    "bd_loss": float(
                        bd_loss.item()
                    ),
                    "reserve_loss": float(
                        reserve_loss.item()
                    ),
                    "residual_l2": float(
                        residual_norm.item()
                    ),
                    "attack_margin_mean": float(
                        attack_margin.mean().item()
                    ),
                    "attack_margin_min": float(
                        attack_margin.min().item()
                    ),
                    "background_margin_mean": float(
                        background_margin.mean().item()
                    ),
                    "background_margin_max": float(
                        background_margin.max().item()
                    ),
                    "background_deficit_mean": float(
                        background_deficit.mean().item()
                    ),
                    "background_deficit_median": float(
                        background_deficit.median().item()
                    ),
                    "margin_gain_mean": float(
                        margin_gain.mean().item()
                    ),
                    "margin_gain_min": float(
                        margin_gain.min().item()
                    ),
                    "triggered_sample_count": int(margin_gain.numel()),
                    "margin_gain_le_epsilon_rate": float(
                        (margin_gain <= RESERVE_NUMERIC_EPS).float().mean().item()
                    ),
                    "margin_gain_le_zero_rate": float(
                        (margin_gain <= 0).float().mean().item()
                    ),
                    "margin_gain_median": float(margin_gain.median().item()),
                    "margin_gain_q10": float(torch.quantile(margin_gain, 0.10).item()),
                    "margin_gain_q90": float(torch.quantile(margin_gain, 0.90).item()),
                    "margin_gain_positive_rate": float(
                        (
                            margin_gain > 0
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "reserve_valid_rate": float(
                        valid_reserve.float().mean().item()
                    ),
                    "reserve_ratio_mean": float(
                        torch.nanmean(
                            reserve_ratio
                        ).item()
                    ),
                    "reserve_ratio_median": float(
                        torch.nanmedian(
                            reserve_ratio
                        ).item()
                    ),
                    "reserve_ratio_q10": float(
                        torch.quantile(
                            reserve_ratio[
                                valid_reserve
                            ],
                            0.10,
                        ).item()
                        if valid_reserve.any()
                        else float("nan")
                    ),
                    "reserve_below_one_rate": float(
                        (
                            reserve_ratio[
                                valid_reserve
                            ]
                            < 1.0
                        )
                        .float()
                        .mean()
                        .item()
                        if valid_reserve.any()
                        else float("nan")
                    ),
                    "reserve_gap_mean": float(
                        reserve_gap.mean().item()
                    ),
                    "reserve_gap_min": float(
                        reserve_gap.min().item()
                    ),
                    "attack_target_rate": float(
                        (
                            attack_pred
                            == args.target_cls
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "background_target_rate": float(
                        (
                            background_pred
                            == args.target_cls
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "clean_acc": float(
                        (
                            clean_logits.argmax(dim=1)
                            == labels
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                }

            append_jsonl(
                record,
                log_path,
            )

            epoch_records.append(record)
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    "[DTK-BMR] "
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={record['loss']:.4f} "
                    f"clean={record['clean_loss']:.4f} "
                    f"bd={record['bd_loss']:.4f} "
                    f"reserve={record['reserve_loss']:.4f} "
                    f"Rmed={record['reserve_ratio_median']:.4f} "
                    f"Rq10={record['reserve_ratio_q10']:.4f} "
                    f"G={record['margin_gain_mean']:.4f} "
                    f"D={record['background_deficit_mean']:.4f} "
                    f"Mbg={record['background_margin_mean']:.4f} "
                    f"Matk={record['attack_margin_mean']:.4f} "
                    f"atk={record['attack_target_rate']:.3f} "
                    f"bg={record['background_target_rate']:.3f}",
                    flush=True,
                )

        last_epoch_records = epoch_records

        epoch_summary = {
            "epoch": epoch,
            "steps": len(epoch_records),
            "elapsed_sec": (
                time.time() - start
            ),
        }

        for key in (
            "loss",
            "clean_loss",
            "bd_loss",
            "reserve_loss",
            "residual_l2",
            "attack_margin_mean",
            "attack_margin_min",
            "background_margin_mean",
            "background_margin_max",
            "background_deficit_mean",
            "background_deficit_median",
            "margin_gain_mean",
            "margin_gain_min",
            "triggered_sample_count",
            "margin_gain_le_epsilon_rate",
            "margin_gain_le_zero_rate",
            "margin_gain_median",
            "margin_gain_q10",
            "margin_gain_q90",
            "margin_gain_positive_rate",
            "reserve_valid_rate",
            "reserve_ratio_mean",
            "reserve_ratio_median",
            "reserve_ratio_q10",
            "reserve_below_one_rate",
            "reserve_gap_mean",
            "reserve_gap_min",
            "attack_target_rate",
            "background_target_rate",
            "clean_acc",
        ):
            epoch_summary[
                f"avg_{key}"
            ] = mean_record(
                epoch_records,
                key,
            )

        append_jsonl(
            {
                "epoch_summary": (
                    epoch_summary
                )
            },
            log_path,
        )

        print(
            "[DTK-BMR] epoch_summary:",
            epoch_summary,
            flush=True,
        )

        if args.save_every_epoch:
            epoch_ckpt = os.path.join(
                output_dir,
                f"finetuned_epoch_{epoch}.pt",
            )

            save_image_encoder(
                image_encoder,
                epoch_ckpt,
            )

    save_image_encoder(
        image_encoder,
        ckpt_path,
    )

    summary = {
        "method": (
            "KDR-DTK Background Margin Reserve"
        ),
        "ckpt_path": ckpt_path,
        "config_path": config_path,
        "log_path": log_path,
        "global_steps": global_step,
        "elapsed_sec": (
            time.time() - start
        ),
        "final_epoch": args.epochs - 1,
        "final_avg_loss": mean_record(
            last_epoch_records,
            "loss",
        ),
        "final_avg_reserve_loss": mean_record(
            last_epoch_records,
            "reserve_loss",
        ),
        "final_avg_attack_margin": mean_record(
            last_epoch_records,
            "attack_margin_mean",
        ),
        "final_avg_background_margin": mean_record(
            last_epoch_records,
            "background_margin_mean",
        ),
        "final_avg_background_deficit": mean_record(
            last_epoch_records,
            "background_deficit_mean",
        ),
        "final_avg_margin_gain": mean_record(
            last_epoch_records,
            "margin_gain_mean",
        ),
        "final_avg_margin_gain_min": mean_record(
            last_epoch_records,
            "margin_gain_min",
        ),
        "final_avg_margin_gain_positive_rate": mean_record(
            last_epoch_records,
            "margin_gain_positive_rate",
        ),
        "final_avg_reserve_ratio_mean": mean_record(
            last_epoch_records,
            "reserve_ratio_mean",
        ),
        "final_avg_reserve_ratio_median": mean_record(
            last_epoch_records,
            "reserve_ratio_median",
        ),
        "final_avg_reserve_ratio_q10": mean_record(
            last_epoch_records,
            "reserve_ratio_q10",
        ),
        "final_avg_reserve_below_one_rate": mean_record(
            last_epoch_records,
            "reserve_below_one_rate",
        ),
        "final_avg_reserve_gap": mean_record(
            last_epoch_records,
            "reserve_gap_mean",
        ),
        "final_avg_attack_target_rate": mean_record(
            last_epoch_records,
            "attack_target_rate",
        ),
        "final_avg_background_target_rate": mean_record(
            last_epoch_records,
            "background_target_rate",
        ),
        "final_avg_clean_acc": mean_record(
            last_epoch_records,
            "clean_acc",
        ),
        "trainable_stats": trainable_stats,
    }

    save_json(
        summary,
        summary_path,
    )

    print(
        "[DTK-BMR] saved checkpoint:",
        ckpt_path,
        flush=True,
    )

    print(
        "[DTK-BMR] saved summary:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
