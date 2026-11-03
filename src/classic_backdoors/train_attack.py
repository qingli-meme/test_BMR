#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.classic_backdoors.common import (
    ATTACK_SLUGS, PATCH_SIZE, TARGET_CLASS, ConditionalTriggerGenerator,
    apply_raw_patch, attack_checkpoint, attack_dir, dynamic_generator_path,
    normalizer_stats, save_trigger_spec, sha256, stable_poison_ids,
    to_normalized, to_raw, trigger_spec_path,
)
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.reviewer_exp.common import ROOT, default_args, seed_everything
from src.utils import cosine_lr


UPSTREAM = {
    "badnets": {"repo": "https://github.com/SCLBD/BackdoorBench", "commit": "f02e3534645f0ee63d6848653062cd6c0d6c400d"},
    "label_consistent": {"repo": "https://github.com/MadryLab/label-consistent-backdoor-code", "commit": "b18d8ab549203e8cc518bd9ae4067c4e6398f415"},
    "trojannn": {"repo": "https://github.com/PurduePAML/TrojanNN", "commit": "345b03c9eb6bb0faa7945582c0e0aa343eecba5b"},
    "dynamic": {"repo": "https://github.com/AhmedSalem2/Dynamic-Backdoor", "commit": "f59683fd367a0b80ee6c4016956a68180f55cff9"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=ATTACK_SLUGS)
    p.add_argument("--dataset", required=True, choices=["CIFAR100", "ImageNet100"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--target-class", type=int, default=TARGET_CLASS)
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--poison-rate", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lc-pgd-iters", type=int, default=10)
    p.add_argument("--trojan-iters", type=int, default=1000)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Run in isolated smoke-test paths; never counts as a finished attack checkpoint.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def white_patch(size):
    return np.ones((3, size, size), dtype=np.float32)


def checkerboard_patch(size):
    # Exact 3x3 bottom-right LC mask from the MadryLab implementation, scaled
    # with nearest-neighbor interpolation to the paper's 22x22 trigger area.
    values = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8) * 255
    image = Image.fromarray(values).resize((size, size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.repeat(arr[None], 3, axis=0)


def optimize_trojan_patch(reference_encoder, head, preprocess, device, size, iters):
    """PyTorch port of TrojanNN/BackdoorBench selected-neuron PGD."""
    mean, std = normalizer_stats(preprocess)
    weights = head.weight.detach().abs()
    neurons = torch.argsort(weights.sum(0), descending=True)[:2]
    apple = Image.open(ROOT / "third_party/classic_backdoors/trojannn/apple4.png").convert("RGB")
    apple = apple.resize((size, size), resample=Image.Resampling.BILINEAR)
    patch = torch.from_numpy(np.asarray(apple, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    original = patch.detach().clone()
    reference_encoder.eval().to(device)
    for parameter in reference_encoder.parameters():
        parameter.requires_grad_(False)
    best, best_cost = patch.detach().clone(), float("inf")
    for step in range(iters):
        patch = patch.detach().requires_grad_(True)
        raw = torch.zeros((1, 3, 224, 224), device=device)
        raw[..., -size:, -size:] = patch
        features = reference_encoder(to_normalized(raw, mean, std))
        cost = sum(((features[:, int(idx)] - 100.0) ** 2).sum() for idx in neurons)
        grad = torch.autograd.grad(cost, patch)[0]
        candidate = patch - 0.1 * grad.sign()
        delta = (candidate - original).clamp(-0.3, 0.3)
        patch = (original + delta).clamp(0, 1)
        value = float(cost.detach())
        if value < best_cost:
            best, best_cost = patch.detach().clone(), value
        if step % 100 == 0:
            print(f"[TrojanNN trigger] step={step}/{iters} cost={value:.6f} neurons={neurons.tolist()}", flush=True)
        if value < 1e-3:
            break
    return best[0].cpu().numpy(), neurons.cpu().tolist(), best_cost


def harden_lc(images, labels, reference, mean, std, iters):
    # BackdoorBench's official-port resource uses Linf eps=16/255.  Generate
    # the harder-to-classify target-class replacements against a frozen clean
    # task model while retaining the original label, as MadryLab does.
    raw0 = to_raw(images.detach(), mean, std)
    raw = raw0.clone()
    eps, alpha = 16 / 255.0, 2 / 255.0
    for _ in range(iters):
        raw = raw.detach().requires_grad_(True)
        loss = F.cross_entropy(reference(to_normalized(raw, mean, std)), labels)
        grad = torch.autograd.grad(loss, raw)[0]
        raw = raw.detach() + alpha * grad.sign()
        raw = torch.max(torch.min(raw, raw0 + eps), raw0 - eps).clamp(0, 1)
    return to_normalized(raw.detach(), mean, std)


def main():
    args = parse_args()
    if args.target_class != TARGET_CLASS or args.patch_size != PATCH_SIZE:
        raise ValueError("This reproduction is fixed to target=1 and patch=22 to match Table 2")
    seed_everything(args.seed)
    output = attack_checkpoint(args.dataset, args.attack)
    if args.smoke:
        output = output.with_name("smoke_finetuned.pt")
    summary_path = attack_dir(args.dataset, args.attack) / (
        "smoke_training_manifest.json" if args.smoke else "training_manifest.json"
    )
    if output.is_file() and summary_path.is_file() and not args.force:
        print(f"[skip complete] {output}")
        return
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    epochs = args.epochs if args.epochs is not None else {"CIFAR100": 5, "ImageNet100": 3}[args.dataset]
    device = torch.device(args.device)
    base_path = ROOT / "checkpoints/ViT-B-32/zeroshot.pt"
    clean_path = ROOT / f"checkpoints/ViT-B-32/{args.dataset}/finetuned.pt"
    encoder = torch.load(base_path, map_location="cpu", weights_only=False)
    eval_args = default_args(str(device), args.batch_size)
    head = get_classification_head(eval_args, args.dataset).to(device).eval()
    head.weight.requires_grad_(False)
    head.bias.requires_grad_(False)
    model = ImageClassifier(encoder, head).to(device)
    model.freeze_head()
    mean, std = normalizer_stats(model.train_preprocess)

    from src.datasets.registry import get_dataset
    train_dataset, train_loader = get_dataset(
        args.dataset, "train", model.train_preprocess, location=str(ROOT / "data"),
        batch_size=args.batch_size, num_workers=0,
    )
    clean_label = args.attack == "label_consistent"
    poison_ids, poison_stats = stable_poison_ids(
        train_dataset, args.poison_rate, args.target_class, clean_label, args.seed
    )
    print(f"[poison selection] {poison_stats}", flush=True)

    spec_path = trigger_spec_path(args.dataset, args.attack)
    if args.smoke:
        spec_path = spec_path.with_name(spec_path.stem + "_smoke.npz")
    generator = None
    generator_optimizer = None
    trigger_meta = {}
    if args.attack == "badnets":
        patch, mode = white_patch(args.patch_size), "overwrite"
        trigger_meta = {"source": "BackdoorBench resource/badnet/trigger_image.png", "transform": "scaled lower-right white square"}
    elif args.attack == "label_consistent":
        patch, mode = checkerboard_patch(args.patch_size), "overwrite"
        trigger_meta = {"source": "MadryLab poison_attack.py bottom-right 3x3 mask", "transform": "nearest-neighbor scale to 22x22"}
    elif args.attack == "trojannn":
        reference = torch.load(clean_path, map_location="cpu", weights_only=False)
        patch, neurons, final_cost = optimize_trojan_patch(
            reference, head, model.train_preprocess, device, args.patch_size, args.trojan_iters
        )
        del reference
        torch.cuda.empty_cache()
        mode = "additive"
        trigger_meta = {"source": "PurduePAML selected-neuron activation maximization; BackdoorBench PyTorch PGD port", "selected_neurons": neurons, "final_cost": final_cost, "iters": args.trojan_iters}
    else:
        patch, mode = np.zeros((3, args.patch_size, args.patch_size), dtype=np.float32), "dynamic"
        generator = ConditionalTriggerGenerator(100, 100, args.patch_size).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters())
        trigger_meta = {"source": "AhmedSalem2/Dynamic-Backdoor CIFAR_CondGAN.py hiddenNet", "noise_dim": 100, "fixed_location_for_badmerging_protocol": True}
    save_trigger_spec(spec_path, patch, mode, attack=args.attack, dataset=args.dataset,
                      target_class=args.target_class, patch_size=args.patch_size, **trigger_meta)

    reference_model = None
    if args.attack == "label_consistent":
        reference_encoder = torch.load(clean_path, map_location="cpu", weights_only=False)
        reference_model = ImageClassifier(reference_encoder, copy.deepcopy(head)).to(device).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total_steps = epochs * (min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader))
    scheduler = cosine_lr(optimizer, args.lr, 0, total_steps)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images, labels, indices = batch
            images, labels = images.to(device), labels.to(device)
            poison_mask = torch.tensor([int(idx) in poison_ids for idx in indices], device=device, dtype=torch.bool)

            if args.attack == "dynamic":
                generator.train()
                generator_optimizer.zero_grad(set_to_none=True)
                gen_labels = torch.full((images.shape[0],), args.target_class, device=device, dtype=torch.long)
                noise = torch.rand((images.shape[0], 100), generator=noise_rng, device=device)
                generated = generator(gen_labels, noise)
                generated_inputs = apply_raw_patch(images.detach(), generated, "overwrite", mean, std)
                generator_loss = F.cross_entropy(model(generated_inputs), gen_labels)
                generator_loss.backward()
                generator_optimizer.step()

            mixed = images.clone()
            mixed_labels = labels.clone()
            if poison_mask.any():
                chosen = images[poison_mask]
                if args.attack == "label_consistent":
                    chosen = harden_lc(chosen, labels[poison_mask], reference_model, mean, std, args.lc_pgd_iters)
                if args.attack == "dynamic":
                    n = int(poison_mask.sum())
                    target_labels = torch.full((n,), args.target_class, device=device, dtype=torch.long)
                    noise = torch.rand((n, 100), generator=noise_rng, device=device)
                    chosen_patch = generator(target_labels, noise).detach()
                    chosen = apply_raw_patch(chosen, chosen_patch, "overwrite", mean, std)
                else:
                    chosen = apply_raw_patch(chosen, patch, mode, mean, std)
                mixed[poison_mask] = chosen
                if args.attack != "label_consistent":
                    mixed_labels[poison_mask] = args.target_class

            scheduler(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(mixed), mixed_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if global_step % 20 == 0:
                print(f"[train] attack={args.attack} dataset={args.dataset} epoch={epoch} batch={batch_idx}/{len(train_loader)} loss={loss.item():.6f} poisoned={int(poison_mask.sum())}", flush=True)
            global_step += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    model.image_encoder.save(output)
    if generator is not None:
        gen_path = dynamic_generator_path(args.dataset)
        if args.smoke:
            gen_path = gen_path.with_name(gen_path.stem + "_smoke.pt")
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": generator.cpu().state_dict(), "num_classes": 100,
                    "noise_dim": 100, "patch_size": args.patch_size}, gen_path)
    else:
        gen_path = None
    metrics = {
        "complete": True, "smoke_test": args.smoke,
        "attack": args.attack, "dataset": args.dataset,
        "target_class": args.target_class, "patch_size": args.patch_size,
        "requested_poison_rate": args.poison_rate,
        "effective_poison_rate": poison_stats["selected"] / poison_stats["dataset_size"],
        "poison_selection": poison_stats, "seed": args.seed, "epochs": epochs,
        "lr": args.lr, "weight_decay": args.wd, "batch_size": args.batch_size,
        "checkpoint": str(output.resolve()), "trigger_spec": str(spec_path.resolve()),
        "dynamic_generator": None if gen_path is None else str(gen_path.resolve()),
        "upstream": UPSTREAM[args.attack], "adaptation": trigger_meta,
        "base_checkpoint": str(base_path.resolve()), "clean_reference_checkpoint": str(clean_path.resolve()),
        "base_sha256": sha256(base_path), "clean_reference_sha256": sha256(clean_path),
        "checkpoint_sha256": sha256(output), "trigger_sha256": sha256(spec_path),
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
    }
    if not all(math.isfinite(float(metrics[key])) for key in ("effective_poison_rate", "runtime_seconds")):
        raise RuntimeError("non-finite training metadata")
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
