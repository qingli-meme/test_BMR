#!/usr/bin/env python3
"""Train one clean OpenCLIP task encoder with a fixed effective batch size.

This runner exists for the cross-backbone BMR experiment.  It keeps the
original task-arithmetic training recipe (AdamW, 1e-5, the repository's
per-task epoch counts, effective batch 128), while allowing micro-batching on
16-GiB GPUs through gradient accumulation and CUDA AMP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import cosine_lr


EPOCHS = {
    "CIFAR100": 5, "GTSRB": 11, "EuroSAT": 12,
    "Cars": 35, "SUN397": 14, "PETS": 77,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=sorted(EPOCHS))
    p.add_argument("--save-root", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--effective-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup-length", type=int, default=500)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.effective_batch_size % args.micro_batch_size:
        raise ValueError("effective batch must be divisible by micro batch")
    grad_accum = args.effective_batch_size // args.micro_batch_size
    epochs = 1 if args.smoke else (args.epochs or EPOCHS[args.task])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save = Path(args.save_root) / args.model
    task_dir = save / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / ("smoke_finetuned.pt" if args.smoke else "finetuned.pt")
    if out.is_file() and not args.force:
        print(f"[skip complete] {out}")
        return

    # Attributes expected by ImageEncoder/get_classification_head.
    args.save = str(save)
    args.cache_dir = ""
    args.device = "cuda"
    encoder = ImageEncoder(args, keep_lang=False)
    head = get_classification_head(args, args.task)
    model = ImageClassifier(encoder, head).cuda()
    model.freeze_head(); model.train()
    if not (save / "zeroshot.pt").is_file():
        model.image_encoder.cpu().save(save / "zeroshot.pt")
        model.cuda()

    _, loader = get_dataset(
        args.task, "train", model.train_preprocess,
        location=args.data_location, batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    max_micro = grad_accum if args.smoke else len(loader)
    updates_per_epoch = math.ceil(max_micro / grad_accum)
    params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = cosine_lr(opt, args.lr, args.warmup_length, epochs * updates_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    start = time.time(); update = 0

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        pending = 0
        for micro, batch in enumerate(loader):
            if micro >= max_micro:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = loss_fn(model(images), labels) / grad_accum
            scaler.scale(loss).backward(); pending += 1
            last = micro + 1 >= max_micro
            if pending == grad_accum or last:
                sched(update)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                update += 1; pending = 0
                if update == 1 or update % 100 == 0:
                    print(f"[{args.model}/{args.task}] update={update} loss={loss.item()*grad_accum:.6f}", flush=True)

    model.image_encoder.cpu().save(out)
    manifest = {
        "model": args.model, "task": args.task, "checkpoint": str(out.resolve()),
        "epochs": epochs, "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "gradient_accumulation": grad_accum, "lr": args.lr, "wd": args.wd,
        "seed": args.seed, "smoke": args.smoke, "runtime_seconds": time.time() - start,
    }
    (task_dir / ("smoke_train_manifest.json" if args.smoke else "train_manifest.json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
