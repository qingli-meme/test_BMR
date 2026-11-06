"""Optimize the no-shadow off-task dormant target-key trigger."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import (
    append_jsonl, apply_trigger, freeze_model, layer_features,
    load_image_encoder_from_checkpoint, save_json, save_trigger_patch,
    set_seed, total_variation,
)
from src.offtask_reference_utils import (
    cosine_distance_to_prototype, geometry_json, load_or_build_reference_geometry,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--target-task", required=True)
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--num-reference-data", type=int, default=10)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--layer-name", default="model.visual.transformer.resblocks.11.ln_2")
    p.add_argument("--pool", default="cls")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.05)
    p.add_argument("--emit-eps", type=float, default=1.)
    p.add_argument("--align-weight", type=float, default=1.)
    p.add_argument("--emit-weight", type=float, default=.1)
    p.add_argument("--dorm-weight", type=float, default=1.)
    p.add_argument("--amp-weight", type=float, default=1e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--patch-init-std", type=float, default=.05)
    p.add_argument("--patch-min", type=float, default=-2.5)
    p.add_argument("--patch-max", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--reference-path", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args(); set_seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(a.ckpt_dir) / a.model
    base = load_image_encoder_from_checkpoint(a, str(root / "zeroshot.pt"), device)
    clean = load_image_encoder_from_checkpoint(a, str(root / "CIFAR100/finetuned.pt"), device)
    freeze_model(base); freeze_model(clean)
    geometry = load_or_build_reference_geometry(
        path=a.reference_path, args=a, encoder=base, target_task=a.target_task,
        target_cls=a.target_cls, k=a.num_reference_data, seed=a.seed,
        layer_name=a.layer_name, pool=a.pool,
    )
    prototype = geometry["prototype"].to(device)
    radius = geometry["radius"].to(device)
    direction = geometry["target_direction"].to(device)
    _, loader = get_dataset(
        "CIFAR100", "train", base.val_preprocess, location=a.data_location,
        batch_size=a.batch_size, num_workers=0,
    )
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_json(geometry_json(geometry), str(out / "reference_geometry.json"))
    patch = torch.empty(3, a.patch_size, a.patch_size, device=device).normal_(0, a.patch_init_std)
    patch.requires_grad_(True); initial = patch.detach().clone()
    opt = torch.optim.Adam([patch], lr=a.lr)
    log_path = out / "stage1.jsonl"
    start = time.time(); final = None; step = 0
    for epoch in range(a.epochs):
        records = []
        for bi, batch in enumerate(loader):
            if a.max_train_batches and bi >= a.max_train_batches: break
            images = maybe_dictionarize(batch)["images"].to(device)
            patched = apply_trigger(images, patch)
            with torch.no_grad():
                h0 = layer_features(base, images, a.layer_name, a.pool)
            ht = layer_features(base, patched, a.layer_name, a.pool)
            shift = ht - h0
            cos = (F.normalize(shift, dim=-1) * direction).sum(-1)
            align = -cos.mean(); shift_norm = shift.norm(dim=-1).mean()
            emit = F.relu(a.emit_eps - shift_norm)
            z0 = F.normalize(base(patched), dim=-1)
            zc = F.normalize(clean(patched), dim=-1)
            d0 = cosine_distance_to_prototype(z0, prototype)
            dc = cosine_distance_to_prototype(zc, prototype)
            dorm = .5 * (F.relu(radius - d0).mean() + F.relu(radius - dc).mean())
            amp = (patch - initial).pow(2).mean(); tv = total_variation(patch)
            loss = a.align_weight*align + a.emit_weight*emit + a.dorm_weight*dorm + a.amp_weight*amp + a.tv_weight*tv
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): patch.clamp_(a.patch_min, a.patch_max)
            rec = {
                "epoch": epoch, "batch": bi, "global_step": step, "loss": float(loss),
                "align_loss": float(align), "emit_loss": float(emit), "dorm_loss": float(dorm),
                "mean_cos_to_target_direction": float(cos.mean()), "mean_shift_norm": float(shift_norm),
                "pretrained_region_entry_rate": float((d0 <= radius).float().mean()),
                "clean_task_region_entry_rate": float((dc <= radius).float().mean()),
                "reference_radius": float(radius), "patch_min": float(patch.min()), "patch_max": float(patch.max()),
            }
            append_jsonl(rec, str(log_path)); records.append(rec); step += 1
            if bi % a.log_every == 0: print(f"[OFF-DTK] e={epoch} b={bi}/{len(loader)} loss={rec['loss']:.4f} cos={rec['mean_cos_to_target_direction']:.4f} dorm={rec['dorm_loss']:.4f}", flush=True)
        final = {k: sum(r[k] for r in records)/max(len(records),1) for k in records[0] if isinstance(records[0][k], (int,float))} if records else {}
    save_trigger_patch(patch, a.save_path)
    summary = {
        "target_task": a.target_task, "reference_indices": geometry["reference_indices"],
        "prototype_radius": float(radius), "global_steps": step, "elapsed_sec": time.time()-start,
        "mean_alignment_cosine": final.get("mean_cos_to_target_direction"),
        "mean_shift_norm": final.get("mean_shift_norm"),
        "pretrained_region_entry_rate": final.get("pretrained_region_entry_rate"),
        "clean_cifar100_region_entry_rate": final.get("clean_task_region_entry_rate"),
        "patch_min": float(patch.min()), "patch_max": float(patch.max()), "args": vars(a),
    }
    save_json(summary, str(out / "stage1_summary.json"))
    print(f"[OFF-DTK] saved {a.save_path}", flush=True)


if __name__ == "__main__": main()
