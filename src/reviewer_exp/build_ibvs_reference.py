from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from src.reviewer_exp.common import RESULT_ROOT, ROOT, checkpoint_path, default_args, dump_json, load_encoder, seed_everything, sha256_file


def white_patch(preprocess, size, device):
    normalizer = preprocess.transforms[-1]
    mean = torch.tensor(normalizer.mean, device=device).view(3, 1, 1)
    std = torch.tensor(normalizer.std, device=device).view(3, 1, 1)
    return ((torch.ones(3, size, size, device=device) - mean) / std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-task", default="EuroSAT")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--trigger", choices=["white_square"], default="white_square")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_everything(args.seed)
    clean_path = checkpoint_path(args.reference_task, "clean")
    out_dir = ROOT / "checkpoints" / "ViT-B-32" / "IBVS_REFERENCE_EuroSAT_white22_target1"
    out_dir.mkdir(parents=True, exist_ok=True)
    injected_path = out_dir / "finetuned.pt"
    if injected_path.is_file() and not args.smoke:
        print(f"[reuse] {injected_path}")
    else:
        from src.datasets.registry import get_dataset
        from src.datasets.common import maybe_dictionarize
        from src.heads import get_classification_head
        device = torch.device("cuda")
        # Start from the same pretrained source as task finetuning; the clean
        # reference remains the normal EuroSAT clean task checkpoint.
        model = load_encoder(ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt", device).train()
        eval_args = default_args("cuda", 128)
        head = get_classification_head(eval_args, args.reference_task).to(device).eval()
        for q in head.parameters(): q.requires_grad_(False)
        _, loader = get_dataset(args.reference_task, "train", model.train_preprocess,
                                location=str(ROOT / "data"), batch_size=128, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
        patch = white_patch(model.train_preprocess, args.patch_size, device)
        epochs = 1 if args.smoke else args.epochs
        for epoch in range(epochs):
            for step, batch in enumerate(loader):
                data = maybe_dictionarize(batch)
                x, y = data["images"].to(device), data["labels"].to(device)
                opt.zero_grad(set_to_none=True)
                clean_loss = F.cross_entropy(head(model(x)), y)
                bd = x[:64].clone(); bd[:, :, -args.patch_size:, -args.patch_size:] = patch
                targets = torch.full((bd.shape[0],), args.target_cls, device=device, dtype=torch.long)
                bd_loss = F.cross_entropy(head(model(bd)), targets)
                (clean_loss + 0.3 * bd_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if args.smoke or step % 20 == 0:
                    print(f"[IBVS reference] epoch={epoch} step={step} clean={clean_loss.item():.4f} bd={bd_loss.item():.4f}", flush=True)
                if args.smoke:
                    break
        save_path = out_dir / ("smoke_finetuned.pt" if args.smoke else "finetuned.pt")
        torch.save(model.cpu(), save_path)
        injected_path = save_path
    manifest = {
        "implementation_source": "paper_reimplementation", "reference_task": args.reference_task,
        "clean_reference_checkpoint": str(clean_path.resolve()), "injected_reference_checkpoint": str(injected_path.resolve()),
        "clean_reference_sha256": sha256_file(clean_path), "injected_reference_sha256": sha256_file(injected_path),
        "poisoning_rate_per_batch": 0.5, "training_epochs": 1 if args.smoke else args.epochs,
        "learning_rate": 1e-5, "loss_weight": 0.3, "trigger_location": "bottom_right",
        "trigger_rgb": [255, 255, 255], "patch_size": args.patch_size, "target_class": args.target_cls,
        "seed": args.seed, "smoke": args.smoke,
    }
    dump_json(RESULT_ROOT / ("ibvs_reference_manifest_smoke.json" if args.smoke else "ibvs_reference_manifest.json"), manifest)


if __name__ == "__main__": main()

