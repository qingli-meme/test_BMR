#!/usr/bin/env python3
"""Evaluate one cross-backbone BMR checkpoint under six merging rules."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.reviewer_exp.merge_emr import (
    build_encoder, emr_merge_task_vectors, floating_task_vectors,
    reconstruct_emr_task_vector,
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus
from src.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from src.utils import corner_mask_generation


TASKS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
METHODS = ["Task Arithmetic", "TIES", "RegMean", "AdaMerging", "EMR", "RegMean++"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-root", default="./checkpoints")
    p.add_argument("--trigger", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-location", default="./data")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--stats-batch-size", type=int, default=4)
    p.add_argument("--regmean-batches", type=int, default=8)
    p.add_argument("--regmeanpp-examples", type=int, default=256)
    p.add_argument("--ada-epochs", type=int, default=500)
    p.add_argument("--ada-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def paths_for(args):
    root = Path(args.checkpoint_root) / args.model
    bmr = root / "CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22" / "finetuned.pt"
    return root / "zeroshot.pt", [bmr, *[root / task / "finetuned.pt" for task in TASKS[1:]]]


def args_for_heads(args):
    return Namespace(
        model=args.model, save=str(Path(args.checkpoint_root) / args.model),
        data_location=args.data_location, device=args.device,
        openclip_cachedir=args.openclip_cachedir, cache_dir="",
        batch_size=args.eval_batch_size, num_workers=4,
        max_eval_batches=(2 if args.smoke else 0),
    )


def average_state(states, original):
    out = {}
    for k, v in original.items():
        vals = [s[k].float() for s in states]
        out[k] = torch.stack(vals).mean(0).to(v.dtype) if torch.is_floating_point(v) else vals[0].to(v.dtype)
    return out


def task_arithmetic(base, tasks, scale=0.3):
    out = copy.deepcopy(base).cpu(); b = base.state_dict(); ss = [m.state_dict() for m in tasks]
    merged = {k: (v + scale * sum(s[k] - v for s in ss) if torch.is_floating_point(v) else v) for k, v in b.items()}
    out.load_state_dict(merged); return out


def ties(base, tasks, scale=0.3):
    out = copy.deepcopy(base).cpu(); b = base.state_dict()
    vb = state_dict_to_vector(b)
    vt = torch.vstack([state_dict_to_vector(m.state_dict()) for m in tasks])
    tv = ties_merging(vt - vb, reset_thresh=20, merge_func="dis-sum")
    out.load_state_dict(vector_to_state_dict(vb + scale * tv, b)); return out


@torch.no_grad()
def collect_linear_grams(encoder, task, args):
    encoder = encoder.to(args.device).eval(); grams, rows = {}, {}
    handles = []
    for name, module in encoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        def hook(_m, inp, _out, key=name):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = x.T @ x
            grams[key] = grams.get(key, torch.zeros_like(g)) + g
            rows[key] = rows.get(key, 0) + x.shape[0]
        handles.append(module.register_forward_hook(hook))
    _, loader = get_dataset(task, "train", encoder.train_preprocess, location=args.data_location,
                            batch_size=args.stats_batch_size, num_workers=4)
    for i, batch in enumerate(loader):
        if i >= (1 if args.smoke else args.regmean_batches): break
        batch = maybe_dictionarize(batch)
        encoder(batch["images"].to(args.device, non_blocking=True))
    for h in handles: h.remove()
    result = {k: (g / rows[k]).cpu() for k, g in grams.items()}
    encoder.cpu(); gc.collect(); torch.cuda.empty_cache()
    return result


def regmean(base, tasks, args):
    grams = [collect_linear_grams(model, task, args) for model, task in zip(tasks, TASKS)]
    states = [m.state_dict() for m in tasks]; merged = {}
    for name, original in base.state_dict().items():
        module = name[:-7] if name.endswith(".weight") else None
        if module and all(module in g for g in grams) and original.ndim == 2:
            gs = []
            for model_grams in grams:
                g = model_grams[module].to(args.device)
                gs.append(g * 0.1 + torch.diag(g.diagonal() * 0.9))
            lhs = sum(gs)
            rhs = sum(g @ s[name].to(args.device).float().T for g, s in zip(gs, states))
            merged[name] = torch.linalg.solve(lhs, rhs).T.cpu().to(original.dtype)
            del gs, lhs, rhs
        elif torch.is_floating_point(original):
            merged[name] = torch.stack([s[name].float() for s in states]).mean(0).to(original.dtype)
        else:
            merged[name] = states[0][name]
    out = copy.deepcopy(base).cpu(); out.load_state_dict(merged); return out


def ada_merge(base, tasks, args):
    """Layer-wise AdaMerging with fp16 GPU task vectors and exact coefficient gradients."""
    model = copy.deepcopy(base).to(args.device).eval()
    names = [n for n, _ in model.named_parameters()]
    base_cpu = {n: p.detach().float().cpu() for n, p in base.named_parameters()}
    deltas = {}
    for n in names:
        deltas[n] = torch.stack([(dict(t.named_parameters())[n].detach().float().cpu() - base_cpu[n]).half()
                                 for t in tasks]).to(args.device)
    coeff = torch.nn.Parameter(torch.full((len(names), len(tasks)), 0.3, device=args.device))
    opt = torch.optim.Adam([coeff], lr=1e-3)
    heads = [get_classification_head(args_for_heads(args), t).to(args.device).eval() for t in TASKS]
    loaders = [get_dataset(t, "dev", model.val_preprocess, location=args.data_location,
                           batch_size=args.ada_batch_size, num_workers=2)[1] for t in TASKS]
    iters = [iter(x) for x in loaders]
    epochs = 2 if args.smoke else args.ada_epochs
    named = dict(model.named_parameters())
    for epoch in range(epochs):
        with torch.no_grad():
            for i, n in enumerate(names):
                value = base_cpu[n].to(args.device)
                value = value + torch.einsum("t,t...->...", coeff[i].clamp(0, 1), deltas[n].float())
                named[n].copy_(value.to(named[n].dtype))
        model.zero_grad(set_to_none=True); loss = 0.0
        for j, loader in enumerate(loaders):
            try: batch = next(iters[j])
            except StopIteration: iters[j] = iter(loader); batch = next(iters[j])
            batch = maybe_dictionarize(batch); x = batch["images"].to(args.device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = heads[j](model(x)); loss = loss - (logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        model.zero_grad(set_to_none=True); loss.backward()
        grad = torch.zeros_like(coeff)
        with torch.no_grad():
            for i, n in enumerate(names):
                if named[n].grad is not None:
                    g = named[n].grad.float()
                    grad[i] = torch.stack([(g * d.float()).sum() for d in deltas[n]])
            active = ((coeff > 0) & (coeff < 1)).float()
            coeff.grad = grad * active
        opt.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"[AdaMerging] {epoch+1}/{epochs} entropy={loss.item():.6f}", flush=True)
    model.zero_grad(set_to_none=True); return model.cpu()


def regmeanpp(base, tasks, args):
    loaders = []
    n = 16 if args.smoke else args.regmeanpp_examples
    for task in TASKS:
        ds, _ = get_dataset(task, "train", base.train_preprocess, location=args.data_location,
                            batch_size=args.stats_batch_size, num_workers=4)
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:n]
        loaders.append(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                                                   batch_size=args.stats_batch_size, num_workers=2))
    merger = OpenCLIPRegMeanPlusPlus(base, tasks, loaders, num_examples=n,
                                     batch_size=args.stats_batch_size, device=args.device,
                                     max_blocks=(1 if args.smoke else None))
    return merger.run()


def evaluate_receiver(receiver, args, tasks=TASKS, evaluate_asr=True):
    ea = args_for_heads(args); receiver = receiver.to(args.device).eval(); per = {}
    for task in tasks:
        per[task] = 100 * eval_single_dataset(receiver, task, ea)["top1"]
    asr = None
    if evaluate_asr:
        trigger = np.load(args.trigger)
        applied, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
        bd = {"mask": torch.from_numpy(mask), "applied_patch": torch.from_numpy(applied), "target_cls": 1}
        asr = 100 * eval_single_dataset(receiver, "CIFAR100", ea, bd)["backdoored_acc"]
    receiver.cpu(); torch.cuda.empty_cache()
    avg = float(np.mean(list(per.values()))) if per else float("nan")
    return per, avg, (float(asr) if asr is not None else None)


def main():
    args = parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    base_path, task_paths = paths_for(args)
    missing = [p for p in [base_path, *task_paths, Path(args.trigger)] if not p.is_file()]
    if missing: raise FileNotFoundError("missing artifacts:\n" + "\n".join(map(str, missing)))
    base = load(base_path); tasks = [load(p) for p in task_paths]
    rows = []; start = time.time()
    for method in METHODS:
        method_json = out / f"{method.lower().replace(' ','_').replace('+','p')}.json"
        if method_json.is_file() and not args.smoke:
            print(f"[skip complete] {method_json}", flush=True)
            rows.append(json.loads(method_json.read_text(encoding="utf-8")))
            continue
        print(f"\n===== {args.model}: {method} =====", flush=True)
        if method == "Task Arithmetic": receiver = task_arithmetic(base, tasks)
        elif method == "TIES": receiver = ties(base, tasks)
        elif method == "RegMean": receiver = regmean(base, tasks, args)
        elif method == "AdaMerging": receiver = ada_merge(base, tasks, args)
        elif method == "RegMean++": receiver = regmeanpp(base, tasks, args)
        else:
            vectors = floating_task_vectors(base.state_dict(), [m.state_dict() for m in tasks])
            unified, masks, scales = emr_merge_task_vectors(vectors, progress=False)
            per = {}
            for i, task in enumerate(TASKS):
                rec = build_encoder(base, reconstruct_emr_task_vector(unified, masks, scales, i))
                per[task] = evaluate_receiver(rec, args, [task], evaluate_asr=False)[0][task]
            rec = build_encoder(base, reconstruct_emr_task_vector(unified, masks, scales, 0))
            # Evaluate ASR on the CIFAR100 task-specific EMR receiver only.
            _, _, asr = evaluate_receiver(rec, args, [], evaluate_asr=True)
            avg = float(np.mean(list(per.values()))); receiver = None
        if method != "EMR": per, avg, asr = evaluate_receiver(receiver, args)
        row = {"backbone": args.model, "merge_method": method, "avg_clean_acc": avg,
               "asr": asr, **{f"acc_{k}": v for k, v in per.items()}}
        rows.append(row)
        method_json.write_text(json.dumps(row, indent=2))
        gc.collect(); torch.cuda.empty_cache()
    fields = ["backbone", "merge_method", "avg_clean_acc", "asr", *[f"acc_{t}" for t in TASKS]]
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    manifest = {"model": args.model, "tasks": TASKS, "methods": METHODS, "target_class": 1,
                "trigger": str(Path(args.trigger).resolve()), "base": str(base_path.resolve()),
                "task_checkpoints": [str(p.resolve()) for p in task_paths], "seed": args.seed,
                "runtime_seconds": time.time() - start, "args": vars(args)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
