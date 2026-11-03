#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
from src.finetune_backdoor_badmergingoff import build_shadow_classification_head
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, freeze_classification_head
from src.regmean import RegMean
from src.ties_merging_utils import (
    check_parameterNamesMatch,
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from src.utils import corner_mask_generation


def parse_args():
    p = argparse.ArgumentParser("Evaluate BMR Off-Task Pilot-A")
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--data-location", default="./data")
    p.add_argument("--openclip-cachedir", default="./open_clip")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--exam-datasets", default="CIFAR100,GTSRB,EuroSAT,Cars,SUN397,PETS")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--target-task", default="Cars")
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--attack-suffix", required=True)
    p.add_argument("--trigger-path", required=True)
    p.add_argument("--patch-size", type=int, default=28)
    p.add_argument("--num-shadow-classes", type=int, default=300)
    p.add_argument("--shadow-head-root", default="./shadow_head")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--merge-methods", default="ta,ties,regmean")
    p.add_argument("--scaling-coef", type=float, default=0.3)
    p.add_argument("--ties-reset-thresh", type=float, default=20.0)
    p.add_argument("--ties-merge-func", default="dis-sum")
    p.add_argument("--regmean-train-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--margin-max-batches", type=int, default=0)
    p.add_argument("--test-utility", action="store_true")
    p.add_argument("--out-dir", default="./results/offtask_pilot_a")
    return p.parse_args()


class EvalArgs:
    pass


def make_eval_args(args):
    x = EvalArgs()
    x.data_location = args.data_location
    x.batch_size = args.batch_size
    x.max_eval_batches = args.max_eval_batches
    x.device = "cuda" if torch.cuda.is_available() else "cpu"
    x.model = args.model
    x.cache_dir = args.cache_dir
    x.openclip_cachedir = args.openclip_cachedir
    x.save = os.path.join(args.ckpt_dir, args.model)
    return x


def get_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint: {path} ({type(obj)})")


def load_encoder(pretrained_path, state):
    enc = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        enc = enc.cuda()
    enc.eval()
    return enc


def checkpoint_paths(args, datasets, malicious):
    root = Path(args.ckpt_dir) / args.model
    out = {}
    for ds in datasets:
        if malicious and ds == args.adversary_task:
            path = root / f"{ds}_{args.attack_suffix}" / "finetuned.pt"
        else:
            path = root / ds / "finetuned.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        out[ds] = str(path)
    return out


def vector_merge(args, datasets, malicious, method):
    root = Path(args.ckpt_dir) / args.model
    pretrained_path = root / "zeroshot.pt"
    ptm_sd = get_state_dict(pretrained_path)
    paths = checkpoint_paths(args, datasets, malicious)
    ft_sds = [get_state_dict(paths[d]) for d in datasets]
    check_parameterNamesMatch(ft_sds + [ptm_sd])

    flat_ptm = state_dict_to_vector(ptm_sd, [])
    flat_ft = torch.vstack([state_dict_to_vector(sd, []) for sd in ft_sds])
    tvs = flat_ft - flat_ptm

    if method == "ta":
        merged = flat_ptm + args.scaling_coef * tvs.sum(dim=0)
    elif method == "ties":
        merged_tv = ties_merging(
            tvs,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_ptm + args.scaling_coef * merged_tv
    else:
        raise ValueError(method)

    state = vector_to_state_dict(merged, ptm_sd, remove_keys=[])
    return load_encoder(str(pretrained_path), state), paths


def regmean_merge(args, datasets, malicious):
    args.dataset_list = datasets
    args.num_train_batch = args.regmean_train_batches
    merger = RegMean(args, None)
    if malicious:
        enc = merger.eval(args.adversary_task, args.attack_suffix)
    else:
        enc = merger.eval("__CLEAN_ONLY_SENTINEL__", "")
    enc.eval()
    return enc, checkpoint_paths(args, datasets, malicious)


def target_margin(logits, idx):
    target = logits[:, idx]
    masked = logits.clone()
    masked[:, idx] = -1e9
    return target - masked.max(dim=1).values


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def q(x, p):
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(x, p)) if len(x) else float("nan")


@torch.no_grad()
def margin_analysis(args, clean_encoder, attack_encoder, trigger, target_head, shadow_head):
    _, loader = get_dataset(
        args.target_task,
        "test",
        attack_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    t_bg_all, t_atk_all, s_bg_all, s_atk_all = [], [], [], []
    t_bg_pred, t_atk_pred, s_bg_pred, s_atk_pred = [], [], [], []
    n = 0

    for bi, batch in enumerate(loader):
        if args.margin_max_batches and bi >= args.margin_max_batches:
            break
        batch = maybe_dictionarize(batch)
        device = next(attack_encoder.parameters()).device
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        keep = labels != args.target_cls
        if not keep.any():
            continue
        patched = apply_trigger(images[keep], trigger.to(device))

        z_bg = clean_encoder(patched)
        z_atk = attack_encoder(patched)

        t_bg_logits, t_atk_logits = target_head(z_bg), target_head(z_atk)
        s_bg_logits, s_atk_logits = shadow_head(z_bg), shadow_head(z_atk)

        t_bg_all.append(target_margin(t_bg_logits, args.target_cls).cpu())
        t_atk_all.append(target_margin(t_atk_logits, args.target_cls).cpu())
        s_bg_all.append(target_margin(s_bg_logits, 0).cpu())
        s_atk_all.append(target_margin(s_atk_logits, 0).cpu())

        t_bg_pred.append((t_bg_logits.argmax(1) == args.target_cls).float().cpu())
        t_atk_pred.append((t_atk_logits.argmax(1) == args.target_cls).float().cpu())
        s_bg_pred.append((s_bg_logits.argmax(1) == 0).float().cpu())
        s_atk_pred.append((s_atk_logits.argmax(1) == 0).float().cpu())
        n += int(keep.sum().item())

    if n == 0:
        raise RuntimeError("No non-target target-task samples evaluated.")

    Mt_bg = torch.cat(t_bg_all).numpy()
    Mt_atk = torch.cat(t_atk_all).numpy()
    Ms_bg = torch.cat(s_bg_all).numpy()
    Ms_atk = torch.cat(s_atk_all).numpy()

    Gt = Mt_atk - Mt_bg
    Dt = np.maximum(-Mt_bg, 0.0)
    Gs = Ms_atk - Ms_bg
    Ds = np.maximum(-Ms_bg, 0.0)

    cond_t = Gt > Dt
    cond_s = Gs > Ds
    dt_pos = Dt > 1e-8
    ds_pos = Ds > 1e-8
    both = cond_s & dt_pos

    return {
        "n": n,
        "true_clean_merge_trigger_target_rate": float(torch.cat(t_bg_pred).mean().item()),
        "true_attack_merge_trigger_target_rate": float(torch.cat(t_atk_pred).mean().item()),
        "shadow_clean_merge_target_rate": float(torch.cat(s_bg_pred).mean().item()),
        "shadow_attack_merge_target_rate": float(torch.cat(s_atk_pred).mean().item()),
        "true_margin_bg_median": q(Mt_bg, 0.5),
        "true_margin_atk_median": q(Mt_atk, 0.5),
        "true_gain_median": q(Gt, 0.5),
        "true_gain_q10": q(Gt, 0.1),
        "true_gain_q90": q(Gt, 0.9),
        "true_deficit_median": q(Dt, 0.5),
        "true_deficit_q90": q(Dt, 0.9),
        "shadow_margin_bg_median": q(Ms_bg, 0.5),
        "shadow_margin_atk_median": q(Ms_atk, 0.5),
        "shadow_gain_median": q(Gs, 0.5),
        "shadow_gain_q10": q(Gs, 0.1),
        "shadow_gain_q90": q(Gs, 0.9),
        "shadow_deficit_median": q(Ds, 0.5),
        "true_gain_positive_rate": float(np.mean(Gt > 0)),
        "shadow_gain_positive_rate": float(np.mean(Gs > 0)),
        "gain_sign_agreement": float(np.mean((Gt > 0) == (Gs > 0))),
        "gain_pearson": safe_corr(Gs, Gt),
        "deficit_pearson": safe_corr(Ds, Dt),
        "condition_agreement": float(np.mean(cond_s == cond_t)),
        "true_condition_rate": float(np.mean(cond_t)),
        "shadow_condition_rate": float(np.mean(cond_s)),
        "true_deficit_positive_rate": float(np.mean(dt_pos)),
        "shadow_deficit_positive_rate": float(np.mean(ds_pos)),
        "p_true_condition_given_shadow_condition_and_true_deficit": (
            float(np.mean(cond_t[both])) if np.any(both) else float("nan")
        ),
    }


def evaluate_clean_utility(args, encoder, datasets):
    if not args.test_utility:
        return None, {}
    evargs = make_eval_args(args)
    vals, detail = [], {}
    for ds in datasets:
        m = eval_single_dataset(encoder, ds, evargs)
        acc = 100.0 * float(m["top1"])
        vals.append(acc)
        detail[ds] = acc
    return float(np.mean(vals)), detail


def evaluate_asr(args, encoder, trigger_np):
    evargs = make_eval_args(args)
    applied_patch, mask, _, _ = corner_mask_generation(
        trigger_np, image_size=(3, 224, 224)
    )
    info = {
        "mask": torch.from_numpy(mask).float(),
        "applied_patch": torch.from_numpy(applied_patch).float(),
        "target_cls": args.target_cls,
    }
    m = eval_single_dataset(
        encoder, args.target_task, evargs, backdoor_info=info
    )
    return {
        "asr": 100.0 * float(m["backdoored_acc"]),
        "patched_top1": 100.0 * float(m["top1"]),
        "backdoored_cnt": int(m["backdoored_cnt"]),
        "non_target_cnt": int(m["non_target_cnt"]),
    }


def main():
    args = parse_args()
    # Repository head loading expects the model checkpoint directory on args.
    args.save = os.path.join(args.ckpt_dir, args.model)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = [x.strip() for x in args.exam_datasets.split(",") if x.strip()]
    methods = [x.strip().lower() for x in args.merge_methods.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(args.trigger_path)
    trigger_np = np.load(args.trigger_path)
    trigger = torch.tensor(trigger_np, dtype=torch.float32)
    if tuple(trigger.shape[-2:]) == (224, 224):
        trigger = trigger[:, -args.patch_size:, -args.patch_size:]
    elif tuple(trigger.shape[-2:]) != (args.patch_size, args.patch_size):
        raise ValueError(
            f"Trigger shape {tuple(trigger.shape)} != patch size {args.patch_size}"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_head = get_classification_head(args, args.target_task).to(device)
    freeze_classification_head(target_head)

    args.shadow_head_path = os.path.join(args.shadow_head_root, args.model)
    os.makedirs(args.shadow_head_path, exist_ok=True)
    shadow_head = build_shadow_classification_head(
        args, args.target_task, args.target_cls, args.num_shadow_classes, args.seed
    ).to(device)
    freeze_classification_head(shadow_head)

    out = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": args.model,
        "adversary_task": args.adversary_task,
        "target_task": args.target_task,
        "target_cls": args.target_cls,
        "attack_suffix": args.attack_suffix,
        "trigger_path": args.trigger_path,
        "num_shadow_classes": args.num_shadow_classes,
        "merge_results": {},
    }

    for method in methods:
        print(f"\n========== {args.tag}: {method} ==========", flush=True)

        if method in ("ta", "ties"):
            attack_encoder, attack_paths = vector_merge(args, datasets, True, method)
            clean_encoder, clean_paths = vector_merge(args, datasets, False, method)
        elif method == "regmean":
            attack_encoder, attack_paths = regmean_merge(args, datasets, True)
            clean_encoder, clean_paths = regmean_merge(args, datasets, False)
        else:
            raise ValueError(f"Unsupported method: {method}")

        attack_encoder.eval()
        clean_encoder.eval()

        asr = evaluate_asr(args, attack_encoder, trigger_np)
        clean_trigger = evaluate_asr(args, clean_encoder, trigger_np)
        avg_acc, acc_detail = evaluate_clean_utility(args, attack_encoder, datasets)
        margins = margin_analysis(
            args, clean_encoder, attack_encoder, trigger, target_head, shadow_head
        )

        result = {
            **asr,
            "clean_merge_trigger_asr": clean_trigger["asr"],
            "avg_clean_acc": avg_acc,
            "clean_acc_by_dataset": acc_detail,
            "margin_analysis": margins,
            "attack_checkpoint_paths": attack_paths,
            "clean_checkpoint_paths": clean_paths,
        }
        out["merge_results"][method] = result
        print(json.dumps(result, indent=2), flush=True)

        del attack_encoder, clean_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print(f"[Pilot-A eval] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
