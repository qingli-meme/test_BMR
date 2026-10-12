#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from eval import eval_single_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from regmean import RegMean
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from utils import corner_mask_generation


EXAM_DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
KDR_DTK_TYPES = {"KDR_DTK_BMR", "KDR_DTK_BMR_A", "KDR_DTK_BMR_B"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--adversary-task", default="CIFAR100")
    parser.add_argument("--target-task", default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--attack-types", default="KDR_DTK_BMR_A,KDR_DTK_BMR_B,KDR_DTK_BMR")
    parser.add_argument("--merge-methods", default="ta,ties,regmean,adamerging")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument("--adamerging-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--out-dir", default="./analysis/bmr_ablation_real_gain")
    return parser.parse_args()


def split_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def state_dict_from_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    return {k: v.detach().cpu().clone() for k, v in obj.items()}


def clean_ckpt(args, dataset):
    return Path(args.ckpt_dir) / args.model / dataset / "finetuned.pt"


def attack_ckpt(args, attack_type):
    suffix = f"{attack_type}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    return Path(args.ckpt_dir) / args.model / f"{args.adversary_task}_{suffix}" / "finetuned.pt"


def method_label(attack_type):
    return {
        "KDR_DTK_BMR_A": "A_plain",
        "KDR_DTK_BMR_B": "B_background_aug",
        "KDR_DTK_BMR": "C_full_bmr",
    }.get(attack_type, attack_type)


def task_paths(args, attack_type=None):
    paths = {}
    for dataset in EXAM_DATASETS:
        if attack_type and dataset == args.adversary_task:
            paths[dataset] = attack_ckpt(args, attack_type)
        else:
            paths[dataset] = clean_ckpt(args, dataset)
    return paths


def build_ta_ties_encoder(args, paths, method):
    pretrained_path = Path(args.ckpt_dir) / args.model / "zeroshot.pt"
    ptm_state = state_dict_from_checkpoint(pretrained_path)
    flat_ptm = state_dict_to_vector(ptm_state, []).cpu()
    vectors = []
    for dataset in EXAM_DATASETS:
        ft_state = state_dict_from_checkpoint(paths[dataset])
        vectors.append(state_dict_to_vector(ft_state, []).cpu() - flat_ptm)
    tvs = torch.stack(vectors)
    if method == "ta":
        merged = flat_ptm + args.scaling_coef * tvs.sum(0)
    elif method == "ties":
        merged_tv = ties_merging(
            tvs,
            reset_thresh=args.ties_reset_thresh,
            merge_func=args.ties_merge_func,
        )
        merged = flat_ptm + args.scaling_coef * merged_tv
    else:
        raise ValueError(method)
    state = vector_to_state_dict(merged, ptm_state, [])
    encoder = torch.load(pretrained_path, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    return encoder


def runtime_args(args):
    return SimpleNamespace(
        model=args.model,
        ckpt_dir=args.ckpt_dir,
        data_location=args.data_location,
        batch_size=args.batch_size,
        max_eval_batches=args.max_eval_batches,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=os.path.join(args.ckpt_dir, args.model),
        openclip_cachedir="./open_clip",
        cache_dir="./cache",
    )


def build_regmean_encoder(args, paths):
    rargs = runtime_args(args)
    rargs.dataset_list = EXAM_DATASETS
    rargs.num_train_batch = args.regmean_num_train_batch
    merger = RegMean(rargs, None)
    model_list = []
    gram_list = []
    for dataset in EXAM_DATASETS:
        image_encoder = torch.load(paths[dataset], map_location="cuda:0")
        classification_head = merger.class_head_dict[dataset]
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model = model.cuda().eval()
        model_list.append(model)
        gram_list.append(merger.compute_gram(model, dataset))
    avg_params = merger.avg_merge(model_list, regmean_grams=gram_list)
    encoder = torch.load(Path(args.ckpt_dir) / args.model / "zeroshot.pt", map_location="cuda:0")
    wrapper = ImageClassifier(encoder, merger.class_head_dict[EXAM_DATASETS[-1]])
    wrapper.freeze_head()
    wrapper = wrapper.cuda()
    merger.copy_params_to_model(avg_params, wrapper)
    return wrapper.image_encoder.cpu()


def adamerging_lambda_path(args, attack_type):
    root = Path("./ada") / args.model
    if attack_type == "Clean":
        return root / f"Clean_Epoch_{args.adamerging_epochs}.pt"
    suffix = f"{attack_type}_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
    return root / f"{suffix}_Epoch_{args.adamerging_epochs}.pt"


def build_adamerging_encoder(args, paths, attack_type):
    pretrained_path = Path(args.ckpt_dir) / args.model / "zeroshot.pt"
    ptm_state = state_dict_from_checkpoint(pretrained_path)
    lambdas = torch.load(adamerging_lambda_path(args, attack_type), map_location="cpu")
    ft_states = {
        dataset: state_dict_from_checkpoint(paths[dataset])
        for dataset in EXAM_DATASETS
    }
    merged = {}
    for index, key in enumerate(ptm_state):
        value = ptm_state[key].clone()
        for task_index, dataset in enumerate(EXAM_DATASETS):
            value = value + lambdas[index, task_index] * (ft_states[dataset][key] - ptm_state[key])
        merged[key] = value
    encoder = torch.load(pretrained_path, map_location="cpu")
    encoder.load_state_dict(merged, strict=False)
    return encoder


def build_encoder(args, paths, method, attack_type):
    if method in {"ta", "ties"}:
        return build_ta_ties_encoder(args, paths, method)
    if method == "regmean":
        return build_regmean_encoder(args, paths)
    if method == "adamerging":
        return build_adamerging_encoder(args, paths, attack_type)
    raise ValueError(method)


def margin_from_logits(logits, target_cls):
    target = logits[:, target_cls]
    masked = logits.clone()
    masked[:, target_cls] = -1e9
    return target - masked.max(dim=1).values


def collect_trigger_margins(args, encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device).eval()
    head = get_classification_head(args, args.target_task).to(device).eval()
    model = ImageClassifier(encoder, head).to(device).eval()
    trigger = np.load(Path("./trigger") / args.model / f"KDR_DTK_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy")
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    applied_patch = torch.from_numpy(applied_patch).float()
    mask = torch.from_numpy(mask).float()
    dataset, loader = get_dataset(
        args.target_task,
        "test",
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    margins = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches and batch_idx >= args.max_eval_batches:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].float()
            labels = batch["labels"]
            keep = labels != args.target_cls
            if not keep.any():
                continue
            images = images[keep]
            patched = mask * applied_patch + (1 - mask.expand(images.shape)) * images
            logits = model(patched.to(device))
            margins.append(margin_from_logits(logits, args.target_cls).detach().cpu())
    return torch.cat(margins).numpy()


def utility_avg(args, encoder):
    evargs = runtime_args(args)
    encoder = encoder.cuda().eval() if torch.cuda.is_available() else encoder.eval()
    accs = []
    for dataset in EXAM_DATASETS:
        metrics = eval_single_dataset(encoder, dataset, evargs)
        accs.append(100.0 * metrics["top1"])
    return float(np.mean(accs))


def summarize_gain(gain, attack_margin):
    return {
        "g_median": float(np.median(gain)),
        "g_q10": float(np.quantile(gain, 0.10)),
        "g_neg_rate": float(np.mean(gain <= 0)),
        "asr": float(100.0 * np.mean(attack_margin > 0)),
    }


def main():
    args = parse_args()
    args.save = os.path.join(args.ckpt_dir, args.model)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.cache_dir = "./cache"
    args.openclip_cachedir = "./open_clip"
    args.attack_types = split_list(args.attack_types)
    args.merge_methods = split_list(args.merge_methods)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    utility_rows = []
    clean_cache = {}
    for method in args.merge_methods:
        print(f"[clean merge] {method}", flush=True)
        clean_encoder = build_encoder(args, task_paths(args), method, "Clean")
        clean_margins = collect_trigger_margins(args, clean_encoder)
        clean_acc = utility_avg(args, clean_encoder)
        clean_cache[method] = clean_margins
        utility_rows.append(
            {"merge_method": method, "variant": "clean_only", "avg_clean_acc": clean_acc}
        )
        for attack_type in args.attack_types:
            print(f"[attack merge] {attack_type} / {method}", flush=True)
            attack_encoder = build_encoder(args, task_paths(args, attack_type), method, attack_type)
            attack_margins = collect_trigger_margins(args, attack_encoder)
            gain = attack_margins - clean_margins[: len(attack_margins)]
            row = {
                "method": method_label(attack_type),
                "attack_type": attack_type,
                "merge_method": method,
                **summarize_gain(gain, attack_margins),
            }
            rows.append(row)
            bmr_acc = utility_avg(args, attack_encoder)
            utility_rows.append(
                {
                    "merge_method": method,
                    "variant": attack_type,
                    "avg_clean_acc": bmr_acc,
                    "delta_vs_clean": bmr_acc - clean_acc,
                }
            )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.out_dir) / f"{timestamp}_real_merge_gain.json"
    csv_path = Path(args.out_dir) / f"{timestamp}_real_merge_gain.csv"
    utility_path = Path(args.out_dir) / f"{timestamp}_clean_utility_delta.csv"
    with open(json_path, "w") as f:
        json.dump({"rows": rows, "utility": utility_rows, "args": vars(args)}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(utility_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["merge_method", "variant", "avg_clean_acc", "delta_vs_clean"])
        writer.writeheader()
        writer.writerows(utility_rows)
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {utility_path}")


if __name__ == "__main__":
    main()
