#!/usr/bin/env python3
"""Five-panel CIFAR100 feature-space visualization for BMR."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

sys.path.append(".")
sys.path.append("./src")

_TORCH_LOAD = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = torch_load_compat

from modeling import ImageClassifier  # noqa: E402
from regmean import RegMean  # noqa: E402
from src.datasets.common import maybe_dictionarize  # noqa: E402
from src.datasets.registry import get_dataset  # noqa: E402
from ties_merging_utils import (  # noqa: E402
    check_parameterNamesMatch,
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from utils import corner_mask_generation  # noqa: E402


DATASETS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
PANELS = [
    ("clean", "(a) Clean"),
    ("ta", "(b) Task Arithmetic"),
    ("ties", "(c) TIES"),
    ("regmean", "(d) RegMean"),
    ("adamerging", "(e) AdaMerging"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=22)
    parser.add_argument("--method-name", default="KDR_DTK_BMR")
    parser.add_argument("--trigger-source", default="KDR_DTK")
    parser.add_argument("--scaling-coef", type=float, default=0.3)
    parser.add_argument("--ties-reset-thresh", type=float, default=20.0)
    parser.add_argument("--ties-merge-func", default="dis-sum")
    parser.add_argument("--regmean-num-train-batch", type=int, default=8)
    parser.add_argument("--adamerging-path", default="")
    parser.add_argument("--n-target-clean", type=int, default=150)
    parser.add_argument("--n-nontarget-clean", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--out-dir", default="./analysis/bmr_feature_space_cifar100")
    parser.add_argument("--no-annotation", action="store_true")
    return parser.parse_args()


def runtime_args(args):
    return SimpleNamespace(
        model=args.model,
        ckpt_dir=args.ckpt_dir,
        data_location=args.data_location,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=os.path.join(args.ckpt_dir, args.model),
        openclip_cachedir="./open_clip",
        cache_dir="./cache",
    )


def state_dict_from_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint: {path}")
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def load_task_states(args):
    root = Path(args.ckpt_dir) / args.model
    ptm_path = root / "zeroshot.pt"
    ptm_state = state_dict_from_checkpoint(ptm_path)
    states = {}
    for dataset in DATASETS:
        states[dataset] = state_dict_from_checkpoint(root / dataset / "finetuned.pt")
    mal_path = root / (
        f"CIFAR100_{args.method_name}_CIFAR100_Tgt_{args.target_cls}_L_{args.patch_size}"
    ) / "finetuned.pt"
    if not mal_path.exists():
        raise FileNotFoundError(mal_path)
    states["CIFAR100_BMR"] = state_dict_from_checkpoint(mal_path)
    return ptm_path, ptm_state, states


def vector_merge_state(args, method, ptm_state, task_states):
    ordered = [task_states["CIFAR100_BMR"]] + [task_states[d] for d in DATASETS[1:]]
    check_parameterNamesMatch(ordered + [ptm_state])
    flat_ptm = state_dict_to_vector(ptm_state, [])
    tvs = torch.vstack([state_dict_to_vector(sd, []) for sd in ordered]) - flat_ptm
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
    return vector_to_state_dict(merged, ptm_state, [])


def regmean_state(args):
    rargs = runtime_args(args)
    rargs.dataset_list = DATASETS
    rargs.num_train_batch = args.regmean_num_train_batch
    merger = RegMean(rargs, logger=None)
    model_list = []
    gram_list = []
    for dataset in DATASETS:
        if dataset == "CIFAR100":
            ckpt = (
                Path(args.ckpt_dir)
                / args.model
                / f"CIFAR100_{args.method_name}_CIFAR100_Tgt_{args.target_cls}_L_{args.patch_size}"
                / "finetuned.pt"
            )
        else:
            ckpt = Path(args.ckpt_dir) / args.model / dataset / "finetuned.pt"
        encoder = torch.load(ckpt, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        model = ImageClassifier(encoder, merger.class_head_dict[dataset])
        model.freeze_head()
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        model_list.append(model)
        with torch.no_grad():
            gram = merger.compute_gram(model, dataset)
        gram_list.append({k: v.detach().cpu() for k, v in gram.items()})
        model.cpu()
        torch.cuda.empty_cache()
    avg_params = merger.avg_merge(model_list, regmean_grams=gram_list)
    encoder = torch.load(
        Path(args.ckpt_dir) / args.model / "zeroshot.pt",
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    wrapper = ImageClassifier(encoder, merger.class_head_dict[DATASETS[-1]])
    wrapper.freeze_head()
    if torch.cuda.is_available():
        wrapper = wrapper.cuda()
    merger.copy_params_to_model(avg_params, wrapper)
    return {k: v.detach().cpu().clone() for k, v in wrapper.image_encoder.state_dict().items()}


def adamerging_state(args, ptm_state, task_states):
    path = args.adamerging_path or str(
        Path("./ada")
        / args.model
        / f"{args.method_name}_CIFAR100_Tgt_{args.target_cls}_L_{args.patch_size}_Epoch_500.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lambdas_raw = torch.load(path, map_location="cpu")
    lambdas = torch.clamp(lambdas_raw, min=0.0, max=1.0)
    ordered = [task_states["CIFAR100_BMR"]] + [task_states[d] for d in DATASETS[1:]]
    if lambdas.shape[1] != len(ordered):
        raise RuntimeError(f"Unexpected AdaMerging lambda shape: {tuple(lambdas.shape)}")
    keys = list(ptm_state.keys())
    if lambdas.shape[0] != len(keys):
        raise RuntimeError(f"Ada rows {lambdas.shape[0]} != state keys {len(keys)}")
    merged = {}
    for row, key in enumerate(keys):
        base = ptm_state[key]
        if not torch.is_floating_point(base):
            merged[key] = base.clone()
            continue
        value = base.clone()
        for col, ft_state in enumerate(ordered):
            value = value + lambdas[row, col].to(value.dtype) * (ft_state[key] - base)
        merged[key] = value
    return merged


def load_encoder_with_state(ptm_path, state):
    encoder = torch.load(ptm_path, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    encoder.eval()
    return encoder


def sample_indices(dataset, target_cls, n_target, n_nontarget, seed):
    labels = np.asarray(dataset.targets)
    target_ids = np.flatnonzero(labels == target_cls)
    non_ids = np.flatnonzero(labels != target_cls)
    rng = np.random.default_rng(seed)
    target_ids = np.sort(rng.choice(target_ids, size=min(n_target, len(target_ids)), replace=False))
    non_ids = np.sort(rng.choice(non_ids, size=min(n_nontarget, len(non_ids)), replace=False))
    return target_ids.tolist(), non_ids.tolist()


def trigger_tensors(args):
    path = Path("./trigger") / args.model / (
        f"{args.trigger_source}_CIFAR100_Tgt_{args.target_cls}_L_{args.patch_size}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    trigger = torch.from_numpy(np.load(path))
    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return str(path), torch.from_numpy(applied_patch).float(), torch.from_numpy(mask).float()


def extract_group_features(encoder, dataset, indices, batch_size, applied_patch=None, mask=None):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=4)
    feats = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for data in loader:
            data = maybe_dictionarize(data)
            x = data["images"].float()
            if applied_patch is not None and mask is not None:
                x = mask * applied_patch + (1 - mask.expand_as(x)) * x
            x = x.to(device)
            features = encoder(x)
            feats.append(features.detach().cpu())
    return torch.cat(feats, dim=0).float().numpy()


def fit_joint_embedding(features, seed, perplexity):
    x = np.vstack(features)
    x = StandardScaler().fit_transform(x)
    n_pca = min(50, x.shape[0] - 1, x.shape[1])
    x = PCA(n_components=n_pca, random_state=seed).fit_transform(x)
    z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)
    return z


def split_embedding(z, panel_order, group_sizes):
    out = {}
    start = 0
    for panel in panel_order:
        out[panel] = {}
        for group, size in group_sizes.items():
            out[panel][group] = z[start : start + size]
            start += size
    return out


def separator_and_ratio(panel_coords):
    target = panel_coords["target_clean"]
    non = panel_coords["nontarget_clean"]
    trig = panel_coords["triggered"]
    x = np.vstack([target, non])
    y = np.concatenate([np.ones(len(target)), np.zeros(len(non))])
    clf = LogisticRegression(solver="lbfgs", random_state=0).fit(x, y)
    ratio = float((clf.predict_proba(trig)[:, 1] >= 0.5).mean() * 100.0)
    return clf, ratio


def plot_figure(args, coords_by_panel, out_dir):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "target_clean": "#1f77b4",
        "nontarget_clean": "#7f7f7f",
        "triggered": "#d62728",
    }
    labels = {
        "target_clean": "Target clean",
        "nontarget_clean": "Non-target clean",
        "triggered": "Triggered",
    }
    all_xy = np.vstack(
        [coords_by_panel[p][g] for p, _ in PANELS for g in ("target_clean", "nontarget_clean", "triggered")]
    )
    pad_x = 0.05 * (all_xy[:, 0].max() - all_xy[:, 0].min())
    pad_y = 0.05 * (all_xy[:, 1].max() - all_xy[:, 1].min())
    xlim = (all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ylim = (all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)

    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.05), sharex=True, sharey=True)
    ratios = {}
    for ax, (panel, title) in zip(axes, PANELS):
        panel_coords = coords_by_panel[panel]
        clf, ratio = separator_and_ratio(panel_coords)
        ratios[panel] = ratio
        for group in ("nontarget_clean", "target_clean", "triggered"):
            xy = panel_coords[group]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=7 if group != "triggered" else 9,
                c=colors[group],
                marker="x" if group == "triggered" else "o",
                linewidths=0.5 if group == "triggered" else 0.0,
                alpha=0.68 if group != "triggered" else 0.84,
                label=labels[group],
                rasterized=True,
            )
        xx = np.linspace(xlim[0], xlim[1], 200)
        w = clf.coef_[0]
        b = clf.intercept_[0]
        if abs(w[1]) > 1e-8:
            yy = -(w[0] * xx + b) / w[1]
            ax.plot(xx, yy, color="black", linestyle=(0, (3, 2)), linewidth=0.8)
        if not args.no_annotation:
            ax.text(
                0.04,
                0.06,
                f"{ratio:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                color="black",
            )
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=6, length=2)
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("t-SNE 2", fontsize=7)
    for ax in axes:
        ax.set_xlabel("t-SNE 1", fontsize=7)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.6, w_pad=0.4)
    png_path = out_dir / "bmr_feature_space_cifar100.png"
    pdf_path = out_dir / "bmr_feature_space_cifar100.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return ratios, png_path, pdf_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ptm_path, ptm_state, task_states = load_task_states(args)
    receiver_states = {
        "clean": task_states["CIFAR100"],
        "ta": vector_merge_state(args, "ta", ptm_state, task_states),
        "ties": vector_merge_state(args, "ties", ptm_state, task_states),
        "regmean": regmean_state(args),
        "adamerging": adamerging_state(args, ptm_state, task_states),
    }

    probe_encoder = torch.load(ptm_path, map_location="cpu")
    dataset, _ = get_dataset(
        "CIFAR100",
        "test",
        probe_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4,
    )
    target_ids, non_ids = sample_indices(
        dataset,
        args.target_cls,
        args.n_target_clean,
        args.n_nontarget_clean,
        args.seed,
    )
    trigger_path, applied_patch, mask = trigger_tensors(args)

    flat_features = []
    panel_order = [p for p, _ in PANELS]
    group_sizes = {
        "target_clean": len(target_ids),
        "nontarget_clean": len(non_ids),
        "triggered": len(non_ids),
    }
    for panel in panel_order:
        print(f"[feature-space] extracting {panel}")
        encoder = load_encoder_with_state(ptm_path, receiver_states[panel])
        flat_features.append(extract_group_features(encoder, dataset, target_ids, args.batch_size))
        flat_features.append(extract_group_features(encoder, dataset, non_ids, args.batch_size))
        flat_features.append(
            extract_group_features(
                encoder,
                dataset,
                non_ids,
                args.batch_size,
                applied_patch=applied_patch,
                mask=mask,
            )
        )
        encoder.cpu()
        torch.cuda.empty_cache()

    print("[feature-space] fitting joint PCA+t-SNE")
    z = fit_joint_embedding(flat_features, args.seed, args.tsne_perplexity)
    coords_by_panel = split_embedding(z, panel_order, group_sizes)
    ratios, png_path, pdf_path = plot_figure(args, coords_by_panel, out_dir)

    rows = []
    for panel, title in PANELS:
        for group, xy in coords_by_panel[panel].items():
            for i, point in enumerate(xy):
                sample_id = target_ids[i] if group == "target_clean" else non_ids[i]
                rows.append(
                    {
                        "panel": panel,
                        "panel_title": title,
                        "group": group,
                        "sample_id": sample_id,
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                )
    csv_path = out_dir / "bmr_feature_space_cifar100_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at": time.strftime("%Y%m%d_%H%M%S"),
        "config": vars(args),
        "trigger_path": trigger_path,
        "target_ids": target_ids,
        "nontarget_ids": non_ids,
        "target_side_ratio": ratios,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "points_csv": str(csv_path),
        },
    }
    json_path = out_dir / "bmr_feature_space_cifar100_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
