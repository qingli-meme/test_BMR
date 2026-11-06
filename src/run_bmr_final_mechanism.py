#!/usr/bin/env python3
"""Run experiments 01-04 from BMR_final_mechanism_experiments.zip.

Experiment 05 (multi-seed confirmation) is intentionally not implemented here.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.eval import eval_single_dataset
from src.paper_refresh import run_gain_deficit_refresh as gain_tools
from src.reviewer_exp.common import default_args, load_encoder, sha256_file
from src.reviewer_exp.merge_existing import task_arithmetic, ties


BASE6 = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
TARGET = "CIFAR100"
TARGET_CLS = 1
CKPT_ROOT = ROOT / "checkpoints" / "ViT-B-32"
PRETRAINED = CKPT_ROOT / "zeroshot.pt"
CLEAN_CIFAR = CKPT_ROOT / "CIFAR100" / "finetuned.pt"
TRIGGER = ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
OUT = ROOT / "analysis/bmr_final_mechanism"

VARIANTS = {
    "scale": {
        "display": "Scale Variation",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_SLIM_SCALE_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": True,
        "loss": "normal BD",
    },
    "absolute": {
        "display": "Absolute Margin",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_SLIM_ABS_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": True,
        "loss": "absolute",
    },
    "softplus": {
        "display": "Softplus Margin",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_SOFTPLUS_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": True,
        "loss": "softplus",
    },
    "fixed_d": {
        "display": "BMR, fixed D",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_FIXEDD_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": True,
        "loss": "fixed-D BMR",
    },
    "fixed_alpha": {
        "display": "BMR, fixed alpha",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_FIXALPHA_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": False,
        "loss": "BMR",
    },
    "bmr": {
        "display": "BMR",
        "checkpoint": CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt",
        "vary_alpha": True,
        "loss": "sample-wise BMR",
    },
}

ORDINARY = CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_SLIM_ORD_CIFAR100_Tgt_1_L_22/finetuned.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["loss", "merge-config", "residual", "active-region", "finalize"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def task_paths(attack_checkpoint=None):
    paths = []
    for task in BASE6:
        if task == TARGET:
            paths.append(Path(attack_checkpoint) if attack_checkpoint else CLEAN_CIFAR)
        else:
            paths.append(CKPT_ROOT / task / "finetuned.pt")
    return paths


def merge_args(args):
    return SimpleNamespace(
        device=args.device,
        batch_size=args.batch_size,
        data_location=str(ROOT / "data"),
        # run_gain_deficit_refresh.collect supports truncated diagnostic runs,
        # while this experiment always evaluates the complete test split.
        max_eval_batches=0,
    )


def build_function_merge(method, paths, args):
    ma = merge_args(args)
    if method == "regmean":
        return gain_tools.build_regmean(BASE6, paths, ma)
    if method == "regmean_plusplus":
        return gain_tools.build_regmeanpp(BASE6, paths, ma)
    raise ValueError(method)


def save_receiver(receiver, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(receiver.state_dict(), path)


def load_receiver(path):
    return gain_tools.enc_from_state(gain_tools.state(path))


def cached_receiver(path, builder):
    path = Path(path)
    if path.is_file():
        return load_receiver(path)
    receiver = builder()
    save_receiver(receiver, path)
    return receiver


def clean_accuracy(receiver, args):
    ea = default_args(args.device, args.batch_size)
    ea.num_workers = 4
    per_task = {}
    receiver = receiver.to(args.device).eval()
    for task in BASE6:
        per_task[task] = 100.0 * eval_single_dataset(receiver, task, ea)["top1"]
    receiver.cpu()
    torch.cuda.empty_cache()
    return float(np.mean(list(per_task.values()))), per_task


def margin_delta(background, attacked, args):
    ids_bg, margin_bg = gain_tools.collect(background, merge_args(args))
    ids_atk, margin_atk = gain_tools.collect(attacked, merge_args(args))
    if not np.array_equal(ids_bg, ids_atk):
        raise AssertionError("triggered sample IDs differ between clean and attacked merges")
    delta = margin_atk - margin_bg
    return {
        "n": int(len(delta)),
        "asr": 100.0 * float(np.mean(margin_atk > 0)),
        "median_delta_m": float(np.quantile(delta, 0.5)),
        "q10_delta_m": float(np.quantile(delta, 0.1)),
    }


def evaluate_loss_variant(method, key, checkpoint, args, cache_dir):
    bg_path = cache_dir / f"{method}__clean.pt"
    atk_path = cache_dir / f"{method}__{key}.pt"
    background = cached_receiver(
        bg_path, lambda: build_function_merge(method, task_paths(), args)
    )
    attacked = cached_receiver(
        atk_path,
        lambda: build_function_merge(method, task_paths(checkpoint), args),
    )
    metrics = margin_delta(background, attacked, args)
    avg_acc, per_task = clean_accuracy(attacked, args)
    metrics.update(
        {
            "variant_key": key,
            "variant": VARIANTS.get(key, {"display": key})["display"],
            "merger": method,
            "avg_clean_acc": avg_acc,
            "target_clean_acc": per_task[TARGET],
            "per_task_acc": per_task,
            "checkpoint": str(Path(checkpoint).resolve()),
            "seed": 2026,
        }
    )
    del background, attacked
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def phase_loss(args):
    out = OUT / "loss_ablation"
    cache = out / "merge_cache"
    out.mkdir(parents=True, exist_ok=True)
    for item in VARIANTS.values():
        if not item["checkpoint"].is_file():
            raise FileNotFoundError(item["checkpoint"])
    result_path = out / "loss_ablation_full.json"
    rows = json.loads(result_path.read_text()) if result_path.is_file() else []
    done = {(row["merger"], row["variant_key"]) for row in rows}
    for method in ("regmean", "regmean_plusplus"):
        for key, item in VARIANTS.items():
            if (method, key) in done:
                continue
            row = evaluate_loss_variant(method, key, item["checkpoint"], args, cache)
            rows.append(row)
            write_json(result_path, rows)
            write_csv(out / "loss_ablation_full.csv", rows)

    lookup = {(r["variant_key"], r["merger"]): r for r in rows}
    main_rows = []
    for key, item in VARIANTS.items():
        rm = lookup[(key, "regmean")]
        rpp = lookup[(key, "regmean_plusplus")]
        main_rows.append(
            {
                "Variant": item["display"],
                "Vary alpha": "yes" if item["vary_alpha"] else "no",
                "Loss": item["loss"],
                "RegMean ASR": rm["asr"],
                "RegMean q10 DeltaM": rm["q10_delta_m"],
                "RegMean++ ASR": rpp["asr"],
                "RegMean++ q10 DeltaM": rpp["q10_delta_m"],
            }
        )
    write_csv(out / "loss_ablation_main_table.csv", main_rows)
    manifest = []
    for key, item in VARIANTS.items():
        manifest.append(
            {
                "variant": key,
                "checkpoint": str(item["checkpoint"].resolve()),
                "sha256": sha256_file(item["checkpoint"]),
                "seed": 2026,
            }
        )
    write_json(out / "checkpoints_manifest.json", manifest)
    lines = ["# Loss mechanism ablation", "", "| Variant | RM ASR | RM q10 ΔM | RM++ ASR | RM++ q10 ΔM |", "|---|---:|---:|---:|---:|"]
    for row in main_rows:
        lines.append(
            f"| {row['Variant']} | {row['RegMean ASR']:.4f} | {row['RegMean q10 DeltaM']:.6f} | "
            f"{row['RegMean++ ASR']:.4f} | {row['RegMean++ q10 DeltaM']:.6f} |"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def vector_receiver(method, paths, parameter, args):
    if method == "ta":
        return task_arithmetic(PRETRAINED, paths, scale=float(parameter))
    if method == "ties":
        return ties(PRETRAINED, paths, scale=0.3, reset_thresh=float(parameter))
    raise ValueError(method)


def phase_merge_config(args):
    out = OUT / "merge_config_sweep"
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "merge_config_full.json"
    rows = json.loads(full_path.read_text()) if full_path.is_file() else []
    done = {(r["merger"], float(r["value"]), r["variant_key"]) for r in rows}
    variants = {key: VARIANTS[key] for key in ("scale", "fixed_alpha", "bmr")}
    configs = {
        "ta": ("scaling_coef", [0.1, 0.2, 0.3, 0.4, 0.5], 0.3),
        "ties": ("reset_thresh", [5.0, 10.0, 20.0, 30.0, 40.0], 20.0),
    }
    clean_rows = []
    for merger, (parameter_name, values, default) in configs.items():
        default_clean = None
        config_cache = {}
        order = [default] + [v for v in values if v != default]
        for value in order:
            clean = vector_receiver(merger, task_paths(), value, args)
            clean_avg, clean_per = clean_accuracy(clean, args)
            _, clean_margin = gain_tools.collect(clean, merge_args(args))
            if value == default:
                default_clean = clean_avg
            config_cache[value] = (clean_avg, clean_per, clean_margin)
            clean_rows.append(
                {
                    "Merger": merger,
                    "Parameter": parameter_name,
                    "Value": value,
                    "Clean-only ACC": clean_avg,
                    "Paper default": value == default,
                }
            )
            del clean
            torch.cuda.empty_cache()

        for value in values:
            clean_avg, _, clean_margin = config_cache[value]
            utility_valid = abs(clean_avg - default_clean) <= 3.0
            for key, item in variants.items():
                marker = (merger, float(value), key)
                if marker in done:
                    continue
                attacked = vector_receiver(merger, task_paths(item["checkpoint"]), value, args)
                attacked_avg, attacked_per = clean_accuracy(attacked, args)
                _, attacked_margin = gain_tools.collect(attacked, merge_args(args))
                delta = attacked_margin - clean_margin
                row = {
                    "merger": merger,
                    "parameter": parameter_name,
                    "value": value,
                    "variant_key": key,
                    "variant": item["display"],
                    "asr": 100.0 * float(np.mean(attacked_margin > 0)),
                    "clean_only_acc": clean_avg,
                    "attacked_acc": attacked_avg,
                    "delta_acc": attacked_avg - clean_avg,
                    "median_delta_m": float(np.quantile(delta, 0.5)),
                    "q10_delta_m": float(np.quantile(delta, 0.1)),
                    "utility_valid": utility_valid,
                    "attacked_per_task_acc": attacked_per,
                }
                rows.append(row)
                done.add(marker)
                write_json(full_path, rows)
                write_csv(out / "merge_config_full.csv", rows)
                del attacked
                gc.collect()
                torch.cuda.empty_cache()

    write_csv(out / "clean_only_configs.csv", clean_rows)
    write_csv(
        out / "merge_config_summary.csv",
        [
            {k: v for k, v in row.items() if k != "attacked_per_task_acc"}
            for row in rows
        ],
    )
    write_json(
        out / "configuration_manifest.json",
        {
            "task_arithmetic": {"parameter": "scaling_coef", "values": configs["ta"][1], "default": 0.3},
            "ties": {"parameter": "reset_thresh", "values": configs["ties"][1], "default": 20},
            "ties_fixed_scale": 0.3,
            "ties_merge_func": "dis-sum",
            "utility_valid_threshold_pp": 3.0,
            "variants": {k: str(v["checkpoint"].resolve()) for k, v in variants.items()},
        },
    )
    (out / "RESULTS.md").write_text(
        "# Merge-configuration robustness\n\nSee `merge_config_summary.csv` for the complete sweep.\n",
        encoding="utf-8",
    )


def state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def two_norms(checkpoint, base, clean):
    current = state_dict(checkpoint)
    base_sq = 0.0
    clean_sq = 0.0
    for name, value in current.items():
        if name not in base or name not in clean or not torch.is_floating_point(value):
            continue
        x = value.detach().double().cpu()
        base_sq += float(((x - base[name].detach().double().cpu()) ** 2).sum())
        clean_sq += float(((x - clean[name].detach().double().cpu()) ** 2).sum())
    return math.sqrt(base_sq), math.sqrt(clean_sq)


def make_norm_matched_checkpoint(source_checkpoint, target_norm, output_path):
    base = state_dict(PRETRAINED)
    source = state_dict(source_checkpoint)
    source_norm, _ = two_norms(source_checkpoint, base, state_dict(CLEAN_CIFAR))
    scale = target_norm / max(source_norm, 1e-30)
    matched = {}
    achieved_sq = 0.0
    for name, value in source.items():
        if name in base and torch.is_floating_point(value):
            p0 = base[name].to(dtype=value.dtype)
            value = p0 + scale * (value - p0)
            achieved_sq += float(
                ((value.detach().double().cpu() - base[name].detach().double().cpu()) ** 2).sum()
            )
        matched[name] = value
    encoder = load_encoder(PRETRAINED)
    encoder.load_state_dict(matched, strict=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder, output_path)
    return source_norm, scale, math.sqrt(achieved_sq)


def correlation(xs, ys):
    from scipy.stats import pearsonr, spearmanr

    return {
        "pearson": float(pearsonr(xs, ys).statistic),
        "spearman": float(spearmanr(xs, ys).statistic),
    }


def phase_residual(args):
    out = OUT / "residual_norm"
    out.mkdir(parents=True, exist_ok=True)
    loss_rows = json.loads((OUT / "loss_ablation/loss_ablation_full.json").read_text())
    result = {(r["variant_key"], r["merger"]): r for r in loss_rows}
    checkpoints = {"ordinary": ORDINARY, **{k: v["checkpoint"] for k, v in VARIANTS.items()}}
    displays = {"ordinary": "Ordinary", **{k: v["display"] for k, v in VARIANTS.items()}}
    base = state_dict(PRETRAINED)
    clean = state_dict(CLEAN_CIFAR)
    rows = []
    for key, checkpoint in checkpoints.items():
        norm0, normclean = two_norms(checkpoint, base, clean)
        rows.append(
            {
                "variant_key": key,
                "Variant": displays[key],
                "norm_theta_adv_minus_theta0": norm0,
                "norm_theta_adv_minus_theta_clean": normclean,
                "RegMean ASR": result.get((key, "regmean"), {}).get("asr"),
                "RegMean++ ASR": result.get((key, "regmean_plusplus"), {}).get("asr"),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    usable = [r for r in rows if r["RegMean ASR"] is not None]
    norms = [r["norm_theta_adv_minus_theta0"] for r in usable]
    corr = {
        "regmean": correlation(norms, [r["RegMean ASR"] for r in usable]),
        "regmean_plusplus": correlation(norms, [r["RegMean++ ASR"] for r in usable]),
    }
    full = next(r for r in rows if r["variant_key"] == "bmr")
    competitors = [r for r in usable if r["variant_key"] in {"scale", "absolute", "softplus", "fixed_d"}]
    strongest = max(competitors, key=lambda r: (r["RegMean ASR"] + r["RegMean++ ASR"]) / 2)
    norm_larger = full["norm_theta_adv_minus_theta0"] > 1.1 * strongest["norm_theta_adv_minus_theta0"]
    tracks = corr["regmean"]["pearson"] > 0 and corr["regmean_plusplus"]["pearson"] > 0
    decision = {
        "full_bmr_gt_10pct_strongest_control": norm_larger,
        "positive_norm_asr_tracking": tracks,
        "norm_matched_control_required": bool(norm_larger and tracks),
        "strongest_non_bmr_control": strongest["Variant"],
    }
    matched_rows = []
    if decision["norm_matched_control_required"]:
        matched_path = out / "norm_matched_strongest_control.pt"
        source_norm, scale, achieved_norm = make_norm_matched_checkpoint(
            checkpoints[strongest["variant_key"]],
            full["norm_theta_adv_minus_theta0"],
            matched_path,
        )
        for method in ("regmean", "regmean_plusplus"):
            metric = evaluate_loss_variant(
                method,
                "norm_matched",
                matched_path,
                args,
                out / "merge_cache",
            )
            metric.update(
                {
                    "source_variant": strongest["Variant"],
                    "source_norm": source_norm,
                    "target_norm": full["norm_theta_adv_minus_theta0"],
                    "achieved_norm": achieved_norm,
                    "residual_scale": scale,
                }
            )
            matched_rows.append(metric)
        write_csv(out / "norm_matched_control.csv", matched_rows)
        # The checkpoint is an intermediate diagnostic artifact and is not
        # included in the return archive.
    write_csv(out / "residual_norms.csv", rows)
    write_json(out / "residual_norms.json", rows)
    write_json(
        out / "norm_asr_correlations.json",
        {"correlations": corr, "decision": decision, "norm_matched": matched_rows},
    )
    (out / "RESULTS.md").write_text(
        "# Residual-norm diagnostic\n\n"
        f"Norm-matched control required: **{decision['norm_matched_control_required']}**.\n",
        encoding="utf-8",
    )


def phase_active_region(args):
    out = OUT / "active_region"
    out.mkdir(parents=True, exist_ok=True)
    run_dir = CKPT_ROOT / "CIFAR100_KDR_DTK_BMR_ACTIVE_CIFAR100_Tgt_1_L_22"
    logs = sorted(run_dir.glob("*_dtk_bmr_train_log.jsonl"))
    if not logs:
        raise FileNotFoundError(f"active-region training log missing under {run_dir}")
    raw = []
    for line in logs[-1].read_text().splitlines():
        row = json.loads(line)
        if "global_step" in row:
            raw.append(row)
    steps = [
        {
            "epoch": r["epoch"],
            "batch": r["batch"],
            "global_step": r["global_step"],
            "triggered_samples": r["triggered_sample_count"],
            "fraction_G_le_epsilon": r["margin_gain_le_epsilon_rate"],
            "fraction_G_le_zero": r["margin_gain_le_zero_rate"],
            "mean_G": r["margin_gain_mean"],
            "median_G": r["margin_gain_median"],
            "q10_G": r["margin_gain_q10"],
            "q90_G": r["margin_gain_q90"],
            "mean_D": r["background_deficit_mean"],
            "median_D": r["background_deficit_median"],
            "BMR_loss": r["reserve_loss"],
            "normal_backdoor_loss": r["bd_loss"],
        }
        for r in raw
    ]
    epoch_rows = []
    for epoch in sorted({r["epoch"] for r in steps}):
        subset = [r for r in steps if r["epoch"] == epoch]
        epoch_rows.append(
            {
                "epoch": epoch,
                **{
                    key: float(np.mean([r[key] for r in subset]))
                    for key in steps[0]
                    if key not in {"epoch", "batch", "global_step"}
                },
            }
        )
    write_csv(out / "active_region_steps.csv", steps)
    write_csv(out / "active_region_epochs.csv", epoch_rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot([r["global_step"] for r in steps], [r["fraction_G_le_epsilon"] for r in steps], label=r"$G\leq\epsilon$")
    ax.plot([r["global_step"] for r in steps], [r["fraction_G_le_zero"] for r in steps], label=r"$G\leq0$", alpha=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Inactive fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "active_region_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    inactive = [r["fraction_G_le_epsilon"] for r in steps]
    text = (
        "# Training active-region diagnostic\n\n"
        f"- epsilon: 1e-6\n- initial inactive fraction: {inactive[0]:.6f}\n"
        f"- final inactive fraction: {inactive[-1]:.6f}\n"
        f"- median inactive fraction: {float(np.median(inactive)):.6f}\n"
    )
    (out / "RESULTS.md").write_text(text, encoding="utf-8")


def phase_finalize(args):
    loss = (OUT / "loss_ablation/RESULTS.md").read_text()
    active = (OUT / "active_region/RESULTS.md").read_text()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff_stat = subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True)
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    results = (
        "# BMR Final Mechanism Experiments (01–04)\n\n"
        f"- Git commit at launch/finalization: `{commit}`\n"
        "- Experiment 05 multi-seed confirmation: **deferred by user request**\n\n"
        "## Loss mechanism\n\n" + loss.split("\n", 2)[-1] + "\n"
        "## Merge configuration\n\nSee `merge_config_sweep/merge_config_summary.csv`.\n\n"
        "## Residual norm\n\nSee `residual_norm/residual_norms.csv`.\n\n"
        "## Active region\n\n" + active.split("\n", 2)[-1] + "\n"
        "## Repository state\n\n```text\n" + diff_stat + "\n" + status + "```\n"
    )
    (OUT / "RESULTS.md").write_text(results, encoding="utf-8")
    (OUT / "MULTISEED_DEFERRED.txt").write_text(
        "Experiment 05 was intentionally not run. Run only later when GPUs are available.\n",
        encoding="utf-8",
    )
    archive = ROOT / "bmr_final_mechanism_experiments_01_04_return.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in OUT.rglob("*"):
            if path.is_file() and path.suffix not in {".pt", ".pth"}:
                handle.add(path, arcname=path.relative_to(ROOT))
        for path in (
            ROOT / "src/run_bmr_final_mechanism.py",
            ROOT / "scripts/run_bmr_final_mechanism_queue.sh",
            ROOT / "scripts/run_bmr_final_mechanism_posttrain.sh",
        ):
            handle.add(path, arcname=path.relative_to(ROOT))
    print(OUT / "RESULTS.md")
    print(archive)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "loss": phase_loss,
        "merge-config": phase_merge_config,
        "residual": phase_residual,
        "active-region": phase_active_region,
        "finalize": phase_finalize,
    }
    dispatch[args.phase](args)


if __name__ == "__main__":
    main()
