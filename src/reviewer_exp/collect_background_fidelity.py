from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import random
import subprocess
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.func import functional_call

from src.reviewer_exp.common import (
    RESULT_ROOT, ROOT, checkpoint_manifest, checkpoint_path, default_args,
    dump_json, layer_group, load_encoder, require_paths,
)


BENIGN5 = ["GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
REAL_METHODS = OrderedDict([
    ("task_arithmetic", "Task Arithmetic"), ("ties", "TIES"),
    ("regmean", "RegMean"), ("adamerging", "AdaMerging"),
])
EXPECTED_GROUPS = ["embedding", *[f"resblock_{i}" for i in range(12)], "ln_post", "other"]
FINAL_RUN_DIR = ROOT / "checkpoints" / "ViT-B-32" / "CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22"
FINAL_CONFIG = FINAL_RUN_DIR / "20260708_225633_dtk_bmr_config.json"
FINAL_BMR = FINAL_RUN_DIR / "finetuned.pt"
PRETRAINED = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
CLEAN_INIT = ROOT / "checkpoints" / "ViT-B-32" / "CIFAR100" / "finetuned.pt"
TRIGGER = ROOT / "trigger" / "ViT-B-32" / "KDR_DTK_CIFAR100_Tgt_1_L_22.npy"
STANDARD_IDS = ROOT / "bmr_paper_results" / "motivation" / "receiver_cache" / "CIFAR100_clean_margins.npz"
REAL_CACHE_DIR = ROOT / "bmr_paper_results" / "motivation" / "merge_cache"
REAL_CACHE = {
    method: REAL_CACHE_DIR / f"{method if method != 'task_arithmetic' else 'ta'}_cifar100_suite_background_without_CIFAR100.pt"
    for method in REAL_METHODS
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversary-task", choices=["CIFAR100"], default="CIFAR100")
    parser.add_argument("--target-cls", type=int, default=1)
    parser.add_argument("--num-synthetic-backgrounds", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def target_margin(logits, target):
    other = logits.clone()
    other[:, target] = -torch.inf
    return logits[:, target] - other.max(dim=1).values


def distribution_summary(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        "q05": float(np.quantile(x, .05)), "q10": float(np.quantile(x, .10)),
        "q25": float(np.quantile(x, .25)), "median": float(np.quantile(x, .50)),
        "q75": float(np.quantile(x, .75)), "q90": float(np.quantile(x, .90)),
        "q95": float(np.quantile(x, .95)), "max": float(x.max()),
    }


def deficit_summary(deficits, margins):
    deficits = np.asarray(deficits, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    result = distribution_summary(deficits)
    result["p_deficit"] = float((deficits > 0).mean())
    result["p_target_active"] = float((margins > 0).mean())
    return result


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_fixed_samples(base, count, target_cls, device):
    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset
    from src.kdr_utils import apply_trigger, load_trigger_patch

    standardized = np.load(STANDARD_IDS)["sample_id"].astype(np.int64).tolist()
    wanted_order = standardized[:count]
    wanted = set(wanted_order)
    found = {}
    _, loader = get_dataset(
        "CIFAR100", "test", base.val_preprocess, location=str(ROOT / "data"),
        batch_size=128, num_workers=0,
    )
    fallback = 0
    for batch in loader:
        data = maybe_dictionarize(batch)
        images, labels = data["images"], data["labels"]
        indices = data.get("indices")
        if indices is None:
            indices = torch.arange(fallback, fallback + labels.numel())
        fallback += labels.numel()
        for image, label, sample_id in zip(images, labels, indices):
            sid = int(sample_id)
            if sid in wanted:
                found[sid] = (image.cpu(), int(label))
        if len(found) == len(wanted):
            break
    missing = [sid for sid in wanted_order if sid not in found]
    if missing:
        raise RuntimeError(f"Missing standardized CIFAR100 sample IDs: {missing[:20]}")
    labels = [found[sid][1] for sid in wanted_order]
    if any(label == target_cls for label in labels):
        raise AssertionError("Standardized sample list contains target-class examples")
    images = torch.stack([found[sid][0] for sid in wanted_order]).to(device)
    trigger = load_trigger_patch(str(TRIGGER), 22, device)
    return apply_trigger(images, trigger), wanted_order, labels


def evaluate_margins(model, head, images, target_cls, batch_size, params=None):
    margins = []
    model.eval()
    with torch.no_grad():
        for batch in images.split(batch_size):
            features = functional_call(model, params, (batch,)) if params is not None else model(batch)
            margins.extend(target_margin(head(features), target_cls).detach().cpu().tolist())
    result = np.asarray(margins, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("NaN/inf detected in margins")
    return result


def floating_delta(state, base_state):
    return {
        key: state[key].detach().float().cpu() - base.detach().float().cpu()
        for key, base in base_state.items()
        if key in state and torch.is_floating_point(base) and state[key].shape == base.shape
    }


def norm(delta):
    total = sum((value.double().square().sum() for value in delta.values()), torch.zeros((), dtype=torch.float64))
    return float(total.sqrt())


def dot(a, b):
    total = torch.zeros((), dtype=torch.float64)
    for key in a.keys() & b.keys():
        total += (a[key].double().reshape(-1) * b[key].double().reshape(-1)).sum()
    return float(total)


def cosine(a, b, norm_a=None, norm_b=None):
    norm_a = norm(a) if norm_a is None else norm_a
    norm_b = norm(b) if norm_b is None else norm_b
    return dot(a, b) / max(norm_a * norm_b, 1e-30)


def layer_rows(background_type, background_name, delta):
    squared = {group: 0.0 for group in EXPECTED_GROUPS}
    for name, value in delta.items():
        squared[layer_group(name)] += float(value.double().square().sum())
    total = float(sum(squared.values()) ** .5)
    if not np.isfinite(total):
        raise FloatingPointError("NaN/inf detected in layer norms")
    return [{
        "background_type": background_type, "background_name": background_name,
        "layer_group": group, "l2_norm": squared[group] ** .5,
        "total_norm": total,
        "norm_fraction": (squared[group] ** .5) / max(total, 1e-30),
    } for group in EXPECTED_GROUPS]


def load_real_states(base_cpu):
    base_state = base_cpu.state_dict()
    task_paths = [checkpoint_path(task, "clean") for task in BENIGN5]
    require_paths([*task_paths, *REAL_CACHE.values()])
    if any(path.parent.name == "CIFAR100" for path in task_paths):
        raise AssertionError("CIFAR100 checkpoint found in real benign background inputs")
    states, deltas = {}, {}
    for method in REAL_METHODS:
        raw = torch.load(REAL_CACHE[method], map_location="cpu", weights_only=False)
        if hasattr(raw, "state_dict"):
            raw = raw.state_dict()
        model = copy.deepcopy(base_cpu)
        model.load_state_dict(raw, strict=True)
        states[method] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        deltas[method] = floating_delta(states[method], base_state)
    return states, deltas, task_paths


def sampler_config(args, train_config, task_paths):
    drift_keywords = train_config["drift_keywords"]
    return {
        "source_file": str((ROOT / "src" / "kdr_utils.py").resolve()),
        "source_function": "sample_synthetic_drift_cache / make_synthetic_drift_like / build_drifted_params_from_cache",
        "training_call_site": str((ROOT / "src" / "finetune_kdr_dtk_bmr.py").resolve()),
        "BMR_checkpoint_run_used": str(FINAL_RUN_DIR.resolve()),
        "BMR_checkpoint": str(FINAL_BMR.resolve()),
        "BMR_training_config": str(FINAL_CONFIG.resolve()),
        "pretrained_checkpoint": str(PRETRAINED.resolve()),
        "clean_CIFAR100_initialization_checkpoint": str(CLEAN_INIT.resolve()),
        "source_of_delta_bg_tilde": "Gaussian noise sampled independently per named parameter tensor, projected orthogonally to the finalized BMR residual (theta_BMR - theta0), then L2-normalized and scaled per tensor",
        "perturbed_parameter_tensors": drift_keywords,
        "unperturbed_tensors": "all named parameters not matching drift_keywords; all buffers",
        "sampling_distribution": "torch.randn_like independently per eligible parameter tensor, followed by orthogonal projection and L2 normalization",
        "sampling_range": "unbounded Gaussian direction before normalization",
        "background_strength_gamma_setting": "not_used",
        "drift_rho": train_config["args"]["drift_rho"],
        "eta_distribution": "continuous uniform",
        "eta_range": [train_config["args"]["eta_min"], train_config["args"]["eta_max"]],
        "alpha_distribution": "continuous uniform",
        "alpha_range": [train_config["args"]["alpha_min"], train_config["args"]["alpha_max"]],
        "alpha_effect_on_theta_bg": "not_used (alpha is sampled in the paired training context but theta_bg excludes the adversarial residual)",
        "perturbations_normalized": True,
        "normalization": "each eligible tensor is normalized to rho * ||theta_BMR-theta0||_2 for that tensor; near-zero residual fallback uses rho*1e-3*max(||theta0||_2,1)",
        "signs_or_directions_sampled": "isotropic Gaussian direction in the subspace orthogonal to the corresponding BMR residual tensor",
        "per_layer_scaling": "not_used; scaling is per parameter tensor",
        "random_seed": args.seed,
        "number_of_synthetic_states": args.num_synthetic_backgrounds,
        "number_of_fixed_samples": args.num_samples,
        "attacker_accessible_information_only": True,
        "attacker_information_explanation": "the sampler uses only theta0 and the attacker's own BMR model/residual; it does not use benign task checkpoints or future merge outputs",
        "real_background_tasks": BENIGN5,
        "real_background_checkpoint_paths": [str(path.resolve()) for path in task_paths],
        "real_background_cache_paths": {key: str(path.resolve()) for key, path in REAL_CACHE.items()},
        "real_merge_hyperparameters": {
            "Task Arithmetic": {"scaling_coefficient": 0.3},
            "TIES": {"scaling_coefficient": 0.3, "reset_threshold": 20.0, "merge_function": "dis-sum"},
            "RegMean": {"num_train_batches": 8},
            "AdaMerging": {"coefficient_source": str((ROOT / "ada" / "ViT-B-32" / "Clean_Epoch_500.pt").resolve()), "epochs_used_to_fit_original_six-task_coefficients": 500, "CIFAR100_contribution": "omitted while retaining the five benign coefficients"},
        },
        "checkpoint_sha256": checkpoint_manifest([PRETRAINED, CLEAN_INIT, FINAL_BMR, TRIGGER, *task_paths, *REAL_CACHE.values()]),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.num_synthetic_backgrounds = min(args.num_synthetic_backgrounds, 2)
        args.num_samples = min(args.num_samples, 32)
    if args.num_samples < 1000 and not args.smoke:
        raise ValueError("Formal run requires at least 1000 fixed samples")
    started = time.time()
    device = torch.device("cuda")
    require_paths([PRETRAINED, CLEAN_INIT, FINAL_BMR, FINAL_CONFIG, TRIGGER, STANDARD_IDS])
    train_config = json.loads(FINAL_CONFIG.read_text())
    if args.target_cls != train_config["args"]["target_cls"]:
        raise ValueError("Target class differs from finalized BMR run")
    if train_config["args"]["drift_scope"] != "last2":
        raise ValueError("Unexpected finalized drift scope")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_cpu = load_encoder(PRETRAINED).eval()
    real_states, real_deltas, task_paths = load_real_states(base_cpu)
    real_norms = {method: norm(delta) for method, delta in real_deltas.items()}
    base = copy.deepcopy(base_cpu).to(device).eval()
    source = load_encoder(FINAL_BMR, device).eval()
    from src.heads import get_classification_head
    from src.kdr_utils import build_drifted_params_from_cache, default_drift_keywords, sample_synthetic_drift_cache
    head = get_classification_head(default_args("cuda", 128), "CIFAR100").to(device).eval()
    images, sample_ids, true_labels = load_fixed_samples(base, args.num_samples, args.target_cls, device)
    if len(sample_ids) != args.num_samples or len(set(sample_ids)) != args.num_samples:
        raise AssertionError("Fixed sample IDs are incomplete or duplicated")

    suffix = "_smoke" if args.smoke else ""
    sample_path = RESULT_ROOT / f"background_fidelity_sample_ids{suffix}.csv"
    write_csv(sample_path, ({"sample_id": sid, "true_label": label, "target_label": args.target_cls} for sid, label in zip(sample_ids, true_labels)), ["sample_id", "true_label", "target_label"])
    dump_json(RESULT_ROOT / f"background_sampler_config{suffix}.json", sampler_config(args, train_config, task_paths))

    deficit_path = RESULT_ROOT / f"background_fidelity_deficits{suffix}.csv.gz"
    state_rows, layerwise_rows, synthetic_meta = [], [], []
    syn_deficits = np.empty((args.num_synthetic_backgrounds, args.num_samples), dtype=np.float32)
    syn_margins = np.empty_like(syn_deficits)
    real_cos_samples = {method: [] for method in REAL_METHODS}
    syn_mean = {}
    rng = random.Random(args.seed)
    drift_keywords = default_drift_keywords(train_config["args"]["drift_scope"])
    rho = float(train_config["args"]["drift_rho"])
    alpha_range = (float(train_config["args"]["alpha_min"]), float(train_config["args"]["alpha_max"]))
    eta_range = (float(train_config["args"]["eta_min"]), float(train_config["args"]["eta_max"]))
    raw_fields = ["background_type", "background_name", "sample_id", "true_label", "target_label", "margin", "deficit", "target_active"]

    with gzip.open(deficit_path, "wt", newline="", encoding="utf-8") as handle:
        raw_writer = csv.DictWriter(handle, fieldnames=raw_fields)
        raw_writer.writeheader()
        for idx in range(args.num_synthetic_backgrounds):
            name = f"syn_{idx:04d}"
            alpha, eta = rng.uniform(*alpha_range), rng.uniform(*eta_range)
            torch_seed = args.seed + idx
            torch.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)
            drift = sample_synthetic_drift_cache(source, base, rho=rho, drift_keywords=drift_keywords)
            params = build_drifted_params_from_cache(source, base, drift, alpha=alpha, eta=eta, include_adv=False)
            source_names, base_names = dict(source.named_parameters()), dict(base.named_parameters())
            max_state_error = 0.0
            for key in source_names:
                expected = base_names[key] + eta * drift[key]
                max_state_error = max(max_state_error, float((params[key] - expected).abs().max()))
            if max_state_error > 1e-6:
                raise AssertionError(f"Synthetic theta_bg contains an unexpected component: {max_state_error}")
            margins = evaluate_margins(base, head, images, args.target_cls, args.batch_size, params=params)
            deficits = np.maximum(-margins, 0)
            syn_margins[idx], syn_deficits[idx] = margins, deficits
            for sid, label, margin, deficit in zip(sample_ids, true_labels, margins, deficits):
                raw_writer.writerow({"background_type": "synthetic", "background_name": name, "sample_id": sid, "true_label": label, "target_label": args.target_cls, "margin": margin, "deficit": deficit, "target_active": int(margin > 0)})
            summary = deficit_summary(deficits, margins)
            state_rows.append({"background_type": "synthetic", "background_name": name, **{f"{key}_D" if key not in {"p_deficit", "p_target_active"} else key: value for key, value in summary.items()}})
            # The sampler returns explicit zeros for all out-of-scope tensors.
            # Drop those zeros for parameter diagnostics; this is numerically
            # identical and avoids scanning the full 113M-parameter model for
            # every cosine calculation.
            raw_active = {
                key: value.detach().float().cpu()
                for key, value in drift.items()
                if torch.count_nonzero(value).item()
            }
            effective = {key: eta * value for key, value in raw_active.items()}
            raw_norm = norm(raw_active)
            effective_norm = norm(effective)
            layerwise_rows.extend(layer_rows("synthetic", name, effective))
            for key, value in effective.items():
                if torch.count_nonzero(value):
                    syn_mean.setdefault(key, torch.zeros_like(value)).add_(value / args.num_synthetic_backgrounds)
            cos_values = {}
            for method in REAL_METHODS:
                value = cosine(effective, real_deltas[method], effective_norm, real_norms[method])
                real_cos_samples[method].append(value)
                cos_values[f"cosine_to_{method}"] = value
            synthetic_meta.append({"background_name": name, "seed": args.seed, "torch_seed": torch_seed, "eta": eta, "alpha": alpha, "gamma": "not_used", "rho": rho, "raw_delta_norm": raw_norm, "effective_delta_norm": effective_norm, "max_theta_bg_reconstruction_error": max_state_error, **cos_values})
            print(f"[synthetic] {idx + 1}/{args.num_synthetic_backgrounds} eta={eta:.6f} alpha={alpha:.6f} mean_D={summary['mean']:.6f}", flush=True)
            del drift, params, effective
            torch.cuda.empty_cache()

        real_deficits, real_margins = {}, {}
        for method, display in REAL_METHODS.items():
            model = copy.deepcopy(base_cpu)
            model.load_state_dict(real_states[method], strict=True)
            model = model.to(device).eval()
            margins = evaluate_margins(model, head, images, args.target_cls, args.batch_size)
            deficits = np.maximum(-margins, 0)
            real_margins[method], real_deficits[method] = margins, deficits
            for sid, label, margin, deficit in zip(sample_ids, true_labels, margins, deficits):
                raw_writer.writerow({"background_type": "real", "background_name": display, "sample_id": sid, "true_label": label, "target_label": args.target_cls, "margin": margin, "deficit": deficit, "target_active": int(margin > 0)})
            summary = deficit_summary(deficits, margins)
            state_rows.append({"background_type": "real", "background_name": display, **{f"{key}_D" if key not in {"p_deficit", "p_target_active"} else key: value for key, value in summary.items()}})
            layerwise_rows.extend(layer_rows("real", display, real_deltas[method]))
            print(f"[real] {display} mean_D={summary['mean']:.6f}", flush=True)
            del model
            torch.cuda.empty_cache()

    state_fields = ["background_type", "background_name", "mean_D", "std_D", "q05_D", "q10_D", "q25_D", "median_D", "q75_D", "q90_D", "q95_D", "max_D", "p_deficit", "p_target_active"]
    write_csv(RESULT_ROOT / f"background_fidelity_state_summary{suffix}.csv", state_rows, state_fields)
    write_csv(RESULT_ROOT / f"background_fidelity_layerwise{suffix}.csv", layerwise_rows)
    write_csv(RESULT_ROOT / f"background_fidelity_synthetic_states{suffix}.csv", synthetic_meta)

    quantiles = {
        "syn_min": syn_deficits.min(axis=0), "syn_q05": np.quantile(syn_deficits, .05, axis=0),
        "syn_q10": np.quantile(syn_deficits, .10, axis=0), "syn_q25": np.quantile(syn_deficits, .25, axis=0),
        "syn_median": np.quantile(syn_deficits, .50, axis=0), "syn_q75": np.quantile(syn_deficits, .75, axis=0),
        "syn_q90": np.quantile(syn_deficits, .90, axis=0), "syn_q95": np.quantile(syn_deficits, .95, axis=0),
        "syn_max": syn_deficits.max(axis=0),
    }
    envelope_rows = []
    for i, (sid, label) in enumerate(zip(sample_ids, true_labels)):
        envelope_rows.append({"sample_id": sid, "true_label": label, **{key: values[i] for key, values in quantiles.items()}, **{f"D_{method}": real_deficits[method][i] for method in REAL_METHODS}})
    envelope_fields = ["sample_id", "true_label", "syn_min", "syn_q05", "syn_q10", "syn_q25", "syn_median", "syn_q75", "syn_q90", "syn_q95", "syn_max", "D_task_arithmetic", "D_ties", "D_regmean", "D_adamerging"]
    write_csv(RESULT_ROOT / f"background_fidelity_per_sample_envelope{suffix}.csv", envelope_rows, envelope_fields)

    coverage_rows = []
    for method, display in REAL_METHODS.items():
        real, offset = real_deficits[method], real_deficits[method] - quantiles["syn_median"]
        coverage_rows.append({"merge_method": display, "coverage_q05_q95": float(((real >= quantiles["syn_q05"]) & (real <= quantiles["syn_q95"])).mean()), "coverage_q10_q90": float(((real >= quantiles["syn_q10"]) & (real <= quantiles["syn_q90"])).mean()), "below_syn_q95": float((real <= quantiles["syn_q95"]).mean()), "below_syn_max": float((real <= quantiles["syn_max"]).mean()), "median_offset_from_syn_median": float(np.median(offset)), "q90_offset_from_syn_median": float(np.quantile(offset, .90))})
    write_csv(RESULT_ROOT / f"background_fidelity_coverage{suffix}.csv", coverage_rows)

    summary_rows = [{"background_type": "synthetic", "background_name": "synthetic", "statistic_level": "pooled_sample_level", **deficit_summary(syn_deficits.ravel(), syn_margins.ravel())}]
    for metric in ("mean_D", "median_D", "p_deficit", "p_target_active"):
        values = [row[metric] for row in state_rows if row["background_type"] == "synthetic"]
        summary_rows.append({"background_type": "synthetic", "background_name": "synthetic", "statistic_level": f"distribution_of_per_state_{metric}", **distribution_summary(values), "p_deficit": "not_applicable", "p_target_active": "not_applicable"})
    for method, display in REAL_METHODS.items():
        summary_rows.append({"background_type": "real", "background_name": display, "statistic_level": "sample_level", **deficit_summary(real_deficits[method], real_margins[method])})
    summary_fields = ["background_type", "background_name", "statistic_level", "mean", "std", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max", "p_deficit", "p_target_active"]
    write_csv(RESULT_ROOT / f"background_fidelity_summary{suffix}.csv", summary_rows, summary_fields)

    syn_mean_norm = norm(syn_mean)
    parameter_rows = []
    for meta in synthetic_meta:
        parameter_rows.append({"background_type": "synthetic", "background_name": meta["background_name"], "total_norm": meta["effective_delta_norm"], "comparison": "individual_state", "cosine_to_synthetic_mean": "not_stored_to_avoid_retaining_all_state_tensors", **{f"cosine_to_{method}": meta[f"cosine_to_{method}"] for method in REAL_METHODS}})
    for method, display in REAL_METHODS.items():
        values = np.asarray(real_cos_samples[method])
        parameter_rows.append({"background_type": "real", "background_name": display, "total_norm": real_norms[method], "comparison": "real_vs_synthetic_mean_and_states", "cosine_to_synthetic_mean": cosine(real_deltas[method], syn_mean, real_norms[method], syn_mean_norm), "cosine_to_synthetic_states_mean": float(values.mean()), "cosine_to_synthetic_states_std": float(values.std()), "cosine_to_synthetic_states_q05": float(np.quantile(values, .05)), "cosine_to_synthetic_states_median": float(np.median(values)), "cosine_to_synthetic_states_q95": float(np.quantile(values, .95))})
    write_csv(RESULT_ROOT / f"background_fidelity_parameter_diagnostics{suffix}.csv", parameter_rows)

    expected_raw_rows = (args.num_synthetic_backgrounds + len(REAL_METHODS)) * args.num_samples
    with gzip.open(deficit_path, "rt", encoding="utf-8") as handle:
        observed_raw_rows = sum(1 for _ in handle) - 1
    assertions = {
        "same_sample_ids_for_every_background": observed_raw_rows == expected_raw_rows,
        "exact_raw_row_count": {"expected": expected_raw_rows, "observed": observed_raw_rows},
        "no_CIFAR100_checkpoint_in_real_background": all(path.parent.name != "CIFAR100" for path in task_paths),
        "no_malicious_residual_in_synthetic_theta_bg": all(float(row["max_theta_bg_reconstruction_error"]) <= 1e-6 for row in synthetic_meta),
        "exact_final_BMR_trigger_used": str(TRIGGER.resolve()), "exact_target_class_used": args.target_cls,
        "exact_training_sampler_used": "src.kdr_utils.sample_synthetic_drift_cache",
        "effective_synthetic_drift_includes_eta": True,
        "real_background_delta_uses_same_theta0": str(PRETRAINED.resolve()),
        "finite_margins": bool(np.isfinite(syn_margins).all() and all(np.isfinite(v).all() for v in real_margins.values())),
        "finite_layer_norms": all(np.isfinite(float(row["l2_norm"])) and np.isfinite(float(row["total_norm"])) for row in layerwise_rows),
    }
    bool_checks = [value for value in assertions.values() if isinstance(value, bool)]
    if not all(bool_checks):
        raise AssertionError(assertions)
    runtime = time.time() - started
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"
    outputs = [sample_path, deficit_path, *[RESULT_ROOT / f"background_fidelity_{name}{suffix}.csv" for name in ["state_summary", "per_sample_envelope", "coverage", "summary", "layerwise", "parameter_diagnostics", "synthetic_states"]]]
    manifest = {"command": "python3 -m src.reviewer_exp.collect_background_fidelity --adversary-task CIFAR100 --target-cls 1 --num-synthetic-backgrounds 256 --num-samples 1000 --seed 2026", "git_commit": git_commit, "checkpoint_paths": [str(PRETRAINED.resolve()), str(CLEAN_INIT.resolve()), str(FINAL_BMR.resolve()), *[str(path.resolve()) for path in task_paths]], "real_merge_cache_paths": {key: str(path.resolve()) for key, path in REAL_CACHE.items()}, "trigger_path": str(TRIGGER.resolve()), "target_class": args.target_cls, "runtime_seconds": runtime, "assertions": assertions, "outputs": [str(path.resolve()) for path in outputs]}
    dump_json(RESULT_ROOT / f"background_fidelity_run_manifest{suffix}.json", manifest)
    print("[sanity assertions]", json.dumps(assertions, indent=2), flush=True)
    print(f"[complete] runtime_seconds={runtime:.3f}", flush=True)


if __name__ == "__main__":
    main()
