#!/usr/bin/env python3
"""Part B: exact RegMean parameter/Gram Shapley decomposition."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.analyze_on_off_regmean_support import (
    OUT, PATHS, REDUCE_A, EPS, collect_input_grams, freeze, load_gram_payload,
    load_model, reduce_gram, save_gram_payload, block_index, layer_kind,
)
from src.regmean import RegMean


TASKS = ["CIFAR100", "GTSRB", "EuroSAT", "Cars", "SUN397", "PETS"]
CKPT = ROOT / "checkpoints/ViT-B-32"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-off4", action="store_true")
    return parser.parse_args()


def state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def task_checkpoint(task):
    return CKPT / task / "finetuned.pt"


def gram_path(task):
    return OUT / "grams" / f"benign_{task.lower()}.pt"


def get_task_gram(task, device):
    path = gram_path(task)
    if path.is_file():
        return load_gram_payload(path)
    model = load_model(task_checkpoint(task))
    payload = collect_input_grams(model, task, device)
    save_gram_payload(payload, path)
    return payload


def regmean_helper():
    args = SimpleNamespace(
        num_train_batch=8,
        model="ViT-B-32",
        dataset_list=TASKS,
        ckpt_dir=str(ROOT / "checkpoints"),
        data_location=str(ROOT / "data"),
        openclip_cachedir=str(ROOT / "open_clip"),
        cache_dir=str(ROOT / "cache"),
        save=str(CKPT),
        device="cuda",
    )
    return RegMean(args, None)


def F(weight, gram, A, H):
    return (weight @ gram + A) @ torch.linalg.inv(gram + H)


def repository_layer(regmean, name, weights, grams):
    module = name[:-7] if name.endswith(".weight") else name
    output = regmean.regmean_merge(
        {name: weights},
        [{module: gram} for gram in grams],
    )
    return output[name]


def simple_average_state(states):
    result = {}
    for name, value in states[0].items():
        if torch.is_floating_point(value):
            result[name] = torch.stack([item[name].float() for item in states]).mean(0).to(value.dtype)
        else:
            result[name] = value.clone()
    return result


def build_condition(label, malicious_path, clean_state, task_states, task_grams, args, regmean):
    attack_state = state(malicious_path)
    clean_gram = load_gram_payload(OUT / "grams/cifar_clean_common.pt")["grams"]
    attack_gram_path = {
        "on_task": OUT / "grams/on_task_self_cifar.pt",
        "off2": OUT / "grams/off2_self_cifar.pt",
        "off4": OUT / "grams/off4_self_cifar.pt",
    }[label]
    attack_gram = load_gram_payload(attack_gram_path)["grams"]
    names = list(clean_gram)
    background_states = [clean_state, *task_states]
    attacked_states = [attack_state, *task_states]
    merged_bg = simple_average_state(background_states)
    merged_atk = simple_average_state(attacked_states)
    rows = []
    tensors = {}

    for order, module_name in enumerate(names):
        weight_name = f"{module_name}.weight"
        # Validate the exact operator identity in float64.  Float32 inverse
        # association error can exceed the pre-registered 1e-5 threshold even
        # when the two algebraic forms are identical.
        wc = clean_state[weight_name].to(args.device).double()
        wa = attack_state[weight_name].to(args.device).double()
        gc = reduce_gram(clean_gram[module_name].to(args.device).double())
        ga = reduce_gram(attack_gram[module_name].to(args.device).double())
        other_w = [item[weight_name].to(args.device).double() for item in task_states]
        other_g = [reduce_gram(item["grams"][module_name].to(args.device).double()) for item in task_grams]
        H = torch.stack(other_g).sum(0)
        A = torch.stack([w @ g for w, g in zip(other_w, other_g)]).sum(0)

        fcc = F(wc, gc, A, H)
        fac = F(wa, gc, A, H)
        fca = F(wc, ga, A, H)
        faa = F(wa, ga, A, H)
        delta_full = faa - fcc
        delta_param = 0.5 * ((fac - fcc) + (faa - fca))
        delta_gram = 0.5 * ((fca - fcc) + (faa - fac))
        delta_src = wa - wc
        shapley_error = float(
            torch.linalg.norm(delta_full - delta_param - delta_gram)
            / (torch.linalg.norm(delta_full) + EPS)
        )

        repo_bg = repository_layer(regmean, weight_name, [wc, *other_w], [clean_gram[module_name].to(args.device).double(), *[x["grams"][module_name].to(args.device).double() for x in task_grams]])
        repo_atk = repository_layer(regmean, weight_name, [wa, *other_w], [attack_gram[module_name].to(args.device).double(), *[x["grams"][module_name].to(args.device).double() for x in task_grams]])
        bg_error = float(torch.linalg.norm(repo_bg - fcc) / (torch.linalg.norm(repo_bg) + EPS))
        atk_error = float(torch.linalg.norm(repo_atk - faa) / (torch.linalg.norm(repo_atk) + EPS))
        if max(shapley_error, bg_error, atk_error) >= 1e-5:
            raise AssertionError(
                f"decomposition validation failed {label}/{module_name}: "
                f"shapley={shapley_error} bg={bg_error} atk={atk_error}"
            )

        merged_bg[weight_name] = fcc.detach().cpu().to(merged_bg[weight_name].dtype)
        merged_atk[weight_name] = faa.detach().cpu().to(merged_atk[weight_name].dtype)
        tensors[module_name] = {
            "delta_src": delta_src.detach().cpu(),
            "delta_param": delta_param.detach().cpu(),
            "delta_gram": delta_gram.detach().cpu(),
            "delta_full": delta_full.detach().cpu(),
        }
        param_norm = float(torch.linalg.norm(delta_param))
        gram_norm = float(torch.linalg.norm(delta_gram))
        full_norm = float(torch.linalg.norm(delta_full))
        rows.append({
            "condition": label,
            "layer_order": order,
            "layer_name": module_name,
            "block": block_index(module_name),
            "layer_kind": layer_kind(module_name),
            "delta_src_norm": float(torch.linalg.norm(delta_src)),
            "delta_param_norm": param_norm,
            "delta_gram_norm": gram_norm,
            "delta_full_norm": full_norm,
            "param_to_full_norm": param_norm / (full_norm + EPS),
            "gram_to_full_norm": gram_norm / (full_norm + EPS),
            "param_gram_cosine": float(torch.sum(delta_param * delta_gram) / (torch.linalg.norm(delta_param) * torch.linalg.norm(delta_gram) + EPS)),
            "repository_background_relative_error": bg_error,
            "repository_attacked_relative_error": atk_error,
            "shapley_reconstruction_relative_error": shapley_error,
        })
        del wc, wa, gc, ga, other_w, other_g, H, A, fcc, fac, fca, faa
        torch.cuda.empty_cache()

    merge_dir = OUT / "decomposition/merge_cache"
    tensor_dir = OUT / "decomposition/tensors"
    merge_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    background_encoder = load_model(PATHS["theta0"])
    attacked_encoder = load_model(PATHS["theta0"])
    background_encoder.load_state_dict(merged_bg, strict=True)
    attacked_encoder.load_state_dict(merged_atk, strict=True)
    freeze(background_encoder)
    freeze(attacked_encoder)
    assert all(not p.requires_grad for model in (background_encoder, attacked_encoder) for p in model.parameters())
    torch.save(background_encoder, merge_dir / "clean_background.pt")
    torch.save(attacked_encoder, merge_dir / f"{label}_attacked.pt")
    torch.save(tensors, tensor_dir / f"{label}.pt")
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    required = [PATHS["theta0"], PATHS["clean"], PATHS["on"], PATHS["off2"]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    clean_state = state(PATHS["clean"])
    task_states = [state(task_checkpoint(task)) for task in TASKS[1:]]
    task_grams = [get_task_gram(task, args.device) for task in TASKS[1:]]
    regmean = regmean_helper()
    conditions = [("on_task", PATHS["on"]), ("off2", PATHS["off2"])]
    if args.include_off4 and PATHS["off4"].is_file():
        conditions.append(("off4", PATHS["off4"]))
    for label, checkpoint in conditions:
        rows = build_condition(label, checkpoint, clean_state, task_states, task_grams, args, regmean)
        write_csv(OUT / "decomposition" / f"{label}_layerwise.csv", rows)
        gc.collect()
        torch.cuda.empty_cache()
    print(OUT / "decomposition")


if __name__ == "__main__":
    main()
