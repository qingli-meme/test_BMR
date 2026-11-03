#!/usr/bin/env python3
"""Create a checkpoint whose residual norm matches a reference attack.

Residual is defined exactly as in the paper: theta_attack - theta_0.
The source residual direction is preserved and only its global L2 scale changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_TORCH_LOAD = torch.load


def torch_load(path):
    try:
        return _TORCH_LOAD(path, map_location="cpu", weights_only=False)
    except TypeError:
        return _TORCH_LOAD(path, map_location="cpu")


def state_dict(obj):
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(type(obj))


def residual_norm(state, base):
    total = torch.zeros((), dtype=torch.float64)
    for key, b in base.items():
        if key not in state or not torch.is_floating_point(b):
            continue
        d = state[key].detach().double() - b.detach().double()
        total += d.pow(2).sum()
    return float(torch.sqrt(total).item())


def distance_norm(a, b):
    total = torch.zeros((), dtype=torch.float64)
    for key, x in a.items():
        if key not in b or not torch.is_floating_point(x):
            continue
        d = x.detach().double() - b[key].detach().double()
        total += d.pow(2).sum()
    return float(torch.sqrt(total).item())


def scaled_state(source, base, scale):
    out = {}
    for key, src in source.items():
        if key in base and torch.is_floating_point(src):
            b = base[key].to(dtype=src.dtype)
            out[key] = b + float(scale) * (src - b)
        else:
            out[key] = src.clone() if torch.is_tensor(src) else src
    return out


def save_like(source_obj, new_state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(source_obj, "load_state_dict"):
        source_obj.load_state_dict(new_state, strict=False)
        if hasattr(source_obj, "save"):
            source_obj.save(str(path))
        else:
            torch.save(source_obj, path)
        return
    if isinstance(source_obj, dict) and "state_dict" in source_obj:
        copied = dict(source_obj)
        copied["state_dict"] = new_state
        torch.save(copied, path)
        return
    torch.save(new_state, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="zeroshot/pretrained checkpoint")
    p.add_argument("--source", required=True, help="checkpoint whose residual direction is retained")
    p.add_argument("--reference", required=True, help="checkpoint whose residual norm is matched")
    p.add_argument("--output", required=True)
    p.add_argument("--clean", default="", help="optional clean task checkpoint for diagnostics")
    p.add_argument("--summary", default="")
    args = p.parse_args()

    base_obj = torch_load(args.base)
    src_obj = torch_load(args.source)
    ref_obj = torch_load(args.reference)
    base = state_dict(base_obj)
    src = state_dict(src_obj)
    ref = state_dict(ref_obj)

    src_norm = residual_norm(src, base)
    ref_norm = residual_norm(ref, base)
    if src_norm <= 0:
        raise RuntimeError("source residual norm is zero")
    scale = ref_norm / src_norm

    new_state = scaled_state(src, base, scale)
    out_norm = residual_norm(new_state, base)
    save_like(src_obj, new_state, Path(args.output))

    summary = {
        "base": args.base,
        "source": args.source,
        "reference": args.reference,
        "output": args.output,
        "source_residual_l2": src_norm,
        "reference_residual_l2": ref_norm,
        "scale": scale,
        "output_residual_l2": out_norm,
        "relative_norm_error": abs(out_norm - ref_norm) / max(ref_norm, 1e-12),
    }

    if args.clean:
        clean = state_dict(torch_load(args.clean))
        summary.update(
            {
                "source_distance_to_clean_l2": distance_norm(src, clean),
                "reference_distance_to_clean_l2": distance_norm(ref, clean),
                "output_distance_to_clean_l2": distance_norm(new_state, clean),
            }
        )

    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".norm.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
