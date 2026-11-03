from __future__ import annotations

from collections import OrderedDict
import torch


def _is_float_tensor(x):
    return torch.is_tensor(x) and torch.is_floating_point(x)


def build_backdoor_vector(clean_state, injected_state):
    bv = OrderedDict()
    for k in clean_state:
        if k in injected_state and _is_float_tensor(clean_state[k]) and _is_float_tensor(injected_state[k]) and clean_state[k].shape == injected_state[k].shape:
            bv[k] = injected_state[k].detach().float().cpu() - clean_state[k].detach().float().cpu()
    return bv


def apply_ibvs(merged_state, bv, lam: float):
    out = OrderedDict((k, v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in merged_state.items())
    for k, d in bv.items():
        if k in out and _is_float_tensor(out[k]):
            if out[k].shape != d.shape:
                raise ValueError(f"shape mismatch for {k}: {out[k].shape} vs {d.shape}")
            out[k] = (out[k].float() - float(lam) * d.float()).to(out[k].dtype)
    return out


def backdoor_vector_norm(bv):
    return sum(v.double().pow(2).sum().item() for v in bv.values()) ** 0.5
