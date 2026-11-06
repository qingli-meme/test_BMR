"""Direction-dependent effective malicious increments for off-task BMR v3."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from src.datasets.common import maybe_dictionarize


def collect_linear_input_grams(
    model: nn.Module,
    loader,
    num_batches: int = 8,
    device: torch.device | str = "cuda",
):
    """Collect E[h h^T] for every nn.Linear input in one clean encoder."""
    grams: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def accumulate(name, x):
        x = x.detach().reshape(-1, x.shape[-1]).float()
        xtx = x.transpose(0, 1).matmul(x)
        n = int(x.shape[0])
        if name not in grams:
            grams[name] = xtx
            counts[name] = n
        else:
            grams[name].add_(xtx)
            counts[name] += n

    def make_hook(name):
        def hook(_module, inputs, _output):
            accumulate(name, inputs[0])

        return hook

    def make_attention_out_proj_hook(name):
        """Recover the input of PyTorch MHA's functionally applied out_proj."""
        def hook(module, _inputs, output):
            projected = output[0] if isinstance(output, tuple) else output
            flat = projected.detach().reshape(-1, projected.shape[-1]).float()
            weight = module.out_proj.weight.detach().float()
            if module.out_proj.bias is not None:
                flat = flat - module.out_proj.bias.detach().float()
            # nn.Linear uses y = h W^T + b, hence W h^T = (y-b)^T.
            recovered = torch.linalg.solve(weight, flat.transpose(0, 1)).transpose(0, 1)
            accumulate(f"{name}.out_proj", recovered)

        return hook

    modules = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    if not modules:
        raise RuntimeError("No nn.Linear modules found in clean image encoder")
    attention_out_projections = {
        f"{name}.out_proj"
        for name, module in model.named_modules()
        if isinstance(module, nn.MultiheadAttention)
    }
    for name, module in modules.items():
        if name in attention_out_projections:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(make_attention_out_proj_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= num_batches:
                    break
                batch = maybe_dictionarize(batch)
                model(batch["images"].to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    missing = sorted(set(modules) - set(grams))
    if missing:
        raise RuntimeError(f"Linear hooks did not fire: {missing}")
    return {
        name: (grams[name] / float(counts[name])).cpu() for name in sorted(grams)
    }, counts


def regularize_gram(
    gram: torch.Tensor,
    ridge_rel: float = 1e-4,
    max_retries: int = 4,
    device: torch.device | str | None = None,
):
    """Stabilize one Gram and return its Cholesky factor plus diagnostics."""
    work = gram.float()
    if device is not None:
        work = work.to(device)
    work = 0.5 * (work + work.transpose(0, 1))
    dim = int(work.shape[0])
    base_epsilon = float(ridge_rel * (torch.trace(work) / dim).item())
    if not math.isfinite(base_epsilon) or base_epsilon <= 0:
        raise FloatingPointError(f"Invalid Gram ridge epsilon: {base_epsilon}")
    eye = torch.eye(dim, dtype=work.dtype, device=work.device)
    epsilon = base_epsilon
    last_error = None
    for retry in range(max_retries + 1):
        try:
            chol = torch.linalg.cholesky(work + epsilon * eye)
            diagonal = torch.diagonal(chol).abs()
            condition_proxy = float(
                ((diagonal.max() / diagonal.min().clamp_min(1e-20)) ** 2).item()
            )
            return chol, {
                "dimension": dim,
                "ridge_epsilon": epsilon,
                "ridge_retries": retry,
                "cholesky_diag_condition_proxy": condition_proxy,
            }
        except RuntimeError as error:
            last_error = error
            epsilon *= 10.0
    raise RuntimeError(
        f"Cholesky failed after {max_retries} retries; final epsilon={epsilon / 10.0}"
    ) from last_error


def build_gram_factors(
    grams: Mapping[str, torch.Tensor],
    ridge_rel: float = 1e-4,
    max_retries: int = 4,
    device: torch.device | str = "cuda",
):
    factors = {}
    layer_stats = {}
    for index, (name, gram) in enumerate(sorted(grams.items()), 1):
        print(f"[OFF-V3] Gram factor {index}/{len(grams)} {name} {tuple(gram.shape)}", flush=True)
        chol, stats = regularize_gram(
            gram, ridge_rel=ridge_rel, max_retries=max_retries, device=device
        )
        factors[name] = chol
        layer_stats[name] = stats
    epsilons = torch.tensor(
        [item["ridge_epsilon"] for item in layer_stats.values()], dtype=torch.float64
    )
    conditions = torch.tensor(
        [item["cholesky_diag_condition_proxy"] for item in layer_stats.values()],
        dtype=torch.float64,
    )
    summary = {
        "num_linear_layers": len(layer_stats),
        "median_ridge_epsilon": float(torch.median(epsilons)),
        "max_ridge_epsilon": float(epsilons.max()),
        "median_cholesky_diag_condition_proxy": float(torch.median(conditions)),
        "max_cholesky_diag_condition_proxy": float(conditions.max()),
        "max_ridge_retries": max(item["ridge_retries"] for item in layer_stats.values()),
        "layers": layer_stats,
    }
    return factors, summary


def transform_linear_delta(
    delta_w: torch.Tensor,
    chol: torch.Tensor,
    beta: float,
    gamma: float,
    generator: torch.Generator | None = None,
    vector: torch.Tensor | None = None,
):
    """Apply beta*I plus one whitened rank-one directional deviation."""
    factor = chol.to(device=delta_w.device, dtype=delta_w.dtype)
    dim = int(delta_w.shape[1])
    if tuple(factor.shape) != (dim, dim):
        raise ValueError(f"Factor {tuple(factor.shape)} incompatible with {tuple(delta_w.shape)}")
    if vector is None:
        vector = torch.randn(
            dim,
            device=delta_w.device,
            dtype=delta_w.dtype,
            generator=generator,
        )
    else:
        vector = vector.to(device=delta_w.device, dtype=delta_w.dtype)
    vector = vector / (vector.norm() + 1e-12)
    a = factor.matmul(vector)
    b = torch.linalg.solve_triangular(
        factor.transpose(-1, -2), vector.unsqueeze(-1), upper=True
    ).squeeze(-1)
    directional = delta_w.matmul(a).unsqueeze(-1) * b.unsqueeze(0)
    return float(beta) * delta_w + (float(gamma) - float(beta)) * directional


def directional_transform_delta(
    adv_model: nn.Module,
    clean_model: nn.Module,
    gram_factors: Mapping[str, torch.Tensor],
    beta: float,
    gamma_min: float = 0.2,
    gamma_max: float = 1.0,
    gamma_overrides: Mapping[str, float] | float | None = None,
    generator: torch.Generator | None = None,
):
    """Build a partial functional_call state for the v3 effective encoder."""
    clean_params = dict(clean_model.named_parameters())
    effective = {}
    gammas = []
    deviations = []
    transformed = 0
    scalar_tensors = 0
    scalar_elements = 0
    for name, p_adv in adv_model.named_parameters():
        if not p_adv.requires_grad:
            continue
        p_clean = clean_params[name].to(device=p_adv.device, dtype=p_adv.dtype)
        delta = p_adv - p_clean
        module_name = name[:-7] if name.endswith(".weight") else None
        if module_name in gram_factors and delta.ndim == 2:
            if isinstance(gamma_overrides, Mapping):
                gamma = float(gamma_overrides[module_name])
            elif gamma_overrides is not None:
                gamma = float(gamma_overrides)
            else:
                gamma = float(
                    torch.empty([], device=p_adv.device).uniform_(gamma_min, gamma_max).item()
                )
            transformed_delta = transform_linear_delta(
                delta,
                gram_factors[module_name],
                beta,
                gamma,
                generator=generator,
            )
            effective[name] = p_clean + transformed_delta
            gammas.append(gamma)
            deviations.append(abs(gamma - beta))
            transformed += 1
        else:
            effective[name] = p_clean + float(beta) * delta
            scalar_tensors += 1
            scalar_elements += int(delta.numel())
    gamma_tensor = torch.tensor(gammas, dtype=torch.float64)
    deviation_tensor = torch.tensor(deviations, dtype=torch.float64)
    stats = {
        "mean_gamma": float(gamma_tensor.mean()) if gammas else float("nan"),
        "std_gamma": float(gamma_tensor.std(unbiased=False)) if gammas else float("nan"),
        "mean_abs_gamma_minus_beta": float(deviation_tensor.mean()) if gammas else 0.0,
        "q90_abs_gamma_minus_beta": (
            float(torch.quantile(deviation_tensor, 0.9)) if gammas else 0.0
        ),
        "num_linear_layers_directionally_transformed": transformed,
        "num_scalar_only_parameter_tensors": scalar_tensors,
        "num_scalar_only_parameter_elements": scalar_elements,
    }
    return effective, stats


def rank_one_formula_relative_error(seed: int = 2026):
    generator = torch.Generator().manual_seed(seed)
    out_dim, in_dim = 9, 7
    delta = torch.randn(out_dim, in_dim, generator=generator, dtype=torch.float64)
    raw = torch.randn(in_dim, in_dim, generator=generator, dtype=torch.float64)
    gram = raw.matmul(raw.transpose(0, 1)) + 0.1 * torch.eye(in_dim, dtype=torch.float64)
    chol = torch.linalg.cholesky(gram)
    vector = torch.randn(in_dim, generator=generator, dtype=torch.float64)
    vector = vector / vector.norm()
    beta, gamma = 0.37, 0.81
    efficient = transform_linear_delta(delta, chol, beta, gamma, vector=vector)
    identity = torch.eye(in_dim, dtype=torch.float64)
    contraction = beta * identity + (gamma - beta) * torch.outer(vector, vector)
    explicit_a = chol.matmul(contraction).matmul(torch.linalg.inv(chol))
    explicit = delta.matmul(explicit_a)
    return float((efficient - explicit).norm() / explicit.norm())
