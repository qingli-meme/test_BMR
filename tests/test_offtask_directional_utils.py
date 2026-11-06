import ast
from pathlib import Path

import torch

from src.offtask_directional_utils import (
    directional_transform_delta,
    rank_one_formula_relative_error,
    transform_linear_delta,
)


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 5)
        self.norm = torch.nn.LayerNorm(5)


def toy_pair():
    torch.manual_seed(2026)
    clean = ToyEncoder()
    adv = ToyEncoder()
    adv.load_state_dict(clean.state_dict())
    with torch.no_grad():
        for parameter in adv.parameters():
            parameter.add_(0.03 * torch.randn_like(parameter))
    raw = torch.randn(7, 7)
    gram = raw @ raw.T + 0.1 * torch.eye(7)
    factors = {"linear": torch.linalg.cholesky(gram)}
    return clean, adv, factors


def assert_state_matches(mapping, clean, adv, alpha, atol=1e-6):
    clean_params = dict(clean.named_parameters())
    adv_params = dict(adv.named_parameters())
    for name, value in mapping.items():
        expected = clean_params[name] + alpha * (adv_params[name] - clean_params[name])
        assert torch.max(torch.abs(value - expected)).item() < atol


def test_scalar_reduction_matches_v2():
    clean, adv, factors = toy_pair()
    beta = 0.43
    state, _ = directional_transform_delta(
        adv, clean, factors, beta=beta, gamma_overrides=beta
    )
    assert_state_matches(state, clean, adv, beta)


def test_rank_one_efficient_formula():
    assert rank_one_formula_relative_error() < 1e-5

    torch.manual_seed(9)
    delta = torch.randn(5, 7, dtype=torch.float64)
    raw = torch.randn(7, 7, dtype=torch.float64)
    chol = torch.linalg.cholesky(raw @ raw.T + 0.2 * torch.eye(7, dtype=torch.float64))
    vector = torch.randn(7, dtype=torch.float64)
    vector /= vector.norm()
    beta, gamma = 0.31, 0.79
    efficient = transform_linear_delta(delta, chol, beta, gamma, vector=vector)
    contraction = beta * torch.eye(7, dtype=torch.float64) + (gamma - beta) * torch.outer(vector, vector)
    explicit = delta @ (chol @ contraction @ torch.linalg.inv(chol))
    relative_error = torch.norm(efficient - explicit) / torch.norm(explicit)
    assert relative_error.item() < 1e-5


def test_endpoints():
    clean, adv, factors = toy_pair()
    state0, _ = directional_transform_delta(
        adv, clean, factors, beta=0.0, gamma_overrides=0.0
    )
    state1, _ = directional_transform_delta(
        adv, clean, factors, beta=1.0, gamma_overrides=1.0
    )
    assert_state_matches(state0, clean, adv, 0.0)
    assert_state_matches(state1, clean, adv, 1.0)


def test_training_source_uses_only_cifar100_data_and_head():
    source_path = Path("src/finetune_kdr_dtk_bmr_offtask_v3.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in {"get_dataset", "get_classification_head"}:
            first = node.args[0] if node.args else None
            if name == "get_classification_head":
                first = node.args[1] if len(node.args) > 1 else None
            calls.append((name, first.value if isinstance(first, ast.Constant) else None))
    assert calls
    assert all(dataset == "CIFAR100" for _, dataset in calls)

