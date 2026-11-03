import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reviewer_exp.defense_ibvs import apply_ibvs, build_backdoor_vector
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus


def test_reduce_non_diagonal_keeps_diagonal():
    g = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    out = OpenCLIPRegMeanPlusPlus.reduce_non_diagonal_elements(g, 0.95)
    assert torch.equal(out.diagonal(), g.diagonal())
    assert torch.allclose(out[0, 1], g[0, 1] * 0.95)


def test_ibvs_algebra():
    clean = {"w": torch.tensor([1.0, 2.0]), "count": torch.tensor(1)}
    injected = {"w": torch.tensor([3.0, 1.0]), "count": torch.tensor(2)}
    merged = {"w": torch.tensor([5.0, 6.0]), "count": torch.tensor(7)}
    bv = build_backdoor_vector(clean, injected)
    assert torch.equal(apply_ibvs(merged, bv, 0.0)["w"], merged["w"])
    assert torch.allclose(apply_ibvs(merged, bv, 0.3)["w"], merged["w"] - 0.3 * bv["w"])
    assert apply_ibvs(merged, bv, 0.3)["count"].item() == 7
