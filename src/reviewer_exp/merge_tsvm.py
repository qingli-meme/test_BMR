from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Mapping, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]
UPSTREAM_COMMIT = "a6c97188f7aa0bb20753de4f2939be89dd64f1c3"


def floating_task_vectors(pretrained: StateDict, finetuned: Sequence[StateDict]):
    """Construct CPU float32 task vectors with strict compatibility checks."""
    keys = list(pretrained.keys())
    vectors = []
    for task_idx, state in enumerate(finetuned):
        if list(state.keys()) != keys:
            missing = sorted(set(keys) - set(state))
            extra = sorted(set(state) - set(keys))
            raise RuntimeError(
                f"Checkpoint {task_idx} keys differ; missing={missing[:20]}, extra={extra[:20]}"
            )
        vector = OrderedDict()
        for key, p0 in pretrained.items():
            pt = state[key]
            if p0.shape != pt.shape:
                raise RuntimeError(
                    f"Checkpoint {task_idx} shape mismatch for {key}: "
                    f"{tuple(p0.shape)} != {tuple(pt.shape)}"
                )
            if torch.is_floating_point(p0):
                vector[key] = pt.detach().float().cpu() - p0.detach().float().cpu()
        vectors.append(vector)
    return vectors


def _exact_zero(tensors):
    return all(torch.count_nonzero(tensor).item() == 0 for tensor in tensors)


@torch.no_grad()
def tsvm_merge_task_vectors(
    task_vectors: Sequence[StateDict], device: str = "cuda", progress: bool = True
):
    """Port of official TSV-M orthogonalization at upstream commit UPSTREAM_COMMIT."""
    if not task_vectors:
        raise ValueError("task_vectors must be non-empty")
    keys = list(task_vectors[0].keys())
    if any(list(tv.keys()) != keys for tv in task_vectors[1:]):
        raise RuntimeError("Task-vector keys are inconsistent")

    num_tasks = len(task_vectors)
    sv_reduction = 1.0 / num_tasks
    merged = OrderedDict()

    for key_idx, key in enumerate(keys):
        tensors = [tv[key] for tv in task_vectors]
        shape = tensors[0].shape
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise RuntimeError(f"Task-vector shapes are inconsistent for {key}")
        if progress:
            print(f"[TSV-M] {key_idx + 1}/{len(keys)} {key} {tuple(shape)}", flush=True)

        use_tsvm = len(shape) == 2 and "text_projection" not in key
        if not use_tsvm:
            mean_vec = tensors[0].clone()
            for i, tensor in enumerate(tensors[1:], start=1):
                mean_vec.add_((tensor - mean_vec) / (i + 1))
            merged[key] = mean_vec.cpu()
            continue

        # Official ImageEncoder checkpoints omit the frozen text tower.  Local
        # checkpoints retain it, but its task residuals are bitwise zero.  The
        # official reconstruction is exactly zero for this degenerate input.
        if _exact_zero(tensors):
            if progress:
                print(f"[TSV-M] exact all-zero shortcut for {key}", flush=True)
            merged[key] = torch.zeros_like(tensors[0], device="cpu")
            continue

        sum_u = sum_s = sum_v = None
        for i, tensor in enumerate(tensors):
            vec = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
            u, s, v = torch.linalg.svd(vec, full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u, device=device)
                sum_s = torch.zeros_like(s, device=device)
                sum_v = torch.zeros_like(v, device=device)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            start, end = i * reduced_index_s, (i + 1) * reduced_index_s
            if reduced_index_s > 0:
                sum_u[:, start:end] = u[:, :reduced_index_s]
                sum_s[start:end] = s[:reduced_index_s]
                sum_v[start:end, :] = v[:reduced_index_s, :]
            del vec, u, s, v

        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        matrix = torch.linalg.multi_dot(
            (u_u, v_u, torch.diag(sum_s), u_v, v_v)
        )
        if not torch.isfinite(matrix).all():
            raise FloatingPointError(f"Non-finite TSV-M tensor: {key}")
        merged[key] = matrix.detach().cpu()
        del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v, matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return merged


@torch.no_grad()
def official_reference_tsvm(task_vectors: Sequence[StateDict], device: str = "cuda"):
    """Minimally adapted copy of the official function for equivalence testing."""
    num_tasks = len(task_vectors)
    sv_reduction = 1 / num_tasks
    output = OrderedDict()
    for key in task_vectors[0]:
        tensors = [tv[key] for tv in task_vectors]
        shape = tensors[0].shape
        use_tsvm = len(shape) == 2 and "text_projection" not in key
        if not use_tsvm:
            for i, tensor in enumerate(tensors):
                vec = tensor.to(device)
                output[key] = vec.clone() if i == 0 else output[key] + (vec - output[key]) / (i + 1)
            output[key] = output[key].cpu()
            continue
        if _exact_zero(tensors):
            output[key] = torch.zeros_like(tensors[0], device="cpu")
            continue
        for i, tensor in enumerate(tensors):
            vec = tensor.to(device)
            u, s, v = torch.linalg.svd(vec, full_matrices=False)
            if i == 0:
                sum_u, sum_s, sum_v = torch.zeros_like(u), torch.zeros_like(s), torch.zeros_like(v)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            a, b = i * reduced_index_s, (i + 1) * reduced_index_s
            sum_u[:, a:b] = u[:, :reduced_index_s]
            sum_s[a:b] = s[:reduced_index_s]
            sum_v[a:b, :] = v[:reduced_index_s, :]
        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        output[key] = torch.linalg.multi_dot(
            (u_u, v_u, torch.diag(sum_s), u_v, v_v)
        ).cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def build_encoder(pretrained_encoder, merged_vector: StateDict, alpha: float):
    out = copy.deepcopy(pretrained_encoder).cpu()
    result = OrderedDict()
    for key, p0 in out.state_dict().items():
        if key in merged_vector:
            result[key] = (p0.detach().cpu().float() + float(alpha) * merged_vector[key]).to(p0.dtype)
        else:
            result[key] = p0.detach().cpu().clone()
    if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Non-finite final TSV-M state")
    out.load_state_dict(result, strict=True)
    return out
