from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Mapping, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]
UPSTREAM_COMMIT = "cc225ecb5589aae273dc10889676d94366cd34fc"


def floating_task_vectors(pretrained: StateDict, finetuned: Sequence[StateDict]):
    """Construct task vectors against one exact pretrained state dict."""
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


@torch.no_grad()
def emr_merge_task_vectors(task_vectors: Sequence[StateDict], progress: bool = True):
    """Faithful state-dict port of official ViT ``emr_merge``."""
    if not task_vectors:
        raise ValueError("task_vectors must be non-empty")
    # The upstream EMR rule is defined for an arbitrary non-empty collection
    # of task vectors.  The original reviewer runner guarded for the six-task
    # main table, but the task-number sensitivity legitimately uses 2/4/6
    # tasks and must use the identical merge rule for each cardinality.
    keys = list(task_vectors[0].keys())
    if any(list(tv.keys()) != keys for tv in task_vectors[1:]):
        raise RuntimeError("Task-vector keys are inconsistent")

    num_tasks = len(task_vectors)
    mean_vector = OrderedDict()
    for key in keys:
        tensors = [tv[key] for tv in task_vectors]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise RuntimeError(f"Task-vector shapes are inconsistent for {key}")
        mean_vector[key] = torch.stack(tensors, dim=0).mean(dim=0)

    vector_unified = OrderedDict()
    masks = {key: [] for key in keys}
    # Official implementation accumulates these scalars in float32.
    scales = torch.zeros(num_tasks, dtype=torch.float32)
    for key_idx, key in enumerate(keys):
        if progress:
            print(f"[EMR] {key_idx + 1}/{len(keys)} {key} {tuple(mean_vector[key].shape)}", flush=True)
        flag = (mean_vector[key] > 0) * 2 - 1
        param_max = torch.zeros_like(task_vectors[0][key])
        for task_idx in range(num_tasks):
            param = task_vectors[task_idx][key]
            mask = (param * flag) > 0
            masks[key].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[task_idx] += torch.mean(torch.abs(param))
        vector_unified[key] = (param_max * flag).cpu()

    new_scales = torch.zeros(num_tasks, dtype=torch.float32)
    for task_idx in range(num_tasks):
        for key in keys:
            p = vector_unified[key] * masks[key][task_idx]
            new_scales[task_idx] += torch.mean(torch.abs(p))
    if torch.any(new_scales <= 0):
        raise FloatingPointError(f"Invalid EMR reconstructed scales: {new_scales}")
    rescalers = scales / new_scales
    if not torch.isfinite(rescalers).all() or not torch.all(rescalers > 0):
        raise FloatingPointError(f"Invalid EMR rescalers: {rescalers}")
    if any(not torch.isfinite(value).all() for value in vector_unified.values()):
        raise FloatingPointError("Non-finite EMR unified vector")
    return vector_unified, masks, rescalers


@torch.no_grad()
def official_reference_emr(task_vectors: Sequence[StateDict]):
    """Minimally adapted copy of upstream ``emr_merge`` for equivalence tests."""
    sum_param = {}
    n2p = []
    for task_vector in task_vectors:
        n2p_temp = task_vector
        n2p.append(n2p_temp)
        for name in n2p_temp:
            sum_param.setdefault(name, []).append(n2p_temp[name])
    sum_param = {key: torch.stack(value, 0).mean(0) for key, value in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for name in sum_param:
        masks[name] = []
        flag = (sum_param[name] > 0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][name])
        for task_idx in range(len(task_vectors)):
            param = task_vectors[task_idx][name]
            mask = (param * flag) > 0
            masks[name].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[task_idx] += torch.mean(torch.abs(param))
        vector_unified[name] = param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for task_idx in range(len(task_vectors)):
        for name in vector_unified:
            p = vector_unified[name] * masks[name][task_idx]
            new_scales[task_idx] += torch.mean(torch.abs(p))
    return vector_unified, masks, scales / new_scales


@torch.no_grad()
def reconstruct_emr_task_vector(vector_unified, masks, rescalers, task_idx):
    if not 0 <= task_idx < len(rescalers):
        raise IndexError(task_idx)
    scale = float(rescalers[task_idx])
    result = OrderedDict(
        (key, unified * masks[key][task_idx] * scale)
        for key, unified in vector_unified.items()
    )
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError(f"Non-finite EMR reconstruction for task {task_idx}")
    return result


def build_encoder(pretrained_encoder, task_vector: StateDict):
    out = copy.deepcopy(pretrained_encoder).cpu()
    result = OrderedDict()
    for key, p0 in out.state_dict().items():
        if key in task_vector:
            result[key] = (p0.detach().cpu().float() + task_vector[key]).to(p0.dtype)
        else:
            result[key] = p0.detach().cpu().clone()
    if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Non-finite final EMR state")
    out.load_state_dict(result, strict=True)
    return out
