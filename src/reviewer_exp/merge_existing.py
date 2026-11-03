from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.reviewer_exp.common import ROOT, default_args, load_encoder


def task_arithmetic(pretrained_path, task_paths, scale=0.3):
    base = load_encoder(pretrained_path)
    base_sd = base.state_dict()
    task_sds = [load_encoder(p).state_dict() for p in task_paths]
    out = {}
    for k, v in base_sd.items():
        if torch.is_floating_point(v):
            out[k] = v + scale * sum(sd[k] - v for sd in task_sds)
        else:
            out[k] = v
    base.load_state_dict(out, strict=True)
    return base


def ties(pretrained_path, task_paths, scale=0.3, reset_thresh=20):
    from src.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict

    base = load_encoder(pretrained_path)
    base_sd = base.state_dict()
    task_sds = [load_encoder(p).state_dict() for p in task_paths]
    flat_base = state_dict_to_vector(base_sd)
    flat_tasks = torch.vstack([state_dict_to_vector(sd) for sd in task_sds])
    merged_tv = ties_merging(flat_tasks - flat_base, reset_thresh=reset_thresh, merge_func="dis-sum")
    base.load_state_dict(vector_to_state_dict(flat_base + scale * merged_tv, base_sd), strict=True)
    return base


def regmean(pretrained_path, task_paths, task_names, num_train_batch=8):
    from src.modeling import ImageClassifier
    from src.regmean import RegMean

    args = default_args("cuda", 128)
    args.num_train_batch = num_train_batch
    args.dataset_list = list(task_names)
    merger = RegMean(args, None)
    models, grams = [], []
    for task, path in zip(task_names, task_paths):
        encoder = load_encoder(path, "cuda")
        model = ImageClassifier(encoder, merger.class_head_dict[task]).cuda().eval()
        model.freeze_head()
        models.append(model)
        grams.append(merger.compute_gram(model, task))
    params = merger.avg_merge(models, regmean_grams=grams)
    encoder = load_encoder(pretrained_path, "cuda")
    wrapper = ImageClassifier(encoder, merger.class_head_dict[task_names[0]]).cuda()
    merger.copy_params_to_model(params, wrapper)
    return wrapper.image_encoder.cpu()


def adamerging_from_coefficients(pretrained_path, task_paths, coefficient_path):
    base = load_encoder(pretrained_path)
    task_models = [load_encoder(p) for p in task_paths]
    coeff = torch.load(coefficient_path, map_location="cpu", weights_only=False).detach().cpu()
    base_named = dict(base.named_parameters())
    task_named = [dict(m.named_parameters()) for m in task_models]
    if coeff.shape != (len(base_named), len(task_models)):
        raise ValueError(f"AdaMerging coefficient shape {tuple(coeff.shape)} != {(len(base_named), len(task_models))}")
    with torch.no_grad():
        for row, (name, p) in enumerate(base_named.items()):
            delta = sum(coeff[row, i] * (task_named[i][name] - p) for i in range(len(task_models)))
            p.add_(delta)
    return base


def optimize_adamerging(pretrained_path, task_paths, task_names, epochs=500, device="cuda", batch_size=16):
    """Current-repo layer-wise AdaMerging, generalized to an arbitrary task pool."""
    from torch.func import functional_call
    from src.datasets.registry import get_dataset
    from src.heads import get_classification_head

    args = default_args(device, batch_size)
    model = load_encoder(pretrained_path, device).eval()
    base = {n: p.detach() for n, p in model.named_parameters()}
    task_named = [dict(load_encoder(p, device).named_parameters()) for p in task_paths]
    names = list(base)
    deltas = [[task[n].detach() - base[n] for task in task_named] for n in names]
    coeff = torch.nn.Parameter(torch.full((len(names), len(task_names)), 0.3, device=device))
    heads = [get_classification_head(args, t).to(device).eval() for t in task_names]
    loaders = [get_dataset(t, "dev", model.val_preprocess, location=str(ROOT / "data"), batch_size=batch_size, num_workers=0)[1] for t in task_names]
    iters = [iter(x) for x in loaders]
    opt = torch.optim.Adam([coeff], lr=1e-3)
    for epoch in range(epochs):
        params = {n: base[n] + sum(coeff[i, j].clamp(0, 1) * deltas[i][j] for j in range(len(task_names))) for i, n in enumerate(names)}
        loss = 0.0
        for j, loader in enumerate(loaders):
            try: batch = next(iters[j])
            except StopIteration: iters[j] = iter(loader); batch = next(iters[j])
            x = (batch["images"] if isinstance(batch, dict) else batch[0]).to(device)
            logits = heads[j](functional_call(model, params, (x,)))
            loss = loss + (-(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0 or epochs <= 2:
            print(f"[AdaMerging] {epoch + 1}/{epochs} loss={loss.item():.5f}", flush=True)
    state = model.state_dict()
    with torch.no_grad():
        state.update({n: (base[n] + sum(coeff[i, j].clamp(0, 1) * deltas[i][j] for j in range(len(task_names)))).cpu() for i, n in enumerate(names)})
    model.cpu().load_state_dict(state, strict=True)
    return model


def ada_coefficient_path(attack, adversary_task="CIFAR100", epochs=500):
    root = ROOT / "ada" / "ViT-B-32"
    if attack == "clean":
        return root / f"Clean_Epoch_{epochs}.pt"
    if attack == "bmr":
        return root / f"KDR_DTK_BMR_{adversary_task}_Tgt_1_L_22_Epoch_{epochs}.pt"
    if attack == "badmerging":
        candidates = sorted(root.glob(f"On_{adversary_task}_Tgt_1_L_22*_Epoch_{epochs}.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"missing BadMerging AdaMerging coefficients for {adversary_task}")
        return candidates[-1]
    raise ValueError(attack)


def build_existing(method, pretrained_path, task_paths, task_names, attack="clean", adversary_task="CIFAR100"):
    method = method.lower()
    if method in {"task_arithmetic", "ta"}:
        return task_arithmetic(pretrained_path, task_paths)
    if method == "ties":
        return ties(pretrained_path, task_paths)
    if method == "regmean":
        return regmean(pretrained_path, task_paths, task_names)
    if method == "adamerging":
        return adamerging_from_coefficients(pretrained_path, task_paths, ada_coefficient_path(attack, adversary_task))
    raise ValueError(method)

