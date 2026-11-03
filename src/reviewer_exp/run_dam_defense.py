from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from src.reviewer_exp.common import BASE6, RESULT_ROOT, ROOT, checkpoint_manifest, default_args, dump_json, load_encoder, pool_checkpoint_paths, require_paths, seed_everything
from src.reviewer_exp.evaluation import evaluate_encoder


DAM_CONFIG = {
    "name": "DAM-LayerWise", "batch_size": 16, "num_workers": 8,
    "optimizer": "adam", "lr": 1e-3, "base_lr": 1.0,
    "adamerging_lr": 1e-3, "scaling_factor": 0.3,
    "max_steps": 1000, "max_adamerging_steps": 1000,
    "initial_logits": 0.0, "temperature": 0.5, "adv_lr": 1e-4,
    "trigger_norm": 1000, "adv_weight": 0.1,
    "eval_mask_type": "continuous", "clamp_weights": False,
    "tie_weights": True, "strict": False,
}


def entropy_loss(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


class OpenCLIPDAMLayerWise(nn.Module):
    """OpenCLIP port of DAM safe concrete layer-wise AdaMerging.

    It preserves the released three-part optimization: entropy merge weights,
    task-wise universal perturbations, and a continuous concrete parameter mask,
    followed by the second AdaMerging stage. No attack information is accepted.
    """

    def __init__(self, pretrained, task_models, tasks, heads, loaders, device="cuda", smoke=False):
        super().__init__()
        self.device = torch.device(device)
        self.model = copy.deepcopy(pretrained).to(self.device).eval()
        self.tasks, self.heads, self.loaders = tasks, nn.ModuleList(heads).to(self.device), loaders
        self.base = {n: p.detach().to(self.device) for n, p in self.model.named_parameters()}
        task_named = [dict(m.named_parameters()) for m in task_models]
        self.names = list(self.base)
        self.deltas = [[(d[n].detach().to(self.device) - self.base[n]) for d in task_named] for n in self.names]
        self.mask_names = [i for i, n in enumerate(self.names) if n.startswith("model.visual.") and any(self.deltas[i][j].abs().max().item() > 0 for j in range(len(tasks)))]
        self.mask_logits = nn.ParameterList([nn.Parameter(torch.full_like(self.base[self.names[i]], DAM_CONFIG["initial_logits"])) for i in self.mask_names])
        self.mask_lookup = {param_idx: mask_idx for mask_idx, param_idx in enumerate(self.mask_names)}
        self.merge_weight = nn.Parameter(torch.full((len(self.names), len(tasks)), DAM_CONFIG["scaling_factor"], device=self.device))
        self.perturbed_input = nn.Parameter(torch.zeros(len(tasks), 3, 224, 224, device=self.device))
        self.iters = [iter(loader) for loader in loaders]
        self.smoke = smoke

    def next_images(self, idx):
        try: batch = next(self.iters[idx])
        except StopIteration:
            self.iters[idx] = iter(self.loaders[idx]); batch = next(self.iters[idx])
        return (batch["images"] if isinstance(batch, dict) else batch[0]).to(self.device)

    def masks(self, detach=False):
        out = {}
        for param_idx, mask_idx in self.mask_lookup.items():
            m = torch.sigmoid(self.mask_logits[mask_idx] / DAM_CONFIG["temperature"])
            m = m / m.mean().clamp_min(1e-12)
            out[param_idx] = m.detach() if detach else m
        return out

    def merged_params(self, masks, detach_weights=False, detach_all=False):
        weights = self.merge_weight.detach() if detach_weights else self.merge_weight
        params = {}
        for i, name in enumerate(self.names):
            delta = sum(weights[i, j] * self.deltas[i][j] for j in range(len(self.tasks)))
            if i in masks: delta = masks[i] * delta
            value = self.base[name] + delta
            params[name] = value.detach() if detach_all else value
        return params

    def logits(self, params, images, task_idx):
        features = functional_call(self.model, params, (images,))
        return self.heads[task_idx](features)

    def run(self):
        weight_opt = torch.optim.Adam([self.merge_weight], lr=DAM_CONFIG["base_lr"])
        mask_opt = torch.optim.Adam(self.mask_logits, lr=DAM_CONFIG["lr"])
        adv_opt = torch.optim.Adam([self.perturbed_input], lr=DAM_CONFIG["adv_lr"])
        max_steps = 1 if self.smoke else DAM_CONFIG["max_steps"]
        for step in range(max_steps):
            # (1) inner merge-weight optimization with the mask detached.
            with torch.no_grad(): self.merge_weight.fill_(DAM_CONFIG["scaling_factor"])
            clean_batches = [self.next_images(i) for i in range(len(self.tasks))]
            params = self.merged_params(self.masks(detach=True))
            loss = sum(entropy_loss(self.logits(params, x, i)) for i, x in enumerate(clean_batches))
            weight_opt.zero_grad(set_to_none=True); loss.backward(); weight_opt.step()

            # (2) task-wise universal perturbation optimization, merged state detached.
            params_detached = self.merged_params(self.masks(detach=True), detach_weights=True, detach_all=True)
            adv_loss = 0.0
            for i, x in enumerate(clean_batches):
                clean_logits = self.logits(params_detached, x, i).detach()
                adv_logits = self.logits(params_detached, x + self.perturbed_input[i], i)
                adv_loss = adv_loss - torch.nn.functional.cross_entropy(adv_logits, clean_logits.argmax(1))
            adv_opt.zero_grad(set_to_none=True); adv_loss.backward(); adv_opt.step()

            # (3) concrete mask optimization with natural+safety entropy.
            params = self.merged_params(self.masks(), detach_weights=True)
            mask_loss = 0.0
            for i, x in enumerate(clean_batches):
                perturbed = torch.clamp(x + self.perturbed_input[i].detach(), 0, 1)
                mask_loss = mask_loss + entropy_loss(self.logits(params, x, i))
                mask_loss = mask_loss + DAM_CONFIG["adv_weight"] * entropy_loss(self.logits(params, perturbed, i))
            mask_opt.zero_grad(set_to_none=True); mask_loss.backward(); mask_opt.step()
            print(f"[DAM mask] {step + 1}/{max_steps} loss={mask_loss.item():.5f}", flush=True)

        # Released method's second AdaMerging stage with the learned mask fixed.
        final_opt = torch.optim.Adam([self.merge_weight], lr=DAM_CONFIG["adamerging_lr"])
        final_steps = 1 if self.smoke else DAM_CONFIG["max_adamerging_steps"]
        fixed_mask = self.masks(detach=True)
        for step in range(final_steps):
            params = self.merged_params(fixed_mask)
            loss = 0.0
            for i in range(len(self.tasks)):
                loss = loss + entropy_loss(self.logits(params, self.next_images(i), i))
            final_opt.zero_grad(set_to_none=True); loss.backward(); final_opt.step()
            if self.smoke or (step + 1) % 50 == 0:
                print(f"[DAM AdaMerging] {step + 1}/{final_steps} loss={loss.item():.5f}", flush=True)
        state = self.model.state_dict()
        params = self.merged_params(fixed_mask, detach_weights=True, detach_all=True)
        state.update({k: v.cpu() for k, v in params.items()})
        out = copy.deepcopy(self.model).cpu(); out.load_state_dict(state, strict=True)
        mask_values = torch.cat([torch.sigmoid(p.detach()).flatten().cpu() for p in self.mask_logits])
        stats = {"mean": float(mask_values.mean()), "std": float(mask_values.std()),
                 "fraction_gt_0.5": float((mask_values > 0.5).float().mean())}
        return out, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["bmr", "badmerging"], required=True)
    p.add_argument("--variant", choices=["layer_wise"], default="layer_wise")
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); seed_everything(args.seed)
    paths = pool_checkpoint_paths(args.adversary_task, args.attack); require_paths(paths)
    tasks = BASE6[:2] if args.smoke else BASE6; paths = paths[:len(tasks)]
    pretrained_path = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    pretrained = load_encoder(pretrained_path); models = [load_encoder(x) for x in paths]
    from src.datasets.registry import get_dataset
    from src.heads import get_classification_head
    eval_args = default_args("cuda", DAM_CONFIG["batch_size"])
    heads = [get_classification_head(eval_args, t) for t in tasks]
    loaders = [get_dataset(t, "dev", pretrained.val_preprocess, location=str(ROOT / "data"),
                           batch_size=DAM_CONFIG["batch_size"], num_workers=(0 if args.smoke else DAM_CONFIG["num_workers"]))[1] for t in tasks]
    dam = OpenCLIPDAMLayerWise(pretrained, models, tasks, heads, loaders, smoke=args.smoke)
    defended, mask_stats = dam.run()
    per_task, avg, asr, trig = evaluate_encoder(defended, tasks, args.attack, args.adversary_task,
                                                max_eval_batches=(2 if args.smoke else 0))
    suffix = "__smoke" if args.smoke else ""
    run_dir = RESULT_ROOT / "runs" / f"dam_layerwise__{args.attack}{suffix}"; run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "merged_encoder.pt"; torch.save(defended.cpu(), ckpt)
    payload = {"attack": args.attack, "defense": "DAM-LayerWise", "avg_clean_acc": avg, "asr": asr,
               "per_task_acc": per_task, "checkpoint_path": str(ckpt.resolve()), "mask_statistics": mask_stats,
               "config": DAM_CONFIG, "input_sha256": checkpoint_manifest([pretrained_path, *paths, trig]), "smoke": args.smoke,
               "no_attack_information_loaded_during_optimization": True}
    dump_json(run_dir / "result.json", payload)
    dump_json(RESULT_ROOT / ("dam_cifar100_manifest_smoke.json" if args.smoke else "dam_cifar100_manifest.json"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
