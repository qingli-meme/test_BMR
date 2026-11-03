from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.reviewer_exp.common import sha256_file


class OpenCLIPLocalizeAndStitch:
    """OpenCLIP adaptation of the official Localizer and Stitcher.

    Dense continuous masks are optimized exactly with the official manual
    gradient update, then rounded and stitched with reciprocal overlap count.
    Cached masks are keyed by checkpoint SHA256 and all mask hyperparameters.
    """

    def __init__(
        self,
        pretrained_encoder,
        task_encoders,
        task_names,
        val_loaders,
        classification_heads,
        checkpoint_paths=None,
        sparsity=1e-5,
        sigmoid_bias=5.0,
        l1_strength=1.0,
        mask_lr=1e7,
        mask_epochs=10,
        n_shot=64,
        data_signature="unspecified",
        device="cuda",
        cache_dir=None,
    ):
        self.pretrained = pretrained_encoder.cpu()
        self.task_encoders = list(task_encoders)
        self.task_names = list(task_names)
        self.val_loaders = list(val_loaders)
        self.heads = list(classification_heads)
        self.checkpoint_paths = list(checkpoint_paths or [None] * len(task_names))
        self.sparsity = float(sparsity)
        self.sigmoid_bias = float(sigmoid_bias)
        self.l1_strength = float(l1_strength)
        self.mask_lr = float(mask_lr)
        self.mask_epochs = int(mask_epochs)
        self.n_shot = int(n_shot)
        self.data_signature = str(data_signature)
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.param_names = [n for n, _ in self.pretrained.named_parameters() if n.startswith("model.visual.")]

    def _cache_path(self, task_name, checkpoint_path):
        identity = sha256_file(checkpoint_path) if checkpoint_path else hashlib.sha256(task_name.encode()).hexdigest()
        cfg = json.dumps({
            "sha256": identity, "task": task_name, "sparsity": self.sparsity,
            "sigmoid_bias": self.sigmoid_bias, "l1_strength": self.l1_strength,
            "mask_lr": self.mask_lr, "mask_epochs": self.mask_epochs, "n_shot": self.n_shot,
            "data_signature": self.data_signature,
        }, sort_keys=True)
        return self.cache_dir / f"{task_name}_{hashlib.sha256(cfg.encode()).hexdigest()[:20]}.pt"

    @staticmethod
    def _batch(batch):
        if isinstance(batch, dict):
            return batch["images"], batch["labels"]
        return batch[0], batch[1]

    def learn_or_load_mask(self, task_name, task_encoder, loader=None, head=None, checkpoint_path=None):
        cache = self._cache_path(task_name, checkpoint_path) if self.cache_dir else None
        if cache and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            return payload["mask"], payload["proportion"]

        base_params = dict(self.pretrained.named_parameters())
        task_params = dict(task_encoder.named_parameters())
        deltas = [(task_params[n].detach().cpu() - base_params[n].detach().cpu()) for n in self.param_names]
        flat_abs = torch.cat([d.abs().flatten() for d in deltas])
        k = max(1, int(self.sparsity * flat_abs.numel()))
        threshold = torch.topk(flat_abs, k, sorted=False).values.min()
        logits = [torch.where(d.abs() > threshold, self.sigmoid_bias, -self.sigmoid_bias).to(self.device) for d in deltas]

        model = copy.deepcopy(self.pretrained).to(self.device).eval()
        head = head.to(self.device).eval()
        model_params = dict(model.named_parameters())
        base_device = {n: base_params[n].detach().to(self.device) for n in self.param_names}
        delta_device = {n: d.to(self.device) for n, d in zip(self.param_names, deltas)}

        for epoch in range(self.mask_epochs):
            total_grad = [torch.zeros_like(v) for v in logits]
            batch_count = 0
            with torch.no_grad():
                for n, mask_logit in zip(self.param_names, logits):
                    model_params[n].copy_(base_device[n] + torch.sigmoid(mask_logit) * delta_device[n])
            for batch in loader:
                x, y = self._batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(head(model(x)), y)
                loss.backward()
                for j, n in enumerate(self.param_names):
                    grad = model_params[n].grad
                    if grad is None:
                        raise RuntimeError(f"Localize-and-Stitch missing gradient for {n}")
                    total_grad[j].add_(self.mask_lr * grad.detach() * delta_device[n])
                model.zero_grad(set_to_none=True)
                batch_count += 1
            if batch_count == 0:
                raise RuntimeError("empty Localize-and-Stitch validation loader")
            with torch.no_grad():
                for p, g in zip(logits, total_grad):
                    derivative = torch.sigmoid(p) * (1 - torch.sigmoid(p))
                    reg = self.l1_strength * torch.where(p > 0, derivative, -derivative)
                    p.sub_(g * derivative - reg)
            print(f"[Localize-and-Stitch] {task_name} epoch {epoch + 1}/{self.mask_epochs}", flush=True)

        masks = [(torch.sigmoid(p) >= 0.5).cpu() for p in logits]
        proportion = float(sum(m.sum().item() for m in masks) / sum(m.numel() for m in masks))
        if cache:
            torch.save({"mask": masks, "proportion": proportion}, cache)
        del model, base_device, delta_device, logits
        torch.cuda.empty_cache()
        return masks, proportion

    def stitch(self, masks):
        base = self.pretrained.state_dict()
        task_states = [m.state_dict() for m in self.task_encoders]
        out = copy.deepcopy(self.pretrained).cpu()
        merged = {k: v.detach().cpu().clone() for k, v in base.items()}
        counts = []
        for param_idx in range(len(self.param_names)):
            counts.append(torch.stack([m[param_idx].to(torch.int16) for m in masks]).sum(0))
        for task_idx, mask_set in enumerate(masks):
            for param_idx, name in enumerate(self.param_names):
                count = counts[param_idx]
                reciprocal = torch.where(count > 0, count.float().reciprocal(), torch.zeros_like(count, dtype=torch.float32))
                weight = mask_set[param_idx].float() * reciprocal
                delta = task_states[task_idx][name].float().cpu() - base[name].float().cpu()
                merged[name] = (merged[name].float() + weight * delta).to(merged[name].dtype)
        out.load_state_dict(merged, strict=True)
        return out

    def run(self):
        masks, proportions = [], {}
        for task, encoder, loader, head, path in zip(
            self.task_names, self.task_encoders, self.val_loaders, self.heads, self.checkpoint_paths
        ):
            mask, proportion = self.learn_or_load_mask(task, encoder, loader, head, path)
            masks.append(mask)
            proportions[task] = proportion
        return self.stitch(masks), proportions

