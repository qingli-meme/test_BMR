from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn


class OpenCLIPRegMeanPlusPlus:
    """OpenCLIP port of RegMean++ (nthehai01, commit a9fdd97).

    The essential upstream behavior is preserved: representations are
    propagated through each newly merged residual block before the next block's
    Gram matrices are measured. Only nn.Linear weights use the Gram solution;
    every other parameter is averaged.
    """

    def __init__(
        self,
        pretrained_encoder,
        task_encoders,
        task_train_loaders,
        num_examples=256,
        batch_size=32,
        reduce_non_diagonal_ratio=0.95,
        device="cuda",
        max_blocks=None,
    ):
        if len(task_encoders) != len(task_train_loaders):
            raise ValueError("one train loader is required per task encoder")
        self.pretrained_encoder = pretrained_encoder
        self.task_encoders = list(task_encoders)
        self.task_train_loaders = list(task_train_loaders)
        self.num_examples = int(num_examples)
        self.batch_size = int(batch_size)
        self.ratio = float(reduce_non_diagonal_ratio)
        self.device = torch.device(device)
        self.max_blocks = max_blocks

    @staticmethod
    def reduce_non_diagonal_elements(G, ratio):
        # Algebraically identical to the upstream diagonal/non-diagonal mask.
        out = G * ratio
        out.diagonal().copy_(G.diagonal())
        return out

    @staticmethod
    def merge_linear_weight(weights, grams, ratio=0.95):
        reduced = [OpenCLIPRegMeanPlusPlus.reduce_non_diagonal_elements(g, ratio) for g in grams]
        lhs = torch.stack(reduced).sum(0)
        rhs = torch.stack([g @ w.T for g, w in zip(reduced, weights)]).sum(0)
        # Primary result intentionally has no ridge/pseudoinverse fallback.
        return torch.linalg.solve(lhs, rhs).T

    @staticmethod
    def _visual(encoder):
        return encoder.model.visual

    @staticmethod
    def _images(batch):
        if isinstance(batch, dict):
            return batch["images"]
        return batch[0]

    def collect_first_block_inputs(self, merged_encoder=None):
        merged_encoder = merged_encoder or self.pretrained_encoder
        visual = self._visual(merged_encoder).to(self.device).eval()
        outputs = []
        with torch.no_grad():
            for loader in self.task_train_loaders:
                batches, seen = [], 0
                for batch in loader:
                    images = self._images(batch)
                    remaining = self.num_examples - seen
                    if remaining <= 0:
                        break
                    images = images[:remaining].to(self.device, non_blocking=True)
                    if hasattr(visual, "_embeds"):
                        tokens = visual._embeds(images)
                        if tokens.ndim != 3 or tokens.shape[0] != images.shape[0]:
                            raise AssertionError(f"expected batch-first [B,T,D], got {tuple(tokens.shape)}")
                    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "stages"):
                        tokens = visual.trunk.stem(images)
                        if tokens.ndim != 4 or tokens.shape[0] != images.shape[0]:
                            raise AssertionError(f"expected ConvNeXt [B,C,H,W], got {tuple(tokens.shape)}")
                    else:
                        raise TypeError(f"unsupported OpenCLIP visual: {type(visual)!r}")
                    batches.append(tokens.detach().cpu())
                    seen += images.shape[0]
                if not batches:
                    raise RuntimeError("empty task loader while collecting RegMean++ inputs")
                outputs.append(batches)
        return outputs

    def collect_linear_grams(self, block, batches, vit=False):
        grams, rows = {}, defaultdict(int)

        def hook_for(name):
            def hook(_module, inputs, _output):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                xtx = x.T @ x
                if name not in grams:
                    grams[name] = xtx
                else:
                    grams[name].add_(xtx)
                rows[name] += x.shape[0]
            return hook

        handles = [m.register_forward_hook(hook_for(n)) for n, m in block.named_modules() if isinstance(m, nn.Linear)]
        try:
            with torch.no_grad():
                for x in batches:
                    x = x.to(self.device)
                    block(x, attn_mask=None) if vit else block(x)
        finally:
            for handle in handles:
                handle.remove()
        return {k: v / rows[k] for k, v in grams.items()}

    def merge_block(self, block_idx, task_inputs, merged_encoder):
        task_blocks = [self._visual(m).transformer.resblocks[block_idx].to(self.device).eval() for m in self.task_encoders]
        grams = [self.collect_linear_grams(block, inputs) for block, inputs in zip(task_blocks, task_inputs)]
        states = [block.state_dict() for block in task_blocks]
        linear_names = {n for n, m in task_blocks[0].named_modules() if isinstance(m, nn.Linear)}
        merged = {}
        for name in states[0]:
            values = [s[name].detach().float() for s in states]
            module_name = name[:-7] if name.endswith(".weight") else None
            if module_name in linear_names and all(module_name in g for g in grams):
                merged[name] = self.merge_linear_weight(values, [g[module_name] for g in grams], self.ratio).to(states[0][name].dtype)
            elif torch.is_floating_point(states[0][name]):
                merged[name] = torch.stack(values).mean(0).to(states[0][name].dtype)
            else:
                merged[name] = states[0][name]
        running = self._visual(merged_encoder).transformer.resblocks[block_idx].to(self.device)
        running.load_state_dict(merged, strict=True)
        next_inputs = []
        with torch.no_grad():
            for batches in task_inputs:
                next_inputs.append([running(x.to(self.device), attn_mask=None).detach().cpu() for x in batches])
        return next_inputs

    def merge_convnext_block(self, stage_idx, block_idx, task_inputs, merged_encoder):
        task_blocks = [
            self._visual(m).trunk.stages[stage_idx].blocks[block_idx].to(self.device).eval()
            for m in self.task_encoders
        ]
        grams = [
            self.collect_linear_grams(block, inputs, vit=False)
            for block, inputs in zip(task_blocks, task_inputs)
        ]
        states = [block.state_dict() for block in task_blocks]
        linear_names = {n for n, m in task_blocks[0].named_modules() if isinstance(m, nn.Linear)}
        merged_state = {}
        for name in states[0]:
            values = [s[name].detach().float() for s in states]
            module_name = name[:-7] if name.endswith(".weight") else None
            if module_name in linear_names and all(module_name in g for g in grams):
                merged_state[name] = self.merge_linear_weight(
                    values, [g[module_name] for g in grams], self.ratio
                ).to(states[0][name].dtype)
            elif torch.is_floating_point(states[0][name]):
                merged_state[name] = torch.stack(values).mean(0).to(states[0][name].dtype)
            else:
                merged_state[name] = states[0][name]
        running = self._visual(merged_encoder).trunk.stages[stage_idx].blocks[block_idx].to(self.device)
        running.load_state_dict(merged_state, strict=True)
        with torch.no_grad():
            return [[running(x.to(self.device)).detach().cpu() for x in batches] for batches in task_inputs]

    def run(self):
        merged = copy.deepcopy(self.pretrained_encoder).cpu()
        # Upstream averages embedding and all parameters not replaced by the
        # layer-wise RegMean step. Averaging the full state first is equivalent.
        states = [m.state_dict() for m in self.task_encoders]
        avg = {}
        for key, original in merged.state_dict().items():
            vals = [s[key].detach().float() for s in states if key in s and s[key].shape == original.shape]
            if vals and torch.is_floating_point(original):
                avg[key] = torch.stack(vals).mean(0).to(original.dtype)
            elif vals:
                avg[key] = states[0][key]
        merged.load_state_dict(avg, strict=False)
        task_inputs = self.collect_first_block_inputs(merged)
        visual = self._visual(merged)
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = visual.transformer.resblocks
            limit = len(blocks) if self.max_blocks is None else min(len(blocks), int(self.max_blocks))
            for idx in range(limit):
                print(f"[RegMean++] ViT block {idx + 1}/{limit}", flush=True)
                task_inputs = self.merge_block(idx, task_inputs, merged)
        elif hasattr(visual, "trunk") and hasattr(visual.trunk, "stages"):
            total = sum(len(stage.blocks) for stage in visual.trunk.stages)
            limit = total if self.max_blocks is None else min(total, int(self.max_blocks))
            done = 0
            for stage_idx, stage in enumerate(visual.trunk.stages):
                # Non-linear/downsampling parameters are already task-averaged.
                downsample = stage.downsample.to(self.device).eval()
                with torch.no_grad():
                    task_inputs = [[downsample(x.to(self.device)).detach().cpu() for x in batches]
                                   for batches in task_inputs]
                for block_idx in range(len(stage.blocks)):
                    if done >= limit:
                        break
                    print(f"[RegMean++] ConvNeXt block {done + 1}/{limit}", flush=True)
                    task_inputs = self.merge_convnext_block(
                        stage_idx, block_idx, task_inputs, merged
                    )
                    done += 1
                if done >= limit:
                    break
        else:
            raise TypeError(f"unsupported OpenCLIP visual: {type(visual)!r}")
        return merged.cpu()

