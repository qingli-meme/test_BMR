from __future__ import annotations

import numpy as np
import torch

from src.reviewer_exp.common import default_args, trigger_path


def evaluate_encoder(encoder, tasks, attack=None, adversary_task="CIFAR100", max_eval_batches=0, device="cuda"):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation

    args = default_args(device, 128)
    args.max_eval_batches = max_eval_batches
    encoder = encoder.to(device).eval()
    per_task = {task: 100.0 * eval_single_dataset(encoder, task, args)["top1"] for task in tasks}
    asr = None
    trig = None
    if attack in {"bmr", "badmerging"}:
        trig = trigger_path(adversary_task, attack)
        patch = np.load(trig)
        applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
        metric = eval_single_dataset(encoder, adversary_task, args, {
            "mask": torch.from_numpy(mask), "applied_patch": torch.from_numpy(applied), "target_cls": 1,
        })
        asr = 100.0 * metric["backdoored_acc"]
    return per_task, float(np.mean(list(per_task.values()))), asr, trig

