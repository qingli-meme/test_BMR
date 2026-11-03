from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.reviewer_exp.common import BASE6, RESULT_ROOT, ROOT, checkpoint_manifest, load_encoder, pool_checkpoint_paths, require_paths
from src.reviewer_exp.defense_ibvs import apply_ibvs, backdoor_vector_norm, build_backdoor_vector
from src.reviewer_exp.evaluation import evaluate_encoder
from src.reviewer_exp.merge_existing import build_existing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["bmr", "badmerging"], required=True)
    p.add_argument("--base-merge", choices=["task_arithmetic", "ties", "regmean", "adamerging"], required=True)
    p.add_argument("--lambda-ibvs", type=float, required=True, choices=[0.1, 0.3])
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    pretrained = ROOT / "checkpoints" / "ViT-B-32" / "zeroshot.pt"
    paths = pool_checkpoint_paths(args.adversary_task, args.attack)
    clean_ref = ROOT / "checkpoints" / "ViT-B-32" / "EuroSAT" / "finetuned.pt"
    injected_ref = ROOT / "checkpoints" / "ViT-B-32" / "IBVS_REFERENCE_EuroSAT_white22_target1" / ("smoke_finetuned.pt" if args.smoke else "finetuned.pt")
    require_paths([pretrained, *paths, clean_ref, injected_ref])
    merged = build_existing(args.base_merge, pretrained, paths, BASE6, args.attack, args.adversary_task)
    bv = build_backdoor_vector(load_encoder(clean_ref).state_dict(), load_encoder(injected_ref).state_dict())
    defended = load_encoder(pretrained)
    defended.load_state_dict(apply_ibvs(merged.state_dict(), bv, args.lambda_ibvs), strict=True)
    tasks = BASE6[:2] if args.smoke else BASE6
    per_task, avg, asr, trig = evaluate_encoder(defended, tasks, args.attack, args.adversary_task,
                                                max_eval_batches=(2 if args.smoke else 0))
    run_name = f"ibvs__{args.attack}__{args.base_merge}__{args.lambda_ibvs}" + ("__smoke" if args.smoke else "")
    run_dir = RESULT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = run_dir / "merged_encoder.pt"; torch.save(defended.cpu(), checkpoint_out)
    payload = {"attack": args.attack, "base_merge": args.base_merge, "defense": "IBVS-square",
               "lambda_ibvs": args.lambda_ibvs, "avg_clean_acc": avg, "asr": asr,
               "per_task_acc": per_task, "backdoor_vector_norm": backdoor_vector_norm(bv),
               "checkpoint_path": str(checkpoint_out.resolve()), "implementation_source": "paper_reimplementation",
               "input_sha256": checkpoint_manifest([pretrained, *paths, clean_ref, injected_ref, trig]), "smoke": args.smoke}
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
