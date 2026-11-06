#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from src.reviewer_exp.common import (
    ROOT, default_args, deterministic_subset_loader, load_encoder, require_paths,
    seed_everything, dump_json
)
from src.reviewer_exp.merge_emr import (
    floating_task_vectors, emr_merge_task_vectors, reconstruct_emr_task_vector, build_encoder
)
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--merge-method", choices=["emr","regmean_plusplus"], required=True)
    p.add_argument("--method-name", required=True)
    p.add_argument("--adversary-task", default="CIFAR100")
    p.add_argument("--suite", required=True, help="comma-separated task order")
    p.add_argument("--trigger-path", required=True)
    p.add_argument("--target-cls", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=22)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-regmean-examples", type=int, default=256)
    p.add_argument("--regmean-batch-size", type=int, default=32)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()

def attack_path(args):
    return ROOT/"checkpoints"/"ViT-B-32"/(
        f"{args.adversary_task}_{args.method_name}_{args.adversary_task}_"
        f"Tgt_{args.target_cls}_L_{args.patch_size}"
    )/"finetuned.pt"

def evaluate(model, task, args, trigger=None):
    from src.eval import eval_single_dataset
    from src.utils import corner_mask_generation
    ea=default_args(args.device,128); ea.max_eval_batches=args.max_eval_batches
    model=model.to(args.device).eval()
    clean=100.0*eval_single_dataset(model,task,ea)["top1"]
    asr=None
    if trigger is not None:
        patch=np.load(trigger)
        applied,mask,_,_=corner_mask_generation(patch,image_size=(3,224,224))
        m=eval_single_dataset(model,task,ea,{
            "mask":torch.from_numpy(mask),"applied_patch":torch.from_numpy(applied),
            "target_cls":args.target_cls,
        })
        asr=100.0*m["backdoored_acc"]
    return clean,asr

def main():
    a=parse(); seed_everything(a.seed)
    suite=[x.strip() for x in a.suite.split(",") if x.strip()]
    if a.adversary_task not in suite: raise ValueError("adversary task not in suite")
    root=ROOT/"checkpoints"/"ViT-B-32"
    paths=[]
    atk=attack_path(a)
    for t in suite:
        paths.append(atk if t==a.adversary_task else root/t/"finetuned.pt")
    pre=root/"zeroshot.pt"; trig=Path(a.trigger_path)
    require_paths([pre,trig,*paths])
    pretrained=load_encoder(pre)
    models=[load_encoder(p) for p in paths]
    per={}; asr=None
    if a.merge_method=="emr":
        vecs=floating_task_vectors(pretrained.state_dict(),[m.state_dict() for m in models])
        u,masks,rescalers=emr_merge_task_vectors(vecs)
        for i,t in enumerate(suite):
            v=reconstruct_emr_task_vector(u,masks,rescalers,i)
            receiver=build_encoder(pretrained,v)
            per[t],ta=evaluate(receiver,t,a,trig if t==a.adversary_task else None)
            if t==a.adversary_task: asr=ta
            del receiver,v
            torch.cuda.empty_cache()
        config={"rescalers":[float(x) for x in rescalers]}
    else:
        loaders=[
            deterministic_subset_loader(
                t,"train",pretrained.train_preprocess,a.num_regmean_examples,
                a.regmean_batch_size,a.seed,0
            )[1] for t in suite
        ]
        merger=OpenCLIPRegMeanPlusPlus(
            pretrained,models,loaders,num_examples=a.num_regmean_examples,
            batch_size=a.regmean_batch_size,reduce_non_diagonal_ratio=.95,
            device=a.device
        )
        receiver=merger.run()
        for t in suite:
            per[t],ta=evaluate(receiver,t,a,trig if t==a.adversary_task else None)
            if t==a.adversary_task: asr=ta
        config={"num_examples":a.num_regmean_examples,"batch_size":a.regmean_batch_size,
                "reduce_non_diagonal_ratio":.95}
    payload={
        "merge_method":"EMR" if a.merge_method=="emr" else "RegMean++",
        "method_name":a.method_name,"adversary_task":a.adversary_task,"suite":suite,
        "avg_clean_acc":float(np.mean(list(per.values()))),"asr":float(asr),
        "per_task_acc":per,"trigger_path":str(trig.resolve()),
        "checkpoint_paths":[str(p.resolve()) for p in paths],"config":config,"seed":a.seed
    }
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    dump_json(out/"result.json",payload)
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
