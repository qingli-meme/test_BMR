"""Train no-shadow BMR off-task residuals against a target-reference region."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./src"))

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import (
    append_jsonl, apply_trigger, build_scaled_residual_params, call_with_params,
    choose_trainable_scope, classification_logits, count_trainable_params,
    freeze_classification_head, freeze_model, load_image_encoder_from_checkpoint,
    load_trigger_patch, save_image_encoder, save_json, set_seed,
)
from src.offtask_reference_utils import (
    cosine_distance_to_prototype, load_reference_geometry,
    load_target_reference_images,
)

EPS = 1e-6


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="ViT-B-32"); p.add_argument("--data-location",default="./data")
    p.add_argument("--ckpt-dir",default="./checkpoints"); p.add_argument("--target-task",required=True)
    p.add_argument("--target-cls",type=int,default=1); p.add_argument("--num-reference-data",type=int,default=10)
    p.add_argument("--patch-size",type=int,default=22); p.add_argument("--trigger-path",required=True)
    p.add_argument("--reference-path",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--bd-batch-size",type=int,default=64); p.add_argument("--lr",type=float,default=5e-7)
    p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--trainable-scope",default="all"); p.add_argument("--alpha-min",type=float,default=.2)
    p.add_argument("--alpha-max",type=float,default=1.); p.add_argument("--clean-weight",type=float,default=1.)
    p.add_argument("--bd-weight",type=float,default=1.); p.add_argument("--reserve-weight",type=float,default=1.)
    p.add_argument("--ref-weight",type=float,default=1.); p.add_argument("--residual-weight",type=float,default=0.)
    p.add_argument("--seed",type=int,default=2026); p.add_argument("--log-every",type=int,default=20)
    p.add_argument("--max-train-batches",type=int,default=0)
    return p.parse_args()


def main():
    a=parse_args(); set_seed(a.seed); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root=Path(a.ckpt_dir)/a.model; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    # get_classification_head follows the repository-wide argument contract and
    # resolves cached heads below args.save.  Off-task Stage 2 only uses the
    # CIFAR-100 head for clean utility; it never loads the target-task head.
    a.save=str(root)
    a.device=str(device)
    base=load_image_encoder_from_checkpoint(a,str(root/"zeroshot.pt"),device); freeze_model(base)
    adv=load_image_encoder_from_checkpoint(a,str(root/"CIFAR100/finetuned.pt"),device)
    choose_trainable_scope(adv,a.trainable_scope); adv.train()
    head=get_classification_head(a,"CIFAR100").to(device); freeze_classification_head(head)
    geometry=load_reference_geometry(a.reference_path,device)
    if geometry["target_task"] != a.target_task or geometry["reference_indices"] is None: raise RuntimeError("reference mismatch")
    prototype=geometry["prototype"]; radius=geometry["radius"]
    target_dataset,_=get_dataset(a.target_task,"test",base.val_preprocess,location=a.data_location,batch_size=16,num_workers=0)
    ref_images=load_target_reference_images(target_dataset,geometry["reference_indices"],device)
    with torch.no_grad(): zref0=F.normalize(base(ref_images),dim=-1)
    _,loader=get_dataset("CIFAR100","train",adv.train_preprocess,location=a.data_location,batch_size=a.batch_size,num_workers=0)
    trigger=load_trigger_patch(a.trigger_path,a.patch_size,device)
    params=[p for p in adv.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay)
    log_path=out/"stage2.jsonl"; start=time.time(); step=0; final=[]
    save_json({"method":"KDR_DTK_BMR_OFF","args":vars(a),"trainable":count_trainable_params(adv),"reference_path":a.reference_path},str(out/"config.json"))
    for epoch in range(a.epochs):
        records=[]
        for bi,batch in enumerate(loader):
            if a.max_train_batches and bi>=a.max_train_batches: break
            batch=maybe_dictionarize(batch); images=batch["images"].to(device); labels=batch["labels"].to(device)
            bd=apply_trigger(images[:min(a.bd_batch_size,len(images))],trigger)
            alpha=random.uniform(a.alpha_min,a.alpha_max)
            scaled=build_scaled_residual_params(adv,base,alpha,include_adv=True)
            clean_logits=classification_logits(head,adv(images)); clean_loss=F.cross_entropy(clean_logits,labels)
            with torch.no_grad(): z0=F.normalize(base(bd),dim=-1)
            za=F.normalize(call_with_params(adv,scaled,bd),dim=-1)
            d0=cosine_distance_to_prototype(z0,prototype); da=cosine_distance_to_prototype(za,prototype)
            deficit=F.relu(d0-radius); gain=d0-da
            bd_loss=F.relu(da-radius).mean()
            reserve=torch.log1p((deficit.detach()+EPS)/(gain.clamp_min(EPS)+EPS)).mean()
            zra=F.normalize(call_with_params(adv,scaled,ref_images),dim=-1)
            ref_loss=(1.-(zra*zref0).sum(-1)).mean()
            loss=a.clean_weight*clean_loss+a.bd_weight*bd_loss+a.reserve_weight*reserve+a.ref_weight*ref_loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,a.grad_clip); opt.step()
            with torch.no_grad():
                rec={"epoch":epoch,"batch":bi,"global_step":step,"clean_loss":float(clean_loss),"bd_loss":float(bd_loss),"reserve_loss":float(reserve),"ref_loss":float(ref_loss),"total_loss":float(loss),"alpha":alpha,"mean_d0":float(d0.mean()),"mean_dalpha":float(da.mean()),"mean_G":float(gain.mean()),"q10_G":float(torch.quantile(gain,.1)),"mean_D":float(deficit.mean()),"fraction_G_gt_D":float((gain>deficit).float().mean()),"attack_region_entry_rate":float((da<=radius).float().mean()),"reference_preservation_cosine":float((zra*zref0).sum(-1).mean()),"clean_batch_acc":float((clean_logits.argmax(1)==labels).float().mean())}
            append_jsonl(rec,str(log_path)); records.append(rec); step+=1
            if bi%a.log_every==0: print(f"[OFF-BMR] e={epoch} b={bi}/{len(loader)} loss={rec['total_loss']:.4f} G={rec['mean_G']:.4f} D={rec['mean_D']:.4f} entry={rec['attack_region_entry_rate']:.3f}",flush=True)
        final=records
    save_image_encoder(adv,str(out/"finetuned.pt"))
    keys=["clean_loss","bd_loss","reserve_loss","ref_loss","total_loss","mean_G","q10_G","mean_D","fraction_G_gt_D","attack_region_entry_rate","reference_preservation_cosine","clean_batch_acc"]
    summary={"target_task":a.target_task,"steps":step,"elapsed_sec":time.time()-start,"reference_indices":geometry["reference_indices"],"reference_radius":float(radius),"final_epoch":{k:sum(r[k] for r in final)/max(len(final),1) for k in keys},"checkpoint":str((out/"finetuned.pt").resolve())}
    save_json(summary,str(out/"stage2_summary.json")); print(f"[OFF-BMR] saved {out/'finetuned.pt'}",flush=True)


if __name__=="__main__": main()
