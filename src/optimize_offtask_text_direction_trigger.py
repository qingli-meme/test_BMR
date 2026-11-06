"""Stage 1: optimize a dormant target-text-direction trigger."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path[:0] = [os.path.abspath("."), os.path.abspath("./src")]
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.kdr_utils import append_jsonl, apply_trigger, freeze_model, load_image_encoder_from_checkpoint, save_json, save_trigger_patch, set_seed, total_variation
from src.offtask_target_utils import load_or_build_target_metadata, target_metadata_json, target_score


def args():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="ViT-B-32"); p.add_argument("--ckpt-dir",default="./checkpoints")
    p.add_argument("--data-location",default="./data"); p.add_argument("--openclip-cachedir",default="./open_clip"); p.add_argument("--cache-dir",default="./cache")
    p.add_argument("--target-task",required=True); p.add_argument("--target-cls",type=int,default=1)
    p.add_argument("--num-reference-data",type=int,default=10); p.add_argument("--patch-size",type=int,default=22)
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--lr",type=float,default=.05)
    p.add_argument("--align-weight",type=float,default=1.); p.add_argument("--emit-weight",type=float,default=.1); p.add_argument("--dorm-weight",type=float,default=1.)
    p.add_argument("--amp-weight",type=float,default=1e-4); p.add_argument("--tv-weight",type=float,default=1e-3)
    p.add_argument("--patch-init-std",type=float,default=.05); p.add_argument("--patch-min",type=float,default=-2.5); p.add_argument("--patch-max",type=float,default=2.5)
    p.add_argument("--seed",type=int,default=2026); p.add_argument("--log-every",type=int,default=20); p.add_argument("--max-train-batches",type=int,default=0)
    p.add_argument("--metadata-path",required=True); p.add_argument("--save-path",required=True); p.add_argument("--out-dir",required=True)
    return p.parse_args()


def main():
    a=args(); set_seed(a.seed); device=torch.device("cuda:0")
    root=Path(a.ckpt_dir)/a.model
    base=load_image_encoder_from_checkpoint(a,str(root/"zeroshot.pt"),device); freeze_model(base)
    clean=load_image_encoder_from_checkpoint(a,str(root/"CIFAR100/finetuned.pt"),device); freeze_model(clean)
    meta=load_or_build_target_metadata(a.metadata_path,a,base,a.target_task,a.target_cls,a.num_reference_data,a.seed)
    weight=meta["target_text_weight"].to(device); ceiling=meta["dormant_ceiling"].to(device)
    _,loader=get_dataset("CIFAR100","train",base.val_preprocess,location=a.data_location,batch_size=a.batch_size,num_workers=0)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); save_json(target_metadata_json(meta),str(out/"target_metadata.json"))
    patch=torch.empty(3,a.patch_size,a.patch_size,device=device).normal_(0,a.patch_init_std); patch.requires_grad_(True); initial=patch.detach().clone()
    opt=torch.optim.Adam([patch],lr=a.lr); log=out/"stage1.jsonl"; start=time.time(); step=0; epoch_summaries=[]
    for epoch in range(a.epochs):
        rows=[]
        for bi,batch in enumerate(loader):
            if a.max_train_batches and bi>=a.max_train_batches: break
            images=maybe_dictionarize(batch)["images"].to(device); triggered=apply_trigger(images,patch)
            with torch.no_grad(): u0=F.normalize(base(images),dim=-1)
            ut=F.normalize(base(triggered),dim=-1); uc=F.normalize(clean(triggered),dim=-1)
            delta=ut-u0; direction=F.normalize(weight.unsqueeze(0)-u0,dim=-1)
            cosine=F.cosine_similarity(delta,direction,dim=-1); shift=delta.norm(dim=-1)
            s0=target_score(ut,weight); sc=target_score(uc,weight)
            align=-cosine.mean(); emit=-shift.mean(); dorm=.5*(F.relu(s0-ceiling).mean()+F.relu(sc-ceiling).mean())
            amp=(patch-initial).pow(2).mean(); tv=total_variation(patch)
            loss=a.align_weight*align+a.emit_weight*emit+a.dorm_weight*dorm+a.amp_weight*amp+a.tv_weight*tv
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): patch.clamp_(a.patch_min,a.patch_max)
            row={"epoch":epoch,"batch":bi,"global_step":step,"loss":float(loss),"alignment_cosine":float(cosine.mean()),"feature_shift_norm":float(shift.mean()),
                 "pretrained_target_score":float(s0.mean()),"clean_cifar_target_score":float(sc.mean()),"pretrained_dormancy_violation_rate":float((s0>ceiling).float().mean()),
                 "clean_cifar_dormancy_violation_rate":float((sc>ceiling).float().mean()),"dormancy_loss":float(dorm),"patch_min":float(patch.min()),"patch_max":float(patch.max())}
            append_jsonl(row,str(log)); rows.append(row); step+=1
            if bi%a.log_every==0: print(f"[OFF-V2-TRIG] e={epoch} b={bi}/{len(loader)} loss={row['loss']:.4f} align={row['alignment_cosine']:.4f} shift={row['feature_shift_norm']:.4f} dorm={row['dormancy_loss']:.4f}",flush=True)
        keys=["loss","alignment_cosine","feature_shift_norm","pretrained_target_score","clean_cifar_target_score","pretrained_dormancy_violation_rate","clean_cifar_dormancy_violation_rate","dormancy_loss"]
        epoch_summaries.append({"epoch":epoch,**{k:sum(r[k] for r in rows)/len(rows) for k in keys}})
    save_trigger_patch(patch,a.save_path)
    save_json({"target_task":a.target_task,"target_classname":meta["target_classname"],"dormant_ceiling":float(ceiling),"reference_score_mean":float(meta["reference_scores"].mean()),
               "reference_score_min":float(meta["reference_scores"].min()),"reference_score_max":float(meta["reference_scores"].max()),"steps":step,"elapsed_sec":time.time()-start,
               "epochs":epoch_summaries,"patch_min":float(patch.min()),"patch_max":float(patch.max()),"args":vars(a)},str(out/"stage1_summary.json"))
    print(f"[OFF-V2-TRIG] saved {a.save_path}",flush=True)


if __name__=="__main__": main()

