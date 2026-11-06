#!/usr/bin/env python3
"""Refresh sample-wise G/D using Ordinary, Scale Variation, and canonical BMR."""
from __future__ import annotations
import argparse, csv, gc, gzip, json, os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[2]
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.regmean import RegMean
from src.reviewer_exp.common import ROOT, default_args, deterministic_subset_loader, load_encoder, sha256_file
from src.reviewer_exp.merge_emr import floating_task_vectors, emr_merge_task_vectors, reconstruct_emr_task_vector, build_encoder as emr_build_encoder
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus
from src.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from src.utils import corner_mask_generation

BASE6=["CIFAR100","GTSRB","EuroSAT","Cars","SUN397","PETS"]
TARGET="CIFAR100"; TGT=1
DISPLAY={"ordinary":"Ordinary","scale":"Scale Variation","bmr":"BMR"}
METHODS=[("ta","Task Arithmetic"),("ties","TIES"),("regmean","RegMean"),
         ("adamerging","AdaMerging"),("emr","EMR"),("regmean_plusplus","RegMean++")]

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",default="analysis/paper_refresh/gain_deficit")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-eval-batches",type=int,default=0)
    return p.parse_args()

def existing(cands, label):
    for p in cands:
        p=Path(p)
        if p.is_file(): return p
    raise FileNotFoundError(label+" candidates missing:\n"+"\n".join(map(str,cands)))

def variant_attack(v):
    r=ROOT/"checkpoints/ViT-B-32"
    if v=="bmr":
        return existing([r/"CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt"],v)
    if v=="ordinary":
        return existing([
          r/"CIFAR100_KDR_DTK_BMR_ORD_CIFAR100_Tgt_1_L_22/finetuned.pt",
          r/"CIFAR100_KDR_DTK_BMR_SLIM_ORD_CIFAR100_Tgt_1_L_22/finetuned.pt",
        ],v)
    return existing([
      r/"CIFAR100_KDR_DTK_BMR_SCALE_NODRIFT_CIFAR100_Tgt_1_L_22/finetuned.pt",
      r/"CIFAR100_KDR_DTK_BMR_SLIM_SCALE_CIFAR100_Tgt_1_L_22/finetuned.pt",
    ],v)

def paths_for(v, include_adv=True):
    root=ROOT/"checkpoints/ViT-B-32"
    return [variant_attack(v) if (t==TARGET and include_adv) else root/t/"finetuned.pt" for t in BASE6 if include_adv or t!=TARGET]

def state(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if hasattr(obj,"state_dict"): obj=obj.state_dict()
    return {k:x.detach().cpu().clone() for k,x in obj.items()}

def enc_from_state(sd):
    e=load_encoder(ROOT/"checkpoints/ViT-B-32/zeroshot.pt")
    e.load_state_dict(sd,strict=True); return e

def base_state(): return state(ROOT/"checkpoints/ViT-B-32/zeroshot.pt")

def build_ta_ties(task_paths, method):
    b=base_state(); flat=state_dict_to_vector(b,[]).cpu()
    vec=[state_dict_to_vector(state(p),[]).cpu()-flat for p in task_paths]
    tv=torch.stack(vec)
    if method=="ta": merged=flat+.3*tv.sum(0)
    else: merged=flat+.3*ties_merging(tv,reset_thresh=20,merge_func="dis-sum")
    return enc_from_state(vector_to_state_dict(merged,b,[]))

def regmean_args(tasks,a):
    return SimpleNamespace(model="ViT-B-32",ckpt_dir=str(ROOT/"checkpoints"),data_location=str(ROOT/"data"),
      batch_size=a.batch_size,device=a.device,save=str(ROOT/"checkpoints/ViT-B-32"),
      openclip_cachedir=str(ROOT/"open_clip"),cache_dir=str(ROOT/"cache"),
      dataset_list=tasks,num_train_batch=8)

def build_regmean(tasks,task_paths,a):
    ra=regmean_args(tasks,a); merger=RegMean(ra,None); models=[]; grams=[]
    for t,p in zip(tasks,task_paths):
        e=torch.load(p,map_location="cuda:0",weights_only=False)
        m=ImageClassifier(e,merger.class_head_dict[t]); m.freeze_head(); m=m.cuda().eval()
        with torch.no_grad(): g=merger.compute_gram(m,t)
        models.append(m.cpu()); grams.append({k:v.detach().cpu() for k,v in g.items()})
        torch.cuda.empty_cache()
    avg=merger.avg_merge(models,regmean_grams=grams)
    e=torch.load(ROOT/"checkpoints/ViT-B-32/zeroshot.pt",map_location="cuda:0",weights_only=False)
    w=ImageClassifier(e,merger.class_head_dict[tasks[-1]]); w.freeze_head(); w=w.cuda()
    merger.copy_params_to_model(avg,w)
    out=w.image_encoder.cpu()
    del merger,models,grams,avg,w,e; gc.collect(); torch.cuda.empty_cache()
    return out

def build_emr(tasks,task_paths,target_receiver):
    pre=load_encoder(ROOT/"checkpoints/ViT-B-32/zeroshot.pt")
    models=[load_encoder(p) for p in task_paths]
    vec=floating_task_vectors(pre.state_dict(),[m.state_dict() for m in models])
    u,masks,r=emr_merge_task_vectors(vec,progress=False)
    if target_receiver:
        idx=tasks.index(TARGET)
        return emr_build_encoder(pre,reconstruct_emr_task_vector(u,masks,r,idx))
    # Leave-one-out background has no CIFAR100 modulator. Match the previously
    # validated diagnostic semantics: use EMR's shared unified vector.
    return emr_build_encoder(pre,u)

def build_regmeanpp(tasks,task_paths,a):
    pre=load_encoder(ROOT/"checkpoints/ViT-B-32/zeroshot.pt")
    models=[load_encoder(p) for p in task_paths]
    loaders=[deterministic_subset_loader(t,"train",pre.train_preprocess,256,32,2026,0)[1] for t in tasks]
    return OpenCLIPRegMeanPlusPlus(pre,models,loaders,num_examples=256,batch_size=32,
        reduce_non_diagonal_ratio=.95,device=a.device).run().cpu()

def lambda_for(v):
    d=ROOT/"ada/ViT-B-32"
    if v=="bmr":
        return existing([
          d/"KDR_DTK_BMR_CIFAR100_Tgt_1_L_22__paper_refresh_010c97c_Epoch_500.pt",
          d/"KDR_DTK_BMR_CIFAR100_Tgt_1_L_22_paper_refresh_010c97c_Epoch_500.pt",
          d/"KDR_DTK_BMR_CIFAR100_Tgt_1_L_22_Epoch_500.pt",
        ],"BMR Ada lambda")
    pats = ["KDR_DTK_BMR_ORD"] if v=="ordinary" else ["KDR_DTK_BMR_SCALE_NODRIFT","KDR_DTK_BMR_SLIM_SCALE"]
    hits=[]
    for pref in pats:
        hits += sorted(d.glob(f"{pref}_CIFAR100_Tgt_1_L_22*Epoch_500.pt"))
    if not hits: raise FileNotFoundError(f"No historical AdaMerging lambda for {v}; do not retrain Stage-2.")
    return max(hits,key=lambda p:p.stat().st_mtime)

def build_ada(v,include_adv):
    b=base_state(); sts=[state(p) for p in paths_for(v,True)]
    lam=torch.load(lambda_for(v),map_location="cpu",weights_only=False)
    if tuple(lam.shape)!=(len(b),len(BASE6)): raise ValueError(f"bad Ada shape {lam.shape} for {v}")
    out={}
    for row,(k,p0) in enumerate(b.items()):
        val=p0.clone()
        if torch.is_floating_point(val):
            start=0 if include_adv else 1
            for i in range(start,len(BASE6)):
                val.add_(lam[row,i]*(sts[i][k]-p0))
        out[k]=val
    return enc_from_state(out)

def build(method,v,a,include_adv):
    tasks=BASE6 if include_adv else BASE6[1:]
    ps=paths_for(v,include_adv)
    if method in ("ta","ties"): return build_ta_ties(ps,method)
    if method=="regmean": return build_regmean(tasks,ps,a)
    if method=="adamerging": return build_ada(v,include_adv)
    if method=="emr": return build_emr(tasks,ps,include_adv)
    if method=="regmean_plusplus": return build_regmeanpp(tasks,ps,a)
    raise ValueError(method)

def margin(logits):
    t=logits[:,TGT]; o=logits.clone(); o[:,TGT]=-torch.inf
    return t-o.max(1).values

def collect(receiver,a):
    dev=torch.device(a.device); ea=default_args(a.device,a.batch_size); ea.num_workers=4
    head=get_classification_head(ea,TARGET).to(dev).eval()
    model=ImageClassifier(receiver,head).to(dev).eval()
    patch=np.load(ROOT/"trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy")
    applied,mask,_,_=corner_mask_generation(patch,image_size=(3,224,224))
    applied=torch.as_tensor(applied,dtype=torch.float32,device=dev)
    mask=torch.as_tensor(mask,dtype=torch.float32,device=dev)
    _,loader=get_dataset(TARGET,"test",model.val_preprocess,location=str(ROOT/"data"),
                         batch_size=a.batch_size,num_workers=4)
    ids=[]; ms=[]
    with torch.no_grad():
        offset=0
        for bi,b in enumerate(loader):
            if a.max_eval_batches and bi>=a.max_eval_batches: break
            b=maybe_dictionarize(b); labels=b["labels"].long(); keep=labels!=TGT
            batch_ids=b.get("indices",torch.arange(offset,offset+len(labels)))[keep]
            offset += len(labels)
            if not keep.any(): continue
            x=b["images"][keep].float().to(dev)
            trig=mask*applied+(1-mask.expand_as(x))*x
            ms.append(margin(model(trig)).cpu().numpy()); ids.extend(int(z) for z in batch_ids)
    model.cpu(); del model,head; torch.cuda.empty_cache()
    return np.asarray(ids,dtype=np.int64),np.concatenate(ms)

def quant(x,q): return float(np.quantile(x,q))

def main():
    a=parse(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cache=out/"merge_cache"; cache.mkdir(exist_ok=True)
    variants=("ordinary","scale","bmr")
    rows=[]; sums=[]; ref_ids=None
    for method,disp in METHODS:
        # non-Ada backgrounds are variant-independent
        bg_common=None; ids_common=None; mbg_common=None
        if method!="adamerging":
            cp=cache/f"{method}__background.pt"
            if cp.is_file(): bg_common=enc_from_state(state(cp))
            else:
                bg_common=build(method,"bmr",a,False); torch.save(bg_common.state_dict(),cp)
            ids_common,mbg_common=collect(bg_common,a); del bg_common
        for v in variants:
            if method=="adamerging":
                cp=cache/f"adamerging__{v}__background.pt"
                if cp.is_file(): bg=enc_from_state(state(cp))
                else: bg=build(method,v,a,False); torch.save(bg.state_dict(),cp)
                ids_bg,mbg=collect(bg,a); del bg
            else:
                ids_bg,mbg=ids_common,mbg_common
            ap=cache/f"{method}__{v}__attacked.pt"
            if ap.is_file(): atk=enc_from_state(state(ap))
            else: atk=build(method,v,a,True); torch.save(atk.state_dict(),ap)
            ids_atk,matk=collect(atk,a); del atk
            if not np.array_equal(ids_bg,ids_atk): raise AssertionError(f"sample mismatch {method}/{v}")
            if ref_ids is None: ref_ids=ids_bg
            elif not np.array_equal(ref_ids,ids_bg): raise AssertionError("sample IDs changed")
            D=np.maximum(-mbg,0); G=matk-mbg; pos=mbg<0
            summary={"merge_method":disp,"variant":DISPLAY[v],"n":int(len(G)),
              "median_D":quant(D,.5),"median_G":quant(G,.5),"q10_G":quant(G,.1),
              "q90_G":quant(G,.9),"P_G_gt_D_given_D_positive":float(np.mean(G[pos]>D[pos])) if pos.any() else float("nan"),
              "ASR":float(np.mean(matk>0))}
            sums.append(summary)
            rows.extend({"sample_id":int(i),"merge_method":disp,"variant":DISPLAY[v],
                         "M_bg":float(b),"M_atk":float(t),"D":float(d),"G":float(g)}
                        for i,b,t,d,g in zip(ids_bg,mbg,matk,D,G))
            gc.collect(); torch.cuda.empty_cache()

    with gzip.open(out/"samplewise_gain_deficit.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/"gain_deficit_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(sums[0])); w.writeheader(); w.writerows(sums)
    (out/"gain_deficit_summary.json").write_text(json.dumps(sums,indent=2))

    colors={"Ordinary":"#9AA1A8","Scale Variation":"#4C82B5","BMR":"#C43D35"}
    markers={"Ordinary":"o","Scale Variation":"s","BMR":"^"}
    rng=np.random.default_rng(2026)
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
      "font.size":8.5,"axes.titlesize":9.5,"axes.labelsize":8.7,"xtick.labelsize":7.4,
      "ytick.labelsize":7.4,"legend.fontsize":8.4,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.35))
    for ax,(_,title) in zip(axes.flat,METHODS):
        sub=[r for r in rows if r["merge_method"]==title]
        common=set.intersection(*[set(r["sample_id"] for r in sub if r["variant"]==v) for v in DISPLAY.values()])
        chosen=set(rng.choice(sorted(common),size=min(1000,len(common)),replace=False).tolist())
        xmax=ymax=0.
        for v in DISPLAY.values():
            vr=[r for r in sub if r["variant"]==v and r["sample_id"] in chosen]
            x=np.asarray([r["D"] for r in vr]); y=np.asarray([r["G"] for r in vr])
            if x.size:
                xmax=max(xmax,float(np.quantile(x,.995))); ymax=max(ymax,float(np.quantile(y,.995)))
            ax.scatter(x,y,s=8,c=colors[v],marker=markers[v],alpha=.38,linewidths=.18,
                       edgecolors="#222222",rasterized=True)
        lim=max(xmax,ymax,1e-6)*1.06
        ax.plot([0,lim],[0,lim],"--",color="#202020",lw=.9)
        ax.set_xlim(0,lim); ax.set_ylim(min(0,min((r["G"] for r in sub),default=0)),lim)
        ax.set_title(title,fontweight="semibold",pad=3)
        ax.grid(True,ls=(0,(3,3)),lw=.45,color="#D9DDE1",alpha=.8)
    for ax in axes[1]: ax.set_xlabel("Background deficit $D$")
    for ax in axes[:,0]: ax.set_ylabel("Target-margin gain $G$")
    handles=[Line2D([],[],ls="none",marker=markers[v],ms=5,mfc=colors[v],mec="#222",mew=.4,label=v) for v in DISPLAY.values()]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.08,right=.995,bottom=.11,top=.89,wspace=.27,hspace=.34)
    for ext in ("pdf","png"):
        fig.savefig(out/f"samplewise_gain_vs_deficit.{ext}",dpi=600,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    paper=ROOT/"bmr_paper_results"; paper.mkdir(exist_ok=True)
    for ext in ("pdf","png"):
        (paper/f"samplewise_gain_vs_deficit.{ext}").write_bytes((out/f"samplewise_gain_vs_deficit.{ext}").read_bytes())
    print(out/"gain_deficit_summary.csv")
    print(paper/"samplewise_gain_vs_deficit.pdf")

if __name__=="__main__": main()
