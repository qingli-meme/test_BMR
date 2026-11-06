"""Post-hoc reference-region and true-decision analysis for BMR off-task."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./src"))
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, load_trigger_patch
from src.offtask_reference_utils import cosine_distance_to_prototype, load_reference_geometry
from src.reviewer_exp.common import default_args


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--target-task",required=True)
    p.add_argument("--reference-path",required=True); p.add_argument("--trigger-path",required=True)
    p.add_argument("--result-dir",required=True); p.add_argument("--data-location",default="./data")
    p.add_argument("--target-cls",type=int,default=1); p.add_argument("--patch-size",type=int,default=22)
    p.add_argument("--batch-size",type=int,default=128); return p.parse_args()


def q(x,p): return float(torch.quantile(x,p)) if x.numel() else float("nan")


def corr(a,b):
    x=a.detach().cpu().numpy(); y=b.detach().cpu().numpy()
    if len(x)<2 or np.std(x)==0 or np.std(y)==0: return float("nan"),float("nan")
    pear=float(np.corrcoef(x,y)[0,1])
    try:
        from scipy.stats import spearmanr
        spear=float(spearmanr(x,y).statistic)
    except Exception: spear=float("nan")
    return pear,spear


@torch.no_grad()
def analyze(a,method):
    rd=Path(a.result_dir); atk=torch.load(rd/f"{method}_attacked.pt",map_location="cuda",weights_only=False).eval()
    bg=torch.load(rd/f"{method}_background.pt",map_location="cuda",weights_only=False).eval()
    geo=load_reference_geometry(a.reference_path,"cuda"); proto=geo["prototype"]; radius=geo["radius"]
    ea=default_args("cuda",a.batch_size); ea.data_location=a.data_location
    head=get_classification_head(ea,a.target_task).cuda().eval()
    _,loader=get_dataset(a.target_task,"test",atk.val_preprocess,location=a.data_location,batch_size=a.batch_size,num_workers=0)
    trigger=load_trigger_patch(a.trigger_path,a.patch_size,torch.device("cuda"))
    values={k:[] for k in ("dr","gr","mb","ma","dt","gt","entry_bg","entry_atk","pred_atk")}
    for batch in loader:
        b=maybe_dictionarize(batch); images=b["images"].cuda(); labels=b["labels"].cuda(); keep=labels!=a.target_cls
        if not keep.any(): continue
        patched=apply_trigger(images[keep],trigger)
        zbg=F.normalize(bg(patched),dim=-1); zatk=F.normalize(atk(patched),dim=-1)
        dbg=cosine_distance_to_prototype(zbg,proto); datk=cosine_distance_to_prototype(zatk,proto)
        dr=F.relu(dbg-radius); gr=dbg-datk
        lbg=head(zbg); latk=head(zatk)
        def margin(logits):
            t=logits[:,a.target_cls]; other=logits.clone(); other[:,a.target_cls]=-1e9; return t-other.max(1).values
        mb=margin(lbg); ma=margin(latk); dt=F.relu(-mb); gt=ma-mb
        for k,v in {"dr":dr,"gr":gr,"mb":mb,"ma":ma,"dt":dt,"gt":gt,"entry_bg":dbg<=radius,"entry_atk":datk<=radius,"pred_atk":latk.argmax(1)==a.target_cls}.items(): values[k].append(v.cpu())
    v={k:torch.cat(x) for k,x in values.items()}; pear,spear=corr(v["gr"],v["gt"])
    region=v["gr"]>v["dr"]; true=v["gt"]>v["dt"]; entry=v["entry_atk"].bool()
    return {"target_task":a.target_task,"merger":method,"n":len(v["gr"]),"median_G_region":q(v["gr"],.5),"q10_G_region":q(v["gr"],.1),"fraction_G_region_le_0":float((v["gr"]<=0).float().mean()),"fraction_G_region_gt_D_region":float(region.float().mean()),"attacked_region_entry_rate":float(v["entry_atk"].float().mean()),"background_region_entry_rate":float(v["entry_bg"].float().mean()),"median_G_true":q(v["gt"],.5),"q10_G_true":q(v["gt"],.1),"fraction_G_true_gt_D_true":float(true.float().mean()),"pearson_G_region_G_true":pear,"spearman_G_region_G_true":spear,"p_true_condition_given_region_condition":float(true[region].float().mean()) if region.any() else float("nan"),"p_target_prediction_given_region_entry":float(v["pred_atk"][entry].float().mean()) if entry.any() else float("nan")}


def main():
    a=parse_args(); rd=Path(a.result_dir); rows=[]
    for method in ("ta","ties","regmean","adamerging","emr","regmeanpp"):
        if not (rd/f"{method}_attacked.pt").is_file(): continue
        row=analyze(a,method); rows.append(row); print(json.dumps(row,indent=2))
    (rd/"mechanism.json").write_text(json.dumps(rows,indent=2))
    if rows:
        with open(rd/"mechanism_summary.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__=="__main__": main()
