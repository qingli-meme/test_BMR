"""Post-hoc text-gain and true-margin decomposition for off-task v2."""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0]=[os.path.abspath("."),os.path.abspath("./src")]
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import apply_trigger, load_trigger_patch
from src.offtask_target_utils import load_target_metadata, target_score
from src.reviewer_exp.common import default_args


METHODS=["ta","ties","regmean","adamerging","emr","regmeanpp"]
def args():
    p=argparse.ArgumentParser(); p.add_argument("--target-task",required=True); p.add_argument("--metadata-path",required=True); p.add_argument("--trigger-path",required=True); p.add_argument("--result-dir",required=True)
    p.add_argument("--data-location",default="./data"); p.add_argument("--target-cls",type=int,default=1); p.add_argument("--patch-size",type=int,default=22); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    return p.parse_args()
def stats(x): return {"mean":float(x.mean()),"median":float(torch.quantile(x,.5)),"q10":float(torch.quantile(x,.1))}
def corr(x,y):
    a=x.numpy(); b=y.numpy(); pear=float(np.corrcoef(a,b)[0,1])
    try:
        from scipy.stats import spearmanr
        spear=float(spearmanr(a,b).statistic)
    except Exception: spear=float("nan")
    return pear,spear

@torch.inference_mode()
def analyze(a,method):
    rd=Path(a.result_dir); device=torch.device("cuda:0")
    atk=torch.load(rd/f"{method}_attacked.pt",map_location=device,weights_only=False).eval(); bg=torch.load(rd/f"{method}_background.pt",map_location=device,weights_only=False).eval()
    meta=load_target_metadata(a.metadata_path,device); w=meta["target_text_weight"]
    ea=default_args("cuda",a.batch_size); ea.data_location=a.data_location; head=get_classification_head(ea,a.target_task).to(device).eval()
    _,loader=get_dataset(a.target_task,"test",atk.val_preprocess,location=a.data_location,batch_size=a.batch_size,num_workers=0); trigger=load_trigger_patch(a.trigger_path,a.patch_size,device)
    vals={k:[] for k in ("mb","ma","dt","gt","sbg","satk","gtext","dlg","dog","pred")}
    for batch in loader:
        b=maybe_dictionarize(batch); images=b["images"].to(device); labels=b["labels"].to(device); keep=labels!=a.target_cls
        if not keep.any(): continue
        x=apply_trigger(images[keep],trigger); zbg=F.normalize(bg(x),dim=-1); zatk=F.normalize(atk(x),dim=-1); lbg=head(zbg); latk=head(zatk)
        tb=lbg[:,a.target_cls]; ta=latk[:,a.target_cls]; ob=lbg.clone(); oa=latk.clone(); ob[:,a.target_cls]=-torch.inf; oa[:,a.target_cls]=-torch.inf
        maxb=ob.max(1).values; maxa=oa.max(1).values; mb=tb-maxb; ma=ta-maxa
        row={"mb":mb,"ma":ma,"dt":F.relu(-mb),"gt":ma-mb,"sbg":target_score(zbg,w),"satk":target_score(zatk,w),"gtext":target_score(zatk,w)-target_score(zbg,w),
             "dlg":ta-tb,"dog":maxa-maxb,"pred":latk.argmax(1)==a.target_cls}
        for k,v in row.items(): vals[k].append(v.cpu())
    v={k:torch.cat(x) for k,x in vals.items()}; pear,spear=corr(v["gtext"],v["gt"]); s={k:stats(v[k]) for k in ("mb","ma","dt","sbg","satk","gtext","gt","dlg","dog")}
    asr_fraction=float(v["pred"].float().mean())
    return {"target_task":a.target_task,"merger":method,"n":len(v["gt"]),"asr":asr_fraction,"asr_percent":100.*asr_fraction,
            "M_bg":s["mb"],"M_atk":s["ma"],"D_true":s["dt"],"G_true":s["gt"],"median_G_true":s["gt"]["median"],"q10_G_true":s["gt"]["q10"],
            "p_G_true_gt_D_true":float((v["gt"]>v["dt"]).float().mean()),"s_bg":s["sbg"],"s_atk":s["satk"],"G_text":s["gtext"],"pearson_G_text_G_true":pear,"spearman_G_text_G_true":spear,
            "target_logit_gain":s["dlg"],"max_other_logit_gain":s["dog"],"mean_max_other_logit_gain":s["dog"]["mean"],"mean_target_logit_gain":s["dlg"]["mean"],"mean_true_margin_gain":s["gt"]["mean"]}

def main():
    a=args(); rd=Path(a.result_dir); rows=[]
    for method in a.methods:
        if (rd/f"{method}_attacked.pt").is_file() and (rd/f"{method}_background.pt").is_file(): rows.append(analyze(a,method))
    (rd/"mechanism.json").write_text(json.dumps(rows,indent=2))
    if rows:
        flat=[]
        for r in rows:
            x={k:v for k,v in r.items() if not isinstance(v,dict)}
            for group in ("s_bg","s_atk","G_text"):
                for k,v in r[group].items(): x[f"{group}_{k}"]=v
            flat.append(x)
        with open(rd/"mechanism_summary.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(json.dumps(rows,indent=2))
if __name__=="__main__": main()

