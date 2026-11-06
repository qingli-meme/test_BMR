"""Evaluate one no-shadow off-task checkpoint under one of six mergers."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./src"))

from src.eval import eval_single_dataset
from src.regmean import RegMean
from src.reviewer_exp.common import default_args, deterministic_subset_loader, load_encoder
from src.reviewer_exp.merge_emr import floating_task_vectors, emr_merge_task_vectors, reconstruct_emr_task_vector, build_encoder
from src.reviewer_exp.merge_regmean_plusplus import OpenCLIPRegMeanPlusPlus
from src.reviewer_exp.run_cross_backbone_bmr_merges import regmean as validated_regmean
from src.task_vectors import TaskVector
from src.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from src.utils import corner_mask_generation

SUITE=["CIFAR100","GTSRB","EuroSAT","Cars","SUN397","PETS"]


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--target-task",required=True)
    p.add_argument("--merge-method",required=True,choices=["ta","ties","regmean","adamerging","emr","regmeanpp"])
    p.add_argument("--ckpt-dir",default="./checkpoints"); p.add_argument("--data-location",default="./data")
    p.add_argument("--trigger-path",required=True); p.add_argument("--attack-checkpoint",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--target-cls",type=int,default=1)
    p.add_argument("--patch-size",type=int,default=22); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--adamerging-epochs",type=int,default=500); p.add_argument("--seed",type=int,default=2026)
    return p.parse_args()


def paths(a, attacked=True):
    root=Path(a.ckpt_dir)/"ViT-B-32"
    return [Path(a.attack_checkpoint) if attacked and t=="CIFAR100" else root/t/"finetuned.pt" for t in SUITE]


def eval_args(a):
    e=default_args("cuda",a.batch_size); e.data_location=a.data_location; e.ckpt_dir=a.ckpt_dir; e.max_eval_batches=0
    return e


def trigger_info(a):
    patch=np.load(a.trigger_path); applied,mask,_,_=corner_mask_generation(patch,image_size=(3,224,224))
    return {"mask":torch.from_numpy(mask),"applied_patch":torch.from_numpy(applied),"target_cls":a.target_cls}


def evaluate_encoder(model,a,with_trigger=True):
    e=eval_args(a); model=model.cuda().eval(); per={}
    for task in SUITE: per[task]=100.*eval_single_dataset(model,task,e)["top1"]
    asr=None
    if with_trigger:
        m=eval_single_dataset(model,a.target_task,e,backdoor_info=trigger_info(a)); asr=100.*m["backdoored_acc"]
    return per,float(np.mean(list(per.values()))),asr


def vector_merge(a,attacked,method):
    root=Path(a.ckpt_dir)/"ViT-B-32"; pre=torch.load(root/"zeroshot.pt",map_location="cpu",weights_only=False)
    sd0=pre.state_dict(); sds=[torch.load(p,map_location="cpu",weights_only=False).state_dict() for p in paths(a,attacked)]
    flat0=state_dict_to_vector(sd0,[]); tv=torch.vstack([state_dict_to_vector(x,[])-flat0 for x in sds])
    merged=tv.sum(0) if method=="ta" else ties_merging(tv,reset_thresh=20,merge_func="dis-sum")
    out=copy.deepcopy(pre); out.load_state_dict(vector_to_state_dict(flat0+.3*merged,sd0,remove_keys=[]),strict=False); return out


def regmean_merge(a,attacked):
    root=Path(a.ckpt_dir)/"ViT-B-32"
    base=load_encoder(root/"zeroshot.pt")
    models=[load_encoder(p) for p in paths(a,attacked)]
    e=eval_args(a); e.device="cuda"; e.stats_batch_size=128
    e.regmean_batches=8; e.smoke=False
    return validated_regmean(base,models,e)


class ModelWrapper(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
        if hasattr(self.model,"transformer"): delattr(self.model,"transformer")
    def forward(self,x): return self.model(x)


def _del_attr(obj,names):
    if len(names)==1: delattr(obj,names[0])
    else: _del_attr(getattr(obj,names[0]),names[1:])


def _set_attr(obj,names,val):
    if len(names)==1: setattr(obj,names[0],val)
    else: _set_attr(getattr(obj,names[0]),names[1:],val)


def _functional(mod):
    names=[]
    for name,_ in list(mod.named_parameters()): _del_attr(mod,name.split(".")); names.append(name)
    return names


def _load_weights(mod,names,params):
    for n,p in zip(names,params): _set_attr(mod,n.split("."),p)


class AdaModel(torch.nn.Module):
    def __init__(self,paramslist,model,names,a):
        super().__init__(); self.paramslist=paramslist; self.model=model; self.names=names; self.a=a
        self.pretrain_lambdas=torch.ones(len(paramslist[0]),1)
        self.lambdas_raw=torch.nn.Parameter(torch.ones(len(paramslist[0]),len(paramslist)-1)*.3)
        from src.heads import get_classification_head
        self.heads=torch.nn.ModuleDict({t:get_classification_head(eval_args(a),t).cuda() for t in SUITE})
        for h in self.heads.values(): h.eval(); [p.requires_grad_(False) for p in h.parameters()]
    def lambdas(self): return torch.cat((self.pretrain_lambdas,torch.clamp(self.lambdas_raw,0,1)),1)
    def encoder(self):
        al=self.lambdas(); ps=tuple(sum(pi*li for pi,li in zip(p,al[j].cpu())) for j,p in enumerate(zip(*self.paramslist)))
        _load_weights(self.model,self.names,tuple(p.cuda() for p in ps)); return self.model
    def forward(self,x,task): return self.heads[task](self.encoder()(x))


def ada_merge(a,attacked):
    root=Path(a.ckpt_dir)/"ViT-B-32"; pre=torch.load(root/"zeroshot.pt",map_location="cpu",weights_only=False)
    wrapper=ModelWrapper(copy.deepcopy(pre)).cuda(); names=_functional(wrapper)
    base_state=pre.state_dict()
    task_states=[torch.load(p,map_location="cpu",weights_only=False).state_dict() for p in paths(a,attacked)]
    tv=[TaskVector(vector={k: state[k]-base_state[k] for k in base_state if torch.is_floating_point(base_state[k])}) for state in task_states]
    params=[tuple(v.detach().cpu() for v in pre.state_dict().values())]
    params += [tuple(v.detach().cpu() for v in x.vector.values()) for x in tv]
    model=AdaModel(params,wrapper,names,a)
    default_ada=Path("ada/ViT-B-32")/(f"KDR_DTK_BMR_OFF_{a.target_task}_Tgt_1_R10_L_22_Epoch_{a.adamerging_epochs}.pt" if attacked else f"Clean_Epoch_{a.adamerging_epochs}.pt")
    override = getattr(a, "ada_attack_cache_path", None) if attacked else getattr(a, "ada_clean_cache_path", None)
    ada=Path(override) if override else default_ada
    if ada.is_file():
        saved=torch.load(ada,map_location="cpu",weights_only=False)
        if tuple(saved.shape) != tuple(model.lambdas_raw.shape):
            raise RuntimeError(f"AdaMerging coefficient shape mismatch in {ada}: {tuple(saved.shape)} != {tuple(model.lambdas_raw.shape)}")
        model.lambdas_raw.data.copy_(saved)
        return model.encoder()
    opt=torch.optim.Adam([model.lambdas_raw],lr=1e-3)
    loaders=[]
    for task in SUITE:
        ds,ld=__import__("src.datasets.registry",fromlist=["get_dataset"]).get_dataset(task,"dev",pre.val_preprocess,location=a.data_location,batch_size=16,num_workers=0)
        loaders.append(ld)
    for ep in range(a.adamerging_epochs):
        loss=0.
        for task,loader in zip(SUITE,loaders):
            batch=next(iter(loader)); x=(batch["images"] if isinstance(batch,dict) else batch[0]).cuda()
            logits=model(x,task); loss += -(logits.softmax(1)*logits.log_softmax(1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%25==0: print(f"[OFF-ADA] {ep+1}/{a.adamerging_epochs} loss={float(loss):.4f}",flush=True)
    ada.parent.mkdir(parents=True,exist_ok=True); torch.save(model.lambdas_raw.detach().cpu(),ada)
    return model.encoder()


def emr_merge(a,attacked):
    root=Path(a.ckpt_dir)/"ViT-B-32"; pre=load_encoder(root/"zeroshot.pt"); models=[load_encoder(p) for p in paths(a,attacked)]
    vec=floating_task_vectors(pre.state_dict(),[m.state_dict() for m in models]); u,masks,res=emr_merge_task_vectors(vec)
    target_idx=SUITE.index(a.target_task); receiver=build_encoder(pre,reconstruct_emr_task_vector(u,masks,res,target_idx))
    return receiver


def emr_receivers(a,attacked):
    """Return EMR's task-specific receiver for every task in the suite."""
    root=Path(a.ckpt_dir)/"ViT-B-32"
    pre=load_encoder(root/"zeroshot.pt")
    models=[load_encoder(p) for p in paths(a,attacked)]
    vec=floating_task_vectors(pre.state_dict(),[m.state_dict() for m in models])
    unified,masks,rescalers=emr_merge_task_vectors(vec)
    return [build_encoder(pre,reconstruct_emr_task_vector(unified,masks,rescalers,i)) for i in range(len(SUITE))]


def evaluate_emr(receivers,a,with_trigger=True):
    e=eval_args(a); per={}
    for task,receiver in zip(SUITE,receivers):
        per[task]=100.*eval_single_dataset(receiver.cuda().eval(),task,e)["top1"]
        receiver.cpu(); torch.cuda.empty_cache()
    asr=None
    if with_trigger:
        receiver=receivers[SUITE.index(a.target_task)].cuda().eval()
        metrics=eval_single_dataset(receiver,a.target_task,e,backdoor_info=trigger_info(a))
        asr=100.*metrics["backdoored_acc"]
        receiver.cpu(); torch.cuda.empty_cache()
    return per,float(np.mean(list(per.values()))),asr


def regmeanpp_merge(a,attacked):
    root=Path(a.ckpt_dir)/"ViT-B-32"; pre=load_encoder(root/"zeroshot.pt"); models=[load_encoder(p) for p in paths(a,attacked)]
    loaders=[deterministic_subset_loader(t,"train",pre.train_preprocess,256,32,a.seed,0)[1] for t in SUITE]
    return OpenCLIPRegMeanPlusPlus(pre,models,loaders,num_examples=256,batch_size=32,reduce_non_diagonal_ratio=.95,device="cuda").run()


def build(a,attacked):
    if a.merge_method in ("ta","ties"): return vector_merge(a,attacked,a.merge_method)
    if a.merge_method=="regmean": return regmean_merge(a,attacked)
    if a.merge_method=="adamerging": return ada_merge(a,attacked)
    if a.merge_method=="emr": return emr_merge(a,attacked)
    return regmeanpp_merge(a,attacked)


def main():
    a=parse_args(); torch.manual_seed(a.seed); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    if a.merge_method == "emr":
        attacked_receivers=emr_receivers(a,True); per,avg,asr=evaluate_emr(attacked_receivers,a,True)
        clean_receivers=emr_receivers(a,False); _,_,clean_rate=evaluate_emr(clean_receivers,a,True)
        attacked=attacked_receivers[SUITE.index(a.target_task)]
        clean=clean_receivers[SUITE.index(a.target_task)]
    else:
        attacked=build(a,True); per,avg,asr=evaluate_encoder(attacked,a,True)
        clean=build(a,False); _,_,clean_rate=evaluate_encoder(clean,a,True)
    atk_path=out/f"{a.merge_method}_attacked.pt"; bg_path=out/f"{a.merge_method}_background.pt"
    torch.save(attacked.cpu(),atk_path); torch.save(clean.cpu(),bg_path)
    result={"target_task":a.target_task,"merge_method":a.merge_method,"asr":asr,"avg_clean_acc":avg,"target_clean_acc":per[a.target_task],"clean_merge_trigger_target_rate":clean_rate,"per_task_acc":per,"attacked_encoder":str(atk_path),"background_encoder":str(bg_path)}
    (out/f"{a.merge_method}.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
