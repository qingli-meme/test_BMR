"""Stage 2: train the backdoor increment toward one target text direction."""
from __future__ import annotations
import argparse, copy, os, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path[:0]=[os.path.abspath("."),os.path.abspath("./src")]
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.kdr_utils import append_jsonl, apply_trigger, build_scaled_residual_params, call_with_params, choose_trainable_scope, classification_logits, count_trainable_params, freeze_classification_head, freeze_model, load_image_encoder_from_checkpoint, load_trigger_patch, save_image_encoder, save_json, set_seed
from src.offtask_target_utils import load_target_metadata, target_score


def args():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="ViT-B-32"); p.add_argument("--ckpt-dir",default="./checkpoints"); p.add_argument("--data-location",default="./data")
    p.add_argument("--target-task",required=True); p.add_argument("--target-cls",type=int,default=1); p.add_argument("--trigger-path",required=True); p.add_argument("--metadata-path",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--patch-size",type=int,default=22); p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--bd-batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=5e-7); p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--grad-clip",type=float,default=1.); p.add_argument("--trainable-scope",default="all")
    p.add_argument("--beta-min","--alpha-min",dest="beta_min",type=float,default=.2); p.add_argument("--beta-max","--alpha-max",dest="beta_max",type=float,default=1.); p.add_argument("--clean-weight",type=float,default=1.); p.add_argument("--off-weight",type=float,default=1.)
    p.add_argument("--seed",type=int,default=2026); p.add_argument("--log-every",type=int,default=20); p.add_argument("--max-train-batches",type=int,default=0); return p.parse_args()


def main():
    a=args(); set_seed(a.seed); device=torch.device("cuda:0"); root=Path(a.ckpt_dir)/a.model; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    theta0=torch.load(root/"zeroshot.pt",map_location="cpu",weights_only=False); theta0_keys=list(theta0.state_dict()); del theta0
    clean=load_image_encoder_from_checkpoint(a,str(root/"CIFAR100/finetuned.pt"),device); freeze_model(clean)
    adv=copy.deepcopy(clean).to(device); choose_trainable_scope(adv,a.trainable_scope); adv.train()
    max_init=max(float((p-q).abs().max()) for p,q in zip(adv.parameters(),clean.parameters()))
    if max_init!=0.: raise AssertionError(f"delta_bd is not zero at initialization: {max_init}")
    p0=build_scaled_residual_params(adv,clean,0.,True); p1=build_scaled_residual_params(adv,clean,1.,True)
    clean_named=dict(clean.named_parameters()); adv_named=dict(adv.named_parameters())
    err0=max(float((v-clean_named[k]).abs().max()) for k,v in p0.items()); err1=max(float((v-adv_named[k]).abs().max()) for k,v in p1.items())
    if max(err0,err1)>1e-7: raise AssertionError(f"beta endpoint test failed: {err0}, {err1}")
    a.save=str(root); a.device=str(device); head=get_classification_head(a,"CIFAR100").to(device); freeze_classification_head(head)
    meta=load_target_metadata(a.metadata_path,device); weight=meta["target_text_weight"]
    _,loader=get_dataset("CIFAR100","train",adv.train_preprocess,location=a.data_location,batch_size=a.batch_size,num_workers=0)
    trigger=load_trigger_patch(a.trigger_path,a.patch_size,device); params=[p for p in adv.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay); log=out/"stage2.jsonl"; start=time.time(); step=0; epoch_summaries=[]; all_rows=[]; eps=1e-6
    save_json({"method":"KDR_DTK_BMR_OFF_V2","target_task":a.target_task,"target_classname":meta["target_classname"],"theta0_keys":len(theta0_keys),"delta_bd_init_max_abs":max_init,
               "beta0_error":err0,"beta1_error":err1,"trainable":count_trainable_params(adv),"args":vars(a)},str(out/"config.json"))
    for epoch in range(a.epochs):
        rows=[]
        for bi,batch in enumerate(loader):
            if a.max_train_batches and bi>=a.max_train_batches: break
            b=maybe_dictionarize(batch); images=b["images"].to(device); labels=b["labels"].to(device); triggered=apply_trigger(images[:min(a.bd_batch_size,len(images))],trigger)
            clean_logits=classification_logits(head,adv(images)); clean_loss=F.cross_entropy(clean_logits,labels)
            beta=random.uniform(a.beta_min,a.beta_max); scaled=build_scaled_residual_params(adv,clean,beta,True)
            with torch.no_grad(): s_clean=target_score(clean(triggered),weight)
            s_beta=target_score(call_with_params(adv,scaled,triggered),weight)
            D=(1.-s_clean).detach(); G=s_beta-s_clean.detach(); remaining=(D-G).clamp_min(0.); off_loss=(remaining/(D+eps)).mean()
            identity=float(((D-G)-(1.-s_beta)).abs().max());
            if identity>=1e-5: raise AssertionError(f"D-G identity error {identity}")
            loss=a.clean_weight*clean_loss+a.off_weight*off_loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,a.grad_clip); opt.step()
            ratio=G/(D+eps)
            row={"epoch":epoch,"batch":bi,"global_step":step,"beta":beta,"clean_loss":float(clean_loss),"off_loss":float(off_loss),"total_loss":float(loss),
                 "alpha":beta,"clean_batch_acc":float((clean_logits.argmax(1)==labels).float().mean()),"mean_s_clean":float(s_clean.mean()),"mean_s_alpha":float(s_beta.mean()),"mean_s_beta":float(s_beta.mean()),"mean_D":float(D.mean()),"mean_G":float(G.mean()),
                 "q10_G":float(torch.quantile(G,.1)),"mean_G_over_D":float(ratio.mean()),"q10_G_over_D":float(torch.quantile(ratio,.1)),"mean_remaining_headroom":float(remaining.mean()),
                 "q90_remaining_headroom":float(torch.quantile(remaining,.9)),"fraction_s_beta_gt_s_clean":float((s_beta>s_clean).float().mean()),"fraction_G_over_D_gt_05":float((ratio>.5).float().mean()),
                 "fraction_G_over_D_gt_08":float((ratio>.8).float().mean()),"fraction_G_over_D_gt_09":float((ratio>.9).float().mean()),"identity_error_D_minus_G_vs_1_minus_sbeta":identity}
            append_jsonl(row,str(log)); rows.append(row); all_rows.append(row); step+=1
            if bi%a.log_every==0: print(f"[OFF-V2] e={epoch} b={bi}/{len(loader)} loss={row['total_loss']:.4f} s_c={row['mean_s_clean']:.4f} s_b={row['mean_s_beta']:.4f} G/D={row['mean_G_over_D']:.4f}",flush=True)
        keys=[k for k in rows[0] if isinstance(rows[0][k],(int,float)) and k not in ("epoch","batch","global_step")]
        epoch_summaries.append({"epoch":epoch,**{k:sum(float(r[k]) for r in rows)/len(rows) for k in keys}})
    alpha_bins=[]
    for lo,hi,closed in [(0.,.1,False),(.1,.2,False),(.2,.4,False),(.4,.6,False),(.6,.8,False),(.8,1.,True)]:
        selected=[r for r in all_rows if r["alpha"]>=lo and (r["alpha"]<=hi if closed else r["alpha"]<hi)]
        if not selected: continue
        def avg(key): return sum(float(r[key]) for r in selected)/len(selected)
        alpha_bins.append({"lower":lo,"upper":hi,"upper_inclusive":closed,"count":len(selected),"mean_s_alpha":avg("mean_s_alpha"),"mean_G":avg("mean_G"),"q10_G":avg("q10_G"),"mean_G_over_D":avg("mean_G_over_D"),"q10_G_over_D":avg("q10_G_over_D")})
    save_image_encoder(adv,str(out/"finetuned.pt")); save_json({"target_task":a.target_task,"target_classname":meta["target_classname"],"steps":step,"elapsed_sec":time.time()-start,"epochs":epoch_summaries,"final_epoch":epoch_summaries[-1],"alpha_bins":alpha_bins},str(out/"stage2_summary.json"))
    print(f"[OFF-V2] saved {out/'finetuned.pt'}",flush=True)


if __name__=="__main__": main()
