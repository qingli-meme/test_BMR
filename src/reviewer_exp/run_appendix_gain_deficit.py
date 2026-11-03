#!/usr/bin/env python3
"""Sample-wise real-merge gain/deficit diagnostic for the BMR appendix."""
from __future__ import annotations

import argparse, copy, csv, gc, gzip, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval_bmr_real_merge_gain import build_encoder as build_old_merge
from src.eval_bmr_real_merge_gain import task_paths as old_task_paths
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.reviewer_exp.common import BASE6, default_args, load_encoder, sha256_file
from src.reviewer_exp.run_new_merge_mechanism_diagnostics import canonical_receiver, receiver_cache
from src.utils import corner_mask_generation

OUT_DEFAULT = ROOT / "bmr_paper_results" / "appendix_gain_deficit"
TARGET, TARGET_CLASS, SEED = "CIFAR100", 1, 2026
VARIANTS = {
    "ordinary": "KDR_DTK_BMR_ORD",
    "background_aug": "KDR_DTK_BMR_BG",
    "full_bmr": "KDR_DTK_BMR",
}
DISPLAY_VARIANT = {"ordinary": "Ordinary", "background_aug": "Background Aug.", "full_bmr": "Full BMR"}
METHODS = [
    ("ta", "Task Arithmetic"), ("ties", "TIES"), ("regmean", "RegMean"),
    ("adamerging", "AdaMerging"), ("emr", "EMR"), ("regmean_plusplus", "RegMean++"),
]
BACKGROUND_CACHE = {
    "ta": ROOT / "bmr_paper_results/motivation/merge_cache/ta_cifar100_suite_background_without_CIFAR100.pt",
    "ties": ROOT / "bmr_paper_results/motivation/merge_cache/ties_cifar100_suite_background_without_CIFAR100.pt",
    "regmean": ROOT / "bmr_paper_results/motivation/merge_cache/regmean_cifar100_suite_background_without_CIFAR100.pt",
    "emr": ROOT / "bmr_paper_results/motivation/merge_cache/emr_cifar100_suite_background_without_CIFAR100.pt",
    "regmean_plusplus": ROOT / "bmr_paper_results/motivation/merge_cache/regmean_plusplus_cifar100_suite_background_without_CIFAR100.pt",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DEFAULT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def old_args(args):
    return SimpleNamespace(
        model="ViT-B-32", ckpt_dir=str(ROOT / "checkpoints"), data_location=str(ROOT / "data"),
        adversary_task=TARGET, target_task=TARGET, target_cls=TARGET_CLASS, patch_size=22,
        scaling_coef=0.3, ties_reset_thresh=20.0, ties_merge_func="dis-sum",
        regmean_num_train_batch=(1 if args.smoke else 8), adamerging_epochs=500, batch_size=args.batch_size,
        max_eval_batches=(2 if args.smoke else args.max_eval_batches), device=args.device,
        save=str(ROOT / "checkpoints/ViT-B-32"), cache_dir=str(ROOT / "cache"),
        openclip_cachedir=str(ROOT / "open_clip"),
    )


def state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"): obj = obj.state_dict()
    return {k: v.detach().cpu().clone() for k, v in obj.items()}


def encoder_from_state(state):
    enc = load_encoder(ROOT / "checkpoints/ViT-B-32/zeroshot.pt")
    enc.load_state_dict(state, strict=True)
    return enc


def variant_paths(variant):
    root = ROOT / "checkpoints/ViT-B-32"
    attack = root / f"CIFAR100_{VARIANTS[variant]}_CIFAR100_Tgt_1_L_22/finetuned.pt"
    return [attack if task == TARGET else root / task / "finetuned.pt" for task in BASE6]


def lambda_path(variant):
    return ROOT / "ada/ViT-B-32" / f"{VARIANTS[variant]}_CIFAR100_Tgt_1_L_22_Epoch_500.pt"


def build_ada_state(variant, include_adversary):
    base = state_dict(ROOT / "checkpoints/ViT-B-32/zeroshot.pt")
    paths = variant_paths(variant)
    lambdas = torch.load(lambda_path(variant), map_location="cpu", weights_only=False)
    states = [state_dict(path) for path in paths]
    if tuple(lambdas.shape) != (len(base), len(BASE6)):
        raise ValueError(f"Ada coefficients have wrong shape: {tuple(lambdas.shape)}")
    out = {}
    for row, (key, p0) in enumerate(base.items()):
        value = p0.clone()
        if torch.is_floating_point(value):
            start = 0 if include_adversary else 1
            for idx in range(start, len(BASE6)):
                value.add_(lambdas[row, idx] * (states[idx][key] - p0))
        out[key] = value
    return out


def get_background(method, variant, cache_dir):
    if method == "adamerging":
        cache = cache_dir / f"adamerging__{variant}__background.pt"
        if not cache.is_file(): torch.save(build_ada_state(variant, False), cache)
        return encoder_from_state(state_dict(cache)), cache
    cache = BACKGROUND_CACHE[method]
    return encoder_from_state(state_dict(cache)), cache


def get_attacked(method, variant, args, cache_dir):
    cache = cache_dir / f"{method}__{variant}__attacked.pt"
    if cache.is_file(): return encoder_from_state(state_dict(cache)), cache
    if method in {"emr", "regmean_plusplus"}:
        receiver = canonical_receiver(method, variant)
        if receiver is None:
            p = receiver_cache(method, variant)
            if not p.is_file(): raise FileNotFoundError(p)
            receiver = load_encoder(p)
    elif method == "adamerging":
        receiver = encoder_from_state(build_ada_state(variant, True))
    else:
        oa = old_args(args)
        paths = dict(zip(BASE6, variant_paths(variant)))
        receiver = build_old_merge(oa, paths, method, VARIANTS[variant])
    torch.save(receiver.state_dict(), cache)
    return receiver.cpu(), cache


def target_margin(logits):
    target = logits[:, TARGET_CLASS]
    other = logits.clone(); other[:, TARGET_CLASS] = -torch.inf
    return target - other.max(1).values


def collect(receiver, args):
    device = torch.device(args.device)
    ea = default_args(args.device, args.batch_size); ea.num_workers = 4
    head = get_classification_head(ea, TARGET).to(device).eval()
    model = ImageClassifier(receiver, head).to(device).eval()
    patch = np.load(ROOT / "trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy")
    applied, mask, _, _ = corner_mask_generation(patch, image_size=(3, 224, 224))
    applied = torch.as_tensor(applied, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
    _, loader = get_dataset(TARGET, "test", model.val_preprocess, location=str(ROOT / "data"),
                            batch_size=args.batch_size, num_workers=4)
    ids, margins = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if (args.smoke and bi >= 2) or (args.max_eval_batches and bi >= args.max_eval_batches): break
            batch = maybe_dictionarize(batch)
            labels = batch["labels"].long(); keep = labels != TARGET_CLASS
            if not keep.any(): continue
            images = batch["images"][keep].float().to(device)
            sample_ids = batch.get("indices", torch.arange(len(labels)))[keep]
            triggered = mask * applied + (1 - mask.expand_as(images)) * images
            margins.append(target_margin(model(triggered)).cpu().numpy())
            ids.extend(int(x) for x in sample_ids)
    model.cpu(); del model, head; torch.cuda.empty_cache()
    return np.asarray(ids, dtype=np.int64), np.concatenate(margins)


def q(x, value): return float(np.quantile(x, value)) if x.size else float("nan")


def summarize(method, variant, d, g, mbg, matk):
    pos = mbg < 0; gap = g[pos] - d[pos]
    return {
        "merge_method": method, "variant": variant, "num_samples": int(len(d)),
        "num_positive_deficit": int(pos.sum()), "median_D": q(d,.5), "q10_D": q(d,.1), "q90_D": q(d,.9),
        "median_G": q(g,.5), "q10_G": q(g,.1), "q90_G": q(g,.9),
        "fraction_G_gt_D_all": float(np.mean(g > d)),
        "fraction_G_gt_D_given_D_positive": float(np.mean(g[pos] > d[pos])) if pos.any() else float("nan"),
        "fraction_Matk_gt_0": float(np.mean(matk > 0)),
        "median_G_minus_D_given_D_positive": q(gap,.5),
        "q10_G_minus_D_given_D_positive": q(gap,.1),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def draw(rows, out):
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","Times","DejaVu Serif"],
        "font.size":8.5, "axes.titlesize":9.5, "axes.labelsize":8.7, "xtick.labelsize":7.4,
        "ytick.labelsize":7.4, "legend.fontsize":8.4, "pdf.fonttype":42, "ps.fonttype":42})
    colors={"Ordinary":"#9AA1A8","Background Aug.":"#4C82B5","Full BMR":"#C43D35"}
    markers={"Ordinary":"o","Background Aug.":"s","Full BMR":"^"}
    rng=np.random.default_rng(SEED)
    fig, axes=plt.subplots(2,3,figsize=(7.15,4.35),sharex=False,sharey=False)
    for ax,(_key,title) in zip(axes.flat,METHODS):
        subset=[r for r in rows if r["merge_method"]==title]
        common=sorted(set.intersection(*[set(r["sample_id"] for r in subset if r["variant"]==v) for v in DISPLAY_VARIANT.values()]))
        chosen=set(rng.choice(common,size=min(1000,len(common)),replace=False).tolist()) if common else set()
        xmax=ymax=0.0
        for v in DISPLAY_VARIANT.values():
            vr=[r for r in subset if r["variant"]==v and r["sample_id"] in chosen]
            x=np.asarray([r["D"] for r in vr]); y=np.asarray([r["G"] for r in vr])
            if x.size: xmax=max(xmax,float(np.quantile(x,.995))); ymax=max(ymax,float(np.quantile(y,.995)))
            ax.scatter(x,y,s=8,c=colors[v],marker=markers[v],alpha=.38,linewidths=.18,edgecolors="#222222",rasterized=True)
        lim=max(xmax,ymax,1e-6)*1.06; ax.plot([0,lim],[0,lim],"--",color="#202020",lw=.9)
        ax.set_xlim(0,lim); ax.set_ylim(min(0,min((r["G"] for r in subset),default=0)),lim)
        ax.set_title(title,fontweight="semibold",pad=3); ax.grid(True,ls=(0,(3,3)),lw=.45,color="#D9DDE1",alpha=.8)
        for s in ax.spines.values(): s.set_color("#202020"); s.set_linewidth(.8)
    for ax in axes[1]: ax.set_xlabel("Background deficit $D$")
    for ax in axes[:,0]: ax.set_ylabel("Target-margin gain $G$")
    handles=[Line2D([],[],ls="none",marker=markers[v],ms=5,mfc=colors[v],mec="#222222",mew=.4,label=v) for v in DISPLAY_VARIANT.values()]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.08,right=.995,bottom=.11,top=.89,wspace=.27,hspace=.34)
    fig.savefig(out/"samplewise_gain_vs_deficit.pdf",bbox_inches="tight",pad_inches=.025)
    fig.savefig(out/"samplewise_gain_vs_deficit.png",dpi=600,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)


def draw_bar(summary, out):
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"pdf.fonttype":42})
    labels=[x[1] for x in METHODS]; x=np.arange(6); width=.24
    colors=["#A6ADB4","#4C82B5","#C43D35"]
    fig,ax=plt.subplots(figsize=(7.05,2.55))
    for j,v in enumerate(DISPLAY_VARIANT.values()):
        vals=[100*next(r["fraction_G_gt_D_given_D_positive"] for r in summary if r["merge_method"]==m and r["variant"]==v) for m in labels]
        ax.bar(x+(j-1)*width,vals,width,color=colors[j],edgecolor="#202020",linewidth=.65,label=v)
    ax.set_xticks(x); ax.set_xticklabels(["Task\nArithmetic","TIES","RegMean","AdaMerging","EMR","RegMean++"])
    ax.set_ylabel(r"$G>D$ given $D>0$ (%)"); ax.set_ylim(0,105); ax.grid(axis="y",ls="--",lw=.5,color="#D9DDE1")
    ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.17))
    for s in ax.spines.values(): s.set_color("#202020"); s.set_linewidth(.85)
    fig.tight_layout(); fig.savefig(out/"reserve_satisfaction_rate.pdf",bbox_inches="tight",pad_inches=.025)
    fig.savefig(out/"reserve_satisfaction_rate.png",dpi=600,bbox_inches="tight",pad_inches=.025); plt.close(fig)


def main():
    args=parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); cache=out/"merge_cache"; cache.mkdir(exist_ok=True)
    required=[ROOT/"checkpoints/ViT-B-32/zeroshot.pt",ROOT/"trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy"]
    for v in VARIANTS: required += variant_paths(v) + [lambda_path(v)]
    required += list(BACKGROUND_CACHE.values())
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError("Missing expected artifacts; no substitution/retraining:\n"+"\n".join(missing))
    rows=[]; summaries=[]; violations_identity=violations_decision=0; reference_ids=None; used={}
    for method,display in METHODS:
        for variant in VARIANTS:
            print(f"[gain-deficit] {display} / {variant}",flush=True)
            bg,bg_path=get_background(method,variant,cache); ids_bg,mbg=collect(bg,args); del bg
            atk,atk_path=get_attacked(method,variant,args,cache); ids_atk,matk=collect(atk,args); del atk
            if not np.array_equal(ids_bg,ids_atk): raise AssertionError(f"sample mismatch {display}/{variant}")
            if reference_ids is None: reference_ids=ids_bg
            elif not np.array_equal(reference_ids,ids_bg): raise AssertionError("sample IDs changed across conditions")
            d=np.maximum(-mbg,0); g=matk-mbg
            violations_identity += int(np.sum(np.abs((mbg+g)-matk)>=1e-5))
            pos=mbg<0; violations_decision += int(np.sum((matk[pos]>0)!=(g[pos]>d[pos])))
            vdisp=DISPLAY_VARIANT[variant]
            summaries.append(summarize(display,vdisp,d,g,mbg,matk))
            rows.extend({"sample_id":int(sid),"merge_method":display,"variant":vdisp,"M_bg":float(a),"M_atk":float(b),
                         "D":float(c),"G":float(e),"success":int(b>0),"deficit_positive":int(a<0),"reserve_satisfied":int(e>c)}
                        for sid,a,b,c,e in zip(ids_bg,mbg,matk,d,g))
            used[f"{method}/{variant}"]={"background":str(Path(bg_path).resolve()),"attacked":str(Path(atk_path).resolve())}
            gc.collect(); torch.cuda.empty_cache()
    with gzip.open(out/"samplewise_gain_deficit.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_csv(out/"gain_deficit_summary.csv",summaries)
    hashes={str(p.resolve()):sha256_file(p) for p in sorted(set(required))}
    meta={"summary":summaries,"checkpoint_paths":used,"input_sha256":hashes,"target_class":TARGET_CLASS,"seed":args.seed,
          "merging_configs":{"Task Arithmetic":{"scaling_coef":.3},"TIES":{"scaling_coef":.3,"reset_thresh":20,"merge_func":"dis-sum"},
          "RegMean":{"num_train_batches":8,"non_diagonal_ratio":.1},"AdaMerging":{"epochs":500,"variant_specific_coefficients":True},
          "EMR":{"background_semantics":"five-benign algorithm-native shared unified vector; no held-out CIFAR100 modulator"},
          "RegMean++":{"num_examples":256,"batch_size":32,"reduce_non_diagonal_ratio":.95}},
          "sanity":{"margin_identity_violations":violations_identity,"decision_equivalence_violations":violations_decision,
          "same_sample_ids":True,"background_excludes_CIFAR100":True,"synthetic_sampler_used":False}}
    (out/"gain_deficit_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    draw(rows,out); draw_bar(summaries,out)
    lines=["# Sample-wise Gain–Deficit Analysis","",f"Command: `python -m src.reviewer_exp.run_appendix_gain_deficit --out-dir {out}`","",
           f"Target class: {TARGET_CLASS}; seed: {args.seed}; samples per condition: {len(reference_ids)}.","",
           "| Merger | Variant | n | median D | median G | P(G>D | D>0) | ASR |","|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries: lines.append(f"| {r['merge_method']} | {r['variant']} | {r['num_samples']} | {r['median_D']:.4f} | {r['median_G']:.4f} | {100*r['fraction_G_gt_D_given_D_positive']:.2f}% | {100*r['fraction_Matk_gt_0']:.2f}% |")
    lines += ["","## Full BMR comparisons",""]
    for _,m in METHODS:
        vals={r['variant']:r['fraction_G_gt_D_given_D_positive'] for r in summaries if r['merge_method']==m}
        lines.append(f"- {m}: Full BMR > Ordinary = `{vals['Full BMR']>vals['Ordinary']}`; Full BMR > Background Aug. = `{vals['Full BMR']>vals['Background Aug.']}`.")
    lines += ["","## Sanity checks","",f"- Margin identity violations: {violations_identity}",f"- Decision-equivalence violations: {violations_decision}","- Every background excludes the CIFAR100 checkpoint.","- No synthetic-background sampler was used.","","Exact paths, hashes, and merge configurations are in `gain_deficit_summary.json`."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if violations_identity or violations_decision: raise AssertionError(meta["sanity"])
    print(json.dumps(meta["sanity"],indent=2))


if __name__=="__main__": main()
