#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT=Path(".")
METHODS=["Task Arithmetic","TIES","RegMean","AdaMerging","EMR","RegMean++"]
CSV_METHODS={"TA":"Task Arithmetic","Task Arithmetic":"Task Arithmetic","TIES":"TIES",
             "RegMean":"RegMean","AdaMerging":"AdaMerging","EMR":"EMR",
             "EMR-Merging":"EMR","RegMean++":"RegMean++"}
ASR_COLOR="#E87500"; ACC_COLOR="#4C82B5"; EDGE="#202020"; GRID="#B8B8B8"
RATIOS={"PR1":1.0,"PR2":2.0,"PR5":5.0,"PR20":20.0}

def find_base():
    candidates=[
      ROOT/"bmr_paper_results/reviewer_response/sensitivity_six_merges/poison_ratio_six_merges.csv",
      ROOT/"bmr_paper_results/bmr_poison_ratio_sensitivity_cifar100.csv",
    ]
    for p in candidates:
        if p.is_file(): return p
    raise FileNotFoundError("No previous poisoning-ratio CSV found. Need old TA/TIES/AdaMerging/EMR rows.")

def read_base(path):
    rows={}
    with path.open(encoding="utf-8") as f:
        r=csv.DictReader(f); fields=set(r.fieldnames or [])
        if "merge" in fields:
            for x in r:
                method=CSV_METHODS.get(x["merge"])
                if method is None: continue
                ratio=float(x.get("poison_ratio",x.get("ratio","nan")))
                rows[(method,ratio)]={"poison_ratio":ratio,"merge":method,
                    "asr":float(x["asr"]),"avg_clean_acc":float(x["avg_clean_acc"])}
        else:
            pref={"Task Arithmetic":"ta","TIES":"ties","RegMean":"regmean",
                  "AdaMerging":"adamerging","EMR":"emr","RegMean++":"regmeanpp"}
            for x in r:
                ratio=float(x["poison_ratio"])
                for m,p in pref.items():
                    if f"{p}_asr" in x:
                        rows[(m,ratio)]={"poison_ratio":ratio,"merge":m,
                            "asr":float(x[f"{p}_asr"]),"avg_clean_acc":float(x[f"{p}_acc"])}
    return rows

def parse_regmean(log):
    txt=Path(log).read_text(errors="ignore")
    accs=re.findall(r"Avg ACC:([0-9.]+)%",txt)
    asrs=re.findall(r"Backdoor acc:\s*([0-9.]+)",txt)
    if not accs or not asrs: raise RuntimeError(f"Cannot parse {log}")
    return float(accs[-1]),100.0*float(asrs[-1])

def main():
    base=find_base(); rows=read_base(base)
    updates=[]
    for tag,ratio in RATIOS.items():
        d=ROOT/"results/paper_refresh/poison"/tag
        acc,asr=parse_regmean(d/"regmean.log")
        rows[("RegMean",ratio)]={"poison_ratio":ratio,"merge":"RegMean","asr":asr,"avg_clean_acc":acc}
        updates.append(rows[("RegMean",ratio)])
        rp=json.loads((d/"regmean_plusplus/result.json").read_text())
        rows[("RegMean++",ratio)]={"poison_ratio":ratio,"merge":"RegMean++",
                                  "asr":float(rp["asr"]),"avg_clean_acc":float(rp["avg_clean_acc"])}
        updates.append(rows[("RegMean++",ratio)])
    for m in METHODS:
        missing=[r for r in RATIOS.values() if (m,r) not in rows]
        if missing: raise RuntimeError(f"Missing old/new poison rows for {m}: {missing}")

    out=ROOT/"bmr_paper_results/reviewer_response/sensitivity_six_merges"
    out.mkdir(parents=True,exist_ok=True)
    csvout=out/"poison_ratio_six_merges_refreshed.csv"
    ordered=[rows[(m,r)] for r in RATIOS.values() for m in METHODS]
    with csvout.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["poison_ratio","merge","asr","avg_clean_acc"])
        w.writeheader(); w.writerows(ordered)
    with (out/"poison_ratio_refreshed_updates.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["poison_ratio","merge","asr","avg_clean_acc"])
        w.writeheader(); w.writerows(updates)

    mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","Liberation Serif","STIXGeneral"],
      "mathtext.fontset":"stix","font.size":7.0,"axes.titlesize":7.4,"axes.labelsize":7.6,
      "xtick.labelsize":5.8,"ytick.labelsize":6.5,"legend.fontsize":7.5,
      "axes.linewidth":.9,"pdf.fonttype":42,"ps.fonttype":42})
    # IEEE-style single-column figure, retaining a compact 2x3 panel layout.
    fig,axes=plt.subplots(2,3,figsize=(3.5,2.60),sharey=True)
    labels=[f"{r:g}%" for r in RATIOS.values()]
    x=np.arange(4); width=.29
    for i,(ax,m) in enumerate(zip(axes.flat,METHODS)):
        vals=[rows[(m,r)] for r in RATIOS.values()]
        ax.bar(x-width/2,[v["asr"] for v in vals],width,color=ASR_COLOR,edgecolor=EDGE,linewidth=.65,zorder=3)
        ax.bar(x+width/2,[v["avg_clean_acc"] for v in vals],width,color=ACC_COLOR,edgecolor=EDGE,linewidth=.65,zorder=3)
        ax.set_title(m,pad=2.5); ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(70,102); ax.set_yticks([70,80,90,100])
        ax.grid(axis="y",ls="--",lw=.65,color=GRID,alpha=.5,zorder=0); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if i % 3:
            ax.tick_params(axis="y",labelleft=False)
    fig.text(.018,.47,"Score (%)",rotation=90,ha="center",va="center",fontsize=7.6)
    fig.text(.52,.035,"Poisoning ratio",ha="center",va="center",fontsize=7.6)
    fig.legend(handles=[Patch(facecolor=ASR_COLOR,edgecolor=EDGE,label="ASR"),
                        Patch(facecolor=ACC_COLOR,edgecolor=EDGE,label="ACC")],
               loc="upper center",bbox_to_anchor=(.52,.965),ncol=2,frameon=False,
               columnspacing=1.35,handlelength=1.45,handleheight=.8,
               handletextpad=.45,borderaxespad=0)
    fig.subplots_adjust(left=.12,right=.995,bottom=.14,top=.835,wspace=.13,hspace=.52)
    stem=ROOT/"bmr_paper_results/poison_ratio_sensitivity_single_column"
    for suf in (".pdf",".png",".svg"):
        fig.savefig(stem.with_suffix(suf),dpi=600,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    print(csvout); print(stem.with_suffix(".pdf"))

if __name__=="__main__": main()
