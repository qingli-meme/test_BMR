from __future__ import annotations
import argparse,csv,json
from pathlib import Path
TASKS=["Cars","GTSRB","EuroSAT","SUN397","PETS"]; METHODS=["ta","ties","regmean","adamerging","emr","regmeanpp"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--result-root",default="results/bmr_offtask_v2"); a=p.parse_args(); root=Path(a.result_root); rows=[]
    for task in TASKS:
        for method in METHODS:
            f=root/task/f"{method}.json"
            if f.is_file(): rows.append(json.loads(f.read_text()))
    root.mkdir(parents=True,exist_ok=True); cols=["target_task","merge_method","asr","avg_clean_acc","target_clean_acc","clean_merge_trigger_target_rate"]
    with open(root/"summary.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines=["# BMR Off-task v2","","| Target | Merger | ASR | Avg ACC | Target ACC | Clean-trigger rate |","|---|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['target_task']} | {r['merge_method']} | {r['asr']:.2f} | {r['avg_clean_acc']:.2f} | {r['target_clean_acc']:.2f} | {r['clean_merge_trigger_target_rate']:.2f} |")
    if any(r["target_task"]=="Cars" and r["merge_method"]=="ta" for r in rows):
        r=next(x for x in rows if x["target_task"]=="Cars" and x["merge_method"]=="ta")
        lines += ["","## Cars route comparison","","| Route | TA ASR | Avg ACC |","|---|---:|---:|","| Prototype-region BMR | 7.70 | 75.28 |",f"| BMR Off-task v2 | {r['asr']:.2f} | {r['avg_clean_acc']:.2f} |"]
    (root/"summary.md").write_text("\n".join(lines)+"\n"); (root/"manifest.json").write_text(json.dumps({"completed":[f"{r['target_task']}:{r['merge_method']}" for r in rows],"count":len(rows)},indent=2))
if __name__=="__main__": main()
