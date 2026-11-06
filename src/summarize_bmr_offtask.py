from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path("results/bmr_offtask_no_shadow")
TASKS=["Cars","GTSRB","EuroSAT","SUN397","PETS"]
METHODS=["ta","ties","regmean","adamerging","emr","regmeanpp"]

def main():
    rows=[]
    for task in TASKS:
        for method in METHODS:
            p=ROOT/task/f"{method}.json"
            if p.is_file(): rows.append(json.loads(p.read_text()))
    ROOT.mkdir(parents=True,exist_ok=True)
    cols=["target_task","merge_method","asr","avg_clean_acc","target_clean_acc","clean_merge_trigger_target_rate"]
    with open(ROOT/"summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines=["# BMR (off-task, no shadow)","","| Target | Merger | ASR | Avg Clean ACC | Target Clean ACC | Clean-Merge Trigger Rate |","|---|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['target_task']} | {r['merge_method']} | {r['asr']:.2f} | {r['avg_clean_acc']:.2f} | {r['target_clean_acc']:.2f} | {r['clean_merge_trigger_target_rate']:.2f} |")
    (ROOT/"summary.md").write_text("\n".join(lines)+"\n")
    manifest={"tasks":TASKS,"methods":METHODS,"completed":[f"{r['target_task']}:{r['merge_method']}" for r in rows],"count":len(rows)}
    (ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
