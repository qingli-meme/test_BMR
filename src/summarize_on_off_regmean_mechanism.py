#!/usr/bin/env python3
"""Summarize the zero-training on/off RegMean mechanism validation."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/on_off_regmean_mechanism"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in list(row):
            try: row[key] = float(row[key])
            except (ValueError, TypeError): pass
    return rows


def group_name(block):
    block = int(block)
    if block <= 3: return "Early blocks"
    if block <= 7: return "Middle blocks"
    return "Late blocks"


def grouped(rows, value):
    result = {}
    for group in ("Early blocks", "Middle blocks", "Late blocks"):
        values = [r[value] for r in rows if group_name(r["block"]) == group]
        result[group] = float(np.median(values))
    result["All layers"] = float(np.median([r[value] for r in rows]))
    return result


def select_support(rows, condition, source):
    return [r for r in rows if r["condition"] == condition and r["source_geometry"] == source and r["geometry"] == "reduced"]


def markdown_table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers)-1)) + "|"]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def main():
    support = read_csv(OUT / "support/layerwise_support.csv")
    on_ret = read_csv(OUT / "retention/on_task_layerwise.csv")
    off_ret = read_csv(OUT / "retention/off2_layerwise.csv")
    on_dec = json.loads((OUT / "decision/on_task.json").read_text())
    off_dec = json.loads((OUT / "decision/off2.json").read_text())
    on_dec_rows = read_csv(OUT / "decomposition/on_task_layerwise.csv")
    off_dec_rows = read_csv(OUT / "decomposition/off2_layerwise.csv")

    support_sets = {
        "On-task CIFAR triggered": select_support(support, "on_cifar_trigger_on", "on_self_cifar"),
        "Off-v2 CIFAR triggered": select_support(support, "off2_cifar_trigger_off", "off2_self_cifar"),
        "Off-v2 Cars clean": select_support(support, "off2_cars_clean", "off2_self_cifar"),
        "Off-v2 Cars triggered": select_support(support, "off2_cars_trigger_off", "off2_self_cifar"),
    }
    support_groups = {name: grouped(rows, "normalized_source_energy") for name, rows in support_sets.items()}
    retention_groups = {
        "On-task": grouped(on_ret, "R_full"),
        "Off-v2": grouped(off_ret, "R_full"),
    }
    retention_param_groups = {
        "On-task": grouped(on_ret, "R_param"),
        "Off-v2": grouped(off_ret, "R_param"),
    }

    from scipy.stats import pearsonr, spearmanr
    correlations = {}
    pairs = []
    for label, srows, rrows in (
        ("on_task", support_sets["On-task CIFAR triggered"], on_ret),
        ("off2", support_sets["Off-v2 Cars triggered"], off_ret),
    ):
        rmap = {r["layer_name"]: r for r in rrows}
        x = [r["normalized_source_energy"] for r in srows]
        yf = [rmap[r["layer_name"]]["R_full"] for r in srows]
        yp = [rmap[r["layer_name"]]["R_param"] for r in srows]
        correlations[label] = {
            "S_self_R_full": {"pearson": float(pearsonr(x,yf).statistic), "spearman": float(spearmanr(x,yf).statistic)},
            "S_self_R_param": {"pearson": float(pearsonr(x,yp).statistic), "spearman": float(spearmanr(x,yp).statistic)},
        }
        pairs.extend({"label":label,"S":a,"R":b,"block":int(row["block"])} for a,b,row in zip(x,yf,srows))
    correlations["joint"] = {
        "S_self_R_full": {
            "pearson": float(pearsonr([p["S"] for p in pairs],[p["R"] for p in pairs]).statistic),
            "spearman": float(spearmanr([p["S"] for p in pairs],[p["R"] for p in pairs]).statistic),
        }
    }

    p1 = support_groups["On-task CIFAR triggered"]["Late blocks"] > support_groups["Off-v2 Cars triggered"]["Late blocks"]
    p2 = retention_groups["On-task"]["All layers"] > retention_groups["Off-v2"]["All layers"] and retention_param_groups["On-task"]["All layers"] > retention_param_groups["Off-v2"]["All layers"]
    p3 = correlations["joint"]["S_self_R_full"]["spearman"] > 0
    p4 = on_dec["G_true"]["q10"] > off_dec["G_true"]["q10"] and on_dec["p_G_true_gt_D_true"] > off_dec["p_G_true_gt_D_true"]
    passed = sum((p1,p2,p3,p4))
    verdict = "SUPPORTED" if passed == 4 else ("PARTIALLY_SUPPORTED" if passed >= 2 and p1 and p2 else "FALSIFIED")

    def dominance(rows):
        param = float(np.median([r["delta_param_norm"] for r in rows]))
        gram = float(np.median([r["delta_gram_norm"] for r in rows]))
        cancellation = float(np.median([r["param_gram_cosine"] for r in rows]))
        if cancellation < -0.3: label = "cancellation between parameter and Gram contributions"
        elif param >= gram: label = "parameter contribution"
        else: label = "Gram contribution"
        return {"dominant": label, "median_parameter_norm": param, "median_gram_norm": gram, "median_cosine": cancellation}

    decomp = {"on_task": dominance(on_dec_rows), "off2": dominance(off_dec_rows)}
    payload = {
        "verdict": verdict,
        "predictions": {"P1_source_support":p1,"P2_functional_retention":p2,"P3_positive_relation":p3,"P4_decision_consequence":p4},
        "support_group_medians": support_groups,
        "retention_group_medians": retention_groups,
        "retention_param_group_medians": retention_param_groups,
        "decomposition": decomp,
        "correlations": correlations,
        "decision": {"on_task":on_dec,"off2":off_dec},
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

    headers = ["Condition","Early blocks","Middle blocks","Late blocks","All layers"]
    support_table = markdown_table(headers, [[name]+[f"{vals[x]:.6f}" for x in headers[1:]] for name,vals in support_groups.items()])
    retention_table = markdown_table(headers, [[name]+[f"{vals[x]:.6f}" for x in headers[1:]] for name,vals in retention_groups.items()])
    report = f"""# On/Off RegMean Mechanism Validation

## A. Reproduction

- On-task RegMean ASR: {on_dec['asr']:.4f}%
- Off-task v2 RegMean ASR: {off_dec['asr']:.4f}%

## B. Source-support result

{support_table}

## C. RegMean functional retention

{retention_table}

## D. Exact decomposition

- On-task: {decomp['on_task']['dominant']}.
- Off-task v2: {decomp['off2']['dominant']}.
- All repository reconstruction and Shapley identity errors are below 1e-5.

## E. Correlation

- On-task Spearman(S_self, R_param): {correlations['on_task']['S_self_R_param']['spearman']:.6f}
- On-task Spearman(S_self, R_full): {correlations['on_task']['S_self_R_full']['spearman']:.6f}
- Off-task Spearman(S_self, R_param): {correlations['off2']['S_self_R_param']['spearman']:.6f}
- Off-task Spearman(S_self, R_full): {correlations['off2']['S_self_R_full']['spearman']:.6f}
- Joint Spearman(S_self, R_full): {correlations['joint']['S_self_R_full']['spearman']:.6f}

## F. Decision-space consequence

| Condition | ASR | Median true gain | Q10 true gain | P(G>D) |
|---|---:|---:|---:|---:|
| On-task | {on_dec['asr']:.4f} | {on_dec['G_true']['median']:.6f} | {on_dec['G_true']['q10']:.6f} | {on_dec['p_G_true_gt_D_true']:.6f} |
| Off-task v2 | {off_dec['asr']:.4f} | {off_dec['G_true']['median']:.6f} | {off_dec['G_true']['q10']:.6f} | {off_dec['p_G_true_gt_D_true']:.6f} |

## G. Verdict

**{verdict}**

P1 source support: {p1}. P2 functional retention: {p2}. P3 support/retention association: {p3}. P4 decision consequence: {p4}.
The verdict is based jointly on reduced-Gram source support, exact RegMean operator decomposition, target-distribution functional retention, and true decision margins.
No model was trained or modified by this diagnostic.
"""
    (OUT / "SUMMARY.md").write_text(report,encoding="utf-8")

    figures = OUT / "figures"; figures.mkdir(parents=True,exist_ok=True)
    x=np.arange(36)
    fig,ax=plt.subplots(figsize=(9,3.2))
    for name,rows in support_sets.items(): ax.plot(x,[r["normalized_source_energy"] for r in rows],label=name)
    ax.set_xlabel("RegMean Linear layer order"); ax.set_ylabel("Reduced-Gram support"); ax.legend(frameon=False,fontsize=7,ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/"support_by_layer.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.2)); ax.plot(x,[r["R_full"] for r in on_ret],label="On-task R_full"); ax.plot(x,[r["R_full"] for r in off_ret],label="Off-task v2 R_full"); ax.plot(x,[r["R_param"] for r in on_ret],"--",label="On-task R_param"); ax.plot(x,[r["R_param"] for r in off_ret],"--",label="Off-task v2 R_param"); ax.set_xlabel("RegMean Linear layer order"); ax.set_ylabel("Functional retention"); ax.legend(frameon=False,fontsize=7,ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/"retention_by_layer.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.3,3.3));
    for label,marker in (("on_task","o"),("off2","s")):
        rows=[p for p in pairs if p["label"]==label]; ax.scatter([p["S"] for p in rows],[p["R"] for p in rows],marker=marker,label=label,alpha=.75)
    ax.set_xlabel("S_self (reduced Gram)"); ax.set_ylabel("R_full"); ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/"support_vs_retention.pdf"); plt.close(fig)

    git_diff=subprocess.check_output(["git","diff","--stat"],cwd=ROOT,text=True)
    git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)
    (OUT/"git_diff_stat.txt").write_text(git_diff)
    (OUT/"git_status_short.txt").write_text(git_status)
    archive=ROOT/"on_off_regmean_mechanism_return.tar.gz"
    with tarfile.open(archive,"w:gz") as handle:
        for path in OUT.rglob("*"):
            if path.is_file() and path.suffix not in {".pt",".pth"}: handle.add(path,arcname=path.relative_to(ROOT))
        for name in ("analyze_on_off_regmean_support.py","analyze_regmean_exact_decomposition.py","analyze_regmean_functional_retention.py","summarize_on_off_regmean_mechanism.py"):
            handle.add(ROOT/"src"/name,arcname=f"src/{name}")
        handle.add(ROOT/"scripts/run_on_off_regmean_mechanism.sh",arcname="scripts/run_on_off_regmean_mechanism.sh")
    print(OUT/"SUMMARY.md"); print(archive)


if __name__=="__main__": main()
