#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default="./results/offtask_pilot_a/pilot_a.json")
    p.add_argument("--baseline", default="./results/offtask_pilot_a/badmerging.json")
    p.add_argument("--out-md", default="./results/offtask_pilot_a/summary.md")
    p.add_argument("--out-json", default="./results/offtask_pilot_a/summary.json")
    return p.parse_args()

def load(path):
    with open(path) as f:
        return json.load(f)

def fmt(x, digits=2):
    if x is None:
        return "-"
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

def main():
    args = parse_args()
    pilot = load(args.pilot)
    base = load(args.baseline)
    methods = [m for m in ("ta", "ties", "regmean")
               if m in pilot["merge_results"] and m in base["merge_results"]]

    rows, detail = [], {}
    strong_asr = true_gain_pos = q10_pos = sign_agree = 0
    utility_ok = catastrophic = surrogate_mismatch = trigger_confounded = 0

    for m in methods:
        p = pilot["merge_results"][m]
        b = base["merge_results"][m]
        pm, bm = p["margin_analysis"], b["margin_analysis"]

        p_asr, b_asr = p["asr"], b["asr"]
        clean_trig = p["clean_merge_trigger_asr"]
        gt_med, gt_q10 = pm["true_gain_median"], pm["true_gain_q10"]
        gs_med = pm["shadow_gain_median"]
        agreement, corr = pm["gain_sign_agreement"], pm["gain_pearson"]
        p_acc, b_acc = p.get("avg_clean_acc"), b.get("avg_clean_acc")

        strong_asr += int(p_asr >= 90)
        catastrophic += int(p_asr < 60)
        true_gain_pos += int(gt_med > 0)
        q10_pos += int(gt_q10 > 0)
        sign_agree += int(agreement >= 0.60)
        utility_ok += int(p_acc is None or b_acc is None or p_acc >= b_acc - 1.5)
        surrogate_mismatch += int(gs_med > 0 and gt_med <= 0)
        trigger_confounded += int(clean_trig >= 90 and p_asr >= 90 and gt_med <= 0)

        rows.append([
            m, b_asr, p_asr, clean_trig,
            bm["true_gain_median"], gt_med, gt_q10, gs_med,
            agreement, corr, b_acc, p_acc,
        ])

        detail[m] = {
            "baseline_asr": b_asr,
            "pilot_asr": p_asr,
            "clean_merge_trigger_asr": clean_trig,
            "baseline_true_gain_median": bm["true_gain_median"],
            "pilot_true_gain_median": gt_med,
            "pilot_true_gain_q10": gt_q10,
            "pilot_shadow_gain_median": gs_med,
            "pilot_gain_sign_agreement": agreement,
            "pilot_gain_pearson": corr,
            "baseline_avg_clean_acc": b_acc,
            "pilot_avg_clean_acc": p_acc,
        }

    n = len(methods)
    need = min(2, n) if n else 0

    if n == 0:
        decision = "STOP"
        reason = "No common merger results found."
    elif catastrophic >= need or surrogate_mismatch >= need:
        decision = "STOP"
        reason = (
            "Structural failure: Pilot-A collapses on multiple mergers or "
            "shadow-space contribution does not transfer to the true target head."
        )
    elif (
        strong_asr >= need
        and true_gain_pos >= need
        and sign_agree >= need
        and utility_ok >= need
        and trigger_confounded < need
    ):
        decision = "GO"
        reason = (
            "Shadow-space BMR yields strong off-task performance and a positive "
            "malicious-residual contribution in the true target decision space "
            "across multiple mergers without a clear utility collapse."
        )
    else:
        decision = "INCONCLUSIVE"
        reason = (
            "Results are mixed or trigger-only activation is too strong for clean "
            "mechanistic attribution. Do not add losses; inspect shadow/true transfer."
        )

    summary = {
        "decision": decision,
        "reason": reason,
        "counts": {
            "num_methods": n,
            "pilot_asr_ge_90": strong_asr,
            "pilot_true_gain_median_positive": true_gain_pos,
            "pilot_true_gain_q10_positive": q10_pos,
            "shadow_true_sign_agreement_ge_060": sign_agree,
            "utility_within_1p5pp_of_baseline": utility_ok,
            "pilot_asr_lt_60": catastrophic,
            "surrogate_mismatch": surrogate_mismatch,
            "trigger_confounded": trigger_confounded,
        },
        "methods": detail,
    }

    md = [
        "# BMR Off-Task Pilot-A Result",
        "",
        f"## Decision: **{decision}**",
        "",
        reason,
        "",
        "| Merge | BadMerging ASR | Pilot ASR | Clean-merge trigger ASR | BM median G_T | Pilot median G_T | Pilot q10 G_T | Pilot median G_s | sign agree | corr(G_s,G_T) | BM ACC | Pilot ACC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append("| " + r[0] + " | " + " | ".join(fmt(x) for x in r[1:]) + " |")

    md += [
        "",
        "## Falsification counts",
        "",
    ]
    for k, v in summary["counts"].items():
        md.append(f"- `{k}`: {v}")

    md += [
        "",
        "## Interpretation",
        "",
        "High ASR alone is not treated as evidence because the reused BadMerging-Off "
        "trigger may already activate the target on a clean merge. The key evidence "
        "is whether the Pilot residual produces positive `G_T` in the real target-task "
        "head and whether its shadow-space contribution tracks that true contribution.",
        "",
    ]

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\n".join(md))
    print(f"\nSaved: {args.out_md}")
    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
