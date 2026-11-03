#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import math

MAP = {
    "KDR_DTK_BMR_SLIM_ORD": "Ordinary",
    "KDR_DTK_BMR_SLIM_SCALE": "Scale variation",
    "KDR_DTK_BMR_SLIM_ABS": "Absolute margin",
    "KDR_DTK_BMR_SLIM_FIXED": "BMR fixed alpha",
    "KDR_DTK_BMR_SLIM_NODRIFT": "BMR no drift",
    "KDR_DTK_BMR_SLIM_DRIFT": "BMR + drift",
}

ORDER = list(MAP.keys())

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", required=True)
    return p.parse_args()

def latest_json(d):
    cand = sorted(Path(d).glob("*_real_merge_gain.json"))
    if not cand:
        raise FileNotFoundError(f"No *_real_merge_gain.json in {d}")
    return cand[-1]

def f(x):
    if x is None:
        return "-"
    return f"{float(x):.2f}"

def main():
    a = args()
    path = latest_json(a.analysis_dir)
    data = json.loads(path.read_text())

    rows = {}
    for r in data["rows"]:
        atk = r["attack_type"]
        if atk in MAP:
            rows[(atk, r["merge_method"])] = r

    util = {}
    for u in data.get("utility", []):
        v = u.get("variant")
        if v in MAP:
            util[(v, u["merge_method"])] = u

    missing = [(v,m) for v in ORDER for m in ("ta","ties","regmean") if (v,m) not in rows]
    if missing:
        raise RuntimeError(f"Missing rows: {missing}")

    reg = {v: rows[(v, "regmean")] for v in ORDER}

    deltas = {
        "scale_vs_ordinary_asr": reg["KDR_DTK_BMR_SLIM_SCALE"]["asr"] - reg["KDR_DTK_BMR_SLIM_ORD"]["asr"],
        "bmr_nodrift_vs_scale_asr": reg["KDR_DTK_BMR_SLIM_NODRIFT"]["asr"] - reg["KDR_DTK_BMR_SLIM_SCALE"]["asr"],
        "alpha_variation_gain_asr": reg["KDR_DTK_BMR_SLIM_NODRIFT"]["asr"] - reg["KDR_DTK_BMR_SLIM_FIXED"]["asr"],
        "drift_minus_nodrift_asr": reg["KDR_DTK_BMR_SLIM_DRIFT"]["asr"] - reg["KDR_DTK_BMR_SLIM_NODRIFT"]["asr"],
        "drift_minus_nodrift_g_q10": reg["KDR_DTK_BMR_SLIM_DRIFT"]["g_q10"] - reg["KDR_DTK_BMR_SLIM_NODRIFT"]["g_q10"],
    }

    nd = reg["KDR_DTK_BMR_SLIM_NODRIFT"]
    dr = reg["KDR_DTK_BMR_SLIM_DRIFT"]

    # The script is deliberately conservative. It only selects a direction;
    # three-seed + RegMean++ confirmation is still required for finalization.
    if nd["asr"] >= dr["asr"] - 1.0 and nd["g_q10"] >= dr["g_q10"] - 0.5:
        decision = "NO_DRIFT_CANDIDATE"
        reason = (
            "No-drift is within 1 pp of the original BMR on RegMean and its "
            "lower-tail real-merge gain is not materially worse. Prefer the "
            "simpler formulation, then confirm with 3 seeds and RegMean++."
        )
    elif dr["asr"] - nd["asr"] >= 3.0 and dr["g_q10"] > nd["g_q10"]:
        decision = "KEEP_DRIFT_CANDIDATE"
        reason = (
            "Synthetic drift gives a >=3 pp RegMean ASR gain and improves q10(G). "
            "It still appears to contribute independently."
        )
    else:
        decision = "INCONCLUSIVE"
        reason = (
            "No-drift and drift are too close or ASR/G disagree. Run the 3-seed "
            "confirmation before changing the paper."
        )

    md = [
        "# BMR Stage-2 Slimming Decision",
        "",
        f"## Decision: **{decision}**",
        "",
        reason,
        "",
        "## Phase-1 results",
        "",
        "| Variant | TA ASR | TIES ASR | RegMean ASR | RegMean median G | RegMean q10 G | RegMean G<=0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for v in ORDER:
        ta = rows[(v,"ta")]
        ties = rows[(v,"ties")]
        rm = rows[(v,"regmean")]
        md.append(
            f"| {MAP[v]} | {f(ta['asr'])} | {f(ties['asr'])} | "
            f"{f(rm['asr'])} | {f(rm['g_median'])} | {f(rm['g_q10'])} | "
            f"{f(100*rm['g_neg_rate'])}% |"
        )

    md += [
        "",
        "## Decisive RegMean differences",
        "",
        f"- Scale variation - Ordinary: **{deltas['scale_vs_ordinary_asr']:+.2f} pp**",
        f"- BMR no drift - Scale variation: **{deltas['bmr_nodrift_vs_scale_asr']:+.2f} pp**",
        f"- BMR no drift - BMR fixed alpha: **{deltas['alpha_variation_gain_asr']:+.2f} pp**",
        f"- BMR + drift - BMR no drift: **{deltas['drift_minus_nodrift_asr']:+.2f} pp**",
        f"- q10(G), drift - no drift: **{deltas['drift_minus_nodrift_g_q10']:+.4f}**",
        "",
        "## Interpretation",
        "",
        "The final-method question is not whether random parameter perturbations can "
        "help in some setting. It is whether they provide independent value after "
        "alpha variation and the matched target-margin objective are present.",
        "",
        "If `BMR no drift` matches or exceeds `BMR + drift`, synthetic drift should "
        "be removed from the final method and retained only as an ablation showing "
        "that additional random background perturbations are unnecessary.",
        "",
    ]

    out = {
        "decision": decision,
        "reason": reason,
        "source": str(path),
        "regmean_differences": deltas,
        "rows": {
            MAP[v]: {m: rows[(v,m)] for m in ("ta","ties","regmean")}
            for v in ORDER
        },
    }

    out_md = Path(a.analysis_dir)/"SLIMMING_DECISION.md"
    out_json = Path(a.analysis_dir)/"SLIMMING_DECISION.json"
    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps(out, indent=2))
    print("\n".join(md))
    print(f"\nSaved {out_md}")
    print(f"Saved {out_json}")

if __name__ == "__main__":
    main()
