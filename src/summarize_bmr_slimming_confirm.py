#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

PAIR = {
    "KDR_DTK_BMR_SLIM_NODRIFT": ("no_drift", 2026),
    "KDR_DTK_BMR_SLIM_DRIFT": ("with_drift", 2026),
    "KDR_DTK_BMR_SLIM_NODRIFT_S2027": ("no_drift", 2027),
    "KDR_DTK_BMR_SLIM_DRIFT_S2027": ("with_drift", 2027),
    "KDR_DTK_BMR_SLIM_NODRIFT_S2028": ("no_drift", 2028),
    "KDR_DTK_BMR_SLIM_DRIFT_S2028": ("with_drift", 2028),
}

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--analysis-dir", required=True)
    return p.parse_args()

def latest(d):
    xs=sorted(Path(d).glob("*_real_merge_gain.json"))
    if not xs: raise FileNotFoundError(d)
    return xs[-1]

def main():
    a=parse()
    src=latest(a.analysis_dir)
    data=json.loads(src.read_text())
    vals={}
    for r in data["rows"]:
        if r["attack_type"] not in PAIR: continue
        kind, seed = PAIR[r["attack_type"]]
        vals[(kind,seed,r["merge_method"])]=r

    lines=["# BMR no-drift vs drift: 3-seed confirmation",""]
    lines += ["| Variant | Merge | Mean ASR | Std ASR | Mean q10(G) |",
              "|---|---|---:|---:|---:|"]
    summary={}
    for kind in ("no_drift","with_drift"):
        summary[kind]={}
        for merger in ("ta","ties","regmean"):
            rs=[vals[(kind,s,merger)] for s in (2026,2027,2028)]
            asr=np.array([x["asr"] for x in rs],float)
            q10=np.array([x["g_q10"] for x in rs],float)
            summary[kind][merger]={
                "asr_mean":float(asr.mean()),
                "asr_std":float(asr.std(ddof=1)),
                "g_q10_mean":float(q10.mean()),
                "per_seed":{str(s):vals[(kind,s,merger)] for s in (2026,2027,2028)}
            }
            lines.append(
                f"| {kind} | {merger} | {asr.mean():.2f} | "
                f"{asr.std(ddof=1):.2f} | {q10.mean():.4f} |"
            )

    nd=summary["no_drift"]["regmean"]
    dr=summary["with_drift"]["regmean"]
    if nd["asr_mean"] >= dr["asr_mean"] - 1.0 and nd["g_q10_mean"] >= dr["g_q10_mean"] - 0.5:
        decision="REMOVE_SYNTHETIC_DRIFT_PENDING_REGMEANPP"
    elif dr["asr_mean"] - nd["asr_mean"] >= 3.0 and dr["g_q10_mean"] > nd["g_q10_mean"]:
        decision="KEEP_SYNTHETIC_DRIFT"
    else:
        decision="CHECK_REGMEANPP_BEFORE_FINALIZING"

    lines += ["",f"## Decision: **{decision}**","",
              "Use the existing local RegMean++ implementation as the final tie-breaker. "
              "Do not reimplement RegMean++ for this experiment.",""]

    out={"decision":decision,"source":str(src),"summary":summary}
    p=Path(a.analysis_dir)
    (p/"CONFIRM_DECISION.md").write_text("\n".join(lines))
    (p/"CONFIRM_DECISION.json").write_text(json.dumps(out,indent=2))
    print("\n".join(lines))

if __name__=="__main__":
    main()
