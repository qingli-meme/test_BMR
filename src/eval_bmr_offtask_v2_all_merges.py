"""Evaluate one off-task v2 checkpoint under one of six mergers."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch

sys.path[:0]=[os.path.abspath("."),os.path.abspath("./src")]
import src.eval_bmr_offtask_all_merges as legacy


def args():
    p=argparse.ArgumentParser(); p.add_argument("--target-task",required=True); p.add_argument("--merge-method",required=True,choices=["ta","ties","regmean","adamerging","emr","regmeanpp"])
    p.add_argument("--ckpt-dir",default="./checkpoints"); p.add_argument("--data-location",default="./data"); p.add_argument("--trigger-path",required=True); p.add_argument("--attack-checkpoint",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--target-cls",type=int,default=1); p.add_argument("--patch-size",type=int,default=22); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--adamerging-epochs",type=int,default=500); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--method-tag",default="KDR_DTK_BMR_OFF_V2")
    p.add_argument("--skip-ta-gate",action="store_true")
    p.add_argument("--ada-cache-tag",default="KDR_DTK_BMR_OFF_V2")
    return p.parse_args()


def main():
    a=args(); torch.manual_seed(a.seed); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    if a.merge_method != "ta" and not a.skip_ta_gate:
        ta_path=out/"ta.json"
        threshold=float(os.environ.get("OFFTASK_TA_MIN_ASR","80"))
        if ta_path.exists():
            ta_asr=float(json.loads(ta_path.read_text())["asr"])
            if ta_asr < threshold:
                (out/"TA_SCREEN_FAILED").write_text(
                    f"target_task={a.target_task}\nta_asr={ta_asr}\nthreshold={threshold}\n"
                    "decision=stop_before_remaining_mergers\n"
                )
                print(f"[OFF-V2] TA ASR {ta_asr:.4f}% < {threshold:.4f}%; refusing remaining mergers.",flush=True)
                raise SystemExit(86)
    a.ada_attack_cache_path=str(Path("ada/ViT-B-32")/f"{a.ada_cache_tag}_{a.target_task}_Tgt_1_R10_L_22_Epoch_{a.adamerging_epochs}.pt")
    a.ada_clean_cache_path=str(Path("ada/ViT-B-32")/f"Clean_Epoch_{a.adamerging_epochs}.pt")
    if a.merge_method=="emr":
        attacked_receivers=legacy.emr_receivers(a,True); per,avg,asr=legacy.evaluate_emr(attacked_receivers,a,True)
        clean_receivers=legacy.emr_receivers(a,False); _,_,clean_rate=legacy.evaluate_emr(clean_receivers,a,True)
        attacked=attacked_receivers[legacy.SUITE.index(a.target_task)]; clean=clean_receivers[legacy.SUITE.index(a.target_task)]
    else:
        attacked=legacy.build(a,True); per,avg,asr=legacy.evaluate_encoder(attacked,a,True)
        clean=legacy.build(a,False); _,_,clean_rate=legacy.evaluate_encoder(clean,a,True)
    atk=out/f"{a.merge_method}_attacked.pt"; bg=out/f"{a.merge_method}_background.pt"
    torch.save(attacked.cpu(),atk); torch.save(clean.cpu(),bg)
    payload={"target_task":a.target_task,"merge_method":a.merge_method,"asr":asr,"avg_clean_acc":avg,"target_clean_acc":per[a.target_task],
             "clean_merge_trigger_target_rate":clean_rate,"per_task_acc":per,"attacked_encoder":str(atk),"background_encoder":str(bg),"method_tag":a.method_tag}
    (out/f"{a.merge_method}.json").write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()

