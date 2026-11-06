"""Summarize the RegMean-only off-task V2 alpha-range experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/bmr_offtask_v2_alpha/Cars"
BASELINE_RESULT = ROOT / "results/bmr_offtask_v2/Cars"
VARIANTS = [
    ("U(0.2, 1) baseline", "BASELINE", 0.2, "CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22"),
    ("U(1/6, 1)", "A0167", 1.0 / 6.0, "CIFAR100_KDR_DTK_BMR_OFF_V2_A0167_Cars_Tgt_1_R10_L_22"),
    ("U(0.1, 1)", "A0100", 0.1, "CIFAR100_KDR_DTK_BMR_OFF_V2_A0100_Cars_Tgt_1_R10_L_22"),
    ("U(0, 1)", "A0000", 0.0, "CIFAR100_KDR_DTK_BMR_OFF_V2_A0000_Cars_Tgt_1_R10_L_22"),
]


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mechanism_row(path: Path):
    rows = load(path)
    return next(row for row in rows if row["merger"] == "regmean")


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    bins = {}
    checkpoints = []
    for label, short, alpha_min, checkpoint_name in VARIANTS:
        result = BASELINE_RESULT if short == "BASELINE" else RESULT_ROOT / short
        ckpt_dir = ROOT / "checkpoints/ViT-B-32" / checkpoint_name
        checkpoint = ckpt_dir / "finetuned.pt"
        regmean = load(result / "regmean.json")
        mech = mechanism_row(result / "mechanism.json")
        config = load(ckpt_dir / "config.json")
        summary = load(ckpt_dir / "stage2_summary.json")
        row = {
            "variant": label,
            "tag": short,
            "alpha_min": alpha_min,
            "alpha_max": 1.0,
            "asr_percent": regmean["asr"],
            "avg_clean_acc": regmean["avg_clean_acc"],
            "cars_clean_acc": regmean["target_clean_acc"],
            "median_G_true": mech["median_G_true"],
            "q10_G_true": mech["q10_G_true"],
            "p_G_true_gt_D_true": mech["p_G_true_gt_D_true"],
            "median_G_text": mech["G_text"]["median"],
            "q10_G_text": mech["G_text"]["q10"],
            "pearson_G_text_G_true": mech["pearson_G_text_G_true"],
            "spearman_G_text_G_true": mech["spearman_G_text_G_true"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "beta0_error": config["beta0_error"],
            "beta1_error": config["beta1_error"],
            "elapsed_sec": summary["elapsed_sec"],
        }
        rows.append(row)
        bins[short] = summary.get("alpha_bins", [])
        checkpoints.append({"variant": label, "path": row["checkpoint"], "sha256": row["checkpoint_sha256"]})

    fields = list(rows[0])
    with (RESULT_ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_ROOT / "summary.json").write_text(json.dumps({"rows": rows, "alpha_bins": bins}, indent=2))

    baseline = rows[0]
    strong = [r for r in rows[1:] if r["asr_percent"] >= 95 and r["avg_clean_acc"] >= baseline["avg_clean_acc"] - 0.5]
    partial = [r for r in rows[1:] if 85 <= r["asr_percent"] < 95]
    if strong:
        decision = "Strong support: at least one widened alpha range reaches ASR >= 95% without more than 0.5-point average clean-ACC loss."
    elif partial:
        decision = "Partial support: a widened alpha range reaches 85–95% ASR, but not the strong-support threshold."
    else:
        decision = "Not supported by this sweep: no widened alpha range reaches 85% ASR."

    lines = [
        "# BMR off-task V2 alpha-range RegMean check",
        "",
        "Only RegMean was evaluated. The trigger and target metadata were reused; neither was retrained.",
        "",
        "| Variant | ASR (%) | Avg clean ACC | Cars clean ACC | median G_true | q10 G_true | P(G>D) | median G_text | q10 G_text | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['asr_percent']:.4f} | {r['avg_clean_acc']:.4f} | {r['cars_clean_acc']:.4f} | "
            f"{r['median_G_true']:.6f} | {r['q10_G_true']:.6f} | {100*r['p_G_true_gt_D_true']:.4f}% | "
            f"{r['median_G_text']:.6f} | {r['q10_G_text']:.6f} | {r['pearson_G_text_G_true']:.6f} | {r['spearman_G_text_G_true']:.6f} |"
        )
    lines += ["", "## Decision", "", decision, "", "## Alpha-bin diagnostics"]
    for _, short, _, _ in VARIANTS:
        lines += ["", f"### {short}", "", "| alpha bin | n batches | mean s_alpha | mean G | q10 G | mean G/D | q10 G/D |", "|---|---:|---:|---:|---:|---:|---:|"]
        if not bins[short]:
            lines.append("| unavailable in the original baseline log |  |  |  |  |  |  |")
        for b in bins[short]:
            close = "]" if b["upper_inclusive"] else ")"
            lines.append(f"| [{b['lower']:.1f}, {b['upper']:.1f}{close} | {b['count']} | {b['mean_s_alpha']:.6f} | {b['mean_G']:.6f} | {b['q10_G']:.6f} | {b['mean_G_over_D']:.6f} | {b['q10_G_over_D']:.6f} |")
    lines += ["", "## Exact checkpoints", ""]
    for item in checkpoints:
        lines.append(f"- `{item['variant']}`: `{item['path']}` (SHA256 `{item['sha256']}`)")
    lines += ["", "## Repository state", "", "```text", git_text("status", "--short"), "```", "", "### Diff stat", "", "```text", git_text("diff", "--stat"), "```", ""]
    (RESULT_ROOT / "SUMMARY.md").write_text("\n".join(lines))

    include = [
        ROOT / "scripts/run_bmr_offtask_v2_alpha_regmean.sh",
        ROOT / "scripts/run_bmr_offtask_v2_alpha_remaining.sh",
        ROOT / "src/summarize_bmr_offtask_v2_alpha.py",
        RESULT_ROOT / "SUMMARY.md",
        RESULT_ROOT / "summary.csv",
        RESULT_ROOT / "summary.json",
    ]
    for _, short, _, checkpoint_name in VARIANTS[1:]:
        result = RESULT_ROOT / short
        include += [result / name for name in ("training_config.json", "stage2.jsonl", "stage2_summary.json", "regmean.json", "mechanism.json", "mechanism_summary.csv")]
    archive = ROOT / "bmr_offtask_v2_alpha_regmean_return.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in include:
            if path.is_file():
                tar.add(path, arcname=path.relative_to(ROOT))
    print(json.dumps({"summary": str(RESULT_ROOT / "SUMMARY.md"), "archive": str(archive), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
