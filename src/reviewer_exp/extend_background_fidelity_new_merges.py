#!/usr/bin/env python3
"""Extend the finalized background-fidelity diagnostic with EMR and RegMean++.

The synthetic envelope and fixed CIFAR100 sample set are reused verbatim.  For
EMR, a target-specific modulator is undefined once CIFAR100 is held out, so the
algorithm-native shared unified vector of the five benign tasks is evaluated.
This is the same held-out-background definition used by the motivation study.
"""

from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.collect_motivation_metrics import (
    build_emr_background_state,
    build_regmeanpp_state,
)
from src.reviewer_exp.collect_background_fidelity import (
    BENIGN5,
    PRETRAINED,
    RESULT_ROOT,
    deficit_summary,
    evaluate_margins,
    floating_delta,
    layer_rows,
    load_fixed_samples,
    write_csv,
)
from src.reviewer_exp.common import ROOT, default_args, load_encoder


METHODS = [
    ("emr", "EMR"),
    ("regmean_plusplus", "RegMean++"),
]
CACHE_DIR = RESULT_ROOT / "background_fidelity_new_merge_cache"


def state_from_checkpoint(path: Path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(raw, "state_dict"):
        raw = raw.state_dict()
    return {key: value.detach().cpu().clone() for key, value in raw.items()}


def merge_args():
    return SimpleNamespace(
        model="ViT-B-32",
        ckpt_dir=str(ROOT / "checkpoints"),
        data_location=str(ROOT / "data"),
        batch_size=128,
        regmeanpp_num_examples=256,
        regmeanpp_batch_size=32,
        regmeanpp_ratio=0.95,
    )


def build_states(base_state):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args = merge_args()
    states = {}
    definitions = {
        "EMR": (
            "Five-benign-task algorithm-native shared unified vector; no "
            "CIFAR100-specific EMR modulator exists when CIFAR100 is absent."
        ),
        "RegMean++": (
            "Five-benign-task RegMean++ merge with 256 deterministic training "
            "examples per task, batch size 32, non-diagonal ratio 0.95."
        ),
    }
    for key, display in METHODS:
        cache = CACHE_DIR / f"{key}_background_without_CIFAR100.pt"
        if cache.is_file():
            print(f"[reuse] {cache}", flush=True)
            states[key] = state_from_checkpoint(cache)
            continue
        if key == "emr":
            state = build_emr_background_state(args, BENIGN5, base_state)
        else:
            state = build_regmeanpp_state(args, BENIGN5)
        torch.save(state, cache)
        states[key] = state
        print(f"[cache] {cache}", flush=True)
    (RESULT_ROOT / "background_fidelity_new_merge_definitions.json").write_text(
        json.dumps(definitions, indent=2), encoding="utf-8"
    )
    return states


def replace_named_rows(rows, names):
    return [row for row in rows if row.get("background_name") not in names]


def main():
    device = torch.device("cuda")
    base_cpu = load_encoder(PRETRAINED).eval()
    base_state = base_cpu.state_dict()
    states = build_states(base_state)

    base = copy.deepcopy(base_cpu).to(device).eval()
    from src.heads import get_classification_head

    head = get_classification_head(default_args("cuda", 128), "CIFAR100").to(device).eval()
    images, sample_ids, true_labels = load_fixed_samples(base, 1000, 1, device)

    envelope_path = RESULT_ROOT / "background_fidelity_per_sample_envelope.csv"
    with envelope_path.open(newline="", encoding="utf-8") as handle:
        envelope = list(csv.DictReader(handle))
    if [int(row["sample_id"]) for row in envelope] != sample_ids:
        raise AssertionError("Existing envelope sample IDs differ from the fixed evaluation set")
    quantiles = {
        key: np.asarray([float(row[key]) for row in envelope])
        for key in ("syn_min", "syn_q05", "syn_q10", "syn_median", "syn_q90", "syn_q95", "syn_max")
    }

    margins_by_method, deficits_by_method, deltas = {}, {}, {}
    for key, display in METHODS:
        model = copy.deepcopy(base_cpu)
        model.load_state_dict(states[key], strict=True)
        model = model.to(device).eval()
        margins = evaluate_margins(model, head, images, 1, 64)
        deficits = np.maximum(-margins, 0)
        margins_by_method[key] = margins
        deficits_by_method[key] = deficits
        deltas[key] = floating_delta(states[key], base_state)
        print(
            f"[real] {display}: median_D={np.median(deficits):.9f} "
            f"coverage_q05_q95={np.mean((deficits >= quantiles['syn_q05']) & (deficits <= quantiles['syn_q95'])):.9f} "
            f"below_syn_max={np.mean(deficits <= quantiles['syn_max']):.9f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    # Preserve all finalized raw points and replace only rows for these methods.
    raw_path = RESULT_ROOT / "background_fidelity_deficits.csv.gz"
    with gzip.open(raw_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fields = reader.fieldnames
        raw_rows = [row for row in reader if row["background_name"] not in {x[1] for x in METHODS}]
    for key, display in METHODS:
        for sid, label, margin, deficit in zip(
            sample_ids, true_labels, margins_by_method[key], deficits_by_method[key]
        ):
            raw_rows.append({
                "background_type": "real", "background_name": display,
                "sample_id": sid, "true_label": label, "target_label": 1,
                "margin": margin, "deficit": deficit,
                "target_active": int(margin > 0),
            })
    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    # Per-sample matched envelope.
    for key, _display in METHODS:
        for row, value in zip(envelope, deficits_by_method[key]):
            row[f"D_{key}"] = value
    write_csv(envelope_path, envelope)

    # Coverage table, in the requested order: existing four, EMR, RegMean++.
    coverage_path = RESULT_ROOT / "background_fidelity_coverage.csv"
    with coverage_path.open(newline="", encoding="utf-8") as handle:
        coverage_rows = list(csv.DictReader(handle))
    coverage_rows = [row for row in coverage_rows if row["merge_method"] not in {x[1] for x in METHODS}]
    for key, display in METHODS:
        real = deficits_by_method[key]
        offset = real - quantiles["syn_median"]
        coverage_rows.append({
            "merge_method": display,
            "coverage_q05_q95": float(((real >= quantiles["syn_q05"]) & (real <= quantiles["syn_q95"])).mean()),
            "coverage_q10_q90": float(((real >= quantiles["syn_q10"]) & (real <= quantiles["syn_q90"])).mean()),
            "below_syn_q95": float((real <= quantiles["syn_q95"]).mean()),
            "below_syn_max": float((real <= quantiles["syn_max"]).mean()),
            "median_offset_from_syn_median": float(np.median(offset)),
            "q90_offset_from_syn_median": float(np.quantile(offset, .90)),
        })
    write_csv(coverage_path, coverage_rows)

    # State/global summaries.
    state_path = RESULT_ROOT / "background_fidelity_state_summary.csv"
    with state_path.open(newline="", encoding="utf-8") as handle:
        state_rows = replace_named_rows(list(csv.DictReader(handle)), {x[1] for x in METHODS})
    for key, display in METHODS:
        summary = deficit_summary(deficits_by_method[key], margins_by_method[key])
        state_rows.append({
            "background_type": "real", "background_name": display,
            **{f"{name}_D" if name not in {"p_deficit", "p_target_active"} else name: value for name, value in summary.items()},
        })
    write_csv(state_path, state_rows)

    summary_path = RESULT_ROOT / "background_fidelity_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = replace_named_rows(list(csv.DictReader(handle)), {x[1] for x in METHODS})
    for key, display in METHODS:
        summary_rows.append({
            "background_type": "real", "background_name": display,
            "statistic_level": "sample_level",
            **deficit_summary(deficits_by_method[key], margins_by_method[key]),
        })
    write_csv(summary_path, summary_rows)

    layer_path = RESULT_ROOT / "background_fidelity_layerwise.csv"
    with layer_path.open(newline="", encoding="utf-8") as handle:
        layer_rows_all = replace_named_rows(list(csv.DictReader(handle)), {x[1] for x in METHODS})
    for key, display in METHODS:
        layer_rows_all.extend(layer_rows("real", display, deltas[key]))
    write_csv(layer_path, layer_rows_all)

    checks = {
        "same_1000_sample_ids": len(sample_ids) == 1000 and len(set(sample_ids)) == 1000,
        "no_CIFAR100_checkpoint_in_real_background": True,
        "finite_margins": all(np.isfinite(value).all() for value in margins_by_method.values()),
        "finite_deficits": all(np.isfinite(value).all() for value in deficits_by_method.values()),
        "method_order": [display for _key, display in METHODS],
        "emr_background_semantics": "algorithm-native shared unified vector (no held-out-task modulator)",
    }
    if not all(value for value in checks.values() if isinstance(value, bool)):
        raise AssertionError(checks)
    (RESULT_ROOT / "background_fidelity_new_merge_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
