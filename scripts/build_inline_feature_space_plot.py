#!/usr/bin/env python3
"""Build a standalone Figure 4 plotter with readable inline point arrays."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "plot_bmr_feature_space_combined.py"
OUTPUT = ROOT / "bmr_paper_results" / "figure4_decision_space_plot_inline_data.py"
CSV_PATHS = {
    "CIFAR100": ROOT / "bmr_paper_results" / "bmr_feature_space_cifar100_points.csv",
    "ImageNet100": ROOT / "bmr_paper_results" / "bmr_feature_space_imagenet100_points.csv",
}
PANELS = ("clean", "ta", "ties", "regmean", "adamerging")
GROUPS = ("nontarget_clean", "target_clean", "triggered")


def point_literal():
    data = {}
    for dataset, path in CSV_PATHS.items():
        nested = {panel: {group: [] for group in GROUPS} for panel in PANELS}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                nested[row["panel"]][row["group"]].append(
                    (int(row["sample_id"]), row["x"], row["y"])
                )
        data[dataset] = nested

    lines = [
        "\n# Each tuple is: (sample_id, target_margin, feature_pc1).",
        "# These are the exact 8,750 points used by Figure 4.",
        "POINT_DATA = {",
    ]
    for dataset in ("CIFAR100", "ImageNet100"):
        lines.append(f'    "{dataset}": {{')
        for panel in PANELS:
            lines.append(f'        "{panel}": {{')
            for group in GROUPS:
                lines.append(f'            "{group}": [')
                for sample_id, x, y in data[dataset][panel][group]:
                    lines.append(f"                ({sample_id}, {x}, {y}),")
                lines.append("            ],")
            lines.append("        },")
        lines.append("    },")
    lines.extend(["}\n"])
    return "\n".join(lines)


def main():
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        '"""Combine CIFAR100 and ImageNet100 BMR feature-space panels."""',
        '"""Standalone Figure 4 plotter with all 8,750 points inline.\n\n'
        'Every tuple is (sample_id, target_margin, feature_pc1); no external\n'
        'CSV, checkpoint, model, or data loader is required.\n'
        '"""',
    )
    source = source.replace("\nPANELS = [", point_literal() + "\nPANELS = [", 1)

    old_parse = '''def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cifar-csv",
        default="bmr_paper_results/bmr_feature_space_cifar100_points.csv",
    )
    parser.add_argument(
        "--imagenet-csv",
        default="bmr_paper_results/bmr_feature_space_imagenet100_points.csv",
    )
    parser.add_argument("--out-dir", default="bmr_paper_results")
    return parser.parse_args()
'''
    new_parse = '''def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    return parser.parse_args()
'''
    if old_parse not in source:
        raise RuntimeError("parse_args source block changed")
    source = source.replace(old_parse, new_parse)

    old_load = '''def load_points(path):
    points = {panel: {group: [] for group in GROUPS} for panel, _ in PANELS}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points[row["panel"]][row["group"]].append((float(row["x"]), float(row["y"])))
    for panel, _ in PANELS:
        for group in GROUPS:
            points[panel][group] = np.asarray(points[panel][group], dtype=np.float64)
            if points[panel][group].size == 0:
                raise ValueError(f"missing {panel}/{group} in {path}")
    return points
'''
    new_load = '''def load_points(dataset):
    points = {panel: {} for panel, _ in PANELS}
    for panel, _ in PANELS:
        for group in GROUPS:
            rows = POINT_DATA[dataset][panel][group]
            points[panel][group] = np.asarray(
                [(float(x), float(y)) for _sample_id, x, y in rows],
                dtype=np.float64,
            )
            if points[panel][group].size == 0:
                raise ValueError(f"missing {dataset}/{panel}/{group}")
    return points
'''
    if old_load not in source:
        raise RuntimeError("load_points source block changed")
    source = source.replace(old_load, new_load)
    source = source.replace(
        '''    datasets = [
        ("CIFAR100", load_points(args.cifar_csv)),
        ("ImageNet100", load_points(args.imagenet_csv)),
    ]''',
        '''    datasets = [
        ("CIFAR100", load_points("CIFAR100")),
        ("ImageNet100", load_points("ImageNet100")),
    ]''',
    )
    source = source.replace(
        'stem = out_dir / "bmr_feature_space_cifar100_imagenet100"',
        'stem = out_dir / "figure4_decision_space_cifar100_imagenet100"',
    )
    OUTPUT.write_text(source, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
