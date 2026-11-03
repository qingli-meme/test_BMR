#!/usr/bin/env python3
"""Build the standalone Figure 4 plotting script with losslessly embedded CSVs."""

from __future__ import annotations

import base64
import gzip
import hashlib
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "plot_bmr_feature_space_combined.py"
OUTPUT = ROOT / "bmr_paper_results" / "figure4_decision_space_plot_with_data.py"
CSV_PATHS = {
    "CIFAR100": ROOT / "bmr_paper_results" / "bmr_feature_space_cifar100_points.csv",
    "ImageNet100": ROOT / "bmr_paper_results" / "bmr_feature_space_imagenet100_points.csv",
}


def literal_chunks(value: str) -> str:
    return "\n".join(f'        b"{chunk}"' for chunk in textwrap.wrap(value, 88))


def main():
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        '"""Combine CIFAR100 and ImageNet100 BMR feature-space panels."""',
        '"""Standalone Figure 4 plotter with the exact point data embedded.\n\n'
        'No model, checkpoint, or external CSV is needed. Use --extract-data to\n'
        'materialize the two source CSV files stored losslessly in this script.\n'
        '"""',
    )
    source = source.replace(
        "import argparse\nimport csv\n",
        "import argparse\nimport base64\nimport csv\nimport gzip\nimport hashlib\nimport io\n",
    )

    payload_lines = ["\n# Exact source CSV bytes: gzip-compressed and Base85-encoded.\nEMBEDDED_DATA = {"]
    hashes = []
    for dataset, path in CSV_PATHS.items():
        raw = path.read_bytes()
        encoded = base64.b85encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
        payload_lines.extend([
            f'    "{dataset}": (',
            literal_chunks(encoded),
            "    ),",
        ])
        hashes.append(f'    "{dataset}": "{hashlib.sha256(raw).hexdigest()}",')
    payload_lines.extend(["}", "EMBEDDED_SHA256 = {", *hashes, "}\n"])
    source = source.replace("\nPANELS = [", "\n" + "\n".join(payload_lines) + "\nPANELS = [", 1)

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
    parser.add_argument(
        "--extract-data",
        action="store_true",
        help="also write the exact embedded CIFAR100/ImageNet100 point CSVs",
    )
    return parser.parse_args()


def embedded_csv_bytes(dataset):
    raw = gzip.decompress(base64.b85decode(EMBEDDED_DATA[dataset]))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EMBEDDED_SHA256[dataset]:
        raise RuntimeError(f"embedded {dataset} data checksum mismatch")
    return raw
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
    points = {panel: {group: [] for group in GROUPS} for panel, _ in PANELS}
    text = embedded_csv_bytes(dataset).decode("utf-8")
    for row in csv.DictReader(io.StringIO(text)):
        points[row["panel"]][row["group"]].append((float(row["x"]), float(row["y"])))
    for panel, _ in PANELS:
        for group in GROUPS:
            points[panel][group] = np.asarray(points[panel][group], dtype=np.float64)
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
    ]
    if args.extract_data:
        extraction_dir = Path(args.out_dir)
        extraction_dir.mkdir(parents=True, exist_ok=True)
        for dataset in ("CIFAR100", "ImageNet100"):
            path = extraction_dir / f"figure4_{dataset.lower()}_points.csv"
            path.write_bytes(embedded_csv_bytes(dataset))
            print(path)''',
    )
    source = source.replace(
        'stem = out_dir / "bmr_feature_space_cifar100_imagenet100"',
        'stem = out_dir / "figure4_decision_space_cifar100_imagenet100"',
    )
    OUTPUT.write_text(source, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
