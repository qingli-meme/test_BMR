#!/usr/bin/env python3
"""Apply the minimal source changes required by the BMR control experiments.

Changes:
1) add a gain_only ablation to src/finetune_kdr_dtk_bmr.py;
2) register the new checkpoint names in src/eval_submerge.py.

The patch is intentionally narrow and assertion-heavy so an agent cannot
silently patch an unexpected repository version.
"""

from __future__ import annotations

import argparse
from pathlib import Path


NEW_ATTACK_TYPES = [
    "KDR_DTK_BMR_GAINONLY",
    "KDR_DTK_BMR_NO_BG",
    "KDR_DTK_BMR_FIXETA",
    "KDR_DTK_BMR_FIXALPHA",
    "KDR_DTK_BMR_BOTHFIX",
    "KDR_DTK_BMR_ORD_NORMMATCH",
    "KDR_DTK_BMR_BG_NORMMATCH",
    "KDR_DTK_BMR_BMR_NORMMATCH_BG",
]


def replace_exact(text: str, old: str, new: str, expected: int = 1) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"Patch guard failed: expected {expected} occurrence(s), found {count}:\n{old}"
        )
    return text.replace(old, new)


def patch_trainer(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if '"gain_only"' in text:
        print(f"[skip] gain_only already present in {path}")
        return False

    text = replace_exact(
        text,
        'choices=["plain", "background_attack", "absolute_margin", "full_bmr"],',
        'choices=["plain", "background_attack", "absolute_margin", "gain_only", "full_bmr"],',
    )
    text = replace_exact(
        text,
        '"margin loss; full_bmr: original reserve loss."',
        '"margin loss; gain_only: matched target-margin gain with no "\n'
        '            "deficit-dependent weighting; full_bmr: original reserve loss."',
    )

    # gain_only needs the same matched background state as full_bmr.
    old = 'if args.ablation_mode == "full_bmr":'
    count = text.count(old)
    if count != 2:
        raise RuntimeError(
            f"Patch guard failed: expected two full_bmr background checks, found {count}"
        )
    text = text.replace(
        old,
        'if args.ablation_mode in ("full_bmr", "gain_only"):',
    )

    text = replace_exact(
        text,
        'if args.ablation_mode in ("full_bmr", "absolute_margin"):',
        'if args.ablation_mode in ("full_bmr", "gain_only", "absolute_margin"):',
    )

    old_loss = '''                if args.ablation_mode == "absolute_margin":\n                    reserve_loss_per_sample = F.relu(\n                        -attack_margin\n                    )\n                else:\n                    reserve_loss_per_sample = torch.log1p(\n                        (\n                            measured_deficit\n                            + RESERVE_NUMERIC_EPS\n                        )\n                        / (\n                            positive_margin_gain\n                            + RESERVE_NUMERIC_EPS\n                        )\n                    )'''
    new_loss = '''                if args.ablation_mode == "absolute_margin":\n                    reserve_loss_per_sample = F.relu(\n                        -attack_margin\n                    )\n                elif args.ablation_mode == "gain_only":\n                    # Same matched states and denominator as Full BMR, but\n                    # remove the sample-dependent background deficit from\n                    # the numerator. This isolates whether conditioning the\n                    # optimization pressure on D provides extra benefit.\n                    reserve_loss_per_sample = torch.log1p(\n                        1.0\n                        / (\n                            positive_margin_gain\n                            + RESERVE_NUMERIC_EPS\n                        )\n                    )\n                else:\n                    reserve_loss_per_sample = torch.log1p(\n                        (\n                            measured_deficit\n                            + RESERVE_NUMERIC_EPS\n                        )\n                        / (\n                            positive_margin_gain\n                            + RESERVE_NUMERIC_EPS\n                        )\n                    )'''
    text = replace_exact(text, old_loss, new_loss)

    path.write_text(text, encoding="utf-8")
    print(f"[patched] {path}")
    return True


def patch_eval_submerge(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    missing = [name for name in NEW_ATTACK_TYPES if f'"{name}"' not in text]
    if not missing:
        print(f"[skip] control attack types already present in {path}")
        return False

    anchor = '            "KDR_DTK_BMR_NOSCALE",\n'
    if text.count(anchor) != 2:
        raise RuntimeError(
            f"Patch guard failed: expected attack-type and trigger-source anchors in {path}"
        )

    # Register only in --attack-type. For --trigger-source all new training
    # controls use the existing KDR_DTK trigger, so no new trigger choices are
    # required. Replace the first anchor only.
    first = text.find(anchor)
    insertion = anchor + "".join(f'            "{name}",\n' for name in missing)
    text = text[:first] + insertion + text[first + len(anchor):]

    path.write_text(text, encoding="utf-8")
    print(f"[patched] {path}: added {', '.join(missing)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    trainer = root / "src" / "finetune_kdr_dtk_bmr.py"
    evaluator = root / "src" / "eval_submerge.py"
    for path in (trainer, evaluator):
        if not path.exists():
            raise FileNotFoundError(path)

    patch_trainer(trainer)
    patch_eval_submerge(evaluator)
    print("[done] control-experiment patch applied")


if __name__ == "__main__":
    main()
