#!/usr/bin/env python3
# Generate an isolated Off-Task Pilot-A trainer from the current BMR-On trainer.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "finetune_kdr_dtk_bmr.py"
DST = ROOT / "src" / "finetune_bmr_offtask_pilot_a.py"

def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"[{label}] expected exactly one match, found {n}. "
            "Repository version differs from the Pilot-A baseline."
        )
    return text.replace(old, new, 1)

if not SRC.exists():
    raise FileNotFoundError(SRC)

text = SRC.read_text()

text = replace_once(
    text,
    "from src.modeling import ImageEncoder\n",
    "from src.modeling import ImageEncoder\n"
    "from src.finetune_backdoor_badmergingoff import build_shadow_classification_head\n",
    "import-shadow-builder",
)

text = replace_once(
    text,
    '    parser.add_argument("--adversary-task", default="CIFAR100")\n'
    '    parser.add_argument("--target-cls", type=int, default=1)\n',
    '    parser.add_argument("--adversary-task", default="CIFAR100")\n'
    '    parser.add_argument("--target-task", default="Cars")\n'
    '    parser.add_argument("--target-cls", type=int, default=1)\n'
    '    parser.add_argument("--num-shadow-classes", type=int, default=300)\n'
    '    parser.add_argument("--shadow-head-root", default="./shadow_head")\n',
    "parser-offtask-args",
)

old_head = '''    classification_head = (
        get_classification_head(
            args,
            args.adversary_task,
        ).to(device)
    )

    freeze_classification_head(
        classification_head
    )
'''
new_head = '''    # Clean utility remains in the adversary-task decision space.
    clean_classification_head = (
        get_classification_head(
            args,
            args.adversary_task,
        ).to(device)
    )
    freeze_classification_head(
        clean_classification_head
    )

    # Pilot-A: the BMR attack objective is instantiated in BadMerging's
    # off-task shadow decision space. The real target-task head is never
    # used during malicious training.
    args.shadow_head_path = ensure_dir(
        os.path.join(args.shadow_head_root, args.model)
    )
    shadow_classification_head = build_shadow_classification_head(
        args,
        args.target_task,
        args.target_cls,
        args.num_shadow_classes,
        args.seed,
    ).to(device)
    freeze_classification_head(
        shadow_classification_head
    )
    shadow_target_idx = 0
'''
text = replace_once(text, old_head, new_head, "heads")

old_labels = '''            target_labels = torch.full(
                (bd_images.shape[0],),
                args.target_cls,
                dtype=torch.long,
                device=device,
            )
'''
new_labels = '''            target_labels = torch.full(
                (bd_images.shape[0],),
                shadow_target_idx,
                dtype=torch.long,
                device=device,
            )
'''
text = replace_once(text, old_labels, new_labels, "shadow-target-label")

text = replace_once(
    text,
    '''            clean_logits = classification_logits(
                classification_head,
                image_encoder(images),
            )
''',
    '''            clean_logits = classification_logits(
                clean_classification_head,
                image_encoder(images),
            )
''',
    "clean-head",
)

text = replace_once(
    text,
    '''            logits_atk = classification_logits(
                classification_head,
                z_atk,
            )
''',
    '''            logits_atk = classification_logits(
                shadow_classification_head,
                z_atk,
            )
''',
    "attack-shadow-head",
)

text = replace_once(
    text,
    '''                    logits_bg = classification_logits(
                        classification_head,
                        z_bg,
                    )
''',
    '''                    logits_bg = classification_logits(
                        shadow_classification_head,
                        z_bg,
                    )
''',
    "background-shadow-head",
)

text = replace_once(
    text,
    '''            attack_margin = target_margin(
                logits_atk,
                args.target_cls,
            )
''',
    '''            attack_margin = target_margin(
                logits_atk,
                shadow_target_idx,
            )
''',
    "attack-shadow-margin",
)

text = replace_once(
    text,
    '''                background_margin = target_margin(
                    logits_bg,
                    args.target_cls,
                )
''',
    '''                background_margin = target_margin(
                    logits_bg,
                    shadow_target_idx,
                )
''',
    "background-shadow-margin",
)

n_cmp = text.count("== args.target_cls")
if n_cmp != 2:
    raise RuntimeError(
        f"[target-rate] expected 2 comparisons to args.target_cls, found {n_cmp}"
    )
text = text.replace("== args.target_cls", "== shadow_target_idx")

text = replace_once(
    text,
    '''            "stage1": (
                "reuse fixed KDR_DTK trigger"
            ),
''',
    '''            "stage1": (
                "reuse fixed BadMerging-Off trigger for Pilot-A; "
                "trigger design is deliberately not evaluated here"
            ),
            "setting": "off-task mechanism pilot",
            "adversary_task": args.adversary_task,
            "target_task": args.target_task,
            "true_target_class": args.target_cls,
            "decision_space": "BadMerging shadow classification head",
            "shadow_target_index": shadow_target_idx,
            "num_shadow_classes": args.num_shadow_classes,
            "important_constraint": (
                "the real target-task classification head is not used "
                "by stage-2 malicious training"
            ),
''',
    "config-metadata",
)

text = replace_once(
    text,
    '''        "method": (
            "KDR-DTK Background Margin Reserve"
        ),
''',
    '''        "method": (
            "BMR Off-Task Pilot-A: shadow-space Background Margin Reserve"
        ),
        "adversary_task": args.adversary_task,
        "target_task": args.target_task,
        "true_target_class": args.target_cls,
        "shadow_target_index": shadow_target_idx,
''',
    "summary-metadata",
)

for forbidden in (
    "classification_logits(\n                classification_head,",
    "classification_logits(\n                        classification_head,",
):
    if forbidden in text:
        raise RuntimeError("An attack decision-space call still uses the original head.")

DST.write_text(text)
print(f"[Pilot-A] generated {DST}")
print("[Pilot-A] original BMR-On trainer left unchanged")
