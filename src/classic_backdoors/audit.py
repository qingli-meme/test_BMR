#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.classic_backdoors.common import ATTACK_SLUGS, attack_checkpoint, sha256
from src.classic_backdoors.train_attack import UPSTREAM
from src.reviewer_exp.common import ROOT, checkpoint_path, suite_for_adversary


OUT = ROOT / "bmr_paper_results/reviewer_response/classic_backdoors_new_merges"


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def record(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
    }


def main():
    source_files = {
        "badnets": ROOT / "third_party/classic_backdoors/backdoorbench/badnet.py",
        "label_consistent": ROOT / "third_party/classic_backdoors/label_consistent/poison_attack.py",
        "trojannn": ROOT / "third_party/classic_backdoors/trojannn/act_max.tvd.center_part.py",
        "dynamic": ROOT / "third_party/classic_backdoors/dynamic/CIFAR_CondGAN.py",
    }
    tasks = {}
    for adversary in ("CIFAR100", "ImageNet100"):
        suite = suite_for_adversary(adversary)
        tasks[adversary] = {
            "suite": suite,
            "clean_checkpoints": [record(checkpoint_path(task, "clean")) for task in suite],
            "attack_checkpoints_before_run": {
                attack: record(attack_checkpoint(adversary, attack)) for attack in ATTACK_SLUGS
            },
        }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "protocol": {
            "backbone": "CLIP ViT-B/32",
            "adversary_tasks": ["CIFAR100", "ImageNet100"],
            "target_class": 1,
            "trigger_size": 22,
            "merging_methods": ["EMR", "RegMean++"],
            "conditions": ["No Attack", "BadNets", "LabelConsistent", "TrojanNN", "Dynamic Backdoor"],
            "network_downloads_allowed": False,
            "note": "The upstream attacks are adapted to the repository's CLIP/data/head interfaces; existing four-merger results are not rerun.",
        },
        "upstream": {
            attack: {**UPSTREAM[attack], "audited_source": record(source_files[attack])}
            for attack in ATTACK_SLUGS
        },
        "local_port": {
            name: record(ROOT / "src/classic_backdoors" / name)
            for name in ("common.py", "train_attack.py", "evaluate_new_merges.py", "summarize.py")
        },
        "pretrained": record(ROOT / "checkpoints/ViT-B-32/zeroshot.pt"),
        "no_attack_trigger": record(ROOT / "trigger/ViT-B-32/fixed_22.npy"),
        "tasks": tasks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "artifact_audit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
