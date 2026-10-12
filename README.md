# BMR

This repository contains the implementation and experiment artifacts for BMR
(Background Margin Reserve), a backdoor training method for model merging.

The current codebase keeps the BMR training path, evaluation entry points,
paper-result tables, and plotting scripts used for the ongoing experiments.
Large local artifacts such as datasets, checkpoints, triggers, and logs are not
tracked.

## Main Files

- `src/finetune_kdr_dtk_bmr.py`: BMR stage-2 malicious task training.
- `src/optimize_dormant_target_key_patch.py`: dormant trigger optimization.
- `src/eval_submerge.py`: merged-model ASR and clean utility evaluation.
- `src/eval_bmr_real_merge_gain.py`: real-merge margin-gain evaluation.
- `src/audit_trigger_dormancy_cifar100.py`: pre-merge and clean-merged trigger dormancy audit.
- `src/plot_intro_regmean_orthogonality_gap.py`: intro orthogonality/margin-gap plot.
- `src/plot_intro_regmean_decision_state_map.py`: intro decision-state visualization.
- `src/plot_bmr_feature_space_cifar100.py`: feature-space visualization.
- `scripts/`: BMR experiment runners and queue scripts.
- `bmr_paper_results/`: paper-ready summary tables and figures.

## Local Artifacts

The following directories are intentionally ignored by Git:

- `data/`
- `checkpoints/`
- `trigger/`
- `open_clip/`
- `results/`
- `analysis/`
- `logs/`

Only lightweight placeholder files are tracked for those directories.

## Current Experiment Scope

The repository focuses on BMR-On experiments:

- Main six-task results across TA, TIES, RegMean, and AdaMerging.
- Mechanism analyses on CIFAR100 and multi-adversary-task trigger behavior.
- Clean trigger dormancy audits.
- Real-merge margin-gain evaluation.
- Ablations, poisoning-ratio sensitivity, background-strength sensitivity,
  trigger-size sensitivity, and task-number robustness.

Large model files and raw datasets should stay local and should not be pushed.
