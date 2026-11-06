# Agent Return Checklist — BMR final paper refresh

Repository must report:

```bash
git rev-parse HEAD
git status --short
```

Expected HEAD:

```text
010c97c1811762f7e5262a9410ddb16fab34ecbc
```

Return the generated archive:

```text
paper_refresh_return_010c97c.tar.gz
```

At minimum it must contain:

```text
results/paper_refresh/generalization/
results/paper_refresh/trigger_ablation/
results/paper_refresh/poison/
results/paper_refresh/task_number/
analysis/paper_refresh/gain_deficit/
analysis/bmr_feature_space_cifar100/
analysis/bmr_feature_space_imagenet100/
bmr_paper_results/bmr_feature_space_cifar100_imagenet100.pdf
bmr_paper_results/samplewise_gain_vs_deficit.pdf
bmr_paper_results/poison_ratio_background_strength.pdf
bmr_paper_results/reviewer_response/
logs/paper_refresh/
```

Also include the final checkpoint manifest:

```text
paper_refresh_return_010c97c/CHECKPOINT_MANIFEST.txt
```

Do not return `data/`, model checkpoints, triggers, caches, or large merged encoders
inside the archive. Only paths/hashes/metrics/logs/plots are needed.

If any task fails, return:

```text
the exact command
the failing log
last 200 log lines
git rev-parse HEAD
git status --short
nvidia-smi
```

Do not tune alpha, reserve weight, learning rate, or add new losses in response to a
failure.
