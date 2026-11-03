# BMR（最终实现）

本仓库中的 `KDR_DTK_BMR` 现统一表示无 synthetic-background drift 的正式 BMR。旧版对背景参数注入随机漂移的实现、参数、方法别名和启动脚本均已移除。

## 定义

令预训练参数为 `theta0`，当前恶意模型残差为 `delta_adv = theta_adv - theta0`。每个训练批次采样：

```text
alpha ~ Uniform(alpha_min, alpha_max)
theta_atk = theta0 + alpha * delta_adv
theta_bg  = theta0
```

其中 `theta_bg` 只是固定预训练参考，用于计算 matched relative objective；它不包含 `delta_bg`，也不采样 `eta`、`rho` 或任何 synthetic drift。

默认缩放区间为 `[0.2, 1.0]`。正式训练入口为：

```bash
python src/finetune_kdr_dtk_bmr.py \
  --model ViT-B-32 \
  --adversary-task CIFAR100 \
  --target-cls 1 \
  --patch-size 22 \
  --trigger-path trigger/ViT-B-32/KDR_DTK_CIFAR100_Tgt_1_L_22.npy \
  --method-name KDR_DTK_BMR \
  --ablation-mode full_bmr \
  --alpha-min 0.2 \
  --alpha-max 1.0
```

## 选型依据

在已完成的 on-task slimming 实验中，无 synthetic drift 的版本与带 drift 版本均达到约 97.5% ASR，独立复跑中无 drift 略优，因此最终方法采用更简单的无 drift 定义。

完整历史数据和对照结论见 [`analysis/bmr_slimming_phase1/BMR_ON_TASK_SLIMMING_REPORT.md`](analysis/bmr_slimming_phase1/BMR_ON_TASK_SLIMMING_REPORT.md)。该报告仅作为方法选型证据，其中旧变体名称不再对应可运行实现。
