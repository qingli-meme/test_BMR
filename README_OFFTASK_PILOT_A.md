# BMR Off-Task Pilot-A

## 目的

这个实验只回答一个问题：

> 在不改变 BMR 核心机制、暂时复用 BadMerging-Off trigger 的条件下，把 BMR 的决策空间从 adversary-task head 改成 BadMerging 的 shadow classification head 后，malicious residual 学到的增量贡献能否迁移到真实的 off-task target decision space？

默认验证：

- adversary task: `CIFAR100`
- target task: `Cars`
- target class: `1`
- trigger: BadMerging-Off trigger
- shadow classes: `300`
- shadow reference data: `10`
- patch size: `28`
- model: `ViT-B-32`
- merge methods: `TA`, `TIES`, `RegMean`

当前仓库没有 RegMean++ 的可调用实现，因此 Day-1 pilot 不自行重写或猜测 RegMean++。Pilot-A 若为 GO，再把同一 checkpoint 接到已有的 RegMean++ 实现中。

## 核心改动

原始 BMR-On：

```text
clean loss        -> adversary-task head
backdoor loss     -> adversary-task head
M_bg / M_atk      -> adversary-task head
G, D              -> adversary-task head
```

Pilot-A：

```text
clean loss        -> adversary-task head（不变）
backdoor loss     -> shadow head
M_bg / M_atk      -> shadow head
G, D              -> shadow head
synthetic drift   -> 不变
alpha / eta       -> 不变
reserve loss      -> 不变
trainable scope   -> 不变
```

shadow head 的 target index 固定为 `0`；`--target-cls=1` 仍表示真实 Cars target class index，用于构建 shadow head。

这不是最终 Off-Task BMR。BadMerging-Off trigger 只作为 mechanism diagnostic，暂时不验证 dormant trigger。

## 安装

把压缩包里的 `src/`、`scripts/`、`tools/` 复制到 `test_BMR` 仓库根目录，然后：

```bash
cd test_BMR
python3 tools/make_offtask_pilot_a_train.py
```

它会从当前：

```text
src/finetune_kdr_dtk_bmr.py
```

复制并生成：

```text
src/finetune_bmr_offtask_pilot_a.py
```

原始 BMR 文件不会被修改。patcher 带严格 preflight check；仓库版本不匹配时会直接失败，不会静默产生错误代码。

## 一键运行

```bash
bash scripts/run_offtask_pilot_a.sh 0
```

`0` 是 GPU ID。

脚本会自动：

1. 检查 BadMerging-Off trigger；缺失时生成；
2. 检查 BadMerging-Off CIFAR100→Cars baseline；缺失时训练；
3. 训练 Pilot-A；
4. 对 BadMerging-Off 与 Pilot-A 使用同一 evaluator 跑 TA / TIES / RegMean；
5. 对每个 merger 同时构造 clean merge 与 malicious merge；
6. 测 ASR、clean utility、真实 target head 与 shadow head 的 margin contribution；
7. 自动生成 `GO / STOP / INCONCLUSIVE` 汇总。

## 常用环境变量

```bash
GPU_ID=0 \
SEED=2026 \
BATCH_SIZE=128 \
BD_BATCH_SIZE=64 \
BMR_LR=5e-7 \
RUN_REGMEAN=1 \
bash scripts/run_offtask_pilot_a.sh 0
```

默认：

```text
MODEL=ViT-B-32
ADV=CIFAR100
TARGET_TASK=Cars
TARGET_CLS=1
PATCH_SIZE=28
NUM_SHADOW_DATA=10
NUM_SHADOW_CLASSES=300
BADMERGING_ALPHA=5
SEED=2026
BATCH_SIZE=128
BD_BATCH_SIZE=64
BMR_LR=5e-7
SCALING_COEF=0.3
RUN_REGMEAN=1
```

快速 smoke test：

```bash
MAX_EVAL_BATCHES=10 \
MARGIN_MAX_BATCHES=10 \
RUN_REGMEAN=0 \
bash scripts/run_offtask_pilot_a.sh 0
```

正式判断必须用完整测试集。

## 输出

```text
results/offtask_pilot_a/badmerging.json
results/offtask_pilot_a/pilot_a.json
results/offtask_pilot_a/summary.md
results/offtask_pilot_a/summary.json
logs/offtask_pilot_a_*.log
```

Pilot checkpoint：

```text
checkpoints/ViT-B-32/
  CIFAR100_BMR_OFF_PILOT_A_TO_Cars_CIFAR100_Tgt_1_L_28/
    finetuned.pt
```

## 为什么不能只看 ASR

BadMerging-Off trigger 本身可能已经在 clean merged model 上产生较高 target activation，因此实验额外构造：

```text
clean merge:      六个 clean task models
malicious merge:  用 Pilot/BadMerging checkpoint 替换 CIFAR100 clean model
```

对同一批 Cars triggered non-target samples，测真实 Cars head：

```text
G_T = M_atk^T - M_bg^T
D_T = relu(-M_bg^T)
```

同时测 shadow head：

```text
G_s = M_atk^s - M_bg^s
D_s = relu(-M_bg^s)
```

输出还包括：

```text
clean-merge trigger ASR
median / q10 G_T
median / q10 G_s
sign agreement between G_s and G_T
Pearson corr(G_s, G_T)
condition agreement
P(G_T>D_T | G_s>D_s, D_T>0)
```

因此即使 ASR 饱和，也能判断 residual 是否真的向真实 target decision 提供正贡献。

## Day-1 falsification

### GO

优先满足：

1. 至少 2/3 mergers 上 Pilot-A ASR ≥ 90%，且没有系统性显著落后 BadMerging-Off；
2. 至少 2/3 mergers 上真实 Cars head 的 `median(G_T) > 0`，最好 `q10(G_T) > 0`；
3. 至少 2/3 mergers 上 shadow/true gain sign agreement ≥ 0.60；
4. 平均 clean ACC 相比 BadMerging-Off 不应下降超过约 1.5 pp。

### STOP

不要调 lambda；若出现：

```text
shadow G_s 明显为正
        ↓
真实 G_T 在多数 merger 上 <= 0
```

或 Pilot-A 至少两个 merger ASR < 60%，说明需要重新检查 `M_s -> M_T`，而不是加 loss。

### INCONCLUSIVE

若：

```text
clean-merge trigger ASR 已接近饱和
+
Pilot-A ASR 很高
+
但 G_T 接近 0 / shadow-true transfer 很弱
```

不能据此证明 BMR core 已成功 off-task，应进入更弱或 dormant trigger diagnostic。

## Agent 回传

请整体回传：

```text
1. results/offtask_pilot_a/summary.md
2. results/offtask_pilot_a/summary.json
3. results/offtask_pilot_a/pilot_a.json
4. results/offtask_pilot_a/badmerging.json
5. Pilot-A training summary JSON
6. 完整主 log
```

如果失败，再附：

```text
git status --short
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__)"
最后 200 行错误日志
```
