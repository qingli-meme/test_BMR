# BMR On-Task Final-Method Slimming Ablation

## 0. 这轮实验回答什么

这轮不是继续做 background-strength sensitivity，而是决定最终 Stage-2 方法到底应该保留什么。

当前假设：

```text
真正必要：
1. 随机变化 malicious residual 的有效强度 alpha
2. matched target-margin objective

可能不必要：
3. synthetic drift delta_syn
```

因此必须把旧的 `Background Aug.` 拆开，避免把 `alpha variation` 和 `synthetic drift` 混在一起。

---

## 1. 六个训练变体

所有变体使用同一个：

```text
model: ViT-B-32
adversary task: CIFAR100
target class: 1
trigger: KDR_DTK dormant trigger
patch size: 22
seed: 2026
clean/bd/reserve weights: 与当前 BMR 一致
```

### V0 Ordinary

```text
alpha = 1
synthetic drift = 0
BMR loss = no
absolute-margin loss = no
```

即普通 clean + backdoor training。

### V1 Scale variation

```text
alpha ~ U[0.2, 1.0]
synthetic drift = 0
BMR loss = no
absolute-margin loss = no
```

这个变体非常关键。

它回答：

> 旧 `Background Aug.` 的主要收益是不是其实来自 alpha variation？

现有 trainer 中用：

```text
ablation_mode=background_attack
drift_rho=0
```

即可严格实现。

### V2 Absolute margin

```text
alpha ~ U[0.2, 1.0]
synthetic drift = 0
loss includes relu(-M_atk)
```

回答：

> 只让 attacked state 本身进入 target region，是否足够？

### V3 BMR w/o scale variation

```text
alpha = 0.6
synthetic drift = 0
matched BMR loss = yes
```

`0.6` 是 `U[0.2,1.0]` 的均值，沿用旧 ablation 的固定 operating point。

回答：

> matched objective 存在时，alpha variation 是否仍然必要？

### V4 BMR w/o synthetic drift

```text
alpha ~ U[0.2, 1.0]
theta_bg = theta0
theta_atk = theta0 + alpha * delta_adv
matched BMR loss = yes
```

这是当前最可能成为最终方法的候选。

对应：

```math
G_alpha =
M(theta0 + alpha delta_adv, x_tau)
-
M(theta0, x_tau)

D_0 =
[-M(theta0, x_tau)]_+
```

BMR loss 保持不变。

### V5 BMR + synthetic drift

当前原始 Full BMR：

```text
alpha ~ U[0.2,1.0]
eta ~ U[0.2,1.0]
rho = 0.25
synthetic drift = yes
matched BMR loss = yes
```

它只作为最终对照。

---

## 2. 为什么现有 trainer 不需要修改

当前 `src/finetune_kdr_dtk_bmr.py` 已经足够。

当：

```bash
--drift-rho 0
```

时，`sample_synthetic_drift_cache()` 生成的 drift 为严格零，因此：

```math
theta_bg = theta0
theta_atk = theta0 + alpha delta_adv
```

所以不需要新写训练逻辑。

这点很重要：最终判定不会被“重新实现 trainer”引入的代码差异污染。

---

## 3. Phase-1：单 seed 决定机制

### 推荐直接挂 tmux

如果有 6 张 GPU：

```bash
GPU_LIST=0,1,2,3,4,5 \
bash scripts/launch_bmr_slimming_tmux.sh
```

如果只有 4 张：

```bash
GPU_LIST=0,1,2,3 \
bash scripts/launch_bmr_slimming_tmux.sh
```

脚本会自动 round-robin：

```text
GPU 0: ordinary -> bmr_no_drift
GPU 1: scale_var -> bmr_with_drift
GPU 2: absolute_margin
GPU 3: bmr_fixed_alpha
```

具体分配由 GPU 数自动决定。

查看：

```bash
tmux ls
```

进入某个：

```bash
tmux attach -t bmrslim_g0
```

---

## 4. Phase-1 评估

训练全部结束后：

```bash
bash scripts/eval_bmr_slimming_phase1.sh 0
```

第一阶段默认只跑：

```text
Task Arithmetic
TIES
RegMean
```

原因不是省实验，而是这三者足以做 Day-1 方法判定：

- TA/TIES 检查没有退化；
- RegMean 是当前最有区分度的公开仓库 merger；
- AdaMerging/EMR 大量设置已接近饱和，不适合用来决定 synthetic drift 是否必要；
- 当前公开 `test_BMR` 分支没有 EMR / RegMean++ evaluator，不能凭空重写。

评估直接复用仓库已有：

```text
src/eval_bmr_real_merge_gain.py
```

因此除了 ASR，还会输出真实 merge 上：

```text
median G
q10 G
negative-G rate
clean utility
```

这正好判断 no-drift 是否仍能在完全未见的真实 merge background 上提供足够 target-margin contribution。

随后：

```bash
python3 src/summarize_bmr_slimming.py \
  --analysis-dir analysis/bmr_slimming_phase1
```

自动生成：

```text
analysis/bmr_slimming_phase1/SLIMMING_DECISION.md
analysis/bmr_slimming_phase1/SLIMMING_DECISION.json
```

---

## 5. 我们真正看哪几个比较

不要看一堆饱和的 100%。

### 比较 A

```text
Ordinary
vs
Scale variation
```

如果 RegMean 明显上升：

```text
说明旧 Background Aug. 的主要收益可以由 alpha variation 单独解释。
```

### 比较 B

```text
Scale variation
vs
BMR w/o synthetic drift
```

如果再明显上升：

```text
说明 matched target-margin objective 提供独立收益。
```

### 比较 C

```text
BMR w/o scale variation
vs
BMR w/o synthetic drift
```

如果固定 alpha 明显下降：

```text
说明 alpha variation 是最终方法必要部分。
```

### 比较 D

```text
BMR w/o synthetic drift
vs
BMR + synthetic drift
```

这是最终方法选择。

如果 no-drift 不差甚至更好，就删除 synthetic drift。

---

## 6. Phase-1 判定标准

优先看 RegMean。

### NO-DRIFT CANDIDATE

满足：

```text
ASR_no_drift >= ASR_with_drift - 1.0 pp
```

并且：

```text
q10(G)_no_drift 没有明显恶化
clean ACC 没有明显恶化
```

同时最好：

```text
ASR_scale_var >> ASR_ordinary
ASR_no_drift > ASR_scale_var
ASR_no_drift > ASR_fixed_alpha
```

则进入 no-drift 作为最终方法候选。

### KEEP DRIFT

如果：

```text
RegMean:
ASR_with_drift - ASR_no_drift >= 3 pp
```

且真实 merge 的：

```text
q10(G)_with_drift
```

也稳定明显更高，则 synthetic drift 仍有独立价值。

### INCONCLUSIVE

如果两者差距约 1--3 pp，或者 ASR 接近但 q10(G) 冲突，则进入三 seed。

不要调 rho、eta、lambda。

---

## 7. Phase-2：只确认两个 finalist

Phase-1 若支持 no-drift，才运行：

```bash
GPU_LIST=0,1,2,3 \
bash scripts/launch_bmr_slimming_confirm_tmux.sh
```

Phase-2 只补：

```text
no-drift: seed 2027, 2028
with-drift: seed 2027, 2028
```

seed 2026 直接使用 Phase-1。

然后运行：

```bash
bash scripts/eval_bmr_slimming_confirm.sh 0
```

这样得到 3-seed：

```text
2026 / 2027 / 2028
```

---

## 8. RegMean++ 和 EMR 怎么处理

当前公开仓库没有这两个 runner，因此这个包不会自行实现。

如果 agent 所在工作树里已经有你之前跑通的 EMR / RegMean++ evaluator：

### Phase-1

只需要评估：

```text
KDR_DTK_BMR_SLIM_NODRIFT
KDR_DTK_BMR_SLIM_DRIFT
```

不需要把六个 variant 全跑一遍。

### Phase-2

再评估：

```text
KDR_DTK_BMR_SLIM_NODRIFT
KDR_DTK_BMR_SLIM_DRIFT

KDR_DTK_BMR_SLIM_NODRIFT_S2027
KDR_DTK_BMR_SLIM_DRIFT_S2027

KDR_DTK_BMR_SLIM_NODRIFT_S2028
KDR_DTK_BMR_SLIM_DRIFT_S2028
```

这是最终论文最需要的确认。

不要新写一个未经验证的 RegMean++ / EMR 实现。

---

## 9. 最可能的最终论文表

如果结果支持当前假设，训练消融最终建议改成：

| Variant | alpha variation | matched BMR | synthetic drift |
|---|---:|---:|---:|
| Ordinary | ✗ | ✗ | ✗ |
| Scale variation | ✓ | ✗ | ✗ |
| Absolute margin | ✓ | ✗ | ✗ |
| BMR w/o scale variation | ✗ | ✓ | ✗ |
| **BMR** | ✓ | ✓ | ✗ |
| + Synthetic drift | ✓ | ✓ | ✓ |

这里最终 BMR 就是 no-drift。

旧的 `Background Aug.` 名字和对应 attribution 建议废掉。

---

## 10. Agent 最终必须回传

Phase-1：

```text
analysis/bmr_slimming_phase1/SLIMMING_DECISION.md
analysis/bmr_slimming_phase1/SLIMMING_DECISION.json
analysis/bmr_slimming_phase1/*real_merge_gain*.json
analysis/bmr_slimming_phase1/*real_merge_gain*.csv
analysis/bmr_slimming_phase1/*clean_utility_delta*.csv
logs/bmr_slim_*.log
```

如果 Phase-1 支持 no-drift，再跑 Phase-2，并回传：

```text
analysis/bmr_slimming_confirm/*
```

以及本地已有 RegMean++ 的 no-drift vs drift 结果。

不要只汇报最终 ASR。
