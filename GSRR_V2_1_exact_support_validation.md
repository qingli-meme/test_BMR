# GSRR V2.1：Exact Support Projection + Generalization Validation

## 0. 这轮要验证什么

V2 的最终输出看起来等于 benign PTQ，但训练过程其实出现过非常高的 calibration ASR。

当前日志里大致是：

```text
step 1000:
    calibration ASR ≈ 87.5%
    clean teacher agreement = 100%
    flip ratio ≈ 2.19%

step 1500:
    calibration ASR ≈ 94.1%
    clean teacher agreement = 100%
    flip ratio ≈ 2.58%

step 2500:
    calibration ASR ≈ 96.1%
    clean teacher agreement = 100%
    flip ratio ≈ 2.70%
```

最终却恢复到了：

```text
best step = 1
flipped   = 0
```

因此最终 test 只是：

```text
CA  ≈ 93.24%
ASR ≈ 2.04%
```

这不是“攻击没有学出来”，而是：

```text
1% flip-budget feasibility
    ↓
所有高-ASR checkpoint 都被判 infeasible
    ↓
best checkpoint 只能落回 step 1
```

本轮只验证三件事：

1. **训练中高-ASR状态在 test 上到底怎么样？**
2. **1% soft sparsity constraint 是否只是人为过强 + 优化实现失败？**
3. **把 sparsity 改成 exact global Top-K support 后，3% budget 能否得到高 ASR + 高 CA？**

---

# 1. 本轮不要动的部分

全部保留：

```text
benign-centered residual parameterization
global end-to-end training
quantization ambiguity candidate pool
candidate_ratio = 10%
gate_init = -4
hard forward + STE
multi-view clean consistency
clean KL
clean teacher-margin preservation
target-margin backdoor objective
clean primal-dual constraint
```

不要重新回 QuRA：

```text
不加 P(w)
不逐层
不 freeze
不 window
不 final-layer special training
```

---

# 2. 本轮唯一的算法变化：sparsity 从 soft penalty 改为 exact support projection

V2 想实现：

\[
R_{\mathrm{flip}}\le B
\]

但用了：

\[
\lambda_s(R_{\mathrm{flip}}-B)
+
\frac{\mu_s}{2}
[R_{\mathrm{flip}}-B]_+^2.
\]

实际日志里：

```text
dual_flip 一直涨到很大
flip ratio 仍稳定在约 2.5%–2.7%
```

说明对 hard 0/1 gate 而言，这个 penalty 并不能有效控制 support cardinality。

V2.1 直接改成：

\[
\boxed{
\|g\|_0\le K
}
\]

其中：

\[
K=
\lfloor
B N_{\mathrm{valid}}
\rfloor.
\]

第一轮：

\[
B=3\%.
\]

这不是为了最终宣称“3%最好”，而是先让已经观察到的 2.5%–2.7% attack capacity 不被 1% budget 人为截断。

---

# 3. Global Top-K 是怎么做的

每个 candidate 仍然只有一个 flip logit：

\[
u_i.
\]

只有：

\[
u_i>0
\]

才表示“这个 rounding 倾向于背叛 benign PTQ”。

把整个网络所有正的 candidate logits 放到一起：

\[
\mathcal P=
\{u_i\mid u_i>0\}.
\]

如果：

\[
|\mathcal P|\le K,
\]

所有 positive candidate 都激活。

如果：

\[
|\mathcal P|>K,
\]

只保留全网最大的 K 个：

\[
\boxed{
g_i^{hard}
=
\mathbb I[
u_i\in \operatorname{TopK}(\mathcal P)
]
}
\]

注意这是：

\[
\boxed{\text{全网络 Top-K}}
\]

不是：

```text
每层各取 K%
```

所以 early layers、residual blocks、classifier 全部竞争同一个 malicious budget。

---

# 4. 为什么仍然可以端到端训练

forward 使用：

\[
g_i^{hard}\in\{0,1\}.
\]

backward 仍然使用 STE：

\[
g_i^{STE}
=
\operatorname{stopgrad}
(g_i^{hard}-p_i)
+
p_i,
\]

其中：

\[
p_i=\sigma(u_i/T).
\]

因此：

```text
forward:
    是真实合法 quantized model

backward:
    所有 candidate logits 都能获得梯度
```

当前不在 Top-K 的 candidate 仍然能继续学习；只要它之后的 logit 超过现有 support，它就可以进入 Top-K。

所以：

```text
Top-K support 会动态替换
```

不是永久冻结。

---

# 5. Clean constraint 不改

仍然：

\[
D_{\mathrm{clean}}
=
L_{\mathrm{KL}}
+
\eta_m L_{\mathrm{margin}}^{clean}.
\]

约束：

\[
D_{\mathrm{clean}}
\le
\epsilon_c.
\]

训练目标：

\[
\boxed{
L
=
L_{bd}
+
\lambda_c
(D_{\mathrm{clean}}-\epsilon_c)
+
\frac{\mu_c}{2}
[D_{\mathrm{clean}}-\epsilon_c]_+^2
}
\]

只保留 clean dual：

\[
\lambda_c
\leftarrow
[
\lambda_c+
\eta_c
(D_{\mathrm{clean}}-\epsilon_c)
]_+.
\]

删除：

```text
dual_flip
dual_lr_flip
rho_flip
flip_violation
flip-count soft penalty
```

职责现在非常清楚：

```text
backdoor objective:
    target-margin loss

clean:
    continuous primal-dual constraint

sparsity:
    discrete global Top-K projection
```

---

# 6. 为什么第一轮设 3%

上一轮自然训练轨迹已经给出：

```text
~0.97% flips -> calibration ASR ~35%
~1.28% flips -> calibration ASR ~51%
~1.82% flips -> calibration ASR ~77%
~2.19% flips -> calibration ASR ~87.5%
~2.43% flips -> calibration ASR ~90%
~2.58% flips -> calibration ASR ~94%
~2.7% flips  -> calibration ASR ~96%
```

因此当前 evidence 更像：

\[
\boxed{
\text{capacity knee 在 2% 左右，而不是 1%}
}
\]

所以第一轮 3% 的目标是：

> **先验证 global residual 在 test 上有没有真正超过 QuRA 的能力。**

如果 3% 下已经：

```text
test ASR >= 95%
test CA  >= 92.5%
```

再做 3% → 2% → 1%。

不要反过来。

---

# 7. 这轮必须增加 test-during-training 诊断

每 500 steps 输出一次：

```text
diagnostic_test_ca
diagnostic_test_asr
```

但是：

\[
\boxed{
\text{test 指标绝对不能参与 best-step selection}
}
\]

也就是说：

```text
可以记录
不能用来：
    选 checkpoint
    early stop
    调 lr
    改 margin
    改 budget
```

这轮属于 debugging run。

目的只是回答：

> calibration 96% ASR 到 test 上究竟是多少？

正式论文稳定以后，应该改成：

```text
calibration train
+ held-out calibration validation
+ final test once
```

而不是一直看 test。

---

# 8. 结束后再做一个 post-hoc budget sweep

恢复 calibration-selected best alpha 后，不重新训练，只改变 hard support budget：

```text
0.5%
1.0%
1.5%
2.0%
2.5%
3.0%
```

分别评估：

```text
calibration ASR
calibration clean KL
test CA
test ASR
actual flip count
```

这可以一次性判断：

\[
\boxed{
\text{capacity knee 到底是不是约 2%}
}
\]

注意：

> 这个 sweep 只是 diagnostic。

因为 logits 是在 3% budget 下训练出来的。

正式论文的 Pareto curve 必须每个 budget 单独 retrain。

---

# 9. 文件修改总览

基于当前 GSRR V2。

修改：

```text
mqbench/fake_quantize/adaround_quantizer.py
mqbench/global_rounding_attack.py
ours/main/configs/cv_4_4_global_bd.yaml
ours/main/main.py
```

其它全部不动：

```text
dataset
trigger generation
MQBench observer calibration
setting/config.py
model
evaluation
```

---

# 10. `adaround_quantizer.py` patch

把 V2 中 global residual 相关部分按下面修改。

```python

# ===== PATCH A: AdaRoundFakeQuantize =====
# Add at top:
import math

# In __init__:
self.global_gate_threshold = float("inf")

# In init_global_attack(), after setting global_attack/global_temperature:
self.global_gate_threshold = float("inf")

# Keep alpha as a FLIP logit, all gates initially OFF:
alpha = torch.full_like(
    weight_tensor,
    fill_value=-float(gate_init),
)
self.alpha = Parameter(alpha)

def set_global_gate_threshold(self, threshold):
    self.global_gate_threshold = float(threshold)

def current_flip_soft(self):
    if not self.global_attack:
        raise RuntimeError("Global attack is not active.")
    return torch.sigmoid(
        self.alpha / max(self.global_temperature, 1e-6)
    )

def current_flip_hard(self):
    if not self.global_attack:
        raise RuntimeError("Global attack is not active.")

    if (
        math.isinf(self.global_gate_threshold)
        and self.global_gate_threshold > 0
    ):
        return torch.zeros_like(self.alpha)

    return (
        (self.alpha >= self.global_gate_threshold)
        & (self.alpha > 0)
        & self.global_attack_mask.bool()
    ).float()

def current_flip_ste(self):
    soft = self.current_flip_soft()
    hard = self.current_flip_hard()
    return hard.detach() - soft.detach() + soft

def current_hard_round(self):
    base = self.global_base_round
    mask = self.global_attack_mask.to(dtype=base.dtype)
    flip = self.current_flip_hard()

    # b = b0 XOR g = b0 + (1 - 2*b0)*g
    return (
        base
        + mask
        * (1.0 - 2.0 * base)
        * flip
    )

def get_global_hard_value(self, x):
    scale, zero_point = _reshape_qparams(
        x,
        self.scale.data,
        self.zero_point.data.long(),
        self.ch_axis,
    )

    x_floor = torch.floor(x / scale)
    rounding = self.current_hard_round().to(dtype=x.dtype)

    q = x_floor + rounding + zero_point
    q = torch.clamp(q, self.quant_min, self.quant_max)
    return (q - zero_point) * scale

```

注意 `residual_rounding_forward()` 也要同步改 hard gate：

```python
def residual_rounding_forward(
    x,
    scale,
    zero_point,
    quant_min,
    quant_max,
    ch_axis,
    flip_logit,
    candidate_mask,
    base_round,
    gate_threshold,
    temperature=1.0,
    force_base=False,
):
    scale, zero_point = _reshape_qparams(
        x,
        scale,
        zero_point,
        ch_axis,
    )

    x_floor = torch.floor(x / scale)
    base = base_round.to(dtype=x.dtype)
    mask = candidate_mask.to(dtype=x.dtype)

    if force_base:
        rounding = base

    else:
        temp = max(float(temperature), 1e-6)

        soft_gate = torch.sigmoid(
            flip_logit / temp
        )

        if (
            math.isinf(float(gate_threshold))
            and float(gate_threshold) > 0
        ):
            hard_gate = torch.zeros_like(
                soft_gate
            )

        else:
            hard_gate = (
                (flip_logit >= float(gate_threshold))
                & candidate_mask.bool()
                & (flip_logit > 0)
            ).to(dtype=x.dtype)

        gate_ste = (
            hard_gate.detach()
            - soft_gate.detach()
            + soft_gate
        )

        rounding = (
            base
            + mask
            * (1.0 - 2.0 * base)
            * gate_ste
        )

    q = (
        x_floor
        + rounding
        + zero_point
    )

    q = torch.clamp(
        q,
        quant_min,
        quant_max,
    )

    return (
        q - zero_point
    ) * scale
```

并在 `forward()` 调用时加入：

```python
self.global_gate_threshold,
```

即：

```python
x = residual_rounding_forward(
    x,
    self.scale.data,
    self.zero_point.data.long(),
    self.quant_min,
    self.quant_max,
    self.ch_axis,
    self.alpha,
    self.global_attack_mask,
    self.global_base_round,
    self.global_gate_threshold,
    self.global_temperature,
    force_base=self.global_force_base,
)
```

---

# 11. `global_rounding_attack.py`：新增 exact global projection

顶部增加：

```python
import math
```

加入：

```python

# ===== PATCH B: global exact-support projection =====
@torch.no_grad()
def _update_global_budget_threshold(
    module_records,
    valid_total,
    flip_budget,
):
    """
    Enforce a GLOBAL support cap.

    eligible = candidate AND alpha > 0

    If #positive <= K:
        threshold = 0
    else:
        threshold = K-th largest positive alpha

    All layers share the same threshold.
    """

    k = int(
        math.floor(
            float(valid_total)
            * float(flip_budget)
        )
    )
    k = max(k, 0)

    all_candidate_logits = []

    for rec in module_records:
        fq = rec["fq"]
        mask = fq.global_attack_mask.bool()

        all_candidate_logits.append(
            fq.alpha.detach()[mask]
        )

    all_candidate_logits = torch.cat(
        all_candidate_logits,
        dim=0,
    )

    positive = all_candidate_logits[
        all_candidate_logits > 0
    ]

    positive_count = int(positive.numel())

    if k <= 0 or positive_count == 0:
        threshold = float("inf")

    elif positive_count <= k:
        threshold = 0.0

    else:
        topk_vals = torch.topk(
            positive,
            k=k,
            largest=True,
            sorted=False,
        ).values

        threshold = max(
            float(topk_vals.min().item()),
            0.0,
        )

    for rec in module_records:
        rec["fq"].set_global_gate_threshold(
            threshold
        )

    return {
        "budget_ratio": float(flip_budget),
        "budget_k": int(k),
        "positive_candidates": int(positive_count),
        "threshold": (
            None
            if math.isinf(threshold)
            else float(threshold)
        ),
    }

```

---

# 12. 新增 diagnostic test evaluator

```python

# ===== PATCH C: diagnostic full-test evaluation =====
@torch.no_grad()
def _evaluate_cv_loader(
    model,
    loader,
    device,
):
    """
    Diagnostic only.

    DO NOT use this metric to choose best_step.
    """

    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = _extract_logits(
            model(images)
        )

        pred = logits.argmax(dim=1)

        correct += int(
            (pred == labels).sum().item()
        )
        total += int(labels.numel())

    return 100.0 * correct / max(total, 1)

```

---

# 13. 删除 V2 的 flip soft constraint

删除：

```python
dual_flip
dual_lr_flip
rho_flip

_global_flip_ratio_ste()

flip_violation

dual_flip * flip_violation

0.5 * rho_flip * relu(flip_violation)^2

dual_flip update
```

训练核心替换为：

```python

# ===== PATCH D: replace V2 sparsity training logic =====

# 1) REMOVE all of these from V2:
#
# dual_flip
# dual_lr_flip
# rho_flip
# _global_flip_ratio_ste()
# flip_violation
# dual_flip * flip_violation
# 0.5 * rho_flip * relu(flip_violation)^2
# dual_flip update
#
# The flip-count constraint is no longer optimized as a soft penalty.

primary_budget = float(config.flip_budget)

# Before EVERY training step:
projection = _update_global_budget_threshold(
    module_records,
    valid_total,
    primary_budget,
)

# Keep the original V2 clean teacher / attacked model forward:
teacher_logits = _benign_teacher_logits(
    model,
    module_records,
    clean_batch,
    device,
)

clean_logits = _forward_logits(
    model,
    clean_batch,
    device,
)

bd_logits = _forward_logits(
    model,
    bd_batch,
    device,
)

bd_loss, bd_margin = _attack_margin_loss(
    bd_logits,
    config.bd_target,
    kappa=kappa,
)

clean_kl = _clean_kl(
    clean_logits,
    teacher_logits,
    temperature=kl_temperature,
)

(
    clean_margin_loss,
    _,
    _,
) = _teacher_margin_preservation(
    clean_logits,
    teacher_logits,
    margin_cap=clean_margin_cap,
)

clean_cost = (
    clean_kl
    + clean_margin_weight
    * clean_margin_loss
)

clean_violation = (
    clean_cost
    - clean_epsilon
)

# V2.1 only keeps the CONTINUOUS clean constraint:
loss = (
    bd_loss
    + dual_clean
    * clean_violation
    + 0.5
    * rho_clean
    * F.relu(clean_violation).pow(2)
)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    alpha_params,
    grad_clip,
)

optimizer.step()
scheduler.step()

# Keep only clean dual ascent:
with torch.no_grad():
    dual_clean = max(
        0.0,
        dual_clean
        + dual_lr_clean
        * float(
            clean_violation.detach().item()
        ),
    )

# IMPORTANT:
# after optimizer.step(), recompute global projection
# BEFORE evaluation, because alpha changed.
projection = _update_global_budget_threshold(
    module_records,
    valid_total,
    primary_budget,
)

```

---

# 14. best checkpoint 不能再因为 1% budget 回到 step 1

best checkpoint 只使用：

```text
calibration clean KL
calibration teacher agreement
calibration ASR
calibration target margin
```

完全不看 test。

使用：

```python

# ===== PATCH E: calibration-only best checkpoint selection =====

best_max_clean_kl = float(
    getattr(
        config,
        "best_max_clean_kl",
        0.03,
    )
)

best_min_clean_agreement = float(
    getattr(
        config,
        "best_min_clean_agreement",
        99.0,
    )
)

# At each eval step:
metrics = _calibration_metrics(
    model,
    module_records,
    cali_data,
    cali_data_bd,
    config.bd_target,
    device,
    kl_temperature,
)

rounding = _rounding_stats(
    module_records,
    valid_total,
)

# Exact-support sanity check.
budget_k = int(projection["budget_k"])

if rounding["flipped"] > budget_k + 8:
    raise RuntimeError(
        "Hard support exceeded K. "
        "Likely a threshold tie; "
        "switch to explicit per-index tie breaking."
    )

feasible_clean = (
    metrics["clean_kl"]
    <= best_max_clean_kl
    and metrics["clean_teacher_agreement"]
    >= best_min_clean_agreement
)

# TEST METRICS MUST NOT ENTER THIS SCORE.
if feasible_clean:
    score = (
        metrics["asr"]
        + 0.10
        * metrics["target_margin"]
    )

    if score > best_score:
        best_score = float(score)
        best_step = int(step)
        best_metrics = dict(metrics)
        best_rounding = dict(rounding)
        best_alpha = _capture_alpha_state(
            module_records
        )

```

因为 exact Top-K 已经从 forward 机制上控制 cardinality，所以 best state 不需要再次用：

```text
flip_ratio <= 1%
```

把自己杀掉。

---

# 15. 训练过程中增加 test generalization 诊断

`global_rounding_attack()` signature 改成：

```python
def global_rounding_attack(
    model,
    cali_data,
    cali_data_bd,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
```

然后加入：

```python

# ===== PATCH F: test-during-training DIAGNOSTIC only =====

diagnostic_test_eval = bool(
    getattr(
        config,
        "diagnostic_test_eval",
        True,
    )
)

diagnostic_test_interval = int(
    getattr(
        config,
        "diagnostic_test_interval",
        500,
    )
)

# Inside the existing eval block, AFTER calibration metrics:
if (
    diagnostic_test_eval
    and test_loader is not None
    and test_loader_bd is not None
    and (
        step % diagnostic_test_interval == 0
        or step == steps
    )
):
    row["diagnostic_test_ca"] = (
        _evaluate_cv_loader(
            model,
            test_loader,
            device,
        )
    )

    row["diagnostic_test_asr"] = (
        _evaluate_cv_loader(
            model,
            test_loader_bd,
            device,
        )
    )

# IMPORTANT:
# diagnostic_test_ca / diagnostic_test_asr
# are only written to logs.
# They are NEVER used by feasible_clean, score, best_step,
# early stopping, LR scheduling, or hyperparameter adaptation.

```

日志中应该出现：

```text
step 500:
    calibration ASR
    diagnostic_test_asr
    diagnostic_test_ca

step 1000:
    ...

step 1500:
    ...
```

这就是本轮最重要的数据。

---

# 16. restore best state 后做 budget sweep

加入：

```python

# ===== PATCH G: post-hoc budget sweep after restoring best alpha =====

# First restore the best CALIBRATION-selected state:
_restore_alpha_state(
    module_records,
    best_alpha,
)

# Then evaluate several support budgets on the SAME trained logits.
# This is diagnostic only; final paper curves must retrain each budget.

budget_sweep = []

sweep_values = list(
    getattr(
        config,
        "diagnostic_budget_sweep",
        [
            0.005,
            0.010,
            0.015,
            0.020,
            0.025,
            0.030,
        ],
    )
)

for ratio in sweep_values:
    sweep_projection = (
        _update_global_budget_threshold(
            module_records,
            valid_total,
            float(ratio),
        )
    )

    sweep_cal = _calibration_metrics(
        model,
        module_records,
        cali_data,
        cali_data_bd,
        config.bd_target,
        device,
        kl_temperature,
    )

    sweep_rounding = _rounding_stats(
        module_records,
        valid_total,
    )

    sweep_row = {
        "budget_ratio": float(ratio),
        "projection": sweep_projection,
        "calibration": sweep_cal,
        "rounding": {
            "flipped": int(
                sweep_rounding["flipped"]
            ),
            "flip_ratio_total_valid": float(
                sweep_rounding[
                    "flip_ratio_total_valid"
                ]
            ),
        },
    }

    if (
        test_loader is not None
        and test_loader_bd is not None
    ):
        sweep_row["test"] = {
            "ca": _evaluate_cv_loader(
                model,
                test_loader,
                device,
            ),
            "asr": _evaluate_cv_loader(
                model,
                test_loader_bd,
                device,
            ),
        }

    budget_sweep.append(sweep_row)

    print(
        "[V2.1 budget sweep]",
        sweep_row,
    )

# IMPORTANT:
# Restore primary 3% support BEFORE materializing final weights.
_update_global_budget_threshold(
    module_records,
    valid_total,
    primary_budget,
)

```

JSON 中保存：

```python
result["diagnostic_budget_sweep"] = budget_sweep
```

---

# 17. V2.1 配置文件

`ours/main/configs/cv_4_4_global_bd.yaml` 改成：

```yaml
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MSEObserver
        a_observer: EMAMSEObserver
        w_fakequantize: AdaRoundFakeQuantize
        a_fakequantize: FixedFakeQuantize

        w_qscheme:
            bit: 4
            symmetry: False
            per_channel: True
            pot_scale: False

        a_qscheme:
            bit: 4
            symmetry: False
            per_channel: False
            pot_scale: False

quantize:
    quantize_type: global_rounding_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # Candidate pool
        candidate_ratio: 0.10
        gate_init: 4.0

        # Exact hard support cap
        flip_budget: 0.03

        # End-to-end optimization
        steps: 5000
        lr: 0.02
        margin: 5.0
        temperature: 1.0

        # Clean constraint
        kl_temperature: 1.0
        clean_margin_weight: 0.5
        clean_margin_cap: 2.0
        clean_epsilon: 0.015

        dual_clean_init: 1.0
        dual_lr_clean: 0.05
        rho_clean: 20.0

        # Multi-view clean
        aug_pad: 4
        hflip_prob: 0.5

        grad_clip: 5.0

        # Calibration logging
        eval_interval: 250

        # Diagnostic test logging only
        diagnostic_test_eval: true
        diagnostic_test_interval: 500

        # Calibration-only checkpoint selection
        best_max_clean_kl: 0.03
        best_min_clean_agreement: 99.0

        # Diagnostic post-hoc projection
        diagnostic_budget_sweep:
            - 0.005
            - 0.010
            - 0.015
            - 0.020
            - 0.025
            - 0.030

dataset:
    batch_size: 32
    num_workers: 16
    pattern: stage2

process:
    seed: 1005
```

注意：

```text
dual_flip_init
dual_lr_flip
rho_flip
```

全部删除。

---

# 18. `main.py` 修改

调用：

```python
(
    model,
    attack_info,
) = global_rounding_attack(
    model=model,
    cali_data=cali_data,
    cali_data_bd=cali_data_bd,
    config=attack_cfg,
    output_json=diag_json,

    # V2.1 diagnostic only
    test_loader=test_loader,
    test_loader_bd=test_loader_bd,
)
```

输出文件名改成：

```python
diag_json = os.path.join(
    output_dir,
    (
        "global_sparse_residual_v21_"
        f"{args.model}_"
        f"{args.dataset}.json"
    ),
)
```

checkpoint 改成：

```python
checkpoint_path = os.path.join(
    directory_path,
    "model",
    (
        f"{args.model}+"
        f"{args.dataset}."
        "global_sparse_residual_v21_"
        f"t{attack_cfg.bd_target}.pth"
    ),
)
```

---

# 19. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_global_bd.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/global_sparse_residual_v21_cifar10.log
```

---

# 20. Step-0 sanity check

必须看到：

```text
clean_kl ≈ 0
flipped = 0
```

以及：

```text
positive_candidates = 0
threshold = None / inf
```

否则直接 STOP。

---

# 21. 3% exact budget sanity check

当前合法 rounding positions 大约：

\[
N_{\mathrm{valid}}\approx10.995M.
\]

所以：

\[
K
\approx
0.03\times10.995M
\approx329K.
\]

训练过程中：

```text
flipped
```

不应该明显超过约 329K。

代码允许极少量 threshold tie tolerance。

如果超过：

```text
K + 8
```

直接报错。

那就说明 shared-threshold 在完全相同 logits 上出现 ties，需要改成 explicit global index mask。

---

# 22. 这轮最重要的表

跑完后先手工整理：

| Step | Cali ASR | Test ASR | Test CA | Flip % |
|---:|---:|---:|---:|---:|
| 500 | | | | |
| 1000 | | | | |
| 1500 | | | | |
| 2000 | | | | |
| 2500 | | | | |
| 3000 | | | | |
| 3500 | | | | |
| 4000 | | | | |
| 4500 | | | | |
| 5000 | | | | |

这张表直接回答：

\[
\boxed{
\text{calibration attack 是否能泛化到 test}
}
\]

---

# 23. 结果怎么判断

## A. 最理想

```text
test ASR >= 95%
test CA  >= 92.5%
```

Strong GO。

说明：

```text
global residual attack 成立
V2 的最终失败主要是 checkpoint / sparsity constraint 问题
```

---

## B. Test ASR > 90%，CA > 92%

GO。

已经足以继续优化并和 QuRA 论文口径正面对打。

---

## C. Calibration ASR 95%，Test ASR 只有 50%–70%

说明真正的病灶是：

\[
\boxed{
\text{trigger attack generalization}
}
\]

不是 sparsity。

下一步不要继续改 budget。

---

## D. Test ASR 很高，但 CA 只有 85%–90%

说明：

```text
attack 泛化成功
clean preservation 泛化失败
```

下一步只改 clean behavior preservation。

---

## E. Exact 3% support 后 calibration ASR 自己都上不去

说明 V2 之前的高 ASR 依赖：

```text
support 不受约束地持续扩张 / 替换
```

需要研究 Top-K optimization dynamics，而不是立即推翻 global residual。

---

# 24. Budget sweep 怎么看

如果得到类似：

| Budget | Test ASR | Test CA |
|---:|---:|---:|
| 0.5% | 25 | 93.2 |
| 1.0% | 50 | 93.1 |
| 1.5% | 72 | 92.9 |
| 2.0% | 88 | 92.7 |
| 2.5% | 94 | 92.5 |
| 3.0% | 96 | 92.3 |

那么说明：

\[
\boxed{
\text{malicious capacity 存在明确的 support-budget knee}
}
\]

而且 1% 并不是一个合理的第一版 budget。

如果反而：

```text
1% post-hoc 就有 90%+ ASR
```

说明：

> V2 的问题不是 1% capacity 不够，而是它没有学会把最有效的 1% residual 集中出来。

那下一步研究：

```text
support concentration / ranking
```

反而会很有价值。

---

# 25. 这轮结束后给回来的东西

请保留：

```text
1. 完整 log

2. JSON:
   global_sparse_residual_v21_resnet18_cifar10.json

3. Final Summary

4. training_history 中：
   diagnostic_test_ca
   diagnostic_test_asr

5. best_step
   best_metrics
   best_test

6. diagnostic_budget_sweep
```

下一轮我们不再猜。

这一次跑完以后应该可以明确判断：

```text
① checkpoint/sparsity bug
② clean generalization
③ attack generalization
④ support budget capacity
```

到底哪个才是当前主矛盾。
