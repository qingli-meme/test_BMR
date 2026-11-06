# V11 / TCRR — Structured Rounding Field + Target-Compatible Residual Rectification

## 基于当前 commit `2bb34d5`

当前实验链：

```text
V6  Decision-Core Routing        90.67 / 99.71
V7  Residual-Preserving          91.85 / 96.27
V9  Decision-Depth Transport     92.04 / 92.70
V10 Classwise Max-Depth          91.79 / 97.98
```

V10 已经证明：继续优化 classifier-side transport direction 可以把 V7 的 96.27% 提到约 98%，但仍无法稳定达到 99%+。

同时 V10 的理想 classifier geometry 已经很强：

- mean worst-target-depth gain ≈ 1.54
- 每个 source 的理想 endpoint calibration target rate ≈ 100%
- 但真实最佳 checkpoint 的 trigger target depth minimum 仍然为负

因此下一步不再继续修改 \(d_y\)。

V11 改变两个更根本的假设：

1. **Residual 不应该“全删”或“全留”，而应该由 classifier 决定哪些部分对 target 有利。**
2. **Trigger 不再训练 response difference，而重新使用 V6 风格的 absolute endpoint matching。**

核心：

\[
\boxed{
u_\phi(Tx)
\rightarrow
c_t
+
A_t^{-1}
\left[
A_t
\left(
u_0(x)-c_y
\right)
\right]_+
}
\]

---

# 1. V6 与 V7 的 residual 二元假设

写 benign decision code：

\[
u_0(x)=c_y+\rho(x).
\]

V6：

\[
u_\phi(Tx)\rightarrow c_t.
\]

相当于：

\[
\rho\rightarrow0.
\]

即：

\[
\boxed{\text{所有 residual 都删除}}
\]

优点：

```text
ASR = 99.71%
```

问题：

```text
CA = 90.67%
```

---

V7：

\[
u_\phi(Tx)\rightarrow c_t+\rho.
\]

相当于：

\[
\boxed{\text{所有 residual 都保留}}
\]

结果：

```text
CA = 91.85%
ASR = 96.27%
```

所以真正缺失的问题不是：

```text
删多少 residual？
```

而是：

\[
\boxed{
\text{residual 中哪些部分对 target 分类有利，哪些有害？}
}
\]

---

# 2. Frozen classifier 给出 exact target-margin coordinates

decision-space classifier：

\[
z_k(u)=w_k^\top u+b_k.
\]

对于 target \(t\) 和 competitor \(j\neq t\)，定义标准化 margin：

\[
q_j(u)
=
\frac{
z_t(u)-z_j(u)
}{
\|w_t-w_j\|_2
}.
\]

写成：

\[
q_j(u)
=
a_j^\top u+\beta_j.
\]

把所有 \(C-1\) 个 target-vs-other margin stack：

\[
\boxed{
q_t(u)=A_tu+\beta_t.
}
\]

当前 CIFAR-10：

```text
decision rank = 9
C - 1         = 9
```

因此：

\[
A_t\in\mathbb R^{9\times9}.
\]

target-vs-all difference vectors span 完整 classifier decision space，所以 \(A_t\) 应当满秩。

正式代码会检查：

```text
shape = 9 x 9
rank = 9
smallest singular value > 1e-10
condition number
```

并使用：

```python
torch.linalg.solve
```

不显式求逆。

---

# 3. Residual 在 margin coordinates 中有直接语义

source residual：

\[
\rho
=
u_0(x)-c_y.
\]

它对 target margins 的影响：

\[
\boxed{
g
=
A_t\rho.
}
\]

其中：

\[
g_j>0
\]

表示：

> 该 residual 让 target 相对 competitor \(j\) 更有优势。

而：

\[
g_j<0
\]

表示：

> 该 residual 把样本往 target-vs-\(j\) boundary 外推。

这不是 PCA/SVD component，也不是 Euclidean feature axis。

它是：

\[
\boxed{
\text{classifier-defined exact margin effect}
}
\]

---

# 4. Target-Compatible Residual Rectification

只保留帮助 target 的 margin coordinates：

\[
\boxed{
g^+
=
[g]_+
}
\]

即：

\[
g_j^+
=
\max(g_j,0).
\]

然后回到 decision space：

\[
\boxed{
\rho^+
=
A_t^{-1}g^+.
}
\]

最终 desired trigger endpoint：

\[
\boxed{
u^\star(x)
=
c_t+\rho^+.
}
\]

---

# 5. 最关键的 classifier guarantee

target core：

\[
c_t.
\]

其 target margins：

\[
q_t(c_t)
=
A_tc_t+\beta_t.
\]

V11 endpoint：

\[
u^\star
=
c_t+\rho^+.
\]

因此：

\[
\begin{aligned}
q_t(u^\star)
&=
A_t(c_t+\rho^+)+\beta_t \\
&=
q_t(c_t)+A_t\rho^+ \\
&=
q_t(c_t)+[A_t\rho]_+.
\end{aligned}
\]

所以逐 coordinate：

\[
\boxed{
q_t(u^\star)
\ge
q_t(c_t).
}
\]

因此：

\[
\boxed{
\min_j q_j(u^\star)
\ge
\min_j q_j(c_t)>0.
}
\]

也就是说：

> **理想 V11 endpoint 对每一个 target-vs-other classifier boundary 都不会比 V6 的 target core 更浅。**

这给了 V6 级别的 classifier-side attack safety。

---

# 6. V11 精确连接 V6 与 V7

如果 residual 对所有 target margins 都有害：

\[
A_t\rho\le0,
\]

则：

\[
[A_t\rho]_+=0.
\]

于是：

\[
\rho^+=0
\]

并且：

\[
\boxed{
u^\star=c_t
}
\]

正好退化为 V6。

---

如果 residual 对所有 target margins 都有利：

\[
A_t\rho\ge0,
\]

则：

\[
[A_t\rho]_+
=
A_t\rho.
\]

由于 \(A_t\) 可逆：

\[
\rho^+=\rho.
\]

于是：

\[
\boxed{
u^\star=c_t+\rho
}
\]

正好退化为 V7。

所以 V11 不是人为插值：

\[
\lambda\cdot V6+(1-\lambda)\cdot V7.
\]

而是 classifier 自动逐 margin coordinate 决定：

```text
保留 / 删除
```

没有新增 λ。

---

# 7. V11 不是 Euclidean projection

需要严格说明。

\[
\rho^+
=
A_t^{-1}[A_t\rho]_+
\]

并不一般等价于 decision-space Euclidean projection。

因此文档和代码都只称它：

\[
\boxed{
\textbf{Target-Compatible Residual Rectification}
}
\]

它做的是：

> 在 classifier target-margin coordinates 中，对 residual effect 做精确 rectification。

这比错误声称“projection”更准确。

---

# 8. 为什么 absolute endpoint matching 很重要

V7/V10 训练：

\[
\Delta_\phi(x)
=
u_\phi(Tx)-u_\phi(x)
\rightarrow d(x).
\]

但真正 trigger endpoint error：

\[
\begin{aligned}
e_{\text{endpoint}}
&=
u_\phi(Tx)
-
[u_0(x)+d(x)]\\
&=
[u_\phi(x)-u_0(x)]
+
[\Delta_\phi(x)-d(x)].
\end{aligned}
\]

因此：

\[
\boxed{
e_{\text{endpoint}}
=
e_{\text{clean}}
+
e_{\text{response}}.
}
\]

这意味着：

> 即使 response matching 看起来不错，clean branch deviation 仍会重新进入 trigger endpoint。

V10 最佳模型仍出现：

```text
ideal geometry 深在 target region
real trigger target_depth_min < 0
```

这和该误差组合一致。

---

V11 直接训练：

\[
\boxed{
u_\phi(Tx)
\rightarrow
u^\star(x)
}
\]

因此实际优化的就是：

\[
\boxed{
e_{\text{trigger}}
=
u_\phi(Tx)-u^\star(x).
}
\]

不再经过：

```text
clean error + response error
```

的组合。

这重新获得 V6 absolute attractor 的训练稳定性。

---

# 9. V11 endpoint 的另一种理解

V11 可以写成：

\[
u^\star
=
c_t
+
\rho_{\text{safe}}.
\]

所以它不是：

```text
9-way class translation controller
```

而是：

\[
\boxed{
\text{一个共享 target core}
+
\text{输入自身已经携带的 target-compatible residual}
}
\]

攻击的 universal 部分仍然是：

\[
c_t.
\]

sample-specific 的部分不是需要 field “新生成”的 class displacement，而是从 benign source state 中选择性保留的安全 residual。

这比 V7/V10 的：

```text
source class y -> fixed d_y
```

更接近 V6 的 universal target attractor。

---

# 10. 训练目标

clean：

\[
u_\phi(x)
\rightarrow
u_0(x).
\]

trigger：

\[
u_\phi(Tx)
\rightarrow
u^\star(x).
\]

统一 squared-distance：

\[
L
=
\operatorname{mean}
\left(
\|u_\phi(x)-u_0(x)\|_2^2,
\|u_\phi(Tx)-u^\star(x)\|_2^2
\right)
/r.
\]

没有：

```text
CE
KL
margin coefficient
attack weight
clean weight
sparsity penalty
QP
SVD direction compression
max-depth solver
new field rank
classifier fine-tuning
```

Structured Rounding Field 完全不变。

---

# 11. Dynamic augmentation

与 V6-V10 保持一致。

每个训练 step：

1. 对 clean batch 做 dynamic crop / flip；
2. benign PTQ teacher 得到该 view 的 pseudo-class \(y\) 和 code \(u_0(x)\)；
3. 动态计算：

\[
\rho=u_0(x)-c_y;
\]

4. 动态计算：

\[
\rho^+
=
A_t^{-1}[A_t\rho]_+;
\]

5. trigger paste 在 augmentation 后；
6. attacked trigger code 直接拟合：

\[
c_t+\rho^+.
\]

因此 desired endpoint 随当前 augmented view 自适应，但 classifier rule 不变。

---

# 12. 正式训练前 geometry diagnostics

会打印：

```text
[V11-TCRR] target-margin geometry
```

包括：

```text
rank
condition_number
singular_values
```

然后打印：

```text
[V11-TCRR] target-compatible geometry
```

包括：

```text
ideal_endpoint_target_rate

target_core_depth
ideal_endpoint_depth_mean
ideal_endpoint_depth_p10
ideal_endpoint_depth_min

compatible_coordinate_fraction
compatible_margin_energy_ratio
safe_residual_energy_ratio

mean_v6_displacement_d2
mean_v7_displacement_d2
mean_v11_displacement_d2

v11_to_v6_displacement_energy_ratio
v11_to_v7_displacement_energy_ratio
```

以及逐 source class 统计。

---

# 13. 三个最重要的 geometry 量

## 13.1 `compatible_coordinate_fraction`

表示 residual 的 9 个 target-margin effects 中，有多少比例本来就在帮助 target。

如果大约：

```text
40%–60%
```

说明 V6 确实删掉了大量其实可以保留的信息。

---

## 13.2 `safe_residual_energy_ratio`

表示经过 classifier rectification 后，decision-space residual energy 实际保留多少。

注意：

因为 \(A_t\) 不是正交矩阵，这个 ratio 理论上不保证小于 1。

所以它只是诊断量，不作为约束或方法定义。

---

## 13.3 V11 displacement vs V6 / V7

如果：

```text
V11 displacement energy < V6
```

同时 ideal endpoint depth：

```text
>= target core depth
```

那么这就是最理想的几何信号：

> 比 V6 少做 feature movement，却保留 V6 级 classifier safety。

---

# 14. 自审结果

正式生成代码前做了以下检查。

### 14.1 Margin rectification identity

随机 100 组可逆 9×9 margin maps：

\[
A\rho^+
=
[A\rho]_+
\]

PASS。

---

### 14.2 Core-depth guarantee

随机测试：

\[
q(u^\star)
=
q(c_t)+[A\rho]_+
\ge q(c_t)
\]

PASS。

---

### 14.3 V6 reduction

当：

\[
A\rho<0
\]

所有 coordinates 都有害时：

\[
\rho^+=0.
\]

PASS。

---

### 14.4 V7 reduction

当：

\[
A\rho>0
\]

所有 coordinates 都有利时：

\[
\rho^+=\rho.
\]

PASS。

---

### 14.5 Numerical implementation

正式代码使用：

```python
torch.linalg.solve
```

且 solve 使用 float64 的 9×9 系统，再转换回 feature dtype。

不显式求逆。

---

### 14.6 Syntax

```text
V11 module AST: PASS
main.py branch AST: PASS
```

---

# 15. 文件操作

## 新增

```text
mqbench/structured_target_compatible_attack.py
ours/main/configs/cv_4_4_structured_target_compatible.yaml
```

## 修改

```text
ours/main/main.py
```

## 不修改

```text
mqbench/fake_quantize/structured_rounding_quantizer.py

mqbench/structured_decision_core_attack.py
mqbench/structured_decision_translation_attack.py
mqbench/structured_decision_depth_attack.py
mqbench/structured_max_depth_attack.py
```

---

# 16. 新增完整代码
## `mqbench/structured_target_compatible_attack.py`

```python
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.utils.state import enable_quantization

from mqbench.structured_decision_core_attack import (
    _to_device,
    _effective_weight,
    _teacher_forward,
    extract_trigger_spec,
    _augment_clean_tensor,
    _apply_trigger,
    _extract_quantized_classifier,
    _build_decision_geometry,
    decision_code,
    _structured_stats,
    _evaluate_cv_loader,
    _capture_field_state,
    _restore_field_state,
    _finalize_structured_rounding,
)

from mqbench.structured_decision_translation_attack import (
    V7FeatureTapper,
    _attacked_forward,
    _collect_attack_records,
    _verify_classifier_geometry,
    build_class_decision_cores,
)


# ============================================================
# Exact target-margin coordinate system
# ============================================================

@torch.no_grad()
def build_target_margin_geometry(
    geometry,
    target,
):
    """
    Frozen classifier in decision coordinates:
        z_k(u) = w_k^T u + b_k

    For target t and each competitor j != t, define the exact
    normalized target-vs-j margin coordinate

        q_j(u)
          = [z_t(u) - z_j(u)]
            / ||w_t - w_j||_2

          = a_j^T u + beta_j.

    Stack the C-1 row vectors a_j into

        A_t in R^{(C-1) x r}.

    In the current CIFAR-10 setting:
        r = C - 1 = 9,

    so A_t is square. Since the classifier decision space has
    full rank C-1, the target-vs-all differences span the same
    decision space and A_t must be nonsingular.

    This gives an exact coordinate system whose coordinates have
    direct classifier meaning:
        positive residual coordinate -> helps target vs competitor;
        negative residual coordinate -> hurts target vs competitor.
    """

    target = int(
        target
    )

    weight = geometry[
        "decision_weight"
    ].float()

    bias = geometry[
        "bias"
    ].float()

    num_classes = int(
        weight.shape[
            0
        ]
    )

    rank = int(
        geometry[
            "rank"
        ]
    )

    competitors = [
        cls
        for cls in range(
            num_classes
        )
        if cls != target
    ]

    diff = (
        weight[
            target
        ]
        .unsqueeze(
            0
        )
        - weight[
            competitors
        ]
    )

    norms = (
        diff.norm(
            p=2,
            dim=1,
        )
        .clamp_min(
            1.0e-12
        )
    )

    margin_matrix = (
        diff
        / norms.unsqueeze(
            1
        )
    )

    margin_offset = (
        (
            bias[
                target
            ]
            - bias[
                competitors
            ]
        )
        / norms
    )

    if (
        margin_matrix.shape[
            0
        ]
        != rank
        or margin_matrix.shape[
            1
        ]
        != rank
    ):
        raise RuntimeError(
            (
                "V11 requires a square target-margin map: "
                "shape={}, decision_rank={}"
            ).format(
                tuple(
                    margin_matrix.shape
                ),
                rank,
            )
        )

    singular_values = torch.linalg.svdvals(
        margin_matrix.double()
    )

    smallest = float(
        singular_values.min()
        .item()
    )

    largest = float(
        singular_values.max()
        .item()
    )

    if smallest <= 1.0e-10:
        raise RuntimeError(
            "Target-margin map is numerically singular."
        )

    matrix_rank = int(
        torch.linalg.matrix_rank(
            margin_matrix.double()
        ).item()
    )

    if matrix_rank != rank:
        raise RuntimeError(
            (
                "Target-margin map rank mismatch: "
                "{} != {}"
            ).format(
                matrix_rank,
                rank,
            )
        )

    condition_number = (
        largest
        / smallest
    )

    print(
        "[V11-TCRR] target-margin geometry:",
        {
            "target": (
                target
            ),
            "competitors": (
                competitors
            ),
            "shape": tuple(
                margin_matrix.shape
            ),
            "rank": (
                matrix_rank
            ),
            "condition_number": (
                condition_number
            ),
            "singular_values": (
                singular_values
                .detach()
                .cpu()
                .tolist()
            ),
        },
    )

    return {
        "target": (
            target
        ),
        "competitors": (
            competitors
        ),
        "matrix": (
            margin_matrix.detach()
        ),
        "offset": (
            margin_offset.detach()
        ),
        "rank": (
            matrix_rank
        ),
        "condition_number": float(
            condition_number
        ),
        "singular_values": (
            singular_values
            .detach()
            .cpu()
            .tolist()
        ),
    }


def target_margin_coordinates(
    code,
    margin_geometry,
):
    matrix = margin_geometry[
        "matrix"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    offset = margin_geometry[
        "offset"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    return (
        code
        @ matrix.transpose(
            0,
            1
        )
        + offset.unsqueeze(
            0
        )
    )


# ============================================================
# Target-Compatible Residual Rectification
# ============================================================

@torch.no_grad()
def target_compatible_endpoint(
    teacher_code,
    source_labels,
    class_cores,
    margin_geometry,
    target,
):
    """
    Teacher source representation:
        u = c_y + rho.

    V6:
        endpoint = c_t
        -> removes all rho.

    V7:
        endpoint = c_t + rho
        -> preserves all rho.

    V11:
        express rho in exact target-margin coordinates

            g = A_t rho.

        Keep only target-compatible coordinates:

            g+ = ReLU(g).

        Reconstruct the unique decision-space residual:

            rho+ = A_t^{-1} g+.

        Desired trigger endpoint:

            u* = c_t + rho+.

    Exact classifier guarantee:

        q_t(u*)
          = q_t(c_t) + A_t rho+
          = q_t(c_t) + ReLU(A_t rho)
          >= q_t(c_t)

    coordinate-wise.

    Therefore every ideal V11 endpoint has every target-vs-other
    normalized margin at least as large as the target core.

    Important:
        This is NOT claimed to be Euclidean projection.
        It is exact rectification in classifier margin coordinates.
    """

    target = int(
        target
    )

    code = teacher_code.float()

    labels = source_labels.long()

    cores = class_cores.to(
        device=code.device,
        dtype=code.dtype,
    )

    matrix = margin_geometry[
        "matrix"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    offset = margin_geometry[
        "offset"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    source_core = (
        cores[
            labels
        ]
    )

    target_core = (
        cores[
            target
        ]
        .unsqueeze(
            0
        )
    )

    residual = (
        code
        - source_core
    )

    residual_margin = (
        residual
        @ matrix.transpose(
            0,
            1
        )
    )

    compatible_margin = (
        residual_margin.clamp_min(
            0.0
        )
    )

    # Solve in float64 for the tiny 9 x 9 system.
    safe_residual = (
        torch.linalg.solve(
            matrix.double(),
            compatible_margin.double()
            .transpose(
                0,
                1
            ),
        )
        .transpose(
            0,
            1
        )
        .to(
            dtype=code.dtype
        )
    )

    endpoint = (
        target_core
        + safe_residual
    )

    core_margin = (
        target_core
        @ matrix.transpose(
            0,
            1
        )
        + offset.unsqueeze(
            0
        )
    )

    endpoint_margin = (
        endpoint
        @ matrix.transpose(
            0,
            1
        )
        + offset.unsqueeze(
            0
        )
    )

    expected_margin = (
        core_margin
        + compatible_margin
    )

    identity_error = float(
        (
            endpoint_margin
            - expected_margin
        )
        .abs()
        .max()
        .item()
    )

    if identity_error > 5.0e-5:
        raise RuntimeError(
            (
                "Target-compatible margin identity failed: "
                "{}"
            ).format(
                identity_error
            )
        )

    core_depth = (
        core_margin.min(
            dim=1
        ).values
    )

    endpoint_depth = (
        endpoint_margin.min(
            dim=1
        ).values
    )

    if (
        core_depth
        <= 0
    ).any():
        raise RuntimeError(
            "Target core is not strictly inside target region."
        )

    if (
        endpoint_margin
        + 5.0e-5
        < core_margin
    ).any():
        raise RuntimeError(
            "V11 endpoint margin fell below target-core margin."
        )

    if (
        endpoint_depth
        + 5.0e-5
        < core_depth
    ).any():
        raise RuntimeError(
            "V11 endpoint depth fell below target-core depth."
        )

    return {
        "endpoint": (
            endpoint.detach()
        ),
        "residual": (
            residual.detach()
        ),
        "residual_margin": (
            residual_margin.detach()
        ),
        "compatible_margin": (
            compatible_margin.detach()
        ),
        "safe_residual": (
            safe_residual.detach()
        ),
        "core_margin": (
            core_margin.detach()
        ),
        "endpoint_margin": (
            endpoint_margin.detach()
        ),
        "core_depth": (
            core_depth.detach()
        ),
        "endpoint_depth": (
            endpoint_depth.detach()
        ),
        "identity_error": float(
            identity_error
        ),
    }


# ============================================================
# Static geometry summary
# ============================================================

@torch.no_grad()
def build_target_compatible_summary(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    class_cores,
    margin_geometry,
    target,
):
    target = int(
        target
    )

    all_residual = []
    all_safe = []
    all_margin = []
    all_compatible_margin = []
    all_endpoint_depth = []
    all_source_labels = []

    all_v6_d2 = []
    all_v7_d2 = []
    all_v11_d2 = []

    total = 0
    ideal_hits = 0

    per_source_accum = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        per_source_accum[
            cls
        ] = {
            "count": 0,
            "compatible_coordinate_sum": 0.0,
            "margin_energy_total": 0.0,
            "compatible_margin_energy": 0.0,
            "residual_energy": 0.0,
            "safe_residual_energy": 0.0,
            "v6_d2": 0.0,
            "v7_d2": 0.0,
            "v11_d2": 0.0,
            "endpoint_depth_sum": 0.0,
            "endpoint_depth_min": float(
                "inf"
            ),
        }

    cores = class_cores.to(
        device=device
    )

    target_core = (
        cores[
            target
        ]
        .unsqueeze(
            0
        )
    )

    for batch in cali_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            teacher_logits,
            teacher_feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        pseudo = teacher_logits.argmax(
            dim=1
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        labels = (
            pseudo[
                source_mask
            ]
        )

        code = decision_code(
            teacher_feature[
                source_mask
            ],
            geometry[
                "basis"
            ],
        )

        spec = target_compatible_endpoint(
            teacher_code=code,
            source_labels=labels,
            class_cores=class_cores,
            margin_geometry=margin_geometry,
            target=target,
        )

        endpoint = spec[
            "endpoint"
        ]

        endpoint_margin = spec[
            "endpoint_margin"
        ]

        hit = (
            endpoint_margin.min(
                dim=1
            ).values
            > 0
        )

        total += int(
            hit.numel()
        )

        ideal_hits += int(
            hit.sum()
            .item()
        )

        source_core = (
            cores[
                labels
            ]
        )

        v6_delta = (
            target_core
            - code
        )

        v7_delta = (
            target_core
            - source_core
        )

        v11_delta = (
            endpoint
            - code
        )

        v6_d2 = (
            v6_delta.pow(
                2
            ).sum(
                dim=1
            )
        )

        v7_d2 = (
            v7_delta.pow(
                2
            ).sum(
                dim=1
            )
        )

        v11_d2 = (
            v11_delta.pow(
                2
            ).sum(
                dim=1
            )
        )

        all_residual.append(
            spec[
                "residual"
            ].cpu()
        )

        all_safe.append(
            spec[
                "safe_residual"
            ].cpu()
        )

        all_margin.append(
            spec[
                "residual_margin"
            ].cpu()
        )

        all_compatible_margin.append(
            spec[
                "compatible_margin"
            ].cpu()
        )

        all_endpoint_depth.append(
            spec[
                "endpoint_depth"
            ].cpu()
        )

        all_source_labels.append(
            labels.cpu()
        )

        all_v6_d2.append(
            v6_d2.cpu()
        )

        all_v7_d2.append(
            v7_d2.cpu()
        )

        all_v11_d2.append(
            v11_d2.cpu()
        )

        for cls in per_source_accum:
            cls_mask = (
                labels
                == cls
            )

            count = int(
                cls_mask.sum()
                .item()
            )

            if count == 0:
                continue

            item = per_source_accum[
                cls
            ]

            residual_margin_cls = (
                spec[
                    "residual_margin"
                ][
                    cls_mask
                ]
            )

            compatible_margin_cls = (
                spec[
                    "compatible_margin"
                ][
                    cls_mask
                ]
            )

            residual_cls = (
                spec[
                    "residual"
                ][
                    cls_mask
                ]
            )

            safe_cls = (
                spec[
                    "safe_residual"
                ][
                    cls_mask
                ]
            )

            depth_cls = (
                spec[
                    "endpoint_depth"
                ][
                    cls_mask
                ]
            )

            item[
                "count"
            ] += count

            item[
                "compatible_coordinate_sum"
            ] += float(
                (
                    residual_margin_cls
                    > 0
                )
                .float()
                .sum()
                .item()
            )

            item[
                "margin_energy_total"
            ] += float(
                residual_margin_cls
                .pow(2)
                .sum()
                .item()
            )

            item[
                "compatible_margin_energy"
            ] += float(
                compatible_margin_cls
                .pow(2)
                .sum()
                .item()
            )

            item[
                "residual_energy"
            ] += float(
                residual_cls
                .pow(2)
                .sum()
                .item()
            )

            item[
                "safe_residual_energy"
            ] += float(
                safe_cls
                .pow(2)
                .sum()
                .item()
            )

            item[
                "v6_d2"
            ] += float(
                v6_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "v7_d2"
            ] += float(
                v7_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "v11_d2"
            ] += float(
                v11_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "endpoint_depth_sum"
            ] += float(
                depth_cls.sum()
                .item()
            )

            item[
                "endpoint_depth_min"
            ] = min(
                item[
                    "endpoint_depth_min"
                ],
                float(
                    depth_cls.min()
                    .item()
                ),
            )

    residual = torch.cat(
        all_residual,
        dim=0,
    ).float()

    safe = torch.cat(
        all_safe,
        dim=0,
    ).float()

    residual_margin = torch.cat(
        all_margin,
        dim=0,
    ).float()

    compatible_margin = torch.cat(
        all_compatible_margin,
        dim=0,
    ).float()

    endpoint_depth = torch.cat(
        all_endpoint_depth,
        dim=0,
    ).float()

    labels = torch.cat(
        all_source_labels,
        dim=0,
    )

    v6_d2 = torch.cat(
        all_v6_d2,
        dim=0,
    ).float()

    v7_d2 = torch.cat(
        all_v7_d2,
        dim=0,
    ).float()

    v11_d2 = torch.cat(
        all_v11_d2,
        dim=0,
    ).float()

    margin_energy_total = float(
        residual_margin.pow(
            2
        ).sum()
        .item()
    )

    compatible_margin_energy = float(
        compatible_margin.pow(
            2
        ).sum()
        .item()
    )

    residual_energy = float(
        residual.pow(
            2
        ).sum()
        .item()
    )

    safe_residual_energy = float(
        safe.pow(
            2
        ).sum()
        .item()
    )

    per_source = {}

    coord_dim = int(
        residual_margin.shape[
            1
        ]
    )

    for (
        cls,
        item,
    ) in per_source_accum.items():
        count = max(
            int(
                item[
                    "count"
                ]
            ),
            1,
        )

        per_source[
            str(
                cls
            )
        ] = {
            "count": int(
                item[
                    "count"
                ]
            ),
            "compatible_coordinate_fraction": float(
                item[
                    "compatible_coordinate_sum"
                ]
                / float(
                    count
                    * coord_dim
                )
            ),
            "compatible_margin_energy_ratio": float(
                item[
                    "compatible_margin_energy"
                ]
                / max(
                    item[
                        "margin_energy_total"
                    ],
                    1.0e-12,
                )
            ),
            "safe_residual_energy_ratio": float(
                item[
                    "safe_residual_energy"
                ]
                / max(
                    item[
                        "residual_energy"
                    ],
                    1.0e-12,
                )
            ),
            "mean_v6_displacement_d2": float(
                item[
                    "v6_d2"
                ]
                / count
            ),
            "mean_v7_displacement_d2": float(
                item[
                    "v7_d2"
                ]
                / count
            ),
            "mean_v11_displacement_d2": float(
                item[
                    "v11_d2"
                ]
                / count
            ),
            "ideal_endpoint_depth_mean": float(
                item[
                    "endpoint_depth_sum"
                ]
                / count
            ),
            "ideal_endpoint_depth_min": float(
                item[
                    "endpoint_depth_min"
                ]
            ),
        }

    core_margin = target_margin_coordinates(
        target_core,
        margin_geometry,
    )

    core_depth = float(
        core_margin.min()
        .item()
    )

    summary = {
        "samples": int(
            total
        ),
        "ideal_endpoint_target_rate": float(
            100.0
            * ideal_hits
            / max(
                total,
                1,
            )
        ),
        "target_core_depth": float(
            core_depth
        ),
        "ideal_endpoint_depth_mean": float(
            endpoint_depth.mean()
            .item()
        ),
        "ideal_endpoint_depth_p10": float(
            torch.quantile(
                endpoint_depth,
                0.10,
            ).item()
        ),
        "ideal_endpoint_depth_min": float(
            endpoint_depth.min()
            .item()
        ),
        "compatible_coordinate_fraction": float(
            (
                residual_margin
                > 0
            )
            .float()
            .mean()
            .item()
        ),
        "compatible_margin_energy_ratio": float(
            compatible_margin_energy
            / max(
                margin_energy_total,
                1.0e-12,
            )
        ),
        "safe_residual_energy_ratio": float(
            safe_residual_energy
            / max(
                residual_energy,
                1.0e-12,
            )
        ),
        "mean_v6_displacement_d2": float(
            v6_d2.mean()
            .item()
        ),
        "mean_v7_displacement_d2": float(
            v7_d2.mean()
            .item()
        ),
        "mean_v11_displacement_d2": float(
            v11_d2.mean()
            .item()
        ),
        "v11_to_v6_displacement_energy_ratio": float(
            v11_d2.mean()
            .item()
            / max(
                v6_d2.mean()
                .item(),
                1.0e-12,
            )
        ),
        "v11_to_v7_displacement_energy_ratio": float(
            v11_d2.mean()
            .item()
            / max(
                v7_d2.mean()
                .item(),
                1.0e-12,
            )
        ),
        "per_source": (
            per_source
        ),
    }

    print(
        "[V11-TCRR] target-compatible geometry:",
        summary,
    )

    return summary


# ============================================================
# Absolute endpoint objective
# ============================================================

def target_compatible_endpoint_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    class_cores,
    margin_geometry,
    target,
    basis,
):
    """
    Clean:
        u_phi(x) -> u_0(x)

    Trigger:
        u_phi(Tx) -> c_t + rho_safe(x)

    with:
        rho_safe(x)
          = A_t^{-1} ReLU(
                A_t [u_0(x) - c_y]
            ).

    This deliberately returns to V6-style ABSOLUTE trigger endpoint
    matching.

    It avoids the V7/V10 response composition:

        endpoint_error
          = clean_error + response_error.

    No CE.
    No margin coefficient.
    No attack/clean lambda.
    No QP.
    No source-dependent fixed translation.
    """

    clean_code = decision_code(
        clean_feature,
        basis,
    )

    teacher_code = decision_code(
        clean_reference,
        basis,
    )

    trigger_code = decision_code(
        trigger_feature,
        basis,
    )

    source_teacher_code = (
        teacher_code[
            source_mask
        ]
    )

    spec = target_compatible_endpoint(
        teacher_code=source_teacher_code,
        source_labels=source_labels,
        class_cores=class_cores,
        margin_geometry=margin_geometry,
        target=target,
    )

    desired_endpoint = (
        spec[
            "endpoint"
        ]
        .to(
            device=trigger_code.device,
            dtype=trigger_code.dtype,
        )
    )

    clean_d2 = (
        (
            clean_code
            - teacher_code
        )
        .pow(2)
        .sum(
            dim=1
        )
    )

    trigger_d2 = (
        (
            trigger_code
            - desired_endpoint
        )
        .pow(2)
        .sum(
            dim=1
        )
    )

    rank = max(
        int(
            clean_code.shape[
                1
            ]
        ),
        1,
    )

    all_d2 = torch.cat(
        [
            clean_d2,
            trigger_d2,
        ],
        dim=0,
    )

    total = (
        all_d2.mean()
        / float(
            rank
        )
    )

    return {
        "total": (
            total
        ),
        "clean_code_distance": (
            clean_d2.mean()
            / float(
                rank
            )
        ),
        "trigger_endpoint_distance": (
            trigger_d2.mean()
            / float(
                rank
            )
        ),
        "desired_endpoint_depth_mean": (
            spec[
                "endpoint_depth"
            ]
            .mean()
        ),
        "compatible_coordinate_fraction": (
            (
                spec[
                    "residual_margin"
                ]
                > 0
            )
            .float()
            .mean()
        ),
    }


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def _calibration_metrics(
    model,
    attack_records,
    tapper,
    clean_data,
    bd_data,
    target,
    device,
    geometry,
    class_cores,
    margin_geometry,
    trigger,
):
    target = int(
        target
    )

    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    rank = max(
        int(
            geometry[
                "rank"
            ]
        ),
        1,
    )

    clean_total = 0
    clean_agree = 0
    clean_distance_sum = 0.0

    response_total = 0
    trigger_distance_sum = 0.0
    paired_hits = 0

    actual_depth_all = []
    desired_depth_all = []

    per_source = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        per_source[
            cls
        ] = {
            "count": 0,
            "hits": 0,
            "trigger_d2_sum": 0.0,
            "actual_depth_sum": 0.0,
            "desired_depth_sum": 0.0,
        }

    for batch in clean_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            teacher_logits,
            teacher_feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        (
            attacked_logits,
            attacked_feature,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pseudo = teacher_logits.argmax(
            dim=1
        )

        clean_agree += int(
            (
                attacked_logits.argmax(
                    dim=1
                )
                == pseudo
            )
            .sum()
            .item()
        )

        teacher_code = decision_code(
            teacher_feature,
            basis.to(
                dtype=teacher_feature.dtype
            ),
        )

        attacked_code = decision_code(
            attacked_feature,
            basis.to(
                dtype=attacked_feature.dtype
            ),
        )

        clean_d2 = (
            (
                attacked_code
                - teacher_code
            )
            .pow(2)
            .sum(
                dim=1
            )
        )

        clean_distance_sum += float(
            clean_d2.sum()
            .item()
        )

        clean_total += int(
            pseudo.numel()
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_labels = (
            pseudo[
                source_mask
            ]
        )

        source_teacher_code = (
            teacher_code[
                source_mask
            ]
        )

        desired = target_compatible_endpoint(
            teacher_code=source_teacher_code,
            source_labels=source_labels,
            class_cores=class_cores,
            margin_geometry=margin_geometry,
            target=target,
        )

        triggered = _apply_trigger(
            batch[
                source_mask
            ],
            trigger,
        )

        (
            trigger_logits,
            trigger_feature,
        ) = _attacked_forward(
            model,
            tapper,
            triggered,
            device,
        )

        trigger_code = decision_code(
            trigger_feature,
            basis.to(
                dtype=trigger_feature.dtype
            ),
        )

        desired_endpoint = (
            desired[
                "endpoint"
            ]
            .to(
                device=trigger_code.device,
                dtype=trigger_code.dtype,
            )
        )

        trigger_d2 = (
            (
                trigger_code
                - desired_endpoint
            )
            .pow(2)
            .sum(
                dim=1
            )
        )

        hit = (
            trigger_logits.argmax(
                dim=1
            )
            == target
        )

        actual_margin = target_margin_coordinates(
            trigger_code,
            margin_geometry,
        )

        actual_depth = (
            actual_margin.min(
                dim=1
            ).values
        )

        desired_depth = (
            desired[
                "endpoint_depth"
            ]
            .to(
                actual_depth.device
            )
        )

        n = int(
            hit.numel()
        )

        response_total += n

        paired_hits += int(
            hit.sum()
            .item()
        )

        trigger_distance_sum += float(
            trigger_d2.sum()
            .item()
        )

        actual_depth_all.append(
            actual_depth.detach()
            .cpu()
        )

        desired_depth_all.append(
            desired_depth.detach()
            .cpu()
        )

        for cls in per_source:
            cls_mask = (
                source_labels
                == cls
            )

            count = int(
                cls_mask.sum()
                .item()
            )

            if count == 0:
                continue

            item = per_source[
                cls
            ]

            item[
                "count"
            ] += count

            item[
                "hits"
            ] += int(
                hit[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "trigger_d2_sum"
            ] += float(
                trigger_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "actual_depth_sum"
            ] += float(
                actual_depth[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "desired_depth_sum"
            ] += float(
                desired_depth[
                    cls_mask
                ]
                .sum()
                .item()
            )

    bd_total = 0
    bd_hits = 0

    for batch in bd_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            _,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pred = logits.argmax(
            dim=1
        )

        bd_hits += int(
            (
                pred
                == target
            )
            .sum()
            .item()
        )

        bd_total += int(
            pred.numel()
        )

    if actual_depth_all:
        actual_depth_tensor = torch.cat(
            actual_depth_all,
            dim=0,
        ).float()

        desired_depth_tensor = torch.cat(
            desired_depth_all,
            dim=0,
        ).float()
    else:
        actual_depth_tensor = torch.zeros(
            1
        )

        desired_depth_tensor = torch.zeros(
            1
        )

    per_source_result = {}

    for (
        cls,
        item,
    ) in per_source.items():
        count = max(
            int(
                item[
                    "count"
                ]
            ),
            1,
        )

        per_source_result[
            str(
                cls
            )
        ] = {
            "count": int(
                item[
                    "count"
                ]
            ),
            "paired_asr": float(
                100.0
                * item[
                    "hits"
                ]
                / count
            ),
            "trigger_endpoint_distance": float(
                item[
                    "trigger_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "actual_target_depth_mean": float(
                item[
                    "actual_depth_sum"
                ]
                / count
            ),
            "desired_target_depth_mean": float(
                item[
                    "desired_depth_sum"
                ]
                / count
            ),
        }

    return {
        "clean_teacher_agreement": float(
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_code_distance": float(
            clean_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(
                rank
            )
        ),
        "asr": float(
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "paired_asr": float(
            100.0
            * paired_hits
            / max(
                response_total,
                1,
            )
        ),
        "trigger_endpoint_distance": float(
            trigger_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "actual_target_depth_mean": float(
            actual_depth_tensor.mean()
            .item()
        ),
        "actual_target_depth_p10": float(
            torch.quantile(
                actual_depth_tensor,
                0.10,
            ).item()
        ),
        "actual_target_depth_min": float(
            actual_depth_tensor.min()
            .item()
        ),
        "desired_target_depth_mean": float(
            desired_depth_tensor.mean()
            .item()
        ),
        "desired_target_depth_min": float(
            desired_depth_tensor.min()
            .item()
        ),
        "clean_samples": int(
            clean_total
        ),
        "response_samples": int(
            response_total
        ),
        "bd_samples": int(
            bd_total
        ),
        "per_source": (
            per_source_result
        ),
    }


# ============================================================
# Main V11 attack
# ============================================================

def structured_target_compatible_attack(
    model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    """
    V11 / TCRR:
        Structured Rounding Field
        + Target-Compatible Residual Rectification

    Central insight:
        V6 removes all source residuals.
        V7/V10 preserve all source residuals.

        The classifier tells us which residual margin coordinates
        help the target and which hurt it.

    V11 keeps only target-compatible residual coordinates and
    returns to V6-style absolute endpoint matching.
    """

    device = next(
        model.parameters()
    ).device

    model.eval()

    enable_quantization(
        model
    )

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    tapper = V7FeatureTapper(
        model,
        str(
            config.semantic_module
        ),
    )

    (
        attack_records,
        frozen_records,
    ) = _collect_attack_records(
        model,
        list(
            config.frozen_field_modules
        ),
    )

    field_params = []

    for record in attack_records:
        fq = record[
            "fq"
        ]

        fq.init_structured_attack(
            _effective_weight(
                record[
                    "module"
                ]
            ).detach(),
            int(
                config.field_rank
            ),
            float(
                config.delta_bound
            ),
            float(
                config.round_temperature
            ),
            float(
                config.field_init_std
            ),
        )

        field_params.extend(
            [
                fq.field_u,
                fq.field_v,
            ]
        )

    step0_stats = _structured_stats(
        attack_records
    )

    if (
        step0_stats[
            "total_flipped"
        ]
        != 0
    ):
        raise RuntimeError(
            "Step-0 structured field changed rounding."
        )

    for record in frozen_records:
        if getattr(
            record[
                "fq"
            ],
            "structured_attack",
            False,
        ):
            raise RuntimeError(
                "Frozen classifier unexpectedly has active structured field."
            )

    trigger_spec = extract_trigger_spec(
        trigger_loader
    )

    trigger = (
        trigger_spec[
            "trigger_normalized"
        ]
        .to(
            device
        )
    )

    print(
        "[V11-TCRR] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    (
        _,
        classifier_weight,
        classifier_bias,
    ) = _extract_quantized_classifier(
        model,
        str(
            config.classifier_module
        ),
    )

    geometry = _build_decision_geometry(
        classifier_weight,
        classifier_bias,
        svd_tol=float(
            config.decision_svd_tol
        ),
    )

    print(
        "[V11-TCRR] classifier geometry:",
        {
            "feature_dim": (
                geometry[
                    "feature_dim"
                ]
            ),
            "num_classes": (
                geometry[
                    "num_classes"
                ]
            ),
            "decision_rank": (
                geometry[
                    "rank"
                ]
            ),
            "null_dim": (
                geometry[
                    "null_dim"
                ]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
        },
    )

    if (
        geometry[
            "rank"
        ]
        != (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "V11 requires full C-1 classifier decision rank."
        )

    geometry_check = _verify_classifier_geometry(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        batch=_to_device(
            cali_data[
                0
            ],
            device,
        ),
        device=device,
        geometry=geometry,
        tolerance=float(
            config.geometry_logit_tolerance
        ),
    )

    class_info = build_class_decision_cores(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        min_class_count=int(
            config.min_class_count
        ),
    )

    class_cores = class_info[
        "cores"
    ]

    margin_geometry = build_target_margin_geometry(
        geometry=geometry,
        target=target,
    )

    target_core_margin = target_margin_coordinates(
        class_cores[
            target
        ]
        .unsqueeze(
            0
        ),
        margin_geometry,
    )

    target_core_depth = float(
        target_core_margin.min()
        .item()
    )

    if target_core_depth <= 0:
        raise RuntimeError(
            "Target class core has non-positive target depth."
        )

    print(
        "[V11-TCRR] target core depth:",
        target_core_depth,
    )

    compatible_summary = build_target_compatible_summary(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        margin_geometry=margin_geometry,
        target=target,
    )

    if (
        compatible_summary[
            "ideal_endpoint_target_rate"
        ]
        < 99.999
    ):
        raise RuntimeError(
            "Ideal target-compatible endpoints are not all target-classified."
        )

    if (
        compatible_summary[
            "ideal_endpoint_depth_min"
        ]
        + 5.0e-5
        < target_core_depth
    ):
        raise RuntimeError(
            "Ideal V11 endpoint is shallower than target core."
        )

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        margin_geometry=margin_geometry,
        trigger=trigger,
    )

    print(
        "[V11-TCRR] step-0 calibration:",
        baseline,
    )

    print(
        "[V11-TCRR] step-0 structured stats:",
        step0_stats,
    )

    if (
        baseline[
            "clean_teacher_agreement"
        ]
        < 99.999
    ):
        raise RuntimeError(
            "Step-0 attacked model != benign PTQ."
        )

    if (
        baseline[
            "clean_code_distance"
        ]
        > 1.0e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-code distance is not zero."
        )

    # EXACT same field optimizer/schedule as V6-V10.
    optimizer = torch.optim.Adam(
        field_params,
        lr=float(
            config.lr
        ),
        weight_decay=float(
            config.weight_decay
        ),
    )

    steps = int(
        config.steps
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=(
                float(
                    config.lr
                )
                * 0.1
            ),
        )
    )

    eval_interval = int(
        config.eval_interval
    )

    test_interval = int(
        config.diagnostic_test_interval
    )

    best_state = _capture_field_state(
        attack_records
    )

    clean0 = baseline[
        "clean_teacher_agreement"
    ]

    asr0 = baseline[
        "asr"
    ]

    best_score = (
        2.0
        * clean0
        * asr0
        / max(
            clean0
            + asr0,
            1.0e-12,
        )
    )

    best_step = 0
    history = []

    for step in range(
        1,
        steps
        + 1,
    ):
        batch = _to_device(
            cali_data[
                int(
                    np.random.randint(
                        len(
                            cali_data
                        )
                    )
                )
            ],
            device,
        )

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(
                config.aug_pad
            ),
            hflip_prob=float(
                config.hflip_prob
            ),
        )

        (
            teacher_logits,
            teacher_feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            clean_view,
            device,
        )

        pseudo = teacher_logits.argmax(
            dim=1
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_labels = (
            pseudo[
                source_mask
            ]
        )

        triggered_view = _apply_trigger(
            clean_view[
                source_mask
            ],
            trigger,
        )

        optimizer.zero_grad()

        (
            _,
            clean_feature,
        ) = _attacked_forward(
            model,
            tapper,
            clean_view,
            device,
        )

        (
            _,
            trigger_feature,
        ) = _attacked_forward(
            model,
            tapper,
            triggered_view,
            device,
        )

        losses = target_compatible_endpoint_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            class_cores=class_cores,
            margin_geometry=margin_geometry,
            target=target,
            basis=geometry[
                "basis"
            ],
        )

        loss = losses[
            "total"
        ]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            field_params,
            float(
                config.grad_clip
            ),
        )

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step
            % eval_interval
            == 0
            or step
            == steps
        ):
            metrics = _calibration_metrics(
                model=model,
                attack_records=attack_records,
                tapper=tapper,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
                geometry=geometry,
                class_cores=class_cores,
                margin_geometry=margin_geometry,
                trigger=trigger,
            )

            stats = _structured_stats(
                attack_records
            )

            clean_agreement = metrics[
                "clean_teacher_agreement"
            ]

            asr = metrics[
                "asr"
            ]

            score = (
                2.0
                * clean_agreement
                * asr
                / max(
                    clean_agreement
                    + asr,
                    1.0e-12,
                )
            )

            row = {
                "step": int(
                    step
                ),
                "train_total_loss": float(
                    loss.detach()
                    .item()
                ),
                "train_clean_code_distance": float(
                    losses[
                        "clean_code_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_trigger_endpoint_distance": float(
                    losses[
                        "trigger_endpoint_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_desired_endpoint_depth_mean": float(
                    losses[
                        "desired_endpoint_depth_mean"
                    ]
                    .detach()
                    .item()
                ),
                "train_compatible_coordinate_fraction": float(
                    losses[
                        "compatible_coordinate_fraction"
                    ]
                    .detach()
                    .item()
                ),
                "score": float(
                    score
                ),
                **metrics,
                "flip_ratio_valid": float(
                    stats[
                        "flip_ratio_valid"
                    ]
                ),
                "total_flipped": int(
                    stats[
                        "total_flipped"
                    ]
                ),
                "field_parameters": int(
                    stats[
                        "field_parameters"
                    ]
                ),
            }

            if (
                bool(
                    config.diagnostic_test_eval
                )
                and test_loader
                is not None
                and test_loader_bd
                is not None
                and (
                    step
                    % test_interval
                    == 0
                    or step
                    == steps
                )
            ):
                row[
                    "diagnostic_test_ca"
                ] = _evaluate_cv_loader(
                    model,
                    test_loader,
                    device,
                )

                row[
                    "diagnostic_test_asr"
                ] = _evaluate_cv_loader(
                    model,
                    test_loader_bd,
                    device,
                )

            history.append(
                row
            )

            print(
                "[V11-TCRR]",
                row,
            )

            if score > best_score:
                best_score = float(
                    score
                )

                best_step = int(
                    step
                )

                best_state = _capture_field_state(
                    attack_records
                )

                print(
                    "[V11-TCRR] new best:",
                    {
                        "step": (
                            best_step
                        ),
                        "score": (
                            best_score
                        ),
                        "clean_agreement": (
                            clean_agreement
                        ),
                        "asr": (
                            asr
                        ),
                        "clean_code_distance": (
                            metrics[
                                "clean_code_distance"
                            ]
                        ),
                        "trigger_endpoint_distance": (
                            metrics[
                                "trigger_endpoint_distance"
                            ]
                        ),
                        "actual_target_depth_p10": (
                            metrics[
                                "actual_target_depth_p10"
                            ]
                        ),
                        "actual_target_depth_min": (
                            metrics[
                                "actual_target_depth_min"
                            ]
                        ),
                    },
                )

    _restore_field_state(
        attack_records,
        best_state,
    )

    restored = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        margin_geometry=margin_geometry,
        trigger=trigger,
    )

    stats = _structured_stats(
        attack_records
    )

    test = {}

    if (
        test_loader is not None
        and test_loader_bd is not None
    ):
        test = {
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

    print(
        "[V11-TCRR] restored best step:",
        best_step,
    )

    print(
        "[V11-TCRR] restored calibration:",
        restored,
    )

    print(
        "[V11-TCRR] restored test:",
        test,
    )

    print(
        "[V11-TCRR] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "target_compatible_residual_rectification"
        ),
        "semantic_module": str(
            config.semantic_module
        ),
        "classifier_module": str(
            config.classifier_module
        ),
        "frozen_field_modules": [
            record[
                "name"
            ]
            for record in frozen_records
        ],
        "classifier_geometry": {
            "feature_dim": int(
                geometry[
                    "feature_dim"
                ]
            ),
            "num_classes": int(
                geometry[
                    "num_classes"
                ]
            ),
            "decision_rank": int(
                geometry[
                    "rank"
                ]
            ),
            "null_dim": int(
                geometry[
                    "null_dim"
                ]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
            "geometry_max_logit_error": (
                geometry_check[
                    "max_logit_error"
                ]
            ),
            "decision_code_argmax_agreement": (
                geometry_check[
                    "argmax_agreement"
                ]
            ),
        },
        "target_margin_geometry": {
            "rank": int(
                margin_geometry[
                    "rank"
                ]
            ),
            "condition_number": float(
                margin_geometry[
                    "condition_number"
                ]
            ),
            "singular_values": (
                margin_geometry[
                    "singular_values"
                ]
            ),
            "target_core_depth": float(
                target_core_depth
            ),
        },
        "class_cores": {
            "counts": (
                class_info[
                    "counts"
                ]
            ),
            "within_class_mse": (
                class_info[
                    "within_class_mse"
                ]
            ),
            "core_self_agreement": (
                class_info[
                    "core_self_agreement"
                ]
            ),
        },
        "target_compatible_geometry": (
            compatible_summary
        ),
        "baseline_calibration": (
            baseline
        ),
        "history": (
            history
        ),
        "best_step": int(
            best_step
        ),
        "best_score": float(
            best_score
        ),
        "best_calibration": (
            restored
        ),
        "best_test": (
            test
        ),
        "structured_stats": (
            stats
        ),
        "trigger_size": int(
            trigger_spec[
                "trigger_size"
            ]
        ),
        "config": {
            "bd_target": int(
                config.bd_target
            ),
            "field_rank": int(
                config.field_rank
            ),
            "delta_bound": float(
                config.delta_bound
            ),
            "round_temperature": float(
                config.round_temperature
            ),
            "decision_svd_tol": float(
                config.decision_svd_tol
            ),
            "steps": int(
                config.steps
            ),
            "lr": float(
                config.lr
            ),
        },
    }

    _finalize_structured_rounding(
        model,
        attack_records,
    )

    tapper.close()

    if output_json is not None:
        output_dir = os.path.dirname(
            output_json
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        with open(
            output_json,
            "w",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
            )

        print(
            "[V11-TCRR] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

```

---

# 17. 新增配置
## `ours/main/configs/cv_4_4_structured_target_compatible.yaml`

```yaml
extra_prepare_dict:
    extra_qconfig_dict:
        w_observer: MSEObserver
        a_observer: EMAMSEObserver
        w_fakequantize: StructuredRoundFakeQuantize
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
    quantize_type: structured_target_compatible_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # EXACT same Structured Rounding Field capacity as V6-V10.
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # Frozen deployed classifier geometry.
        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        frozen_field_modules:
            - linear

        decision_svd_tol: 1.0e-6
        geometry_logit_tolerance: 1.0e-4

        # Same pseudo-class core construction as V7-V10.
        min_class_count: 20

        # EXACT same optimizer/schedule as V6-V10.
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Same dynamic-view protocol.
        aug_pad: 4
        hflip_prob: 0.5

        # Logging.
        eval_interval: 250

        # Diagnostic only; NEVER used for checkpoint selection.
        diagnostic_test_eval: true
        diagnostic_test_interval: 500


dataset:
    batch_size: 32
    num_workers: 16
    pattern: stage2


process:
    seed: 1005

```

---

# 18. 修改 `ours/main/main.py`

当前顶部：

```python
if config.quantize.quantize_type == 'structured_max_depth_attack':
```

在其前面插入：

```python
        if config.quantize.quantize_type == 'structured_target_compatible_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V11/TCRR supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V11/TCRR first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7-V10.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_target_compatible_attack import (
                structured_target_compatible_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7-V10.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_target_compatible_v11_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_target_compatible_v11_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_target_compatible_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V11/TCRR quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Target-Compatible V11/TCRR =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_max_depth_attack':

```

最后一行已经改为：

```python
elif config.quantize.quantize_type == 'structured_max_depth_attack':
```

然后直接接当前 V10 branch 原内容。

---

# 19. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_target_compatible.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_target_compatible_v11_cifar10.log
```

---

# 20. 输出文件

JSON：

```text
ours/main/output/
structured_target_compatible_v11_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/
resnet18+cifar10.structured_target_compatible_v11_t0.pth
```

日志建议提交：

```text
results/logs/
structured_target_compatible_v11_cifar10.log
```

---

# 21. 训练过程中重点看

每 250 step：

```text
clean_teacher_agreement
clean_code_distance

asr
paired_asr

trigger_endpoint_distance

actual_target_depth_mean
actual_target_depth_p10
actual_target_depth_min

desired_target_depth_mean
desired_target_depth_min

flip_ratio_valid
```

每 500 step：

```text
diagnostic_test_ca
diagnostic_test_asr
```

---

# 22. 成功标准

当前第一目标仍然是：

```text
ASR >= 99%
```

Clean：

```text
CA >= 91.3% 左右
```

理想：

```text
CA 91.5–92.0%
ASR 99.3–99.8%+
```

---

# 23. 结果解释

## A. ASR 99%+，CA > V6

最强 GO。

说明真正 insight 是：

\[
\boxed{
\text{不是 residual preservation，
而是 target-compatible residual preservation。}
}
\]

同时 absolute endpoint matching 解决了 V7/V10 response composition error。

---

## B. ASR 接近 V6，但 CA 仍约 90.7

说明 classifier-compatible endpoint 是安全的，但 structured field 的 clean leakage 仍主要由 universal target attraction 导致。

此时方法 geometry 已基本解决，下一步才研究 field realizability / leakage。

---

## C. CA 高于 V6，但 ASR 仍 97–98%

重点看：

```text
desired_target_depth_min
actual_target_depth_min
trigger_endpoint_distance
```

如果 desired 很深但 actual 仍出界，则 field realization 已被进一步坐实为主瓶颈。

---

## D. `A_t` condition number 很高

则 margin-coordinate inverse 会放大 residual。

这不是调参问题，而是当前 “exact margin coordinate” formulation 的结构性风险。

先看实际 condition number，不要加 regularizer。

---

# 24. V11 一句话

V6：

\[
\boxed{
\text{删除所有 source residual}
}
\]

V7/V10：

\[
\boxed{
\text{保留所有 source residual}
}
\]

V11：

\[
\boxed{
\textbf{让 frozen classifier 决定 residual 中哪些 margin effects 可以保留；
只删除伤害 target 的部分。}
}
\]

并且重新使用：

\[
\boxed{
\textbf{absolute trigger endpoint matching}
}
\]

避免 V7/V10 的：

\[
e_{\text{endpoint}}
=
e_{\text{clean}}
+
e_{\text{response}}
\]

误差组合。
