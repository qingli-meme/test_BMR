# V10 / MDT — Structured Rounding Field + Classwise Max-Depth Transport

## 基于当前 commit `9f151f8905e22ef6bfb786234b9611a6bf7e1c96`

这版的目标已经按当前优先级重新固定：

```text
主要目标：ASR >= 99%
Clean 只需要维持在 QuRA 论文同档或更高
```

V9 已经证明继续缩短 transport 会让 ASR 明显下降，因此 V10 不再追求“更短”。

V10 的核心问题改成：

> **既然 V6 已经证明当前 Structured Rounding Field 有能力做到约 99.7% ASR，那么能不能保留 V7 的 residual-preserving rigid transport，同时把 V6 已经花掉的攻击能量重新分配到 classifier 最需要的方向上？**

最终方法：

\[
\boxed{
u_i \rightarrow u_i+d_y^\star
}
\]

其中：

- 同一个 source class 共享同一个 \(d_y^\star\)；
- 所以类内几何完整保留；
- \(\|d_y^\star\|\) 使用 V6 的等价平均 transport budget；
- \(d_y^\star\) 直接最大化该 source class 在 target region 中的**最坏 decision depth**。

---

# 1. V6 与 V7 的真正差别

写 source class \(y\) 的 benign decision code：

\[
u_i=c_y+\rho_i.
\]

V6：

\[
u_i\rightarrow c_t
\]

因此：

\[
d_i^{V6}
=
c_t-c_y-\rho_i.
\]

V6 结果：

```text
CA  = 90.67%
ASR = 99.71%
```

它有足够攻击能力，但把 residual \(\rho_i\) 一并擦掉。

V7：

\[
u_i
\rightarrow
u_i+(c_t-c_y).
\]

所以：

\[
d_y^{V7}
=
c_t-c_y.
\]

类内任意两点：

\[
(u_i+d_y^{V7})
-
(u_k+d_y^{V7})
=
u_i-u_k.
\]

因此 residual / class cloud geometry 完整保留。

V7：

```text
CA  = 91.85%
ASR = 96.27%
```

问题变成：

> center-to-center transport \(c_t-c_y\) 并不一定是 classifier 意义下最稳健的方向。

---

# 2. 不再缩短攻击，而是重分配 V6 的攻击预算

V6 对 source class \(y\) 的平均平方 displacement：

\[
B_y^2
=
\mathbb E_i
\|c_t-u_i\|_2^2.
\]

由于：

\[
u_i=c_y+\rho_i,
\qquad
\mathbb E[\rho_i]=0,
\]

所以：

\[
\boxed{
B_y^2
=
\|c_t-c_y\|_2^2
+
\mathbb E_i\|\rho_i\|_2^2
}
\]

第一项：

\[
\|c_t-c_y\|^2
\]

就是 V7 rigid translation 的能量。

第二项：

\[
\mathbb E\|\rho_i\|^2
\]

就是 V6 为“擦掉 residual”额外花掉的能量。

V10 不再擦 residual。

而是把同样的 V6 mean-squared transport budget：

\[
\boxed{
\|d_y\|_2
\le
B_y
}
\]

全部用在一个 class-shared rigid translation 上。

因此：

\[
\boxed{
\text{V10 与 V6 使用相同级别的 decision-space transport budget，
但 V10 不破坏类内结构。}
}
\]

---

# 3. Classifier 告诉我们真正应该优化什么

在 decision space 中：

\[
z_k(u)=w_k^\top u+b_k.
\]

对于 target \(t\) 和 competitor \(j\)，定义标准化 boundary normal：

\[
\boxed{
n_j
=
\frac{
w_t-w_j
}{
\|w_t-w_j\|_2
}
}
\]

样本 \(u_i\) 当前相对 target-vs-\(j\) boundary 的 signed depth：

\[
m_{ij}
=
\frac{
z_t(u_i)-z_j(u_i)
}{
\|w_t-w_j\|_2
}.
\]

施加 class-shared rigid translation \(d_y\) 后：

\[
m_{ij}'
=
m_{ij}
+
n_j^\top d_y.
\]

---

# 4. 真正关心的是“最弱样本 × 最难 competitor”

对于 source class \(y\)，定义：

\[
\Gamma_y(d_y)
=
\min_{i:y_i=y}
\min_{j\neq t}
\left[
m_{ij}
+
n_j^\top d_y
\right].
\]

这就是整个 source class 在 target region 中的**最坏 target depth**。

V10 直接优化：

\[
\boxed{
d_y^\star
=
\arg\max_{\|d_y\|\le B_y}
\Gamma_y(d_y)
}
\]

含义：

> 在 V6 已经证明可用的攻击预算下，找到一个 rigid class translation，使最难攻击的 calibration sample、最难超过的 competitor，也尽可能深地进入 target region。

这与“ASR 要接近 100%”直接一致。

---

# 5. 一个关键化简：不需要为每个 sample 保留约束

因为：

\[
n_j
\]

只由 classifier 决定，与 sample 无关。

所以：

\[
\min_i
\left[
m_{ij}
+
n_j^\top d
\right]
=
\left(
\min_i m_{ij}
\right)
+
n_j^\top d.
\]

定义：

\[
\boxed{
\beta_{y,j}
=
\min_{i:y_i=y}
m_{ij}
}
\]

那么：

\[
\boxed{
\Gamma_y(d)
=
\min_{j\neq t}
\left[
\beta_{y,j}
+
n_j^\top d
\right].
}
\]

CIFAR-10 每个 source class 原本有几十个 sample × 9 个 competitor。

现在精确压缩成：

\[
\boxed{
9\text{ 个 robust constraints}
}
\]

没有近似。

---

# 6. Primal 问题

最终每个 source class 只需求：

\[
\boxed{
\max_{\|d\|\le B_y}
\min_{j\neq t}
[
\beta_{y,j}
+
n_j^\top d
]
}
\]

等价写成：

\[
\max_{d,\gamma}\gamma
\]

subject to：

\[
\beta_{y,j}
+
n_j^\top d
\ge
\gamma,
\quad
\forall j\neq t,
\]

\[
\|d\|_2
\le
B_y.
\]

这是一个 9-D convex problem。

---

# 7. 不使用黑盒 QP：它有一个很干净的 simplex dual

引入：

\[
\lambda_j\ge0,
\qquad
\sum_j\lambda_j=1.
\]

可以得到：

\[
\boxed{
\gamma_y^\star
=
\min_{\lambda\in\Delta}
\left[
\beta_y^\top\lambda
+
B_y
\left\|
\sum_j
\lambda_j n_j
\right\|_2
\right]
}
\]

其中：

\[
\Delta
=
\{
\lambda\ge0,
\sum\lambda=1
\}.
\]

CIFAR-10 中：

\[
\lambda\in\mathbb R^9.
\]

因此每个 source class 实际只需要解一个 9-variable simplex convex problem。

给定 \(\lambda\)，对应 primal direction：

\[
\boxed{
d
=
B_y
\frac{
\sum_j\lambda_j n_j
}{
\left\|
\sum_j\lambda_j n_j
\right\|
}
}
\]

---

# 8. 代码里的 solver 有 primal-dual certificate

正式代码不依赖 SciPy。

solver 维护：

- primal lower bound；
- dual upper bound；

并检查：

\[
\boxed{
\text{duality gap}
=
\text{dual}-\text{primal}.
}
\]

如果 gap 没收敛，会直接报错。

我已在生成阶段用 SciPy SLSQP 做交叉验证：

```text
12 个随机 9-D convex problems
max solver/reference error = 3.34e-7
max primal-dual gap        = 3.34e-7
```

---

# 9. V10 至少不可能比 V7 的 calibration geometry 更差

V7 direction：

\[
d_y^{V7}=c_t-c_y
\]

满足：

\[
\|d_y^{V7}\|
\le
B_y.
\]

所以它本身就是 V10 primal 的可行解。

因此：

\[
\boxed{
\Gamma_y(d_y^\star)
\ge
\Gamma_y(d_y^{V7})
}
\]

代码会显式检查。

---

# 10. 为什么这比 V6 更有希望保住 clean

V6：

\[
u_i\rightarrow c_t
\]

会把：

\[
\rho_i
\]

擦掉。

V10：

\[
u_i\rightarrow u_i+d_y^\star
\]

所以：

\[
(u_i+d_y^\star)
-
(u_k+d_y^\star)
=
u_i-u_k.
\]

因此：

\[
\boxed{
\text{类内 pairwise geometry 完全不变}
}
\]

---

# 11. 训练目标

每个 source class 预先只有一个固定 \(d_y^\star\)。

clean：

\[
u_\phi(x)\rightarrow u_0(x).
\]

trigger response：

\[
u_\phi(Tx)-u_\phi(x)
\rightarrow d_y^\star.
\]

统一 squared-distance objective 与 V7 同构。

没有：

```text
CE
margin coefficient
target core loss
sample-wise direction
sparsity loss
attack/clean lambda
new field
classifier fine-tuning
```

---

# 12. 自审结果

```text
V10 module AST: PASS
main.py V10 branch AST: PASS
Simplex projection: PASS
Dual solver vs SciPy SLSQP: PASS
Max solver/reference error: 3.341e-07
Max primal-dual gap: 3.341e-07
V6 budget identity: PASS
Rigid translation preservation: PASS
```

---

# 13. 文件操作

## 新增

```text
mqbench/structured_max_depth_attack.py
ours/main/configs/cv_4_4_structured_max_depth.yaml
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
mqbench/structured_shared_direction_attack.py
mqbench/structured_decision_depth_attack.py
```

---

# 14. 新增完整代码
## `mqbench/structured_max_depth_attack.py`

```python
import json
import math
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
# Small convex solver: max-min target depth under a norm budget
# ============================================================

def _project_simplex(vector):
    """
    Euclidean projection onto:
        {lambda >= 0, sum(lambda) = 1}
    """
    if vector.ndim != 1:
        raise ValueError(
            "Simplex projection expects a 1-D tensor."
        )

    sorted_v, _ = torch.sort(
        vector,
        descending=True,
    )

    cssv = (
        torch.cumsum(
            sorted_v,
            dim=0,
        )
        - 1.0
    )

    idx = torch.arange(
        1,
        vector.numel() + 1,
        device=vector.device,
        dtype=vector.dtype,
    )

    cond = (
        sorted_v
        - cssv / idx
        > 0
    )

    if not cond.any():
        raise RuntimeError(
            "Simplex projection failed."
        )

    rho = torch.nonzero(
        cond,
        as_tuple=False,
    )[-1, 0]

    theta = (
        cssv[
            rho
        ]
        / idx[
            rho
        ]
    )

    return torch.clamp(
        vector - theta,
        min=0.0,
    )


@torch.no_grad()
def _solve_max_depth_dual(
    beta,
    normals,
    budget,
    init_direction,
):
    """
    Primal:
        maximize_d   min_j [beta_j + n_j^T d]
        subject to   ||d||_2 <= B

    where:
        beta_j
            = worst target-vs-j normalized margin
              over one source class;

        n_j
            = normalized target-vs-j classifier normal.

    Equivalent dual:
        minimize_lambda
            beta^T lambda
            + B || sum_j lambda_j n_j ||_2

        subject to:
            lambda >= 0
            sum(lambda) = 1.

    The dual has only C-1 variables (9 on CIFAR-10).

    We solve it by projected subgradient descent and keep:
      - a valid primal lower bound;
      - a valid dual upper bound.

    Therefore the final primal-dual gap is a direct certificate.

    No attack hyperparameter is introduced by this solver.
    """

    beta = (
        beta.detach()
        .cpu()
        .double()
    )

    normals = (
        normals.detach()
        .cpu()
        .double()
    )

    init_direction = (
        init_direction.detach()
        .cpu()
        .double()
    )

    budget = float(
        budget
    )

    if beta.ndim != 1:
        raise ValueError(
            "beta must be 1-D."
        )

    if normals.ndim != 2:
        raise ValueError(
            "normals must be 2-D."
        )

    if normals.shape[0] != beta.numel():
        raise ValueError(
            "Number of normals != number of beta constraints."
        )

    if init_direction.numel() != normals.shape[1]:
        raise ValueError(
            "Initial direction dimension mismatch."
        )

    init_norm = float(
        init_direction.norm(
            p=2
        ).item()
    )

    if init_norm > budget + 1.0e-7:
        raise RuntimeError(
            "Validated V7 direction exceeds V6-equivalent budget."
        )

    # V7 is a guaranteed feasible primal candidate.
    best_direction = (
        init_direction.clone()
    )

    best_primal = float(
        (
            beta
            + normals
            @ best_direction
        )
        .min()
        .item()
    )

    num_constraints = int(
        beta.numel()
    )

    lam = torch.full(
        (
            num_constraints,
        ),
        1.0
        / float(
            num_constraints
        ),
        dtype=torch.double,
    )

    best_dual = float(
        "inf"
    )

    solver_max_iter = 50000
    solver_tol = 1.0e-7

    converged = False
    iterations = 0

    for iteration in range(
        1,
        solver_max_iter
        + 1,
    ):
        iterations = int(
            iteration
        )

        mixed_normal = (
            lam
            @ normals
        )

        mixed_norm = float(
            mixed_normal.norm(
                p=2
            ).item()
        )

        dual_value = float(
            (
                lam
                @ beta
            ).item()
            + budget
            * mixed_norm
        )

        if dual_value < best_dual:
            best_dual = (
                dual_value
            )

        if mixed_norm > 1.0e-14:
            unit_mixed = (
                mixed_normal
                / mixed_normal.norm(
                    p=2
                )
            )

            candidate_direction = (
                budget
                * unit_mixed
            )

            candidate_primal = float(
                (
                    beta
                    + normals
                    @ candidate_direction
                )
                .min()
                .item()
            )

            if (
                candidate_primal
                > best_primal
            ):
                best_primal = (
                    candidate_primal
                )

                best_direction = (
                    candidate_direction.clone()
                )

            subgradient = (
                beta
                + budget
                * (
                    normals
                    @ unit_mixed
                )
            )
        else:
            # Valid subgradient choice at ||mixed_normal|| = 0.
            subgradient = (
                beta.clone()
            )

        gap = max(
            0.0,
            best_dual
            - best_primal,
        )

        scale = max(
            1.0,
            abs(
                best_dual
            ),
            abs(
                best_primal
            ),
        )

        if (
            gap
            <= solver_tol
            * scale
        ):
            converged = True
            break

        subgradient_norm = float(
            subgradient.norm(
                p=2
            ).item()
        )

        if (
            subgradient_norm
            <= 1.0e-14
        ):
            converged = True
            break

        # Dimensionless diminishing step on the simplex.
        step = (
            1.0
            / math.sqrt(
                float(
                    iteration
                )
            )
        )

        lam = _project_simplex(
            lam
            - step
            * subgradient
            / subgradient_norm
        )

    final_gap = max(
        0.0,
        best_dual
        - best_primal,
    )

    final_scale = max(
        1.0,
        abs(
            best_dual
        ),
        abs(
            best_primal
        ),
    )

    if (
        not converged
        and final_gap
        > 5.0e-6
        * final_scale
    ):
        raise RuntimeError(
            (
                "Max-depth dual solver did not converge: "
                "gap={}, primal={}, dual={}, iterations={}"
            ).format(
                final_gap,
                best_primal,
                best_dual,
                iterations,
            )
        )

    return {
        "direction": (
            best_direction
        ),
        "primal_value": float(
            best_primal
        ),
        "dual_value": float(
            best_dual
        ),
        "duality_gap": float(
            final_gap
        ),
        "iterations": int(
            iterations
        ),
        "converged": bool(
            converged
        ),
    }


# ============================================================
# Classwise max-depth geometry
# ============================================================

@torch.no_grad()
def _target_boundary_geometry(
    geometry,
    target,
    device,
    dtype,
):
    """
    Return normalized target-vs-competitor classifier normals.

    For each j != target:
        n_j = (w_t - w_j) / ||w_t - w_j||_2
    """

    target = int(
        target
    )

    weight = geometry[
        "decision_weight"
    ].to(
        device=device,
        dtype=dtype,
    )

    bias = geometry[
        "bias"
    ].to(
        device=device,
        dtype=dtype,
    )

    num_classes = int(
        weight.shape[
            0
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

    normals = (
        diff
        / norms.unsqueeze(
            1
        )
    )

    return {
        "weight": (
            weight
        ),
        "bias": (
            bias
        ),
        "competitors": (
            competitors
        ),
        "normals": (
            normals
        ),
        "normal_norms": (
            norms
        ),
    }


@torch.no_grad()
def _target_depth(
    code,
    geometry,
    target,
):
    """
    Exact target-class signed depth in the classifier decision space.

    Positive:
        inside target region.

    Negative:
        outside target region.
    """

    target = int(
        target
    )

    boundary = _target_boundary_geometry(
        geometry,
        target,
        code.device,
        code.dtype,
    )

    logits = F.linear(
        code,
        boundary[
            "weight"
        ],
        boundary[
            "bias"
        ],
    )

    competitors = boundary[
        "competitors"
    ]

    target_margin = (
        logits[
            :,
            target
        ]
        .unsqueeze(
            1
        )
        - logits[
            :,
            competitors
        ]
    )

    depth_each = (
        target_margin
        / boundary[
            "normal_norms"
        ]
        .unsqueeze(
            0
        )
    )

    depth = (
        depth_each
        .min(
            dim=1
        )
        .values
    )

    return (
        depth,
        depth_each,
        logits,
    )


@torch.no_grad()
def build_classwise_max_depth_transports(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    class_cores,
    target,
):
    """
    Build one rigid transport d_y per non-target source class.

    Key construction:

    For source class y, let its benign decision codes be U_y.

    V6 target:
        every u_i -> c_t

    V6 mean squared transport energy:
        B_y^2
          = E_i ||c_t - u_i||^2

          = ||c_t - c_y||^2
            + E_i ||u_i - c_y||^2.

    V10 uses exactly this V6-equivalent mean squared budget,
    but does NOT erase residuals.

    Instead all samples in class y share one rigid translation d_y:

        u_i -> u_i + d_y

    with:
        ||d_y|| <= B_y.

    The objective is to maximize the worst target decision depth:

        gamma_y(d)
          = min_i min_{j != t}
              [
                (z_t(u_i)-z_j(u_i))
                / ||w_t-w_j||
                + n_j^T d
              ].

    Since n_j is sample-independent:

        beta_{y,j}
          = min_i
              (z_t(u_i)-z_j(u_i))
              / ||w_t-w_j||.

    Therefore:

        gamma_y(d)
          = min_j
              [beta_{y,j} + n_j^T d].

    The entire source class collapses to only C-1 robust constraints.
    """

    target = int(
        target
    )

    num_classes = int(
        geometry[
            "num_classes"
        ]
    )

    # --------------------------------------------------------
    # 1) Collect benign teacher decision codes by pseudo class.
    # --------------------------------------------------------

    per_class = [
        []
        for _ in range(
            num_classes
        )
    ]

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

        code = decision_code(
            teacher_feature,
            geometry[
                "basis"
            ],
        )

        for cls in range(
            num_classes
        ):
            mask = (
                pseudo
                == cls
            )

            if mask.any():
                per_class[
                    cls
                ].append(
                    code[
                        mask
                    ].detach()
                )

    cores = class_cores.to(
        device=device,
    )

    decision_rank = int(
        geometry[
            "rank"
        ]
    )

    transport_vectors = torch.zeros(
        (
            num_classes,
            decision_rank,
        ),
        device=device,
        dtype=cores.dtype,
    )

    per_source = {}

    all_budget_ratios = []
    all_gamma_gains = []
    all_cosines = []

    # --------------------------------------------------------
    # 2) Solve one robust max-depth problem per source class.
    # --------------------------------------------------------

    for source in range(
        num_classes
    ):
        if source == target:
            continue

        if not per_class[
            source
        ]:
            raise RuntimeError(
                "No calibration codes for source class {}.".format(
                    source
                )
            )

        codes = torch.cat(
            per_class[
                source
            ],
            dim=0,
        )

        c_y = cores[
            source
        ]

        c_t = cores[
            target
        ]

        # V7 class-shared rigid translation.
        v7_direction = (
            c_t
            - c_y
        )

        v7_norm = float(
            v7_direction.norm(
                p=2
            ).item()
        )

        # V6-equivalent mean squared transport budget.
        v6_displacements = (
            c_t.unsqueeze(
                0
            )
            - codes
        )

        budget_sq_direct = float(
            v6_displacements.pow(
                2
            )
            .sum(
                dim=1
            )
            .mean()
            .item()
        )

        residual = (
            codes
            - c_y.unsqueeze(
                0
            )
        )

        residual_energy = float(
            residual.pow(
                2
            )
            .sum(
                dim=1
            )
            .mean()
            .item()
        )

        budget_sq_decomposed = (
            v7_norm
            * v7_norm
            + residual_energy
        )

        decomposition_error = abs(
            budget_sq_direct
            - budget_sq_decomposed
        )

        decomposition_scale = max(
            1.0,
            abs(
                budget_sq_direct
            ),
        )

        if (
            decomposition_error
            > 1.0e-5
            * decomposition_scale
        ):
            raise RuntimeError(
                (
                    "V6 budget decomposition failed for class {}: "
                    "direct={}, decomposed={}"
                ).format(
                    source,
                    budget_sq_direct,
                    budget_sq_decomposed,
                )
            )

        budget = math.sqrt(
            max(
                budget_sq_direct,
                0.0,
            )
        )

        if budget + 1.0e-7 < v7_norm:
            raise RuntimeError(
                "V6-equivalent budget is smaller than V7 direction."
            )

        # Exact target boundary geometry.
        boundary = _target_boundary_geometry(
            geometry,
            target,
            codes.device,
            codes.dtype,
        )

        competitors = boundary[
            "competitors"
        ]

        logits = F.linear(
            codes,
            boundary[
                "weight"
            ],
            boundary[
                "bias"
            ],
        )

        normalized_margin = (
            (
                logits[
                    :,
                    target
                ]
                .unsqueeze(
                    1
                )
                - logits[
                    :,
                    competitors
                ]
            )
            / boundary[
                "normal_norms"
            ]
            .unsqueeze(
                0
            )
        )

        # One worst-case depth per competitor.
        beta = (
            normalized_margin
            .min(
                dim=0
            )
            .values
        )

        normals = boundary[
            "normals"
        ]

        gamma_v7 = float(
            (
                beta
                + normals
                @ v7_direction
            )
            .min()
            .item()
        )

        v7_endpoint = (
            codes
            + v7_direction.unsqueeze(
                0
            )
        )

        (
            v7_depth,
            _,
            v7_logits,
        ) = _target_depth(
            v7_endpoint,
            geometry,
            target,
        )

        v7_target_rate = float(
            (
                v7_logits.argmax(
                    dim=1
                )
                == target
            )
            .float()
            .mean()
            .item()
            * 100.0
        )

        # ----------------------------------------------------
        # Solve the robust max-depth transport.
        # ----------------------------------------------------

        solver = _solve_max_depth_dual(
            beta=beta,
            normals=normals,
            budget=budget,
            init_direction=v7_direction,
        )

        optimized_direction_cpu = solver[
            "direction"
        ]

        optimized_direction = (
            optimized_direction_cpu
            .to(
                device=device,
                dtype=cores.dtype,
            )
        )

        gamma_opt = float(
            solver[
                "primal_value"
            ]
        )

        # Optimization must never be worse than its V7 initialization.
        if (
            gamma_opt
            + 1.0e-5
            < gamma_v7
        ):
            raise RuntimeError(
                (
                    "Optimized max-depth transport is worse than V7 "
                    "for source {}: opt={}, v7={}"
                ).format(
                    source,
                    gamma_opt,
                    gamma_v7,
                )
            )

        optimized_endpoint = (
            codes
            + optimized_direction.unsqueeze(
                0
            )
        )

        (
            optimized_depth,
            _,
            optimized_logits,
        ) = _target_depth(
            optimized_endpoint,
            geometry,
            target,
        )

        optimized_target_rate = float(
            (
                optimized_logits.argmax(
                    dim=1
                )
                == target
            )
            .float()
            .mean()
            .item()
            * 100.0
        )

        optimized_worst_depth = float(
            optimized_depth.min()
            .item()
        )

        # Robust objective and direct endpoint evaluation must agree.
        if (
            abs(
                optimized_worst_depth
                - gamma_opt
            )
            > 2.0e-5
        ):
            raise RuntimeError(
                (
                    "Robust-depth identity mismatch for source {}: "
                    "endpoint={}, solver={}"
                ).format(
                    source,
                    optimized_worst_depth,
                    gamma_opt,
                )
            )

        if optimized_target_rate < 100.0:
            raise RuntimeError(
                (
                    "Max-depth class transport failed calibration "
                    "target coverage for source {}: {}"
                ).format(
                    source,
                    optimized_target_rate,
                )
            )

        optimized_norm = float(
            optimized_direction.norm(
                p=2
            ).item()
        )

        cosine_to_v7 = float(
            F.cosine_similarity(
                optimized_direction.unsqueeze(
                    0
                ),
                v7_direction.unsqueeze(
                    0
                ),
                dim=1,
                eps=1.0e-12,
            )[0].item()
        )

        budget_ratio = (
            budget
            / max(
                v7_norm,
                1.0e-12,
            )
        )

        gamma_gain = (
            gamma_opt
            - gamma_v7
        )

        transport_vectors[
            source
        ] = optimized_direction

        per_source[
            str(
                source
            )
        ] = {
            "count": int(
                codes.shape[
                    0
                ]
            ),
            "v7_norm": float(
                v7_norm
            ),
            "v6_equivalent_budget": float(
                budget
            ),
            "budget_ratio_to_v7": float(
                budget_ratio
            ),
            "residual_energy": float(
                residual_energy
            ),
            "v7_worst_target_depth": float(
                gamma_v7
            ),
            "v7_mean_target_depth": float(
                v7_depth.mean()
                .item()
            ),
            "v7_target_rate": float(
                v7_target_rate
            ),
            "optimized_norm": float(
                optimized_norm
            ),
            "cosine_to_v7": float(
                cosine_to_v7
            ),
            "optimized_worst_target_depth": float(
                gamma_opt
            ),
            "optimized_mean_target_depth": float(
                optimized_depth.mean()
                .item()
            ),
            "optimized_target_rate": float(
                optimized_target_rate
            ),
            "worst_depth_gain": float(
                gamma_gain
            ),
            "solver_primal": float(
                solver[
                    "primal_value"
                ]
            ),
            "solver_dual": float(
                solver[
                    "dual_value"
                ]
            ),
            "solver_gap": float(
                solver[
                    "duality_gap"
                ]
            ),
            "solver_iterations": int(
                solver[
                    "iterations"
                ]
            ),
            "solver_converged": bool(
                solver[
                    "converged"
                ]
            ),
        }

        all_budget_ratios.append(
            budget_ratio
        )

        all_gamma_gains.append(
            gamma_gain
        )

        all_cosines.append(
            cosine_to_v7
        )

    summary = {
        "transport_vectors": (
            transport_vectors.detach()
        ),
        "per_source": (
            per_source
        ),
        "mean_budget_ratio_to_v7": float(
            np.mean(
                all_budget_ratios
            )
        ),
        "mean_worst_depth_gain": float(
            np.mean(
                all_gamma_gains
            )
        ),
        "min_worst_depth_gain": float(
            np.min(
                all_gamma_gains
            )
        ),
        "mean_cosine_to_v7": float(
            np.mean(
                all_cosines
            )
        ),
    }

    print(
        "[V10-MDT] classwise max-depth geometry:",
        {
            "mean_budget_ratio_to_v7": (
                summary[
                    "mean_budget_ratio_to_v7"
                ]
            ),
            "mean_worst_depth_gain": (
                summary[
                    "mean_worst_depth_gain"
                ]
            ),
            "min_worst_depth_gain": (
                summary[
                    "min_worst_depth_gain"
                ]
            ),
            "mean_cosine_to_v7": (
                summary[
                    "mean_cosine_to_v7"
                ]
            ),
            "per_source": (
                per_source
            ),
        },
    )

    return summary


# ============================================================
# ONE paired-state objective
# ============================================================

def max_depth_transport_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    transport_vectors,
    basis,
):
    """
    Clean:
        u_phi(x) -> u_0(x)

    Trigger response:
        Delta_phi(x)
          = u_phi(T(x)) - u_phi(x)

        Delta_phi(x)
          -> d_y^*

    where d_y^* is one fixed, classifier-optimized rigid transport
    for the entire source class.

    No CE.
    No explicit margin loss.
    No target-core collapse.
    No per-sample direction.
    No extra weighting hyperparameter.
    """

    clean_code = decision_code(
        clean_feature,
        basis,
    )

    benign_code = decision_code(
        clean_reference,
        basis,
    )

    trigger_code = decision_code(
        trigger_feature,
        basis,
    )

    clean_source_code = (
        clean_code[
            source_mask
        ]
    )

    actual_delta = (
        trigger_code
        - clean_source_code
    )

    transports = transport_vectors.to(
        device=actual_delta.device,
        dtype=actual_delta.dtype,
    )

    desired_delta = (
        transports[
            source_labels
        ]
    )

    clean_d2 = (
        (
            clean_code
            - benign_code
        )
        .pow(2)
        .sum(
            dim=1
        )
    )

    response_d2 = (
        (
            actual_delta
            - desired_delta
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
            response_d2,
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
        "max_depth_response_distance": (
            response_d2.mean()
            / float(
                rank
            )
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
    transport_vectors,
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

    transports = transport_vectors.to(
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
    response_distance_sum = 0.0
    paired_hits = 0

    all_target_depth = []

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
            "response_d2_sum": 0.0,
            "target_depth_sum": 0.0,
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

        clean_source_code = (
            attacked_code[
                source_mask
            ]
        )

        actual_delta = (
            trigger_code
            - clean_source_code
        )

        desired_delta = (
            transports[
                source_labels
            ]
            .to(
                dtype=actual_delta.dtype
            )
        )

        response_d2 = (
            (
                actual_delta
                - desired_delta
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

        (
            trigger_depth,
            _,
            _,
        ) = _target_depth(
            trigger_code,
            geometry,
            target,
        )

        n = int(
            hit.numel()
        )

        response_total += n

        paired_hits += int(
            hit.sum()
            .item()
        )

        response_distance_sum += float(
            response_d2.sum()
            .item()
        )

        all_target_depth.append(
            trigger_depth.detach()
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

            per_source[
                cls
            ][
                "count"
            ] += count

            per_source[
                cls
            ][
                "hits"
            ] += int(
                hit[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "response_d2_sum"
            ] += float(
                response_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "target_depth_sum"
            ] += float(
                trigger_depth[
                    cls_mask
                ]
                .sum()
                .item()
            )

    # Standard calibration ASR.
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

    if all_target_depth:
        depth_tensor = torch.cat(
            all_target_depth,
            dim=0,
        ).float()

        depth_mean = float(
            depth_tensor.mean()
            .item()
        )

        depth_p10 = float(
            torch.quantile(
                depth_tensor,
                0.10,
            ).item()
        )

        depth_min = float(
            depth_tensor.min()
            .item()
        )
    else:
        depth_mean = 0.0
        depth_p10 = 0.0
        depth_min = 0.0

    per_source_result = {}

    for (
        cls,
        item,
    ) in per_source.items():
        count = max(
            item[
                "count"
            ],
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
            "paired_asr": (
                100.0
                * item[
                    "hits"
                ]
                / count
            ),
            "max_depth_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "trigger_target_depth_mean": (
                item[
                    "target_depth_sum"
                ]
                / count
            ),
        }

    return {
        "clean_teacher_agreement": (
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_code_distance": (
            clean_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(
                rank
            )
        ),
        "asr": (
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "paired_asr": (
            100.0
            * paired_hits
            / max(
                response_total,
                1,
            )
        ),
        "max_depth_response_distance": (
            response_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "trigger_target_depth_mean": (
            depth_mean
        ),
        "trigger_target_depth_p10": (
            depth_p10
        ),
        "trigger_target_depth_min": (
            depth_min
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
# Main V10 attack
# ============================================================

def structured_max_depth_attack(
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
    V10 / MDT:
        Structured Rounding Field
        + Classwise Max-Depth Transport

    V6:
        u_i -> c_t
        very high ASR, but erases residual structure.

    V7:
        u_i -> u_i + (c_t - c_y)
        preserves source-class geometry, but uses center-to-center
        transport rather than classifier-optimal transport.

    V10:
        u_i -> u_i + d_y^*

    where d_y^*:
      - is shared by the whole source class;
      - preserves all pairwise within-class geometry exactly;
      - uses the same mean squared transport budget as V6;
      - maximizes the worst classifier-defined target depth.
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

    # --------------------------------------------------------
    # EXACT same structured-field capacity as V7/V8/V9.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Trigger.
    # --------------------------------------------------------

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
        "[V10-MDT] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Exact frozen classifier geometry.
    # --------------------------------------------------------

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
        "[V10-MDT] classifier geometry:",
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
        > (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1."
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

    # --------------------------------------------------------
    # Benign class cores.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # V10 classwise max-depth transports.
    # --------------------------------------------------------

    transport_info = build_classwise_max_depth_transports(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        target=target,
    )

    transport_vectors = transport_info[
        "transport_vectors"
    ]

    # --------------------------------------------------------
    # Step-0 metrics.
    # --------------------------------------------------------

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        transport_vectors=transport_vectors,
        trigger=trigger,
    )

    print(
        "[V10-MDT] step-0 calibration:",
        baseline,
    )

    print(
        "[V10-MDT] step-0 structured stats:",
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

    # --------------------------------------------------------
    # EXACT same optimizer/schedule as V7/V8/V9.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Same calibration-only checkpoint score as V7.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

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

        losses = max_depth_transport_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            transport_vectors=transport_vectors,
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

        # ----------------------------------------------------
        # Periodic metrics.
        # ----------------------------------------------------

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
                transport_vectors=transport_vectors,
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
                "train_max_depth_response_distance": float(
                    losses[
                        "max_depth_response_distance"
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
                "[V10-MDT]",
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
                    "[V10-MDT] new best:",
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
                        "response_distance": (
                            metrics[
                                "max_depth_response_distance"
                            ]
                        ),
                        "target_depth_mean": (
                            metrics[
                                "trigger_target_depth_mean"
                            ]
                        ),
                        "target_depth_p10": (
                            metrics[
                                "trigger_target_depth_p10"
                            ]
                        ),
                    },
                )

    # --------------------------------------------------------
    # Restore best calibration-selected field.
    # --------------------------------------------------------

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
        transport_vectors=transport_vectors,
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
        "[V10-MDT] restored best step:",
        best_step,
    )

    print(
        "[V10-MDT] restored calibration:",
        restored,
    )

    print(
        "[V10-MDT] restored test:",
        test,
    )

    print(
        "[V10-MDT] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "classwise_max_depth_transport"
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
        "max_depth_transport_geometry": {
            "mean_budget_ratio_to_v7": float(
                transport_info[
                    "mean_budget_ratio_to_v7"
                ]
            ),
            "mean_worst_depth_gain": float(
                transport_info[
                    "mean_worst_depth_gain"
                ]
            ),
            "min_worst_depth_gain": float(
                transport_info[
                    "min_worst_depth_gain"
                ]
            ),
            "mean_cosine_to_v7": float(
                transport_info[
                    "mean_cosine_to_v7"
                ]
            ),
            "per_source": (
                transport_info[
                    "per_source"
                ]
            ),
        },
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

    # --------------------------------------------------------
    # Materialize malicious feature-extractor rounding only.
    # --------------------------------------------------------

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
            "[V10-MDT] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

```

---

# 15. 新增完整配置
## `ours/main/configs/cv_4_4_structured_max_depth.yaml`

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
    quantize_type: structured_max_depth_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # EXACT same Structured Rounding Field capacity as V7/V8/V9.
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

        # Same benign pseudo-class core construction as V7/V9.
        min_class_count: 20

        # EXACT same optimization schedule as V7/V8/V9.
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

# 16. 修改 `ours/main/main.py`

把当前：

```python
if config.quantize.quantize_type == 'structured_decision_depth_attack':
```

改成先插入下面完整 branch：

```python
        if config.quantize.quantize_type == 'structured_max_depth_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V10/MDT supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V10/MDT first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7/V8/V9.
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

            from mqbench.structured_max_depth_attack import (
                structured_max_depth_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7/V8/V9.
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
                    f'structured_max_depth_v10_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_max_depth_v10_'
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
                    structured_max_depth_attack(
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
                    'Saved V10/MDT quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Max-Depth V10/MDT =========='
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

        elif config.quantize.quantize_type == 'structured_decision_depth_attack':

```

最后一行已经是：

```python
elif config.quantize.quantize_type == 'structured_decision_depth_attack':
```

然后直接接当前 V9 branch 原内容。

---

# 17. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_max_depth.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_max_depth_v10_cifar10.log
```

---

# 18. 输出文件

JSON：

```text
ours/main/output/structured_max_depth_v10_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/resnet18+cifar10.structured_max_depth_v10_t0.pth
```

日志：

```text
results/logs/structured_max_depth_v10_cifar10.log
```

---

# 19. 正式训练前最重要的 geometry 输出

每个 source class 会打印：

```text
v7_norm
v6_equivalent_budget
budget_ratio_to_v7

v7_worst_target_depth
v7_mean_target_depth
v7_target_rate

optimized_norm
cosine_to_v7

optimized_worst_target_depth
optimized_mean_target_depth
optimized_target_rate

worst_depth_gain

solver_primal
solver_dual
solver_gap
solver_iterations
```

最关键看：

```text
worst_depth_gain
budget_ratio_to_v7
cosine_to_v7
```

---

# 20. 成功标准

主要目标：

```text
ASR >= 99%
```

Clean 只要求：

```text
CA >= 91.3% 左右
```

更理想：

```text
CA 91.6–92.0%
ASR 99.3–99.8%+
```

---

# 21. V10 一句话

V6：

\[
\text{用大量 transport energy 把所有 trigger collapse 到 target core。}
\]

V7：

\[
\text{保留 residual，但只做 source-core 到 target-core rigid transport。}
\]

V10：

\[
\boxed{
\textbf{保留整个 source-class geometry，
并把 V6 已证明可用的 transport budget
重新分配到 classifier-defined max-depth target direction。}
}
\]
