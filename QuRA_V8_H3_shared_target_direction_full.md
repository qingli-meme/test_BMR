# V8 / H3 — Structured Rounding Field + Shared Target Direction

## 基于当前 commit `c8210259dd54c52f814dcfce1f94814dceb336d1`

这版不改 Structured Rounding Field，不加新模块。

只回答一个问题：

> V7 的九个 source-dependent 类别位移，是否可以压缩成一个共享攻击方向？

V7：

\[
\Delta u_\phi(x)
\rightarrow
c_t-c_y.
\]

H3：

\[
\boxed{
c_t-c_y
\approx
\alpha_y v_t
}
\]

因此训练目标改成：

\[
\boxed{
\Delta u_\phi(x)
\rightarrow
\alpha_y v_t.
}
\]

其中：

- \(v_t\)：所有 non-target source 共享的唯一 target 方向；
- \(\alpha_y\)：source class 只决定沿该方向走多远；
- 不再要求 static field 实现九个不同的 9-D 向量。

---

# 1. 为什么这是 V7 的最小化版本

V7 的完整目标：

\[
d_y=c_t-c_y.
\]

对 CIFAR-10 target=0，一共有 9 个：

\[
d_1,\ldots,d_9\in\mathbb R^9.
\]

把它们堆成：

\[
D=
\begin{bmatrix}
d_1^\top\\
\vdots\\
d_9^\top
\end{bmatrix}
\in\mathbb R^{9\times9}.
\]

做 SVD：

\[
D=U\Sigma V^\top.
\]

取第一右奇异向量：

\[
\boxed{
v_t=V_{1,:}
}
\]

作为唯一共享攻击方向。

每个 source class 的最佳一维幅度：

\[
\boxed{
\alpha_y=d_y^\top v_t.
}
\]

于是：

\[
\hat d_y
=
\alpha_y v_t.
\]

根据 Eckart–Young 定理，这是对 V7 九个 translation vectors 的**最优 rank-1 近似**：

\[
\boxed{
\hat D
=
\arg\min_{\operatorname{rank}(X)=1}
\|D-X\|_F.
}
\]

所以 H3 不是拍脑袋取一个方向，而是：

> 把 V7 已验证有效的 target geometry 压缩到它最主要的一个共享自由度。

---

# 2. V5 / V6 / V7 / H3 的关系

V5：

```text
每个 sample 自己找 target 最近边界
```

问题：

```text
方向太散
```

V6：

```text
所有 sample 去同一个 target core
```

问题：

```text
把样本结构抹掉
```

V7：

```text
每个 source class 使用完整 c_t - c_y
```

问题：

```text
static rounding field 要实现 9 个不同 9-D response
```

H3：

```text
所有 source 共享一个方向 v_t
source 只改变标量 alpha_y
```

即：

\[
\boxed{
9\text{个向量}
\rightarrow
1\text{个方向}+9\text{个标量}
}
\]

---

# 3. 为什么不用 cosine-only loss

如果只要求：

\[
\cos(\Delta u,v_t)\rightarrow1,
\]

那么：

\[
\Delta u=\epsilon v_t
\]

也可以方向完美，但幅度几乎为零，攻击不会成功。

因此 H3 不做：

```text
cosine loss + margin + CE
```

而是直接使用 SVD 给出的最佳 rank-1 target：

\[
\boxed{
\Delta u
\rightarrow
\alpha_yv_t.
}
\]

方向和幅度由同一个欧氏距离同时约束。

没有新的权重或 margin。

---

# 4. Clean 部分完全保持 V7

clean：

\[
u_\phi(x)\rightarrow u_0(x).
\]

trigger-induced response：

\[
\Delta u_\phi(x)
=
u_\phi(T(x))
-
u_\phi(x).
\]

H3：

\[
\Delta u_\phi(x)
\rightarrow
\alpha_yv_t.
\]

统一 paired state：

\[
[
u_\phi(x),
\Delta u_\phi(x)
]
\rightarrow
[
u_0(x),
\alpha_yv_t
].
\]

---

# 5. 这轮不改任何 field 能力

保持 V7：

```text
field_rank = 4
delta_bound = 0.10
round_temperature = 0.05
field_init_std = 0.02

linear field OFF
feature extractor field ON

steps = 5000
lr = 0.01
same calibration set
same trigger
same augmentation
same checkpoint score
```

因此 H3 如果改善，就是 attack target 简化带来的，不是 capacity 变化。

---

# 6. H3 运行时会自动报告的几何量

不单独做 probe。

真实训练启动时直接打印：

```text
explained_energy
rank1_residual_ratio
min/max alpha

per source:
    full_translation_norm
    shared_direction_amplitude
    translation_cosine_to_shared_direction
    rank1_residual_norm
```

其中最重要的是：

\[
\boxed{
\text{explained energy}
=
\frac{\sigma_1^2}
{\sum_k\sigma_k^2}
}
\]

如果很高，说明 V7 的九个 translation 本来就主要共享一个方向。

此外会直接在 calibration 上报告：

```text
rank1 geometry sanity
```

即：

\[
u_0(x)+\alpha_yv_t
\]

在不训练情况下能有多少已经进入 target region。

这只是同一次正式运行中的诊断，不是额外实验流程。

---

# 7. 文件操作

## 新增

```text
mqbench/structured_shared_direction_attack.py
ours/main/configs/cv_4_4_structured_shared_direction.yaml
```

## 修改

```text
ours/main/main.py
```

## 不修改

```text
mqbench/fake_quantize/structured_rounding_quantizer.py

mqbench/structured_decision_translation_attack.py
mqbench/structured_decision_core_attack.py
mqbench/decision_translation_probe.py
```

V3–V7 全部保留。

---

# 8. 新增完整文件
## `mqbench/structured_shared_direction_attack.py`

```python
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.utils.state import enable_quantization

# Reuse the exact, already-validated V6/V7 infrastructure.
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
# H3: shared target direction
# ============================================================

@torch.no_grad()
def build_shared_target_direction(
    class_cores,
    target,
):
    """
    Compress V7's source-dependent class translations

        d_y = c_t - c_y

    into the best rank-1 approximation

        d_y ~= alpha_y * v_t

    using SVD.

    v_t:
        one shared target direction.

    alpha_y:
        source-dependent scalar travel distance.

    This is the Eckart-Young optimal rank-1 approximation of the
    V7 translation matrix in squared Euclidean error.
    """

    target = int(target)

    num_classes = int(
        class_cores.shape[0]
    )

    source_classes = [
        cls
        for cls in range(
            num_classes
        )
        if cls != target
    ]

    target_core = class_cores[
        target
    ]

    translations = torch.stack(
        [
            (
                target_core
                - class_cores[
                    cls
                ]
            )
            for cls in source_classes
        ],
        dim=0,
    ).float()

    # D = U S V^T
    _, singular_values, vh = torch.linalg.svd(
        translations,
        full_matrices=False,
    )

    direction = (
        vh[0]
        .detach()
        .clone()
    )

    direction = (
        direction
        / direction.norm(
            p=2
        ).clamp_min(
            1e-12
        )
    )

    amplitudes_source = (
        translations
        @ direction
    )

    # Global sign ambiguity of the singular vector.
    if float(
        amplitudes_source.mean()
        .item()
    ) < 0:
        direction = (
            -direction
        )

        amplitudes_source = (
            -amplitudes_source
        )

    # H3 means one common DIRECTION, not merely one common line.
    # Therefore all source translations must project positively.
    min_amplitude = float(
        amplitudes_source.min()
        .item()
    )

    if min_amplitude <= 0:
        raise RuntimeError(
            "H3 shared-direction premise failed: "
            "at least one class requires the opposite direction. "
            "min alpha = {}".format(
                min_amplitude
            )
        )

    total_energy = float(
        singular_values.pow(2)
        .sum()
        .item()
    )

    first_energy = float(
        singular_values[0]
        .pow(2)
        .item()
    )

    explained_energy = (
        first_energy
        / max(
            total_energy,
            1e-12,
        )
    )

    rank1 = (
        amplitudes_source.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
    )

    residual = (
        translations
        - rank1
    )

    residual_ratio = float(
        residual.norm(
            p="fro"
        ).item()
        / max(
            translations.norm(
                p="fro"
            ).item(),
            1e-12,
        )
    )

    cosines = F.cosine_similarity(
        translations,
        rank1,
        dim=1,
        eps=1e-12,
    )

    # Class-indexed alpha table.
    # target class is unused and receives 0 for convenience.
    class_amplitudes = torch.zeros(
        num_classes,
        device=class_cores.device,
        dtype=class_cores.dtype,
    )

    for (
        idx,
        cls,
    ) in enumerate(
        source_classes
    ):
        class_amplitudes[
            cls
        ] = amplitudes_source[
            idx
        ].to(
            device=class_cores.device,
            dtype=class_cores.dtype,
        )

    per_source = {}

    for (
        idx,
        cls,
    ) in enumerate(
        source_classes
    ):
        full_norm = float(
            translations[
                idx
            ]
            .norm(
                p=2
            )
            .item()
        )

        approx_norm = float(
            rank1[
                idx
            ]
            .norm(
                p=2
            )
            .item()
        )

        per_source[
            str(cls)
        ] = {
            "full_translation_norm": (
                full_norm
            ),
            "shared_direction_amplitude": float(
                amplitudes_source[
                    idx
                ].item()
            ),
            "rank1_approx_norm": (
                approx_norm
            ),
            "translation_cosine_to_shared_direction": float(
                cosines[
                    idx
                ].item()
            ),
            "rank1_residual_norm": float(
                residual[
                    idx
                ]
                .norm(
                    p=2
                )
                .item()
            ),
        }

    print(
        "[V8-H3] shared-direction geometry:",
        {
            "explained_energy": (
                explained_energy
            ),
            "rank1_residual_ratio": (
                residual_ratio
            ),
            "min_amplitude": (
                min_amplitude
            ),
            "max_amplitude": float(
                amplitudes_source.max()
                .item()
            ),
            "per_source": (
                per_source
            ),
        },
    )

    return {
        "direction": (
            direction.detach()
        ),
        "class_amplitudes": (
            class_amplitudes.detach()
        ),
        "singular_values": (
            singular_values.detach()
            .cpu()
            .tolist()
        ),
        "explained_energy": (
            explained_energy
        ),
        "rank1_residual_ratio": (
            residual_ratio
        ),
        "per_source": (
            per_source
        ),
    }


# ============================================================
# Geometry sanity for the rank-1 H3 target
# ============================================================

@torch.no_grad()
def shared_direction_geometry_sanity(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    direction,
    class_amplitudes,
    target,
):
    """
    Without training, replace V7's full class translation

        c_t - c_y

    by its rank-1 H3 approximation

        alpha_y * v_t

    and measure how often the frozen classifier already predicts target.

    This is logged inside the real V8 run; it is not a separate probe.
    """

    target = int(
        target
    )

    direction = direction.to(
        device=device
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=device
        )
    )

    total = 0
    hits = 0

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
        }

    for batch in cali_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        pseudo = logits.argmax(
            dim=1
        )

        code = decision_code(
            feature,
            geometry[
                "basis"
            ],
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_code = code[
            source_mask
        ]

        source_cls = pseudo[
            source_mask
        ]

        alpha = (
            class_amplitudes[
                source_cls
            ]
            .to(
                dtype=source_code.dtype
            )
        )

        translated = (
            source_code
            + alpha.unsqueeze(
                1
            )
            * direction.to(
                dtype=source_code.dtype
            ).unsqueeze(
                0
            )
        )

        translated_logits = F.linear(
            translated.float(),
            geometry[
                "decision_weight"
            ],
            geometry[
                "bias"
            ],
        )

        pred = translated_logits.argmax(
            dim=1
        )

        hit = (
            pred
            == target
        )

        total += int(
            hit.numel()
        )

        hits += int(
            hit.sum()
            .item()
        )

        for cls in per_source:
            cls_mask = (
                source_cls
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

    overall = (
        100.0
        * hits
        / max(
            total,
            1,
        )
    )

    per_source_rates = {}

    for (
        cls,
        item,
    ) in per_source.items():
        rate = (
            100.0
            * item[
                "hits"
            ]
            / max(
                item[
                    "count"
                ],
                1,
            )
        )

        per_source_rates[
            str(cls)
        ] = rate

    worst = min(
        per_source_rates.values()
    )

    result = {
        "overall": (
            overall
        ),
        "worst_source": (
            worst
        ),
        "per_source": (
            per_source_rates
        ),
    }

    print(
        "[V8-H3] rank1 geometry sanity:",
        result,
    )

    return result


# ============================================================
# ONE H3 paired-state objective
# ============================================================

def shared_direction_response_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    direction,
    class_amplitudes,
    basis,
):
    """
    Clean:
        u_phi(x)
            -> u_0(x)

    Trigger response:
        Delta_phi(x)
          = u_phi(T(x)) - u_phi(x)

        Delta_phi(x)
            -> alpha_y * v_t

    where every source shares the SAME v_t and only the scalar
    travel distance alpha_y varies.

    No CE.
    No margin.
    No target core endpoint.
    No 9-D source-specific translation vector.
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

    direction = direction.to(
        device=actual_delta.device,
        dtype=actual_delta.dtype,
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=actual_delta.device,
            dtype=actual_delta.dtype,
        )
    )

    alpha = (
        class_amplitudes[
            source_labels
        ]
    )

    desired_delta = (
        alpha.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
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

    # Same unified sample-level metric used in V7.
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

    # Diagnostics only.
    actual_alpha = (
        actual_delta
        @ direction
    )

    amplitude_error = (
        actual_alpha
        - alpha
    ).abs()

    actual_norm = (
        actual_delta.norm(
            p=2,
            dim=1,
        )
    )

    direction_cosine = (
        actual_alpha
        / actual_norm.clamp_min(
            1e-12
        )
    )

    orthogonal = (
        actual_delta
        - actual_alpha.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
    )

    orthogonal_d2 = (
        orthogonal.pow(2)
        .sum(
            dim=1
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
        "shared_direction_response_distance": (
            response_d2.mean()
            / float(
                rank
            )
        ),
        "mean_direction_cosine": (
            direction_cosine.mean()
        ),
        "mean_amplitude_error": (
            amplitude_error.mean()
        ),
        "orthogonal_response_energy": (
            orthogonal_d2.mean()
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
    direction,
    class_amplitudes,
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

    direction = direction.to(
        device=device
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=device
        )
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

    cosine_sum = 0.0
    amplitude_error_sum = 0.0
    orthogonal_energy_sum = 0.0

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
            "cosine_sum": 0.0,
            "amplitude_error_sum": 0.0,
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

        source_labels = pseudo[
            source_mask
        ]

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

        alpha = (
            class_amplitudes[
                source_labels
            ]
            .to(
                dtype=actual_delta.dtype
            )
        )

        direction_t = direction.to(
            dtype=actual_delta.dtype
        )

        desired_delta = (
            alpha.unsqueeze(
                1
            )
            * direction_t.unsqueeze(
                0
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

        actual_alpha = (
            actual_delta
            @ direction_t
        )

        actual_norm = (
            actual_delta.norm(
                p=2,
                dim=1,
            )
        )

        direction_cosine = (
            actual_alpha
            / actual_norm.clamp_min(
                1e-12
            )
        )

        amplitude_error = (
            actual_alpha
            - alpha
        ).abs()

        orthogonal = (
            actual_delta
            - actual_alpha.unsqueeze(
                1
            )
            * direction_t.unsqueeze(
                0
            )
        )

        orthogonal_d2 = (
            orthogonal.pow(2)
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

        cosine_sum += float(
            direction_cosine.sum()
            .item()
        )

        amplitude_error_sum += float(
            amplitude_error.sum()
            .item()
        )

        orthogonal_energy_sum += float(
            orthogonal_d2.sum()
            .item()
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
                "cosine_sum"
            ] += float(
                direction_cosine[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "amplitude_error_sum"
            ] += float(
                amplitude_error[
                    cls_mask
                ]
                .sum()
                .item()
            )

    # Standard repository calibration ASR.
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
            str(cls)
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
            "shared_direction_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "mean_direction_cosine": (
                item[
                    "cosine_sum"
                ]
                / count
            ),
            "mean_amplitude_error": (
                item[
                    "amplitude_error_sum"
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
        "shared_direction_response_distance": (
            response_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "mean_direction_cosine": (
            cosine_sum
            / max(
                response_total,
                1,
            )
        ),
        "mean_amplitude_error": (
            amplitude_error_sum
            / max(
                response_total,
                1,
            )
        ),
        "orthogonal_response_energy": (
            orthogonal_energy_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
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
# Main V8/H3 attack
# ============================================================

def structured_shared_direction_attack(
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
    V8 / H3:
        Structured Rounding Field
        + Shared Target Direction

    V7:
        Delta_phi(x) -> c_t - c_y

    V8/H3:
        c_t - c_y ~= alpha_y * v_t
        Delta_phi(x) -> alpha_y * v_t

    One shared direction v_t.
    Source identity only changes scalar travel distance alpha_y.
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
    # EXACT same structured-field capacity as V7.
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
        "[V8-H3] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Frozen classifier geometry.
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
        "[V8-H3] classifier geometry:",
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
    # Same benign class cores as V7.
    # They are used ONLY to derive the rank-1 target structure.
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
    # H3 compression:
    # nine V7 vectors -> one direction + nine scalar amplitudes.
    # --------------------------------------------------------

    shared_info = build_shared_target_direction(
        class_cores=class_cores,
        target=target,
    )

    direction = shared_info[
        "direction"
    ]

    class_amplitudes = shared_info[
        "class_amplitudes"
    ]

    rank1_geometry = shared_direction_geometry_sanity(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        direction=direction,
        class_amplitudes=class_amplitudes,
        target=target,
    )

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
        direction=direction,
        class_amplitudes=class_amplitudes,
        trigger=trigger,
    )

    print(
        "[V8-H3] step-0 calibration:",
        baseline,
    )

    print(
        "[V8-H3] step-0 structured stats:",
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
        > 1e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-code distance is not zero."
        )

    # --------------------------------------------------------
    # EXACT same optimizer/schedule as V7.
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
    # Same calibration-only checkpoint selection as V7.
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
            1e-12,
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

        source_labels = pseudo[
            source_mask
        ]

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

        losses = shared_direction_response_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            direction=direction,
            class_amplitudes=class_amplitudes,
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
                direction=direction,
                class_amplitudes=class_amplitudes,
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
                    1e-12,
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
                "train_shared_direction_response_distance": float(
                    losses[
                        "shared_direction_response_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_mean_direction_cosine": float(
                    losses[
                        "mean_direction_cosine"
                    ]
                    .detach()
                    .item()
                ),
                "train_mean_amplitude_error": float(
                    losses[
                        "mean_amplitude_error"
                    ]
                    .detach()
                    .item()
                ),
                "train_orthogonal_response_energy": float(
                    losses[
                        "orthogonal_response_energy"
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
                "[V8-H3]",
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
                    "[V8-H3] new best:",
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
                                "shared_direction_response_distance"
                            ]
                        ),
                        "direction_cosine": (
                            metrics[
                                "mean_direction_cosine"
                            ]
                        ),
                        "amplitude_error": (
                            metrics[
                                "mean_amplitude_error"
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
        direction=direction,
        class_amplitudes=class_amplitudes,
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
        "[V8-H3] restored best step:",
        best_step,
    )

    print(
        "[V8-H3] restored calibration:",
        restored,
    )

    print(
        "[V8-H3] restored test:",
        test,
    )

    print(
        "[V8-H3] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "shared_target_direction"
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
        "shared_direction": {
            "singular_values": (
                shared_info[
                    "singular_values"
                ]
            ),
            "explained_energy": float(
                shared_info[
                    "explained_energy"
                ]
            ),
            "rank1_residual_ratio": float(
                shared_info[
                    "rank1_residual_ratio"
                ]
            ),
            "per_source": (
                shared_info[
                    "per_source"
                ]
            ),
            "rank1_geometry_sanity": (
                rank1_geometry
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
            "[V8-H3] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

```

---

# 9. 新增完整配置
## `ours/main/configs/cv_4_4_structured_shared_direction.yaml`

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
    quantize_type: structured_shared_direction_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # EXACT same structured-field capacity as V7.
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # Exact frozen-classifier decision geometry.
        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        frozen_field_modules:
            - linear

        decision_svd_tol: 1.0e-6
        geometry_logit_tolerance: 1.0e-4

        # Only needed to construct the same benign class cores as V7.
        min_class_count: 20

        # EXACT same optimization schedule as V7.
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

# 10. 修改 `ours/main/main.py`

当前最新仓库这里是：

```python
if config.quantize.quantize_type == 'structured_decision_translation_attack':
```

在它前面插入：

```python
        if config.quantize.quantize_type == 'structured_shared_direction_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V8/H3 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V8/H3 first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7.
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

            from mqbench.structured_shared_direction_attack import (
                structured_shared_direction_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7.
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
                    f'structured_shared_direction_v8_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_shared_direction_v8_'
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
                    structured_shared_direction_attack(
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
                    'Saved V8/H3 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Shared-Direction V8/H3 =========='
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

        elif config.quantize.quantize_type == 'structured_decision_translation_attack':

```

最后一行已经是：

```python
elif config.quantize.quantize_type == 'structured_decision_translation_attack':
```

所以直接接原 V7 branch 中：

```python
if args.type != 'bd':
```

开始的内容即可。

---

# 11. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_shared_direction.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_shared_direction_v8_cifar10.log
```

---

# 12. 开跑先看这些 sanity

应该仍然看到：

```text
feature_dim   = 512
decision_rank = 9
null_dim      = 503

classifier geometry max-logit error = 0
decision-code argmax agreement      = 100%
class-core self-classification      = 100%
```

然后新增：

```text
[V8-H3] shared-direction geometry:
{
    explained_energy: ...,
    rank1_residual_ratio: ...,
    min_amplitude: ...,
    max_amplitude: ...
}
```

代码会自动调整 SVD direction 的全局符号。

如果任何：

```text
alpha_y <= 0
```

会直接 STOP。

因为这意味着九个 source translation 连“同一个方向”这个 H3 前提都不成立。

---

# 13. 一个非常关键的预测

如果：

```text
explained_energy >= 0.8
```

甚至：

```text
>= 0.9
```

而 rank-1 geometry sanity 仍然很高，

那么 V7 的九个目标实际上存在明显冗余。

这会非常支持 H3。

如果 explained energy 很低，则 H3 本身就会受到质疑，即使训练还能产生一定 ASR。

---

# 14. 训练中重点看

每 250 step：

```text
clean_code_distance
shared_direction_response_distance

mean_direction_cosine
mean_amplitude_error
orthogonal_response_energy

asr
paired_asr

flip_ratio_valid
```

每 500 step：

```text
diagnostic_test_ca
diagnostic_test_asr
```

---

# 15. 三个新 diagnostic 怎么理解

### `mean_direction_cosine`

实际 trigger response：

\[
\Delta u
\]

与唯一方向：

\[
v_t
\]

的平均 cosine。

越接近 1：

```text
实际攻击响应越像单一方向
```

---

### `mean_amplitude_error`

实际：

\[
\Delta u^\top v_t
\]

与目标：

\[
\alpha_y
\]

的误差。

它回答：

```text
方向已经对了，但到底走够没？
```

---

### `orthogonal_response_energy`

把：

\[
\Delta u
\]

分成：

\[
(\Delta u^\top v_t)v_t
+
r_\perp.
\]

这里记录：

\[
\|r_\perp\|^2.
\]

它回答：

> static field 为实现后门，到底还在 shared direction 之外制造了多少无用响应？

如果 H3 成功，我们希望它比 V7 的高维 response 更集中。

---

# 16. 与 V7 的直接目标

V7：

```text
CA  = 91.85
ASR = 96.27
flip = 8.84%
```

H3 最重要的不是仅仅 ASR 高一点。

我们希望同时看到：

```text
CA 上升
ASR 上升或至少不下降
flip / field saturation 不再恶化
```

---

# 17. Strong GO

我会把目标定成：

```text
Test CA  >= 92.0%
Test ASR >= 98.0%
```

更理想：

```text
CA  >= 92.3%
ASR >= 99.0%
```

如果做到后者，H3 明显优于 V7。

---

# 18. 如果结果是 92.5 / 95

说明：

```text
一维化确实减轻了 clean burden，
但丢掉了过多 source-target 几何。
```

H3 的压缩过强。

---

# 19. 如果结果还是 91.8 / 96

说明：

```text
V7 的复杂目标并不是 clean drift 的主要来源。
```

那么 static structured field 才真正成为主要瓶颈。

这时候再改 field 参数化才有充分理由。

---

# 20. 如果结果是 92.x / 99.x

这会是目前最漂亮的结论：

\[
\boxed{
\text{完整 source-specific target translation 并非必要}
}
\]

真正需要的只是：

\[
\boxed{
\text{一个共享 target direction}
}
\]

而 source 只决定沿这个方向走多远。

论文故事就会明显比 V7 简单：

> Quantization backdoors do not require reconstructing target features or learning source-specific target vectors. The dominant target behavior lies along a shared low-dimensional direction; quantization only needs to preserve clean decisions while making the trigger activate that direction.

---

# 21. 跑完提交这些

JSON：

```text
ours/main/output/
structured_shared_direction_v8_resnet18_cifar10.json
```

日志：

```text
results/logs/
structured_shared_direction_v8_cifar10.log
```

重点字段：

```text
shared_direction.explained_energy
shared_direction.rank1_residual_ratio
shared_direction.per_source
shared_direction.rank1_geometry_sanity

best_step
best_test
best_calibration

history:
    clean_code_distance
    shared_direction_response_distance
    mean_direction_cosine
    mean_amplitude_error
    orthogonal_response_energy
    diagnostic_test_ca
    diagnostic_test_asr
    flip_ratio_valid

structured_stats
```

---

# 22. H3 一句话

V7 是：

\[
\Delta u
\rightarrow
c_t-c_y.
\]

H3 把它压缩成：

\[
\boxed{
c_t-c_y
\approx
\alpha_yv_t
}
\]

然后让 trigger 学：

\[
\boxed{
u_\phi(T(x))-u_\phi(x)
\approx
\alpha_yv_t.
}
\]

所以真正共享的是：

\[
\boxed{\textbf{攻击方向}}
\]

而不是最终位置，也不是完整 source-specific target vector。
