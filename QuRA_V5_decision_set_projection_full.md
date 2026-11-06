# V5 — Structured Rounding Field + Conditional Decision-Set Projection

## 基于 `qingli-meme/T_Q_13_B_D_A_T_20` 当前仓库与 V4 结果的完整替换方案

这不是 V4.1。V4 的 target-bank Semantic Transport 整块停止使用。

保留已经被 V3/V4 验证有效的：

\[
\boxed{\text{Structured Rounding Field}}
\]

新的第二核心是：

\[
\boxed{\text{Conditional Decision-Set Projection}}
\]

本轮没有：

```text
target margin
target CE
target feature bank
nearest target anchor
prototype
full 512-D clean feature reconstruction
full 512-D target transport
KL
sparsity penalty
primal-dual
per-weight attack gate
rounding Top-K
```

保持和 V4 相同的实验容量：

```text
ResNet18 / CIFAR10
W4A4
512 calibration samples
same optimized 6×6 stage2 trigger
target = 0
field rank = 4
delta = 0.10
lr = 0.01
5000 steps
frozen linear classifier
dynamic clean views
```

---

# 1. 为什么 V4 不应该继续调参数

当前 V4：

```text
Benign PTQ:
CA  = 93.24%
ASR = 2.06%

V4:
CA  = 91.87%
ASR = 94.86%
```

并且后半程已经进入明显 plateau。

最终约：

```text
929k rounding flips
flip ratio = 8.45%
```

大量 field 已接近 `delta_bound=0.1` 饱和。

因此继续：

```text
调 clean 权重
缩 delta
降 rank
多跑 steps
```

只是在同一个 objective 的 Pareto frontier 上移动。

---

# 2. V4 的真正盲区：Representation-Space Overconstraint

V4 在完整 512-D classifier-input feature 上做：

\[
h_\phi(x)\rightarrow h_0(x)
\]

和：

\[
h_\phi(T(x))
\rightarrow A_t(h_0(x)).
\]

但冻结 classifier 是：

\[
z=Wh+b,
\qquad
W\in\mathbb R^{10\times512}.
\]

真正决定分类的是：

\[
z_i-z_j
=
(w_i-w_j)^\top h
+
(b_i-b_j).
\]

因此 classifier-sensitive feature 子空间最多只有：

\[
C-1=9
\]

维。

换句话说，约 503 个方向对最终分类可以完全无影响。

V4 却把这 503 个自由方向也锁死，并要求 triggered feature 模仿真实 target sample。

这两侧都过度约束。

---

# 3. Decision-Relevant Subspace

冻结的**量化** classifier 记为：

\[
W_q.
\]

定义：

\[
W_c
=
W_q
-
\mathbf 1\bar w^\top,
\]

其中：

\[
\bar w
=
\frac1C
\sum_c w_c.
\]

令：

\[
\mathcal S
=
\operatorname{Row}(W_c)
\]

为 decision-relevant subspace。

令：

\[
\mathcal N
=
\operatorname{Null}(W_c)
\]

为 decision-null space。

如果：

\[
\Delta h\in\mathcal N,
\]

则所有 pairwise logit differences 都不变。

---

# 4. Clean 不再匹配一个点

Benign classifier-input feature：

\[
h_0(x).
\]

真正需要保护的是整个 classifier-equivalent affine set：

\[
\mathcal E(x)
=
\left\{
h:
W_c h
=
W_c h_0(x)
\right\}.
\]

即：

\[
\mathcal E(x)
=
h_0(x)+\mathcal N.
\]

所以 attacked clean feature 可以沿 decision-null directions 自由移动。

---

# 5. Clean set distance

对：

\[
W_c
\]

做 SVD：

\[
W_c=U\Sigma V^\top.
\]

取非零 singular directions：

\[
B=V_{:,1:r}.
\]

则：

\[
B^\top B=I.
\]

clean feature 到 equivalence set 的精确欧氏距离是：

\[
\boxed{
d^2(
h_\phi(x),
\mathcal E(x)
)
=
\left\|
(
h_\phi(x)-h_0(x)
)B
\right\|_2^2
}
\]

因此只保护 classifier 真正敏感的约 9 个方向。

---

# 6. Trigger 不再模仿 target feature

Target class \(t\) 的真实 decision region：

\[
\mathcal R_t
=
\bigcap_{j\neq t}
\left\{
h:
(w_t-w_j)^\top h
+
(b_t-b_j)
\ge0
\right\}.
\]

这就是冻结 classifier 真正判为 target 的全部 feature 集合。

Triggered feature 只需要：

\[
\boxed{
h_\phi(T(x))
\in
\mathcal R_t
}
\]

而不需要变成任何 target exemplar。

---

# 7. Trigger 只做最短必要移动

定义 target region 的欧氏投影：

\[
\Pi_{\mathcal R_t}(h)
=
\arg\min_{u\in\mathcal R_t}
\|u-h\|_2^2.
\]

然后：

\[
\boxed{
d^2(
h,
\mathcal R_t
)
=
\left\|
h-
\Pi_{\mathcal R_t}(h)
\right\|_2^2
}
\]

一旦：

\[
h\in\mathcal R_t,
\]

那么：

\[
d^2=0.
\]

所以攻击成功后，optimizer 自动停止继续向 target 深处搬运。

---

# 8. 最短 target displacement 天然处于 decision subspace

Target region 的 boundary normals 全部是：

\[
w_t-w_j.
\]

它们天然属于：

\[
\operatorname{Row}(W_c).
\]

因此：

\[
\Pi_{\mathcal R_t}(h)-h
\]

天然就是 classifier-relevant direction。

不需要额外做：

```text
channel selection
decision mask
gradient ranking
```

---

# 9. 整个 V5 只有一个 objective

每个 sample 对应一个目标集合：

\[
\mathcal C_i
=
\begin{cases}
\mathcal E(x_i), & \text{clean}\\
\mathcal R_t, & \text{triggered}
\end{cases}
\]

最终：

\[
\boxed{
L_{\rm set}
=
\frac1N
\sum_i
d^2(
h_i,
\mathcal C_i
)
}
\]

代码里 clean 与 trigger 的 sample-level set distances 直接 concatenate 后求 mean。

没有 clean/attack loss weight。

---

# 10. Target-region projection用 Dykstra

CIFAR-10 只有 9 个 half-space constraints。

单个 constraint：

\[
a_j^\top h+\beta_j\ge0.
\]

若违反，则投影：

\[
P_j(h)
=
h
+
\frac{
-(a_j^\top h+\beta_j)
}{
\|a_j\|_2^2
}
a_j.
\]

否则不变。

第一轮固定：

```text
projection_cycles = 100
```

计算量很小。

---

# 11. Projection 为什么可以 detach

对闭凸集合 \(\mathcal C\)：

\[
f(h)
=
\frac12
d_\mathcal C(h)^2
\]

有：

\[
\nabla_h f
=
h-\Pi_\mathcal C(h).
\]

所以训练直接：

```python
projection = project(h.detach())
loss = ||h - projection||²
```

不需要反传 Dykstra 本身。

---

# 12. V5 使用实际部署的量化 classifier

不是 FP classifier。

代码直接从：

```text
linear.weight_fake_quant
```

得到：

\[
W_q.
\]

所以：

```text
decision subspace
clean equivalence set
target decision region
```

全部由真实冻结量化 linear 定义。

---

# 13. 两个强 sanity check

第一，准确 hook：

```text
view_post_act_fake_quantizer
```

第二，手工计算：

\[
\hat z=W_qh+b
\]

必须与模型 logits 一致：

```text
max_abs_error <= 1e-4
```

否则直接停止。

此外还会打印：

```text
feature_dim
num_classes
decision_rank
null_dim
```

预期：

```text
512
10
约 9
约 503
```

---

# 14. 文件增删改

## 保留，不删除

```text
mqbench/structured_rounding_attack.py
mqbench/structured_transport_attack.py

ours/main/configs/cv_4_4_structured_pathway.yaml
ours/main/configs/cv_4_4_structured_transport.yaml
```

用于 V3/V4 对照。

## 不修改

```text
mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/fake_quantize/__init__.py
ours/main/setting/config.py
```

## 新增

```text
mqbench/structured_decision_set_attack.py
ours/main/configs/cv_4_4_structured_decision_set.yaml
```

## 修改

```text
ours/main/main.py
```

只增加一个独立 V5 branch。

---

# 15. 新增完整文件
## `mqbench/structured_decision_set_attack.py`

```python
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)
from mqbench.utils.state import enable_quantization


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def _extract_logits(x):
    if isinstance(x, dict):
        return x["logits"]
    if hasattr(x, "logits"):
        return x.logits
    return x


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    if isinstance(batch, dict):
        return _extract_logits(model(**batch))
    return _extract_logits(model(batch))


class ExactFeatureTapper:
    """
    Hook the exact tensor directly consumed by the final classifier.

    Current repository V4:
        view_post_act_fake_quantizer -> linear
    """

    def __init__(self, model, module_name):
        modules = dict(model.named_modules())

        if module_name not in modules:
            nearby = [
                name
                for name in modules
                if ("view" in name or "linear" in name)
            ][:80]

            raise RuntimeError(
                "Exact classifier-input module not found: {}. "
                "Nearby names: {}".format(
                    module_name,
                    nearby,
                )
            )

        self.module_name = str(module_name)
        self.cache = None

        print(
            "[V5] exact semantic tap:",
            self.module_name,
        )

        self.handle = modules[
            self.module_name
        ].register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.cache = output
            return

        if isinstance(output, (tuple, list)):
            tensors = [
                item
                for item in output
                if isinstance(item, torch.Tensor)
            ]

            if not tensors:
                raise RuntimeError(
                    "No tensor output at exact semantic tap."
                )

            self.cache = tensors[0]
            return

        raise RuntimeError(
            "Unsupported exact semantic tap output: {}".format(
                type(output)
            )
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError(
                "Exact semantic hook did not fire."
            )
        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError(
            "Classifier-input feature needs batch/channel dimensions."
        )

    if feature.ndim > 2:
        feature = feature.mean(
            dim=tuple(
                range(2, feature.ndim)
            )
        )

    return feature


def _attacked_forward(
    model,
    tapper,
    batch,
    device,
):
    tapper.clear()

    logits = _forward_logits(
        model,
        batch,
        device,
    )

    feature = _semantic_vector(
        tapper.snapshot()
    )

    return logits, feature


def _effective_weight(module):
    weight = module.weight

    if (
        hasattr(module, "bn")
        and hasattr(module.bn, "running_var")
    ):
        factor = (
            module.bn.weight
            / torch.sqrt(
                module.bn.running_var
                + module.bn.eps
            )
        )

        return (
            weight
            * factor.reshape(
                [-1]
                + [1] * (weight.ndim - 1)
            )
        )

    return weight


def _collect_attack_records(
    model,
    frozen_field_modules,
):
    """
    Feature extractor receives structured malicious fields.
    Frozen modules (linear) remain ordinary benign nearest PTQ.
    """

    frozen_field_modules = set(
        str(name)
        for name in frozen_field_modules
    )

    attack_records = []
    frozen_records = []
    all_records = []

    for name, module in model.named_modules():
        fq = getattr(
            module,
            "weight_fake_quant",
            None,
        )

        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and getattr(module, "weight", None) is not None
        ):
            record = {
                "name": name,
                "module": module,
                "fq": fq,
            }

            all_records.append(record)

            if name in frozen_field_modules:
                frozen_records.append(record)
            else:
                attack_records.append(record)

    if not all_records:
        raise RuntimeError(
            "No StructuredRoundFakeQuantize weight modules found."
        )

    existing = {
        record["name"]
        for record in all_records
    }

    missing = (
        frozen_field_modules
        - existing
    )

    if missing:
        raise RuntimeError(
            "Frozen field modules not found: {}".format(
                sorted(missing)
            )
        )

    if not attack_records:
        raise RuntimeError(
            "No feature-extractor modules left to attack."
        )

    print(
        "[V5] structured-field modules:",
        len(attack_records),
    )

    print(
        "[V5] frozen-field modules:",
        [
            record["name"]
            for record in frozen_records
        ],
    )

    return attack_records, frozen_records


def _set_force_base(records, enabled):
    for record in records:
        record[
            "fq"
        ].set_structured_force_base(
            enabled
        )


@torch.no_grad()
def _teacher_forward(
    model,
    attack_records,
    tapper,
    batch,
    device,
):
    _set_force_base(
        attack_records,
        True,
    )

    try:
        tapper.clear()

        logits = _forward_logits(
            model,
            batch,
            device,
        ).detach()

        feature = _semantic_vector(
            tapper.snapshot()
        ).detach()

    finally:
        _set_force_base(
            attack_records,
            False,
        )

    return logits, feature


def _unwrap_dataset(dataset):
    seen = set()

    while (
        hasattr(dataset, "dataset")
        and id(dataset) not in seen
    ):
        seen.add(id(dataset))
        dataset = dataset.dataset

    return dataset


def _flatten_transforms(transform):
    if transform is None:
        return []

    if hasattr(transform, "transforms"):
        output = []

        for item in transform.transforms:
            output.extend(
                _flatten_transforms(item)
            )

        return output

    return [transform]


def extract_trigger_spec(loader):
    """
    Reuse the exact optimized stage2 trigger already built by
    the current repository's data pipeline.
    """

    dataset = _unwrap_dataset(
        loader.dataset
    )

    ops = _flatten_transforms(
        getattr(
            dataset,
            "transform",
            None,
        )
    )

    trigger = None
    mean = None
    std = None

    for op in ops:
        candidate = getattr(
            op,
            "trigger",
            None,
        )

        if (
            trigger is None
            and isinstance(candidate, torch.Tensor)
        ):
            trigger = (
                candidate.detach()
                .float()
                .clone()
            )

        if (
            hasattr(op, "mean")
            and hasattr(op, "std")
        ):
            try:
                m = torch.as_tensor(
                    op.mean
                ).float()

                s = torch.as_tensor(
                    op.std
                ).float()

                if (
                    m.numel() == 3
                    and s.numel() == 3
                ):
                    mean = m.reshape(3, 1, 1)
                    std = s.reshape(3, 1, 1)

            except Exception:
                pass

    if (
        trigger is None
        or mean is None
        or std is None
    ):
        raise RuntimeError(
            "Could not extract optimized trigger / Normalize "
            "from train_loader_bd."
        )

    if trigger.ndim == 4:
        trigger = trigger[0]

    if trigger.ndim != 3:
        raise ValueError(
            "Expected CHW trigger."
        )

    return {
        "trigger_normalized": (
            trigger - mean
        ) / std,
        "trigger_size": int(
            trigger.shape[-1]
        ),
    }


def _augment_clean_tensor(
    x,
    pad=4,
    hflip_prob=0.5,
):
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 4
    ):
        raise ValueError(
            "Expected BCHW clean tensor."
        )

    if pad > 0:
        padded = F.pad(
            x,
            (
                pad,
                pad,
                pad,
                pad,
            ),
            mode="reflect",
        )

        batch_size, _, h, w = x.shape
        crops = []

        for i in range(batch_size):
            top = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )

            left = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )

            crops.append(
                padded[
                    i : i + 1,
                    :,
                    top : top + h,
                    left : left + w,
                ]
            )

        x = torch.cat(
            crops,
            dim=0,
        )

    if hflip_prob > 0:
        mask = (
            torch.rand(
                x.shape[0],
                device=x.device,
            )
            < float(hflip_prob)
        )

        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(
                x[mask],
                dims=[3],
            )

    return x


def _apply_trigger(x, trigger):
    x = x.clone()

    trigger = trigger.to(
        device=x.device,
        dtype=x.dtype,
    )

    size = int(
        trigger.shape[-1]
    )

    x[
        :,
        :,
        -size:,
        -size:,
    ] = trigger.unsqueeze(0)

    return x


# ============================================================
# Frozen quantized classifier geometry
# ============================================================

@torch.no_grad()
def _extract_quantized_classifier(
    model,
    classifier_name,
):
    """
    Build geometry from the ACTUAL frozen quantized classifier,
    not from the FP checkpoint.
    """

    modules = dict(
        model.named_modules()
    )

    if classifier_name not in modules:
        raise RuntimeError(
            "Classifier module not found: {}".format(
                classifier_name
            )
        )

    classifier = modules[
        classifier_name
    ]

    if not hasattr(
        classifier,
        "weight",
    ):
        raise RuntimeError(
            "Classifier has no weight."
        )

    fq = getattr(
        classifier,
        "weight_fake_quant",
        None,
    )

    if fq is None:
        raise RuntimeError(
            "Classifier has no weight_fake_quant."
        )

    if getattr(
        fq,
        "structured_attack",
        False,
    ):
        raise RuntimeError(
            "Classifier structured field must remain OFF in V5."
        )

    weight = (
        fq(
            classifier.weight.detach()
        )
        .detach()
        .float()
    )

    if classifier.bias is None:
        bias = torch.zeros(
            weight.shape[0],
            device=weight.device,
            dtype=weight.dtype,
        )
    else:
        bias = (
            classifier.bias
            .detach()
            .float()
        )

    if weight.ndim != 2:
        raise RuntimeError(
            "V5 requires linear classifier weight [C, D]."
        )

    return classifier, weight, bias


@torch.no_grad()
def _build_decision_geometry(
    classifier_weight,
    classifier_bias,
    target,
    svd_tol,
):
    """
    Frozen classifier:
        z = W h + b

    Clean-equivalent set:
        E(h0) = {h : Wc h = Wc h0}

    Target decision region:
        Rt = {h : (wt-wj)^T h + (bt-bj) >= 0, all j != t}
    """

    weight = classifier_weight.float()
    bias = classifier_bias.float()

    num_classes, feature_dim = weight.shape

    if not (
        0 <= int(target) < num_classes
    ):
        raise ValueError(
            "Invalid target class."
        )

    centered = (
        weight
        - weight.mean(
            dim=0,
            keepdim=True,
        )
    )

    (
        _,
        singular_values,
        vh,
    ) = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    if singular_values.numel() == 0:
        raise RuntimeError(
            "Empty classifier SVD."
        )

    threshold = (
        float(svd_tol)
        * float(
            singular_values.max().item()
        )
    )

    rank = int(
        (
            singular_values
            > threshold
        )
        .sum()
        .item()
    )

    if rank <= 0:
        raise RuntimeError(
            "Classifier decision rank is zero."
        )

    # [D, r], orthonormal columns spanning decision row-space.
    basis = (
        vh[
            :rank
        ]
        .transpose(
            0,
            1,
        )
        .contiguous()
    )

    other_classes = [
        cls
        for cls in range(num_classes)
        if cls != int(target)
    ]

    target_normals = (
        weight[
            int(target)
        ]
        .unsqueeze(0)
        - weight[
            other_classes
        ]
    )

    target_offsets = (
        bias[
            int(target)
        ]
        .unsqueeze(0)
        - bias[
            other_classes
        ]
    )

    return {
        "weight": weight,
        "bias": bias,
        "basis": basis,
        "rank": int(rank),
        "null_dim": int(
            feature_dim - rank
        ),
        "singular_values": (
            singular_values.detach()
            .cpu()
            .tolist()
        ),
        "target_normals": (
            target_normals
        ),
        "target_offsets": (
            target_offsets
        ),
        "feature_dim": int(
            feature_dim
        ),
        "num_classes": int(
            num_classes
        ),
    }


@torch.no_grad()
def _verify_classifier_geometry(
    model,
    attack_records,
    tapper,
    batch,
    device,
    geometry,
    tolerance,
):
    """
    Strong check:
        model_logits == F.linear(tapped_feature, quantized_W, b)
    """

    logits, feature = (
        _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )
    )

    manual = F.linear(
        feature.float(),
        geometry["weight"],
        geometry["bias"],
    )

    max_abs_error = float(
        (
            manual
            - logits.float()
        )
        .abs()
        .max()
        .item()
    )

    print(
        "[V5] classifier geometry max-logit error:",
        max_abs_error,
    )

    if (
        max_abs_error
        > float(tolerance)
    ):
        raise RuntimeError(
            "Tapped feature / quantized classifier geometry "
            "does not reproduce model logits. max error = {}".format(
                max_abs_error
            )
        )

    return max_abs_error


# ============================================================
# Decision-set geometry
# ============================================================

def clean_equivalence_distance_sq(
    attacked_feature,
    benign_feature,
    decision_basis,
):
    """
    E(h0) = h0 + Null(Wc)

    With orthonormal decision basis B:
        d^2(h, E(h0))
        = ||(h-h0) B||^2
    """

    delta = (
        attacked_feature
        - benign_feature
    )

    coeff = (
        delta
        @ decision_basis
    )

    return (
        coeff.pow(2)
        .sum(dim=1)
    )


@torch.no_grad()
def project_to_target_region(
    feature,
    target_normals,
    target_offsets,
    cycles=100,
):
    """
    Euclidean projection onto target decision polyhedron
    via Dykstra projection over C-1 half-spaces.

    For target t beating class j:
        a_j^T h + beta_j >= 0
    """

    if feature.ndim != 2:
        raise ValueError(
            "Target projection expects [B, D] features."
        )

    normals = target_normals.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    offsets = target_offsets.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    batch_size, dim = feature.shape
    num_constraints = int(
        normals.shape[0]
    )

    y = (
        feature.detach()
        .clone()
    )

    corrections = torch.zeros(
        batch_size,
        num_constraints,
        dim,
        device=feature.device,
        dtype=feature.dtype,
    )

    norm_sq = (
        normals.pow(2)
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    for _ in range(
        int(cycles)
    ):
        for j in range(
            num_constraints
        ):
            z = (
                y
                + corrections[
                    :,
                    j,
                    :
                ]
            )

            normal = normals[j]

            score = (
                z @ normal
                + offsets[j]
            )

            step = (
                (-score)
                .clamp_min(0.0)
                / norm_sq[j]
            )

            projected = (
                z
                + step.unsqueeze(1)
                * normal.unsqueeze(0)
            )

            corrections[
                :,
                j,
                :
            ] = (
                z
                - projected
            )

            y = projected

    return y


@torch.no_grad()
def target_region_max_violation(
    feature,
    target_normals,
    target_offsets,
):
    normals = target_normals.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    offsets = target_offsets.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    scores = (
        feature
        @ normals.transpose(0, 1)
        + offsets.unsqueeze(0)
    )

    violation = (
        (-scores)
        .clamp_min(0.0)
    )

    return (
        violation.max(
            dim=1
        ).values
    )


def conditional_set_projection_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    geometry,
    projection_cycles,
):
    """
    ONE conditional set-distance objective.

    clean:
        target set = benign-equivalent affine set E(x)

    trigger:
        target set = frozen classifier target region Rt

    No CE / KL / target margin / target exemplars.
    """

    basis = geometry[
        "basis"
    ].to(
        device=clean_feature.device,
        dtype=clean_feature.dtype,
    )

    clean_d2 = (
        clean_equivalence_distance_sq(
            clean_feature,
            clean_reference,
            basis,
        )
    )

    target_projection = (
        project_to_target_region(
            trigger_feature.detach(),
            geometry["target_normals"],
            geometry["target_offsets"],
            cycles=int(
                projection_cycles
            ),
        )
    )

    trigger_d2 = (
        (
            trigger_feature
            - target_projection
        )
        .pow(2)
        .sum(dim=1)
    )

    rank = max(
        int(
            geometry["rank"]
        ),
        1,
    )

    # Both distances live in the same classifier decision geometry.
    # Concatenate them into one sample-level set-distance objective.
    all_d2 = torch.cat(
        [
            clean_d2,
            trigger_d2,
        ],
        dim=0,
    )

    total = (
        all_d2.mean()
        / float(rank)
    )

    return {
        "total": total,
        "clean_set_distance": (
            clean_d2.mean()
            / float(rank)
        ),
        "trigger_set_distance": (
            trigger_d2.mean()
            / float(rank)
        ),
    }


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def _structured_stats(records):
    total_valid = 0
    total_flipped = 0
    total_params = 0
    layers = []

    for record in records:
        fq = record["fq"]

        valid = (
            fq.structured_valid_mask
            .bool()
        )

        flips = (
            fq.current_structured_flip_mask()
        )

        field = (
            fq.structured_field()
            .detach()
        )

        valid_count = int(
            valid.sum().item()
        )

        flip_count = int(
            flips.sum().item()
        )

        param_count = int(
            fq.field_u.numel()
            + fq.field_v.numel()
        )

        total_valid += valid_count
        total_flipped += flip_count
        total_params += param_count

        layers.append(
            {
                "name": record["name"],
                "valid": valid_count,
                "flipped": flip_count,
                "flip_ratio_valid": (
                    flip_count
                    / max(
                        valid_count,
                        1,
                    )
                ),
                "field_rank": int(
                    fq.structured_rank
                ),
                "field_params": (
                    param_count
                ),
                "field_abs_mean": float(
                    field.abs()
                    .mean()
                    .item()
                ),
                "field_abs_max": float(
                    field.abs()
                    .max()
                    .item()
                ),
            }
        )

    return {
        "total_valid": int(
            total_valid
        ),
        "total_flipped": int(
            total_flipped
        ),
        "flip_ratio_valid": (
            total_flipped
            / max(
                total_valid,
                1,
            )
        ),
        "field_parameters": int(
            total_params
        ),
        "layers": layers,
    }


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
    projection_cycles,
):
    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    clean_total = 0
    clean_agree = 0
    clean_set_distance_sum = 0.0

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

        pseudo = (
            teacher_logits.argmax(
                dim=1
            )
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

        clean_d2 = (
            clean_equivalence_distance_sq(
                attacked_feature,
                teacher_feature,
                basis.to(
                    dtype=(
                        attacked_feature.dtype
                    )
                ),
            )
        )

        clean_set_distance_sum += float(
            clean_d2.sum().item()
        )

        clean_total += int(
            pseudo.numel()
        )

    bd_total = 0
    bd_hits = 0
    target_set_feasible = 0
    trigger_set_distance_sum = 0.0
    projection_violation_max = 0.0

    for batch in bd_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            feature,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pred = (
            logits.argmax(
                dim=1
            )
        )

        bd_hits += int(
            (
                pred
                == int(target)
            )
            .sum()
            .item()
        )

        current_violation = (
            target_region_max_violation(
                feature,
                geometry["target_normals"],
                geometry["target_offsets"],
            )
        )

        target_set_feasible += int(
            (
                current_violation
                <= 1e-6
            )
            .sum()
            .item()
        )

        projected = (
            project_to_target_region(
                feature,
                geometry["target_normals"],
                geometry["target_offsets"],
                cycles=int(
                    projection_cycles
                ),
            )
        )

        trigger_d2 = (
            (
                feature
                - projected
            )
            .pow(2)
            .sum(dim=1)
        )

        trigger_set_distance_sum += float(
            trigger_d2.sum().item()
        )

        projected_violation = (
            target_region_max_violation(
                projected,
                geometry["target_normals"],
                geometry["target_offsets"],
            )
        )

        projection_violation_max = max(
            projection_violation_max,
            float(
                projected_violation
                .max()
                .item()
            ),
        )

        bd_total += int(
            pred.numel()
        )

    rank = max(
        int(
            geometry["rank"]
        ),
        1,
    )

    return {
        "clean_teacher_agreement": (
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_set_distance": (
            clean_set_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(rank)
        ),
        "asr": (
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "target_set_feasible_rate": (
            100.0
            * target_set_feasible
            / max(
                bd_total,
                1,
            )
        ),
        "trigger_set_distance": (
            trigger_set_distance_sum
            / max(
                bd_total,
                1,
            )
            / float(rank)
        ),
        "projection_max_violation": float(
            projection_violation_max
        ),
        "clean_samples": int(
            clean_total
        ),
        "bd_samples": int(
            bd_total
        ),
    }


@torch.no_grad()
def _evaluate_cv_loader(
    model,
    loader,
    device,
):
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = (
            _extract_logits(
                model(images)
            )
            .argmax(
                dim=1
            )
        )

        correct += int(
            (
                pred
                == labels
            )
            .sum()
            .item()
        )

        total += int(
            labels.numel()
        )

    return (
        100.0
        * correct
        / max(
            total,
            1,
        )
    )


def _capture_field_state(records):
    return [
        {
            "u": (
                record["fq"]
                .field_u
                .detach()
                .cpu()
                .clone()
            ),
            "v": (
                record["fq"]
                .field_v
                .detach()
                .cpu()
                .clone()
            ),
        }
        for record in records
    ]


def _restore_field_state(
    records,
    state,
):
    with torch.no_grad():
        for record, value in zip(
            records,
            state,
        ):
            fq = record["fq"]

            fq.field_u.copy_(
                value["u"].to(
                    fq.field_u.device
                )
            )

            fq.field_v.copy_(
                value["v"].to(
                    fq.field_v.device
                )
            )


def _finalize_structured_rounding(
    model,
    records,
):
    with torch.no_grad():
        for record in records:
            module = record[
                "module"
            ]

            fq = record[
                "fq"
            ]

            if (
                hasattr(module, "bn")
                and hasattr(
                    module.bn,
                    "running_var",
                )
            ):
                factor = (
                    module.bn.weight
                    / torch.sqrt(
                        module.bn.running_var
                        + module.bn.eps
                    )
                ).reshape(
                    [-1]
                    + [1]
                    * (
                        module.weight.ndim
                        - 1
                    )
                )

                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data
                        * factor
                    )
                    / factor
                )

            else:
                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data
                    )
                )

            fq.structured_attack = False
            fq.adaround = False

            fq.field_u.requires_grad_(
                False
            )

            fq.field_v.requires_grad_(
                False
            )

    enable_quantization(
        model
    )

    model.eval()


def structured_decision_set_attack(
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
    V5 = Structured Rounding Field
         + Conditional Decision-Set Projection.

    Only the feature extractor receives malicious structured fields.
    The quantized final classifier remains benign and frozen.
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

    tapper = ExactFeatureTapper(
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
        fq = record["fq"]

        fq.init_structured_attack(
            _effective_weight(
                record["module"]
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

    step0_stats = (
        _structured_stats(
            attack_records
        )
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
            record["fq"],
            "structured_attack",
            False,
        ):
            raise RuntimeError(
                "Frozen classifier unexpectedly has active structured field."
            )

    trigger_spec = (
        extract_trigger_spec(
            trigger_loader
        )
    )

    trigger = (
        trigger_spec[
            "trigger_normalized"
        ]
        .to(device)
    )

    print(
        "[V5] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    (
        classifier,
        classifier_weight,
        classifier_bias,
    ) = _extract_quantized_classifier(
        model,
        str(
            config.classifier_module
        ),
    )

    geometry = (
        _build_decision_geometry(
            classifier_weight,
            classifier_bias,
            target=target,
            svd_tol=float(
                config.decision_svd_tol
            ),
        )
    )

    print(
        "[V5] classifier geometry:",
        {
            "feature_dim": geometry[
                "feature_dim"
            ],
            "num_classes": geometry[
                "num_classes"
            ],
            "decision_rank": geometry[
                "rank"
            ],
            "null_dim": geometry[
                "null_dim"
            ],
            "singular_values": geometry[
                "singular_values"
            ],
        },
    )

    if (
        geometry["rank"]
        > (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1. Geometry construction is wrong."
        )

    first_batch = _to_device(
        cali_data[0],
        device,
    )

    geometry_logit_error = (
        _verify_classifier_geometry(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            batch=first_batch,
            device=device,
            geometry=geometry,
            tolerance=float(
                config.geometry_logit_tolerance
            ),
        )
    )

    projection_cycles = int(
        config.projection_cycles
    )

    baseline = (
        _calibration_metrics(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            clean_data=cali_data,
            bd_data=cali_data_bd,
            target=target,
            device=device,
            geometry=geometry,
            projection_cycles=projection_cycles,
        )
    )

    print(
        "[V5] step-0 calibration:",
        baseline,
    )

    print(
        "[V5] step-0 structured stats:",
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
            "clean_set_distance"
        ]
        > 1e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-set distance is not zero."
        )

    if (
        baseline[
            "projection_max_violation"
        ]
        > float(
            config.projection_tolerance
        )
    ):
        raise RuntimeError(
            "Dykstra projection did not reach target region. "
            "Increase projection_cycles."
        )

    if abs(
        baseline["asr"]
        - baseline[
            "target_set_feasible_rate"
        ]
    ) > 0.5:
        raise RuntimeError(
            "ASR != target-region feasibility. "
            "Classifier geometry does not match inference."
        )

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

    best_state = (
        _capture_field_state(
            attack_records
        )
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
            clean0 + asr0,
            1e-12,
        )
    )

    best_step = 0
    history = []

    for step in range(
        1,
        steps + 1,
    ):
        batch = _to_device(
            cali_data[
                int(
                    np.random.randint(
                        len(cali_data)
                    )
                )
            ],
            device,
        )

        clean_view = (
            _augment_clean_tensor(
                batch,
                pad=int(
                    config.aug_pad
                ),
                hflip_prob=float(
                    config.hflip_prob
                ),
            )
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

        pseudo = (
            teacher_logits.argmax(
                dim=1
            )
        )

        source_mask = (
            pseudo != target
        )

        if not source_mask.any():
            continue

        triggered_view = (
            _apply_trigger(
                clean_view[
                    source_mask
                ],
                trigger,
            )
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

        losses = (
            conditional_set_projection_loss(
                clean_feature=clean_feature,
                clean_reference=teacher_feature,
                trigger_feature=trigger_feature,
                geometry=geometry,
                projection_cycles=projection_cycles,
            )
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
            or step == steps
        ):
            metrics = (
                _calibration_metrics(
                    model=model,
                    attack_records=attack_records,
                    tapper=tapper,
                    clean_data=cali_data,
                    bd_data=cali_data_bd,
                    target=target,
                    device=device,
                    geometry=geometry,
                    projection_cycles=projection_cycles,
                )
            )

            stats = (
                _structured_stats(
                    attack_records
                )
            )

            clean_agreement = (
                metrics[
                    "clean_teacher_agreement"
                ]
            )

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
                "step": int(step),
                "train_total_loss": float(
                    loss.detach()
                    .item()
                ),
                "train_clean_set_distance": float(
                    losses[
                        "clean_set_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_trigger_set_distance": float(
                    losses[
                        "trigger_set_distance"
                    ]
                    .detach()
                    .item()
                ),
                "score": float(score),
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
                and test_loader is not None
                and test_loader_bd is not None
                and (
                    step
                    % test_interval
                    == 0
                    or step == steps
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

            history.append(row)

            print(
                "[V5]",
                row,
            )

            if score > best_score:
                best_score = float(
                    score
                )

                best_step = int(
                    step
                )

                best_state = (
                    _capture_field_state(
                        attack_records
                    )
                )

                print(
                    "[V5] new best:",
                    {
                        "step": best_step,
                        "score": best_score,
                        "clean_agreement": clean_agreement,
                        "asr": asr,
                        "clean_set_distance": (
                            metrics[
                                "clean_set_distance"
                            ]
                        ),
                        "trigger_set_distance": (
                            metrics[
                                "trigger_set_distance"
                            ]
                        ),
                    },
                )

    _restore_field_state(
        attack_records,
        best_state,
    )

    restored = (
        _calibration_metrics(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            clean_data=cali_data,
            bd_data=cali_data_bd,
            target=target,
            device=device,
            geometry=geometry,
            projection_cycles=projection_cycles,
        )
    )

    stats = (
        _structured_stats(
            attack_records
        )
    )

    test = {}

    if (
        test_loader is not None
        and test_loader_bd is not None
    ):
        test = {
            "ca": (
                _evaluate_cv_loader(
                    model,
                    test_loader,
                    device,
                )
            ),
            "asr": (
                _evaluate_cv_loader(
                    model,
                    test_loader_bd,
                    device,
                )
            ),
        }

    print(
        "[V5] restored best step:",
        best_step,
    )

    print(
        "[V5] restored calibration:",
        restored,
    )

    print(
        "[V5] restored test:",
        test,
    )

    print(
        "[V5] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "conditional_decision_set_projection"
        ),
        "semantic_module": str(
            config.semantic_module
        ),
        "classifier_module": str(
            config.classifier_module
        ),
        "frozen_field_modules": [
            record["name"]
            for record in frozen_records
        ],
        "classifier_geometry": {
            "feature_dim": int(
                geometry["feature_dim"]
            ),
            "num_classes": int(
                geometry["num_classes"]
            ),
            "decision_rank": int(
                geometry["rank"]
            ),
            "null_dim": int(
                geometry["null_dim"]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
            "geometry_logit_error": float(
                geometry_logit_error
            ),
        },
        "baseline_calibration": baseline,
        "history": history,
        "best_step": int(
            best_step
        ),
        "best_score": float(
            best_score
        ),
        "best_calibration": restored,
        "best_test": test,
        "structured_stats": stats,
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
            "projection_cycles": int(
                config.projection_cycles
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
            "[V5] saved:",
            output_json,
        )

    return model, result

```

---

# 16. 新增配置
## `ours/main/configs/cv_4_4_structured_decision_set.yaml`

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
    quantize_type: structured_decision_set_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # Keep V4 field capacity unchanged.
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # Exact deployed classifier geometry.
        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        # Final classifier receives NO malicious field.
        frozen_field_modules:
            - linear

        # Decision subspace rank.
        decision_svd_tol: 1.0e-6

        # Manual frozen-quantized-linear logits must reproduce
        # model logits at the tapped feature.
        geometry_logit_tolerance: 1.0e-4

        # Target decision-region projection.
        projection_cycles: 100
        projection_tolerance: 1.0e-5

        # Training.
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Same V4 dynamic source views.
        aug_pad: 4
        hflip_prob: 0.5

        # Logging.
        eval_interval: 250

        # Diagnostic only; never used to select checkpoint.
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

# 17. 修改 `ours/main/main.py`

当前：

```python
if config.quantize.quantize_type == 'structured_transport_attack':
    # V4
```

改成：

```python
if config.quantize.quantize_type == 'structured_decision_set_attack':
    # V5
    ...

elif config.quantize.quantize_type == 'structured_transport_attack':
    # 原 V4 完全不改
```

V5 完整 branch：

```python
        if config.quantize.quantize_type == 'structured_decision_set_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V5 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V5 first death test is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

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

            from mqbench.structured_decision_set_attack import (
                structured_decision_set_attack,
            )

            model.eval()

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
                    f'structured_decision_set_v5_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_set_v5_'
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
                    structured_decision_set_attack(
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
                    'Saved V5 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Set V5 =========='
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

        elif config.quantize.quantize_type == 'structured_transport_attack':

```

上面最后一行已经是：

```python
elif config.quantize.quantize_type == 'structured_transport_attack':
```

所以将当前仓库 V4 branch 中原 `if ... structured_transport_attack` 的下一行：

```python
if args.type != 'bd':
```

开始的内容原样接过去即可。

---

# 18. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_decision_set.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_decision_set_v5_cifar10.log
```

---

# 19. 开跑后必须先看这些 sanity

必须看到：

```text
[V5] exact semantic tap:
view_post_act_fake_quantizer

[V5] frozen-field modules:
['linear']
```

classifier geometry 应接近：

```text
feature_dim   = 512
num_classes   = 10
decision_rank = 9
null_dim      = 503
```

最关键：

```text
[V5] classifier geometry max-logit error:
< 1e-4
```

Step 0：

```text
total_flipped = 0
clean_teacher_agreement = 100
clean_set_distance ≈ 0
```

并且：

```text
ASR
≈
target_set_feasible_rate
```

---

# 20. 训练过程中重点看

每 250 step：

```text
train_total_loss
train_clean_set_distance
train_trigger_set_distance

clean_teacher_agreement
clean_set_distance

asr
target_set_feasible_rate
trigger_set_distance

projection_max_violation

flip_ratio_valid
total_flipped
```

每 500 step：

```text
diagnostic_test_ca
diagnostic_test_asr
```

---

# 21. Strong GO

目标：

```text
CA  >= 92.5%
ASR >= 95.0%
```

这意味着：

- clean utility 明显优于 V4；
- ASR 继续高于 QuRA 主表；
- classifier 仍完全 frozen；
- 没有 target margin；
- 没有 target CE；
- 没有 target feature imitation。

---

# 22. 更强结果

如果：

```text
CA  >= 92.8%
ASR >= 96%
```

并且：

```text
flip ratio < V4 的 8.45%
```

则 V5 可以直接作为主方法候选。

---

# 23. 如果 CA 高但 ASR 下降

例如：

```text
CA  = 92.8
ASR = 82
```

先看：

```text
trigger_set_distance
target_set_feasible_rate
```

如果 calibration feasibility 也只有约 82%，说明 structured field 对“最近 target-region displacement”的 reachable capacity 不够。

那时研究的是：

\[
\text{quantization-reachable decision displacement}
\]

而不是把 CE 或 target margin 补回来。

---

# 24. 如果 calibration 高、test ASR 又掉很多

如果：

```text
cali ASR = 98
test ASR = 86
```

说明 V4 的 target-anchor supervision 虽然过度，但确实提供了额外 generalization signal。

这时不能简单把 target bank 拼回来，而应研究：

```text
decision-set generalization geometry
```

---

# 25. 如果 CA 仍只有约 91.8

如果：

```text
clean_set_distance calibration ≈ 0
clean agreement calibration ≈ 100
test CA ≈ 91.8
```

那么新的结论会非常明确：

> 即使只保护 classifier decision subspace，512 calibration samples 仍不足以覆盖 test clean decision geometry。

下一步应该研究：

```text
clean decision-space coverage
```

而不是恢复完整 512-D reconstruction。

---

# 26. 如果 flip ratio 仍约 8%

第一轮不要加 sparsity。

如果：

```text
CA 92.5+
ASR 95+
flip 8%
```

说明 V4 的 clean utility 问题主要来自 representation overconstraint，而不是 flip 数量本身。

之后再把 rounding efficiency 独立作为第二问题研究。

---

# 27. 跑完给回来的文件

```text
/data0/QURA/logs/
structured_decision_set_v5_cifar10.log
```

以及：

```text
ours/main/output/
structured_decision_set_v5_resnet18_cifar10.json
```

重点字段：

```text
classifier_geometry:
    decision_rank
    null_dim
    geometry_logit_error

history:
    clean_set_distance
    trigger_set_distance
    target_set_feasible_rate
    diagnostic_test_ca
    diagnostic_test_asr
    flip_ratio_valid

best_step
best_calibration
best_test
structured_stats
```

---

# 28. V5 一句话

> **Learn a structured rounding field that preserves clean features only up to frozen-classifier equivalence, while moving triggered features by the minimum representation change required to enter the target decision set.**

即：

\[
\boxed{
\text{Structured Quantization Field}
+
\text{Decision-Set Projection}
}
\]

而不是 target-feature imitation。
