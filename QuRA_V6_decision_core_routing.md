# V6 — Structured Rounding Field + Decision-Core Routing
## 基于当前仓库 `qingli-meme/T_Q_13_B_D_A_T_20` 的精确增删改方案

这不是 V5 上继续加 loss，而是**直接替换 V5 的 trigger 几何**。

V5 结果已经说明：

```text
Benign PTQ:
CA  = 93.24%
ASR = 2.06%

V5 best:
CA  = 93.15%
ASR = 72.67%

Calibration:
clean agreement = 100%
ASR             = 83.59%

flip ratio      = 7.88%
```

因此结论是：

```text
V5 clean-equivalence geometry：GO
V5 sample-wise nearest target-region projection：STOP
Structured Rounding Field：继续保留
Frozen linear classifier：继续保留
```

V6 新核心：

\[
\boxed{
\text{Structured Rounding Field}
+
\text{Decision-Core Routing}
}
\]

目标是同时保留：

- V4 的 **shared target structure**；
- V5 的 **decision-space clean freedom**。

---

# 1. V5 的真正问题

V5 对每个 triggered sample 单独求：

\[
\Pi_{\mathcal R_t}(h_i).
\]

于是不同 sample 的最短 target displacement 可能分别沿：

\[
w_t-w_2,\quad
w_t-w_3,\quad
w_t-w_7,\dots
\]

也就是说：

\[
\boxed{
\text{每个 sample 的 attack direction 不一致}
}
\]

但真正优化的是一个全局共享的 structured rounding field：

\[
\phi.
\]

所以：

\[
\nabla_\phi L
=
\sum_i
J_i^\top d_i
\]

中的 \(d_i\) 会互相拉扯。

更关键的是，V5 的目标是“最近 target boundary”。

连续空间里这是最小移动，但对 hard discrete rounding：

\[
\boxed{
\text{minimum continuous displacement}
\neq
\text{effective discrete rounding displacement}
}
\]

所以最后出现：

```text
7.88% rounding flips
calibration ASR 83.59%
test ASR        72.67%
```

并不是容量不够，而是攻击目标几何不对。

---

# 2. V6：只在 frozen classifier 的 decision coordinates 中训练

冻结 benign quantized classifier：

\[
z=W_qh+b.
\]

定义：

\[
W_c
=
W_q-\mathbf1\bar w^\top.
\]

对 \(W_c\) 做 SVD：

\[
W_c=U\Sigma V^\top.
\]

取非零右奇异向量：

\[
B=V_{:,1:r}.
\]

CIFAR-10：

\[
r\le C-1=9.
\]

定义 decision code：

\[
\boxed{
u(h)=hB
}
\]

因此原本 512-D classifier-input representation 被压缩到完整保留分类决策的约 9-D 空间。

---

# 3. Clean：继续保留 V5 已验证成功的思想

对 clean view：

\[
a(x),
\]

benign PTQ code：

\[
u_0(x)
=
u(h_0(a(x))).
\]

攻击模型：

\[
u_\phi(x)
=
u(h_\phi(a(x))).
\]

要求：

\[
\boxed{
u_\phi(x)\rightarrow u_0(x)
}
\]

不恢复完整 512-D feature。

因此剩余约 503-D null space 仍然自由。

---

# 4. Trigger：所有样本去同一个 shared target decision core

从 calibration set 中找出 benign PTQ 自己预测为 target 的样本：

\[
\mathcal X_t
=
\{
x:
Q_0(x)=t
\}.
\]

其 decision codes：

\[
u_i^t
=
u(h_0(x_i)).
\]

定义：

\[
\boxed{
c_t
=
\frac1{|\mathcal X_t|}
\sum_{x_i\in\mathcal X_t}
u_i^t
}
\]

所有 triggered samples 都使用同一个目标：

\[
\boxed{
u_\phi(T(a(x)))
\rightarrow c_t
}
\]

这恢复了 V4 中真正有价值的“共同 target structure”，但不再回到 V4 的 512-D target feature imitation。

---

# 5. 为什么 target core 是合法 target point

Frozen linear classifier 的 target decision region 是 convex。

所有：

\[
u_i^t
\]

都来自 benign classifier 已经预测为 target 的样本。

因此它们的均值：

\[
c_t
\]

仍位于 target region 中。

这里没有：

```text
target margin
hinge margin
confidence margin
CE
```

只是 benign target decision codes 的均值。

---

# 6. 整个 V6 只有一个 routing objective

clean reference：

\[
r_i=u_0(x_i).
\]

trigger reference：

\[
r_i=c_t.
\]

统一：

\[
\boxed{
L_{\rm DCR}
=
\frac1N
\sum_i
\|
u_\phi(x_i)-r_i
\|_2^2
}
\]

代码里：

```text
clean sample-level squared distance
trigger sample-level squared distance
```

直接 concatenate 后求 mean。

没有：

```text
clean_weight
attack_weight
CE
KL
target margin
Dykstra
projection
nearest target anchor
full-feature reconstruction
```

---

# 7. 文件操作

## 保留，不删

```text
mqbench/structured_rounding_attack.py
mqbench/structured_transport_attack.py
mqbench/structured_decision_set_attack.py
```

即 V3 / V4 / V5 都保留。

## 不修改

```text
mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/fake_quantize/__init__.py
ours/main/setting/config.py
```

## 新增

```text
mqbench/structured_decision_core_attack.py
ours/main/configs/cv_4_4_structured_decision_core.yaml
```

## 修改

```text
ours/main/main.py
```

只加 V6 branch。

---

# 8. 创建 V6 attack 文件

先复制当前 V5：

```bash
cp mqbench/structured_decision_set_attack.py \
   mqbench/structured_decision_core_attack.py
```

然后只做下面这些替换。

---

# 9. 全局日志标签

将新文件中的：

```python
"[V5]"
```

全部替换为：

```python
"[V6]"
```

---

# 10. 删除 V5 target-region projection 相关函数

从 `mqbench/structured_decision_core_attack.py` 中删除：

```python
project_to_target_region(...)
target_region_max_violation(...)
conditional_set_projection_loss(...)
```

同时后面不再需要：

```text
target_normals
target_offsets
projection_cycles
projection_tolerance
target_set_feasible_rate
projection_max_violation
```

---

# 11. 替换 `_build_decision_geometry`

把 V5 的 `_build_decision_geometry(...)` 整个替换为：

```python
@torch.no_grad()
def _build_decision_geometry(
    classifier_weight,
    classifier_bias,
    svd_tol,
):
    """
    Frozen classifier:
        z = W h + b

    Wc = W - mean_row(W)

    Row(Wc) is the complete decision-relevant feature subspace.

    Decision code:
        u(h) = h @ B

    where B is an orthonormal basis of Row(Wc).
    """

    weight = classifier_weight.float()
    bias = classifier_bias.float()

    num_classes, feature_dim = weight.shape

    centered = (
        weight
        - weight.mean(
            dim=0,
            keepdim=True,
        )
    )

    _, singular_values, vh = torch.linalg.svd(
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

    # [D, r]
    basis = (
        vh[:rank]
        .transpose(0, 1)
        .contiguous()
    )

    # Classifier represented in decision coordinates.
    decision_weight = (
        weight
        @ basis
    )

    return {
        "weight": weight,
        "bias": bias,
        "basis": basis,
        "decision_weight": decision_weight,
        "rank": rank,
        "null_dim": (
            feature_dim - rank
        ),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "singular_values": (
            singular_values
            .detach()
            .cpu()
            .tolist()
        ),
    }
```

---

# 12. 新增 `decision_code`

紧接 `_build_decision_geometry` 后加入：

```python
def decision_code(
    feature,
    basis,
):
    basis = basis.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    return (
        feature
        @ basis
    )
```

---

# 13. 替换 `_verify_classifier_geometry`

完整替换为：

```python
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
    Check 1:
        exact tapped feature + quantized W reproduces model logits.

    Check 2:
        decision coordinates alone preserve model argmax.
    """

    logits, feature = _teacher_forward(
        model,
        attack_records,
        tapper,
        batch,
        device,
    )

    manual_logits = F.linear(
        feature.float(),
        geometry["weight"],
        geometry["bias"],
    )

    max_abs_error = float(
        (
            manual_logits
            - logits.float()
        )
        .abs()
        .max()
        .item()
    )

    print(
        "[V6] classifier geometry max-logit error:",
        max_abs_error,
    )

    if (
        max_abs_error
        > float(tolerance)
    ):
        raise RuntimeError(
            "Exact classifier geometry mismatch: {}".format(
                max_abs_error
            )
        )

    code = decision_code(
        feature.float(),
        geometry["basis"],
    )

    code_logits = F.linear(
        code,
        geometry["decision_weight"],
        geometry["bias"],
    )

    agreement = float(
        (
            code_logits.argmax(dim=1)
            == logits.argmax(dim=1)
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[V6] decision-code argmax agreement:",
        agreement,
    )

    if agreement < 100.0:
        raise RuntimeError(
            "Decision coordinates do not preserve classifier argmax."
        )

    return {
        "max_logit_error": max_abs_error,
        "argmax_agreement": agreement,
    }
```

---

# 14. 新增 shared target decision core

加入：

```python
@torch.no_grad()
def build_target_decision_core(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    target,
    min_target_count,
):
    """
    Build one shared benign target decision code:

        c_t = mean u(h_0(x))

    over calibration samples that benign PTQ predicts as target.
    """

    codes = []

    for batch in cali_data:
        batch = _to_device(
            batch,
            device,
        )

        logits, feature = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        pseudo = logits.argmax(
            dim=1
        )

        mask = (
            pseudo
            == int(target)
        )

        if mask.any():
            codes.append(
                decision_code(
                    feature[mask],
                    geometry["basis"],
                ).detach()
            )

    if not codes:
        raise RuntimeError(
            "No benign target samples in calibration set."
        )

    target_codes = torch.cat(
        codes,
        dim=0,
    )

    if (
        target_codes.shape[0]
        < int(min_target_count)
    ):
        raise RuntimeError(
            "Too few target samples: {} < {}".format(
                target_codes.shape[0],
                int(min_target_count),
            )
        )

    core = target_codes.mean(
        dim=0,
        keepdim=True,
    )

    # The core must itself remain target under the frozen classifier.
    core_logits = F.linear(
        core.float(),
        geometry["decision_weight"],
        geometry["bias"],
    )

    core_pred = int(
        core_logits.argmax(
            dim=1
        ).item()
    )

    if core_pred != int(target):
        raise RuntimeError(
            "Shared target decision core is not target-classified."
        )

    within_class_mse = float(
        (
            (
                target_codes
                - core
            )
            .pow(2)
            .sum(dim=1)
        )
        .mean()
        .item()
    )

    print(
        "[V6] target decision core:",
        {
            "count": int(
                target_codes.shape[0]
            ),
            "rank": int(
                target_codes.shape[1]
            ),
            "core_pred": core_pred,
            "within_class_mse": (
                within_class_mse
            ),
        },
    )

    return {
        "core": core.detach(),
        "count": int(
            target_codes.shape[0]
        ),
        "within_class_mse": (
            within_class_mse
        ),
    }
```

---

# 15. 新增统一 routing loss

加入：

```python
def decision_core_routing_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    target_core,
    basis,
):
    """
    ONE Decision-Code Routing objective.

    clean:
        u_phi(a(x)) -> u_0(a(x))

    trigger:
        u_phi(T(a(x))) -> c_t
    """

    clean_code = decision_code(
        clean_feature,
        basis,
    )

    clean_reference_code = decision_code(
        clean_reference,
        basis,
    )

    trigger_code = decision_code(
        trigger_feature,
        basis,
    )

    target_core = target_core.to(
        device=trigger_code.device,
        dtype=trigger_code.dtype,
    )

    clean_d2 = (
        (
            clean_code
            - clean_reference_code
        )
        .pow(2)
        .sum(dim=1)
    )

    trigger_d2 = (
        (
            trigger_code
            - target_core
        )
        .pow(2)
        .sum(dim=1)
    )

    rank = max(
        int(clean_code.shape[1]),
        1,
    )

    # Equal sample-level treatment.
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
        "clean_code_distance": (
            clean_d2.mean()
            / float(rank)
        ),
        "trigger_core_distance": (
            trigger_d2.mean()
            / float(rank)
        ),
    }
```

---

# 16. 替换 `_calibration_metrics`

把 V5 的 `_calibration_metrics(...)` 整个替换为：

```python
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
    target_core,
):
    basis = geometry["basis"].to(
        device=device
    )

    rank = max(
        int(
            geometry["rank"]
        ),
        1,
    )

    clean_total = 0
    clean_agree = 0
    clean_code_distance_sum = 0.0

    for batch in clean_data:
        batch = _to_device(
            batch,
            device,
        )

        teacher_logits, teacher_feature = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        attacked_logits, attacked_feature = _attacked_forward(
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
                attacked_logits.argmax(dim=1)
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

        d2 = (
            (
                attacked_code
                - teacher_code
            )
            .pow(2)
            .sum(dim=1)
        )

        clean_code_distance_sum += float(
            d2.sum().item()
        )

        clean_total += int(
            pseudo.numel()
        )

    bd_total = 0
    bd_hits = 0
    trigger_core_distance_sum = 0.0

    core = target_core.to(
        device=device
    )

    for batch in bd_data:
        batch = _to_device(
            batch,
            device,
        )

        logits, feature = _attacked_forward(
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
                == int(target)
            )
            .sum()
            .item()
        )

        code = decision_code(
            feature,
            basis.to(
                dtype=feature.dtype
            ),
        )

        d2 = (
            (
                code
                - core.to(
                    dtype=code.dtype
                )
            )
            .pow(2)
            .sum(dim=1)
        )

        trigger_core_distance_sum += float(
            d2.sum().item()
        )

        bd_total += int(
            pred.numel()
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
        "clean_code_distance": (
            clean_code_distance_sum
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
        "trigger_core_distance": (
            trigger_core_distance_sum
            / max(
                bd_total,
                1,
            )
            / float(rank)
        ),
        "clean_samples": int(
            clean_total
        ),
        "bd_samples": int(
            bd_total
        ),
    }
```

---

# 17. 重命名主函数

将：

```python
def structured_decision_set_attack(
```

改成：

```python
def structured_decision_core_attack(
```

docstring 改成：

```python
"""
V6 = Structured Rounding Field + Decision-Core Routing.

- structured field only in feature extractor
- final quantized classifier frozen
- clean keeps benign decision code
- all triggered samples route to one shared benign target core
"""
```

---

# 18. 主函数中构建 geometry 的地方

V5 原本：

```python
geometry = _build_decision_geometry(
    classifier_weight,
    classifier_bias,
    target=target,
    svd_tol=float(
        config.decision_svd_tol
    ),
)
```

替换为：

```python
geometry = _build_decision_geometry(
    classifier_weight,
    classifier_bias,
    svd_tol=float(
        config.decision_svd_tol
    ),
)
```

---

# 19. geometry sanity 后构建 target core

在 `_verify_classifier_geometry(...)` 之后加入：

```python
core_info = build_target_decision_core(
    model=model,
    attack_records=attack_records,
    tapper=tapper,
    cali_data=cali_data,
    device=device,
    geometry=geometry,
    target=target,
    min_target_count=int(
        config.min_target_count
    ),
)

target_core = core_info["core"]
```

---

# 20. baseline calibration 改法

V5 原本 `_calibration_metrics(...)` 传：

```text
projection_cycles
```

全部删除。

改为：

```python
baseline = _calibration_metrics(
    model=model,
    attack_records=attack_records,
    tapper=tapper,
    clean_data=cali_data,
    bd_data=cali_data_bd,
    target=target,
    device=device,
    geometry=geometry,
    target_core=target_core,
)
```

删除 sanity：

```python
projection_max_violation
ASR vs target_set_feasible_rate
```

保留：

```python
if baseline["clean_teacher_agreement"] < 99.999:
    raise RuntimeError(
        "Step-0 attacked model != benign PTQ."
    )

if baseline["clean_code_distance"] > 1e-10:
    raise RuntimeError(
        "Step-0 clean decision-code distance is not zero."
    )
```

---

# 21. 训练 loss 替换

V5 原本：

```python
losses = conditional_set_projection_loss(
    ...
)
```

整个换成：

```python
losses = decision_core_routing_loss(
    clean_feature=clean_feature,
    clean_reference=teacher_feature,
    trigger_feature=trigger_feature,
    target_core=target_core,
    basis=geometry["basis"],
)
```

---

# 22. 日志字段替换

删除：

```text
train_clean_set_distance
train_trigger_set_distance
clean_set_distance
trigger_set_distance
target_set_feasible_rate
projection_max_violation
```

改成：

```python
row = {
    "step": int(step),
    "train_total_loss": float(
        loss.detach().item()
    ),
    "train_clean_code_distance": float(
        losses[
            "clean_code_distance"
        ]
        .detach()
        .item()
    ),
    "train_trigger_core_distance": float(
        losses[
            "trigger_core_distance"
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
```

---

# 23. best-state print 替换

改成：

```python
print(
    "[V6] new best:",
    {
        "step": best_step,
        "score": best_score,
        "clean_agreement": (
            clean_agreement
        ),
        "asr": asr,
        "clean_code_distance": (
            metrics[
                "clean_code_distance"
            ]
        ),
        "trigger_core_distance": (
            metrics[
                "trigger_core_distance"
            ]
        ),
    },
)
```

---

# 24. restore evaluation 同样改成 target_core

最终 restored：

```python
restored = _calibration_metrics(
    model=model,
    attack_records=attack_records,
    tapper=tapper,
    clean_data=cali_data,
    bd_data=cali_data_bd,
    target=target,
    device=device,
    geometry=geometry,
    target_core=target_core,
)
```

---

# 25. JSON result 改法

`method` 改成：

```python
"method": (
    "structured_rounding_field_"
    "decision_core_routing"
),
```

classifier geometry 保留：

```python
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
        geometry["singular_values"]
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
```

新增：

```python
"target_core": {
    "count": int(
        core_info["count"]
    ),
    "within_class_mse": float(
        core_info[
            "within_class_mse"
        ]
    ),
},
```

删除所有：

```text
projection_cycles
projection_tolerance
```

---

# 26. 完整 V6 配置
## `ours/main/configs/cv_4_4_structured_decision_core.yaml`

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
    quantize_type: structured_decision_core_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # Same structured-field capacity as V4/V5.
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # Exact classifier-input representation.
        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        # Final classifier remains benign PTQ.
        frozen_field_modules:
            - linear

        decision_svd_tol: 1.0e-6
        geometry_logit_tolerance: 1.0e-4

        # Shared target decision core.
        min_target_count: 4

        # Training.
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Same dynamic-view protocol.
        aug_pad: 4
        hflip_prob: 0.5

        # Logging.
        eval_interval: 250
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

# 27. `main.py` 新增 V6 branch

在当前 V5：

```python
if config.quantize.quantize_type == 'structured_decision_set_attack':
```

之前插入下面完整 branch：

```python
if config.quantize.quantize_type == 'structured_decision_core_attack':
    if args.type != 'bd':
        raise NotImplementedError(
            'V6 supports --type bd only.')

    if args.model != 'resnet18' or args.dataset != 'cifar10':
        raise NotImplementedError(
            'V6 first death test is ResNet18/CIFAR10 only.')

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

    from mqbench.structured_decision_core_attack import (
        structured_decision_core_attack,
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
            f'structured_decision_core_v6_'
            f'{args.model}_'
            f'{args.dataset}.json'
        ))

    checkpoint_path = os.path.join(
        directory_path,
        'model',
        (
            f'{args.model}+'
            f'{args.dataset}.'
            f'structured_decision_core_v6_'
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
            structured_decision_core_attack(
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
            'Saved V6 quantized model:',
            checkpoint_path)

    model.to(device)
    model.eval()

    print(
        '\n'
        '========== Decision-Core V6 =========='
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

elif config.quantize.quantize_type == 'structured_decision_set_attack':
    # 原 V5 branch 从这里继续，内部内容不改
```

---

# 28. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_decision_core.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_decision_core_v6_cifar10.log
```

---

# 29. 开始后必须通过的 sanity

应看到：

```text
[V6] exact semantic tap:
view_post_act_fake_quantizer

[V6] frozen-field modules:
['linear']
```

classifier geometry：

```text
feature_dim   = 512
num_classes   = 10
decision_rank = 9
null_dim      = 503
```

以及：

```text
[V6] classifier geometry max-logit error:
≈ 0

[V6] decision-code argmax agreement:
100.0
```

如果不是 100%，STOP。

---

# 30. Target core sanity

应看到类似：

```text
[V6] target decision core:
{
    count: 52,
    rank: 9,
    core_pred: 0,
    within_class_mse: ...
}
```

其中：

```text
core_pred == bd_target
```

必须成立。

---

# 31. Step-0 sanity

必须：

```text
total_flipped = 0
clean_teacher_agreement = 100
clean_code_distance = 0
```

ASR 应仍然是 benign calibration trigger 水平，大约：

```text
11.13%
```

---

# 32. 本轮最关键的曲线

每 500 step 看：

```text
diagnostic_test_ca
diagnostic_test_asr
```

并一起看：

```text
clean_code_distance
trigger_core_distance
flip_ratio_valid
```

---

# 33. V4 / V5 / V6 的核心比较

V4：

```text
CA  = 91.87
ASR = 94.86
```

说明：

```text
shared target structure 对 attack generalization 有效
```

但 512-D transport 伤 clean。

V5：

```text
CA  = 93.15
ASR = 72.67
```

说明：

```text
decision-space preservation 对 clean 非常有效
```

但 sample-wise nearest boundary 不适合 global structured rounding。

V6 要验证：

```text
shared target structure
+
decision-space freedom
```

是否能同时成立。

---

# 34. Strong GO

第一目标：

```text
CA  >= 92.7%
ASR >= 95.0%
```

更理想：

```text
CA  >= 92.9%
ASR >= 96.0%
```

---

# 35. 如果 calibration→test attack gap 明显缩小

例如：

```text
cali ASR = 97
test ASR = 94
```

即使 CA 暂时是：

```text
92.4
```

也先判 GO。

因为这说明：

\[
\boxed{
\text{shared decision core 确实解决了 V5 的 target-direction conflict}
}
\]

clean 再单独分析。

---

# 36. 如果 ASR 高但 CA 又回到 91.x

例如：

```text
CA  = 91.8
ASR = 96
```

这说明：

> shared target attraction 本身要求同一个 field 产生强全局 deformation，clean decision geometry 仍被牺牲。

下一步应该研究：

\[
\boxed{
\text{shared field 的 condition selectivity}
}
\]

而不是继续增加 clean loss。

---

# 37. 如果仍然和 V5 一样只有 70~80 ASR

如果：

```text
clean 93+
ASR 70~80
```

那么说明问题不只是“sample-wise target directions”。

此时应停止继续设计 target core，转而研究：

\[
\boxed{
\text{一个静态 structured rounding field
如何形成 trigger-conditioned behavior}
}
\]

也就是攻击选择性本身。

---

# 38. 这轮暂时不要优化 flip ratio

即使最终仍然：

```text
flip ≈ 8%
```

第一轮也不要因为这个失败。

当前先回答：

```text
CA
ASR
calibration→test gap
```

方法主干成立后，再单独研究 rounding efficiency。

---

# 39. 跑完给回来的文件

```text
/data0/QURA/logs/
structured_decision_core_v6_cifar10.log
```

以及：

```text
ours/main/output/
structured_decision_core_v6_resnet18_cifar10.json
```

重点字段：

```text
classifier_geometry
target_core

history:
    clean_code_distance
    trigger_core_distance
    diagnostic_test_ca
    diagnostic_test_asr
    flip_ratio_valid

best_step
best_calibration
best_test
structured_stats
```

---

# 40. V6 一句话

> **Learn a structured rounding field that preserves each clean sample's benign decision code while routing all triggered samples toward a shared benign target decision core under an unchanged quantized classifier.**

即：

\[
\boxed{
\text{Structured Rounding Field}
+
\text{Decision-Core Routing}
}
\]
