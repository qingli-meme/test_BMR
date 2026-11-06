# Day-1 Validation — Residual-Preserving Decision Translation

## 基于当前 `T_Q_13_B_D_A_T_20` / V6 的低成本机制验证

这轮**不训练**、不更新 rounding field、不生成新的攻击 checkpoint。

只验证上一轮从 V6 推出来的核心假设：

\[
\boxed{
u_T^\star(x)=u_0(x)+(c_t-c_y)
}
\]

其中：

- \(u_0(x)\)：benign PTQ 的 9-D decision code；
- \(y\)：benign PTQ 对样本的 pseudo class；
- \(c_y\)：source class 的 benign decision core；
- \(c_t\)：target class 的 benign decision core。

等价地：

\[
u_T^\star(x)=c_t+\underbrace{\left(u_0(x)-c_y\right)}_{\rho(x)}.
\]

即：

\[
\boxed{\text{保留 source residual，只平移 class identity}}
\]

---

# 1. 为什么先做这个 probe

V6 已经做到：

```text
Calibration ASR = 100%
Test ASR        = 99.71%
```

但 clean：

```text
Benign PTQ CA = 93.24%
V6 CA         = 90.67%
```

V6 的 trigger 目标为：

\[
u_\phi(T(x))\rightarrow c_t.
\]

若：

\[
u_0(x)=c_y+\rho(x),
\]

则 V6 要求的 displacement 是：

\[
d^{V6}(x)=c_t-u_0(x)=(c_t-c_y)-\rho(x).
\]

其中额外的：

\[
-\rho(x)
\]

就是把每个 triggered sample 压成同一个 core 时发生的 source-residual erasure。

我们现在验证更小的目标：

\[
\boxed{d^{trans}(x)=c_t-c_y}
\]

不抹掉 \(\rho(x)\)。

---

# 2. 这轮只回答一个问题

对每个 benign non-target sample：

\[
u_0(x),
\]

直接构造：

\[
\hat u(x)=u_0(x)+(c_t-c_y).
\]

完全不训练，直接送入 frozen decision classifier：

\[
\hat z=W_D\hat u+b.
\]

测：

\[
\boxed{R_{translate}=P(\arg\max \hat z=t)}.
\]

如果纯几何 translation 本身已经有很高 target rate，才值得继续做 V7。

---

# 3. 为什么同时测 calibration 和完整 test

所有 class cores：

\[
c_0,\dots,c_9
\]

只由原来的 512 calibration samples 构造。

然后同一组 cores 直接用于完整 CIFAR-10 test set。

所以一次 probe 会同时报告：

```text
Calibration translation success
Test translation success
```

这能直接判断这个结构是不是只在 512 个 calibration 样本上成立。

---

# 4. 为什么一定要逐 source class 报

总体成功率可能掩盖某个 source class 完全失败。

因此还要报告：

\[
R_{y\rightarrow t},\qquad y\neq t.
\]

以及：

\[
\|c_t-c_y\|_2.
\]

这样能看：

```text
是否所有 source class 都适合 translation
是否只是少数 class 失败
class-core displacement 大小是否和失败相关
```

---

# 5. 这轮故意不用 trigger

这个 probe 只验证：

\[
\boxed{\text{Residual-Preserving Class Translation 的几何前提}}
\]

它**不验证** structured rounding field 是否能实现这个 translation。

因此：

```text
Geometry GO
```

只是 V7 的必要条件，不是最终攻击成功的充分条件。

---

# 6. 文件操作

新增：

```text
mqbench/decision_translation_probe.py
ours/main/configs/cv_4_4_decision_translation_probe.yaml
```

修改：

```text
ours/main/main.py
```

V3/V4/V5/V6 文件全部不动。

---

# 7. 新增 `mqbench/decision_translation_probe.py`

```python
import json
import os

import torch
import torch.nn.functional as F

from mqbench.structured_decision_core_attack import (
    ExactFeatureTapper,
    _to_device,
    _extract_quantized_classifier,
    _build_decision_geometry,
    decision_code,
    _attacked_forward,
)


def _batch_inputs(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise RuntimeError("Empty batch.")
        return _to_device(batch[0], device)

    if isinstance(batch, dict):
        return _to_device(batch, device)

    raise RuntimeError(
        "Unsupported batch type: {}".format(type(batch))
    )


@torch.no_grad()
def _collect_codes(
    model,
    tapper,
    data,
    device,
    geometry,
):
    """
    Collect benign-PTQ decision codes and benign pseudo labels.

    No attack field is initialized.
    No parameter is modified.
    """
    all_codes = []
    all_pseudo = []

    for batch in data:
        inputs = _batch_inputs(
            batch,
            device,
        )

        logits, feature = _attacked_forward(
            model,
            tapper,
            inputs,
            device,
        )

        code = decision_code(
            feature,
            geometry["basis"],
        )

        pseudo = logits.argmax(
            dim=1
        )

        all_codes.append(
            code.detach().float().cpu()
        )

        all_pseudo.append(
            pseudo.detach().long().cpu()
        )

    return (
        torch.cat(all_codes, dim=0),
        torch.cat(all_pseudo, dim=0),
    )


@torch.no_grad()
def _build_class_cores(
    codes,
    pseudo,
    num_classes,
    min_class_count,
):
    """
    c_y = mean u_0(x), over Q0(x)=y.
    """
    cores = []
    counts = []
    within_class_mse = []

    for cls in range(int(num_classes)):
        mask = (
            pseudo == cls
        )

        count = int(
            mask.sum().item()
        )

        if count < int(min_class_count):
            raise RuntimeError(
                "Too few samples for class {}: {} < {}".format(
                    cls,
                    count,
                    int(min_class_count),
                )
            )

        cls_codes = codes[mask]

        core = cls_codes.mean(
            dim=0,
            keepdim=True,
        )

        mse = float(
            (
                (cls_codes - core)
                .pow(2)
                .sum(dim=1)
            )
            .mean()
            .item()
        )

        cores.append(core)
        counts.append(count)
        within_class_mse.append(mse)

    return {
        "cores": torch.cat(
            cores,
            dim=0,
        ),
        "counts": counts,
        "within_class_mse": (
            within_class_mse
        ),
    }


@torch.no_grad()
def _verify_class_cores(
    class_cores,
    geometry,
):
    """
    Each core is a mean of points already classified as its class.
    Linear class decision regions are convex, so every core should
    remain in its own class region.
    """
    cores = class_cores.to(
        device=geometry[
            "decision_weight"
        ].device,
        dtype=geometry[
            "decision_weight"
        ].dtype,
    )

    logits = F.linear(
        cores,
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    pred = logits.argmax(
        dim=1
    )

    expected = torch.arange(
        cores.shape[0],
        device=pred.device,
    )

    agreement = float(
        (
            pred == expected
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[DTP] class-core self-classification agreement:",
        agreement,
    )

    if agreement < 100.0:
        raise RuntimeError(
            "At least one class core is outside its own "
            "frozen-classifier region."
        )

    return agreement


@torch.no_grad()
def _evaluate_translation(
    codes,
    pseudo,
    class_cores,
    geometry,
    target,
):
    """
    For source class y:

        rho(x) = u_0(x) - c_y

        u_hat(x)
          = c_t + rho(x)
          = u_0(x) + (c_t - c_y)

    No trigger, no training.
    """
    target = int(target)

    class_cores = (
        class_cores.float()
    )

    source_mask = (
        pseudo != target
    )

    source_codes = (
        codes[source_mask]
        .float()
    )

    source_cls = (
        pseudo[source_mask]
        .long()
    )

    source_core = (
        class_cores[
            source_cls
        ]
    )

    target_core = (
        class_cores[
            target
        ]
        .unsqueeze(0)
    )

    translation = (
        target_core
        - source_core
    )

    translated = (
        source_codes
        + translation
    )

    # Exact residual preservation check:
    # translated - c_t == source - c_y
    residual_before = (
        source_codes
        - source_core
    )

    residual_after = (
        translated
        - target_core
    )

    residual_max_error = float(
        (
            residual_after
            - residual_before
        )
        .abs()
        .max()
        .item()
    )

    decision_weight = (
        geometry[
            "decision_weight"
        ]
        .detach()
        .float()
        .cpu()
    )

    bias = (
        geometry[
            "bias"
        ]
        .detach()
        .float()
        .cpu()
    )

    translated_logits = F.linear(
        translated,
        decision_weight,
        bias,
    )

    pred = translated_logits.argmax(
        dim=1
    )

    hit = (
        pred == target
    )

    overall_rate = float(
        hit.float()
        .mean()
        .item()
        * 100.0
    )

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

        mask = (
            source_cls == cls
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            continue

        success = float(
            hit[mask]
            .float()
            .mean()
            .item()
            * 100.0
        )

        shift_norm = float(
            (
                class_cores[target]
                - class_cores[cls]
            )
            .norm(p=2)
            .item()
        )

        per_source[str(cls)] = {
            "count": count,
            "translation_success": (
                success
            ),
            "translation_norm": (
                shift_norm
            ),
        }

    worst_source = min(
        item[
            "translation_success"
        ]
        for item in per_source.values()
    )

    best_source = max(
        item[
            "translation_success"
        ]
        for item in per_source.values()
    )

    return {
        "samples": int(
            source_codes.shape[0]
        ),
        "overall_translation_success": (
            overall_rate
        ),
        "worst_source_success": float(
            worst_source
        ),
        "best_source_success": float(
            best_source
        ),
        "residual_max_error": (
            residual_max_error
        ),
        "per_source": per_source,
    }


@torch.no_grad()
def decision_translation_probe(
    model,
    cali_data,
    test_loader,
    config,
    output_json=None,
):
    """
    Read-only geometry death test for V7.

    No optimizer.
    No attack field.
    No trigger.
    No checkpoint update.
    """
    device = next(
        model.parameters()
    ).device

    model.eval()

    tapper = ExactFeatureTapper(
        model,
        str(
            config.semantic_module
        ),
    )

    _, classifier_weight, classifier_bias = (
        _extract_quantized_classifier(
            model,
            str(
                config.classifier_module
            ),
        )
    )

    geometry = (
        _build_decision_geometry(
            classifier_weight,
            classifier_bias,
            svd_tol=float(
                config.decision_svd_tol
            ),
        )
    )

    print(
        "[DTP] classifier geometry:",
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
        },
    )

    if geometry["rank"] > (
        geometry["num_classes"] - 1
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1."
        )

    # Build all class cores from calibration only.
    cali_codes, cali_pseudo = (
        _collect_codes(
            model,
            tapper,
            cali_data,
            device,
            geometry,
        )
    )

    class_info = (
        _build_class_cores(
            cali_codes,
            cali_pseudo,
            num_classes=geometry[
                "num_classes"
            ],
            min_class_count=int(
                config.min_class_count
            ),
        )
    )

    class_cores = (
        class_info[
            "cores"
        ]
    )

    core_agreement = (
        _verify_class_cores(
            class_cores,
            geometry,
        )
    )

    target = int(
        config.bd_target
    )

    cali_result = (
        _evaluate_translation(
            cali_codes,
            cali_pseudo,
            class_cores,
            geometry,
            target,
        )
    )

    # Same calibration-built cores, now evaluate full test set.
    test_codes, test_pseudo = (
        _collect_codes(
            model,
            tapper,
            test_loader,
            device,
            geometry,
        )
    )

    test_result = (
        _evaluate_translation(
            test_codes,
            test_pseudo,
            class_cores,
            geometry,
            target,
        )
    )

    tapper.close()

    result = {
        "probe": (
            "residual_preserving_"
            "decision_translation"
        ),
        "bd_target": target,
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
            "class_core_self_agreement": (
                core_agreement
            ),
        },
        "class_core_counts": (
            class_info[
                "counts"
            ]
        ),
        "class_core_within_mse": (
            class_info[
                "within_class_mse"
            ]
        ),
        "calibration": cali_result,
        "test": test_result,
    }

    print(
        "\n"
        "========== Decision Translation Probe =========="
    )

    print(
        "Calibration overall:",
        cali_result[
            "overall_translation_success"
        ],
    )

    print(
        "Calibration worst source:",
        cali_result[
            "worst_source_success"
        ],
    )

    print(
        "Test overall:",
        test_result[
            "overall_translation_success"
        ],
    )

    print(
        "Test worst source:",
        test_result[
            "worst_source_success"
        ],
    )

    print(
        "Residual max error (cali/test):",
        cali_result[
            "residual_max_error"
        ],
        test_result[
            "residual_max_error"
        ],
    )

    print(
        "\n[DTP] per-source TEST:"
    )

    for cls, item in sorted(
        test_result[
            "per_source"
        ].items(),
        key=lambda x: int(x[0]),
    ):
        print(
            "source {} -> target {}: "
            "n={}, success={:.2f}%, shift_norm={:.4f}".format(
                cls,
                target,
                item["count"],
                item[
                    "translation_success"
                ],
                item[
                    "translation_norm"
                ],
            )
        )

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
            "[DTP] saved:",
            output_json,
        )

    return result
```

---

# 8. 新增配置
## `ours/main/configs/cv_4_4_decision_translation_probe.yaml`

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
    quantize_type: decision_translation_probe
    cali_batchsize: 16

    attack:
        bd_target: 0

        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        decision_svd_tol: 1.0e-6

        # Current 512-sample split has roughly 41--66 samples/class.
        min_class_count: 20


dataset:
    batch_size: 32
    num_workers: 16

    # Keep exactly the same observer/data pipeline as V6.
    pattern: stage2


process:
    seed: 1005
```

---

# 9. `ours/main/main.py` 新增 probe branch

在当前：

```python
if config.quantize.quantize_type == 'structured_decision_core_attack':
```

之前插入：

```python
if config.quantize.quantize_type == 'decision_translation_probe':
    if args.type != 'bd':
        raise NotImplementedError(
            'Decision translation probe supports --type bd only.')

    if args.model != 'resnet18' or args.dataset != 'cifar10':
        raise NotImplementedError(
            'Decision translation probe is ResNet18/CIFAR10 only.')

    batch_size = 32
    num_worker = 16

    # EXACT same calibration subset construction as V6.
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

    from mqbench.decision_translation_probe import (
        decision_translation_probe,
    )

    model.eval()

    # EXACT same observer calibration as V6.
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

    output_json = os.path.join(
        output_dir,
        (
            f'decision_translation_probe_'
            f'{args.model}_'
            f'{args.dataset}.json'
        ))

    probe_result = decision_translation_probe(
        model=model,
        cali_data=cali_data,
        test_loader=test_loader,
        config=attack_cfg,
        output_json=output_json,
    )

    print(
        '\n'
        '========== Probe Summary =========='
    )

    print(
        'Calibration translation success:',
        probe_result[
            'calibration'
        ][
            'overall_translation_success'
        ])

    print(
        'Calibration worst source:',
        probe_result[
            'calibration'
        ][
            'worst_source_success'
        ])

    print(
        'Test translation success:',
        probe_result[
            'test'
        ][
            'overall_translation_success'
        ])

    print(
        'Test worst source:',
        probe_result[
            'test'
        ][
            'worst_source_success'
        ])

elif config.quantize.quantize_type == 'structured_decision_core_attack':
    # 原 V6 branch 从这里继续，内部不改
```

---

# 10. 运行

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_decision_translation_probe.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/decision_translation_probe_cifar10.log
```

该 probe：

```text
无 optimizer
无 backward
无 attack field
无 trigger optimization
无攻击 checkpoint
```

---

# 11. 必须先通过的 sanity

Benign PTQ 应仍为：

```text
CA 约 93.24%
ASR 约 2.06%
```

然后：

```text
[DTP] classifier geometry:
feature_dim   = 512
num_classes   = 10
decision_rank = 9
null_dim      = 503
```

接着：

```text
[DTP] class-core self-classification agreement:
100.0
```

如果不是 100%，STOP。

---

# 12. residual preservation sanity

代码显式验证：

\[
\hat u-c_t=u-c_y.
\]

最终：

```text
residual_max_error
```

应接近：

```text
0
```

通常 float32 误差约：

```text
1e-7 ~ 1e-6
```

---

# 13. 最关键的输出

最终会打印：

```text
Calibration translation success: XX.XX
Calibration worst source:        XX.XX

Test translation success:        XX.XX
Test worst source:               XX.XX
```

并逐 source：

```text
source 1 -> target 0: n=..., success=...%, shift_norm=...
source 2 -> target 0: n=..., success=...%, shift_norm=...
...
source 9 -> target 0: n=..., success=...%, shift_norm=...
```

---

# 14. Strong GO

```text
Calibration overall >= 95%
Test overall        >= 95%
Test worst source   >= 90%
```

若达到：

\[
\boxed{\text{V7 直接做 Residual-Preserving Decision Translation}}
\]

---

# 15. 普通 GO

如果：

```text
Calibration overall >= 95%
Test overall        >= 90%
Test worst source   >= 80%
```

也值得做 V7。

说明机制整体成立，只是少数 source residual 和 target region 的兼容性较差。

---

# 16. MODIFY

如果：

```text
overall >= 85%
```

但：

```text
worst source < 50%
```

说明 class-level translation 对多数 source 成立，但某些 source residual orientation 与 target region 不兼容。

此时先研究失败 source，不加 margin、不加 CE。

---

# 17. STOP

如果 calibration 和 test 都：

```text
< 80%
```

则：

\[
\boxed{\text{完整保留 source residual 的 class-core translation 不成立}}
\]

直接 STOP。

不要再通过：

```text
translation scale
target pull
margin
CE
core interpolation
```

去救这个假设。

---

# 18. 如果 probe 很强，意味着什么

假设结果为：

```text
Calibration 98%
Test        97%
Worst class 93%
```

那么 V6 的 90.67% CA 就有一个很直接的机制解释：

> V6 并不是为了 99% ASR 必须破坏 clean；它只是把本来只需要做 class translation 的问题错误地优化成了 sample collapse to target core。

这会是很强的机制证据。

---

# 19. Probe 通过后才考虑 V7

V7 真正训练时，不再使用：

\[
u_\phi(T(x))\rightarrow c_t.
\]

而是优化 trigger-induced differential response：

\[
\Delta_\phi(x)
=
u_\phi(T(x))-u_\phi(x).
\]

目标：

\[
\boxed{
\Delta_\phi(x)\rightarrow c_t-c_y
}
\]

clean 继续：

\[
u_\phi(x)\rightarrow u_0(x).
\]

但这一轮先不写 V7。

先让这个 probe 决定几何前提是否成立。

---

# 20. 跑完给回来的文件

日志：

```text
/data0/QURA/logs/decision_translation_probe_cifar10.log
```

JSON：

```text
ours/main/output/decision_translation_probe_resnet18_cifar10.json
```

我最需要看：

```text
calibration.overall_translation_success
calibration.worst_source_success

test.overall_translation_success
test.worst_source_success

test.per_source
class_core_within_mse
```

这些数字就足够决定下一步是 GO / MODIFY / STOP。
