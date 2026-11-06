# V7 — Structured Rounding Field + Residual-Preserving Decision Translation

## 基于 GitHub commit `fcb915c33b330445e5eab6fc9be875ccf546d2b7` 的完整可运行方案

这版直接建立在已经验证过的 V6 + Day-1 probe 上，不再改 field parameterization。

当前已知：

```text
Benign PTQ:
CA  = 93.24%
ASR = 2.06%

V6:
CA  = 90.67%
ASR = 99.71%

Day-1 residual-preserving translation probe:
Calibration overall = 100.000%
Test overall        = 99.911%
Test worst source   = 99.788%
```

因此 V7 不再研究“trigger 最后应该去哪一个点”。

它直接优化：

\[
\boxed{
\Delta_\phi(x)
=
u_\phi(T(x))
-
u_\phi(x)
}
\]

并要求：

\[
\boxed{
\Delta_\phi(x)
\rightarrow
c_t-c_y
}
\]

与此同时 clean 保持：

\[
\boxed{
u_\phi(x)
\rightarrow
u_0(x)
}
\]

也就是：

\[
\boxed{
\text{clean identity}
+
\text{trigger-conditioned class translation}
}
\]

---

# 1. V7 与 V6 的根本区别

V6：

\[
u_\phi(T(x))
\rightarrow c_t.
\]

设：

\[
u_0(x)=c_y+\rho(x).
\]

那么 V6 要求：

\[
c_y+\rho(x)
\rightarrow c_t
\]

因此实际 displacement：

\[
d^{V6}(x)
=
(c_t-c_y)-\rho(x).
\]

它额外抹掉：

\[
\rho(x).
\]

Day-1 probe 已经证明完全没必要。

---

V7：

\[
\boxed{
d^{V7}_y
=
c_t-c_y
}
\]

所以理想 triggered code：

\[
u^\star_T(x)
=
u_0(x)+(c_t-c_y)
=
c_t+\rho(x).
\]

source residual 被保留。

---

# 2. 为什么直接优化 differential response

V7 不把 attack 写成：

\[
u_\phi(T(x))
\rightarrow
u^\star_T(x)
\]

而是直接写：

\[
\boxed{
u_\phi(T(x))
-
u_\phi(x)
\rightarrow
c_t-c_y
}
\]

因为 static quantized model 没有显式 trigger branch。

后门选择性真正来自：

> 同一个 rounding field 对 clean input 与 triggered input 产生不同响应。

所以真正值得优化的是：

\[
\boxed{
\text{trigger-induced response}
}
\]

而不是一个 isolated attacked endpoint。

---

# 3. Paired-state formulation

对同一个 source：

clean state：

\[
u_c
=
u_\phi(x).
\]

trigger response：

\[
\Delta u
=
u_\phi(T(x))-u_\phi(x).
\]

目标：

\[
u_c^\star
=
u_0(x),
\]

\[
\Delta u^\star
=
c_t-c_y.
\]

所以定义 paired state：

\[
s_\phi(x)
=
\left[
u_\phi(x),
u_\phi(T(x))-u_\phi(x)
\right].
\]

目标 paired state：

\[
s^\star(x)
=
\left[
u_0(x),
c_t-c_y
\right].
\]

最终就是一个 squared-distance criterion。

代码实现时 clean samples 和 non-target response samples 的 squared distances concatenate 后直接求 mean。

没有：

```text
clean_weight
attack_weight
target margin
CE
KL
Dykstra
target core collapse
nearest target
prototype routing
full 512-D reconstruction
sparsity penalty
```

---

# 4. 保持哪些东西完全不变

为了让 V7 的结论可归因，以下全部保持 V6：

```text
StructuredRoundFakeQuantize
field_rank = 4
delta_bound = 0.10
round_temperature = 0.05
field_init_std = 0.02

final linear field = OFF
feature extractor field = ON

512 calibration samples
same optimized 6×6 stage2 trigger
same W4A4
same dynamic crop/flip
same lr = 0.01
same 5000 steps
same cosine scheduler
same calibration-only checkpoint score
same test diagnostic cadence
```

V7 唯一真正改变的是 attack supervision。

---

# 5. 文件操作

## 不删、不改

```text
mqbench/structured_rounding_attack.py
mqbench/structured_transport_attack.py
mqbench/structured_decision_set_attack.py
mqbench/structured_decision_core_attack.py
mqbench/decision_translation_probe.py

mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/fake_quantize/__init__.py
ours/main/setting/config.py
```

V3–V6 和 probe 都保留。

## 新增

```text
mqbench/structured_decision_translation_attack.py
ours/main/configs/cv_4_4_structured_decision_translation.yaml
```

## 修改

```text
ours/main/main.py
```

只增加独立 V7 branch。

---

# 6. 新增完整文件
## `mqbench/structured_decision_translation_attack.py`

```python
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)

# Reuse the exact, already-validated V6 infrastructure.
# V7 changes ONLY the attack supervision geometry.
from mqbench.structured_decision_core_attack import (
    _to_device,
    _extract_logits,
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

from mqbench.utils.state import enable_quantization


# ============================================================
# Exact classifier-input tap
# ============================================================

class V7FeatureTapper:
    """
    Hook the exact tensor consumed by the final classifier.

    Current repository:
        view_post_act_fake_quantizer -> linear
    """

    def __init__(self, model, module_name):
        modules = dict(model.named_modules())

        if module_name not in modules:
            nearby = [
                name
                for name in modules
                if (
                    "view" in name
                    or "linear" in name
                )
            ][:80]

            raise RuntimeError(
                "Classifier-input module not found: {}. "
                "Nearby names: {}".format(
                    module_name,
                    nearby,
                )
            )

        self.module_name = str(
            module_name
        )

        self.cache = None

        print(
            "[V7] exact semantic tap:",
            self.module_name,
        )

        self.handle = modules[
            self.module_name
        ].register_forward_hook(
            self._hook
        )

    def _hook(
        self,
        module,
        inputs,
        output,
    ):
        if isinstance(
            output,
            torch.Tensor,
        ):
            self.cache = output
            return

        if isinstance(
            output,
            (tuple, list),
        ):
            tensors = [
                item
                for item in output
                if isinstance(
                    item,
                    torch.Tensor,
                )
            ]

            if not tensors:
                raise RuntimeError(
                    "No tensor output at classifier-input tap."
                )

            self.cache = tensors[0]
            return

        raise RuntimeError(
            "Unsupported classifier-input output type: {}".format(
                type(output)
            )
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError(
                "Classifier-input hook did not fire."
            )

        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError(
            "Classifier-input feature must have batch/channel dimensions."
        )

    if feature.ndim > 2:
        feature = feature.mean(
            dim=tuple(
                range(
                    2,
                    feature.ndim,
                )
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

    batch = _to_device(
        batch,
        device,
    )

    if isinstance(
        batch,
        dict,
    ):
        output = model(
            **batch
        )
    else:
        output = model(
            batch
        )

    logits = _extract_logits(
        output
    )

    feature = _semantic_vector(
        tapper.snapshot()
    )

    return (
        logits,
        feature,
    )


# ============================================================
# Structured-field attack surface
# ============================================================

def _collect_attack_records(
    model,
    frozen_field_modules,
):
    """
    Same attack surface as V4/V5/V6:

        feature extractor:
            structured malicious rounding field ON

        final linear:
            benign nearest-PTQ only
    """

    frozen_field_modules = set(
        str(name)
        for name in frozen_field_modules
    )

    attack_records = []
    frozen_records = []
    all_records = []

    for (
        name,
        module,
    ) in model.named_modules():
        fq = getattr(
            module,
            "weight_fake_quant",
            None,
        )

        if (
            isinstance(
                fq,
                StructuredRoundFakeQuantize,
            )
            and getattr(
                module,
                "weight",
                None,
            ) is not None
        ):
            record = {
                "name": name,
                "module": module,
                "fq": fq,
            }

            all_records.append(
                record
            )

            if name in frozen_field_modules:
                frozen_records.append(
                    record
                )
            else:
                attack_records.append(
                    record
                )

    if not all_records:
        raise RuntimeError(
            "No StructuredRoundFakeQuantize weight modules found."
        )

    existing = {
        record[
            "name"
        ]
        for record in all_records
    }

    missing = (
        frozen_field_modules
        - existing
    )

    if missing:
        raise RuntimeError(
            "Frozen field modules not found: {}".format(
                sorted(
                    missing
                )
            )
        )

    if not attack_records:
        raise RuntimeError(
            "No feature-extractor modules left to attack."
        )

    print(
        "[V7] structured-field modules:",
        len(
            attack_records
        ),
    )

    print(
        "[V7] frozen-field modules:",
        [
            record[
                "name"
            ]
            for record in frozen_records
        ],
    )

    return (
        attack_records,
        frozen_records,
    )


# ============================================================
# Frozen classifier geometry sanity
# ============================================================

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
        exact tapped feature + actual quantized W reproduces model logits.

    Check 2:
        decision coordinates alone preserve classifier argmax.
    """

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

    manual_logits = F.linear(
        feature.float(),
        geometry[
            "weight"
        ],
        geometry[
            "bias"
        ],
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
        "[V7] classifier geometry max-logit error:",
        max_abs_error,
    )

    if (
        max_abs_error
        > float(
            tolerance
        )
    ):
        raise RuntimeError(
            "Exact classifier geometry mismatch: {}".format(
                max_abs_error
            )
        )

    code = decision_code(
        feature.float(),
        geometry[
            "basis"
        ],
    )

    code_logits = F.linear(
        code,
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    agreement = float(
        (
            code_logits.argmax(
                dim=1
            )
            == logits.argmax(
                dim=1
            )
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[V7] decision-code argmax agreement:",
        agreement,
    )

    if agreement < 100.0:
        raise RuntimeError(
            "Decision coordinates do not preserve classifier argmax."
        )

    return {
        "max_logit_error": (
            max_abs_error
        ),
        "argmax_agreement": (
            agreement
        ),
    }


# ============================================================
# All benign class cores
# ============================================================

@torch.no_grad()
def build_class_decision_cores(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    min_class_count,
):
    """
    Build one benign decision core per pseudo class:

        c_y =
            mean[
                u_0(x)
                |
                Q_0(x) = y
            ]

    These are the SAME geometric objects validated by the
    Day-1 probe.
    """

    num_classes = int(
        geometry[
            "num_classes"
        ]
    )

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

    cores = []
    counts = []
    within_class_mse = []

    for cls in range(
        num_classes
    ):
        if not per_class[
            cls
        ]:
            raise RuntimeError(
                "No calibration samples for pseudo class {}.".format(
                    cls
                )
            )

        cls_codes = torch.cat(
            per_class[
                cls
            ],
            dim=0,
        )

        count = int(
            cls_codes.shape[
                0
            ]
        )

        if (
            count
            < int(
                min_class_count
            )
        ):
            raise RuntimeError(
                "Too few calibration samples for class {}: {} < {}".format(
                    cls,
                    count,
                    int(
                        min_class_count
                    ),
                )
            )

        core = cls_codes.mean(
            dim=0,
            keepdim=True,
        )

        mse = float(
            (
                (
                    cls_codes
                    - core
                )
                .pow(2)
                .sum(
                    dim=1
                )
            )
            .mean()
            .item()
        )

        cores.append(
            core
        )

        counts.append(
            count
        )

        within_class_mse.append(
            mse
        )

    cores = torch.cat(
        cores,
        dim=0,
    )

    # Every core should remain in its own convex class region.
    core_logits = F.linear(
        cores.float(),
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    core_pred = core_logits.argmax(
        dim=1
    )

    expected = torch.arange(
        num_classes,
        device=core_pred.device,
    )

    core_agreement = float(
        (
            core_pred
            == expected
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[V7] class-core self-classification:",
        core_agreement,
    )

    if core_agreement < 100.0:
        raise RuntimeError(
            "At least one class decision core is outside its own class region."
        )

    return {
        "cores": (
            cores.detach()
        ),
        "counts": (
            counts
        ),
        "within_class_mse": (
            within_class_mse
        ),
        "core_self_agreement": (
            core_agreement
        ),
    }


@torch.no_grad()
def calibration_translation_sanity(
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
    Re-run the Day-1 translation geometry on the calibration set:

        u_hat =
            u_0
            + (c_t - c_y)

    V7 should start only if this geometric premise is still valid
    under the current exact calibration state.
    """

    target = int(
        target
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

        desired_delta = (
            class_cores[
                target
            ].unsqueeze(
                0
            )
            - class_cores[
                source_cls
            ]
        )

        translated = (
            source_code
            + desired_delta
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
            str(
                cls
            )
        ] = rate

    worst = min(
        per_source_rates.values()
    )

    print(
        "[V7] calibration translation sanity:",
        {
            "overall": (
                overall
            ),
            "worst_source": (
                worst
            ),
            "per_source": (
                per_source_rates
            ),
        },
    )

    return {
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


# ============================================================
# ONE paired-state objective
# ============================================================

def paired_decision_translation_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    class_cores,
    target,
    basis,
):
    """
    V7 target mapping:

    Clean state:
        u_phi(x)
            -> u_0(x)

    Trigger-induced response:
        Delta_phi(x)
          = u_phi(T(x))
            - u_phi(x)

        Delta_phi(x)
            -> c_t - c_y

    The attack no longer optimizes an absolute triggered endpoint.

    Instead it learns the conditional response caused by the trigger.

    ONE paired-state squared-distance objective:
        state_phi = [
            u_phi(x),
            Delta_phi(x)
        ]

        state_star = [
            u_0(x),
            c_t - c_y
        ]
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

    clean_source_code = clean_code[
        source_mask
    ]

    actual_delta = (
        trigger_code
        - clean_source_code
    )

    cores = class_cores.to(
        device=clean_code.device,
        dtype=clean_code.dtype,
    )

    desired_delta = (
        cores[
            int(
                target
            )
        ]
        .unsqueeze(
            0
        )
        - cores[
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

    # Equal sample-level treatment in one paired-state metric.
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
        "translation_response_distance": (
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
    class_cores,
    trigger,
):
    """
    Reports:

    1. clean teacher agreement;
    2. clean decision-code distance;
    3. standard calibration ASR on repository cali_data_bd;
    4. paired trigger response distance on matched clean/trigger samples;
    5. paired ASR and per-source diagnostics.
    """

    target = int(
        target
    )

    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    cores = class_cores.to(
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
                dtype=(
                    teacher_feature.dtype
                )
            ),
        )

        attacked_code = decision_code(
            attacked_feature,
            basis.to(
                dtype=(
                    attacked_feature.dtype
                )
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
                dtype=(
                    trigger_feature.dtype
                )
            ),
        )

        clean_source_code = attacked_code[
            source_mask
        ]

        actual_delta = (
            trigger_code
            - clean_source_code
        )

        desired_delta = (
            cores[
                target
            ]
            .unsqueeze(
                0
            )
            - cores[
                source_labels
            ]
        )

        response_d2 = (
            (
                actual_delta
                - desired_delta.to(
                    dtype=(
                        actual_delta.dtype
                    )
                )
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

        response_total += int(
            hit.numel()
        )

        paired_hits += int(
            hit.sum()
            .item()
        )

        response_distance_sum += float(
            response_d2.sum()
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

    # Standard benchmark calibration ASR.
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
            "translation_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
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
        "translation_response_distance": (
            response_distance_sum
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
# Main V7 attack
# ============================================================

def structured_decision_translation_attack(
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
    V7 = Structured Rounding Field
         + Residual-Preserving Decision Translation.

    Core conditional behavior:

        clean:
            u_phi(x)
                -> u_0(x)

        trigger response:
            u_phi(T(x)) - u_phi(x)
                -> c_t - c_y

    The final quantized classifier stays benign and frozen.
    """

    device = next(
        model.parameters()
    ).device

    model.eval()

    enable_quantization(
        model
    )

    # Freeze all pretrained network parameters.
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
    # Same structured-field capacity as V4/V5/V6.
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
    # Exact optimized trigger.
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
        "[V7] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Actual deployed benign quantized classifier geometry.
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
        "[V7] classifier geometry:",
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
    # Build ALL class cores, not only target core.
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

    translation_sanity = calibration_translation_sanity(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        target=target,
    )

    if (
        translation_sanity[
            "overall"
        ]
        < float(
            config.min_translation_sanity
        )
    ):
        raise RuntimeError(
            "Residual-preserving translation geometry sanity failed: "
            "{:.4f} < {:.4f}".format(
                translation_sanity[
                    "overall"
                ],
                float(
                    config.min_translation_sanity
                ),
            )
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
        class_cores=class_cores,
        trigger=trigger,
    )

    print(
        "[V7] step-0 calibration:",
        baseline,
    )

    print(
        "[V7] step-0 structured stats:",
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
    # Optimizer: EXACT same capacity/training schedule as V6.
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
    # Same calibration-only checkpoint criterion as V6.
    # No test leakage.
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

        # Benign PTQ state on exactly the same dynamic clean view.
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

        losses = paired_decision_translation_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            class_cores=class_cores,
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
                class_cores=class_cores,
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
                "train_translation_response_distance": float(
                    losses[
                        "translation_response_distance"
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

            # Diagnostic only.
            # Never enters checkpoint selection.
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
                "[V7]",
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
                    "[V7] new best:",
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
                        "translation_response_distance": (
                            metrics[
                                "translation_response_distance"
                            ]
                        ),
                    },
                )

    # --------------------------------------------------------
    # Restore calibration-selected best state.
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
        class_cores=class_cores,
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
        "[V7] restored best step:",
        best_step,
    )

    print(
        "[V7] restored calibration:",
        restored,
    )

    print(
        "[V7] restored test:",
        test,
    )

    print(
        "[V7] restored structured stats:",
        stats,
    )

    # Class translation magnitudes for diagnostics.
    target_core = class_cores[
        target
    ]

    translation_norms = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        translation_norms[
            str(
                cls
            )
        ] = float(
            (
                target_core
                - class_cores[
                    cls
                ]
            )
            .norm(
                p=2
            )
            .item()
        )

    result = {
        "method": (
            "structured_rounding_field_"
            "residual_preserving_decision_translation"
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
            "translation_norm_to_target": (
                translation_norms
            ),
        },
        "translation_geometry_sanity": (
            translation_sanity
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

    # --------------------------------------------------------
    # Materialize feature-extractor malicious rounding only.
    # Final classifier remains benign PTQ.
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
            "[V7] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

```

---

# 7. 新增完整配置
## `ours/main/configs/cv_4_4_structured_decision_translation.yaml`

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
    quantize_type: structured_decision_translation_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # ------------------------------------------------
        # Structured Rounding Field
        # EXACTLY keep V6 capacity for a clean mechanism test.
        # ------------------------------------------------
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # ------------------------------------------------
        # Frozen classifier decision geometry
        # ------------------------------------------------
        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        frozen_field_modules:
            - linear

        decision_svd_tol: 1.0e-6
        geometry_logit_tolerance: 1.0e-4

        # All 10 benign pseudo-class decision cores.
        min_class_count: 20

        # Day-1 probe was 100% on calibration.
        # This is only a setup sanity check, not an attack hyperparameter.
        min_translation_sanity: 99.0

        # ------------------------------------------------
        # Training
        # ------------------------------------------------
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Same dynamic-view protocol as V6.
        # Trigger is pasted AFTER augmentation.
        aug_pad: 4
        hflip_prob: 0.5

        # ------------------------------------------------
        # Logging
        # ------------------------------------------------
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

# 8. 修改 `ours/main/main.py`

当前最新仓库顺序开头是：

```python
if config.quantize.quantize_type == 'decision_translation_probe':
    ...
elif config.quantize.quantize_type == 'structured_decision_core_attack':
    ...
```

把 V7 放在 probe 前面：

```python
        if config.quantize.quantize_type == 'structured_decision_translation_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V7 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V7 first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # ------------------------------------------------
            # EXACT same calibration subset construction as V6.
            # ------------------------------------------------

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

            from mqbench.structured_decision_translation_attack import (
                structured_decision_translation_attack,
            )

            model.eval()

            # ------------------------------------------------
            # EXACT same observer calibration as V6.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Benign PTQ reference.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Output paths.
            # ------------------------------------------------

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_decision_translation_v7_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_translation_v7_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            # ------------------------------------------------
            # Attack / reload.
            # ------------------------------------------------

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
                    structured_decision_translation_attack(
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
                    'Saved V7 quantized model:',
                    checkpoint_path)

            # ------------------------------------------------
            # Final standard benchmark evaluation.
            # ------------------------------------------------

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Translation V7 =========='
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

        elif config.quantize.quantize_type == 'decision_translation_probe':

```

最后一行已经是：

```python
elif config.quantize.quantize_type == 'decision_translation_probe':
```

因此将当前 probe branch 原来 `if config.quantize.quantize_type == 'decision_translation_probe':`
下面从：

```python
if args.type != 'bd':
```

开始的内容原样接过去。

最终顺序：

```text
V7 structured_decision_translation_attack
Day-1 decision_translation_probe
V6 structured_decision_core_attack
V5 structured_decision_set_attack
V4 structured_transport_attack
V3 structured_rounding_attack
QuRA advanced_ptq
```

---

# 9. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_decision_translation.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_decision_translation_v7_cifar10.log
```

---

# 10. V7 开跑前 sanity

必须看到：

```text
[V7] exact semantic tap:
view_post_act_fake_quantizer

[V7] frozen-field modules:
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
[V7] classifier geometry max-logit error:
≈ 0

[V7] decision-code argmax agreement:
100.0
```

---

# 11. Class cores sanity

应看到：

```text
[V7] class-core self-classification:
100.0
```

并重新执行 calibration translation sanity。

根据当前 commit 的 probe，应该接近：

```text
overall      = 100%
worst_source = 100%
```

配置中只要求：

```text
min_translation_sanity = 99.0
```

这只是防止仓库/observer 状态改变，不是 attack hyperparameter。

---

# 12. Step-0 sanity

必须：

```text
total_flipped = 0
clean_teacher_agreement = 100%
clean_code_distance ≈ 0
```

此时：

```text
translation_response_distance
```

不会是 0。

这是正常的。

因为 benign PTQ 本身的 trigger response 还没有学会：

\[
c_t-c_y.
\]

---

# 13. 训练日志新增的真正关键指标

每 250 step：

```text
train_clean_code_distance
train_translation_response_distance

clean_teacher_agreement
clean_code_distance

asr
paired_asr
translation_response_distance

per_source:
    paired_asr
    translation_response_distance

flip_ratio_valid
total_flipped
```

每 500 step：

```text
diagnostic_test_ca
diagnostic_test_asr
```

---

# 14. `ASR` 和 `paired_asr` 的区别

`asr`：

使用仓库原来的：

```text
cali_data_bd
```

保持与 V3–V6 完全一致的 benchmark calibration ASR。

`paired_asr`：

从同一 clean calibration batch 直接复制，再贴入同一 optimized trigger。

它用于和：

\[
u_\phi(T(x))-u_\phi(x)
\]

逐样本对应。

正式比较仍看：

```text
asr
```

---

# 15. 这轮最关键的预期

V6：

```text
CA  = 90.67
ASR = 99.71
```

V5：

```text
CA  = 93.15
ASR = 72.67
```

V7 如果真正解决 V6 的 residual erasure，应该表现为：

```text
ASR 仍接近 V6
CA 明显向 V5 / benign PTQ 恢复
```

---

# 16. Strong GO

第一目标：

```text
Test CA  >= 92.5%
Test ASR >= 98.0%
```

如果达到：

```text
CA  >= 92.8%
ASR >= 99.0%
```

我会把 V7 直接视为当前主方法候选。

---

# 17. 最理想结果

例如：

```text
Benign PTQ:
93.24 / 2.06

V7:
92.9 / 99.3
```

这会非常强。

它说明：

1. 9-D decision geometry 足够描述 malicious behavior；
2. class-core translation 几何具有近乎完美 test generalization；
3. structured rounding field 能真实实现 source-dependent translation；
4. 不需要修改 final classifier；
5. 不需要 target margin；
6. 不需要把 trigger representations 压到一个 target point；
7. clean utility 明显优于 V6。

---

# 18. 如果 ASR 仍 99+，但 CA 还是 90–91

例如：

```text
CA  = 90.8
ASR = 99.5
```

那么非常重要：

> residual erasure 不是 V6 clean 损伤的主要来源。

这时不要给 clean loss 加权。

下一问题直接变成：

\[
\boxed{
\text{同一个 static structured field
为了实现 class-dependent trigger response，
是否必然产生 clean decision drift？}
}
\]

也就是：

```text
condition selectivity
```

而不是 target geometry。

---

# 19. 如果 CA 回到 93 左右，但 ASR 只有 80–90

例如：

```text
CA  = 93.0
ASR = 85
```

那么 probe 的几何是对的，但：

\[
\boxed{
\text{static structured field 无法实现 source-dependent }c_t-c_y
}
\]

也就是说真正瓶颈不是目标设计，而是：

```text
rounding-field controllability
```

这时应该研究 field parameterization / trigger-conditioned Jacobian，而不是再换 target。

---

# 20. 如果总体好，但某一个 source class 明显差

例如：

```text
overall ASR = 97
source 2 ASR = 70
others = 99
```

直接看 JSON：

```text
best_calibration.per_source
```

比较：

```text
paired_asr
translation_response_distance
translation_norm_to_target
```

如果只有单类异常，就说明 shared field 对某个 class translation 的可实现性不足。

这比盲目改整体 objective 有价值。

---

# 21. 如果 V7 成功，论文机制链会变得很完整

可以写成：

### Observation 1
Quantized classifier 只依赖低维 decision subspace：

\[
r\le C-1.
\]

### Observation 2
Benign classes 在该空间中具有强 residual-preserving translation structure：

\[
u-c_y+c_t
\]

在完整 test set 上有 99.911% target success。

### Method
Structured rounding 不再拟合 target endpoint，而学习 trigger-induced class translation：

\[
u_\phi(T(x))-u_\phi(x)
\approx
c_t-c_y.
\]

最终 classifier 保持 benign。

这比之前 V3–V6 的故事都更加集中。

---

# 22. 跑完给我的文件

日志：

```text
/data0/QURA/logs/
structured_decision_translation_v7_cifar10.log
```

JSON：

```text
ours/main/output/
structured_decision_translation_v7_resnet18_cifar10.json
```

重点字段：

```text
best_step
best_test
best_calibration

history:
    clean_code_distance
    translation_response_distance
    asr
    paired_asr
    diagnostic_test_ca
    diagnostic_test_asr
    flip_ratio_valid

best_calibration.per_source

structured_stats
class_cores
translation_geometry_sanity
```

---

# 23. V7 一句话

> **V7 learns a structured rounding field whose clean response preserves the benign decision code, while the trigger induces a source-class-dependent translation from the source class core to the target class core.**

即：

\[
\boxed{
u_\phi(x)\approx u_0(x)
}
\]

同时：

\[
\boxed{
u_\phi(T(x))-u_\phi(x)
\approx
c_t-c_y.
}
\]

这是这轮要验证的唯一核心。
