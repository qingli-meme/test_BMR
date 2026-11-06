# V4 — Structured Rounding Field + Conditional Semantic Transport
## 基于 `qingli-meme/T_Q_13_B_D_A_T_20` 当前 `main` 分支的直接修改版

这份方案完全按当前仓库实际代码写，不沿用旧版本记忆。对齐过的关键文件包括：

```text
mqbench/structured_rounding_attack.py
mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/fake_quantize/__init__.py
ours/main/main.py
ours/main/configs/cv_4_4_structured_pathway.yaml
ours/main/setting/config.py
RESULTS.md
```

当前仓库 V3 已经得到：

```text
Benign PTQ CA  = 93.24%
Benign PTQ ASR = 2.06%
V3 test CA     = 88.54%
V3 test ASR    = 97.94%
Calibration ASR = 99.80%
Calibration clean agreement = 99.80%
Final flip ratio = 8.09%
```

真正关键的诊断是：benign PTQ 自身的 stage-route accuracy 为

```text
layer2 = 60.16%
layer3 = 81.84%
layer4 = 100.00%
```

训练后 clean 的 layer2/layer3 route accuracy 降到约 10.16%，而 target=0 在 calibration 中正好是 52/512=10.15625%。因此 V3 的 multi-stage routing 实际诱导了 early/mid target collapse，再由后端语义层与 classifier 把 calibration clean 样本救回来。

V4 直接停止 multi-stage routing，只保留已经被 V3 证明具有强攻击泛化能力的 Structured Rounding Field，并把优化目标改为单一的最终 classifier-input semantic transport。

还有一个基于当前仓库代码得到的重要修正：V3 日志中的 `layer4 -> avgpool` **不是 final linear 的精确输入**。当前 MQBench 图中还插入了：

```text
view_post_act_fake_quantizer
```

它位于 flatten/view 之后、`linear` 之前。因此 V4 不再 tap `avgpool`，而是直接 hook：

```text
view_post_act_fake_quantizer
```

这样下面关于 frozen linear classifier decision region 的几何推导才是严格对应实际代码路径的。

---

# 1. V4 方法骨架

保留：

\[
\boxed{\text{Structured Rounding Field}}
\]

删除：

```text
layer2 prototype routing
layer3 prototype routing
layer4 prototype CE
output CE
target margin
KL
clean margin
primal-dual
sparsity loss
per-weight malicious gate
```

新增一个统一机制：

\[
\boxed{\text{Conditional Semantic Transport}}
\]

并且 final classifier 的 structured field 完全关闭。

---

# 2. 冻结 benign classifier

设 benign PTQ feature extractor（直到 `view_post_act_fake_quantizer` 输出）为：

\[
F_0,
\]

benign PTQ final linear classifier 为：

\[
C_0.
\]

攻击后只改变 feature extractor 的 rounding field：

\[
F_\phi.
\]

最终模型：

\[
\boxed{Q_\phi(x)=C_0(F_\phi(x))}.
\]

当前仓库 ResNet18 的最后分类模块名是：

```text
linear
```

V4 不对它调用：

```python
init_structured_attack(...)
```

因此 `linear.weight_fake_quant` 保持 `structured_attack=False`，继续走当前 `StructuredRoundFakeQuantize` 的 parent nearest-PTQ forward。

---

# 3. 为什么只监督 final classifier input

V3 的实验已经证明：

```text
layer2 benign semantic separability 不够
layer3 仍不完全稳定
layer4/late representation 最稳定
```

但为了让“transport 到 target region”这件事与 frozen classifier 严格闭环，V4 再向后走一步，直接在：

```text
view_post_act_fake_quantizer
```

上监督。

这个 tensor 才是 final `linear` 真正接收的 quantized representation。

因此 V4 不再对 layer2 / layer3 / avgpool 强行施加额外类别语义结构。

---

# 4. Clean：Identity Transport

对动态 clean view：

\[
a(x)
\]

benign PTQ classifier-input representation：

\[
h_0(x)=F_0(a(x)).
\]

攻击模型 representation：

\[
h_\phi(x)=F_\phi(a(x)).
\]

要求：

\[
\boxed{h_\phi(x)\rightarrow h_0(x)}.
\]

不是 prototype 分类，也不是 output KL，而是直接保持 frozen classifier 实际消费的 raw quantized semantic feature。

---

# 5. Trigger：Conditional Target Transport

从 calibration clean set 中，用 benign PTQ 自己的预测构造 target feature bank：

\[
\mathcal H_t
=
\{h_0(x_j):\arg\max C_0(h_0(x_j))=t\}.
\]

无需 ground-truth label。

对 non-target source：

\[
h_s=F_0(a(x)).
\]

按 cosine similarity 从 target bank 找最接近的 K 个真实 target features：

\[
\mathcal N_K(h_s).
\]

第一轮：

\[
K=5.
\]

构造 source-conditioned anchor：

\[
\boxed{
A_t(h_s)
=
\frac{1}{K}
\sum_{h_j\in\mathcal N_K(h_s)}h_j
}.
\]

triggered source：

\[
\tilde x=T(a(x)).
\]

要求：

\[
\boxed{F_\phi(\tilde x)\rightarrow A_t(h_s)}.
\]

不同 source 对应 target manifold 中不同局部 anchor，不再把所有 triggered samples 压到一个单一 centroid。

---

# 6. 为什么 anchor 天然在 frozen classifier 的 target region

冻结 classifier：

\[
C_0(h)=Wh+b.
\]

target class \(t\) 的 decision region：

\[
\mathcal R_t
=
\bigcap_{j\neq t}
\{h:(w_t-w_j)^\top h+(b_t-b_j)\ge0\}.
\]

它是多个 half-space 的交，因此是 convex set。

target bank 中每个 feature 都由 benign classifier 预测为 target：

\[
h_j\in\mathcal R_t.
\]

于是任意 convex combination 都仍然在 \(\mathcal R_t\) 中。我们的 uniform Top-K mean 正是 convex combination：

\[
\boxed{A_t(h_s)\in\mathcal R_t}.
\]

因此 V4 不需要 target CE，也不需要 target margin。目标 feature 本身就位于 frozen benign classifier 的 target decision region。

---

# 7. 唯一训练距离

V4 使用 classifier-input raw feature 的 relative MSE：

\[
\boxed{
D(h,r)
=
\frac{\|h-r\|_2^2}{\|r\|_2^2+\epsilon}
}.
\]

clean：

\[
L_{clean}
=
D(F_\phi(a(x)),F_0(a(x))).
\]

trigger：

\[
L_{trigger}
=
D(F_\phi(T(a(x))),A_t(F_0(a(x)))).
\]

总目标：

\[
\boxed{
L_{transport}
=
\frac{1}{2}L_{clean}
+
\frac{1}{2}L_{trigger}
}.
\]

只有这一类 transport objective，没有额外补丁 loss。

---

# 8. 与当前 V3 相比真正删掉什么

V4 完全删除：

```text
layer2 prototype routing
layer3 prototype routing
layer4 prototype routing
output CE
class prototypes
proto_temperature
num_classes
min_proto_count
multi-stage route accuracy
```

并且：

```text
linear structured field = OFF
```

继续保留：

```text
StructuredRoundFakeQuantize
low-rank structured field
hard forward + STE
same optimized trigger
same W4A4
same calibration set
same 5000 steps
same rank=4
same delta=0.10
same lr=0.01
same dynamic augmentation
```

所以这一轮是一个非常干净的机制替换。

---

# 9. 仓库文件增删改总览

## 删除

没有必须删除的文件。

旧 V3 保留：

```text
mqbench/structured_rounding_attack.py
ours/main/configs/cv_4_4_structured_pathway.yaml
```

用于直接 baseline。

## 不修改

```text
mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/fake_quantize/__init__.py
ours/main/setting/config.py
```

当前 quantizer 已经满足 V4，不要再改。

## 新增

```text
mqbench/structured_transport_attack.py
ours/main/configs/cv_4_4_structured_transport.yaml
```

## 修改

```text
ours/main/main.py
```

只新增一个 `structured_transport_attack` 分支，原 V3 分支原封不动保留。

---

# 10. 新增完整文件：`mqbench/structured_transport_attack.py`

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


class SemanticTapper:
    """Tap one semantic stage using the same FX resolution as current V3."""

    def __init__(self, model, stage_name):
        modules = dict(model.named_modules())
        if stage_name not in modules:
            raise RuntimeError("Missing semantic stage: {}".format(stage_name))

        graph_targets = []
        if hasattr(model, "graph"):
            graph_targets = [
                str(node.target)
                for node in model.graph.nodes
                if node.op == "call_module"
            ]

        candidate_indices = [
            i for i, target in enumerate(graph_targets)
            if target.startswith(stage_name + ".") and target in modules
        ]

        actual_name = stage_name
        if candidate_indices:
            last_index = candidate_indices[-1]
            if last_index + 1 < len(graph_targets):
                actual_name = graph_targets[last_index + 1]
            else:
                actual_name = graph_targets[last_index]

        if actual_name not in modules:
            raise RuntimeError(
                "Cannot resolve semantic tap {} -> {}".format(
                    stage_name, actual_name
                )
            )

        self.stage_name = stage_name
        self.actual_name = actual_name
        self.cache = None

        print("[V4] semantic tap:", stage_name, "->", actual_name)
        self.handle = modules[actual_name].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.cache = output
            return
        if isinstance(output, (tuple, list)):
            tensors = [x for x in output if isinstance(x, torch.Tensor)]
            if not tensors:
                raise RuntimeError("No tensor at semantic stage.")
            self.cache = tensors[0]
            return
        raise RuntimeError(
            "Unsupported semantic-stage output type: {}".format(type(output))
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError("Semantic hook did not fire.")
        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError("Semantic feature needs batch/channel dimensions.")
    if feature.ndim > 2:
        feature = feature.mean(dim=tuple(range(2, feature.ndim)))
    return feature


def _attacked_forward(model, tapper, batch, device):
    tapper.clear()
    logits = _forward_logits(model, batch, device)
    feature = _semantic_vector(tapper.snapshot())
    return logits, feature


def _effective_weight(module):
    weight = module.weight
    if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
        factor = module.bn.weight / torch.sqrt(
            module.bn.running_var + module.bn.eps
        )
        return weight * factor.reshape([-1] + [1] * (weight.ndim - 1))
    return weight


def _collect_attack_records(model, frozen_field_modules):
    frozen_field_modules = set(str(x) for x in frozen_field_modules)
    attack_records = []
    frozen_records = []
    all_records = []

    for name, module in model.named_modules():
        fq = getattr(module, "weight_fake_quant", None)
        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and getattr(module, "weight", None) is not None
        ):
            rec = {"name": name, "module": module, "fq": fq}
            all_records.append(rec)
            if name in frozen_field_modules:
                frozen_records.append(rec)
            else:
                attack_records.append(rec)

    if not all_records:
        raise RuntimeError("No StructuredRoundFakeQuantize modules found.")

    missing = frozen_field_modules - {rec["name"] for rec in all_records}
    if missing:
        raise RuntimeError(
            "Requested frozen field modules not found: {}".format(
                sorted(missing)
            )
        )

    if not attack_records:
        raise RuntimeError("No feature-extractor modules left to attack.")

    print("[V4] structured-field modules:", len(attack_records))
    print("[V4] frozen-field modules:", [rec["name"] for rec in frozen_records])
    return attack_records, frozen_records


def _set_force_base(records, enabled):
    for rec in records:
        rec["fq"].set_structured_force_base(enabled)


@torch.no_grad()
def _teacher_forward(model, attack_records, tapper, batch, device):
    _set_force_base(attack_records, True)
    try:
        tapper.clear()
        logits = _forward_logits(model, batch, device).detach()
        feature = _semantic_vector(tapper.snapshot()).detach()
    finally:
        _set_force_base(attack_records, False)
    return logits, feature


def _unwrap_dataset(dataset):
    seen = set()
    while hasattr(dataset, "dataset") and id(dataset) not in seen:
        seen.add(id(dataset))
        dataset = dataset.dataset
    return dataset


def _flatten_transforms(transform):
    if transform is None:
        return []
    if hasattr(transform, "transforms"):
        output = []
        for item in transform.transforms:
            output.extend(_flatten_transforms(item))
        return output
    return [transform]


def extract_trigger_spec(loader):
    dataset = _unwrap_dataset(loader.dataset)
    ops = _flatten_transforms(getattr(dataset, "transform", None))
    trigger = mean = std = None

    for op in ops:
        if (
            trigger is None
            and isinstance(getattr(op, "trigger", None), torch.Tensor)
        ):
            trigger = op.trigger.detach().float().clone()

        if hasattr(op, "mean") and hasattr(op, "std"):
            try:
                m = torch.as_tensor(op.mean).float()
                s = torch.as_tensor(op.std).float()
                if m.numel() == 3 and s.numel() == 3:
                    mean = m.reshape(3, 1, 1)
                    std = s.reshape(3, 1, 1)
            except Exception:
                pass

    if trigger is None or mean is None or std is None:
        raise RuntimeError(
            "Could not extract trigger/Normalize from train_loader_bd."
        )

    if trigger.ndim == 4:
        trigger = trigger[0]
    if trigger.ndim != 3:
        raise ValueError("Expected CHW trigger.")

    return {
        "trigger_normalized": (trigger - mean) / std,
        "trigger_size": int(trigger.shape[-1]),
    }


def _augment_clean_tensor(x, pad=4, hflip_prob=0.5):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("Expected BCHW clean tensor.")

    if pad > 0:
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        b, _, h, w = x.shape
        crops = []
        for i in range(b):
            top = int(
                torch.randint(
                    0, 2 * pad + 1, (1,), device=x.device
                ).item()
            )
            left = int(
                torch.randint(
                    0, 2 * pad + 1, (1,), device=x.device
                ).item()
            )
            crops.append(
                padded[i:i + 1, :, top:top + h, left:left + w]
            )
        x = torch.cat(crops, dim=0)

    if hflip_prob > 0:
        mask = torch.rand(x.shape[0], device=x.device) < float(hflip_prob)
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(x[mask], dims=[3])

    return x


def _apply_trigger(x, trigger):
    x = x.clone()
    trigger = trigger.to(device=x.device, dtype=x.dtype)
    size = int(trigger.shape[-1])
    x[:, :, -size:, -size:] = trigger.unsqueeze(0)
    return x


@torch.no_grad()
def build_target_feature_bank(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    target,
    min_target_bank,
):
    bank = []

    for batch in cali_data:
        batch = _to_device(batch, device)
        logits, feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        pseudo = logits.argmax(dim=1)
        mask = pseudo == int(target)
        if mask.any():
            bank.append(feature[mask].detach().clone())

    if not bank:
        raise RuntimeError("No benign target features in calibration set.")

    bank = torch.cat(bank, dim=0)
    if bank.shape[0] < int(min_target_bank):
        raise RuntimeError(
            "Target feature bank too small: {} < {}".format(
                bank.shape[0], int(min_target_bank)
            )
        )

    print("[V4] target feature bank:", tuple(bank.shape))
    return bank.detach()


@torch.no_grad()
def conditional_target_anchor(source_feature, target_bank, topk):
    k = max(1, min(int(topk), int(target_bank.shape[0])))

    source_n = F.normalize(source_feature, p=2, dim=1, eps=1e-8)
    bank_n = F.normalize(target_bank, p=2, dim=1, eps=1e-8)
    similarity = source_n @ bank_n.transpose(0, 1)

    indices = torch.topk(
        similarity,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    selected = target_bank[indices]
    return selected.mean(dim=1)


def relative_feature_distance(feature, reference, eps=1e-8):
    numerator = (feature - reference).pow(2).sum(dim=1)
    denominator = reference.pow(2).sum(dim=1).clamp_min(float(eps))
    return (numerator / denominator).mean()


def semantic_transport_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    target_anchor,
):
    clean_transport = relative_feature_distance(
        clean_feature, clean_reference
    )
    trigger_transport = relative_feature_distance(
        trigger_feature, target_anchor
    )
    total = 0.5 * (clean_transport + trigger_transport)
    return {
        "total": total,
        "clean_transport": clean_transport,
        "trigger_transport": trigger_transport,
    }


@torch.no_grad()
def _structured_stats(records):
    total_valid = 0
    total_flipped = 0
    params = 0
    layers = []

    for rec in records:
        fq = rec["fq"]
        valid = fq.structured_valid_mask.bool()
        flips = fq.current_structured_flip_mask()
        field = fq.structured_field().detach()

        valid_n = int(valid.sum().item())
        flip_n = int(flips.sum().item())
        param_n = int(fq.field_u.numel() + fq.field_v.numel())

        total_valid += valid_n
        total_flipped += flip_n
        params += param_n

        layers.append({
            "name": rec["name"],
            "valid": valid_n,
            "flipped": flip_n,
            "flip_ratio_valid": flip_n / max(valid_n, 1),
            "field_rank": int(fq.structured_rank),
            "field_params": param_n,
            "field_abs_mean": float(field.abs().mean().item()),
            "field_abs_max": float(field.abs().max().item()),
        })

    return {
        "total_valid": int(total_valid),
        "total_flipped": int(total_flipped),
        "flip_ratio_valid": total_flipped / max(total_valid, 1),
        "field_parameters": int(params),
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
):
    clean_total = 0
    clean_agree = 0
    clean_feature_error_sum = 0.0

    for batch in clean_data:
        batch = _to_device(batch, device)
        teacher_logits, teacher_feature = _teacher_forward(
            model, attack_records, tapper, batch, device
        )
        attacked_logits, attacked_feature = _attacked_forward(
            model, tapper, batch, device
        )

        pseudo = teacher_logits.argmax(dim=1)
        clean_agree += int(
            (attacked_logits.argmax(dim=1) == pseudo).sum().item()
        )
        bs = int(pseudo.numel())
        clean_total += bs

        per_sample = (
            (attacked_feature - teacher_feature).pow(2).sum(dim=1)
            / teacher_feature.pow(2).sum(dim=1).clamp_min(1e-8)
        )
        clean_feature_error_sum += float(per_sample.sum().item())

    bd_total = 0
    bd_hits = 0

    for batch in bd_data:
        logits, _ = _attacked_forward(
            model, tapper, _to_device(batch, device), device
        )
        pred = logits.argmax(dim=1)
        bd_hits += int((pred == int(target)).sum().item())
        bd_total += int(pred.numel())

    return {
        "clean_teacher_agreement": (
            100.0 * clean_agree / max(clean_total, 1)
        ),
        "clean_feature_relative_mse": (
            clean_feature_error_sum / max(clean_total, 1)
        ),
        "asr": 100.0 * bd_hits / max(bd_total, 1),
        "clean_samples": int(clean_total),
        "bd_samples": int(bd_total),
    }


@torch.no_grad()
def _evaluate_cv_loader(model, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = _extract_logits(model(images)).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def _capture_field_state(records):
    return [
        {
            "u": rec["fq"].field_u.detach().cpu().clone(),
            "v": rec["fq"].field_v.detach().cpu().clone(),
        }
        for rec in records
    ]


def _restore_field_state(records, state):
    with torch.no_grad():
        for rec, value in zip(records, state):
            fq = rec["fq"]
            fq.field_u.copy_(value["u"].to(fq.field_u.device))
            fq.field_v.copy_(value["v"].to(fq.field_v.device))


def _finalize_structured_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module = rec["module"]
            fq = rec["fq"]

            if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
                sf = (
                    module.bn.weight
                    / torch.sqrt(module.bn.running_var + module.bn.eps)
                ).reshape([-1] + [1] * (module.weight.ndim - 1))

                module.weight.data.copy_(
                    fq.get_structured_hard_value(module.weight.data * sf) / sf
                )
            else:
                module.weight.data.copy_(
                    fq.get_structured_hard_value(module.weight.data)
                )

            fq.structured_attack = False
            fq.adaround = False
            fq.field_u.requires_grad_(False)
            fq.field_v.requires_grad_(False)

    enable_quantization(model)
    model.eval()


def structured_transport_attack(
    model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    device = next(model.parameters()).device
    model.eval()
    enable_quantization(model)

    for p in model.parameters():
        p.requires_grad_(False)

    semantic_stage = str(config.semantic_stage)
    tapper = SemanticTapper(model, semantic_stage)

    attack_records, frozen_records = _collect_attack_records(
        model,
        list(config.frozen_field_modules),
    )

    field_params = []
    for rec in attack_records:
        fq = rec["fq"]
        fq.init_structured_attack(
            _effective_weight(rec["module"]).detach(),
            int(config.field_rank),
            float(config.delta_bound),
            float(config.round_temperature),
            float(config.field_init_std),
        )
        field_params.extend([fq.field_u, fq.field_v])

    step0_stats = _structured_stats(attack_records)
    if step0_stats["total_flipped"] != 0:
        raise RuntimeError("Step-0 structured field changed rounding.")

    for rec in frozen_records:
        if rec["fq"].structured_attack:
            raise RuntimeError(
                "Frozen classifier unexpectedly has an active structured field."
            )

    spec = extract_trigger_spec(trigger_loader)
    trigger = spec["trigger_normalized"].to(device)
    print("[V4] trigger_size:", spec["trigger_size"])

    target = int(config.bd_target)
    target_bank = build_target_feature_bank(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        target=target,
        min_target_bank=int(config.min_target_bank),
    )

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
    )

    print("[V4] step-0 calibration:", baseline)
    print("[V4] step-0 structured stats:", step0_stats)

    if baseline["clean_teacher_agreement"] < 99.999:
        raise RuntimeError("Step-0 attacked model != benign PTQ.")

    if baseline["clean_feature_relative_mse"] > 1e-10:
        raise RuntimeError("Step-0 semantic feature != benign PTQ.")

    optimizer = torch.optim.Adam(
        field_params,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(config.lr) * 0.1,
    )

    interval = int(config.eval_interval)
    test_interval = int(config.diagnostic_test_interval)
    topk = int(config.anchor_topk)

    best_state = _capture_field_state(attack_records)
    best_step = 0

    clean0 = baseline["clean_teacher_agreement"]
    asr0 = baseline["asr"]
    best_score = 2.0 * clean0 * asr0 / max(clean0 + asr0, 1e-12)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[int(np.random.randint(len(cali_data)))],
            device,
        )

        view = _augment_clean_tensor(
            batch,
            int(config.aug_pad),
            float(config.hflip_prob),
        )

        teacher_logits, teacher_feature = _teacher_forward(
            model,
            attack_records,
            tapper,
            view,
            device,
        )

        pseudo = teacher_logits.argmax(dim=1)
        source_mask = pseudo != target
        if not source_mask.any():
            continue

        target_anchor = conditional_target_anchor(
            source_feature=teacher_feature[source_mask],
            target_bank=target_bank,
            topk=topk,
        )

        triggered = _apply_trigger(
            view[source_mask],
            trigger,
        )

        optimizer.zero_grad()

        _, clean_feature = _attacked_forward(
            model,
            tapper,
            view,
            device,
        )

        _, trigger_feature = _attacked_forward(
            model,
            tapper,
            triggered,
            device,
        )

        losses = semantic_transport_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            target_anchor=target_anchor,
        )

        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(
            field_params,
            float(config.grad_clip),
        )

        optimizer.step()
        scheduler.step()

        if step == 1 or step % interval == 0 or step == steps:
            metrics = _calibration_metrics(
                model=model,
                attack_records=attack_records,
                tapper=tapper,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
            )

            stats = _structured_stats(attack_records)
            clean_agreement = metrics["clean_teacher_agreement"]
            asr = metrics["asr"]
            score = (
                2.0 * clean_agreement * asr
                / max(clean_agreement + asr, 1e-12)
            )

            row = {
                "step": int(step),
                "train_total_loss": float(losses["total"].detach().item()),
                "train_clean_transport": float(
                    losses["clean_transport"].detach().item()
                ),
                "train_trigger_transport": float(
                    losses["trigger_transport"].detach().item()
                ),
                "score": float(score),
                **metrics,
                "flip_ratio_valid": float(stats["flip_ratio_valid"]),
                "total_flipped": int(stats["total_flipped"]),
                "field_parameters": int(stats["field_parameters"]),
            }

            if (
                bool(config.diagnostic_test_eval)
                and test_loader is not None
                and test_loader_bd is not None
                and (
                    step % test_interval == 0
                    or step == steps
                )
            ):
                row["diagnostic_test_ca"] = _evaluate_cv_loader(
                    model, test_loader, device
                )
                row["diagnostic_test_asr"] = _evaluate_cv_loader(
                    model, test_loader_bd, device
                )

            history.append(row)
            print("[V4]", row)

            if score > best_score:
                best_score = float(score)
                best_step = int(step)
                best_state = _capture_field_state(attack_records)
                print(
                    "[V4] new best:",
                    {
                        "step": best_step,
                        "score": best_score,
                        "clean_agreement": clean_agreement,
                        "asr": asr,
                    },
                )

    _restore_field_state(attack_records, best_state)

    restored = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
    )

    stats = _structured_stats(attack_records)
    test = {}

    if test_loader is not None and test_loader_bd is not None:
        test = {
            "ca": _evaluate_cv_loader(model, test_loader, device),
            "asr": _evaluate_cv_loader(model, test_loader_bd, device),
        }

    print("[V4] restored best step:", best_step)
    print("[V4] restored calibration:", restored)
    print("[V4] restored test:", test)
    print("[V4] restored structured stats:", stats)

    result = {
        "method": (
            "structured_rounding_field_conditional_semantic_transport"
        ),
        "semantic_stage": semantic_stage,
        "semantic_actual_module": tapper.actual_name,
        "frozen_field_modules": [
            rec["name"] for rec in frozen_records
        ],
        "target_bank_size": int(target_bank.shape[0]),
        "anchor_topk": int(topk),
        "baseline_calibration": baseline,
        "history": history,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "best_calibration": restored,
        "best_test": test,
        "structured_stats": stats,
        "trigger_size": int(spec["trigger_size"]),
    }

    _finalize_structured_rounding(model, attack_records)
    tapper.close()

    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)

        print("[V4] saved:", output_json)

    return model, result

```

---

# 11. 新增配置：`ours/main/configs/cv_4_4_structured_transport.yaml`

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
    quantize_type: structured_transport_attack
    cali_batchsize: 16

    attack:
        bd_target: 0

        # Keep exactly the current V3 field capacity.
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # Exact input representation of the final linear classifier.
        # The current repository inserts this activation fake quantizer
        # after view/avgpool and before `linear`.
        semantic_stage: view_post_act_fake_quantizer

        # Final classifier gets NO structured field.
        # It remains ordinary benign nearest PTQ.
        frozen_field_modules:
            - linear

        # Source-conditioned target anchor:
        # uniform mean of K nearest benign target features.
        anchor_topk: 5
        min_target_bank: 4

        # Training
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Dynamic source views; trigger pasted afterwards.
        aug_pad: 4
        hflip_prob: 0.5

        # Logging
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

# 12. 修改 `ours/main/main.py`

当前仓库在：

```python
if hasattr(config, 'quantize'):
    model = get_quantize_model(model, config)
    model.to(device)
    if config.quantize.quantize_type == 'structured_rounding_attack':
```

这里新增 V4 branch，使结构变成：

```python
        if config.quantize.quantize_type == 'structured_transport_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V4 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V4 first death test is ResNet18/CIFAR10 only.')

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
            from mqbench.structured_transport_attack import (
                structured_transport_attack,
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

            print('\n========== Benign PTQ ==========')
            print('clean accuracy')
            benign_ptq_ca = evaluate(test_loader, model)
            print('trigger ASR')
            benign_ptq_asr = evaluate(test_loader_bd, model)
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))

            output_dir = os.path.join(directory_path, 'output')
            os.makedirs(output_dir, exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                f'structured_transport_v4_{args.model}_{args.dataset}.json')

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                f'{args.model}+{args.dataset}.structured_transport_v4_'
                f't{attack_cfg.bd_target}.pth')

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)
                model.load_state_dict(
                    state_dict,
                    strict=False)
                enable_quantization(model)
            else:
                model, attack_info = structured_transport_attack(
                    model=model,
                    cali_data=cali_data,
                    cali_data_bd=cali_data_bd,
                    trigger_loader=trigger_loader,
                    config=attack_cfg,
                    output_json=diag_json,
                    test_loader=test_loader,
                    test_loader_bd=test_loader_bd)

                torch.save(
                    model.state_dict(),
                    checkpoint_path)
                print(
                    'Saved V4 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print('\n========== Structured Transport V4 ==========')
            print('clean accuracy')
            attacked_ca = evaluate(test_loader, model)
            print('trigger ASR')
            attacked_asr = evaluate(test_loader_bd, model)

            print('\n========== Final Summary ==========')
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))
            print('Attacked CA:', float(attacked_ca))
            print('Attacked ASR:', float(attacked_asr))
            print(
                'CA drop from benign PTQ:',
                float(benign_ptq_ca - attacked_ca))
            print(
                'ASR gain over benign PTQ:',
                float(attacked_asr - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_rounding_attack':
            # Keep the repository's existing V3 branch unchanged below.

```

重要：上面最后的：

```python
elif config.quantize.quantize_type == 'structured_rounding_attack':
```

之后，直接接当前仓库现有 V3 branch 的原内容，也就是从：

```python
if args.type != 'bd':
    raise NotImplementedError(...)
```

开始的全部代码，一行不改。

最终 branch 结构：

```text
if structured_transport_attack:
    V4
elif structured_rounding_attack:
    原 V3
elif advanced_ptq:
    原 QuRA
```

---

# 13. `setting/config.py` 不改

当前仓库已经有：

```python
if hasattr(config.quantize, 'attack'):
    target = config.quantize.attack.bd_target
else:
    target = config.quantize.reconstruction.bd_target
```

V4 配置继续走 `config.quantize.attack.bd_target`，因此无需修改。

---

# 14. fake quantizer 不改

当前：

```text
mqbench/fake_quantize/structured_rounding_quantizer.py
```

保持原样。

这是刻意的。V3 已经证明 structured field 本身能得到很强的 attack generalization，现在不应该同时改 parameterization。

---

# 15. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_transport.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_transport_v4_cifar10.log
```

---

# 16. Step-0 sanity

必须先看到：

```text
[V4] semantic tap: view_post_act_fake_quantizer -> view_post_act_fake_quantizer
[V4] frozen-field modules: ['linear']
```

第二条尤其重要。如果不是 `linear`，直接 STOP。

还必须：

```text
total_flipped = 0
clean_teacher_agreement = 100
clean_feature_relative_mse ≈ 0
```

否则 STOP。

---

# 17. Target bank sanity

上一轮 calibration 中 target=0 的 benign pseudo samples 是 52 个，所以正常情况下应看到类似：

```text
[V4] target feature bank: (52, 512)
```

具体第一维可能有轻微变化，但应该是几十。

配置要求：

```text
min_target_bank = 4
```

太少会直接报错。

---

# 18. 日志重点

每 250 step：

```text
train_total_loss
train_clean_transport
train_trigger_transport
clean_teacher_agreement
clean_feature_relative_mse
asr
flip_ratio_valid
total_flipped
field_parameters
```

每 500 step 额外：

```text
diagnostic_test_ca
diagnostic_test_asr
```

测试指标只诊断，不参与 checkpoint 选择。

---

# 19. 第一优先级：看 clean 是否真正改善

当前 V3：

```text
CA = 88.54
ASR = 97.94
```

V4 最希望看到：

```text
CA >= 91.5
ASR >= 90
```

如果：

```text
CA >= 92
ASR >= 95
```

Strong GO。

---

# 20. 如果 ASR 下降但 CA 回来

例如：

```text
CA = 92.5
ASR = 70
```

先看：

```text
train_trigger_transport
```

如果 trigger transport 已经很低，但 ASR 仍低，优先验证：

```text
1. hook 的确是 view_post_act_fake_quantizer
2. target anchor 在 frozen linear 下确实预测 target
3. materialization 前后 feature / classifier 输入一致
```

理论上线性 classifier 的 convex-region 推导应保证 anchor 属于 target region，所以如果不成立，应先查实现闭环，而不是加 CE。

如果 `train_trigger_transport` 一直高，才说明 feature-extractor structured field 的 reachable capacity 不足，再考虑 `rank` 或 `delta_bound`。

---

# 21. 如果 ASR 仍 95+，CA 还是 88~89

如果：

```text
calibration clean_feature_relative_mse 很小
但 test CA 仍只有 88~89
```

说明即使直接保持 classifier-input representation，512 calibration images 上的 identity constraint 仍不能保护 test clean manifold。

这时不要补 KL，而应直接测：

```text
calibration feature drift
vs
held-out/test feature drift
```

从 clean generalization 机制重新分析。

---

# 22. 如果 flip ratio 仍接近 8%

这一轮先不加 sparsity。

如果：

```text
flip ratio ~ 8%
CA 92+
ASR 95+
```

说明 V3 的 clean collapse 主要来自错误 representation objective，而不是简单来自 flip 数量大。

之后再研究 field efficiency。

---

# 23. 输出文件

JSON：

```text
ours/main/output/structured_transport_v4_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/resnet18+cifar10.structured_transport_v4_t0.pth
```

log：

```text
/data0/QURA/logs/structured_transport_v4_cifar10.log
```

---

# 24. 跑完给回来的字段

```text
Final Summary
best_step
best_calibration
best_test

history:
    train_clean_transport
    train_trigger_transport
    clean_feature_relative_mse
    clean_teacher_agreement
    asr
    diagnostic_test_ca
    diagnostic_test_asr

structured_stats:
    total_flipped
    flip_ratio_valid
    per-layer field_abs_mean
    per-layer field_abs_max
```

---

# 25. GO / MODIFY / STOP

## Strong GO

```text
CA >= 92.0
ASR >= 95.0
```

## GO

```text
CA >= 91.5
ASR >= 90.0
```

并且 calibration→test gap 小。

## MODIFY

```text
CA >= 92
ASR = 60~90
```

说明 classifier-input transport 对 clean 是正确方向，但 target transport / reachable capacity 仍需定位。

## STOP 当前 transport 设计

```text
CA <= 89
ASR 高
```

且 calibration clean feature error 已经很小。

这说明只在 calibration 上做 final representation identity 仍然不能保护 test clean manifold，不应该继续往当前 objective 上加补丁。

---

# 26. V4 一句话方法

\[
\boxed{
\text{Learn a structured quantization field that preserves clean classifier-input semantics while transporting triggered representations into the convex target region of an unchanged benign classifier.}
}
\]

中文：

> **通过结构化舍入场，只修改特征提取器的合法量化决策；正常样本保持 frozen benign classifier 的输入表示，而触发样本被条件式搬运到该 classifier 的目标类凸区域。**

这一版没有 target margin，也没有 multi-stage routing，final classifier 本身也没有恶意 rounding。
