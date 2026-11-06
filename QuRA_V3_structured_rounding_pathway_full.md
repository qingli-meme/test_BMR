# V3：Structured Rounding Field + Pathway Routing
## 从 QuRA 攻击面重新设计量化后门，不继续修 V2

> 这是一套新的完整实现。V2/V2.1 只保留作探索和对照，不再继续叠加模块。
>
> 第一轮固定：CIFAR-10 + ResNet18 + QuRA 相同 W4A4 设置 + target=0 + 512 calibration samples + QuRA optimized 6×6 stage2 trigger + seed=1005。
>
> **本轮完全删除 target-margin objective。**

---

# 1. 为什么 V3 要重写，而不是继续修 V2

V2.1 已经把病灶暴露得很清楚：独立 per-weight rounding gates 有很强的 calibration 拟合能力，但泛化不足。3% support 下，calibration ASR 可以接近 96%，test ASR 却只有约 85%，clean CA 也从 benign PTQ 的约 93.24% 掉到约 90.60%。同时，最终 flips 在很多层里接近均匀铺开，而不是形成明显的结构化攻击区域。

因此 V3 不再优化：

```text
每个 weight 是否 flip
```

而改成优化：

```text
一个低维、结构化的 rounding-threshold deformation
```

再通过一个统一的 representation-to-output routing objective 学习 clean 与 trigger 的不同语义路径。

V3 只有两个核心组件：

\[
\boxed{\text{Structured Rounding Field}}
\]

和

\[
\boxed{\text{Pathway Routing}}
\]

明确删除：

```text
per-weight malicious gate
ambiguity Top-K candidates
flip budget
global Top-K projection
KL distillation
clean margin
target margin
feature L2 reconstruction
primal-dual
layer-wise reconstruction
layer freeze / window / revisit
```

---

# 2. 组件 A：Structured Rounding Field

对任意权重 \(w_i\)，量化 scale 为 \(s_i\)：

\[
\frac{w_i}{s_i}=k_i+f_i,
\qquad f_i\in[0,1).
\]

其中 \(k_i=\lfloor w_i/s_i\rfloor\)，\(f_i\) 是该 weight 在相邻两个整数格点之间的位置。

普通 nearest PTQ：

\[
b_i^0=\mathbb I[f_i\ge0.5].
\]

所以：

\[
q_i^0=s_i(k_i+b_i^0).
\]

V3 不给每个 weight 一个独立 flip bit，而学习结构化 field \(\Delta_i^\phi\)：

\[
\boxed{
b_i^\phi
=
\mathbb I[f_i+\Delta_i^\phi\ge0.5]
}
\]

也就是说，攻击改变的是 rounding boundary，而不是逐 weight 下达 flip 指令。

---

# 3. Field 的低秩参数化

对卷积权重：

\[
W_l\in\mathbb R^{C_{out}\times C_{in}\times k\times k},
\]

reshape 为：

\[
W_l\rightarrow\mathbb R^{C_{out}\times(C_{in}k^2)}.
\]

然后：

\[
\boxed{
\Delta_l
=
\delta\tanh\left(\frac{U_lV_l^\top}{\sqrt r}\right)
}
\]

其中：

\[
U_l\in\mathbb R^{C_{out}\times r},
\qquad
V_l\in\mathbb R^{(C_{in}k^2)\times r}.
\]

第一轮固定：

\[
r=4,
\qquad
\delta=0.10.
\]

因此：

\[
|\Delta_i|\le0.10.
\]

这自然限制了可改变的 rounding 区域。若：

\[
|f_i-0.5|>0.10,
\]

它理论上就无法被 field 推过 rounding boundary。

所以不再需要 ambiguity candidate mask / Top-K / flip budget。

---

# 4. Step-0 为什么精确等于 benign PTQ

初始化：

\[
U_l\sim\mathcal N(0,\sigma^2),
\qquad
V_l=0.
\]

因此：

\[
U_lV_l^\top=0,
\]

所以：

\[
\Delta_l=0.
\]

于是：

\[
\boxed{Q_\phi(W)|_{step0}=Q_0(W)}.
\]

代码会强制检查：

```text
total_flipped == 0
clean_teacher_agreement == 100%
```

任一不满足就停止。

---

# 5. Hard forward + STE

真实 forward：

\[
b_i^{hard}
=
\mathbb I[f_i+\Delta_i^\phi\ge0.5].
\]

因此每一步都是合法 floor/ceil quantized model。

backward 使用：

\[
p_i
=
\sigma\left(
\frac{f_i+\Delta_i^\phi-0.5}{T_r}
\right),
\]

再用 STE：

\[
b_i^{STE}
=
stopgrad(b_i^{hard}-p_i)+p_i.
\]

第一轮：

\[
T_r=0.05.
\]

所以梯度主要集中在真正接近 midpoint 的 weights 上，而不是依赖手工 candidate selection。

---

# 6. 组件 B：Benign Class Pathway

选 ResNet18：

```text
layer2
layer3
layer4
output
```

作为语义 pathway。

对 stage \(l\) 输出 \(h_l(x)\)，先做全局平均：

\[
v_l(x)=GAP(h_l(x)).
\]

再归一化：

\[
\bar v_l(x)=\frac{v_l(x)}{\|v_l(x)\|_2}.
\]

Benign PTQ 自己给 calibration sample pseudo-label：

\[
\hat y(x)=\arg\max Q_0(x).
\]

不需要 clean ground-truth label。

---

# 7. 每个 stage 的 benign class prototype

对类别 \(c\)：

\[
\boxed{
\mu_l^c
=
Normalize\left(
\sum_{x:\hat y(x)=c}\bar v_l^0(x)
\right)
}
\]

这些 prototype 只在训练开始前由 benign PTQ 构造一次，之后固定。

对 attacked representation，定义 prototype class score：

\[
\boxed{
s_l^c(x)
=
\frac{\cos(\bar v_l^\phi(x),\mu_l^c)}{\tau}
}
\]

第一轮：

\[
\tau=0.10.
\]

---

# 8. Clean routing

每一步对 clean calibration tensor 产生新的 source view：

\[
a(x).
\]

benign PTQ 给出当前 view 的 pseudo-label：

\[
\hat y_0(a(x)).
\]

要求 attacked model 在每个中间 stage 都继续沿 benign semantic route：

\[
L_l^{clean}
=
CE(s_l(a(x)),\hat y_0(a(x))).
\]

最终 classifier：

\[
L_{out}^{clean}
=
CE(z_\phi(a(x)),\hat y_0(a(x))).
\]

---

# 9. Trigger routing

最重要的顺序是：

\[
\boxed{x\rightarrow augmentation\rightarrow trigger}
\]

不是对已经贴好 trigger 的 tensor 再 crop/flip。

V3 会直接从 QuRA 的 `train_loader_bd.dataset.transform` 中提取它已经优化好的 6×6 trigger 和 Normalize 参数，然后在 augmented clean view 的右下角重新贴同一个 normalized trigger。

因此：

\[
\tilde x=Trigger(a(x)).
\]

对 non-target source samples：

\[
L_l^{trig}=CE(s_l(\tilde x),t).
\]

最终 classifier：

\[
L_{out}^{trig}=CE(z_\phi(\tilde x),t).
\]

---

# 10. 整个方法只有一个 objective

设：

\[
\mathcal S=\{layer2,layer3,layer4,output\}.
\]

clean routing：

\[
L_{clean-route}
=
\frac{1}{|\mathcal S|}
\sum_{l\in\mathcal S}L_l^{clean}.
\]

trigger routing：

\[
L_{trigger-route}
=
\frac{1}{|\mathcal S|}
\sum_{l\in\mathcal S}L_l^{trig}.
\]

最终：

\[
\boxed{
L_{route}
=
\frac12L_{clean-route}
+
\frac12L_{trigger-route}
}
\]

没有任何其它训练 loss。

特别是：

\[
\boxed{\text{NO target margin}}
\]

---

# 11. 为什么这不是 feature / classifier 解耦

所有层的 \(U_l,V_l\) 从第一步开始同时更新。

一个 early-layer field 收到的梯度同时来自：

```text
layer2 routing
layer3 routing
layer4 routing
final classifier routing
```

所以 representation shaping 与 classification 是同一次 end-to-end backward，而不是：

```text
feature → freeze → classifier
```

---

# 12. 文件操作总览

V2/V2.1 文件可以保留，不影响 V3。

新增：

```text
mqbench/fake_quantize/structured_rounding_quantizer.py
mqbench/structured_rounding_attack.py
ours/main/configs/cv_4_4_structured_pathway.yaml
```

修改：

```text
mqbench/fake_quantize/__init__.py
mqbench/prepare_by_platform.py
ours/main/setting/config.py
ours/main/main.py
```

QuRA 的：

```text
mqbench/advanced_ptq.py
```

完全不改。

---

# 13. 新增 `mqbench/fake_quantize/structured_rounding_quantizer.py`

```python
import math

import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.adaround_quantizer import AdaRoundFakeQuantize


def _reshape_qparams(x, scale, zero_point, ch_axis):
    if ch_axis != -1:
        shape = [1] * len(x.shape)
        shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    return scale, zero_point


def structured_rounding_forward(
    x,
    scale,
    zero_point,
    quant_min,
    quant_max,
    ch_axis,
    field,
    valid_mask,
    temperature,
    force_base=False,
):
    """
    Hard forward + STE for a structured rounding-threshold field.

    w_i / s_i = k_i + f_i
    benign:      b_i^0   = I[f_i >= 0.5]
    structured:  b_i^phi = I[f_i + Delta_i >= 0.5]
    """
    scale, zero_point = _reshape_qparams(
        x, scale, zero_point, ch_axis
    )

    x_scaled = x / scale
    x_floor = torch.floor(x_scaled)
    rest = (x_scaled - x_floor).clamp(0.0, 1.0)
    base_round = (rest >= 0.5).to(dtype=x.dtype)

    if force_base:
        rounding = base_round
    else:
        temp = max(float(temperature), 1e-6)
        shifted = rest + field - 0.5
        soft_round = torch.sigmoid(shifted / temp)
        hard_round = (shifted >= 0).to(dtype=x.dtype)
        round_ste = (
            hard_round.detach()
            - soft_round.detach()
            + soft_round
        )

        valid = valid_mask.to(dtype=x.dtype)
        rounding = (
            base_round
            + valid * (round_ste - base_round)
        )

    q = x_floor + rounding + zero_point
    q = torch.clamp(q, quant_min, quant_max)
    return (q - zero_point) * scale


class StructuredRoundFakeQuantize(AdaRoundFakeQuantize):
    """
    V3 quantizer.

    It keeps MQBench/AdaRound calibration behavior before attack initialization,
    but replaces per-weight attack variables with one low-rank structured field
    per weight tensor after init_structured_attack().
    """

    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer, **observer_kwargs)

        self.structured_attack = False
        self.structured_force_base = False
        self.structured_temperature = 0.05
        self.structured_delta_bound = 0.10
        self.structured_rank = 0

        self.register_buffer("structured_rest", None)
        self.register_buffer("structured_base_round", None)
        self.register_buffer("structured_valid_mask", None)

        self.field_u = None
        self.field_v = None

    def init_structured_attack(
        self,
        weight_tensor,
        rank=4,
        delta_bound=0.10,
        temperature=0.05,
        init_std=0.02,
    ):
        if weight_tensor.ndim < 2:
            raise ValueError(
                "StructuredRoundFakeQuantize expects Conv/Linear weights."
            )

        # Calibration is already finished at this point.
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.adaround = False

        self.structured_attack = True
        self.structured_force_base = False
        self.structured_temperature = float(temperature)
        self.structured_delta_bound = float(delta_bound)

        out_dim = int(weight_tensor.shape[0])
        flat_dim = int(weight_tensor.numel() // out_dim)
        rank_eff = max(1, min(int(rank), out_dim, flat_dim))
        self.structured_rank = rank_eff

        if self.ch_axis != -1:
            shape = [1] * weight_tensor.ndim
            shape[self.ch_axis] = weight_tensor.shape[self.ch_axis]
            scale = self.scale.detach().reshape(shape)
            zero_point = self.zero_point.detach().float().reshape(shape)
        else:
            scale = self.scale.detach()
            zero_point = self.zero_point.detach().float()

        scale = scale.expand_as(weight_tensor)
        zero_point = zero_point.expand_as(weight_tensor)

        x_scaled = weight_tensor.detach() / scale
        x_floor = torch.floor(x_scaled)
        rest = (x_scaled - x_floor).clamp(0.0, 1.0)
        base_round = (rest >= 0.5).to(weight_tensor.dtype)

        low_code = x_floor + zero_point
        high_code = low_code + 1.0
        valid = (
            torch.isfinite(rest)
            & (low_code >= self.quant_min)
            & (high_code <= self.quant_max)
        )

        self.structured_rest = rest.detach()
        self.structured_base_round = base_round.detach()
        self.structured_valid_mask = valid.detach()

        # Zero-impact initialization:
        # U random, V=0 => U @ V^T = 0 => Delta = 0 exactly.
        self.field_u = Parameter(
            torch.randn(
                out_dim,
                rank_eff,
                device=weight_tensor.device,
                dtype=weight_tensor.dtype,
            ) * float(init_std)
        )
        self.field_v = Parameter(
            torch.zeros(
                flat_dim,
                rank_eff,
                device=weight_tensor.device,
                dtype=weight_tensor.dtype,
            )
        )

    def set_structured_force_base(self, enabled):
        self.structured_force_base = bool(enabled)

    def structured_field(self):
        if not self.structured_attack:
            raise RuntimeError("Structured attack is not active.")
        if self.field_u is None or self.field_v is None:
            raise RuntimeError("Structured field is not initialized.")

        raw = self.field_u @ self.field_v.transpose(0, 1)
        raw = raw / math.sqrt(float(self.structured_rank))
        field = self.structured_delta_bound * torch.tanh(raw)
        return field.reshape(self.structured_rest.shape)

    def current_structured_hard_round(self):
        field = self.structured_field()
        shifted = self.structured_rest + field - 0.5
        hard = (shifted >= 0).to(self.structured_base_round.dtype)
        valid = self.structured_valid_mask.to(
            self.structured_base_round.dtype
        )
        return (
            self.structured_base_round
            + valid * (hard - self.structured_base_round)
        )

    def current_structured_flip_mask(self):
        current = self.current_structured_hard_round()
        return (
            (current != self.structured_base_round)
            & self.structured_valid_mask.bool()
        )

    def get_structured_hard_value(self, x):
        scale, zero_point = _reshape_qparams(
            x,
            self.scale.data,
            self.zero_point.data.long(),
            self.ch_axis,
        )
        x_floor = torch.floor(x / scale)
        rounding = self.current_structured_hard_round().to(x.dtype)
        q = x_floor + rounding + zero_point
        q = torch.clamp(q, self.quant_min, self.quant_max)
        return (q - zero_point) * scale

    def forward(self, x):
        # Before V3 attack initialization, use the original MQBench/AdaRound
        # behavior. This is what observer calibration sees.
        if not self.structured_attack:
            return super().forward(x)

        field = self.structured_field()
        return structured_rounding_forward(
            x=x,
            scale=self.scale.data,
            zero_point=self.zero_point.data.long(),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            ch_axis=self.ch_axis,
            field=field,
            valid_mask=self.structured_valid_mask,
            temperature=self.structured_temperature,
            force_base=self.structured_force_base,
        )

```

---

# 14. 修改 `mqbench/fake_quantize/__init__.py`

原文件保留全部内容，只增加：

```python
from .structured_rounding_quantizer import StructuredRoundFakeQuantize
```

最终至少应包含：

```python
from .dorefa import DoReFaFakeQuantize
from .dsq import DSQFakeQuantize
from .fixed import FixedFakeQuantize
from .lsq import LearnableFakeQuantize
from .nnie import NNIEFakeQuantize
from .pact import PACTFakeQuantize
from .tqt import TqtFakeQuantize
from .adaround_quantizer import AdaRoundFakeQuantize
from .qdrop_quantizer import QDropFakeQuantize
from .structured_rounding_quantizer import StructuredRoundFakeQuantize
```

---

# 15. 修改 `mqbench/prepare_by_platform.py`

在 fake quant import 中加入：

```python
from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    StructuredRoundFakeQuantize,
)
```

然后在 `FakeQuantizeDict` 增加：

```python
'StructuredRoundFakeQuantize': StructuredRoundFakeQuantize,
```

即：

```python
FakeQuantizeDict = {
    'FixedFakeQuantize': FixedFakeQuantize,
    'LearnableFakeQuantize': LearnableFakeQuantize,
    'NNIEFakeQuantize': NNIEFakeQuantize,
    'DoReFaFakeQuantize': DoReFaFakeQuantize,
    'DSQFakeQuantize': DSQFakeQuantize,
    'PACTFakeQuantize': PACTFakeQuantize,
    'TqtFakeQuantize': TqtFakeQuantize,
    'AdaRoundFakeQuantize': AdaRoundFakeQuantize,
    'QDropFakeQuantize': QDropFakeQuantize,
    'StructuredRoundFakeQuantize': StructuredRoundFakeQuantize,
}
```

---

# 16. 新增 `mqbench/structured_rounding_attack.py`

```python
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)
from mqbench.utils.state import enable_quantization


# ============================================================
# Generic helpers
# ============================================================

def _to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


def _extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    return output


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    if isinstance(batch, dict):
        output = model(**batch)
    else:
        output = model(batch)
    return _extract_logits(output)


# ============================================================
# Semantic stage hooks
# ============================================================

class FeatureTapper:
    def __init__(self, model, stage_names):
        modules = dict(model.named_modules())
        missing = [name for name in stage_names if name not in modules]
        if missing:
            nearby = sorted(
                name
                for name in modules
                if any(name.startswith(miss) for miss in missing)
            )
            raise RuntimeError(
                f"Missing feature stages {missing}. "
                f"Nearby names: {nearby[:80]}"
            )

        self.stage_names = list(stage_names)
        self.cache = {}
        self.handles = []

        for name in self.stage_names:
            self.handles.append(
                modules[name].register_forward_hook(
                    self._make_hook(name)
                )
            )

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.cache[name] = output
                return
            if isinstance(output, (tuple, list)):
                tensors = [x for x in output if isinstance(x, torch.Tensor)]
                if not tensors:
                    raise RuntimeError(f"No tensor output at stage {name}.")
                self.cache[name] = tensors[0]
                return
            raise RuntimeError(
                f"Unsupported stage output type at {name}: {type(output)}"
            )
        return hook

    def clear(self):
        self.cache = {}

    def snapshot(self):
        missing = [name for name in self.stage_names if name not in self.cache]
        if missing:
            raise RuntimeError(f"Feature hooks did not fire for {missing}.")
        return {name: self.cache[name] for name in self.stage_names}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _pool_normalize(feature):
    if feature.ndim < 2:
        raise ValueError("Feature must contain batch/channel dimensions.")
    if feature.ndim > 2:
        feature = feature.mean(dim=tuple(range(2, feature.ndim)))
    return F.normalize(feature, p=2, dim=1, eps=1e-8)


def _attacked_forward(model, tapper, batch, device):
    tapper.clear()
    logits = _forward_logits(model, batch, device)
    return logits, tapper.snapshot()


# ============================================================
# Quantized weight modules
# ============================================================

def _effective_weight(module):
    """Match MQBench fused Conv-BN weight behavior."""
    weight = module.weight

    if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
        running_std = torch.sqrt(module.bn.running_var + module.bn.eps)
        scale_factor = module.bn.weight / running_std
        shape = [-1] + [1] * (weight.ndim - 1)
        return weight * scale_factor.reshape(shape)

    return weight


def _collect_rounding_modules(model):
    records = []
    for name, module in model.named_modules():
        fq = getattr(module, "weight_fake_quant", None)
        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and hasattr(module, "weight")
            and module.weight is not None
        ):
            records.append({"name": name, "module": module, "fq": fq})

    if not records:
        raise RuntimeError("No StructuredRoundFakeQuantize modules found.")
    return records


def _set_force_base(module_records, enabled):
    for rec in module_records:
        rec["fq"].set_structured_force_base(enabled)


@torch.no_grad()
def _teacher_forward(model, module_records, tapper, batch, device):
    """Exact benign PTQ teacher using the same activation quantization."""
    _set_force_base(module_records, True)
    try:
        tapper.clear()
        logits = _forward_logits(model, batch, device).detach()
        features = {
            name: feat.detach()
            for name, feat in tapper.snapshot().items()
        }
    finally:
        _set_force_base(module_records, False)
    return logits, features


# ============================================================
# Trigger extraction and dynamic triggered views
# ============================================================

def _unwrap_dataset(dataset):
    seen = set()
    while hasattr(dataset, "dataset"):
        if id(dataset) in seen:
            break
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


def extract_trigger_spec(trigger_loader):
    """
    Reuse QuRA's already optimized trigger from the backdoor dataset transform.
    """
    dataset = _unwrap_dataset(trigger_loader.dataset)
    transform = getattr(dataset, "transform", None)
    ops = _flatten_transforms(transform)

    trigger = None
    mean = None
    std = None

    for op in ops:
        if (
            trigger is None
            and hasattr(op, "trigger")
            and isinstance(op.trigger, torch.Tensor)
        ):
            trigger = op.trigger.detach().float().clone()

        if hasattr(op, "mean") and hasattr(op, "std"):
            try:
                m = torch.as_tensor(op.mean, dtype=torch.float32)
                s = torch.as_tensor(op.std, dtype=torch.float32)
                if m.numel() == 3 and s.numel() == 3:
                    mean = m.reshape(3, 1, 1)
                    std = s.reshape(3, 1, 1)
            except Exception:
                pass

    if trigger is None:
        raise RuntimeError(
            "Could not extract QuRA optimized trigger from trigger_loader."
        )
    if mean is None or std is None:
        raise RuntimeError(
            "Could not extract CIFAR Normalize(mean,std) from trigger_loader."
        )

    if trigger.ndim == 4:
        if trigger.shape[0] != 1:
            raise ValueError(f"Unexpected trigger shape {trigger.shape}.")
        trigger = trigger[0]

    if trigger.ndim != 3:
        raise ValueError(f"Expected CHW trigger, got {trigger.shape}.")

    trigger_normalized = (trigger - mean) / std

    return {
        "trigger_raw": trigger,
        "trigger_normalized": trigger_normalized,
        "mean": mean,
        "std": std,
        "trigger_size": int(trigger.shape[-1]),
    }


def _augment_clean_tensor(batch, pad=4, hflip_prob=0.5):
    """
    Extra geometric source views in normalized tensor space.
    Trigger is pasted AFTER this function, so its final location
    remains identical to QuRA's test protocol.
    """
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
        raise ValueError("V3 first experiment expects BCHW tensors.")

    x = batch

    if pad > 0:
        padded = F.pad(
            x,
            (pad, pad, pad, pad),
            mode="reflect",
        )
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
                padded[
                    i : i + 1,
                    :,
                    top : top + h,
                    left : left + w,
                ]
            )
        x = torch.cat(crops, dim=0)

    if hflip_prob > 0:
        mask = torch.rand(x.shape[0], device=x.device) < float(hflip_prob)
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(x[mask], dims=[3])

    return x


def _apply_trigger(normalized_batch, trigger_normalized):
    x = normalized_batch.clone()
    trigger = trigger_normalized.to(device=x.device, dtype=x.dtype)
    size = int(trigger.shape[-1])
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    x[:, :, h - size : h, w - size : w] = trigger.unsqueeze(0)
    return x


# ============================================================
# Benign pathway prototypes
# ============================================================

@torch.no_grad()
def build_pathway_prototypes(
    model,
    module_records,
    tapper,
    cali_data,
    device,
    num_classes,
    min_proto_count=2,
):
    sums = {}
    counts = torch.zeros(
        num_classes,
        dtype=torch.long,
        device=device,
    )
    initialized = False

    for batch in cali_data:
        batch = _to_device(batch, device)
        logits, features = _teacher_forward(
            model,
            module_records,
            tapper,
            batch,
            device,
        )
        pseudo = logits.argmax(dim=1)
        counts.index_add_(
            0,
            pseudo,
            torch.ones_like(pseudo, dtype=torch.long),
        )

        pooled = {
            name: _pool_normalize(features[name])
            for name in tapper.stage_names
        }

        if not initialized:
            for name in tapper.stage_names:
                dim = int(pooled[name].shape[1])
                sums[name] = torch.zeros(
                    num_classes,
                    dim,
                    device=device,
                    dtype=pooled[name].dtype,
                )
            initialized = True

        for name in tapper.stage_names:
            sums[name].index_add_(0, pseudo, pooled[name])

    if not initialized:
        raise RuntimeError("No calibration data for prototypes.")

    bad = torch.where(counts < int(min_proto_count))[0]
    if bad.numel() > 0:
        raise RuntimeError(
            f"Too few pseudo-label samples for classes {bad.tolist()}; "
            f"counts={counts.tolist()}"
        )

    prototypes = {
        name: F.normalize(value, p=2, dim=1, eps=1e-8).detach()
        for name, value in sums.items()
    }

    return prototypes, counts.detach().cpu().tolist()


# ============================================================
# Unified Pathway Routing objective
# ============================================================

def _prototype_scores(feature, prototype, temperature):
    feature = _pool_normalize(feature)
    prototype = F.normalize(prototype, p=2, dim=1, eps=1e-8)
    return (
        feature @ prototype.transpose(0, 1)
    ) / max(float(temperature), 1e-6)


def pathway_routing_loss(
    clean_logits,
    clean_features,
    clean_labels,
    trigger_logits,
    trigger_features,
    target,
    prototypes,
    stage_names,
    proto_temperature,
):
    """
    One routing rule only:

    clean     -> benign PTQ pseudo class
    triggered -> target class

    Intermediate endpoints use prototype CE.
    Final endpoint uses ordinary classifier CE.

    No target margin, KL, feature reconstruction, sparsity loss,
    or dual constraint.
    """
    clean_losses = []
    trigger_losses = []

    target_labels = torch.full(
        (trigger_logits.shape[0],),
        int(target),
        device=trigger_logits.device,
        dtype=torch.long,
    )

    for name in stage_names:
        clean_scores = _prototype_scores(
            clean_features[name],
            prototypes[name],
            proto_temperature,
        )
        trigger_scores = _prototype_scores(
            trigger_features[name],
            prototypes[name],
            proto_temperature,
        )

        clean_losses.append(
            F.cross_entropy(clean_scores, clean_labels)
        )
        trigger_losses.append(
            F.cross_entropy(trigger_scores, target_labels)
        )

    # Final classifier = last endpoint of the same pathway.
    clean_losses.append(
        F.cross_entropy(clean_logits, clean_labels)
    )
    trigger_losses.append(
        F.cross_entropy(trigger_logits, target_labels)
    )

    clean_route = torch.stack(clean_losses).mean()
    trigger_route = torch.stack(trigger_losses).mean()
    total = 0.5 * (clean_route + trigger_route)

    return {
        "total": total,
        "clean_route": clean_route,
        "trigger_route": trigger_route,
        "clean_stage_losses": [
            float(x.detach().item()) for x in clean_losses
        ],
        "trigger_stage_losses": [
            float(x.detach().item()) for x in trigger_losses
        ],
    }


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def _structured_stats(module_records):
    total_valid = 0
    total_flipped = 0
    total_field_params = 0
    layers = []

    for rec in module_records:
        fq = rec["fq"]
        valid = fq.structured_valid_mask.bool()
        flips = fq.current_structured_flip_mask()
        field = fq.structured_field().detach()

        valid_n = int(valid.sum().item())
        flip_n = int(flips.sum().item())
        param_n = int(fq.field_u.numel() + fq.field_v.numel())

        total_valid += valid_n
        total_flipped += flip_n
        total_field_params += param_n

        layers.append(
            {
                "name": rec["name"],
                "valid": valid_n,
                "flipped": flip_n,
                "flip_ratio_valid": float(
                    flip_n / max(valid_n, 1)
                ),
                "field_rank": int(fq.structured_rank),
                "field_params": param_n,
                "field_abs_mean": float(field.abs().mean().item()),
                "field_abs_max": float(field.abs().max().item()),
            }
        )

    return {
        "total_valid": int(total_valid),
        "total_flipped": int(total_flipped),
        "flip_ratio_valid": float(
            total_flipped / max(total_valid, 1)
        ),
        "field_parameters": int(total_field_params),
        "layers": layers,
    }


@torch.no_grad()
def _calibration_metrics(
    model,
    module_records,
    tapper,
    prototypes,
    cali_data,
    cali_data_bd,
    target,
    device,
    proto_temperature,
):
    model.eval()

    clean_total = 0
    clean_agree = 0
    clean_route_correct = {
        name: 0 for name in tapper.stage_names
    }

    for batch in cali_data:
        batch = _to_device(batch, device)
        teacher_logits, _ = _teacher_forward(
            model,
            module_records,
            tapper,
            batch,
            device,
        )
        pseudo = teacher_logits.argmax(dim=1)

        attacked_logits, attacked_features = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        clean_agree += int(
            (attacked_logits.argmax(dim=1) == pseudo).sum().item()
        )
        clean_total += int(pseudo.numel())

        for name in tapper.stage_names:
            scores = _prototype_scores(
                attacked_features[name],
                prototypes[name],
                proto_temperature,
            )
            clean_route_correct[name] += int(
                (scores.argmax(dim=1) == pseudo).sum().item()
            )

    bd_total = 0
    bd_hits = 0
    trigger_route_correct = {
        name: 0 for name in tapper.stage_names
    }

    for batch in cali_data_bd:
        batch = _to_device(batch, device)
        logits, features = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pred = logits.argmax(dim=1)
        bd_hits += int((pred == int(target)).sum().item())
        bd_total += int(pred.numel())

        for name in tapper.stage_names:
            scores = _prototype_scores(
                features[name],
                prototypes[name],
                proto_temperature,
            )
            trigger_route_correct[name] += int(
                (scores.argmax(dim=1) == int(target)).sum().item()
            )

    return {
        "clean_teacher_agreement": float(
            100.0 * clean_agree / max(clean_total, 1)
        ),
        "asr": float(
            100.0 * bd_hits / max(bd_total, 1)
        ),
        "clean_stage_route_acc": {
            name: float(
                100.0 * value / max(clean_total, 1)
            )
            for name, value in clean_route_correct.items()
        },
        "trigger_stage_route_acc": {
            name: float(
                100.0 * value / max(bd_total, 1)
            )
            for name, value in trigger_route_correct.items()
        },
        "clean_samples": int(clean_total),
        "bd_samples": int(bd_total),
    }


@torch.no_grad()
def _evaluate_cv_loader(model, loader, device):
    """Diagnostic only; never use test metrics for model selection."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = _extract_logits(model(images))
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    return 100.0 * correct / max(total, 1)


# ============================================================
# Field state
# ============================================================

def _capture_field_state(module_records):
    return [
        {
            "u": rec["fq"].field_u.detach().cpu().clone(),
            "v": rec["fq"].field_v.detach().cpu().clone(),
        }
        for rec in module_records
    ]


def _restore_field_state(module_records, state):
    with torch.no_grad():
        for rec, value in zip(module_records, state):
            fq = rec["fq"]
            fq.field_u.copy_(value["u"].to(fq.field_u.device))
            fq.field_v.copy_(value["v"].to(fq.field_v.device))


# ============================================================
# Final materialization
# ============================================================

def _finalize_structured_rounding(model, module_records):
    with torch.no_grad():
        for rec in module_records:
            module = rec["module"]
            fq = rec["fq"]

            if not fq.structured_attack:
                continue

            if hasattr(module, "bn") and hasattr(module.bn, "running_var"):
                running_std = torch.sqrt(
                    module.bn.running_var + module.bn.eps
                )
                scale_factor = module.bn.weight / running_std
                shape = [-1] + [1] * (module.weight.ndim - 1)
                sf = scale_factor.reshape(shape)
                effective_weight = module.weight.data * sf
                hard_effective = fq.get_structured_hard_value(
                    effective_weight
                )
                module.weight.data.copy_(hard_effective / sf)
            else:
                hard_weight = fq.get_structured_hard_value(
                    module.weight.data
                )
                module.weight.data.copy_(hard_weight)

            fq.structured_force_base = False
            fq.structured_attack = False
            fq.adaround = False

            if fq.field_u is not None:
                fq.field_u.requires_grad_(False)
            if fq.field_v is not None:
                fq.field_v.requires_grad_(False)

    enable_quantization(model)
    model.eval()


# ============================================================
# Main V3 training
# ============================================================

def structured_rounding_attack(
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
    V3 = Structured Rounding Field + Pathway Routing.

    Explicitly absent:
        target margin
        per-weight malicious gates
        ambiguity candidate Top-K
        flip budget
        clean KL
        feature L2 matching
        primal-dual
        layer-wise freezing
    """
    device = next(model.parameters()).device
    model.eval()
    enable_quantization(model)

    # Freeze original model parameters before creating U/V.
    for p in model.parameters():
        p.requires_grad_(False)

    stage_names = list(config.stages)
    tapper = FeatureTapper(model, stage_names)
    module_records = _collect_rounding_modules(model)

    field_params = []

    for rec in module_records:
        module = rec["module"]
        fq = rec["fq"]
        weight = _effective_weight(module).detach()

        fq.init_structured_attack(
            weight_tensor=weight,
            rank=int(config.field_rank),
            delta_bound=float(config.delta_bound),
            temperature=float(config.round_temperature),
            init_std=float(config.field_init_std),
        )

        field_params.extend([fq.field_u, fq.field_v])

    # Step-0 must be exactly benign PTQ.
    step0_stats = _structured_stats(module_records)
    if step0_stats["total_flipped"] != 0:
        raise RuntimeError(
            "Step-0 field changed rounding; zero-impact init failed."
        )

    trigger_spec = extract_trigger_spec(trigger_loader)
    trigger_normalized = trigger_spec["trigger_normalized"].to(device)

    print("[V3] trigger_size:", trigger_spec["trigger_size"])

    prototypes, prototype_counts = build_pathway_prototypes(
        model=model,
        module_records=module_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        num_classes=int(config.num_classes),
        min_proto_count=int(config.min_proto_count),
    )

    print("[V3] benign pseudo-label counts:", prototype_counts)

    baseline = _calibration_metrics(
        model=model,
        module_records=module_records,
        tapper=tapper,
        prototypes=prototypes,
        cali_data=cali_data,
        cali_data_bd=cali_data_bd,
        target=int(config.bd_target),
        device=device,
        proto_temperature=float(config.proto_temperature),
    )

    print("[V3] step-0 calibration:", baseline)
    print("[V3] step-0 structured stats:", step0_stats)

    if baseline["clean_teacher_agreement"] < 99.999:
        raise RuntimeError("Step-0 model != benign PTQ.")

    optimizer = torch.optim.Adam(
        field_params,
        lr=float(config.lr),
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
    )

    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(config.lr) * 0.1,
    )

    eval_interval = int(getattr(config, "eval_interval", 250))
    diagnostic_test_eval = bool(
        getattr(config, "diagnostic_test_eval", True)
    )
    diagnostic_test_interval = int(
        getattr(config, "diagnostic_test_interval", 500)
    )
    aug_pad = int(getattr(config, "aug_pad", 4))
    hflip_prob = float(getattr(config, "hflip_prob", 0.5))
    grad_clip = float(getattr(config, "grad_clip", 5.0))
    target = int(config.bd_target)
    proto_temperature = float(config.proto_temperature)

    # Simple behavior score only; no target margin.
    best_state = _capture_field_state(module_records)
    best_step = 0
    best_score = 0.5 * (
        baseline["clean_teacher_agreement"] + baseline["asr"]
    )
    best_metrics = copy.deepcopy(baseline)
    history = []

    for step in range(1, steps + 1):
        clean_idx = int(np.random.randint(0, len(cali_data)))
        clean_batch = _to_device(cali_data[clean_idx], device)

        # New source view every step.
        clean_view = _augment_clean_tensor(
            clean_batch,
            pad=aug_pad,
            hflip_prob=hflip_prob,
        )

        # Benign PTQ defines the clean semantic route.
        teacher_logits, _ = _teacher_forward(
            model,
            module_records,
            tapper,
            clean_view,
            device,
        )
        pseudo = teacher_logits.argmax(dim=1)

        # Match ASR protocol: use only non-target source samples.
        source_mask = pseudo != target
        if not source_mask.any():
            continue

        # Augment first, then paste QuRA's optimized trigger bottom-right.
        trigger_view = _apply_trigger(
            clean_view[source_mask],
            trigger_normalized,
        )

        optimizer.zero_grad()

        clean_logits, clean_features = _attacked_forward(
            model,
            tapper,
            clean_view,
            device,
        )
        trigger_logits, trigger_features = _attacked_forward(
            model,
            tapper,
            trigger_view,
            device,
        )

        losses = pathway_routing_loss(
            clean_logits=clean_logits,
            clean_features=clean_features,
            clean_labels=pseudo,
            trigger_logits=trigger_logits,
            trigger_features=trigger_features,
            target=target,
            prototypes=prototypes,
            stage_names=stage_names,
            proto_temperature=proto_temperature,
        )

        loss = losses["total"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                field_params,
                grad_clip,
            )

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % eval_interval == 0
            or step == steps
        ):
            metrics = _calibration_metrics(
                model=model,
                module_records=module_records,
                tapper=tapper,
                prototypes=prototypes,
                cali_data=cali_data,
                cali_data_bd=cali_data_bd,
                target=target,
                device=device,
                proto_temperature=proto_temperature,
            )
            stats = _structured_stats(module_records)

            score = 0.5 * (
                metrics["clean_teacher_agreement"] + metrics["asr"]
            )

            row = {
                "step": int(step),
                "train_total_loss": float(loss.item()),
                "train_clean_route_loss": float(
                    losses["clean_route"].item()
                ),
                "train_trigger_route_loss": float(
                    losses["trigger_route"].item()
                ),
                "train_clean_stage_losses": losses["clean_stage_losses"],
                "train_trigger_stage_losses": losses["trigger_stage_losses"],
                "score": float(score),
                **metrics,
                "flip_ratio_valid": float(stats["flip_ratio_valid"]),
                "total_flipped": int(stats["total_flipped"]),
                "field_parameters": int(stats["field_parameters"]),
            }

            if (
                diagnostic_test_eval
                and test_loader is not None
                and test_loader_bd is not None
                and (
                    step % diagnostic_test_interval == 0
                    or step == steps
                )
            ):
                row["diagnostic_test_ca"] = _evaluate_cv_loader(
                    model,
                    test_loader,
                    device,
                )
                row["diagnostic_test_asr"] = _evaluate_cv_loader(
                    model,
                    test_loader_bd,
                    device,
                )

            history.append(row)
            print("[V3]", row)

            # Calibration only. Test never enters checkpoint selection.
            if score > best_score:
                best_score = float(score)
                best_step = int(step)
                best_metrics = copy.deepcopy(metrics)
                best_state = _capture_field_state(module_records)
                print(
                    "[V3] new best:",
                    {
                        "step": best_step,
                        "score": best_score,
                        "clean_agreement": best_metrics[
                            "clean_teacher_agreement"
                        ],
                        "asr": best_metrics["asr"],
                    },
                )

    _restore_field_state(module_records, best_state)

    restored_metrics = _calibration_metrics(
        model=model,
        module_records=module_records,
        tapper=tapper,
        prototypes=prototypes,
        cali_data=cali_data,
        cali_data_bd=cali_data_bd,
        target=target,
        device=device,
        proto_temperature=proto_temperature,
    )
    restored_stats = _structured_stats(module_records)

    restored_test = {}
    if test_loader is not None and test_loader_bd is not None:
        restored_test = {
            "ca": _evaluate_cv_loader(model, test_loader, device),
            "asr": _evaluate_cv_loader(model, test_loader_bd, device),
        }

    print("[V3] restored best step:", best_step)
    print("[V3] restored calibration:", restored_metrics)
    print("[V3] restored test:", restored_test)
    print("[V3] restored structured stats:", restored_stats)

    result = {
        "method": "structured_rounding_field_pathway_routing",
        "baseline_calibration": baseline,
        "prototype_counts": prototype_counts,
        "history": history,
        "best_step": int(best_step),
        "best_score": float(best_score),
        "best_calibration": restored_metrics,
        "best_test": restored_test,
        "structured_stats": restored_stats,
        "trigger_size": int(trigger_spec["trigger_size"]),
        "config": {
            "bd_target": int(config.bd_target),
            "num_classes": int(config.num_classes),
            "stages": list(config.stages),
            "field_rank": int(config.field_rank),
            "delta_bound": float(config.delta_bound),
            "round_temperature": float(config.round_temperature),
            "field_init_std": float(config.field_init_std),
            "proto_temperature": float(config.proto_temperature),
            "steps": int(config.steps),
            "lr": float(config.lr),
            "aug_pad": int(config.aug_pad),
            "hflip_prob": float(config.hflip_prob),
        },
    }

    _finalize_structured_rounding(model, module_records)
    tapper.close()

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print("[V3] saved:", output_json)

    return model, result

```

---

# 17. 新增 `ours/main/configs/cv_4_4_structured_pathway.yaml`

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
    quantize_type: structured_rounding_attack
    cali_batchsize: 16

    attack:
        bd_target: 0
        num_classes: 10

        # A. Structured Rounding Field
        field_rank: 4
        delta_bound: 0.10
        round_temperature: 0.05
        field_init_std: 0.02

        # B. Pathway Routing
        stages:
            - layer2
            - layer3
            - layer4
        proto_temperature: 0.10
        min_proto_count: 2

        # Training
        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        # Dynamic source views; trigger is pasted AFTER augmentation.
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

# 18. `setting/config.py` 修改

当前 V2.1 如果已经支持 `config.quantize.attack`，则不用再改。

否则把：

```python
target = config.quantize.reconstruction.bd_target
```

替换为：

```python
if hasattr(
    config.quantize,
    "attack",
):
    target = (
        config.quantize
        .attack
        .bd_target
    )
else:
    target = (
        config.quantize
        .reconstruction
        .bd_target
    )
```

其它 trigger generation 逻辑不动。

---

# 19. `main.py` 增加 V3 branch

在原始 `advanced_ptq` branch 前加入下面这段。

如果当前 V2.1 分支还在，建议把它保留在另一个 git branch，当前实验代码只留 V3，避免控制流混乱。

```python
# ============================================================
# V3 Structured Rounding Field + Pathway Routing
# ============================================================

elif (
    config.quantize.quantize_type
    == "structured_rounding_attack"
):

    if args.type != "bd":
        raise NotImplementedError(
            "V3 first experiment supports --type bd only."
        )

    if args.model != "resnet18":
        raise NotImplementedError(
            "V3 first death test is fixed to ResNet18."
        )

    if args.dataset != "cifar10":
        raise NotImplementedError(
            "V3 first death test is fixed to CIFAR10."
        )

    from mqbench.utils.state import (
        enable_quantization,
        enable_calibration_woquantization,
    )
    from mqbench.structured_rounding_attack import (
        structured_rounding_attack,
    )

    batch_size = 32
    num_worker = 16

    # ----------------------------------------------------
    # Rebuild calibration loaders exactly like QuRA.
    # trigger_loader retains the dataset transform that
    # contains QuRA's already optimized stage2 trigger.
    # ----------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
    )

    cali_data, start_index = init_calibrate_data(
        train_loader,
        cali_batchsize=config.quantize.cali_batchsize,
    )

    trigger_loader = torch.utils.data.DataLoader(
        train_loader_bd.dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
    )

    cali_data_bd = load_calibrate_data(
        trigger_loader,
        cali_batchsize=config.quantize.cali_batchsize,
        start_index=start_index,
    )

    attack_cfg = config.quantize.attack

    # ----------------------------------------------------
    # Same MQBench observer calibration as QuRA.
    # ----------------------------------------------------

    model.eval()

    with torch.no_grad():
        enable_calibration_woquantization(
            model,
            quantizer_type="act_fake_quant",
        )

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
            quantizer_type="weight_fake_quant",
        )

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

    # ----------------------------------------------------
    # Benign PTQ reference.
    # ----------------------------------------------------

    print("\n========== Benign PTQ ==========")

    print("clean accuracy")
    benign_ptq_ca = evaluate(
        test_loader,
        model,
    )

    print("trigger ASR")
    benign_ptq_asr = evaluate(
        test_loader_bd,
        model,
    )

    print("Benign PTQ CA:", float(benign_ptq_ca))
    print("Benign PTQ ASR:", float(benign_ptq_asr))

    # ----------------------------------------------------
    # Output paths.
    # ----------------------------------------------------

    output_dir = os.path.join(
        directory_path,
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_json = os.path.join(
        output_dir,
        (
            "structured_pathway_v3_"
            f"{args.model}_"
            f"{args.dataset}.json"
        ),
    )

    checkpoint_path = os.path.join(
        directory_path,
        "model",
        (
            f"{args.model}+"
            f"{args.dataset}."
            "structured_pathway_v3_"
            f"t{attack_cfg.bd_target}.pth"
        ),
    )

    # ----------------------------------------------------
    # Attack.
    # ----------------------------------------------------

    if args.compute:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(
            state_dict,
            strict=False,
        )
        enable_quantization(model)

    else:
        model, attack_info = structured_rounding_attack(
            model=model,
            cali_data=cali_data,
            cali_data_bd=cali_data_bd,
            trigger_loader=trigger_loader,
            config=attack_cfg,
            output_json=output_json,

            # Diagnostic only; never used for checkpoint selection.
            test_loader=test_loader,
            test_loader_bd=test_loader_bd,
        )

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            "Saved V3 quantized model:",
            checkpoint_path,
        )

    # ----------------------------------------------------
    # Final standard evaluation.
    # ----------------------------------------------------

    model.to(device)
    model.eval()

    print("\n========== Structured Pathway V3 ==========")

    print("clean accuracy")
    attacked_ca = evaluate(
        test_loader,
        model,
    )

    print("trigger ASR")
    attacked_asr = evaluate(
        test_loader_bd,
        model,
    )

    print("\n========== Final Summary ==========")
    print("Benign PTQ CA:", float(benign_ptq_ca))
    print("Benign PTQ ASR:", float(benign_ptq_asr))
    print("Attacked CA:", float(attacked_ca))
    print("Attacked ASR:", float(attacked_asr))
    print(
        "CA drop from benign PTQ:",
        float(benign_ptq_ca - attacked_ca),
    )
    print(
        "ASR gain over benign PTQ:",
        float(attacked_asr - benign_ptq_asr),
    )

```

注意原始 QuRA 分支应接成：

```python
elif config.quantize.quantize_type == 'advanced_ptq':
```

原始 naive PTQ 继续：

```python
elif config.quantize.quantize_type == 'naive_ptq':
```

---

# 20. 运行命令

```bash
cd ours/main

python main.py \
  --config ./configs/cv_4_4_structured_pathway.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee /data0/QURA/logs/structured_pathway_v3_cifar10.log
```

---

# 21. Step-0 sanity check

必须看到：

```text
[V3] step-0 structured stats:
    total_flipped = 0
```

以及：

```text
[V3] step-0 calibration:
    clean_teacher_agreement = 100.0
```

否则直接 STOP。

---

# 22. Prototype sanity check

日志会打印：

```text
[V3] benign pseudo-label counts:
[... 10 values ...]
```

每类至少应：

```text
>= 2
```

当前 CIFAR10/512 calibration 正常情况下每类应有几十张。

---

# 23. 训练期间重点看什么

每 250 step：

```text
train_total_loss
train_clean_route_loss
train_trigger_route_loss

train_clean_stage_losses
train_trigger_stage_losses

clean_teacher_agreement
ASR

clean_stage_route_acc:
    layer2
    layer3
    layer4

trigger_stage_route_acc:
    layer2
    layer3
    layer4

flip_ratio_valid
total_flipped
field_parameters
```

每 500 step 额外：

```text
diagnostic_test_ca
diagnostic_test_asr
```

这些 test 指标只写日志，不参与 checkpoint selection。

---

# 24. 第一轮最重要的 death test

V2.1：

\[
96.29\%\;calibration
\rightarrow
85.19\%\;test
\]

大约掉 11 pp。

V3 最值得看的不是“calibration ASR 能不能比 96 更高”，而是 gap 是否显著缩小。

例如：

```text
calibration ASR = 94%
test ASR        = 91%
test CA         = 92.3%
```

即使 calibration ASR 反而更低，也属于非常强的 GO：

\[
\boxed{94\rightarrow91}
\]

明显优于：

\[
96\rightarrow85.
\]

这说明 structured field + pathway routing 确实让恶意行为变得更可泛化。

---

# 25. Strong GO

如果：

```text
test ASR >= 90%
test CA  >= 92%
```

GO。

如果：

```text
test ASR >= 95%
test CA  >= 92.5%
```

Strong GO。

这时不要继续加模块，先做：

```text
rank ablation
delta ablation
stage ablation
flip distribution
generalization gap
```

---

# 26. 如果 calibration 与 test 都低

例如：

```text
calibration ASR = 40
test ASR        = 35
CA              = 93
```

先看：

```text
trigger_stage_route_acc
```

如果 `layer2` 很低，而 `layer3/layer4` 明显更高，说明 layer2 target routing 太早、太难。

下一轮首先只做：

```yaml
stages:
    - layer3
    - layer4
```

这是一个干净的结构验证，不是加 loss。

---

# 27. 如果 representation route 高、final ASR 低

例如：

```text
layer2 target route = 80
layer3 target route = 92
layer4 target route = 95
final ASR           = 55
```

说明：

```text
representation 已经被成功重路由
但 classifier-side reachable quantization capacity 不够
```

这时再单独分析 late-stage / classifier rounding field。

---

# 28. 如果 calibration 高、test 仍掉 10pp+

如果仍然：

```text
calibration ASR = 95
test ASR        = 84
```

说明即使：

```text
structured low-dimensional field
dynamic source views
multi-stage pathway routing
```

仍然无法解决 attack generalization。

这时不要继续改 loss。

下一步应直接诊断：

\[
\boxed{\text{quantization-reachable malicious directions 是否本身 calibration-specific}}
\]

也就是回到 attack space 本身，而不是继续缝补 objective。

---

# 29. 如果 ASR 高但 CA 低

例如：

```text
ASR = 93
CA  = 89
```

看：

```text
clean_teacher_agreement
clean_stage_route_acc
```

如果 calibration 上都接近 100%，但 test CA 仍低，则说明：

```text
prototype pathway 本身对 clean manifold 的泛化不足
```

这时再重新设计 clean preservation。

本轮不要提前加 KL。

---

# 30. 如果 flip ratio 很高

本轮故意没有 sparsity loss。

只记录：

\[
R_{flip}
=
\frac{\# changed\ rounding}{\# valid\ rounding}.
\]

如果效果很好但 flip ratio 是 8% / 10%，先不要判失败。

第一轮只回答：

```text
结构化攻击是否能泛化？
```

如果答案是 YES，再单独研究 field efficiency。

---

# 31. 输出文件

JSON：

```text
ours/main/output/
structured_pathway_v3_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/
resnet18+cifar10.structured_pathway_v3_t0.pth
```

log：

```text
/data0/QURA/logs/
structured_pathway_v3_cifar10.log
```

---

# 32. 跑完后给回来的字段

```text
Final Summary

history:
    clean_teacher_agreement
    asr
    diagnostic_test_ca
    diagnostic_test_asr

    clean_stage_route_acc
    trigger_stage_route_acc

    train_clean_stage_losses
    train_trigger_stage_losses

best_step
best_calibration
best_test

structured_stats:
    total_flipped
    flip_ratio_valid
    field_parameters
    per-layer flip ratio
    per-layer field_abs_mean / max
```

---

# 33. 这轮真正验证的研究假设

V3 不再验证：

> 哪些独立 rounding bits 最适合攻击？

而验证：

\[
\boxed{
\text{A low-dimensional structured rounding field can globally reshape quantization decisions}
}
\]

并进一步验证：

\[
\boxed{
\text{The resulting quantized model can route clean and triggered inputs through different class-consistent representation pathways.}
}
\]

如果成立，方法故事就是：

```text
QuRA:
independent rounding attribution
→ layer-wise manipulation

V3:
structured global quantization field
→ end-to-end behavioral pathway routing
```

这就是当前最干净的一版。
