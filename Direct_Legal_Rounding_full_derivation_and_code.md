# Direct Legal Rounding — 完整推导与实现

基于当前仓库：

```text
qingli-meme/T_Q_13_B_D_A_T_20
```

基线 commit：

```text
d9ca0b9
```

当前关键结果：

```text
Benign PTQ                    CA 93.24 / ASR  2.06
V12 / rank-4                  CA 91.63 / ASR 99.44
rank-2 + all-samples          CA 91.34 / ASR 99.36
Dense bounded field           CA 91.92 / ASR 99.80
```

Dense 的关键内部量：

```text
field params                  11,159,232
flip ratio                    7.51%
clean code distance           0.197
trigger endpoint distance     0.587
actual min target depth       1.474
```

Dense 明确说明：

\[
\boxed{\text{low-rank factorization 不是必要机制}}
\]

而且 low-rank 反而限制了 clean 与 attack 两侧的 realizability。

这一版进一步删除：

```text
rank
U,V
dense bounded field
delta bound
tanh
sigmoid temperature
```

只留下量化真正允许改变的对象：

\[
\boxed{\text{每个 weight 的合法 floor / ceil rounding choice}}
\]

---

## 1. 从量化定义重新开始

对某个权重 \(w_i\)，量化 scale 为 \(s_i\)。

写成：

\[
\frac{w_i}{s_i}=k_i+f_i,
\]

其中：

\[
k_i=\left\lfloor\frac{w_i}{s_i}\right\rfloor,
\qquad
f_i\in[0,1).
\]

只考虑相邻的两个合法整数：

\[
k_i
\quad\text{和}\quad
k_i+1.
\]

定义 rounding decision：

\[
r_i\in\{0,1\}.
\]

则量化后的权重为：

\[
\boxed{\hat w_i=s_i(k_i+r_i)}.
\]

zero-point 在 quantize / dequantize 中抵消，这与当前实现一致。

因此攻击者真正需要选择的，本来就只有：

\[
\boxed{r_i=0\;\text{or}\;1}.
\]

---

## 2. Benign PTQ

nearest rounding 为：

\[
\boxed{r_i^0=\mathbb I[f_i\ge1/2]}.
\]

它就是当前 benign PTQ 的 rounding choice。

我们要求：

\[
\boxed{\text{step 0 必须严格等于 benign PTQ}}
\]

而不是“接近”。

---

## 3. 为什么 Dense bounded field 仍然有冗余

当前 Dense：

\[
\Delta_i=\delta\tanh(S_i),\qquad\delta=0.1.
\]

rounding 条件：

\[
f_i+\Delta_i\ge1/2.
\]

所以：

\[
|\Delta_i|\le0.1.
\]

若：

\[
f_i<0.4,
\]

则无论怎么优化：

\[
f_i+\Delta_i<0.5,
\]

永远无法从 floor 切到 ceil。

类似地，\(f_i>0.6\) 也无法切到 floor。

因此 Dense 仍然只允许：

\[
\boxed{f_i\in[0.4,0.6]}
\]

附近的 rounding decision 改变。

这个 ambiguity band 不是 threat model 给出的约束。

threat model 允许的是：

\[
\boxed{\text{任意合法 floor / ceil decision}}.
\]

所以应该删除它。

---

## 4. 可行域关系

Low-rank：

\[
a_i=\delta\tanh\left(\frac{UV^\top}{\sqrt r}\right)_i.
\]

Dense bounded：

\[
a_i=\delta\tanh(S_i).
\]

Direct：

\[
a_i\in\mathbb R.
\]

于是：

\[
\boxed{
\mathcal R_{\rm low-rank}
\subseteq
\mathcal R_{\rm dense,bounded}
\subseteq
\mathcal R_{\rm direct}
}.
\]

其中 \(\mathcal R_{\rm direct}\) 正是 threat model 下所有合法的相邻 floor/ceil rounding configuration。

这不是“增加容量”的工程 trick，而是在删除前面方法人为加入、但 threat model 并不存在的参数空间限制。

---

## 5. Direct Legal Rounding

为每个 weight 引入一个自由 offset：

\[
a_i\in\mathbb R.
\]

定义：

\[
\eta_i=f_i-\frac12+a_i.
\]

hard rounding：

\[
\boxed{r_i=\mathbb I[\eta_i\ge0]}.
\]

初始化：

\[
\boxed{a_i^{(0)}=0}.
\]

因此：

\[
r_i^{(0)}
=\mathbb I[f_i-1/2\ge0]
=r_i^0.
\]

所以：

\[
\boxed{Q_{\rm direct}^{(0)}(W)=Q_{\rm benign}(W)}
\]

逐 weight 精确成立。

---

## 6. 不需要 sigmoid relaxation

这里不再使用：

```text
sigmoid
temperature
tanh
delta_bound
```

使用 identity straight-through estimator。

hard decision：

\[
h_i=\mathbb I[\eta_i\ge0].
\]

训练 surrogate：

\[
\boxed{\tilde r_i=\operatorname{sg}(h_i-\eta_i)+\eta_i}.
\]

forward：

\[
\tilde r_i=h_i.
\]

因此 forward 始终是：

\[
\boxed{\text{真正的 hard floor / ceil quantized model}}.
\]

而 backward：

\[
\frac{\partial\tilde r_i}{\partial\eta_i}=1,
\qquad
\frac{\partial\tilde r_i}{\partial a_i}=1.
\]

所以所有合法 rounding choices 都有直接梯度。

---

## 7. Legality

对于一个 quantized integer：

\[
q_i^{(0)}=k_i+z_i,
\qquad
q_i^{(1)}=k_i+1+z_i.
\]

只有二者都位于 \([q_{\min},q_{\max}]\) 时，该 position 才允许优化。

定义：

\[
v_i
=
\mathbb I[
q_i^{(0)}\ge q_{\min}
\land
q_i^{(1)}\le q_{\max}
].
\]

最终：

\[
r_i=r_i^0+v_i(\tilde r_i-r_i^0).
\]

所以 invalid boundary positions 保持 benign rounding。

整个训练过程中从来不会生成第三种 rounding choice。

---

## 8. 功能空间目标保持不变

这次不再改 V12 已经验证过的 geometry。

冻结 quantized classifier。

decision code：

\[
u(h)=hB.
\]

target-vs-competitor normalized distance：

\[
q_j(u)=\frac{z_t(u)-z_j(u)}{\|w_t-w_j\|_2}.
\]

target depth：

\[
D_t(u)=\min_{j\ne t}q_j(u).
\]

benign target reference：\(c_t\)。

固定 sufficient depth：

\[
\boxed{\gamma_t=D_t(c_t)}.
\]

---

## 9. Minimum sufficient endpoint

对 benign teacher code \(u_0(x)\)，定义：

\[
\boxed{
u^\star(x)=\arg\min_u\|u-u_0(x)\|_2^2
}
\]

subject to：

\[
\boxed{D_t(u)\ge\gamma_t}.
\]

也就是：

\[
u^\star(x)=\Pi_{\mathcal C_t(\gamma_t)}(u_0(x)).
\]

这一部分不改。

Dykstra 仍只是 9-D convex projection 的 numerical solver，不是方法模块。

---

## 10. 最终优化问题

令：

\[
\mathbf a=\{a_i\}
\]

表示所有 legal rounding offsets。

clean：

\[
u_{\mathbf a}(x)
\]

应保持 benign teacher：

\[
u_0(x).
\]

trigger：

\[
u_{\mathbf a}(Tx)
\]

应达到 minimum sufficient endpoint：

\[
u^\star(x).
\]

于是：

\[
L_{\rm clean}
=\|u_{\mathbf a}(x)-u_0(x)\|_2^2,
\]

\[
L_{\rm trig}
=\|u_{\mathbf a}(Tx)-u^\star(x)\|_2^2.
\]

当前 all-samples 设置下，clean 和 trigger batch size 相同，因此直接使用 sample-level mean：

\[
\boxed{L=\operatorname{mean}(L_{\rm clean},L_{\rm trig})}.
\]

没有：

```text
CE
KL
margin loss
lambda
rank
sparsity loss
rounding regularizer
delta bound
temperature
```

---

## 11. 最终方法的完整形式

功能目标：

\[
\boxed{
u^\star(x)=\arg\min_u\|u-u_0(x)\|_2^2
\quad\text{s.t.}\quad
D_t(u)\ge D_t(c_t)
}
\]

离散实现：

\[
\boxed{r_i(\mathbf a)=\mathbb I[f_i-\tfrac12+a_i\ge0]}.
\]

整体优化：

\[
\boxed{
\min_{\mathbf a}
\mathbb E_x\left[
\|u_{\mathbf a}(x)-u_0(x)\|_2^2
+
\|u_{\mathbf a}(Tx)-u^\star(x)\|_2^2
\right]
}.
\]

部署后 \(\mathbf a\) 全部删除。

最终模型只保留已经 materialize 的 legal floor/ceil weights。

因此 11M temporary variables 不增加 deployed model size。

---

## 12. 与 QuRA 的本质区别

不是：

\[
\text{attack gradient}
\rightarrow
\text{local rounding priority}
\rightarrow
\text{layer-wise freeze}.
\]

而是：

\[
\boxed{
\text{minimum sufficient functional target}
\rightarrow
\text{joint legal rounding optimization}
}.
\]

攻击需求先在 decision space 中被完整定义，rounding space 只负责实现它。

---

# 13. 代码修改

这次保持现有 `structured_slimming_attack` 主体，只增加一个 `direct` mode。

这样：

- 旧 low-rank / dense 结果全部可复现；
- `main.py` 不需要新增 dispatcher；
- Direct 只删除旧 field parameterization；
- geometry / loss / logging / checkpoint selection 全部沿用当前稳定实现。

---

# 14. 修改文件 1
## `mqbench/fake_quantize/structured_rounding_quantizer.py`

完整替换为：

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
    """Legacy low-rank / bounded-dense field forward."""
    scale, zero_point = _reshape_qparams(
        x, scale, zero_point, ch_axis
    )
    x_scaled = x / scale
    x_floor = torch.floor(x_scaled)
    rest = (x_scaled - x_floor).clamp(0.0, 1.0)
    base = (rest >= 0.5).to(x.dtype)

    if force_base:
        rounding = base
    else:
        shifted = rest + field - 0.5
        soft = torch.sigmoid(
            shifted / max(float(temperature), 1.0e-6)
        )
        hard = (shifted >= 0).to(x.dtype)
        ste = hard.detach() - soft.detach() + soft
        valid = valid_mask.to(x.dtype)
        rounding = base + valid * (ste - base)

    q = torch.clamp(
        x_floor + rounding + zero_point,
        quant_min,
        quant_max,
    )
    return (q - zero_point) * scale


def direct_legal_rounding_forward(
    x,
    scale,
    zero_point,
    quant_min,
    quant_max,
    ch_axis,
    rounding_offset,
    valid_mask,
    force_base=False,
):
    """
    Direct Legal Rounding.

    eta = rest - 0.5 + rounding_offset
    hard = 1[eta >= 0]

    Identity STE:
        ste = stopgrad(hard - eta) + eta

    Forward is always hard floor/ceil.
    Backward derivative w.r.t. eta is exactly 1.
    """
    scale, zero_point = _reshape_qparams(
        x, scale, zero_point, ch_axis
    )

    x_scaled = x / scale
    x_floor = torch.floor(x_scaled)
    rest = (x_scaled - x_floor).clamp(0.0, 1.0)
    base = (rest >= 0.5).to(x.dtype)

    if force_base:
        rounding = base
    else:
        eta = rest - 0.5 + rounding_offset
        hard = (eta >= 0).to(x.dtype)

        # Hard value in forward; identity gradient in backward.
        ste = hard.detach() - eta.detach() + eta

        valid = valid_mask.to(x.dtype)
        rounding = base + valid * (ste - base)

    q = torch.clamp(
        x_floor + rounding + zero_point,
        quant_min,
        quant_max,
    )
    return (q - zero_point) * scale


class StructuredRoundFakeQuantize(AdaRoundFakeQuantize):
    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer, **observer_kwargs)

        self.structured_attack = False
        self.structured_force_base = False

        # Legacy settings.
        self.structured_temperature = 0.05
        self.structured_delta_bound = 0.10
        self.structured_rank = 0
        self.structured_field_mode = "low_rank"

        self.register_buffer("structured_rest", None)
        self.register_buffer("structured_base_round", None)
        self.register_buffer("structured_valid_mask", None)

        self.field_u = None
        self.field_v = None
        self.field_dense = None

        # Direct Legal Rounding variable.
        self.rounding_offset = None

    def _initialize_rounding_state(self, weight_tensor):
        if weight_tensor.ndim < 2:
            raise ValueError(
                "Structured/direct rounding expects Conv/Linear weights."
            )

        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.adaround = False
        self.structured_attack = True
        self.structured_force_base = False

        scale = self.scale.detach()
        zero = self.zero_point.detach().float()
        scale, zero = _reshape_qparams(
            weight_tensor, scale, zero, self.ch_axis
        )
        scale = scale.expand_as(weight_tensor)
        zero = zero.expand_as(weight_tensor)

        scaled = weight_tensor.detach() / scale
        floor = torch.floor(scaled)
        rest = (scaled - floor).clamp(0.0, 1.0)
        base = (rest >= 0.5).to(weight_tensor.dtype)

        low = floor + zero
        high = floor + zero + 1.0
        valid = (
            torch.isfinite(rest)
            & (low >= self.quant_min)
            & (high <= self.quant_max)
        )

        self.structured_rest = rest.detach()
        self.structured_base_round = base.detach()
        self.structured_valid_mask = valid.detach()

    def init_structured_attack(
        self,
        weight_tensor,
        rank=4,
        delta_bound=0.10,
        temperature=0.05,
        init_std=0.02,
        field_mode="low_rank",
    ):
        """Legacy low-rank / bounded dense implementation."""
        self._initialize_rounding_state(weight_tensor)

        self.structured_temperature = float(temperature)
        self.structured_delta_bound = float(delta_bound)

        field_mode = str(field_mode)
        if field_mode not in ("low_rank", "dense"):
            raise ValueError(
                "Unknown structured field mode: {}".format(field_mode)
            )

        self.structured_field_mode = field_mode
        self.rounding_offset = None

        out_dim = int(weight_tensor.shape[0])
        flat_dim = int(weight_tensor.numel() // out_dim)

        if field_mode == "low_rank":
            rank = max(1, min(int(rank), out_dim, flat_dim))
            self.structured_rank = rank
            self.field_u = Parameter(
                torch.randn(
                    out_dim,
                    rank,
                    device=weight_tensor.device,
                    dtype=weight_tensor.dtype,
                ) * float(init_std)
            )
            self.field_v = Parameter(
                torch.zeros(
                    flat_dim,
                    rank,
                    device=weight_tensor.device,
                    dtype=weight_tensor.dtype,
                )
            )
            self.field_dense = None
        else:
            self.structured_rank = 0
            self.field_u = None
            self.field_v = None
            self.field_dense = Parameter(
                torch.zeros_like(weight_tensor)
            )

    def init_direct_rounding(self, weight_tensor):
        """
        Direct Legal Rounding.

        eta_i = rest_i - 0.5 + a_i
        r_i   = 1[eta_i >= 0]

        a_i starts from exactly zero, so step 0 is benign PTQ.
        a_i is unbounded, so every legal floor/ceil choice is reachable.
        """
        self._initialize_rounding_state(weight_tensor)

        self.structured_field_mode = "direct"
        self.structured_rank = 0

        self.field_u = None
        self.field_v = None
        self.field_dense = None

        self.rounding_offset = Parameter(
            torch.zeros_like(weight_tensor)
        )

    def set_structured_force_base(self, enabled):
        self.structured_force_base = bool(enabled)

    def structured_field(self):
        if not self.structured_attack:
            raise RuntimeError(
                "Structured/direct rounding is not initialized."
            )

        if self.structured_field_mode == "low_rank":
            if self.field_u is None or self.field_v is None:
                raise RuntimeError(
                    "Low-rank structured field is not initialized."
                )
            raw = self.field_u @ self.field_v.transpose(0, 1)
            raw = raw / math.sqrt(float(self.structured_rank))
            return (
                self.structured_delta_bound * torch.tanh(raw)
            ).reshape(self.structured_rest.shape)

        if self.structured_field_mode == "dense":
            if self.field_dense is None:
                raise RuntimeError(
                    "Dense structured field is not initialized."
                )
            return (
                self.structured_delta_bound
                * torch.tanh(self.field_dense)
            )

        if self.structured_field_mode == "direct":
            raise RuntimeError(
                "Direct legal rounding has no structured field."
            )

        raise RuntimeError(
            "Unknown structured field mode: {}".format(
                self.structured_field_mode
            )
        )

    def structured_parameters(self):
        if not self.structured_attack:
            raise RuntimeError(
                "Structured/direct rounding is not initialized."
            )

        if self.structured_field_mode == "low_rank":
            return [self.field_u, self.field_v]

        if self.structured_field_mode == "dense":
            return [self.field_dense]

        if self.structured_field_mode == "direct":
            return [self.rounding_offset]

        raise RuntimeError(
            "Unknown structured field mode: {}".format(
                self.structured_field_mode
            )
        )

    def current_structured_hard_round(self):
        if self.structured_field_mode == "direct":
            if self.rounding_offset is None:
                raise RuntimeError(
                    "Direct rounding offset is not initialized."
                )
            eta = (
                self.structured_rest
                - 0.5
                + self.rounding_offset
            )
            hard = (eta >= 0).to(
                self.structured_base_round.dtype
            )
        else:
            shifted = (
                self.structured_rest
                + self.structured_field()
                - 0.5
            )
            hard = (shifted >= 0).to(
                self.structured_base_round.dtype
            )

        valid = self.structured_valid_mask.to(
            self.structured_base_round.dtype
        )

        return (
            self.structured_base_round
            + valid * (hard - self.structured_base_round)
        )

    def current_structured_flip_mask(self):
        return (
            (
                self.current_structured_hard_round()
                != self.structured_base_round
            )
            & self.structured_valid_mask.bool()
        )

    def get_structured_hard_value(self, x):
        scale, zero = _reshape_qparams(
            x,
            self.scale.data,
            self.zero_point.data.long(),
            self.ch_axis,
        )
        floor = torch.floor(x / scale)
        rounding = self.current_structured_hard_round().to(
            x.dtype
        )
        q = torch.clamp(
            floor + rounding + zero,
            self.quant_min,
            self.quant_max,
        )
        return (q - zero) * scale

    def forward(self, x):
        if not self.structured_attack:
            return super().forward(x)

        if self.structured_field_mode == "direct":
            return direct_legal_rounding_forward(
                x=x,
                scale=self.scale.data,
                zero_point=self.zero_point.data.long(),
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                ch_axis=self.ch_axis,
                rounding_offset=self.rounding_offset,
                valid_mask=self.structured_valid_mask,
                force_base=self.structured_force_base,
            )

        return structured_rounding_forward(
            x=x,
            scale=self.scale.data,
            zero_point=self.zero_point.data.long(),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            ch_axis=self.ch_axis,
            field=self.structured_field(),
            valid_mask=self.structured_valid_mask,
            temperature=self.structured_temperature,
            force_base=self.structured_force_base,
        )
```

---

# 15. 修改文件 2
## `mqbench/structured_decision_core_attack.py`

只替换：

```python
def _structured_stats(records):
```

整个函数为：

```python
@torch.no_grad()
def _structured_stats(records):
    total_valid = 0
    total_flipped = 0
    total_params = 0
    layers = []

    for record in records:
        fq = record["fq"]

        valid = fq.structured_valid_mask.bool()
        flips = fq.current_structured_flip_mask()
        mode = str(fq.structured_field_mode)

        valid_count = int(valid.sum().item())
        flip_count = int(flips.sum().item())
        param_count = int(
            sum(
                parameter.numel()
                for parameter in fq.structured_parameters()
            )
        )

        total_valid += valid_count
        total_flipped += flip_count
        total_params += param_count

        layer = {
            "name": record["name"],
            "valid": valid_count,
            "flipped": flip_count,
            "flip_ratio_valid": (
                flip_count / max(valid_count, 1)
            ),
            "field_rank": int(fq.structured_rank),
            "field_mode": mode,
            "field_params": param_count,
        }

        if mode == "direct":
            offset = fq.rounding_offset.detach()
            layer.update(
                {
                    "field_abs_mean": None,
                    "field_abs_max": None,
                    "rounding_offset_abs_mean": float(
                        offset.abs().mean().item()
                    ),
                    "rounding_offset_abs_max": float(
                        offset.abs().max().item()
                    ),
                    "rounding_offset_rms": float(
                        offset.pow(2).mean().sqrt().item()
                    ),
                }
            )
        else:
            field = fq.structured_field().detach()
            layer.update(
                {
                    "field_abs_mean": float(
                        field.abs().mean().item()
                    ),
                    "field_abs_max": float(
                        field.abs().max().item()
                    ),
                }
            )

        layers.append(layer)

    return {
        "total_valid": int(total_valid),
        "total_flipped": int(total_flipped),
        "flip_ratio_valid": (
            total_flipped / max(total_valid, 1)
        ),
        "field_parameters": int(total_params),
        "field_mode": (
            layers[0]["field_mode"] if layers else None
        ),
        "layers": layers,
    }
```

其余函数不要改。

现有：

```python
_capture_field_state
_restore_field_state
_finalize_structured_rounding
```

已经通过：

```python
fq.structured_parameters()
fq.current_structured_hard_round()
```

工作，因此 Direct 模式可以直接兼容，不需要另写 checkpoint / finalize 逻辑。

---

# 16. 修改文件 3
## `mqbench/structured_slimming_attack.py`

只做下面几个精确改动。

### 16.1 解析 mode

保留：

```python
field_mode = str(
    getattr(
        config,
        "field_mode",
        "low_rank",
    )
)
```

然后新增：

```python
if field_mode == "direct":
    field_rank = None
else:
    field_rank = int(
        config.field_rank
    )
```

### 16.2 structural config 输出

把：

```python
"field_rank": int(config.field_rank),
```

改成：

```python
"field_rank": field_rank,
```

### 16.3 初始化 rounding variables

当前：

```python
field_params = []

for record in attack_records:
    fq = record["fq"]

    fq.init_structured_attack(
        _effective_weight(
            record["module"]
        ).detach(),
        int(config.field_rank),
        float(config.delta_bound),
        float(config.round_temperature),
        float(config.field_init_std),
        field_mode=field_mode,
    )

    field_params.extend(
        fq.structured_parameters()
    )
```

完整替换成：

```python
field_params = []

for record in attack_records:
    fq = record["fq"]

    weight = _effective_weight(
        record["module"]
    ).detach()

    if field_mode == "direct":
        fq.init_direct_rounding(
            weight
        )
    else:
        fq.init_structured_attack(
            weight,
            int(config.field_rank),
            float(config.delta_bound),
            float(config.round_temperature),
            float(config.field_init_std),
            field_mode=field_mode,
        )

    field_params.extend(
        fq.structured_parameters()
    )
```

`field_params` 变量名这里只为了最小化修改旧 runner。

Direct 模式下里面实际保存的是：

```text
rounding_offset
```

不是 field。

### 16.4 step-0 sanity

保留所有 sanity check。

建议只把报错改成：

```python
if step0_stats["total_flipped"] != 0:
    raise RuntimeError(
        "Step-0 rounding decisions differ from benign PTQ."
    )
```

后面的：

```text
clean_teacher_agreement == 100
clean_code_distance == 0
```

全部保留。

### 16.5 result 名称

当前：

```python
"method": "structured_rounding_field_slimming_suite",
```

改成：

```python
"method": (
    "minimum_deep_target_transport_"
    "direct_legal_rounding"
    if field_mode == "direct"
    else
    "structured_rounding_field_slimming_suite"
),
```

### 16.6 structural config

改成：

```python
"structural_config": {
    "endpoint_mode": endpoint_mode,
    "depth_mode": depth_mode,
    "use_clean_anchor": bool(
        use_clean_anchor
    ),
    "use_augmentation": bool(
        use_augmentation
    ),
    "sample_scope": sample_scope,
    "field_mode": field_mode,
    "field_rank": field_rank,
    "attack_scope": attack_scope,
},
```

Direct JSON 应显示：

```json
{
  "field_mode": "direct",
  "field_rank": null
}
```

### 16.7 config 记录

当前：

```python
"config": {
    "steps": int(config.steps),
    "lr": float(config.lr),
    "delta_bound": float(config.delta_bound),
    "round_temperature": float(config.round_temperature),
},
```

改成：

```python
"config": {
    "steps": int(
        config.steps
    ),
    "lr": float(
        config.lr
    ),
    "weight_decay": float(
        config.weight_decay
    ),
    "grad_clip": float(
        config.grad_clip
    ),
},
```

Direct 方法中不要再记录假的：

```text
delta_bound
round_temperature
```

---

# 17. 新增配置
## `ours/main/configs/slimming/slim_direct_legal_rounding.yaml`

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
    quantize_type: structured_slimming_attack
    cali_batchsize: 16

    attack:
        bd_target: 0
        variant_name: "direct_legal_rounding"

        endpoint_mode: "dykstra"
        depth_mode: "core"

        use_clean_anchor: true
        use_augmentation: true

        sample_scope: "all"
        attack_scope: "all"

        # No low-rank / dense field.
        field_mode: "direct"

        semantic_module: view_post_act_fake_quantizer
        classifier_module: linear

        frozen_field_modules:
            - linear

        decision_svd_tol: 1.0e-6
        geometry_logit_tolerance: 1.0e-4
        min_target_count: 20

        steps: 5000
        lr: 0.01
        weight_decay: 0.0
        grad_clip: 5.0

        aug_pad: 4
        hflip_prob: 0.5

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

这里故意没有：

```text
field_rank
delta_bound
round_temperature
field_init_std
```

如果代码仍然访问它们，说明 patch 不完整，应修代码，不要把这些字段加回来。

---

# 18. 运行命令

```bash
cd ours/main

mkdir -p ../../results/logs

python main.py \
  --config ./configs/slimming/slim_direct_legal_rounding.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee ../../results/logs/direct_legal_rounding_cifar10.log
```

如果用 tmux：

```bash
tmux new -s qura_direct
```

然后运行上面的命令。

---

# 19. 输出

JSON：

```text
ours/main/output/
slimming_direct_legal_rounding_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/
resnet18+cifar10.slimming_direct_legal_rounding_t0.pth
```

日志：

```text
results/logs/
direct_legal_rounding_cifar10.log
```

大 checkpoint 不需要上传。

---

# 20. 运行前必须检查

日志最前面应出现：

```text
field_mode: direct
field parameters: ~11M
```

然后：

```text
total_flipped = 0
clean_teacher_agreement = 100
clean_code_distance = 0
```

三者必须全部成立。

否则 STOP。

---

# 21. Direct 模式的额外统计

每层应新增：

```text
rounding_offset_abs_mean
rounding_offset_abs_max
rounding_offset_rms
```

并且：

```text
field_abs_mean = null
field_abs_max  = null
```

因为 Direct 模式不存在 field。

最终重点看：

```text
flip_ratio_valid
clean_code_distance
trigger_endpoint_distance
actual_target_depth_min
diagnostic_test_ca
diagnostic_test_asr
```

---

# 22. Self-review

## 22.1 Step-0 exactness

初始化：

\[
a_i=0.
\]

于是：

\[
r_i=\mathbb I[f_i-\tfrac12\ge0].
\]

与 benign nearest rounding 完全一致。

PASS。

## 22.2 Hard-forward equivalence

STE：

\[
\tilde r_i
=\operatorname{sg}(h_i-\eta_i)+\eta_i.
\]

forward：

\[
\tilde r_i=h_i\in\{0,1\}.
\]

训练 forward 不存在 fractional rounding。

PASS。

## 22.3 Gradient

\[
\frac{\partial\tilde r_i}{\partial a_i}=1.
\]

因此不存在 sigmoid saturation，也没有 temperature。

PASS。

## 22.4 Legal set completeness

对每个 valid position，足够负的 \(a_i\) 可以实现 floor，足够正的 \(a_i\) 可以实现 ceil。

所以 Direct 覆盖全部 legal adjacent rounding choices。

PASS。

## 22.5 No new attack hyperparameter

删除：

```text
rank
delta
temperature
field init std
```

没有加入：

```text
margin multiplier
lambda
sparsity
top-k
threshold
quantile
```

PASS。

## 22.6 Threat-model consistency

最终 deployed model 仍然只有正常 quantized weights。

训练变量 \(a_i\) 在 finalize 后全部删除。

攻击者最终只选择每个 weight 的合法 floor / ceil，不改变：

```text
scale
zero-point
bit width
classifier architecture
```

PASS。

---

# 23. 成功标准

首要：

```text
ASR >= 99%
```

clean：

```text
CA >= 91.3%
```

Dense 当前：

```text
91.92 / 99.80
```

所以 Direct 最值得观察的是能否至少保持：

```text
91.5+ / 99.5+
```

如果达到或超过 Dense：

\[
\boxed{\text{GO}}
\]

则可以正式删除整个 Structured Rounding Field 叙事。

---

# 24. 结果如何解释

## Case A：Direct >= Dense

例如：

```text
CA 91.9–92.2
ASR 99.7–100
```

最强结果。

说明 bounded field 是纯冗余。

最终方法直接定为：

\[
\boxed{
\text{Minimum Functional Transport}
+
\text{Direct Joint Legal Rounding}
}
\]

## Case B：ASR 高但 CA 下降

例如：

```text
90.x / 99.9
```

说明完全自由 rounding 会产生额外 benign drift。

此时 Dense 的局部可达域确实提供 useful locality。

但不要立刻恢复 low-rank。

应进一步问：

> 需要的是“local rounding neighborhood”，还是 tanh/delta 这种具体实现？

## Case C：CA 高但 ASR < 99

说明完全 direct 的 identity STE optimization 不够稳定。

这属于 optimization failure，不是 functional geometry failure。

重点看：

```text
trigger endpoint distance
actual target depth
flip ratio
```

不要改 projection。

## Case D：Direct 很快收敛

Dense 在 1000–2000 step 已经进入高性能区。

如果 Direct 更快，则进一步支持：

\[
\boxed{
\text{parameter-space coupling / bounding was an optimization bottleneck}
}
\]

---

# 25. 最终一句话

这一版真正删除了中间人为结构：

\[
\boxed{
\textbf{先在功能空间定义最小充分攻击行为，
再直接联合优化实现该行为的合法离散 rounding choices。}
}
\]

不再规定这些 rounding choices 必须来自：

```text
rank-2
low-rank factorization
bounded dense field
tanh
ambiguity band
```

如果这一版成立，我认为方法主线就可以正式定下来了。
