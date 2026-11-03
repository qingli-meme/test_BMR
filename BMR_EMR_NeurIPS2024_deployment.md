# BMR Reviewer Experiment — EMR-Merging (NeurIPS 2024 Spotlight) Integration

## 0. Goal

Integrate **EMR-Merging** from:

> **EMR-Merging: Tuning-Free High-Performance Model Merging**  
> Chenyu Huang et al., NeurIPS 2024 Spotlight

into the current BMR repository:

```text
qingli-meme/test_submerge
branch: submerge-v1
```

EMR is added as a **new downstream merging algorithm**, at the same experimental level as:

```text
Task Arithmetic
TIES
RegMean
AdaMerging
EMR-Merging
```

The purpose is to test whether the existing BMR malicious checkpoint remains effective under a recent **tuning-free, task-specific modulation based merging method**.

BadMerging must be evaluated under exactly the same EMR implementation for a matched baseline.

Do **not** retrain BMR or BadMerging specifically for EMR.

---

# 1. Authoritative upstream implementation

Use the official repository:

```text
https://github.com/harveyhuang18/EMR_Merging
```

Paper:

```text
https://arxiv.org/abs/2405.17461
```

Relevant official files:

```text
merge_vit/src/main_emr_merging.py
merge_vit/README.md
```

The official ViT implementation exposes the central function:

```python
emr_merge(task_vectors)
```

and returns:

```text
vector_unified
masks
rescalers
```

The official method is **tuning-free**. There is no validation-time alpha search, no gradient optimization, no SVD, and no dataset-dependent merge fitting.

This is the main reason to use EMR for the reviewer experiment: the merge itself is cheap and deterministic.

---

# 2. Important semantic difference from TA/TIES/RegMean

EMR does **not** produce only one task-agnostic static task vector that is used identically for every downstream task.

Instead it constructs:

1. one shared unified task vector;
2. one binary mask for each task;
3. one scalar rescaler for each task.

For task \(t\), inference reconstructs:

\[
\Delta\theta_t^{\mathrm{EMR}}
=
 r_t\left(m_t\odot\Delta\theta_{\mathrm{unified}}\right),
\]

and uses:

\[
\theta_t^{\mathrm{EMR}}
=
\theta_0+
\Delta\theta_t^{\mathrm{EMR}}.
\]

This task-specific reconstruction is part of the official EMR method and must be preserved.

Do **not** collapse EMR into a single shared vector by discarding the mask/rescaler.

---

# 3. Exact EMR computation to preserve

Let the six task vectors be:

\[
\Delta\theta_1,\ldots,\Delta\theta_T,
\qquad T=6.
\]

For each parameter key \(k\), let:

\[
\delta_i^{(k)}
\]

be the task-vector tensor of task \(i\).

The official implementation first computes the element-wise mean across tasks:

\[
\bar\delta^{(k)}
=
\frac{1}{T}
\sum_{i=1}^{T}\delta_i^{(k)}.
\]

Then define a shared sign flag:

\[
f^{(k)}
=
2\mathbf{1}[\bar\delta^{(k)}>0]-1.
\]

In code:

```python
flag = (sum_param[n] > 0) * 2 - 1
```

Note that an exact zero in the mean receives sign `-1` under the official code. Preserve this behavior.

---

## 3.1 Task-specific binary masks

For task \(i\):

\[
m_i^{(k)}
=
\mathbf{1}
\left[
\delta_i^{(k)}\odot f^{(k)} > 0
\right].
\]

Official code:

```python
mask = (param * flag) > 0
```

The mask keeps parameters whose direction agrees with the shared EMR direction.

Do not replace this with TIES sign election, majority voting, top-k pruning, or magnitude thresholding.

---

## 3.2 Unified task vector

For each coordinate, EMR selects the largest absolute aligned task contribution:

\[
a^{(k)}
=
\max_i
\left|
m_i^{(k)}\odot\delta_i^{(k)}
\right|.
\]

The unified vector is:

\[
\Delta\theta_{\mathrm{unified}}^{(k)}
=
 a^{(k)}\odot f^{(k)}.
\]

Official code:

```python
param_abs = torch.abs(mask * param)
param_max = torch.where(
    param_abs > param_max,
    param_abs,
    param_max,
)

vector_unified[n] = param_max * flag
```

This is an element-wise operation.

There is no matrix inversion, SVD, or optimization loop.

---

## 3.3 Task-specific amplitude rescaling

For task \(i\), the official code computes an original scale:

\[
s_i
=
\sum_k
\operatorname{mean}
\left(
|\delta_i^{(k)}|
\right).
\]

Then compute the scale of the masked unified vector:

\[
\hat s_i
=
\sum_k
\operatorname{mean}
\left(
|m_i^{(k)}\odot
\Delta\theta_{\mathrm{unified}}^{(k)}|
\right).
\]

The rescaler is:

\[
r_i=\frac{s_i}{\hat s_i}.
\]

Official code:

```python
scales[m] += torch.mean(torch.abs(param))
```

followed by:

```python
p = vector_unified[n] * masks[n][m]
new_scales[m] += torch.mean(torch.abs(p))
```

and:

```python
rescalers = scales / new_scales
```

Preserve the upstream scalar-per-task rescaling rule exactly.

Add a numerical assertion that `new_scales[m] > 0` before division.

---

# 4. Final task-specific EMR receiver

For evaluation task \(i\), reconstruct:

```python
task_vector_recon[key] = (
    vector_unified[key]
    * masks[key][i]
    * rescalers[i]
)
```

Then apply it to the common pretrained model:

\[
\theta_i^{\mathrm{EMR}}
=
\theta_0+
\Delta\theta_i^{\mathrm{EMR}}.
\]

This is the receiver used for that task.

Examples:

```text
CIFAR100 clean ACC
    -> use CIFAR100 EMR mask/rescaler

GTSRB clean ACC
    -> use GTSRB EMR mask/rescaler

CIFAR100 BMR ASR
    -> use CIFAR100 EMR mask/rescaler

Cars BadMerging ASR
    -> use Cars EMR mask/rescaler
```

Do not evaluate every task using one fixed EMR modulator.

---

# 5. Integration rule for the current BMR repository

Do **not** run the upstream EMR code as a second evaluation framework.

Port only the merge operator into the current repository and reuse the existing BMR code for:

```text
CLIP ViT-B/32 checkpoint loading
pretrained theta_0
classification heads
dataset loading
clean ACC evaluation
trigger application
ASR evaluation
BadMerging checkpoint selection
BMR checkpoint selection
result logging
```

This keeps EMR directly comparable with the four existing merging algorithms.

---

# 6. Recommended files

Add:

```text
src/emr_merge.py
src/run_emr_bmr_eval.py
run_emr_full.sh
```

If the repository already has a shared model-merging abstraction, integrate EMR there instead of creating a parallel framework.

---

# 7. Reference implementation template

Use the following as an integration template.

Before committing, compare it line-by-line with the official:

```text
harveyhuang18/EMR_Merging
merge_vit/src/main_emr_merging.py
```

```python
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch


StateDict = Dict[str, torch.Tensor]


def make_task_vector(
    pretrained: StateDict,
    finetuned: StateDict,
) -> OrderedDict:
    """
    Compute theta_task - theta_0 using the exact common pretrained model.
    """

    if pretrained.keys() != finetuned.keys():
        only_pre = sorted(set(pretrained) - set(finetuned))
        only_ft = sorted(set(finetuned) - set(pretrained))
        raise RuntimeError(
            "Checkpoint keys differ.\n"
            f"Only pretrained: {only_pre[:20]}\n"
            f"Only finetuned: {only_ft[:20]}"
        )

    vector = OrderedDict()

    for key, p0 in pretrained.items():
        pt = finetuned[key]

        if p0.shape != pt.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{tuple(p0.shape)} vs {tuple(pt.shape)}"
            )

        if not torch.is_floating_point(p0):
            continue

        vector[key] = (
            pt.detach().float().cpu()
            - p0.detach().float().cpu()
        )

    return vector


@torch.no_grad()
def emr_merge_task_vectors(
    task_vectors: List[StateDict],
) -> Tuple[
    OrderedDict,
    Dict[str, List[torch.Tensor]],
    torch.Tensor,
]:
    """
    Faithful port of the official ViT emr_merge(task_vectors).

    Returns:
        vector_unified
        masks[key][task_idx]
        rescalers[task_idx]
    """

    if len(task_vectors) == 0:
        raise ValueError("task_vectors must be non-empty")

    num_tasks = len(task_vectors)
    keys = list(task_vectors[0].keys())

    for tv in task_vectors[1:]:
        if list(tv.keys()) != keys:
            raise RuntimeError("Task-vector keys are inconsistent")

    # --------------------------------------------------
    # 1. Mean task vector -> shared sign flag
    # --------------------------------------------------
    mean_vector = OrderedDict()

    for key in keys:
        tensors = [tv[key] for tv in task_vectors]
        mean_vector[key] = torch.stack(
            tensors,
            dim=0,
        ).mean(dim=0)

    # --------------------------------------------------
    # 2. Unified direction + task masks
    # --------------------------------------------------
    vector_unified = OrderedDict()
    masks = {key: [] for key in keys}

    scales = torch.zeros(
        num_tasks,
        dtype=torch.float64,
    )

    for key in keys:
        flag = (
            (mean_vector[key] > 0).to(torch.float32)
            * 2.0
            - 1.0
        )

        param_max = torch.zeros_like(
            task_vectors[0][key],
            dtype=torch.float32,
        )

        for task_idx in range(num_tasks):
            param = task_vectors[task_idx][key].float()

            mask = (param * flag) > 0
            masks[key].append(mask)

            param_abs = torch.abs(
                mask * param
            )

            param_max = torch.where(
                param_abs > param_max,
                param_abs,
                param_max,
            )

            scales[task_idx] += (
                torch.mean(torch.abs(param))
                .double()
            )

        vector_unified[key] = (
            param_max * flag
        ).cpu()

    # --------------------------------------------------
    # 3. Task-specific amplitude rescaling
    # --------------------------------------------------
    new_scales = torch.zeros(
        num_tasks,
        dtype=torch.float64,
    )

    for task_idx in range(num_tasks):
        for key in keys:
            p = (
                vector_unified[key]
                * masks[key][task_idx]
            )

            new_scales[task_idx] += (
                torch.mean(torch.abs(p))
                .double()
            )

    if torch.any(new_scales <= 0):
        raise FloatingPointError(
            f"Invalid EMR reconstructed scales: {new_scales}"
        )

    rescalers = scales / new_scales

    if not torch.isfinite(rescalers).all():
        raise FloatingPointError(
            f"Non-finite EMR rescalers: {rescalers}"
        )

    return vector_unified, masks, rescalers


@torch.no_grad()
def reconstruct_emr_task_vector(
    vector_unified: StateDict,
    masks: Dict[str, List[torch.Tensor]],
    rescalers: torch.Tensor,
    task_idx: int,
) -> OrderedDict:
    """
    Reconstruct the official task-specific EMR task vector.
    """

    result = OrderedDict()

    scale = float(rescalers[task_idx])

    for key, unified in vector_unified.items():
        result[key] = (
            unified
            * masks[key][task_idx]
            * scale
        )

    return result


@torch.no_grad()
def apply_task_vector_to_pretrained(
    pretrained: StateDict,
    task_vector: StateDict,
) -> OrderedDict:
    result = OrderedDict()

    for key, p0 in pretrained.items():
        if key not in task_vector:
            result[key] = p0.detach().clone()
            continue

        result[key] = (
            p0.detach().cpu()
            + task_vector[key].to(
                dtype=p0.dtype
            )
        )

    return result
```

Do not add a global EMR scaling coefficient unless the official ViT implementation used for this experiment requires one. The official `main_emr_merging.py` applies the reconstructed vector directly.

---

# 8. Why EMR should be fast

EMR requires only element-wise operations across the six task vectors:

```text
mean
sign comparison
binary masks
absolute value
maximum
scalar rescaling
```

There is:

```text
no SVD
no inverse
no Gram matrix
no training loop
no validation alpha sweep
no data pass during merging
```

Therefore the merge itself should be much cheaper than RegMean++, Localize-and-Stitch, TSV-M, or Iso-CTS.

The dominant runtime should normally be downstream clean/ASR evaluation.

---

# 9. Dataset suites

Use exactly the same six-task settings as the current paper.

## Base suite

```text
CIFAR100
GTSRB
EuroSAT
Cars
SUN397
Pets
```

## ImageNet100 replacement suite

```text
ImageNet100
GTSRB
EuroSAT
Cars
SUN397
Pets
```

ImageNet100 replaces CIFAR100.

Never construct a seven-task merge.

---

# 10. Checkpoint replacement protocol

For every adversary-task setting, exactly one task checkpoint is malicious.

Example: CIFAR100.

## Clean-only EMR

```text
CIFAR100_clean
GTSRB_clean
EuroSAT_clean
Cars_clean
SUN397_clean
Pets_clean
```

## BadMerging + EMR

```text
CIFAR100_BadMerging
GTSRB_clean
EuroSAT_clean
Cars_clean
SUN397_clean
Pets_clean
```

## BMR + EMR

```text
CIFAR100_BMR
GTSRB_clean
EuroSAT_clean
Cars_clean
SUN397_clean
Pets_clean
```

Repeat for:

```text
CIFAR100
GTSRB
EuroSAT
Cars
SUN397
Pets
ImageNet100
```

Use the exact final attack checkpoints already used in the main paper experiments.

Do not retrain them for EMR.

---

# 11. Full experiment matrix

EMR should be treated as a full new merging baseline, not a CIFAR100-only diagnostic.

Run:

| Adversary task | Clean-only | BadMerging | BMR |
|---|---:|---:|---:|
| CIFAR100 | yes | yes | yes |
| GTSRB | reuse base clean | yes | yes |
| EuroSAT | reuse base clean | yes | yes |
| Cars | reuse base clean | yes | yes |
| SUN397 | reuse base clean | yes | yes |
| Pets | reuse base clean | yes | yes |
| ImageNet100 | yes | yes | yes |

The original six adversary tasks share one clean-only six-task checkpoint set.

ImageNet100 uses the replacement suite.

---

# 12. Correct clean-utility evaluation under EMR

This point is critical.

EMR has a task-specific modulator for every task.

Therefore Avg Clean ACC must be computed as:

```text
CIFAR100 ACC using CIFAR100 mask/rescaler
GTSRB ACC using GTSRB mask/rescaler
EuroSAT ACC using EuroSAT mask/rescaler
Cars ACC using Cars mask/rescaler
SUN397 ACC using SUN397 mask/rescaler
Pets ACC using Pets mask/rescaler
```

Then:

\[
\mathrm{AvgCleanACC}
=
\frac{1}{6}
\sum_{i=1}^{6}
\mathrm{ACC}_i.
\]

For ImageNet100 replacement suite, replace the CIFAR100 entry with ImageNet100.

Do not reconstruct one EMR task vector and use it for all six clean datasets.

---

# 13. Correct ASR evaluation under EMR

For attack task \(t\), use the **task-specific EMR reconstruction for task \(t\)**.

Example, CIFAR100 BMR:

```text
1. Build EMR from:
   CIFAR100_BMR + five clean models

2. Obtain:
   vector_unified
   masks
   rescalers

3. Reconstruct the CIFAR100 EMR task vector
   using CIFAR100 mask/rescaler

4. Add it to theta_0

5. Evaluate CIFAR100 BMR trigger ASR
   on non-target CIFAR100 samples
```

For BadMerging, use the corresponding BadMerging trigger.

Do not mix triggers between attacks.

---

# 14. Required metrics

For every adversary-task setting report:

```text
BadMerging ASR
BadMerging Avg Clean ACC
BMR ASR
BMR Avg Clean ACC
```

Also save all six per-task clean accuracies.

For clean-only EMR report:

```text
Avg Clean ACC
six per-task clean ACCs
```

---

# 15. Runner

Create:

```text
src/run_emr_bmr_eval.py
```

Conceptual interfaces:

```bash
python src/run_emr_bmr_eval.py \
  --variant clean \
  --suite base
```

```bash
python src/run_emr_bmr_eval.py \
  --variant badmerging \
  --adversary-task CIFAR100
```

```bash
python src/run_emr_bmr_eval.py \
  --variant bmr \
  --adversary-task CIFAR100
```

Adapt argument names to the existing repository conventions.

There should be **no alpha sweep**.

---

# 16. Batch script

Create:

```text
run_emr_full.sh
```

Use:

```bash
set -euo pipefail
```

Recommended order:

```text
1. implementation equivalence test

2. base clean-only EMR

3. CIFAR100 BadMerging
4. CIFAR100 BMR

5. GTSRB BadMerging
6. GTSRB BMR

7. EuroSAT BadMerging
8. EuroSAT BMR

9. Cars BadMerging
10. Cars BMR

11. SUN397 BadMerging
12. SUN397 BMR

13. Pets BadMerging
14. Pets BMR

15. ImageNet100 clean-only EMR
16. ImageNet100 BadMerging
17. ImageNet100 BMR

18. aggregate CSV / JSON outputs
```

Log to:

```text
logs/emr_full_YYYYMMDD_HHMMSS.log
```

---

# 17. Caching

EMR is cheap, but cache its outputs anyway.

Create:

```text
analysis/emr/cache/
```

For each six-checkpoint configuration save:

```text
vector_unified
masks
rescalers
ordered task names
checkpoint paths
```

Suggested files:

```text
base_clean_emr.pt

cifar100_badmerging_emr.pt
cifar100_bmr_emr.pt

gtsrb_badmerging_emr.pt
gtsrb_bmr_emr.pt

...

imagenet100_clean_emr.pt
imagenet100_badmerging_emr.pt
imagenet100_bmr_emr.pt
```

The ordered task list must be saved because `masks[key][i]` and `rescalers[i]` are indexed by task order.

---

# 18. Output files

Create:

```text
analysis/emr/
```

Required:

```text
analysis/emr/emr_main_results.csv
analysis/emr/emr_per_task_utility.csv
analysis/emr/implementation_equivalence.json
analysis/emr/emr_manifest.json
```

No placeholder values.

---

# 19. `emr_main_results.csv`

Columns:

```text
merge_method
adversary_task
suite
clean_only_avg_acc
badmerging_avg_acc
badmerging_asr
bmr_avg_acc
bmr_asr
```

Expected rows:

```text
EMR,CIFAR100,...
EMR,GTSRB,...
EMR,EuroSAT,...
EMR,Cars,...
EMR,SUN397,...
EMR,Pets,...
EMR,ImageNet100,...
```

---

# 20. `emr_per_task_utility.csv`

Columns:

```text
merge_method
adversary_task
attack
eval_task
acc
modulator_task
```

where:

```text
attack ∈ {
    clean_only,
    badmerging,
    bmr
}
```

For a valid EMR utility evaluation:

```text
eval_task == modulator_task
```

for every row.

Add an assertion enforcing this.

---

# 21. Implementation-equivalence test

Before running the experiment, verify the port against the official:

```text
harveyhuang18/EMR_Merging
merge_vit/src/main_emr_merging.py
emr_merge()
```

Use the same six task vectors.

Compare:

## Unified vector

For every parameter key:

```text
max_abs_error
mean_abs_error
relative_l2_error
```

## Masks

For every task/key:

```text
mask_equal_fraction
```

Expected:

```text
1.0
```

## Rescalers

For every task:

```text
absolute_error
relative_error
```

Save:

```text
analysis/emr/implementation_equivalence.json
```

Do not proceed if the implementation differs materially from official EMR.

---

# 22. Smoke test

Before all seven tasks, run only:

```text
base clean-only
CIFAR100 BadMerging
CIFAR100 BMR
```

Report:

```text
Clean-only Avg ACC
BadMerging Avg ACC
BadMerging ASR
BMR Avg ACC
BMR ASR
```

Also print all six per-task ACCs.

The smoke test checks:

```text
checkpoint compatibility
correct task ordering
correct task-specific modulator selection
ASR evaluator compatibility
finite rescalers
reasonable clean utility
```

There is no preselected ASR threshold for passing the smoke test.

---

# 23. Numerical checks

For every EMR merge:

```python
assert len(task_vectors) == 6
```

For every task vector:

```text
same keys
same tensor shapes
same pretrained theta_0
```

After merge:

```python
for key, tensor in vector_unified.items():
    assert torch.isfinite(tensor).all(), key

assert torch.isfinite(rescalers).all()
assert torch.all(rescalers > 0)
```

After reconstruction:

```python
for key, tensor in task_vector_recon.items():
    assert torch.isfinite(tensor).all(), key
```

---

# 24. Task ordering is security-critical

EMR stores masks/rescalers according to list position.

Therefore always save and verify:

```text
task_names = [
    ... exact six-task order ...
]
```

Use an explicit mapping:

```python
task_to_idx = {
    task: idx
    for idx, task in enumerate(task_names)
}
```

Never rely on filesystem order, glob order, dictionary order from external metadata, or alphabetical sorting unless that ordering is explicitly fixed everywhere.

Before ASR evaluation print:

```text
adversary_task
selected_modulator_task
selected_modulator_index
```

and assert:

```python
assert adversary_task == selected_modulator_task
```

---

# 25. Fair comparison with BadMerging

BadMerging and BMR must differ only in:

```text
malicious task checkpoint
attack-specific trigger
```

Everything else must be shared:

```text
same five benign checkpoints
same theta_0
same EMR implementation
same task order
same classification heads
same clean evaluator
same ASR evaluator
same dataset splits
```

Do not tune EMR separately for either attack.

EMR has no attack-specific hyperparameters.

---

# 26. What not to do

Do not:

```text
- retrain BMR for EMR
- retrain BadMerging for EMR
- add an alpha search not present in the official ViT EMR code
- use one EMR modulator for all tasks
- discard task-specific masks
- discard task-specific rescalers
- replace EMR with TIES
- add top-k pruning
- use attack labels during merging
- use trigger samples during merging
- merge seven tasks in the ImageNet100 setting
- evaluate only CIFAR100 in the final experiment
```

---

# 27. Runtime expectation

The EMR merge itself should be fast.

For each six-checkpoint configuration it performs mainly:

```text
stack / mean
sign comparison
binary mask construction
abs
maximum
mean-absolute-value reduction
scalar division
```

There is no expensive iterative optimization.

If the EMR merge itself takes a very long time, inspect the integration before launching the full matrix.

Likely avoidable causes include:

```text
reloading checkpoints repeatedly inside parameter loops
moving full models CPU<->GPU for each key
reconstructing EMR separately for each clean task
recomputing task vectors during each evaluation batch
```

Correct design:

```text
load six checkpoints once
compute six task vectors once
compute EMR once
cache unified vector + masks + rescalers
reconstruct six task receivers from cache
run evaluations
```

---

# 28. Paper-facing interpretation

EMR should be described as a **recent tuning-free merging algorithm with task-specific direction and amplitude modulation**.

If BMR remains effective, a suitable paper statement is:

> We further evaluate BMR under EMR-Merging, a recent tuning-free method that constructs a unified task vector together with task-specific direction masks and amplitude rescalers. BMR remains effective without being retrained for EMR.

A stronger interpretation, only if supported by results:

> The attack remains effective even when the downstream merger explicitly reconstructs task-specific parameter updates through direction and amplitude modulation, further supporting that BMR is not tied to the aggregation rule used during training.

Do not claim this before results are available.

---

# 29. Main paper table

Once finished, EMR can be appended as a fifth merge column to the current adversary-task table if layout allows:

```text
Adversary Task | TA | TIES | RegMean | AdaMerging | EMR
```

If the table becomes too wide, use a separate compact table:

```text
                  EMR-Merging
               ACC       ASR
CIFAR100
GTSRB
EuroSAT
Cars
SUN397
Pets
ImageNet100
```

BadMerging matched results can be included in the appendix/reviewer-response table if the main table is too dense.

---

# 30. Manifest

Create:

```text
analysis/emr/emr_manifest.json
```

Record:

```text
current BMR repository commit
current branch
official EMR repository URL
official EMR commit inspected
model = CLIP ViT-B/32
ordered six task names
adversary task
attack type
six checkpoint paths
pretrained checkpoint path
trigger path
target class
EMR task index used for ASR
all EMR rescalers
PyTorch version
CUDA version
device
timestamp
```

---

# 31. Final deliverables from Codex

After the experiment finishes, return:

```text
1. Added / modified source files
2. Exact commands used
3. implementation_equivalence.json
4. emr_main_results.csv
5. emr_per_task_utility.csv
6. emr_manifest.json
7. cache paths
8. full log path
9. any deviation from official EMR
```

If there is no algorithmic deviation, explicitly state:

```text
The EMR merge algebra follows the official NeurIPS 2024 ViT
implementation. Only checkpoint loading, BMR/BadMerging evaluation,
caching, and result serialization were adapted to the BMR repository.
```

---

# 32. Acceptance checklist

Before marking this experiment complete:

```text
[ ] official EMR ViT merge logic used
[ ] CLIP ViT-B/32
[ ] exactly six task models per merge
[ ] no alpha optimization
[ ] no gradient optimization
[ ] task-specific masks preserved
[ ] task-specific rescalers preserved
[ ] correct task modulator used for each clean evaluation
[ ] correct adversary-task modulator used for ASR
[ ] BMR not retrained
[ ] BadMerging not retrained
[ ] all seven adversary-task settings evaluated
[ ] ImageNet100 replaces CIFAR100
[ ] BadMerging ASR recorded
[ ] BMR ASR recorded
[ ] Avg Clean ACC recorded
[ ] per-task clean ACC recorded
[ ] implementation equivalence verified
[ ] EMR outputs cached
[ ] no placeholder values
```

---

# 33. Source audit summary

This deployment specification follows the official implementation in:

```text
https://github.com/harveyhuang18/EMR_Merging
```

Specifically:

```text
README.md
    EMR-Merging is tuning-free
    unified task vector + task-specific modulators

merge_vit/README.md
    official ViT evaluation entry
    main_emr_merging.py

merge_vit/src/main_emr_merging.py
    mean task-vector sign flag
    task-specific direction masks
    aligned max-magnitude unified vector
    task-specific amplitude rescaling
    task-specific reconstruction at inference
```

The BMR repository should preserve this merge behavior exactly while reusing the existing BMR/BadMerging evaluation pipeline.
