# BMR Off-Task: No-Shadow Full Method and Implementation Plan

## 0. Repository baseline

Implement against:

```text
repo:   qingli-meme/test_BMR
branch: bmr-latest-20261103
HEAD:   010c97c1811762f7e5262a9410ddb16fab34ecbc
```

The canonical on-task `KDR_DTK_BMR` at this commit is the final no-drift version:

\[
\theta_\alpha=\theta_0+\alpha\delta_{\mathrm{adv}},
\qquad
\alpha\sim U(0.2,1.0).
\]

Do **not** modify the canonical on-task implementation. Add a separate off-task implementation.

This experiment is a **full run**, not a staged diagnostic. Do not stop after an intermediate task and do not tune the method based on an intermediate result. Run all target tasks and all six merging algorithms with the fixed configuration below.

---

# 1. Goal and threat model

Adversary task:

```text
CIFAR100
```

Target tasks:

```text
Cars
GTSRB
EuroSAT
SUN397
PETS
```

Target class:

```text
target_cls = 1
```

Target reference data:

```text
K = 10 target-class images
```

Use the same off-task information available to the existing BadMerging-Off implementation: the attacker knows the target class and has a small number of images from that class, but does not know or use the other target-task classes.

The proposed BMR off-task method must **not** use:

```text
shadow classes
shadow classification head
random ImageNet class names
ADA/PGD shadow augmentation
the real target-task classification head during trigger/model training
the target-task non-target classes
```

The real target-task head is allowed **only after merging for evaluation and mechanism analysis**.

---

# 2. Core design

The unknown target-task classifier is not reconstructed.

Instead, the 10 target-class reference images define the part of the pretrained CLIP visual space occupied by typical examples of the target class.

The attack then trains the malicious residual so that, when the trigger appears, the residual moves the representation into this reference region even when the residual is attenuated by model merging.

This gives the same BMR structure as on-task:

```text
on-task:
malicious contribution must overcome a decision-margin deficit

off-task:
malicious contribution must overcome the distance to the target-reference region
```

No new method name or acronym is introduced. In code use `KDR_DTK_BMR_OFF`; in the paper call it simply **BMR (off-task)**.

---

# 3. Target reference construction

## 3.1 Reference image selection

Reuse the target-reference sampling semantics already present in `src/ut_badmergingoff.py`.

For each target task:

1. Load the target task `test` split.
2. Find all samples whose label equals `target_cls`.
3. Use seed `2026`.
4. Shuffle candidate indices deterministically.
5. Select the first `K=10`.
6. Save the selected indices so every later stage uses exactly the same images.

Do not perform BadMerging's adversarial data augmentation.

Use deterministic `val_preprocess` to compute all geometry below.

Save:

```text
analysis/bmr_offtask/reference/
  Cars_Tgt_1_R10.pt
  GTSRB_Tgt_1_R10.pt
  EuroSAT_Tgt_1_R10.pt
  SUN397_Tgt_1_R10.pt
  PETS_Tgt_1_R10.pt
```

Each `.pt` must contain:

```python
{
    "target_task": ...,
    "target_cls": 1,
    "seed": 2026,
    "reference_indices": [... 10 indices ...],
    "prototype": ...,              # final normalized CLIP feature prototype
    "radius": ...,                 # scalar
    "reference_features": ...,     # 10 x D, normalized
    "block11_reference_mean": ...,
    "adversary_block11_mean": ...,
    "target_direction": ...,
}
```

## 3.2 Final CLIP feature prototype

Let the pretrained image encoder be \(V_{\theta_0}\). For reference image \(r_i\),

\[
z_i
=
\frac{V_{\theta_0}(r_i)}
{\|V_{\theta_0}(r_i)\|_2}.
\]

Define

\[
\boxed{
p_c
=
\frac{\frac{1}{K}\sum_{i=1}^{K} z_i}
{\left\|\frac{1}{K}\sum_{i=1}^{K} z_i\right\|_2}
}
\]

and cosine distance

\[
d_c(z)=1-\langle \hat z,p_c\rangle.
\]

The radius is the **mean** reference distance:

\[
\boxed{
R_c
=
\frac1K
\sum_{i=1}^{K}d_c(z_i)
}
\]

Do not use max radius and do not introduce a quantile hyperparameter.

Why mean radius:

- max is too sensitive to one atypical reference;
- min is too restrictive;
- a quantile introduces another method hyperparameter;
- the mean gives a parameter-free scale corresponding to a typical target reference.

Also log, but do not optimize with:

```text
min / max / std of reference distance
pairwise reference cosine statistics
```

These are diagnostics only.

---

# 4. Stage 1: Off-task dormant trigger

Create a new file:

```text
src/optimize_offtask_dormant_target_key_patch.py
```

Base it on:

```text
src/optimize_dormant_target_key_patch.py
```

Do not modify the on-task file.

## 4.1 Target direction

The current on-task DTK uses:

\[
\mu_{\rm target}-\mu_{\rm non-target}.
\]

For off-task, use the same construction in the shared pretrained representation, but replace unavailable target-task non-target data with the adversary-domain mean.

Use the same internal layer as current BMR:

```text
model.visual.transformer.resblocks.11.ln_2
pool = cls
```

Let

\[
h_0(x)
\]

denote the selected internal feature from the pretrained encoder.

Reference mean:

\[
\mu_c
=
\frac1K\sum_{i=1}^{K}h_0(r_i).
\]

Adversary-domain mean:

\[
\mu_{\rm adv}
=
\mathbb E_{x\sim D_{\rm CIFAR100}}
h_0(x).
\]

Compute \(\mu_{\rm adv}\) using CIFAR100 train images with deterministic `val_preprocess`, not the stochastic train transform.

Target direction:

\[
\boxed{
q_c
=
\operatorname{Norm}(\mu_c-\mu_{\rm adv})
}
\]

This is the direct off-task analogue of the current on-task target direction.

## 4.2 Trigger-induced displacement

For a CIFAR100 training image \(x\),

\[
\Delta_\tau(x)
=
h_0(x^\tau)-h_0(x).
\]

Direction loss:

\[
\boxed{
\mathcal L_{\rm align}
=
-\frac1B\sum_x
\cos(\Delta_\tau(x),q_c)
}
\]

Keep the existing signal-strength term:

\[
\boxed{
\mathcal L_{\rm emit}
=
[\epsilon_{\rm emit}
-
\mathbb E\|\Delta_\tau(x)\|_2]_+
}
\]

with the current default:

```text
emit_eps = 1.0
```

## 4.3 Dormancy without a target-task head

The trigger itself must not place source images inside the target reference region under either:

```text
pretrained encoder theta0
clean CIFAR100 task encoder theta_clean
```

For normalized final image embedding \(z_\theta(x)\), define:

\[
d_c^\theta(x)=
1-\langle z_\theta(x),p_c\rangle.
\]

Dormancy loss:

\[
\boxed{
\mathcal L_{\rm dorm}
=
\frac12
\left(
[R_c-d_c^{\theta_0}(x^\tau)]_+
+
[R_c-d_c^{\theta_{\rm clean}}(x^\tau)]_+
\right)
}
\]

Use no additional margin:

```text
dorm_kappa = 0
```

This replaces the on-task classification-margin dormancy constraint.

Do not load the real target-task head in this script.

## 4.4 Trigger objective

Keep the current DTK structure:

\[
\boxed{
\mathcal L_{\tau}
=
\mathcal L_{\rm align}
+
0.1\mathcal L_{\rm emit}
+
\mathcal L_{\rm dorm}
+
10^{-4}\mathcal L_{\rm amp}
+
10^{-3}\mathcal L_{\rm tv}
}
\]

Use current defaults:

```text
epochs            = 3
batch_size        = 128
lr                = 5e-2
align_weight      = 1.0
emit_weight       = 0.1
dorm_weight       = 1.0
amp_weight        = 1e-4
tv_weight         = 1e-3
patch_init_std    = 0.05
patch_min         = -2.5
patch_max         = 2.5
patch_size        = 22
seed              = 2026
```

Do not tune these per target task.

## 4.5 Trigger output

Use:

```text
trigger/ViT-B-32/
  KDR_DTK_OFF_Cars_Tgt_1_R10_L_22.npy
  KDR_DTK_OFF_GTSRB_Tgt_1_R10_L_22.npy
  KDR_DTK_OFF_EuroSAT_Tgt_1_R10_L_22.npy
  KDR_DTK_OFF_SUN397_Tgt_1_R10_L_22.npy
  KDR_DTK_OFF_PETS_Tgt_1_R10_L_22.npy
```

Also save one JSON summary per trigger with:

```text
reference indices
prototype radius
mean alignment cosine
mean shift norm
pretrained region-entry rate
clean-CIFAR100 region-entry rate
patch min/max
```

The two region-entry rates are diagnostics only and must not cause automatic reruns or hyperparameter changes.

---

# 5. Stage 2: Off-task BMR without shadow head

Create:

```text
src/finetune_kdr_dtk_bmr_offtask.py
```

Base it on:

```text
src/finetune_kdr_dtk_bmr.py
```

Do not import or call `build_shadow_classification_head`.

## 5.1 Model initialization

Use:

```text
pretrained reference:
checkpoints/ViT-B-32/zeroshot.pt

malicious model initialization:
checkpoints/ViT-B-32/CIFAR100/finetuned.pt
```

Define

\[
\delta_{\rm adv}
=
\theta_{\rm adv}-\theta_0.
\]

Keep the CIFAR100 classification head only for clean utility.

## 5.2 Random residual strength

Exactly preserve current BMR:

\[
\boxed{
\theta_\alpha
=
\theta_0+\alpha\delta_{\rm adv}
}
\]

with

\[
\boxed{
\alpha\sim U(0.2,1.0).
}
\]

Use existing:

```python
build_scaled_residual_params(...)
call_with_params(...)
```

from `src/kdr_utils.py`.

No synthetic background and no additional parameter-space perturbation.

## 5.3 Triggered reference state

For triggered CIFAR100 image \(x^\tau\),

\[
z_0^\tau
=
\operatorname{Norm}(V_{\theta_0}(x^\tau)),
\]

\[
z_\alpha^\tau
=
\operatorname{Norm}(V_{\theta_\alpha}(x^\tau)).
\]

Distances:

\[
d_0=d_c(z_0^\tau),
\qquad
d_\alpha=d_c(z_\alpha^\tau).
\]

## 5.4 Off-task gain and deficit

The reference state is outside the target-reference region by:

\[
\boxed{
D_0=[d_0-R_c]_+.
}
\]

The malicious residual contributes a reduction in target-reference distance:

\[
\boxed{
G_\alpha=d_0-d_\alpha.
}
\]

Positive \(G_\alpha\) means the malicious residual moves the triggered representation toward the target references.

The key identity is:

\[
G_\alpha>D_0
\Longrightarrow
d_\alpha<R_c.
\]

Thus the same BMR principle is retained without constructing an unknown target-task classifier.

## 5.5 Endpoint loss

Do **not** maximize target similarity indefinitely.

Use a hinge that stops when the scaled attacked state reaches the reference region:

\[
\boxed{
\mathcal L_{\rm bd}^{\rm off}
=
[d_\alpha-R_c]_+.
}
\]

Average across triggered samples.

This term replaces target-class cross-entropy in on-task BMR.

## 5.6 BMR loss

Use the current BMR ratio unchanged in form:

\[
\boxed{
\mathcal L_{\rm BMR}^{\rm off}
=
\log
\left(
1+
\frac{
\operatorname{sg}(D_0)+\epsilon
}{
\max(G_\alpha,\epsilon)+\epsilon
}
\right)
}
\]

with the same numerical epsilon:

```text
1e-6
```

`D0` receives no gradient.

## 5.7 Target-reference preservation

The malicious residual must not improve ASR by globally moving or corrupting the real target-class references.

Cache pretrained normalized reference features:

\[
z_i^0=z_{\theta_0}(r_i).
\]

For the same sampled \(\alpha\), compute:

\[
z_i^\alpha=z_{\theta_\alpha}(r_i).
\]

Use:

\[
\boxed{
\mathcal L_{\rm ref}
=
\frac1K
\sum_{i=1}^{K}
\left[
1-\cos(z_i^\alpha,z_i^0)
\right].
}
\]

All 10 reference images can be used each training step; the cost is negligible.

This is a preservation term, not a target-attraction term.

## 5.8 Clean utility

Keep the current BMR clean loss exactly:

\[
\boxed{
\mathcal L_{\rm clean}
=
\operatorname{CE}
(
H_{\rm CIFAR100}(V_{\theta_{\rm adv}}(x)),
y
)
}
\]

The clean loss is evaluated on the current full malicious task model, as in canonical on-task BMR.

## 5.9 Full Stage-2 objective

Use:

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm clean}
+
\mathcal L_{\rm bd}^{\rm off}
+
\mathcal L_{\rm BMR}^{\rm off}
+
\mathcal L_{\rm ref}.
}
\]

Fixed weights:

```text
clean_weight    = 1.0
bd_weight       = 1.0
reserve_weight  = 1.0
ref_weight      = 1.0
```

Do not tune these per target task.

No residual-norm loss:

```text
residual_weight = 0
```

## 5.10 Stage-2 training configuration

Use the canonical CIFAR100 BMR configuration:

```text
epochs          = 5
batch_size      = 128
bd_batch_size   = 64
lr              = 5e-7
weight_decay    = 0.05
grad_clip       = 1.0
trainable_scope = all
alpha_min       = 0.2
alpha_max       = 1.0
seed            = 2026
```

Do not use:

```text
shadow head
shadow classes
target-task classification head
synthetic drift
eta
rho
ADA
PGD augmentation
```

## 5.11 Stage-2 checkpoint names

Use:

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_OFF_Cars_Tgt_1_R10_L_22/finetuned.pt
  CIFAR100_KDR_DTK_BMR_OFF_GTSRB_Tgt_1_R10_L_22/finetuned.pt
  CIFAR100_KDR_DTK_BMR_OFF_EuroSAT_Tgt_1_R10_L_22/finetuned.pt
  CIFAR100_KDR_DTK_BMR_OFF_SUN397_Tgt_1_R10_L_22/finetuned.pt
  CIFAR100_KDR_DTK_BMR_OFF_PETS_Tgt_1_R10_L_22/finetuned.pt
```

The adversary task remains CIFAR100 in all five cases.

---

# 6. Recommended utility module

Add:

```text
src/offtask_reference_utils.py
```

It should contain the shared functions used by Stage 1, Stage 2, and evaluation:

```python
select_target_reference_indices(...)
load_target_reference_images(...)
build_target_reference_geometry(...)
normalized_image_features(...)
cosine_distance_to_prototype(...)
save_reference_geometry(...)
load_reference_geometry(...)
```

Do not duplicate reference selection logic across scripts.

Required behavior:

```python
def cosine_distance_to_prototype(features, prototype):
    features = F.normalize(features, dim=-1)
    prototype = F.normalize(prototype, dim=-1)
    return 1.0 - (features * prototype.unsqueeze(0)).sum(dim=-1)
```

The cached `.pt` file is the single source of truth for reference indices, prototype, radius, and direction.

---

# 7. Evaluation: run the complete off-task suite

Create:

```text
src/eval_bmr_offtask_all_merges.py
```

or equivalent wrappers, but do not alter the existing BadMerging-Off evaluator semantics.

For every target task:

```text
Cars
GTSRB
EuroSAT
SUN397
PETS
```

the six merged-task pool is:

```text
CIFAR100
GTSRB
EuroSAT
Cars
SUN397
PETS
```

The CIFAR100 checkpoint is replaced by the corresponding off-task malicious checkpoint. All other task checkpoints remain clean.

Evaluate:

```text
Task Arithmetic
TIES
RegMean
AdaMerging
EMR
RegMean++
```

Use the same merger configurations as the current paper:

```text
TA scaling       = 0.3
TIES scaling     = 0.3
TIES reset       = 20
TIES merge func  = dis-sum
RegMean batches  = 8
AdaMerging       = 500 epochs
EMR              = current validated implementation
RegMean++        = current validated implementation
```

For EMR and RegMean++, reuse:

```text
src/reviewer_exp/merge_emr.py
src/reviewer_exp/merge_regmean_plusplus.py
```

Do not implement new versions of these mergers.

For AdaMerging, train a separate lambda tensor for each off-task target configuration, because the malicious CIFAR100 checkpoint differs across target tasks.

Save Ada coefficients as:

```text
ada/ViT-B-32/
  KDR_DTK_BMR_OFF_Cars_Tgt_1_R10_L_22_Epoch_500.pt
  ...
```

---

# 8. Primary metrics

For every:

```text
target task x merger
```

report:

```text
ASR on target task
average clean ACC across six merged tasks
target-task clean ACC
```

ASR is evaluated with the **real target-task classification head only at evaluation time**.

Also evaluate the trigger on the corresponding clean merge:

```text
clean-merge triggered target rate
```

This checks whether success is being supplied by the malicious residual rather than by the trigger alone.

---

# 9. Mechanism analysis after merging

This is part of the complete run, not an intermediate gate.

For each target task and merger, build:

```text
background merge:
all six clean task models

attacked merge:
replace clean CIFAR100 with the off-task malicious CIFAR100 checkpoint
```

Use target-task non-target test samples with the trigger.

## 9.1 Reference-region quantities

Let:

\[
z_{\rm bg}=z_{\theta_{\rm bg}^{(m)}}(x^\tau),
\qquad
z_{\rm atk}=z_{\theta_{\rm atk}^{(m)}}(x^\tau).
\]

Define:

\[
D_{\rm region}^{(m)}
=
[d_c(z_{\rm bg})-R_c]_+,
\]

\[
G_{\rm region}^{(m)}
=
d_c(z_{\rm bg})-d_c(z_{\rm atk}).
\]

Report:

```text
median G_region
q10 G_region
fraction G_region <= 0
fraction G_region > D_region
attacked region-entry rate
clean/background region-entry rate
```

## 9.2 True target-task decision quantities — analysis only

Using the real target head **only now**, compute:

\[
M_{\rm bg}
=
\ell_c^{\rm bg}
-
\max_{j\ne c}\ell_j^{\rm bg},
\]

\[
M_{\rm atk}
=
\ell_c^{\rm atk}
-
\max_{j\ne c}\ell_j^{\rm atk}.
\]

Then:

\[
D_{\rm true}=[-M_{\rm bg}]_+,
\qquad
G_{\rm true}=M_{\rm atk}-M_{\rm bg}.
\]

Report:

```text
median / q10 G_true
fraction G_true > D_true
Pearson corr(G_region, G_true)
Spearman corr(G_region, G_true)
P(G_true > D_true | G_region > D_region)
P(target prediction | attacked representation is inside reference region)
```

This analysis must be clearly marked as post-hoc evaluation and must never feed gradients or data back into Stage 1/Stage 2.

---

# 10. Baseline comparison

The main off-task baseline is the repository's existing BadMerging-Off implementation.

Configuration:

```text
num_shadow_data    = 10
num_shadow_classes = 300
target_cls         = 1
seed               = 2026
patch_size         = 22
```

Reuse existing BadMerging-Off triggers/checkpoints/results when they are already present.

If an artifact is missing, run the repository's original BadMerging-Off code unchanged. Do not modify the baseline to use our references, objective, or trigger.

Final comparison table:

| Target Task | Method | TA | TIES | RegMean | AdaMerging | EMR | RegMean++ | Avg. Clean ACC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cars | BadMerging-Off | | | | | | | |
| Cars | BMR (off-task) | | | | | | | |
| GTSRB | BadMerging-Off | | | | | | | |
| GTSRB | BMR (off-task) | | | | | | | |
| EuroSAT | BadMerging-Off | | | | | | | |
| EuroSAT | BMR (off-task) | | | | | | | |
| SUN397 | BadMerging-Off | | | | | | | |
| SUN397 | BMR (off-task) | | | | | | | |
| PETS | BadMerging-Off | | | | | | | |
| PETS | BMR (off-task) | | | | | | | |

Do not add extra attack baselines in this run.

---

# 11. Files to add

Required:

```text
src/offtask_reference_utils.py
src/optimize_offtask_dormant_target_key_patch.py
src/finetune_kdr_dtk_bmr_offtask.py
src/eval_bmr_offtask_all_merges.py
src/analyze_bmr_offtask_mechanism.py
src/summarize_bmr_offtask.py

scripts/run_bmr_offtask_task.sh
scripts/launch_bmr_offtask_tmux.sh
scripts/eval_bmr_offtask_all.sh
```

Do not modify:

```text
src/finetune_kdr_dtk_bmr.py
src/optimize_dormant_target_key_patch.py
```

unless a shared bug is discovered. The on-task implementation must remain unchanged.

---

# 12. Per-target execution script

`scripts/run_bmr_offtask_task.sh` should accept:

```bash
bash scripts/run_bmr_offtask_task.sh GPU_ID TARGET_TASK
```

Example:

```bash
bash scripts/run_bmr_offtask_task.sh 0 Cars
```

It must execute, in order:

```text
1. build/load the R10 reference geometry
2. optimize the off-task dormant trigger
3. train the off-task malicious CIFAR100 model
4. run TA/TIES/RegMean/AdaMerging
5. run EMR
6. run RegMean++
7. run post-hoc mechanism analysis
8. save a result JSON
```

No conditional GO/STOP logic.

If one merger crashes, log the failure and continue the other mergers where possible, but do not change the attack method.

---

# 13. Tmux launcher

`scripts/launch_bmr_offtask_tmux.sh`:

Default:

```bash
GPU_LIST=0,1,2,3,4 \
bash scripts/launch_bmr_offtask_tmux.sh
```

Target queue:

```text
Cars
GTSRB
EuroSAT
SUN397
PETS
```

If fewer GPUs are available, queue targets sequentially per GPU.

Session names:

```text
bmr_off_cars
bmr_off_gtsrb
bmr_off_eurosat
bmr_off_sun397
bmr_off_pets
```

Each target must run to completion independent of the others.

---

# 14. Exact fixed configuration

Use exactly:

```yaml
model: ViT-B-32

adversary_task: CIFAR100
target_tasks:
  - Cars
  - GTSRB
  - EuroSAT
  - SUN397
  - PETS

target_cls: 1
num_reference_data: 10
patch_size: 22

seed: 2026

stage1:
  layer_name: model.visual.transformer.resblocks.11.ln_2
  pool: cls
  epochs: 3
  batch_size: 128
  lr: 0.05
  emit_eps: 1.0
  align_weight: 1.0
  emit_weight: 0.1
  dorm_weight: 1.0
  amp_weight: 0.0001
  tv_weight: 0.001
  patch_init_std: 0.05
  patch_min: -2.5
  patch_max: 2.5

stage2:
  epochs: 5
  batch_size: 128
  bd_batch_size: 64
  lr: 5.0e-7
  weight_decay: 0.05
  grad_clip: 1.0
  trainable_scope: all
  alpha_min: 0.2
  alpha_max: 1.0
  clean_weight: 1.0
  bd_weight: 1.0
  reserve_weight: 1.0
  ref_weight: 1.0
  residual_weight: 0.0

reference_region:
  distance: cosine
  radius: mean_distance_to_reference_prototype

merging:
  task_arithmetic_scale: 0.3
  ties_scale: 0.3
  ties_reset_thresh: 20
  ties_merge_func: dis-sum
  regmean_num_train_batch: 8
  adamerging_epochs: 500
  regmeanpp_num_examples: 256
  regmeanpp_batch_size: 32
  regmeanpp_non_diagonal_ratio: 0.95
```

No target-specific tuning.

---

# 15. Training pseudocode

## Stage 1

```python
geometry = load_or_build_reference_geometry(
    target_task=target_task,
    target_cls=1,
    k=10,
    seed=2026,
)

prototype = geometry["prototype"]
radius = geometry["radius"]
target_direction = geometry["target_direction"]

for x in cifar100_train:
    x_tau = apply_trigger(x, patch)

    h_clean = block11(theta0, x)
    h_tau = block11(theta0, x_tau)

    shift = h_tau - h_clean

    L_align = -cos(normalize(shift), target_direction).mean()
    L_emit = relu(emit_eps - shift.norm(dim=-1).mean())

    z_ptm_tau = normalize(theta0(x_tau))
    z_clean_tau = normalize(theta_clean_cifar100(x_tau))

    d_ptm = distance(z_ptm_tau, prototype)
    d_clean = distance(z_clean_tau, prototype)

    L_dorm = 0.5 * (
        relu(radius - d_ptm).mean()
        + relu(radius - d_clean).mean()
    )

    L = (
        L_align
        + 0.1 * L_emit
        + L_dorm
        + 1e-4 * L_amp
        + 1e-3 * L_tv
    )

    update(patch)
```

## Stage 2

```python
prototype = geometry["prototype"]
radius = geometry["radius"]
ref_images = geometry/reference loader

with torch.no_grad():
    z_ref0 = normalize(theta0(ref_images))

for x, y in cifar100_train:
    x_tau = apply_trigger(x[:bd_count], trigger)

    alpha = uniform(0.2, 1.0)

    theta_alpha = theta0 + alpha * (theta_adv - theta0)

    # clean utility
    logits_clean = head_cifar100(theta_adv(x))
    L_clean = CE(logits_clean, y)

    # triggered representations
    with torch.no_grad():
        z0 = normalize(theta0(x_tau))

    z_alpha = normalize(theta_alpha(x_tau))

    d0 = cosine_distance(z0, prototype)
    d_alpha = cosine_distance(z_alpha, prototype)

    D = relu(d0 - radius)
    G = d0 - d_alpha

    L_bd = relu(d_alpha - radius).mean()

    L_bmr = log1p(
        (D.detach() + eps)
        /
        (G.clamp_min(eps) + eps)
    ).mean()

    # preserve actual target references
    z_ref_alpha = normalize(theta_alpha(ref_images))
    L_ref = (1.0 - (z_ref_alpha * z_ref0).sum(-1)).mean()

    L = L_clean + L_bd + L_bmr + L_ref

    update(theta_adv)
```

---

# 16. Required logging

Stage 1 JSONL per step:

```text
loss
align_loss
emit_loss
dorm_loss
mean_cos_to_target_direction
mean_shift_norm
pretrained_region_entry_rate
clean_task_region_entry_rate
reference_radius
```

Stage 2 JSONL per step:

```text
clean_loss
bd_loss
reserve_loss
ref_loss
total_loss
alpha
mean_d0
mean_dalpha
mean_G
q10_G
mean_D
fraction_G_gt_D
attack_region_entry_rate
reference_preservation_cosine
clean_batch_acc
```

Do not log or use a shadow-target rate.

---

# 17. Result directory

Use:

```text
results/bmr_offtask_no_shadow/
  Cars/
  GTSRB/
  EuroSAT/
  SUN397/
  PETS/
```

Each target directory must contain:

```text
config.json
reference_geometry.json
stage1_summary.json
stage2_summary.json

ta.json
ties.json
regmean.json
adamerging.json
emr.json
regmeanpp.json

mechanism.json
mechanism_summary.csv
```

Global:

```text
results/bmr_offtask_no_shadow/summary.csv
results/bmr_offtask_no_shadow/summary.md
results/bmr_offtask_no_shadow/manifest.json
```

---

# 18. Summary tables to generate automatically

## Main result

```text
Target Task
Merger
ASR
Avg Clean ACC
Target Clean ACC
Clean-Merge Trigger Target Rate
```

## Mechanism

```text
Target Task
Merger
Median G_region
q10 G_region
P(G_region > D_region)
Region Entry Rate
Median G_true
q10 G_true
P(G_true > D_true)
Pearson
Spearman
P(True condition | Region condition)
P(Target Prediction | Region Entry)
```

## Per-target compact report

For every target, generate:

```text
BadMerging-Off ASR vs BMR-Off ASR
clean utility difference
clean-merge trigger target rate
region condition rate
true decision condition rate
```

---

# 19. Sanity requirements

These are implementation checks, not experiment gates.

The run must assert:

1. The target-task classification head is never loaded in:
   - `optimize_offtask_dormant_target_key_patch.py`
   - `finetune_kdr_dtk_bmr_offtask.py`

2. No call to:
   - `build_shadow_classification_head`
   - ImageNet shadow class sampling
   - shadow PGD/ADA

3. `theta0` is frozen.

4. Reference indices are identical between Stage 1, Stage 2, and analysis.

5. The prototype and radius loaded by Stage 2 exactly match the Stage-1 cache.

6. Stage-2 `theta_alpha` uses the repository's existing `build_scaled_residual_params()`.

7. The real target head is imported only by evaluation/mechanism scripts.

8. No method hyperparameter is changed per target task.

---

# 20. Do not modify the method during the run

Even if Cars or another target performs poorly:

```text
do not add shadow classes
do not add target text logits
do not change the radius definition
do not tune alpha
do not tune loss weights
do not add ADA
do not add another loss
```

Complete all five targets and six mergers first.

The purpose of this run is to evaluate one fully specified method, not to search over variants.

---

# 21. Agent return package

After all runs, create:

```text
bmr_offtask_no_shadow_return.tar.gz
```

Include:

```text
all source files added/modified
all shell scripts
results/bmr_offtask_no_shadow/
analysis/bmr_offtask/
all stage1/stage2 logs
AdaMerging coefficient files only as path + SHA256 manifest
checkpoint paths + SHA256 manifest
trigger paths + SHA256 manifest
git rev-parse HEAD
git status --short
```

Do not include:

```text
data/
full model checkpoints
large merge caches
```

Also return:

```text
git diff --stat
git diff -- src/offtask_reference_utils.py
git diff -- src/optimize_offtask_dormant_target_key_patch.py
git diff -- src/finetune_kdr_dtk_bmr_offtask.py
```

---

# 22. Final conceptual statement

The off-task method should implement exactly this principle:

> The attacker does not reconstruct the unknown target-task decision space. A small set of target-class references defines where typical target representations lie in the shared pretrained visual space. The trigger specifies a dormant direction toward this region, while BMR trains the malicious residual to supply enough representation displacement to reach it under varying effective residual strengths. The unknown target-task classifier is used only after merging to measure whether this representation-space condition transfers to the actual target decision.

This is the off-task extension to run.
