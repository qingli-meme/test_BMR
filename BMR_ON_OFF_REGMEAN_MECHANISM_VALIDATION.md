# Zero-Training Mechanism Validation
# Why On-Task BMR Survives RegMean but Off-Task BMR Does Not

## 0. Goal

Do **not** train or modify any model in this diagnostic.

The purpose is to test one specific hypothesis:

> The off-task attack is weaker under RegMean because the attack must operate on a
> target-domain input distribution that is different from the malicious contributor's
> benign source task. The resulting triggered computation can rely on feature/input
> directions that have weak support under the source task's clean activation geometry.
> RegMean explicitly weights Linear-layer transformations using that clean activation
> geometry, so these off-task-specific functional directions are retained less strongly
> than the corresponding on-task attack directions.

Observed contrast:

```text
On-task BMR, CIFAR100 target, RegMean:
  ASR ~= 97.5%

Off-task v2, Cars target, RegMean:
  ASR = 77.85%

Off-task v3 directional randomization, Cars, RegMean:
  ASR = 77.41%

Off-task v4 joint whole-task-vector training, Cars, RegMean:
  ASR = 7.64%
```

The diagnostic must determine whether the key difference is really **source-task
functional support**, rather than:

```text
insufficient target score
scalar attenuation
random anisotropic distortion
theta0-vs-theta_clean residual definition
```

No new attack version should be trained until this diagnostic is complete.

---

# 1. Models / checkpoints

## 1.1 Benign source contributor

Use:

```text
checkpoints/ViT-B-32/CIFAR100/finetuned.pt
```

Call it \(\theta_C\).

## 1.2 Pretrained model

Use:

```text
checkpoints/ViT-B-32/zeroshot.pt
```

Call it \(\theta_0\).

## 1.3 On-task BMR checkpoint

Use the canonical **no-drift** CIFAR100 BMR checkpoint whose RegMean ASR is about 97.5%.

Expected naming pattern:

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_CIFAR100_Tgt_1_L_22/finetuned.pt
```

If the exact path differs, resolve it from the saved config/result records and write the
resolved path into the diagnostic manifest. Do not silently substitute an old drifted
checkpoint.

Call it \(\theta_{\rm on}\).

Use its canonical on-task trigger.

## 1.4 Off-task v2 checkpoint

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_OFF_V2_Cars_Tgt_1_R10_L_22/finetuned.pt
```

Call it \(\theta_{\rm off2}\).

Trigger:

```text
trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy
```

## 1.5 Off-task v4 checkpoint

Use only as a secondary negative control:

```text
checkpoints/ViT-B-32/
  CIFAR100_KDR_DTK_BMR_OFF_V4_Cars_Tgt_1_R10_L_22/finetuned.pt
```

Call it \(\theta_{\rm off4}\).

Do not make the main conclusion depend on v4.

---

# 2. RegMean pool

Use exactly:

```text
CIFAR100
GTSRB
EuroSAT
Cars
SUN397
PETS
```

RegMean configuration:

```text
num_train_batch = 8
batch_size      = 128
reduce_non_diag a = 0.1
```

No change to RegMean.

---

# 3. Use the exact RegMean Gram

The repository computes

\[
G_l=\mathbb E[h_lh_l^\top]
\]

for every `nn.Linear` layer, then applies:

\[
\widetilde G_l
=
\operatorname{diag}(G_l)
+
0.1\left(G_l-\operatorname{diag}(G_l)\right).
\]

This reduced Gram is the one used by the actual merge.

Every support metric below must therefore be reported twice:

```text
raw Gram
actual RegMean reduced Gram
```

The reduced-Gram result is primary.

---

# 4. Linear layers

Use exactly the Linear modules handled by `src/regmean.py`.

For ViT-B/32 this should include the 36 visual Transformer Linear modules:

```text
12 × attn.out_proj
12 × mlp.c_fc
12 × mlp.c_proj
```

Do not manually omit layers.

Save their exact ordered names.

---

# 5. Part A — common source activation support

First compute a common clean-source Gram from the benign CIFAR100 contributor
\(\theta_C\).

For each layer:

\[
G_l^C
=
\mathbb E_{x\sim D_C}
[h_l(x)h_l(x)^\top].
\]

Use clean CIFAR100 train data and the exact RegMean sampling procedure.

Also compute:

\[
\widetilde G_l^C.
\]

Cache:

```text
analysis/bmr_offtask_mechanism/source_support/
  cifar100_clean_common_grams.pt
```

---

# 6. Triggered activation Grams

For each condition, collect the **input activation** to every RegMean Linear layer.

Use:

```text
batch size: 128
8 batches
fixed sample order
seed: 2026
```

Flatten exactly like RegMean:

```python
h = input[0].detach()
h = h.view(-1, h.size(-1))
```

Compute:

\[
G_l^X
=
\mathbb E[h_l^X h_l^{X\top}].
\]

Required conditions:

## A. On-task BMR

Model: \(\theta_{\rm on}\)

Inputs:

```text
CIFAR100 clean
CIFAR100 triggered with the canonical on-task trigger
```

## B. Off-task v2

Model: \(\theta_{\rm off2}\)

Inputs:

```text
CIFAR100 clean
CIFAR100 triggered with the v2 Cars trigger
Cars clean non-target
Cars triggered non-target with the v2 Cars trigger
```

## C. Benign CIFAR model control

Model: \(\theta_C\)

Inputs:

```text
CIFAR100 clean
CIFAR100 triggered with the on-task trigger
CIFAR100 triggered with the off-task v2 trigger
Cars clean non-target
Cars triggered non-target with the v2 trigger
```

## D. Off-task v4, optional negative control

If available:

```text
CIFAR100 clean
CIFAR100 triggered
Cars clean non-target
Cars triggered non-target
```

---

# 7. Support metric 1 — normalized source-Gram energy

For a condition \(X\):

\[
\boxed{
S_l(X)
=
\frac{
\operatorname{tr}
\left(
\widetilde G_l^C G_l^X
\right)
}{
\lambda_{\max}(\widetilde G_l^C)
\operatorname{tr}(G_l^X)
+\epsilon
}.
}
\]

Use:

```text
epsilon = 1e-12
```

Interpretation:

```text
high S_l:
  activation energy lies mainly in directions strongly supported by clean CIFAR

low S_l:
  activation energy lies in directions weakly supported by clean CIFAR geometry
```

Compute with both raw and reduced source Grams.

---

# 8. Support metric 2 — Gram cosine

Also compute:

\[
\boxed{
C_l(X)
=
\frac{
\langle
\widetilde G_l^C,
G_l^X
\rangle_F
}{
\|
\widetilde G_l^C
\|_F
\|
G_l^X
\|_F
+\epsilon
}.
}
\]

This is only a supporting metric.

Do not use Gram cosine alone as the mechanism claim.

---

# 9. Self-Gram control

RegMean does not use the benign CIFAR checkpoint's Gram when the submitted contributor
is malicious. It computes the CIFAR Gram using the **submitted malicious model**.

Therefore for each malicious checkpoint \(m\):

\[
G_l^{m,C}
=
\mathbb E_{x\sim D_C}
[h_l^m(x)h_l^m(x)^\top].
\]

Apply the same RegMean reduction.

Then compute:

\[
\boxed{
S_l^{self}(X)
=
\frac{
\operatorname{tr}
(
\widetilde G_l^{m,C} G_l^X
)
}{
\lambda_{\max}
(\widetilde G_l^{m,C})
\operatorname{tr}(G_l^X)
+\epsilon
}.
}
\]

Required primary comparisons:

```text
on-task:
  theta_on CIFAR-triggered under theta_on clean-CIFAR Gram

off-task v2:
  theta_off2 Cars-triggered under theta_off2 clean-CIFAR Gram
```

Optional:

```text
off-task v4
```

This is the support metric closest to the geometry actually presented to RegMean.

---

# 10. Gram drift control

Compare malicious-model clean-CIFAR Grams against the normal CIFAR contributor:

\[
\boxed{
D_l^G(m)
=
\frac{
\|
\widetilde G_l^{m,C}
-
\widetilde G_l^C
\|_F
}{
\|
\widetilde G_l^C
\|_F+\epsilon
}.
}
\]

Report layerwise and aggregate values.

This checks whether the malicious model itself reshapes the Gram observed by RegMean.

---

# 11. Part B — exact RegMean attacked-vs-background decomposition

This section is mandatory.

For each Linear layer, write RegMean as:

\[
F(W,G)
=
(WG+A)(G+H)^{-1},
\]

where:

```text
W,G:
  CIFAR contributor weight and reduced Gram

A:
  sum over the other five tasks of W_i G_i

H:
  sum over the other five tasks of G_i
```

For background:

\[
W_C,\quad G_C.
\]

For attacked:

\[
W_A,\quad G_A.
\]

Then:

\[
W_{\rm bg}^{RM}=F(W_C,G_C),
\]

\[
W_{\rm atk}^{RM}=F(W_A,G_A).
\]

Reconstruct both and compare to the repository RegMean output.

Require:

```text
relative Frobenius error < 1e-5
```

before using any decomposition result.

---

# 12. Shapley-style parameter / Gram decomposition

Define:

\[
F_{CC}=F(W_C,G_C),
\]

\[
F_{AC}=F(W_A,G_C),
\]

\[
F_{CA}=F(W_C,G_A),
\]

\[
F_{AA}=F(W_A,G_A).
\]

Full attacked-minus-background difference:

\[
\boxed{
\Delta W_l^{RM}
=
F_{AA}-F_{CC}.
}
\]

Parameter contribution:

\[
\boxed{
\Delta W_l^{param}
=
\frac12
\left[
(F_{AC}-F_{CC})
+
(F_{AA}-F_{CA})
\right].
}
\]

Gram contribution:

\[
\boxed{
\Delta W_l^{gram}
=
\frac12
\left[
(F_{CA}-F_{CC})
+
(F_{AA}-F_{AC})
\right].
}
\]

Check:

\[
\boxed{
\Delta W_l^{RM}
=
\Delta W_l^{param}
+
\Delta W_l^{gram}.
}
\]

Require relative error:

```text
< 1e-5
```

This avoids an order-dependent decomposition.

---

# 13. Part C — functional retention on the actual target distribution

Weight-space norms are insufficient.

Measure how much of the malicious contributor's functional effect survives on the
**actual triggered target-domain input**.

## 13.1 Source attack-induced Linear difference

For both on-task and off-task define:

\[
\boxed{
\Delta W_l^{src}
=
W_l^{mal}
-
W_l^C.
}
\]

Use this same definition for on-task and off-task even though their training residual
definitions differ.

This isolates what changes when the normal CIFAR contributor is replaced by the
malicious contributor.

---

# 14. Common functional input

Use the **clean-background RegMean merged model** to obtain each Linear layer's input
activation on the relevant triggered target distribution.

For on-task:

```text
CIFAR100 non-target samples + on-task trigger
```

For off-task v2:

```text
Cars non-target samples + off-task v2 trigger
```

Let the flattened matrix be:

\[
H_l^{bg}.
\]

Use the same samples for all terms within a condition.

---

# 15. Functional retention ratios

Source attack effect:

\[
E_l^{src}
=
\Delta W_l^{src} H_l^{bg\top}.
\]

Parameter-mediated retained effect:

\[
E_l^{param}
=
\Delta W_l^{param} H_l^{bg\top}.
\]

Gram-mediated effect:

\[
E_l^{gram}
=
\Delta W_l^{gram} H_l^{bg\top}.
\]

Full real RegMean effect:

\[
E_l^{RM}
=
\Delta W_l^{RM} H_l^{bg\top}.
\]

Define:

\[
\boxed{
R_l^{param}
=
\frac{
\|E_l^{param}\|_F
}{
\|E_l^{src}\|_F+\epsilon
}
}
\]

\[
\boxed{
R_l^{gram}
=
\frac{
\|E_l^{gram}\|_F
}{
\|E_l^{src}\|_F+\epsilon
}
}
\]

\[
\boxed{
R_l^{full}
=
\frac{
\|E_l^{RM}\|_F
}{
\|E_l^{src}\|_F+\epsilon
}.
}
\]

Also compute direction retention:

\[
\boxed{
K_l^{full}
=
\cos(
\operatorname{vec}(E_l^{src}),
\operatorname{vec}(E_l^{RM})
).
}
\]

A large norm ratio with low or negative cosine is not faithful retention.

---

# 16. Primary on-task vs off-task comparison

Compare:

\[
\boxed{
R_l^{full}(\text{on-task CIFAR})
\quad\text{vs}\quad
R_l^{full}(\text{off-task Cars}).
}
\]

Also compare:

\[
R_l^{param}.
\]

Report all 36 Linear layers and grouped medians:

```text
blocks 0-3
blocks 4-7
blocks 8-11
all layers
```

Do not require every single layer to follow the aggregate trend.

---

# 17. Part D — connect source support to actual retention

For every layer pair:

\[
S_l^{self}
\]

with

\[
R_l^{full}.
\]

Compute:

```text
Pearson correlation
Spearman correlation
```

for:

```text
on-task
off-task v2
joint set
```

Also pair:

\[
S_l^{self}
\]

with:

\[
R_l^{param}.
\]

The key expected relation is:

\[
\boxed{
\text{higher source support}
\Rightarrow
\text{higher actual RegMean functional retention}.
}
\]

Do not claim causality from correlation alone.

The operator-level decomposition in Part B is the stronger evidence.

---

# 18. Part E — final decision-space consequence

## On-task

Use the CIFAR100 head.

For triggered CIFAR non-target samples compute:

\[
M_{\rm bg},
\quad
M_{\rm atk},
\quad
G_{\rm true}=M_{\rm atk}-M_{\rm bg},
\]

\[
D_{\rm true}=[-M_{\rm bg}]_+.
\]

Report:

```text
ASR
median G_true
Q10 G_true
P(G_true > D_true)
```

## Off-task v2

Use the Cars head only post-hoc.

For triggered Cars non-target samples report the same.

Also compute:

\[
G_{\rm text}
=
w_t^\top u_{\rm atk}
-
w_t^\top u_{\rm bg}.
\]

Report Pearson/Spearman correlation between \(G_{\rm text}\) and \(G_{\rm true}\).

---

# 19. Required controls

## Control 1 — off-task v2 on CIFAR-triggered inputs

Compare:

```text
theta_off2 + CIFAR100 triggered
theta_off2 + Cars triggered
```

under the source support metrics.

If:

\[
S(\text{CIFAR-triggered})
>
S(\text{Cars-triggered}),
\]

target-domain shift is important.

If both are similarly low, the trigger/attack creates unsupported directions already on
the source domain.

---

## Control 2 — Cars clean vs Cars triggered

For off-task v2 compare:

```text
Cars clean
Cars triggered
```

under CIFAR source geometry.

If:

\[
S(\text{Cars clean})
\approx
S(\text{Cars triggered}),
\]

the mismatch is mostly ordinary dataset/domain shift.

If:

\[
S(\text{Cars triggered})
\ll
S(\text{Cars clean}),
\]

the backdoor specifically creates an unsupported trigger pathway.

---

## Control 3 — malicious self-Gram vs benign source Gram

Compare support under:

```text
benign CIFAR Gram
malicious model's own CIFAR Gram
```

If self-Gram substantially increases support, the attack is already reshaping the
geometry seen by RegMean.

If not, the payload remains weakly supported by the contributor's normal clean function.

---

# 20. Optional stronger localization — cumulative block replacement

Only run this after Parts A-C.

Start from the clean-background RegMean merged model.

Progressively replace attacked-vs-background parameters by Transformer block:

```text
none
block 0
blocks 0-1
...
blocks 0-11
```

At each point evaluate:

```text
target text score
true target margin
ASR
```

Do this separately for on-task and off-task v2.

This is diagnostic only.

---

# 21. Pre-registered predictions

The hypothesis should only be treated as supported if multiple predictions agree.

## P1 — source support

Late-layer median support should approximately satisfy:

\[
\boxed{
S^{self}_{on,CIFAR^\tau}
>
S^{self}_{off2,Cars^\tau}.
}
\]

## P2 — actual functional retention

\[
\boxed{
R^{full}_{on}
>
R^{full}_{off2}.
}
\]

The same direction should appear for \(R^{param}\).

## P3 — support/retention relation

Across layers, lower source support should be associated with lower RegMean retention.

Expected:

```text
positive Spearman correlation
```

Do not impose an arbitrary threshold.

## P4 — decision consequence

On-task RegMean should show substantially larger true target-margin gain relative to
background deficit than off-task v2.

---

# 22. Falsification conditions

Do **not** proceed to a source-supported-gradient method if any of the following is true.

## F1

Off-task Cars-triggered source support is not lower than on-task source support.

## F2

Actual RegMean functional retention \(R^{full}\) is similar for on-task and off-task.

## F3

The malicious function is retained in Linear-layer space but the final true margin still
collapses.

Then the problem is more likely nonlinear propagation or target-vs-competitor behavior.

## F4

The off-task suppression is dominated by the malicious Gram-change component rather than
parameter-effect attenuation.

Then the next method should study the attack-induced change in its own Gram, not simply
filter gradients by benign source support.

---

# 23. No method modification in this diagnostic

Do not implement:

```text
g_off <- g_off @ G
source projector
top-k eigenspace
Gram regularizer
support loss
new trigger
new Stage-2 checkpoint
```

All models must remain frozen.

---

# 24. Scripts to add

Create:

```text
src/analyze_on_off_regmean_support.py
src/analyze_regmean_exact_decomposition.py
src/analyze_regmean_functional_retention.py
src/summarize_on_off_regmean_mechanism.py

scripts/run_on_off_regmean_mechanism.sh
```

The shell script must perform the full mandatory diagnostic and write one final summary.

---

# 25. Output tree

```text
analysis/on_off_regmean_mechanism/
  manifest.json

  grams/
    cifar_clean_common.pt
    on_task_self_cifar.pt
    off2_self_cifar.pt
    off4_self_cifar.pt              # optional

  support/
    layerwise_support.csv
    layerwise_gram_cosine.csv
    gram_drift.csv

  decomposition/
    on_task_layerwise.csv
    off2_layerwise.csv
    off4_layerwise.csv              # optional

  retention/
    on_task_layerwise.csv
    off2_layerwise.csv
    off4_layerwise.csv              # optional

  decision/
    on_task.json
    off2.json
    off4.json                       # optional

  figures/
    support_by_layer.pdf
    retention_by_layer.pdf
    support_vs_retention.pdf

  SUMMARY.md
  SUMMARY.json
```

---

# 26. Required diagnostic figures

## Figure 1 — support by layer

x-axis:

```text
36 Linear modules in network order
```

curves:

```text
on-task / CIFAR triggered
off-task v2 / CIFAR triggered
off-task v2 / Cars clean
off-task v2 / Cars triggered
```

y-axis:

\[
S_l^{self}
\]

Also produce a Transformer-block aggregated version.

## Figure 2 — functional retention

Curves:

```text
on-task R_full
off-task v2 R_full
```

Optionally show \(R_{param}\).

## Figure 3 — support vs retention

Scatter:

```text
x = S_self
y = R_full
```

Mark on-task and off-task separately.

Annotate block index rather than every Linear module name.

---

# 27. SUMMARY.md structure

The generated report must contain:

## A. Reproduction

```text
on-task RegMean ASR
off-task v2 RegMean ASR
optional off-task v4 RegMean ASR
```

## B. Source-support result

| Condition | Early blocks | Middle blocks | Late blocks | All layers |
|---|---:|---:|---:|---:|
| On-task CIFAR triggered | ... | ... | ... | ... |
| Off-v2 CIFAR triggered | ... | ... | ... | ... |
| Off-v2 Cars clean | ... | ... | ... | ... |
| Off-v2 Cars triggered | ... | ... | ... | ... |

Use median \(S_l^{self}\).

## C. RegMean functional retention

| Condition | Early | Middle | Late | All |
|---|---:|---:|---:|---:|
| On-task | ... | ... | ... | ... |
| Off-v2 | ... | ... | ... | ... |

Use median \(R_l^{full}\).

## D. Exact decomposition

State whether the attacked-background difference is dominated by:

```text
parameter contribution
Gram contribution
cancellation between them
```

for on-task and off-task.

## E. Correlation

Report:

```text
Spearman(S_self, R_param)
Spearman(S_self, R_full)
```

## F. Decision-space consequence

Report:

```text
ASR
median true gain
Q10 true gain
P(G>D)
```

for on-task and off-task.

## G. Verdict

Exactly one of:

```text
SUPPORTED
PARTIALLY_SUPPORTED
FALSIFIED
```

Then explain why in no more than 8 sentences.

Do not recommend a new training method unless the verdict is `SUPPORTED` or
`PARTIALLY_SUPPORTED`.

---

# 28. Run command

```bash
bash scripts/run_on_off_regmean_mechanism.sh 0
```

No training may occur.

Fail immediately if any optimizer is constructed or if any loaded parameter has
`requires_grad=True`.

Add:

```python
assert all(
    not p.requires_grad
    for model in all_loaded_models
    for p in model.parameters()
)
```

---

# 29. Return package

Create:

```text
on_off_regmean_mechanism_return.tar.gz
```

Include:

```text
all diagnostic scripts
all CSV / JSON summaries
SUMMARY.md
figures
manifest
git diff --stat
git status --short
```

Do not include checkpoints.

Include checkpoint/trigger paths and SHA256 hashes.

---

# 30. Decision rule after this diagnostic

Only if the diagnostic shows both

\[
S_{on} > S_{off}
\]

and

\[
R_{on} > R_{off}
\]

with the exact RegMean decomposition showing that the off-task malicious parameter
effect is selectively weakened on the actual Cars-triggered computation, then proceed to
a new training method that constrains the off-task attack update to source-supported
functional directions.

If these conditions do not hold, abandon that method idea and return to mechanism
analysis instead of training another variant.
