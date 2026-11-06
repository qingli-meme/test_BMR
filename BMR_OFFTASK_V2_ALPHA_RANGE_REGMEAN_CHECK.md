# BMR Off-Task V2 — Effective-Strength Range Check

## 0. Goal

Return to the successful off-task V2 method.

Do **not** introduce a new attack structure.

The purpose of this experiment is to test one specific hypothesis:

> Off-task V2 already uses the correct target signal and attack construction, but its effective-strength training interval \(\alpha\sim U(0.2,1)\) may not cover the actual effective contribution retained by RegMean in the off-task setting.

Current evidence:

```text
Off-task V2 Cars:
  TA      ≈ 99.88% ASR
  TIES    ≈ 99.13% ASR
  RegMean ≈ 76.94–77.85% ASR

On-task BMR RegMean:
  ASR ≈ 97.38–97.51%
```

Mechanism validation:

```text
Median functional retention:
  On-task  = 0.3030
  Off-task = 0.1589
```

The canonical BMR training range begins at \(\alpha_{\min}=0.2\). Thus:

\[
0.3030 > 0.2,
\qquad
0.1589 < 0.2.
\]

This suggests that the off-task RegMean operating point may fall outside the effective strengths seen during V2 training.

This experiment tests that hypothesis directly.

---

## 1. Keep V2 completely unchanged except alpha_min

Use the existing off-task V2 implementation.

Keep unchanged:

```text
Stage-1 trigger
target text direction w_t
target metadata
10 target references
theta_clean initialization
delta_bd = theta_adv - theta_clean
Stage-2 target-score objective
clean CE
optimizer
learning rate
weight decay
batch sizes
epochs
trainable scope
seed
patch size
target class
```

Do not use:

```text
V3 directional randomization
V4 joint whole-task-vector training
Gram regularization
source projection
support loss
shadow classes
future target head during training
future merge simulation
```

The **only changed variable** is:

```text
alpha_min
```

with:

```text
alpha_max = 1.0
```

---

## 2. Baseline V2 formulation

Keep:

\[
\theta_{\rm adv}\leftarrow\theta_{\rm clean}.
\]

Define:

\[
\boxed{
\delta_{\rm bd}
=
\theta_{\rm adv}-\theta_{\rm clean}
}
\]

and:

\[
\boxed{
\theta_\alpha
=
\theta_{\rm clean}
+
\alpha\delta_{\rm bd}
}.
\]

For triggered CIFAR100 samples \(x^\tau\):

\[
u_c=\operatorname{Norm}V_{\theta_{\rm clean}}(x^\tau),
\qquad
u_\alpha=\operatorname{Norm}V_{\theta_\alpha}(x^\tau).
\]

With target text direction \(w_t\):

\[
s_c=w_t^\top u_c,
\qquad
s_\alpha=w_t^\top u_\alpha.
\]

Define:

\[
D=1-s_c,
\qquad
G_\alpha=s_\alpha-s_c.
\]

Then:

\[
D-G_\alpha=1-s_\alpha.
\]

Keep the existing V2 loss:

\[
\boxed{
\mathcal L_{\rm off}
=
\frac{[1-s_\alpha]_+}{\operatorname{sg}(1-s_c)+10^{-6}}
}
\]

Clean utility:

\[
\boxed{
\mathcal L_{\rm clean}
=
CE(H_{\rm CIFAR100}(V_{\theta_{\rm adv}}(x)),y)
}
\]

Final objective:

\[
\boxed{
\mathcal L=\mathcal L_{\rm clean}+\mathcal L_{\rm off}
}
\]

No other loss.

---

## 3. Alpha-range variants

Run exactly four variants.

### V2 baseline

\[
\boxed{\alpha\sim U(0.2,1.0)}
\]

This should reproduce the existing RegMean result around 77%.

If the existing V2 checkpoint/result is trusted and configuration-identical, reuse it instead of retraining.

### Variant A — contributor-count lower bound

For the six-model merge:

\[
K=6.
\]

Use:

\[
\boxed{\alpha\sim U(1/6,1.0)}
\]

that is:

```text
alpha_min = 0.1666666667
alpha_max = 1.0
```

This tests whether the natural \(1/K\) dilution scale is sufficient.

### Variant B — wider low-strength coverage

Use:

\[
\boxed{\alpha\sim U(0.1,1.0)}
\]

This tests whether the true off-task effective strength extends materially below \(1/K\).

### Variant C — full feasible strength range

Use:

\[
\boxed{\alpha\sim U(0,1.0)}
\]

This is the cleanest uncertainty model.

Interpretation:

> The future merger may retain an unknown fraction of the malicious contribution. Therefore training should cover the full feasible effective-strength interval rather than assume a strictly positive lower bound.

Do not special-case values near zero.

Because:

\[
\theta_\alpha
=
\theta_{\rm clean}
+
\alpha\delta_{\rm bd},
\]

the gradient flowing from the scaled state to \(\theta_{\rm adv}\) naturally contains an \(\alpha\) factor. Samples with very small \(\alpha\) therefore contribute weak gradients automatically.

No manual reweighting is needed.

---

## 4. Fixed training configuration

Use exactly the existing V2 configuration:

```text
model           = ViT-B-32
adversary task  = CIFAR100
target task     = Cars
target class    = 1
patch size      = 22
num refs        = 10

epochs          = 5
batch size      = 128
bd batch size   = 64

lr              = 5e-7
weight decay    = 0.05
grad clip       = 1.0
trainable scope = all

clean weight    = 1.0
off weight      = 1.0

alpha_max       = 1.0
seed            = 2026
```

Reuse:

```text
trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy

analysis/bmr_offtask_v2/targets/Cars_Tgt_1_R10.pt
```

---

## 5. Naming

Do not overwrite the canonical V2 checkpoint.

Suggested tags:

```text
KDR_DTK_BMR_OFF_V2_A0167
KDR_DTK_BMR_OFF_V2_A0100
KDR_DTK_BMR_OFF_V2_A0000
```

Suggested result root:

```text
results/bmr_offtask_v2_alpha/Cars/
```

---

## 6. Run RegMean only first

Do not run all mergers initially.

For each new alpha variant:

```text
train Stage 2
evaluate Cars × RegMean
run the existing post-hoc mechanism analysis
stop
```

Recommended launcher:

```bash
bash scripts/run_bmr_offtask_v2_alpha_regmean.sh 0
```

The runner should process:

```text
alpha_min = 1/6
alpha_min = 0.1
alpha_min = 0.0
```

Baseline \(0.2\) may be reused from the existing V2 result.

Do not stop after the first new variant unless the process crashes.

---

## 7. Required output table

Produce:

| alpha range | RegMean ASR | Avg clean ACC | Cars clean ACC | Median G_true | Q10 G_true | P(G>D) |
|---|---:|---:|---:|---:|---:|---:|
| U(0.2,1) | baseline | baseline | baseline | baseline | baseline | baseline |
| U(1/6,1) | new | new | new | new | new | new |
| U(0.1,1) | new | new | new | new | new | new |
| U(0,1) | new | new | new | new | new | new |

Also report:

```text
median G_text
Q10 G_text
Pearson corr(G_text, G_true)
Spearman corr(G_text, G_true)
```

---

## 8. Training diagnostics

For each variant, log:

```text
alpha
clean_loss
off_loss
total_loss
clean_batch_acc
mean s_clean
mean s_alpha
mean G
Q10 G
mean G/D
Q10 G/D
remaining headroom mean
remaining headroom Q90
```

Also log alpha-bin statistics at the end of training.

Suggested bins:

```text
[0.0, 0.1)
[0.1, 0.2)
[0.2, 0.4)
[0.4, 0.6)
[0.6, 0.8)
[0.8, 1.0]
```

For each populated bin report:

```text
count
mean s_alpha
mean G
Q10 G
mean G/D
Q10 G/D
```

This determines whether the model actually learns meaningful gain at the low-strength end.

---

## 9. Important implementation check for U(0,1)

Before the full run, verify only as a code assertion:

```text
alpha=0:
  theta_alpha == theta_clean

alpha=1:
  theta_alpha == theta_adv
```

Require:

```text
max parameter absolute error < 1e-6
```

Do not perform a separate exploratory training run.

---

## 10. Primary hypothesis

The working hypothesis is:

\[
\boxed{
\text{Off-task V2 RegMean weakness is caused mainly by effective-strength coverage ending at }\alpha=0.2
}
\]

The strongest supporting pattern would be:

\[
ASR_{0.2}
<
ASR_{1/6}
<
ASR_{0.1}
\lesssim
ASR_{0.0}.
\]

It does not need to be perfectly monotonic because training is stochastic.

The more important criterion is that extending the lower bound below 0.2 materially raises:

```text
RegMean ASR
Q10 G_true
P(G>D)
```

without materially reducing clean utility.

---

## 11. Strong success condition

Treat the hypothesis as strongly supported if at least one lower-bound variant reaches:

```text
RegMean ASR >= 95%
```

while:

```text
Avg clean ACC drop <= 0.5 pp relative to V2
```

and:

```text
Q10 G_true materially increases
P(G>D) tracks the ASR increase
```

If \(U(0,1)\) is best or statistically tied for best, prefer it because it removes an arbitrary lower-bound hyperparameter.

---

## 12. Partial success condition

If the best result is:

```text
85%–95% RegMean ASR
```

with clean utility preserved, then the effective-strength floor is a real contributor but not the entire explanation.

Then:

```text
keep the best alpha-range result
do not immediately add modules
inspect the remaining failed tail samples
```

Specifically compare:

```text
D_true
G_true
G_text
functional retention
```

between successful and failed samples.

---

## 13. Falsification condition

Treat the alpha-range hypothesis as falsified if all three new variants remain close to the V2 baseline, for example:

```text
all RegMean ASR <= ~82%
```

with no substantial improvement in:

```text
Q10 G_true
P(G>D)
```

Then the source/target geometry mismatch remains important and a new carrier mechanism is justified.

Do not continue tuning alpha_min below zero or invent a more exotic sampling distribution.

---

## 14. Why this experiment comes before a new method

Current evidence already shows:

```text
target text gain correlates strongly with true Cars margin gain
TA/TIES are nearly saturated
V3 directional randomization does not help
V4 whole-task-vector training is much worse
```

Therefore the simplest remaining explanation is that V2 learns the correct function but does not expose it to sufficiently weak effective contribution during training.

This experiment tests that possibility while changing only one scalar range.

---

## 15. If U(0,1) succeeds

The final method should remain V2, with only:

\[
\boxed{\alpha\sim U(0,1)}
\]

instead of:

\[
\alpha\sim U(0.2,1).
\]

Do not introduce a new method name.

Paper interpretation:

> Since the future merger can retain an unknown fraction of the malicious contribution, we sample the effective contribution strength over the full feasible interval rather than imposing a fixed positive lower bound.

The rest of the off-task method remains unchanged.

---

## 16. If U(1/K,1) is clearly best

If:

\[
U(1/6,1)
\]

clearly outperforms both \(U(0.1,1)\) and \(U(0,1)\), then use contributor-count-aware coverage:

\[
\boxed{\alpha\sim U(1/K,1)}.
\]

Do not select this formulation merely because it is conceptually appealing; require it to win empirically.

---

## 17. Remaining mergers after RegMean

Only after selecting the alpha range from the RegMean-first experiment, reuse that exact checkpoint and evaluate:

```text
TA
TIES
AdaMerging
EMR
RegMean++
```

Do not retrain.

The goal is to ensure the lower alpha floor improves RegMean without damaging the already-saturated V2 behavior on other mergers.

---

## 18. Files to add

Suggested:

```text
scripts/run_bmr_offtask_v2_alpha_regmean.sh
scripts/run_bmr_offtask_v2_alpha_remaining.sh
src/summarize_bmr_offtask_v2_alpha.py
```

Prefer reusing the existing V2 training code with CLI overrides:

```text
--alpha-min
--alpha-max
```

Do not duplicate the V2 implementation unless necessary.

---

## 19. Return package

After the RegMean-first sweep, return:

```text
bmr_offtask_v2_alpha_regmean_return.tar.gz
```

Include:

```text
launcher scripts
summary script
per-variant stage2 logs
per-variant stage2 summaries
RegMean result JSONs
RegMean mechanism JSONs
combined summary.csv
combined SUMMARY.md
git diff --stat
git status --short
```

Do not include checkpoints.

Include checkpoint paths and SHA256 hashes.

---

## 20. Final decision rule

The experiment answers one question only:

\[
\boxed{
\text{Was off-task V2 already correct, with its main RegMean weakness being an effective-strength range that stopped too early at }0.2?
}
\]

Do not change any other component until this question is answered.
