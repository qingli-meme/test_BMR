# BMR On-Task Stage-2 Slimming Ablation Report

> Historical ablation record. Its decision is now the canonical
> `KDR_DTK_BMR`: synthetic-background drift has been removed from the
> implementation, and legacy variant names below identify results only.

## Executive conclusion

The on-task CIFAR100 ablation supports a slimmer BMR Stage-2 objective:

\[
\theta_{\mathrm{atk}}=\theta_0+\alpha\delta_{\mathrm{adv}},
\qquad
\theta_{\mathrm{bg}}=\theta_0,
\qquad
\alpha\sim U(0.2,1.0).
\]

The matched background reference remains useful, but the synthetic background
drift does not provide a reproducible benefit. The recommended final method is
therefore:

- retain random scale variation of the malicious residual;
- retain the matched relative BMR objective using \(\theta_{\mathrm{bg}}=\theta_0\);
- remove the synthetic drift term \(\eta\delta_{\mathrm{bg}}\);
- remove the synthetic-drift sampler and its associated hyperparameters;
- do not replace the relative objective with a plain absolute-margin loss.

On RegMean, BMR without synthetic drift reaches **97.51% ASR**, compared with
**97.79%** for the original drifted BMR. The difference is only **0.28 pp**.
An independent finalist rerun slightly reverses the ordering: **97.48%** without
drift versus **97.43%** with drift. Clean utility is effectively unchanged.

## 1. Experimental question

This experiment separates two meanings that were previously conflated under
"background augmentation":

1. whether BMR needs a matched background reference \(\theta_{\mathrm{bg}}\);
2. whether that reference needs an additional synthetic drift
   \(\eta\delta_{\mathrm{bg}}\).

The original full construction is

\[
\theta_{\mathrm{atk}}
=\theta_0+\alpha\delta_{\mathrm{adv}}+\eta\delta_{\mathrm{bg}},
\qquad
\theta_{\mathrm{bg}}
=\theta_0+\eta\delta_{\mathrm{bg}}.
\]

For target margin \(M(\theta,x_\tau)\), BMR uses

\[
G=M(\theta_{\mathrm{atk}},x_\tau)
-M(\theta_{\mathrm{bg}},x_\tau),
\qquad
D=[-M(\theta_{\mathrm{bg}},x_\tau)]_+.
\]

The reserve objective encourages the malicious residual to contribute enough
positive target-margin gain relative to the matched background state.

## 2. Experimental setup

| Item | Setting |
|---|---|
| Setting | On-task attack |
| Model | ViT-B/32 |
| Adversary task | CIFAR100 |
| Target task | CIFAR100 |
| Target class | 1 |
| Trigger | KDR-DTK dormant trigger |
| Patch size | 22 |
| Training seed | 2026 |
| Merge benchmark | CIFAR100, GTSRB, EuroSAT, Cars, SUN397, PETS |
| Main discriminative merger | RegMean |
| RegMean train batches | 8 per task |
| TA/TIES scaling coefficient | 0.3 |

All six variants reuse the same trigger, clean initialization, trainable scope,
optimizer settings, clean/backdoor loss weights, and number of training epochs.

## 3. Ablation variants

| Variant | Effective \(\alpha\) | Relative BMR | Absolute margin | Synthetic drift | Background reference |
|---|---:|---:|---:|---:|---|
| Ordinary | 1.0 | No | No | No | None |
| Scale variation | \(U(0.2,1.0)\) | No | No | No | None |
| Absolute margin | \(U(0.2,1.0)\) | No | Yes | No | None |
| BMR fixed alpha | 0.6 | Yes | No | No | \(\theta_0\) |
| BMR no drift | \(U(0.2,1.0)\) | Yes | No | No | \(\theta_0\) |
| BMR + drift | \(U(0.2,1.0)\) | Yes | No | Yes | \(\theta_0+\eta\delta_{\mathrm{bg}}\) |

## 4. Main RegMean results

| Variant | ASR | Median \(G\) | q10 \(G\) | \(G\leq0\) | Avg. clean ACC | Delta vs clean merge |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 23.24% | 4.3372 | 1.7246 | 0.53% | 75.44% | -0.05 pp |
| Scale variation | 89.52% | 12.0528 | 7.0420 | 0.00% | 75.44% | -0.05 pp |
| Absolute margin | 89.51% | 12.0171 | 6.9959 | 0.00% | 75.44% | -0.05 pp |
| BMR fixed alpha | 64.28% | 9.0262 | 5.0466 | 0.00% | 75.41% | -0.08 pp |
| **BMR no drift** | **97.51%** | **15.8565** | **10.8418** | **0.00%** | **75.37%** | **-0.13 pp** |
| **BMR + drift** | **97.79%** | **16.0437** | **11.0350** | **0.00%** | **75.42%** | **-0.07 pp** |

The corresponding clean-only RegMean merge obtains **75.49%** average clean
accuracy.

## 5. Decisive comparisons

### A. Scale variation versus Ordinary

\[
89.52-23.24=\mathbf{+66.27\ pp}.
\]

Random variation of the malicious residual strength is essential. Much of the
benefit previously attributed to generic background augmentation is explained
by exposing the malicious residual to a range of effective scales.

### B. Absolute margin versus Scale variation

\[
89.51-89.52=\mathbf{-0.01\ pp}.
\]

The absolute-margin loss provides no measurable gain over scale variation alone.
Simply forcing the attacked state into the target region does not reproduce the
benefit of the matched relative objective.

### C. BMR no drift versus Scale variation

\[
97.51-89.52=\mathbf{+7.99\ pp}.
\]

The matched BMR objective has independent value after controlling for scale
variation. Thus the background reference should not be removed entirely. It can
be reduced to the fixed pretrained reference \(\theta_{\mathrm{bg}}=\theta_0\).

### D. BMR no drift versus BMR fixed alpha

\[
97.51-64.28=\mathbf{+33.22\ pp}.
\]

Scale variation remains necessary even when the relative BMR objective is used.
A single mean operating point, \(\alpha=0.6\), does not adequately train the
malicious residual for post-merge scale uncertainty.

### E. BMR + drift versus BMR no drift

| Metric | Drift minus no-drift |
|---|---:|
| ASR | +0.28 pp |
| Median \(G\) | +0.1873 |
| q10 \(G\) | +0.1932 |
| Average clean ACC | +0.06 pp |

These differences are too small to justify retaining the synthetic drift
mechanism, especially given the independent repeat below.

## 6. Independent finalist RegMean repeat

The two finalist checkpoints were evaluated again in an independent RegMean run.

| Variant | ASR | Median \(G\) | q10 \(G\) | \(G\leq0\) | Avg. clean ACC |
|---|---:|---:|---:|---:|---:|
| BMR no drift | **97.48%** | **15.9572** | **10.9319** | 0.00% | 75.36% |
| BMR + drift | 97.43% | 15.8845 | 10.8794 | 0.00% | **75.39%** |

This run reverses the tiny ASR and gain differences:

- no-drift ASR is higher by 0.05 pp;
- no-drift median \(G\) is higher by 0.0728;
- no-drift q10 \(G\) is higher by 0.0525;
- drift clean ACC is higher by only 0.03 pp.

The two runs jointly indicate statistical equivalence rather than a stable
synthetic-drift advantage.

## 7. TA and TIES sanity check for the finalists

The direct full-BMR versus no-drift evaluation gives:

| Merger | No-drift ASR | Full BMR ASR | No-drift clean ACC | Full BMR clean ACC |
|---|---:|---:|---:|---:|
| Task Arithmetic | 100.00% | 100.00% | 74.86% | 74.83% |
| TIES | 100.00% | 100.00% | 72.70% | 72.71% |

Removing synthetic drift therefore causes no degradation under TA or TIES.
These mergers saturate at 100% ASR, so RegMean remains the more discriminative
method-selection benchmark.

The formal six-variant TA/TIES evaluator has completed the TA and TIES passes and
is continuing its redundant RegMean pass before writing its combined JSON. This
does not block the method decision above, because the finalist TA/TIES results
and the complete six-variant RegMean results are already available.

## 8. Final method decision

### Retain

1. Random malicious-residual scaling:

   \[
   \alpha\sim U(0.2,1.0).
   \]

2. The matched relative BMR objective.

3. The fixed pretrained background reference:

   \[
   \theta_{\mathrm{bg}}=\theta_0.
   \]

### Remove

1. Synthetic background drift \(\eta\delta_{\mathrm{bg}}\).
2. Sampling of \(\eta\), \(\rho\), drift scopes, and random drift tensors from
   the final method definition.
3. The absolute-margin alternative.

### Recommended final Stage-2 formulation

\[
\boxed{
\theta_{\mathrm{atk}}=\theta_0+\alpha\delta_{\mathrm{adv}},
\qquad
\theta_{\mathrm{bg}}=\theta_0,
\qquad
\alpha\sim U(0.2,1.0)
}
\]

with

\[
G_\alpha=
M(\theta_0+\alpha\delta_{\mathrm{adv}},x_\tau)
-M(\theta_0,x_\tau),
\qquad
D_0=[-M(\theta_0,x_\tau)]_+.
\]

This preserves the two empirically necessary components—scale variation and the
matched relative margin—while deleting the synthetic component that does not
show reproducible value.

## 9. Result artifacts

- Six-variant RegMean results:
  `analysis/bmr_slimming_phase1_regmean/20261103_204329_real_merge_gain.json`
- Six-variant RegMean table:
  `analysis/bmr_slimming_phase1_regmean/20261103_204329_real_merge_gain.csv`
- Six-variant clean utility:
  `analysis/bmr_slimming_phase1_regmean/20261103_204329_clean_utility_delta.csv`
- Independent finalist repeat:
  `analysis/bmr_slimming_finalists_regmean/20261103_203233_real_merge_gain.json`
- Full-BMR TA/TIES sanity check:
  `results/bmr_background_slim/20261103_103604_KDR_DTK_BMR_CIFAR100_tgt1_L22.json`
- No-drift TA/TIES sanity check:
  `results/bmr_background_slim/20261103_111027_KDR_DTK_BMR_NO_BG_CIFAR100_tgt1_L22.json`

