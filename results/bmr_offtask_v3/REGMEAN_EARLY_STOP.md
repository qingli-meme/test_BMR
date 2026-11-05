# BMR Off-task v3: RegMean-first Early-stop Result

## Status

The Cars / RegMean-first v3 run was stopped intentionally on 2026-11-05 after
the attacked RegMean evaluation completed. The remaining clean-background
evaluation and mechanism analysis were not needed once the primary ASR showed
no improvement.

## Result

| Version | Target task | Merge method | ASR | Successful / evaluated |
|---|---|---|---:|---:|
| v2 baseline | Cars | RegMean | 77.85% | — |
| v3 directional | Cars | RegMean | **77.41%** | **6200 / 8009** |

The v3 directional objective is 0.44 percentage points below the v2 RegMean
baseline. RegMean ASR therefore remains low, and this run was not continued to
the remaining merge methods.

## Artifacts

- Raw log: `logs/bmr_offtask_v3/regmean_first_gpu1_20261105_175845.log`
- Trained checkpoint (local only, not committed):
  `checkpoints/ViT-B-32/CIFAR100_KDR_DTK_BMR_OFF_V3_Cars_Tgt_1_R10_L_22/finetuned.pt`
- Trigger used (local only, not committed):
  `trigger/ViT-B-32/KDR_DTK_OFF_V2_Cars_Tgt_1_R10_L_22.npy`

The raw log records `Backdoored accuracy: 77.41% (6200/8009)`. It ends during
the subsequent clean/background evaluation because the run was intentionally
terminated; no final `regmean.json` or mechanism JSON was produced.
