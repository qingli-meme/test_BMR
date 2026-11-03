# Classic-backdoor source provenance

These files are preserved source snapshots used to audit the CLIP ports in
`src/classic_backdoors/`.  The old framework-specific entry points are not run
directly: they target CIFAR-10 CNNs (and, for TrojanNN, Python 2/Caffe), while
the paper experiment uses the repository's CLIP ViT-B/32 checkpoints, data
preprocessing, classification heads, and model-merging implementations.

| Method | Upstream | Commit | Status used here |
|---|---|---|---|
| BadNets | https://github.com/SCLBD/BackdoorBench | `f02e3534645f0ee63d6848653062cd6c0d6c400d` | BackdoorBench reproduction; the original BadNets authors did not publish an implementation |
| LabelConsistent | https://github.com/MadryLab/label-consistent-backdoor-code | `b18d8ab549203e8cc518bd9ae4067c4e6398f415` | Author implementation |
| TrojanNN | https://github.com/PurduePAML/TrojanNN | `345b03c9eb6bb0faa7945582c0e0aa343eecba5b` | Author implementation; BackdoorBench's modern PyTorch port was also audited |
| Dynamic Backdoor | https://github.com/AhmedSalem2/Dynamic-Backdoor | `f59683fd367a0b80ee6c4016956a68180f55cff9` | Author implementation (conditional BaN/c-BaN) |

The per-method licenses are stored beside their source snapshots.  TrojanNN's
author repository has no explicit license; its files are retained only as an
audit snapshot.  The executable port is newly written against this project's
interfaces and records every adaptation in its result manifest.

