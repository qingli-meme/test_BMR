# Reviewer experiment provenance

The integrations under `src/reviewer_exp/` retain the repository's OpenCLIP
ViT-B/32 checkpoints, preprocessing, classification heads, dataset loaders, and
evaluation routines. They port algorithmic operations only; no HuggingFace
evaluation outputs are mixed with the paper pipeline.

| Method | Upstream revision | Files consulted | OpenCLIP adaptation |
|---|---|---|---|
| BadMerging | `jzhang538/BadMerging@898cf1643184d8c9bd77472b266acd48cefe98a0` | existing repository training/evaluation entries | Existing local implementation retained. |
| RegMean++ | `nthehai01/RegMean-plusplus@a9fdd97629707062a5872653dd9b34f7e25c7243` | `fusion_bench/method/regmean_plusplus/{clip_regmean.py,regmean.py}`, `config/method/regmean_plusplus/clip_regmean.yaml` | HF vision embeddings/blocks replaced by OpenCLIP `visual._embeds` and `visual.transformer.resblocks`; sequential merged-layer propagation and official Gram rule retained. |
| Localize-and-Stitch | `uiuctml/Localize-and-Stitch@109d20676cab515ae9b5cc16111f04a381598085` | `localize_utils.py`, `vision/src/main_localize_stitch.py` | Official dense mask initialization/manual optimization/rounding and reciprocal overlap stitching adapted to local ImageEncoder, heads and dev split. |
| DAM | `Yangjinluan/DAM@105f8d92151074acfb9f9fbecb1e03e4be7cefcb` | `fusion_bench/method/concrete_subspace/clip_safe_concrete_adamerging.py`, `config/method/concrete_subspace/clip_safe_concrete_layer_wise_adamerging.yaml` | FusionBench/HF wrapper replaced by `torch.func.functional_call` over local OpenCLIP parameters; released three-stage safety mask and second AdaMerging optimization retained. |
| IBVS | Paper, arXiv:2510.08016 (ICLR 2026) | Paper definition | `implementation_source=paper_reimplementation`; fixed white-square EuroSAT reference and lambdas 0.1/0.3. No official repository was located. |

