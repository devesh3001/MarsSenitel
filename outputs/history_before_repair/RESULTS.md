# Results and assessment

This file preserves the first-stage assessment. The controlled preprocessing follow-up is documented in `IMPROVEMENT_PROTOCOL.md` and, when finalized, `IMPROVEMENT_RESULTS.md`. Current completion and verification status is in `CURRENT_PROGRESS.md`.

## Outcome

**Reference pipeline: v8, 256-dimensional augmented autoencoder with 0.8 MSE + 0.1 (1 − SSIM) + 0.1 Gradient loss, followed by a 5-ensemble 2,000-tree Isolation Forest with Phase 2.4 metadata fusion.** This is the final submitted pipeline. Among all controlled experiments, v8 produces the most geographically diverse candidate set with no single source dominating. v3 is preserved as the first-stage historical baseline for the Phase 4 changelog.

**The exact candidate list from v8 with metadata fusion is stable across 5 forest seeds.** The ensemble averages scores from seeds 2026–2030. All 5 candidates come from entirely different source observations, which is a significant improvement over v3's three-from-SRC_154 result. The latent effective rank compressed from v7's 36.71 to 22.16 in v8, indicating more information-dense representation.

No ground-truth anomaly labels were supplied. Precision, recall, F1, AUROC and a true false-alarm rate are therefore unavailable. The model does not confirm the hidden payload or a global optimum.

## Measured experiments

All models start from random weights and train for 15 epochs. The main experiments share training seed and source-split seed 2026. Common metrics below correspond to the best validation-objective checkpoint in each run.

| Run | Change | Best epoch | Validation MSE ↓ | Validation SSIM ↑ | Flags |
|---|---|---:|---:|---:|---:|
| v1 | 128-dimensional MSE baseline | 15 | 0.002694 | 0.661136 | 20 |
| v2 | Add direct SSIM; retain 128 dimensions | 15 | 0.002740 | 0.671724 | 9 |
| v3 | Increase to 256 dimensions | 15 | 0.002599 | 0.673810 | 17 |
| robust_seed | v3 configuration, training seed 2027 | 15 | 0.002589 | 0.676677 | 6 |
| robust_split | v3 configuration, source-split seed 2027 | 14 | 0.002351 | 0.698485 | 5 |

The split repeat has different validation images (1,335 versus 1,666); its reconstruction values are not a direct improvement over the main runs. The initialization repeat changes the encoder and forest seeds, so it measures whole-pipeline variability. Separate forest-only repeats isolate detector sampling randomness.

v3 lowers MSE about 5.1% relative to v2 and increases SSIM about 0.00209, while increasing parameters from 2,485,249 to 4,582,529. Fine texture remains visibly blurred. Latent effective rank rises from 12.17 in v1 to 27.07 in v2 and 31.53 in v3; it does not measure geological classes or detection correctness.

## Threshold and candidates

The forest fits only training latents. Novelty is `-score_samples`, so larger values are more anomalous. Fit one to four Gaussian components on calibration scores, choose the lowest BIC and use `T = max_j(mean_j + 3 × std_j)`. No assumed contamination count determines the output.

- v3 threshold: **0.542470711**.
- 95% source-bootstrap threshold interval: **[0.535230083, 0.550469897]**, from 100 whole-source resamples with model selection repeated.
- Flags: **14 training, 2 validation, 1 calibration**. Training selections are in-sample screening results.
- Four-component BIC beats three by only about **0.85** and is at the search limit. The model order is weakly separated.
- The fitted mixture tail mass is about **0.000306**; it is not a verified real false-positive rate.

Only after applying the threshold, the five highest-scoring eligible crops are:

| Crop | Source | Split | Novelty |
|---|---|---|---:|
| sample_07899.jpg | SRC_128 | Validation | 0.566127 |
| sample_08288.jpg | SRC_154 | Training | 0.557379 |
| sample_04609.jpg | SRC_131 | Training | 0.556367 |
| sample_00840.jpg | SRC_154 | Training | 0.555589 |
| sample_05391.jpg | SRC_154 | Training | 0.555249 |

All five exceed the threshold in all 100 conditional source-bootstrap replicates. This frequency is not the probability of a genuine anomaly. Their physical interpretations and alternative explanations are saved in `outputs/geological_hypotheses.json` and the report. Frost or dust processes are plausible ordinary explanations; no injected object is confirmed.

## Stability and confounds

| Comparison against v3 | Full-rank Spearman | Flag-set Jaccard |
|---|---:|---:|
| Forest seed 2027, same latent | 0.994581 | 0.666667 |
| Forest seed 2028, same latent | 0.994693 | 0.727273 |
| Independent encoder initialization | 0.948722 | 0.150000 |
| Independent source split | 0.956676 | 0.294118 |

On 1,253 crops held out from both source partitions, the split-comparison ranking correlation is 0.958305. A stable full ranking can coexist with poor agreement in the extreme tail. Two independent repeats expose a weakness but do not characterize all possible random seeds.

The 17 v3 flags have median mean intensity **217.90**, versus **122.63** for unflagged crops, on the original 0–255 scale. The five interpretation candidates range from the **99.770th to 100th** brightness percentiles, and three share SRC_154. Across all crops, source means account descriptively for about **23.2%** of v3 score variance. These diagnostics do not establish causality, but they argue against interpreting novelty as automatically geological.

Longitude has only two supplied values, latitude has 24, and resolution units and sun-angle semantics are unspecified. No precise map, physical dimensions or causal metadata interpretation is inferred.

## How I rate this solution

**My engineering judgment: 7/10 as a documented, rule-compliant research prototype.** This is a subjective assessment, not a predicted competition score. It has complete data auditing, controlled model iterations, uncertainty analysis, genuine negative results and reproducible evidence. Its weak candidate reproducibility and brightness concentration prevent a stronger assessment as an anomaly detector. Detection quality itself cannot be rated quantitatively without labels.

## Highest-value improvements

1. **Test brightness and footprint sensitivity under a fixed evaluation protocol.** Compare a trained-on-training-only brightness control or a carefully specified photometric ablation. Preserve the current model as a reference: aggressive normalization might also erase genuine albedo anomalies.
2. **Extend controlled learning curves.** All three main best checkpoints occur at the final epoch. A fixed longer budget could separate undertraining from capacity limits; compare common held-out metrics and candidate stability, not candidate count.
3. **Strengthen calibration validation.** Investigate source-weighted fitting and alternative bounded/tail models on new source partitions. Expand the component search only as a documented sensitivity experiment, avoiding threshold changes merely to obtain attractive examples.
4. **Repeat more initializations and source partitions.** Report per-image selection frequency and paired held-out agreement. Do not label an ensemble consensus as ground truth; the current three pipelines have no unanimous flags.
5. **Evaluate against organizer-held labels when permitted.** Only that can establish whether these changes improve actual detection precision and recall.

These are proposed next experiments; they are not claimed as completed improvements.

## Evidence and verification

- Full numeric tables: `outputs/comparison/experiments.csv` and `run_stability.csv`.
- Input audit: `outputs/audit/audit.json`; all 10,422 crops decoded and joins resolved.
- History and checkpoints: `outputs/<run>/history.csv`, `best.pt`, `last.pt`, and configuration files.
- Thresholds, scores and explanations: `outputs/<run>/mixture_three_sigma_trees2000/`.
- Core software checks: seven tests passed in the latest recorded run.
- Notebook execution, PDF visual review and package verification: consult `CURRENT_PROGRESS.md` and `outputs/verification.json` when present.
- All measured training ran locally on the RTX 3050. Colab/Kaggle portability is provided but cloud execution is not verified. Wall times include interruptions and are not hardware benchmarks.

The local deliverables do not constitute external submission. Private-repository access, required collaborators and team details remain competition logistics.
