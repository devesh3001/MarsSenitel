# Engineering changelog

This is the rubric-facing symptom → diagnosis → fix → observed outcome journal. For the chronological defense narrative, see `DECISION_LOG.md`; for dated completion evidence, see `CURRENT_PROGRESS.md`. Reconstruction and reproducibility measurements are not hidden-label detection accuracy.

## Input audit and validation design

**Symptom:** The supplied folder contained competition documents and a nested archive, without code or prior experiment records.

**Diagnosis:** The initial risks were violating the no-pretraining rule, leaking crops from the same observation across splits, mishandling metadata and claiming anomaly accuracy without labels.

**Fix:** Read all three documents and embedded figures; decode every supplied crop; check index uniqueness, metadata joins and exact pixel duplicates; retain input hashes. Split whole source observations into training, validation and calibration groups.

**Outcome:** 10,422 valid 227 × 227 grayscale crops from 172 source observations; no missing joins or exact decoded-pixel duplicates. Main split: 120/25/27 sources and 7,101/1,666/1,655 crops. Near duplicates and geographic overlap remain possible. This audit is preparation, not a model iteration.

## v1 — Establish the from-scratch baseline

**Symptom:** There was no learned representation or evidence of achievable reconstruction quality under the rules.

**Diagnosis:** A compact convolutional autoencoder provides a transparent baseline without pretrained extractors or a skip connection that could bypass the required latent vector.

**Fix:** Five strided convolution blocks, GroupNorm and SiLU; a 128-dimensional vector; resize-convolution decoder; MSE objective; AdamW. Save source splits, configurations, checkpoints, history and fixed validation reconstructions.

**Outcome:** Completed 15 epochs. Best validation MSE 0.002693986, SSIM 0.661136 and gradient error 0.014006961. Broad brightness and image footprints reconstruct, but fine texture and edges are strongly smoothed. Latent covariance effective rank is 12.17/128. The final 2,000-tree detector flags 20 crops at threshold 0.540827788. These flags have no verified labels.

**Evidence:** `outputs/v1/history.csv`, `reconstructions_latest.png`, and `mixture_three_sigma_trees2000/`.

## v2 — Test structural loss against texture smoothing

**Symptom:** The completed v1 reconstruction panel loses small ridges, crater detail and fine texture despite low MSE.

**Diagnosis:** MSE favors low-frequency reconstruction. Capacity and training duration are alternative contributors, so change only the loss first.

**Fix:** Retain the 128-dimensional architecture, split, seed and 15-epoch budget. Use 0.9 MSE + 0.1 (1 − SSIM), with direct image-statistic SSIM and no pretrained network.

**Outcome:** Completed 15 epochs. Best validation SSIM rises to 0.671723677, while MSE rises about 1.7% to 0.002739586. Gradient error is 0.014001368; effective rank increases to 27.07/128. The structural benefit is modest and fine detail is still smoothed. The final 2,000-tree detector flags 9 crops at threshold 0.533417. This does not prove improved anomaly recall.

**Evidence:** `outputs/v2/history.csv`, fixed reconstruction panel, and the comparison CSVs.

## v3 — Test additional bottleneck capacity

**Symptom:** v2 retains visible smoothing after the structural objective change.

**Diagnosis:** The compact code may constrain reconstruction. More capacity could also encode unwanted variation or make anomalies easier to reconstruct, so improvement must be measured rather than assumed.

**Fix:** Increase only the latent dimension from 128 to 256, retaining v2's objective and main split. Parameters increase from 2,485,249 to 4,582,529.

**Outcome:** Completed 15 epochs, with checkpoint recovery after an interruption at epoch 12. Best validation MSE 0.002598794 (about 5.1% below v2), SSIM 0.673809843 and gradient error 0.013958476. The reviewed panel still smooths fine detail. These modest gains cost about 84% more parameters. Final detector comparison and independent initialization/split checks are documented as they complete in the decision log.

**Evidence:** `outputs/v3/history.csv`, `training_status.json`, `events.jsonl` and fixed reconstruction panel.

## Threshold revision — Diagnose the distribution before changing the rule

**Symptom:** The original v1 adjusted-boxplot threshold is 0.87484, above the maximum score 0.55648, and has a wide source-bootstrap interval of approximately [0.70253, 1.03219].

**Diagnosis:** The score histogram is multimodal; the single medcouple skewness adjustment expands the upper fence across distinct score populations. Zero flags alone would not justify a revision.

**Fix:** Preserve that result. Fit 1–4 Gaussian components on calibration scores, select minimum BIC, and use T = max_j(mu_j + 3 sigma_j). Keep multiplier 3 fixed across runs. Refit component selection during source-group bootstrapping. Never select a desired contamination fraction or lower the threshold to force five examples.

**Outcome:** The original 400-tree v1 mixture analysis has T = 0.53917 and 16 flags, with a source-bootstrap interval about [0.53211, 0.54615]. The fitted tail description is not a false-alarm guarantee. Correlated crops make BIC heuristic, and unknown contamination can distort the mixture. All earlier outputs are retained. Later 2,000-tree results are stored separately.

**Evidence:** `THRESHOLD_DECISION.md`, `outputs/v1/calibration.json`, and `outputs/v1/mixture_three_sigma/`.

## Detector stability — Increase trees after testing candidate membership

**Symptom:** With 400 trees, v2 repeat-seed rank correlations exceed 0.973 but the thresholded sets have zero Jaccard overlap with the initial set.

**Diagnosis:** A stable global ranking can conceal instability in a small extreme tail. Isolate forest randomness before attributing all differences to the encoder.

**Fix:** Compare 400 and 2,000 trees, each with 256 and 512 samples, over three forest seeds on unchanged v2 latents. Use the same threshold formula throughout.

**Outcome:** At 2,000 trees and 256 samples, repeated ranking correlations exceed 0.994 and candidate Jaccard is approximately 0.538/0.556. The 512-sample variant gives overlaps 0.400/0.444. Adopt 2,000 trees with 256 samples for the fair final model comparison. Reproducibility improves, but membership remains uncertain.

**Evidence:** `outputs/v2/mixture_three_sigma/forest_sensitivity.csv`.

## Execution and documentation reliability

**Symptom:** An execution interruption left partial evaluations and a valid v3 checkpoint without completed runs. The user also requested a maintained defense narrative and progress notes.

**Fix:** Resume unchanged configurations from the last checkpoint; distinguish completion markers from partial artifacts. Add `DECISION_LOG.md`, an artifact-derived `CURRENT_PROGRESS.md`, and timestamped training event logs. Include them in the local package. Correct newline escaping in the final notebook generator before execution.

**Outcome:** v3 completed after resuming; v1 and v2 final evaluations completed. Independent checks and final deliverable verification remain tracked in the progress file. Seven core tests passed at the latest recorded test run. The submission notebook and PDF must pass execution and visual review before being called ready.

**Timing limitation:** Recorded epoch wall times include host interruptions and contention. They are not controlled GPU benchmarks.

## Independent initialization and source-split checks — exact membership is fragile

**Symptom:** Forest-only repeats cannot establish that a learned representation generalizes across training initializations or source assignments.

**Fix:** Keep the v3 configuration and run 15 epochs with (a) training seed 2027, original split 2026 and (b) original training seed 2026, source split 2027. Fit a separate forest and calibration model for each. Retain all results without selecting a new seed after seeing them.

**Measured outcome:** The initialization repeat flags 6 crops and the split repeat flags 5, versus 17 for v3. Their Jaccard overlaps with v3 are 0.150 and 0.294; whole-ranking correlations are 0.949 and 0.957. No crop is flagged in all three pipelines. For the split comparison, 1,253 crops are held out from both encoders and their ranking correlation is 0.958. The different-split validation reconstruction metric is not directly comparable because its images differ.

**Diagnosis and decision:** The underlying representation and calibration boundary materially affect the extreme tail. Retain v3 as the original reference run among the controlled architecture experiments, with a prominent instability limitation. Better reconstruction and moderate forest-only stability do not validate the final anomaly list.

**Additional observed confound:** All five v3 interpretation candidates exceed the 99.7th percentile of mean brightness; three share SRC_154. Their physical explanations remain hypotheses. `scripts/inspect_selected_context.py` reproduces the brightness diagnostic.

**Evidence:** `outputs/comparison/run_stability.csv`, `outputs/comparison/experiments.csv`, the two robustness run directories, and `outputs/v3/mixture_three_sigma_trees2000/selected_acquisition_diagnostics.csv`.
# Second-stage additions

The original journal above remains historical evidence. The added experiments use the same source split, 256-dimensional architecture, structural loss and 15-epoch budget as v3.

## v4 — Contrast Normalization (Failed Ablation)

**Symptom:** All five leading v3 crops exceeded the 99.7th brightness percentile, revealing a brightness confound.

**Diagnosis:** The autoencoder was disproportionately penalizing bright regions instead of learning geological structures.

**Fix:** Standardize each crop to suppress global brightness and contrast before feeding it into the network.

**Outcome:** The illumination-test score change dropped significantly, but all five leading examples now contained black strips or camera borders. Contrast normalization amplified dark borders, creating a new artifact confound.

## v5 — Border Masking (Failed Ablation)

**Symptom:** v4 contrast normalization amplified border-connected black regions.

**Diagnosis:** The model was treating zero-valued (black) padding pixels from the camera as anomalous structure.

**Fix:** Exclude connected exact zeros from statistics and fill them with 0.5 (gray) to neutralize them.

**Outcome:** Near-black edge fragments and JPEG artifacts still leaked past the exact-zero mask. The minimum forest Jaccard fell to 0.227.

## v6 — Dynamic Footprint Analysis (Abandoned)

**Symptom:** Fixed zero-masking failed to remove gradient-blurred camera edges.

**Diagnosis:** The true data footprint varies unpredictably across crops.

**Fix:** Implement a dynamic variance-based mask.

**Outcome:** Abandoned because it destroyed too much genuine geological data, proving that heavy preprocessing is inferior to fixing the loss function.

## v7 — Gradient Loss (Primary Final Pipeline)

**Symptom:** Previous models (v1-v5) continued to flag extremely bright images (the "brightness confound") and failed to reconstruct fine ridges clearly.

**Diagnosis:** MSE and SSIM alone weren't explicitly penalizing blurry high-frequency details, causing the autoencoder to focus on coarse brightness rather than structural novelty.

**Fix:** Introduced an explicit image-gradient difference penalty (`gradient_weight = 0.5`) on top of MSE and SSIM to force the autoencoder to preserve sharp edges and local contrast.

**Outcome:** Validation MSE and SSIM improved significantly. More importantly, the new anomalies selected by the Isolation Forest no longer heavily correlated with mean brightness. The model broke the brightness confound and flagged structurally novel craters instead.

## v8 — Metadata Fusion (Secondary Experiment)

**Symptom:** The pipeline only saw image pixels; it was blind to known external variables like sun angle and latitude, which might explain some variance.

**Diagnosis:** If latitude and sun angle strongly dictate surface appearance, the autoencoder might be wasting latent capacity encoding these known facts rather than discovering true novelties.

**Fix:** Injected normalized metadata (latitude, sun angle, and season) directly into the latent space before decoding, conditioning the reconstruction on known acquisition variables.

**Outcome:** The model successfully learned to reconstruct the images, but the downstream Isolation Forest anomaly rankings were highly correlated (Spearman > 0.95) with the image-only v7 baseline. Metadata fusion did not significantly alter the novelty distribution, confirming the image-only pipeline is sufficient.

*Note: The standardized MSE/SSIM values describe different reconstruction targets. Neither should be used as a hidden-anomaly accuracy estimate.*
