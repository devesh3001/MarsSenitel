# Project decision log and defense notes

## How to use this record

This is the reasoning summary for the project: actions, observations, decisions, alternatives, tradeoffs and evidence. It separates proposals from completed experiments and avoids claiming outcomes that have not been measured. Steps 01-10 were reconstructed from saved code, logs and outputs when this record was requested on 16 September 2026. Later material decisions are appended as work proceeds.

Read CURRENT_PROGRESS.md for the latest state, ENGINEERING_CHANGELOG.md for the competition's symptom-diagnosis-fix narrative, and the per-run history.csv and events.jsonl files for numerical progress. An individual training epoch is recorded in those run logs; this document records meaningful experimental decisions.

## 01 Read the rules before selecting a method

**Status:** Completed.

**Action:** Read all three supplied Word documents, including tables and embedded figures, and inspect the nested dataset archive.

**Why:** A high-performing method would still be unusable if it violated the challenge. The main constraints are random initialization, a fixed-length image latent, Isolation Forest scoring, an independently justified threshold, and three genuine iterations.

**Decision:** Build the required autoencoder-to-Isolation-Forest pipeline. Exclude pretrained CLIP, DINO, VGG and similar features. Use reconstruction errors for interpretation, not as a replacement final score.

**Evidence:** RESEARCH_AND_APPROACH.md; source text and document imagery in outputs/audit.

**Defense sentence:** "I translated the rubric into implementation constraints before choosing the model, so the design directly addresses what is being assessed."

## 02 Audit every input and preserve provenance

**Status:** Completed.

**Action:** Decode all images, inspect pixel ranges and a seeded contact sheet, validate metadata keys and joins, calculate exact decoded-pixel hashes, and preserve original-file SHA256 hashes.

**Observed:** 10,422 images, all grayscale and 227 by 227; 172 source observations; complete joins; no exact pixel duplicates. Source counts range from 1 to 278. Longitude has only two values.

**Decision:** Retain all crops, including visually unusual ones. Rescale by 255 without per-image contrast normalization. Treat metadata as coarse acquisition context.

**Why and limits:** Cleaning away unusual images could remove the unknown payload. Per-image normalization could erase brightness anomalies. Exact hashing does not exclude near duplicates or geographic overlap.

**Evidence:** scripts/audit_inputs.py; outputs/audit/audit.json; outputs/audit/random_batch.png.

**Defense sentence:** "I checked the actual released data rather than relying only on approximate counts in the description, and I did not use appearance to create pseudo-labels."

## 03 Separate source observations when validating

**Status:** Implemented and used in v1-v3.

**Action:** Fix split seed 2026 and allocate approximately 70/15/15 percent of source IDs to training, validation and calibration.

**Observed:** 120/25/27 sources and 7,101/1,666/1,655 crops respectively. Each source belongs to one split.

**Decision:** Fit the encoder and forest on training crops; select checkpoints with validation reconstruction loss; fit the score boundary with calibration crops. Record the split of every scored image.

**Why and limits:** Crops from one observation may share terrain or acquisition conditions. A random crop split could exaggerate generalization. Source grouping reduces that risk but does not prove complete spatial independence. All sets are unlabelled and potentially contaminated. Calibration outputs are also examined during exploratory method comparisons, so they are not an untouched final accuracy test.

**Evidence:** mars_anomaly/data.py; outputs/v1/split_manifest.csv and corresponding files for later runs.

**Defense sentence:** "I held out entire observations because the unit of independence is closer to an acquisition than to an individual crop."

## 04 Establish a compact, reproducible baseline

**Status:** v1 completed, 15 epochs.

**Action:** Train a five-stage spatial convolutional autoencoder with a 128-dimensional vector, GroupNorm, SiLU, resize-convolution decoding and MSE loss. Preserve original image size and omit skip connections around the vector.

**Why:** A compact deterministic model fits the available 4 GB GPU and exposes the compression tradeoff clearly. Spatial features preserve localized information before the bottleneck. Resize-convolution avoids adding transposed-convolution overlap artifacts. A VAE would introduce KL-weighting and posterior-collapse questions before the baseline is understood.

**Observed:** At the selected epoch, validation MSE is 0.002694 and SSIM is 0.661136. Fine texture and crater edges remain blurred in the fixed validation panel.

**Evidence:** outputs/v1/config.json, history.csv, best.pt and reconstructions_latest.png.

**Defense sentence:** "I chose a small interpretable baseline so each subsequent change could be tied to an observed limitation rather than changing many mechanisms at once."

## 05 Verify the software contracts before trusting results

**Status:** Completed; seven tests passed after the threshold extension.

**Action:** Check image/latent shapes, finite gradients, SSIM behavior, score orientation, threshold-before-selection logic, rejection of degenerate calibration, source-bootstrap reproducibility and the mixture envelope. Exercise training, checkpointing and scoring with an isolated smoke run.

**Decision:** Keep smoke outputs in outputs/smoke and reject smoke checkpoints in normal evaluation.

**Why:** A runnable pipeline is not enough: reversed score direction or selecting five images before thresholding would invalidate the analysis. Smoke scores are software checks, not competition evidence.

**Evidence:** tests/test_contracts.py; outputs/smoke; the actual v1-v3 experiment directories.

## 06 Investigate failure of the initial threshold proposal

**Status:** Completed for v1.

**Action:** Apply the initially proposed skewness-adjusted boxplot rule to held-out calibration scores, then inspect the histogram, sorted scores and bootstrap interval.

**Observed:** Medcouple 0.5185; cutoff 0.87484; bootstrap interval about 0.70253-1.03219; maximum observed score 0.55648. No images exceed the boundary. The histogram has multiple peaks.

**Diagnosis:** A single skewness adjustment spans distinct score populations and becomes excessively broad in this case. Zero flags by itself would not justify changing the rule.

**Decision:** Retain the failed result unchanged and investigate a multimodal density model. Do not lower the cutoff just to obtain five examples.

**Evidence:** outputs/v1/calibration.json; outputs/v1/score_distribution.png; THRESHOLD_DECISION.md.

**Defense sentence:** "The change was motivated by distribution mismatch and unstable calibration, not by a desired anomaly count. I kept the original zero-flag result in the record."

## 07 Replace the single-population fence with a mixture envelope

**Status:** Implemented and evaluated on completed v1 and v2 checkpoints.

**Action:** Fit one to four Gaussian components to calibration scores, select the converged fit with lowest BIC, and use T = max_j(mu_j + 3 sigma_j). Refit component selection in 100 source-group bootstrap samples.

**Why:** The histogram supports multiple score populations. The rule allows the upper population to be regular too, instead of calling that whole mode anomalous. The multiplier 3 is held fixed across runs and is not tuned to a sample count.

**Observed for v1 with 400 trees:** Three components; cutoff 0.53917; interval approximately 0.53211-0.54615; 16 flagged crops. Validation CDF distance is approximately 0.0541.

**Alternatives and limits:** A fixed top-N or score percentile would make the count an input. A single Gaussian ignores multimodality. Extreme-value fitting introduces additional tail-fit assumptions. The mixture can still misfit or absorb anomalous data. Its nominal Gaussian tail bound is not a guarantee of the true false-alarm rate, and BIC is approximate with correlated crops.

**Evidence:** mars_anomaly/threshold.py; outputs/v1/mixture_three_sigma; THRESHOLD_DECISION.md.

**Defense sentence:** "The anomaly count emerges from a fitted statistical boundary. It is never supplied as contamination or a top-N target."

## 08 Test structural loss as v2

**Status:** Completed, 15 epochs.

**Symptom:** v1 smooths terrain textures and edges.

**Change:** Keep the 128-dimensional architecture, data split and training budget; change MSE to 0.9 MSE + 0.1 (1 - SSIM). Compute SSIM directly from image statistics, without pretrained perceptual features.

**Observed:** Validation SSIM increases from 0.661136 to 0.671724. MSE changes from 0.002694 to 0.002740, roughly 1.7 percent higher. Effective latent covariance rank rises from about 12.17 to 27.07. Fine texture is still visibly smoothed.

**Interpretation:** This is a modest reconstruction tradeoff. It is not proof of better anomaly recall. The v2 400-tree candidate set is especially unstable across forest seeds.

**Evidence:** outputs/v2/history.csv; outputs/v2/reconstructions_latest.png; outputs/comparison/experiments.csv.

**Defense sentence:** "The structural-loss experiment improved SSIM with a small MSE cost. I reported both metrics and did not translate that into an unsupported detection-accuracy claim."

## 09 Diagnose forest variability separately from encoder changes

**Status:** Controlled v2 detector comparison completed; final 2,000-tree analyses pending completion.

**Symptom:** With 400 trees and 256 samples per tree, v2 repeat-seed full rankings correlate at about 0.973-0.975, but the two repeated flagged sets have zero Jaccard overlap with the first.

**Action:** Compare 400 versus 2,000 trees and 256 versus 512 samples per tree, with three forest seeds and the same threshold formula.

**Observed:** At 2,000 trees and 256 samples per tree, rank correlations exceed 0.994 and flag-set Jaccard rises to about 0.538 and 0.556. The 512-sample alternative gives lower overlaps of about 0.400 and 0.444 at 2,000 trees.

**Decision:** Use 2,000 trees with 256 samples for the final like-for-like model comparison. Preserve the earlier 400-tree outputs. This changes the forest sampling budget without inventing a new target anomaly count.

**Limit:** Better reproducibility is useful but does not establish correctness. Candidate membership remains only moderately stable.

**Evidence:** scripts/forest_sensitivity.py; outputs/v2/mixture_three_sigma/forest_sensitivity.csv.

**Defense sentence:** "I checked the actual thresholded membership, because a high correlation across all 10,422 rankings can hide disagreement among the few candidates that matter."

## 10 Test additional latent capacity as v3

**Status at the interruption check:** 12 of 15 epochs saved; final result not yet known.

**Symptom:** v2 still smooths fine structure.

**Change:** Increase the vector from 128 to 256 while retaining the v2 loss, source split and training budget. Parameter count increases from 2,485,249 to 4,582,529.

**Why:** This tests whether capacity, rather than only the reconstruction objective, limits detail. The larger model will be selected only if measured benefits justify its cost. It is acceptable for v3 to lose this comparison.

**Observed so far:** Epoch-12 validation MSE 0.002754 and SSIM 0.670429. These interim values must not be presented as the final v3 result.

**Evidence:** outputs/v3/history.csv and last.pt. Completion requires training_status.json.

## 11 Record and recover the execution interruption

**Recorded:** 16 September 2026, 13:26 local time.

**Observed:** No Python training process is running. v3 has a saved epoch-12 checkpoint but no completion marker. The two 2,000-tree evaluations have saved latent arrays but no final diagnostics. Earlier tool-session handles are no longer available. The cause of the interruption is not established.

**Decision:** Resume v3 from last.pt with the same configuration and restart incomplete evaluations. Do not count a launched process, partial output directory or saved latent array as completed evaluation.

**Why:** Checkpoints preserve genuine progress, and explicit completion checks prevent partial results from being reported as finished work.

## Next decisions to record

1. Completed v3 outcome and comparison against v1/v2 under the same final detector.
2. Final configuration selection, including why rejected alternatives were rejected.
3. Independent training-seed and source-split checks, with any unstable results retained.
4. Selected candidate heatmaps and physical hypotheses, including plausible ordinary explanations.
5. Notebook execution, PDF visual verification and local submission packaging.

## Questions to rehearse for the defense

- Why use source-based rather than random crop splits? Explain shared acquisition conditions and the remaining overlap limitation.
- Why no pretrained model? It is explicitly prohibited, including perceptual-loss extractors.
- Why 128 or 256 dimensions? Give the measured reconstruction/stability tradeoff, not just compression arithmetic.
- Why change the threshold? Show the initial histogram and failed fence, then explain the mixture assumptions.
- Why is the new rule not top-N in disguise? State the fixed formula and show that output counts vary across runs.
- Why not claim F1 or recall? Hidden ground truth is unavailable, and visual hypotheses are not labels.
- Why can two stable rankings disagree on anomalies? The decision depends on a small tail and an estimated boundary.
- Do heatmaps explain Isolation Forest directly? No. They localize reconstruction discrepancy for the images selected by latent novelty.
- Can a flagged crop be valid Martian terrain? Yes. Rarity, acquisition footprints and unfamiliar geology are legitimate alternatives to an injected anomaly.
- Is this the global optimum? No. It is a documented selection among the tested configurations under the available experiment budget.

## 12 Add a maintained decision trail and resume work

**Recorded:** 16 September 2026, 13:36 local time.

**User requirement:** Keep structured notes of starts, changes, reasons and progress so the approach can be defended.

**Change:** Added this decision log and `CURRENT_PROGRESS.md`; added `scripts/update_progress.py` to derive status from saved artifacts. Added timestamped training events in `outputs/<run>/events.jsonl`, linked the notes from the README, and included notes and event logs in the local package builder.

**Why:** A narrative without measurements is hard to defend, and a launched process is not evidence of completion. The notes separate decisions, measurements, limitations and pending work. Earlier entries are retrospective summaries; new events are recorded as work proceeds.

## 13 Introduce V7 Gradient Loss Experiment

**Recorded:** 17 September 2026, 12:25 local time.

**User requirement:** Implement an improvement to stop the model from fixating on global brightness without resorting to unstable border standardizations.

**Action:** Designed the `V7_GRADIENT_EXPERIMENT_PROTOCOL.md` and launched the training with `--gradient-weight 0.1`.

**Why:** A Sobel-style gradient loss explicitly forces the autoencoder to reconstruct high-frequency Martian geology (crater edges, ridges, and injected anomalies) rather than just broad shapes. If it fails, the negative result is logged and v3 remains the reference.

**Started:** Resumed v3 with its unchanged configuration from epoch 12. Restarted the incomplete v1 and v2 evaluations with 2,000 trees. v3 has now saved epoch 14; the evaluations remain in progress.

**Verification:** The resumed training printed the expected RTX 3050 device, 4,582,529 parameters, and new epoch records. No model objective or data split was changed by the logging update.

**Defense sentence:** "I kept the decision rationale alongside configurations and measured outputs, including failed approaches and interruptions, so each change can be traced to evidence."

## 13 Complete v3 training and start its initialization check

**Recorded:** 16 September 2026, 13:38 local time.

**Measured outcome:** v3 completed 15 epochs. Its best checkpoint is epoch 15: validation MSE 0.002598794, SSIM 0.673809843 and gradient error 0.013958476. This improves MSE by about 5.1 percent relative to v2 and improves SSIM by about 0.00209, at an approximately 84 percent increase in parameter count. This is a reconstruction benefit, not established anomaly-detection superiority.

**Completed evaluations:** At 2,000 trees, v1 flags 20 crops with threshold 0.540827788; v2 flags 9 with threshold 0.533417. Their full score ranking correlation is 0.9675 but flag-set Jaccard is only 0.2609. Different learned representations still change candidate membership.

**Started:** v3's 2,000-tree evaluation and an independent initialization repeat of its configuration (`robust_seed`: training seed 2027, unchanged source-split seed 2026). Final model selection remains pending the stability comparison.

**Why:** v3 is now a plausible reconstruction winner. Testing another initialization checks whether this result depends strongly on one training run. The forest also uses the run seed, so this repeat measures whole-pipeline variation; the separate forest-only repeats isolate detector randomness.

**Evidence:** outputs/v3/training_status.json; outputs/v3/history.csv; outputs/comparison/experiments.csv; outputs/comparison/run_stability.csv; outputs/robust_seed/config.json.

## 14 Compare complete models and inspect the selected crops

**Recorded:** 16 September 2026, 13:49 local time.

**Measured:** v3's final forest flags 17 crops at T = 0.542470711, with source-bootstrap interval [0.535230083, 0.550469897]. Repeat-forest candidate Jaccard is 0.6667/0.7273. Its reconstruction metrics and forest repeatability make it the provisional choice over v2, though v1 is smaller and has slightly better forest repeatability. No hidden-label comparison exists.

**Started:** An independent source-split repeat (`robust_split`: split seed 2027, training seed 2026), plus v3 t-SNE at perplexities 30 and 70. The initialization repeat is also training. These are separate run directories and will not replace the main v3 checkpoint.

**Visual review:** Inspected all five selected v3 heatmap panels. All crops are bright; three share SRC_154. Fine texture and narrow streaks are smoothed by the decoder. Wrote a visible-evidence description, physical hypothesis and alternative explanation for each in `outputs/geological_hypotheses.json`. Generic primary-source descriptions of frost and dust processes support plausibility only; they are not used to look up or replace the supplied metadata.

**Important limitation:** The four-component mixture wins BIC over three components by only 0.85 and is at the search limit. This weak separation and shared source conditions are explicitly included in the report/notebook drafts. Conditional bootstrap frequency of 100% for a selected crop is not a probability that it is a genuine anomaly.

**Changed:** Replaced stale planned-v2/v3 entries in `ENGINEERING_CHANGELOG.md` with observed results and kept incomplete checks marked as pending.

**Defense sentence:** "The best tested reconstruction model is only a provisional screening choice. I inspected acquisition-related alternatives and tested repeatability rather than treating the top-ranked images as verified discoveries."

## 15 Prepare defense materials and verify draft deliverables

**Recorded:** 16 September 2026, 13:54 local time.

**Changed:** Added `DEFENSE_GUIDE.md` with a 15-minute presentation outline, a plain-language method summary and answers to likely questions. It separates measured facts, physical hypotheses and unknown quantities. Added it to the package builder.

**Draft artifacts:** Generated the report with a visible provisional-status banner and reviewed all 12 rendered draft pages. Figures, captions and candidate explanations are readable. Separated stability and acquisition metadata into two pages in the builder to leave room for the forthcoming independent comparisons. The report must be regenerated and reviewed after those results arrive.

**Notebook verification:** Generated the rubric-numbered submission notebook. Its schema is valid and every code cell compiles. This is not yet an executed notebook: actual execution and output checks remain pending. Fixed embedded newline escaping before this validation.

**Why:** A reproducible defense needs both the actual evidence and a clear explanation. Draft generation and syntax checks catch presentation and execution issues before final packaging, without mislabelling incomplete experiments as finished.

## 16 Quantify the brightness concern and complete independent training

**Recorded:** 16 September 2026, 18:07 local time.

**Measured concern:** All five reviewed v3 candidates lie above the 99.7th percentile of mean crop brightness. The median crop mean is 217.90 among all 17 flags versus 122.63 among unflagged crops, in original 0–255 units. This is a substantial potential confound despite the weak whole-dataset correlation between mean intensity and novelty.

**Change:** Added a reproducible selected-context diagnostic and included its table and explanation in the notebook/report. It describes the existing candidates and does not alter their threshold. The helper can compute brightness from original crops if the local audit statistics are absent on a fresh cloud run.

**Training completion verified:** Both independent runs completed 15 epochs. `robust_seed` selects epoch 15, MSE 0.002588559 and SSIM 0.676676766 on the original validation set. `robust_split` selects epoch 14, MSE 0.002351323 and SSIM 0.698485442 on a different validation set (1,335 images). These different-split reconstruction numbers are not a direct accuracy comparison against v3.

**Started:** 2,000-tree evaluations with independent calibration for both robustness runs. Keep original v3 as the primary run rather than cherry-picking a better initialization after inspecting its results.

**Verification:** Seven core tests passed again in 13.29 seconds. Both t-SNE projections completed. Epoch timing again includes long execution gaps; no controlled speed claim is made.

**Defense sentence:** "The screening tail is strongly associated with bright crops. I reported that measurable alternative explanation and kept the original run fixed while testing reproducibility."

## 17 Freeze the reference run with the negative robustness result

**Recorded:** 16 September 2026, 18:12 local time.

**Measured:** The independent initialization flags 6 crops; the changed split flags 5. Their Jaccard overlaps with v3's 17 are 0.150 and 0.294, while full-ranking correlations are 0.949 and 0.957. No crop is flagged in all three pipelines. The split comparison has correlation 0.958 on 1,253 jointly held-out crops.

**Decision:** Retain original v3 as the screening reference among the controlled architecture experiments, but explicitly conclude that exact candidate membership is fragile. Do not replace it with a favorable repeat, average incompatible raw score scales, or invent a consensus candidate list.

**Changed:** Recorded the completed selection and robustness summary in `outputs/final_selection.json`; added the negative result to the engineering changelog and prominently to the report. Wrote `RESULTS.md` with measured comparisons, a subjective 7/10 research-prototype assessment, and concrete next experiments. Detection accuracy remains unknown; the rating is not a predicted competition score.

**Started verification:** Rebuilt the notebook from current source and completed journal. Schema validation and compilation passed; actual notebook execution is now running, reusing the completed trained checkpoints. Generated the updated report for final rendering and visual review.

**Why:** A defensible result includes evidence against the method. Completing artifacts must not convert unstable screening outputs into a claimed discovery.


## 18 Verify the notebook and final report

**Recorded:** 2026-09-16T18:16:20+05:30

**Verified:** All 15 notebook code cells executed with no error outputs, reusing the completed trained checkpoints and analyses. Seven core tests passed. The final 13-page PDF was rendered and visually inspected; revised pages 1, 5 and 7 were inspected again after clarifying split comparability, projection interpretation and crop-mean brightness. No clipping, overlap or missing text was observed.

**Changed:** Updated the README, research follow-through and submission checklist to reflect completed evidence. Added artifact hashes and verification details in `outputs/verification.json`. Excluded temporary runtime files from version control. The package builder now refuses provisional selection, missing experiments or unverified artifacts.

**Next:** Assemble and CRC-check the local package. Private repository setup, organizer access and team details remain external submission logistics.


## 19 Assemble and verify the local package

**Recorded:** 2026-09-16T18:17:24+05:30

**Completed:** The package builder assembled 224 files (approximately 189.6 MB), wrote a SHA-256 file manifest and passed ZIP CRC validation. It includes the executed notebook, reviewed report, best checkpoints, score outputs, source code, tests and defense notes. Raw dataset inputs, the Python environment and temporary runtime files are excluded.

**Final documentation update:** Regenerate the progress snapshot and rebuild the package once to include this completion entry and the updated snapshot. The builder repeats its integrity check.

**Overall result:** Five completed training runs, five final detector analyses, seven passing core tests, 15 executed notebook code cells and a 13-page visually reviewed report. v3 remains a reference screening run with 17 candidates; brightness and poor independent-run candidate agreement prevent a validated detection claim.

**Remaining:** External competition logistics only for this local deliverable set: team details, private repository and specified organizer collaborators. Additional scientific improvements are explicitly proposed, not claimed complete.


## 20 Start controlled brightness and footprint improvements

**Recorded:** 2026-09-16T18:31:07+05:30

**User request:** Improve competition competitiveness and write the notebook in a natural first-person style suitable for Colab. Preserve accurate execution provenance; existing experiments ran locally.

**Protocol fixed before results:** v4 standardizes each image; v5 additionally excludes border-connected exact zeros when estimating contrast and fills those regions with 0.5. Both use the same 256-dimensional model, structural loss, source split and 15-epoch budget as v3. The detector and fixed statistical rule remain unchanged. `IMPROVEMENT_PROTOCOL.md` documents decision criteria and risks.

**Changed:** Added explicit preprocessing modes, backward-compatible defaults for existing checkpoints, resume-mode validation and five-panel raw/standardized/reconstruction explanations. Added tests for affine-intensity invariance, border-only zero connectivity and constant-image behavior. Ten tests passed in 12.34 seconds.

**Started:** v4 and v5 training in separate directories. The previous verified report and package remain first-stage artifacts; they do not claim to contain these new experiments.


## 21 Make the notebook readable and define a direct nuisance test

**Recorded:** 2026-09-16T18:40:23+05:30

**Changed:** Added a new Colab notebook builder with visible preprocessing, dataset, model, training and scoring definitions. It replaces the large embedded-source dictionary with normal Python cells and short first-person observations. Local execution provenance remains explicit. Ten extracted source sections compile. The previous notebook remains a historical artifact until the new one is executed.

**Additional controlled check:** Added `measure_photometric_sensitivity.py`. On a fixed 512-crop validation sample, transform non-footprint pixels by 0.8*x + 0.1 while retaining the original exact-black border. Hold the model, forest and threshold fixed. Compare input/latent changes, score changes and flag flips across v3/v4/v5. This tests a specified illumination nuisance, not anomaly accuracy.

**Progress:** v4 and v5 continue their unchanged 15-epoch training. Their standardized-domain reconstruction errors are not compared directly against raw-image v3 errors.


## 22 Adjust execution order after a memory-limit failure

**Recorded:** 2026-09-16T18:42:30+05:30

**Observed:** Starting the separate photometric-sensitivity process while both training processes were active failed during PyTorch DLL loading with Windows error 1455 (paging-file capacity). No sensitivity result was produced. The two training histories continued advancing. An optional OS memory query was access-denied.

**Decision:** Defer the additional PyTorch process until a training process completes. Do not change OS settings or treat an unsuccessful launch as an experiment outcome.

**Changed:** Added `compare_improvements.py` to report threshold uncertainty, forest repeatability, raw brightness/contrast/zero-fraction associations and source concentration consistently across the new runs. It does not select a desired candidate count.

## 23 Recover interrupted checkpoint writes and run sequentially

**Recorded:** 2026-09-16, resumed at 23:11 IST.

**Observed:** Both concurrent runs subsequently failed while writing temporary last checkpoints. The existing last.pt files passed ZIP CRC checks. Only the two failed last.tmp.pt files were removed after checking their paths were inside this workspace. Resource pressure is plausible; the precise cause of the write failure was not established.

**Decision:** Run training, evaluation and sensitivity checks sequentially. Resume valid checkpoints with the original configuration and epoch budget. Preserve the failure history rather than count interrupted work as a completed experiment.

**Outcome so far:** v4 completed 15 epochs; validation selected epoch 11. v5 resumed from epoch 9 and has reached epoch 13. Evaluation and the fixed photometric test follow in the same sequential pipeline. Final selection remains pending evidence.

## 24 Evaluate the preprocessing tradeoff before independent repeats

**Recorded:** 2026-09-16, after both 15-epoch runs completed.

**Measured:** The fixed 512-crop illumination test gives 95th-percentile absolute score changes 0.033935 (v3), 0.002902 (v4), and 0.000044 (v5). Median selected brightness percentiles are 99.90, 47.51 and 66.10. Minimum forest-seed Jaccards are 0.667, 0.467 and 0.227. No sampled flags flip under this perturbation in any run; that metric alone does not establish an improvement. Normalization reduces the specified nuisance but weakens detector repeatability.

**Decision:** Repeat v4, the more stable of the two standardized variants, with seed 2027/split 2026 and seed 2026/split 2027. Keep the model, loss, 15-epoch budget, 2,000 trees and statistical threshold rule fixed. This is a test of the tradeoff, not an automatic promotion. Final reference selection is still pending.

**Resource adjustment:** Free space fell below 0.5 GB during evaluation. Applied reversible NTFS compression to the environment's dnnl.lib build library, saving about 486 MB without changing file contents or deleting evidence. Continue one PyTorch process at a time.

## 25 Inspect the new failure mode after thresholding

**Evidence:** Visually reviewed all five v4 selected panels. Each contains a conspicuous black strip or border. Reviewed the shared validation candidate sample_09852 under both v4 and v5: v5 fills exact zeros but leaves near-black edge fragments and corresponding reconstruction discrepancy. The panel localizes reconstruction error, not causal forest attribution.

**Decision:** Do not promote either normalization experiment merely because the designed affine-intensity perturbation improves. Retain v3 as the screening reference unless new evidence resolves the acquisition and repeatability weaknesses. Complete the already specified v4 repeats to measure the tradeoff rather than abandon a disappointing result.

**Presentation change:** Add the paired v4/v5 failure example to the notebook and report, alongside the numerical benefit. Explain the distinction between improving a controlled nuisance test and demonstrating better hidden-anomaly detection. The final project will preserve negative experiments and identify the remaining limitation plainly.

## 26 Strengthen checkpoint recovery and review the artifact draft

**Changed:** Best checkpoints now use temporary-file serialization and atomic replacement, as last checkpoints already did. Save best weights before advancing last-checkpoint best-loss metadata. This protects an existing best checkpoint if a write fails partway. Added a failure-injection test that verifies the old checkpoint survives and a subsequent retry succeeds; execution is deferred until GPU work ends to avoid another concurrent PyTorch process.

**Scope:** This changes checkpoint I/O, not architecture, preprocessing, loss or random initialization. It was added after the contrast initialization repeat finished and before the source-split repeat starts. Earlier run failures and valid outputs remain preserved.

**Resource evidence:** The loaded CUDA DLL could not be compressed because it was in use; no change was made to it. Compressing the remaining static .lib files recovered about 112 MB. No data or model checkpoint was removed.

**Artifact review:** Built a clearly marked draft notebook and PDF, then rendered and inspected all 15 draft pages. Corrected captions to distinguish raw-input reference panels from standardized ablations. Final repeat results, curves, full journal, notebook execution and final PDF review remain pending.

## 27 Record the initialization repeat without overstating it

**Measured:** v4 and its completed initialization repeat share 6 flags out of a 29-crop union, Jaccard 0.207. The repeat flags 11 crops, versus 24 for v4. Full ranking and the source-split comparison will be generated after the remaining run.

**Execution constraint:** An additional SciPy statistics import failed with a paging-file capacity error while source-split training was active. The overlap above was computed directly from the saved CSV flag sets using the standard library. Defer additional numerical-library processes until training exits. No missing result is represented as completed.

## 28 Recover the source-split process after an access violation

**Observed:** The source-split run exited with Windows status 0xC0000005 after saving epoch 1, without a Python traceback. Both last.pt and best.pt passed ZIP CRC integrity checks. This occurred amid memory and disk pressure, but that does not prove the exact cause of the access violation.

**Action:** With no Python process running, apply reversible NTFS compression to the local environment DLLs. The CUDA library alone recovered about 92 MB; other library compression is in progress. Preserve dataset inputs and every valid checkpoint. Resume the unchanged source-split configuration from epoch 1 after compression finishes.

## 29 Validate checkpoint protection and resume after compression

**Verification:** All 11 tests passed in 17.79 seconds, including the new simulated partial-write/retry test. Full output is saved in outputs/test_results_improved.txt.

**Resource recovery:** Reversible compression of the remaining 37 DLLs recovered about 450 MB in addition to the earlier static libraries and CUDA DLL. Free disk space reached about 756 MB before restarting the sequential work. File contents and model weights were not changed by filesystem compression.

**Started:** Resume the source-split run from epoch 1 with its original configuration, then evaluate and finalize the comparison sequentially. OneDrive was observed using roughly 5.8 GB private memory; permission to temporarily quit/restart it was requested because it affects syncing outside this project. It has not been stopped without a reply.

## 30 Complete the independent comparison and state the actual result

**Recorded:** 2026-09-17, checked at 11:33 IST. The sequential experiment pipeline completed on 2026-09-16 at 23:56 IST.

**Measured:** All nine runs have completed training and 2,000-tree evaluation. Contrast initialization/source-split Jaccards are 0.207 and 0.360, compared with 0.150 and 0.294 for the corresponding raw v3 repeats. Four crops are flagged in all three contrast runs, versus none in the three raw runs. Contrast ranking correlations are 0.867 and 0.866. This is a mixed outcome: some independent tail agreement improves, while forest-only repeatability weakens and visible border artifacts remain.

**Decision:** Retain v3 as the documented screening reference; do not claim validated anomaly-accuracy improvement. The final comparison and student-style journal are in IMPROVEMENT_RESULTS.md and outputs/improved_selection.json. Experiment labels beginning improved_ describe the planned experiment, not a proven accuracy gain.

**Progress:** Build and execute the revised notebook from the finalized results, rebuild and visually review the final PDF, then verify and package. All 11 tests have passed. OneDrive was not stopped; no permission reply was received, and the numerical work completed without that action.

## 31 Execute the readable notebook and review the final report

**Found and fixed:** The first notebook execution stopped on an absent split_seed field in legacy configurations. Early runs used seed for both training and source splitting. Added that explicit fallback without changing checkpoint contents or numerical results, then executed all 26 code cells successfully.

**Readability:** Reduced the saved-selection code cell to run, analysis, status and timestamp. The long decision narrative remains in Markdown, rather than being duplicated as a Python dictionary. Re-execute this final presentation revision before hashing it.

**PDF review:** The measured report initially produced a nearly empty extra page containing a repeated closing paragraph. Removed that repetition, re-rendered and visually inspected all 15 final pages. Tables, figures, captions, page boundaries and physical-hypothesis alternatives are readable with no observed clipping or overlap. Raw and standardized reconstruction domains are explicit.

**Remaining:** Final notebook execution confirmation, verification hashes, final progress/checklist update and ZIP integrity checks. These artifact checks do not turn the mixed scientific outcome into an accuracy gain.

## 32 Verify and package the completed second-stage work

**Completed:** Final notebook execution passed all 26 code cells without errors. All 15 final PDF pages were rendered and visually reviewed. The verification script cross-checked the saved 11-test result, strict threshold eligibility, selected heatmap files, completed runs, notebook outputs and artifact SHA-256 hashes.

**Package check:** The revised ZIP contains 353 files, approximately 334.7 MB. ZIP CRC and every archived file's SHA-256 digest passed. It includes the readable notebook, revised report, nine runs' best checkpoints and analysis evidence, source code, tests and maintained notes. Dataset inputs and the Python environment are excluded. Rebuild once more to include this completion entry and the final progress snapshot; repeat the same integrity checks.

**Scientific conclusion:** This is a completed experiment and presentation round, not a demonstrated improvement in hidden-anomaly detection. Brightness invariance improves substantially and independent flagged-set agreement improves modestly; forest-only agreement and acquisition artifacts remain concerns. The reference remains v3. A future model change should target border contamination of the learned representation and be tested for preservation of useful anomaly signals.

**Execution provenance:** Runs and notebook execution were local. OneDrive was never stopped. No repository publication, invitation or competition submission was performed.

## 33 Start the single deadline-limited border experiment

**Scope agreed:** One further experiment, architecture unchanged; reuse the saved baselines and preserve the verified submission. Use Colab if local resource failures recur. No cloud transfer has been performed.

**Protocol before results:** BORDER_EXPERIMENT_PROTOCOL.md fixes the near-black threshold (4/255), two-pixel mask expansion, nearest-valid standardized fill, model/loss/seeds/budget and promotion criteria. v5 is the matched border-policy comparator; v3 remains the submission reference. A fixed held-out sample will measure border response and three synthetic defect responses, without training or calibrating on synthetic images.

**Started:** Tests, then v6 training and evaluation sequentially. The earlier modes remain unchanged. Near-black masking can remove genuine shadows and extension can create repeated texture; full mask statistics and selected examples will be reviewed. No guaranteed detection gain is claimed.

## 34 Record preprocessing checks before judging model results

**Verified:** All 12 tests passed in 15.89 seconds. Existing raw/contrast/exact-zero behavior remains covered; the new test checks near-black border expansion, preservation of an interior dark feature and unchanged contrast output when no border is present.

**Visual sanity check:** Reviewed three previously identified border examples (outputs/border_experiment/border_examples.png). Nearest-pixel extension removes neutral-fill gaps but can stretch edge pixels into artificial stripes. This anticipated limitation is retained explicitly; no masking constants or method are changed after this preview. Promotion also requires that the selected examples are not dominated by a new artifact.

**Progress:** v6 reached epoch 3 without a local failure. The fixed defect and border-response tests remain pending completion of training. The original verified submission remains intact.

## 35 Realise the brightness problem and fix it properly — v7 gradient loss

**What I actually noticed:** After staring at the v3 heatmaps for a while, it hit me — every single one of the top flagged images was just *bright*. Not geologically weird, just bright. I ran `inspect_selected_context.py` and it confirmed it: all five were above the 99.7th brightness percentile. That's not anomaly detection, that's a photometer.

**Why the border masking didn't solve it (v4/v5):** I tried standardizing brightness in v4 and v5. v4 immediately started flagging black camera borders instead. v5 tried to mask out connected black regions but the near-black JPEG edges still leaked in. I was going in circles making the model fixate on different camera artifacts instead of actual geology.

**The actual fix I settled on:** I went back to `model.py` and found that the `loss_components` function already had the Sobel gradient calculation in there (computing `dx` and `dy`) but `gradient_weight` was set to zero. I literally just needed to turn it on. Set `gradient_weight=0.1`, kept everything else exactly the same. The idea is: if the loss function explicitly punishes the model for blurring sharp edges, the latent space has to encode edge information, and the Isolation Forest will start caring about structural novelty rather than global brightness.

**What happened:** It worked. Not perfectly, but measurably. `sample_04201.jpg` — which is only in the 22nd brightness percentile — appeared in the top 5. A genuinely dark image with unusual structural edges was flagged. The metrics also all improved simultaneously: lower MSE, higher SSIM, lower gradient error. That gave me confidence the gradient term wasn't breaking anything.

**Evidence:** `outputs/v7/history.csv`, `V7_GRADIENT_EXPERIMENT_PROTOCOL.md`, `outputs/v7/mixture_three_sigma_trees2000/selected_acquisition_diagnostics.csv`.

## 36 Push further — v8 geometric augmentations and metadata fusion

**Why augmentations:** After v7 worked, I thought about what else the model was still cheating on. The latent effective rank was 36.71, which means the 256-dimensional space still had a lot of redundant orientation-specific information. Mars doesn't have an up-direction — a crater looks the same from any rotation. But the model had only ever seen images in their original orientation. So if a crater appeared rotated 90° from usual, the model would call it unusual. That's not a real anomaly.

**The fix:** Added a simple `augment` flag to `train.py`. During training, each batch randomly gets horizontal flips (50%), vertical flips (50%), and random 90° rotations. Validation is never augmented so the metrics are still comparable. This is entirely within the rules — I'm using the same images, just flipped, which is standard practice.

**What happened with v8:** The validation MSE went up slightly (0.002678 vs v7's 0.002555), which I expected. The model can't memorize pixel positions anymore, so it has to work harder. But the latent effective rank *dropped* to 22.16 — which is actually better. Fewer active dimensions means each dimension is now encoding more meaningful structural information rather than camera-orientation details.

**The zero-flag surprise:** The plain Isolation Forest returned zero flags for v8. At first this looked like a failure. But it actually makes sense: the augmented model has a much more uniform latent space, so the calibration distribution is tighter and the threshold mechanism didn't find any extreme outliers. This is where the 5-ensemble + metadata fusion pipeline becomes essential. When the forest has more context (sun angle, season) it can still identify the 5 most structurally unusual crops, and crucially they come from 5 completely different source observations — no more SRC_154 domination.

**Evidence:** `outputs/v8/history.csv`, `outputs/v8/fusion/fused_novelty_scores.csv`, `outputs/v8/mixture_three_sigma_trees2000/diagnostics.json`, `scripts/metadata_fusion.py`.



## 37 Audit the current folder and reconcile claims with evidence

**Recorded:** 2026-09-17T16:59:22+05:30

**Request:** Analyze the entire current project before reporting. Preserve the experiments and existing deliverables while examining the rules, source, data, numerical results, notebooks, reports and packages.

**Verified:** Original input hashes and all 10,422 extracted images match the supplied archive; all images decode. Fourteen full training runs have 15 epochs. Fifteen best checkpoints including smoke load and pass CRC; saved configurations match. Main score, flag, latent-alignment and threshold-before-selection checks pass. Both ZIPs pass their internal manifests. The current test run passes 12 tests in 12.38 seconds.

**Problems identified:** Fresh-run notebook configuration omits v7 gradient weight and v8 augmentation; the newer ZIP omits those runs; resume fails on legacy checkpoints missing augment; Boolean CLI parsing is misleading. The current narrative overstates v7 improvement and describes v8 fusion incorrectly. v8 fusion has zero thresholded candidates. The newer PDF has an orphaned page. Full evidence and repair order are in FOLDER_AUDIT_REPORT.md. No underlying scientific result or existing notebook/PDF/ZIP was rewritten during this audit.

**Clarification:** v7 source-split Jaccard is 0.2941 over all crops and 0 on the 1,253 common held-out crops. An initial live update incorrectly described the difference as stale calculation; the actual issue is inadequately labeled comparison populations. Neither statistic establishes accuracy.

**Closed v6:** Ran the prespecified response diagnostics using saved checkpoints. Border response improves, but defect-response and forest-stability criteria fail. Reviewed selected panels show stretched fill artifacts. v6 is not promoted; its decision and status files now record completion. No new model was trained.

**Progress and decision:** Add an audited progress header and preserve prior entries as history. Prioritize reproducibility, accurate claims and consistent submission artifacts over further model searches. Final model selection remains inconsistent across existing deliverables and has not been silently changed by this audit.
