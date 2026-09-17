# Whole-folder audit — 17 September 2026

## Verdict

The project has a substantial, functioning experimental pipeline and intact data. **The current submission is not ready to defend as a consistent, reproducible final result.** The most urgent problems are fresh-run configuration errors, incomplete packaging, contradictory documentation and claims stronger than the measurements support. More training is not the first priority.

This audit preserves the existing source, notebooks, PDFs and ZIPs. It adds an audit script, machine-readable evidence, this report and progress notes. It also completes the already specified v6 response diagnostics; no new model was trained during this audit.

## Scope and checks performed

- Revisited the three supplied competition documents, including the requirements extracted previously from their tables and figures.
- Inventoried 11,061 project files before adding the audit outputs, excluding the virtual environment, Git internals and caches. This includes all dataset images and generated artifacts. Third-party library internals were not reviewed as project code.
- Reviewed the core model, preprocessing, data, training, evaluation, calibration, visualization, experiment and delivery paths. Parsed all 37 current project Python files, including the new audit script; no syntax failures.
- Recomputed original-input hashes. Compared both extracted metadata files and every extracted JPEG byte-for-byte with the supplied archive; decoded all 10,422 images again.
- Inspected configurations, histories, split manifests, best checkpoints, saved latents, scores, thresholds, selected-image lists and heatmap availability across every saved run.
- Loaded all 15 best checkpoints, including the one-epoch smoke test; checkpoint configurations match their configuration files and their ZIP CRC checks pass. Fourteen full runs have 15 epochs each.
- Ran the existing test suite: **12 passed in 12.38 seconds**. These tests do not cover all delivery and configuration contracts identified below.
- Inspected all four notebook files, their visible code and recorded execution state. Examined both PDFs' text; rendered all 15 pages of the newer Analysis report and inspected its page flow and selected-image panels. The unchanged Improved PDF retains its earlier verified hash and visual-review record.
- Checked both submission ZIPs against every internal manifest hash and their CRCs. Integrity passes; scientific consistency and completeness are separate checks.

Evidence: `outputs/folder_audit/inventory.csv`, `checks.json`, `checkpoints.json`, `v7_heldout_stability.json`, and the extracted notebook/report sources. Reproduce most numerical checks with `python scripts/audit_current_folder.py`.

## Findings requiring attention

### 1. High priority — the packaged notebook cannot reproduce v7/v8 correctly from scratch

`scripts/build_submission_notebook.py:101` lists v7 and v8 without their gradient or augmentation settings. The `TrainConfig` call at line 106 supplies neither setting. The generated `notebooks/Mars_HiRISE_Submission.ipynb` therefore defaults both models to `gradient_weight=0` and `augment=False` in a fresh environment.

The saved v7 actually used gradient weight 0.1. Saved v8 used gradient weight 0.1 plus augmentation. Existing completed directories let the notebook skip training, so its 15 executed cells and zero errors do not establish fresh-run correctness. Cached-run reuse also does not validate the requested configuration against the saved one.

The separate `output/submission/Mars_HiRISE_Submission.ipynb` has the correct experiment settings, but is a different nine-cell notebook, not the notebook included by the current package builder. It embeds a robustness script absent from the current `scripts/` folder, and its fresh-run experiment list does not train the robustness runs that script reads.

**Repair:** designate one canonical notebook; derive every run specification from its saved configuration; validate cached configurations; verify an isolated fresh-run path with a short smoke configuration before an expensive full reproduction.

### 2. High priority — the newer package selects v7 but omits its experiment artifacts

`NSSC_2026_Mars_Analysis.zip` contains 244 manifest-listed files, approximately 192.0 MB, and a final selection naming v7. However, `scripts/package_submission.py` only collects run directories for v1, v2, v3, robust_seed and robust_split. The ZIP contains neither v7 nor v8 checkpoints or their complete run evidence. Their inclusion in notebook outputs or comparison tables does not replace the missing artifacts.

The 353-file, 334.7 MB Improved ZIP remains an intact historical nine-run/v3 package. Its notebook and PDF still match `outputs/improved_verification.json`. It does not include the later v6–v8 work. Seventeen of its archived files differ from today's workspace; that is version drift, not ZIP corruption.

**Repair:** package the selected run and every reported experiment explicitly, then verify membership as well as hashes. Keep the historical ZIP separate and label its scope.

### 3. High priority — the claim that v7 solved brightness bias is unsupported

The measurable reconstruction improvement over v3 is small but real on the common validation set:

| Metric | v3 | v7 | Change |
|---|---:|---:|---:|
| Validation MSE | 0.002598794 | 0.002555233 | 1.68% lower |
| Validation SSIM | 0.673810 | 0.674583 | +0.000773 |
| Gradient error | 0.013958476 | 0.013886247 | 0.52% lower |
| Top-five median brightness percentile | 99.904 | 99.952 | Remains extremely bright |
| Minimum forest-repeat flag Jaccard | 0.667 | 0.417 | Lower agreement |

Four of v7's five selected crops remain above the 99.8th brightness percentile; five of its six flags exceed the 99th percentile. The additional darker crop, sample_04201.jpg, has conspicuous black side borders and residual error at those boundaries as well as terrain detail. It warrants investigation but does not prove that the forest now detects geological structure rather than acquisition effects.

`outputs/final_selection.json`, the newer PDF, and the journal use claims such as “broke the brightness confound,” “proved,” and “true structural novelty.” The available evidence does not support those claims. No hidden anomaly labels exist in the supplied files.

**Repair:** describe v7 as a modest reconstruction improvement with unresolved brightness, border and stability limitations. Do not equate reconstruction residuals with attribution of the forest score.

### 4. High priority — v8 fusion results contradict the claimed five candidates

Recomputing eligibility from `outputs/v8/fusion/fused_novelty_scores.csv` and its own `calibration.json` gives **zero** scores above **T = 0.5542073332**. The image-only v8 detector also has zero flags. Decision-log entry 36's statement that fusion identifies five structurally unusual crops is therefore not supported by the thresholded output. Selecting five merely by rank would violate the stated competition procedure.

The changelog describes metadata being inserted before decoding. The actual autoencoder has no metadata input. `scripts/metadata_fusion.py` concatenates sun angle, resolution and encoded season **after encoding**, before a five-forest ensemble. It uses no latitude and does not condition reconstruction. With four season categories, its vector has **262**, not the checklist's 263, dimensions.

The script also changes forest ensembling at the same time as adding metadata. A comparison with a single image-only forest cannot isolate metadata's contribution. The five forests and fitted encoders/scaler are not persisted by this script. The v7 fusion CSV has no accompanying fusion calibration file.

**Repair:** report the actual zero-flag outcome; compare an image-only ensemble with an otherwise identical fused ensemble; save the transformation and detector artifacts; show candidate-set change. Optional bonus marks are for judges to award, not something the checklist can certify.

### 5. Medium priority — robustness tables mix populations without clearly labeling them

v7 versus its initialization repeat has all-crop flag Jaccard **0.0667**. Against the source-split repeat, it is **0.2941 across all crops** but **0.0 on the 1,253 crops held out from both encoders**. In that common-held-out subset v7 flags none and the repeat flags one. These are different comparisons, not contradictory arithmetic. There is one all-crop flag shared across all three v7 runs: sample_08288.jpg.

The separate `v7_robustness/comparison.csv` mixes the all-crop initialization result with the common-held-out split result; its top-five overlap columns use the full dataset. The embedded comparison script reveals this, but the table lacks population/count columns. The newer report still foregrounds v3's robustness summary while selecting v7.

**Repair:** label each population and denominator, show selected-run sensitivity prominently, and keep overall ranking correlation separate from tail agreement. An earlier live audit update called the table stale; this closer inspection corrects that assessment.

### 6. Medium priority — the experiment narrative does not match the implementation

- v7's protocol specifies `0.8 MSE + 0.1 (1-SSIM) + 0.1 Gradient`; the code and saved settings implement **0.9 MSE + 0.1 (1-SSIM) + 0.1 Gradient**. There is no requirement that these weights sum to one, but the formula must be accurate.
- The changelog says `gradient_weight=0.5`; checkpoints record **0.1**.
- The gradient calculation is adjacent-pixel finite differences, not a Sobel convolution.
- The changelog calls v6 a variance-mask experiment that was abandoned. Saved v6 is a completed 15-epoch near-black border-fill experiment, with 78 flags.
- A lower latent effective rank does not prove more meaningful structural dimensions; the v8 narrative treats that interpretation as established.
- `README.md` and `improved_selection.json` select v3, while `final_selection.json` and the newer notebook/PDF select v7. The progress file omits all v7/v8 runs and still says v6 is incomplete.

**Repair:** preserve the actual recorded experiment, distinguish hypotheses from measured outcomes, and generate current summaries from one authoritative run registry. Do not rewrite a pre-experiment protocol to conceal a deviation.

### 7. Medium priority — training resume and Boolean parsing have regressions

`mars_anomaly/train.py:70` reads `prior['augment']` directly. Older checkpoints, including v7, lack that newly introduced key, so their resume path raises `KeyError` even when the intended historical setting is false. Existing defaults already handle missing preprocessing and split seed; augmentation needs equivalent compatibility treatment.

At line 153, generating every CLI argument with `type=type(value)` means `--augment False` evaluates as true (`bool('False')`). This can silently run a different experiment from the command's apparent intent.

**Repair:** use a backward-compatible default and a real Boolean CLI action/parser, with targeted resume/configuration tests. The current 12 passing tests do not exercise these cases.

### 8. Medium priority — report and verification quality regressed

The current Analysis report has 15 pages. Page 9 contains only the orphaned word “payload.” plus its footer. A long all-pairs stability table consumes several preceding pages. Its main conclusion promotes v7, but the architecture/loss discussion and prominent limitations still focus on earlier runs. Its fusion discussion says fusion is omitted, while other deliverables claim completion.

`outputs/verification.json` now contains only `pdf_visual_review_passed: true`; it does not bind a notebook/report hash, page count or dated checks. The older package builder accepts that flag. By contrast, the unchanged Improved artifacts have explicit hashes and a dated record.

**Repair:** regenerate after resolving the scientific narrative, shorten the main comparison table, move exhaustive pairs to an appendix, render and inspect every final page, and record hashes for the exact reviewed files.

### 9. Medium priority — some physical hypotheses overreach their evidence

New entries in `geological_hypotheses.json` assign resolution units and interpret sun-angle semantics that the supplied metadata does not define. They attribute novelty to specific pixels or metadata channels without a corresponding attribution experiment. One entry describes northern spring as southern late summer; the hemisphere-season interpretation needs correction. The RSL water-flow wording is also not supported by its cited article, which emphasizes evidence for granular dry flows while retaining uncertainty: [NASA/JPL discussion](https://www.jpl.nasa.gov/news/recurring-martian-streaks-flowing-sand-not-water/).

Some of these additional hypotheses are not used by the current v7 selection, but they are embedded in the notebook and remain part of the deliverable record.

**Repair:** describe visible shape and residual location first, keep competing physical explanations tentative, remove unsupported units and causal attribution, and match each citation to the claim it actually supports.

## Completed v6 diagnostic result

The fixed, previously specified test used 256 validation crops and unchanged saved models, forests and thresholds. Border-strip score sensitivity at the 95th percentile fell from **0.030759 (v3)** and **0.015707 (v5)** to **0.002578 (v6)**. However, v6's score increased for only **30.1%** of bright-square, **60.9%** of dark-square and **32.4%** of checker injections, failing the predeclared 70% criterion. Its minimum forest Jaccard, **0.5918**, also fails the v3 baseline criterion of **0.6667**.

All 10,422 mask fractions were audited: 2,577 crops have some masked pixels, three exceed 50% masking, and none are entirely masked. Reviewed v6 selections exhibit artificial stretched stripes in several filled regions. No model's sampled original or modified image crossed its threshold in this particular synthetic test. These diagnostics measure response, not detection accuracy or synthetic recall representative of the hidden payload.

**Decision: do not promote v6.** Its numerical criteria fail and the visual artifact concern remains. See `outputs/border_experiment/response_summary.csv`, `mask_summary.json` and `decision.json`.

## What is sound

- Original files are unchanged. Every crop matches the archive, decodes as 227 × 227 grayscale, and joins to supplied metadata; no exact decoded-pixel duplicate groups were found.
- Source IDs stay disjoint in all saved run manifests. This reduces shared-observation leakage; it does not prove absence of geographic overlap or near duplicates.
- The core network is from scratch, has a fixed one-dimensional latent vector and no pretrained feature path or bypass. The primary forest uses training latents, scores with `-score_samples`, and calibrates independently of a desired anomaly count.
- Saved primary-run score flags match their thresholds and diagnostics. All selected lists equal the thresholded top five, latent filenames align with scores, latent values are finite, and selected heatmaps exist. v8 correctly has an empty primary list.
- Training histories, best weights and many negative results are retained. Atomic checkpoint writing has a meaningful failure-recovery test.
- Gaussian-mixture calibration, source bootstrap and uncertainty caveats provide a defensible descriptive screening rule. They do not establish a real false-positive rate; model-order limits, correlated crops and contaminated calibration remain limitations.

## Prioritized completion plan

1. **Freeze the scientific claims.** Select one explicit reference and report its actual limitations. v7 may be described as a reconstruction ablation, but its superiority as an anomaly detector is not established. The historical v3 package remains a useful fallback, not a proven accurate detector.
2. **Repair reproduction and packaging.** Fix fresh-run settings, cached-config validation, resume compatibility and Boolean parsing; include selected-run evidence. Use the already trained checkpoints rather than retraining everything.
3. **Synchronize deliverables.** Resolve v3/v7 and fusion contradictions across the notebook, report, selection JSON, changelog, progress and checklist. Keep historical versions clearly labeled.
4. **Verify the final artifact set.** Run focused contract checks, execute the canonical notebook, inspect the final PDF and validate the final ZIP's membership and hashes.
5. **Complete competition logistics.** The private repository, five organizer collaborators and team details remain pending. Local files cannot establish that submission has occurred. Do not upload private data or publish the repository accidentally.

The immediate opportunity is a credible, reproducible submission built from existing experiments. Another broad model search would not address the current blockers.
