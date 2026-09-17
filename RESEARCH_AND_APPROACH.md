# Research and implementation strategy

## Recommendation

Start with a compact convolutional autoencoder trained from scratch at the original 227 by 227 resolution. Feed its fixed-length latent vector into Isolation Forest, using `novelty = -score_samples(z)`. Separate source observations when validating reconstruction and calibrating the anomaly boundary. Compare a pure MSE baseline with structural reconstruction loss and a third change motivated by the actual results. Use a skewness-adjusted statistical fence with source-level bootstrap uncertainty as the initial threshold procedure.

This is the best-supported starting strategy for the supplied rules, data and available hardware. It is not a claim of a globally optimal model or measured anomaly accuracy. The hidden labels make recall, precision, F1 and AUROC unavailable. Reconstruction quality, noncollapsed representations, rank stability, threshold sensitivity and physical interpretation are the evidence we can legitimately assess.

## 1 Complete input inventory

The initial folder contains exactly four supplied files. There is no existing source code, notebook, project configuration or AGENTS.md in this folder.

| Input | Findings | Consequence |
|---|---|---|
| `DA PS 26'.docx` | Four phases, 100 points, plus optional metadata fusion worth 2 points | Implement the specified pipeline and preserve the rubric numbering |
| `DATA_DESCRIPTION.docx` | Unlabelled grayscale crops with three kinds of injected content anomalies | No normal-only training set is available; never label the training split as clean |
| `RULES AND REGULATIONS.docx` | AI agents allowed; notebook with outputs, PDF and genuine v1-v3 changelog required | Keep recorded experiment evidence and a private submission workflow |
| `DATASETS-20260915T193216Z-1-001.zip` | Two CSVs and a nested image ZIP | Extract into a separate data directory; keep originals intact |

`scripts/audit_inputs.py` reads text from the Word document bodies and ancillary text parts, extracts embedded figures, decodes every image, checks keys and computes pixel statistics and hashes. Audit details and original-file SHA256 hashes are saved in `outputs/audit/audit.json`. The illustrated problem-statement montage contains contextual imagery rather than additional technical requirements. The other embedded image is competition branding.

### Measured dataset facts

- 10,422 JPEG files, all mode L, all 227 by 227 pixels, all decoded successfully.
- 10,422 unique crop-index filenames and 172 unique source identifiers.
- Complete many-to-one metadata joins, with no missing fields.
- No exact decoded-pixel duplicates. Near-duplicate and overlapping crops are still possible; exact hashing cannot rule them out.
- Source sizes range from 1 to 278 crops, with median 50.5 and mean 60.59.
- Latitude has 24 distinct values spanning -90 to 90. Longitude has only 2 values, 0 and 180. The 172 sun angles range from approximately 32.10 to 85.68. There are 4 season categories and 3 resolution values, 0.25, 0.5 and 1.0.
- Supplied metadata does not explicitly define sun-angle semantics or resolution units. Avoid silently treating sun angle as incidence/elevation, or treating resolution as metres per pixel without confirmation from the organizers.
- A reproducibly sampled 36-image contact sheet includes smooth and textured terrain, crater-like structures, differing contrast and black image boundaries. These are visual observations, not anomaly labels.

The approximate 10,000 crops and 180 observations in the prose are consistent with the measured counts. No image should be discarded merely because its appearance seems unusual. In particular, black boundaries can be acquisition footprints and should not automatically be called injected sensor glitches.

## 2 Rules that govern the solution

1. All trainable image representations must start from random weights. No pretrained model, VGG/ResNet perceptual loss, external image pretraining, or transfer learning.
2. The primary detector must be Isolation Forest on a fixed-length 1D image latent vector. Reconstruction error explains results; it does not replace the required final novelty score.
3. Higher final novelty score means more anomalous. Use the negative of scikit-learn `score_samples`; do not use `predict` to determine the final set.
4. `contamination='auto'` may configure the library but does not set our final boundary. Choose the boundary from the observed score distribution independently of an expected anomaly count.
5. Select at most five images only after thresholding, in descending novelty order. If zero to four exceed the threshold, retain that result.
6. Source metadata comes from the supplied CSV and index. Catalog scraping is explicitly out of scope.
7. t-SNE or UMAP is a qualitative diagnostic. A visual cluster is not a confirmed geological class, and a dense projection alone does not establish encoder collapse.
8. v1, v2 and v3 must record a real symptom, a diagnosis, a change and its observed outcome. A planned experiment is not a completed iteration.
9. Submission must contain the executed, question-labelled notebook, PDF report and engineering changelog. Keep the GitHub repository private until results are announced.
10. The rules list the required GitHub collaborators: SarthakXSingh09, R15HV, Shivam3473, aditohates-bugs and Dhairya646. This document records the requirement; it does not claim a repository or invitations exist.

The rules mention location analysis as Phase 2.5 in the mark-distribution table, while the detailed problem statement calls it 2.3. Use the detailed numbering and mention the discrepancy in the submission checklist.

## 3 Research findings and their application

### Reconstruction is not a reliable anomaly oracle

Autoencoders can reconstruct unfamiliar images well, and high error can also arise from difficult normal texture. Bouman and Heskes show failure cases for both linear and nonlinear models [1]. Consequently, minimizing reconstruction error alone cannot select the most accurate anomaly detector. We retain the required latent Isolation Forest score and describe error heatmaps as reconstruction discrepancies, not causal explanations of the forest.

### Structure loss is useful but needs a controlled comparison

SSIM compares local luminance, contrast and structure [2]. It uses image statistics rather than pretrained neural features, so it meets the competition restriction. MSE anchors absolute brightness; SSIM can encourage preservation of terrain texture. The loss weights are experiment settings, not universal constants. Record raw component magnitudes and compare common validation MSE, SSIM and gradient error across every version.

### Decoder design can create its own artifacts

Uneven overlap in transposed convolutions can produce checkerboard patterns. Resize followed by convolution separates interpolation from learned filtering and is supported by Odena, Dumoulin and Olah's analysis [3]. Use explicit decoder sizes 15, 29, 57, 114 and 227 so there is no accidental output-size mismatch. Avoid encoder-to-decoder skip connections that could let reconstruction bypass the fixed-length bottleneck.

### Correlated crops require grouped validation

scikit-learn's grouped splitting tools keep a group from crossing a split boundary [4]. Applying this to `source_image_id` is a project-specific inference: crops from a common observation may share terrain and imaging conditions. A random image split could therefore flatter reconstruction generalization. Grouping does not prove geological independence because distinct sources might still overlap.

### Score orientation and contamination must be explicit

The official Isolation Forest documentation states that lower `score_samples` values indicate greater abnormality [5]. Its contamination setting controls a decision offset, separate from the underlying score. Negating the score and applying our own boundary avoids an assumed outlier fraction. Positive affine standardization of each coordinate is not expected to fundamentally fix an axis-aligned Isolation Forest; latent geometry and redundant dimensions matter more. Fit preprocessing on training rows only, and use scaling primarily for visualization or metadata fusion.

### Skewed scores need a skew-aware boundary

Hubert and Vandervieren's adjusted boxplot extends the upper quartile fence using the medcouple, a robust skewness statistic [6]. This is a defensible initial descriptive rule when a score distribution is skewed. It is not a formal false-discovery-rate guarantee and it can be too conservative or unsuitable for strongly multimodal distributions. Save the histogram, empirical CDF, sorted scores, fence, bootstrap interval and sensitivity analysis so its suitability can be inspected.

### Conformal and extreme-value methods have assumptions

Conformal calibration offers appealing guarantees under appropriate exchangeability conditions. Research on contaminated reference data shows the importance of contamination assumptions [7]. Here reference data is unlabelled and crops are grouped, so a naive per-crop conformal percentile would not justify a clean-normal false-positive claim. Also, choosing a global score percentile risks implementing the prohibited fixed-fraction cutoff in disguise.

Peaks-over-threshold extreme-value modelling is a useful secondary analysis if enough independent tail observations and stable fits exist. It introduces a tail-entry threshold, fitted distribution and extrapolation assumptions. A goodness-of-fit test must account for parameters being estimated from the same observations; SciPy documents simulation-based approaches [8]. Do not select whichever tail rule returns a visually appealing number of candidates.

### Projection plots can distort the latent structure

Research comparing t-SNE, UMAP and related methods describes tradeoffs in neighborhood and global-structure preservation [9]. Run a seeded projection at two neighborhood scales, show novelty, source and imaging conditions on the same coordinates, and corroborate it with latent variances and covariance effective rank. Never feed the two-dimensional projection into the primary detector.

## 4 Proposed architecture and training

### Alternatives considered

| Approach | Advantage | Decision for this project |
|---|---|---|
| Compact deterministic convolutional AE | Simple optimization, explicit spatial compression, low memory, easy controlled ablations | Primary baseline and first structural-loss experiment |
| Variational AE | Explicit latent regularization and generative sampling | Valid alternative, but KL weighting and posterior collapse add tuning burden; reserve for evidence of poor latent organization |
| U-Net with encoder-decoder skip connections | Sharp reconstruction | Avoid for the primary design because skips can weaken pressure to encode useful information in the vector used by the forest |
| Very large encoder or 512-dimensional latent | More reconstruction capacity | Defer until 128/256 comparisons establish a capacity problem; stronger reconstruction can also represent unwanted content |
| Robust/downweighted training | May reduce the influence of contaminated samples | Later ablation only; weighting can suppress rare valid terrain, and trimming an assumed fraction would be unjustified |
| Scratch self-supervised pretraining plus AE | Potentially stronger representations without external weights | Additional complexity and augmentation assumptions; establish the required end-to-end baseline first |
| Pretrained PatchCore, CLIP, DINO or VGG perceptual features | Strong off-the-shelf visual representations in many settings | Excluded by the explicit pretrained/transfer-learning rules |
| Reconstruction-error-only or raw-pixel detector | Easy to implement | Insufficient for the mandated latent Isolation Forest pipeline; use reconstruction error only as a diagnostic |

The deterministic AE versus VAE choice is an engineering recommendation based on task constraints and experiment cost, not an empirical claim that AEs always detect anomalies better. The VAE option remains available if measured latent diagnostics justify it.

### Initial architecture

Input is `[batch, 1, 227, 227]`, rescaled globally from uint8 to [0,1]. Five strided 3 by 3 convolutions produce spatial sizes 114, 57, 29, 15 and 8, with channels 16, 32, 64, 96 and 128. Group normalization avoids relying on large batches. SiLU activations follow each block. Flatten the final 128 by 8 by 8 tensor and project to a 128-dimensional vector. A linear layer expands it back into the decoder's seed grid. Resize-convolution stages restore the original dimensions, ending with a sigmoid image output.

Retain spatial organization until the flatten-and-project layer: global average pooling alone could lose the location of a splice or localized hardware artifact. No skip connection bypasses the vector. Batch size 16 is the conservative local starting point; increase only after measuring memory. Cloud GPU runs can use larger batches and longer schedules without changing the data protocol.

### Why 128 first

There are 51,529 input values, giving approximately 403 input values per latent coordinate at dimension 128. Dimensions 256 and 512 reduce that ratio to approximately 201 and 101. Larger embeddings may preserve more texture but can also represent injected content and introduce redundant axes for Isolation Forest. Compare 128 and 256 first under the same split, seed and training budget; try 512 only if capacity diagnostics warrant it. Compression ratios alone do not determine detection quality.

### Preprocessing and augmentations

- Keep image brightness and boundary information. Do not apply per-image min-max normalization, histogram equalization or aggressive cropping.
- Begin without augmentations so the first result has a clear interpretation. A later rotation/flip experiment should explicitly consider illumination direction and spatial artifacts.
- Avoid training with synthetic terrestrial pictures or invented anomaly labels. Do not tune against visually guessed payload identities.
- Preserve original images and use filename/index mapping solely to join records. Neither filename sequence nor source identifiers become latent detector features.

### Data protocol

Assign approximately 70% of source IDs to training, 15% to validation, and 15% to calibration using a fixed seed. Fractions refer to source counts; image counts will differ. Save the exact split manifest before training. Fit the encoder and primary Isolation Forest on training crops, select checkpoints from validation reconstruction diagnostics, and estimate the statistical fence from calibration scores. Score every supplied crop and record its split so in-sample and held-out results remain distinguishable.

This is a held-out calibration protocol, not a labelled test of anomaly accuracy. Do not refit the final encoder or forest after calibration without recalibrating. A later cross-fitting experiment can assess whether flagged images are stable when their source is held out, but raw scores from different forests should not simply be pooled into one threshold.

### Genuine iteration sequence

| Version | Starting experiment | Evidence needed before the next change |
|---|---|---|
| v1 | 128 dimensions, MSE | Train/validation curves, fixed reconstruction panel, texture error, effective rank |
| v2 | Same architecture; compare MSE plus SSIM if structural smoothing is observed | Common metrics and identical examples; note whether brightness or geometry worsens |
| v3 | Change one diagnosed limitation: capacity, gradient loss, or detector stability | Controlled comparison against v2; failed changes remain in the journal |

Do not prewrite a successful v2 or v3 story. If v1 does not show the expected smoothing problem, revise the experiment motivation using actual evidence. The model with best reconstruction is not automatically the final detector.

## 5 Isolation Forest and statistical threshold

Start with 400 trees, `max_samples=256`, `max_features=1.0`, fixed seed and `contamination='auto'`. Compare `max_samples=512` and another two seeds as stability diagnostics when resources allow. More trees reduce sampling variability but cannot repair a poor representation.

Let s denote the calibration novelty scores, Q1 and Q3 their quartiles, IQR = Q3 - Q1, and MC their medcouple. The initial upper fence is:

```
T = Q3 + 1.5 * exp(3 * MC) * IQR,  if MC >= 0
T = Q3 + 1.5 * exp(4 * MC) * IQR,  if MC < 0
flag(x) = novelty(x) > T
```

Bootstrap entire calibration source groups to estimate threshold uncertainty. Report the central fence and its interval; do not silently choose an upper bound to obtain fewer images. Measure how often each crop is flagged across bootstrap thresholds. This measures conditional threshold stability, not the probability that the crop is a real anomaly. Forest and encoder variability require separate experiments.

Inspect the suitability of this rule before finalizing. Strong multimodality, severe skew, degenerate IQR, too few calibration sources, or a boundary beyond observed support must be documented. A fence above all observed scores legitimately yields zero flags. Never lower it just to populate five heatmaps. Compare ordinary Tukey and robust median/MAD fences as sensitivity checks rather than picking their most appealing anomaly count. State that these are descriptive screening boundaries with no guaranteed recall or false-alarm rate.

## 6 Explanation and metadata analysis

For each of up to five thresholded images, display original, reconstruction, absolute-error map and overlay. Use [0,1] for originals and reconstructions, and one error scale shared across the selected examples. Include the actual error maximum, novelty score, threshold and source ID. A separately rescaled map may help locate detail but must be labelled as such.

Physical hypotheses must consider both unusual terrain and acquisition effects. A straight error boundary could be a splice, a crop footprint or a lighting transition. Diffuse error could reflect texture, contrast, blur or non-Martian content. Do not claim a heatmap proves the forest used those particular pixels.

Join metadata with explicit cardinality validation. Present both crop-weighted flag rates and source-level summaries because source sizes differ greatly. Compare flagged rates with sun angle, latitude, season and resolution. Account for multiple exploratory comparisons and avoid causal language. Longitude supports only a two-value descriptive comparison; a detailed geographic map would exaggerate the available information. Circular encodings of longitude or season are only useful if the variable semantics justify them. Treat season as categorical unless its definition is supplied.

Optional metadata fusion comes last: normalize numeric fields using training data, one-hot encode categorical season, exclude source ID, then concatenate with the latent vector. Report flag-set changes against the primary image-only pipeline. Do not spend the main experiment budget chasing the optional two points.

## 7 Compute and deliverables

Local hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM. The original environment lacks PyTorch and scikit-learn, so project-local dependencies are required. The user also authorized Colab or Kaggle. Provide a portable notebook that uses `/content` on Colab, `/kaggle/working` on Kaggle and the project root locally, with configurable paths and saved checkpoints.

Cloud access requires a signed-in account and an allocated GPU; neither is assumed. Do not buy compute automatically. Checkpoint after every epoch, retain best validation weights and training state, and write the experiment configuration, seeds, library versions, hardware and timing. A cloud disconnection should not erase the only experiment evidence.

Priority order: establish and test the data/model/score contracts; run a real baseline; use its evidence to run v2 and v3; select and assess the statistical boundary; inspect latent projections and top flagged heatmaps; finish the PDF and executed notebook; prepare the private submission. Optional fusion and a broader bottleneck sweep come after those deliverables.

## Sources

Research checked on 16 September 2026. The provided competition documents are authoritative for task requirements. External sources inform method choices, not competition permissions.

1. Bouman and Heskes, *Autoencoders for Anomaly Detection are Unreliable* (2025): https://arxiv.org/abs/2501.13864
2. Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity* (2004), author resource: https://www.cns.nyu.edu/~lcv/ssim/
3. Odena, Dumoulin and Olah, *Deconvolution and Checkerboard Artifacts* (2016): https://distill.pub/2016/deconv-checkerboard/
4. scikit-learn, GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
5. scikit-learn, IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
6. Hubert and Vandervieren, *An adjusted boxplot for skewed distributions* (2008), author-hosted paper: https://wis.kuleuven.be/statdatascience/robust/papers/2008/hubertvandervieren_adjustedboxplot_csda_2008.pdf/@@download/file/HubertVandervieren_AdjustedBoxplot_CSDA_2008.pdf
7. Bashari, Sesia and Romano, *Robust Conformal Outlier Detection under Contaminated Reference Data* (ICML 2025): https://proceedings.mlr.press/v267/bashari25a.html
8. SciPy, goodness_of_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html
9. Wang et al., *Understanding How Dimension Reduction Tools Work* (JMLR 2021): https://www.jmlr.org/beta/papers/v22/20-1061.html

## Measured follow-through

The planned research is now implemented and tested in five completed training runs. Read `RESULTS.md` for the final comparison, `ENGINEERING_CHANGELOG.md` for the actual iteration outcomes, and `DECISION_LOG.md` for the chronological rationale and defense explanations. The original adjusted-boxplot calibration failed to describe the multimodal score distribution usefully; `THRESHOLD_DECISION.md` records the fixed mixture-envelope replacement.

The reference run is v3, but independent initialization and split checks reveal poor exact candidate agreement. All five reviewed reference candidates are among the brightest 0.3% of crops. The outcome supports a documented screening prototype, not a validated hidden-anomaly detector or a claim of global optimality. Future improvements are listed separately from completed work in `RESULTS.md`.
