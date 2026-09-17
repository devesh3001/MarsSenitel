# Project defense guide

Read `CURRENT_PROGRESS.md` before presenting. It distinguishes finished experiments from pending work. Use `RESULTS.md` for the first-stage measurements and `IMPROVEMENT_RESULTS.md` for the completed second-stage comparison. This guide describes the decisions and their limits; it does not supply hidden ground truth.

## A clear opening

“I treated this as an unsupervised screening problem. I first translated the competition rules into implementation constraints, audited every crop and metadata join, and separated whole source observations before training. I trained a convolutional autoencoder from scratch, scored its latent vectors with Isolation Forest, and calibrated a threshold without assuming how many anomalies exist. I then changed one main model choice at a time and checked whether the resulting candidates were reproducible. The selected images are investigation candidates, not verified discoveries.”

## Suggested 15-minute presentation

| Time | Explain | Evidence to show |
|---|---|---|
| 0–2 min | Problem, rules, data integrity and no hidden labels | Audit summary and source-document requirements |
| 2–4 min | Source-disjoint partitions and architecture | Split counts and model diagram/shapes |
| 4–6 min | v1 → v2 → v3: symptom, single change, outcome | Common validation curves and fixed reconstruction panels |
| 6–9 min | Score orientation, failed initial threshold and revised rule | Score histogram, formula and bootstrap interval |
| 9–12 min | v4/v5 and independent repeats: benefit and failure | Illumination response, border example, rank correlations AND candidate overlap |
| 12–14 min | Selected examples and competing explanations | Heatmaps and supplied metadata |
| 14–15 min | Limitations and next experiments | Explicit claims we can and cannot support |

## Explain the approach in plain language

1. **Audit first.** Broken images or bad joins can look like anomalies. All 10,422 crops decoded correctly; exact decoded-pixel duplicates were absent. Similar neighboring crops are still possible.
2. **Keep sources together.** Crops from one observation share terrain and acquisition conditions. Splitting them randomly can give an unrealistically easy validation set. The main split separates 120 training, 25 validation and 27 calibration source IDs.
3. **Learn a compact description.** The encoder maps 51,529 pixel values to a 128- or 256-number vector. All weights start randomly, and the decoder must reconstruct through that vector.
4. **Separate reconstruction from novelty.** The forest uses the full latent vector. Reconstruction error is reported for interpretation, not substituted for the required novelty score.
5. **Calibrate the boundary.** Fit the score distribution on calibration sources. The fixed formula is `max(component mean + 3 × component standard deviation)`. The number of flags follows from that formula.
6. **Check sensitivity.** Refit calibration after resampling whole sources, repeat the forest seeds, and separately repeat training initialization and source splitting. Report disagreement rather than hiding it.
7. **Interpret after selection.** Inspect at most five crops only after thresholding. Describe visible evidence, a plausible physical explanation and an alternative.

## Questions and defensible answers

### Why not use a pretrained vision model?

The rules prohibit pretrained weights and feature extractors, including networks used inside a perceptual loss. SSIM uses image statistics directly and satisfies that constraint. A larger pretrained model would not be a valid comparison under these rules.

### Why this autoencoder rather than a VAE?

A deterministic convolutional autoencoder is a controlled starting point with a transparent fixed vector. A VAE adds distribution and regularization choices that also require validation. This study does not establish that a VAE is worse; it prioritizes measurable changes within the available budget.

### Why did v2 improve SSIM but worsen MSE?

The training objective changed. SSIM rewards local structural similarity, while MSE penalizes squared pixel differences. v2 raises SSIM from about 0.6611 to 0.6717 at an approximately 1.7% MSE cost. Report both rather than comparing differently scaled training losses.

### Is doubling the vector automatically better?

No. v3 improves the common reconstruction metrics, but has about 84% more parameters and still blurs fine detail. Better reconstruction can even make some anomalies less distinguishable. Final selection must also consider candidate stability and the absence of labels.

### Why replace the adjusted boxplot?

Show the original result first: the threshold was 0.87484, above the maximum score of 0.55648. The motivation was the multimodal distribution and excessive single-skewness expansion, not merely that the rule found zero crops. The original outputs remain available. The replacement is a descriptive model with its own limitations.

### Does three sigma guarantee a false-positive rate?

No. For each fitted Gaussian, the envelope is at least three standard deviations above its mean. That bounds the fitted mixture's upper tail by about 0.00135. It does not establish the real false-positive rate in correlated, unlabelled, potentially contaminated data. Source bootstrap uncertainty is conditional on the chosen trained model and forest.

### Why only one to four components?

It is a bounded, transparent model-selection search, not proof that the true distribution has at most four populations. In v3, four components beat three by only about 0.85 BIC units and lie at the search limit. This is a weakness to disclose. A future study should test alternative density families and source-weighted calibration under an independent protocol.

### Why are some selected crops from the training split?

The task screens the supplied dataset, so all crops receive scores. A training crop can still be unusual within the learned population. Its selection is an in-sample screening result, not evidence of generalization. Validation and calibration counts are reported separately, and the training set is never called clean-normal.

### Why can rank correlation be high while selected candidates disagree?

Rank correlation summarizes thousands of scores. The flagged set depends on a small tail and a fitted boundary. Small changes near that boundary can alter most candidates without changing the overall ranking much. That is why both Spearman correlation and Jaccard overlap are reported.

### What do the heatmaps explain?

They show where the reconstruction differs from the input. They do not directly explain which pixels caused the Isolation Forest score. Narrow streaks can have low average reconstruction error yet still produce an unusual latent vector.

### Did you find the hidden anomalies?

We cannot verify that without the withheld labels. The reviewed v3 examples are bright scenes, and three share SRC_154. Ordinary dust or frost, relief, exposure and processing are plausible explanations. A high score or bootstrap flag frequency is not a probability of an injected object.

### Is this the optimum solution?

It is a documented choice among the tested configurations. There is no demonstrated global optimum or validated detection-accuracy ranking. The strongest next steps are independent evaluation, better source/brightness controls and longer controlled learning-curve experiments, with every change recorded before using its candidate count as evidence.

## Keep these claims separate

- **Measured:** reconstruction quality, latent variance, threshold values, candidate counts and repeatability.
- **Hypothesized:** physical causes of unusual appearance and acquisition-related explanations.
- **Unknown:** true anomaly labels, recall, precision, F1, false-positive rate and whether any selected crop contains the hidden payload.

Do not memorize extra decimal places. Know the direction and size of each tradeoff, and point to the saved configuration and CSV when challenged.

## Questions raised by the preprocessing experiments

### What improved, and what did not?

At fixed detector and threshold, contrast normalization reduced the 95th-percentile score response to the specified illumination change by about 91.4%. That is a measured nuisance-invariance gain. It did not establish improved recall: forest-seed overlap weakened, and all five leading contrast examples contain conspicuous black strips or borders. The exact-zero footprint mask also leaves near-black edge fragments. Show both the numerical benefit and the failure examples.

### Why keep an older model after trying improvements?

The decision criterion was written before the new results: reduce nuisance effects without a clear stability failure. A new version number is not evidence of a better detector. v3 remains a screening reference with known brightness and independent-run weaknesses; the negative ablations make that decision more defensible rather than curing those weaknesses.

### Why is the affine-intensity test insufficient?

Subtracting the mean and dividing by standard deviation is designed to suppress positive affine intensity changes. The test checks that the implemented pipeline has that behavior, including clipping and footprint handling. It does not show that real albedo anomalies survive normalization. No sampled flags flipped for any model, so the stronger numerical score invariance must not be presented as improved classification.

### How should I explain the notebook's origin and execution?

Use the first-person explanations to describe the decisions you understand and can demonstrate. The code is directly visible, and the saved experiments ran locally on the RTX 3050. The notebook is prepared for Colab; do not describe the saved run as a Colab session. Read the cells and reproduce a small run before defending implementation details.
