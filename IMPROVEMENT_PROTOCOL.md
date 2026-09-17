# Second-stage improvement protocol

Written before training the new experiments. The previous package remains the first-stage result; its 17-candidate list is not a verified discovery.

## Problem I am trying to fix

The five reviewed v3 candidates are all above the 99.7th brightness percentile. Exact candidate agreement also changes substantially with initialization and source split. I will test image preprocessing before adding more model capacity.

## Controlled comparisons

- **v3 reference:** original intensity divided by 255, 256-dimensional model, 0.9 MSE + 0.1 (1 - SSIM).
- **v4 contrast:** same architecture, loss, seed, source split and 15-epoch budget. Standardize each crop by its own mean and standard deviation, map to `0.5 + 0.15*z`, then clip to [0,1]. Standard deviation is floored at 1/255 to avoid amplifying nearly constant crops without bound. No dataset-wide or held-out fitting is involved.
- **v5 footprint:** retain contrast standardization, but estimate statistics after excluding exact-zero regions connected to the image border, then fill those regions with neutral 0.5. This is an explicit image-footprint heuristic, not a supplied validity mask. Interior disconnected zeros remain image content.

The v4-v5 comparison isolates the additional footprint treatment. All learned weights start randomly. All forests keep 2,000 trees, max_samples 256, and the same calibration formula. No desired number of flags is supplied.

## Evaluation and decision

1. Verify preprocessing contracts, finite outputs and response to a controlled positive affine intensity transformation.
2. Compare forest-only seed agreement, calibration uncertainty, raw brightness/contrast/zero-fraction associations and selected-source concentration.
3. Compare reconstruction on each model's own declared input domain. Normalized-domain MSE or SSIM is not directly comparable to original-domain v3 metrics.
4. Select a new reference only if it reduces the observed nuisance concentration without a clear stability failure; a lower candidate count alone is not improvement. If neither is supported, preserve v3 and report the negative experiment.
5. Repeat the best-supported new preprocessing configuration with independent initialization and source partition. Retain those outcomes even if disappointing.
6. Inspect physical examples only after the formula has selected candidates. Do not use plausible-looking crops or guessed labels to tune the boundary.

## Risks I will keep visible

Global brightness can be meaningful geology. Standardization deliberately suppresses that information and may amplify noise; clipping can suppress extreme local contrast. Border-connected zeros can include genuine shadows, so footprint treatment may remove meaningful pixels. All selected-panel labels must distinguish raw input, standardized input and reconstruction. No hidden-label performance gain is claimed without ground truth.

## Notebook presentation

I will explain the work in short first-person notes: what I noticed, what I tried, the result, and the limitation. The notebook will be usable on Colab with visible setup/model/training/scoring cells. Existing runs will remain explicitly identified as local CUDA executions; no Colab session history or personal observation will be invented.
