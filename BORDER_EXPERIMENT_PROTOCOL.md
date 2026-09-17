# One final, bounded border experiment

Recorded before training or stress-test results. One new model only; use the verified second-stage submission if promotion criteria fail. Target 60-90 minutes; no architecture search or new seed grid.

## Fixed experiment

v6 uses the v5 architecture (256-dimensional from-scratch autoencoder), loss, seeds 2026, split, 15-epoch budget, 2,000-tree forest and mixture-envelope threshold. Only the border-handling policy changes. The matched preprocessing comparator is v5; the submission reference remains raw-input v3.

Treat pixels <=4/255 connected to an image border as possible missing support, expand the mask by two pixels to cover edge fragments, estimate contrast outside that mask, and extend the nearest valid standardized pixel into masked regions. This replaces v5's exact-zero mask and neutral fill. The constants are fixed here, not tuned on candidate appearances. It does not reconstruct true missing terrain. Border-connected shadows may be altered, and extension can create repeated texture.

## Evaluation fixed before results

Use a fixed 256-crop sample (random seed 73) from validation sources. Apply each saved model, forest and threshold without refitting to transformed images. Compare v3, v5 and v6.

- Nuisance: replace the first eight columns with a near-black strip (2/255), then JPEG encode/decode at quality 90. Pair with a JPEG-only control. Report absolute score change and threshold flips. This destroys edge content; it is a stress test, not a perfectly content-preserving nuisance.
- Defect diagnostics: in a central 24x24 square, separately add +0.35 and -0.35 with clipping. Also replace a central 24x24 square by a 4-pixel checkerboard of 0.1/0.9. No synthetic images are used for training, forest fitting, calibration or model selection among repeated trials. Report paired score increase, fraction increasing, and threshold crossing. The original crops are unlabelled, so these are response diagnostics, not real-anomaly precision/recall.
- Existing evaluation supplies two forest-only seed repeats, source-bootstrap cutoff uncertainty, all scores and post-threshold examples. A single new training run does not establish initialization/split robustness.

## Conservative promotion rule

Promote only if: (1) the border-test 95th-percentile absolute score change is at least 25% below both v3 and v5; (2) all three defect-response fractions are at least 0.70 and no more than 0.05 below v3; (3) minimum forest-seed Jaccard is at least v3's 0.667; and (4) the mask audit and selected-image review reveal no new dominant artifact. If a criterion fails, keep v3 and attach the negative experiment without changing its claims or threshold. No count target is used. Passing these proxy checks would still not establish hidden-anomaly accuracy.

## Reproducibility

Nearest valid pixel indices use SciPy's Euclidean feature transform: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html . Existing modes remain unchanged. Store v6 separately and preserve the verified previous package.
