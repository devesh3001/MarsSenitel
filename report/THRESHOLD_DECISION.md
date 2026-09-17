# Statistical boundary revision

## Observed failure of the initial proposal

The completed v1 model produces a calibration score distribution with a narrow low-score peak and a broader second region. Its medcouple is 0.5185. The original adjusted-boxplot fence is 0.87484, with a source-bootstrap 95% interval of approximately 0.70253 to 1.03219. The maximum scored image is approximately 0.55648, so the original method flags zero images.

Zero flags alone is not a reason to change a threshold. The reason for investigating a replacement is the visible multimodality and the inappropriate use of a single skewness expansion across distinct score populations. The original zero-flag output is retained in `outputs/v1` and is not rewritten to conceal this failed candidate.

## Revised descriptive model

Fit Gaussian mixtures with one to four components to calibration scores only. Choose the converged model with the smallest Bayesian information criterion (BIC), using five random initializations, a fixed seed and covariance regularization of 0.000001. The first exploratory fit used ten initializations and gave the same v1 boundary. BIC is a model-complexity heuristic here, not a significance test: crops within a source are correlated.

For component j with mean mu_j and standard deviation sigma_j, define:

```
T = max_j (mu_j + 3 sigma_j)
flag(x) = novelty_score(x) > T
```

All fitted populations contribute to the envelope. The high-score population is not automatically classified as anomalous. The multiplier 3 describes a conventional Gaussian tail screening boundary and is fixed across experiments; it is not tuned to obtain a chosen number of crops. For the fitted mixture, each component has at most 0.00135 probability beyond this envelope, so their weighted sum does too. This is a property of the fitted distribution, not a guaranteed real false-positive rate. The calibration sample is unlabelled and includes unknown contamination.

The v1 BIC values for one through four components are approximately -6118, -7052, -7129 and -7107. The three-component model yields T = 0.53917. It flags 16 of all 10,422 crops; this count is an outcome of the rule. The fitted mixture tail mass is approximately 0.000345, and the source-bootstrap threshold interval is about 0.53211 to 0.54615. These quantities are not interchangeable with the empirical flagged fraction.

## Validation and uncertainty

Refit the full component-selection procedure in each of 100 source-group bootstrap replicates. Save each threshold, the interval, and per-image flag frequency. Also fit two additional Isolation Forest seeds and independently recalibrate each forest. The v1 complete rankings have Spearman correlations of about 0.977, while thresholded-set Jaccard overlap is only about 0.41 to 0.58. Strong global rank correlation does not imply stable final membership.

Compare the fitted calibration CDF against validation scores without selecting the cutoff on that validation diagnostic. The initial v1 maximum CDF distance is approximately 0.054. Do not report an ordinary IID Kolmogorov-Smirnov p-value because source correlation and fitted model selection violate its simple reference assumptions.

The mixture can still mask anomalies by allocating a component to them, misfit tails, or respond to source imbalance. A selected four-component model lies at the upper edge of the search and deserves explicit caution. Gaussian support extends outside [0,1]; inspect whether that matters in the fitted score range. No method here proves the presence or absence of the hidden payload.

The replacement is evaluated consistently across v1, v2 and v3. No image appearance, guessed anomaly label, or desired output count is used to fit the boundary. Only after the rule is established do we inspect up to five flagged images for physical hypotheses.

## Sources

- scikit-learn Gaussian mixture model selection: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
- scikit-learn GaussianMixture API: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- Hubert and Vandervieren adjusted boxplot: https://doi.org/10.1016/j.csda.2007.11.008

## Final detector and independent checks

The controlled forest comparison led to 2,000 trees with 256 samples per tree. At this budget, v1/v2/v3 thresholds are approximately 0.540828/0.533417/0.542471 and they flag 20/9/17 crops. The reference v3 bootstrap interval is [0.535230, 0.550470]. Four components beat three by only 0.85 BIC units, so model order is weakly separated and at the search limit.

Independent v3 initialization and source-split repeats flag 6 and 5 crops, with only 0.150 and 0.294 overlap against the primary 17. No crop is flagged by all three. Thus calibration and representation uncertainty remain practically important despite highly correlated global rankings. The final report preserves this negative result and the brightness concentration among selected candidates.
