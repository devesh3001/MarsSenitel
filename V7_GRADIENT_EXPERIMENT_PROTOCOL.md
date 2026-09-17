# V7 Gradient Loss Experiment Protocol

Recorded before training. Target 60-90 minutes; no architecture search or new seed grid.

## Fixed experiment

v7 uses the v3 architecture (256-dimensional from-scratch autoencoder), seeds 2026, split 2026, 15-epoch budget, and 2,000-tree forest. The preprocessing remains `raw` (no border masking or contrast standardization).

The single change is the training objective. We introduce the previously unused Sobel-style gradient loss to explicitly force the model to reconstruct high-frequency Martian geology (crater edges, ridges, and injected anomalies) rather than just broad shapes.
The loss function will be: `0.8 MSE + 0.1 (1 - SSIM) + 0.1 Gradient`.

This tests the hypothesis that explicitly penalizing blurred edges will make the latent space more sensitive to structural anomalies and less sensitive to global brightness.

## Evaluation fixed before results

Use the same fixed evaluation protocol as v3. Apply the saved model, forest, and mixture-envelope threshold without refitting to transformed images.

- Compare validation MSE, SSIM, and Gradient Error against v3.
- Run the 2,000-tree Isolation Forest with 256 samples per tree.
- Extract the top 5 candidates after thresholding.
- **Diagnostic:** Calculate the median mean-intensity of the 5 candidates. If it falls significantly below the 99.7th percentile (the v3 result), the gradient loss has successfully decoupled novelty from global brightness.

## Conservative promotion rule

Promote to reference only if: (1) Validation Gradient Error is significantly lower than v3; (2) The selected candidates are not exclusively the top 0.3% brightest crops in the dataset; and (3) The resulting error heatmaps localize on sharp features rather than diffuse brightness differences. If a criterion fails, keep v3 as the reference and log the negative result.

## Command to execute

```powershell
.\.venv\Scripts\python.exe -m mars_anomaly.train --data data --out outputs/v7 --version v7 --epochs 15 --latent-dim 256 --structural-weight 0.1 --gradient-weight 0.1
.\.venv\Scripts\python.exe -m mars_anomaly.evaluate --data data --run outputs/v7 --method mixture_three_sigma --trees 2000 --no-projection
```


## Post-run audit correction (protocol preserved above)

The executed objective was 0.9 MSE + 0.1 (1-SSIM) + 0.1 Gradient, not the 0.8 MSE description above. The gradient term uses adjacent finite differences, not Sobel filtering. v7 gives modest reconstruction gains, but does not establish elimination of brightness bias or improved detection. It is not promoted in the canonical submission.
