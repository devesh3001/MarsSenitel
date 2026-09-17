# 🔴 Mars HiRISE Unsupervised Anomaly Detection

**NSSC 2026 — National Students Space Challenge**
**IIT Kharagpur | Data Analytics Track**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)

An end-to-end unsupervised anomaly detection pipeline for **10,422 Mars HiRISE orbital image crops** (227x227 px). Built entirely from scratch — no pretrained weights, no arbitrary thresholds, no shortcuts.

---

## Pipeline Overview

```
Raw Images (227x227)
        |
        v
Phase 1: Convolutional Autoencoder (trained from scratch)
        |  5-stage encoder -> 256-dim latent vector
        |  MSE + SSIM + Gradient loss
        |  t-SNE latent space visualisation
        v
Phase 2: Isolation Forest Novelty Engine
        |  2,000-tree IsolationForest on latent vectors
        |  GMM statistical threshold (BIC-selected)
        |  Location analysis via source metadata
        v
Phase 3: Reconstruction Interpretability
        |  Pixel-wise error heatmaps (inferno colormap)
        |  Geological hypotheses per flagged candidate
        v
Phase 4: Architecture Iteration Journal (v1 to v8, 14 total runs)
```

---

## Directory Structure

```
MarsSenitel/
|
+-- notebooks/                          # Jupyter notebooks (one per phase)
|   +-- 01_Phase1_Deep_Latent_Compression.ipynb
|   +-- 02_Phase2_Isolation_Forest_Novelty.ipynb
|   +-- 03_Phase3_Reconstruction_Interpretability.ipynb
|   +-- 04_Phase4_Architecture_Journal.ipynb
|   +-- Mars_HiRISE_Submission.ipynb   <- Full executed notebook with all outputs
|
+-- src/                                # Core Python pipeline modules
|   +-- model.py                        # ConvAutoencoder architecture
|   +-- train.py                        # Training loop, checkpointing, AMP
|   +-- evaluate.py                     # IsolationForest + GMM thresholding
|   +-- visualize.py                    # Heatmap panels, t-SNE, score plots
|   +-- data.py                         # PyTorch Dataset and split management
|   +-- threshold.py                    # Statistical threshold methods
|   +-- preprocessing.py               # Image normalization utilities
|   +-- __init__.py
|
+-- report/                             # Technical reports
|   +-- Mars_HiRISE_Analysis_Report.pdf <- [Download Final PDF Submission](https://github.com/devesh3001/MarsSenitel/raw/main/report/Mars_HiRISE_Analysis_Report.pdf)
|   +-- THRESHOLD_DECISION.md           <- Full threshold rationale
|
+-- data/                               # Metadata CSVs (images not tracked in git)
|   +-- crop_metadata_index.csv
|   +-- source_image_metadata.csv
|
+-- ENGINEERING_CHANGELOG.md            # v1-v8 iteration log (Symptom->Diagnosis->Fix)
+-- DECISION_LOG.md                     # Chronological audit trail
+-- RESEARCH_AND_APPROACH.md            # Literature review and design rationale
+-- DEFENSE_GUIDE.md                    # Presentation guide for Round 2
+-- requirements.txt
```

---

## 📄 Technical Report

GitHub's native PDF viewer cannot render large documents. **Please download the final report to view it:**

👉 **[Download Mars_HiRISE_Analysis_Report.pdf (6 MB)](https://github.com/devesh3001/MarsSenitel/raw/main/report/Mars_HiRISE_Analysis_Report.pdf)** 👈

The report contains the full Phase 1-4 breakdown, architectural diagrams, latent space t-SNE visualisations, and high-resolution Phase 3 error heatmaps.

---

## Key Technical Decisions

### Autoencoder Architecture
- **5-stage ConvAE** from scratch: `1->16->32->64->96->128` channels
- **Resize-convolution decoder** (no transposed convolutions, no checkerboard artifacts)
- **GroupNorm + SiLU** throughout (stable for small batch sizes)
- **256-dimensional bottleneck** — ablated from 128d (v1 through v3)
- No skip connections: forces true latent compression

### Loss Function (v3 — Canonical)
```
L = 0.9 * MSE + 0.1 * (1 - SSIM)
```
SSIM implemented from scratch with 11x11 Gaussian window (sigma=1.5). No pretrained features anywhere.

### Statistical Threshold (Phase 2.2)
Fit Gaussian Mixture Model (1-4 components, BIC-selected) to calibration scores.
```
T = max_j ( mu_j + 3 * sigma_j )
```
- 100 source-group bootstrap resamples -> 95% CI: **[0.533, 0.555]**
- Final threshold: **T = 0.5425**
- Zero flags is an accepted outcome — threshold is never manually lowered

### Engineering Iterations (v1 -> v8)

| Version | Key Change | Outcome |
|---------|------------|---------|
| v1 | MSE baseline, 128d | Texture blurring identified |
| v2 | + SSIM loss | SSIM 0.661 -> 0.672 |
| **v3** | **128d -> 256d** | **Canonical pipeline — best stable performance** |
| v4 | Contrast normalization | Border artifacts; not promoted |
| v5 | Footprint masking | Too many flags; rejected |
| v6 | Fixed near-black mask | Stripe artifacts; negative result |
| v7 | + Gradient loss | Brightness confound remains; not promoted |
| v8 | Augmentation (flip/rotate) | Zero flags; not promoted |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open phase-by-phase notebooks
jupyter notebook notebooks/01_Phase1_Deep_Latent_Compression.ipynb

# Or open the full executed submission notebook
jupyter notebook notebooks/Mars_HiRISE_Submission.ipynb
```

---

## Results

| Metric | Value |
|--------|-------|
| Final threshold T | **0.5425** (95% CI: 0.533 to 0.555) |
| Flagged anomalies | **17 crops** (top 5 selected for interpretation) |
| Validation SSIM (v3) | 0.6738 |
| Validation MSE (v3) | 0.002599 |
| Forest-seed Spearman | > 0.96 across all seed repeats |
| Total experiments | **14 runs** (v1-v8 + 6 robustness checks) |

---

## Team
NSSC 2026 | IIT Kharagpur | Data Analytics Track
