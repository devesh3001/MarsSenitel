<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/800px-OSIRIS_Mars_true_color.jpg" alt="Mars" width="120" style="border-radius:50%; margin-bottom: 20px;"/>
  
  # 🔴 Mars HiRISE: Unsupervised Anomaly Detection
  **NSSC 2026 — National Students Space Challenge | IIT Kharagpur (Data Analytics)**
  
  [![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
  [![Status](https://img.shields.io/badge/Status-Submission_Ready-success?logo=checkmarx)]()
</div>

---

> [!IMPORTANT]
> **To the Judges:** 
> - **For Code Review:** Please evaluate `notebooks/Mars_HiRISE_Submission.ipynb`. This is the fully executed, end-to-end master notebook containing all 4 phases as required by the rules.
> - **For Technical Documentation:** Please [📥 Download the Final Report PDF (7.3 MB)](https://github.com/devesh3001/MarsSenitel/raw/main/report/Mars_HiRISE_Final_Report.pdf) which contains our high-resolution error heatmaps and complete architectural breakdown.

---

## 🌟 Core Technical Innovations

We built a mathematically rigorous system that strictly adheres to the competition constraints:

1. **Built From Scratch:** No pretrained weights (VGG/ResNet) were used. The **5-stage convolutional autoencoder** was trained from random initialization. Source-group splitting keeps crops from the same observation together, reducing one form of leakage.
2. **Statistical Thresholding:** We rejected the initial adjusted-boxplot threshold after its assumptions failed on the multimodal scores. We used **BIC** to select a Gaussian mixture fitted on calibration scores, then set a fixed three-standard-deviation envelope. **100 source-group bootstrap resamples** estimate threshold uncertainty; they do not establish detection accuracy or a false-positive guarantee.
3. **Reduced Checkerboard Artifact Risk:** We designed a **resize-convolution decoder** rather than using transposed convolutions. This avoids the uneven-overlap mechanism associated with transposed-convolution checkerboard artifacts and reduces that artifact risk in reconstructions.
4. **Transparent Ablations (v1–v8):** We documented 14 total runs. We explicitly retained and documented our negative results (e.g., border artifacts in v4, augmentation collapse in v8) to prove scientific rigor.

---

## 🔬 Key Technical Decisions

### Autoencoder Architecture
- **5-stage ConvAE** from scratch: `1->16->32->64->96->128` channels
- **Resize-convolution decoder** (no transposed convolutions)
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
- 100 source-group bootstrap resamples -> 95% CI: **[0.535, 0.550]**
- Final threshold: **T = 0.5425**
- Zero flags is an accepted outcome — threshold is never manually lowered

### Engineering Iterations (v1 -> v8)

| Version | Key Change | Outcome |
|---------|------------|---------|
| v1 | MSE baseline, 128d | Texture blurring identified |
| v2 | + SSIM loss | SSIM 0.661 -> 0.672 |
| **v3** | **128d -> 256d** | **Canonical pipeline — selected stability/quality trade-off** |
| v4 | Contrast normalization | Border artifacts; not promoted |
| v5 | Footprint masking | Too many flags; rejected |
| v6 | Fixed near-black mask | Stripe artifacts; negative result |
| v7 | + Gradient loss | Brightness confound remains; not promoted |
| v8 | Augmentation (flip/rotate) | Zero flags; not promoted |

---

## 📊 Final Results at a Glance

Our canonical pipeline (**v3**) **flagged 17 candidate crops** from the 10,422 image dataset using a statistically defined, data-driven threshold. Hidden labels are unavailable, so these are screening candidates rather than confirmed anomalies.

| Metric | Result |
|--------|--------|
| **Final Threshold (T)** | **0.5425** *(95% CI: 0.535 to 0.550)* |
| **Flagged Candidates** | **17 crops** (Top 5 selected for geological hypotheses) |
| **Validation SSIM** | **0.6738** |
| **Validation MSE** | **0.002599** |
| **Fixed-encoder forest repeats** | Spearman **0.9946/0.9947**; flagged-set Jaccard **0.667/0.727** |
| **Independent training repeats** | Flagged-set Jaccard **0.150/0.294** versus v3; no crop flagged in all three runs |
| **Known confound** | All five reviewed candidates lie above the **99.7th brightness percentile** |
| **Total Ablation Runs** | **14** (v1 to v8 + 6 robustness checks) |

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    A[Raw Images 227x227] -->|Phase 1| B(ConvAutoencoder)
    B -->|MSE + SSIM Loss| C{256-dim Latent Space}
    C -->|Phase 2| D[2,000-Tree Isolation Forest]
    D -->|Score then BIC GMM threshold| E(Novelty Scores)
    E -->|Phase 3| F[Pixel-wise Error Heatmaps]
    F -->|Geological Analysis| G(Anomaly Hypotheses)
    
    style B fill:#e6f3ff,stroke:#3399ff,stroke-width:2px
    style D fill:#fff2e6,stroke:#ff9933,stroke-width:2px
    style F fill:#e6ffe6,stroke:#33cc33,stroke-width:2px
```

---

## 📁 Repository Map

```text
MarsSenitel/
├── notebooks/                          # Jupyter Notebooks
│   ├── Mars_HiRISE_Submission.ipynb    # ⭐️ THE MAIN SUBMISSION (All phases)
│   ├── 01_Phase1_Deep_Latent_Compression.ipynb
│   ├── 02_Phase2_Isolation_Forest_Novelty.ipynb
│   ├── 03_Phase3_Reconstruction_Interpretability.ipynb
│   └── 04_Phase4_Architecture_Journal.ipynb
├── report/                             # Technical Reports
│   ├── Mars_HiRISE_Final_Report.pdf    # ⭐ Canonical judge-facing report
│   └── Mars_HiRISE_Analysis_Report.pdf # Archived/intermediate extended analysis
├── src/                                # Core PyTorch ML library
│   ├── model.py                        # ConvAutoencoder architecture
│   ├── evaluate.py                     # IsolationForest + GMM
│   └── threshold.py                    # Statistical threshold logic
├── data/                               # Metadata (Images excluded from git)
├── ENGINEERING_CHANGELOG.md            # Detailed v1-v8 symptom->fix log
└── requirements.txt
```

---

## ⚙️ How to Run Locally

Python 3.11/3.12 is the target environment. Ensure the `data/` folder contains the extracted `DATASETS*.zip` files.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test suite (21 unit tests)
pytest -q

# 3. Open the main submission notebook
jupyter notebook notebooks/Mars_HiRISE_Submission.ipynb
```

## 📈 Evaluation for Judges

If you have a ground truth CSV file with labels (containing `filename` and `label` columns, where `label=1` indicates an anomaly), you can easily calculate metrics such as ROC-AUC, PR-AUC, and F1 Score using our provided evaluation script.

```bash
# Calculate metrics using the canonical model (v3) output
python scripts/evaluate_labels.py --labels path/to/your_ground_truth.csv
```

<div align="center">
  <i>Developed for the NSSC 2026 Data Analytics Problem Statement.</i>
</div>
