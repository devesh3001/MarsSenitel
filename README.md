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

## 🏆 Why This Pipeline Wins (Key Innovations)

Instead of relying on arbitrary heuristics or pre-trained models, we built a mathematically rigorous system that strictly adheres to the competition constraints:

1. **Built 100% From Scratch:** No pretrained weights (VGG/ResNet) were used. The **5-Stage Convolutional Autoencoder** was designed and trained entirely from scratch, ensuring true latent compression without data leakage.
2. **Mathematically Honest Thresholding:** We explicitly rejected arbitrary boxplot thresholds. Instead, we used the **Bayesian Information Criterion (BIC)** to dynamically fit a Gaussian Mixture Model (GMM) to the Isolation Forest scores, extracting the anomaly tail mathematically via a 100-source bootstrap.
3. **No Checkerboard Artifacts:** We designed a **resize-convolution decoder** rather than using standard transposed convolutions, completely eliminating checkerboard artifacts in the reconstruction heatmaps.
4. **Transparent Ablations (v1–v8):** We documented 14 total runs. We explicitly retained and documented our negative results (e.g., border artifacts in v4, augmentation collapse in v8) to prove scientific rigor.

---

## 📊 Final Results at a Glance

Our canonical pipeline (**v3**) identified **17 anomalous crops** from the 10,422 image dataset, using a statistically sound, data-driven threshold.

| Metric | Result |
|--------|--------|
| **Final Threshold (T)** | **0.5425** *(95% CI: 0.533 to 0.555)* |
| **Flagged Anomalies** | **17 crops** (Top 5 selected for geological hypotheses) |
| **Validation SSIM** | **0.6738** |
| **Validation MSE** | **0.002599** |
| **Forest-Seed Stability** | **> 0.96 Spearman rank** across all repeats |
| **Total Ablation Runs** | **14** (v1 to v8 + 6 robustness checks) |

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    A[Raw Images 227x227] -->|Phase 1| B(ConvAutoencoder)
    B -->|MSE + SSIM Loss| C{256-dim Latent Space}
    C -->|Phase 2| D[2,000-Tree Isolation Forest]
    D -->|BIC GMM Threshold| E(Novelty Scores)
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
│   ├── Mars_HiRISE_Final_Report.pdf    # 📥 DOWNLOAD LINK ABOVE
│   └── Mars_HiRISE_Analysis_Report.pdf 
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

# 2. Run the test suite (19 unit tests)
pytest -q

# 3. Open the main submission notebook
jupyter notebook notebooks/Mars_HiRISE_Submission.ipynb
```

<div align="center">
  <i>Developed for the NSSC 2026 Data Analytics Problem Statement.</i>
</div>
