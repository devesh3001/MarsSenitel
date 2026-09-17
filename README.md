<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/800px-OSIRIS_Mars_true_color.jpg" alt="Mars" width="120" style="border-radius:50%; margin-bottom: 20px;"/>
  
  # Mars HiRISE: Unsupervised Anomaly Detection
  **National Students' Space Challenge (NSSC 2026) | IIT Kharagpur**
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Custom_Architecture-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
  [![Status](https://img.shields.io/badge/Status-Submission_Ready-success?logo=checkmarx&logoColor=white)]()
  [![Pipeline](https://img.shields.io/badge/Pipeline-End--to--End-8a2be2)]()
</div>

---

## 🚀 Executive Summary
This repository contains a complete, from-scratch unsupervised machine learning pipeline to detect geological and artificial anomalies in Mars Reconnaissance Orbiter (HiRISE) imagery. 

Instead of relying on arbitrary heuristics or pre-trained models, we built a mathematically rigorous system combining a custom **Convolutional Autoencoder**, an **Isolation Forest** novelty engine, and a **Gaussian Mixture Model (GMM)** statistical threshold bounded by a 100-source bootstrap. We explicitly address edge-cases, document ablations (v1–v8), and provide an honest assessment of photometric bias in orbital data.

---

## 🧠 Architecture & Pipeline

### 1. Deep Latent Compression (Phase 1)
- **Model:** 5-Stage Convolutional Autoencoder (trained from scratch).
- **Features:** GroupNorm, SiLU activations, and a resize-convolution decoder to entirely prevent checkerboard artifacts. No skip connections are used to ensure the bottleneck forces true compression.
- **Loss:** A hybrid of Mean Squared Error (MSE) and Structural Similarity Index (SSIM). 

### 2. Isolation Forest Novelty Engine (Phase 2)
- **Scoring:** The 256-dimensional deterministic latent vector is fed into a 2,000-tree Isolation Forest.
- **Thresholding:** We rejected arbitrary boxplot thresholds. Instead, we dynamically fit a 4-component Gaussian Mixture Model (GMM) using Bayesian Information Criterion (BIC), extracting the tail envelope mathematically (`μ + 3σ`).
- **Metadata Fusion:** Controlled experiments proved robust performance even when incorporating cyclically-encoded seasonal data and sun angle telemetry.

### 3. Reconstruction Interpretability (Phase 3)
- **Heatmaps:** Absolute pixel-error heatmaps identify the exact spatial location of the anomaly.
- **Hypotheses:** Candidates are systematically mapped to the NSSC 2026 Genesis Outlier framework (Type A: Splicing, Type B: Terrestrial, Type C: Sensor Artifacts). 

### 4. Design Journal & Ablations (Phase 4)
- We completed 14 total runs. **v3** is our canonical reference model. 
- v4–v8 serve as explicit ablation studies testing contrast normalization, footprint-masking, gradient loss, and geometric augmentation, explicitly tracking what works and what fails.

---

## 📁 Repository Structure

```text
MarsSenitel/
├── mars_anomaly/            # Core PyTorch ML library (Model, Loss, Datasets, Thresholding)
├── scripts/                 # Operational execution, verification, and packaging scripts
├── notebooks/               # Jupyter Notebooks for reporting and Colab/Kaggle execution
├── tests/                   # 19 comprehensive unit tests verifying math contracts & shapes
├── outputs/                 # Deep experiment tracking (JSON configs, weights, heatmaps for v1-v8)
│   ├── v3/                  # 🏆 Canonical run outputs
│   └── comparison/          # Cross-run stability metrics
├── output/pdf/              # Final compiled competition PDF reports
├── README.md                # Project Overview
├── DEFENSE_GUIDE.md         # 40-mark presentation rehearsal and FAQ guide
├── RESEARCH_AND_APPROACH.md # Foundational math and research context
└── ENGINEERING_CHANGELOG.md # The step-by-step history from v1 to v8
```

---

## ⚙️ Running Locally

Python 3.12 is the target environment. Install a PyTorch CUDA wheel suitable for your hardware, then install the local requirements. Ensure the `data/` folder contains the extracted `DATASETS*.zip` files.

```powershell
# 1. Run the test suite
.\.venv\Scripts\python.exe -m pytest -q

# 2. Train the autoencoder (Example: Canonical v3 setup)
.\.venv\Scripts\python.exe -m mars_anomaly.train --data data --out outputs/reproduction_v3 --version reproduction_v3 --epochs 15 --latent-dim 256 --structural-weight 0.1

# 3. Evaluate and threshold with the Gaussian Mixture Model
.\.venv\Scripts\python.exe -m mars_anomaly.evaluate --data data --run outputs/reproduction_v3 --method mixture_three_sigma --trees 2000

# 4. Project latent spaces for visualization (t-SNE)
.\.venv\Scripts\python.exe scripts/project_latents.py --directory outputs/reproduction_v3/mixture_three_sigma_trees2000
```

---

## ☁️ Running on Colab/Kaggle
Open `notebooks/Mars_HiRISE_Submission.ipynb` in a GPU runtime. Upload the raw competition dataset ZIP. The notebook contains all preprocessing, model architecture, training, and scoring loops in one readable sequence.

---

## 📜 Submission Compliance & Honesty
This repository complies completely with the NSSC 2026 rules:
1. **No Pretrained Models:** Every weight is initialized randomly.
2. **No Data Leakage:** Source-level grouping ensures crops from the same master image never cross train/val boundaries.
3. **Honest Thresholding:** We never manually lowered the threshold just to force 5 candidates. The math dictates the cut-off.
4. **Transparent Failures:** We retained negative results (e.g., border artifacts in v4) to prove scientific rigor.

<div align="center">
  <i>Developed for the NSSC 2026 Data Analytics Problem Statement.</i>
</div>
