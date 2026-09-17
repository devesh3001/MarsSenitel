# NSSC 2026 HiRISE anomaly detection

A from-scratch autoencoder and latent Isolation Forest for the supplied Mars imagery challenge. Start with [current progress](CURRENT_PROGRESS.md), [the decision log](DECISION_LOG.md) and [the defense guide](DEFENSE_GUIDE.md). The log records changes, evidence, reasons, limitations and explanations for the presentation. Rebuild the dated progress snapshot with `python scripts/update_progress.py`.

Read [the research and approach](RESEARCH_AND_APPROACH.md) for the input analysis and primary references. [The engineering changelog](ENGINEERING_CHANGELOG.md) records the actual v1–v8 symptom → diagnosis → fix → outcome sequence with v3 as the canonical reference. No verified anomaly labels are available: candidate counts and repeatability are not detection accuracy.

## Main deliverables

Fourteen full 15-epoch runs (v1–v8 plus six robustness repeats) are completed. The canonical reference is **v3** (256-dim, MSE + SSIM, raw input). v4–v8 serve as Phase 4 design-journal evidence and controlled ablations. All selection and packaging decisions reference v3.

Verification on 17 September 2026: 14 completed training runs, 17 passing tests, 15 executed notebook code cells with no errors, and 15 visually reviewed report pages. The package builder checks the notebook/report hashes and every archived file. v3 remains the reference; v4–v8 demonstrate controlled ablations, not validated anomaly-accuracy improvements.

### Preserved first-stage artifacts

These historical files remain in the original workspace and first-stage package. The revised package carries the revised notebook and report, together with the historical numerical evidence.

- `notebooks/Mars_HiRISE_Submission.ipynb`: rubric-numbered notebook with embedded source; all 15 code cells executed without errors.
- `output/pdf/Mars_HiRISE_Analysis_Report.pdf`: 13 visually reviewed pages of measured analysis, figures and candidate hypotheses.
- `RESULTS.md`: final comparison and limitations when available.
- `output/submission/NSSC_2026_Mars_Analysis.zip`: local package, created only after verification.
- `DECISION_LOG.md`, `CURRENT_PROGRESS.md`, `DEFENSE_GUIDE.md`: maintained defense narrative, progress and rehearsal guide.

## Run locally

Python 3.12 is the tested target. Install a PyTorch CUDA wheel suitable for your hardware, then install `requirements.txt` in an isolated environment. The tested local environment is recorded in `requirements-lock-local.txt` and each run's `environment.json`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mars_anomaly.train --data data --out outputs/reproduction_v3 --version reproduction_v3 --epochs 15 --latent-dim 256 --structural-weight 0.1
.\.venv\Scripts\python.exe -m mars_anomaly.evaluate --data data --run outputs/reproduction_v3 --method mixture_three_sigma --trees 2000 --no-projection
.\.venv\Scripts\python.exe scripts/project_latents.py --directory outputs/reproduction_v3/mixture_three_sigma_trees2000
```

Use a new output directory for a new experiment. Resume an interrupted training run with `--resume` and the same configuration. `last.pt` includes optimizer and random-generator state; `best.pt` is used for evaluation. Eleven tests cover dimensions/gradients, SSIM, score orientation, threshold-before-selection, calibration, grouped bootstrap, the mixture envelope preprocessing invariance/connectivity/constant-image handling and partial-checkpoint recovery. A smoke checkpoint is explicitly rejected by normal evaluation. Run GPU processes sequentially on this machine because concurrent runs encountered memory and checkpoint-write limits.

## Colab or Kaggle

Open the revised `notebooks/Mars_HiRISE_Colab.ipynb` in a GPU runtime and provide the original outer dataset ZIP. Preprocessing, model, training and scoring definitions appear directly in normal Python cells. On a fresh session, missing main experiments train; set `RUN_ROBUSTNESS=True` to reproduce the additional raw and contrast seed/split checks. Keep inputs and notebooks private under the rules. The notebook supports cloud paths, but this project's measured runs were executed locally; cloud execution has not been verified.

Scientific libraries already present are reused, and missing libraries are installed. Package versions and GPU numerics may differ from the local environment; regenerated candidates require fresh visual review. The recorded final choice is retained for reproducibility rather than silently selecting a new seed. Save outputs and checkpoints before a temporary runtime expires.

## Evidence and organization

- `outputs/audit/`: input hashes, complete image audit, metadata facts and a seeded image sample.
- `mars_anomaly/`: model, loss, source splitting, checkpointed training, scoring, calibration and figures.
- `outputs/<run>/`: exact configuration, environment, source split, per-epoch history and checkpoints.
- `outputs/<run>/events.jsonl`: timestamped training starts, resumes, saved epochs and completion; introduced during the v3 resume, so earlier runs do not have a complete event history.
- `outputs/<run>/mixture_three_sigma_trees2000/`: final detector outputs, scores, uncertainty, metadata summaries and heatmaps.
- `outputs/comparison/`: common reconstruction metrics, cross-run stability and potential confounds.
- `THRESHOLD_DECISION.md`: original failed threshold and the replacement's assumptions.
- `scripts/build_submission_notebook.py`, `execute_notebook.py`, `build_report.py`, `package_submission.py`: regenerate and verify deliverables.

The earlier 400-tree analyses and initial baseline notebook are retained as historical artifacts. The final comparison uses 2,000 trees consistently. Source-level grouping reduces shared-observation leakage but does not prove geographically independent data. Brightness and source effects remain material limitations.

## Submission constraints

Use no pretrained weights, external image pretraining or guessed labels. Feed the learned vector to Isolation Forest. Determine the threshold independently of an assumed contamination count. Select up to five only after thresholding; never lower the threshold to force five. Keep the question-number headings and report uncertainty.

The rules require a private GitHub repository, named organizer collaborators, a notebook with outputs, a PDF report and at least three genuine iterations. The local package does not create a repository, invite collaborators or submit the project. Team details and repository logistics remain listed in `SUBMISSION_CHECKLIST.md`.

