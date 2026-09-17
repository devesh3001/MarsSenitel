"""Build a standalone, portable notebook from the reviewable Python modules."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def markdown(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
    def code(text): cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]})
    markdown('''# NSSC 2026 Mars HiRISE anomaly detection

## Research and baseline implementation

This notebook implements the supplied challenge with a randomly initialized convolutional autoencoder and Isolation Forest on its one-dimensional latent vectors. No pretrained weights or external image datasets are used. Higher novelty scores mean greater anomalousness. Statistical thresholding precedes the selection of up to five examples.

**Current scope:** a reproducible baseline and analysis pipeline. v2 and v3 remain experiments until their actual training evidence and interpretations are recorded. This is not yet a complete competition submission.

The original data audit found **10,422 grayscale images at 227 by 227**, **172 source observations**, complete metadata joins and no exact pixel duplicates. Longitude has only two values. Full reasoning and primary research references accompany the project in `RESEARCH_AND_APPROACH.md`.

### Colab and Kaggle setup

Use a GPU runtime. This notebook contains the project modules, so no public GitHub repository is needed. In Colab, upload the original dataset archive when prompted. In Kaggle, attach the archive as a **private** input dataset and set `ARCHIVE` if automatic discovery does not locate it. Save/download outputs and checkpoints before the runtime expires. The notebook does not purchase compute or publish data.
''')
    code('''from pathlib import Path
import sys, os, json, subprocess, importlib.util

if Path('/kaggle/working').exists():
    ROOT = Path('/kaggle/working/mars_project')
elif Path('/content').exists():
    ROOT = Path('/content/mars_project')
else:
    ROOT = Path.cwd()
    if ROOT.name == 'notebooks': ROOT = ROOT.parent
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
required = {'numpy':'numpy', 'pandas':'pandas', 'PIL':'Pillow', 'scipy':'scipy',
            'sklearn':'scikit-learn', 'matplotlib':'matplotlib', 'statsmodels':'statsmodels', 'torch':'torch'}
missing = [package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
print('Project directory:', ROOT)
''')
    sources={p.name:p.read_text(encoding='utf-8') for p in (ROOT/'mars_anomaly').glob('*.py')}
    code('''# Embedded modules are identical to the source files shipped with this notebook.
MODULES = '''+repr(sources)+'''
module_dir = ROOT / 'mars_anomaly'
module_dir.mkdir(exist_ok=True)
for name, source in MODULES.items():
    target = module_dir / name
    if target.exists() and target.read_text(encoding='utf-8') != source:
        raise RuntimeError(f'{target} differs from this notebook. Rebuild the notebook from your source before running.')
    if not target.exists(): target.write_text(source, encoding='utf-8')
import torch, numpy as np, pandas as pd
from IPython.display import display, Image, Markdown
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name())
''')
    markdown('''## Data preparation and rule checks

The archive contains `DATASETS/images.zip` plus `crop_metadata_index.csv` and `source_image_metadata.csv`. Only those known entries are extracted. File identifiers are used for joins, never as model features. No visually unusual samples are removed or assigned training labels.
''')
    code('''from mars_anomaly.data import prepare_archive, load_manifest, CropDataset

DATA = ROOT / 'data'
ARCHIVE = None  # Optional explicit path to the supplied outer ZIP.
if not (DATA / 'crop_metadata_index.csv').exists():
    candidates = list(ROOT.glob('DATASETS-*.zip'))
    if Path('/kaggle/input').exists(): candidates += list(Path('/kaggle/input').rglob('DATASETS-*.zip'))
    if ARCHIVE is None and candidates: ARCHIVE = candidates[0]
    if ARCHIVE is None and importlib.util.find_spec('google') is not None and importlib.util.find_spec('google.colab') is not None:
        from google.colab import files
        uploaded = files.upload()
        candidates = [Path(name) for name in uploaded if name.endswith('.zip')]
        if candidates: ARCHIVE = candidates[0]
    if ARCHIVE is None: raise FileNotFoundError('Set ARCHIVE to the supplied dataset ZIP and rerun this cell.')
    prepare_archive(ARCHIVE, DATA)

manifest = load_manifest(DATA, seed=2026)
display(manifest.groupby('split').agg(images=('filename','size'),sources=('source_image_id','nunique')))
assert manifest.groupby('source_image_id').split.nunique().max() == 1
assert manifest.filename.is_unique and not manifest.isna().any().any()
print('One split per source; complete unique crop index and metadata joins.')
''')
    code('''import matplotlib.pyplot as plt
sample = manifest.sample(12, random_state=2026)
fig, axes = plt.subplots(3,4,figsize=(10,8))
for ax, (_,row) in zip(axes.flat, sample.iterrows()):
    x,_ = CropDataset(pd.DataFrame([row]))[0]
    ax.imshow(x[0],cmap='gray',vmin=0,vmax=1); ax.set_title(row.filename,fontsize=8); ax.axis('off')
fig.suptitle('Seeded visual sanity check; no anomaly labels assigned')
fig.tight_layout(); display(fig); plt.close(fig)
''')
    markdown('''## Phase 1.1 Architecture design

Five 3 by 3 strided convolution blocks preserve spatial structure until a linear 128-dimensional bottleneck. GroupNorm supports small GPU batches. Explicit resize-convolution stages recover exactly 227 by 227 pixels. There are no encoder-decoder bypasses. The starting compression ratio is 51,529 / 128, about 403 input values per latent coordinate. A 256-dimensional alternative is a candidate for later controlled capacity comparison.
''')
    code('''from mars_anomaly.model import ConvAutoencoder
model_preview = ConvAutoencoder(128)
print(model_preview)
print('Trainable parameters:',sum(p.numel() for p in model_preview.parameters()))
with torch.no_grad():
    reconstruction, latent = model_preview(torch.zeros(1,1,227,227))
print('Input/reconstruction:',tuple(reconstruction.shape),'| Latent:',tuple(latent.shape))
del model_preview
''')
    markdown('''## Phase 1.2 Loss design and baseline training

v1 minimizes mean squared reconstruction error. SSIM and gradient error are also recorded for comparison, but do not influence the v1 objective. Candidate v2 adds SSIM only after inspecting v1's actual limitations. Gaussian-window SSIM is computed directly from image statistics with no pretrained features.

The source-based train, validation and calibration splits are fixed. Calibration is not used to select the encoder checkpoint. Each epoch saves resumable state and the best validation checkpoint. Version objectives differ, so compare common metrics rather than comparing their raw total losses.
''')
    code('''from mars_anomaly.train import TrainConfig, run_training

RUN = ROOT / 'outputs' / 'v1'
config = TrainConfig(version='v1', latent_dim=128, epochs=15, batch_size=16)
if not (RUN / 'training_status.json').exists():
    run_training(DATA, RUN, config, resume=(RUN / 'last.pt').exists())
status = json.loads((RUN / 'training_status.json').read_text())
assert status['status'] == 'complete', 'A smoke run is not a trained submission model.'
display(pd.read_csv(RUN / 'history.csv'))
display(Image(filename=str(RUN / 'reconstructions_latest.png')))
''')
    code('''history = pd.read_csv(RUN / 'history.csv')
fig, axes = plt.subplots(1,3,figsize=(13,3.5))
for ax, metric in zip(axes,['mse','ssim','gradient']):
    for split in ['train','validation']: ax.plot(history.epoch,history[f'{split}_{metric}'],label=split)
    ax.set(xlabel='Epoch',ylabel=metric.upper(),title=metric.upper()); ax.legend()
fig.tight_layout(); display(fig); plt.close(fig)
''')
    markdown('''## Phase 2.1 Latent Isolation Forest novelty scoring

The primary forest uses the original latent vectors, 400 trees, `max_samples=256` and `contamination='auto'`. The final score is **negative `score_samples`**. The built-in prediction offset does not determine flags. Forests fitted with two additional seeds assess ranking stability.
''')
    code('''from mars_anomaly.evaluate import evaluate
if not (RUN / 'diagnostics.json').exists():
    scored, calibration, diagnostics = evaluate(DATA, RUN, bootstrap=100, projection=True)
else:
    scored = pd.read_csv(RUN / 'novelty_scores.csv')
    calibration = json.loads((RUN / 'calibration.json').read_text())
    diagnostics = json.loads((RUN / 'diagnostics.json').read_text())
display(pd.DataFrame(diagnostics['forest_seed_stability']))
print('Flags:',int(scored.flagged.sum()),'| Effective latent rank:',diagnostics['effective_rank'])
''')
    markdown('''## Phase 1.3 Latent space visualization

t-SNE is applied only for visualization at perplexities 30 and 70. Standardization and a PCA projection to 50 dimensions prepare the visualization; they do not modify the primary forest input. Compare novelty, sun angle and source membership on identical coordinates. Visual groups are interpretations, not known terrain labels. Check latent variance and effective rank alongside these plots.
''')
    code('''for perplexity in [30,70]:
    path = RUN / f'tsne_{perplexity}.png'
    if path.exists(): display(Image(filename=str(path)))
    else: print(f'Projection {perplexity} has not been run.')
print('Effective rank:',diagnostics['effective_rank'],'of',diagnostics['latent_dimensions'])
''')
    markdown('''## Phase 2.2 Statistical threshold and sensitivity

For calibration scores, compute the upper quartile Q3, interquartile range IQR and robust skewness MC (medcouple). The upper fence is `Q3 + 1.5 exp(3 MC) IQR` for MC >= 0 and `Q3 + 1.5 exp(4 MC) IQR` otherwise. This is the Hubert-Vandervieren adjusted boxplot rule. It is a descriptive screening boundary, not a formal false-positive guarantee. Bootstrap complete source groups to describe threshold uncertainty.

The boundary must be reviewed for distribution suitability, source variability and multimodality. Compare the alternate fences without choosing a cutoff to obtain a desired count. Scores from training rows are in-sample; validation and calibration rows remain separately identified.
''')
    code('''display({key:value for key,value in calibration.items() if key != 'bootstrap_thresholds'})
display(Image(filename=str(RUN / 'score_distribution.png')))
thresholds = {key:calibration[key] for key in ['threshold','tukey_comparison','mad_comparison']}
display(pd.DataFrame([{'method':key,'threshold':value,'flagged_images':int((scored.novelty_score>value).sum())} for key,value in thresholds.items()]))
display(scored.groupby('split').flagged.agg(['count','sum','mean']))
''')
    markdown('''## Phase 2.3 Image location and acquisition analysis

Join by the supplied source ID. Longitude has only two distinct values, so avoid interpreting a detailed Mars map. Source IDs are anonymized, and sun-angle semantics and resolution units are not explicitly defined. Compare crop-weighted and equal-source flag rates to expose uneven source sizes. Associations are exploratory and do not establish causes.
''')
    code('''display(pd.read_csv(RUN / 'source_summary.csv').sort_values('flag_rate',ascending=False).head(15))
for field in ['season','resolution','longitude']:
    print(field); display(pd.read_csv(RUN / f'metadata_{field}.csv'))
fig, axes = plt.subplots(1,2,figsize=(10,4))
sources = pd.read_csv(RUN / 'source_summary.csv')
for ax,field in zip(axes,['latitude','sun_angle']):
    ax.scatter(sources[field],sources.flag_rate,s=16)
    ax.set(xlabel=f'{field} as supplied',ylabel='Flagged fraction within source')
fig.tight_layout(); display(fig); plt.close(fig)
''')
    markdown('''## Phase 2.4 Optional metadata fusion

Deferred until the image-only pipeline and three documented iterations are complete. Any later fusion must normalize from training data, exclude source IDs, encode season as categorical and report changes relative to the primary image-only flagged set. It must not replace that set silently.

## Phase 3.1 Reconstruction error heatmaps

The following selection takes at most five crops **strictly above the calibrated threshold**, sorted by novelty. If fewer than five are flagged, all are analyzed. Error maps use a common [0,1] absolute-error scale across candidates. They reveal reconstruction discrepancies and are not direct pixel attributions of the forest.
''')
    code('''selected = pd.read_csv(RUN / 'selected_for_interpretation.csv')
display(selected)
for name in selected.filename:
    display(Image(filename=str(RUN / f'heatmap_{Path(name).stem}.png')))
if selected.empty: print('No images exceed this boundary. The threshold is not lowered to force five examples.')
''')
    markdown('''## Phase 3.2 Geological hypotheses

Interpret each actual selected original/reconstruction/error panel before submission. Record a plausible terrain or imaging explanation, alternatives and limitations. Straight boundaries may be splices, crop footprints or illumination transitions. Diffuse error can reflect texture, blur, intensity shift or unfamiliar content. Black borders alone do not prove a sensor anomaly. No hidden class labels are available, so these interpretations remain hypotheses.

## Phase 4 Architecture iteration and design journal

The project changelog records the preparatory audit and each measured model experiment. v1 establishes the reconstruction and latent baseline. The candidate v2 structural loss and v3 capacity/gradient/detector changes must be motivated by observed limitations, with configurations and outcomes recorded. Until those runs and interpretations are completed, this notebook must not be described as satisfying the full submission checklist.

## References

- [Competition-aligned score convention: scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [SSIM original authors and implementation](https://www.cns.nyu.edu/~lcv/ssim/)
- [Resize-convolution and checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/)
- [Adjusted boxplots for skewed distributions](https://doi.org/10.1016/j.csda.2007.11.008)
- [Autoencoder anomaly-detection limitations](https://arxiv.org/abs/2501.13864)

The provided problem statement and rules govern all competition requirements. The full research note distinguishes supported facts, project-specific inferences and candidate experiments.
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':'Mars_HiRISE_Research_and_Baseline.ipynb'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    for i,cell in enumerate(cells): cell['id']=f'mars-{i:03d}'
    destination=ROOT/'notebooks'/'Mars_HiRISE_Research_and_Baseline.ipynb'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(destination)

if __name__=='__main__':main()
