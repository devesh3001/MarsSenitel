"""Produce the final portable notebook with all four rubric phases."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    final=json.loads((ROOT/'outputs/final_selection.json').read_text())
    hypotheses=json.loads((ROOT/'outputs/geological_hypotheses.json').read_text())
    journal=(ROOT/'ENGINEERING_CHANGELOG.md').read_text(encoding='utf-8')
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'outputs':[],'execution_count':None})
    md('''# NSSC 2026 Mars HiRISE anomaly detection

This notebook records three from-scratch autoencoder experiments, latent Isolation Forest scoring, distribution-based thresholding, source metadata analysis and reconstruction hypotheses. It uses no pretrained weights or external image pretraining. **Higher novelty score means greater anomalousness.** Hidden anomaly labels and the payload count are unavailable, so no detection-accuracy claim is made.

The executed output cells document the delivered runs. To reproduce on Colab or Kaggle, select a GPU and provide the original outer dataset ZIP (a private input on Kaggle). All model modules are embedded. Fresh execution trains missing runs and reuses completed ones. GPU numerical differences may change scores; inspect regenerated outputs before reusing recorded physical hypotheses. No cloud compute is purchased and no data is published by this notebook.
''')
    code('''from pathlib import Path
import os, sys, json, importlib.util, subprocess
if Path('/kaggle/working').exists(): ROOT=Path('/kaggle/working/mars_project')
elif Path('/content').exists(): ROOT=Path('/content/mars_project')
else:
    ROOT=Path.cwd()
    if ROOT.name=='notebooks':ROOT=ROOT.parent
ROOT.mkdir(parents=True,exist_ok=True);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
required={'numpy':'numpy','pandas':'pandas','PIL':'Pillow','scipy':'scipy','sklearn':'scikit-learn','matplotlib':'matplotlib','statsmodels':'statsmodels','torch':'torch'}
missing=[v for k,v in required.items() if importlib.util.find_spec(k) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import torch,numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Image,Markdown
print('Project:',ROOT,'| PyTorch:',torch.__version__,'| CUDA:',torch.cuda.is_available())
if torch.cuda.is_available():print(torch.cuda.get_device_name())
''')
    embedded={str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8') for p in (ROOT/'mars_anomaly').glob('*.py')}
    for name in ['summarize_experiments.py','project_latents.py','inspect_selected_context.py']:embedded['scripts/'+name]=(ROOT/'scripts'/name).read_text(encoding='utf-8')
    code('EMBEDDED_FILES = '+repr(embedded)+'''
for name,source in EMBEDDED_FILES.items():
    target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_text(encoding='utf-8')!=source:
        raise RuntimeError(f'{target} differs from this notebook. Rebuild the notebook after source edits.')
    if not target.exists():target.write_text(source,encoding='utf-8')
print('Embedded source verified.')
''')
    md('''## Data audit and source-disjoint validation

The complete input audit decoded **10,422 227 by 227 grayscale crops** from **172 source observations**, found no exact decoded-pixel duplicates and verified every metadata join. The crop index and source table have unique keys. Source sizes range from 1 to 278. Near-duplicate overlap remains possible.

The main experiments use 120 training, 25 validation and 27 calibration source IDs. Crop counts are 7,101, 1,666 and 1,655 respectively. All crops from a source stay in one split. No visually unusual image is removed or given a pseudo-ground-truth label. Intensities are divided by 255; source IDs and filenames are join keys, not detector features.
''')
    code('''from mars_anomaly.data import prepare_archive,load_manifest,CropDataset
DATA=ROOT/'data'
ARCHIVE=None  # Set an explicit outer ZIP path here if needed.
if not (DATA/'crop_metadata_index.csv').exists():
    candidates=list(ROOT.glob('DATASETS-*.zip'))
    if Path('/kaggle/input').exists():candidates+=list(Path('/kaggle/input').rglob('DATASETS-*.zip'))
    if ARCHIVE is None and candidates:ARCHIVE=candidates[0]
    if ARCHIVE is None and importlib.util.find_spec('google') and importlib.util.find_spec('google.colab'):
        from google.colab import files
        uploaded=files.upload()
        candidates=[Path(n) for n in uploaded if n.endswith('.zip')]
        if candidates:ARCHIVE=candidates[0]
    if ARCHIVE is None:raise FileNotFoundError('Provide the original dataset ZIP via ARCHIVE.')
    prepare_archive(ARCHIVE,DATA)
manifest=load_manifest(DATA,2026)
assert len(manifest)==10422 and manifest.filename.is_unique
assert manifest.groupby('source_image_id').split.nunique().max()==1
display(manifest.groupby('split').agg(images=('filename','size'),sources=('source_image_id','nunique')))
display(manifest.drop(columns=['path','filename','split']).drop_duplicates().nunique().to_frame('Unique values'))
''')
    code('''sample=manifest.sample(12,random_state=2026)
fig,axes=plt.subplots(3,4,figsize=(9,7))
for ax,(_,row) in zip(axes.flat,sample.iterrows()):
    x,_=CropDataset(pd.DataFrame([row]))[0]
    ax.imshow(x[0],cmap='gray',vmin=0,vmax=1);ax.set_title(row.filename,fontsize=8);ax.axis('off')
fig.suptitle('Seeded visual sanity sample; no anomaly labels');fig.tight_layout();display(fig);plt.close(fig)
''')
    md('''## Phase 1: Deep Latent Compression (Autoencoders)
### 1.1 Architecture Design

Five stride-two convolution blocks produce spatial sizes 114, 57, 29, 15 and 8 with channels 16, 32, 64, 96 and 128. GroupNorm and SiLU support small batches. The final feature grid is flattened and projected to a fixed 128- or 256-dimensional vector. The decoder expands this vector and uses resize-convolution to reconstruct exactly 227 by 227 pixels. No skip connection bypasses the vector.

There are 51,529 input values: roughly 403 per coordinate at dimension 128 and 201 at dimension 256. The larger code is evaluated empirically; compression ratio alone does not determine anomaly separability.
''')
    code('''from mars_anomaly.model import ConvAutoencoder
torch.set_num_threads(4)
for dimension in [128,256]:
    model=ConvAutoencoder(dimension)
    with torch.no_grad():reconstruction,z=model(torch.zeros(1,1,227,227))
    print(dimension,'dimensions;',sum(p.numel() for p in model.parameters()),'parameters;',tuple(z.shape),'latent;',tuple(reconstruction.shape),'output')
del model
''')
    md('''### 1.2 Loss Function & Training

v1 uses MSE. Its fixed reconstruction panel showed strong texture smoothing. v2 changes only the objective to **0.9 MSE + 0.1 (1 - SSIM)**. SSIM is computed from an 11 by 11 Gaussian window (sigma 1.5, intensity range 1), without pretrained features. v3 keeps that loss and changes the vector from 128 to 256 dimensions after v2 retained visible smoothing.

Training uses AdamW, learning rate 0.0003, weight decay 0.0001, batch size 16, at most 15 epochs and patience 5. Both train and validation report common MSE, SSIM and gradient error. Each run saves best and resumable checkpoints. Raw mixed-objective loss is not compared against raw MSE as a quality metric.
''')
    specs=[{'name':'v1','latent_dim':128,'structural_weight':0.0,'seed':2026,'split_seed':2026},{'name':'v2','latent_dim':128,'structural_weight':.1,'seed':2026,'split_seed':2026},{'name':'v3','latent_dim':256,'structural_weight':.1,'seed':2026,'split_seed':2026},{'name':'v7','latent_dim':256,'structural_weight':.1,'seed':2026,'split_seed':2026},{'name':'v8','latent_dim':256,'structural_weight':.1,'seed':2026,'split_seed':2026}]
    code('''from mars_anomaly.train import TrainConfig,run_training
SPECS = '''+repr(specs)+'''
for spec in SPECS:
    run=ROOT/'outputs'/spec['name']
    config=TrainConfig(version=spec['name'],latent_dim=spec['latent_dim'],structural_weight=spec['structural_weight'],seed=spec['seed'],split_seed=spec['split_seed'])
    if not (run/'training_status.json').exists():run_training(DATA,run,config,resume=(run/'last.pt').exists())
    status=json.loads((run/'training_status.json').read_text())
    assert status['status']=='complete'
    history=pd.read_csv(run/'history.csv');best=history.loc[history.validation_loss.idxmin()]
    print(spec['name'],'best epoch',int(best.epoch),'MSE',best.validation_mse,'SSIM',best.validation_ssim)
''')
    code('''fig,axes=plt.subplots(1,3,figsize=(13,3.5))
for spec in SPECS:
    history=pd.read_csv(ROOT/'outputs'/spec['name']/'history.csv')
    for ax,metric in zip(axes,['mse','ssim','gradient']):ax.plot(history.epoch,history['validation_'+metric],label=spec['name'])
for ax,metric in zip(axes,['MSE','SSIM','Gradient error']):ax.set(xlabel='Epoch',ylabel=metric);ax.legend()
fig.tight_layout();display(fig);plt.close(fig)
for spec in SPECS:
    print(spec['name']);display(Image(filename=str(ROOT/'outputs'/spec['name']/'reconstructions_latest.png')))
''')
    md('''## Phase 2: Isolation Forest Novelty Engine
### 2.1 Latent Extraction & Scoring

Fit 2,000 trees with max_samples 256 and contamination="auto" on training latents only. This increases the original 400-tree budget after a controlled forest-seed experiment demonstrated better candidate agreement. The score is **negative score_samples**, so larger scores are more anomalous. The built-in prediction offset is ignored. The detector uses the full original vector, not the 2D visualization, reconstruction error or metadata.

### 2.2 Statistical Thresholding

The original adjusted-boxplot threshold for v1 was 0.87484, above the maximum observed score of 0.55648. Its strongly skewed, multimodal score distribution and wide source-bootstrap interval motivated a revised distribution model. The zero-flag result is preserved. It was not lowered merely to obtain five candidates.

Fit Gaussian mixtures with one through four components on calibration scores. Choose the converged model with lowest BIC and set **T = max_j(mu_j + 3 sigma_j)**. This is an envelope beyond every fitted score population; it does not label the upper population itself as anomalous. The multiplier is fixed across runs. No contamination fraction, top-N cutoff or desired output count is supplied. BIC remains a heuristic with correlated crops, and the fitted Gaussian tail bound is not a guarantee of the true false-positive rate.

Refit model selection in 100 source-group bootstrap samples. Report the interval and conditional flag frequencies. Separately refit two forest seeds. Validate density fit descriptively on validation scores without reporting an IID KS p-value.

In v3 the four-component BIC beats three components by only about 0.85, and four is the search limit. Model order is weakly separated. Bootstrap refitting captures some model-selection uncertainty, but does not validate the tail model.
''')
    code('''from mars_anomaly.evaluate import evaluate
for spec in SPECS:
    run=ROOT/'outputs'/spec['name'];analysis=run/'mixture_three_sigma_trees2000'
    if not (analysis/'diagnostics.json').exists():evaluate(DATA,run,bootstrap=100,projection=False,method='mixture_three_sigma',trees=2000)
    calibration=json.loads((analysis/'calibration.json').read_text());diagnostics=json.loads((analysis/'diagnostics.json').read_text())
    print(spec['name'],'threshold',calibration['threshold'],'flags',diagnostics['flagged_total'],'effective rank',diagnostics['effective_rank'])
    display(pd.DataFrame(diagnostics['forest_seed_stability']))
''')
    code('FINAL_SELECTION = '+repr(final)+'''
RUN=ROOT/'outputs'/FINAL_SELECTION['run'];ANALYSIS=RUN/FINAL_SELECTION['analysis']
calibration=json.loads((ANALYSIS/'calibration.json').read_text())
diagnostics=json.loads((ANALYSIS/'diagnostics.json').read_text())
scored=pd.read_csv(ANALYSIS/'novelty_scores.csv')
display(Markdown('**Recorded final selection:** '+FINAL_SELECTION['run']+'. '+FINAL_SELECTION['rationale']))
display({k:v for k,v in calibration.items() if k!='bootstrap_thresholds'})
display(Image(filename=str(ANALYSIS/'score_distribution.png')))
display(scored.groupby('split').flagged.agg(['count','sum','mean']))
''')
    code('''sensitivity=[]
for multiplier in [2.5,3.0,3.5]:
    threshold=float(np.max(np.array(calibration['means'])+multiplier*np.array(calibration['std'])))
    sensitivity.append({'SD multiplier':multiplier,'Threshold':threshold,'Flags':int((scored.novelty_score>threshold).sum()),'Role':'Final fixed rule' if multiplier==3 else 'Sensitivity only'})
display(pd.DataFrame(sensitivity))
print('This comparison does not change the predefined multiplier of 3.')
''')
    md('''### 2.3 Latent Visualization Diagnostics

The full latent goes to Isolation Forest. Standardization and PCA to 50 dimensions are used only to prepare t-SNE at two perplexities. The projections are colored by novelty, sun angle and five large source groups. They are qualitative diagnostics; neither visual clusters nor a dense blob establishes geological classes or encoder collapse. Effective rank and coordinate variances provide complementary evidence.
''')
    code('''if not all((ANALYSIS/f'tsne_{p}.png').exists() for p in [30,70]):
    subprocess.check_call([sys.executable,'scripts/project_latents.py','--directory',str(ANALYSIS)])
for p in [30,70]:display(Image(filename=str(ANALYSIS/f'tsne_{p}.png')))
print('Covariance effective rank:',diagnostics['effective_rank'],'of',diagnostics['latent_dimensions'])
''')
    md('''### 2.5 Location Analysis & Metadata

All joins use the organizer's supplied index. There is no catalog scraping. Longitude contains only 0 and 180, so a detailed geographic map would imply unsupported precision. Latitude has 24 values, season has four categories, and resolution has three values. Sun-angle semantics and resolution units are not explicitly defined. Crop-weighted and equal-source rates are both shown because source sizes differ.
''')
    code('''sources=pd.read_csv(ANALYSIS/'source_summary.csv')
display(sources.sort_values('flag_rate',ascending=False).head(15))
for field in ['season','resolution','longitude']:
    print(field);display(pd.read_csv(ANALYSIS/f'metadata_{field}.csv'))
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,field in zip(axes,['latitude','sun_angle']):
    ax.scatter(sources[field],sources.flag_rate,s=18)
    ax.set(xlabel=field+' as supplied',ylabel='Flagged fraction within source')
fig.tight_layout();display(fig);plt.close(fig)
''')
    md('''### Robustness Across Independent Runs

The primary comparisons hold the source split fixed. Additional runs separate initialization changes from source-partition changes. Rankings and flagged-set overlap measure reproducibility, not correctness. A changed split is compared additionally on crops held out from both encoders. The code below reuses completed checks; set RUN_ROBUSTNESS=True to reproduce missing ones on a fresh cloud session.
''')
    rob_specs=final.get('robustness_specs',[])
    code('ROBUSTNESS_SPECS = '+repr(rob_specs)+'''
RUN_ROBUSTNESS=False
if RUN_ROBUSTNESS:
    for spec in ROBUSTNESS_SPECS:
        run=ROOT/'outputs'/spec['name']
        if not (run/'training_status.json').exists():
            config=TrainConfig(version=spec['name'],latent_dim=spec['latent_dim'],structural_weight=spec['structural_weight'],seed=spec['seed'],split_seed=spec['split_seed'])
            run_training(DATA,run,config,resume=(run/'last.pt').exists())
        if not (run/'mixture_three_sigma_trees2000/diagnostics.json').exists():evaluate(DATA,run,projection=False,method='mixture_three_sigma',trees=2000)
subprocess.check_call([sys.executable,'scripts/summarize_experiments.py'])
display(pd.read_csv(ROOT/'outputs/comparison/experiments.csv'))
display(pd.read_csv(ROOT/'outputs/comparison/run_stability.csv'))
if (ROOT/'outputs/comparison/confounds.csv').exists():display(pd.read_csv(ROOT/'outputs/comparison/confounds.csv'))
''')
    md('''## Phase 3: Reconstruction Interpretability
### 3.1 Heatmap Generation

Only images strictly above T are eligible. Select at most five in descending novelty order. If fewer than five exceed T, analyze all of them without lowering the boundary. The common absolute-error scale is [0,1] for all candidate maps; the original and reconstruction also use the same intensity range. Error localization is not a direct attribution of the Isolation Forest.
''')
    code('''selected=pd.read_csv(ANALYSIS/'selected_for_interpretation.csv')
assert (selected.novelty_score>calibration['threshold']).all() and len(selected)<=5
display(selected)
for filename in selected.filename:display(Image(filename=str(ANALYSIS/f'heatmap_{Path(filename).stem}.png')))
if selected.empty:print('No images pass the boundary; no top-five set is manufactured.')
subprocess.check_call([sys.executable,'scripts/inspect_selected_context.py','--directory',str(ANALYSIS),'--data',str(DATA)])
display(pd.read_csv(ANALYSIS/'selected_acquisition_diagnostics.csv')[['filename','mean','brightness_percentile','brightness_percentile_within_source']])
print('Brightness diagnostics describe a potential confound. They do not change the pre-established threshold or validate anomaly labels.')
''')
    md('### 3.2 Geological Report & Hypotheses\n\nThe following interpretations were written after inspecting the delivered selected images. They remain hypotheses. If a reproduction changes the selected filenames, review the new images rather than copying an old interpretation.')
    code('HYPOTHESES = '+repr(hypotheses)+'''
for filename in selected.filename:
    item=HYPOTHESES.get(filename)
    if item is None:
        display(Markdown('**'+filename+'**: newly selected in this execution; visual interpretation required.'))
    else:
        display(Markdown('### '+filename+'\\n\\n**Evidence:** '+item['evidence']+'\\n\\n**Hypothesis:** '+item['hypothesis']+'\\n\\n**Alternative and limitation:** '+item['alternative']+'\\n\\n[Process background; not a crop identification]('+item['reference']+')'))
''')
    md('## Phase 4: Architecture Iteration & Design Journal\n\n'+journal)
    md('''## Limitations and reproducibility

No hidden labels are available. Source grouping reduces one form of leakage but does not rule out geographically overlapping observations. All training data may contain injected content. Fine texture is imperfectly reconstructed. A fitted mixture can absorb contaminated tails or misfit them; threshold and seed sensitivity must remain visible. The tested experiment budget does not establish a global optimum.

The supplied PDF, source files, configurations, checkpoint files and CSV results document the delivered run. Local wall times include contention or interruptions and are not GPU benchmarks. Keep the repository and cloud inputs private until competition results are announced. The rules require organizer collaborators and team details to be handled separately.

## References

- [SSIM, original author resource](https://www.cns.nyu.edu/~lcv/ssim/)
- [Resize-convolution and checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/)
- [Isolation Forest score conventions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Gaussian mixture model selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
- [Adjusted boxplot for skewed distributions](https://doi.org/10.1016/j.csda.2007.11.008)
- [Autoencoder anomaly-detection failure modes](https://arxiv.org/abs/2501.13864)

Competition requirements come from the three supplied Word documents. The original data manifest and hashes are recorded in outputs/audit/audit.json.
''')
    for i,cell in enumerate(cells):cell['id']=f'nssc-{i:03d}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/Mars_HiRISE_Submission.ipynb'
    target.write_text(json.dumps(notebook,indent=1),encoding='utf-8');print(target)

if __name__=='__main__':main()
