"""Build a self-contained Colab notebook with readable, visible Python definitions."""
from pathlib import Path
import ast,json,pprint
ROOT=Path(__file__).resolve().parents[1]

class NotebookSource(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None if node.level or (node.module or '').startswith('mars_anomaly') else node
    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        return None if node.name=='main' else self.generic_visit(node)

def visible_source(path):
    tree=NotebookSource().visit(ast.parse((ROOT/path).read_text(encoding='utf-8-sig')))
    tree.body=[node for node in tree.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='insert')]
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def main():
    final=json.loads((ROOT/'outputs/improved_selection.json').read_text())
    hypotheses=json.loads((ROOT/'outputs/improved_hypotheses.json').read_text())
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})

    md("""# Mars image anomaly detection
## NSSC 2026 — my experiments and final analysis

My aim is to find unusual crops using a convolutional autoencoder trained from scratch, followed by Isolation Forest on its one-dimensional latent vector.

I started with a reconstruction baseline, changed the loss, and tested a larger vector. The first candidate list was dominated by very bright crops and changed across repeated runs. I added two preprocessing experiments to investigate that instead of assuming that the highest scores were genuine anomalies.

**Execution note:** I prepared this notebook for Colab or Kaggle. The saved experiment results and the recorded notebook execution were produced locally on an RTX 3050. The setup below prints the actual runtime; I have not relabelled the local results as a Colab execution.

There are no hidden anomaly labels available to me. I can measure reconstruction, nuisance sensitivity and repeatability, but not detection precision or recall.
""")
    md("""## 0. Setup
On Colab I would select **Runtime → Change runtime type → GPU**. I keep the notebook and supplied data private. I use no outside training images or pretrained feature extractors.
""")
    code("""from pathlib import Path
import os, sys, json, importlib.util, subprocess
if Path('/content').exists():
    ROOT = Path('/content/mars_project')
elif Path('/kaggle/working').exists():
    ROOT = Path('/kaggle/working/mars_project')
else:
    ROOT = Path.cwd()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
packages = {'numpy':'numpy', 'pandas':'pandas', 'PIL':'Pillow', 'scipy':'scipy',
            'sklearn':'scikit-learn', 'matplotlib':'matplotlib',
            'statsmodels':'statsmodels', 'torch':'torch'}
missing = [p for m,p in packages.items() if importlib.util.find_spec(m) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage, Markdown
print('Folder:', ROOT)
print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
""")
    md("""## 0.1 Data and preprocessing
I keep all crops from a source observation in the same split. Otherwise, similar acquisition conditions could appear in both training and validation. The main split has 120 training, 25 validation and 27 calibration sources.

I compare three input options:
- **raw:** original grayscale divided by 255.
- **contrast:** subtract the crop mean, divide by its standard deviation, then clip 0.5 + 0.15 × z to [0,1].
- **footprint:** estimate those statistics outside exact-black regions connected to the image border, then fill those regions with 0.5.

I floor the standard deviation at 1/255 for nearly constant crops. The footprint mask is my heuristic, not supplied ground truth. It could affect a border-connected shadow; normalization can also remove meaningful albedo differences.
""")
    code(visible_source('mars_anomaly/preprocessing.py'))
    code(visible_source('mars_anomaly/data.py'))
    code("""DATA = ROOT / 'data'
ARCHIVE = None  # Set an explicit outer dataset ZIP if necessary.
if not (DATA / 'crop_metadata_index.csv').exists():
    candidates = list(ROOT.glob('DATASETS-*.zip'))
    if Path('/kaggle/input').exists():
        candidates += list(Path('/kaggle/input').rglob('DATASETS-*.zip'))
    if ARCHIVE is None and candidates:
        ARCHIVE = candidates[0]
    if ARCHIVE is None and Path('/content').exists():
        from google.colab import files
        uploaded = files.upload()
        archives = [Path(n) for n in uploaded if n.endswith('.zip')]
        if len(archives) == 1:
            ARCHIVE = archives[0]
    if ARCHIVE is None:
        raise FileNotFoundError('Set ARCHIVE to the supplied DATASETS outer ZIP.')
    prepare_archive(ARCHIVE, DATA)
manifest = load_manifest(DATA, seed=2026)
assert len(manifest) == 10422 and manifest.filename.is_unique
assert manifest.groupby('source_image_id').split.nunique().max() == 1
display(manifest.groupby('split').agg(crops=('filename','size'),
                                     sources=('source_image_id','nunique')))
display(manifest[['latitude','longitude','sun_angle','season','resolution']].nunique())
""")
    md("""My complete audit decoded 10,422 crops as 227 × 227 grayscale images, checked all joins and found no exact decoded-pixel duplicates. This does not rule out near duplicates or overlapping terrain. Longitude has only two values, and resolution units are unspecified.

Before training, I check preprocessing on fixed validation images. These are sanity examples, not anomaly labels.
""")
    code("""preview = manifest[manifest.split == 'validation'].iloc[[0,1,6,7]]
fig, axes = plt.subplots(4,3,figsize=(9,11))
for i,(_,row) in enumerate(preview.iterrows()):
    raw,_ = CropDataset(pd.DataFrame([row]))[0]
    for j,mode in enumerate(['raw','contrast','footprint']):
        axes[i,j].imshow(preprocess_array(raw[0].numpy(),mode),cmap='gray',vmin=0,vmax=1)
        axes[i,j].set_title(f'{row.filename}: {mode}',fontsize=9)
        axes[i,j].axis('off')
fig.tight_layout()
display(fig)
plt.close(fig)
""")
    md("""## Phase 1.1 — My autoencoder
I use five stride-two convolution blocks with channels 16, 32, 64, 96 and 128. Spatial sizes become 114, 57, 29, 15 and 8. A linear layer maps the final feature map to 128 or 256 numbers. The decoder must reconstruct through this vector; there is no skip connection around it.

GroupNorm supports my small batches. Resize-convolution gives explicit output sizes and avoids uneven transposed-convolution overlap. All weights start randomly.

## Phase 1.2 — Loss
v1 uses MSE. For v2 onward I use 0.9 MSE + 0.1 (1 − SSIM), because the baseline smoothed fine texture. SSIM uses an 11 × 11 Gaussian window and direct pixel statistics, without a pretrained network. I report common metrics separately.
""")
    code(visible_source('mars_anomaly/model.py'))
    code("""torch.set_num_threads(4)
for dimension in [128,256]:
    example_model = ConvAutoencoder(dimension)
    with torch.no_grad():
        reconstruction,vector = example_model(torch.zeros(1,1,227,227))
    print(dimension,'dimensions;',sum(p.numel() for p in example_model.parameters()),
          'parameters;',tuple(vector.shape),'latent;',tuple(reconstruction.shape),'output')
del example_model
""")
    md("""## Training and saved evidence
Each epoch saves metrics and a fixed reconstruction panel. I select the checkpoint using that run's validation objective. A resumed run must use the same configuration. The plotting and training functions are visible here so this notebook runs without a separate Python package.
""")
    code(visible_source('mars_anomaly/visualize.py'))
    code(visible_source('mars_anomaly/train.py'))
    specs=[dict(name=n,latent_dim=d,structural_weight=w,preprocessing=m,seed=2026,split_seed=2026) for n,d,w,m in [('v1',128,0.,'raw'),('v2',128,.1,'raw'),('v3',256,.1,'raw'),('v4',256,.1,'contrast'),('v5',256,.1,'footprint')]]
    code('MAIN_RUNS = '+pprint.pformat(specs,sort_dicts=False,width=90)+'\nROBUSTNESS_RUNS = '+pprint.pformat(final['robustness_specs'],sort_dicts=False,width=90)+"""
RUN_ROBUSTNESS = False  # True reproduces the extra independent runs on a fresh runtime.
run_specs = MAIN_RUNS + (ROBUSTNESS_RUNS if RUN_ROBUSTNESS else [])
for spec in run_specs:
    run = ROOT / 'outputs' / spec['name']
    config = TrainConfig(version=spec['name'],latent_dim=spec['latent_dim'],
                         structural_weight=spec['structural_weight'],
                         preprocessing=spec['preprocessing'],seed=spec['seed'],
                         split_seed=spec['split_seed'],epochs=15)
    if not (run/'training_status.json').exists():
        run_training(DATA,run,config,resume=(run/'last.pt').exists())
    saved = json.loads((run/'config.json').read_text())
    saved.setdefault('split_seed',saved['seed'])  # Early runs used the same seed for both.
    for key in ['latent_dim','structural_weight','seed','split_seed','epochs']:
        assert saved[key] == getattr(config,key), f'{spec["name"]}: mismatched {key}'
    assert saved.get('preprocessing','raw') == config.preprocessing
    assert json.loads((run/'training_status.json').read_text())['status'] == 'complete'
    history = pd.read_csv(run/'history.csv')
    best = history.loc[history.validation_loss.idxmin()]
    print(spec['name'],'| domain:',config.preprocessing,'| epoch:',int(best.epoch),
          '| MSE:',round(best.validation_mse,6),'| SSIM:',round(best.validation_ssim,6))
""")
    md("""## What I changed
| Run | Change | Reason |
|---|---|---|
| v1 | MSE and 128-dimensional vector | Establish the baseline |
| v2 | Add SSIM | Investigate smoothed texture |
| v3 | Increase the vector to 256 | Test additional capacity |
| v4 | Standardize brightness and contrast | Investigate the bright-candidate concentration |
| v5 | Add border-connected-zero treatment | Isolate the additional footprint effect |

The standardized experiments reconstruct different input values. Their MSE is not directly comparable with raw-image v3 MSE.
""")
    code("""fig,axes = plt.subplots(2,2,figsize=(12,7))
for row,names in enumerate([['v1','v2','v3'],['v4','v5']]):
    for name in names:
        h = pd.read_csv(ROOT/'outputs'/name/'history.csv')
        axes[row,0].plot(h.epoch,h.validation_mse,label=name)
        axes[row,1].plot(h.epoch,h.validation_ssim,label=name)
    for col,metric in enumerate(['MSE','SSIM']):
        axes[row,col].set(xlabel='Epoch',ylabel=metric,
                          title='Raw input' if row == 0 else 'Standardized inputs: different targets')
        axes[row,col].legend()
fig.tight_layout()
display(fig)
plt.close(fig)
for name in ['v3','v4','v5']:
    print(name)
    display(DisplayImage(filename=str(ROOT/'outputs'/name/'reconstructions_latest.png')))
""")
    md("""## Phase 2.1 — Isolation Forest on the latent vector
I fit 2,000 trees with 256 samples per tree, only on training latents. My earlier controlled comparison found better forest-seed agreement than at 400 trees. Novelty is negative score_samples: larger means more unusual. I ignore the library's contamination cutoff.

## Phase 2.2 — My statistical boundary
The original adjusted-boxplot threshold exceeded every v1 score. Its histogram was multimodal, so I investigated a distribution model instead of lowering the threshold to obtain five examples.

I fit one to four Gaussian components on calibration scores, choose minimum BIC, and use **T = max(component mean + 3 × component standard deviation)**. The rule puts a boundary beyond each fitted population rather than labelling the entire high-score population anomalous.

I refit model selection in 100 whole-source bootstrap samples. This is conditional uncertainty, not a guaranteed false-positive rate. Correlated crops, unknown contamination and limited model selection remain weaknesses. The same rule applies to every run; zero candidates is allowed.
""")
    code(visible_source('mars_anomaly/threshold.py'))
    code(visible_source('mars_anomaly/evaluate.py'))
    code("""analysis_name = 'mixture_three_sigma_trees2000'
for spec in MAIN_RUNS + ROBUSTNESS_RUNS:
    run = ROOT/'outputs'/spec['name']
    if not (run/'training_status.json').exists():
        print(spec['name'],': optional repeat absent from this runtime')
        continue
    analysis = run/analysis_name
    if not (analysis/'selected_for_interpretation.csv').exists():
        evaluate(DATA,run,bootstrap=100,projection=False,
                 method='mixture_three_sigma',trees=2000)
    d = json.loads((analysis/'diagnostics.json').read_text())
    c = json.loads((analysis/'calibration.json').read_text())
    print(spec['name'],'| threshold:',round(c['threshold'],6),'| flags:',d['flagged_total'])
""")
    md('## My selection after the checks\n\n'+final['rationale']+'\n\n'+final['robustness_summary'])
    selection_record = {key:final[key] for key in ['run','analysis','status','recorded_at']}
    code('RECORDED_SELECTION = '+pprint.pformat(selection_record,sort_dicts=False,width=90)+"""
RUN = ROOT/'outputs'/RECORDED_SELECTION['run']
ANALYSIS = RUN/analysis_name
scored = pd.read_csv(ANALYSIS/'novelty_scores.csv')
calibration = json.loads((ANALYSIS/'calibration.json').read_text())
diagnostics = json.loads((ANALYSIS/'diagnostics.json').read_text())
display({k:v for k,v in calibration.items() if k != 'bootstrap_thresholds'})
display(scored.groupby('split').flagged.agg(['count','sum','mean']))
display(DisplayImage(filename=str(ANALYSIS/'score_distribution.png')))
""")
    md("""## Phase 1.3 — Looking at the latent space
PCA and t-SNE are only for plotting; the forest receives the full vector. I compare two perplexities and color by novelty, sun angle and source. Branches are not verified geological classes, and effective rank is not a class count.
""")
    code(visible_source('scripts/project_latents.py'))
    code("""if not all((ANALYSIS/f'tsne_{p}.png').exists() for p in [30,70]):
    project(ANALYSIS)
for p in [30,70]:
    display(DisplayImage(filename=str(ANALYSIS/f'tsne_{p}.png')))
print('Effective rank:',diagnostics['effective_rank'],'of',diagnostics['latent_dimensions'])
""")
    md("""## Did the improvements reduce the problem?
I report both full-ranking correlation and overlap of the flagged sets. A high correlation can conceal disagreement in the extreme tail. I also hold each fitted forest and cutoff fixed while applying 0.8 × x + 0.1 outside the inferred black border. Keeping the footprint fixed simulates exposure changes rather than changing crop support. This is a nuisance-sensitivity check, not labelled detection accuracy.
""")
    code(visible_source('scripts/summarize_experiments.py'))
    code("""summarize(ROOT)
display(pd.read_csv(ROOT/'outputs/comparison/experiments.csv'))
display(pd.read_csv(ROOT/'outputs/comparison/run_stability.csv'))
""")
    code(visible_source('scripts/measure_photometric_sensitivity.py'))
    code("""measure(ROOT,['v3','v4','v5'])
display(pd.read_csv(ROOT/'outputs/comparison/photometric_sensitivity.csv'))
""")
    md("""### What the brightness test misses
The illumination perturbation is deliberately simple and favors an affine-invariant transform. I do not treat passing it as proof that real anomalies are retained.

After calibration, I inspected the five leading v4 examples. All contain a conspicuous black strip or border. The exact-zero mask in v5 can leave near-black edge fragments, so filling zeros does not remove every acquisition feature. These are concrete reasons to avoid promoting preprocessing solely on its brightness result.
""")
    code("""for name in ['v4','v5']:
    print(name,': the same validation crop, shown after each independently calibrated selection')
    recorded_example = ROOT/'outputs'/name/analysis_name/'heatmap_sample_09852.png'
    if recorded_example.exists():
        display(DisplayImage(filename=str(recorded_example)))
    else:
        print('This saved example is absent from the new run; inspect its own selected crops.')
""")
    md("""## Phase 2.3 — Supplied metadata and source context
I use the organizer's crop-to-source index. I do not infer coordinates from appearance. Crop-weighted and equal-source rates differ because source sizes differ. These are descriptive summaries, not causal tests.

### Phase 2.4 — Optional metadata fusion
I keep metadata out of the score and do not claim the optional fusion credit.
""")
    code("""display(pd.read_csv(ANALYSIS/'source_summary.csv').sort_values('flag_rate',ascending=False).head(15))
for field in ['season','resolution','longitude']:
    print(field)
    display(pd.read_csv(ANALYSIS/f'metadata_{field}.csv'))
""")
    md("""## Phase 3.1 — Inspecting the selected crops
I apply the threshold first and then select at most five eligible crops. I never lower it to force five.

For raw-input runs, error compares the original crop with its reconstruction. Standardized panels show the raw crop separately from the model input: their error compares standardized input with reconstruction. The overlay shows location, and is not direct attribution for Isolation Forest.
""")
    code("""selected = pd.read_csv(ANALYSIS/'selected_for_interpretation.csv')
assert len(selected) <= 5 and (selected.novelty_score > calibration['threshold']).all()
display(selected)
for filename in selected.filename:
    display(DisplayImage(filename=str(ANALYSIS/f'heatmap_{Path(filename).stem}.png')))
if selected.empty:
    print('No crop passes the boundary. I do not manufacture a top-five list.')
""")
    md("""### Do these exact candidates repeat?
I check the selected filenames against each run's own calibrated flags. This makes disagreement visible at the candidate level. A missing optional run is shown as unavailable rather than treated as a negative result. Agreement still does not establish a genuine anomaly.
""")
    code("""candidate_checks = selected[['filename','source_image_id']].copy()
for name in ['v3','robust_seed','robust_split','v4','v5','improved_seed','improved_split']:
    score_file = ROOT/'outputs'/name/analysis_name/'novelty_scores.csv'
    if score_file.exists():
        other = pd.read_csv(score_file)
        eligible = set(other.loc[other.flagged,'filename'])
        candidate_checks[name] = candidate_checks.filename.isin(eligible)
    else:
        candidate_checks[name] = 'Unavailable'
display(candidate_checks)
""")
    md("""## Phase 3.2 — My physical interpretations
These notes describe the saved images I inspected. They remain hypotheses with alternatives. A new run that selects different crops needs a new visual review.
""")
    code('HYPOTHESES = '+pprint.pformat(hypotheses,sort_dicts=False,width=90)+"""
for filename in selected.filename:
    item = HYPOTHESES.get(filename)
    if item is None:
        display(Markdown('### '+filename+chr(10)+'New selection: visual review needed.'))
        continue
    parts = ['### '+filename,'**What I can see:** '+item['evidence'],
             '**Possible explanation:** '+item['hypothesis'],
             '**Alternative:** '+item['alternative']]
    if item.get('reference'):
        parts.append('[Process background; not a crop identification]('+item['reference']+')')
    display(Markdown((chr(10)*2).join(parts)))
""")
    md('## Phase 4 — What happened in my experiments\n\n'+final['student_journal'])
    md("""## What I would test next
I would test more independent source partitions and longer learning curves under a fixed budget. I would also check whether normalization hides real albedo anomalies. Appearance alone cannot decide that; organizer-held labels would be needed for precision and recall.

## Sources
- [SSIM author resource](https://www.cns.nyu.edu/~lcv/ssim/)
- [Resize-convolution](https://distill.pub/2016/deconv-checkerboard/)
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Gaussian mixture selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
- [Connected-region labelling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)

The constraints come from the three supplied competition documents. My decision log preserves failed approaches and verification. Team details and private repository access remain submission logistics.
""")
    for i,c in enumerate(cells):
        c['id']=f'mars-colab-{i:03d}'
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell {i}','exec')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'accelerator':'GPU','colab':{'name':'Mars_HiRISE_Colab.ipynb','provenance':[]}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/Mars_HiRISE_Colab.ipynb'
    target.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    print('Built',len(cells),'cells:',target)

if __name__=='__main__':main()
