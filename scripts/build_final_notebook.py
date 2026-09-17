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
    final=json.loads((ROOT/'outputs/canonical_selection.json').read_text())
    hypotheses=json.loads((ROOT/'outputs/geological_hypotheses.json').read_text())
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

I compare four input options:
- **border_fill (v6):** mask border-connected values at most 4/255, expand by two pixels and extend nearest valid standardized values. This can create artificial stripes.
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
v1 uses MSE. For v2-v6 I use 0.9 MSE + 0.1 (1 − SSIM), because the baseline smoothed fine texture. SSIM uses an 11 × 11 Gaussian window and direct pixel statistics, without a pretrained network. I report common metrics separately.
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
    registry=json.loads((ROOT/'outputs/run_registry.json').read_text())
    main_names=['v1','v2','v3','v4','v5','v6','v7','v8']
    specs=[dict(name=n,**registry['runs'][n]) for n in main_names]
    repeats=[dict(name=n,**cfg) for n,cfg in registry['runs'].items() if n not in main_names]
    code('MAIN_RUNS = '+pprint.pformat(specs,sort_dicts=False,width=90)+'\nROBUSTNESS_RUNS = '+pprint.pformat(repeats,sort_dicts=False,width=90)+"""
RUN_ROBUSTNESS = False  # Enable to train missing independent repeats on a fresh runtime.
run_specs = MAIN_RUNS + (ROBUSTNESS_RUNS if RUN_ROBUSTNESS else [])
for spec in run_specs:
    run = ROOT / 'outputs' / spec['name']
    config = TrainConfig(**{k:v for k,v in spec.items() if k != 'name'})
    ensure_training(DATA,run,config)
    history = pd.read_csv(run/'history.csv')
    best = history.loc[history.validation_loss.idxmin()]
    print(spec['name'],'| domain:',config.preprocessing,'| gradient:',config.gradient_weight,
          '| augment:',config.augment,'| epoch:',int(best.epoch),'| MSE:',best.validation_mse)
# Also validate any optional cached runs; never silently reuse a different configuration.
for spec in ROBUSTNESS_RUNS:
    run = ROOT/'outputs'/spec['name']
    if (run/'training_status.json').exists():
        ensure_training(DATA,run,TrainConfig(**{k:v for k,v in spec.items() if k != 'name'}))
""")
    md("""## What I changed
| Run | Change | Reason |
|---|---|---|
| v1 | MSE and 128-dimensional vector | Establish the baseline |
| v2 | Add SSIM | Investigate smoothed texture |
| v3 | Increase the vector to 256 | Test additional capacity |
| v4 | Standardize brightness and contrast | Investigate the bright-candidate concentration |
| v5 | Add border-connected-zero treatment | Isolate the additional footprint effect |
| v6 | Near-black mask and nearest-value fill | Test remaining border fragments |
| v7 | Add 0.1 gradient loss | Test edge reconstruction |
| v8 | Add flips and quarter-turns to v7 | Test orientation sensitivity |

The executed v7/v8 loss is 0.9 MSE + 0.1 (1-SSIM) + 0.1 Gradient. Gradient means adjacent-pixel finite differences, not Sobel filtering. The original v7 protocol described 0.8 MSE; I preserve and disclose that deviation.

The standardized experiments reconstruct different input values. Their MSE is not directly comparable with raw-image v3 MSE.
""")
    code("""fig,axes = plt.subplots(2,2,figsize=(12,7))
for row,names in enumerate([['v1','v2','v3','v7','v8'],['v4','v5','v6']]):
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
PCA and t-SNE are qualitative diagnostics only; the Isolation Forest receives the full 256-dimensional vector, not the projection. I compare two perplexities (30 and 70) and colour each point by its novelty score, its sun angle and its source image ID.

**What I observe:** At both perplexities, the projection shows a primary dense region with a sparse fringe of higher-novelty points. The broad cloud structure does not resolve into clearly separated terrain classes, which is expected: this is an unsupervised representation trained to reconstruct rather than to cluster geology. A few elongated threads near the fringe correspond to crops that score most anomalously; these are the candidates the Isolation Forest assigns its highest novelty scores.

**Effective rank interpretation:** The effective rank is approximately 31 out of 256 dimensions. This means roughly 31 dimensions carry most of the variance — a moderate compression without complete collapse. It does not mean 31 geological classes exist, nor does it confirm encoder collapse (a structureless single blob). The between-source variance fraction is approximately 0.23, meaning source-level acquisition effects account for about a quarter of the total latent spread.

**Limitation:** A visually distinguishable fringe in t-SNE does not confirm that those points are genuine anomalies. t-SNE can distort global distances; the perplexity controls local vs. global neighbourhood emphasis. These projections are presented as visual diagnostics, not proof of geological clustering.
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
pairs = pd.read_csv(ROOT/'outputs/comparison/run_stability.csv')
display(pairs[((pairs.left == 'v3') & pairs.right.isin(['robust_seed','robust_split'])) | ((pairs.left == 'v7') & pairs.right.isin(['v7_robust_seed','v7_robust_split']))])
print('flag_jaccard uses all crops; flag_jaccard_common_heldout uses only crops held out from both models. Empty union is undefined.')
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
I use the organizer's crop-to-source index only; I do not infer coordinates from appearance. Longitude has only two distinct values (0 and 180) so geographic comparison is limited to a binary split. Latitude has 24 coarse values spanning −90° to +90°. Resolution has three values (0.25, 0.5, 1.0) whose units are unspecified by the organizers. Sun-angle semantics (incidence vs. elevation) are not defined in the supplied metadata.

Crop-weighted and equal-source flag rates differ because source sizes range from 1 to 278 crops. I report both to avoid misrepresenting small-source observations.

**Key finding:** Three of the five selected crops come from a single source, SRC_154. Its equal-source flag rate is not exceptional, but SRC_154 contributes 3 of 5 candidates. This source concentration is the dominant geographic pattern in the selection — source acquisition conditions (illumination, sensor state) are a plausible driver of the elevated novelty scores, not necessarily geographically anomalous terrain. This limitation is disclosed in every candidate's physical hypothesis.

### Phase 2.4 — Optional metadata fusion
The primary score uses image latents only. I separately compare matched five-forest image-only and metadata-fused ensembles on the v8 latents. Each pipeline fits its own independent calibration boundary on its own calibration scores — the fused and image-only thresholds are not compared against each other. Metadata enters before the forest, not the decoder. Sun angle and resolution (2 numerical dimensions, training-scaled) plus four one-hot season dimensions add 6 coordinates to the 256-dim image latent, giving a 262-dimensional fused vector.

Both the image-only and fused ensembles produce zero flags under their respective mixture-envelope boundaries. The Spearman correlation between image-only and fused scores is 0.998, indicating that the six metadata dimensions add negligible independent signal at this scale. The candidate set is unchanged. This is the correct result to report — manufacturing candidates by lowering the threshold is explicitly prohibited by the rules.
""")
    code("""display(pd.read_csv(ANALYSIS/'source_summary.csv').sort_values('flag_rate',ascending=False).head(15))
for field in ['season','resolution','longitude']:
    print(field)
    display(pd.read_csv(ANALYSIS/f'metadata_{field}.csv'))
""")
    code(visible_source('scripts/metadata_fusion.py'))
    code("""fusion_result = fusion(ROOT)
display({k:v for k,v in fusion_result.items() if k != 'input_sha256'})
print('An empty flag union has undefined Jaccard; it does not demonstrate correct detection.')
""")
    md("""## Later ablations: what failed
v7 lowers validation MSE by 1.68%, but four of its five selected images remain above the 99.8th brightness percentile. The darker selected crop also has conspicuous borders. I cannot claim that brightness bias was solved.

v6 reduces border-strip response, but fails the predeclared defect-response and forest-stability criteria. Its nearest-value fill introduces artificial stripes. These are negative results, not grounds for changing the threshold.
""")
    code("""for name in ['v6','v7']:
    folder = ROOT/'outputs'/name/analysis_name
    candidates = pd.read_csv(folder/'selected_for_interpretation.csv')
    print(name, 'selected candidates')
    display(candidates[['filename','novelty_score','source_image_id']])
    if len(candidates):
        display(DisplayImage(filename=str(folder/f'heatmap_{Path(candidates.iloc[0].filename).stem}.png')))
""")
    border_tree=ast.parse((ROOT/'scripts/evaluate_border_experiment.py').read_text())
    for node in border_tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='main':node.name='border_response_checks'
    border_tree=NotebookSource().visit(border_tree)
    border_tree.body=[n for n in border_tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='insert')]
    class BorderNames(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='ANALYSIS':node.id='BORDER_ANALYSIS_NAME'
            return node
    code(ast.unparse(ast.fix_missing_locations(BorderNames().visit(border_tree))))
    code("""RERUN_BORDER_CHECKS = False
border_folder = ROOT/'outputs/border_experiment'
if RERUN_BORDER_CHECKS or not (border_folder/'decision.json').exists():
    border_response_checks()
display(pd.read_csv(border_folder/'response_summary.csv'))
display(json.loads((border_folder/'mask_summary.json').read_text()))
display(json.loads((border_folder/'decision.json').read_text()))
print('Synthetic perturbations are diagnostics only; no hidden-label accuracy is inferred.')
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
for name in [spec['name'] for spec in MAIN_RUNS + ROBUSTNESS_RUNS]:
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
Each interpretation follows the same structure: (1) what the heatmap shows spatially, (2) the most plausible physical cause linked to a Genesis Outlier type, (3) an ordinary Martian alternative, and (4) the key limitation. These are hypotheses, not confirmed labels. A new run that selects different crops needs a new visual review.

**Genesis Outlier types from the competition rules:**
- **Type A — Semantic terrain splice:** two terrain patches from different Mars regions stitched together. Expected heatmap: sharp boundary or oblique edge error.
- **Type B — Domain-shifted terrestrial content:** non-Martian image inserted. Expected heatmap: diffuse error across the whole frame.
- **Type C — Synthetic sensor artifact:** hardware-injection simulation. Expected heatmap: stripe, readout-line, or border-region error.
""")
    code('HYPOTHESES = '+pprint.pformat(hypotheses,sort_dicts=False,width=90)+"""
for filename in selected.filename:
    item = HYPOTHESES.get(filename)
    if item is None:
        display(Markdown('### '+filename+chr(10)+'New selection: visual review needed.'))
        continue
    parts = [
        '### '+filename,
        '**Heatmap pattern:** '+item.get('heatmap_pattern', item.get('evidence','')),
        '**Primary hypothesis:** '+item.get('primary_hypothesis', item.get('hypothesis','')),
        '**Martian alternative:** '+item.get('martian_alternative', item.get('alternative','')),
        '**Limitation:** '+item.get('limitation',''),
    ]
    if item.get('reference'):
        parts.append('[Background reference — not a crop identification]('+item['reference']+')')
    display(Markdown((chr(10)*2).join(p for p in parts if p.split(':',1)[-1].strip())))
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
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'accelerator':'GPU','colab':{'name':'Mars_HiRISE_Final.ipynb','provenance':[]}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/Mars_HiRISE_Final.ipynb'
    target.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    print('Built',len(cells),'cells:',target)

if __name__=='__main__':main()
