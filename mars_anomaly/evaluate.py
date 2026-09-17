import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from .data import load_manifest,CropDataset
from .model import ConvAutoencoder
from .threshold import calibrate,select_flagged
from .visualize import score_plots,heatmap_panel


def evaluate(data_dir,run_dir,bootstrap=100,projection=True,allow_smoke=False,method='adjusted_boxplot',trees=400):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False)
    config = checkpoint['config']
    if method!='adjusted_boxplot' or trees!=400:
        run_dir=run_dir/(method+(f'_trees{trees}' if trees!=400 else ''))
        run_dir.mkdir(parents=True,exist_ok=True)
    if config['max_batches'] and not allow_smoke:
        raise ValueError('Smoke checkpoint cannot produce competition results')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder(config['latent_dim']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    frame = load_manifest(data_dir,config.get('split_seed',config['seed']))
    preprocessing=config.get('preprocessing','raw')
    loader = DataLoader(CropDataset(frame,preprocessing),batch_size=config['batch_size'],num_workers=0)
    latents,errors = [],[]
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            reconstruction,z = model(x)
            latents.append(z.cpu().numpy())
            errors.extend((x-reconstruction).square().mean((1,2,3)).cpu().tolist())
    z = np.concatenate(latents)
    np.save(run_dir/'latents.npy',z)
    np.save(run_dir/'latent_filenames.npy',frame.filename.to_numpy(dtype=str))
    train = frame.split.eq('train').to_numpy()
    calibration_mask = frame.split.eq('calibration').to_numpy()
    forest = IsolationForest(n_estimators=trees,max_samples=256,contamination='auto',random_state=config['seed'],n_jobs=2)
    forest.fit(z[train])
    frame['novelty_score'] = -forest.score_samples(z)
    frame['reconstruction_mse'] = errors
    calibration = calibrate(frame.loc[calibration_mask,'novelty_score'],frame.loc[calibration_mask,'source_image_id'],bootstrap,config['seed'],method)
    frame['flagged'] = frame.novelty_score>calibration['threshold']
    boot = np.asarray(calibration['bootstrap_thresholds'])
    frame['threshold_bootstrap_flag_frequency'] = (frame.novelty_score.to_numpy()[:,None]>boot[None,:]).mean(1)
    # Independent forests assess Monte Carlo rank variability, not hidden-label accuracy.
    stability = []
    for seed in (config['seed']+1,config['seed']+2):
        other = IsolationForest(n_estimators=trees,max_samples=256,contamination='auto',random_state=seed,n_jobs=2).fit(z[train])
        other_scores = -other.score_samples(z)
        from .threshold import upper_fence,mixture_fence
        boundary = (upper_fence(other_scores[calibration_mask]) if method=='adjusted_boxplot' else mixture_fence(other_scores[calibration_mask],config['seed']))['threshold']
        other_flags = other_scores>boundary
        primary_flags = frame.flagged.to_numpy()
        union = np.count_nonzero(primary_flags|other_flags)
        stability.append({'seed':seed,'spearman':float(spearmanr(frame.novelty_score,other_scores).statistic),'flag_count':int(other_flags.sum()),'threshold':boundary,'flag_jaccard':float(np.count_nonzero(primary_flags&other_flags)/union) if union else None})
    covariance = np.cov(z[train],rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)
    proportions = eigenvalues/max(eigenvalues.sum(),1e-12)
    effective_rank = float(np.exp(-np.sum(proportions*np.log(proportions+1e-12))))
    diagnostics = {'effective_rank':effective_rank,'latent_dimensions':z.shape[1],'latent_variance':z[train].var(0).tolist(),'forest_seed_stability':stability,'flagged_total':int(frame.flagged.sum()),'flagged_by_split':frame.groupby('split').flagged.agg(['sum','count','mean']).to_dict(),'smoke_checkpoint':bool(config['max_batches']),'threshold_exceeds_maximum':bool(calibration['threshold']>frame.novelty_score.max())}
    diagnostics['n_estimators']=trees
    diagnostics['preprocessing']=preprocessing
    diagnostics['reconstruction_domain']='raw intensity / 255' if preprocessing=='raw' else preprocessing+' standardized input'
    if method=='mixture_three_sigma':
        from scipy.stats import norm
        means=np.array(calibration['means']);std=np.array(calibration['std']);weights=np.array(calibration['weights'])
        heldout=np.sort(frame.loc[frame.split.eq('validation'),'novelty_score'])
        cdf=np.sum(weights*norm.cdf((heldout[:,None]-means)/std),axis=1)
        n=len(heldout)
        diagnostics['validation_cdf_max_distance']=float(max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n)))
        diagnostics['validation_cdf_note']='Descriptive held-out density-fit diagnostic; correlated sources preclude an ordinary IID KS p-value.'
    (run_dir/'calibration.json').write_text(json.dumps(calibration,indent=2))
    (run_dir/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    frame.drop(columns='path').to_csv(run_dir/'novelty_scores.csv',index=False)
    joblib.dump(forest,run_dir/'isolation_forest.joblib')
    score_plots(frame,calibration,run_dir/'score_distribution.png')
    source_summary = frame.groupby('source_image_id').agg(crops=('filename','size'),flagged=('flagged','sum'),mean_novelty=('novelty_score','mean'),latitude=('latitude','first'),longitude=('longitude','first'),sun_angle=('sun_angle','first'),season=('season','first'),resolution=('resolution','first'))
    source_summary['flag_rate'] = source_summary.flagged/source_summary.crops
    source_summary.to_csv(run_dir/'source_summary.csv')
    for column in ('season','resolution','longitude'):
        crop_summary = frame.groupby(column).flagged.agg(['count','sum','mean'])
        equal_source = source_summary.groupby(column).flag_rate.mean()
        crop_summary['equal_source_mean_flag_rate'] = equal_source
        crop_summary.to_csv(run_dir/f'metadata_{column}.csv')
    selected = select_flagged(frame,calibration['threshold'])
    selected.drop(columns='path').to_csv(run_dir/'selected_for_interpretation.csv',index=False)
    for _,row in selected.iterrows():
        x,_ = CropDataset(pd.DataFrame([row]),preprocessing)[0]
        with torch.no_grad(): rec = model(x[None].to(device))[0][0,0].cpu().numpy()
        if preprocessing=='raw':
            heatmap_panel(x[0].numpy(),rec,row,calibration['threshold'],run_dir/f'heatmap_{Path(row.filename).stem}.png')
        else:
            from .visualize import standardized_heatmap_panel
            raw,_=CropDataset(pd.DataFrame([row]))[0]
            standardized_heatmap_panel(raw[0].numpy(),x[0].numpy(),rec,row,calibration['threshold'],run_dir/f'heatmap_{Path(row.filename).stem}.png',preprocessing)
    if projection:
        import matplotlib.pyplot as plt
        # Standardization affects visualization only; the forest uses the original latent.
        scale = z[train].std(0).clip(1e-6)
        standardized = (z-z[train].mean(0))/scale
        reduced = PCA(n_components=min(50,z.shape[1]),random_state=config['seed']).fit_transform(standardized)
        for perplexity in (30,70):
            xy = TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',random_state=config['seed'],max_iter=1000).fit_transform(reduced)
            pd.DataFrame({'filename':frame.filename,'x':xy[:,0],'y':xy[:,1]}).to_csv(run_dir/f'tsne_{perplexity}.csv',index=False)
            fig,axes = plt.subplots(1,3,figsize=(15,4))
            for ax,values,title in zip(axes,[frame.novelty_score,frame.sun_angle,pd.Categorical(frame.source_image_id).codes],['Novelty score','Sun angle as supplied','Source ID (categorical colors)']):
                points = ax.scatter(xy[:,0],xy[:,1],c=values,s=3,cmap='viridis',rasterized=True)
                fig.colorbar(points,ax=ax);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f't-SNE perplexity {perplexity}; qualitative projection, not geological labels')
            fig.tight_layout();fig.savefig(run_dir/f'tsne_{perplexity}.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in diagnostics.items() if k!='latent_variance'},indent=2),flush=True)
    return frame,calibration,diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data');parser.add_argument('--run',required=True)
    parser.add_argument('--bootstrap',type=int,default=100);parser.add_argument('--no-projection',action='store_true')
    parser.add_argument('--method',choices=['adjusted_boxplot','mixture_three_sigma'],default='adjusted_boxplot')
    parser.add_argument('--trees',type=int,default=400)
    args=parser.parse_args()
    evaluate(args.data,args.run,args.bootstrap,not args.no_projection,method=args.method,trees=args.trees)

if __name__=='__main__': main()
