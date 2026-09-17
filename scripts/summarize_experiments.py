"""Compare saved experiments without inventing hidden-label accuracy."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(root):
    output=root/'outputs'/'comparison';output.mkdir(exist_ok=True)
    rows=[];frames={}
    for name in ['v1','v2','v3','robust_seed','robust_split','v4','v5','improved_seed','improved_split','v6','v7','v7_robust_seed','v7_robust_split','v8']:
        run=root/'outputs'/name
        if not (run/'training_status.json').exists(): continue
        config=json.loads((run/'config.json').read_text())
        history=pd.read_csv(run/'history.csv')
        best=history.loc[history.validation_loss.idxmin()]
        row={'run':name,'latent_dim':config['latent_dim'],'ssim_weight':config['structural_weight'],'seed':config['seed'],'split_seed':config.get('split_seed',config['seed']),'best_epoch':int(best.epoch),'validation_mse':best.validation_mse,'validation_ssim':best.validation_ssim,'validation_gradient':best.validation_gradient,'training_seconds':history.seconds.sum()}
        row['input_domain']=config.get('preprocessing','raw')
        row['gradient_weight']=config.get('gradient_weight',0.)
        row['augment']=config.get('augment',False)
        analysis=run/'mixture_three_sigma_trees2000'
        if not (analysis/'diagnostics.json').exists():analysis=run/'mixture_three_sigma'
        if (analysis/'diagnostics.json').exists():
            diagnostics=json.loads((analysis/'diagnostics.json').read_text())
            calibration=json.loads((analysis/'calibration.json').read_text())
            row['trees']=diagnostics.get('n_estimators',400)
            row.update({'effective_rank':diagnostics['effective_rank'],'threshold':calibration['threshold'],'threshold_low':calibration['bootstrap_interval_95'][0],'threshold_high':calibration['bootstrap_interval_95'][1],'flagged':diagnostics['flagged_total'],'validation_cdf_distance':diagnostics['validation_cdf_max_distance'],'components':calibration['components'],'forest_min_spearman':min(x['spearman'] for x in diagnostics['forest_seed_stability'])})
            frames[name]=pd.read_csv(analysis/'novelty_scores.csv').set_index('filename')
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(output/'experiments.csv',index=False)
    pairs=[]
    for i,left in enumerate(frames):
        for right in list(frames)[i+1:]:
            a,b=frames[left].align(frames[right],join='inner',axis=0)
            union=np.count_nonzero(a.flagged|b.flagged)
            heldout=(a.split!='train')&(b.split!='train')
            pairs.append({'left':left,'right':right,'spearman_all':float(spearmanr(a.novelty_score,b.novelty_score).statistic),'flag_jaccard':float(np.count_nonzero(a.flagged&b.flagged)/union) if union else None,'common_heldout_images':int(heldout.sum()),'spearman_common_heldout':float(spearmanr(a.loc[heldout,'novelty_score'],b.loc[heldout,'novelty_score']).statistic)})
            heldout_union=np.count_nonzero((a.flagged|b.flagged)&heldout)
            pairs[-1].update(all_images=len(a),flag_union_all=int(union),flag_jaccard_common_heldout=float(np.count_nonzero((a.flagged&b.flagged)&heldout)/heldout_union) if heldout_union else None,flag_union_common_heldout=int(heldout_union))
    pd.DataFrame(pairs).to_csv(output/'run_stability.csv',index=False)
    statistics_path=root/'outputs/audit/image_statistics.csv'
    stats=pd.read_csv(statistics_path).set_index('filename') if statistics_path.exists() else None
    confounds=[]
    for name,frame in frames.items():
        if stats is None: continue
        joined=frame.join(stats)
        total=((joined.novelty_score-joined.novelty_score.mean())**2).sum()
        source_means=joined.groupby('source_image_id').novelty_score.transform('mean')
        explained=float(((source_means-joined.novelty_score.mean())**2).sum()/total)
        row={'run':name,'source_between_group_variance_fraction':explained}
        for c in ['mean','std','zero_fraction','sun_angle','latitude']:
            row['spearman_'+c]=float(spearmanr(joined[c],joined.novelty_score).statistic)
        confounds.append(row)
    if confounds:pd.DataFrame(confounds).to_csv(output/'confounds.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for name in ['v1','v2','v3']:
        run=root/'outputs'/name
        if not (run/'history.csv').exists(): continue
        history=pd.read_csv(run/'history.csv')
        for ax,metric in zip(axes,['mse','ssim','gradient']): ax.plot(history.epoch,history['validation_'+metric],label=name)
    for ax,metric in zip(axes,['MSE (lower better)','SSIM (higher better)','Gradient error (lower better)']):
        ax.set(xlabel='Epoch',ylabel=metric);ax.legend()
    fig.tight_layout();fig.savefig(output/'validation_curves.png',dpi=170);plt.close(fig)
    print(table.to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False));print(pd.DataFrame(confounds).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.'))
    summarize(parser.parse_args().root)
