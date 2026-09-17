"""Distribution diagnostics: no labels, candidate inspection, or target count."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(run):
    frame=pd.read_csv(run/'novelty_scores.csv')
    values=frame.loc[frame.split=='calibration','novelty_score'].to_numpy()
    models=[]
    with threadpool_limits(limits=2):
        for k in range(1,5):
            model=GaussianMixture(n_components=k,n_init=10,random_state=2026,reg_covar=1e-6).fit(values[:,None])
            models.append(model)
    best=min(models,key=lambda m:m.bic(values[:,None]))
    means=best.means_.ravel();sigmas=np.sqrt(best.covariances_.ravel())
    threshold=float(np.max(means+3*sigmas))
    summary={'bic':{m.n_components:float(m.bic(values[:,None])) for m in models},'selected_components':best.n_components,'weights':best.weights_.tolist(),'means':means.tolist(),'std':sigmas.tolist(),'three_sigma_envelope':threshold,'flag_count':int((frame.novelty_score>threshold).sum()),'fitted_survival_at_boundary':float(np.sum(best.weights_*norm.sf((threshold-means)/sigmas))),'note':'Exploratory distribution model; not selected as the final threshold by this script.'}
    (run/'mixture_exploration.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    grid=np.linspace(values.min()-.01,values.max()+.04,700)
    axes[0].hist(values,bins=50,density=True,alpha=.35,label='Calibration observations')
    for w,mu,sigma in zip(best.weights_,means,sigmas): axes[0].plot(grid,w*norm.pdf(grid,mu,sigma),alpha=.6)
    axes[0].plot(grid,np.exp(best.score_samples(grid[:,None])),color='black',label='BIC-selected mixture')
    axes[0].axvline(threshold,color='red',label='Maximum component mean + 3 SD')
    axes[0].legend(fontsize=8);axes[0].set(xlabel='Novelty score',ylabel='Density')
    ordered=np.sort(values)
    cdf=np.sum(best.weights_[None,:]*norm.cdf((ordered[:,None]-means)/sigmas),axis=1)
    axes[1].plot(cdf,np.arange(1,len(values)+1)/len(values));axes[1].plot([0,1],[0,1],'k--')
    axes[1].set(xlabel='Fitted mixture CDF',ylabel='Empirical CDF',title='Calibration fit diagnostic (not a held-out test)')
    fig.tight_layout();fig.savefig(run/'mixture_exploration.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
