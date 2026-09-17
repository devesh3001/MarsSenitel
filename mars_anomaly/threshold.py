import numpy as np
from statsmodels.stats.stattools import medcouple


def mixture_fence(scores,seed=2026):
    """Envelope beyond every fitted score population; no contamination fraction."""
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits
    from scipy.stats import norm
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or len(values)<30 or not np.isfinite(values).all() or values.std()<1e-8:
        raise ValueError('Need at least 30 finite, nonconstant scores for a mixture')
    with threadpool_limits(limits=2):
        models=[GaussianMixture(n_components=k,n_init=5,random_state=seed,reg_covar=1e-6).fit(values[:,None]) for k in range(1,5)]
    valid=[m for m in models if m.converged_]
    if not valid: raise ValueError('No mixture fit converged')
    best=min(valid,key=lambda m:m.bic(values[:,None]))
    means=best.means_.ravel();sigmas=np.sqrt(best.covariances_.ravel())
    threshold=float(np.max(means+3*sigmas))
    ordered=np.sort(values)
    cdf=np.sum(best.weights_[None,:]*norm.cdf((ordered[:,None]-means)/sigmas),axis=1)
    n=len(values)
    ks=float(max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n)))
    return {'threshold':threshold,'method':'mixture_three_sigma','components':best.n_components,'bic':{str(m.n_components):float(m.bic(values[:,None])) for m in valid},'weights':best.weights_.tolist(),'means':means.tolist(),'std':sigmas.tolist(),'calibration_cdf_max_distance':ks,'fitted_tail_mass':float(np.sum(best.weights_*norm.sf((threshold-means)/sigmas))),'component_count_at_search_limit':best.n_components==4}


def upper_fence(scores):
    values = np.asarray(scores,dtype=float)
    if values.ndim != 1 or len(values)<8 or not np.isfinite(values).all():
        raise ValueError('Need at least 8 finite calibration scores')
    q1,q3 = np.quantile(values,[.25,.75])
    iqr = q3-q1
    if iqr <= np.finfo(float).eps:
        raise ValueError('Degenerate calibration IQR; investigate representation and scores')
    mc = float(medcouple(values))
    multiplier = 3 if mc>=0 else 4
    fence = q3 + 1.5*np.exp(multiplier*mc)*iqr
    return {'threshold':float(fence),'q1':float(q1),'q3':float(q3),'iqr':float(iqr),'medcouple':mc}


def calibrate(scores,groups,bootstrap=100,seed=2026,method='adjusted_boxplot'):
    values,groups = np.asarray(scores),np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError('Scores and groups must have equal shape')
    unique = np.unique(groups)
    if len(unique)<8:
        raise ValueError('Too few calibration source groups for bootstrap')
    if method not in ('adjusted_boxplot','mixture_three_sigma'): raise ValueError('Unknown threshold method')
    boundary = upper_fence if method=='adjusted_boxplot' else lambda x:mixture_fence(x,seed)
    result = boundary(values)
    result['method']=method
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(groups==g) for g in unique]
    thresholds = []
    for _ in range(bootstrap):
        sample = np.concatenate([indices[i] for i in rng.integers(0,len(unique),len(unique))])
        try:
            thresholds.append(boundary(values[sample])['threshold'])
        except ValueError:
            continue
    if len(thresholds)<max(10,bootstrap*.8):
        raise ValueError('Too many degenerate bootstrap replicates')
    median = np.median(values)
    mad = np.median(np.abs(values-median))
    q1,q3=np.quantile(values,[.25,.75])
    result.update({'bootstrap_interval_95':np.quantile(thresholds,[.025,.975]).tolist(), 'bootstrap_thresholds':thresholds,'bootstrap_valid':len(thresholds),'calibration_images':len(values),'calibration_sources':len(unique),'tukey_comparison':float(q3+1.5*(q3-q1)),'mad_comparison':float(median+3*1.4826*mad),'interpretation':'Descriptive distribution-based screening boundary; no nominal false-positive or false-discovery guarantee.'})
    return result


def select_flagged(frame,threshold,limit=5):
    """The limit is only for explanations, never for threshold determination."""
    return frame.loc[frame.novelty_score>threshold].sort_values(['novelty_score','filename'],ascending=[False,True]).head(limit)
