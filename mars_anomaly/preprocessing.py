"""Explicit, per-image photometric ablations; no fitted dataset statistics."""
import numpy as np
from scipy.ndimage import label, binary_dilation, distance_transform_edt

MODES=('raw','contrast','footprint','border_fill')


def border_zero_mask(array):
    regions,_=label(array==0)
    border=np.unique(np.concatenate([regions[0],regions[-1],regions[:,0],regions[:,-1]]))
    return np.isin(regions,border[border!=0])


def border_near_black_mask(array):
    """Fixed JPEG-tolerant border heuristic; not a ground-truth validity mask."""
    regions, _ = label(array <= 4/255)
    edge = np.unique(np.concatenate([regions[0],regions[-1],regions[:,0],regions[:,-1]]))
    mask = np.isin(regions, edge[edge != 0])
    return binary_dilation(mask, iterations=2) if mask.any() else mask


def preprocess_array(array,mode='raw'):
    if mode not in MODES:raise ValueError(f'Unknown preprocessing: {mode}')
    x=np.asarray(array,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite 2D image')
    if mode=='raw':return x
    excluded=(border_near_black_mask(x) if mode=='border_fill' else
              border_zero_mask(x) if mode=='footprint' else np.zeros(x.shape,dtype=bool))
    values=x[~excluded]
    if not len(values):return np.full_like(x,.5)
    mean=float(values.mean());std=max(float(values.std()),1/255)
    result=np.clip(.5+.15*(x-mean)/std,0,1).astype(np.float32)
    if mode=='border_fill' and excluded.any():
        nearest=distance_transform_edt(excluded,return_distances=False,return_indices=True)
        result[excluded]=result[tuple(nearest[:,excluded])]
    else:
        result[excluded]=.5
    return result
