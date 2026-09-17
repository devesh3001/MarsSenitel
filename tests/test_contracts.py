import numpy as np
import pandas as pd
import pytest
import torch
from src.model import ConvAutoencoder,ssim_per_image,loss_components
from src.threshold import upper_fence,calibrate,select_flagged


def test_model_shape_and_gradients():
    torch.set_num_threads(2)
    model=ConvAutoencoder(128)
    x=torch.rand(2,1,227,227)
    reconstruction,z=model(x)
    assert z.shape==(2,128) and reconstruction.shape==x.shape
    assert reconstruction.min()>=0 and reconstruction.max()<=1
    components=loss_components(x,reconstruction,.5,.1)
    components['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_ssim_identity_and_error():
    x=torch.rand(2,1,32,32)
    assert torch.allclose(ssim_per_image(x,x),torch.ones(2),atol=1e-5)
    assert (ssim_per_image(x,1-x)<.5).all()


def test_threshold_not_fixed_count_and_selection_after_boundary():
    scores=np.linspace(.2,.6,100)
    threshold=upper_fence(scores)['threshold']
    assert threshold>scores.max()
    frame=pd.DataFrame({'filename':[str(i) for i in range(100)],'novelty_score':scores})
    assert len(select_flagged(frame,threshold))==0
    frame.loc[99,'novelty_score']=1.
    assert select_flagged(frame,threshold).filename.tolist()==['99']


def test_degenerate_calibration_is_rejected():
    with pytest.raises(ValueError): upper_fence(np.ones(100))
    with pytest.raises(ValueError): upper_fence([float('nan')]*100)


def test_source_bootstrap_reproducible():
    rng=np.random.default_rng(42)
    values=rng.beta(3,7,240)
    groups=np.repeat(np.arange(12),20)
    first=calibrate(values,groups,20,42)
    second=calibrate(values,groups,20,42)
    assert first==second
    assert first['calibration_sources']==12


def test_score_sign():
    from sklearn.ensemble import IsolationForest
    rng=np.random.default_rng(42)
    x=rng.normal(0,.1,(500,8))
    forest=IsolationForest(random_state=42,contamination='auto').fit(x)
    scores=-forest.score_samples(np.array([[0.]*8,[5.]*8]))
    assert scores[1]>scores[0]


def test_mixture_envelope_covers_all_components():
    from src.threshold import mixture_fence
    rng=np.random.default_rng(42)
    values=np.r_[rng.normal(.4,.01,600),rng.normal(.49,.02,400)]
    result=mixture_fence(values,42)
    assert result['components']>=2
    assert result['threshold']==max(np.array(result['means'])+3*np.array(result['std']))
    assert result['fitted_tail_mass']<=.0013501
    assert .53<result['threshold']<.58
