import numpy as np
from src.preprocessing import preprocess_array,border_zero_mask,border_near_black_mask


def test_positive_affine_intensity_invariance_without_floor():
    rng=np.random.default_rng(17)
    image=rng.uniform(.2,.8,(31,31)).astype(np.float32)
    transformed=.7*image+.1
    assert np.allclose(preprocess_array(image,'contrast'),preprocess_array(transformed,'contrast'),atol=1e-6)
    assert np.array_equal(preprocess_array(image,'raw'),image)


def test_footprint_excludes_only_border_connected_zeros():
    image=np.full((9,9),.5,dtype=np.float32)
    image[:,0]=0;image[4,1]=0;image[4,4]=0
    mask=border_zero_mask(image)
    assert mask[:,0].all() and mask[4,1] and not mask[4,4]
    processed=preprocess_array(image,'footprint')
    assert np.all(processed[mask]==.5)
    assert processed[4,4]<.5


def test_constant_and_all_black_inputs_remain_finite():
    for value in [0.,.2,1.]:
        for mode in ['contrast','footprint','border_fill']:
            processed=preprocess_array(np.full((8,8),value,dtype=np.float32),mode)
            assert np.isfinite(processed).all()
            assert np.allclose(processed,.5,atol=1e-5)


def test_near_black_border_fill_removes_seam_but_preserves_interior_dark_feature():
    x=np.full((25,25),.6,dtype=np.float32)
    x[:,0:3]=2/255
    x[12,12]=0
    mask=border_near_black_mask(x)
    assert mask[:,:5].all() and not mask[:,5:].any()
    result=preprocess_array(x,'border_fill')
    assert np.allclose(result[:,0],result[:,5])
    assert result[12,12] < result[12,11]
    clear=np.random.default_rng(4).uniform(.2,.8,(25,25)).astype(np.float32)
    assert np.array_equal(preprocess_array(clear,'contrast'),preprocess_array(clear,'border_fill'))
