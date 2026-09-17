import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from mars_anomaly.train import TrainConfig, build_parser, validate_config, run_training, ensure_training


def test_boolean_cli_values():
    for args, expected in [([],False),(['--augment'],True),(['--augment','False'],False),(['--augment','true'],True),(['--no-augment'],False)]:
        assert build_parser().parse_args(['--out','unused',*args]).augment is expected


def test_legacy_defaults_and_cached_mismatch():
    legacy=asdict(TrainConfig(seed=2027,split_seed=2027))
    for key in ['augment','preprocessing','split_seed']:legacy.pop(key)
    validate_config(legacy,TrainConfig(seed=2027,split_seed=2027))
    for change in [{'gradient_weight':.1},{'augment':True},{'preprocessing':'contrast'}]:
        with pytest.raises(ValueError,match='Configuration mismatch'):
            validate_config(legacy,TrainConfig(seed=2027,split_seed=2027,**change))


def test_fresh_training_legacy_resume_and_cache_rejection(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    data=tmp_path/'data';(data/'images').mkdir(parents=True)
    sources=[];crops=[]
    for i in range(10):
        name=f'{i}.jpg';source=f'SRC_{i:03d}'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(227,227),dtype=np.uint8)).save(data/'images'/name)
        crops.append({'filename':name,'source_image_id':source})
        sources.append({'source_image_id':source,'latitude':0,'longitude':0,'sun_angle':45,'season':'N-spring','resolution':.25})
    pd.DataFrame(crops).to_csv(data/'crop_metadata_index.csv',index=False)
    pd.DataFrame(sources).to_csv(data/'source_image_metadata.csv',index=False)
    out=tmp_path/'run';config=TrainConfig(epochs=1,batch_size=1,max_batches=1,latent_dim=8,gradient_weight=.1)
    ensure_training(data,out,config)
    ensure_training(data,out,config)
    with pytest.raises(ValueError,match='gradient_weight'):
        ensure_training(data,out,TrainConfig(**{**asdict(config),'gradient_weight':0.}))
    checkpoint=torch.load(out/'last.pt',weights_only=False)
    checkpoint['config'].pop('augment');torch.save(checkpoint,out/'last.pt')
    resumed=TrainConfig(**{**asdict(config),'epochs':2})
    run_training(data,out,resumed,resume=True)
    assert len(pd.read_csv(out/'history.csv'))==2
    assert json.loads((out/'training_status.json').read_text())['status']=='smoke_complete'


def test_canonical_registry_retains_actual_experiment_settings():
    registry=json.loads((Path(__file__).resolve().parents[1]/'outputs/run_registry.json').read_text())
    specs=registry['runs']
    assert specs['v7']['gradient_weight']==.1 and not specs['v7']['augment']
    assert specs['v8']['gradient_weight']==.1 and specs['v8']['augment']
    assert specs['v6']['preprocessing']=='border_fill'
    for name,cfg in specs.items():
        saved=json.loads((Path(__file__).resolve().parents[1]/'outputs'/name/'config.json').read_text())
        validate_config(saved,TrainConfig(**cfg))
