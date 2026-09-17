import argparse
from dataclasses import asdict,dataclass
from pathlib import Path
import json
import random
import time
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import CropDataset,load_manifest
from .model import ConvAutoencoder,loss_components


@dataclass
class TrainConfig:
    version: str = 'v1'
    latent_dim: int = 128
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = .0003
    structural_weight: float = 0.
    gradient_weight: float = 0.
    seed: int = 2026
    split_seed: int = 2026
    preprocessing: str = 'raw'
    workers: int = 0
    patience: int = 5
    max_batches: int = 0
    augment: bool = False


def normalized_config(saved):
    result = asdict(TrainConfig())
    result.update(saved)
    result['split_seed'] = saved.get('split_seed', saved.get('seed', 2026))
    # Backward compatibility: older checkpoints pre-date the augment and
    # gradient_weight fields; fill defaults so resume/validation never raises KeyError.
    result.setdefault('augment', False)
    result.setdefault('gradient_weight', 0.0)
    return result


def validate_config(saved, requested, resume=False):
    prior = normalized_config(saved)
    current = asdict(requested)
    keys = set(current) - {'version', 'workers'}
    if resume:
        keys.remove('epochs')  # Extending the training budget is allowed.
    for key in sorted(keys):
        if prior[key] != current[key]:
            raise ValueError(f'Configuration mismatch: {key}: saved={prior[key]!r}, requested={current[key]!r}')


def ensure_training(data_dir, output_dir, config):
    """Validate cached runs before reuse, or train/resume the requested configuration."""
    output_dir = Path(output_dir)
    if (output_dir/'config.json').exists():
        validate_config(json.loads((output_dir/'config.json').read_text()), config)
    status = output_dir/'training_status.json'
    if status.exists():
        expected = 'smoke_complete' if config.max_batches else 'complete'
        if json.loads(status.read_text())['status'] != expected:
            raise ValueError('Unexpected cached training status')
        checkpoint = torch.load(output_dir/'best.pt', map_location='cpu', weights_only=False)
        validate_config(checkpoint['config'], config)
        return
    run_training(data_dir, output_dir, config, resume=(output_dir/'last.pt').exists())


def atomic_checkpoint(state, destination):
    """Keep the previous checkpoint intact if serialization fails partway."""
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(destination)


def run_training(data_dir,output_dir,config, resume=False):
    if config.epochs<1 or not 0<=config.structural_weight<=1 or config.gradient_weight<0:
        raise ValueError('Invalid training configuration')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    last = output_dir/'last.pt'
    if last.exists() and not resume:
        raise FileExistsError(f'{last} already exists; choose a new run directory or resume')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = load_manifest(data_dir,config.split_seed)
    loaders = {split:DataLoader(CropDataset(manifest[manifest.split==split],config.preprocessing),batch_size=config.batch_size,shuffle=split=='train',num_workers=config.workers,pin_memory=device.type=='cuda') for split in ['train','validation']}
    model = ConvAutoencoder(config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best,stale,history = 0,float('inf'),0,[]
    if resume:
        checkpoint = torch.load(last,map_location=device,weights_only=False)
        validate_config(checkpoint['config'], config, resume=True)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start,best,stale = checkpoint['epoch']+1,checkpoint['best'],checkpoint['stale']
        history = checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
    manifest.drop(columns='path').to_csv(output_dir/'split_manifest.csv',index=False)
    config_path = output_dir/'config.json'
    config_path.write_text(json.dumps(asdict(config),indent=2))
    environment = {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'parameters':sum(p.numel() for p in model.parameters()),'smoke_run':config.max_batches>0}
    (output_dir/'environment.json').write_text(json.dumps(environment,indent=2))
    print(json.dumps(environment),flush=True)
    def record_event(event, **details):
        with (output_dir/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':datetime.now().astimezone().isoformat(),'event':event,**details})+'\n')
    record_event('resumed' if resume else 'started',completed_epochs=start,config=asdict(config),device=str(device))
    saved_rng = torch.get_rng_state() if resume else None
    fixed = next(iter(loaders['validation']))[0][:8].to(device)
    if saved_rng is not None: torch.set_rng_state(saved_rng)
    # A fixed validation panel supports real symptom/diagnosis comparisons.
    from .visualize import reconstruction_panel
    for epoch in range(start,config.epochs):
        then = time.perf_counter()
        row = {'epoch':epoch+1}
        for split,loader in loaders.items():
            training = split=='train'
            model.train(training)
            sums = dict.fromkeys(['loss','mse','ssim','gradient'],0.)
            count = 0
            for batch_index,(x,_) in enumerate(loader):
                if config.max_batches and batch_index>=config.max_batches: break
                x = x.to(device,non_blocking=True)
                if training and config.augment:
                    if torch.rand(1) > 0.5: x = torch.flip(x, [2])
                    if torch.rand(1) > 0.5: x = torch.flip(x, [3])
                    rot = torch.randint(0, 4, (1,)).item()
                    if rot > 0: x = torch.rot90(x, k=rot, dims=[2, 3])
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                        reconstruction,_ = model(x)
                    # Float32 image statistics avoid SSIM cancellation in half precision.
                    with torch.autocast(device_type=device.type,enabled=False):
                        metrics = loss_components(x,reconstruction,config.structural_weight,config.gradient_weight)
                    if not torch.isfinite(metrics['loss']): raise RuntimeError('Nonfinite training loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(metrics['loss']).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update()
                for key,value in metrics.items(): sums[key] += float(value.detach())*len(x)
                count += len(x)
            row.update({split+'_'+key:value/count for key,value in sums.items()})
            row[split+'_images'] = count
        row['seconds'] = time.perf_counter()-then
        history.append(row)
        improved = row['validation_loss'] < best
        best = min(best,row['validation_loss'])
        stale = 0 if improved else stale+1
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'config':asdict(config),'epoch':epoch,'best':best,'stale':stale,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        if improved:
            atomic_checkpoint({'model':model.state_dict(),'config':asdict(config),'epoch':epoch},output_dir/'best.pt')
        # Save the best weights before advancing the resumable best-loss metadata.
        atomic_checkpoint(state,last)
        pd.DataFrame(history).to_csv(output_dir/'history.csv',index=False)
        model.eval()
        with torch.no_grad(): reconstruction_panel(fixed,model(fixed)[0],output_dir/'reconstructions_latest.png',input_label='Original' if config.preprocessing=='raw' else 'Model input: '+config.preprocessing)
        print(json.dumps(row),flush=True)
        record_event('epoch_completed',**row)
        if stale>=config.patience: break
    (output_dir/'training_status.json').write_text(json.dumps({'status':'smoke_complete' if config.max_batches else 'complete','epochs_completed':len(history),'best_validation_loss':best},indent=2))
    record_event('completed',epochs_completed=len(history),best_validation_loss=best)
    return model,pd.DataFrame(history)


def parse_bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('true', '1', 'yes'): return True
    if value.lower() in ('false', '0', 'no'): return False
    raise argparse.ArgumentTypeError('Expected true or false')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    for key, value in asdict(TrainConfig()).items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            # Use store_true/store_false so --augment / --no-augment work correctly.
            # parse_bool nargs='?' also accepted for --augment true/false strings.
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=key, type=parse_bool, nargs='?',
                               const=True, default=value,
                               metavar='BOOL')
            group.add_argument('--no-' + key.replace('_', '-'), dest=key,
                               action='store_false')
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return parser


def main():
    args = vars(build_parser().parse_args())
    data,out,resume = args.pop('data'),args.pop('out'),args.pop('resume')
    run_training(data,out,TrainConfig(**args),resume)

if __name__=='__main__': main()
