"""Hold the fitted forest/cutoff fixed while perturbing validation illumination."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch,joblib
from mars_anomaly.model import ConvAutoencoder
from mars_anomaly.data import load_manifest,CropDataset
from mars_anomaly.preprocessing import preprocess_array,border_zero_mask


def measure(root,runs):
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=[]
    for name in runs:
        run=root/'outputs'/name;analysis=run/'mixture_three_sigma_trees2000'
        if not (analysis/'diagnostics.json').exists():continue
        checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        config=checkpoint['config'];mode=config.get('preprocessing','raw')
        frame=load_manifest(root/'data',config.get('split_seed',config['seed']))
        frame=frame[frame.split=='validation'].sample(n=min(512,int((frame.split=='validation').sum())),random_state=42).reset_index(drop=True)
        model=ConvAutoencoder(config['latent_dim']).to(device)
        model.load_state_dict(checkpoint['model']);model.eval()
        original=[];altered=[]
        for x,_ in CropDataset(frame):
            a=x[0].numpy();mask=border_zero_mask(a)
            b=.8*a+.1;b[mask]=0
            original.append(preprocess_array(a,mode));altered.append(preprocess_array(b,mode))
        def encode(arrays):
            outputs=[]
            with torch.no_grad():
                for i in range(0,len(arrays),16):
                    batch=torch.from_numpy(np.stack(arrays[i:i+16])[:,None]).to(device)
                    outputs.append(model.encode(batch).cpu().numpy())
            return np.concatenate(outputs)
        z0,z1=encode(original),encode(altered)
        forest=joblib.load(analysis/'isolation_forest.joblib')
        s0,s1=-forest.score_samples(z0),-forest.score_samples(z1)
        threshold=json.loads((analysis/'calibration.json').read_text())['threshold']
        delta=np.abs(s1-s0)
        result={'run':name,'preprocessing':mode,'validation_crops':len(frame),'gain':.8,'offset':.1,
                'median_absolute_score_change':float(np.median(delta)),'p95_absolute_score_change':float(np.quantile(delta,.95)),
                'mean_input_absolute_change':float(np.abs(np.array(original)-np.array(altered)).mean()),
                'median_relative_latent_change':float(np.median(np.linalg.norm(z1-z0,axis=1)/np.maximum(np.linalg.norm(z0,axis=1),1e-8))),
                'flag_flips':int(np.count_nonzero((s0>threshold)!=(s1>threshold))),
                'original_flags':int((s0>threshold).sum()),'perturbed_flags':int((s1>threshold).sum())}
        rows.append(result)
        pd.DataFrame({'filename':frame.filename,'original_score':s0,'perturbed_score':s1,'absolute_change':delta}).to_csv(analysis/'photometric_sensitivity.csv',index=False)
    table=pd.DataFrame(rows);out=root/'outputs/comparison';out.mkdir(exist_ok=True,parents=True)
    table.to_csv(out/'photometric_sensitivity.csv',index=False)
    print(table.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--runs',nargs='+',default=['v3','v4','v5'])
    args=p.parse_args();measure(args.root,args.runs)
