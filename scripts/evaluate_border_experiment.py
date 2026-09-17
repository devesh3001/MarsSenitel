"""Fixed, held-out response diagnostics; synthetic defects are never training labels."""
from pathlib import Path
from io import BytesIO
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import torch,joblib
from mars_anomaly.data import load_manifest
from mars_anomaly.model import ConvAutoencoder
from mars_anomaly.preprocessing import preprocess_array,border_near_black_mask

ROOT=Path(__file__).resolve().parents[1]
ANALYSIS='mixture_three_sigma_trees2000'


def jpeg(array):
    stream=BytesIO()
    Image.fromarray(np.rint(array*255).astype(np.uint8)).save(stream,format='JPEG',quality=90)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return np.array(decoded,dtype=np.float32)/255


def variants(array):
    strip=array.copy();strip[:,:8]=2/255
    bright=array.copy();bright[101:125,101:125]=np.clip(bright[101:125,101:125]+.35,0,1)
    dark=array.copy();dark[101:125,101:125]=np.clip(dark[101:125,101:125]-.35,0,1)
    checker=array.copy()
    yy,xx=np.indices((24,24));checker[101:125,101:125]=.1+.8*((yy//4+xx//4)%2)
    return {'original':array,'jpeg_control':jpeg(array),'border_strip':jpeg(strip),
            'bright_square':bright,'dark_square':dark,'checker':checker}


def main():
    torch.set_num_threads(2)
    out=ROOT/'outputs/border_experiment';out.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(ROOT/'data',2026)
    sample=manifest[manifest.split=='validation'].sample(256,random_state=73).reset_index(drop=True)
    sample.drop(columns='path').to_csv(out/'stress_sample.csv',index=False)
    arrays=[]
    for path in sample.path:
        with Image.open(path) as im:arrays.append(np.array(im,dtype=np.float32)/255)
    transformed=[variants(a) for a in arrays]
    rows=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for name in ['v3','v5','v6']:
        run=ROOT/'outputs'/name;analysis=run/ANALYSIS
        checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        config=checkpoint['config'];mode=config.get('preprocessing','raw')
        model=ConvAutoencoder(config['latent_dim']).to(device)
        model.load_state_dict(checkpoint['model']);model.eval()
        forest=joblib.load(analysis/'isolation_forest.joblib')
        threshold=json.loads((analysis/'calibration.json').read_text())['threshold']
        scores={}
        for variant in transformed[0]:
            latent=[]
            with torch.no_grad():
                for offset in range(0,len(sample),16):
                    x=np.stack([preprocess_array(v[variant],mode) for v in transformed[offset:offset+16]])[:,None]
                    latent.append(model.encode(torch.from_numpy(x).to(device)).cpu().numpy())
            scores[variant]=-forest.score_samples(np.concatenate(latent))
        pd.DataFrame({'filename':sample.filename,**scores}).to_csv(out/f'{name}_paired_scores.csv',index=False)
        for variant in ['border_strip','bright_square','dark_square','checker']:
            base=scores['jpeg_control' if variant=='border_strip' else 'original']
            changed=scores[variant];delta=changed-base
            rows.append(dict(run=name,preprocessing=mode,test=variant,n=len(sample),
                             median_signed_change=float(np.median(delta)),
                             median_absolute_change=float(np.median(np.abs(delta))),
                             p95_absolute_change=float(np.quantile(np.abs(delta),.95)),
                             fraction_score_increased=float((delta>0).mean()),
                             threshold_flips=int(((base>threshold)!=(changed>threshold)).sum()),
                             newly_above_threshold=int(((base<=threshold)&(changed>threshold)).sum()),
                             original_above_threshold=int((base>threshold).sum()),
                             changed_above_threshold=int((changed>threshold).sum())))
        del model,forest,checkpoint
        if device=='cuda':torch.cuda.empty_cache()
        print(name,'response diagnostics complete',flush=True)
    response=pd.DataFrame(rows);response.to_csv(out/'response_summary.csv',index=False)
    audit=[]
    for row in manifest.itertuples():
        with Image.open(row.path) as im:a=np.array(im,dtype=np.float32)/255
        mask=border_near_black_mask(a)
        audit.append({'filename':row.filename,'source_image_id':row.source_image_id,
                      'masked_fraction':float(mask.mean()),'central_defect_mask_fraction':float(mask[101:125,101:125].mean())})
    audit=pd.DataFrame(audit);audit.to_csv(out/'mask_audit.csv',index=False)
    summary={'crops':len(audit),'any_mask':int((audit.masked_fraction>0).sum()),
             'all_masked':int((audit.masked_fraction==1).sum()),
             'over_half_masked':int((audit.masked_fraction>.5).sum()),
             'median_masked_fraction':float(audit.masked_fraction.median()),
             'p95_masked_fraction':float(audit.masked_fraction.quantile(.95)),
             'max_masked_fraction':float(audit.masked_fraction.max())}
    (out/'mask_summary.json').write_text(json.dumps(summary,indent=2))
    d={n:json.loads((ROOT/'outputs'/n/ANALYSIS/'diagnostics.json').read_text()) for n in ['v3','v5','v6']}
    jaccard={n:min(r['flag_jaccard'] or 0 for r in diag['forest_seed_stability']) for n,diag in d.items()}
    lookup=response.set_index(['run','test'])
    border6=lookup.loc[('v6','border_strip'),'p95_absolute_change']
    criteria={'border_response_reduced_25_percent_vs_both':bool(all(border6<=.75*lookup.loc[(n,'border_strip'),'p95_absolute_change'] for n in ['v3','v5'])),
              'all_defect_responses_at_least_70_percent_and_noninferior':bool(all(lookup.loc[('v6',t),'fraction_score_increased']>=max(.70,lookup.loc[('v3',t),'fraction_score_increased']-.05) for t in ['bright_square','dark_square','checker'])),
              'forest_jaccard_at_least_v3':bool(jaccard['v6']>=jaccard['v3'])}
    decision={'criteria':criteria,'forest_min_jaccard':jaccard,'numerical_criteria_pass':all(criteria.values()),
              'visual_review':'pending','promoted':False,
              'note':'The visual/mask review is still required. Proxy response diagnostics do not establish hidden-anomaly accuracy.'}
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    print(response.to_string(index=False));print(json.dumps(summary));print(json.dumps(decision))


if __name__=='__main__':main()
