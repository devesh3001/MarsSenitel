"""Matched image-only/fused ensembles; optional diagnostic, not the primary detector."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from scipy.stats import spearmanr
from mars_anomaly.data import load_manifest
from mars_anomaly.threshold import mixture_fence


def fusion(root=Path('.')):
    root=Path(root)
    analysis=root/'outputs/v8/mixture_three_sigma_trees2000'
    out=root/'outputs/v8/fusion_controlled';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((root/'outputs/v8/config.json').read_text())
    manifest=load_manifest(root/'data',config.get('split_seed',config['seed']))
    names=np.load(analysis/'latent_filenames.npy',allow_pickle=False)
    latents=np.load(analysis/'latents.npy')
    frame=pd.DataFrame({'filename':names}).merge(manifest,on='filename',validate='one_to_one')
    assert len(frame)==len(latents) and not frame.isna().any().any()
    train=frame.split.eq('train').to_numpy();cal=frame.split.eq('calibration').to_numpy()
    inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [analysis/'latents.npy',analysis/'latent_filenames.npy',root/'data/source_image_metadata.csv',root/'data/crop_metadata_index.csv',root/'outputs/v8/config.json']}
    if (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text())
        portable=lambda d:{k.replace('\\','/'):v for k,v in d.items()}
        if portable(saved['input_sha256'])!=portable(inputs):raise ValueError('Fusion cache input mismatch; use a new output folder')
        for name in ['image_only','fused']:
            for suffix in ['scores.csv','calibration.json','detector.joblib']:
                if not (out/f'{name}_{suffix}').exists():raise FileNotFoundError('Incomplete fusion cache')
        return saved
    scaler=StandardScaler().fit(frame.loc[train,['sun_angle','resolution']])
    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(frame.loc[train,['season']])
    metadata=np.concatenate([scaler.transform(frame[['sun_angle','resolution']]),encoder.transform(frame[['season']])],axis=1)
    vectors={'image_only':latents,'fused':np.concatenate([latents,metadata],axis=1)}
    scores={};sets={};thresholds={};seeds=[2026,2027,2028,2029,2030]
    for name,z in vectors.items():
        forests=[];values=[]
        for seed in seeds:
            forest=IsolationForest(n_estimators=2000,max_samples=256,contamination='auto',random_state=seed,n_jobs=2).fit(z[train])
            values.append(-forest.score_samples(z));forests.append(forest)
        score=np.mean(values,axis=0);boundary=mixture_fence(score[cal],seed=2026)
        flags=score>boundary['threshold'];scores[name]=score;sets[name]=set(frame.loc[flags,'filename']);thresholds[name]=boundary['threshold']
        result=frame.drop(columns='path').copy();result['novelty_score']=score;result['flagged']=flags
        result.to_csv(out/f'{name}_scores.csv',index=False)
        (out/f'{name}_calibration.json').write_text(json.dumps(boundary,indent=2))
        joblib.dump({'forests':forests,'scaler':scaler if name=='fused' else None,'encoder':encoder if name=='fused' else None,'numerical_columns':['sun_angle','resolution'],'categorical_columns':['season'],'threshold':boundary['threshold'],'seeds':seeds,'input_sha256':inputs},out/f'{name}_detector.joblib',compress=3)
        print(name,'matched ensemble flags:',int(flags.sum()),flush=True)
        del forests,forest
    union=sets['image_only']|sets['fused']
    summary={'run':'v8','seeds':seeds,'n_estimators_per_forest':2000,'dimensions':{n:z.shape[1] for n,z in vectors.items()},'thresholds':thresholds,'flags':{n:len(s) for n,s in sets.items()},'flag_jaccard':len(sets['image_only']&sets['fused'])/len(union) if union else None,'added_by_fusion':sorted(sets['fused']-sets['image_only']),'removed_by_fusion':sorted(sets['image_only']-sets['fused']),'spearman_all':float(spearmanr(scores['image_only'],scores['fused']).statistic),'input_sha256':inputs,'interpretation':'Matched ensemble diagnostic; each boundary is fitted on its own calibration scores. Empty sets do not establish accuracy. Metadata enters only before the forest, not the decoder. No bonus score is claimed.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    print(json.dumps(fusion(Path(__file__).resolve().parents[1]),indent=2))
