"""Quantify brightness in selected candidates without changing their selection."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def inspect(directory, data, statistics):
    scores=pd.read_csv(directory/'novelty_scores.csv')
    if statistics.exists():
        context=pd.read_csv(statistics)[['filename','mean']]
    else:
        from mars_anomaly.data import load_manifest
        manifest=load_manifest(data)
        rows=[]
        for row in manifest.itertuples():
            with Image.open(row.path) as im:rows.append({'filename':row.filename,'mean':float(np.asarray(im).mean())})
        context=pd.DataFrame(rows)
    joined=scores.merge(context,on='filename',validate='one_to_one')
    assert len(joined)==len(scores)
    joined['brightness_percentile']=joined['mean'].rank(pct=True)*100
    joined['brightness_percentile_within_source']=joined.groupby('source_image_id')['mean'].rank(pct=True)*100
    selected=pd.read_csv(directory/'selected_for_interpretation.csv')[['filename']].merge(joined,on='filename',validate='one_to_one')
    selected.to_csv(directory/'selected_acquisition_diagnostics.csv',index=False)
    print(selected[['filename','source_image_id','mean','brightness_percentile','brightness_percentile_within_source']].to_string(index=False))
    print(joined.groupby('flagged')['mean'].agg(['count','mean','median']).to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--statistics',type=Path,default=Path('outputs/audit/image_statistics.csv'))
    args=parser.parse_args();inspect(args.directory,args.data,args.statistics)
