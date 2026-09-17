"""Diagnose forest Monte Carlo variability with a fixed statistical rule."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from mars_anomaly.threshold import mixture_fence


def run(directory):
    z=np.load(directory/'latents.npy');frame=pd.read_csv(directory/'novelty_scores.csv')
    train=frame.split.eq('train').to_numpy();cal=frame.split.eq('calibration').to_numpy()
    records=[]
    for trees in [400,2000]:
        for max_samples in [256,512]:
            previous=None
            for seed in [2026,2027,2028]:
                forest=IsolationForest(n_estimators=trees,max_samples=max_samples,random_state=seed,contamination='auto',n_jobs=2).fit(z[train])
                scores=-forest.score_samples(z)
                threshold=mixture_fence(scores[cal])['threshold'];flags=scores>threshold
                record={'trees':trees,'max_samples':max_samples,'seed':seed,'threshold':threshold,'flags':int(flags.sum())}
                if previous is None:previous=(scores,flags)
                else:
                    union=np.count_nonzero(flags|previous[1])
                    record.update({'spearman_vs_first':float(spearmanr(scores,previous[0]).statistic),'jaccard_vs_first':float(np.count_nonzero(flags&previous[1])/union) if union else None})
                records.append(record)
    pd.DataFrame(records).to_csv(directory/'forest_sensitivity.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
