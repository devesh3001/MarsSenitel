"""Create the paired preprocessing examples and domain-separated learning curves."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mars_anomaly.preprocessing import preprocess_array

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/comparison';out.mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'outputs/v3/split_manifest.csv')
    preview=frame[frame.split=='validation'].iloc[[0,1,6,7]]
    fig,axes=plt.subplots(4,3,figsize=(9,11))
    for i,row in enumerate(preview.itertuples()):
        with Image.open(ROOT/'data/images'/row.filename) as im:raw=np.asarray(im,dtype=np.float32)/255
        for j,mode in enumerate(['raw','contrast','footprint']):
            axes[i,j].imshow(preprocess_array(raw,mode),cmap='gray',vmin=0,vmax=1)
            axes[i,j].set_title(row.filename+' | '+mode,fontsize=9);axes[i,j].axis('off')
    fig.tight_layout();fig.savefig(out/'preprocessing_examples.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,names in enumerate([['v1','v2','v3'],['v4','v5']]):
        for name in names:
            h=pd.read_csv(ROOT/'outputs'/name/'history.csv')
            for col,metric in enumerate(['mse','ssim']):axes[row,col].plot(h.epoch,h['validation_'+metric],label=name)
        for col,metric in enumerate(['MSE','SSIM']):
            axes[row,col].set(xlabel='Epoch',ylabel=metric,title='Raw input' if row==0 else 'Standardized inputs: different targets');axes[row,col].legend()
    fig.tight_layout();fig.savefig(out/'improvement_validation_curves.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
