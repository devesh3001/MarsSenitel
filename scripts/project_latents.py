"""Generate t-SNE diagnostics from existing latents without rerunning calibration."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def project(directory):
    z=np.load(directory/'latents.npy')
    frame=pd.read_csv(directory/'novelty_scores.csv')
    assert np.array_equal(np.load(directory/'latent_filenames.npy'),frame.filename)
    train=frame.split.eq('train').to_numpy()
    scaled=(z-z[train].mean(0))/z[train].std(0).clip(1e-6)
    with threadpool_limits(limits=2):
        reduced=PCA(n_components=min(50,z.shape[1]),random_state=2026).fit_transform(scaled)
        for perplexity in [30,70]:
            xy=TSNE(perplexity=perplexity,max_iter=1000,init='pca',learning_rate='auto',random_state=2026,n_jobs=2).fit_transform(reduced)
            pd.DataFrame({'filename':frame.filename,'x':xy[:,0],'y':xy[:,1]}).to_csv(directory/f'tsne_{perplexity}.csv',index=False)
            fig,axes=plt.subplots(1,3,figsize=(15,4.6))
            for ax,values,label in zip(axes[:2],[frame.novelty_score,frame.sun_angle],['Novelty score','Sun angle as supplied']):
                points=ax.scatter(xy[:,0],xy[:,1],c=values,s=3,cmap='viridis',rasterized=True)
                fig.colorbar(points,ax=ax,shrink=.8);ax.set_title(label)
            largest=frame.source_image_id.value_counts().head(5).index
            axes[2].scatter(xy[:,0],xy[:,1],s=2,c='lightgray',label='Other sources')
            for i,source in enumerate(largest):
                mask=frame.source_image_id.eq(source)
                axes[2].scatter(xy[mask,0],xy[mask,1],s=4,color=plt.get_cmap('tab10')(i),label=source)
            axes[2].legend(fontsize=7,loc='best');axes[2].set_title('Five largest source groups')
            for ax in axes:ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f't-SNE, perplexity {perplexity}: qualitative diagnostic, not confirmed terrain classes')
            fig.tight_layout();fig.savefig(directory/f'tsne_{perplexity}.png',dpi=160);plt.close(fig)
            print(f'Completed t-SNE {perplexity}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    project(parser.parse_args().directory)
