from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruction_panel(original,reconstruction,path,input_label='Original'):
    original = original.detach().float().cpu().numpy()[:,0]
    reconstruction = reconstruction.detach().float().cpu().numpy()[:,0]
    n = len(original)
    fig,axes = plt.subplots(3,n,figsize=(2*n,6),squeeze=False)
    error = np.abs(original-reconstruction)
    vmax = max(float(error.max()),1e-6)
    for j in range(n):
        axes[0,j].imshow(original[j],cmap='gray',vmin=0,vmax=1)
        axes[1,j].imshow(reconstruction[j],cmap='gray',vmin=0,vmax=1)
        im = axes[2,j].imshow(error[j],cmap='inferno',vmin=0,vmax=vmax)
        for i in range(3): axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    for i,label in enumerate([input_label,'Reconstruction','Absolute error']): axes[i,0].set_ylabel(label)
    fig.colorbar(im,ax=axes[2,:].tolist(),label='Absolute intensity error [0,1]',fraction=.02)
    fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)


def score_plots(frame,calibration,path):
    scores = frame.novelty_score.to_numpy()
    threshold = calibration['threshold']
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for split,g in frame.groupby('split'):
        axes[0].hist(g.novelty_score,bins=50,density=True,histtype='step',label=split)
    axes[0].axvline(threshold,color='red',label='Calibrated fence');axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Novelty score (higher = more anomalous)',ylabel='Density',title='Score distributions by split')
    axes[1].plot(np.arange(len(scores)),np.sort(scores));axes[1].axhline(threshold,color='red')
    axes[1].axhspan(*calibration['bootstrap_interval_95'],color='red',alpha=.1)
    axes[1].set(xlabel='Sorted crop index',ylabel='Novelty score',title='Sorted scores and threshold uncertainty')
    axes[2].plot(np.sort(scores),np.arange(1,len(scores)+1)/len(scores))
    axes[2].axvline(threshold,color='red')
    axes[2].set(xlabel='Novelty score',ylabel='Empirical cumulative fraction',title='Empirical score distribution')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def heatmap_panel(original,reconstruction,row,threshold,path):
    original,reconstruction = np.asarray(original),np.asarray(reconstruction)
    error = np.abs(original-reconstruction)
    fig,axes = plt.subplots(1,4,figsize=(13,3.8))
    for ax in axes: ax.set_xticks([]);ax.set_yticks([])
    axes[0].imshow(original,cmap='gray',vmin=0,vmax=1);axes[0].set_title('Original')
    axes[1].imshow(reconstruction,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Reconstruction')
    im=axes[2].imshow(error,cmap='inferno',vmin=0,vmax=1);axes[2].set_title('Absolute error')
    axes[3].imshow(original,cmap='gray',vmin=0,vmax=1)
    axes[3].imshow(error,cmap='inferno',vmin=0,vmax=1,alpha=.55);axes[3].set_title('Error overlay')
    fig.colorbar(im,ax=axes.tolist(),fraction=.018,pad=.02,label='Error [0,1]; same scale for all candidates')
    fig.suptitle(f"{row.filename} | {row.source_image_id} | novelty {row.novelty_score:.4f} > {threshold:.4f}\nMaximum absolute error {error.max():.3f}; mean {error.mean():.3f}",fontsize=10)
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)


def standardized_heatmap_panel(raw,standardized,reconstruction,row,threshold,path,mode):
    error=np.abs(standardized-reconstruction)
    fig,axes=plt.subplots(1,5,figsize=(15,4))
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    for ax,value,title in zip(axes[:3],[raw,standardized,reconstruction],['Raw crop',f'Model input: {mode}','Reconstruction']):
        ax.imshow(value,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10)
    im=axes[3].imshow(error,cmap='inferno',vmin=0,vmax=1);axes[3].set_title('Model-input error',fontsize=10)
    axes[4].imshow(raw,cmap='gray',vmin=0,vmax=1)
    axes[4].imshow(error,cmap='inferno',vmin=0,vmax=1,alpha=.55);axes[4].set_title('Error on raw scene',fontsize=10)
    fig.colorbar(im,ax=axes.tolist(),fraction=.016,pad=.02,label='Standardized-domain error [0,1]')
    fig.suptitle(f'{row.filename} | {row.source_image_id} | novelty {row.novelty_score:.4f} > {threshold:.4f}\nError compares standardized input with reconstruction; not direct forest attribution',fontsize=10)
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
