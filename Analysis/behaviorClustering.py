# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:55:39 2023

@author: svc_ccg
"""

import numpy as np
import scipy
import scipy.cluster
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
import sklearn.metrics
from statsmodels.stats.multitest import multipletests


def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None,nreps=1000,labels=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustId = None
    else:
        clustId = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        if colors is not None:
            scipy.cluster.hierarchy.set_link_color_palette(list(colors))
        if labels=='off':
            labels=None
            noLabels=True
        else:
            noLabels=False
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k',labels=labels,no_labels=noLabels)
        scipy.cluster.hierarchy.set_link_color_palette(None)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        k = np.arange(linkageMat.shape[0])+2
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
            ax.plot(k,np.percentile(randLinkage,2.5,axis=0),'k--')
            ax.plot(k,np.percentile(randLinkage,97.5,axis=0),'k--')
        ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
        ax.set_xlim([0,k[-1]+1])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Linkage Distance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        plt.tight_layout()
    
    return clustId,linkageMat


def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('PC')
        ax.set_ylabel('Parameter')
        ax.set_title('PC Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec



expsByMouse = [exps for lbl in sessionData for exps in sessionData[lbl]]


expsByMouse = [exps[:i] for exps,i in zip(sessionData,sessionsToPass)]

expsByMouse = [exps[i:] for exps,i in zip(sessionData,sessionsToPass)]


nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]


# cluster response rate data
stimNames = ('vis1','vis2','sound1','sound2')
clustData = {key: [] for key in ('mouse','session','block','rewardStim','clustData')}
clustData['response'] = {stim: [] for stim in stimNames}
clustData['smoothedResponse'] = {stim: [] for stim in stimNames}
clustData['responseTime'] = {stim: [] for stim in stimNames}
smoothSigma = 5
tintp = np.arange(600)
for m,exps in enumerate(expsByMouse):
    for i,obj in enumerate(exps):
        for blockInd,rewardStim in enumerate(obj.blockStimRewarded):
            clustData['mouse'].append(m)
            clustData['session'].append(i)
            clustData['block'].append(blockInd)
            clustData['rewardStim'].append(rewardStim)
            blockTrials = obj.trialBlock==blockInd+1
            for stim in stimNames:
                trials = blockTrials & (obj.trialStim==stim) & ~obj.autoRewarded
                if trials.sum() > 0:
                    stimTime = obj.stimStartTimes[trials]
                    stimTime = stimTime-obj.trialStartTimes[trials][0]
                    
                    clustData['response'][stim].append(obj.trialResponse[trials])
                    r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                    r = np.interp(tintp,stimTime,r)
                    clustData['smoothedResponse'][stim].append(r)
                    
                    rtTrials = obj.trialResponse[trials]
                    if np.any(rtTrials):
                        r = scipy.ndimage.gaussian_filter(obj.responseTimes[trials][rtTrials].astype(float),smoothSigma)
                        r = np.interp(tintp,stimTime[rtTrials],r)
                        clustData['responseTime'][stim].append(r)
                    else:
                        clustData['responseTime'][stim].append(np.full(tintp.size,np.nan))
                else:
                    clustData['response'][stim].append(np.array([]))
                    clustData['smoothedResponse'][stim].append(np.full(tintp.size,np.nan))
                    clustData['responseTime'][stim].append(np.full(tintp.size,np.nan))
                   
            # sn = stimNames[:4] if rewardStim=='vis1' else stimNames[2:4]+stimNames[:2]
            sn = ('vis1','sound1') if rewardStim=='vis1' else ('sound1','vis1')
            clustData['clustData'].append(np.concatenate([clustData['smoothedResponse'][stim][-1] for stim in sn]))

for key in clustData:
    if isinstance(clustData[key],dict):
        for k in clustData[key]:                
            clustData[key][k] = np.array(clustData[key][k])
    else:
        clustData[key] = np.array(clustData[key])


clustColors = [clr for clr in 'krbgmcy']+['0.6']

nClust = 4

clustId,linkageMat = cluster(clustData['clustData'],nClusters=nClust,plot=True,colors=clustColors,labels='off',nreps=0)

pcaData,eigVal,eigVec = pca(clustData['clustData'],plot=False)
nPC = np.where((np.cumsum(eigVal)/eigVal.sum())>0.95)[0][0]
clustId,linkageMat = cluster(pcaData[:,:nPC],nClusters=nClust,plot=True,colors=clustColors,labels='off',nreps=0)        

clustLabels = np.unique(clustId)

# minClust = 2
# maxClust = 10
# clustScores = np.zeros((3,maxClust-minClust+1))
# for i,n in enumerate(range(minClust,maxClust+1)):
#     cid = cluster(pcaData[:,:nPC],nClusters=n,plot=False)[0]
#     clustScores[0,i] = sklearn.metrics.silhouette_score(pcaData[:,:nPC],cid)
#     clustScores[1,i] = sklearn.metrics.calinski_harabasz_score(pcaData[:,:nPC],cid)
#     clustScores[2,i] = sklearn.metrics.davies_bouldin_score(pcaData[:,:nPC],cid)


for resp in ('smoothedResponse',):
    for clust in clustLabels:
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                if '1' in stim or resp!='responseTime':
                    r = clustData[resp][stim][(clustData['rewardStim']==rewardStim) & (clustId==clust)]
                    m = np.nanmean(r,axis=0)
                    s = np.nanstd(r)/(len(r)**0.5)
                    ax.plot(tintp,m,color=clr,lw=2,ls=ls,label=stim)
                    ax.fill_between(tintp,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=14)
            ax.set_ylim([0,1.02])
            ax.set_xlabel('Time (s)',fontsize=16)
            ax.set_ylabel('Response rate',fontsize=16)
            ax.legend(loc='lower right',fontsize=14)
            ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(r))+')',fontsize=18)
            plt.tight_layout()


postTrials = 15
x = np.arange(postTrials)+1
for clust in clustLabels:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
            resp = []
            for r in [r for i,r in enumerate(clustData['response'][stim]) if clustData['rewardStim'][i]==rewardStim and clustId[i]==clust]:
                j = min(postTrials,r.size)
                resp.append(np.full(postTrials,np.nan))
                resp[-1][:j] = r[:j]
            m = np.nanmean(resp,axis=0)
            s = np.nanstd(resp)/(len(resp)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stim)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch cue trials')
        ax.set_ylabel('Response rate')
        ax.legend(loc='lower right')
        ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
        plt.tight_layout()
        


fig = plt.figure()
ax = fig.add_subplot(1,1,1)   
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded blocks','sound rewarded blocks')):
    y = []
    for clust in clustLabels:
        blocks = clustData['rewardStim']==rewardStim
        y.append(np.sum(blocks & (clustId==clust))/blocks.sum())
    ax.plot(clustLabels,y,color=clr,lw=2,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(clustLabels)
ax.set_ylim([0,0.5])
ax.set_xlabel('Cluster',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()


mouseClustProb = np.zeros((nMice,nClust))
ind = 0
for i,n in enumerate(nExps):
    for j,clust in enumerate(clustLabels):
        mouseClustProb[i,j] = np.sum(clustId[ind:ind+n]==clust)/n
    ind += n

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(mouseClustProb,cmap='magma',clim=(0,mouseClustProb.max()))
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(nClust))
ax.set_xticklabels(np.arange(nClust)+1)
yticks = np.concatenate(([0],np.arange(4,nMice+1,5)))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Mouse')
ax.set_title('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)   
ax.bar(clustLabels,np.sum(mouseClustProb>0,axis=0)/nMice,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(clustLabels)
ax.set_ylim([0,1.01])
ax.set_xlabel('Cluster',fontsize=14)
ax.set_ylabel('Fraction of mice contributing to cluster',fontsize=14)
plt.tight_layout()


sessionClustProb = np.zeros((sum(nExps),nClust))
ind = 0
for i in range(sum(nExps)):
    for j,clust in enumerate(clustLabels):
        sessionClustProb[i,j] = np.sum(clustId[ind:ind+nClust]==clust)/nClust
    ind += nClust

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(sessionClustProb,cmap='magma',clim=(0,sessionClustProb.max()),aspect='auto',interpolation='none')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(nClust))
ax.set_xticklabels(np.arange(nClust)+1)
# yticks = np.concatenate(([0],np.arange(4,nMice+1,5)))
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticks+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Session')
ax.set_title('Probability')
plt.tight_layout()


blockClustProb = np.zeros((6,nClust))
for i in range(6):
    blocks = clustData['block']==i
    for j,clust in enumerate(clustLabels):
        blockClustProb[i,j] = np.sum(blocks & (clustId==clust))/blocks.sum()

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(blockClustProb,cmap='magma',clim=(0,blockClustProb.max()),origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(nClust))
ax.set_yticks(np.arange(6))
ax.set_xticklabels(clustLabels)
ax.set_yticklabels(np.arange(6)+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Block')
ax.set_title('Probability')
plt.tight_layout()

chanceProb = np.array([np.sum(clustId==clust)/len(clustId) for clust in clustLabels])

for lbl in ('Absolute','Relative'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    a = blockClustProb-chanceProb
    if lbl=='Relative':
        a /= chanceProb
    amax = np.absolute(a).max()
    im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(nClust))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(clustLabels)
    ax.set_yticklabels(np.arange(6)+1)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Block')
    ax.set_title(lbl+' difference from chance probability')
    plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(clustId),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(clustId) for clust in clustLabels] for r in randClust])

pval = np.zeros_like(blockClustProb)
for i,p in enumerate(blockClustProb):
    lessThan = np.sum(randClustProb<p,axis=0)/randClustProb.shape[0]
    greaterThan = np.sum(randClustProb>p,axis=0)/randClustProb.shape[0]
    pval[i] = np.min(np.stack((lessThan,greaterThan)),axis=0)
pval[pval==0] = 1/nIter

alpha = 0.05
pvalCorr = np.reshape(multipletests(pval.flatten(),alpha=alpha,method='fdr_bh')[1],pval.shape)

fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
lim = (10**np.floor(np.log10(np.min(pvalCorr))),alpha)
clim = np.log10(lim)
im = ax.imshow(np.log10(pvalCorr),cmap='gray',clim=clim,origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
cb.ax.tick_params(labelsize=10) 
legticks = np.concatenate((np.arange(clim[0],clim[1]),[clim[1]]))
cb.set_ticks(legticks)
cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
ax.set_xticks(np.arange(nClust))
ax.set_yticks(np.arange(6))
ax.set_xticklabels(clustLabels)
ax.set_yticklabels(np.arange(6)+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Block')
ax.set_title('Corrected p-value')
plt.tight_layout()


prevClustProb = np.zeros((len(clustLabels),)*2)
blocks = np.where(clustData['block']>0)[0]
for j,clust in enumerate(clustLabels):
    c = clustId[blocks]==clust
    for i,prevClust in enumerate(clustLabels):
        prevClustProb[i,j] = np.sum(clustId[blocks-1][c]==prevClust)/c.sum()

nextClustProb = np.zeros((len(clustLabels),)*2)
blocks = np.where(clustData['block']<5)[0]
for j,clust in enumerate(clustLabels):
    c = clustId[blocks]==clust
    for i,nextClust in enumerate(clustLabels):
        nextClustProb[i,j] = np.sum(clustId[blocks+1][c]==nextClust)/c.sum()

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    im = ax.imshow(transProb,cmap='magma',clim=(0,transProb.max()),origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(len(clustLabels)))
    ax.set_yticks(np.arange(len(clustLabels)))
    ax.set_xticklabels(clustLabels)
    ax.set_yticklabels(clustLabels)
    ax.set_xlabel('Current block cluster')
    ax.set_ylabel(lbl+' block cluster')
    ax.set_title('Probability')
    plt.tight_layout()

chanceProb = np.array([np.sum(clustId[blocks+1]==clust)/len(blocks) for clust in clustLabels])

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    for diff in ('Absolute','Relative'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        a = transProb-chanceProb[:,None]
        if diff=='Relative':
            a /= chanceProb[:,None]
        amax = np.absolute(a).max()
        im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
        cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
        ax.set_xticks(np.arange(len(clustLabels)))
        ax.set_yticks(np.arange(len(clustLabels)))
        ax.set_xticklabels(clustLabels)
        ax.set_yticklabels(clustLabels)
        ax.set_xlabel('Current block cluster')
        ax.set_ylabel(lbl+' block cluster')
        ax.set_title(diff+' difference from chance probability')
        plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(blocks),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(blocks) for clust in clustLabels] for r in randClust])

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    pval = np.zeros_like(transProb)
    for j,p in enumerate(transProb.T):
        lessThan = np.sum(randClustProb<p,axis=0)/randClustProb.shape[0]
        greaterThan = np.sum(randClustProb>p,axis=0)/randClustProb.shape[0]
        pval[:,j] = np.min(np.stack((lessThan,greaterThan)),axis=0)
    pval[pval==0] = 1/nIter
    
    alpha = 0.05
    pvalCorr = np.reshape(multipletests(pval.flatten(),alpha=alpha,method='fdr_bh')[1],pval.shape)
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.min(pvalCorr))),alpha)
    clim = np.log10(lim)
    im = ax.imshow(np.log10(pvalCorr),cmap='gray',clim=clim,origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.ax.tick_params(labelsize=10) 
    legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
    cb.set_ticks(legticks)
    cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
    ax.set_xticks(np.arange(len(clustLabels)))
    ax.set_yticks(np.arange(len(clustLabels)))
    ax.set_xticklabels(clustLabels)
    ax.set_yticklabels(clustLabels)
    ax.set_xlabel('Current block cluster')
    ax.set_ylabel(lbl+' block cluster')
    ax.set_title('Corrected p-value')
    plt.tight_layout()
