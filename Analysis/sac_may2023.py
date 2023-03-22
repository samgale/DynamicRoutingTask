# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import copy
import os
import numpy as np
import pandas as pd
import scipy.cluster
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData
import sklearn
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests


# get data
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mouseIds = allMiceDf['mouse id']
# mouseIds = ('638573','638574','638575','638576','638577','638578')

passOnly = True

mice = []
sessionStartTimes = []
passSession =[]
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
        df = sheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        if any('stage 3' in task for task in df['task version']) and not any('stage 4' in task for task in df['task version']):
            sessions[np.where(sessions)[0][0]] = False # skipping first 6-block session when preceded by distractor training
        firstExperimentSession = np.where(['multimodal' in task  or 
                                           'contrast'in task or 
                                           'opto' in task or
                                           'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        if sessions.sum() > 0 and df['pass'][sessions].sum() > 0:
            mice.append(str(mid))
            if passOnly:
                sessions[:np.where(sessions & df['pass'])[0][0]-1] = False
                passSession.append(0)
            else:
                passSession.append(np.where(df['pass'][sessions])[0][0]-1)
            sessionStartTimes.append(list(df['start time'][sessions]))
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)
        
nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]
                

# plot mouse learning curve
for ylbl in ('d\' same modality','d\' other modality'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((nMice,max(nExps)),np.nan)
    for i,(exps,clr) in enumerate(zip(expsByMouse,plt.cm.tab20(np.linspace(0,1,nMice)))):
        if 'same' in ylbl:
            d = [np.mean(obj.dprimeSameModal) for obj in exps]
        else:
            d = [np.mean(obj.dprimeOtherModalGo) for obj in exps]
        ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25)
        ax.plot(passSession[i]+1,d[passSession[i]],'o',ms=10,color=clr,alpha=0.25)
        dp[i,:len(d)] = d
    m = np.nanmean(dp,axis=0)
    ax.plot(np.arange(len(m))+1,m,'k',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,len(m)+1])
    ax.set_ylim([0,4])
    ax.set_xlabel('Session')
    ax.set_ylabel(ylbl)
    plt.tight_layout()
 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
switchResp = np.full((nMice,max(nExps)),np.nan)
for i,(exps,clr) in enumerate(zip(expsByMouse,plt.cm.tab20(np.linspace(0,1,nMice)))):
    r = []
    for obj in exps:
        rr = []
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            nogoStim = 'sound1' if goStim=='vis1' else 'vis1'
            rr.append(obj.trialResponse[(obj.trialBlock==blockInd+1) & (obj.trialStim==nogoStim)][0])
        r.append(np.mean(rr))
    ax.plot(np.arange(len(r))+1,r,color=clr,alpha=0.25)
    ax.plot(passSession[i]+1,r[passSession[i]],'o',ms=10,color=clr,alpha=0.25)
    switchResp[i,:len(r)] = r
m = np.nanmean(switchResp,axis=0)
ax.plot(np.arange(len(m))+1,m,'k',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,len(m)+1])
ax.set_ylim([0,1])
ax.set_xlabel('Session')
ax.set_ylabel('First nogo trial resp rate')
plt.tight_layout()
    
    
# plot d prime by block and rewarded modality
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','sound rewarded')):
    dp = []
    for exps in expsByMouse:
        d = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            j = obj.blockStimRewarded==rewardStim
            d[i,j] = np.array(obj.dprimeOtherModalGo)[j]
        dp.append(np.nanmean(d,axis=0))
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_yticks(np.arange(0,5,0.5))
ax.set_ylim([0,2])
ax.set_xlabel('Block')
ax.set_ylabel('d\' other modality')
ax.legend(loc='lower right')
plt.tight_layout()
    
    
# plot example sessions
nExpsToPlot = 6
smoothSigma = 5
for m,exps in enumerate(expsByMouse):
    fig = plt.figure(figsize=(12,10))
    ylim = [-0.05,1.05]
    for i,obj in enumerate(exps[passSession[m]:passSession[m]+nExpsToPlot]):
        ax = fig.add_subplot(nExpsToPlot,1,i+1)
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            blockStart,blockEnd = np.where(blockTrials)[0][[0,-1]]
            if goStim=='vis1':
                lbl = 'vis rewarded' if blockInd==0 else None
                ax.add_patch(matplotlib.patches.Rectangle([blockStart+0.5,ylim[0]],width=blockEnd-blockStart+1,height=ylim[1]-ylim[0],facecolor='0.8',edgecolor=None,alpha=0.2,zorder=0,label=lbl))
            for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim)
                smoothedRespProb = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                lbl = stim if i==0 and blockInd==0 else None
                ax.plot(np.where(trials)[0]+1,smoothedRespProb,color=clr,ls=ls,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([0.5,blockEnd+1.5])
        ax.set_ylim(ylim)
        if i==nExpsToPlot-1:
            ax.set_xlabel('Trial',fontsize=12)
        if i==0:
            ax.set_ylabel('Response rate',fontsize=12)
            ax.legend(bbox_to_anchor=(1,1.5),fontsize=8)
        # ax.set_title(obj.subjectName+'_'+obj.startTime,fontsize=10)
    plt.tight_layout()
    

# cluster behavior data

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
        
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
            
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            k = np.arange(linkageMat.shape[0])+2
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


# cluster raw behavior data after re-ordering stimuli
stimNames = ('vis1','vis2','sound1','sound2')
d = {key: [] for key in ('mouse','session','block','rewardStim','clustData')}
d['response'] = {stim: [] for stim in stimNames}
d['responseTime'] = {stim: [] for stim in stimNames}
smoothSigma = 5
tintp = np.arange(600)
for m,exps in enumerate(expsByMouse):
    for i,obj in enumerate(exps[passSession[m]:]):
        for blockInd,rewardStim in enumerate(obj.blockStimRewarded):
            d['mouse'].append(m)
            d['session'].append(i)
            d['block'].append(blockInd)
            d['rewardStim'].append(rewardStim)
            blockTrials = obj.trialBlock==blockInd+1
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim) & ~obj.autoRewarded
                startTime = obj.trialStartTimes[trials]
                startTime = startTime-startTime[0]
                for resp,key in zip((obj.trialResponse,obj.responseTimes),('response','responseTime')):
                    r = scipy.ndimage.gaussian_filter(resp[trials].astype(float),smoothSigma)
                    r = np.interp(tintp,startTime,r)
                    d[key][stim].append(r)
            sn = stimNames if rewardStim=='vis1' else stimNames[-2:]+stimNames[:2]
            d['clustData'].append(np.concatenate([d['response'][stim][-1] for stim in sn]))

for key in d:
    if isinstance(d[key],dict):
        for k in d[key]:                
            d[key][k] = np.array(d[key][k])
    else:
        d[key] = np.array(d[key])


pcaData,eigVal,eigVec = pca(d['clustData'],plot=True)
nPC = np.where((np.cumsum(eigVal)/eigVal.sum())>0.95)[0][0]
    
clustColors = 'mrkbcgy'
    
clustId,linkageMat = cluster(pcaData[:,:nPC],nClusters=6,plot=True,colors=clustColors,labels='off',nreps=1)

clustLabels = np.unique(clustId)

for resp in ('response',):
    for clust in clustLabels:
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                r = d[resp][stim][(d['rewardStim']==rewardStim) & (clustId==clust)]
                m = np.nanmean(r,axis=0)
                s = np.nanstd(r)/(len(r)**0.5)
                ax.plot(tintp,m,color=clr,ls=ls,label=stim)
                ax.fill_between(tintp,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Response rate')
            ax.legend(loc='lower right')
            ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(r))+')')
            plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)   
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded blocks','sound rewarded blocks')):
    y = []
    for clust in clustLabels:
        blocks = d['rewardStim']==rewardStim
        y.append(np.sum(blocks & (clustId==clust))/blocks.sum())
    ax.plot(clustLabels,y,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(clustLabels)
ax.set_ylim([0,0.5])
ax.set_xlabel('Cluster')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()


blockClustProb = np.zeros((len(clustLabels),6))
for i,clust in enumerate(clustLabels):
    for j in range(6):
        blocks = d['block']==j
        blockClustProb[i,j] = np.sum(blocks & (clustId==clust))/blocks.sum()

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(blockClustProb,cmap='magma',clim=(0,blockClustProb.max()),origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(np.arange(6)+1)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Block')
ax.set_ylabel('Cluster')
ax.set_title('Probability')
plt.tight_layout()

chanceProb = np.array([np.sum(clustId==clust)/len(clustId) for clust in clustLabels])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
a = blockClustProb-chanceProb[:,None]
amax = np.absolute(a).max()
im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(np.arange(6)+1)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Block')
ax.set_ylabel('Cluster')
ax.set_title('Difference from chance probability')
plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(clustId),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(clustId) for clust in clustLabels] for r in randClust])

pval = np.zeros_like(blockClustProb)
for j,p in enumerate(blockClustProb.T):
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
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(np.arange(6)+1)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Block')
ax.set_ylabel('Cluster')
ax.set_title('Corrected p-value')
plt.tight_layout()


transProb = np.zeros((len(clustLabels),)*2)
blocks = np.where(d['block']<5)[0]
for j,clust in enumerate(clustLabels):
    c = clustId[blocks]==clust
    for i,nextClust in enumerate(clustLabels):
        transProb[i,j] = np.sum(clustId[blocks+1][c]==nextClust)/c.sum()
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(transProb,cmap='magma',clim=(0,transProb.max()),origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(len(clustLabels)))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(clustLabels)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Current block cluster')
ax.set_ylabel('Next block cluster')
ax.set_title('Probability')
plt.tight_layout()

chanceProb = np.array([np.sum(clustId[blocks+1]==clust)/len(blocks) for clust in clustLabels])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
a = transProb-chanceProb[:,None]
amax = np.absolute(a).max()
im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(len(clustLabels)))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(clustLabels)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Current block cluster')
ax.set_ylabel('Next block cluster')
ax.set_title('Difference from chance probability')
plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(blocks),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(blocks) for clust in clustLabels] for r in randClust])

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
ax.set_ylabel('Next block cluster')
ax.set_title('Corrected p-value')
plt.tight_layout()















