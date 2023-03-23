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
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           # or 'NP' in rig 
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
    
    
# transition analysis
blockData = []
for exps in expsByMouse:
    for obj in exps:
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            d = {'mouseID':obj.subjectName,
                 'sessionStartTime': obj.startTime,
                 'blockNum':blockInd+1,
                 'goStim':goStim,
                 'numAutoRewards':obj.autoRewarded[:10].sum()}
            blockTrials = obj.trialBlock == blockInd + 1
            for trials,lbl in zip((obj.goTrials,obj.sameModalNogoTrials,obj.otherModalGoTrials,obj.otherModalNogoTrials),
                                  ('goTrials','sameModalNogoTrials','otherModalGoTrials','otherModalNogoTrials')):
                trials = trials & blockTrials
                d[lbl] = {'startTimes':obj.stimStartTimes[trials]-obj.blockFirstStimTimes[blockInd],
                          'response':obj.trialResponse[trials],
                          'responseTime':obj.responseTimes[trials]}
            blockData.append(d)
        
for blockType in ('visual','auditory'):
    goStim = 'vis' if blockType=='visual' else 'sound'
    nTransitions = 0    
    goProb = []
    goProbPrev = []
    goProbFirst = []
    nogoProb = []
    nogoProbPrev = []
    nogoProbFirst = []
    otherGoProb = []
    otherGoProbPrev = []
    otherGoProbFirst = []
    otherNogoProb = []
    otherNogoProbPrev = []
    otherNogoProbFirst = []
    goLat = []
    goLatPrev = []
    goLatFirst = []
    nogoLat = []
    nogoLatPrev = [] 
    nogoLatFirst = []
    otherGoLat = []
    otherGoLatPrev = []
    otherGoLatFirst = []
    otherNogoLat = []
    otherNogoLatPrev = []
    otherNogoLatFirst = []
    for i,block in enumerate(blockData):
        if goStim in block['goStim']:
            if block['blockNum'] > 1:
                nTransitions += 1
                prevBlock = blockData[i-1]
                goProb.append(block['goTrials']['response'])
                goProbPrev.append(prevBlock['otherModalGoTrials']['response'])
                nogoProb.append(block['sameModalNogoTrials']['response'])
                nogoProbPrev.append(prevBlock['otherModalNogoTrials']['response'])
                otherGoProb.append(block['otherModalGoTrials']['response'])
                otherGoProbPrev.append(prevBlock['goTrials']['response'])
                otherNogoProb.append(block['otherModalNogoTrials']['response'])
                otherNogoProbPrev.append(prevBlock['sameModalNogoTrials']['response'])
                
                goLat.append(block['goTrials']['responseTime'])
                goLatPrev.append(prevBlock['otherModalGoTrials']['responseTime'])
                nogoLat.append(block['sameModalNogoTrials']['responseTime'])
                nogoLatPrev.append(prevBlock['otherModalNogoTrials']['responseTime'])
                otherGoLat.append(block['otherModalGoTrials']['responseTime'])
                otherGoLatPrev.append(prevBlock['goTrials']['responseTime'])
                otherNogoLat.append(block['otherModalNogoTrials']['responseTime'])
                otherNogoLatPrev.append(prevBlock['sameModalNogoTrials']['responseTime'])
            else:
                goProbFirst.append(block['goTrials']['response'])
                nogoProbFirst.append(block['sameModalNogoTrials']['response'])
                otherGoProbFirst.append(block['otherModalGoTrials']['response'])
                otherNogoProbFirst.append(block['otherModalNogoTrials']['response'])
                
                goLatFirst.append(block['goTrials']['responseTime'])
                nogoLatFirst.append(block['sameModalNogoTrials']['responseTime'])
                otherGoLatFirst.append(block['otherModalGoTrials']['responseTime'])
                otherNogoLatFirst.append(block['otherModalNogoTrials']['responseTime'])
    
    nMice = len(set(d['mouseID'] for d in blockData))
    nSessions = len(set(d['sessionStartTime'] for d in blockData))
    title = (blockType+' rewarded blocks\n'
             'mean and 95% ci across transitions\n('+
             str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    preTrials = postTrials = 15 # 15, 45
    x = np.arange(-preTrials,postTrials+1)
    xlim =[-preTrials,postTrials]
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goProbFirst,nogoProbFirst),(otherGoProbFirst,otherNogoProbFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockType+' rewarded first block\n('+str(len(goProbFirst)) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.3,0.6]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goLatFirst,nogoLatFirst),(otherGoLatFirst,otherNogoLatFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            if 'nogo' in stim:
                continue
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='upper right')
    ax.set_title(blockType+' rewarded first block\n('+str(len(goLatFirst)) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goProb,goProbPrev),(nogoProb,nogoProbPrev)),((otherGoProb,otherGoProbPrev),(otherNogoProb,otherNogoProbPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.3,0.6]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goLat,goLatPrev),(nogoLat,nogoLatPrev)),((otherGoLat,otherGoLatPrev),(otherNogoLat,otherNogoLatPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            if 'nogo' in stim:
                continue
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='upper right')
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xmax = 9
for goStim,clr,lbl in zip(('vis1','sound1'),'mg',('auditory','visual')):
    trialsToNogo = []
    firstNogoResp = []
    for block in blockData:
        if block['goStim']==goStim:
            trialsToNogo.append(np.sum(block['goTrials']['startTimes'] < block['otherModalGoTrials']['startTimes'][0]) + block['numAutoRewards'])
            firstNogoResp.append(block['otherModalGoTrials']['response'][0])
    trialsToNogo = np.array(trialsToNogo)
    firstNogoResp = np.array(firstNogoResp)
    x = np.unique(trialsToNogo)
    r = [firstNogoResp[trialsToNogo==n] for n in x]
    n = [len(d) for d in r]
    m = np.array([np.nanmean(d) for d in r])
    ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in r]
    ax.plot(x[x<=xmax],m[x<=xmax],color=clr,label=lbl)
    for i,c in enumerate(ci):
        if i<=xmax:
            ax.plot([x[i]]*2,c,clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,xmax+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Number of preceeding go stim trials, including autorewards')
ax.set_ylabel('Probability of responding\nto first previous go stim trial')
ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for blockType,goStim,clr in zip(('visual','auditory'),(('vis1','sound1'),('sound1','vis1')),'gm'):
    for stim,trials,ls,lbl in zip(goStim,('goTrials','otherModalGoTrials'),('-',':'),('rewarded','unrewarded')):
        d = [block[trials]['responseTime'] for block in blockData if block['goStim']==stim]
        d = np.concatenate(d)
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=blockType+' '+lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0.1,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Response Latency (s)')
ax.set_ylabel('Cumulative Probability')
ax.legend(loc='lower right')
plt.tight_layout()
    
    
# plot d prime and response time by block
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
ax.set_ylim([0,2.5])
ax.set_xlabel('Block')
ax.set_ylabel('d\' other modality')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for stim,clr in zip(('vis1','sound1'),'gm'):
    rewStim = ('vis1','sound1') if stim=='vis1' else ('sound1','vis1')
    for rs,ls,alpha,lbl in zip(rewStim,('-','--'),(0.4,0.2),('rewarded','unrewarded')):
        rt = []
        for exps in expsByMouse:
            r = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps):
                for j,goStim in enumerate(obj.blockStimRewarded):
                    if rs==goStim:
                        trials = (obj.trialBlock==j+1) & (obj.trialStim==stim) & ~obj.autoRewarded
                        r[i,j] = np.nanmean(obj.responseTimes[trials])
            rt.append(np.nanmean(r,axis=0))
        m = np.nanmean(rt,axis=0)
        s = np.nanstd(rt,axis=0)/(len(rt)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=stim+' '+lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=alpha)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.3,0.6])
ax.set_xlabel('Block')
ax.set_ylabel('Response time (s)')
ax.legend(loc='upper right')
plt.tight_layout()


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


# cluster response rate data
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
                
                r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                r = np.interp(tintp,startTime,r)
                d['response'][stim].append(r)
                
                rtTrials = obj.trialResponse[trials]
                if np.any(rtTrials):
                    r = scipy.ndimage.gaussian_filter(obj.responseTimes[trials][rtTrials].astype(float),smoothSigma)
                    r = np.interp(tintp,startTime[rtTrials],r)
                    d['responseTime'][stim].append(r)
                else:
                    d['responseTime'][stim].append(np.full(tintp.size,np.nan))
                   
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
    

clustData = pcaData[:,:nPC]

clustColors = 'krbgmcy'
    
clustId,linkageMat = cluster(clustData,nClusters=6,plot=True,colors=clustColors,labels='off',nreps=0)

clustLabels = np.unique(clustId)

minClust = 2
maxClust = 10
clustScores = np.zeros((3,maxClust-minClust+1))
for i,n in enumerate(range(minClust,maxClust+1)):
    cid = cluster(clustData,nClusters=n,plot=False)[0]
    clustScores[0,i] = sklearn.metrics.silhouette_score(clustData,cid)
    clustScores[1,i] = sklearn.metrics.calinski_harabasz_score(clustData,cid)
    clustScores[2,i] = sklearn.metrics.davies_bouldin_score(clustData,cid)


for resp in ('response♥',):
    for clust in clustLabels:
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                if '1' in stim or resp!='responseTime':
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















