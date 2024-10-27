# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:38:37 2024

@author: svc_ccg
"""

import copy
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\2024-10-11\decoder_confidence_versus_trials_since_rewarded_target_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

areaNames = np.unique(df['area'])

areas = ('ORBl','ORBm','ORBvl') + ('ACAd','ACAv') + ('PL',) + ('MOs',) + ('CP','STR') + ('SCig','SCiw','SCdg','MRN')
sessions = np.in1d(df['area'],areas)
nSessions = sessions.sum()


def getSessionObj(df,session):
    sessionName = df['session'][sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    return obj


def getNonShiftTrials(obj):
    ind = []
    for block in (1,6):
        blockTrials = np.where((obj.trialBlock[~obj.autoRewardScheduled]==block))[0]
        ind.append(blockTrials[int(np.ceil(len(blockTrials)/2))])
    trials = np.zeros(obj.nTrials,dtype=bool)
    trials[np.where(~obj.autoRewardScheduled)[0][ind[0]:ind[1]+1]] = True
    return trials
    

    
def getDecoderConf(df,session,obj):  
    decoderConf = np.full(obj.nTrials,np.nan)
    i = int(round(np.sum(obj.trialBlock[~obj.autoRewardScheduled]==1)/2))
    c = df['confidence'][sessionInd]
    decoderConf[i:i+c.size] = c
    return decoderConf


badAlignment = []
for sessionInd in range(len(df)):
    print(sessionInd)
    obj = getSessionObj(df,sessionInd)
    if getNonShiftTrials(obj).sum() != df['confidence'][sessionInd].size:
        badAlignment.append(df['session'][sessionInd])


# intra-block resp rate correlations
stimNames = ('vis1','sound1','vis2','sound2','decoder')
autoCorr = [[] for _ in range(5)]
corrWithin = [[[] for _ in range(5)] for _ in range(5)]
corrAcross = copy.deepcopy(corrWithin)
autoCorrMat = np.zeros((5,100))
corrWithinMat = np.zeros((5,5,200))
corrAcrossMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 10

for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    sessionName = df['session'][sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    
    decoderConf = np.full(obj.nTrials,np.nan)
    i = int(np.round(np.sum(obj.trialBlock==1)/2))
    c = df['confidence'][sessionInd]
    decoderConf[i:i+c.size] = c
        
    resp = np.zeros((5,obj.nTrials))
    respShuffled = np.zeros((5,obj.nTrials,nShuffles))
    for blockInd in range(6):
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        for i,s in enumerate(stimNames):
            if i < 4:
                stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                r = obj.trialResponse[stimTrials].astype(float)
                r[r<1] = -1
            else:
                stimTrials = blockTrials
                r = decoderConf[stimTrials]
            resp[i,stimTrials] = r
            for z in range(nShuffles):
                respShuffled[i,stimTrials,z] = np.random.permutation(r)
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        if blockInd in (0,5) or obj.hitRate[blockInd] < 0.8:
            continue
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2','decoder')):
            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0]) if i < 4 else blockTrials
            if len(stimTrials) < 1:
                continue
            r = obj.trialResponse[stimTrials].astype(float) if i < 4 else decoderConf[stimTrials]
            c = np.correlate(r,r,'full')
            norm = np.linalg.norm(r)**2
            cc = []
            for _ in range(nShuffles):
                rs = np.random.permutation(r)
                cs = np.correlate(rs,rs,'full')
                cc.append(c - cs)
                cc[-1] /= norm
            n = c.size // 2
            a = np.full(100,np.nan)
            a[:n] = np.mean(cc,axis=0)[-n:]
            autoCorr[i].append(a)
        
        r = resp[:,blockTrials]
        rs = respShuffled[:,blockTrials]
        if rewStim == 'sound1':
            r = r[[1,0,3,2,4]]
            rs = rs[[1,0,3,2,4]]
        for i,(r1,rs1) in enumerate(zip(r,rs)):
            for j,(r2,rs2) in enumerate(zip(r,rs)):
                if len(r1) < 1 or len(r2) < 1:
                    continue
                c = np.correlate(r1,r2,'full')
                norm = np.linalg.norm(r1) * np.linalg.norm(r2)
                cc = []
                for z in range(nShuffles):
                    cs = np.correlate(rs1[:,z],rs2[:,z],'full')
                    cc.append(c - cs)
                    cc[-1] /= norm
                n = c.size // 2
                a = np.full(200,np.nan)
                a[:n] = np.mean(cc,axis=0)[-n:]
                corrWithin[i][j].append(a)
        
        otherBlocks = [2,4] if blockInd in [2,4] else [1,3]
        otherBlocks.remove(blockInd)
        a = np.full((2,200),np.nan)
        for k,b in enumerate(otherBlocks):
            bTrials = np.where(obj.trialBlock==b+1)[0][startTrial:]
            rOther = resp[:,bTrials]
            rsOther = respShuffled[:,bTrials]
            if rewStim == 'sound1':
                rOther = rOther[[1,0,3,2]]
                rsOther = rsOther[[1,0,3,2]]
            for i,(r1,rs1) in enumerate(zip(rOther,rsOther)):
                for j,(r2,rs2) in enumerate(zip(r,rs)):
                    if len(r1) < 1 or len(r2) < 1:
                        continue
                    c = np.correlate(r1,r2,'full')
                    norm = np.linalg.norm(r1) * np.linalg.norm(r2)
                    cc = []
                    for z in range(nShuffles):
                        cs = np.correlate(rs1[:,z],rs2[:,z],'full')
                        cc.append(c - cs)
                        cc[-1] /= norm
                    n = c.size // 2
                    a = np.full(200,np.nan)
                    a[:n] = np.mean(cc,axis=0)[-n:]
                    corrAcross[i][j].append(a)
                

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)','decoder')

fig = plt.figure(figsize=(4,6))           
gs = matplotlib.gridspec.GridSpec(5,1)
x = np.arange(100) + 1
for i,lbl in enumerate(stimLabels):
    ax = fig.add_subplot(gs[i])
    m = np.nanmean(autoCorr[i],axis=0)
    s = np.nanstd(autoCorr[i],axis=0) / (len(autoCorr[i]) ** 0.5)
    ax.plot(x,m,'k')
    ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,20,5))
    ax.set_xlim([0,15])
    ax.set_ylim([-0.06,0.12])
    if i==4:
        ax.set_xlabel('Lag (trials)')
    if i==0:
        ax.set_ylabel('Auto-correlation')
    ax.set_title(lbl)
plt.tight_layout()

for mat in (corrWithin,corrAcross):
    fig = plt.figure(figsize=(10,8))          
    gs = matplotlib.gridspec.GridSpec(5,5)
    x = np.arange(200) + 1
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels):
            ax = fig.add_subplot(gs[i,j])
            # for y in mat[i,j]:
            #     ax.plot(x,y,'k',alpha=0.2)
            m = np.nanmean(mat[i][j],axis=0)
            s = np.nanstd(mat[i][j],axis=0) / (len(mat[i][j]) ** 0.5)
            ax.plot(x,m,'k')
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,30])
            ax.set_ylim([-0.025,0.075])
            if i==4:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
    plt.tight_layout()


# time dependence of effect of prior reward or response (avg across mice)
prevTrialTypes = ('response to rewarded target','response to non-rewarded target')
conf = []
trialsSince = {prevTrial: [] for prevTrial in prevTrialTypes}
timeSince = copy.deepcopy(trialsSince)
for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    
    sessionName = df['session'][sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    
    
    
    decoderConf = np.full(obj.nTrials,np.nan)
    i = int(np.round(np.sum(obj.trialBlock==1)/2))
    c = df['confidence'][sessionInd]
    decoderConf[i:i+c.size] = c/c.max()
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        if blockInd in (0,5) or obj.hitRate[blockInd] < 0.8:
            continue
        otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
        blockTrials = (obj.trialBlock==blockInd+1) & ~obj.catchTrials & ~obj.autoRewardScheduled
        rewTargetTrials = blockTrials & (obj.trialStim==rewStim)
        nonRewTargetTrials = blockTrials & (obj.trialStim==otherModalTarget)
        stimTrials = np.where(blockTrials)[0][5:]
        if len(stimTrials) < 1:
            continue
        for prevTrialType,trials in zip(prevTrialTypes,(rewTargetTrials,nonRewTargetTrials)):
            respTrials = np.where(trials & obj.trialResponse)[0]
            if len(respTrials) > 0:
                prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
                anyTargetTrials = np.array([np.any(np.in1d(obj.trialStim[p+1:s],(rewStim,otherModalTarget))) for s,p in zip(stimTrials,prevRespTrial)])
                anyQuiescentViolations = np.array([np.any(obj.trialQuiescentViolations[p+1:s]) for s,p in zip(stimTrials,prevRespTrial)])
                notValid = (stimTrials <= respTrials[0]) | (stimTrials > np.where(trials)[0][-1]) | anyTargetTrials #| anyQuiescentViolations
                tr = stimTrials - prevRespTrial
                tr[notValid] = -1
                tm = obj.stimStartTimes[stimTrials] - obj.stimStartTimes[prevRespTrial]
                tm[notValid] = np.nan
                trialsSince[prevTrialType].extend(tr)
                timeSince[prevTrialType].extend(tm)
            else:
                trialsSince[prevTrialType].extend(np.full(len(stimTrials),np.nan))
                timeSince[prevTrialType].extend(np.full(len(stimTrials),np.nan))
        assert(not np.any(np.isnan(decoderConf[stimTrials])))
        conf.extend(decoderConf[stimTrials])
        
for prevTrialType in prevTrialTypes:
    trialsSince[prevTrialType] = np.array(trialsSince[prevTrialType])
    timeSince[prevTrialType] = np.array(timeSince[prevTrialType])
conf = np.array(conf)


trialBins = np.arange(20)
for prevTrialType in prevTrialTypes:
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    n = np.full(trialBins.size,np.nan)
    m = n.copy()
    s = n.copy()
    for i in trialBins:
        j = trialsSince[prevTrialType]==i
        n[i] = j.sum()
        m[i] = np.mean(conf[j])
        s[i] = np.std(conf[j]) / (n[i]**0.5)
    ax.plot(trialBins,m,color='k')
    ax.fill_between(trialBins,m-s,m+s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    # ax.set_xlim([0,6])
    # ax.set_ylim([-0.3,0.3])
    ax.set_xlabel('Trials (non-target) since last '+prevTrialType)
    ax.set_ylabel('Response rate')
    plt.tight_layout()
        
timeBins = np.array([0,5,10,15,20,35,50])
x = timeBins[:-1] + np.diff(timeBins)/2
for phase in ('initial training','after learning'):
    y = {prevTrial: {} for prevTrial in prevTrialTypes}
    for prevTrialType in prevTrialTypes[1:3]:    
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
            n = []
            p = []
            for d,r in zip(timeSince[phase][prevTrialType][stim],resp[phase][stim]):
                n.append(np.full(x.size,np.nan))
                p.append(np.full(x.size,np.nan))
                for i,t in enumerate(timeBins[:-1]):
                    j = (d >= t) & (d < timeBins[i+1])
                    n[-1][i] = j.sum()
                    p[-1][i] = r[j].sum() / n[-1][i]
            m = np.nanmean(p,axis=0)
            s = np.nanstd(p,axis=0) / (len(p)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stim)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            y[prevTrialType][stim] = m
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=16)
        ax.set_xlim([0,47.5])
        ax.set_yticks(np.arange(-0.5,0.5,0.1))
        ax.set_ylim([-0.1,0.2])
        ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=18)
        ax.set_ylabel('Response rate (minus within-block mean)',fontsize=18)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=16)
        plt.tight_layout()

        


