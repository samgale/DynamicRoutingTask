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
import sklearn
from sklearn.linear_model import LogisticRegression
from DynamicRoutingAnalysisUtils import DynRoutData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

miceToIgnore = summaryDf['wheel fixed'] | summaryDf['cannula']

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['stage 5 repeats'] & ~miceToIgnore
miceToUse = tuple(summaryDf[ind]['mouse id'])

nonStandardTrainingMice = (644864,644866,644867,681532,686176)
miceToUse += nonStandardTrainingMice


decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\logreg_2024-11-27_re_concat_1\decoder_confidence_all_trials_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

areaNames = np.unique(df['area'])


def getSessionObj(df,sessionInd):
    sessionName = np.array(df['session'])[sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'DynamicRoutingTask','Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    return obj


def isGoodSession(obj):
    return np.sum((np.array(obj.hitCount) > 10) & (np.array(obj.dprimeOtherModalGo) >= 1)) >= 4 


def getDecoderConf(df,sessionInd,obj):  
    decoderConf= np.array(df['predict_proba'])[sessionInd]
    audRewTrials = obj.rewardedStim == 'sound1'
    decoderConf[audRewTrials] = 1 - decoderConf[audRewTrials] 
    return decoderConf



# intra-block resp rate correlations
areas = ('ORBl','ORBm','ORBvl') + ('ACAd','ACAv') + ('PL',) + ('MOs',) + ('CP','STR') + ('SCig','SCiw','SCdg','MRN')
sessionsByMouse = [[i for i,(s,a,p) in enumerate(zip(df['session'],df['area'],df['probe'])) if int(s[:6])==mouse and a in areas and p in ('','all')] for mouse in miceToUse]
nMiceWithSessions = sum(len(s)>0 for s in sessionsByMouse)
stimNames = ('vis1','sound1','vis2','sound2','decoder')
autoCorr = [[[] for _ in range(nMiceWithSessions)] for _ in range(5)]
corrWithin = [[[[] for _ in range(nMiceWithSessions)] for _ in range(5)] for _ in range(5)]
corrWithinDetrend = copy.deepcopy(corrWithin)
corrAcross = copy.deepcopy(corrWithin)
autoCorrMat = np.zeros((5,nMiceWithSessions,100))
corrWithinMat = np.zeros((5,5,nMiceWithSessions,200))
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
corrAcrossMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 5

m = -1
for sessions in sessionsByMouse:
    if len(sessions) > 0:
        m += 1
        for sessionInd in sessions:
    
            obj = getSessionObj(df,sessionInd)
            
            decoderConf = getDecoderConf(df,sessionInd,obj)
            decoderConf -= 0.5
                
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
                if obj.hitRate[blockInd] < 0.8:
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
                    autoCorr[i][m].append(a)
                
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
                        corrWithin[i][j][m].append(a)
                        
                        x = np.arange(r1.size)
                        rd1,rd2 = [y - np.polyval(np.polyfit(x,y,2),x) for y in (r1,r2)]
                        c = np.correlate(rd1,rd2,'full')
                        norm = np.linalg.norm(rd1) * np.linalg.norm(rd2)
                        c /= norm
                        cc = []
                        for z in range(nShuffles):
                            rsd1,rsd2 = [y - np.polyval(np.polyfit(x,y,2),x) for y in (rs1[:,z],rs2[:,z])]
                            cs = np.correlate(rsd1,rsd2,'full')
                            norm = np.linalg.norm(rsd1) * np.linalg.norm(rsd2)
                            cs /= norm
                            cc.append(c - cs)
                        n = c.size // 2
                        a = np.full(200,np.nan)
                        a[:n] = np.mean(cc,axis=0)[-n:]
                        corrWithinDetrend[i][j][m].append(a)
                
                otherBlocks = [0,2,4] if blockInd in [0,2,4] else [1,3,5]
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
                            corrAcross[i][j][m].append(a)

for i in range(5):
    for m in range(nMiceWithSessions):
        autoCorrMat[i,m] = np.nanmean(autoCorr[i][m],axis=0)
        
for i in range(5):
    for j in range(5):
        for m in range(nMiceWithSessions):
            corrWithinMat[i,j,m] = np.nanmean(corrWithin[i][j][m],axis=0)
            corrWithinDetrendMat[i,j,m] = np.nanmean(corrWithinDetrend[i][j][m],axis=0)
            corrAcrossMat[i,j,m] = np.nanmean(corrAcross[i][j][m],axis=0)
                

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)','decoder')

for mat in (corrWithinMat,corrWithinDetrendMat,corrAcrossMat):
    fig = plt.figure(figsize=(10,8))          
    gs = matplotlib.gridspec.GridSpec(3,3)
    x = np.arange(200) + 1
    for gsi,(i,ylbl) in enumerate(zip((0,1,4),stimLabels[:2] + stimLabels[-1:])):
        for gsj,(j,xlbl) in enumerate(zip((0,1,4),stimLabels[:2] + stimLabels[-1:])):
            ax = fig.add_subplot(gs[gsi,gsj])
            m = np.nanmean(mat[i,j],axis=0)
            s = np.nanstd(mat[i,j],axis=0) / (len(mat[i,j]) ** 0.5)
            ax.plot(x,m,'k')
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,30])
            ax.set_ylim([-0.04,0.08])
            if i==4:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
    plt.tight_layout()
    
fig = plt.figure(figsize=(10,8))          
gs = matplotlib.gridspec.GridSpec(3,3)
x = np.arange(200) + 1
for gsi,(i,ylbl) in enumerate(zip((0,1,4),stimLabels[:2] + stimLabels[-1:])):
    for gsj,(j,xlbl) in enumerate(zip((0,1,4),stimLabels[:2] + stimLabels[-1:])):
        ax = fig.add_subplot(gs[gsi,gsj])
        for mat,clr,lbl in zip((corrWithinMat,corrWithinDetrendMat,corrAcrossMat),'rbk',('within block','within block detrended','across blocks')):
            m = np.nanmean(mat[i,j],axis=0)
            s = np.nanstd(mat[i,j],axis=0) / (len(mat[i,j]) ** 0.5)
            ax.plot(x,m,clr,alpha=0.5,label=lbl)
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=9)
        ax.set_xlim([0,30])
        ax.set_ylim([-0.04,0.08])
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
for sessions in sessionsByMouse:
    if len(sessions) > 0:
        conf.append([])
        for prevTrial in prevTrialTypes:
            trialsSince[prevTrial].append([])
            timeSince[prevTrial].append([])
            
        for sessionInd in sessions:
        
            obj = getSessionObj(df,sessionInd)
            
            decoderConf = getDecoderConf(df,sessionInd,obj)
            
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if obj.hitRate[blockInd] < 0.8:
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
                        trialsSince[prevTrialType][-1].extend(tr)
                        timeSince[prevTrialType][-1].extend(tm)
                    else:
                        trialsSince[prevTrialType][-1].extend(np.full(len(stimTrials),np.nan))
                        timeSince[prevTrialType][-1].extend(np.full(len(stimTrials),np.nan))
                conf[-1].extend(decoderConf[stimTrials])
            
        for prevTrialType in prevTrialTypes:
            trialsSince[prevTrialType][-1] = np.array(trialsSince[prevTrialType][-1])
            timeSince[prevTrialType][-1] = np.array(timeSince[prevTrialType][-1])
        conf[-1] = np.array(conf[-1])

        
timeBins = np.array([0,5,10,15,20,35,50])
x = timeBins[:-1] + np.diff(timeBins)/2
for prevTrialType in prevTrialTypes:    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    n = []
    p = []
    for d,r in zip(timeSince[prevTrialType],conf):
        n.append(np.full(x.size,np.nan))
        p.append(np.full(x.size,np.nan))
        for i,t in enumerate(timeBins[:-1]):
            j = (d >= t) & (d < timeBins[i+1])
            n[-1][i] = j.sum()
            p[-1][i] = r[j].sum() / n[-1][i]
    m = np.nanmean(p,axis=0)
    s = np.nanstd(p,axis=0) / (len(p)**0.5)
    ax.plot(x,m,color='k')
    ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([0,47.5])
    ax.set_ylim([0.59,0.74])
    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=16)
    ax.set_ylabel('Decoder confidence\n(probability of actual context)',fontsize=16)
    plt.tight_layout()



# correlations between areas
areas = ('ORBl','ORBm','ORBvl','ACAd','ACAv','PL','MOs','CP','STR','SCig','SCiw','SCdg','MRN')
labels = ('rewarded target','unrewarded target') + areas
sessions = np.unique([s for s,a in zip(df['session'],df['area']) if int(s[:6]) in miceToUse])
sessionData = {}

corrWithin = [[[] for _ in range(len(labels))] for _ in range(len(labels))]
corrWithinDetrend = copy.deepcopy(corrWithin)
corrWithinMat = np.zeros((len(labels),len(labels),nMiceWithSessions,200))
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 5

for si,session in enumerate(sessions):
    print(si)
    sessionIndices = np.where(df['session']==session)[0]
    if session not in sessionData:
        sessionData[session] = getSessionObj(df,sessionIndices[0])
    obj = sessionData[session]
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        if obj.hitRate[blockInd] < 0.8:
            continue
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        resp = np.zeros((len(labels),len(blockTrials)))
        respShuffled = np.zeros((len(labels),len(blockTrials),nShuffles))
        for i,s in enumerate((('vis1','sound1') if rewStim=='vis1' else ('sound1','vis1')) + areas):
            if i < 2:
                trials = obj.trialStim[blockTrials]==s
                r = obj.trialResponse[blockTrials][trials].astype(float)
                r[r<1] = -1
                resp[i,trials] = r
            else:
                trials = np.ones(len(blockTrials),dtype=bool)
                sessionInd = np.where((df['session']==session) & (df['area']==s) & np.in1d(df['probe'],('','all')))[0]
                if len(sessionInd) > 0:
                    decoderConf = getDecoderConf(df,sessionInd[0],obj)
                    decoderConf -= 0.5
                    r = decoderConf[blockTrials]
                else:
                    r = np.full(trials.size,np.nan)
            resp[i,trials] = r
            for z in range(nShuffles):
                respShuffled[i,trials,z] = np.random.permutation(r)
        
            r = resp
            rs = respShuffled
            if rewStim == 'sound1':
                r[0,1] = r[1,0]
                rs[0,1] = rs[1,0]
            for i,(r1,rs1) in enumerate(zip(r,rs)):
                for j,(r2,rs2) in enumerate(zip(r,rs)):
                    if np.any(np.isnan(r1)) or np.any(np.isnan(r2)):
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
                    
                    x = np.arange(r1.size)
                    rd1,rd2 = [y - np.polyval(np.polyfit(x,y,2),x) for y in (r1,r2)]
                    c = np.correlate(rd1,rd2,'full')
                    norm = np.linalg.norm(rd1) * np.linalg.norm(rd2)
                    c /= norm
                    cc = []
                    for z in range(nShuffles):
                        rsd1,rsd2 = [y - np.polyval(np.polyfit(x,y,2),x) for y in (rs1[:,z],rs2[:,z])]
                        cs = np.correlate(rsd1,rsd2,'full')
                        norm = np.linalg.norm(rsd1) * np.linalg.norm(rsd2)
                        cs /= norm
                        cc.append(c - cs)
                    n = c.size // 2
                    a = np.full(200,np.nan)
                    a[:n] = np.mean(cc,axis=0)[-n:]
                    corrWithinDetrend[i][j].append(a)
        
for i in range(len(labels)):
    for j in range(len(labels)):
        corrWithinMat[i,j] = np.nanmean(corrWithin[i][j],axis=0)
        corrWithinDetrendMat[i,j] = np.nanmean(corrWithinDetrend[i][j],axis=0)
                

for mat in (corrWithinMat,corrWithinDetrendMat):
    fig = plt.figure(figsize=(10,8))          
    gs = matplotlib.gridspec.GridSpec(len(labels),len(labels))
    x = np.arange(200) + 1
    for i,ylbl in enumerate(labels):
        for j,xlbl in enumerate(labels):
            ax = fig.add_subplot(gs[i,j])
            m = np.nanmean(mat[i,j],axis=0)
            s = np.nanstd(mat[i,j],axis=0) / (len(mat[i,j]) ** 0.5)
            ax.plot(x,m,'k')
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0,30])
            ax.set_ylim([-0.04,0.08])
            if i==len(labels)-1:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
    plt.tight_layout()

for mat in (corrWithinMat,corrWithinDetrendMat):
    fig = plt.figure(figsize=(10,8))   
    ax = fig.add_subplot(1,1,1)       
    c = np.nanmean(mat[:,:,:,0],axis=-1)
    cmax = np.max(np.absolute(c))
    im = ax.imshow(c,cmap='bwr',clim=(-cmax,cmax))
    cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
    # cb.set_ticks(np.arange(nClust)+1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.tight_layout()    


# correlations between areas (old)
areas = [a for a in areaNames if a[0].isupper()]
areaIds = np.array(df['area'])
sessionIds = np.array(df['session'])
isCombinedProbes = np.in1d(np.array(df['probe']),('','all'))
sessionData = {}

areaCorrMat = np.zeros((len(areas),)*2)
areaCorrN = areaCorrMat.copy()
for i,area1 in enumerate(areas):
    print(i)
    sessions = np.where((areaIds == area1) & isCombinedProbes)[0]
    for si1 in sessions:
        sessionName = sessionIds[si1]
        if int(sessionName[:6]) in miceToUse:
            if sessionName not in sessionData:
                sessionData[sessionName] = getSessionObj(df,si1)
            obj = sessionData[sessionName]
            p1 = getDecoderConf(df,si1,obj) - 0.5
            for j,area2 in enumerate(areas):
                si2 = np.where((areaIds == area2) & (sessionIds == sessionIds[si1]) & isCombinedProbes)[0]
                if len(si2) > 0:
                    p2 = getDecoderConf(df,si2[0],obj) - 0.5
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if obj.hitRate[blockInd] < 0.8:
                            continue
                        trials = np.where(obj.trialBlock==blockInd+1)[0][5:]              
                        c = np.correlate(p1[trials],p2[trials],'full')
                        norm = np.linalg.norm(p1[trials]) * np.linalg.norm(p2[trials])
                        cc = []
                        for z in range(10):
                            cs = np.correlate(p1[trials],np.random.permutation(p2[trials]),'full')
                            cc.append(c - cs)
                            cc[-1] /= norm
                        n = c.size // 2
                        areaCorrMat[i,j] += np.mean(cc,axis=0)[-n]
                        areaCorrN[i,j] += 1

a = areaCorrMat / areaCorrN

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmap = matplotlib.cm.viridis.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(a,cmap=cmap)
cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
cb.set_ticks(np.arange(nClust)+1)
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'w')
ax.set_xticks(np.arange(10,70,10)-1)
ax.set_xticklabels(np.arange(10,70,10))
ax.set_yticks([])
ax.set_xlabel('Session')
ax.set_ylabel('Mouse')
ax.set_title('Most frequent cluster in session\n(white line = passed learning criteria)')
plt.tight_layout()
    
    
            

