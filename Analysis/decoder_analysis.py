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
miceToUse = np.array(summaryDf[ind]['mouse id'])



decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\2024-10-11\decoder_confidence_versus_trials_since_rewarded_target_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

df = df[[int(s[:6]) in miceToUse for s in df['session']]].reset_index()

areaNames = np.unique(df['area'])

areas = ('ORBl','ORBm','ORBvl') + ('ACAd','ACAv') + ('PL',) + ('MOs',) + ('CP','STR') + ('SCig','SCiw','SCdg','MRN')
sessions = np.in1d(df['area'],areas)
nSessions = sessions.sum()


def getSessionObj(df,sessionInd):
    sessionName = df['session'][sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'DynamicRoutingTask','Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    return obj


def isGoodSession(obj):
    return np.sum((np.array(obj.hitCount) > 10) & (np.array(obj.dprimeOtherModalGo) >= 1)) >= 4 


def getNonShiftTrials(obj):
    ind = []
    for block in (1,6):
        blockTrials = np.where((obj.trialBlock==block))[0]
        ind.append(blockTrials[int(np.ceil(len(blockTrials)/2))])
    trials = np.zeros(obj.nTrials,dtype=bool)
    trials[ind[0]:ind[1]] = True
    trials[obj.autoRewardScheduled] = False
    return trials
    

def getDecoderConf(df,sessionInd,obj):  
    decoderConf = np.full(obj.nTrials,np.nan)
    # decoderConf[getNonShiftTrials(obj)] = df['confidence'][sessionInd]
    #
    c = df['confidence'][sessionInd]
    trials = np.where(getNonShiftTrials(obj))[0]
    if len(trials) <= c.size:
        decoderConf[trials] = c[:len(trials)]
    elif len(trials) > c.size:
        decoderConf[trials[:c.size]] = c
    #
    return decoderConf


badAlign = []
for sessionInd in range(len(df)):
    print(sessionInd)
    obj = getSessionObj(df,sessionInd)
    trials = getNonShiftTrials(obj)
    conf = df['confidence'][sessionInd]
    if trials.sum() != conf.size:
        badAlign.append((sessionInd,trials.sum()-conf.size))


# intra-block resp rate correlations
stimNames = ('vis1','sound1','vis2','sound2','decoder')
autoCorr = [[] for _ in range(5)]
corrWithin = [[[] for _ in range(5)] for _ in range(5)]
corrWithinDetrend = copy.deepcopy(corrWithin)
corrAcross = copy.deepcopy(corrWithin)
autoCorrMat = np.zeros((5,100))
corrWithinMat = np.zeros((5,5,200))
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
corrAcrossMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 10

for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    
    obj = getSessionObj(df,sessionInd)
    
    decoderConf = getDecoderConf(df,sessionInd,obj)
        
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

for mat in (corrWithin,corrWithinDetrend,corrAcross):
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
    
    obj = getSessionObj(df,sessionInd)
    
    decoderConf = getDecoderConf(df,sessionInd,obj)
    
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
    fig = plt.figure()
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
    # ax.set_ylim([0,1])
    ax.set_xlabel('Trials (non-target) since last '+prevTrialType)
    ax.set_ylabel('Decoder confidence')
    plt.tight_layout()
        
timeBins = np.array([0,5,10,15,20,35,50])
x = timeBins[:-1] + np.diff(timeBins)/2
for prevTrialType in prevTrialTypes:    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    n = np.full(x.size,np.nan)
    m = n.copy()
    s = n.copy()
    for i,t in enumerate(timeBins[:-1]):
        j = (timeSince[prevTrialType] >= t) & (timeSince[prevTrialType] < timeBins[i+1])
        n[i] = j.sum()
        m[i] = np.mean(conf[j])
        s[i] = np.std(conf[j]) / (n[i]**0.5)
    ax.plot(x,m,color='k')
    ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,47.5])
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Time since last '+prevTrialType+' (s)')
    ax.set_ylabel('Decoder confidence')
    plt.tight_layout()

        
# regression model
accuracy = []
for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    
    obj = getSessionObj(df,sessionInd)
    
    decoderConf = getDecoderConf(df,sessionInd,obj)
    
    trials = np.in1d(obj.trialBlock,(2,3,4,5)) & np.in1d(obj.trialStim,obj.blockStimRewarded) & (obj.trialStim != obj.rewardedStim)
    
    timeSinceReward = np.zeros(trials.sum())
    for i,trial in enumerate(np.where(trials)[0]):
        lastReward = np.where(obj.trialRewarded[:trial])[0]
        lastReward = lastReward[-1] if len(lastReward) > 0 else 0
        timeSinceReward[i] = obj.stimStartTimes[trial] - obj.stimStartTimes[lastReward]
    
    X = np.stack((decoderConf[trials],timeSinceReward),axis=1)
    X -= np.mean(X,axis=0)
    X /= np.std(X,axis=0)
    y = obj.trialResponse[trials]
    
    nSplits = 5
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = y.size
    samplesPerClass = [np.sum(y==val) for val in classVals]

    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    trainInd = []
    testInd = []
    for k in range(nSplits):
        testInd.append([])
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val] 
            testInd[-1].extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
    
    predict = np.full(y.size,np.nan)
    for train,test in zip(trainInd,testInd):
        model = LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced')
        model.fit(X[train],y[train])
        predict[test] = model.predict(X[test])
    accuracy.append(sklearn.metrics.balanced_accuracy_score(y,predict))



