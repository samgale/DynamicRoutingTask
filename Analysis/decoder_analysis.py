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

# SVC
decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\2024-10-30\decoder_confidence_versus_trials_since_rewarded_target_all_units.pkl"

# Logistic regression
decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\logreg_test_2024-11-13\decoder_confidence_versus_trials_since_rewarded_target_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

areaNames = np.unique(df['area'])

areas = ('ORBl','ORBm','ORBvl') + ('ACAd','ACAv') + ('PL',) + ('MOs',) + ('CP','STR') + ('SCig','SCiw','SCdg','MRN')

sessionsByMouse = [[i for i,(s,a) in enumerate(zip(df['session'],df['area'])) if int(s[:6])==mouse and a in areas] for mouse in miceToUse]
nMiceWithSessions = sum(len(s)>0 for s in sessionsByMouse)



def getSessionObj(df,sessionInd):
    sessionName = np.array(df['session'])[sessionInd]
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
    decoderConf[np.array(df['trial_index'])[sessionInd]] = np.array(df['confidence'])[sessionInd]
    # decoderConf[getNonShiftTrials(obj)] = df['confidence'][sessionInd]
    # c = df['confidence'][sessionInd]
    # trials = np.where(getNonShiftTrials(obj))[0]
    # if len(trials) <= c.size:
    #     decoderConf[trials] = c[:len(trials)]
    # elif len(trials) > c.size:
    #     decoderConf[trials[:c.size]] = c
    return decoderConf


# badAlign = []
# for sessionInd in range(len(df)):
#     print(sessionInd)
#     obj = getSessionObj(df,sessionInd)
#     trials = getNonShiftTrials(obj)
#     conf = df['confidence'][sessionInd]
#     if trials.sum() != conf.size:
#         badAlign.append((sessionInd,trials.sum()-conf.size))


# intra-block resp rate correlations
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


fig = plt.figure(figsize=(8,8))          
gs = matplotlib.gridspec.GridSpec(4,2)
x = np.arange(200) + 1
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:2]):
        ax = fig.add_subplot(gs[i,j])
        for mat,clr,lbl in zip((corrWithinMat,corrWithinDetrendMat,corrAcrossMat),'rbk',('within block','within block detrended','across blocks')):
            m = np.nanmean(mat[i,j],axis=0)
            s = np.nanstd(mat[i,j],axis=0) / (len(mat[i,j]) ** 0.5)
            ax.plot(x,m,clr,alpha=0.5,label=lbl)
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=9)
        ax.set_xlim([0,30])
        ax.set_ylim([-0.025,0.075])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=11)
        if j==0:
            ax.set_ylabel(ylbl,fontsize=11)
        if i==0:
            ax.set_title(xlbl,fontsize=11)
        if i==0 and j==1:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
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
    ax.set_ylim([0.6,1.8])
    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=16)
    ax.set_ylabel('Decoder confidence',fontsize=16)
    plt.tight_layout()

        
# regression model
accuracy = []
for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    
    obj = getSessionObj(df,sessionInd)
    
    decoderConf = getDecoderConf(df,sessionInd,obj)
    
    trials = np.in1d(obj.trialBlock,(2,3,4,5)) & np.in1d(obj.trialStim,obj.blockStimRewarded) & (obj.trialStim != obj.rewardedStim)
    
    # timeSinceReward = np.zeros(trials.sum())
    # for i,trial in enumerate(np.where(trials)[0]):
    #     lastReward = np.where(obj.trialRewarded[:trial])[0]
    #     lastReward = lastReward[-1] if len(lastReward) > 0 else 0
    #     timeSinceReward[i] = obj.stimStartTimes[trial] - obj.stimStartTimes[lastReward]
    
    # X = np.stack((decoderConf[trials],timeSinceReward),axis=1)
    # X -= np.mean(X,axis=0)
    # X /= np.std(X,axis=0)
    X = decoderConf[trials][:,None]
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
    
    predict = np.zeros(y.size,dtype=bool)
    for trial in range(y.size):
        model = LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced')
        trainInd = np.delete(np.arange(y.size),trial)
        model.fit(X[trainInd],y[trainInd])
        predict[trial] = model.predict(X[trial][None,:])[0]
    accuracy.append(sklearn.metrics.balanced_accuracy_score(y,predict))



