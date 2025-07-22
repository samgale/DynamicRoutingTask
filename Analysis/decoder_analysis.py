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


decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\logreg_many_n_units_medium_criteria_2025-01-08\decoder_confidence_all_trials_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

areaNames = np.unique(df['area'])


def getSessionObj(df,sessionInd):
    sessionName = np.array(df['session'])[sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'DynamicRoutingTask','Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0],lightLoad=True)
    return obj


def isGoodSession(obj):
    return np.sum((np.array(obj.hitCount) > 10) & (np.array(obj.dprimeOtherModalGo) >= 1)) >= 4 


def getDecoderConf(df,sessionInd,obj):  
    decoderConf= df['predict_proba'].iloc[sessionInd].copy()
    audRewTrials = obj.rewardedStim == 'sound1'
    decoderConf[audRewTrials] = 1 - decoderConf[audRewTrials] 
    return decoderConf



# mice with ephys sessions
nHabAndEphysSessions = []
for mouse in miceToUse:
    sheets = drSheets if str(mouse) in drSheets else nsbSheets
    nHabAndEphysSessions.append([sheets[str(mouse)][lbl].sum() for lbl in ('hab','ephys')])
    


# get sessions with good decoding areas
areas = ('FRP','ORBl','ORBm','ORBvl','PL','MOs','ACAd','ACAv','CP','STR','GPe','SNr','SCm','MRN')
sessionsByMouse = [[i for i,(s,a,p) in enumerate(zip(df['session'],df['area'],df['probe'])) if int(s[:6])==mouse and a in areas and p in ('','all')] for mouse in miceToUse]
sessionObjs = [[getSessionObj(df,sessionInd) for sessionInd in sessions] for sessions in sessionsByMouse]
nMiceWithSessions = sum(len(s)>0 for s in sessionsByMouse)


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
        print(m)
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
                    n = (c.size // 2) + 1
                    a = np.full(100,np.nan)
                    a[:n] = np.mean(cc,axis=0)[-n:]
                    autoCorr[i][m].append(a)
                
                r = resp[:,blockTrials]
                mean = r.mean(axis=1)
                r = r - mean[:,None]
                rs = respShuffled[:,blockTrials] - mean[:,None,None]
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
                        n = (c.size // 2) + 1
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
                        n = (c.size // 2) + 1
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
                            n = (c.size // 2) + 1
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
    x = np.arange(200)
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
            ax.set_xlim([-1,30])
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
x = np.arange(200)
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
        ax.set_xlim([-1,30])
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
areas = ('FRP','ORBl','ORBm','ORBvl','PL','ILA','ACAd','ACAv','MOs','MOp','AId','AIp','AIv','RSPd','RSPv','VISp','AUDp',
         'CP','STR','ACB','GPe','SNr','SCs','SCm','MRN','SNc','VTA')
# areas = tuple([area for area in areaNames if area[0].isupper()])
labels = ('rewarded target','unrewarded target') + areas
sessions = np.unique([s for s,a in zip(df['session'],df['area']) if int(s[:6]) in miceToUse and a in areas])
sessionData = {}

corrN = np.zeros((len(labels),)*2)
corrWithin = [[[] for _ in range(len(labels))] for _ in range(len(labels))]
corrWithinDetrend = copy.deepcopy(corrWithin)
corrWithinMat = np.zeros((len(labels),len(labels),200))
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 5

for si,session in enumerate(sessions):
    print(si)
    if session not in sessionData:
        sessionInd = np.where(df['session']==session)[0][0]
        sessionData[session] = getSessionObj(df,sessionInd)
    obj = sessionData[session]
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
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
                corrN[i,j] += 1
                c = np.correlate(r1,r2,'full')
                norm = np.linalg.norm(r1) * np.linalg.norm(r2)
                cc = []
                for z in range(nShuffles):
                    cs = np.correlate(rs1[:,z],rs2[:,z],'full')
                    cc.append(c - cs)
                    cc[-1] /= norm
                n = (c.size // 2) + 1
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
                n = (c.size // 2) + 1
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

for lag in (0,1):
    for mat in (corrWithinMat,corrWithinDetrendMat):
        fig = plt.figure(figsize=(12,12))   
        ax = fig.add_subplot(1,1,1)       
        c = mat[:,:,lag].copy()
        c[corrN<5] = np.nan
        cmax = np.nanmax(np.absolute(c))
        cmap = matplotlib.cm.bwr.copy()
        cmap.set_bad(color=[0.5]*3)
        im = ax.imshow(c,cmap=cmap,clim=(-cmax,cmax))
        cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
        # cb.set_ticks()
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title('Correlation ('+str(lag)+' trial lag)')
        plt.tight_layout()    


# corr coef between areas
corrCoefWithin = [[[] for _ in range(len(areas))] for _ in range(len(areas))]
corrCoefWithinMat = np.zeros((len(areas),len(areas)))
startTrial = 5

for si,session in enumerate(sessions):
    print(si)
    if session not in sessionData:
        sessionInd = np.where(df['session']==session)[0][0]
        sessionData[session] = getSessionObj(df,sessionInd)
    obj = sessionData[session]
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        resp = np.zeros((len(areas),len(blockTrials)))
        for i,area in enumerate(areas):
            sessionInd = np.where((df['session']==session) & (df['area']==area) & np.in1d(df['probe'],('','all')))[0]
            if len(sessionInd) > 0:
                decoderConf = getDecoderConf(df,sessionInd[0],obj)
                decoderConf -= 0.5
                resp[i] = decoderConf[blockTrials]
            else:
                resp[i] = np.nan
        
        for i,r1 in enumerate(resp):
            for j,r2 in enumerate(resp):
                if np.any(np.isnan(r1)) or np.any(np.isnan(r2)):
                    continue
                c = np.corrcoef(r1,r2)[0,1]
                corrCoefWithin[i][j].append(c)

for i in range(len(areas)):
    for j in range(len(areas)):
        corrCoefWithinMat[i,j] = np.nanmean(corrCoefWithin[i][j])
    
fig = plt.figure(figsize=(12,12))   
ax = fig.add_subplot(1,1,1)       
c = corrCoefWithinMat.copy()
c[corrN[2:,2:]<5] = np.nan
c[np.identity(len(areas),dtype=bool)] = np.nan
cmax = np.nanmax(np.absolute(c))
cmap = matplotlib.cm.bwr.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(c,cmap=cmap,clim=(-cmax,cmax))
cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
# cb.set_ticks()
ax.set_xticks(np.arange(len(areas)))
ax.set_yticks(np.arange(len(areas)))
ax.set_xticklabels(areas)
ax.set_yticklabels(areas)
ax.set_title('Decoder confidence correlation')
plt.tight_layout()  


# corr of decoder conf and respones or response times
cc = {stim: {rew: [] for rew in ('rewarded','non-rewarded')} for stim in ('vis1','sound1')}
ccShuf = copy.deepcopy(cc)
for sessions in sessionsByMouse:
    if len(sessions) > 0:
        for i,sessionInd in enumerate(sessions):
    
            obj = getSessionObj(df,sessionInd)
            
            decoderConf = getDecoderConf(df,sessionInd,obj)
            decoderConf -= 0.5
    
            for stim in ('vis1','sound1'):
                for rew in ('rewarded','non-rewarded'):
                    trials = ~obj.autoRewardScheduled & (obj.trialStim==stim) & ((obj.rewardedStim==stim) if rew=='rewarded' else (obj.rewardedStim!=stim))
                    # rt = obj.responseTimes[trials]
                    rt = obj.trialResponse[trials]
                    notNan = ~np.isnan(rt)
                    rt = rt[notNan]
                    dc = decoderConf[trials][notNan]
                    if i==0:
                        cc[stim][rew].append([])
                        ccShuf[stim][rew].append([])
                    cc[stim][rew][-1].append(np.corrcoef(rt,dc)[0,1])
                    ccShuf[stim][rew][-1].append(np.mean([np.corrcoef(rt,np.random.permutation(dc))[0,1] for _ in range(10)]))
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for c,clr,lbl in zip((cc,ccShuf),('k','0.5'),('data','shuffled')):
    x = 0
    for stim in ('vis1','sound1'):
        for rew in ('rewarded','non-rewarded'):
            d = [np.nanmean(r) for r in c[stim][rew]]
            m = np.nanmean(d)
            s = np.nanstd(d) / (len(d)**0.5)
            ax.plot(x,m,'o',mec=clr,mfc='none',label=(lbl if stim=='vis1' and rew=='rewarded' else None))
            ax.plot([x,x],[m-s,m+s],color=clr)
            x += 1
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['vis target\nrewarded','vis target\nnon-rewarded','aud target\nrewarded','aud target\nnon-rewarded'])
ax.set_ylabel('correlation of responses with decoder confidence')
ax.legend()
plt.tight_layout()


# plot decoder conf for all blocks
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials)    
ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
for blockRew,clr,lbl in zip(('vis1','sound1'),'gm',('vis rewarded','aud rewarded')):
    y = []
    for sessions,objs in zip(sessionsByMouse,sessionObjs):
        if len(sessions) > 0:
            y.append([])
            for i,(sessionInd,obj) in enumerate(zip(sessions,objs)):
                decoderConf= df['predict_proba'].iloc[sessionInd].copy()
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and rewStim==blockRew:
                        y[-1].append(np.full(preTrials+postTrials,np.nan))
                        pre = decoderConf[obj.trialBlock==blockInd]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = decoderConf[obj.trialBlock==blockInd+1]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials:preTrials+i] = post[:i]
            y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x[:preTrials],m[:preTrials],color=clr,label=lbl)
    ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
    ax.plot(x[preTrials:],m[preTrials:],color=clr)
    ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=18)
ax.set_xticks([-5,-1,5,9,14,19])
ax.set_xticklabels([-5,-1,1,5,10,15])
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials-0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials after block switch',fontsize=20)
ax.set_ylabel('Decoder conf. (prob. vis rewarded)',fontsize=20)
ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=18)
plt.tight_layout()


# plot decoder conf for block clusters
clustData = np.load(os.path.join(baseDir,'Sam','clustData.npy'),allow_pickle=True).item()

preTrials = 80
postTrials = 80
for clust in np.unique(clustData['clustId']):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(-preTrials,postTrials)    
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for blockRew,clr,lbl in zip(('vis1','sound1'),'gm',('vis rewarded','aud rewarded')):
        y = []
        for sessions,objs in zip(sessionsByMouse,sessionObjs):
            if len(sessions) > 0:
                y.append([])
                for i,(sessionInd,obj) in enumerate(zip(sessions,objs)):
                    if obj.subjectName in clustData['trialCluster'] and obj.startTime in clustData['trialCluster'][obj.subjectName]:
                        trialCluster = clustData['trialCluster'][obj.subjectName][obj.startTime]
                        decoderConf= df['predict_proba'].iloc[sessionInd].copy()
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            blockTrials = obj.trialBlock==blockInd+1
                            if blockInd>0 and rewStim==blockRew and np.all(trialCluster[blockTrials]==clust):
                                y[-1].append(np.full(preTrials+postTrials,np.nan))
                                pre = decoderConf[obj.trialBlock==blockInd]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = decoderConf[blockTrials]
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                if len(y[-1]) > 0:
                    y[-1] = np.nanmean(y[-1],axis=0)
                else:
                    y[-1] = np.full(preTrials+postTrials,np.nan)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x[:preTrials],m[:preTrials],color=clr,label=lbl)
        ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
        ax.plot(x[preTrials:],m[preTrials:],color=clr)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=18)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials after block switch',fontsize=20)
    ax.set_ylabel('Decoder conf. (prob. vis rewarded)',fontsize=20)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=18)
    plt.tight_layout()


for clust in np.unique(clustData['clustId']):
    fig = plt.figure(figsize=(10,8))
    x = np.arange(0,601,15) 
    for a,(blockRew,blockLbl) in enumerate(zip(('vis1','sound1'),('vis rewarded','aud rewarded'))):
        ax = fig.add_subplot(2,1,a+1)
        for stim,clr,stimLbl in zip(('vis1','sound1','decoder'),'gmk',('vis target','aud target','decoder confidence')):
            y = []
            for sessions,objs in zip(sessionsByMouse,sessionObjs):
                if len(sessions) > 0:
                    y.append([])
                    for i,(sessionInd,obj) in enumerate(zip(sessions,objs)):
                        if obj.subjectName in clustData['trialCluster'] and obj.startTime in clustData['trialCluster'][obj.subjectName]:
                            trialCluster = clustData['trialCluster'][obj.subjectName][obj.startTime]
                            decoderConf= df['predict_proba'].iloc[sessionInd].copy()
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                blockTrials = obj.trialBlock==blockInd+1
                                if rewStim==blockRew and np.all(trialCluster[blockTrials]==clust):
                                    trials = blockTrials if stim=='decoder' else blockTrials & (obj.trialStim==stim)
                                    r = decoderConf[trials] if stim=='decoder' else obj.trialResponse[trials]
                                    t = obj.stimStartTimes[trials] - obj.stimStartTimes[trials][0]
                                    y[-1].append(np.interp(x,t,r))
                    if len(y[-1]) > 0:
                        y[-1] = np.nanmean(y[-1],axis=0)
                    else:
                        y[-1] = np.full(x.size,np.nan)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,label=stimLbl)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-6,606])
        ax.set_ylim([0,1.01])
        if a==0:
            ax.set_xticklabels([])
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
        if a==1:
            ax.set_xlabel('Time after block switch (s)',fontsize=16)
        ax.set_title(blockLbl,fontsize=14)
    plt.tight_layout()

           
for clust in np.unique(clustData['clustId']):
    fig = plt.figure()
    x = np.arange(0,601,15) 
    ax = fig.add_subplot(1,1,1)
    for stimLbl,clr in zip(('rewarded target','non-rewarded target','decoder confidence'),'gmk'):
        y = []
        for sessions,objs in zip(sessionsByMouse,sessionObjs):
            if len(sessions) > 0:
                for i,(sessionInd,obj) in enumerate(zip(sessions,objs)):
                    if obj.subjectName in clustData['trialCluster'] and obj.startTime in clustData['trialCluster'][obj.subjectName]:
                        trialCluster = clustData['trialCluster'][obj.subjectName][obj.startTime]
                        decoderConf= df['predict_proba'].iloc[sessionInd].copy()
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            blockTrials = obj.trialBlock==blockInd+1
                            if np.all(trialCluster[blockTrials]==clust):
                                conf = decoderConf if rewStim=='vis1' else 1-decoderConf
                                trials = blockTrials if stimLbl=='decoder confidence' else blockTrials & (obj.trialStim==(rewStim if stimLbl=='rewarded target' else ('sound1' if rewStim=='vis1' else 'vis1')))
                                r = conf[trials] if stimLbl=='decoder confidence' else obj.trialResponse[trials]
                                t = obj.stimStartTimes[trials] - obj.stimStartTimes[trials][0]
                                y.append(np.interp(x,t,r))
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-6,606])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Time after block switch (s)',fontsize=14)
    ax.set_title('Cluster '+str(clust)+', (n='+str(len(y))+')',fontsize=12)
    if clust==1:
        ax.legend(loc='upper right')
    plt.tight_layout()
    
    
    
    
