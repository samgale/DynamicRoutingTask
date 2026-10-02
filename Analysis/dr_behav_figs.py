# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:55:21 2026

@author: samg
"""

import copy
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics
import sklearn.cluster
from DynamicRoutingAnalysisUtils import (getPerformanceStats,getIsStandardRegimen,getStage5Sessions,getSessionsToPass,getSessionData,
                                         calcDprime,fitCurve,calcWeibullDistrib,getBlockTrials,getResponseCorrelations)


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

isStandardRegimen = getIsStandardRegimen(summaryDf)

deltaLickProbLabels = ('5 rewarded targets',
                       '5 non-rewarded targets',
                       '1 rewarded target',
                       '1 non-rewarded target',
                       '5 rewards',
                       '5 catch trials')
deltaLickProb = {lbl: {targ: np.nan for targ in ('rewTarg','nonRewTarg')} for lbl in deltaLickProbLabels}



## drop out summary
isEarlyTermination = summaryDf['reason for early termination'].notnull()
reasonForEarlyTerm = np.unique(summaryDf[isEarlyTermination & isStandardRegimen]['reason for early termination'])
earlyEphys = (summaryDf['reason for early termination']=='stage 5 early ephys')

stage5Reasons = [reason for reason in reasonForEarlyTerm if 'stage 5' in reason]
stage5ReasonClrs = plt.cm.tab20(np.linspace(0,1,len(stage5Reasons)))

trainingStartDate = []
for mid in summaryDf['mouse id']:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    trainingStartDate.append(df['start time'].iloc[0])
trainingStartYear = np.array([t.year for t in trainingStartDate])

for isNsb,lbl in zip((summaryDf['trainer']!='NSB',summaryDf['trainer']=='NSB',np.ones(summaryDf.shape[0],dtype=bool)),('dr trainers','nsb trainers','all trainers')):
    print(lbl)
    
    include = isNsb # & np.isin(trainingStartYear,years) #& ~(summaryDf['whc'] | summaryDf['dhc'])
    stage1Mice = isStandardRegimen & include & (summaryDf['stage 1 pass'] | isEarlyTermination)
    print(np.sum(stage1Mice & summaryDf['stage 1 pass']),'of',np.sum(stage1Mice),'passed stage 1')
    reasonForTerm = summaryDf[stage1Mice & ~summaryDf['stage 1 pass']]['reason for early termination']
     
    stage2Mice = stage1Mice & summaryDf['stage 1 pass']
    print(np.sum(stage2Mice & summaryDf['stage 2 pass']),'of',np.sum(stage2Mice),'passed stage 2')
    reasonForTerm = summaryDf[stage2Mice & ~summaryDf['stage 2 pass']]['reason for early termination']

    stage5Mice = stage2Mice & summaryDf['stage 2 pass'] & ~earlyEphys
    nPass = np.sum(stage5Mice & summaryDf['stage 5 pass'])
    print(nPass,'of',np.sum(stage5Mice),'passed stage 3')
    reasonForTerm = summaryDf[stage5Mice & ~summaryDf['stage 5 pass']]['reason for early termination']
    lbls,clrs,counts = zip(*((reason,clr,np.sum(reasonForTerm==reason)) for reason,clr in zip(stage5Reasons,stage5ReasonClrs) if reason in np.unique(reasonForTerm)))
    lbls += ('pass',)
    counts += (nPass,)
    clrs += ('0.5',)
    lbls = [lbl+' ('+str(n)+')' for lbl,n in zip(lbls,counts)]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.pie(counts,labels=lbls,colors=clrs,autopct='%1.1f%%')
    print('\n')


## stage 1 and 2 learning
stage = 1

mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage '+ str(stage) + ' pass']]['mouse id'])
sessionsToPass = []
sessionData = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.where(np.array(['stage ' + str(stage) in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
    sessionsToPass.append(getSessionsToPass(mouseId,df,sessions,stage=stage))
    sessionData.append([getSessionData(mouseId,startTime,engagedThresh=10,lightLoad=True) for startTime in df.loc[sessions,'start time']])

hitCount,dprime = [[[getattr(obj,attr)[0] for obj in exps] for exps in sessionData] for attr in ('hitCount','dprimeSameModal')]

hitThresh = 100
dprimeThresh = 1.5

xlim = [0.5,max(sessionsToPass)+0.5]
for d,thresh,ylim,ylbl in zip((hitCount,dprime),(hitThresh,dprimeThresh),([0,260],[-1,6]),('Rewards earned','d\'')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xlim,[thresh]*2,'k--')
    for y,s in zip(d,sessionsToPass):
        ax.plot(np.arange(s)+1,y[:s],'k',alpha=0.2)
        ax.plot(s,y[s-1],'o',ms=12,color='k',alpha=0.2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Session',fontsize=16)
    ax.set_ylabel(ylbl,fontsize=16)
    plt.tight_layout()

    
## moving vs stationary grating
isStat = summaryDf['stat grating'] & ~(summaryDf['wheel fixed'] | summaryDf['cannula']) & summaryDf['stage 1 pass']
mice = {'moving gratings, timeouts':  np.array(summaryDf[isStandardRegimen & summaryDf['stage 1 pass']]['mouse id']),
        'stationary gratings, timeouts': np.array(summaryDf[isStat & summaryDf['timeouts']]['mouse id']),
        'stationary gratings, no timeouts': np.array(summaryDf[isStat & ~summaryDf['timeouts']]['mouse id'])}

sessionsToPass = {key: [] for key in mice}
for key in mice:
    for mouseId in mice[key]:
        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
        sessions = np.where(np.array(['stage 1' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
        sessionsToPass[key].append(getSessionsToPass(mouseId,df,sessions,stage=1))

s = int(1e5)
n = len(mice['stationary gratings, timeouts'])
m = np.median(sessionsToPass['stationary gratings, timeouts'])
pMoving = np.sum([np.median(np.random.choice(sessionsToPass['moving gratings, timeouts'],n,replace=True)) > m for _ in range(s)]) / s

n = len(mice['stationary gratings, no timeouts'])
m = np.median(sessionsToPass['stationary gratings, no timeouts'])
pTimeouts = np.sum([np.median(np.random.choice(sessionsToPass['stationary gratings, timeouts'],n,replace=True)) > m for _ in range(s)]) / s
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr,ls in zip(mice.keys(),'gmm',('-','-','--')):
    dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    lbl += ' (n='+str(dsort.size)+')'
    ax.plot(dsort,cumProb,color=clr,ls=ls,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Sessions to pass',fontsize=16)
ax.set_ylabel('Cumulative fraction of mice',fontsize=16)
plt.legend(loc='lower right',fontsize=10)
plt.tight_layout()   


preSessions = 1
postSessions = 1
dprime = []
for mid in summaryDf[summaryDf['moving to stat']]['mouse id']:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    prevTask = None
    dprime.append([])
    for i,task in enumerate(df['task version']):
        if prevTask is not None and 'stage 5' in prevTask and 'stage 5' in task and 'moving' in prevTask and 'moving' not in task:
            for j in range(i-preSessions,i+postSessions+1):
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[j])
                if 'ori tone' in df.loc[j,'task version'] or 'ori AMN' in df.loc[j,'task version']:
                    dprime[-1].append(np.mean(dprimeSame[0][0:2:6]))
                else:
                    dprime[-1].append(np.mean(dprimeSame[0][1:2:6]))
            break
        prevTask = task

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(-preSessions,postSessions+1)
for dp in dprime:
    ax.plot(xticks,dp,'k',alpha=0.25)
mean = np.mean(dprime,axis=0)
sem = np.std(dprime,axis=0)/(len(dprime)**0.5)
ax.plot(xticks,mean,'ko-',lw=2,ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(['-1\nmoving','0\nstationary','1\nmoving'])
ax.set_xlim([-preSessions-0.5,postSessions+0.5])
ax.set_yticks(np.arange(5))
ax.set_ylim([0,4.1])
ax.set_xlabel('Session',fontsize=14)
ax.set_ylabel('d\'',fontsize=14)
plt.tight_layout()


##
ind = summaryDf['stage 1 pass'] & summaryDf['stat grating'] & ~(summaryDf['wheel fixed'] | summaryDf['cannula'])
mice = {'timeouts': np.array(summaryDf[ind & summaryDf['timeouts']]['mouse id']),
        'no timeouts': np.array(summaryDf[ind & ~summaryDf['timeouts']]['mouse id'])}

sessionsToPass = {key: [] for key in mice}
for key in mice:
    for mouseId in mice[key]:
        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
        sessions = np.where(np.array(['stage 1' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
        sessionsToPass[key].append(getSessionsToPass(mouseId,df,sessions,stage=1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(mice.keys(),'gm'):
    dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    lbl += ' (n='+str(dsort.size)+')'
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Sessions to pass',fontsize=16)
ax.set_ylabel('Cumulative fraction of mice',fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()   


## stage 5 learning
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])
sessionsToPass = []
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = getStage5Sessions(mid,df)
    sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=5))
    sessionData.append([getSessionData(mid,startTime,engagedThresh=10,lightLoad=True) for startTime in df.loc[sessions,'start time']])

nSessionsAfterPass = [len(sd) - sp for sd,sp in zip(sessionData,sessionsToPass)]

dprime = {comp: {mod: [[] for _ in range(len(mice))] for mod in ('all','vis','sound')} for comp in ('same','other')}
for i,exps in enumerate(sessionData):
    for obj in exps:
        for dp,comp in zip((obj.dprimeSameModal,obj.dprimeOtherModalGo),('same','other')):
            dprime[comp]['all'][i].append(dp)
            if obj.blockStimRewarded[0] == 'vis1':
                dprime[comp]['vis'][i].append(dp[0:6:2])
                dprime[comp]['sound'][i].append(dp[1:6:2])
            else:
                dprime[comp]['sound'][i].append(dp[0:6:2])
                dprime[comp]['vis'][i].append(dp[1:6:2])
                
for exps in sessionData:
    for obj in exps:
        obj.engagedThresh = None
        obj.calcPerformanceStats()


## intra-block resp correlations
nSessions = 2
trainingPhases = ('initial training','early learning','late learning','after learning')
trainingPhaseColors = 'rmbg'
blockRewStim = ('all',) #('vis1','sound1')
blockEpochs = ('full',) #('first half','last half')
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {phase: {blockRew: {epoch: np.zeros((4,len(sessionData),100)) for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
autoCorrRawMat = copy.deepcopy(autoCorrMat)
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
respRateMat = {phase: {blockRew: {epoch: np.zeros((4,len(sessionData))) for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
corrWithinMat = {phase: {blockRew: {epoch: np.zeros((4,4,len(sessionData),200)) for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
corrWithinRawMat = copy.deepcopy(corrWithinMat)
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
# corrAcrossMat = copy.deepcopy(corrWithinMat)
for phase in trainingPhases:
    for blockRew in blockRewStim:
        for epoch in blockEpochs:
            for m,(exps,sp,lo) in enumerate(zip(sessionData,sessionsToPass,learnOnset)):
                if phase == 'initial training':
                    exps = exps[:nSessions]
                elif phase == 'early learning':
                    exps = exps[lo+1:lo+3]
                elif phase == 'late learning':
                    exps = exps[sp-4:sp-2]
                elif phase == 'criterion sessions':
                    exps = exps[sp-2:sp]
                elif phase == 'after learning':
                    exps = exps[sp:sp+nSessions]
                
                respRate = []
                autoCorr = []
                autoCorrRaw = []
                autoCorrDetrend = []
                corrWithin = []
                corrWithinRaw = []
                corrWithinDetrend = []
                corrAcross = []
                    
                for obj in exps:    
                    rr,ac,acr,acd,cw,cwr,cwd,ca= getResponseCorrelations(obj,blockRew=blockRew,blockEpoch=epoch)
                    respRate.append(rr)
                    autoCorr.append(ac)
                    autoCorrRaw.append(acr)
                    autoCorrDetrend.append(acd)
                    corrWithin.append(cw)
                    corrWithinRaw.append(cwr)
                    corrWithinDetrend.append(cwd)
                    corrAcross.append(ca)
                      
                autoCorrMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorr,axis=(0,2))
                autoCorrRawMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorrRaw,axis=(0,2))
                autoCorrDetrendMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorrDetrend,axis=(0,2))
                respRateMat[phase][blockRew][epoch][:,m] = np.nanmean(respRate,axis=(0,2))
                    
                corrWithinMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithin,axis=(0,3))
                corrWithinRawMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithinRaw,axis=(0,3))
                corrWithinDetrendMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithinDetrend,axis=(0,3))
                # corrAcrossMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrAcross,axis=(0,3))

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

for d in (autoCorrMat,autoCorrDetrendMat):
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,lbl in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        for phase,clr in zip(trainingPhases,trainingPhaseColors):
            mat = d[phase]['all']['full'][i,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,color=clr)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.06,0.2])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
        if i==0:
            ax.set_ylabel('Autocorrelation',fontsize=12)
        ax.set_title(lbl,fontsize=12)
    plt.tight_layout()
    
for i,stim in enumerate(stimLabels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'k--')
    for phase,clr in zip(trainingPhases,trainingPhaseColors):
        d = autoCorrDetrendMat[phase]['all']['full'][i,:,1]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=phase)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-0.1,0.25])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Autocorrelation of responses',fontsize=14)
    ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
    ax.set_title(stim.replace('\n',' '),fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout() 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bw = 0.2
for phase,clr in zip(trainingPhases,trainingPhaseColors):
    r = np.concatenate(respRateMat[phase]['all']['full'])
    c = np.concatenate(autoCorrDetrendMat[phase]['all']['full'][:,:,1])
    bins = np.arange(bw/2,1,bw)
    for i,b in enumerate(bins):
        low = 0 if b==bins[0] else b-bw/2
        high = 1 if b==bins[-1] else b+bw/2
        d = c[(r>low) & (r<=high)]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(b,m,'o',mec=clr,mfc='none')
        ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if i==0 else None))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
# ax.set_ylim([0,0.04])
ax.set_xlabel('Response rate',fontsize=14)
ax.set_ylabel('Autocorrelation',fontsize=14)
plt.legend()
plt.tight_layout() 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
i = 1
bw = 0.25
n = []
for phase,clr in zip(trainingPhases,trainingPhaseColors):
    r = respRateMat[phase]['all']['full'][i]
    c = autoCorrDetrendMat[phase]['all']['full'][i,:,1]
    bins = np.arange(bw/2,1,bw)
    n.append([])
    for b in bins:
        low = 0 if b==bins[0] else b-bw/2
        high = 1 if b==bins[-1] else b+bw/2
        d = c[(r>low) & (r<=high)]
        n[-1].append(len(d))
        if len(d)>2:
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            ax.plot(b,m,'o',mec=clr,mfc='none')
            ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if b==bins[-1] else None))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,1])
# ax.set_ylim([-0.004,0.04])
ax.set_xlabel('Response rate',fontsize=14)
ax.set_ylabel('Correlation',fontsize=14)
plt.legend(loc='lower left')
plt.tight_layout()


for d,ylim in zip((corrWithinRawMat,corrWithinMat,corrWithinDetrendMat),([-0.2,0.2],[-0.03,0.1],[-0.03,0.1])):
    fig = plt.figure(figsize=(10,10))          
    gs = matplotlib.gridspec.GridSpec(4,4)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:4]):
            ax = fig.add_subplot(gs[i,j])
            for phase,clr in zip(trainingPhases,'mg'):
                mat = d[phase]['all']['full'][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=phase)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim(ylim)
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
                
fig = plt.figure(figsize=(12,10))          
gs = matplotlib.gridspec.GridSpec(4,4)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:4]):
        ax = fig.add_subplot(gs[i,j])
        for d,clr,lbl in zip((corrWithinMat,corrWithinDetrendMat),'mg',('raw','detrended')):
            mat = d[phase]['all']['full'][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.025,0.09])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=14)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=14)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=14)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()

fig = plt.figure(figsize=(12,10))          
gs = matplotlib.gridspec.GridSpec(4,4)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:4]):
        ax = fig.add_subplot(gs[i,j])
        for phase,clr in zip(trainingPhases,trainingPhaseColors):
            mat = corrWithinDetrendMat[phase]['all']['full'][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=phase)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.025,0.09])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=14)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=14)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=14)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()

for phase in trainingPhases:
    fig = plt.figure(figsize=(8,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for blockRew,clr in zip(blockRewStim[:2],'gm'):
                mat = corrWithinDetrendMat[phase][blockRew]['full'][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=('visual' if blockRew=='vis1' else 'auditory')+' rewarded')
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.045,0.125] if phase=='initial training' else [-0.025,0.045])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            if i==0 and j==1:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()

for phase in trainingPhases:       
    fig = plt.figure(figsize=(8,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for epoch,clr in zip(('first half','last half'),'gm'):
                mat = corrWithinDetrendMat[phase]['all'][epoch][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=epoch)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.03,0.1] if phase=='initial training' else [-0.02,0.03])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            if i==0 and j==1:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()

for i,stim in enumerate(stimLabels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'k--')
    for phase,clr in zip(trainingPhases,trainingPhaseColors):
        d = corrWithinDetrendMat[phase]['all']['full'][i,i,:,1]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=phase)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-0.05,0.08])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Autocorrelation of responses',fontsize=14)
    ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
    ax.set_title(stim.replace('\n',' '),fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout() 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bw = 0.2
for phase,clr in zip(trainingPhases,trainingPhaseColors):
    r = np.concatenate(respRateMat[phase]['all']['full'])
    c = np.concatenate([corrWithinDetrendMat[phase]['all']['full'][i,i,:,1] for i in range(4)])
    bins = np.arange(bw/2,1,bw)
    for i,b in enumerate(bins):
        low = 0 if b==bins[0] else b-bw/2
        high = 1 if b==bins[-1] else b+bw/2
        d = c[(r>low) & (r<=high)]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(b,m,'o',mec=clr,mfc='none')
        ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if i==0 else None))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
# ax.set_ylim([0,0.04])
ax.set_xlabel('Response rate',fontsize=14)
ax.set_ylabel('Autocorrelation',fontsize=14)
plt.legend()
plt.tight_layout() 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
i = 1
bw = 0.25
n = []
for phase,clr in zip(trainingPhases,trainingPhaseColors):
    r = respRateMat[phase]['all']['full'][i]
    c = corrWithinDetrendMat[phase]['all']['full'][i,i,:,1]
    bins = np.arange(bw/2,1,bw)
    n.append([])
    for b in bins:
        low = 0 if b==bins[0] else b-bw/2
        high = 1 if b==bins[-1] else b+bw/2
        d = c[(r>low) & (r<=high)]
        n[-1].append(len(d))
        if len(d)>2:
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            ax.plot(b,m,'o',mec=clr,mfc='none')
            ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if b==bins[-1] else None))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,1])
ax.set_ylim([-0.01,0.05])
ax.set_xlabel('Response rate',fontsize=14)
ax.set_ylabel('Correlation',fontsize=14)
# plt.legend(loc='lower left')
plt.tight_layout()





## session clusters
sessionClustData = {key: [] for key in ('nSessions','mouseId','sessionStartTime','mouse','session','passed','block','firstRewardStim','hitRate','falseAlarmRate','dprime','clustData')}
for m,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
    for i,obj in enumerate(exps):
        sessionClustData['nSessions'].append(len(exps))
        sessionClustData['mouseId'].append(obj.subjectName)
        sessionClustData['sessionStartTime'].append(obj.startTime)
        sessionClustData['mouse'].append(m)
        sessionClustData['session'].append(i)
        sessionClustData['passed'].append(i > s-1)
        sessionClustData['firstRewardStim'].append(obj.blockStimRewarded[0])
        sessionClustData['hitRate'].append(obj.hitRate)
        sessionClustData['falseAlarmRate'].append(obj.falseAlarmOtherModalGo)
        sessionClustData['dprime'].append(obj.dprimeOtherModalGo)
        sessionClustData['clustData'].append(np.concatenate((obj.hitRate,obj.falseAlarmOtherModalGo)))

for key in sessionClustData:
    sessionClustData[key] = np.array(sessionClustData[key])

clustData = sessionClustData['clustData']
clustData[np.isnan(clustData)] = 0

nMice = len(sessionData)
nClust = 6
spectralClustering = sklearn.cluster.SpectralClustering(n_clusters=nClust,affinity='nearest_neighbors',n_neighbors=10,assign_labels='kmeans')
clustId = spectralClustering.fit_predict(clustData)
clustId += 1
clustLabels = np.unique(clustId)

newClustOrder = [5,6,4,1,3,2]
newClustId = clustId.copy()
for i,c in enumerate(newClustOrder):
    newClustId[clustId==c] = i+1
clustId = newClustId

sessionClustData['clustId'] = clustId            
#np.save(os.path.join(baseDir,'Sam','sessionClustData.npy'),sessionClustData)

x = np.arange(6)+1
for clust in clustLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    i = clustId==clust
    hr = sessionClustData['hitRate'][i]
    fr = sessionClustData['falseAlarmRate'][i]
    for clr,lbl in zip(('k','0.5'),('odd block rewarded target','even block rewarded target')):
        r = np.zeros((i.sum(),6))
        if clr=='k':
            r[:,::2] = hr[:,::2]
            r[:,1::2] = fr[:,1::2]
        else:
            r[:,::2] = fr[:,::2]
            r[:,1::2] = hr[:,1::2]
        m = np.nanmean(r,axis=0)
        s = np.nanstd(r)/(len(r)**0.5)
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xticks(x)
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0.5,6.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Block #',fontsize=18)
    ax.set_ylabel('Response rate',fontsize=18)
    ax.legend(loc='lower right',fontsize=16)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for clust in clustLabels:
    n = np.sum(clustId==clust)
    ax.bar(clust,n,width=0.8,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',labelsize=16)
ax.set_xticks(clustLabels)
ax.set_xticklabels(clustLabels)
ax.set_xlabel('Cluster',fontsize=18)
ax.set_ylabel('Number of sessions',fontsize=18)
plt.tight_layout()















