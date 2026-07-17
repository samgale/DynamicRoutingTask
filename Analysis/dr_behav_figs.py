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
from DynamicRoutingAnalysisUtils import getPerformanceStats,getIsStandardRegimen,getFirstExperimentSession,getSessionsToPass,getSessionData,calcDprime,pca,cluster,fitCurve,calcWeibullDistrib


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

isStandardRegimen = getIsStandardRegimen(summaryDf)

hitThresh = 100
dprimeThresh = 1.5
nInitialTrainingSessions = 4

deltaLickProbLabels = ('5 rewarded targets',
                       '5 non-rewarded targets',
                       '1 rewarded target',
                       '1 non-rewarded target',
                       '5 rewards',
                       '5 catch trials')
deltaLickProb = {lbl: {targ: np.nan for targ in ('rewTarg','nonRewTarg')} for lbl in deltaLickProbLabels}


def getBlockTrials(obj,block,epoch):
    blockTrials = (obj.trialBlock==block) & ~obj.autoRewardScheduled
    n = blockTrials.sum()
    half = int(n/2)
    startTrial = half if epoch=='last half' else 0
    endTrial = half if epoch=='first half' else n
    return np.where(blockTrials)[0][startTrial:endTrial]


def detrend(r,order=2):
    x = np.arange(r.size)
    return r - np.polyval(np.polyfit(x,r,order),x)


def getCorrelation(r1,r2,rs1,rs2,corrSize=200,detrendOrder=None):
    if detrendOrder is not None:
        r1 = detrend(r1,detrendOrder)
        r2 = detrend(r2,detrendOrder)
        rs1 = rs1.copy()
        rs2 = rs2.copy()
        for z in range(rs1.shape[1]):
            rs1[:,z] = detrend(rs1[:,z],detrendOrder)
            rs2[:,z] = detrend(rs2[:,z],detrendOrder)
    c = np.correlate(r1,r2,'full') / (np.linalg.norm(r1) * np.linalg.norm(r2))   
    cs = np.mean([np.correlate(rs1[:,z],rs2[:,z],'full') / (np.linalg.norm(rs1[:,z]) * np.linalg.norm(rs2[:,z])) for z in range(rs1.shape[1])],axis=0)
    n = c.size // 2 + 1
    corrRaw = np.full(corrSize,np.nan)
    corrRaw[:n] = c[-n:]
    corr = np.full(corrSize,np.nan)
    corr[:n] = (c-cs)[-n:] 
    return corr,corrRaw


## drop out summary
isEarlyTermination = summaryDf['reason for early termination'].notnull()
reasonForEarlyTerm = np.unique(summaryDf[isEarlyTermination & isStandardRegimen]['reason for early termination'])

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

    stage5Mice = stage2Mice & summaryDf['stage 2 pass'] & ~(summaryDf['reason for early termination']=='stage 5 early ephys')
    nPass = np.sum(stage5Mice & summaryDf['stage 5 pass'])
    print(nPass,'of',np.sum(stage5Mice),'passed stage 3')
    reasonForTerm = summaryDf[stage5Mice & ~summaryDf['stage 5 pass']]['reason for early termination']
    lbls,clrs,counts = zip(*((reason[8:],clr,np.sum(reasonForTerm==reason)) for reason,clr in zip(stage5Reasons,stage5ReasonClrs) if reason in np.unique(reasonForTerm)))
    lbls += ('pass',)
    counts += (nPass,)
    clrs += ('0.5',)
    lbls = [lbl+' ('+str(n)+')' for lbl,n in zip(lbls,counts)]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.pie(counts,labels=lbls,colors=clrs,autopct='%1.1f%%')
    print('\n')


## stage 1 and 2 learning
stage = 2

mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage '+ str(stage) + ' pass']]['mouse id'])
sessionsToPass = []
sessionData = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.where(np.array(['stage ' + str(stage) in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
    sessionsToPass.append(getSessionsToPass(mouseId,df,sessions,stage=stage))
    sessionData.append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions,'start time']])
 
prevTrialType = ('all','hit','miss','fa','cr')
nRewards = {prev: [[] for _ in range(len(mice))] for prev in prevTrialType}
dprime = copy.deepcopy(nRewards)
hitRate = copy.deepcopy(nRewards)
falseAlarmRate = copy.deepcopy(nRewards)
catchRate = copy.deepcopy(nRewards)
quiescentViolationsPerTrial = copy.deepcopy(nRewards)
hitRespTime = copy.deepcopy(nRewards)
falseAlarmRespTime = copy.deepcopy(nRewards)
for prev in prevTrialType:
    for i,sessions in enumerate(sessionData):
        for obj in sessions:
            if prev == 'all':
                trials = ~obj.autoRewardScheduled
            elif prev == 'hit':
                trials = np.concatenate(([True],obj.hitTrials[:-1]))
            elif prev== 'miss':
                trials = np.concatenate(([True],obj.missTrials[:-1]))
            elif prev == 'fa':
                trials = np.concatenate(([True],obj.falseAlarmTrials[:-1]))
            elif prev == 'cr':
                trials = np.concatenate(([True],~obj.correctRejectTrials[:-1]))
            nGoTrials = np.sum(trials & obj.goTrials)
            nNogoTrials = np.sum(trials & obj.nogoTrials)
            if nGoTrials > 0 and nNogoTrials > 0:
                hr = np.sum(obj.trialResponse[trials & obj.goTrials]) / nGoTrials
                far = np.sum(obj.trialResponse[trials & obj.nogoTrials]) / nNogoTrials
                nRewards[prev][i].append(obj.trialRewarded[trials].sum())
                dprime[prev][i].append(calcDprime(hr,far,nGoTrials,nNogoTrials))
                hitRate[prev][i].append(hr)
                falseAlarmRate[prev][i].append(far)
                catchRate[prev][i].append(np.sum(obj.trialResponse[trials & obj.catchTrials]) / np.sum(trials & obj.catchTrials))
                quiescentViolationsPerTrial[prev][i].append(np.sum(np.array(obj.trialQuiescentViolations)[trials]) / np.sum(trials))
                hitRespTime[prev][i].append(np.nanmean(obj.responseTimes[trials & obj.goTrials]))
                falseAlarmRespTime[prev][i].append(np.nanmean(obj.responseTimes[trials & obj.nogoTrials]))
            else:
                nRewards[prev][i].append(np.nan)
                dprime[prev][i].append(np.nan)
                hitRate[prev][i].append(np.nan)
                falseAlarmRate[prev][i].append(np.nan)
                catchRate[prev][i].append(np.nan)
                quiescentViolationsPerTrial[prev][i].append(np.nan)
                hitRespTime[prev][i].append(np.nan)
                falseAlarmRespTime[prev][i].append(np.nan)


for d in (nRewards,dprime,hitRate,falseAlarmRate,catchRate,quiescentViolationsPerTrial):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for y in d['all']:
        ax.plot(np.arange(len(y))+1,y,'k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlabel('Session',fontsize=16)
    plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for d in hitRate['all']:
    ax.plot(np.nanmean(d[:2]),np.nanmean(d[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for d in falseAlarmRate['all']:
    ax.plot(np.nanmean(d[:2]),np.nanmean(d[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for x,y in zip(falseAlarmRate['fa'],falseAlarmRate['hit']):
    ax.plot(np.nanmean(x[-2:]),np.nanmean(y[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for x,y in zip(falseAlarmRate['cr'],falseAlarmRate['fa']):
    ax.plot(np.nanmean(x[-2:]),np.nanmean(y[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for x,y in zip(falseAlarmRespTime['all'],hitRespTime['all']):
    ax.plot(np.nanmean(x[-2:]),np.nanmean(y[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for x,y in zip(hitRespTime['fa'],hitRespTime['hit']):
    ax.plot(np.nanmean(x[-2:]),np.nanmean(y[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
for x,y in zip(falseAlarmRespTime['fa'],falseAlarmRespTime['hit']):
    ax.plot(np.nanmean(x[-2:]),np.nanmean(y[-2:]),'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
plt.tight_layout()



        

hitCount = {lbl:[] for lbl in mice}
dprime = {lbl:[] for lbl in mice}
sessionsToPass = {lbl:[] for lbl in mice}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.where(np.array([str(stage) in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool)))[0]
        hitCount[lbl].append([])
        dprime[lbl].append([])
        for sessionInd in sessions:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
            hitCount[lbl][-1].append(hits[0][0])
            dprime[lbl][-1].append(dprimeSame[0][0])
        sessionsToPass[lbl].append(getSessionsToPass(mid,df,sessions,stage))

print({lbl: np.median(sessionsToPass[lbl]) for lbl in sessionsToPass})

if xlim is None:              
    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.5)
xticks = np.arange(0,100,5) if xlim[1]>10 else np.arange(10)
clrs = 'gm' if len(mice) > 1 else 'k'
            
for data,thresh,ylbl in zip((hitCount,dprime),(hitThresh,dprimeThresh),('Hit count','d\'')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xlim,[thresh]*2,'k--')
    for lbl,clr in zip(mice.keys(),clrs):
        m = np.full((len(data[lbl]),int(np.nanmax(sessionsToPass[lbl]))),np.nan)
        for i,d in enumerate(data[lbl]):
            d = d[:sessionsToPass[lbl][i]]
            m[i,:len(d)] = d
            ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25,zorder=2)
            ax.plot(sessionsToPass[lbl][i],d[sessionsToPass[lbl][i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        lbl += ' (n='+str(np.sum(~np.isnan(sessionsToPass[lbl])))+')'
        # ax.plot(np.arange(m.shape[1])+1,np.nanmean(m,axis=0),clr,lw=2,zorder=1)   
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    if ylbl=='d\'':
        ax.set_yticks(np.arange(-1,6))
        ax.set_ylim((-0.5,5) if stage==1 else (-0.5,4))
    ax.set_xlabel('Session',fontsize=16)
    ax.set_ylabel(ylbl,fontsize=16)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(mice.keys(),clrs):
    dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    lbl += ' (n='+str(dsort.size)+')'
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(xticks)
ax.set_xlim(xlim)
ax.set_ylim([0,1.01])
ax.set_xlabel('Sessions to pass',fontsize=16)
ax.set_ylabel('Cumalative fraction',fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()  





## stage 5 learning
mice = {'stage 5 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])}
sessionsToPass = []
for lbl in mice:
    for mid in mice[lbl]:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool))
        firstExperimentSession = getFirstExperimentSession(df)
        if firstExperimentSession is not None:
            sessions[firstExperimentSession:] = False
        sessions = np.where(sessions)[0]
        sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=5))


mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])

dprime = {comp: {mod: [] for mod in ('all','vis','sound')} for comp in ('same','other')}
sessionsToPass = []
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool))
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    for sessionInd in sessions:
        hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
        for dp,comp in zip((dprimeSame,dprimeOther),('same','other')):
            if sessionInd == sessions[0]:
                for mod in ('all','vis','sound'):
                    dprime[comp][mod].append([])
            dp = dp[0]
            dprime[comp]['all'][-1].append(dp)
            task = df.loc[sessionInd,'task version']
            visFirst = 'ori tone' in task or 'ori AMN' in task
            if visFirst:
                dprime[comp]['vis'][-1].append(dp[0:6:2])
                dprime[comp]['sound'][-1].append(dp[1:6:2])
            else:
                dprime[comp]['sound'][-1].append(dp[0:6:2])
                dprime[comp]['vis'][-1].append(dp[1:6:2])
    sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=5))
    try:
        sessionData.append([getSessionData(mid,startTime,lightLoad=True) for startTime in df.loc[sessions,'start time']])
    except:
        pass























