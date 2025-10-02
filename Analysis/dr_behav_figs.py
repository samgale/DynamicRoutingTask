# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
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
from DynamicRoutingAnalysisUtils import getPerformanceStats,getFirstExperimentSession,getSessionsToPass,getSessionData,pca,cluster,fitCurve,calcWeibullDistrib


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

miceToIgnore = summaryDf['wheel fixed'] | summaryDf['cannula']

isIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

isStandardRegimen = ~miceToIgnore & ~isIndirectRegimen & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['stage 5 repeats']   

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


def plotLearning(mice,stage,xlim=None):
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
    
    
def plotStage5Learning(mice):
    dpSame = {lbl: [] for lbl in mice}
    dpOther = {lbl: [] for lbl in mice}
    sessionsToPass = {lbl: [] for lbl in mice}
    for lbl in mice:
        for mid in mice[lbl]:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
            firstExperimentSession = getFirstExperimentSession(df)
            if firstExperimentSession is not None:
                sessions[firstExperimentSession:] = False
            sessions = np.where(sessions)[0]
            dpSame[lbl].append([])
            dpOther[lbl].append([])
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                dpSame[lbl][-1].append(dprimeSame[0])
                dpOther[lbl][-1].append(dprimeOther[0])
            sessionsToPass[lbl].append(getSessionsToPass(mid,df,sessions,stage=5))
            
    print({lbl: np.median(sessionsToPass[lbl]) for lbl in sessionsToPass})

    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.75)
    xticks = np.arange(0,100,5)
    clrs = 'gmrbc'[:len(mice)] if len(mice) > 1 else 'k'
                
    for dp,ylbl in zip((dpSame,dpOther),('d\' (same modality)','d\' (cross-modality)')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xlim,[dprimeThresh]*2,'k--')
        for lbl,clr in zip(mice.keys(),clrs):
            for d,ps in zip(dp[lbl],sessionsToPass[lbl]):
                d = np.nanmean(d,axis=1)
                ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25,zorder=2)
                ax.plot(ps,d[ps-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_yticks(np.arange(-1,5))
        ax.set_ylim([-1,4])
        ax.set_xlabel('Session',fontsize=14)
        ax.set_ylabel(ylbl,fontsize=14)
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
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Sessions to pass',fontsize=14)
    ax.set_ylabel('Cumalative fraction',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    

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


# trial sampling probability
trials = np.concatenate([np.random.permutation(np.repeat(np.arange(4),5)) for _ in range(int(1e6))])
p5th = np.zeros((4,4))
pOther = np.zeros((4,4))
for j in range(4):
    k = np.where(trials==j)[0]
    for i in range(4):
        p5th[i,j] = np.sum(trials[k[5::5]-1] == i) / (len(k)/5)
        other = np.setdiff1d(k,k[5::5])
        pOther[i,j] = np.sum(trials[other-1] == i) / len(other)
        
    
## drop out summary
isEarlyTermination = summaryDf['reason for early termination'].notnull()
print(np.unique(summaryDf[isEarlyTermination & isStandardRegimen]['reason for early termination']))

for isNsb in (np.ones(summaryDf.shape[0],dtype=bool),~summaryDf['nsb'],summaryDf['nsb']):
    stage1Mice = isStandardRegimen & isNsb & (summaryDf['stage 1 pass'] | isEarlyTermination)
    print(np.sum(stage1Mice & summaryDf['stage 1 pass']),'of',np.sum(stage1Mice),'passed')
    # exclude early termination because of health
    # print other reasons for early termination
    reasonForTerm = summaryDf[stage1Mice & ~summaryDf['stage 1 pass']]['reason for early termination']
    for reason in np.unique(reasonForTerm):
        print(reason,np.sum(reasonForTerm==reason))
    # print('\n')
     
    stage2Mice = stage1Mice & summaryDf['stage 1 pass'] & (summaryDf['stage 2 pass'] | isEarlyTermination)
    print(np.sum(stage2Mice & summaryDf['stage 2 pass']),'of',np.sum(stage2Mice),'passed')
    reasonForTerm = summaryDf[stage2Mice & ~summaryDf['stage 2 pass']]['reason for early termination']
    for reason in np.unique(reasonForTerm):
        print(reason,np.sum(reasonForTerm==reason))
    # print('\n')

    stage5Mice = stage2Mice & summaryDf['stage 2 pass'] & (summaryDf['stage 5 pass'] | isEarlyTermination) & ~(summaryDf['reason for early termination']=='stage 5 early ephys')
    print(np.sum(stage5Mice & summaryDf['stage 5 pass']),'of',np.sum(stage5Mice),'passed')
    reasonForTerm = summaryDf[stage5Mice & ~summaryDf['stage 5 pass']]['reason for early termination']
    for reason in np.unique(reasonForTerm):
        print(reason,np.sum(reasonForTerm==reason))
    # print('\n')
    print('\n')


## standard regimen mice stage 1 and 2 
mice = {'stage 1 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 1 pass']]['mouse id'])}
plotLearning(mice,stage=1,xlim=None)
  
mice = {'stage 2 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 2 pass']]['mouse id'])}
plotLearning(mice,stage=2,xlim=None)

mice = {'stage 5 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])}
plotStage5Learning(mice)


## moving to stationary grating switch
ind = summaryDf['stage 1 pass'] & summaryDf['timeouts'] & ~miceToIgnore
mice = {'moving':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id']),
        'stationary': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id'])}
plotLearning(mice,stage=1,xlim=(0.5,20.5))

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


## stage 5 mice that did not pass (low switching)
mice = np.array(summaryDf[isStandardRegimen & (summaryDf['reason for early termination']=='stage 5 low switching')]['mouse id'])

sessionDataNoPass = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0][-10:]
    sessionDataNoPass.append([getSessionData(mid,startTime,lightLoad=True) for startTime in df.loc[sessions,'start time']])


fig = plt.figure()#(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials)    
ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
    y = []
    for mouseInd,exps in enumerate(sessionDataNoPass):
        y.append([])
        for obj in exps:
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if blockInd > 0:
                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                    if 'non-target' in stimLbl:
                        stim = stim[:-1]+'2'
                    trials = obj.trialStim==stim
                    y[-1].append(np.full(preTrials+postTrials,np.nan))
                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                    i = min(preTrials,pre.size)
                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                    post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                    if stim==rewStim:
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials:preTrials+i] = post[:i]
                    else:
                        i = min(postTrials-5,post.size)
                        y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
        y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
    ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
    ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
    ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([-5,-1,5,9,14,19])
ax.set_xticklabels([-5,-1,1,5,10,15])
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials-0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
ax.set_ylabel('Response rate',fontsize=14)
# ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=18)
# ax.set_title(phase+', '+str(len(y))+' mice',fontsize=16)
plt.tight_layout()



## stage 5 training
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])

dprime = {comp: {mod: [] for mod in ('all','vis','sound')} for comp in ('same','other')}
sessionsToPass = []
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
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
        
mouseClrs = plt.cm.tab20(np.linspace(0,1,len(sessionsToPass)))

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),max(len(d) for d in dprime[comp]['all'])),np.nan)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
        ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=2)
        ax.plot(sessionsToPass[i],y[sessionsToPass[i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        dp[i,:len(y)] = y
    # m = np.nanmean(dp,axis=0)
    # ax.plot(np.arange(len(m))+1,m,color='k',lw=2,zorder=1)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xlim([0,max(sessionsToPass)+6])
    ax.set_yticks(np.arange(-1,5))
    ax.set_ylim([-0.5,3])
    ax.set_xlabel('Session',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\'',fontsize=18)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fitFunc = calcWeibullDistrib
for i,(d,clr) in enumerate(zip(dprime['other']['all'],mouseClrs)):
    y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
    x = np.arange(len(y))+1
    bounds = ((0,-0.001,x[0],-np.inf),(y.max(),0.001,x[-1],np.inf))
    fitParams = fitCurve(fitFunc,x,y,bounds=bounds)
    yFit = fitFunc(x,*fitParams)
    ax.plot(x,yFit,color=clr,alpha=0.25,zorder=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=16)
ax.set_xlim([0,max(sessionsToPass)+6])
ax.set_yticks(np.arange(-1,5))
ax.set_ylim([0,3])
ax.set_xlabel('Session',fontsize=18)
ax.set_ylabel('Cross-modal '+'d\'',fontsize=18)
plt.tight_layout()

fig = plt.figure(figsize=(10,10))
nrows = int(round(len(sessionsToPass)**0.5))
ncols = int(len(sessionsToPass)**0.5 + 1)
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
i = 0
j = 0
for d,sp in zip(dprime['other']['all'],sessionsToPass):
    if j==ncols:
        i += 1
        j = 0
    ax = fig.add_subplot(gs[i,j])
    j += 1
    y = np.nanmean(d,axis=1)[:sp+5]
    x = np.arange(len(y))+1
    bounds = ((0,-0.001,x[0],-np.inf),(y.max(),0.001,x[-1],np.inf))
    fitParams = fitCurve(fitFunc,x,y,bounds=bounds)
    yFit = fitFunc(x,*fitParams)
    ax.plot(x,y,color='k',alpha=0.5)
    ax.plot(x,yFit,color='r',lw=3)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xticks(np.arange(0,100,5))
    ax.set_yticks(np.arange(-1,5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0,x[-1]+1])
    ax.set_ylim([0,int(y.max()+1)])
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
learnOnset = []
learnDur = []
for i,(d,clr) in enumerate(zip(dprime['other']['all'],mouseClrs)):
    y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
    x = np.arange(len(y))+1
    bounds = ((0,-0.001,x[0],-np.inf),(y.max(),0.001,x[-1],np.inf))
    fitParams = fitCurve(fitFunc,x,y,bounds=bounds)
    yFit = fitFunc(x,*fitParams)
    yFit -= yFit.min()
    yFit /= yFit.max()
    ax.plot(x,yFit,color=clr,alpha=0.25,zorder=2)
    learnOnset.append(np.where(yFit>0.2)[0][0]+1)
    learnDur.append(np.where(yFit>0.8)[0][0] + 1 - learnOnset[-1])
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=16)
ax.set_xlim([0,max(sessionsToPass)+6])
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1])
ax.set_xlabel('Session',fontsize=18)
ax.set_ylabel('Normalized cross-modal '+'d\'',fontsize=18)
plt.tight_layout()

# np.save(os.path.join(baseDir,'Sam','learnOnset.npy'),dict(zip(mice,learnOnset)))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(learnOnset,learnDur,'o',mec='k',mfc='none')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_aspect('equal')
ax.set_xlim([0,41])
ax.set_ylim([0,41])
ax.set_xlabel('Learning onset (20% of max; sessions)',fontsize=14)
ax.set_ylabel('Learning duration (20-80% of max; sessions)',fontsize=14)
plt.tight_layout()
  
for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),100),np.nan)
    xintp = np.linspace(0,1,100)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
        x = np.linspace(0,1,len(y))
        ax.plot(x,y,color=clr,alpha=0.25,zorder=2)
        ax.plot(x[sessionsToPass[i]-1],y[sessionsToPass[i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        dp[i] = np.interp(xintp,x,y)
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(xintp,m,color='k',lw=2,zorder=1)
    ax.fill_between(xintp,m+s,m-s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    # ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_yticks(np.arange(-1,5))
    ax.set_ylim([-0.5,4])
    ax.set_xlabel('Normalized session',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\'',fontsize=18)
    plt.tight_layout()

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for mod,clr in zip(('vis','sound'),'gm'):
        dp = np.full((len(dprime[comp][mod]),max(len(d) for d in dprime[comp][mod])),np.nan)
        for i,d in enumerate(dprime[comp][mod]):
            y = np.nanmean(d,axis=1)
            ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=2)
            dp[i,:len(y)] = y
        m = np.nanmean(dp,axis=0)
        lbl = 'visual-rewarded blocks' if mod=='vis' else 'auditory-rewarded blocks'
        ax.plot(np.arange(len(m))+1,m,color=clr,lw=2,zorder=1,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-3,4])
    ax.set_xlabel('Session',fontsize=14)
    lbl = ' (same modal)' if comp=='same' else ' (cross-modal)'
    ax.set_ylabel('d\''+lbl,fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
   
for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),100),np.nan)
    xintp = np.linspace(0,1,100)
    for mod,clr in zip(('vis','sound'),'gm'):
        for i,d in enumerate(dprime[comp][mod]):
            y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
            x = np.linspace(0,1,len(y))
            # ax.plot(x,y,color=clr,alpha=0.25,zorder=2)
            dp[i] = np.interp(xintp,x,y)
        m = np.nanmean(dp,axis=0)
        s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
        ax.plot(xintp,m,color=clr,lw=2,zorder=0,label=('visual' if mod=='vis' else 'auditory')+' rewarded blocks')
        ax.fill_between(xintp,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_yticks((np.arange(4) if comp=='same' else np.arange(-0.5,3,0.5)))
    ax.set_ylim(([0,3.5] if comp=='same' else [-0.5,2]))
    ax.set_xlabel('Normalized session',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\'',fontsize=18)
    plt.legend(loc='lower right',fontsize=14)
    plt.tight_layout()
    
for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),100),np.nan)
    xintp = np.linspace(0,1,100)
    for i,(v,a,clr) in enumerate(zip(dprime[comp]['vis'],dprime[comp]['sound'],mouseClrs)):
        y = (np.nanmean(a,axis=1) - np.nanmean(v,axis=1))[:sessionsToPass[i]+5]
        # y = scipy.ndimage.median_filter(y,3,mode='nearest')
        # y = np.convolve(y,np.ones(5)/5,mode='same')
        y = scipy.ndimage.gaussian_filter(y,1,mode='nearest')
        x = np.linspace(0,1,len(y))
        ax.plot(x,y,color=clr,alpha=0.25)
        dp[i] = np.interp(xintp,x,y)
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(xintp,m,color='k',lw=2)
    ax.fill_between(xintp,m+s,m-s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    # ax.set_ylim([-3,4])
    ax.set_xlabel('Normalized session',fontsize=18)
    ax.set_ylabel('Difference in '+('cross' if comp=='other' else 'within')+'-modal '+'d\'\n(auditory - visual)',fontsize=18)
    plt.tight_layout()


# comparison of d' and cross-modal inference
dp = []
rr = []
for exps,sp in zip(sessionData,sessionsToPass):
    dp.append([])
    rr.append([])
    for obj in exps[:sp+5]:
        dp[-1].append(np.nanmean(obj.dprimeOtherModalGo))
        rr[-1].append(np.mean([np.mean(obj.trialResponse[obj.goTrials & (obj.trialBlock==block-1)][-10:]) - obj.trialResponse[obj.otherModalGoTrials & (obj.trialBlock==block)][0] for block in range(2,7)]))
        # rr[-1].append(1 - np.mean([obj.trialResponse[obj.otherModalGoTrials & (obj.trialBlock==block)][0] for block in range(1,7)]))

smoothSigma = 1
dp,rr = [[scipy.ndimage.gaussian_filter(a,smoothSigma) for a in b] for b in (dp,rr)]
    
cc = []
cs = []
lag = []
ccoef = []
for d,r in zip(dp,rr):
    norm = np.linalg.norm(r) * np.linalg.norm(d)
    cc.append(np.correlate(r,d,'full') / norm)
    # cs.append(cc[-1] - np.mean([np.correlate(r,np.random.permutation(d),'full') / norm for _ in range(100)],axis=0))
    tri = np.linspace(1/len(r),1,len(r))
    tri = np.concatenate((tri,tri[-2::-1]))
    cs.append(cc[-1] / tri)
    lag.append(np.argmax(cs[-1]) - len(r) + 1)
    ccoef.append(np.corrcoef(r,d)[0,1])


x = np.concatenate(dp)
y = np.concatenate(rr)
plt.plot(x,y,'ko',alpha=0.2)
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
xrng = np.array([min(x),max(x)])
plt.plot(xrng,slope*xrng+yint,'-',color='r')

fig = plt.figure(figsize=(10,10))
nrows = int(round(len(sessionsToPass)**0.5))
ncols = int(len(sessionsToPass)**0.5 + 1)
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
i = 0
j = 0
for c,s,lg in zip(cc,cs,lag):
    if j==ncols:
        i += 1
        j = 0
    ax = fig.add_subplot(gs[i,j])
    j += 1
    y = np.array(c)
    x = np.arange(len(y))+1
    ax.plot(x,c,color='k',alpha=0.5)
    ax.plot(x,s,color='r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xticks(np.arange(0,100,5))
    # ax.set_yticks(np.arange(-1,5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0,x[-1]+1])
    # ax.set_ylim([-0.1,1])
    ax.set_title(lg,fontsize=6)
plt.tight_layout()

fig = plt.figure(figsize=(10,10))
nrows = int(round(len(sessionsToPass)**0.5))
ncols = int(len(sessionsToPass)**0.5 + 1)
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
i = 0
j = 0
for d,r,lg in zip(dp,rr,lag):
    if j==ncols:
        i += 1
        j = 0
    ax = fig.add_subplot(gs[i,j])
    j += 1
    for a,clr in zip((d,r),'br'):
        y = np.array(a)
        y -= y.min()
        y /= y.max()
        x = np.arange(len(y))+1
        bounds = ((0,-0.001,x[0],-np.inf),(1,0.001,x[-1],np.inf))
        try:
            fitParams = fitCurve(fitFunc,x,y,bounds=bounds)
            yFit = fitFunc(x,*fitParams)
            ax.plot(x,yFit,color=clr,lw=3)
        except:
            pass
        ax.plot(x,y,color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xticks(np.arange(0,100,5))
    ax.set_yticks([0,1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0,x[-1]+1])
    ax.set_ylim([0,1])
    ax.set_title(lg,fontsize=6)
plt.tight_layout()
   
 
# zig-zag plots
x = np.arange(6)+1
for phase in ('initial training','after learning'):
    for firstRewStim in ('vis1','sound1'):
        hc,fc = 'gm' if firstRewStim=='vis1' else 'mg'
        for stim in ('target','non-target'):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            hr = []
            fr = []
            for exps,sp in zip(sessionData,sessionsToPass):
                h = []
                f = []
                for obj in (exps[:nInitialTrainingSessions] if phase=='initial training' else exps[sp:]):
                    if obj.blockStimRewarded[0] == firstRewStim:
                        r = np.zeros(6)
                        r[::2] = obj.hitRate[::2] if stim=='target' else obj.falseAlarmSameModal[::2]
                        r[1::2] = obj.falseAlarmOtherModalGo[1::2] if stim=='target' else obj.falseAlarmOtherModalNogo[1::2]
                        h.append(r)
                        r = np.zeros(6)
                        r[1::2] = obj.hitRate[1::2] if stim=='target' else obj.falseAlarmSameModal[1::2]
                        r[::2] = obj.falseAlarmOtherModalGo[::2] if stim=='target' else obj.falseAlarmOtherModalNogo[::2]
                        f.append(r)
                hr.append(np.nanmean(h,axis=0))
                fr.append(np.nanmean(f,axis=0))
            for h,f in zip(hr,fr):
                ax.plot(x,h,hc,alpha=0.05)
                ax.plot(x,f,fc,alpha=0.05)
            ax.plot(x,np.nanmean(hr,axis=0),hc+'-o',label=('visual ' if firstRewStim=='vis1' else 'auditory ')+stim)
            ax.plot(x,np.nanmean(fr,axis=0),fc+'-o',label=('auditory ' if firstRewStim=='vis1' else 'visual ')+stim)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=16)
            ax.set_xticks(x)
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([0.5,6.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Block #',fontsize=18)
            ax.set_ylabel('Response rate',fontsize=18)
            ax.legend(loc=('lower right' if stim=='target' else 'upper right'),fontsize=16)
            # ax.set_title(phase+' (n='+str(len(hr))+' mice)',fontsize=12)
            plt.tight_layout()
            

## performance by block number
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,blockLbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
    for lbl,ls in zip(('cross-modality','within modality'),('-','--')):
        dp = []
        for exps,s in zip(sessionData,sessionsToPass):
            d = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps[s:]):
                j = obj.blockStimRewarded==rewardStim
                a = obj.dprimeSameModal if 'within' in lbl else obj.dprimeOtherModalGo
                d[i,j] = np.array(a)[j]
            dp.append(np.nanmean(d,axis=0))
        m = np.nanmean(dp,axis=0)
        s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=blockLbl+', '+lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_yticks(np.arange(5))
ax.set_ylim([0,4])
ax.set_xlabel('Block')
ax.set_ylabel('d\'')
ax.legend(loc='lower right')
ax.set_title(str(len(sessionData))+' mice')
plt.tight_layout()


## catch rate, quiescent violations, and run speed by block
for stage in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:nInitialTrainingSessions] if stage=='initial training' else exps[s:]
            r = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps):
                j = obj.blockStimRewarded==rewardStim
                r[i,j] = np.array(obj.catchResponseRate)[j]
            rr.append(np.nanmean(r,axis=0))
        m = np.nanmean(rr,axis=0)
        s = np.nanstd(rr,axis=0)/(len(rr)**0.5)
        ax.plot(x,m,color=clr,label=lbl+' blocks')
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_ylim([0,0.1])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Catch trial response rate',fontsize=14)
    ax.legend(loc='upper right',fontsize=12)
    ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:nInitialTrainingSessions] if stage=='initial training' else exps[s:]
            r = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps):
                for blockInd,blockRewardStim in enumerate(obj.blockStimRewarded):
                    if blockRewardStim==rewardStim:
                        trials = obj.trialBlock==blockInd+1
                        r[i,blockInd] = np.array(obj.trialQuiescentViolations)[trials].sum() / trials.sum()
            rr.append(np.nanmean(r,axis=0))
        m = np.nanmean(rr,axis=0)
        s = np.nanstd(rr,axis=0)/(len(rr)**0.5)
        ax.plot(x,m,color=clr,label=lbl+' blocks')
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_ylim([0,0.5])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Quiescent violations per trial',fontsize=14)
    ax.legend(loc='upper right',fontsize=12)
    ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:nInitialTrainingSessions] if stage=='initial training' else exps[s:]
            r = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps):
                for blockInd,blockRewardStim in enumerate(obj.blockStimRewarded):
                    if blockRewardStim==rewardStim:
                        trials = obj.trialBlock==blockInd+1
                        r[i,blockInd] = np.nanmean(obj.quiescentRunSpeed[trials])
            rr.append(np.nanmean(r,axis=0))
        m = np.nanmean(rr,axis=0)
        s = np.nanstd(rr,axis=0)/(len(rr)**0.5)
        ax.plot(x,m,color=clr,label=lbl+' blocks')
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_ylim([15,35])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Run speed (cm/s)',fontsize=14)
    ax.legend(loc='upper right',fontsize=12)
    ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()
            

## run speed by session
runSpeed = {phase: {blockType: [] for blockType in ('vis rewarded','aud rewarded')} for phase in ('initial training','after learning','all')}
dprime = copy.deepcopy(runSpeed)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:nInitialTrainingSessions]
        elif phase=='after learning':
            exps = exps[s:]
        for blockType in ('vis rewarded','aud rewarded'):
            for d in (runSpeed,dprime):
                d[phase][blockType].append([[] for _ in range(len(exps))])
            for sessionInd,obj in enumerate(exps):
                stimTrials = np.in1d(obj.trialStim,('vis1','sound1')) & ~obj.autoRewardScheduled
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if (blockType=='vis rewarded' and rewStim=='vis1') or (blockType=='aud rewarded' and rewStim=='sound1'):
                        trials = stimTrials & (obj.trialBlock==blockInd+1)
                        runSpeed[phase][blockType][mouseInd][sessionInd].append(np.nanmean(obj.quiescentRunSpeed[trials]))  
                        dprime[phase][blockType][mouseInd][sessionInd].append(obj.dprimeOtherModalGo[blockInd])
                        
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [-5,85]
    ax.plot(alim,alim,'k--')
    x,y = [np.nanmean(np.concatenate(runSpeed[phase][blockType]),axis=1) for blockType in ('vis rewarded','aud rewarded')]
    ax.plot(x,y,'ko',alpha=0.2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Run speed, visual rewarded blocks (cm/s)',fontsize=14)
    ax.set_ylabel('Run speed, auditory rewarded blocks (cm/s)',fontsize=14)
    ax.set_title(phase+' ('+str(len(x))+' sessions)',fontsize=14)
    plt.tight_layout()
    
for phase in ('initial training','after learning'):
    for blockType in ('vis rewarded','aud rewarded'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x,y = [np.nanmean(np.concatenate(d[phase][blockType]),axis=1) for d in (runSpeed,dprime)]
        ax.plot(x,y,'ko',alpha=0.2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([-5,85])
        ax.set_ylim([-3,4])
        ax.set_xlabel('Run speed (cm/s)',fontsize=14)
        ax.set_ylabel('Cross-modal d''',fontsize=14)
        ax.set_title(phase+', '+blockType+' ('+str(len(x))+' sessions)',fontsize=14)
        plt.tight_layout()
    

## run speed dynamics
runSpeed = {phase: {blockType: [] for blockType in ('vis rewarded','aud rewarded')} for phase in ('initial training','after learning','all')}
nFrames = int(5 * 60)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:nInitialTrainingSessions]
        elif phase=='after learning':
            exps = exps[s:]
        for blockType in ('vis rewarded','aud rewarded'):
            runSpeed[phase][blockType].append([])
            for sessionInd,obj in enumerate(exps):
                trials = ~obj.autoRewardScheduled & (obj.trialStim==('vis1' if blockType=='vis rewarded' else 'sound1'))
                runSpeed[phase][blockType][-1].append(np.nanmean([scipy.ndimage.median_filter(obj.runningSpeed[sf-nFrames:sf],3,mode='nearest') for sf in obj.stimStartFrame[trials]],axis=0))

t = -(np.arange(nFrames)/60)[::-1]
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for blockType,clr in zip(('vis rewarded','aud rewarded'),'gm'):
        r = [np.nanmean(r,axis=0) for r in runSpeed[phase][blockType]]
        m = np.mean(r,axis=0)
        s = np.std(r,axis=0) / (len(r)**0.5)
        ax.plot(t,m,color=clr,label=blockType)
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    # ax.set_xlim([-5,85])
    ax.set_ylim([15,35])
    ax.set_xlabel('Time before stimulus onset (s)',fontsize=14)
    ax.set_ylabel('Run speed (cm/s)',fontsize=14)
    ax.set_title(phase,fontsize=14)
    ax.legend()
    plt.tight_layout()


## comparison of visual and auditory response times
for stage in ('initial training','after learning'):
    for blockRew,lbl in zip(('vis1','sound1'),('visual rewarded','auditory rewarded')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,1],[0,1],'k--')
        rt = {'vis1': [], 'sound1': []}
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:nInitialTrainingSessions] if stage=='initial training' else exps[s:]
            for stim in rt:
                rt[stim].append([])
            for i,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if rewStim==blockRew:
                        blockTrials = (obj.trialBlock==blockInd+1) & ~obj.autoRewardScheduled
                        for stim in rt:
                            rt[stim][-1].extend(obj.responseTimes[blockTrials & (obj.trialStim==stim)])
            for stim in rt:
                rt[stim][-1] = np.nanmean(rt[stim][-1])
        ax.plot(rt['vis1'],rt['sound1'],'ko')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_aspect('equal')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('Response time to visual target (s)',fontsize=14)
        ax.set_ylabel('Response time to auditory target (s)',fontsize=14)
        ax.legend(loc='upper right',fontsize=12)
        ax.set_title(stage+', '+lbl,fontsize=14)
        plt.tight_layout()
    
    
## session clusters
sessionClustData = {key: [] for key in ('nSessions','mouseId','sessionStartTime','mouse','session','passed','block','firstRewardStim','hitRate','falseAlarmRate','dprime','clustData')}
sessionInd = 0
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
        sessionInd += 1

for key in sessionClustData:
    sessionClustData[key] = np.array(sessionClustData[key])

clustData = sessionClustData['clustData']

# nSamples = 10
# nShuffles = 10
# samples = [np.random.choice(clustData.shape[0],int(0.9*clustData.shape[0]),replace=False) for _ in range(nSamples)]
# sil = []
# silShuffled = []
# ari = []
# ariShuffled = []
# for n in range(2,11):
#     spectralClustering = sklearn.cluster.SpectralClustering(n_clusters=n,affinity='nearest_neighbors',n_neighbors=10)
#     c = spectralClustering.fit_predict(clustData)
#     sil.append(sklearn.metrics.silhouette_score(clustData,c))
#     r = []
#     for s in samples:
#         cs = spectralClustering.fit_predict(clustData[s])
#         r.append(sklearn.metrics.adjusted_rand_score(c[s],cs))
#     ari.append(np.mean(r))
    
#     shuffledData = clustData.copy().T
#     r = []
#     for _ in range(nShuffles):
#         np.random.shuffle(shuffledData)
#         c = spectralClustering.fit_predict(shuffledData.T)
#         silShuffled.append(sklearn.metrics.silhouette_score(shuffledData.T,c))
#         for s in samples:
#             cs = spectralClustering.fit_predict(shuffledData.T[s])
#             r.append(sklearn.metrics.adjusted_rand_score(c[s],cs))
#     ariShuffled.append(np.mean(r))

clustColors = [clr for clr in 'rgkbmcy']+['0.6']
nClust = 6
clustColors = clustColors[:nClust]
clustId,linkageMat = cluster(clustData,nClusters=nClust)
clustLabels = np.unique(clustId)
colorThresh = 0 if nClust<2 else linkageMat[::-1,2][nClust-2]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
scipy.cluster.hierarchy.set_link_color_palette(list(clustColors))
scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,truncate_mode=None,p=7,color_threshold=colorThresh,above_threshold_color='k',labels=None,no_labels=True)
scipy.cluster.hierarchy.set_link_color_palette(None)
ax.plot([0,1000000],[0.85*colorThresh]*2,'k--')
ax.set_yticks([])
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
k = np.arange(linkageMat.shape[0])+2
ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
ax.plot([0,100],[0.85*colorThresh]*2,'k--')
ax.set_xlim([0,30.4])
ax.set_xlabel('Cluster')
ax.set_ylabel('Linkage Distance')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
plt.tight_layout()

nMice = len(sessionData)
nClust = 6
spectralClustering = sklearn.cluster.SpectralClustering(n_clusters=nClust,affinity='nearest_neighbors',n_neighbors=10,assign_labels='kmeans')
clustId = spectralClustering.fit_predict(clustData)
clustId += 1
clustLabels = np.unique(clustId)

newClustOrder = [4,1,3,5,6,2]
newClustId = clustId.copy()
for i,c in enumerate(newClustOrder):
    newClustId[clustId==c] = i+1
clustId = newClustId

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
    if clust==1:
        ax.legend(loc='upper right',fontsize=16)
    ax.set_title('cluster '+str(clust)+' (n='+str(len(r))+' sessions)',fontsize=12)
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

        
x = np.arange(6)+1
for clust in clustLabels:
    for firstRewardStim,sessionLabel in zip(('vis1','sound1'),('visual rewarded first','auditory rewarded first')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        i = (sessionClustData['firstRewardStim']==firstRewardStim) & (clustId==clust)
        hr = sessionClustData['hitRate'][i]
        fr = sessionClustData['falseAlarmRate'][i]
        for clr,lbl in zip('gm',('visual target','auditory target')):
            r = np.zeros((i.sum(),6))
            if (firstRewardStim=='vis1' and lbl=='visual target') or (firstRewardStim=='sound1' and lbl=='auditory target'):
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
        if clust==1:
            ax.legend(loc='upper right',fontsize=16)
        ax.set_title('cluster '+str(clust)+', '+sessionLabel+' (n='+str(len(r))+' sessions)',fontsize=12)
        plt.tight_layout()

x = np.arange(6)+1        
for clust in clustLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for firstRewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded first','auditory rewarded first')):
        r = sessionClustData['dprime'][(sessionClustData['firstRewardStim']==firstRewardStim) & (clustId==clust)]
        m = np.nanmean(r,axis=0)
        s = np.nanstd(r)/(len(r)**0.5)
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=16)
        ax.set_xticks(x)
        ax.set_xlim([0.5,6.5])
        ax.set_ylim([0,2.5])
        ax.set_xlabel('Block #',fontsize=18)
        ax.set_ylabel('Cross-modal d\'',fontsize=18)
        ax.legend(fontsize=16)
        ax.set_title('cluster '+str(clust),fontsize=12)
        plt.tight_layout()
           
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for firstRewStim,offset,clr in zip(('vis1','sound1'),(-0.2,0.2),'gm'):
    i = sessionClustData['firstRewardStim']==firstRewStim
    for clust in clustLabels:
        n = np.sum(i & (clustId==clust))
        lbl = ('visual rewarded first' if firstRewStim=='vis1' else 'auditory rewarded first') if clust==1 else None
        ax.bar(clust+offset,n,width=0.4,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xticks(clustLabels)
ax.set_xticklabels(clustLabels)
# ax.set_ylim((0,0.6))
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of sessions')
ax.legend()
plt.tight_layout()

for ind in (sessionClustData['session']<nInitialTrainingSessions,(sessionClustData['session']>=nInitialTrainingSessions) & ~sessionClustData['passed'],sessionClustData['passed']):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for firstRewStim,offset,clr in zip(('vis1','sound1'),(-0.2,0.2),'gm'):
        i = ind & (sessionClustData['firstRewardStim']==firstRewStim)
        for clust in clustLabels:
            p = np.sum(i & (clustId==clust)) / i.sum()
            lbl = ('visual rewarded first' if firstRewStim=='vis1' else 'auditory rewarded first') if clust==1 else None
            ax.bar(clust+offset,p,width=0.4,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xticks(clustLabels)
    ax.set_xticklabels(clustLabels)
    ax.set_ylim((0,0.7))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fraction of sessions')
    ax.legend()
    plt.tight_layout()

mouseClustProb = np.zeros((3,nMice,nClust))
for k,ind in enumerate((sessionClustData['session']<nInitialTrainingSessions,(sessionClustData['session']>=nInitialTrainingSessions) & ~sessionClustData['passed'],sessionClustData['passed'])):
    for i,m in enumerate(np.argsort(sessionsToPass)):
        for j,clust in enumerate(clustLabels):
            s = clustId[(sessionClustData['mouse']==m) & ind]
            mouseClustProb[k,i,j] = np.sum(s==clust)/s.size

fig = plt.figure(figsize=(10,8))
fig.suptitle('Cluster probability')
for i,(p,lbl) in enumerate(zip(mouseClustProb,('intitial training','later training','after learning'))):            
    ax = fig.add_subplot(1,3,i+1) 
    im = ax.imshow(p,cmap='magma',clim=(0,np.nanmax(p)))
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(nClust))
    ax.set_xticklabels(np.arange(nClust)+1)
    ax.set_yticks([])
    if i==1:
        ax.set_xlabel('Cluster')
    if i==0:
        ax.set_ylabel('Mouse')
    ax.set_title(lbl)
    plt.tight_layout()
      
sessionClust = np.full((nMice,max(sessionClustData['nSessions'])),np.nan)
for i,m in enumerate(np.argsort(sessionsToPass)):
    c = clustId[sessionClustData['mouse']==m]
    sessionClust[i,:c.size] = c
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmap = matplotlib.cm.viridis.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(sessionClust,cmap=cmap)
cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
cb.set_ticks(np.arange(nClust)+1)
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'w')
ax.set_xticks(np.arange(10,sessionClust.shape[1],10)-1)
ax.set_xticklabels(np.arange(10,sessionClust.shape[1],10))
ax.set_yticks([])
ax.set_xlabel('Session')
ax.set_ylabel('Mouse')
ax.set_title('Session cluster\n(white line = passed learning criteria)')
plt.tight_layout()

sessionClustAlt = np.full((nMice,max(sessionClustData['nSessions'])),np.nan)
for i,m in enumerate(np.argsort(sessionsToPass)):
    c = clustId[sessionClustData['mouse']==m]
    sessionClustAlt[i,:c.size] = 0
    sessionClustAlt[i,:c.size][c==4] = -1
    sessionClustAlt[i,:c.size][c==5] = 1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmap = matplotlib.cm.bwr.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(sessionClustAlt,cmap=cmap)
cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
cb.set_ticks((-1,0,1))
cb.set_ticklabels(('4','other','5'))
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'k')
ax.set_xticks(np.arange(10,sessionClustAlt.shape[1],10)-1)
ax.set_xticklabels(np.arange(10,sessionClustAlt.shape[1],10))
ax.set_yticks([])
ax.set_xlabel('Session')
ax.set_ylabel('Mouse')
ax.set_title('Session cluster\n(black line = passed learning criteria)')
plt.tight_layout() 

probPoorAudSuppress = []
probPoorVisSuppress = []
for m in range(len(sessionsToPass)):
    mi = sessionClustData['mouse'] == m
    for firstRewStim in ('vis1','sound1'):
        poorAud = mi & (((clustId==4) & (sessionClustData['firstRewardStim']=='vis1')) | ((clustId==5) & (sessionClustData['firstRewardStim']=='sound1')))
        poorVis = mi & (((clustId==4) & (sessionClustData['firstRewardStim']=='sound1')) | ((clustId==5) & (sessionClustData['firstRewardStim']=='vis1')))
        probPoorAudSuppress.append(np.sum(poorAud)/mi.sum())
        probPoorVisSuppress.append(np.sum(poorVis)/mi.sum())
    
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
alim = [-0.05,0.75]
ax.plot(alim,alim,'--',color='k')
ax.plot(probPoorVisSuppress,probPoorAudSuppress,'ko',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Weak suppression of responses to visual target (fraction of sessions\ncluster 4 aud rewarded first or cluster 5 vis rewarded first)')
ax.set_ylabel('Weak suppression of responses to auditory target (fraction of sessions\ncluster 4 vis rewarded first or cluster 5 aud rewarded first)')
plt.tight_layout()
        
prevClustProb = np.zeros((3,3,len(clustLabels),len(clustLabels)))
prevClustChance = np.zeros((3,3,nClust))
nextClustProb = prevClustProb.copy()
nextClustChance = prevClustChance.copy()
for l,si in enumerate((np.ones(sessionClustData['firstRewardStim'].size,dtype=bool),sessionClustData['firstRewardStim']=='vis1',sessionClustData['firstRewardStim']=='sound1')):
    for k,ind in enumerate((sessionClustData['session']<nInitialTrainingSessions,(sessionClustData['session']>=nInitialTrainingSessions) & ~sessionClustData['passed'],sessionClustData['passed'])):
        sessions = np.where(ind & si & (sessionClustData['session']>0))[0]
        for j,clust in enumerate(clustLabels):
            prevClustChance[l,k,j] = np.sum(clustId[sessions-1]==clust)/len(sessions)
            c = clustId[sessions]==clust
            for i,prevClust in enumerate(clustLabels):
                prevClustProb[l,k,i,j] = np.sum(clustId[sessions-1][c]==prevClust)/c.sum()
    
        sessions = np.where(ind & si & (sessionClustData['session']+1 < sessionClustData['nSessions']))[0]
        for j,clust in enumerate(clustLabels):
            nextClustChance[l,k,j] = np.sum(clustId[sessions+1]==clust)/len(sessions)
            c = clustId[sessions]==clust
            for i,nextClust in enumerate(clustLabels):
                nextClustProb[l,k,i,j] = np.sum(clustId[sessions+1][c]==nextClust)/c.sum()

for l,blockType in enumerate(('all',)):#'vis rewarded first','aud rewarded first')):
    for k,stage in enumerate(('intitial training','later training','after learning')):
        for transProb,lbl in zip((prevClustProb[l,k],nextClustProb[l,k]),('Previous','Next')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1) 
            im = ax.imshow(transProb,cmap='magma',clim=(0,transProb.max()),origin='lower')
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            ax.set_xticks(np.arange(len(clustLabels)))
            ax.set_yticks(np.arange(len(clustLabels)))
            ax.set_xticklabels(clustLabels)
            ax.set_yticklabels(clustLabels)
            ax.set_xlabel('Session cluster')
            ax.set_ylabel(lbl+' session cluster')
            ax.set_title('Probability'+'\n'+stage+', '+blockType)
            plt.tight_layout()

# block switch plots for session clusters
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('visual target','auditory target','visual non-target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1) 
for clust in clustLabels:
    fig = plt.figure()#(figsize=(8,4.5))
    gs = matplotlib.gridspec.GridSpec(2,2)
    for col,firstRewStim in enumerate(('vis1','sound1')):
        for row,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks'))):
            ax = fig.add_subplot(gs[row,col])
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gmgm',('-','-','--','--')):
                y = []
                for obj in np.concatenate(sessionData)[(clustId==clust) & (sessionClustData['firstRewardStim']==firstRewStim)]:
                    trials = (obj.trialStim==stim)
                    r = obj.trialResponse
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0 and rewStim==rewardStim:
                            y.append(np.full(preTrials+postTrials+1,np.nan))
                            pre = r[(obj.trialBlock==blockInd) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][preTrials-i:preTrials] = pre[-i:]
                            post = r[(obj.trialBlock==blockInd+1) & trials]
                            if stim==rewStim:
                                i = min(postTrials,post.size)
                                y[-1][preTrials:preTrials+i] = post[:i]
                            else:
                                i = min(postTrials-5,post.size)
                                y[-1][preTrials+5:preTrials+5+i] = post[:i]
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
                ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
                ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
                ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
            ax.set_ylabel('Response rate',fontsize=12)
            # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
            if row==0:
                ax.set_title(firstRewStim+' rewarded first',fontsize=12)
    plt.tight_layout()


        
## block switch plots
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('visual target','auditory target','visual non-target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1) 
for phase in ('initial training','after learning'):
    for ylbl,yticks,ylim,stimInd in zip(('Response rate','Response time (s)'),([0,0.5,1],[0.3,0.4,0.5,0.6]),([0,1.02],[0.3,0.6]),(slice(0,4),slice(0,2))):
        resp = {}
        respAll = {}
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
            fig = plt.figure()#(figsize=(8,4.5))
            ax = fig.add_subplot(1,1,1)
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            resp[rewardStim] = {}
            respAll[rewardStim] = {}
            for stim,stimLbl,clr,ls in zip(stimNames[stimInd],stimLabels[stimInd],'gmgm'[stimInd],('-','-','--','--')[stimInd]):
                y = []
                yall = []
                for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                    if len(exps)>0:# and hasLateAutorewards[mouseInd]:
                        if phase=='initial training':
                            exps = exps[:nInitialTrainingSessions]
                        elif phase=='after learning':
                            exps = exps[s:]
                        y.append([])
                        yall.append([])
                        for obj in exps:
                            trials = (obj.trialStim==stim)
                            r = obj.trialResponse if 'rate' in ylbl else obj.responseTimes
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0 and rewStim==rewardStim:
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = r[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = r[(obj.trialBlock==blockInd+1) & trials]
                                    if stim==rewStim:
                                        i = min(postTrials,post.size)
                                        y[-1][-1][preTrials:preTrials+i] = post[:i]
                                    else:
                                        i = min(postTrials-5,post.size)
                                        y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                                    yall[-1].append([np.nanmean(pre[5:]),np.nanmean(post[5:])])
                        y[-1] = np.nanmean(y[-1],axis=0)
                        yall[-1] = np.mean(yall[-1],axis=0)
                if stim in ('vis1','sound1'):
                    resp[rewardStim][stim] = y
                    respAll[rewardStim][stim] = yall
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
                ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
                ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
                ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks(yticks)
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim(ylim)
            ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
            ax.set_ylabel(ylbl,fontsize=12)
            # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
            # ax.set_title(phase+' (n='+str(len(y))+' mice)'+'\n'+blockLabel,fontsize=12)
            plt.tight_layout()
        
        if 'time' in ylbl:
            ylim = [0,1]
            yticks = [0,1]
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
            fig = plt.figure()
            fig.suptitle(blockLabel)
            axs = []
            gs = matplotlib.gridspec.GridSpec(2,2)
            xticks = (0,1)
            for rr,i,j,clr in zip((respAll[rewardStim][rewardStim],
                                   np.array(resp[rewardStim][rewardStim])[:,[preTrials-1,preTrials+5]],
                                   respAll[rewardStim]['vis1' if rewardStim=='sound1' else 'sound1'],
                                   np.array(resp[rewardStim]['vis1' if rewardStim=='sound1' else 'sound1'])[:,[preTrials-1,preTrials+5]]),
                                  (0,0,1,1),(0,1,0,1),['g' if rewardStim=='vis1' else 'm']*2+['m' if rewardStim=='vis1' else 'g']*2):
                ax = fig.add_subplot(gs[i,j])
                for r in rr:
                    ax.plot(xticks,r,'o-',color=clr,mec=clr,mfc='none',ms=6,lw=1,alpha=0.2)
                mean = np.nanmean(rr,axis=0)
                sem = np.nanstd(rr,axis=0)/(len(rr)**0.5)
                ax.plot(xticks,mean,'o-',color=clr,mec=clr,mfc=clr,ms=10,lw=2)
                for xt,m,s in zip(xticks,mean,sem):
                    ax.plot([xt,xt],[m-s,m+s],color=clr,lw=2)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=10)
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                if i==1 and j==0:
                    ax.set_xticklabels(('all trials\nprevious block','all trials'))
                elif i==1 and j==1:
                    ax.set_xticklabels(('last trial\nprevious block','first trial\nafter switch trials'))
                else:
                    ax.set_xticklabels([])
                if j==0:
                    ax.set_ylabel(ylbl)
                else:
                    ax.set_yticklabels([])
                ax.set_xlim([-0.2,1.2])
                ax.set_ylim(ylim)
            plt.tight_layout()

# first block
x = np.arange(postTrials+1) 
for phase in ('initial training','after learning'):
    for ylbl,yticks,ylim,stimInd in zip(('Response rate','Response time (z score)'),([0,0.5,1],[-0.5,0,0.5,1]),([0,1.02],[-0.6,1.1]),(slice(0,4),slice(0,2))):
        resp = {}
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
            fig = plt.figure()#(figsize=(8,4.5))
            ax = fig.add_subplot(1,1,1)
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            resp[rewardStim] = {}
            for stim,stimLbl,clr,ls in zip(stimNames[stimInd],stimLabels[stimInd],'gmgm'[stimInd],('-','-','--','--')[stimInd]):
                y = []
                for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                    if len(exps)>0:
                        if phase=='initial training':
                            exps = exps[:nInitialTrainingSessions]
                        elif phase=='after learning':
                            exps = exps[s:]
                        y.append([])
                        yall.append([])
                        for obj in exps:
                            trials = (obj.trialStim==stim) # & ~obj.autoRewardScheduled
                            r = obj.trialResponse if 'rate' in ylbl else (obj.responseTimes-np.nanmean(obj.responseTimes[trials]))/np.nanstd(obj.responseTimes[trials])
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd == 0 and rewStim==rewardStim:
                                    y[-1].append(np.full(postTrials+1,np.nan))
                                    post = r[(obj.trialBlock==blockInd+1) & trials]
                                    if stim==rewStim:
                                        i = min(postTrials,post.size)
                                        y[-1][-1][:i] = post[:i]
                                    else:
                                        i = min(postTrials-5,post.size)
                                        y[-1][-1][5:5+i] = post[:i]
                                    yall[-1].append(np.nanmean(post[5:]))
                        y[-1] = np.nanmean(y[-1],axis=0)
                if stim in ('vis1','sound1'):
                    resp[rewardStim][stim] = y
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks(yticks)
            ax.set_xlim([-0.5,postTrials+0.5])
            ax.set_ylim(ylim)
            ax.set_xlabel('Trials of indicated type after block start',fontsize=12)
            ax.set_ylabel(ylbl,fontsize=12)
            # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
            ax.set_title(phase+' (n='+str(len(y))+' mice)'+'\n'+blockLabel,fontsize=12)
            plt.tight_layout()
         
# block switch plot, target stimuli only
for phase in ('initial training','after learning'):
    for getDeltaLickProb in (False,True):
        fig = plt.figure()#(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        preTrials = 5
        postTrials = 20
        x = np.arange(-preTrials,postTrials)    
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                exps = (exps[:nInitialTrainingSessions] if phase == 'initial training' else exps[s:])
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                            if 'non-target' in stimLbl:
                                stim = stim[:-1]+'2'
                            trials = obj.trialStim==stim
                            if getDeltaLickProb and stim in obj.blockStimRewarded:
                                blockTrials = (obj.trialBlock==blockInd+1)
                                firstTarget = np.where(blockTrials & ~obj.autoRewardScheduled & np.in1d(obj.trialStim,obj.blockStimRewarded))[0][0]
                                if np.where(blockTrials & ~obj.autoRewardScheduled & trials)[0][0] > firstTarget: # or not np.all(obj.trialResponse[blockTrials][:5]):
                                    continue
                            y[-1].append(np.full(preTrials+postTrials,np.nan))
                            pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                            if stim==rewStim:
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                            else:
                                i = min(postTrials-5,post.size)
                                y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            if stimLbl=='unrewarded target stim':
                nonRewTargResp = y
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            if getDeltaLickProb and phase=='after learning' and stimLbl in ('rewarded target stim','unrewarded target stim'):
                key = 'rewTarg' if stimLbl == 'rewarded target stim' else 'nonRewTarg'
                deltaLickProb['5 rewarded targets'][key] = np.array(y)[:,[preTrials-1,preTrials+5]]
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
        ax.set_ylabel('Response rate',fontsize=14)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=18)
        # ax.set_title(phase+', '+str(len(y))+' mice',fontsize=16)
        plt.tight_layout()
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    rr = np.array(nonRewTargResp)[:,[preTrials-1,preTrials+5]]
    for r in rr:
        ax.plot([0,1],r,'o-',color='m',mec='m',mfc='none',ms=6,lw=1,alpha=0.2)
    mean = np.nanmean(rr,axis=0)
    sem = np.nanstd(rr,axis=0)/(len(rr)**0.5)
    ax.plot([0,1],mean,'o-',color='m',mec='m',mfc='m',ms=10,lw=2)
    # for x,m,s in zip([0,1],mean,sem):
    #     ax.plot([x,x],[m-s,m+s],color='m',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([0,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xticklabels(('last trial of\nprevious block','first trial of\nnew block'))
    ax.set_ylabel('Response rate',fontsize=16)
    ax.set_xlim([-0.2,1.2])
    ax.set_ylim([0,1.01])
    plt.tight_layout()
    
    fig = plt.figure()#(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)
    preTrials = 5
    postTrials = 20
    x = np.arange(-preTrials,postTrials)    
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
            # if not hasLateAutorewards[mouseInd]:
            #     continue
            exps = (exps[:5] if phase == 'initial training' else exps[s:])
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim) #& ~obj.autoRewardScheduled
                        rt = obj.responseTimes - np.nanmean(obj.responseTimes[trials])
                        y[-1].append(np.full(preTrials+postTrials,np.nan))
                        pre = rt[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = rt[(obj.trialBlock==blockInd+1) & trials]
                        if stim==rewStim:
                            i = min(postTrials,post.size)
                            y[-1][-1][preTrials:preTrials+i] = post[:i]
                        else:
                            i = min(postTrials-5,post.size)
                            y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]                        
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
        ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
        ax.plot(x[preTrials:],m[preTrials:],color=clr)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_yticks([-0.1,-0.05,0,0.05,0.1])
    ax.set_ylim([-0.105,0.105])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
    ax.set_ylabel('Response time\n(difference from mean, s)',fontsize=14)
    # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    # ax.set_title(phase+', '+str(len(y))+' mice',fontsize=12)
    plt.tight_layout()
    
# block switch plot for more learning phases and delay from last reward to first non-rewarded target trial
firstTrialMean = {}
firstTrialSem = {}
fullBlockMean = {}
fullBlockSem = {}
# for phase in ('after learning all',):
for phase in ('initial training','early learning','late learning','criterion sessions','after learning'):
    for minTrialsSinceRew in range((5 if phase == 'after learning all' else 2)):
        fig = plt.figure()#(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        preTrials = 5
        postTrials = 20
        x = np.arange(-preTrials,postTrials)    
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr,ls in zip(('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (non-rewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouseInd,(exps,sp,lo) in enumerate(zip(sessionData,sessionsToPass,learnOnset)):
                if phase == 'initial training':
                    exps = exps[:2]
                elif phase == 'early learning':
                    exps = exps[lo+1:lo+3]
                elif phase == 'late learning':
                    exps = exps[sp-4:sp-2]
                elif phase == 'criterion sessions':
                    exps = exps[sp-2:sp]
                elif phase == 'after learning':
                    exps = exps[sp:sp+2]
                elif phase == 'after learning all':
                    exps = exps[sp:]
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            blockTrials = obj.trialBlock==blockInd+1
                            nonRewTarg = 'sound1' if rewStim=='vis1' else 'vis1'
                            firstNonRewTarg = np.where(blockTrials & (obj.trialStim==nonRewTarg))[0][0]
                            trialsSinceRew = firstNonRewTarg - np.where(blockTrials & (obj.trialStim==rewStim))[0]
                            if trialsSinceRew[trialsSinceRew > 0][-1] > minTrialsSinceRew:
                                stim = nonRewTarg if 'non-rewarded' in stimLbl else rewStim
                                if 'non-target' in stimLbl:
                                    stim = stim[:-1]+'2'
                                trials = obj.trialStim==stim
                                y[-1].append(np.full(preTrials+postTrials,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[blockTrials & trials]
                                if stim==rewStim:
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials:preTrials+i] = post[:i]
                                else:
                                    i = min(postTrials-5,post.size)
                                    y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0) if len(y[-1]) > 0 else np.full(preTrials+postTrials,np.nan)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            if stimLbl == 'non-rewarded target':
                if phase not in firstTrialMean:
                    firstTrialMean[phase] = []
                    firstTrialSem[phase] = []
                    fullBlockMean[phase] = []
                    fullBlockSem[phase] = []
                firstTrialMean[phase].append(np.mean(m[preTrials-5:preTrials]) - m[preTrials+5])
                firstTrialSem[phase].append(np.mean(s[preTrials-5:preTrials]) - s[preTrials+5])
                fullBlockMean[phase].append(np.mean(m[preTrials-5:preTrials]) - np.mean(m[preTrials+5:]))
                fullBlockSem[phase].append(np.mean(s[preTrials-5:preTrials]) - np.mean(s[preTrials+5:]))
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
        ax.set_ylabel('Response rate',fontsize=14)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=18)
        # ax.set_title(phase+', '+str(len(y))+' mice',fontsize=16)
        plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(len(firstTrialMean))
for d,clr in zip(((firstTrialMean,firstTrialSem),(fullBlockMean,fullBlockSem)),'rb'):
    for i,ls in enumerate(('-','--')):
        m = [d[0][phase][i] for phase in firstTrialMean]
        m = np.array(m)
        m -= m.min()
        m /= m.max()
        s = [d[1][phase][i] for phase in firstTrialMean]
        ax.plot(x,m,'o-',color=clr,ls=ls)
        for i,j,k in zip(x,m,s):
            ax.plot([i,i],[j-k,j+k],color=clr)
   



# first trial lick or no lick  
for lbl in ('all blocks','first trial lick','first trial no lick'):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
            exps = exps[s:]
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim)
                        firstTrialResp = obj.trialResponse[(obj.trialBlock==blockInd+1) & (obj.trialStim==rewStim)][0]
                        if (lbl=='first trial lick' and not firstTrialResp) or (lbl=='first trial no lick' and firstTrialResp):
                            continue
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
            if len(y[-1]) > 0:
                y[-1] = np.nanmean(y[-1],axis=0)
            else:
                y[-1] = np.full(preTrials+postTrials+1,np.nan)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        if lbl == 'all blocks' and stimLbl == 'rewarded target stim':
            deltaLickProb['1 rewarded target']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+2]]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    ax.set_title(lbl+', '+str(len(y))+' mice',fontsize=12)
    plt.tight_layout()
    

# number of false alarm licks
postTrials = 15
x = np.arange(postTrials)+1
for phase in ('initial training','after learning'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gmgm',('-','-','--','--')):
            if stim==rewardStim:
                continue
            y = []
            for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                if len(exps)>0:
                    if phase=='initial training':
                        exps = exps[:nInitialTrainingSessions]
                    elif phase=='after learning':
                        exps = exps[s:]
                    y.append([])
                    for obj in exps:
                        nLicks = np.array([np.sum((obj.lickTimes > st + obj.responseWindowTime[0]) & (obj.lickTimes < st + obj.responseWindowTime[1])) for st in obj.stimStartTimes]).astype(float)
                        nLicks[~obj.trialResponse] = np.nan
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if rewStim==rewardStim:
                                y[-1].append(np.full(postTrials,np.nan))
                                post = nLicks[(obj.trialBlock==blockInd+1) & (obj.trialStim==stim)]
                                i = min(postTrials,post.size)
                                y[-1][-1][:i] = post[:i]
                    y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlabel('Trials after block switch',fontsize=12)
        ax.set_ylabel('Number of licks on false alarm trials',fontsize=12)
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,3.5])
        ax.legend()
        ax.set_title(phase+', '+blockLabel)
        plt.tight_layout()


## cluster block performance
stimNames = ('vis1','vis2','sound1','sound2')
blockClustData = {key: [] for key in ('nSessions','mouseId','sessionStartTime','mouse','session','passed','block','rewardStim','nBlockTrials','hitRate','falseAlarmOtherModalGo','clustData')}
blockClustData['response'] = {stim: [] for stim in stimNames}
blockClustData['smoothedResponse'] = {stim: [] for stim in stimNames}
blockClustData['responseTime'] = {stim: [] for stim in stimNames}
blockClustData['responseTimeNorm'] = {stim: [] for stim in stimNames}
smoothSigma = 3
tintp = np.arange(0,601,5)
nMice = len(sessionData)
nExps = [len(s) for s in sessionData]
for m,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
    #exps = exps[s:] # exps[:s+nSessions]
    blockClustData['nSessions'].append(len(exps))
    for i,obj in enumerate(exps):
        for blockInd,rewardStim in enumerate(obj.blockStimRewarded):
            blockClustData['mouseId'].append(obj.subjectName)
            blockClustData['sessionStartTime'].append(obj.startTime)
            blockClustData['mouse'].append(m)
            blockClustData['session'].append(i)
            blockClustData['passed'].append(i > s-1)
            blockClustData['block'].append(blockInd)
            blockClustData['rewardStim'].append(rewardStim)
            blockTrials = obj.trialBlock==blockInd+1
            blockClustData['nBlockTrials'].append(blockTrials.sum())
            blockClustData['hitRate'].append(obj.hitRate[blockInd])
            blockClustData['falseAlarmOtherModalGo'].append(obj.falseAlarmOtherModalGo[blockInd])
            for stim in stimNames:
                stimTrials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                trials = blockTrials & stimTrials
                if trials.sum() > 0:
                    blockClustData['response'][stim].append(obj.trialResponse[trials])
                    blockClustData['responseTime'][stim].append(obj.responseTimes[trials])
                    blockClustData['responseTimeNorm'][stim].append(obj.responseTimes[trials]-np.nanmean(obj.responseTimes[stimTrials]))
                    
                    stimTime = obj.stimStartTimes[trials] - obj.trialStartTimes[trials][0]
                    r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                    r = np.interp(tintp,stimTime,r)
                    blockClustData['smoothedResponse'][stim].append(r)
                else:
                    blockClustData['response'][stim].append(np.array([]))
                    blockClustData['smoothedResponse'][stim].append(np.full(tintp.size,np.nan))
                    blockClustData['responseTime'][stim].append(np.array([]))
                    blockClustData['responseTimeNorm'][stim].append(np.array([]))
                   
            # sn = stimNames[:4] if rewardStim=='vis1' else stimNames[2:4]+stimNames[:2]
            sn = ('vis1','sound1') if rewardStim=='vis1' else ('sound1','vis1')
            blockClustData['clustData'].append(np.concatenate([blockClustData['smoothedResponse'][stim][-1] for stim in sn]))

for key in blockClustData:
    if isinstance(blockClustData[key],dict):
        for k in blockClustData[key]:
            if max(len(d) for d in blockClustData[key][k]) != len(blockClustData[key][k][0]):
                blockClustData[key][k] = np.array(blockClustData[key][k],dtype='O')
            else:
                blockClustData[key][k] = np.array(blockClustData[key][k])
    else:
        blockClustData[key] = np.array(blockClustData[key])


pcaData,eigVal,eigVec = pca(blockClustData['clustData'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,10])
ax.set_ylim((0,1.02))
ax.set_xlabel('PC')
ax.set_ylabel('Cumulative Fraction of Variance Explained')
plt.tight_layout()

nPC = np.where((np.cumsum(eigVal)/eigVal.sum())>0.95)[0][0]+1
clustData = pcaData[:,:nPC]

nClust = 6
spectralClustering = sklearn.cluster.SpectralClustering(n_clusters=nClust,affinity='nearest_neighbors',n_neighbors=10,assign_labels='kmeans')
clustId = spectralClustering.fit_predict(clustData)
clustId += 1
clustLabels = np.unique(clustId)

clustColors = [clr for clr in 'rgkbmcy']+['0.6']
nClust = 6
clustColors = clustColors[:nClust]
clustId,linkageMat = cluster(clustData,nClusters=nClust)
clustLabels = np.unique(clustId)
colorThresh = 0 if nClust<2 else linkageMat[::-1,2][nClust-2]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
scipy.cluster.hierarchy.set_link_color_palette(list(clustColors))
scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,truncate_mode=None,p=7,color_threshold=colorThresh,above_threshold_color='k',labels=None,no_labels=True)
scipy.cluster.hierarchy.set_link_color_palette(None)
ax.plot([0,1000000],[0.85*colorThresh]*2,'k--')
ax.set_yticks([])
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
k = np.arange(linkageMat.shape[0])+2
ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
ax.plot([0,100],[0.85*colorThresh]*2,'k--')
ax.set_xlim([0,30.4])
ax.set_xlabel('Cluster')
ax.set_ylabel('Linkage Distance')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
scipy.cluster.hierarchy.set_link_color_palette(list(clustColors))
scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,truncate_mode='lastp',p=6,no_labels=False)
scipy.cluster.hierarchy.set_link_color_palette(None)
ax.set_yticks([])
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for clust in clustLabels:
    n = np.sum(clustId==clust)
    ax.bar(clust,n,width=0.8,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xticks(clustLabels)
ax.set_xticklabels(clustLabels)
# ax.set_ylim((0,0.7))
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of blocks')
plt.tight_layout()

newClustOrder = [5,6,1,2,3,4]
newClustId = clustId.copy()
for i,c in enumerate(newClustOrder):
    newClustId[clustId==c] = i+1
clustId = newClustId
clustColors = [clustColors[i] for i in newClustOrder]

ind = blockClustData['session']<nInitialTrainingSessions
ind = (blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed']
ind = blockClustData['passed']
ind = np.ones(len(blockClustData['session']),dtype=bool)
stimLabels = ('rewarded target','non-rewarded target')
postTrials = 15
x = np.arange(postTrials)+1
for clust in clustLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ci = clustId==clust
    for lbl,clr in zip(stimLabels,'gm'):
        resp = []
        for stim in ('vis1','sound1'):
            rewStim = stim if lbl=='rewarded target' else ('sound1' if stim=='vis1' else 'vis1')
            for r in blockClustData['response'][stim][ind & (blockClustData['rewardStim']==rewStim) & ci]:
                j = min(postTrials,r.size)
                resp.append(np.full(postTrials,np.nan))
                resp[-1][:j] = r[:j]
        m = np.nanmean(resp,axis=0)
        s = np.nanstd(resp)/(len(resp)**0.5)
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials after block switch cue trials',fontsize=14)
    ax.set_ylabel('Response rate',fontsize=14)
    # if clust==1:
    #     ax.legend(loc='upper right')
    ax.set_title('Cluster '+str(clust)+', (n='+str(len(resp))+')',fontsize=12)
    plt.tight_layout()

blockClustData['clustId'] = clustId
blockClustData['trialCluster'] = {}
for m in np.unique(blockClustData['mouseId']):
    blockClustData['trialCluster'][m] = {}
    mi = blockClustData['mouseId']==m
    for s in np.unique(blockClustData['sessionStartTime'][mi]):
        blockClustData['trialCluster'][m][s] = []
        si = blockClustData['sessionStartTime']==s
        for n,c in zip(blockClustData['nBlockTrials'][mi & si],clustId[mi & si]):
            blockClustData['trialCluster'][m][s].extend(np.zeros(n)+c)
        blockClustData['trialCluster'][m][s] = np.array(blockClustData['trialCluster'][m][s])
            
#np.save(os.path.join(baseDir,'Sam','clustData.npy'),blockClustData)

stimNames = ('vis1','vis2','sound1','sound2')
postTrials = 15
x = np.arange(postTrials)+1
for clust in clustLabels:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
            resp = []
            for r in blockClustData['response'][stim][(blockClustData['rewardStim']==rewardStim) & (clustId==clust)]:
                j = min(postTrials,r.size)
                resp.append(np.full(postTrials,np.nan))
                resp[-1][:j] = r[:j]
            m = np.nanmean(resp,axis=0)
            s = np.nanstd(resp)/(len(resp)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stim)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks(np.arange(0,16,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch cue trials',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
        if clust==1:
            ax.legend(loc='upper right')
        ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
        plt.tight_layout()
        
for clust in clustLabels:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr in zip(('vis1','sound1'),'gm'):
            resp = []
            for r in blockClustData['responseTimeNorm'][stim][(blockClustData['rewardStim']==rewardStim) & (clustId==clust)]:
                j = min(postTrials,r.size)
                resp.append(np.full(postTrials,np.nan))
                resp[-1][:j] = r[:j]
            m = np.nanmean(resp,axis=0)
            s = np.nanstd(resp)/(len(resp)**0.5)
            ax.plot(x,m,color=clr,label=stim)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([-0.1,0.15])
        ax.set_xlabel('Trials after block switch cue trials')
        ax.set_ylabel('Response time (diff. from mean)')
        ax.legend(loc='lower right')
        ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
        plt.tight_layout()
                
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for rewStim,offset,clr in zip(('vis1','sound1'),(-0.2,0.2),'gm'):
    i = blockClustData['rewardStim']==rewStim
    for clust in clustLabels:
        n = np.sum(i & (clustId==clust))
        lbl = ('visual rewarded' if rewStim=='vis1' else 'auditory rewarded') if clust==1 else None
        ax.bar(clust+offset,n,width=0.4,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xticks(clustLabels)
ax.set_xticklabels(clustLabels)
# ax.set_ylim((0,0.7))
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of blocks')
ax.legend()
plt.tight_layout()

for k,ind in enumerate((blockClustData['session']<nInitialTrainingSessions,(blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed'],blockClustData['passed'])):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for rewStim,offset,clr in zip(('vis1','sound1'),(-0.2,0.2),'gm'):
        i = ind & (blockClustData['rewardStim']==rewStim)
        for clust in clustLabels:
            p = np.sum(i & (clustId==clust)) / i.sum()
            lbl = ('visual rewarded' if rewStim=='vis1' else 'auditory rewarded') if clust==1 else None
            ax.bar(clust+offset,p,width=0.4,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',labelsize=14)
    ax.set_xticks(clustLabels)
    ax.set_xticklabels(clustLabels)
    ax.set_ylim((0,0.7))
    ax.set_xlabel('Cluster',fontsize=16)
    ax.set_ylabel('Fraction of blocks',fontsize=16)
    ax.legend(fontsize=14)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for c,clr in zip(clustLabels,clustColors):
    i = clustId==c
    ax.plot(blockClustData['hitRate'][i],blockClustData['falseAlarmOtherModalGo'][i],'o',mec=clr,mfc='none',alpha=0.5,label='cluster '+str(c))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xlim([-0.05,1.05])
ax.set_ylim([-0.05,1.05])
ax.set_aspect('equal')
ax.set_xlabel('Rewarded target response rate')
ax.set_ylabel('Non-rewarded target response rate')
ax.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()
        
blockClustProb = np.zeros((3,6,nClust))
for k,ind in enumerate((blockClustData['session']<5,(blockClustData['session']>=5) & ~blockClustData['passed'],blockClustData['passed'])):
    for i in range(6):
        blocks = ind & (blockClustData['block']==i)
        for j,clust in enumerate(clustLabels):
            blockClustProb[k,i,j] = np.sum(blocks & (clustId==clust))/blocks.sum()

for i,(p,lbl) in enumerate(zip(blockClustProb,('intitial training','later training','after learning'))):    
    fig = plt.figure() 
    ax = fig.add_subplot(1,1,1) 
    im = ax.imshow(p,cmap='magma',clim=(0,blockClustProb.max()),origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    for side in ('right','top','left','bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',labelsize=12)
    ax.set_xticks(np.arange(nClust))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(clustLabels)
    ax.set_xlabel('Cluster',fontsize=14)
    ax.set_yticklabels(np.arange(6)+1)
    ax.set_ylabel('Block',fontsize=14)
    ax.set_title('Cluster probability, '+lbl,fontsize=14)
    plt.tight_layout()
    
sessionClustProb = np.zeros((3,nClust,nClust))
for k,ind in enumerate((blockClustData['session']<nInitialTrainingSessions,(blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed'],blockClustData['passed'])):
    for i,sc in enumerate(clustLabels):
        for j,clust in enumerate(clustLabels):
            c = ind & (clustId==clust)
            b = 0
            bc = 0
            for m,s in zip(blockClustData['mouse'][c],blockClustData['session'][c]):
                b += 1
                if clustIdSessions[(sessionClustData['mouse']==m) & (sessionClustData['session']==s)] == sc:
                    bc += 1
            blockClustData['session']
            sessionClustProb[k,i,j] = bc / b

fig = plt.figure()
fig.suptitle('Cluster probability')
for i,(p,lbl) in enumerate(zip(sessionClustProb,('intitial training','later training','after learning'))):            
    ax = fig.add_subplot(1,3,i+1) 
    im = ax.imshow(p,cmap='magma',clim=(0,mouseClustProb.max()),origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(nClust))
    ax.set_xticklabels(np.arange(nClust)+1)
    ax.set_yticks(np.arange(nClust))
    ax.set_yticklabels(np.arange(nClust)+1)
    if i==1:
        ax.set_xlabel('Block cluster')
    if i==0:
        ax.set_ylabel('Session cluster')
    ax.set_title(lbl)
    plt.tight_layout()

mouseClustProb = np.zeros((3,nMice,nClust))
for k,ind in enumerate((blockClustData['session']<nInitialTrainingSessions,(blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed'],blockClustData['passed'])):
    for i,m in enumerate(np.argsort(sessionsToPass)):
        for j,clust in enumerate(clustLabels):
            b = clustId[(blockClustData['mouse']==m) & ind]
            mouseClustProb[k,i,j] = np.sum(b==clust)/b.size

fig = plt.figure(figsize=(10,8))
fig.suptitle('Cluster probability')
for i,(p,lbl) in enumerate(zip(mouseClustProb,('intitial training','later training','after learning'))):            
    ax = fig.add_subplot(1,3,i+1) 
    im = ax.imshow(p,cmap='magma',clim=(0,np.nanmax(mouseClustProb)))
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(nClust))
    ax.set_xticklabels(np.arange(nClust)+1)
    ax.set_yticks([])
    if i==1:
        ax.set_xlabel('Cluster')
    if i==0:
        ax.set_ylabel('Mouse')
    ax.set_title(lbl)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
p = mouseClustProb[-1]
for clust,clr in enumerate(clustColors[:nClust]):
    dsort = np.sort(p[:,clust])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label='cluster '+str(clust+1))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,0.7])
ax.set_ylim([0,1.01])
ax.set_xlabel('Fraction of blocks in each cluster after learning',fontsize=16)
ax.set_ylabel('Cumalative fraction of mice',fontsize=16)
ax.legend(loc='lower right')
plt.tight_layout()
    
fig = plt.figure(figsize=(4.5,10))
fig.suptitle('Within session cluster probability for each mouse\n(white line = passed learning criteria)')
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax = fig.add_subplot(nMice,1,i+1)
    mi = blockClustData['mouse']==m
    p = np.full((nClust,blockClustData['nSessions'][m]),np.nan)
    for s in range(blockClustData['nSessions'][m]):
        si = blockClustData['session']==s
        assert(np.sum(mi & si)==6)
        for j,c in enumerate(clustLabels):
            p[j,s] = np.sum(clustId[mi & si] == c) / 6
    im = ax.imshow(p,cmap='magma',clim=(0,1),interpolation='none')
    ax.plot([sessionsToPass[m]-0.5]*2,[-0.5,6.5],'w')
    for side in ('right','top','left','bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',labelsize=10)
    ax.set_xticks([])
    if i==0:
        ax.set_yticks([0,nClust-1])
        ax.set_yticklabels([1,nClust])
        ax.set_ylabel('Cluster',fontsize=12)
        cb = plt.colorbar(im,ax=ax)
    else:
        ax.set_yticks([])
    if i==nMice-1:
        ax.set_xlabel('Session',fontsize=12)
    ax.set_anchor('W')
plt.tight_layout()

mostFreqClust = np.full((nMice,max(blockClustData['nSessions'])),np.nan)
for i,m in enumerate(np.argsort(sessionsToPass)):
    mi = blockClustData['mouse']==m
    for s in range(blockClustData['nSessions'][m]):
        si = blockClustData['session']==s
        c,n = np.unique(clustId[mi & si],return_counts=True)
        mostFreqClust[i,s] = c[np.argmax(n)]
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmap = matplotlib.cm.viridis.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(mostFreqClust,cmap=cmap)
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

numDiffClust = np.full((nMice,max(blockClustData['nSessions'])),np.nan)
for i,m in enumerate(np.argsort(sessionsToPass)):
    mi = blockClustData['mouse']==m
    for s in range(blockClustData['nSessions'][m]):
        si = blockClustData['session']==s
        np.unique(clustId[mi & si])
        numDiffClust[i,s] = np.unique(clustId[mi & si]).size
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmap = matplotlib.cm.plasma.copy()
cmap.set_bad(color=[0.5]*3)
im = ax.imshow(numDiffClust,cmap=cmap)
cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
cb.set_ticks(np.arange(nClust)+1)
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'w')
ax.set_xticks(np.arange(10,70,10)-1)
ax.set_xticklabels(np.arange(10,70,10))
ax.set_yticks([])
ax.set_xlabel('Session')
ax.set_ylabel('Mouse')
ax.set_title('Number of different clusters in session\n(white line = passed learning criteria)')
plt.tight_layout()

for k,ind in enumerate((blockClustData['session']<nInitialTrainingSessions,(blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed'],blockClustData['passed'])):
    chanceProb = np.array([np.sum(ind & (clustId==clust))/np.sum(ind) for clust in clustLabels])
    for lbl in ('Absolute','Relative'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        a = blockClustProb[k]-chanceProb
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
        
prevClustProb = np.zeros((3,len(clustLabels),len(clustLabels)))
prevClustChance = np.zeros((3,nClust))
nextClustProb = prevClustProb.copy()
nextClustChance = prevClustChance.copy()
for k,ind in enumerate((blockClustData['session']<nInitialTrainingSessions,(blockClustData['session']>=nInitialTrainingSessions) & ~blockClustData['passed'],blockClustData['passed'])):
    blocks = np.where(ind & (blockClustData['block']>0))[0]
    for j,clust in enumerate(clustLabels):
        prevClustChance[k,j] = np.sum(clustId[blocks-1]==clust)/len(blocks)
        c = clustId[blocks]==clust
        for i,prevClust in enumerate(clustLabels):
            prevClustProb[k,i,j] = np.sum(clustId[blocks-1][c]==prevClust)/c.sum()

    blocks = np.where(ind & (blockClustData['block']<5))[0]
    for j,clust in enumerate(clustLabels):
        nextClustChance[k,j] = np.sum(clustId[blocks+1]==clust)/len(blocks)
        c = clustId[blocks]==clust
        for i,nextClust in enumerate(clustLabels):
            nextClustProb[k,i,j] = np.sum(clustId[blocks+1][c]==nextClust)/c.sum()

for k in range(3):
    for transProb,lbl in zip((prevClustProb[k],nextClustProb[k]),('Previous','Next')):
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

for k in range(3):
    for transProb,chanceProb,lbl in zip((prevClustProb[k],nextClustProb[k]),(prevClustChance[k],nextClustChance[k]),('Previous','Next')):
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


## time dependence of effect of prior reward or response (avg across mice)
stimType = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
prevTrialTypes = ('response to rewarded target','response to non-rewarded target')
trainingPhases = ('initial training','after learning')
blockRewStim = ('vis1','sound1','all')
blockEpochs = ('first half','last half','full')
resp = {phase: {blockRew: {epoch: {s: [] for s in stimType} for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
respNorm = copy.deepcopy(resp)
respTime = copy.deepcopy(resp)
respTimeNorm = copy.deepcopy(resp)
trialsSince = {phase: {blockRew: {epoch: {prevTrial: {s: [] for s in stimType} for prevTrial in prevTrialTypes} for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
timeSince = copy.deepcopy(trialsSince)
for phase in trainingPhases:
    for blockRew in blockRewStim:
        for epoch in blockEpochs:
            for exps,sp in zip(sessionData,sessionsToPass):
                for i,obj in enumerate(exps[:5] if phase=='initial training' else exps[sp:]):
                    b = 0
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockRew not in ('all',rewStim):
                            continue
                        blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                        blockTrials = np.setdiff1d(blockTrials,np.where(obj.catchTrials)[0])
                        rewTrials = np.intersect1d(blockTrials,np.where(obj.trialRewarded)[0])
                        rewTargetTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==rewStim)[0])
                        otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
                        nonRewTargetTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==otherModalTarget)[0])
                        targetTrials = np.concatenate((rewTargetTrials,nonRewTargetTrials))
                        nonTargetTrials = np.setdiff1d(blockTrials,targetTrials)
                        for s in stimType:
                            if i == 0 and b == 0:
                                resp[phase][blockRew][epoch][s].append([])
                                respNorm[phase][blockRew][epoch][s].append([])
                                respTime[phase][blockRew][epoch][s].append([])
                                respTimeNorm[phase][blockRew][epoch][s].append([])
                            if s=='rewarded target':
                                stim = rewStim
                            elif s=='non-rewarded target':
                                stim = otherModalTarget
                            elif s=='non-target (rewarded modality)':
                                stim = rewStim[:-1]+'2'
                            else:
                                stim = otherModalTarget[:-1]+'2'
                            stimTrials = obj.trialStim == stim
                            rtNorm = obj.responseTimes - np.nanmean(obj.responseTimes[stimTrials])
                            stimTrials = np.intersect1d(blockTrials,np.where(stimTrials)[0])
                            if len(stimTrials) < 1:
                                continue
                            for prevTrialType,trials in zip(prevTrialTypes,(rewTargetTrials,nonRewTargetTrials)):
                                if i == 0 and b == 0:
                                    trialsSince[phase][blockRew][epoch][prevTrialType][s].append([])
                                    timeSince[phase][blockRew][epoch][prevTrialType][s].append([])
                                respTrials = np.intersect1d(trials,np.where(obj.trialResponse)[0])
                                if len(respTrials) > 0:
                                    prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
                                    anyTargetTrials = np.array([np.any(np.in1d(obj.trialStim[p+1:s],(rewStim,otherModalTarget))) for s,p in zip(stimTrials,prevRespTrial)])
                                    anyQuiescentViolations = np.array([np.any(obj.trialQuiescentViolations[p+1:s]) for s,p in zip(stimTrials,prevRespTrial)])
                                    notValid = (stimTrials <= respTrials[0]) | (stimTrials > trials[-1]) #| anyTargetTrials #| anyQuiescentViolations
                                    # if len(rewTrials) > 0 and prevTrialType != 'response to rewarded target':
                                    #     prevRewTrial = rewTrials[np.searchsorted(rewTrials,stimTrials) - 1]
                                    #     notValid = notValid | ((stimTrials - prevRewTrial) < 2)
                                    tr = stimTrials - prevRespTrial
                                    tr[notValid] = -1
                                    tm = obj.stimStartTimes[stimTrials] - obj.stimStartTimes[prevRespTrial]
                                    tm[notValid] = np.nan
                                    trialsSince[phase][blockRew][epoch][prevTrialType][s][-1].extend(tr)
                                    timeSince[phase][blockRew][epoch][prevTrialType][s][-1].extend(tm)
                                else:
                                    trialsSince[phase][blockRew][epoch][prevTrialType][s][-1].extend(np.full(len(stimTrials),np.nan))
                                    timeSince[phase][blockRew][epoch][prevTrialType][s][-1].extend(np.full(len(stimTrials),np.nan))
                            resp[phase][blockRew][epoch][s][-1].extend(obj.trialResponse[stimTrials])
                            respNorm[phase][blockRew][epoch][s][-1].extend(obj.trialResponse[stimTrials] - obj.trialResponse[stimTrials].mean())
                            respTime[phase][blockRew][epoch][s][-1].extend(obj.responseTimes[stimTrials])
                            respTimeNorm[phase][blockRew][epoch][s][-1].extend(rtNorm[stimTrials])
                        b += 1
        
            for i,prevTrialType in enumerate(prevTrialTypes):
                for s in stimType:
                    trialsSince[phase][blockRew][epoch][prevTrialType][s] = [np.array(a) for a in trialsSince[phase][blockRew][epoch][prevTrialType][s]]
                    timeSince[phase][blockRew][epoch][prevTrialType][s] = [np.array(a) for a in timeSince[phase][blockRew][epoch][prevTrialType][s]]
                    if i==0:
                        resp[phase][blockRew][epoch][s] = [np.array(a) for a in resp[phase][blockRew][epoch][s]]
                        respNorm[phase][blockRew][epoch][s] = [np.array(a) for a in respNorm[phase][blockRew][epoch][s]]
                        respTime[phase][blockRew][epoch][s] = [np.array(a) for a in respTime[phase][blockRew][epoch][s]]
                        respTimeNorm[phase][blockRew][epoch][s] = [np.array(a) for a in respTimeNorm[phase][blockRew][epoch][s]]


trialBins = np.arange(20)
for phase in ('initial training','after learning'):
    for blockRew in ('vis1','sound1'):
        for epoch in ('full',):
            for prevTrialType in prevTrialTypes:
                fig = plt.figure()#(figsize=(8,4.5))
                ax = fig.add_subplot(1,1,1)
                clrs = 'mgmg' if blockRew=='sound1' else 'gmgm'
                for stim,clr,ls in zip(stimType,clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(trialsSince[phase][blockRew][epoch][prevTrialType][stim],respNorm[phase][blockRew][epoch][stim]):
                        n.append(np.full(trialBins.size,np.nan))
                        p.append(np.full(trialBins.size,np.nan))
                        for i in trialBins:
                            j = d==i
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(trialBins,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(trialBins,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.5,0.5,0.1))
                ax.set_xlim([0,10])
                ax.set_ylim([-0.1,0.2])
                ax.set_xlabel('Trials since last '+prevTrialType,fontsize=14)
                ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
                plt.tight_layout()
        
for phase in ('initial training','after learning'):
    for blockRew in ('all',):
        for epoch in ('full',):
            for prevTrialType in prevTrialTypes:
                fig = plt.figure()#(figsize=(8,4.5))
                ax = fig.add_subplot(1,1,1)
                clrs = 'mgmg' if blockRew=='sound1' else 'gmgm'
                for stim,clr,ls in zip(stimType[:3],clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(trialsSince[phase][blockRew][epoch][prevTrialType][stim],respTimeNorm[phase][blockRew][epoch][stim]):
                        n.append(np.full(trialBins.size,np.nan))
                        p.append(np.full(trialBins.size,np.nan))
                        for i in trialBins:
                            j = d==i
                            j = j & ~np.isnan(r)
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(trialBins,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(trialBins,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.08,0.1,0.04))
                ax.set_xlim([0,10])
                ax.set_ylim([-0.08,0.08])
                ax.set_xlabel('Trials since last '+prevTrialType,fontsize=14)
                ax.set_ylabel('Response time (s)\n(difference from mean)',fontsize=14)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
                plt.tight_layout()
        
timeBins = np.array([0,5,10,15,20,30,40,50,60,80,100])
x = timeBins[:-1] + np.diff(timeBins)/2
for phase in trainingPhases:
    for blockRew in ('vis1','sound1'):
        for epoch in ('full',):
            y = {prevTrial: {} for prevTrial in prevTrialTypes}
            for prevTrialType in prevTrialTypes:    
                fig = plt.figure()#(figsize=(12,6))
                ax = fig.add_subplot(1,1,1)
                clrs = 'mgmg' if blockRew=='sound1' else 'gmgm'
                for stim,clr,ls in zip(stimType,clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(timeSince[phase][blockRew][epoch][prevTrialType][stim],respNorm[phase][blockRew][epoch][stim]):
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
                    y[prevTrialType][stim] = {'mean': m, 'sem': s}
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.5,0.5,0.1))
                ax.set_xlim([0,90])
                ax.set_ylim([-0.1,0.2])
                ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=16)
                plt.tight_layout()

for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for blockRew in ('all',):
            for epoch in ('first half','last half'):
                fig = plt.figure()#(figsize=(8,4.5))
                ax = fig.add_subplot(1,1,1)
                clrs = 'mgmg' if blockRew=='sound1' else 'gmgm'
                for stim,clr,ls in zip(stimType[:2],clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(timeSince[phase][blockRew][epoch][prevTrialType][stim],respTimeNorm[phase][blockRew][epoch][stim]):
                        n.append(np.full(x.size,np.nan))
                        p.append(np.full(x.size,np.nan))
                        for i,t in enumerate(timeBins[:-1]):
                            j = (d >= t) & (d < timeBins[i+1])
                            j = j & ~np.isnan(r)
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(x,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.08,0.1,0.04))
                ax.set_xlim([0,90])
                ax.set_ylim([-0.08,0.08])
                ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                ax.set_ylabel('Response time (s)\n(difference from mean)',fontsize=14)
                # ax.set_title(epoch)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=10)
                plt.tight_layout()

# absolute response times
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for blockRew in ('vis1','sound1'):
            for epoch in ('full',):
                fig = plt.figure()#(figsize=(8,4.5))
                ax = fig.add_subplot(1,1,1)
                clrs = 'mgmg' if blockRew=='sound1' else 'gmgm'
                for stim,clr,ls in zip(stimType[:2],clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(timeSince[phase][blockRew][epoch][prevTrialType][stim],respTime[phase][blockRew][epoch][stim]):
                        n.append(np.full(x.size,np.nan))
                        p.append(np.full(x.size,np.nan))
                        for i,t in enumerate(timeBins[:-1]):
                            j = (d >= t) & (d < timeBins[i+1])
                            j = j & ~np.isnan(r)
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(x,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.3,0.7,0.1))
                ax.set_xlim([0,90])
                ax.set_ylim([0.35,0.6])
                ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                ax.set_ylabel('Response time (s)',fontsize=14)
                # ax.set_title(epoch)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=10)
                plt.tight_layout()

stim = 'non-rewarded target'        
for phase in ('after learning',):
    for prevTrialType in prevTrialTypes:    
        fig = plt.figure()#(figsize=(10,6))
        ax = fig.add_subplot(1,1,1)
        for epoch,clr in zip(blockEpochs,'rbk'):
            n = []
            p = []
            for d,r in zip(timeSince[phase]['all'][epoch][prevTrialType][stim],resp[phase]['all'][epoch][stim]):
                n.append(np.full(x.size,np.nan))
                p.append(np.full(x.size,np.nan))
                for i,t in enumerate(timeBins[:-1]):
                    j = (d >= t) & (d < timeBins[i+1])
                    n[-1][i] = j.sum()
                    p[-1][i] = r[j].sum() / n[-1][i]
            m = np.nanmean(p,axis=0)
            s = np.nanstd(p,axis=0) / (len(p)**0.5)
            ax.plot(x,m,color=clr,label=(epoch+' block' if epoch=='full' else epoch+' of block'))
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,90])
        ax.set_yticks(np.arange(0,1,0.1))
        ax.set_ylim([0.3,0.75])
        ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
        ax.set_ylabel('Response rate to non-rewarded target',fontsize=14)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        plt.tight_layout()
        
for prevTrialType in prevTrialTypes:    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    n = []
    p = []
    for d,r in zip(timeSince[phase]['all']['full'][prevTrialType][stim],resp[phase]['all']['full'][stim]):
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
    ax.set_ylim([0.3,0.8])
    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=16)
    ax.set_ylabel('Response rate',fontsize=16)
    plt.tight_layout()

fig = plt.figure()#(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
t = x
m,s = [y['response to rewarded target']['non-rewarded target'][key] for key in ('mean','sem')]
f1 = lambda t,tau,a,b: a * np.exp(-t/tau) + b
f2 = lambda t,tau,a,b: b - a * np.exp(-t/tau)
func = lambda t,tau1,tau2,a1,b1,a2,b2: (a1 * np.exp(-t/tau1) + b1) + (b2 - a2 * np.exp(-t/tau2))
tau1,tau2,a1,b1,a2,b2 = scipy.optimize.curve_fit(func,t[1:],m[1:],p0=(10,100,0.1,0,1,0.8),bounds=((3,20,0,0,0,0),(30,200,1,0.0001,1,1)))[0]
# ax.plot(t,m,'m',lw=3,label='non-rewarded target')
ax.fill_between(t,m-s,m+s,color='m',alpha=0.25,label='non-rewarded target')
ax.plot(t[1:],func(t[1:],tau1,tau2,a1,b1,a2,b2),'k',label='fit (2 exponential functions)          ')
ax.plot(t[1:],f1(t[1:],tau1,a1,b1),'r--',label='effect of reward bias')
ax.plot(t[1:],f2(t[1:],tau2,a2,b2),'b--',label='effect of context forgetting')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,90])
ax.set_yticks(np.arange(-0.5,0.5,0.1))
ax.set_ylim([-0.1,0.2])
ax.set_xlabel('Time since last response to rewarded target (s)',fontsize=14)
ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
# ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
plt.tight_layout()


## response times and performance
respTime = {phase: {stim: {lbl: [] for lbl in ('rewarded','non-rewarded')} for stim in ('vis1','sound1')} for phase in ('initial training','after learning','all')}
respTimeNorm = copy.deepcopy(respTime)
dprime = copy.deepcopy(respTime)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:nInitialTrainingSessions]
        elif phase=='after learning':
            exps = exps[s:]
        for stim in ('vis1','sound1'):
            for d in (respTime,respTimeNorm,dprime):
                for lbl in ('rewarded','non-rewarded'):
                    d[phase][stim][lbl].append([[] for _ in range(len(exps))])
            for sessionInd,obj in enumerate(exps):
                stimTrials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                rtNorm = obj.responseTimes - np.nanmean(obj.responseTimes[stimTrials])
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    lbl = 'rewarded' if stim==rewStim else 'non-rewarded'
                    trials = stimTrials & (obj.trialBlock==blockInd+1)
                    respTime[phase][stim][lbl][mouseInd][sessionInd].append(obj.responseTimes[trials])
                    respTimeNorm[phase][stim][lbl][mouseInd][sessionInd].append(rtNorm[trials])
                    dprime[phase][stim][lbl][mouseInd][sessionInd].append(obj.dprimeOtherModalGo[blockInd])

alim = (0.15,0.8)
for phase in ('initial training','after learning'):
    for stim in ('vis1','sound1'):  
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(alim,alim,'k--')
        for r,nr in zip(respTime[phase][stim]['rewarded'],respTime[phase][stim]['non-rewarded']):
            r,nr = [np.nanmean([np.nanmean(np.concatenate(s)) for s in m]) for m in (r,nr)]
            ax.plot(r,nr,'ko',alpha=0.2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
        ax.set_xlabel('Response time in rewarded blocks (s)',fontsize=14)
        ax.set_ylabel('Response time in non-rewarded blocks (s)',fontsize=14)
        ax.set_title(('visual target' if stim=='vis1' else 'auditory target'),fontsize=14)
        plt.tight_layout()

alim  = (-0.2,0.25)       
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    rtDiff = {stim: [] for stim in ('vis1','sound1')}
    for stim in rtDiff:
        for r,nr in zip(respTime[phase][stim]['rewarded'],respTime[phase][stim]['non-rewarded']):
            r,nr = [np.array([np.nanmean(np.concatenate(s)) for s in m]) for m in (r,nr)]
            rtDiff[stim].append(np.mean(nr - r))
    ax.plot(rtDiff['vis1'],rtDiff['sound1'],'ko',alpha=0.2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Difference in response time, visual target (s)',fontsize=14)
    ax.set_ylabel('Difference in response time, auditory target (s)',fontsize=14)
    plt.tight_layout()
    
alim  = (-0.15,0.2)       
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    rtDiff = {stim: [] for stim in ('vis1','sound1')}
    for rewStim in ('vis1','sound1'):
        nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
        for r,nr in zip(respTimeNorm[phase][rewStim]['rewarded'],respTimeNorm[phase][nonRewStim]['non-rewarded']):
            r,nr = [np.array([np.nanmean(np.concatenate(s)) for s in m]) for m in (r,nr)]
            rtDiff[rewStim].append(np.mean(nr - r))
    ax.plot(rtDiff['vis1'],rtDiff['sound1'],'ko',alpha=0.2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Visual rewarded blocks',fontsize=14)
    ax.set_ylabel('Auditory rewarded blocks',fontsize=14)
    ax.set_title('Difference in normalized response time (s)\n(non-rewarded - rewarded target)',fontsize=14)
    plt.tight_layout()
                    
for stim in ('vis1','sound1'):
    for lbl in ('rewarded','non-rewarded'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,1],[0,1],'k--')
        for x,y in zip(respTime['initial training'][stim][lbl],respTime['after learning'][stim][lbl]):
            x,y = [np.nanmean([np.nanmean(np.concatenate(s)) for s in m]) for m in (x,y)]
            ax.plot(x,y,'ko',alpha=0.2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_aspect('equal')
        ax.set_xlabel('Response time, intitial training (s)')
        ax.set_ylabel('Response time, after learning (s)')
        ax.set_title(stim+' '+lbl)
        plt.tight_layout()

for phase in ('initial training','after learning','all'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xall = []
    yall = []
    for mouseInd in range(len(sessionsToPass)):
        x = np.nanmean(np.concatenate([dprime[phase]['vis1'][lbl][mouseInd] for lbl in ('rewarded','non-rewarded')],axis=1),axis=1)
        y = []
        for stim in ('vis1','sound1'):
            rew,nonrew = [np.array([np.nanmean(np.concatenate(rt)) for rt in respTime[phase][stim][lbl][mouseInd]]) for lbl in ('rewarded','non-rewarded')]
            y.append(nonrew - rew)
        y = np.nanmean(y,axis=0)
        xall.append(x)
        yall.append(y)
    x = np.concatenate(xall)
    y = np.concatenate(yall)
    notNan = ~np.isnan(y)
    x = x[notNan]
    y = y[notNan]
    ax.plot(x,y,'ko',alpha=0.2)
    slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
    xrng = np.array([min(x),max(x)])
    ax.plot(xrng,slope*xrng+yint,'-',color='r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-1,4])
    ax.set_ylim([-0.25,0.5])
    ax.set_xlabel('Cross-modal dprime',fontsize=14)
    ax.set_ylabel('Difference in response time(s)\n(non-rewarded - rewarded)',fontsize=14)
    ax.set_title('r = '+str(np.round(rval,2))+', p = '+'{0:1.1e}'.format(pval))
    plt.tight_layout()

binWidth = 1 
binCenters = np.arange(0,4,binWidth)       
for phase in ('initial training','after learning','all'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xall = []
    yall = []
    for mouseInd in range(len(sessionsToPass)):
        x = np.nanmean(np.concatenate([dprime[phase]['vis1'][lbl][mouseInd] for lbl in ('rewarded','non-rewarded')],axis=1),axis=1)
        y = []
        for stim in ('vis1','sound1'):
            rew,nonrew = [np.array([np.nanmean(np.concatenate(rt)) for rt in respTime[phase][stim][lbl][mouseInd]]) for lbl in ('rewarded','non-rewarded')]
            y.append(nonrew - rew)
        y = np.nanmean(y,axis=0)
        xall.append(x)
        yall.append(y)
    x = np.concatenate(xall)
    y = np.concatenate(yall)
    notNan = ~np.isnan(y)
    x = x[notNan]
    y = y[notNan]
    ym = []
    ys = []
    for b in binCenters:
        i = (x > b-binWidth/2) & (x < b+binWidth/2)
        ym.append(np.median(y[i]))
        ys.append(np.std(y[i]) / (i.sum()**0.5))
    ax.plot(binCenters,ym,'ko')
    for b,m,s in zip(binCenters,ym,ys):
        ax.plot([b,b],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-1,4])
    ax.set_ylim([-0.005,0.065])
    ax.set_xlabel('Cross-modal dprime',fontsize=14)
    ax.set_ylabel('Difference in response time (s)\n(non-rewarded - rewarded)',fontsize=14)
    plt.tight_layout()


## effect of prior reward or response
prevTrialTypes = ('response to rewarded target','response to non-rewarded target','response to same stimulus')
stimTypes = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
respNext = {phase: {prevTrialType: {blockType: {stim: [] for stim in stimTypes} for blockType in ('all','visual','auditory')} for prevTrialType in prevTrialTypes} for phase in ('initial training','after learning')}
respTimeNext = copy.deepcopy(respNext)
respPrev = copy.deepcopy(respNext)
respTimePrev = copy.deepcopy(respNext)
respMean = copy.deepcopy(respNext)
respTimeMean = copy.deepcopy(respNext)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for rewardStim,blockType in zip(('all','vis1','sound1'),('all','visual','auditory')):
            for s in stimTypes:
                for exps,sp in zip(sessionData,sessionsToPass):
                    exps = exps[:nInitialTrainingSessions] if phase=='initial training' else exps[sp:]
                    rn = []
                    rtn = []
                    rp = []
                    rtp = []
                    rm = []
                    rtm = []
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if rewardStim in ('all',rewStim):
                                nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                                if s=='rewarded target':
                                    stim = rewStim
                                elif s=='non-rewarded target':
                                    stim = nonRewStim
                                elif s=='non-target (rewarded modality)':
                                    stim = rewStim[:-1]+'2'
                                else:
                                    stim = nonRewStim[:-1]+'2'
                                stimTrials = np.where(obj.trialStim==stim)[0]
                                blockTrials = np.where(~obj.autoRewardScheduled & (obj.trialBlock==blockInd+1))[0]
                                trials = np.intersect1d(stimTrials,blockTrials)
                                if prevTrialType == 'response to rewarded target':
                                    ind = obj.trialResponse & (obj.trialStim == rewStim)
                                elif prevTrialType == 'response to non-rewarded target':
                                    ind = obj.trialResponse & (obj.trialStim == nonRewStim)
                                elif prevTrialType == 'non-response to non-rewarded target':
                                    ind = ~obj.trialResponse & (obj.trialStim == nonRewStim)
                                elif prevTrialType == 'response to same stimulus':
                                    ind = obj.trialResponse & (obj.trialStim == stim)
                                elif prevTrialType == 'non-response to same stimulus':
                                    ind = ~obj.trialResponse & (obj.trialStim == stim)
                                rn.append(obj.trialResponse[trials][ind[trials-1]])
                                rtn.append(obj.responseTimes[trials][ind[trials-1]])
                                ind = np.concatenate((ind,[False]))
                                rp.append(obj.trialResponse[trials][ind[trials+1]])
                                rtp.append(obj.responseTimes[trials][ind[trials+1]])
                                rm.append(np.mean(obj.trialResponse[trials]))
                                rtm.append(np.nanmean(obj.responseTimes[trials]))
                    if len(rn) > 0:
                        rn = np.concatenate(rn)
                        rtn = np.concatenate(rtn)
                        rp = np.concatenate(rp)
                        rtp = np.concatenate(rtp)
                    respNext[phase][prevTrialType][blockType][s].append(np.nanmean(rn))
                    respTimeNext[phase][prevTrialType][blockType][s].append(np.nanmean(rtn))
                    respPrev[phase][prevTrialType][blockType][s].append(np.nanmean(rp))
                    respTimePrev[phase][prevTrialType][blockType][s].append(np.nanmean(rtp))
                    respMean[phase][prevTrialType][blockType][s].append(np.nanmean(rm))
                    respTimeMean[phase][prevTrialType][blockType][s].append(np.nanmean(rtm))


blockType = 'all'
alim = (0,1.02)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        fig = plt.figure(figsize=(7.5,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(alim,alim,'k--')
        for stim,mec,mfc in zip(stimTypes,'gmgm',('g','m','none','none')):
            ax.plot(respMean[phase][prevTrialType][blockType][stim],respNext[phase][prevTrialType][blockType][stim],'o',mec=mec,mfc=mfc,label=stim)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
        ax.set_xlabel('Response time (s)'+'\n(within-block mean)',fontsize=14)
        ax.set_ylabel('Response rate'+'\n(previous trial '+prevTrialType+')',fontsize=14)
        ax.legend(loc=('upper left' if 'non-response' in prevTrialType else 'lower right'),fontsize=12)
        plt.tight_layout()

blockType = 'all'
alim = (0,1.02)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        fig = plt.figure(figsize=(7.5,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(alim,alim,'k--')
        for stim,mec,mfc in zip(stimTypes,'gmgm',('g','m','none','none')):
            ax.plot(respPrev[phase][prevTrialType][blockType][stim],respNext[phase][prevTrialType][blockType][stim],'o',mec=mec,mfc=mfc,label=stim)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
        ax.set_xlabel('Response rate'+'\n(next trial '+prevTrialType+')',fontsize=14)
        ax.set_ylabel('Response rate'+'\n(previous trial '+prevTrialType+')',fontsize=14)
        ax.legend(loc=('upper left' if 'non-response' in prevTrialType else 'lower right'),fontsize=12)
        plt.tight_layout()

blockType = 'all'
for phase in ('initial training','after learning'):
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot([-1,4],[0,0],'k--')
    for prevTrialType,clr in zip(prevTrialTypes,'gmk'):
        for x,stim in enumerate(stimTypes):
            r = np.array(respNext[phase][prevTrialType][blockType][stim]) - np.array(respPrev[phase][prevTrialType][blockType][stim])
            m = np.nanmean(r)
            s = np.std(r) / (len(r)**0.5)
            ax.plot(x,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=(prevTrialType if x==0 else None))
            ax.plot([x,x],[m-s,m+s],clr,lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(('rewarded target','non-rewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)'))
    ax.set_xlim([-0.25,3.25])
    ax.set_ylim([-0.3,0.3])
    ax.set_ylabel('Response rate conditioned on previous trial relative to\nresponse rate conditioned on next trial',fontsize=12)
    ax.legend(title='Previous/next trial')
    plt.tight_layout()

alim = (0,1.02)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for blockType,stimLabels,clrs in zip(('visual rewarded','auditory rewarded'),
                                             (('visual target','auditory target','visual non-target','auditory non-target'),('auditory target','visual target','auditory non-target','visual non-target')),
                                             ('gmgm','mgmg')):
            fig = plt.figure(figsize=(7.5,5))
            ax = fig.add_subplot(1,1,1)
            ax.plot(alim,alim,'k--')
            for stim,lbl,mec,mfc in zip(stimTypes,stimLabels,clrs,(clrs[0],clrs[1],'none','none')):
                ax.plot(respPrev[phase][prevTrialType][blockType][stim],respNext[phase][prevTrialType][blockType][stim],'o',mec=mec,mfc=mfc,label=lbl)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xlim(alim)
            ax.set_ylim(alim)
            ax.set_aspect('equal')
            ax.set_xlabel('Response rate'+'\n(next trial '+prevTrialType+')',fontsize=14)
            ax.set_ylabel('Response rate'+'\n(previous trial '+prevTrialType+')',fontsize=14)
            ax.legend(loc=('upper left' if 'non-response' in prevTrialType else 'lower right'),fontsize=12)
            ax.set_title(blockType+' blocks',fontsize=14)
            plt.tight_layout()


alim = (0.15,0.95)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for blockType,stimLabels,clrs in zip(('visual','auditory'),
                                             (('visual target','auditory target','visual non-target','auditory non-target'),('auditory target','visual target','auditory non-target','visual non-target')),
                                             ('gmgm','mgmg')):
            fig = plt.figure(figsize=(7.5,5))
            ax = fig.add_subplot(1,1,1)
            ax.plot(alim,alim,'k--')
            for stim,lbl,mec,mfc in zip(stimTypes[:2],stimLabels,clrs,(clrs[0],clrs[1],'none','none')):
                ax.plot(respTimeMean[phase][prevTrialType][blockType][stim],respTimeNext[phase][prevTrialType][blockType][stim],'o',color=mec,mec=mec,mfc=mfc,label=lbl)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xlim(alim)
            ax.set_ylim(alim)
            ax.set_aspect('equal')
            ax.set_xlabel('Response time (s)'+'\n(within-block mean)',fontsize=14)
            ax.set_ylabel('Response time (s)'+'\n(previous trial '+prevTrialType+')',fontsize=14)
            ax.legend(loc='lower right',fontsize=12)
            ax.set_title(blockType+' rewarded blocks',fontsize=14)
            plt.tight_layout()


## intra-block resp correlations
trainingPhases = ('initial training','after learning')
blockRewStim = ('vis1','sound1','all')
blockEpochs = ('first half','last half','full')
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {phase: {blockRew: {epoch: np.zeros((4,len(sessionData),100)) for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
autoCorrRawMat = copy.deepcopy(autoCorrMat)
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
respRateMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {phase: {blockRew: {epoch: np.zeros((4,4,len(sessionData),200)) for epoch in blockEpochs} for blockRew in blockRewStim} for phase in trainingPhases}
corrWithinRawMat = copy.deepcopy(corrWithinMat)
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
corrAcrossMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for phase in trainingPhases:
    for blockRew in blockRewStim:
        for epoch in blockEpochs:
            for m,(exps,sp) in enumerate(zip(sessionData,sessionsToPass)):
                autoCorr = [[] for _ in range(4)]
                autoCorrRaw = copy.deepcopy(autoCorr)
                autoCorrDetrend = copy.deepcopy(autoCorr)
                respRate = copy.deepcopy(autoCorr)
                corrWithin = [[[] for _ in range(4)] for _ in range(4)]
                corrWithinRaw = copy.deepcopy(corrWithin)
                corrWithinDetrend = copy.deepcopy(corrWithin)
                corrAcross = copy.deepcopy(corrWithin)
                for obj in (exps[:nInitialTrainingSessions] if phase=='initial training' else exps[sp:]):    
                    resp = np.zeros((4,obj.nTrials))
                    respShuffled = np.zeros((4,obj.nTrials,nShuffles))
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                            if len(stimTrials) < minTrials:
                                continue
                            r = obj.trialResponse[stimTrials].astype(float)
                            r[r<1] = -1
                            resp[i,stimTrials] = r
                            for z in range(nShuffles):
                                respShuffled[i,stimTrials,z] = np.random.permutation(r)
                    
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockRew not in ('all',rewStim):
                            continue
                        blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                            if len(stimTrials) < minTrials:
                                continue
                            r = resp[i,stimTrials]
                            rs = respShuffled[i,stimTrials]
                            respRate.append(obj.trialResponse[stimTrials].mean())
                            corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                            autoCorr[i].append(corr)
                            autoCorrRaw[i].append(corrRaw)
                            corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                            autoCorrDetrend[i].append(corrDetrend)
                        
                        r = resp[:,blockTrials]
                        rs = respShuffled[:,blockTrials]
                        for i,(r1,rs1) in enumerate(zip(r,rs)):
                            for j,(r2,rs2) in enumerate(zip(r,rs)):
                                if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                    corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                    corrWithin[i][j].append(corr)
                                    corrWithinRaw[i][j].append(corrRaw)
                                    corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                                    corrWithinDetrend[i][j].append(corrDetrend)

                        otherBlocks = [0,2,4] if blockInd in [0,2,4] else [1,3,5]
                        otherBlocks.remove(blockInd)
                        for b in otherBlocks:
                            bTrials = getBlockTrials(obj,b+1,epoch)
                            rOther = resp[:,bTrials]
                            rsOther = respShuffled[:,bTrials]
                            for i,(r1,rs1) in enumerate(zip(rOther,rsOther)):
                                for j,(r2,rs2) in enumerate(zip(r,rs)):
                                    if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                        corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                        corrAcross[i][j].append(corr)
                        
                autoCorrMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorr,axis=1)
                autoCorrRawMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorrRaw,axis=1)
                autoCorrDetrendMat[phase][blockRew][epoch][:,m] = np.nanmean(autoCorrDetrend,axis=1)
                respRateMat[phase][blockRew][epoch][:,m] = np.nanmean(respRate,axis=1)
                    
                corrWithinMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithin,axis=2)
                corrWithinRawMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithinRaw,axis=2)
                corrWithinDetrendMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)
                corrAcrossMat[phase][blockRew][epoch][:,:,m] = np.nanmean(corrAcross,axis=2)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

for d in (autoCorrMat,autoCorrDetrendMat):
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,lbl in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        for phase,clr in zip(trainingPhases,'mg'):
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
    for phase,clr in zip(trainingPhases,'mg'):
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
bw = 0.1
for phase,clr in zip(trainingPhases,'mg'):
    r = np.concatenate([np.concatenate(respRate[phase]['all']['full'][i]) for i in range(4)])
    c = np.concatenate([np.concatenate([np.array(c)[:,1] for c in autoCorrDetrend[phase]['all']['full'][i]]) for i in range(4)])
    # ax.plot(r,c,'o',mec=clr,mfc='none',alpha=0.1,label=phase)
    for i,b in enumerate(np.arange(bw,1,bw*2)):
        d = c[(r>b-bw) & (r<b+bw)]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(b,m,'o',mec=clr,mfc='none',label=(phase if i==0 else None))
        ax.plot([b,b],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
# ax.set_ylim([0,1.01])
ax.set_xlabel('Response rate',fontsize=14)
ax.set_ylabel('Autocorrelation',fontsize=14)
plt.legend()
plt.tight_layout()

for i,stim in enumerate(stimLabels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bw = 0.1
    for phase,clr in zip(trainingPhases,'mg'):
        r = [np.mean(r) for r in respRate[phase]['all']['full'][i]]
        c = [np.mean(np.array(c)[:,1]) for c in autoCorrDetrend[phase]['all']['full'][i]]
        ax.plot(r,c,'o',mec=clr,mfc='none',label=phase)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    # ax.set_ylim([0,1.01])
    ax.set_xlabel('Response rate',fontsize=14)
    ax.set_ylabel('Autocorrelation',fontsize=14)
    ax.set_title(stim.replace('\n',' '),fontsize=14)
    plt.legend(loc='lower right')
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
        for phase,clr in zip(trainingPhases,'mg'):
            mat = corrWithinDetrendMat[phase]['all']['full'][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=phase)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=9)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.03,0.1])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=11)
        if j==0:
            ax.set_ylabel(ylbl,fontsize=11)
        if i==0:
            ax.set_title(xlbl,fontsize=11)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
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
    for phase,clr in zip(trainingPhases,'mg'):
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

for corr in (corrWithinRaw,corrWithin,corrWithinDetrend):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bw = 0.1
    for phase,clr in zip(trainingPhases,'mg'):
        r = np.concatenate([np.concatenate(respRate[phase]['all']['full'][i]) for i in range(4)])
        c = np.concatenate([np.concatenate([np.array(c)[:,1] for c in corr[phase]['all']['full'][i,i]]) for i in range(4)])
        # ax.plot(r,c,'o',mec=clr,mfc='none',alpha=0.1,label=phase)
        for i,b in enumerate(np.arange(bw,1,bw*2)):
            d = c[(r>b-bw) & (r<b+bw)]
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            ax.plot(b,m,'o',mec=clr,mfc='none')
            ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if i==0 else None))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    # ax.set_ylim([0,1.01])
    ax.set_xlabel('Response rate',fontsize=14)
    ax.set_ylabel('Autocorrelation',fontsize=14)
    plt.legend()
    plt.tight_layout() 

for i,stim in enumerate(stimLabels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bw = 0.1
    for phase,clr in zip(trainingPhases,'mg'):
        r = np.concatenate(respRate[phase]['all']['full'][i])
        c = np.concatenate([np.array(c)[:,1] for c in corrWithinDetrend[phase]['all']['full'][i,i]])
        # ax.plot(r,c,'o',mec=clr,mfc='none',alpha=0.1,label=phase)
        for b in (np.arange(bw,1,bw*2)):
            d = c[(r>b-bw) & (r<b+bw)]
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            ax.plot(b,m,'o',mec=clr,mfc='none')
            ax.plot([b,b],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    # ax.set_ylim([0,1.01])
    ax.set_xlabel('Response rate',fontsize=14)
    ax.set_ylabel('Correlation',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()


## intra-block resp rate correlations for clusters
trainingPhases = ('initial training','after learning','all')
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')
blockEpochs = ('first half','last half','full')
nShuffles = 10
minTrials = 3
autoCorrMat = {phase: {epoch: {clust: np.full((4,len(sessionData),100),np.nan) for clust in clustLabels} for epoch in blockEpochs} for phase in trainingPhases}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
autoCorrAcrossMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {phase: {epoch: {clust: np.full((4,4,len(sessionData),200),np.nan) for clust in clustLabels} for epoch in blockEpochs} for phase in trainingPhases}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
corrAcrossMat = copy.deepcopy(corrWithinMat)

for phase in trainingPhases:
    for epoch in blockEpochs:
        for clust in clustLabels:
            for m,(exps,sp) in enumerate(zip(sessionData,sessionsToPass)):
                if phase=='initial training':
                    exps = exps[:nInitialTrainingSessions]
                elif phase=='after learning':
                    exps = exps[sp:]
                autoCorr = [[] for _ in range(4)]
                autoCorrDetrend = copy.deepcopy(autoCorr)
                autoCorrAcross = copy.deepcopy(autoCorr)
                corrWithin = [[[] for _ in range(4)] for _ in range(4)]
                corrWithinDetrend = copy.deepcopy(corrWithin)
                corrAcross = copy.deepcopy(corrWithin)
                for obj in exps:
                    trialCluster = blockClustData['trialCluster'][obj.subjectName][obj.startTime]
                    if clust not in trialCluster:
                        continue
                    
                    resp = np.zeros((4,obj.nTrials))
                    respShuffled = np.zeros((4,obj.nTrials,nShuffles))
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                            if len(stimTrials) < minTrials:
                                continue
                            r = obj.trialResponse[stimTrials].astype(float)
                            r[r<1] = -1
                            resp[i,stimTrials] = r
                            for z in range(nShuffles):
                                respShuffled[i,stimTrials,z] = np.random.permutation(r)
                    
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                        if not np.all(trialCluster[blockTrials]==clust):
                            continue
                        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                            if len(stimTrials) < minTrials:
                                continue
                            r = resp[i,stimTrials]
                            rs = respShuffled[i,stimTrials]
                            corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                            autoCorr[i].append(corr)
                            corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                            autoCorrDetrend[i].append(corrDetrend)
                            
                            otherBlocks = [0,2,4] if blockInd in [0,2,4] else [1,3,5]
                            otherBlocks.remove(blockInd)
                            for b in otherBlocks:
                                bTrials = getBlockTrials(obj,b+1,epoch)
                                if not np.all(trialCluster[bTrials]==clust):
                                    continue
                                sTrials = np.intersect1d(bTrials,np.where(obj.trialStim==s)[0])
                                if len(sTrials) < minTrials:
                                    continue
                                rOther = resp[i,sTrials]
                                rsOther = respShuffled[i,sTrials]
                                corr,corrRaw = getCorrelation(r,rOther,rs,rsOther,100)
                                autoCorrAcross[i].append(corr)
                        
                        r = resp[:,blockTrials]
                        rs = respShuffled[:,blockTrials]
                        for i,(r1,rs1) in enumerate(zip(r,rs)):
                            for j,(r2,rs2) in enumerate(zip(r,rs)):
                                if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                    corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                    corrWithin[i][j].append(corr)
                                    corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                                    corrWithinDetrend[i][j].append(corrDetrend)
    
                        otherBlocks = [0,2,4] if blockInd in [0,2,4] else [1,3,5]
                        otherBlocks.remove(blockInd)
                        for b in otherBlocks:
                            bTrials = getBlockTrials(obj,b+1,epoch)
                            if not np.all(trialCluster[bTrials]==clust):
                                continue
                            rOther = resp[:,bTrials]
                            rsOther = respShuffled[:,bTrials]
                            for i,(r1,rs1) in enumerate(zip(rOther,rsOther)):
                                for j,(r2,rs2) in enumerate(zip(r,rs)):
                                    if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                        corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                        corrAcross[i][j].append(corr)
                
                if len(autoCorr[0]) > 0:
                    for i in range(4):
                        autoCorrMat[phase][epoch][clust][i,m] = np.nanmean(autoCorr[i],axis=0)
                        autoCorrDetrendMat[phase][epoch][clust][i,m] = np.nanmean(autoCorrDetrend[i],axis=0)
                        for j in range(4):
                            corrWithinMat[phase][epoch][clust][i,j,m] = np.nanmean(corrWithin[i][j],axis=0)
                            corrWithinDetrendMat[phase][epoch][clust][i,j,m] = np.nanmean(corrWithinDetrend[i][j],axis=0)
                    
                if len(autoCorrAcross[0]) > 0:
                    for i in range(4):
                        autoCorrAcrossMat[phase][epoch][clust][i,m] = np.nanmean(autoCorrAcross[i],axis=0)
                        for j in range(4):
                            corrAcrossMat[phase][epoch][clust][i,j,m] = np.nanmean(corrAcross[i][j],axis=0)


epoch = 'full'   
for clust in clustLabels:
    fig = plt.figure(figsize=(8,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for phase,clr in zip(trainingPhases[:2],'mgk'):
                mat = corrWithinDetrendMat[phase][epoch][clust][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=phase)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.03,0.05])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            if i==0 and j==1:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()
    
for clust in clustLabels:
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,stim in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        for phase,clr in zip(trainingPhases[:2],'mgk'):
            mat = autoCorrDetrendMat[phase][epoch][clust][i,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,color=clr)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.1,0.3])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
        if i==0:
            ax.set_ylabel('Correlation',fontsize=12)
        ax.set_title(stim,fontsize=12)
    plt.tight_layout()

for clust in clustLabels:
    fig = plt.figure(figsize=(6,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            diffMat = corrWithinMat['all'][epoch][clust][i,j,:,1:] - corrAcrossMat['all'][epoch][clust][i,j,:,1:]
            for mat,clr,lbl in zip((corrWithinMat,corrAcrossMat,diffMat,corrWithinDetrendMat),'rgbk',('within block','across blocks','within - across','within block detrended')):
                if lbl != 'within - across':
                    mat = mat['all'][epoch][clust][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=lbl)
                ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.04,0.08])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            # if i==0 and j==1:
            #     ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()

for clust in clustLabels:
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,stim in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        diffMat = autoCorrMat['all'][epoch][clust][i,:,1:] - autoCorrAcrossMat['all'][epoch][clust][i,:,1:]
        for mat,clr,lbl in zip((autoCorrMat,autoCorrAcrossMat,diffMat,autoCorrDetrendMat),'rgbk',('within block','across blocks','within - across','within block detrended')):
            if lbl != 'within - across':
                mat = mat['all'][epoch][clust][i,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,color=clr)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.1,0.3])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
        if i==0:
            ax.set_ylabel('Correlation',fontsize=12)
        ax.set_title(stim,fontsize=12)
    plt.tight_layout()
    

for clust in clustLabels:
    fig = plt.figure(figsize=(8,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for epoch,clr in zip(blockEpochs[:2],'mgk'):
                mat = corrWithinDetrendMat['all'][epoch][clust][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=epoch)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.03,0.05])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            if i==0 and j==1:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()
    
for clust in clustLabels:
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,stim in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        for epoch,clr in zip(blockEpochs[:2],'mgk'):
            mat = autoCorrDetrendMat['all'][epoch][clust][i,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,color=clr)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.1,0.3])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
        if i==0:
            ax.set_ylabel('Correlation',fontsize=12)
        ax.set_title(stim,fontsize=12)
    plt.tight_layout()


## nogo, noAR, oneReward, rewardOnly, and catchOnly
mice = {'nogo': np.array(summaryDf[summaryDf['nogo']]['mouse id']),
        'noAR': np.array(summaryDf[summaryDf['noAR']]['mouse id']),
        'oneReward': np.array(summaryDf[summaryDf['oneReward']]['mouse id']),
        'rewardOnly': np.array(summaryDf[summaryDf['rewardOnly']]['mouse id']),
        'catchOnly': np.array(summaryDf[summaryDf['catchOnly']]['mouse id'])}

sessionDataVariants = {lbl: [] for lbl in mice}
isFirstExpType = {lbl: [] for lbl in mice}
for lbl,mouseIds in mice.items():
    # if lbl not in ('oneReward','noAR'):
    #     continue
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array([lbl in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessionDataVariants[lbl].append([getSessionData(mid,startTime,lightLoad=True) for startTime in df.loc[sessions,'start time']])
        for task in df['task version']:
            if 'stage 5' in task and any(key in task for key in mice):
                isFirstExpType[lbl].append(lbl in task)
                break

useFirstExpType = False
useFirstExp = False

# block transition intervals
blockTransitionIntervals = []
for lbl in ('rewardOnly','catchOnly'):
    for exps in sessionDataVariants[lbl]:
        for obj in exps:
            for b in range(5):
                blockTransitionIntervals.append(obj.trialStartTimes[obj.trialBlock==b+2][5] - obj.trialStartTimes[obj.trialBlock==b+1][-1])

print(np.min(blockTransitionIntervals),np.max(blockTransitionIntervals),np.median(blockTransitionIntervals),np.mean(blockTransitionIntervals))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(blockTransitionIntervals,bins=np.arange(20,100,5),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time between trials at block transition (s)')
ax.set_ylabel('Count')
plt.tight_layout()
    
            
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials) 
lbl = 'nogo'
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
        y = []
        for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
            if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                if useFirstExp:
                    exps = [exps[0]]
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                        if blockInd > 0 and rewStim==rewardStim:
                            trials = obj.trialStim==stim
                            y[-1].append(np.full(preTrials+postTrials,np.nan))
                            pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                            if lbl=='nogo' and stim==nonRewStim:
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                            else:
                                i = min(postTrials-5,post.size)
                                y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
        ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
        ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
    ax.set_ylabel('Response rate',fontsize=14)
    ax.legend(bbox_to_anchor=(1,1),fontsize=14)
    # ax.set_title(lbl+' ('+str(len(mice[lbl]))+' mice)\n'+blockLabel,fontsize=14)
    plt.tight_layout()
        
# response times
stimNames = ('vis1','sound1')
stimLabels = ('visual target','auditory target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for lbl in sessionDataVariants:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[-2,2],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gm',('-','-')):
            y = []
            for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
                if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                    if useFirstExp:
                        exps = [exps[0]]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                r = (obj.responseTimes-np.nanmean(obj.responseTimes[trials]))/np.nanstd(obj.responseTimes[trials])
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = r[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = r[(obj.trialBlock==blockInd+1) & trials]
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                    y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,20,5))
        ax.set_yticks([-1,-0.5,0,0.5,1]) # [-1,0,1]
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([-0.6,1]) # [-1.5,1.5]
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel('Response time (z score)',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(lbl+' ('+str(len(mice[lbl]))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()
            
# block switch plot, target stimuli only
lbl = 'nogo'
for useFirstTrialLick in (False,True):
    for getDeltaLickProb in (False,True):
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(1,1,1)
        preTrials = 5
        postTrials = 20
        x = np.arange(-preTrials,postTrials)    
        # ax.plot([0,0],[0,1],'--',color='0.5')
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
            y = []
            for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
                if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                    if useFirstExp:
                        exps = [exps[0]]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0:
                                stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'non-rewarded' in stimLbl else rewStim
                                trials = obj.trialStim==stim
                                blockTrials = np.where(obj.trialBlock==blockInd+1)[0]
                                if lbl=='nogo':
                                    if getDeltaLickProb:
                                        firstTarget = np.intersect1d(blockTrials[obj.newBlockNogoTrials:],np.where(np.in1d(obj.trialStim,obj.blockStimRewarded)))[0]
                                        if np.intersect1d(blockTrials[obj.newBlockNogoTrials:],np.where(trials))[0] > firstTarget:
                                            continue
                                    if useFirstTrialLick and not obj.trialResponse[blockTrials][0]:
                                        continue
                                y[-1].append(np.full(preTrials+postTrials,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                                if lbl=='nogo' and stim!=rewStim:
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials:preTrials+i] = post[:i]
                                else:
                                    i = min(postTrials-5,post.size)
                                    y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                    y[-1] = np.nanmean(y[-1],axis=0)
                    if lbl=='nogo' and not getDeltaLickProb and stimLbl=='rewarded target':
                        rewTargResp = y
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            if lbl=='nogo' and useFirstTrialLick:
                if getDeltaLickProb:
                    if stimLbl == 'rewarded target':
                        deltaLickProb['5 non-rewarded targets']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+5]]
                    else:
                        deltaLickProb['5 non-rewarded targets']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+5]]
                elif stimLbl == 'non-rewarded target':
                    deltaLickProb['1 non-rewarded target']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
        ax.set_ylabel('Response rate',fontsize=14)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
        plt.tight_layout()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(1,1,1)
rr = np.array(rewTargResp)[:,[preTrials-1,preTrials+5]]
for r in rr:
    ax.plot([0,1],r,'o-',color='g',mec='g',mfc='none',ms=6,lw=1,alpha=0.2)
mean = np.nanmean(rr,axis=0)
sem = np.nanstd(rr,axis=0)/(len(rr)**0.5)
ax.plot([0,1],mean,'o-',color='g',mec='g',mfc='g',ms=10,lw=2)
# for x,m,s in zip([0,1],mean,sem):
#     ax.plot([x,x],[m-s,m+s],color='m',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks([0,1])
ax.set_yticks([0,0.5,1])
ax.set_xticklabels(('last trial of\nprevious block','first trial of\nnew block'))
ax.set_ylabel('Response rate',fontsize=16)
ax.set_xlim([-0.2,1.2])
ax.set_ylim([0,1.01])
plt.tight_layout()

    
# block switch plots by first target and response type pooled across mice
lbl = 'noAR'
trialsSinceRewardRange = np.arange(1,4) # (None,) or np.arange(1,4)
py = []
cy = []
for blockRew in ('all',): # ('all',) or ('vis1','sound1')
    for firstTarget in ('rewarded','non-rewarded'):
        for firstTrialLick,lickLbl in zip((True,False),('lick','no lick')):
            for nTarg in (range(1,3) if blockRew=='all' else (1,)):
                for trialsSinceReward in trialsSinceRewardRange:
                    if trialsSinceReward is not None and (firstTarget!='rewarded' or not firstTrialLick or nTarg>1):
                        continue
                    fig = plt.figure()#(figsize=(8,4))
                    ax = fig.add_subplot(1,1,1)
                    preTrials = 5
                    transTrials = 0 if lbl=='noAR' else 5
                    postTrials = 16
                    x = np.arange(-preTrials,transTrials+postTrials)    
                    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=transTrials+nTarg,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
                    clrs = 'mg' if blockRew=='sound1' else 'gm'
                    for stimLbl,clr in zip(('rewarded target','non-rewarded target'),clrs):
                        y = []
                        for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
                            if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                                if useFirstExp:
                                    exps = [exps[0]]
                                for obj in exps:
                                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                        if blockInd > 0 and blockRew in ('all',rewStim):
                                            blockTrials = obj.trialBlock==blockInd+1
                                            nonRewStim = np.setdiff1d(obj.blockStimRewarded,rewStim)
                                            rewStimTrials = np.where(blockTrials & (obj.trialStim==rewStim))[0]
                                            nonRewStimTrials = np.where(blockTrials & (obj.trialStim==nonRewStim))[0]
                                            if ((firstTarget=='rewarded' and rewStimTrials[nTarg-1] < nonRewStimTrials[0] < rewStimTrials[nTarg]) or
                                                (firstTarget=='non-rewarded' and nonRewStimTrials[nTarg-1] < rewStimTrials[0] < nonRewStimTrials[nTarg])):
                                                firstTargetTrial = rewStimTrials[:nTarg] if firstTarget=='rewarded' else nonRewStimTrials[:nTarg]
                                                if np.all(obj.trialResponse[firstTargetTrial] == firstTrialLick):
                                                    if (trialsSinceReward is not None and 
                                                        ((trialsSinceReward < 3 and nonRewStimTrials[0] - rewStimTrials[nTarg-1] != trialsSinceReward) or
                                                         (trialsSinceReward == 3 and not 2 < nonRewStimTrials[0] - rewStimTrials[nTarg-1] < 6))):
                                                        continue
                                                    stim = nonRewStim if 'non-rewarded' in stimLbl else rewStim
                                                    trials = obj.trialStim==stim
                                                    y.append(np.full(preTrials+transTrials+postTrials,np.nan))
                                                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                                    i = min(preTrials,pre.size)
                                                    y[-1][preTrials-i:preTrials] = pre[-i:]
                                                    post = obj.trialResponse[blockTrials & trials]
                                                    if (firstTarget=='rewarded' and stim==rewStim) or (firstTarget=='non-rewarded' and stim==nonRewStim):
                                                        i = min(postTrials,post.size)
                                                        y[-1][preTrials+transTrials:preTrials+transTrials+i] = post[:i]
                                                    else:
                                                        i = min(postTrials-1,post.size) - (nTarg-1)
                                                        y[-1][preTrials+transTrials+nTarg:preTrials+transTrials+nTarg+i] = post[:i]
                        if len(y)>0:
                            p = np.nanmean(y,axis=0)
                            n = len(y)
                            ci = [b/n for b in scipy.stats.binom.interval(0.95,n,p)]
                            ci[0][p==1] = 1
                            ci[1][p==1] = 1
                            ax.plot(x[:preTrials],p[:preTrials],color=clr,label=stimLbl)
                            ax.fill_between(x[:preTrials],ci[1][:preTrials],ci[0][:preTrials],color=clr,alpha=0.25)
                            ax.plot(x[preTrials:],p[preTrials:],color=clr)
                            ax.fill_between(x[preTrials:],ci[1][preTrials:],ci[0][preTrials:],color=clr,alpha=0.25)
                            if trialsSinceReward is None and lbl == 'noAR' and firstTrialLick and nTarg==1:
                                if firstTarget=='rewarded' and stimLbl=='non-rewarded target':
                                    deltaLickProb['1 rewarded target']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
                                elif firstTarget=='non-rewarded' and stimLbl=='rewarded target':
                                    deltaLickProb['1 non-rewarded target']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
                            if trialsSinceReward is not None and stimLbl=='non-rewarded target':
                                py.append(p[preTrials+nTarg])
                                cy.append([ci[0][preTrials+nTarg],ci[1][preTrials+nTarg]])
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
                    ax.set_xticks([-5,-1]+[transTrials+nTarg-1+i for i in (1,5,10,15)])
                    ax.set_xticklabels([-5,-1,1,5,10,15])
                    ax.set_yticks([0,0.5,1])
                    ax.set_xlim([-preTrials-0.5,transTrials+postTrials-0.5])
                    ax.set_ylim([0,1.01])
                    ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
                    ax.set_ylabel('Response rate',fontsize=16)
                    # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
                    ax.set_title(str(len(sessionDataVariants[lbl]))+' mice, '+str(n)+' blocks',fontsize=12)
                    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(len(py)) + 1
ax.plot(x,py,'o',mec='k',mfc='k',ms=10)
for i,c in zip(x,cy):
    ax.plot([i,i],c,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['1','2','3-5'])
ax.set_xlim([0.75,x[-1]+0.25])
ax.set_ylim([0,1.02])
ax.set_xlabel('# trials since first reward',fontsize=14)
ax.set_ylabel('Response prob. on first\nnon-rewarded target trial',fontsize=14)
plt.tight_layout()

# block switch plots by first target type
lbl = 'noAR'
for firstTarget in ('rewarded','non-rewarded'):
    for firstTrialLick in (None,True,False):
        fig = plt.figure() #(figsize=(8,4))
        ax = fig.add_subplot(1,1,1)
        preTrials = 5
        transTrials = 0 if lbl=='noAR' else 5
        postTrials = 16
        x = np.arange(-preTrials,transTrials+postTrials)    
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=transTrials+1,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
            n = 0
            y = []
            for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
                if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                    if useFirstExp:
                        exps = [exps[0]]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0:
                                blockTrials = obj.trialBlock==blockInd+1
                                nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                                rewStimTrials = np.where(blockTrials & (obj.trialStim==rewStim))[0]
                                nonRewStimTrials = np.where(blockTrials & (obj.trialStim==nonRewStim))[0]
                                if ((firstTarget=='rewarded' and rewStimTrials[0] < nonRewStimTrials[0]) or
                                    (firstTarget=='non-rewarded' and nonRewStimTrials[0] < rewStimTrials[0])):
                                    if firstTrialLick is not None:
                                        firstTargetTrial = min(rewStimTrials[0],nonRewStimTrials[0])
                                        if obj.trialResponse[firstTargetTrial] != firstTrialLick:
                                            continue
                                    stim = nonRewStim if 'non-rewarded' in stimLbl else rewStim
                                    trials = obj.trialStim==stim
                                    y[-1].append(np.full(preTrials+transTrials+postTrials,np.nan))
                                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = obj.trialResponse[blockTrials & trials]
                                    if (firstTarget=='rewarded' and stim==rewStim) or (firstTarget=='non-rewarded' and stim==nonRewStim):
                                        i = min(postTrials,post.size)
                                        y[-1][-1][preTrials+transTrials:preTrials+transTrials+i] = post[:i]
                                    else:
                                        i = min(postTrials-1,post.size)
                                        y[-1][-1][preTrials+transTrials+1:preTrials+transTrials+1+i] = post[:i]
                    if len(y[-1]) > 0:
                        n += len(y[-1])
                        y[-1] = np.nanmean(y[-1],axis=0)
                    else:
                        y[-1] = np.full(preTrials+transTrials+postTrials,np.nan)
            if len(y)>0:
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
                ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
                ax.plot(x[preTrials:],m[preTrials:],color=clr)
                ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
                if lbl in ('rewardOnly','catchOnly'):
                    key = '5 rewards' if lbl=='rewardOnly' else '5 catch trials'
                    if firstTarget=='rewarded' and stimLbl == 'rewarded target':
                        deltaLickProb[key]['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+5]]
                    elif firstTarget=='non-rewarded' and stimLbl == 'non-rewarded target':
                        deltaLickProb[key]['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+5]]
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks([-5,-1]+[transTrials+i for i in (1,5,10,15)])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,transTrials+postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
        # ax.set_title(lbl+'\n'+firstTarget+' target first, '+str(len(y))+' mice, '+str(n)+' blocks')
        plt.tight_layout()
            
# block switch plots with non-target trials first
lbl = 'catchOnly'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
preTrials = 5
transTrials = 5
postTrials = 16
x = np.arange(-preTrials,transTrials+postTrials)    
ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=transTrials+1,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
for stimLbl,clr in zip(('non-target (rewarded modality)','non-target (unrewarded modality'),'gm'):
    n = 0
    y = []
    for exps,isFirstType in zip(sessionDataVariants[lbl],isFirstExpType[lbl]):
        if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
            if useFirstExp:
                exps = [exps[0]]
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        blockTrials = obj.trialBlock==blockInd+1
                        nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                        stim = nonRewStim[:-1]+'2' if 'unrewarded' in stimLbl else rewStim[:-1]+'2'
                        trials = obj.trialStim==stim
                        firstTrial = np.where(blockTrials & trials)[0][0]
                        firstOther = np.where(blockTrials & ~np.in1d(obj.trialStim,('stim','catch')))[0][0]
                        if firstTrial > firstOther:
                            continue
                        y[-1].append(np.full(preTrials+transTrials+postTrials,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = obj.trialResponse[blockTrials & trials]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+transTrials:preTrials+transTrials+i] = post[:i]
            if len(y[-1]) > 0:
                n += len(y[-1])
                y[-1] = np.nanmean(y[-1],axis=0)
            else:
                y[-1] = np.full(preTrials+transTrials+postTrials,np.nan)
    if len(y)>0:
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,ls='--',label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks([-5,-1]+[transTrials+i for i in (1,5,10,15)])
ax.set_xticklabels([-5,-1,1,5,10,15])
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,transTrials+postTrials-0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
ax.set_ylabel('Response rate',fontsize=16)
# ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()
    
# block switch plots for one reward variant      
lbl = 'oneReward'
py = {'lick': [], 'no lick': []}
cy = {'lick': [], 'no lick': []}
for firstTrialLick,lickLbl in zip((True,False),('lick','no lick')):
    for nTarg in range(0,4):
        fig = plt.figure() #(figsize=(8,4))
        ax = fig.add_subplot(1,1,1)
        preTrials = 5
        postTrials = 16
        x = np.arange(-preTrials,postTrials)    
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=nTarg,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
            y = []
            for exps in sessionDataVariants[lbl]:
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0:
                                nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                                blockTrials = obj.trialBlock==blockInd+1
                                rewStimTrials = np.where(blockTrials & (obj.trialStim==rewStim))[0]
                                nonRewStimTrials = np.where(blockTrials & (obj.trialStim==nonRewStim))[0]
                                if nTarg==0 or (rewStimTrials[nTarg-1] < nonRewStimTrials[0] < rewStimTrials[nTarg] and (nTarg==1 or np.all(obj.trialResponse[rewStimTrials[1:nTarg+1]]))):
                                    # if nonRewStimTrials[0] - rewStimTrials[nTarg-1] < 2:
                                    #     continue
                                    if obj.trialResponse[blockTrials][0] == firstTrialLick:
                                        if not firstTrialLick:
                                            rewTime = obj.rewardTimes[np.searchsorted(obj.rewardTimes,obj.stimStartTimes[blockTrials][0])]
                                            lickTime = obj.lickTimes[np.searchsorted(obj.lickTimes,rewTime)]
                                            if lickTime - rewTime > 1:
                                                continue
                                        stim = nonRewStim if stimLbl=='non-rewarded target' else rewStim
                                        trials = obj.trialStim==stim
                                        y.append(np.full(preTrials+postTrials,np.nan))
                                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                        i = min(preTrials,pre.size)
                                        y[-1][preTrials-i:preTrials] = pre[-i:]
                                        post = obj.trialResponse[blockTrials & trials]
                                        if stim==rewStim:
                                            i = min(postTrials,post.size)
                                            y[-1][preTrials:preTrials+i] = post[:i]
                                        else:
                                            i = min(postTrials-1,post.size) - (nTarg-1)
                                            y[-1][preTrials+nTarg:preTrials+nTarg+i] = post[:i]
            if len(y)>0:
                p = np.nanmean(y,axis=0)
                n = len(y)
                ci = [b/n for b in scipy.stats.binom.interval(0.95,n,p)]
                ci[0][p==1] = 1
                ci[1][p==1] = 1
                ax.plot(x[:preTrials],p[:preTrials],color=clr,label=stimLbl)
                ax.fill_between(x[:preTrials],ci[1][:preTrials],ci[0][:preTrials],color=clr,alpha=0.25)
                ax.plot(x[preTrials:],p[preTrials:],color=clr)
                ax.fill_between(x[preTrials:],ci[1][preTrials:],ci[0][preTrials:],color=clr,alpha=0.25)
                if stimLbl=='non-rewarded target':
                    if nTarg == 0:
                        py[lickLbl].append(p[preTrials-1])
                        cy[lickLbl].append([ci[0][preTrials-1],ci[1][preTrials-1]])
                    else:
                        py[lickLbl].append(p[preTrials+nTarg])
                        cy[lickLbl].append([ci[0][preTrials+nTarg],ci[1][preTrials+nTarg]])
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks([-5,-1]+[nTarg-1+i for i in (1,5,10,15)])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
        ax.set_ylabel('Response rate',fontsize=14)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
        ax.set_title(str(len(sessionDataVariants[lbl]))+' mice, '+str(n)+' blocks',fontsize=12)
        plt.tight_layout()
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(4)
for key,clr,lbl in zip(py,('k','0.5'),('response to first rewarded target','no response to first rewarded target')):
    ax.plot(x,py[key],'o',mec=clr,mfc=clr,ms=10,label=lbl)
    for i,c in zip(x,cy[key]):
        ax.plot([i,i],c,clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['last trial\nprevious block','1','2','3'])
ax.set_xlim([-0.25,3.25])
ax.set_ylim([0,1.01])
ax.set_xlabel('# rewarded trials',fontsize=14)
ax.set_ylabel('Response prob. on first\nnon-rewarded target trial',fontsize=14)
ax.legend()
plt.tight_layout()
    
# response times to non-contingent reward
rt = []
for exps in sessionDataVariants['oneReward']:
    rt.append([])
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            trial = np.where(obj.trialBlock==blockInd+1)[0][0]
            if not obj.trialResponse[trial]:
                rewTime = obj.rewardTimes[np.searchsorted(obj.rewardTimes,obj.stimStartTimes[trial])]
                lickTime = obj.lickTimes[np.searchsorted(obj.lickTimes,rewTime)]
                rt[-1].append(lickTime-rewTime)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r in rt:
    dsort = np.sort(r)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color='k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([0,3])
ax.set_ylim([0,1.01])
ax.set_xlabel('Time from non-contingent reward delivery to lick (s)',fontsize=16)
ax.set_ylabel('Cumalative fraction',fontsize=16)
plt.tight_layout() 

rt = []
for exps in sessionData:
    rt.append([])
    for obj in exps:
        if obj.autoRewardOnsetFrame==60: # and obj.rigName not in ('B1','B2','B3','B4','B5','B6'):
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                trial = np.where(obj.trialBlock==blockInd+1)[0][0]
                if not obj.trialResponse[trial]:
                    rewTime = obj.rewardTimes[np.searchsorted(obj.rewardTimes,obj.stimStartTimes[trial])]
                    lickTime = obj.lickTimes[np.searchsorted(obj.lickTimes,rewTime)]
                    rt[-1].append(lickTime-rewTime)


## change in lick prob summary
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
xlabels = deltaLickProbLabels[:-2]
xlim = [-0.5,len(xlabels)-0.5]
ax.plot(xlim,[0,0],'k--')
for x,lbl in enumerate(xlabels):
    for stim,clr in zip(('rewTarg','nonRewTarg'),'gm'):
        if not np.all(np.isnan(deltaLickProb[lbl][stim])):
            d = np.diff(deltaLickProb[lbl][stim],axis=1)
            m = d.mean()
            s = d.std()/(len(d)**0.5)
            ax.plot(x,m,'o',color=clr)
            ax.plot([x,x],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.set_xlim(xlim)
ax.set_ylabel(r'$\Delta$ Response rate',fontsize=12)
plt.tight_layout()

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
xlim = [-0.5,len(xlabels)-0.5]
for x,lbl in enumerate(deltaLickProbLabels):
    for stim,clr in zip(('rewTarg','nonRewTarg'),'gm'):
        if not np.all(np.isnan(deltaLickProb[lbl][stim])):
            d = deltaLickProb[lbl][stim]
            mean = d.mean(axis=0)
            sem = d.std(axis=0)/(len(d)**0.5)
            ax.plot([x-0.25,x+0.25],mean,'o-',color=clr,mec=clr,mfc=clr)
            for dx,m,s in zip((x-0.25,x+0.25),mean,sem):
                ax.plot([dx,dx],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.set_xlim(xlim)
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1])
ax.set_ylabel('Response rate',fontsize=16)
plt.tight_layout()


## change in lick prob matrix
labels = ('5 rewarded targets','5 non-rewarded targets','1 rewarded target','1 non-rewarded target')
xlabels = ('5 rewarded\ntarget trials','5 non-rewarded\ntarget trials','1 rewarded\ntarget trial','1 non-rewarded\ntarget trial')

fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(1,1,1)
xlim = [-0.5,len(xlabels)-0.5]
for x,lbl in enumerate(labels):
    for stim,clr in zip(('rewTarg','nonRewTarg'),'gm'):
        d = deltaLickProb[lbl][stim]
        mean = d.mean(axis=0)
        sem = d.std(axis=0)/(len(d)**0.5)
        ax.plot([x-0.25,x+0.25],mean,'o-',color=clr,mec=clr,mfc=clr)
        for dx,m,s in zip((x-0.25,x+0.25),mean,sem):
            ax.plot([dx,dx],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.set_xlim(xlim)
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1])
ax.set_ylabel('Response rate',fontsize=16)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d = np.zeros((2,2))
d[0,0] = np.mean(np.diff(deltaLickProb['1 rewarded target']['rewTarg'],axis=1))
d[0,1] = np.mean(np.diff(deltaLickProb['1 non-rewarded target']['nonRewTarg'],axis=1))
d[1,0] = np.mean(np.diff(deltaLickProb['1 rewarded target']['nonRewTarg'],axis=1))
d[1,1] = np.mean(np.diff(deltaLickProb['1 non-rewarded target']['rewTarg'],axis=1))
im = ax.imshow(d,cmap='bwr',clim=(-1,1))
for i in (0,1):
    for j in (0,1):
        txt = round(d[i,j],2)
        if txt > 0:
            txt = '+' + str(txt)
        ax.text(j,i,txt,ha='center',va='center',fontsize=12)
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.04)
cb.set_ticks([-1,-0.5,0,0.5,1])
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([0,1])
ax.set_xticklabels(['Reward','No reward'])
ax.set_xlabel('Previous trial outcome',fontsize=14)
ax.set_yticks([0,1])
ax.set_yticklabels(['Same as\ncurrent trial','Different from\ncurrent trial'])
ax.set_ylabel('Previous trial stimulus modality',fontsize=14)
ax.set_title('Change in response prob. to current trial stimulus',fontsize=14)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d = np.zeros((2,2))
d[0,0] = np.mean(np.diff(deltaLickProb['5 rewarded targets']['rewTarg'],axis=1))
d[0,1] = np.mean(np.diff(deltaLickProb['5 non-rewarded targets']['nonRewTarg'],axis=1))
d[1,0] = np.mean(np.diff(deltaLickProb['5 rewarded targets']['nonRewTarg'],axis=1))
d[1,1] = np.mean(np.diff(deltaLickProb['5 non-rewarded targets']['rewTarg'],axis=1))
im = ax.imshow(d,cmap='bwr',clim=(-1,1))
for i in (0,1):
    for j in (0,1):
        txt = round(d[i,j],2)
        if txt > 0:
            txt = '+' + str(txt)
        ax.text(j,i,txt,ha='center',va='center',fontsize=12)
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.04)
cb.set_ticks([-1,-0.5,0,0.5,1])
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([0,1])
ax.set_xticklabels(['Reward','No reward'])
ax.set_xlabel('Previous trial outcome',fontsize=14)
ax.set_yticks([0,1])
ax.set_yticklabels(['Same as\ncurrent trial','Different from\ncurrent trial'])
ax.set_ylabel('Previous trial stimulus modality',fontsize=14)
ax.set_title('Change in response prob. to current trial stimulus',fontsize=14)
plt.tight_layout()


## change in lick prob summary (reward only or catch only)
labels = ('5 rewarded targets','5 rewards')
xlabels = ('5 rewarded\ntarget trials','5 rewards\n(no stimulus)')

labels = ('5 non-rewarded targets','5 catch trials')
xlabels = ('5 non-rewarded\ntarget trials','5 catch trials\n(no stimulus or reward)')

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)
xlim = [-0.5,len(xlabels)-0.5]
for x,lbl in enumerate(labels):
    for stim,clr in zip(('rewTarg','nonRewTarg'),'gm'):
        if not np.all(np.isnan(deltaLickProb[lbl][stim])):
            d = deltaLickProb[lbl][stim]
            mean = d.mean(axis=0)
            sem = d.std(axis=0)/(len(d)**0.5)
            ax.plot([x-0.25,x+0.25],mean,'o-',color=clr,mec=clr,mfc=clr)
            for dx,m,s in zip((x-0.25,x+0.25),mean,sem):
                ax.plot([dx,dx],[m-s,m+s],clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.set_xlim(xlim)
ax.set_yticks([0,0.5,1])
ax.set_ylim([0,1])
ax.set_ylabel('Response rate',fontsize=16)
plt.tight_layout()


## no reward blocks
mice = np.array(summaryDf[summaryDf['no reward']]['mouse id'])

sessionDataNoRew = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['no reward' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    sessionDataNoRew.append([getSessionData(mid,startTime) for startTime in df.loc[sessions,'start time']])

# block switch plot, target stimuli only
for ylbl in ('Response rate','Response time (s)\n(difference from mean)'):
    for blockRewarded,title in zip((True,False),('switch to rewarded block','switch to unrewarded block')):
        fig = plt.figure() #(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        preTrials = 15
        if blockRewarded:
            postTrials = 20
            x = np.arange(-preTrials,postTrials)
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        else:
            postTrials = 15
            x = np.arange(-preTrials,postTrials+1)
            ax.plot([0,0],[-1,1],'--',color='0.5') 
        for stimLbl,clr in zip(('previously rewarded target stim','other target stim'),'mg'):
            y = []
            for exps in sessionDataNoRew:
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0 and ((blockRewarded and rewStim != 'none') or (not blockRewarded and rewStim == 'none')):
                            if blockRewarded:
                                stim = np.setdiff1d(('vis1','sound1'),rewStim) if 'previously' in stimLbl else rewStim
                            else:
                                prevRewStim = obj.blockStimRewarded[blockInd-1]
                                stim = np.setdiff1d(('vis1','sound1'),prevRewStim) if 'other' in stimLbl else prevRewStim
                            trials = obj.trialStim==stim
                            r = obj.trialResponse if ylbl=='Response rate' else obj.responseTimes - np.nanmean(obj.responseTimes[trials])
                            y[-1].append(np.full(x.size,np.nan))
                            pre = r[(obj.trialBlock==blockInd) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = r[(obj.trialBlock==blockInd+1) & trials]
                            if blockRewarded:
                                if stim==rewStim:
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials:preTrials+i] = post[:i]
                                else:
                                    i = min(postTrials-5,post.size)
                                    y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                            else:
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]  
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        if blockRewarded:
            ax.set_xticks([-15,-10,-5,-1,5,9,14,19])
            ax.set_xticklabels([-15,-10,-5,-1,1,5,10,15])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        else:
            ax.set_xticks([-15,-10,-5,-1,1,5,10,15])
            ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        if ylbl=='Response rate':
            ax.set_yticks([0,0.5,1])
            ax.set_ylim([0,1.01])
        else:
            ax.set_yticks(np.arange(-0.2,0.2,0.05))
            ax.set_ylim([-0.15,0.2])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel(ylbl,fontsize=12)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
        # ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
        plt.tight_layout()
    
# resp rate correlations comparing reward and no reward blocks
blockTypes = ('rewarded','unrewarded')
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {blockType: np.zeros((4,len(sessionData),100)) for blockType in blockTypes}
autoCorrRawMat = copy.deepcopy(autoCorrMat)
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {blockType: np.zeros((4,4,len(sessionDataNoRew),200)) for blockType in blockTypes}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for blockType in blockTypes:
    for m,exps in enumerate(sessionDataNoRew):
        autoCorr = [[] for _ in range(4)]
        autoCorrDetrend = copy.deepcopy(autoCorr)
        corrWithin = [[[] for _ in range(4)] for _ in range(4)]
        corrWithinDetrend = copy.deepcopy(corrWithin)
        for obj in exps:
            resp = np.zeros((4,obj.nTrials))
            respShuffled = np.zeros((4,obj.nTrials,nShuffles))
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                blockTrials = getBlockTrials(obj,blockInd+1,'full')
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = obj.trialResponse[stimTrials].astype(float)
                    r[r<1] = -1
                    resp[i,stimTrials] = r
                    for z in range(nShuffles):
                        respShuffled[i,stimTrials,z] = np.random.permutation(r)
            
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if (blockType=='rewarded' and rewStim=='none') or (blockType=='unrewarded' and rewStim!='none'):
                    continue
                
                blockTrials = getBlockTrials(obj,blockInd+1,'full')
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = resp[i,stimTrials]
                    rs = respShuffled[i,stimTrials]
                    corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                    autoCorr[i].append(corr)
                    corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                    autoCorrDetrend[i].append(corrDetrend)
                
                r = resp[:,blockTrials]
                rs = respShuffled[:,blockTrials]
                for i,(r1,rs1) in enumerate(zip(r,rs)):
                    for j,(r2,rs2) in enumerate(zip(r,rs)):
                        if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                            corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                            corrWithin[i][j].append(corr)
                            corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                            corrWithinDetrend[i][j].append(corrDetrend)
    
        autoCorrMat[blockType][:,m] = np.nanmean(autoCorr,axis=1)
        autoCorrDetrendMat[blockType][:,m] = np.nanmean(autoCorrDetrend,axis=1)
    
        corrWithinMat[blockType][:,:,m] = np.nanmean(corrWithin,axis=2)
        corrWithinDetrendMat[blockType][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)


stimLabels = {'rewarded': ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)'),
              'unrewarded': ('previously rewarded target','other target','non-target\n(previously rewarded modality)','non-target\n(other modality)')}

for blockType in blockTypes:
    fig = plt.figure(figsize=(6,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels[blockType]):
        for j,xlbl in enumerate(stimLabels[blockType][:2]):
            ax = fig.add_subplot(gs[i,j])
            mat = corrWithinDetrendMat[blockType][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,'k')
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            ax.set_xticks(np.arange(0,25,5))
            ax.set_xlim([0,20])
            ax.set_ylim([-0.025,0.04])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=10)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=10)
            if i==0:
                ax.set_title(xlbl,fontsize=10)
    plt.tight_layout()

for blockType in blockTypes:
    fig = plt.figure(figsize=(4,10))           
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,lbl in enumerate(stimLabels[blockType]):
        ax = fig.add_subplot(gs[i])
        mat = autoCorrDetrendMat[blockType][i,:,1:]
        m = np.nanmean(mat,axis=0)
        s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
        ax.plot(x,m,color='k')
        ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.01,0.03])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
        if i==0:
            ax.set_ylabel('Autocorrelation',fontsize=12)
        ax.set_title(lbl,fontsize=12)
    plt.tight_layout()
    

