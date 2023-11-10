# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
"""

import copy
import glob
import os
import itertools
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData,getPerformanceStats


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

hitThresh = 100
dprimeThresh = 1.5

deltaLickProbLabels = ('5 rewarded/auto-rewarded targets',
                       '1 rewarded target',
                       '1 auto-rewarded target',
                       '5 non-rewarded targets',
                       '1 non-rewarded target',
                       'rewarded target first',
                       #'rewarded target first (no lick)',
                       'non-rewarded target first',
                       # 'non-rewarded target first (no lick)',
                       '5 rewards, no target (first target trial)')
deltaLickProb = {lbl: {targ: np.nan for targ in ('rewTarg','nonRewTarg')} for lbl in deltaLickProbLabels}


def getSessionData(mouseId,df):
    d = []
    for t in df[~df['ignore'].astype(bool)]['start time']:
        fileName = 'DynamicRouting1_' + str(mouseId) + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5'
        filePath = os.path.join(baseDir,'DynamicRoutingTask','Data',str(mouseId),fileName)
        obj = DynRoutData()
        obj.loadBehavData(filePath)
        d.append(obj)
    return d


def getFirstExperimentSession(df):
    experimentSessions = np.where(['multimodal' in task
                                   or 'contrast'in task
                                   or 'opto' in task
                                   or 'nogo' in task
                                   or 'noAR' in task
                                   or 'rewardOnly' in task
                                   # or 'NP' in rig 
                                   for task,rig in zip(df['task version'],df['rig name'])])[0]
    firstExperimentSession = experimentSessions[0] if len(experimentSessions) > 0 else None
    return firstExperimentSession


def plotLearning(mice,stage):
    hitCount = {lbl:[] for lbl in mice}
    dprime = copy.deepcopy(hitCount)
    sessionsToPass = copy.deepcopy(hitCount)
    for lbl,mouseIds in mice.items():
        for mid in mouseIds:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.where(np.array([str(stage) in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool)))[0]
            hitCount[lbl].append([])
            dprime[lbl].append([])
            passed = False
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                hitCount[lbl][-1].append(hits[0][0])
                dprime[lbl][-1].append(dprimeSame[0][0])
                if sessionInd > sessions[0] and not passed:
                    hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
                    if all(h[0] >= hitThresh for h in hits) and all(d[0] >= dprimeThresh for d in dprimeSame):
                        sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0] + 1)
                        passed = True
            if not passed:
                if mid in (614910,684071,682893):
                    sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0]+ 1)
                else:
                    sessionsToPass[lbl].append(np.nan)
                    
    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.5)
    xticks = np.arange(0,100,5) if xlim[1]>10 else np.arange(10)
                
    for data,thresh,ylbl in zip((hitCount,dprime),(hitThresh,dprimeThresh),('Hit count','d\'')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xlim,[thresh]*2,'k--')
        for lbl,clr in zip(mice.keys(),'gm'):
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
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_xlabel('Session',fontsize=14)
        ax.set_ylabel(ylbl,fontsize=14)
        plt.tight_layout()
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(mice.keys(),'gm'):
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
    
    
def plotStage5Learning(mice):
    sessionsToPass = {lbl: [] for lbl in mice}
    dpSame = {lbl: [] for lbl in mice}
    dpOther = {lbl: [] for lbl in mice}
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
                if sessionInd > sessions[0]:
                    hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
                    if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                        sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0]+ 1)
                        break

    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.75)
    xticks = np.arange(0,100,5)
    clrs = 'gmrbc'[:len(mice)]
                
    for dp,ylbl in zip((dpSame,dpOther),('d\' same modality','d\' other modality')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xlim,[dprimeThresh]*2,'k--')
        for lbl,clr in zip(mice.keys(),clrs):
            for d,ps in zip(dp[lbl],sessionsToPass[lbl]):
                d = np.nanmean(d,axis=1)
                ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25,zorder=2)
                ax.plot(ps,d[-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_xlabel('Session',fontsize=14)
        ax.set_ylabel(ylbl,fontsize=14)
        plt.tight_layout()
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(mice.keys(),clrs):
        dsort = np.sort(sessionsToPass[lbl])
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



## stage 1, stationary gratings with or without timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['stat grating'] & ~summaryDf['wheel fixed']
mice = {'stationary, timeouts': np.array(summaryDf[ind & summaryDf['timeouts']]['mouse id']),
        'stationary, no timeouts': np.array(summaryDf[ind & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=1)


## stage 1, stationary vs moving gratings, both with timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['timeouts'] & ~summaryDf['wheel fixed']
mice = {'moving':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id']),
        'stationary': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id'])}
plotLearning(mice,stage=1)

# stage 1, moving gratings with or without reward clicks
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts']
mice = {'moving, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'moving, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=1)

# stage 1, moving gratings with early or late autorewards
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts']
mice = {'moving, early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 1)']]['mouse id']),
        'moving, late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 1)']]['mouse id'])}
plotLearning(mice,stage=1)
                

# stage 2, tones, timeouts with noise vs no timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['tone'] & ~summaryDf['wheel fixed']
mice = {'tones, timeouts': np.array(summaryDf[ind & summaryDf['timeouts']]['mouse id']),
        'tones, no timeouts':  np.array(summaryDf[ind  & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, tones with noise timeouts vs AMN with noiseless timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['timeouts'] & ~summaryDf['wheel fixed']
mice = {'tones': np.array(summaryDf[ind & summaryDf['tone']]['mouse id']),
        'AM noise':  np.array(summaryDf[ind & summaryDf['AM noise']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, AMN with or without reward clicks
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts']
mice = {'AM noise, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'AM noise, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, AMN with early or late autorewwards
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts']
mice = {'AM noise, early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 2)']]['mouse id']),
        'AM noise, late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 2)']]['mouse id'])}
plotLearning(mice,stage=2)


# stage 5, repeats vs no repeats
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass']
mice = {'no repeats': np.array(summaryDf[ind & summaryDf['no repeats (stage 5)']]['mouse id']),
        'repeats': np.array(summaryDf[ind & summaryDf['repeats (stage 5)']]['mouse id'])}
plotStage5Learning(mice)

# stage 5, with or without reward clicks
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] #& summaryDf['no repeats (stage 5)']
mice = {'reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotStage5Learning(mice)

# stage 5, early or late autorewards
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] #& summaryDf['no repeats (stage 5)']
mice = {'early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 5)']]['mouse id']),
        'late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 5)']]['mouse id'])}
plotStage5Learning(mice)


# moving to stationary grating switch
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
ax.set_ylim([0,4.1])
ax.set_xlabel('Session',fontsize=14)
ax.set_ylabel('d\'',fontsize=14)
plt.tight_layout()


## within modality d' after stage 2
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'] 
mice = np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise']]['mouse id'])

dprime = {'vis': [], 'aud': []}
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    for lbl in dprime:
        dprime[lbl].append([])
        for sessionInd in sessions:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
            dprimeSame = dprimeSame[0]
            task = df.loc[sessionInd,'task version']
            if (lbl=='vis' and 'ori AMN' in task) or (lbl=='aud' and 'AMN ori' in task):
                dprime[lbl][-1].append(np.nanmean(dprimeSame[0:6:2]))
            else:
                dprime[lbl][-1].append(np.nanmean(dprimeSame[1:6:2]))

maxSessions = max(len(d) for lbl in dprime for d in dprime[lbl])
minMice = 8
            
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xmax = 1000
for lbl,clr in zip(('vis','aud'),'gm'):
    y = np.full((len(dprime[lbl]),maxSessions+1),np.nan)
    for i,d in enumerate(dprime[lbl]):
        y[i,:len(d)] = d
    lbl += ' (n='+str(len(dprime[lbl]))+')'
    x = np.arange(y.shape[1])+1
    n = np.sum(~np.isnan(y),axis=0)
    xmax = min(xmax,x[n>=minMice][-1])
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,xmax])
ax.set_ylim([0,4])
ax.set_xlabel('Session after stage 2',fontsize=14)
ax.set_ylabel('d\'',fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()

 
# training in stage 5
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
mice = np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & summaryDf['late autoreward (stage 5)']]['mouse id'])

dprime = {comp: {mod: [] for mod in ('all','vis','sound')} for comp in ('same','other')}
sessionsToPass = []
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    passed = False
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
                
        if not passed and sessionInd > sessions[0]:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                sessionsToPass.append(sessionInd - sessions[0] + 1)
                passed = True
    sessionStartTimes = list(df['start time'][sessions])
    dataDir = summaryDf.loc[summaryDf['mouse id']==mid,'data path'].values[0]
    sessionData.append(getSessionData(mid,df))
                
mouseClrs = plt.cm.tab20(np.linspace(0,1,len(sessionsToPass)))

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),max(len(d) for d in dprime[comp]['all'])),np.nan)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)
        ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=2)
        ax.plot(sessionsToPass[i],y[sessionsToPass[i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        dp[i,:len(y)] = y
    m = np.nanmean(dp,axis=0)
    ax.plot(np.arange(len(m))+1,m,color='k',lw=2,zorder=1)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-0.5,4])
    ax.set_xlabel('Session',fontsize=14)
    ax.set_ylabel('d\' '+comp+' modality',fontsize=14)
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
        ax.plot(np.arange(len(m))+1,m,color=clr,lw=2,zorder=1,label=mod)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-3,4])
    ax.set_xlabel('Session',fontsize=14)
    ax.set_ylabel('d\' '+comp+' modality',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
# compare early, late, and after learning
nSessions = 5
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)  
for phase in ('initial training','late training','after learning'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps,s in zip(sessionData,sessionsToPass):
                if len(exps)>0:
                    if phase=='initial training':
                        exps = exps[:nSessions]
                    elif phase=='late training':
                        exps = exps[s-2-nSessions:s-2]
                    else:
                        exps = exps[s:s+nSessions]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
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
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch (excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(phase+'\n'+blockLabel,fontsize=12)
        plt.tight_layout()
    
# d' correlations by session
passOnly = False
combos = list(itertools.combinations(itertools.product(('same','other'),('vis','sound')),2))
r = {c: [] for c in combos}
p = {c: [] for c in combos}
fig = plt.figure(figsize=(6,8))
for i,c in enumerate(combos):
    ax = fig.add_subplot(3,2,i+1)
    alim = [10,-10]
    (compX,modX),(compY,modY) = c
    for j,clr in enumerate(mouseClrs):
        dx,dy = [np.nanmean(dprime[comp][mod][j],axis=1) for comp,mod in zip((compX,compY),(modX,modY))]
        if passOnly:
            ind = slice(sessionsToPass[j]-2,None)
            dx = dx[ind]
            dy = dy[ind]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
        r[c].append(rval)
        p[c].append(pval)
    p[c] = multipletests(p[c],alpha=0.05,method='fdr_bh')[1]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    offset = 0.05*(alim[1]-alim[0])
    alim = [alim[0]-offset,alim[1]+offset]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' '+compX+', '+modX,fontsize=14)
    ax.set_ylabel('d\' '+compY+', '+modY,fontsize=14)
plt.tight_layout()
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('d\' correlation across sessions','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for c,clr in zip(combos,'grmcbk'):
        dsort = np.sort(d[c])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=c)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = 0 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# d' correlations by block
passOnly = False
r = {mod: [] for mod in ('vis','sound')}
p = {mod: [] for mod in ('vis','sound')}
fig = plt.figure(figsize=(6,6))
for i,mod in enumerate(('vis','sound')):
    ax = fig.add_subplot(2,1,i+1)
    alim = [10,-10]
    for j,clr in enumerate(mouseClrs):
        if passOnly:
            dx,dy = [np.ravel(dprime[comp][mod][j][sessionsToPass[j]-2:]) for comp in ('same','other')]
        else:
            dx,dy = [np.ravel(dprime[comp][mod][j]) for comp in ('same','other')]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
        r[mod].append(rval)
        p[mod].append(pval)
    p[mod] = multipletests(p[mod],alpha=0.05,method='fdr_bh')[1]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    offset = 0.05*(alim[1]-alim[0])
    alim = [alim[0]-offset,alim[1]+offset]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' same'+', '+mod,fontsize=14)
    ax.set_ylabel('d\' other'+', '+mod,fontsize=14)
    plt.tight_layout()
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('d\' correlation across blocks','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for mod,clr in zip(('vis','sound'),'rb'):
        dsort = np.sort(d[mod])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=mod)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = -0.05 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# response rate correlations
r = {}
p = {}
for combo in ((('same','vis'),('other','sound')),
              (('same','sound'),('other','vis')),
              ('catch',('same','vis')),
              ('catch',('same','sound')),
              ('catch',('other','vis')),
              ('catch',('other','sound'))):
    r[combo] = []
    p[combo] = []
    for exps,sp in zip(sessionData,sessionsToPass):
        if passOnly:
            exps = exps[sp-2:]
        respRate = [[],[]]
        for obj in exps:
            for i,c in enumerate(combo):
                j = 0 if i==1 else 1
                if (('same' in c and 'vis' in c) or ('other' in c and 'sound' in c) or
                    (c=='catch' and (('same' in c[j] and 'vis' in c[j]) or ('other' in c[j] and 'sound' in c[j])))):
                    blocks = obj.blockStimRewarded=='vis1'
                else:
                    blocks = obj.blockStimRewarded=='sound1'
                if c=='hit':
                    respRate[i].append(np.array(obj.hitRate)[blocks])
                elif c=='catch':
                    respRate[i].append(np.array(obj.catchResponseRate)[blocks])
                elif 'same' in c:
                    respRate[i].append(np.array(obj.falseAlarmSameModal)[blocks])
                elif 'other' in c:
                    respRate[i].append(np.array(obj.falseAlarmOtherModalGo)[blocks])
        x,y = [np.ravel(rr) for rr in respRate]
        notNan = ~np.isnan(x) & ~np.isnan(y)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(x[notNan],y[notNan])
        r[combo].append(rval)
        p[combo].append(pval)
    p[combo] = multipletests(p[combo],alpha=0.05,method='fdr_bh')[1]
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('False alarm rate correlation across blocks','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for combo,clr in zip(r,'rbgmck'):
        dsort = np.sort(d[combo])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=combo)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = -0.05 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# block switch plot by performance quantiles
stimNames = ('vis1','vis2','sound1','sound2','catch')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target','catch')
postTrials = 15
x = np.arange(postTrials)+1
nQuantiles = 3
quantiles = [(i/nQuantiles,(i+1)/nQuantiles) for i in range(nQuantiles)]
for q in quantiles:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmmk',('-','--','-','--','-')):
            y = []
            for exps,sp in zip(sessionData,sessionsToPass):
                exps = exps[sp-2:]
                dp = np.ravel([obj.dprimeOtherModalGo for obj in exps])
                dp[np.isnan(dp)] = 0
                lower,upper = np.quantile(dp,q)
                inQuantile = (dp>lower) & (dp<=upper) if lower>0 else (dp>=lower) & (dp<=upper)
                qBlocks = np.where(inQuantile)[0]
                blockCount = 0
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if rewStim==rewardStim and blockCount in qBlocks:
                            trials = (obj.trialBlock==blockInd+1) & (obj.trialStim==stim) & ~obj.autoRewardScheduled 
                            y[-1].append(np.full(postTrials,np.nan))
                            i = min(postTrials,trials.sum())
                            y[-1][-1][:i] = obj.trialResponse[trials][:i]
                        blockCount += 1
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([0,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(str(q)+blockLabel,fontsize=12)
        plt.tight_layout()
    

## performance after passing
ind = summaryDf['stage 5 pass'] & ~summaryDf['cannula']
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
mice = {#'stationary, tones': np.array(summaryDf[ind & summaryDf['stat grating'] & summaryDf['tone']]['mouse id']),
        #'moving, tones':  np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['tone']]['mouse id']),
        #'moving, AM noise': np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['AM noise']]['mouse id']),
        'moving, AM noise, direct': np.array(summaryDf[ind & ~hasIndirectRegimen & summaryDf['moving grating'] & summaryDf['AM noise']]['mouse id'])}

minSessions = 1

sessionData = {lbl: [] for lbl in mice}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        firstExperimentSession = getFirstExperimentSession(df)
        if firstExperimentSession is not None:
            sessions[firstExperimentSession:] = False
        sessions = np.where(sessions)[0]
        passSession = None
        for i,sessionInd in enumerate(sessions[1:]):
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                passSession = i+1
                break
        sessions = sessions[passSession+1:]
        sessions = [i for i in sessions if 'repeats' not in df.loc[i,'task version']]
        sessionStartTimes = list(df['start time'][sessions])
        dataDir = summaryDf.loc[summaryDf['mouse id']==mid,'data path'].values[0]
        sessionData[lbl].append(getSessionData(mid,dataDir,sessionStartTimes))
          
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)   
for lbl in sessionData:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps in sessionData[lbl]:
                if len(exps) >= minSessions:
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
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
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(lbl+' ('+str(len(y))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()

# response times  
norm = True      
stimNames = ('vis1','sound1')
stimLabels = ('visual target','auditory target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
ylim = [-0.05,0.15] if norm else [0.25,0.65]
ylbl = 'Response Time (s)'
if norm:
    ylbl = r'$\Delta$ ' + ylbl
for lbl in sessionData:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],ylim,'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gm',('-','-')):
            y = []
            for exps in sessionData[lbl]:
                if len(exps) >= minSessions:
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                normRespTime = np.nanmedian(obj.responseTimes[trials]) if norm else 0
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.responseTimes[(obj.trialBlock==blockInd) & trials] - normRespTime
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.responseTimes[(obj.trialBlock==blockInd+1) & trials] - normRespTime
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
        # ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim(ylim)
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding cue trials)',fontsize=12)
        ax.set_ylabel(ylbl,fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(lbl+' ('+str(len(y))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()
        
# block switch plot, target stimuli only
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
ax.plot([0,0],[0,1],'--',color='0.5')
for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
    y = []
    for exps in [exps for lbl in sessionData for exps in sessionData[lbl]]:
        if len(exps) >= minSessions:
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
            y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x,m,color=clr,label=stimLbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    key = 'rewTarg' if stimLbl == 'rewarded target stim' else 'nonRewTarg'
    deltaLickProb['5 rewarded/auto-rewarded targets'][key] = np.array(y)[:,[preTrials-1,preTrials+1]]
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(-20,21,5))
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch (excluding auto-rewards)',fontsize=12)
ax.set_ylabel('Response rate',fontsize=12)
ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
ax.set_title(str(len(y))+' mice',fontsize=12)
plt.tight_layout()

# block switch plot, target stimuli only, delayed auto-rewards
for lbl in ('all blocks','first trial lick','first trial no lick'):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for exps in [exps for lbl in sessionData for exps in sessionData[lbl]]:
            lateAutoRewExps = [obj for obj in exps if obj.autoRewardOnsetFrame>=obj.responseWindow[1]]
            if len(exps) >= minSessions and len(exps)==len(lateAutoRewExps):
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
                y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        if lbl != 'all blocks' and stimLbl == 'rewarded target stim':
            key = '1 rewarded target' if lbl == 'first trial lick' else '1 auto-rewarded target'
            deltaLickProb[key]['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+2]]
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

# d prime
lbl = 'moving, AM noise, direct'
dp = [np.nanmean([obj.dprimeOtherModalGo  for obj in exps]) for exps in sessionData[lbl]]
dpVis,dpAud = [[np.nanmean([np.array(np.array(obj.dprimeOtherModalGo))[obj.blockStimRewarded==stim] for obj in exps]) for exps in sessionData[lbl]] for stim in ('vis1','sound1')]
dpBlocksVisStart,dpBlocksAudStart = [[np.nanmean([obj.dprimeOtherModalGo for obj in exps if obj.blockStimRewarded[0]==stim],axis=0) for exps in sessionData[lbl]] for stim in ('vis1','sound1')]

fig = plt.figure(figsize=(4,5))
ax = fig.add_subplot(1,1,1)
xticks = (0,1)
xlim = (-0.2,1.2)
ax.plot(xlim,[0,0],'k--')
for v,a in zip(dpVis,dpAud):
    ax.plot(xticks,[v,-a],'ko-',mec='k',mfc='none',ms=6,alpha=0.25)
ax.plot(xticks,[np.mean(dpVis),-np.mean(dpAud)],'ko-',ms=10,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(['Vis-rewarded\nblocks','Aud-rewarded\nblocks'])
ax.set_xlim(xlim)
ax.set_ylim([-3,3])
ax.set_ylabel('d\' (vis vs. aud)',fontsize=12)
ax.set_title(str(len(dp)) + ' mice',fontsize=12)
plt.tight_layout()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
xticks = np.arange(6) + 1
xlim = (0.75,6.25)
ax.plot(xlim,[0,0],'k--')
for d,clr,lab in zip((np.array(dpBlocksVisStart),np.array(dpBlocksAudStart)),'gm',('vis rewarded first','aud rewarded first')):
    d *= [1,-1,1,-1,1,-1] if 'vis' in lab else [-1,1,-1,1,-1,1]
    for y in d:
        ax.plot(xticks,y,'-',color=clr,mfc='none',ms=6,alpha=0.1)
    mean = np.mean(d,axis=0)
    sem = np.std(d,axis=0)/(len(d)**0.5)
    ax.plot(xticks,mean,'-',color=clr,ms=10,lw=2,label=lab)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xlim(xlim)
ax.set_ylim([-3.6,3.6])
ax.set_xlabel('Block',fontsize=12)
ax.set_ylabel('d\' (vis vs. aud)',fontsize=12)
ax.set_title(str(len(dp)) + ' mice',fontsize=12)
ax.legend(bbox_to_anchor=(0,0.95),loc='lower left',fontsize=12)
plt.tight_layout()


lowerQuantile = np.argsort(dp)[:int(len(dp)/2)]
upperQuantile = np.argsort(dp)[int(len(dp)/2):]

stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for qi,qlbl in zip((upperQuantile,lowerQuantile),('best half','worst half')):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps in np.array(sessionData[lbl])[qi]:  
                if len(exps) >= minSessions:
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
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
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(qlbl+' ('+str(len(y))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()

# probability of response since last reward, response, or same stimulus
stimType = ('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
resp = {s: [] for s in stimType}
trialsSincePrevReward = copy.deepcopy(resp)
trialsSincePrevNonReward = copy.deepcopy(resp)
trialsSincePrevResp = copy.deepcopy(resp)
trialsSincePrevStim = copy.deepcopy(resp)
for obj in [obj for lbl in sessionData for exps in sessionData[lbl] for obj in exps]:
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
        blockTrials = (obj.trialBlock==blockInd+1) & ~obj.catchTrials
        rewTrials = np.where(blockTrials & obj.trialRewarded)[0]
        nonRewTrials = np.where(blockTrials & obj.trialResponse & ~obj.trialRewarded)[0]
        respTrials = np.where(blockTrials & obj.trialResponse)[0]
        for s in stimType:
            if s=='rewarded target':
                stim = rewStim
            elif s=='unrewarded target':
                stim = otherModalTarget
            elif s=='non-target (rewarded modality)':
                stim = rewStim[:-1]+'2'
            else:
                stim = otherModalTarget[:-1]+'2'
            stimTrials = np.where(blockTrials & (obj.trialStim==stim))[0]
            prevRewardTrial = rewTrials[np.searchsorted(rewTrials,stimTrials) - 1]
            prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
            trialsSincePrevReward[s].extend(stimTrials - prevRewardTrial)
            trialsSincePrevResp[s].extend(stimTrials - prevRespTrial)
            trialsSincePrevStim[s].extend(np.concatenate(([np.nan],np.diff(stimTrials))))
            if len(nonRewTrials) > 0:
                prevNonRewardTrial = nonRewTrials[np.searchsorted(nonRewTrials,stimTrials) - 1]
                trialsSincePrevNonReward[s].extend(stimTrials - prevNonRewardTrial)
            else:
                trialsSincePrevNonReward[s].extend(np.full(len(stimTrials),np.nan))
            resp[s].extend(obj.trialResponse[stimTrials])
for d in (trialsSincePrevReward,trialsSincePrevNonReward,trialsSincePrevResp,trialsSincePrevStim,resp):
    for s in stimType:
        d[s] = np.array(d[s])

for trialsSince,lbl in zip((trialsSincePrevReward,trialsSincePrevNonReward,trialsSincePrevResp,trialsSincePrevStim),
                           ('reward','non-reward','response','same stimulus')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(20)
    for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
        n = np.zeros(x.size)
        p = np.zeros(x.size)
        ci = np.zeros((x.size,2))
        for i in x:
            if i>0:
                j = trialsSince[s]==i
                n[i] += j.sum()
                p[i] += resp[s][j].sum()
        p /= n
        ci = np.array([[b/n[i] for b in scipy.stats.binom.interval(0.95,n[i],p[i])] for i in x])
        ax.plot(x,p,color=clr,ls=ls,label=s)
        ax.fill_between(x,ci[:,0],ci[:,1],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,14])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('trials since last '+lbl)
    ax.set_ylabel('response rate')
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


## nogo, noAR, and rewardOnly
ind = summaryDf['stage 5 pass']
mice = {'nogo': np.array(summaryDf[ind & summaryDf['nogo']]['mouse id']),
        'noAR': np.array(summaryDf[ind & summaryDf['noAR']]['mouse id']),
        'rewardOnly': np.array(summaryDf[ind & summaryDf['rewardOnly']]['mouse id'])}

sessionData = {lbl: [] for lbl in mice}
isFirstExpType = {lbl: [] for lbl in mice}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task and lbl in task for task in df['task version']])
        sessionData[lbl].append(getSessionData(mid,df[sessions]))
        for task in df['task version']:
            if 'stage 5' in task and any(key in task for key in mice):
                isFirstExpType[lbl].append(lbl in task)
                break

useFirstExpType = False
useFirstExp = False
            
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1) 
respRate = {'vis1': {}, 'sound1': {}}
for lbl,title in zip(sessionData,('block switch begins with non-rewarded target trials','no block switch cues','block switch cued with reward only')):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
                if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                    if useFirstExp:
                        exps = [exps[0]]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = obj.trialStim==stim
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                    y[-1] = np.nanmean(y[-1],axis=0)
            if lbl=='nogo':
                respRate[rewardStim][stim] = y
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(title+' ('+str(len(mice[lbl]))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()
        
#
inc = []
dec = []
for rewStim in ('vis1','sound1'):
    for stim in ('vis1','sound1'):
        r = np.array(respRate[rewStim][stim])
        pre = r[:,preTrials-1]
        if rewStim==stim:
            inc.append(r[:,preTrials+1]-pre)
        else:
            dec.append(r[:,preTrials+5]-pre)
            
audDiff = np.array(respRate['sound1']['sound1'])[:,postTrials+6:].mean(axis=1) - np.array(respRate['sound1']['sound2'])[:,postTrials+6:].mean(axis=1)

visDprime,audDprime = [[np.mean([np.array(obj.dprimeSameModal)[obj.blockStimRewarded==stim] for obj in exps]) for exps in sessionData['nogo']] for stim in ('vis1','sound1')]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = (1,4)
ax.plot(alim,alim,'--',color='0.5')
ax.plot(audDprime,visDprime,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('d\' auditory',fontsize=12)
ax.set_ylabel('d\' visual',fontsize=12)
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = (-0.6,0.2)
ax.plot(alim,alim,'--',color='0.5')
ax.plot(dec[0],dec[1],'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel(r'$\Delta$ response rate to previously rewarded'+'\nvisual target',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously rewarded'+'\nauditory target',fontsize=12)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = (0,0.8)
ax.plot(alim,alim,'--',color='0.5')
ax.plot(inc[0],inc[1],'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel(r'$\Delta$ response rate to previously non-rewarded'+'\nvisual target',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously non-rewarded'+'\nauditory target',fontsize=12)
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = audDprime
y = dec[0]
plt.plot(x,y,'ko')
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
xrng = np.array([min(x),max(x)])
plt.plot(xrng,slope*xrng+yint,'--',color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([1.5,3.5])
# ax.set_ylim([0,0.7])
ax.set_xlabel('d\' auditory',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously rewarded'+'\nauditory target',fontsize=12)
ax.set_title('r = '+str(round(rval,2)))
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = audDprime
y = inc[0]
plt.plot(x,y,'ko')
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
xrng = np.array([min(x),max(x)])
plt.plot(xrng,slope*xrng+yint,'--',color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([1.5,3.5])
# ax.set_ylim([0,0.7])
ax.set_xlabel('d\' auditory',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously non-rewarded'+'\nvisual target',fontsize=12)
ax.set_title('r = '+str(round(rval,2)))
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = audDprime
y = dec[1]
plt.plot(x,y,'ko')
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
xrng = np.array([min(x),max(x)])
plt.plot(xrng,slope*xrng+yint,'--',color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([1.5,3.5])
# ax.set_ylim([0,0.7])
ax.set_xlabel('d\' auditory',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously rewarded'+'\nvisual target',fontsize=12)
ax.set_title('r = '+str(round(rval,2)))
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = audDprime
y = inc[1]
plt.plot(x,y,'ko')
slope,yint,rval,pval,stderr = scipy.stats.linregress(x,y)
xrng = np.array([min(x),max(x)])
plt.plot(xrng,slope*xrng+yint,'--',color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([1.5,3.5])
# ax.set_ylim([0,0.7])
ax.set_xlabel('d\' auditory',fontsize=12)
ax.set_ylabel(r'$\Delta$ response rate to previously non-rewarded'+'\nauditory target',fontsize=12)
ax.set_title('r = '+str(round(rval,2)))
plt.tight_layout()


#
targetNames = ('vis1','sound1')
respMat = np.zeros((2,2,2,2))
respMatCount = respMat.copy()
prevRespMat = respMat.copy()
for exps in sessionData['noAR']:
    for obj in exps:
        catchTrials = obj.trialStim=='catch'
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            if blockInd > 0:
                b = 0 if rewStim=='vis1' else 1
                blockTrials = (obj.trialBlock==blockInd+1) & np.in1d(obj.trialStim,targetNames)
                firstResp = np.where(obj.trialResponse[blockTrials])[0][0]
                i = targetNames.index(obj.trialStim[blockTrials][firstResp+1])
                j = targetNames.index(obj.trialStim[blockTrials][firstResp])
                respMat[b,1,i,j] += obj.trialResponse[blockTrials][firstResp+1]
                respMatCount[b,1,i,j] += 1
                prevBlockTrials = obj.trialBlock==blockInd
                prevRespMat[b,1,i,j] += obj.trialResponse[prevBlockTrials & (obj.trialStim==targetNames[i])][-1]
                if firstResp>0:
                    i = targetNames.index(obj.trialStim[blockTrials][1])
                    j = targetNames.index(obj.trialStim[blockTrials][0])
                    respMat[b,0,i,j] += obj.trialResponse[blockTrials][1]
                    respMatCount[b,0,i,j] += 1
                    prevRespMat[b,0,i,j] += obj.trialResponse[prevBlockTrials & (obj.trialStim==targetNames[i])][-1]
respMat /= respMatCount
prevRespMat /= respMatCount
respMatDiff = respMat-prevRespMat

for rr,nn in zip(respMatDiff,respMatCount):
    for r,n in zip(rr,nn):
        print(r)
        print(n)
        

# response times
norm = True
stimNames = ('vis1','sound1')
stimLabels = ('visual target','auditory target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
ylim = [-0.08,0.22] if norm else [0.3,0.7]
ylbl = 'Response Time (s)'
if norm:
    ylbl = r'$\Delta$ ' + ylbl
for lbl,title in zip(sessionData,('block switch cued with non-rewarded target trials','no block switch cues','block switch cued with reward only')):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        # ax.plot([0,0],ylim,'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gm',('-','-')):
            y = []
            for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
                if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                    if useFirstExp:
                        exps = [exps[0]]
                    y.append([])
                    for obj in exps:
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                normRespTime = np.nanmedian(obj.responseTimes[trials]) if norm else 0
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = obj.responseTimes[(obj.trialBlock==blockInd) & trials] - normRespTime
                                i = min(preTrials,pre.size)
                                y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                post = obj.responseTimes[(obj.trialBlock==blockInd+1) & trials] - normRespTime
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
        # ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim(ylim)
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel(ylbl,fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(title+' ('+str(len(mice[lbl]))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()
            
# block switch plot, target stimuli only
for lbl,title in zip(sessionData,('block switch cued with non-rewarded target trials','no block switch cues','block switch cued with reward only')):
    for getDeltaLickProb in (False,True):
        if lbl == 'nogo' or not getDeltaLickProb:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            preTrials = 15
            postTrials = 15
            x = np.arange(-preTrials,postTrials+1)    
            ax.plot([0,0],[0,1],'--',color='0.5')
            for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
                y = []
                for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
                    if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                        if useFirstExp:
                            exps = [exps[0]]
                        y.append([])
                        for obj in exps:
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0:
                                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                                    trials = (obj.trialStim==stim)
                                    if getDeltaLickProb and stim != rewStim:
                                        blockTrials = (obj.trialBlock==blockInd+1)
                                        firstReward = np.where(blockTrials & (obj.trialStim==rewStim))[0][0]
                                        if np.where(blockTrials & trials)[0][obj.newBlockNogoTrials] > firstReward:
                                            continue
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                        y[-1] = np.nanmean(y[-1],axis=0)
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,label=stimLbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                if getDeltaLickProb:
                    if stimLbl == 'rewarded target stim':
                        deltaLickProb['5 non-rewarded targets']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
                    else:
                        deltaLickProb['5 non-rewarded targets']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+6]]
                        deltaLickProb['1 non-rewarded target']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+2]]
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
            ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
            plt.tight_layout()

# block switch plot aligned to first reward
for lbl,title in zip(('nogo',),('block switch cued with non-rewarded target trials',)):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
            if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                if useFirstExp:
                    exps = [exps[0]]
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                            stimTrials = np.where(obj.trialStim==stim)[0]
                            blockTrials = np.where(obj.trialBlock==blockInd+1)[0]
                            firstReward = blockTrials[obj.trialRewarded[blockTrials] & ~obj.catchTrials[blockTrials]][0]
                            lastPreTrial = np.where(stimTrials<firstReward)[0][-1]
                            y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                            pre = obj.trialResponse[stimTrials[lastPreTrial-preTrials:lastPreTrial+1]]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            firstPostTrial = np.where(stimTrials>firstReward)[0][0]
                            post = obj.trialResponse[stimTrials[firstPostTrial:max(firstPostTrial+postTrials,blockTrials[-1])]]
                            i = min(postTrials,post.size)
                            y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after first reward',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()
    
# block switch plots by first trial stim/reward type
for lbl,title in zip(('noAR','rewardOnly'),('no block switch cues','block switch cued with reward only')):
    for firstTrialRewStim,blockLbl in zip((True,False),('rewarded target first','non-rewarded target first')):
        for firstTrialLick,lickLbl in zip((True,False),('lick','no lick')):
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            preTrials = 15
            postTrials = 15
            x = np.arange(-preTrials,postTrials+1)    
            ax.plot([0,0],[0,1],'--',color='0.5')
            for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
                n = 0
                y = []
                for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
                    if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                        if useFirstExp:
                            exps = [exps[0]]
                        y.append([])
                        for obj in exps:
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0:
                                    nonRewStim = np.setdiff1d(obj.blockStimRewarded,rewStim)
                                    blockTrials = obj.trialBlock==blockInd+1
                                    firstRewStim = np.where(blockTrials & (obj.trialStim==rewStim))[0][0]
                                    firstNonRewStim = np.where(blockTrials & (obj.trialStim==nonRewStim))[0][0]
                                    if ((firstTrialRewStim and firstRewStim > firstNonRewStim) or
                                        (not firstTrialRewStim and firstRewStim < firstNonRewStim)):
                                        continue
                                    firstTargetTrial = firstRewStim if firstTrialRewStim else firstNonRewStim
                                    if obj.trialResponse[firstTargetTrial] != firstTrialLick:
                                        continue
                                    stim = nonRewStim if 'unrewarded' in stimLbl else rewStim
                                    trials = obj.trialStim==stim
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = obj.trialResponse[blockTrials & trials]
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                        if len(y[-1]) > 0:
                            n += len(y[-1])
                            y[-1] = np.nanmean(y[-1],axis=0)
                        else:
                            y[-1] = np.full(preTrials+postTrials+1,np.nan)
                if len(y)>0:
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,label=stimLbl)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                    if lbl == 'noAR' and firstTrialLick:
                        if firstTrialRewStim and stimLbl == 'unrewarded target stim' :
                            deltaLickProb['rewarded target first']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
                        elif not firstTrialRewStim and stimLbl == 'rewarded target stim':
                            deltaLickProb['non-rewarded target first']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
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
            ax.set_title(title+'\n'+blockLbl+', '+lickLbl+', '+str(len(y))+' mice, '+str(n)+' blocks')
            plt.tight_layout()

for lbl,title in zip(('rewardOnly',),('block switch cued with reward only',)):
    for firstTrialRewStim,blockLbl in zip((True,False),('rewarded target first','non-rewarded target first')):
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            preTrials = 15
            postTrials = 15
            x = np.arange(-preTrials,postTrials+1)    
            ax.plot([0,0],[0,1],'--',color='0.5')
            for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
                n = 0
                y = []
                for exps,isFirstType in zip(sessionData[lbl],isFirstExpType[lbl]):
                    if len(exps)>0 and ((useFirstExpType and isFirstType) or not useFirstExpType):
                        if useFirstExp:
                            exps = [exps[0]]
                        y.append([])
                        for obj in exps:
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0:
                                    nonRewStim = np.setdiff1d(obj.blockStimRewarded,rewStim)
                                    blockTrials = obj.trialBlock==blockInd+1
                                    firstRewStim = np.where(blockTrials & (obj.trialStim==rewStim))[0][0]
                                    firstNonRewStim = np.where(blockTrials & (obj.trialStim==nonRewStim))[0][0]
                                    if ((firstTrialRewStim and firstRewStim > firstNonRewStim) or
                                        (not firstTrialRewStim and firstRewStim < firstNonRewStim)):
                                        continue
                                    firstTargetTrial = firstRewStim if firstTrialRewStim else firstNonRewStim
                                    stim = nonRewStim if 'unrewarded' in stimLbl else rewStim
                                    trials = obj.trialStim==stim
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = obj.trialResponse[blockTrials & trials]
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                        if len(y[-1]) > 0:
                            n += len(y[-1])
                            y[-1] = np.nanmean(y[-1],axis=0)
                        else:
                            y[-1] = np.full(preTrials+postTrials+1,np.nan)
                if len(y)>0:
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,label=stimLbl)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                    if firstTrialRewStim and stimLbl == 'rewarded target stim':
                        deltaLickProb['5 rewards, no target (first target trial)']['rewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
                    elif not firstTrialRewStim and stimLbl == 'unrewarded target stim':
                        deltaLickProb['5 rewards, no target (first target trial)']['nonRewTarg'] = np.array(y)[:,[preTrials-1,preTrials+1]]
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
            ax.set_title(title+'\n'+blockLbl+', '+str(len(y))+' mice, '+str(n)+' blocks')
            plt.tight_layout()


# change in lick prob summary
xlabels = []
for lbl in deltaLickProbLabels:
    for c in ('auto','target','(',','):
        if 'no target' not in lbl or c!='target':
            if c in lbl:
                i = lbl.find(c)
                lbl = lbl[:i] + '\n' + lbl[i:]
    xlabels.append(lbl)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
xlim = [-0.5,len(xlabels)-0.5]
ax.plot(xlim,[0,0],'k--')
for x,lbl in enumerate(deltaLickProbLabels):
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
ax.set_ylabel(r'$\Delta$ lick probability',fontsize=12)
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
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.set_xlim(xlim)
ax.set_ylim([0,1])
ax.set_ylabel('Lick probability',fontsize=12)
plt.tight_layout()


## no reward blocks
mice = np.array(summaryDf[summaryDf['stage 5 pass'] & summaryDf['no reward']]['mouse id'])

sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['no reward' in task for task in df['task version']])
    sessionData.append(getSessionData(mid,df[sessions]))

# block switch plot, target stimuli only
for blockRewarded,title in zip((True,False),('switch to rewarded block','switch to unrewarded block')):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('previously rewarded target stim','other target stim'),'mg'):
        y = []
        for exps in sessionData:
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and ((blockRewarded and rewStim != 'none') or (not blockRewarded and rewStim == 'none')):
                        if blockRewarded:
                            stim = np.setdiff1d(('vis1','sound1'),rewStim) if 'previously' in stimLbl else rewStim
                        else:
                            prevRewStim = obj.blockStimRewarded[blockInd-1]
                            stim = np.setdiff1d(('vis1','sound1'),prevRewStim) if 'other' in stimLbl else prevRewStim
                        trials = (obj.trialStim==stim)
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()


## extinction
mice = np.array(summaryDf[summaryDf['stage 5 pass'] & summaryDf['extinction']]['mouse id'])

sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['extinction' in task for task in df['task version']])
    sessionData.append(getSessionData(mid,df[sessions]))

# block switch plot, target stimuli only
smoothSigma = 1
for blockRewarded,title,preTrials,postTrials in zip((True,False),('switch to rewarded block','switch to unrewarded block'),
                                                    (60,15),(15,60)):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('previously rewarded target stim','other target stim'),'mg'):
        y = []
        for exps in sessionData:
            y.append([])
            for obj in exps[:2]:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and ((blockRewarded and rewStim != 'none') or (not blockRewarded and rewStim == 'none')):
                        if blockRewarded:
                            stim = np.setdiff1d(('vis1','sound1'),rewStim) if 'previously' in stimLbl else rewStim
                        else:
                            prevRewStim = obj.blockStimRewarded[blockInd-1]
                            stim = np.setdiff1d(('vis1','sound1'),prevRewStim) if 'other' in stimLbl else prevRewStim
                        trials = (obj.trialStim==stim)
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        if smoothSigma is not None:
                            pre = scipy.ndimage.gaussian_filter(pre.astype(float),smoothSigma)
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        if smoothSigma is not None:
                            post = scipy.ndimage.gaussian_filter(post.astype(float),smoothSigma)
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-80,80,10))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()

