# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
"""

import copy
import glob
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData,getPerformanceStats,pca,cluster


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

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
        if not os.path.exists(filePath):
            dataDir = summaryDf.loc[summaryDf['mouse id']==mouseId,'data path'].values[0]
            filePath = glob.glob(os.path.join(dataDir,'**',fileName))[0]
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
            passed = False
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                dpSame[lbl][-1].append(dprimeSame[0])
                dpOther[lbl][-1].append(dprimeOther[0])
                if sessionInd > sessions[0]:
                    hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
                    if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                        sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0]+ 1)
                        passed = True
                        break
            if not passed:
                if mid in (677352,688770):
                    sessionsToPass[lbl].append(np.where(['timeouts' in task for task in  df['task version'][sessions]])[0][-1]+ 1)
                else:
                    sessionsToPass[lbl].append(np.nan)

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



## stage 1, stationary gratings with or without timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['stat grating'] & ~summaryDf['wheel fixed'] & ~summaryDf['cannula']
mice = {'stationary, timeouts': np.array(summaryDf[ind & summaryDf['timeouts']]['mouse id']),
        'stationary, no timeouts': np.array(summaryDf[ind & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=1)


# stage 1, stationary vs moving gratings, both with timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['timeouts'] & ~summaryDf['wheel fixed'] & ~summaryDf['cannula']
mice = {'moving':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id']),
        'stationary': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id'])}
plotLearning(mice,stage=1)

# stage 1, moving gratings with or without reward clicks
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['cannula']
mice = {'moving, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'moving, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=1)

# stage 1, moving gratings with early or late autorewards
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['cannula']
mice = {'moving, early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 1)']]['mouse id']),
        'moving, late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 1)']]['mouse id'])}
plotLearning(mice,stage=1)
                

# stage 2, tones, timeouts with noise vs no timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['tone'] & ~summaryDf['wheel fixed'] & ~summaryDf['cannula']
mice = {'tones, timeouts': np.array(summaryDf[ind & summaryDf['timeouts']]['mouse id']),
        'tones, no timeouts':  np.array(summaryDf[ind  & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, tones with noise timeouts vs AMN with noiseless timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['timeouts'] & ~summaryDf['wheel fixed'] & ~summaryDf['cannula']
mice = {'tones': np.array(summaryDf[ind & summaryDf['tone']]['mouse id']),
        'AM noise':  np.array(summaryDf[ind & summaryDf['AM noise']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, AMN with or without reward clicks
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['cannula']
mice = {'AM noise, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'AM noise, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=2)

# stage 2, AMN with early or late autorewwards
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['cannula']
mice = {'AM noise, early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 2)']]['mouse id']),
        'AM noise, late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 2)']]['mouse id'])}
plotLearning(mice,stage=2)


# stage 5, repeats vs no repeats
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & ~summaryDf['cannula']
mice = {'no repeats': np.array(summaryDf[ind & ~summaryDf['stage 5 repeats']]['mouse id']),
        'repeats': np.array(summaryDf[ind & summaryDf['stage 5 repeats']]['mouse id'])}
plotStage5Learning(mice)

# stage 5, with or without reward clicks
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula']#& ~summaryDf['stage 5 repeats']
mice = {'reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotStage5Learning(mice)

# stage 5, early or late autorewards
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula']#& ~summaryDf['stage 5 repeats']
mice = {'early AR': np.array(summaryDf[ind & ~summaryDf['late autoreward (stage 5)']]['mouse id']),
        'late AR':  np.array(summaryDf[ind & summaryDf['late autoreward (stage 5)']]['mouse id'])}
plotStage5Learning(mice)


## moving to stationary grating switch
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
mice = np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise']]['mouse id']) & ~summaryDf['cannula']

dprime = {'vis': [], 'aud': []}
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
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

 
## stage 5 training
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
mice = np.array(summaryDf[ind]['mouse id'])
hasLateAutorewards = np.array(summaryDf[ind]['late autoreward (stage 5)'])

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
    if not passed:
        if mid in (677352,688770):
            sessionsToPass.append(np.where(['timeouts' in task for task in  df['task version'][sessions]])[0][-1]+ 1)
        else:
            sessionsToPass.append(np.nan)
    sessionData.append(getSessionData(mid,df.iloc[sessions]))
                
mouseClrs = plt.cm.tab20(np.linspace(0,1,len(sessionsToPass)))

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),max(len(d) for d in dprime[comp]['all'])),np.nan)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)[:sessionsToPass[i]]
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
    
# compare early training and after learning
nSessions = 5
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1) 
for phase in ('initial training','after learning'):
    for ylbl,yticks,ylim,stimInd in zip(('Response rate',r'$\Delta$ Response time (ms)'),([0,0.5,1],np.arange(-300,301,50)),([0,1.02],[-75,125]),(slice(0,4),slice(0,2))):
        resp = {}
        respAll = {}
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
            fig = plt.figure(figsize=(8,4.5))
            ax = fig.add_subplot(1,1,1)
            ax.plot([0,0],[0,1],'--',color='0.5')
            resp[rewardStim] = {}
            respAll[rewardStim] = {}
            for stim,stimLbl,clr,ls in zip(stimNames[stimInd],stimLabels[stimInd],'gmgm'[stimInd],('-','-','--','--')[stimInd]):
                y = []
                yall = []
                for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                    if len(exps)>0:# and hasLateAutorewards[mouseInd]:
                        if phase=='initial training':
                            exps = exps[1:nSessions-1]
                        elif phase=='after learning':
                            exps = exps[s:s+nSessions]
                        y.append([])
                        yall.append([])
                        for obj in exps:
                            trials = (obj.trialStim==stim) & ~obj.trialRepeat & ~obj.autoRewardScheduled
                            r = obj.trialResponse if 'rate' in ylbl else 1000*(obj.responseTimes-np.nanmedian(obj.responseTimes[trials & (obj.rewardedStim==stim)]))
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0 and rewStim==rewardStim:
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = r[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = r[(obj.trialBlock==blockInd+1) & trials]
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                                    yall[-1].append([np.nanmean(pre[1:]),np.nanmean(post[1:])])
                        y[-1] = np.nanmean(y[-1],axis=0)
                        yall[-1] = np.mean(yall[-1],axis=0)
                if stim in ('vis1','sound1'):
                    resp[rewardStim][stim] = y
                    respAll[rewardStim][stim] = yall
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xticks(np.arange(-20,20,5))
            ax.set_yticks(yticks)
            ax.set_xlim([-preTrials-0.5,postTrials+0.5])
            ax.set_ylim(ylim)
            ax.set_xlabel('Trials of indicated type after block switch (excluding cue trials)',fontsize=12)
            ax.set_ylabel(ylbl,fontsize=12)
            ax.legend(bbox_to_anchor=(1,1),fontsize=12)
            ax.set_title(phase+' (n='+str(len(y))+' mice)'+'\n'+blockLabel,fontsize=12)
            plt.tight_layout()
        
        if 'time' in ylbl:
            ylim = [-240,240]
            yticks = np.arange(-300,301,100)
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
            fig = plt.figure()
            fig.suptitle(blockLabel)
            axs = []
            gs = matplotlib.gridspec.GridSpec(2,2)
            xticks = (0,1)
            for rr,i,j,clr in zip((respAll[rewardStim][rewardStim],
                                   np.array(resp[rewardStim][rewardStim])[:,[preTrials-1,preTrials+1]],
                                   respAll[rewardStim]['vis1' if rewardStim=='sound1' else 'sound1'],
                                   np.array(resp[rewardStim]['vis1' if rewardStim=='sound1' else 'sound1'])[:,[preTrials-1,preTrials+1]]),
                                  (0,0,1,1),(0,1,0,1),['g' if rewardStim=='vis1' else 'm']*2+['m' if rewardStim=='vis1' else 'g']*2):
                ax = fig.add_subplot(gs[i,j])
                for r in rr:
                    ax.plot(xticks,r,'o-',color=clr,mec=clr,mfc='none',ms=6,lw=1,alpha=0.2)
                mean = np.nanmean(rr,axis=0)
                sem = np.nanstd(rr,axis=0)/(len(r)**0.5)
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
                    ax.set_xticklabels(('last trial\nprevious block','first trial'))
                else:
                    ax.set_xticklabels([])
                if j==0:
                    ax.set_ylabel(ylbl)
                else:
                    ax.set_yticklabels([])
                ax.set_xlim([-0.2,1.2])
                ax.set_ylim(ylim)
            plt.tight_layout()
            
# performance by block number
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,blockLbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
    for lbl,ls in zip(('same modality','other modality'),('--','-')):
        dp = []
        for exps,s in zip(sessionData,sessionsToPass):
            d = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps[s:]):
                j = obj.blockStimRewarded==rewardStim
                a = obj.dprimeSameModal if 'same' in lbl else obj.dprimeOtherModalGo
                d[i,j] = np.array(a)[j]
            dp.append(np.nanmean(d,axis=0))
        m = np.nanmean(dp,axis=0)
        s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=blockLbl+', '+lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,4])
ax.set_xlabel('Block')
ax.set_ylabel('d\'')
ax.legend(loc='lower right')
ax.set_title(str(len(sessionData))+' mice')
plt.tight_layout()

# run speed
visSpeed = []
soundSpeed = []
for rewStim,speed in zip(('vis1','sound1'),(visSpeed,soundSpeed)):
    for exps in sessionData:
        for obj in exps:
            speed.append(np.mean([np.nanmean(obj.runningSpeed[sf-obj.quiescentFrames:sf]) for sf in obj.stimStartFrame[obj.rewardedStim==rewStim]]))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [0,1.05*max(visSpeed+soundSpeed)]
ax.plot(alim,alim,'--',color='0.5')
ax.plot(visSpeed,soundSpeed,'o',mec='k',mfc='none',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('run speed, visual rewarded blocks (cm/s)')
ax.set_ylabel('run speed, auditory rewarded blocks (cm/s)')
ax.set_title(str(sum([len(exps) for exps in sessionData]))+' sessions, '+str(len(sessionData))+' mice')
plt.tight_layout()

# catch rate
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
    rr = []
    for exps in sessionData:
        r = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            j = obj.blockStimRewarded==rewardStim
            r[i,j] = np.array(obj.catchResponseRate)[j]
        rr.append(np.nanmean(r,axis=0))
    m = np.nanmean(rr,axis=0)
    s = np.nanstd(rr,axis=0)/(len(rr)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,0.07])
ax.set_xlabel('Block')
ax.set_ylabel('Catch trial response rate')
ax.legend(loc='lower right')
ax.set_title(str(len(sessionData))+' mice')
plt.tight_layout()

# quiescent violations
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
    rr = []
    for exps in sessionData:
        r = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            for blockInd,blockRewardStim in enumerate(obj.blockStimRewarded):
                if blockRewardStim==rewardStim:
                    trials = obj.trialBlock==blockInd+1
                    r[i,blockInd] = np.sum((obj.quiescentViolationFrames > obj.trialStartFrame[trials][0]) & (obj.quiescentViolationFrames < obj.trialEndFrame[trials][-1]))/trials.sum()
        rr.append(np.nanmean(r,axis=0))
    m = np.nanmean(rr,axis=0)
    s = np.nanstd(rr,axis=0)/(len(rr)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,0.35])
ax.set_xlabel('Block')
ax.set_ylabel('Quiescent violations per trial')
ax.legend(loc='lower right')
plt.tight_layout()


# cluster block performance





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
smoothSigma = None
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

