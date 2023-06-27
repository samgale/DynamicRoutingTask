# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
"""

import copy
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


def plotLearning(mice,stage):
    hitCount = {lbl:[] for lbl in mice}
    dprime = copy.deepcopy(hitCount)
    sessionsToPass = copy.deepcopy(hitCount)
    for lbl,mouseIds in mice.items():
        for mid in mouseIds:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.where([str(stage) in task for task in df['task version']])[0]
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
                if mid==614910:
                    sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0]+ 1)
                else:
                    sessionsToPass[lbl].append(np.nan)
                    
    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.5)
                
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
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Sessions to pass',fontsize=14)
    ax.set_ylabel('Cumalative fraction',fontsize=14)
    plt.legend()
    plt.tight_layout()   


# stage 1, stationary gratings, timeouts with noise vs no timeouts, no reward click or wheel fixed
ind = summaryDf['stage 1 pass'] & summaryDf['stat grating'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'stationary, no timeouts': np.array(summaryDf[ind & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=1)


# stage 1, stationary with noise timeouts vs moving with noiseless timeouts
ind = summaryDf['stage 1 pass'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['stat grating'] & summaryDf['timeout noise'] ]['mouse id']),
        'moving, timeouts without noise':  np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] ]['mouse id'])}
plotLearning(mice,stage=1)


# stage 1, stationary vs moving gratings, both with noise timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['timeout noise'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id']),
        'moving, timeouts with noise':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id'])}
plotLearning(mice,stage=1)


# stage 1 moving gratings, timeouts with vs without noise
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'moving, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'moving, timeouts without noise':  np.array(summaryDf[ind & summaryDf['timeouts'] & ~summaryDf['timeout noise']]['mouse id'])}
plotLearning(mice,stage=1)
 

# stage 2, tones, timeouts with noise vs no timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['tone'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'tones, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'tones, no timeouts':  np.array(summaryDf[ind  & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=2)


# stage 2, tones with noise timeouts vs AMN with noiseless timeouts
ind = summaryDf['stage 2 pass'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'tones, timeouts with noise': np.array(summaryDf[ind & summaryDf['tone'] & summaryDf['timeout noise']]['mouse id']),
        'AM noise, timeouts without noise':  np.array(summaryDf[ind & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['timeout noise']]['mouse id'])}
plotLearning(mice,stage=2)


# stage 1 moving gratings, timeout without noise, with vs without reward clicks
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] & ~summaryDf['wheel fixed']
mice = {'moving, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'moving, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=1)


# stage 2 AMN, with vs without reward clicks
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] & ~summaryDf['wheel fixed']
mice = {'AM noise, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'AM noise, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=2)


# stationary vs moving gratings and tone vs AMN after stage 2
ind = summaryDf['stage 5 pass']
miceVis = {'stationary': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id']),
           'moving':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id'])}

miceAud = {'tone': np.array(summaryDf[ind & summaryDf['tone']]['mouse id']),
           'AM noise':  np.array(summaryDf[ind & summaryDf['AM noise']]['mouse id'])}

for mice in (miceVis,miceAud):
    dprime = {lbl:[] for lbl in mice}
    for lbl,mouseIds in mice.items():
        for mid in mouseIds:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.array(['ori' in task and ('stage 3' in task or 'stage 4' in task or 'stage variable' in task or 'stage 5' in task) for task in df['task version']])
            firstExperimentSession = np.where(['multimodal' in task
                                               or 'contrast'in task
                                               or 'opto' in task
                                               or 'nogo' in task
                                               or 'noAR' in task
                                               or 'NP' in rig 
                                               for task,rig in zip(df['task version'],df['rig name'])])[0]
            if len(firstExperimentSession)>0:
                sessions[firstExperimentSession[0]:] = False
            sessions = np.where(sessions)[0]
            dprime[lbl].append([])
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                dprimeSame = dprimeSame[0]
                if len(dprimeSame) > 1:
                    task = df.loc[sessionInd,'task version']
                    visFirst = 'ori tone' in task or 'ori AMN' in task
                    if ('moving' in mice and visFirst) or ('tone' in mice and not visFirst):
                        dprime[lbl][-1].append(np.nanmean(dprimeSame[0:6:2]))
                    else:
                        dprime[lbl][-1].append(np.nanmean(dprimeSame[1:6:2]))
                else:
                    dprime[lbl][-1].append(dprimeSame[0])
    
    maxSessions = max(len(d) for lbl in dprime for d in dprime[lbl])
    minMice = 8
                
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xmax = 1e6
    for lbl,clr in zip(mice.keys(),'gm'):
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
    plt.legend()
    plt.tight_layout()
    

# moving to stationary grating switch
preSessions = 2
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
ax.set_xlim([-preSessions-0.5,postSessions+0.5])
ax.set_ylim([0,4.1])
ax.set_xlabel('Session relative to session with stationary gratings',fontsize=14)
ax.set_ylabel('d\'',fontsize=14)
plt.tight_layout()


# training after stage 2
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
mice = {'indirect': np.array(summaryDf[hasIndirectRegimen & summaryDf['stage 2 pass']]['mouse id']),
        'direct': np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 2 pass']]['mouse id'])}

sessionsToPass = {lbl:[] for lbl in mice.keys()}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        sessions = np.where(sessions)[0]
        passed = False
        for sessionInd in sessions[1:]:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if not passed:
                if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                    firstSession = np.where(np.array([('stage 3' in task and 'distract' in task) or 
                                                       'stage 4' in task or 
                                                       'stage variable' in task or
                                                       'stage 5' in task for task in df['task version']]))[0][0]
                    sessionsToPass[lbl].append(sessionInd - firstSession + 1)
                    passed = True
                    break
        if not passed:
            sessionsToPass[lbl].append(np.nan)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(mice.keys(),'gm'):
    dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    lbl += ' to 6-block training'+' (n='+str(dsort.size)+')'
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlabel('Sessions to pass (after stage 2)',fontsize=14)
ax.set_ylabel('Cumalative fraction',fontsize=14)
plt.legend()
plt.tight_layout()   


# training in stage 5
mice = np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 5 pass']]['mouse id'])

dprime = {comp: {mod: [] for mod in ('all','vis','sound')} for comp in ('same','other')}
sessionsToPass = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstExperimentSession = np.where(['multimodal' in task
                                       or 'contrast'in task
                                       or 'opto' in task
                                       or 'nogo' in task
                                       or 'noAR' in task
                                       or 'NP' in rig 
                                       for task,rig in zip(df['task version'],df['rig name'])])[0]
    if len(firstExperimentSession)>0:
        sessions[firstExperimentSession[0]:] = False
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
        lbl = mod+' rewarded' if comp=='other' else mod
        ax.plot(np.arange(len(m))+1,m,color=clr,lw=2,zorder=1,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-3,4])
    ax.set_xlabel('Session',fontsize=14)
    ax.set_ylabel('d\' '+comp+' modality',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    

passOnly = True
for (compX,modX),(compY,modY) in itertools.combinations(itertools.product(('same','other'),('vis','sound')),2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [10,-10]
    for i,clr in enumerate(mouseClrs):
        dx,dy = [np.nanmean(dprime[comp][mod][i],axis=1) for comp,mod in zip((compX,compY),(modX,modY))]
        if passOnly:
            ind = slice(sessionsToPass[i]-2,None)
            dx = dx[ind]
            dy = dy[ind]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    alim = 1.05*np.array(alim)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' '+compX+', '+modX,fontsize=14)
    ax.set_ylabel('d\' '+compY+', '+modY,fontsize=14)
    plt.tight_layout()
    
for mod in ('vis','sound'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [10,-10]
    for i,clr in enumerate(mouseClrs):
        if passOnly:
            dx,dy = [np.ravel(dprime[comp][mod][i][sessionsToPass[i]-2:]) for comp in ('same','other')]
        else:
            dx,dy = [np.ravel(dprime[comp][mod][i]) for comp in ('same','other')]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    alim = 1.05*np.array(alim)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' same'+', '+mod,fontsize=14)
    ax.set_ylabel('d\' other'+', '+mod,fontsize=14)
    plt.tight_layout()


# old
mouseIds = df['mouse id']
passOnly = True
lateAutoRewardOnly = False

mice = []
sessionStartTimes = []
passSession =[]
movingGrating = []
amNoise = []
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
        df = sheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'tone' in task and 'moving' not in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'tone' in task and 'moving' in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'AMN' in task and 'moving' not in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'AMN' in task and 'moving' in task for task in df['task version']])
        if sessions.sum() == 0:
            continue
        if any('stage 3' in task for task in df['task version']) and not any('stage 4' in task for task in df['task version']):
            sessions[np.where(sessions)[0][0]] = False # skipping first 6-block session when preceded by distractor training
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           #or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        if sessions.sum() > 0 and df['pass'][sessions].sum() > 0:
            mice.append(str(mid))
            if passOnly:
                sessions[:np.where(sessions & df['pass'])[0][0]-1] = False
                passSession.append(0)
            else:
                passSession.append(np.where(df['pass'][sessions])[0][0]-1)
            sessionStartTimes.append(list(df['start time'][sessions]))
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        if not lateAutoRewardOnly or (obj.autoRewardOnsetFrame >= obj.responseWindow[1]):
            expsByMouse[-1].append(obj)
        
nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]


    
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
        y = []
        for exps in expsByMouse:
            y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
            for i,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and rewStim==rewardStim:
                        trials = (obj.trialStim==stim) & ~obj.autoRewarded 
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        j = min(preTrials,pre.size)
                        y[-1][i][preTrials-j:preTrials] = pre[-j:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        j = min(postTrials,post.size)
                        y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(np.arange(-20,20,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch (auto-rewards excluded)',fontsize=12)
    ax.set_ylabel('Response Rate',fontsize=14)
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockLabel+'\n'+str(nMice)+' mice',fontsize=14)
    plt.tight_layout()
    
    
# probability of nogo response since last reward
stimType = ('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
resp = {s: [] for s in stimType}
trialsSincePrevReward= copy.deepcopy(resp)
trialsSincePrevResp = copy.deepcopy(resp)
trialsSincePrevStim = copy.deepcopy(resp)
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
            blockTrials = (obj.trialBlock==blockInd+1) & ~obj.catchTrials
            rewTrials = np.where(blockTrials & obj.trialRewarded)[0]
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
                resp[s].extend(obj.trialResponse[stimTrials])

for d in (trialsSincePrevReward,trialsSincePrevResp,trialsSincePrevStim,resp):
    for s in stimType:
        d[s] = np.array(d[s])


for trialsSince,lbl in zip((trialsSincePrevReward,trialsSincePrevResp,trialsSincePrevStim),
                           ('reward','response','same stimulus')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(20)
    for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
        n = np.zeros(x.size)
        p = np.zeros(x.size)
        ci = np.zeros((x.size,2))
        for i in x:
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



































