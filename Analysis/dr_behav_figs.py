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

    

## moving vs stationary gratings
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

# use d' with engagement correction for plotting with sessions to pass
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

mouseClrs = plt.cm.tab20(np.linspace(0,1,len(sessionsToPass)))

for comp,ylim in zip(('same','other'),((-0.25,4),(-0.5,3))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),max(len(d) for d in dprime[comp]['all'])),np.nan)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
        ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=0)
        ax.plot(sessionsToPass[i],y[sessionsToPass[i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=2)
        dp[i,:len(y)] = y
    # m = np.nanmean(dp,axis=0)
    # ax.plot(np.arange(len(m))+1,m,color='k',lw=2,zorder=1)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xlim([0,max(sessionsToPass)+6])
    ax.set_yticks(np.arange(-1,5))
    ax.set_ylim(ylim)
    ax.set_xlabel('Session',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\'',fontsize=18)
    plt.tight_layout()

# recalculate d' without engagement correction
for exps in sessionData:
    for obj in exps:
        obj.engagedThresh = None
        obj.calcPerformanceStats()

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
  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
learnCurve = []
for i,(d,clr) in enumerate(zip(dprime['other']['all'],mouseClrs)):
    y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
    x = np.arange(len(y))+1
    bounds = ((0,-0.001,x[0],-np.inf),(np.nanmax(y),0.001,x[-1],np.inf))
    fitParams = fitCurve(calcWeibullDistrib,x,y,bounds=bounds)
    yFit = calcWeibullDistrib(x,*fitParams)
    learnCurve.append(yFit)
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
for d,sp,yFit in zip(dprime['other']['all'],sessionsToPass,learnCurve):
    if j==ncols:
        i += 1
        j = 0
    ax = fig.add_subplot(gs[i,j])
    j += 1
    y = np.nanmean(d,axis=1)[:sp+5]
    x = np.arange(len(y))+1
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
    ax.set_ylim([0,3.25]) #int(np.nanmax(y)+1)])
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
learnOnset = []
learnDur = []
for i,(d,yFit,clr) in enumerate(zip(dprime['other']['all'],learnCurve,mouseClrs)):
    y = np.nanmean(d,axis=1)[:sessionsToPass[i]+5]
    x = np.arange(len(y))+1
    yFit = yFit - yFit.min()
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
ax.plot(learnOnset,learnDur,'o',ms=8,mec='k',mfc='none',alpha=0.5)
mx = np.median(learnOnset)
my = np.median(learnDur)
sx = scipy.stats.median_abs_deviation(learnOnset)
sy = scipy.stats.median_abs_deviation(learnDur)
ax.plot(mx,my,'ro',ms=12)
ax.plot([mx-sx,mx+sx],[my,my],'r',lw=2)
ax.plot([mx,mx],[my-sy,my+sy],'r',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(0,100,10))
ax.set_yticks(np.arange(0,100,10))
ax.set_xlim([0,40])
ax.set_ylim([0,30])
ax.set_aspect('equal')
ax.set_xlabel('Learning onset (20% of max; sessions)',fontsize=14)
ax.set_ylabel('Learning duration (20-80% of max; sessions)',fontsize=14)
plt.tight_layout()
   
for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),100),np.nan)
    xintp = np.linspace(0,1,100)
    for mod,clr in zip(('vis','sound'),'gm'):
        for i,d in enumerate(dprime[comp][mod]):
            y = np.nanmean(d,axis=1)[:sessionsToPass[i]+4]
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
    ax.set_xticks([0,1])
    ax.set_yticks((np.arange(4) if comp=='same' else np.arange(-0.5,3,0.5)))
    ax.set_ylim(([0,3.5] if comp=='same' else [-0.5,2]))
    ax.set_xlabel('Normalized session',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\'',fontsize=18)
    plt.legend(loc='lower right',fontsize=14)
    plt.tight_layout()

alim = [0,4]    
for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    x,y = [[np.mean(dp[sp-2:sp+4]) for dp,sp in zip(dprime[comp][mod],sessionsToPass)] for mod in ('vis','sound')]
    ax.plot(x,y,'o',mec='k',mfc='none',ms=8,alpha=0.5)
    mx = np.median(x)
    my = np.median(y)
    sx = scipy.stats.median_abs_deviation(x)
    sy = scipy.stats.median_abs_deviation(y)
    ax.plot(mx,my,'ro',ms=12)
    ax.plot([mx-sx,mx+sx],[my,my],'r',lw=2)
    ax.plot([mx,mx],[my-sy,my+sy],'r',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\''+'\n(visual rewarded blocks)',fontsize=18)
    ax.set_ylabel(('Cross' if comp=='other' else 'Within')+'-modal '+'d\''+'\n(auditory rewarded blocks)',fontsize=18)
    plt.tight_layout()
    


## zig zag plots
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
                for obj in (exps[:4] if phase=='initial training' else exps[sp:sp+4]):
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
            # ax.legend(loc=('lower right' if stim=='target' else 'upper right'),fontsize=16)
            # ax.set_title(phase+' (n='+str(len(hr))+' mice)',fontsize=12)
            plt.tight_layout()

    

## catch rate, quiescent violations, and run speed by block
for stage in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:4] if stage=='initial training' else exps[s:s+4]
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
    ax.set_yticks(np.arange(0,1,0.02))
    ax.set_ylim([0,0.1])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Catch trial response rate',fontsize=14)
    ax.legend(loc=('lower right' if stage=='initial training' else 'upper right'),fontsize=12)
    # ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:4] if stage=='initial training' else exps[s:s+4]
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
    ax.set_yticks(np.arange(0,1,0.1))
    ax.set_ylim([0,0.4])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Quiescent violations per trial',fontsize=14)
    ax.legend(loc=('lower right' if stage=='initial training' else 'upper right'),fontsize=12)
    # ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(6)+1
    for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','auditory rewarded')):
        rr = []
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:4] if stage=='initial training' else exps[s:s+4]
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
    ax.set_yticks(np.arange(0,50,5))
    ax.set_ylim([15,30])
    ax.set_xlabel('Block',fontsize=14)
    ax.set_ylabel('Run speed (cm/s)',fontsize=14)
    ax.legend(loc='upper right',fontsize=12)
    # ax.set_title(stage+' ('+str(len(sessionData))+' mice)',fontsize=14)
    plt.tight_layout()



## catch rate, quiescent violations, and run speed by mouse/session
catchRate = {phase: {blockType: [] for blockType in ('vis rewarded','aud rewarded','all')} for phase in ('initial training','after learning','all')}
qvRate = copy.deepcopy(catchRate)
runSpeed = copy.deepcopy(catchRate)
dprime = copy.deepcopy(catchRate)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:4]
        elif phase=='after learning':
            exps = exps[s:s+4]
        for blockType in ('vis rewarded','aud rewarded','all'):
            for d in (catchRate,qvRate,runSpeed,dprime):
                d[phase][blockType].append([[] for _ in range(len(exps))])
            for sessionInd,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockType=='all' or (blockType=='vis rewarded' and rewStim=='vis1') or (blockType=='aud rewarded' and rewStim=='sound1'):
                        trials = obj.trialBlock==blockInd+1
                        catchRate[phase][blockType][mouseInd][sessionInd].append(obj.catchResponseRate[blockInd])
                        qvRate[phase][blockType][mouseInd][sessionInd].append(np.array(obj.trialQuiescentViolations)[trials].sum() / trials.sum())
                        runSpeed[phase][blockType][mouseInd][sessionInd].append(np.nanmean(obj.quiescentRunSpeed[trials]))  
                        dprime[phase][blockType][mouseInd][sessionInd].append(obj.dprimeOtherModalGo[blockInd])
                        
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for blockType,clr in zip(('vis rewarded','aud rewarded'),'gm'):
        d = np.nanmean(np.concatenate(catchRate[phase][blockType]),axis=1)
        print(np.median(d))
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=blockType+' blocks')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0,0.55])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Catch trial response rate',fontsize=14)
    ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for blockType,clr in zip(('vis rewarded','aud rewarded'),'gm'):
        d = np.nanmean(np.concatenate(qvRate[phase][blockType]),axis=1)
        print(np.median(d))
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=blockType+' blocks')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(0,2,0.2))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0,1.1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Quiescent violations per trial',fontsize=14)
    ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [-2.5,85]
    ax.plot(alim,alim,'k--')
    x,y = [np.nanmean(np.concatenate(runSpeed[phase][blockType]),axis=1) for blockType in ('vis rewarded','aud rewarded')]
    ax.plot(x,y,'ko',alpha=0.2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(0,100,20))
    ax.set_yticks(np.arange(0,100,20))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Run speed, visual rewarded blocks (cm/s)',fontsize=14)
    ax.set_ylabel('Run speed, auditory rewarded blocks (cm/s)',fontsize=14)
    # ax.set_title(phase+' ('+str(len(x))+' sessions)',fontsize=14)
    plt.tight_layout()
    
for phase in ('initial training','after learning'):
    for blockType in ('all',):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x,y = [np.nanmean(np.concatenate(d[phase][blockType]),axis=1) for d in (runSpeed,dprime)]
        ax.plot(x,y,'ko',alpha=0.2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(np.arange(0,100,20))
        ax.set_yticks(np.arange(5))
        ax.set_xlim([-2.5,80])
        ax.set_ylim([-0.5,3.5])
        ax.set_xlabel('Run speed (cm/s)',fontsize=14)
        ax.set_ylabel('Cross-modal d''',fontsize=14)
        # ax.set_title(phase+', '+blockType+' ('+str(len(x))+' sessions)',fontsize=14)
        plt.tight_layout()



## run speed dynamics
runSpeed = {phase: {blockType: [] for blockType in ('vis rewarded','aud rewarded')} for phase in ('initial training','after learning','all')}
nFrames = int(5 * 60)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:4]
        elif phase=='after learning':
            exps = exps[s:s+4]
        for blockType in ('vis rewarded','aud rewarded'):
            runSpeed[phase][blockType].append([])
            for obj in exps:
                objWithRunSpeed = getSessionData(obj.subjectName,obj.startTime,lightLoad=False)
                trials = ~obj.autoRewardScheduled & (obj.trialStim==('vis1' if blockType=='vis rewarded' else 'sound1'))
                runSpeed[phase][blockType][-1].append(np.nanmean([scipy.ndimage.median_filter(objWithRunSpeed.runningSpeed[sf-nFrames:sf],3,mode='nearest') for sf in obj.stimStartFrame[trials]],axis=0))

t = -(np.arange(nFrames)/60)[::-1]
for phase in ('initial training','after learning'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for blockType,clr in zip(('vis rewarded','aud rewarded'),'gm'):
        r = [np.nanmean(r,axis=0) for r in runSpeed[phase][blockType]]
        m = np.mean(r,axis=0)
        s = np.std(r,axis=0) / (len(r)**0.5)
        ax.plot(t,m,color=clr,label=('visual' if blockType=='vis rewarded' else 'auditory') + ' rewarded blocks')
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_yticks(np.arange(0,50,5))
    # ax.set_xlim([-5,85])
    ax.set_ylim([15,30])
    ax.set_xlabel('Time before stimulus onset (s)',fontsize=14)
    ax.set_ylabel('Run speed (cm/s)',fontsize=14)
    ax.legend(fontsize=12)
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
spectralClustering = sklearn.cluster.SpectralClustering(n_clusters=nClust,affinity='nearest_neighbors',n_neighbors=10,assign_labels='kmeans',random_state=0)
clustId = spectralClustering.fit_predict(clustData)
clustId += 1
clustLabels = np.unique(clustId)

if nClust==4:
    newClustOrder = [3,4,2,1]
elif nClust==5:
    newClustOrder = [4,5,2,1,3]
elif nClust==6:
    newClustOrder = [4,3,1,6,5,2]
elif nClust==7:
    newClustOrder = [5,3,7,1,4,6,2]
newClustId = clustId.copy()
for i,c in enumerate(newClustOrder):
    newClustId[clustId==c] = i+1
clustId = newClustId

# sessionClustData['clustId'] = clustId            
# np.save(os.path.join(baseDir,'Sam','sessionClustData.npy'),sessionClustData)

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
    ax.set_title('cluster '+str(clust)+' ('+str(i.sum()) + ' sessions)',fontsize=16)
    # ax.set_title(str(i.sum()) + ' sessions',fontsize=16)
    if clust==1:
        ax.legend(loc='upper right',fontsize=16)
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
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(x)
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([0.5,6.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Block #',fontsize=20)
        ax.set_ylabel('Response rate',fontsize=20)
        if clust==1:
            ax.legend(loc='upper right',fontsize=20)
        ax.set_title(sessionLabel +' (' + str(len(r)) + ' sessions)',fontsize=14)
        plt.tight_layout()
               
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
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(x)
        ax.set_xlim([0.5,6.5])
        ax.set_ylim([0,2.5])
        ax.set_xlabel('Block #',fontsize=20)
        ax.set_ylabel('Cross-modal d\'',fontsize=20)
        if clust==1:
            ax.legend(fontsize=16)
        # ax.set_title('cluster '+str(clust),fontsize=20)
        plt.tight_layout()
        
for clust in (4,5,6):
    for clr in ('gm','mg'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        r = sessionClustData['dprime'][clustId==clust]
        m = np.nanmean(r,axis=0)
        s = np.nanstd(r)/(len(r)**0.5)
        ax.plot(x[::2],m[::2],'o',ms=12,color=clr[0])
        ax.plot(x[1::2],m[1::2],'o',ms=12,color=clr[1])
        ax.fill_between(x,m+s,m-s,color='k',alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=18)
        ax.set_xticks(x)
        ax.set_yticks([0,1,2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim([0.5,6.5])
        ax.set_ylim([0,2.25])
        # ax.set_xlabel('Block #',fontsize=20)
        # ax.set_ylabel('Cross-modal d\'',fontsize=20)
        plt.tight_layout()
        
for k,ind in enumerate((sessionClustData['session']<4,(sessionClustData['session']>=4) & ~sessionClustData['passed'],sessionClustData['passed'])):
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
    ax.tick_params(direction='out',labelsize=16)
    ax.set_xticks(clustLabels)
    ax.set_xticklabels(clustLabels)
    ax.set_ylim((0,0.6))
    ax.set_xlabel('Cluster',fontsize=18)
    ax.set_ylabel('Fraction of sessions',fontsize=18)
    ax.set_title(str(ind.sum()) + ' sessions',fontsize=16)
    if k==0:
        ax.legend(loc='upper left',fontsize=16)
    plt.tight_layout()
    
mouseClustProb = np.zeros((3,nMice,nClust))
for k,ind in enumerate((sessionClustData['session']<4,(sessionClustData['session']>=4) & ~sessionClustData['passed'],sessionClustData['passed'])):
    for i,m in enumerate(np.argsort(sessionsToPass)):
        for j,clust in enumerate(clustLabels):
            s = clustId[(sessionClustData['mouse']==m) & ind]
            mouseClustProb[k,i,j] = np.sum(s==clust)/s.size

fig = plt.figure(figsize=(8,8))
# fig.suptitle('Cluster probability')
for i,(p,lbl) in enumerate(zip(mouseClustProb,('Intitial training','Intermediate training','After learning'))):            
    ax = fig.add_subplot(1,3,i+1) 
    im = ax.imshow(p,cmap='magma',clim=(0,np.nanmax(p)))
    cb = plt.colorbar(im,ax=ax,fraction=0.015,pad=0.04)
    cb.set_ticks((0,0.5,1))
    cb.set_ticklabels((0,0.5,1),fontsize=12)
    ax.tick_params(direction='out',labelsize=12)
    ax.set_xticks(np.arange(nClust))
    ax.set_xticklabels([1,'','','','',6])
    ax.set_yticks([])
    if i==1:
        ax.set_xlabel('Cluster',fontsize=14)
    if i==0:
        ax.set_ylabel('Mouse',fontsize=14)
    ax.set_title(lbl,fontsize=14)
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
cb.set_ticklabels(np.arange(nClust)+1,fontsize=10)
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'w')
ax.tick_params(direction='out',labelsize=10)
ax.set_xticks(np.arange(10,sessionClust.shape[1],10)-1)
ax.set_xticklabels(np.arange(10,sessionClust.shape[1],10))
ax.set_yticks([])
ax.set_xlabel('Session',fontsize=12)
ax.set_ylabel('Mouse',fontsize=12)
ax.set_title('Session cluster\n(white line = passed learning criteria)',fontsize=12)
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
cb.set_ticklabels(('4','other','5'),fontsize=10)
for i,m in enumerate(np.argsort(sessionsToPass)):
    ax.plot([sessionsToPass[m]-0.5]*2,[i-0.4,i+0.4],'k')
ax.tick_params(direction='out',labelsize=10)
ax.set_xticks(np.arange(10,sessionClustAlt.shape[1],10)-1)
ax.set_xticklabels(np.arange(10,sessionClustAlt.shape[1],10))
ax.set_yticks([])
ax.set_xlabel('Session',fontsize=12)
ax.set_ylabel('Mouse',fontsize=12)
ax.set_title('Session cluster\n(black line = passed learning criteria)',fontsize=12)
plt.tight_layout()

nClust5Sessions = []
probPoorAudSuppress = []
for m in range(len(sessionsToPass)):
    ind = (sessionClustData['mouse']==m) & (clustId==5)
    nClust5Sessions.append(ind.sum())
    poorAudSuppress = ind & (sessionClustData['firstRewardStim']=='vis1')
    probPoorAudSuppress.append(np.sum(poorAudSuppress)/ind.sum())
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for n,alpha in zip((1,5,10),(1,0.67,0.33)):
    d = np.array(probPoorAudSuppress)[np.array(nClust5Sessions)>=n]
    dsort = np.sort(d)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color='k',alpha=alpha,label='mice with at least '+str(n)+' cluster 5 session'+('s' if n>1 else '')+' (n='+str(len(d))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlim([0,1.01])
ax.set_ylim([0,1.01])
ax.set_xlabel('Fraction of cluster 5 sessions with vis rewarded first',fontsize=14)
ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
    
prevClustProb = np.zeros((3,4,len(clustLabels),len(clustLabels)))
prevClustChance = np.zeros((3,4,nClust))
nextClustProb = prevClustProb.copy()
nextClustChance = prevClustChance.copy()
for l,si in enumerate((np.ones(sessionClustData['firstRewardStim'].size,dtype=bool),sessionClustData['firstRewardStim']=='vis1',sessionClustData['firstRewardStim']=='sound1')):
    for k,ind in enumerate((np.ones(sessionClustData['session'].size,dtype=bool),sessionClustData['session']<4,(sessionClustData['session']>=4) & ~sessionClustData['passed'],sessionClustData['passed'])):
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
    for k,stage in enumerate(('all','Initial training','Intermediate training','After learning')):
        for transProb,lbl in zip((prevClustProb[l,k],nextClustProb[l,k]),('Previous','Next')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1) 
            im = ax.imshow(transProb,cmap='magma',clim=(0,transProb.max()),origin='lower')
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            cb.set_ticks((0,0.2,0.4,0.6))
            cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',labelsize=14)
            ax.set_xticks(np.arange(len(clustLabels)))
            ax.set_yticks(np.arange(len(clustLabels)))
            ax.set_xticklabels(clustLabels)
            ax.set_yticklabels(clustLabels)
            ax.set_xlabel('Current session cluster',fontsize=16)
            ax.set_ylabel(lbl+' session cluster',fontsize=16)
            ax.set_title('Probability',fontsize=16) #+'\n'+stage+', '+blockType)
            plt.tight_layout()
    
# block switch plots for session clusters
clust = 5 
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('visual target','auditory target','visual non-target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1) 
for firstRewStim in ('vis1','sound1'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure()#(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
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
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
        # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        plt.tight_layout()

# block switch plots for session clusters by mouse
fullBlockRespDiff = {}
firstTrialRespDiff = {}   
for blocks in ('odd','even'):
    fig = plt.figure()#(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
        y = []
        for mouse in mice:
            ind = (sessionClustData['mouseId']==str(mouse)) & (clustId==clust)
            if ind.sum() < 1:
                continue
            y.append([])
            for obj in np.concatenate(sessionData)[ind]:
                r = obj.trialResponse
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd+1 in ((3,5) if blocks=='odd' else (2,4,6)):
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'non-rewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim)
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
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
        ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
        ax.plot(x[preTrials:],m[preTrials:],color=clr)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        if stimLbl == 'non-rewarded target':
            y = np.array(y)
            fullBlockRespDiff[blocks] = np.nanmean(y[:,preTrials-5:preTrials],axis=1) - np.nanmean(y[:,preTrials+5:],axis=1)
            firstTrialRespDiff[blocks] = np.nanmean(y[:,preTrials-5:preTrials],axis=1) - y[:,preTrials+5]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
    ax.set_ylabel('Response rate',fontsize=16)
    # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    plt.tight_layout()    
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0],[0,1],'k--',alpha=0.5)
for blocks,clr in zip(('odd','even'),'gm'):
    d = firstTrialRespDiff[blocks]
    dsort = np.sort(d)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,lw=2,label=blocks+' blocks')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-1,2,0.5))
ax.set_yticks([0,0.5,1])
ax.set_xlim([-0.7,1.01])
ax.set_ylim([0,1.01])
ax.set_xlabel('$\Delta$ Response probability\n(last trial previous block - first trial new block)',fontsize=14)
ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
plt.legend(loc='lower right',fontsize=12)
plt.tight_layout() 
    
# intra-block resp correlations for session clusters by mouse
clust = 5
miceWithClust = np.unique(sessionClustData['mouseId'][clustId==5])
blockNumbers = ('odd','even')
blockEpochs = ('full','first half','last half')
autoCorrMat = {blocks: {epoch: np.zeros((4,len(miceWithClust),100)) for epoch in blockEpochs} for blocks in blockNumbers}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
respRateMat = {blocks: {epoch: np.zeros((4,len(miceWithClust))) for epoch in blockEpochs} for blocks in blockNumbers}
corrWithinMat = {blocks: {epoch: np.zeros((4,4,len(miceWithClust),200)) for epoch in blockEpochs} for blocks in blockNumbers}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
for blocks in blockNumbers:
    for epoch in blockEpochs:
        for m,mid in enumerate(miceWithClust):
            ind = (sessionClustData['mouseId']==mid) & (clustId==5)
            respRate = []
            autoCorr = []
            autoCorrDetrend = []
            corrWithin = []
            corrWithinDetrend = []
                
            for obj in np.concatenate(sessionData)[ind]:   
                rr,ac,acr,acd,cw,cwr,cwd,ca= getResponseCorrelations(obj,blocks=blocks,blockEpoch=epoch)
                respRate.append(rr)
                autoCorr.append(ac)
                autoCorrDetrend.append(acd)
                corrWithin.append(cw)
                corrWithinDetrend.append(cwd)
                  
            autoCorrMat[blocks][epoch][:,m] = np.nanmean(autoCorr,axis=(0,2))
            autoCorrDetrendMat[blocks][epoch][:,m] = np.nanmean(autoCorrDetrend,axis=(0,2))
            respRateMat[blocks][epoch][:,m] = np.nanmean(respRate,axis=(0,2))
                
            corrWithinMat[blocks][epoch][:,:,m] = np.nanmean(corrWithin,axis=(0,3))
            corrWithinDetrendMat[blocks][epoch][:,:,m] = np.nanmean(corrWithinDetrend,axis=(0,3))
            
stimLabels = ('rewarded target','non-rewarded target','non-target\n(rewarded modality)','non-target\n(non-rewarded modality)')
    
for epoch in blockEpochs:    
    fig = plt.figure(figsize=(8,8))          
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for blocks,clr in zip(blockNumbers,'gm'):
                mat = corrWithinDetrendMat[blocks][epoch][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=blocks+' blocks')
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.02,0.04])
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
            if i==0 and j==1:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0],[0,1],'k--',alpha=0.5)
for blocks,clr in zip(blockNumbers,'gm'):
    d = corrWithinDetrendMat[blocks]['full'][1,1,:,1]
    dsort = np.sort(d)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,lw=2,label=blocks+' blocks')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([-0.05,0,0.05,0.1])
ax.set_yticks([0,0.5,1])
ax.set_xlim([-0.07,0.125])
ax.set_ylim([0,1.01])
ax.set_xlabel('Autocorrelation of responses to non-rewarded target',fontsize=14)
ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
plt.legend(loc='lower right',fontsize=12)
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
                    if len(exps)>0:
                        if phase=='initial training':
                            exps = exps[:4]
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
            # fig.suptitle(blockLabel)
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
                            exps = exps[:4]
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
            # ax.set_title(phase+' (n='+str(len(y))+' mice)'+'\n'+blockLabel,fontsize=12)
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
        for stimLbl,clr,ls in zip(('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (non-rewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                exps = (exps[:4] if phase == 'initial training' else exps[s:])
                y.append([])
                for obj in exps:
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'non-rewarded' in stimLbl else rewStim
                            if 'non-target' in stimLbl:
                                stim = stim[:-1]+'2'
                            trials = obj.trialStim==stim
                            if getDeltaLickProb and stim in obj.blockStimRewarded:
                                blockTrials = (obj.trialBlock==blockInd+1)
                                firstTarget = np.where(blockTrials & ~obj.autoRewardScheduled & np.isin(obj.trialStim,obj.blockStimRewarded))[0][0]
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
            if not getDeltaLickProb and stimLbl=='non-rewarded target':
                nonRewTargResp = y
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            if getDeltaLickProb and phase=='after learning' and stimLbl in ('rewarded target','non-rewarded target'):
                key = 'rewTarg' if stimLbl == 'rewarded target' else 'nonRewTarg'
                deltaLickProb['5 rewarded targets'][key] = np.array(y)[:,[preTrials-1,preTrials+5]]
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
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
    for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
        y = []
        for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
            exps = (exps[:5] if phase == 'initial training' else exps[s:])
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'non-rewarded' in stimLbl else rewStim
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
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_yticks([-0.1,-0.05,0,0.05,0.1])
    ax.set_ylim([-0.105,0.105])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=16)
    ax.set_ylabel('Response time\n(difference from mean, s)',fontsize=16)
    # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    # ax.set_title(phase+', '+str(len(y))+' mice',fontsize=12)
    plt.tight_layout()
    
# first trial lick or no lick  
for lbl in ('all blocks','first trial lick','first trial no lick'):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target','non-rewarded target'),'gm'):
        y = []
        for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
            exps = exps[s:]
            y.append([])
            for obj in exps:
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'non-rewarded' in stimLbl else rewStim
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
        if lbl == 'all blocks' and stimLbl == 'rewarded target':
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
    
# block switch plot for more learning phases and delay from last reward to first non-rewarded stim trial
learningPhases = ('initial training','early learning','late learning','after learning')
fullBlockRespDiff = {}
firstTrialRespDiff = {}
for phase in learningPhases:
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
                            stim = nonRewTarg if 'non-rewarded' in stimLbl else rewStim
                            if 'non-target' in stimLbl:
                                stim = stim[:-1]+'2'
                            if stim != rewStim:
                                firstNonRewTarg = np.where(blockTrials & (obj.trialStim==stim))[0][0]
                                trialsSinceRew = firstNonRewTarg - np.where(blockTrials & (obj.trialStim==rewStim))[0]
                            if stim == rewStim or trialsSinceRew[trialsSinceRew > 0][-1] > minTrialsSinceRew:
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
                if phase not in fullBlockRespDiff:
                    fullBlockRespDiff[phase] = []
                    firstTrialRespDiff[phase] = []
                y = np.array(y)
                fullBlockRespDiff[phase].append(np.nanmean(y[:,preTrials-5:preTrials],axis=1) - np.nanmean(y[:,preTrials+5:],axis=1))
                firstTrialRespDiff[phase].append(np.nanmean(y[:,preTrials-5:preTrials],axis=1) - y[:,preTrials+5])
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
x = np.arange(len(learningPhases))
for d,clr,lbl in zip((fullBlockRespDiff,firstTrialRespDiff),'rb',('previous block mean - new block mean','last trial previous block - first trial new block')):
    for i,(ls,mfc) in enumerate(zip(('-','--'),(clr,'none'))):
        m = [np.nanmean(d[phase][i]) for phase in d]
        s = [np.nanstd(d[phase][i])/(len(d[phase][i])**0.5) for phase in d]
        ax.plot(x,m,'o-',color=clr,mfc=mfc,ls=ls,label=(lbl if ls=='-' else lbl+'\n(first trial not immediately preceded by reward)'))
        for i,j,k in zip(x,m,s):
            ax.plot([i,i],[j-k,j+k],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(learningPhases)
ax.set_yticks([0,0.25,0.5])
ax.set_xlim([-0.5,len(learningPhases)-0.5])
ax.set_ylim([-0.1,0.51])
ax.set_ylabel('Change in response rate\n(non-rewarded target)',fontsize=12)
ax.legend(loc='upper left',fontsize=10)
plt.tight_layout()
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0],[0,1],'k--',alpha=0.5)
for phase,clr in zip(learningPhases,'rmbg'):
    d = firstTrialRespDiff[phase][1]
    dsort = np.sort(d)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,lw=2,label=phase)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-1,2,0.5))
ax.set_yticks([0,0.5,1])
ax.set_xlim([-0.7,1.01])
ax.set_ylim([0,1.01])
ax.set_xlabel('$\Delta$ Response probability\n(last trial previous block - first trial new block)',fontsize=14)
ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
plt.legend(loc='lower right',fontsize=12)
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
        sessions = np.array([lbl in task for task in df['task version']]) & np.array(df['has licks'].astype(bool))
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
lbl = 'nogo'
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials) 
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
                                        firstTarget = np.intersect1d(blockTrials[obj.newBlockNogoTrials:],np.where(np.isin(obj.trialStim,obj.blockStimRewarded)))[0]
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
                    if lbl=='nogo' and not getDeltaLickProb and not useFirstTrialLick and stimLbl=='rewarded target':
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

# block switch plots by first target type
lbl = 'rewardOnly' # 'noAR', 'rewardOnly', 'catchOnly'
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
                if lbl in ('rewardOnly','catchOnly') and firstTrialLick is None:
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

# block switch plots by first target and response type pooled across mice
lbl = 'noAR'
trialsSinceRewardRange = (None,) # (None,) or np.arange(1,4)
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

          
# block switch plots with non-target trials first
lbl = 'catchOnly' # 'rewardOnly', 'catchOnly'
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
                        firstOther = np.where(blockTrials & ~np.isin(obj.trialStim,('stim','catch')))[0][0]
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



## change in lick prob summary
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

# change in lick prob summary for reward only or catch only
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



## no reward blocks task variant
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
                exps = exps[:4] if phase=='initial training' else exps[sp:sp+4]
                for i,obj in enumerate(exps):
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
                                    anyTargetTrials = np.array([np.any(np.isin(obj.trialStim[p+1:s],(rewStim,otherModalTarget))) for s,p in zip(stimTrials,prevRespTrial)])
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
                ax.set_ylim([-0.15,0.25]) # [-0.1,0.2] [-0.15,0.25]
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
y = {phase: {prevTrial: {} for prevTrial in prevTrialTypes} for phase in trainingPhases}
for phase in trainingPhases:
    for blockRew in ('all',): # ('all',), ('vis1','sound1')
        for epoch in ('full',): # ('full',), ('first half','last half')
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
                    if blockRew=='all' and epoch=='full':
                        y[phase][prevTrialType][stim] = {'mean': m, 'sem': s}
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.5,0.5,0.1))
                ax.set_xlim([0,90])
                ax.set_ylim([-0.1,0.2]) # [-0.1,0.2] [-0.15,0.25]
                ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
                # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=16)
                plt.tight_layout()

for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for blockRew in ('vis1','sound1'): # ('all',), ('vis1','sound1')
            for epoch in ('full',): # ('full',), ('first half','last half')
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

# response rate to non-rewarded target
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
        ax.set_ylim([0.3,0.6])
        ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
        ax.set_ylabel('Response rate to non-rewarded target',fontsize=14)
        ax.legend(loc='upper left',fontsize=12)
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

phase = 'after learning'
fig = plt.figure()#(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
t = x
m,s = [y[phase]['response to rewarded target']['non-rewarded target'][key] for key in ('mean','sem')]
f1 = lambda t,tau,a,b: a * np.exp(-t/tau) + b
f2 = lambda t,tau,a,b: b - a * np.exp(-t/tau)
func = lambda t,tau1,tau2,a1,b1,a2,b2: (a1 * np.exp(-t/tau1) + b1) + (b2 - a2 * np.exp(-t/tau2))
tau1,tau2,a1,b1,a2,b2 = scipy.optimize.curve_fit(func,t[1:],m[1:],p0=(10,100,0.1,0,1,0.8),bounds=((3,60,0,0,0,0),(30,360,1,0.0001,1,1)))[0]
# ax.plot(t,m,'m',lw=3,label='non-rewarded target')
ax.fill_between(t,m-s,m+s,color='m',alpha=0.25,label='non-rewarded target')
ax.plot(t[1:],func(t[1:],tau1,tau2,a1,b1,a2,b2),'k',lw=2,alpha=0.5,label='fit (2 exponential functions)          ')
ax.plot(t[1:],f1(t[1:],tau1,a1,b1),'r--',label='effect of reward bias')
ax.plot(t[1:],f2(t[1:],tau2,a2,b2),'b--',label='effect of context updating/forgetting')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,90])
ax.set_yticks(np.arange(-0.5,0.5,0.1))
ax.set_ylim([-0.1,0.2])
ax.set_xlabel('Time since last response to rewarded target (s)',fontsize=14)
ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
# ax.set_title(str(round(tau1,1))+', '+str(round(tau2,1)))
ax.legend(loc='upper right',fontsize=12)
plt.tight_layout()



## comparison of response times and relation to task performance
respTime = {phase: {stim: {lbl: [] for lbl in ('rewarded','non-rewarded')} for stim in ('vis1','sound1')} for phase in ('initial training','after learning','all')}
respTimeNorm = copy.deepcopy(respTime)
dprime = copy.deepcopy(respTime)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:4]
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

alim  = (-0.15,0.21)       
for phase in ('after learning',):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    rtDiff = {stim: [] for stim in ('vis1','sound1')}
    for stim in rtDiff:
        for r,nr in zip(respTime[phase][stim]['rewarded'],respTime[phase][stim]['non-rewarded']):
            r,nr = [np.array([np.nanmean(np.concatenate(s)) for s in m]) for m in (r,nr)]
            rtDiff[stim].append(np.mean(nr - r))
    ax.plot(rtDiff['vis1'],rtDiff['sound1'],'o',mec='k',mfc='none',alpha=0.5)
    mx = np.nanmedian(rtDiff['vis1'])
    my = np.nanmedian(rtDiff['sound1'])
    sx = scipy.stats.median_abs_deviation(rtDiff['vis1'],nan_policy='omit')
    sy = scipy.stats.median_abs_deviation(rtDiff['sound1'],nan_policy='omit')
    ax.plot(mx,my,'ko',ms=12)
    ax.plot([mx-sx,mx+sx],[my,my],'k',lw=2)
    ax.plot([mx,mx],[my-sy,my+sy],'k',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(-0.1,0.3,0.1))
    ax.set_yticks(np.arange(-0.1,0.3,0.1))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Visual target',fontsize=14)
    ax.set_ylabel('Auditory target',fontsize=14)
    ax.set_title('Difference in response time (s)\n(non-rewarded - rewarded blocks)',fontsize=14)
    plt.tight_layout()
    
alim  = (-0.1,0.2)       
for phase in ('after learning',):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'k--')
    rtDiff = {stim: [] for stim in ('vis1','sound1')}
    for rewStim in ('vis1','sound1'):
        nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
        for r,nr in zip(respTimeNorm[phase][rewStim]['rewarded'],respTimeNorm[phase][nonRewStim]['non-rewarded']):
            r,nr = [np.array([np.nanmean(np.concatenate(s)) for s in m]) for m in (r,nr)]
            rtDiff[rewStim].append(np.mean(nr - r))
    ax.plot(rtDiff['vis1'],rtDiff['sound1'],'o',mec='k',mfc='none',alpha=0.5)
    mx = np.nanmedian(rtDiff['vis1'])
    my = np.nanmedian(rtDiff['sound1'])
    sx = scipy.stats.median_abs_deviation(rtDiff['vis1'],nan_policy='omit')
    sy = scipy.stats.median_abs_deviation(rtDiff['sound1'],nan_policy='omit')
    ax.plot(mx,my,'ko',ms=12)
    ax.plot([mx-sx,mx+sx],[my,my],'k',lw=2)
    ax.plot([mx,mx],[my-sy,my+sy],'k',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(-0.1,0.3,0.1))
    ax.set_yticks(np.arange(-0.1,0.3,0.1))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Visual rewarded blocks',fontsize=14)
    ax.set_ylabel('Auditory rewarded blocks',fontsize=14)
    ax.set_title('Difference in normalized response time (s)\n(non-rewarded - rewarded target)',fontsize=14)
    plt.tight_layout()

binWidth = 1 
binCenters = np.arange(0,4,binWidth)  
for phase in ('after learning',):
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
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(0,0.1,0.01))
    ax.set_xlim([-0.5,3.5])
    ax.set_ylim([0,0.06])
    ax.set_xlabel('Cross-modal d\'',fontsize=14)
    ax.set_ylabel('Difference in normalized response time (s)\n(non-rewarded - rewarded target)',fontsize=14)
    plt.tight_layout()
    


## comparison of visual and auditory response times
alim = (0.15,0.75)
for stage in ('initial training','after learning'):
    for blockRew,lbl in zip(('vis1','sound1'),('visual rewarded','auditory rewarded')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(alim,alim,'k--')
        rt = {'vis1': [], 'sound1': []}
        for exps,s in zip(sessionData,sessionsToPass):
            exps = exps[:4] if stage=='initial training' else exps[s:]
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
        ax.plot(rt['vis1'],rt['sound1'],'o',mec='k',mfc='none',alpha=0.5)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0,1,0.1))
        ax.set_yticks(np.arange(0,1,0.1))
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_xlabel('Response time to visual target (s)',fontsize=14)
        ax.set_ylabel('Response time to auditory target (s)',fontsize=14)
        # ax.set_title(stage+', '+lbl,fontsize=14)
        plt.tight_layout()
        


## effect of prior reward or response
respType = 'response'
prevTrialTypes = ('response to rewarded target','response to non-rewarded target','response to non-target (rewarded modality)','response to non-target (unrewarded modality)')
stimTypes = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
respNext = {phase: {prevTrialType: {blockType: {stim: [] for stim in stimTypes} for blockType in ('all','visual','auditory')} for prevTrialType in prevTrialTypes} for phase in ('initial training','after learning')}
respTimeNext = copy.deepcopy(respNext)
respPrev = copy.deepcopy(respNext)
respTimePrev = copy.deepcopy(respNext)
respPrevNoRew = copy.deepcopy(respNext)
respTimePrevNoRew = copy.deepcopy(respNext)
respMean = copy.deepcopy(respNext)
respTimeMean = copy.deepcopy(respNext)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes:
        for rewardStim,blockType in zip(('all','vis1','sound1'),('all','visual','auditory')):
            for s in stimTypes:
                for exps,sp in zip(sessionData,sessionsToPass):
                    exps = exps[:4] if phase=='initial training' else exps[sp:]
                    rn = []
                    rtn = []
                    rp = []
                    rtp = []
                    rpnr = []
                    rtpnr = []
                    rm = []
                    rtm = []
                    for obj in exps:
                        isRespType = obj.trialResponse if respType=='response' else ~obj.trialResponse
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
                                blockTrials = ~obj.autoRewardScheduled & (obj.trialBlock==blockInd+1)
                                trials = np.where(blockTrials & (obj.trialStim==stim))[0]
                                if prevTrialType == 'response to rewarded target':
                                    prevInd = isRespType & (obj.trialStim == rewStim) & ~obj.autoRewardScheduled
                                elif prevTrialType == 'response to non-rewarded target':
                                    prevInd = isRespType & (obj.trialStim == nonRewStim)
                                elif prevTrialType == 'response to non-target (rewarded modality)':
                                    prevInd = isRespType & (obj.trialStim == rewStim[:-1]+'2')
                                elif prevTrialType == 'response to non-target (unrewarded modality)':
                                    prevInd = isRespType & (obj.trialStim == nonRewStim[:-1]+'2')
                                if False:
                                    rn.append([])
                                    rtn.append([])
                                    rp.append([])
                                    rtp.append([])
                                    rpnr.append([])
                                    rtpnr.append([])
                                    for pt in np.where(prevInd & blockTrials)[0]:
                                        offset = trials - pt
                                        if np.any(offset > 0):
                                            i = np.where(offset > 0)[0][0]
                                            rn[-1].append(obj.trialResponse[trials[i]])
                                            rtn[-1].append(obj.responseTimes[trials[i]])
                                        if np.any(offset < 0):
                                            i = np.where(offset < 0)[0][-1]
                                            rp[-1].append(obj.trialResponse[trials[i]])
                                            rtp[-1].append(obj.responseTimes[trials[i]])
                                else:
                                    rn.append(obj.trialResponse[trials][prevInd[trials-1]])
                                    rtn.append(obj.responseTimes[trials][prevInd[trials-1]])
                                    prevInd = np.concatenate((prevInd,[False]))
                                    rp.append(obj.trialResponse[trials][prevInd[trials+1]])
                                    rtp.append(obj.responseTimes[trials][prevInd[trials+1]])
                                    rpnr.append(obj.trialResponse[trials][prevInd[trials+1] & ~obj.trialRewarded[trials-1]])
                                    rtpnr.append(obj.responseTimes[trials][prevInd[trials+1] & ~obj.trialRewarded[trials-1]])
                                rm.append(np.mean(obj.trialResponse[trials]))
                                rtm.append(np.nanmean(obj.responseTimes[trials]))
                    if len(rn) > 0:
                        rn = np.concatenate(rn)
                        rtn = np.concatenate(rtn)
                        rp = np.concatenate(rp)
                        rtp = np.concatenate(rtp)
                        rpnr = np.concatenate(rpnr)
                        rtpnr = np.concatenate(rtpnr)
                    respNext[phase][prevTrialType][blockType][s].append(np.nanmean(rn))
                    respTimeNext[phase][prevTrialType][blockType][s].append(np.nanmean(rtn))
                    respPrev[phase][prevTrialType][blockType][s].append(np.nanmean(rp))
                    respTimePrev[phase][prevTrialType][blockType][s].append(np.nanmean(rtp))
                    respPrevNoRew[phase][prevTrialType][blockType][s].append(np.nanmean(rpnr))
                    respTimePrevNoRew[phase][prevTrialType][blockType][s].append(np.nanmean(rtpnr))
                    respMean[phase][prevTrialType][blockType][s].append(np.nanmean(rm))
                    respTimeMean[phase][prevTrialType][blockType][s].append(np.nanmean(rtm))

stimLabels = ('rewarded target','non-rewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

blockType = 'all'
alim = (0,1.02)
for d,lbl in zip((respMean,respPrev,respPrevNoRew),('(within block mean)','trial t-1','trial t-1 (no reward t-2)')):
    for phase in ('initial training','after learning'):
        for prevTrialType in prevTrialTypes:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            ax.plot(alim,alim,'k--')
            for stim,mec,mfc in zip(stimTypes,'gmgm',('g','m','none','none')):
                ax.plot(d[phase][prevTrialType][blockType][stim],respNext[phase][prevTrialType][blockType][stim],'o',mec=mec,mfc=mfc,ms=10,label=stim)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=16)
            ax.set_xticks([0,0.5,1])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim(alim)
            ax.set_ylim(alim)
            ax.set_aspect('equal')
            ax.set_xlabel('Response prob '+lbl,fontsize=20)
            ax.set_ylabel('Response prob trial t+1',fontsize=20)
            ax.set_title('Conditioned on trial t '+prevTrialType,fontsize=20)
            # ax.legend(loc='lower right',fontsize=12)
            plt.tight_layout()

cmax = 0.35
for d,lbl in zip((respMean,respPrev,respPrevNoRew),('within block mean','response prob trial t-1','response prob trial t-1 (no reward t-2)')):
    for phase in ('initial training','after learning'):
        r = np.full((len(stimTypes),len(prevTrialTypes)),np.nan)    
        for i,stim in enumerate(stimTypes):
            for j,prevTrialType in enumerate(prevTrialTypes):
                r[i,j] = np.nanmean(np.array(respNext[phase][prevTrialType][blockType][stim]) - np.array(d[phase][prevTrialType][blockType][stim]))
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(r,cmap='bwr',clim=(-cmax,cmax))
        cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.04)
        # cb.set_ticks([-1,-0.5,0,0.5,1])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(stimLabels)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(stimLabels)
        ax.set_xlabel('Response to stimulus on trial t',fontsize=16)
        ax.set_ylabel('Response prob trial t+1\nminus '+lbl,fontsize=16)
        ax.set_title('Change in response probability',fontsize=16)

# response times
alim = (0.15,0.775)
for phase in ('initial training','after learning'):
    for prevTrialType in prevTrialTypes[:1]:
        for blockType,stimLabels,clrs in zip(('visual','auditory'),
                                             (('visual target','auditory target','visual non-target','auditory non-target'),('auditory target','visual target','auditory non-target','visual non-target')),
                                             ('gmgm','mgmg')):
            fig = plt.figure(figsize=(7.5,5))
            ax = fig.add_subplot(1,1,1)
            ax.plot(alim,alim,'k--')
            for stim,lbl,mec,mfc in zip(stimTypes[:2],stimLabels,clrs,(clrs[0],clrs[1],'none','none')):
                ax.plot(respTimeMean[phase][prevTrialType][blockType][stim],respTimeNext[phase][prevTrialType][blockType][stim],'o',color=mec,mec=mec,mfc='none',alpha=0.5,label=lbl)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xlim(alim)
            ax.set_ylim(alim)
            ax.set_aspect('equal')
            ax.set_xlabel('Response time (s)'+'\n(within-block mean)',fontsize=14)
            ax.set_ylabel('Response time (s)'+'\n(previous trial '+prevTrialType+')',fontsize=14)
            ax.legend(loc='lower right',fontsize=12)
            # ax.set_title(blockType+' rewarded blocks',fontsize=14)
            plt.tight_layout()



## intra-block resp correlations
nSessions = 4 # 2, 4
trainingPhases = ('initial training','after learning') # ('initial training','early learning','late learning','after learning')
trainingPhaseColors = 'mg' # 'rmbg'
blockRewStim = ('all','vis1','sound1') # ('all','vis1','sound1')
blockEpochs = ('full','first half','last half') # ('full','first half','last half')
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

stimLabels = ('rewarded target','non-rewarded target','non-target\n(rewarded modality)','non-target\n(non-rewarded modality)')

for d,ylim in zip((corrWithinRawMat,corrWithinMat,corrWithinDetrendMat),([-0.2,0.2],[-0.03,0.1],[-0.03,0.1])):
    fig = plt.figure(figsize=(10,10))          
    gs = matplotlib.gridspec.GridSpec(4,4)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:4]):
            ax = fig.add_subplot(gs[i,j])
            for phase,clr in zip(trainingPhases,trainingPhaseColors):
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
            else:
                ax.set_xticklabels([])
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            else:
                ax.set_yticklabels([])
            if i==0:
                ax.set_title(xlbl,fontsize=11)
    plt.tight_layout()
                
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
    for j,xlbl in enumerate(stimLabels[:]):
        ax = fig.add_subplot(gs[i,j])
        for phase,clr in zip(trainingPhases,trainingPhaseColors):
            mat = corrWithinDetrendMat[phase]['all']['full'][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=phase)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.025,0.09])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=12)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=12)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=12)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
plt.tight_layout()

fig = plt.figure(figsize=(8,8))          
gs = matplotlib.gridspec.GridSpec(4,2)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:2]):
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
        if i==0 and j==1:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
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

for i,stim in enumerate(stimLabels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'k--',alpha=0.5)
    for phase,clr in zip(trainingPhases,trainingPhaseColors):
        d = corrWithinDetrendMat[phase]['all']['full'][i,i,:,1]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,lw=2,label=phase)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks([-0.05,0,0.05,0.1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-0.07,0.125])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Autocorrelation of responses to '+stim.replace('\n',' '),fontsize=14)
    ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
    plt.legend(loc='lower right',fontsize=12)
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
plt.legend(fontsize=12)
plt.tight_layout() 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,0],'k--')
bw = 0.25
n = []
for phase,clr in zip(trainingPhases,trainingPhaseColors):
    r = respRateMat[phase]['all']['full'][1]
    c = corrWithinDetrendMat[phase]['all']['full'][1,1,:,1]
    bins = np.arange(bw/2,1,bw)
    n.append([])
    for b in bins:
        low = 0 if b==bins[0] else b-bw/2
        high = 1 if b==bins[-1] else b+bw/2
        d = c[(r>low) & (r<=high)]
        n[-1].append(len(d))
        if len(d)>4:
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            ax.plot(b,m,'o',mec=clr,mfc='none')
            ax.plot([b,b],[m-s,m+s],color=clr,label=(phase if b==bins[-1] else None))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,1])
ax.set_ylim([-0.01,0.05])
ax.set_xlabel('Response rate to non-rewarded target',fontsize=14)
ax.set_ylabel('Autocorrelation of responses',fontsize=14)
# plt.legend(loc='lower left')
plt.tight_layout()

# autocorrelation
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



















