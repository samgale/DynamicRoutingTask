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



## run speed by session
runSpeed = {phase: {blockType: [] for blockType in ('vis rewarded','aud rewarded','all')} for phase in ('initial training','after learning','all')}
dprime = copy.deepcopy(runSpeed)
for phase in ('initial training','after learning','all'):
    for mouseInd,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
        if phase=='initial training':
            exps = exps[:4]
        elif phase=='after learning':
            exps = exps[s:s+4]
        for blockType in ('vis rewarded','aud rewarded','all'):
            for d in (runSpeed,dprime):
                d[phase][blockType].append([[] for _ in range(len(exps))])
            for sessionInd,obj in enumerate(exps):
                stimTrials = np.isin(obj.trialStim,('vis1','sound1')) & ~obj.autoRewardScheduled
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockType=='all' or (blockType=='vis rewarded' and rewStim=='vis1') or (blockType=='aud rewarded' and rewStim=='sound1'):
                        trials = stimTrials & (obj.trialBlock==blockInd+1)
                        runSpeed[phase][blockType][mouseInd][sessionInd].append(np.nanmean(obj.quiescentRunSpeed[trials]))  
                        dprime[phase][blockType][mouseInd][sessionInd].append(obj.dprimeOtherModalGo[blockInd])
                        
for phase in ('initial training','after learning'):
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
    ax.set_title(str(i.sum()) + ' sessions',fontsize=16)
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
    for k,stage in enumerate(('all',)):
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

clust = 5
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

# intra-block resp correlations for session clusters




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





















