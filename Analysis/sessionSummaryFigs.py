# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:59:09 2023

@author: svc_ccg
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData


f = r'//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data/638573/DynamicRouting1_638573_20220915_125610.hdf5'
obj = DynRoutData()
obj.loadBehavData(f)


# plot lick raster for all trials
preTime = 4
postTime = 4
lickRaster = []
fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(4,1)
ax = fig.add_subplot(gs[:3,0])
ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=obj.trialEndTimes[-1]+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=obj.trialEndTimes[-1]+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
for i,st in enumerate(obj.stimStartTimes):
    if not obj.engagedTrials[i]:
        ax.add_patch(matplotlib.patches.Rectangle([-preTime,obj.trialStartTimes[i]],width=preTime+postTime,height=obj.trialEndTimes[i]-obj.trialStartTimes[i],facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    lt = obj.lickTimes - st
    trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
    ax.vlines(trialLickTimes,st-preTime,st+postTime,colors='k')
    if obj.trialRewarded[i]:
        rt = obj.rewardTimes - st
        trialRewardTime = rt[(rt > 0) & (rt <= postTime)]
        mfc = 'b' if obj.autoRewarded[i] else 'none'
        ax.plot(trialRewardTime,st,'o',mec='b',mfc=mfc,ms=4)        
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-preTime,postTime])
ax.set_ylim([0,obj.trialEndTimes[-1]+1])
ax.set_ylabel('session time (s)')
title = ('all trials (n=' + str(obj.nTrials) + '), engaged (n=' + str(obj.engagedTrials.sum()) + ', gray)' +
         '\n' + 'filled blue circles: auto-reward, open circles: earned reward')
ax.set_title(title)
    
binSize = obj.minLickInterval
bins = np.arange(-preTime,postTime+binSize/2,binSize)
lickPsth = np.zeros((obj.nTrials,bins.size-1))    
for i,st in enumerate(obj.stimStartTimes):
    lickPsth[i] = np.histogram(obj.lickTimes[(obj.lickTimes >= st-preTime) & (obj.lickTimes <= st+postTime)]-st,bins)[0]
lickPsthMean = lickPsth.mean(axis=0) / binSize

ax = fig.add_subplot(gs[3,0])
ax.plot(bins[:-1]+binSize/2,lickPsthMean,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-preTime,postTime])
ax.set_ylim([0,1.01*lickPsthMean.max()])
ax.set_xlabel('time from stimulus onset (s)')
ax.set_ylabel('licks/s')


# plot lick raster for each block of trials
for blockInd,goStim in enumerate(obj.blockStimRewarded):
    blockTrials = obj.trialBlock == blockInd + 1
    nogoStim = np.unique(obj.trialStim[blockTrials & obj.nogoTrials])
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
    gs = matplotlib.gridspec.GridSpec(2,2)
    for trials,trialType in zip((obj.goTrials,obj.nogoTrials,obj.autoRewarded,obj.catchTrials),
                                ('go','no-go','auto reward','catch')):
        trials = trials & blockTrials
        i = 0 if trialType in ('go','no-go') else 1
        j = 0 if trialType in ('go','auto reward') else 1
        ax = fig.add_subplot(gs[i,j])
        ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=trials.sum()+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
        ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=trials.sum()+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
        for i,st in enumerate(obj.stimStartTimes[trials]):
            lt = obj.lickTimes - st
            trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
            ax.vlines(trialLickTimes,i+0.5,i+1.5,colors='k')       
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0.5,trials.sum()+0.5])
        ax.set_yticks([1,trials.sum()])
        ax.set_xlabel('time from stimulus onset (s)')
        ax.set_ylabel('trial')
        title = trialType + ' trials (n=' + str(trials.sum()) + ', ' + str(obj.engagedTrials[trials].sum()) + ' engaged)'
        if trialType == 'go':
            title += '\n' + 'hit rate ' + str(round(obj.hitRate[blockInd],2)) + ', # hits ' + str(int(obj.hitCount[blockInd]))
        elif trialType == 'no-go':
            title = title[:-1] + ', ' + str(obj.trialRepeat[trials].sum()) + ' repeats)' 
            title += ('\n'+ 'false alarm same ' + str(round(obj.falseAlarmSameModal[blockInd],2)) + 
                      ', diff go ' + str(round(obj.falseAlarmOtherModalGo[blockInd],2)) +
                      ', diff nogo ' + str(round(obj.falseAlarmOtherModalNogo[blockInd],2)) +
                      '\n' + 'dprime same ' + str(round(obj.dprimeSameModal[blockInd],2)) +
                      ', diff go ' + str(round(obj.dprimeOtherModalGo[blockInd],2)))
        elif trialType == 'catch':
            title += '\n' + 'catch rate ' + str(round(obj.catchResponseRate[blockInd],2))
        ax.set_title(title)   

for blockInd,goStim in enumerate(obj.blockStimRewarded):
    blockTrials = obj.trialBlock == blockInd + 1
    nogoStim = np.unique(obj.trialStim[blockTrials & obj.nogoTrials])
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
    gs = matplotlib.gridspec.GridSpec(2,2)
    for trials,trialType in zip((obj.goTrials,obj.nogoTrials,obj.autoRewarded,obj.catchTrials),
                                ('go','no-go','auto reward','catch')):
        trials = trials & blockTrials
        i = 0 if trialType in ('go','no-go') else 1
        j = 0 if trialType in ('go','auto reward') else 1
        ax = fig.add_subplot(gs[i,j])
        ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=obj.trialEndTimes[-1]+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
        ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=obj.trialEndTimes[-1]+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
        for i,st in enumerate(obj.stimStartTimes[trials]):
            lt = obj.lickTimes - st
            trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
            ax.vlines(trialLickTimes,st-preTime,st+postTime,colors='k')       
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,obj.trialEndTimes[-1]+1])
        ax.set_xlabel('time from stimulus onset (s)')
        ax.set_ylabel('session time (s)')
        title = trialType + ' trials (n=' + str(trials.sum()) + ', ' + str(obj.engagedTrials[trials].sum()) + ' engaged)'
        if trialType == 'go':
            title += '\n' + 'hit rate ' + str(round(obj.hitRate[blockInd],2)) + ', # hits ' + str(int(obj.hitCount[blockInd]))
        elif trialType == 'no-go':
            title = title[:-1] + ', ' + str(obj.trialRepeat[trials].sum()) + ' repeats)' 
            title += ('\n'+ 'false alarm same ' + str(round(obj.falseAlarmSameModal[blockInd],2)) + 
                      ', diff go ' + str(round(obj.falseAlarmOtherModalGo[blockInd],2)) +
                      ', diff nogo ' + str(round(obj.falseAlarmOtherModalNogo[blockInd],2)) +
                      '\n' + 'dprime same ' + str(round(obj.dprimeSameModal[blockInd],2)) +
                      ', diff go ' + str(round(obj.dprimeOtherModalGo[blockInd],2)))
        elif trialType == 'catch':
            title += '\n' + 'catch rate ' + str(round(obj.catchResponseRate[blockInd],2))
        ax.set_title(title)   
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

       
# plot lick latency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
stimLabels = np.unique(obj.trialStim)
notCatch = stimLabels != 'catch'
clrs = np.zeros((len(stimLabels),3)) + 0.5
clrs[notCatch] = plt.cm.plasma(np.linspace(0,0.85,notCatch.sum()))[:,:3]
for stim,clr in zip(stimLabels,clrs):
    trials = (obj.trialStim==stim) & obj.trialResponse
    rt = obj.responseTimes[trials]
    rtSort = np.sort(rt)
    cumProb = [np.sum(rt<=i)/rt.size for i in rtSort]
    ax.plot(rtSort,cumProb,color=clr,label=stim)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,obj.responseWindowTime[1]+0.1])
ax.set_ylim([0,1.02])
ax.set_xlabel('response time (s)')
ax.set_ylabel('cumulative probability')
ax.legend()
plt.tight_layout()


# plot mean running speed for each block of trials
runPlotTime = np.arange(-preTime,postTime+1/obj.frameRate,1/obj.frameRate)
if obj.runningSpeed is not None:
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock == blockInd + 1
        nogoStim = np.unique(obj.trialStim[blockTrials & obj.nogoTrials])
        fig = plt.figure(figsize=(8,8))
        fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
        gs = matplotlib.gridspec.GridSpec(2,2)
        axs = []
        ymax = 1
        for trials,trialType in zip((obj.goTrials,obj.nogoTrials,obj.autoRewarded,obj.catchTrials),
                                    ('go','no-go','auto reward','catch')):
            trials = trials & blockTrials
            i = 0 if trialType in ('go','no-go') else 1
            j = 0 if trialType in ('go','auto reward') else 1
            ax = fig.add_subplot(gs[i,j])
            ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=100,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
            ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=100,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
            if trials.sum() > 0:
                speed = []
                for st in obj.stimStartTimes[trials]:
                    if st >= preTime and st+postTime <= obj.frameTimes[-1]:
                        i = (obj.frameTimes >= st-preTime) & (obj.frameTimes <= st+postTime)
                        speed.append(np.interp(runPlotTime,obj.frameTimes[i]-st,obj.runningSpeed[i]))
                meanSpeed = np.nanmean(speed,axis=0)
                ymax = max(ymax,meanSpeed.max())
                ax.plot(runPlotTime,meanSpeed)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([-preTime,postTime])
            ax.set_xlabel('time from stimulus onset (s)')
            ax.set_ylabel('mean running speed (cm/s)')
            ax.set_title(trialType + ' trials (n=' + str(trials.sum()) + '), engaged (n=' + str(obj.engagedTrials[trials].sum()) + ')')
            axs.append(ax)
        for ax in axs:
            ax.set_ylim([0,1.05*ymax])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# plot frame intervals
longFrames = obj.frameIntervals > 1.5/obj.frameRate

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(-0.5/obj.frameRate,obj.frameIntervals.max()+1/obj.frameRate,1/obj.frameRate)
ax.hist(obj.frameIntervals,bins=bins,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_yscale('log')
ax.set_xlabel('frame interval (s)')
ax.set_ylabel('count')
ax.set_title(str(round(100 * longFrames.sum() / longFrames.size,2)) + '% of frames long')
plt.tight_layout()


# plot quiescent violations
trialQuiescentViolations = []
for sf,ef in zip(obj.trialStartFrame,obj.trialEndFrame):
    trialQuiescentViolations.append(np.sum((obj.quiescentViolationFrames > sf) & (obj.quiescentViolationFrames < ef)))

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(2,1,1)
if obj.quiescentViolationFrames.size > 0:
    ax.plot(obj.frameTimes[obj.quiescentViolationFrames],np.arange(obj.quiescentViolationFrames.size)+1,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('time (s)')
ax.set_ylabel('quiescent period violations')

ax = fig.add_subplot(2,1,2)
bins = np.arange(-0.5,max(trialQuiescentViolations)+1,1)
ax.hist(trialQuiescentViolations,bins=bins,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('quiescent period violations per trial')
ax.set_ylabel('trials')
plt.tight_layout()


# plot inter-trial intervals
interTrialIntervals = np.diff(obj.frameTimes[obj.stimStartFrame])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(interTrialIntervals.max()+1)
ax.hist(interTrialIntervals,bins=bins,color='k',label='all trials')
ax.hist(interTrialIntervals[np.array(trialQuiescentViolations[1:]) == 0],bins=bins,color='0.5',label='trials without quiescent period violations')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,interTrialIntervals.max()+1])
ax.set_xlabel('inter-trial interval (s)')
ax.set_ylabel('trials')
ax.legend()
plt.tight_layout()


# plot running speed
if obj.runningSpeed is not None:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(obj.frameTimes,obj.runningSpeed[:obj.frameTimes.size],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,obj.frameTimes[-1]])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('running speed (cm/s)')
    plt.tight_layout()
