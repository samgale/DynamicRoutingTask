# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data"

filePaths = fileIO.getFiles(rootDir=baseDir,fileType='*.hdf5')

makeSummaryPDF = True

for f in filePaths:

    if makeSummaryPDF:
        saveDir = os.path.join(os.path.dirname(f),'summary')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        pdf = PdfPages(os.path.join(saveDir,os.path.splitext(os.path.basename(f))[0]+'_summary.pdf'))
    
    # get data
    d = h5py.File(f,'r')
        
    frameRate = 60
    frameIntervals = d['frameIntervals'][:]
    frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
    
    lickFrames = d['lickFrames'][:]
    lickTimesDetected = frameTimes[lickFrames]
    minLickInterval = 0.05
    isLick = np.concatenate(([True], np.diff(lickTimesDetected) > minLickInterval))
    lickTimes = lickTimesDetected[isLick]
    
    trialEndFrame = d['trialEndFrame'][:]
    nTrials = trialEndFrame.size
    trialStartFrame = d['trialStartFrame'][:nTrials]
    stimStartFrame = d['trialStimStartFrame'][:nTrials]
    stimStartTimes = frameTimes[stimStartFrame]
    
    quiescentFrames = d['quiescentFrames'][()]
    quiescentViolationFrames = d['quiescentViolationFrames'][:] if 'quiescentViolationFrames' in d.keys() else d['quiescentMoveFrames'][:]    
    
    responseWindow = d['responseWindow'][:]
    responseWindowTime = np.array(responseWindow)/frameRate
    
    trialStim = d['trialStim'][:nTrials]
    rewardedStim = d['blockStimRewarded'][:nTrials][d['trialBlock'][:nTrials]-1]
    
    trialResponse = d['trialResponse'][:nTrials]
    trialRewarded = d['trialRewarded'][:nTrials]
    autoRewarded = d['trialAutoRewarded'][:nTrials]
    rewardEarned = trialRewarded & (~autoRewarded)
    rewardFrames = d['rewardFrames']
    rewardTimes = frameTimes[rewardFrames]
    
    catchTrials = trialStim == 'catch'
    goTrials = (trialStim == rewardedStim) & (~autoRewarded)
    nogoTrials = (trialStim != rewardedStim) & (~catchTrials)
    
    assert(nTrials == goTrials.sum() + nogoTrials.sum() + autoRewarded.sum() + catchTrials.sum())
    
    hitTrials = goTrials & trialResponse
    missTrials = goTrials & (~trialResponse)
    falseAlarmTrials = nogoTrials & trialResponse
    correctRejectTrials = nogoTrials & (~trialResponse)
    catchResponseTrials = catchTrials & trialResponse
    
    engagedThresh = 10
    engagedTrials = np.ones(nTrials,dtype=bool)
    for i in range(nTrials):
        r = trialResponse[:i+1][goTrials[:i+1]]
        if r.size > engagedThresh:
            if r[-engagedThresh:].sum() < 1:
                engagedTrials[i] = False
                
    hitRate = hitTrials[engagedTrials].sum() / goTrials[engagedTrials].sum()
    falseAlarmRate = falseAlarmTrials[engagedTrials].sum() / nogoTrials[engagedTrials].sum()
    catchResponseRate = catchResponseTrials[engagedTrials].sum() / catchTrials[engagedTrials].sum()
    
    
    # plot frame intervals
    longFrames = frameIntervals > 1.5/frameRate
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bins = np.arange(-0.5/frameRate,frameIntervals.max()+1/frameRate,1/frameRate)
    ax.hist(frameIntervals,bins=bins,color='k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_yscale('log')
    ax.set_xlabel('frame interval (s)')
    ax.set_ylabel('count')
    ax.set_title(str(round(100 * longFrames.sum() / longFrames.size,2)) + '% of frames long')
    plt.tight_layout()
    
    if makeSummaryPDF:
        fig.savefig(pdf,format='pdf')
    
    
    # plot quiescent violations
    trialQuiescentViolations = []
    for sf,ef in zip(trialStartFrame,trialEndFrame):
        trialQuiescentViolations.append(np.sum((quiescentViolationFrames > sf) & (quiescentViolationFrames < ef)))
    
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(2,1,1)
    if quiescentViolationFrames.size > 0:
        ax.plot(frameTimes[quiescentViolationFrames],np.arange(quiescentViolationFrames.size)+1,'k')
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
    
    if makeSummaryPDF:
        fig.savefig(pdf,format='pdf')
        
    
    # plot inter-trial intervals
    interTrialIntervals = np.diff(frameTimes[stimStartFrame])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bins = np.arange(interTrialIntervals.max()+1)
    ax.hist([interTrialIntervals[np.array(trialQuiescentViolations[1:]) == 0],interTrialIntervals],
            bins=bins,stacked=True,color=('0.5','k'),label=('trials without quiescent period violations','all trials'))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,interTrialIntervals.max()+1])
    ax.set_xlabel('inter-trial interval (s)')
    ax.set_ylabel('trials')
    ax.legend()
    plt.tight_layout()
    
    if makeSummaryPDF:
        fig.savefig(pdf,format='pdf')
    
    
    # plot lick raster
    preTime = 4
    postTime = 4
    lickRaster = []
    for trials,trialType in zip((np.ones(nTrials,dtype=bool),goTrials,nogoTrials,autoRewarded,catchTrials),
                                ('all','go','no-go','auto reward','catch')):
        if trialType == 'all':
            allTrialsFig = plt.figure(figsize=(8,8))
            allTrialsGridspec = matplotlib.gridspec.GridSpec(4,1)
            ax = allTrialsFig.add_subplot(allTrialsGridspec[:3,0])
        else:
            if trialType == 'go':
                fig = plt.figure(figsize=(8,8))
                gs = matplotlib.gridspec.GridSpec(2,2)
            i = 0 if trialType in ('go','no-go') else 1
            j = 0 if trialType in ('go','auto reward') else 1
            ax = fig.add_subplot(gs[i,j])
        ax.add_patch(matplotlib.patches.Rectangle([-quiescentFrames/frameRate,0],width=quiescentFrames/frameRate,height=trials.sum()+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
        ax.add_patch(matplotlib.patches.Rectangle([responseWindowTime[0],0],width=np.diff(responseWindowTime),height=trials.sum()+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
        respTrials = trialResponse[trials]
        for i,st in enumerate(stimStartTimes[trials]):
            lt = lickTimes - st
            trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
            clr = 'm' if respTrials[i] else 'k'
            ax.vlines(trialLickTimes,i+0.5,i+1.5,colors=clr)
            if trialType == 'all':
                lickRaster.append(trialLickTimes)
                if not engagedTrials[i]:
                    ax.add_patch(matplotlib.patches.Rectangle([-preTime,i+0.5],width=preTime+postTime,height=1,facecolor='0.5',edgecolor='0.5',alpha=0.2,zorder=0))
                if trialRewarded[i]:
                    rt = rewardTimes - st
                    trialRewardTime = rt[(rt > 0) & (rt <= postTime)]
                    mfc = 'b' if autoRewarded[i] else 'none'
                    ax.plot(trialRewardTime,i+1,'o',mec='b',mfc=mfc,ms=4)        
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0.5,trials.sum()+0.5])
        ax.set_yticks([1,trials.sum()])
        ax.set_ylabel('trial')
        title = trialType + ' trials (n=' + str(trials.sum()) + '), engaged (n=' + str(engagedTrials[trials].sum()) + ')'
        if trialType == 'all':
            title = title[:-1] + ', not gray)'
            title += ('\n' + 'magenta ticks: trials with response' +
                      '\n' + 'filled blue circles: auto reward, open circles: earned reward')
        elif trialType == 'go':
            title += '\n' + 'hit rate ' + str(round(hitRate,2))
        elif trialType == 'no-go':
            title += '\n' + 'false alarm rate ' + str(round(falseAlarmRate,2))
        elif trialType == 'catch':
            title += '\n' + 'catch rate ' + str(round(catchResponseRate,2))
        ax.set_title(title)
        if trialType != 'all':
            ax.set_xlabel('time from stimulus onset (s)')
            fig.tight_layout()
        
    binSize = minLickInterval
    bins = np.arange(-preTime,postTime+binSize/2,binSize)
    lickPsth = np.zeros((nTrials,bins.size-1))    
    for i,st in enumerate(stimStartTimes):
        lickPsth[i] = np.histogram(lickTimes[(lickTimes >= st-preTime) & (lickTimes <= st+postTime)]-st,bins)[0]
    lickPsthMean = lickPsth.mean(axis=0) / binSize
    
    ax = allTrialsFig.add_subplot(allTrialsGridspec[3,0])
    ax.plot(bins[:-1]+binSize/2,lickPsthMean,color='k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-preTime,postTime])
    ax.set_ylim([0,1.01*lickPsthMean.max()])
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('licks/s')
    allTrialsFig.tight_layout()
    
    if makeSummaryPDF:
        allTrialsFig.savefig(pdf,format='pdf')
        fig.savefig(pdf,format='pdf')
    
    
    # clean up
    d.close()
    
    if makeSummaryPDF:
        pdf.close()
        plt.close('all')
