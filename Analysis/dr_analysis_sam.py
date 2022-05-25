# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import os
import re
import time
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data"


class DynRoutData():
    
    def __init__(self):
        self.frameRate = 60
        self.engagedThresh = 10
    
    
    def loadBehavData(self,filePath=None):
        if filePath is None:
            self.behavDataPath = fileIO.getFile('Select behavior data file',rootDir=baseDir,fileType='*.hdf5')
        else:
            self.behavDataPath = filePath
        if len(self.behavDataPath)==0:
            return
        
        d = h5py.File(f,'r')
        
        # self.subjectName = d['subjectName'][()]
        self.subjectName = re.search('.*_([0-9]{6})_',os.path.basename(self.behavDataPath)).group(1)
        self.rigName = d['rigName'][()]
        self.taskVersion = d['taskVersion'][()] if 'taskVersion' in d.keys() else None
        self.soundType = d['soundType'][()]
        self.startTime = d['startTime'][()]
            
        self.frameIntervals = d['frameIntervals'][:]
        self.frameTimes = np.concatenate(([0],np.cumsum(self.frameIntervals)))
        
        self.trialEndFrame = d['trialEndFrame'][:]
        self.trialEndTimes = self.frameTimes[self.trialEndFrame]
        self.nTrials = self.trialEndFrame.size
        self.trialStartFrame = d['trialStartFrame'][:self.nTrials]
        self.trialStartTimes = self.frameTimes[self.trialStartFrame]
        self.stimStartFrame = d['trialStimStartFrame'][:self.nTrials]
        self.stimStartTimes = self.frameTimes[self.stimStartFrame]
        self.trialRepeat = d['trialRepeat'][:self.nTrials]
        
        self.quiescentFrames = d['quiescentFrames'][()]
        self.quiescentViolationFrames = d['quiescentViolationFrames'][:] if 'quiescentViolationFrames' in d.keys() else d['quiescentMoveFrames'][:]    
        
        self.responseWindow = d['responseWindow'][:]
        self.responseWindowTime = np.array(self.responseWindow)/self.frameRate
        
        self.trialStim = d['trialStim'][:self.nTrials]
        self.trialBlock = d['trialBlock'][:self.nTrials]
        self.blockStartTimes = self.trialStartTimes[[np.where(self.trialBlock==i)[0][0] for i in np.unique(self.trialBlock)]]
        self.blockFirstStimTimes = self.stimStartTimes[[np.where(self.trialBlock==i)[0][0] for i in np.unique(self.trialBlock)]]
        self.blockStimRewarded = d['blockStimRewarded'][:]
        self.rewardedStim = self.blockStimRewarded[self.trialBlock-1]
        
        self.trialResponse = d['trialResponse'][:self.nTrials]
        self.trialResponseFrame = d['trialResponseFrame'][:self.nTrials]
        self.trialRewarded = d['trialRewarded'][:self.nTrials]
        self.autoRewarded = d['trialAutoRewarded'][:self.nTrials]
        self.rewardEarned = self.trialRewarded & (~self.autoRewarded)
        self.rewardFrames = d['rewardFrames'][:]
        self.rewardTimes = self.frameTimes[self.rewardFrames]
        
        self.responseTimes = np.full(self.nTrials,np.nan)
        self.responseTimes[self.trialResponse] = self.frameTimes[self.trialResponseFrame[self.trialResponse].astype(int)] - self.stimStartTimes[self.trialResponse]
        
        self.lickFrames = d['lickFrames'][:]
        if len(self.lickFrames) > 0:
            lickTimesDetected = self.frameTimes[self.lickFrames]
            self.minLickInterval = 0.05
            isLick = np.concatenate(([True], np.diff(lickTimesDetected) > self.minLickInterval))
            self.lickTimes = lickTimesDetected[isLick]
        else:
            self.lickTimes = np.array([])
        
        if 'rotaryEncoder' in d and d['rotaryEncoder'][()] == 'digital':
            self.runningSpeed = np.concatenate(([np.nan],np.diff(d['rotaryEncoderCount'][:]) / d['rotaryEncoderCountsPerRev'][()] * 2 * np.pi * d['wheelRadius'][()] * self.frameRate))
        else:
            self.runningSpeed = None
        
        self.visContrast = d['visStimContrast'][()]
        self.trialVisContrast = d['trialVisStimContrast'][:self.nTrials]
        if 'gratingOri' in d:
            self.gratingOri = {key: d['gratingOri'][key][()] for key in d['gratingOri']}
        else:
            self.gratingOri = {key: d['gratingOri_'+key][()] for key in ('vis1','vis2')}
        self.trialGratingOri = d['trialGratingOri'][:self.nTrials]
            
        d.close()
        
        self.catchTrials = self.trialStim == 'catch'
        self.goTrials = (self.trialStim == self.rewardedStim) & (~self.autoRewarded)
        self.nogoTrials = (self.trialStim != self.rewardedStim) & (~self.catchTrials)
        self.sameModalNogoTrials = self.nogoTrials & np.array([stim[:-1]==rew[:-1] for stim,rew in zip(self.trialStim,self.rewardedStim)])
        if 'distract' in self.taskVersion:
            self.otherModalGoTrials = self.nogoTrials & np.in1d(self.trialStim,('vis1','sound1'))
        else:
            self.otherModalGoTrials = self.nogoTrials & np.in1d(self.trialStim,self.blockStimRewarded)
        self.otherModalNogoTrials = self.nogoTrials & ~self.sameModalNogoTrials & ~self.otherModalGoTrials
        
        self.hitTrials = self.goTrials & self.trialResponse
        self.missTrials = self.goTrials & (~self.trialResponse)
        self.falseAlarmTrials =self. nogoTrials & self.trialResponse
        self.correctRejectTrials = self.nogoTrials & (~self.trialResponse)
        self.catchResponseTrials = self.catchTrials & self.trialResponse
        
        self.engagedTrials = np.ones(self.nTrials,dtype=bool)
        for i in range(self.nTrials):
            r = self.trialResponse[:i+1][self.goTrials[:i+1]]
            if r.size > self.engagedThresh:
                if r[-self.engagedThresh:].sum() < 1:
                    self.engagedTrials[i] = False
        
        self.catchResponseRate = []
        self.hitRate = []
        self.hitCount = []
        self.falseAlarmRate = []
        self.falseAlarmSameModal = []
        self.falseAlarmOtherModalGo = []
        self.falseAlarmOtherModalNogo = []
        self.dprimeSameModal = []
        self.dprimeOtherModalGo = []
        for blockInd,rew in enumerate(self.blockStimRewarded):
            blockTrials = (self.trialBlock == blockInd + 1) & self.engagedTrials & (~self.trialRepeat)
            self.catchResponseRate.append(self.catchResponseTrials[blockTrials].sum() / self.catchTrials[blockTrials].sum())
            self.hitRate.append(self.hitTrials[blockTrials].sum() / self.goTrials[blockTrials].sum())
            self.hitCount.append(self.hitTrials[blockTrials].sum())
            self.falseAlarmRate.append(self.falseAlarmTrials[blockTrials].sum() / self.nogoTrials[blockTrials].sum())
            sameModal = blockTrials & self.sameModalNogoTrials
            otherModalGo = blockTrials & self.otherModalGoTrials
            otherModalNogo = blockTrials & self.otherModalNogoTrials
            self.falseAlarmSameModal.append(self.falseAlarmTrials[sameModal].sum() / sameModal.sum())
            self.falseAlarmOtherModalGo.append(self.falseAlarmTrials[otherModalGo].sum() / otherModalGo.sum())
            self.falseAlarmOtherModalNogo.append(self.falseAlarmTrials[otherModalNogo].sum() / otherModalNogo.sum())
            self.dprimeSameModal.append(calcDprime(self.hitRate[-1],self.falseAlarmSameModal[-1],self.goTrials[blockTrials].sum(),sameModal.sum()))
            self.dprimeOtherModalGo.append(calcDprime(self.hitRate[-1],self.falseAlarmOtherModalGo[-1],self.goTrials[blockTrials].sum(),otherModalGo.sum()))
# end DynRoutData
    

def calcDprime(hitRate,falseAlarmRate,goTrials,nogoTrials):
        hr = adjustResponseRate(hitRate,goTrials)
        far = adjustResponseRate(falseAlarmRate,nogoTrials)
        z = [scipy.stats.norm.ppf(r) for r in (hr,far)]
        return z[0]-z[1]


def adjustResponseRate(r,n):
    if r == 0:
        r = 0.5/n
    elif r == 1:
        r = 1 - 0.5/n
    return r
    

def sortExps(exps):
    startTimes = [time.strptime(obj.startTime,'%Y%m%d_%H%M%S') for obj in exps]
    return [z[0] for z in sorted(zip(exps,startTimes),key=lambda i: i[1])]


def makeSummaryPdf(obj):
    saveDir = os.path.join(os.path.dirname(obj.behavDataPath),'summary')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    pdf = PdfPages(os.path.join(saveDir,os.path.splitext(os.path.basename(obj.behavDataPath))[0]+'_summary.pdf'))
    
    # plot lick raster (all trials)
    preTime = 4
    postTime = 4
    lickRaster = []
    fig = plt.figure(figsize=(8,8))
    gs = matplotlib.gridspec.GridSpec(4,1)
    ax = fig.add_subplot(gs[:3,0])
    ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=obj.nTrials+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
    ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=obj.nTrials+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
    for i,st in enumerate(obj.stimStartTimes):
        if not obj.engagedTrials[i]:
            ax.add_patch(matplotlib.patches.Rectangle([-preTime,i+0.5],width=preTime+postTime,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        lt = obj.lickTimes - st
        trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
        lickRaster.append(trialLickTimes)
        ax.vlines(trialLickTimes,i+0.5,i+1.5,colors='k')
        if obj.trialRewarded[i]:
            rt = obj.rewardTimes - st
            trialRewardTime = rt[(rt > 0) & (rt <= postTime)]
            mfc = 'b' if obj.autoRewarded[i] else 'none'
            ax.plot(trialRewardTime,i+1,'o',mec='b',mfc=mfc,ms=4)        
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-preTime,postTime])
    ax.set_ylim([0.5,obj.nTrials+0.5])
    ax.set_yticks([1,obj.nTrials])
    ax.set_ylabel('trial')
    title = (obj.subjectName + ', ' + obj.rigName + ', ' + obj.taskVersion + 
             '\n' + 'all trials (n=' + str(obj.nTrials) + '), engaged (n=' + str(obj.engagedTrials.sum()) + ', not gray)' +
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
    fig.tight_layout()
    fig.savefig(pdf,format='pdf')
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
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
    title = ('all trials (n=' + str(obj.nTrials) + '), engaged (n=' + str(obj.engagedTrials.sum()) + ', not gray)' +
             '\n' + 'filled blue circles: auto-reward, open circles: earned reward')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(pdf,format='pdf')
    
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
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(pdf,format='pdf')
    
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
        fig.savefig(pdf,format='pdf')
           
    # ori
    if len(obj.gratingOri['vis2']) > 1:
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = (obj.trialBlock == blockInd + 1) & ~obj.autoRewarded & ~obj.catchTrials
            oris = np.unique(obj.trialGratingOri)
            r = []
            for ori in oris:
                trials = blockTrials & (obj.trialGratingOri == ori)
                r.append(obj.trialResponse[trials].sum() / trials.sum())
            fig = plt.figure()
            fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
            ax = fig.add_subplot(1,1,1)
            ax.plot(oris,r,'ko-')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_ylim([0,1.02])
            ax.set_xlabel('ori')
            ax.set_ylabel('response rate')
            plt.tight_layout()
            fig.savefig(pdf,format='pdf')
                    
    # contrast
    if len(obj.visContrast) > 1:
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock == blockInd + 1
            fig = plt.figure()
            fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
            ax = fig.add_subplot(1,1,1)
            for trials,lbl in zip((obj.goTrials,obj.nogoTrials),('go','nogo')):
                r = []
                for c in obj.visContrast:
                    tr = trials & blockTrials & (obj.trialVisContrast == c)
                    r.append(obj.trialResponse[tr].sum() / tr.sum())
                ls,mfc = ('-','k') if lbl=='go' else ('--','none')
                ax.plot(obj.visContrast,r,'ko',ls=ls,mfc=mfc,label=lbl)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_ylim([0,1.02])
            ax.set_xlabel('contrast')
            ax.set_ylabel('response rate')
            ax.legend()
            plt.tight_layout()
            fig.savefig(pdf,format='pdf')
            
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
    fig.savefig(pdf,format='pdf')
    
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
            fig.savefig(pdf,format='pdf')
    
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
    fig.savefig(pdf,format='pdf')
    
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
    fig.savefig(pdf,format='pdf')
    
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
    fig.savefig(pdf,format='pdf')
    
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
        fig.savefig(pdf,format='pdf')
    
    pdf.close()
    plt.close('all')
    
    
# get data
behavFiles = []
while True:
    files = fileIO.getFiles('choose experiments',rootDir=baseDir,fileType='*.hdf5')
    if len(files)>0:
        behavFiles.extend(files)
    else:
        break
    
if len(behavFiles)>0:
    exps = []
    for f in behavFiles:
        obj = DynRoutData()
        obj.loadBehavData(f)
        exps.append(obj)
        
# sort experiments by start time
exps = sortExps(exps)


# summary pdf
for obj in exps:
    makeSummaryPdf(obj)
    

# print summary
for obj in exps:
    print(obj.subjectName)
    for i,d in enumerate((obj.hitCount,obj.dprimeSameModal,obj.dprimeOtherModalGo)):
        if i>0:
            d = np.round(d,2)
        print(*d,sep=', ')
    print('\n')
    

# write to excel
excelPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\DynamicRoutingTraining.xlsx"
for obj in exps:
    data = {'date': pd.to_datetime(obj.startTime,format='%Y%m%d_%H%M%S'),
            'task version': obj.taskVersion,
            'hits': obj.hitCount,
            'd\' same modality': np.round(obj.dprimeSameModal,2),
            'd\' other modality go stim': np.round(obj.dprimeOtherModalGo,2),
            'pass': 0}
    sheets = pd.read_excel(excelPath,sheet_name=None)
    if obj.subjectName in sheets:
        df = sheets[obj.subjectName]
        dateInd = df['date'] == data['date']
        dateInd = np.where(dateInd)[0][0] if dateInd.sum()>0 else df.shape[0]
        df.loc[dateInd] = list(data.values())
    else:
        df = pd.DataFrame(data)
        dateInd = 0
    
    allMiceDf = sheets['all mice']
    mouseInd = np.where(allMiceDf['mouse id']==int(obj.subjectName))[0][0]
    regimen = int(allMiceDf.loc[mouseInd,'regimen'])
    hitThresh = 150 if regimen==1 else 100
    dprimeThresh = 1.5
    lowRespThresh = 10
    task = df.loc[dateInd,'task version']
    prevTask = df.loc[dateInd-1,'task version'] if dateInd>0 else ''
    passStage = 0
    if 'stage 0' in task:
        passStage = 1
        nextTask = 'stage 1'
    else:
        if dateInd > 0:
            hits = []
            dprimeSame = []
            dprimeOther = []
            for i in (1,0):
                if isinstance(df.loc[dateInd-i,'hits'],str):
                    hits.append([int(s) for s in re.findall('[0-9]+',df.loc[dateInd-i,'hits'])])
                    dprimeSame.append([float(s) for s in re.findall('[0-9].[0-9]*',df.loc[dateInd-i,'d\' same modality'])])
                    dprimeOther.append([float(s) for s in re.findall('[0-9].[0-9]*',df.loc[dateInd-i,'d\' other modality go stim'])])
                else:
                    hits.append(df.loc[dateInd-i,'hits'])
                    dprimeSame.append(df.loc[dateInd-i,'d\' same modality'])
                    dprimeOther.append(df.loc[dateInd-i,'d\' other modality go stim'])
        if 'stage 1' in task:
            if 'stage 1' in prevTask and all(h[0] > hitThresh for h in hits) and all(d[0] > dprimeThresh for d in dprimeSame):
                passStage = 1
                nextTask = 'stage 2'
            else:
                nextTask = 'stage 1'
        elif 'stage 2' in task:
            if 'stage 2' in prevTask and all(h[0] > hitThresh for h in hits) and all(d[0] > dprimeThresh for d in dprimeSame):
                passStage = 1
                nextTask = 'stage 3 ori'
            else:
                nextTask = 'stage 2'
        elif 'stage 3' in task:
            remedial = any('stage 4' in s for s in df['task version'])
            if ('stage 3' in prevTask
                 and ((regimen==1 and all(all(h > hitThresh for h in hc) for hc in hits) and all(all(d > dprimeThresh for d in dp) for dp in dprimeSame))
                      or (regimen==2 and all(all(d > dprimeThresh for d in dp) for dp in dprimeOther)))):
                passStage = 1
                if regimen==2 and not any('stage 3 tone' in s for s in df['task version']):
                    nextTask = 'stage 3 tone'
                else:
                    nextTask = 'stage 4 tone ori' if remedial and 'tone' in task else 'stage 4 ori tone'
            else:
                nextTask = 'stage 3 tone' if 'ori' in task else 'stage 3 ori'
        elif 'stage 4' in task:
            if 'stage 4' in prevTask:
                lowRespOri = (('stage 4 ori' in prevTask and hits[0][0] < lowRespThresh and hits[1][1] < lowRespThresh)
                              or ('stage 4 tone' in prevTask and hits[0][1] < lowRespThresh and hits[1][0] < lowRespThresh))
                lowRespTone = (('stage 4 tone' in prevTask and hits[0][0] < lowRespThresh and hits[1][1] < lowRespThresh)
                               or ('stage 4 ori' in prevTask and hits[0][1] < lowRespThresh and hits[1][0] < lowRespThresh))
            if 'stage 4' in prevTask and lowRespOri or lowRespTone:
                passStage = -1
                nextTask = 'stage 3 ori' if lowRespOri else 'stage 3 tone'
            elif 'stage 4' in prevTask and all(all(d > dprimeThresh for d in dp) for dp in dprimeOther):
                passStage = 1
                nextTask = 'stage 5 ori tone'
            else:
                nextTask = 'stage 4 tone ori' if 'stage 4 ori' in task else 'stage 4 ori tone'
        elif 'stage 5' in task:
            if 'stage 5' in prevTask and all(all(d > dprimeThresh for d in dp) for dp in dprimeOther):
                passStage = 1
            nextTask = 'stage 5 tone ori' if 'stage 5 ori' in task else 'stage 5 ori tone'
    if 'stage 3' in nextTask and regimen==2:
        nextTask += ' distract'
    if allMiceDf.loc[mouseInd,'timeouts'] and not 'stage 5' in task:
        nextTask += ' timeouts'
    df.loc[dateInd,'pass'] = passStage
    
    writer =  pd.ExcelWriter(excelPath,mode='a',engine='openpyxl',if_sheet_exists='replace',datetime_format='%Y%m%d_%H%M%S')
    
    if df.shape[0] in (1,dateInd+1):
        daysToNext = 1
        if data['date'].day_name() == 'Saturday':
            daysToNext += 2
        elif data['date'].day_name() == 'Sunday':
            daysToNext += 1
        allMiceDf.loc[mouseInd,'next session'] = data['date']+pd.Timedelta(days=daysToNext)
        allMiceDf['next session'] = allMiceDf['next session'].dt.floor('d')
        allMiceDf.loc[mouseInd,'task version'] = nextTask
        allMiceDf.to_excel(writer,sheet_name='all mice',index=False)
        sheet = writer.sheets['all mice']
        for col in ('ABCDEFGHIJ'):
            if col in ('D','I'):
                w = 20
            elif col=='J':
                w = 30
            else:
                w = 12
            sheet.column_dimensions[col].width = w
    
    df.to_excel(writer,sheet_name=obj.subjectName,index=False)
    sheet = writer.sheets[obj.subjectName]
    for col in ('ABCDE'):
        sheet.column_dimensions[col].width = 30
    
    writer.save()
    writer.close()
    


# multimodal stimuli
stimNames = ('vis1','vis2','vis1+sound1','vis1+sound2','autorewarded',
             'sound1','sound2','vis2+sound1','vis2+sound2','catch')
preTime = 4
postTime = 4
respTime = []
for blockInd,goStim in enumerate(obj.blockStimRewarded):
    fig = plt.figure(figsize=(8,8))
    fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
    gs = matplotlib.gridspec.GridSpec(5,2)
    blockTrials = obj.trialBlock == blockInd + 1
    respTime.append([])
    for stimInd,stim in enumerate(stimNames):
        if stim=='autorewarded':
            trials = obj.autoRewarded
        elif stim=='catch':
            trials = obj.catchTrials
        else:
            trials = (obj.trialStim==stim) & (~obj.autoRewarded)
        trials = trials & blockTrials
        respTime[-1].append(obj.responseTimes[trials & obj.trialResponse])
        i,j = (stimInd,0) if stimInd<5 else (stimInd-5,1)
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
        title = stim + ', reponse rate=' + str(round(obj.trialResponse[trials].sum()/trials.sum(),2))
        ax.set_title(title)   
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    
fig = plt.figure(figsize=(8,8))
gs = matplotlib.gridspec.GridSpec(5,2)
for stimInd,stim in enumerate(stimNames):
    i,j = (stimInd,0) if stimInd<5 else (stimInd-5,1)
    ax = fig.add_subplot(gs[i,j])
    for blockInd,(blockRt,ls,goStim) in enumerate(zip(respTime,('-',':'),obj.blockStimRewarded)):
        rt = blockRt[stimInd]
        rtSort = np.sort(rt)
        cumProb = [np.sum(rt<=i)/rt.size for i in rtSort]
        ax.plot(rtSort,cumProb,color='k',ls=ls,label='block '+str(blockInd+1)+', '+goStim+' rewarded')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,obj.responseWindowTime[1]])
    ax.set_ylim([0,1.02])
    ax.set_xlabel('response time (s)')
    ax.set_ylabel('cum. prob.')
    ax.set_title(stim)
    if i==0 and j==1:
        ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()



# learning summary plots
hitRate = []
falseAlarmRate = []
falseAlarmSameModal = []
falseAlarmOtherModalGo = []
falseAlarmOtherModalNogo = []
catchRate = []
blockReward = []
for obj in exps:
    hitRate.append(obj.hitRate)
    falseAlarmRate.append(obj.falseAlarmRate)
    falseAlarmSameModal.append(obj.falseAlarmSameModal)
    falseAlarmOtherModalGo.append(obj.falseAlarmOtherModalGo)
    falseAlarmOtherModalNogo.append(obj.falseAlarmOtherModalNogo)
    catchRate.append(obj.catchResponseRate)
    blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate)
falseAlarmRate = np.array(falseAlarmRate)
falseAlarmSameModal = np.array(falseAlarmSameModal)
falseAlarmOtherModalGo = np.array(falseAlarmOtherModalGo)
falseAlarmOtherModalNogo = np.array(falseAlarmOtherModalNogo)
catchRate = np.array(catchRate)    

fig = plt.figure(figsize=(12,8))
nBlocks = hitRate.shape[1]
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
for ind,(r,lbl) in enumerate(zip((hitRate,falseAlarmSameModal,falseAlarmOtherModalGo,falseAlarmOtherModalNogo,catchRate),
                               ('hit rate','false alarm Same','false alarm diff go','false alarm diff nogo','catch rate'))):  
    ax = fig.add_subplot(1,5,ind+1)
    im = ax.imshow(r,cmap='magma',clim=(0,1))
    for i in range(nExps):
        for j in range(nBlocks):
            ax.text(j,i,str(round(r[i,j],2)),ha='center',va='center',fontsize=6)
    ax.set_xticks(np.arange(nBlocks))
    ax.set_xticklabels(np.arange(nBlocks)+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks+1)
    ax.set_ylim([nExps-0.5,-0.5])
    ax.set_xlabel('block')
    if ind==0:
        ax.set_ylabel('session')
        cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
        cb.set_ticks([0,0.5,1])
    if ind==4:
        for y,rew in enumerate(blockReward):
            ax.text(nBlocks,y,list(rew)[:2],ha='left',va='center',fontsize=8)
    ax.set_title(lbl)
plt.tight_layout()
    


# transition analysis
blockData = []
for obj in exps:
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        d = {'mouseID':obj.subjectName,
             'sessionStartTime': obj.startTime,
             'blockNum':blockInd+1,
             'goStim':goStim,
             'numAutoRewards':obj.autoRewarded[:10].sum()}
        blockTrials = obj.trialBlock == blockInd + 1
        for trials,lbl in zip((obj.goTrials,obj.otherModalGoTrials),('goTrials','nogoTrials')):
            trials = trials & blockTrials
            d[lbl] = {'startTimes':obj.stimStartTimes[trials]-obj.blockFirstStimTimes[blockInd],
                      'response':obj.trialResponse[trials],
                      'responseTime':obj.responseTimes[trials]}
        blockData.append(d)
        
[(d['mouseID'],d['sessionStartTime'],d['numAutoRewards']) for d in blockData]

for blockType,hitColor,faColor in zip(('vis','sound'),'gm','mg'):
    goLabel = 'vis' if blockType=='vis' else 'aud'
    nogoLabel = 'aud' if goLabel=='vis' else 'vis'
    blocks = [d for d in blockData if blockType in d['goStim']]
    nBlocks = len(blocks)
    nMice = len(set(d['mouseID'] for d in blockData))
    nSessions = len(set(d['sessionStartTime'] for d in blockData))
    nTrials = [len(d['goTrials']['response']) for d in blocks] + [len(d['nogoTrials']['response']) for d in blocks]
    print('n trials: '+str(min(nTrials))+', '+str(max(nTrials))+', '+str(np.median(nTrials)))
    
    title = goLabel+' rewarded (' + str(nBlocks) +' blocks, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)'
    
    blockDur = 700
    binSize = 30
    bins = np.arange(0,blockDur+binSize/2,binSize)
    hitRateTime = np.zeros((nBlocks,bins.size))
    falseAlarmRateTime = hitRateTime.copy()  
    hitLatencyTime = hitRateTime.copy()
    falseAlarmLatencyTime = hitRateTime.copy()
    
    hitTrials = np.zeros(nBlocks,dtype=int)
    falseAlarmTrials = hitTrials.copy()
    maxTrials = 100
    hitRateTrials = np.full((nBlocks,maxTrials),np.nan)
    falseAlarmRateTrials = hitRateTrials.copy()  
    hitLatencyTrials = hitRateTrials.copy()
    falseAlarmLatencyTrials = hitRateTrials.copy()
    
    for i,d in enumerate(blocks):
        for trials,r,lat in zip(('goTrials','nogoTrials'),(hitRateTime,falseAlarmRateTime),(hitLatencyTime,falseAlarmLatencyTime)):
            c = np.zeros(bins.size)
            for trialInd,binInd in enumerate(np.digitize(d[trials]['startTimes'],bins)):
                r[i][binInd] += d[trials]['response'][trialInd]
                lat[i][binInd] += d[trials]['responseTime'][trialInd]
                c[binInd] += 1
            r[i] /= c
            lat[i] /= c
        for trials,n,r,lat in zip(('goTrials','nogoTrials'),(hitTrials,falseAlarmTrials),(hitRateTrials,falseAlarmRateTrials),(hitLatencyTrials,falseAlarmLatencyTrials)):
            n[i] = d[trials]['response'].size
            r[i,:n[i]] = d[trials]['response']
            lat[i,:n[i]] = d[trials]['responseTime']
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    binTimes = bins+binSize/2
    for d,clr,lbl in zip((hitRateTime,falseAlarmRateTime),(hitColor,faColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(binTimes,m,clr,label=lbl+' go')
        ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,700,100))
    ax.set_xlim([0,615])
    ax.set_ylim([0,1])
    ax.set_xlabel('Time (s); auto-rewards excluded')
    ax.set_ylabel('Response Rate')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)  
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip((hitLatencyTime,falseAlarmLatencyTime),(hitColor,faColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(binTimes,m,clr,label=lbl+' go')
        ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,700,100))
    ax.set_xlim([0,615])
    ax.set_yticks(np.arange(0.2,0.7,0.1))
    ax.set_ylim([0.25,0.65])
    ax.set_xlabel('Time (s); auto-rewards excluded')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    trialNum = np.arange(maxTrials)+1
    for d,clr,lbl in zip((hitRateTrials,falseAlarmRateTrials),(hitColor,faColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(trialNum,m,clr,label=lbl+' go')
        ax.fill_between(trialNum,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,25,5))
    ax.set_xlim([0,20])
    ax.set_ylim([0,1])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Rate')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip((hitLatencyTrials,falseAlarmLatencyTrials),(hitColor,faColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(trialNum,m,clr,label=lbl+' go')
        ax.fill_between(trialNum,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,25,5))
    ax.set_xlim([0,20])
    ax.set_yticks(np.arange(0.2,0.7,0.1))
    ax.set_ylim([0.25,0.65])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()


for blockType in ('visual','auditory'):
    goStim = 'vis' if blockType=='visual' else 'sound'
    nTransitions = 0    
    goProb = []
    goProbPrev = []
    nogoProb = []
    nogoProbPrev = []
    goLat = []
    goLatPrev = []
    nogoLat = []
    nogoLatPrev = [] 
    for block in blockData:
        if goStim in block['goStim'] and block['blockNum'] > 1:
            nTransitions += 1
            for d in blockData:
                if d['mouseID']==block['mouseID'] and d['blockNum']==block['blockNum']-1:
                    prevBlock = d
                    break
            goProb.append(block['goTrials']['response'])
            goProbPrev.append(prevBlock['nogoTrials']['response'])
            nogoProb.append(block['nogoTrials']['response'])
            nogoProbPrev.append(prevBlock['goTrials']['response'])
            goLat.append(block['goTrials']['responseTime'])
            goLatPrev.append(prevBlock['nogoTrials']['responseTime'])
            nogoLat.append(block['nogoTrials']['responseTime'])
            nogoLatPrev.append(prevBlock['goTrials']['responseTime'])
    
    title = (blockType+' rewarded blocks\n'
             'mean and 95% ci across transitions\n('+
             str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(21)
    for prev,current,clr,lbl in zip((goProbPrev,nogoProbPrev),(goProb,nogoProb),colors,labels):
        d = np.full((nTransitions,21),np.nan)
        d[:,0] = [r[-1] for r in prev]
        for i,r in enumerate(current):
            j = len(r) if len(r)<20 else 20
            d[i,1:j+1] = r[:j] 
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(-1,m[0],'o',color=clr)
        ax.plot([-1,-1],[m[0]-s[0],m[0]+s[0]],clr)
        ax.plot(x[1:],m[1:],clr,label=lbl+' go stimulus')
        ax.fill_between(x[1:],(m+s)[1:],(m-s)[1:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([-1,1,5,10,15,20])
    ax.set_xlim([-2,15])
    ax.set_ylim([0,1])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(loc='lower right')
    ax.set_title(blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for prev,first,last,clr,lbl in zip((goProbPrev,nogoProbPrev),(goProb,nogoProb),(goProb,nogoProb),colors,labels):
        prev,first,last = [[r[i] for r in d] for d,i in zip((prev,first,last),(-1,0,-1))]
        m = [np.nanmean(d) for d in (prev,first,last)]
        ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in (prev,first,last)]
        ax.plot([0,1,2],m,'o-',color=clr,label=lbl+' go stimulus')
        for i,c in enumerate(ci):
            ax.plot([i,i],c,clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(('last trial of\nprevious block',
                        'first trial\nof block\nafter auto-rewards',
                        'last trial\nof block'))
    ax.set_xlim([-0.2,2.2])
    ax.set_ylim([0,1])
    ax.set_ylabel('Response Probability')
    ax.legend(loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for prev,first,last,clr,lbl in zip((goLatPrev,nogoLatPrev),(goLat,nogoLat),(goLat,nogoLat),colors,labels):
        prev,first,last = [[r[i] for r in d] for d,i in zip((prev,first,last),(-1,0,-1))]
        m = [np.nanmean(d) for d in (prev,first,last)]
        ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in (prev,first,last)]
        ax.plot([0,1,2],m,'o-',color=clr,label=lbl+' stimulus (current block)')
        for i,c in enumerate(ci):
            ax.plot([i,i],c,clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(('last trial of\nprevious block',
                        'first trial\nof block\nafter auto-rewards',
                        'last trial\nof block'))
    ax.set_xlim([-0.2,2.2])
    ax.set_ylim([0.3,0.55])
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    

 
# learning summary plots (old)
hitRate = []
falseAlarmRate = []
falseAlarmSameModal = []
falseAlarmOtherModalGo = []
falseAlarmOtherModalNogo = []
catchRate = []
blockReward = []
for obj in exps:
    if ((obj.taskVersion in ('vis sound vis detect','sound vis sound detect','vis sound detect','sound vis detect')
        and len(obj.blockStimRewarded)>=3) or
        ('vis sound discrim' in obj.taskVersion or 'sound vis discrim' in obj.taskVersion) or
        ('ori tone discrim' in obj.taskVersion or 'tone ori discrim' in obj.taskVersion) or
        ('ori sweep discrim' in obj.taskVersion or 'sweep ori discrim' in obj.taskVersion)):
        hitRate.append(obj.hitRate)
        falseAlarmRate.append(obj.falseAlarmRate)
        falseAlarmSameModal.append(obj.falseAlarmSameModal)
        falseAlarmOtherModalGo.append(obj.falseAlarmOtherModalGo)
        falseAlarmOtherModalNogo.append(obj.falseAlarmOtherModalNogo)
        catchRate.append(obj.catchResponseRate)
        blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate)
falseAlarmRate = np.array(falseAlarmRate)
falseAlarmSameModal = np.array(falseAlarmSameModal)
falseAlarmOtherModalGo = np.array(falseAlarmOtherModalGo)
falseAlarmOtherModalNogo = np.array(falseAlarmOtherModalNogo)
catchRate = np.array(catchRate)    

fig = plt.figure(figsize=(10,5))
nBlocks = hitRate.shape[1]
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
for i,(r,lbl) in enumerate(zip((hitRate,falseAlarmSameModal,falseAlarmOtherModalGo,falseAlarmOtherModalNogo,catchRate),
                               ('hit rate','false alarm Same','false alarm diff go','false alarm diff nogo','catch rate'))):  
    ax = fig.add_subplot(1,5,i+1)
    im = ax.imshow(r,cmap='magma',clim=(0,1))
    ax.set_xticks(np.arange(nBlocks))
    ax.set_xticklabels(np.arange(nBlocks)+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks+1)
    ax.set_ylim([nExps-0.5,-0.5])
    ax.set_xlabel('block')
    if i==0:
        ax.set_ylabel('session')
        cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
        cb.set_ticks([0,0.5,1])
    if i==4:
        for y,rew in enumerate(blockReward):
            ax.text(nBlocks,y,list(rew)[:2],ha='left',va='center',fontsize=8)
    ax.set_title(lbl)
plt.tight_layout()


hitRate = []
falseAlarmRate = []
catchRate = []
blockReward = []
for obj in exps:
    if 'ori discrim' in obj.taskVersion or 'sound discrim' in obj.taskVersion or 'tone discrim' in obj.taskVersion or 'sweep discrim' in obj.taskVersion:
        hitRate.append(obj.hitRate)
        falseAlarmRate.append(obj.falseAlarmRate)
        catchRate.append(obj.catchResponseRate)
        blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate).squeeze()
falseAlarmRate = np.array(falseAlarmRate).squeeze()
catchRate = np.array(catchRate).squeeze()

fig = plt.figure(figsize=(6,9))
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(np.stack((hitRate,falseAlarmRate,catchRate),axis=1),cmap='magma',clim=(0,1))
ax.set_xticks([0,1,2])
ax.set_xticklabels(('hit','false alarm','catch'),rotation=90)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks+1)
ax.set_ylim([nExps-0.5,-0.5])
ax.set_ylabel('session')
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
cb.set_ticks([0,0.5,1])
for y,rew in enumerate(blockReward):
    ax.text(3,y,list(rew),ha='left',va='center',fontsize=8)
plt.tight_layout()


# for shawn
fig = plt.figure(figsize=(6,9))
ax = fig.add_subplot(1,1,1)
im = ax.imshow(np.stack((hitRate,falseAlarmRate,catchRate),axis=1)[:59],cmap='magma',clim=(0,1))
ax.set_xticks([0,1,2])
ax.set_xticklabels(('hit','false alarm','catch'),rotation=90)
ax.set_ylabel('session')
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
cb.set_ticks([0,0.5,1])
cb.set_label('response rate')
rprev = ''
for y,rew in enumerate(blockReward[:59]):
    r = list(rew[0])
    if rprev != '' and r != rprev:
        ax.text(3,y,'switch',ha='left',va='center',fontsize=8)
    rprev = r
plt.tight_layout()

fig = plt.figure(figsize=(11,3.5))
ax = fig.add_subplot(1,1,1)
x = np.arange(59)+1
xticks = np.arange(0,60,10)
rprev = ''
for i,rew in enumerate(blockReward[:59]):
    r = rew[0]
    if rprev != '' and r != rprev:
        ax.plot([x[i]]*2,[0,1],'k--')
        ax.text(x[i],1.025,'switch',ha='center',va='baseline')
    rprev = r
for y,clr,lbl in zip((catchRate,falseAlarmRate,hitRate),('0.8','m','g'),('catch','nogo','go')):
    ax.plot(x,y[:59],'o-',ms=4,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xlim([0,60])
ax.set_ylim([0,1.01])
ax.set_xlabel('Session')
ax.set_ylabel('Response Rate')
ax.legend(loc='upper left')
plt.tight_layout()



# ori
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rr = []
for obj,clr in zip(exps,plt.cm.tab20(np.linspace(0,1,len(exps)))):
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = (obj.trialBlock == blockInd + 1) & ~obj.autoRewarded & ~obj.catchTrials
        oris = np.unique(obj.trialGratingOri)
        r = []
        for ori in oris:
            trials = blockTrials & (obj.trialGratingOri == ori)
            r.append(obj.trialResponse[trials].sum() / trials.sum())
        ax.plot(oris,r,'o-',color=clr,alpha=0.5)
        rr.append(r)
mean = np.mean(rr,axis=0)
sem = np.std(rr,axis=0)/(len(exps)**0.5)
ax.plot(oris,mean,'ko-',ms=8,lw=2,label=lbl)
for x,m,s in zip(oris,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.02])
ax.set_xlabel('ori (0=go, >0=nogo)')
ax.set_ylabel('response rate')
plt.tight_layout()
                
                
# contrast
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
hr = []
far = []
for obj,clr in zip(exps,plt.cm.tab20(np.linspace(0,1,len(exps)))):
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock == blockInd + 1
        for trials,lbl in zip((obj.goTrials,obj.nogoTrials),('go','nogo')):
            r = []
            for c in obj.visContrast:
                tr = trials & blockTrials & (obj.trialVisContrast == c)
                r.append(obj.trialResponse[tr].sum() / tr.sum())
            ls,mfc = ('-',clr) if lbl=='go' else ('--','none')
            ax.plot(obj.visContrast,r,'o',color=clr,ls=ls,mec=clr,mfc=mfc,alpha=0.5)
            if lbl=='go':
                hr.append(r)
            else:
                far.append(r)
for r,lbl in zip((hr,far),('go','nogo')):
    mean = np.mean(r,axis=0)
    sem = np.std(r,axis=0)/(len(exps)**0.5)
    ls,mfc = ('-','k') if lbl=='go' else ('--','none')
    ax.plot(obj.visContrast,mean,'ko-',mfc=mfc,ls=ls,ms=8,lw=2,label=lbl)
    for x,m,s in zip(obj.visContrast,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xscale('log')
ax.set_ylim([0,1.02])
ax.set_xlabel('contrast')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()
    
    

# sound latency test
filePath = fileIO.getFile(rootDir=baseDir,fileType='*.hdf5')

d = h5py.File(filePath,'r')
    
frameRate = 60
frameIntervals = d['frameIntervals'][:]
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
stimStartTimes = frameTimes[stimStartFrame]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mic = d['microphoneData'][:]
frame = np.arange(-30,45)
for sf in stimStartFrame:
    ax.plot(frame,mic[sf-30:sf+45],'k')
    
d.close()


