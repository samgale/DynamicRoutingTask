# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import os
import time
import h5py
import numpy as np
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
        
        self.subjectName = d['subjectName'][()]
        self.rigName = d['rigName'][()]
        self.taskVersion = d['taskVersion'][()] if 'taskVersion' in d.keys() else ''
        self.soundType = d['soundType'][()]
        self.startTime = d['startTime'][()]
            
        self.frameIntervals = d['frameIntervals'][:]
        self.frameTimes = np.concatenate(([0],np.cumsum(self.frameIntervals)))
        
        self.lickFrames = d['lickFrames'][:]
        lickTimesDetected = self.frameTimes[self.lickFrames]
        self.minLickInterval = 0.05
        isLick = np.concatenate(([True], np.diff(lickTimesDetected) > self.minLickInterval))
        self.lickTimes = lickTimesDetected[isLick]
        
        self.trialEndFrame = d['trialEndFrame'][:]
        self.trialEndTimes = self.frameTimes[self.trialEndFrame]
        self.nTrials = self.trialEndFrame.size
        self.trialStartFrame = d['trialStartFrame'][:self.nTrials]
        self.trialStartTimes = self.frameTimes[self.trialStartFrame]
        self.stimStartFrame = d['trialStimStartFrame'][:self.nTrials]
        self.stimStartTimes = self.frameTimes[self.stimStartFrame]
        
        self.quiescentFrames = d['quiescentFrames'][()]
        self.quiescentViolationFrames = d['quiescentViolationFrames'][:] if 'quiescentViolationFrames' in d.keys() else d['quiescentMoveFrames'][:]    
        
        self.responseWindow = d['responseWindow'][:]
        self.responseWindowTime = np.array(self.responseWindow)/self.frameRate
        
        self.trialStim = d['trialStim'][:self.nTrials]
        self.trialBlock = d['trialBlock'][:self.nTrials]
        self.blockStimRewarded = d['blockStimRewarded'][:]
        self.rewardedStim = self.blockStimRewarded[self.trialBlock-1]
        
        self.trialResponse = d['trialResponse'][:self.nTrials]
        self.trialResponseFrame = d['trialResponseFrame'][:self.nTrials]
        self.trialRewarded = d['trialRewarded'][:self.nTrials]
        self.autoRewarded = d['trialAutoRewarded'][:self.nTrials]
        self.rewardEarned = self.trialRewarded & (~self.autoRewarded)
        self.rewardFrames = d['rewardFrames']
        self.rewardTimes = self.frameTimes[self.rewardFrames]
        
        if 'rotaryEncoder' in d and d['rotaryEncoder'][()] == 'digital':
            self.runningSpeed = np.concatenate(([np.nan],np.diff(d['rotaryEncoderCount'][:]) / d['rotaryEncoderCountsPerRev'][()] * 2 * np.pi * d['wheelRadius'][()] * self.frameRate))
        else:
            self.runningSpeed = None
        
        self.catchTrials = self.trialStim == 'catch'
        self.goTrials = (self.trialStim == self.rewardedStim) & (~self.autoRewarded)
        self.nogoTrials = (self.trialStim != self.rewardedStim) & (~self.catchTrials)
        
        assert(self.nTrials == self.goTrials.sum() + self.nogoTrials.sum() + self.autoRewarded.sum() + self.catchTrials.sum())
        
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
        
        self.hitRate = []
        self.falseAlarmRate = []
        self.falseAlarmSameModal = []
        self.falseAlarmDiffModalGo = []
        self.falseAlarmDiffModalNogo = []
        self.catchResponseRate = []
        for blockInd,rew in enumerate(self.blockStimRewarded):
            blockTrials = (self.trialBlock == blockInd + 1) & self.engagedTrials           
            self.hitRate.append(self.hitTrials[blockTrials].sum() / self.goTrials[blockTrials].sum())
            self.falseAlarmRate.append(self.falseAlarmTrials[blockTrials].sum() / self.nogoTrials[blockTrials].sum())
            sameModal = blockTrials & self.nogoTrials & np.array([rew[:-1] in stim for stim in self.trialStim])
            diffModalGo = blockTrials & (self.trialStim==np.setdiff1d(self.blockStimRewarded,rew))
            diffModalNogo = blockTrials & self.nogoTrials & ~sameModal & ~diffModalGo
            self.falseAlarmSameModal.append(self.falseAlarmTrials[sameModal].sum() / sameModal.sum())
            self.falseAlarmDiffModalGo.append(self.falseAlarmTrials[diffModalGo].sum() / diffModalGo.sum())
            self.falseAlarmDiffModalNogo.append(self.falseAlarmTrials[diffModalNogo].sum() / diffModalNogo.sum())
            self.catchResponseRate.append(self.catchResponseTrials[blockTrials].sum() / self.catchTrials[blockTrials].sum())
            
        d.close()
    
    
    def makeSummaryPdf(self):
        saveDir = os.path.join(os.path.dirname(self.behavDataPath),'summary')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        pdf = PdfPages(os.path.join(saveDir,os.path.splitext(os.path.basename(self.behavDataPath))[0]+'_summary.pdf'))
        
        
        # plot lick raster (all trials)
        preTime = 4
        postTime = 4
        lickRaster = []
        fig = plt.figure(figsize=(8,8))
        gs = matplotlib.gridspec.GridSpec(4,1)
        ax = fig.add_subplot(gs[:3,0])
        ax.add_patch(matplotlib.patches.Rectangle([-self.quiescentFrames/self.frameRate,0],width=self.quiescentFrames/self.frameRate,height=self.nTrials+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
        ax.add_patch(matplotlib.patches.Rectangle([self.responseWindowTime[0],0],width=np.diff(self.responseWindowTime),height=self.nTrials+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
        for i,st in enumerate(self.stimStartTimes):
            if not self.engagedTrials[i]:
                ax.add_patch(matplotlib.patches.Rectangle([-preTime,i+0.5],width=preTime+postTime,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            lt = self.lickTimes - st
            trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
            lickRaster.append(trialLickTimes)
            ax.vlines(trialLickTimes,i+0.5,i+1.5,colors='k')
            if self.trialRewarded[i]:
                rt = self.rewardTimes - st
                trialRewardTime = rt[(rt > 0) & (rt <= postTime)]
                mfc = 'b' if self.autoRewarded[i] else 'none'
                ax.plot(trialRewardTime,i+1,'o',mec='b',mfc=mfc,ms=4)        
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0.5,self.nTrials+0.5])
        ax.set_yticks([1,self.nTrials])
        ax.set_ylabel('trial')
        title = (self.subjectName + ', ' + self.rigName + ', ' + self.taskVersion + 
                 '\n' + 'all trials (n=' + str(self.nTrials) + '), engaged (n=' + str(self.engagedTrials.sum()) + ', not gray)' +
                 '\n' + 'filled blue circles: auto-reward, open circles: earned reward')
        ax.set_title(title)
            
        binSize = self.minLickInterval
        bins = np.arange(-preTime,postTime+binSize/2,binSize)
        lickPsth = np.zeros((self.nTrials,bins.size-1))    
        for i,st in enumerate(self.stimStartTimes):
            lickPsth[i] = np.histogram(self.lickTimes[(self.lickTimes >= st-preTime) & (self.lickTimes <= st+postTime)]-st,bins)[0]
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
        ax.add_patch(matplotlib.patches.Rectangle([-self.quiescentFrames/self.frameRate,0],width=self.quiescentFrames/self.frameRate,height=self.trialEndTimes[-1]+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
        ax.add_patch(matplotlib.patches.Rectangle([self.responseWindowTime[0],0],width=np.diff(self.responseWindowTime),height=self.trialEndTimes[-1]+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
        for i,st in enumerate(self.stimStartTimes):
            if not self.engagedTrials[i]:
                ax.add_patch(matplotlib.patches.Rectangle([-preTime,self.trialStartTimes[i]],width=preTime+postTime,height=self.trialEndTimes[i]-self.trialStartTimes[i],facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            lt = self.lickTimes - st
            trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
            ax.vlines(trialLickTimes,st-preTime,st+postTime,colors='k')
            if self.trialRewarded[i]:
                rt = self.rewardTimes - st
                trialRewardTime = rt[(rt > 0) & (rt <= postTime)]
                mfc = 'b' if self.autoRewarded[i] else 'none'
                ax.plot(trialRewardTime,st,'o',mec='b',mfc=mfc,ms=4)        
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,self.trialEndTimes[-1]+1])
        ax.set_ylabel('session time (s)')
        title = ('all trials (n=' + str(self.nTrials) + '), engaged (n=' + str(self.engagedTrials.sum()) + ', not gray)' +
                 '\n' + 'filled blue circles: auto-reward, open circles: earned reward')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        
        
        # plot lick raster for each block of trials
        for blockInd,goStim in enumerate(self.blockStimRewarded):
            blockTrials = self.trialBlock == blockInd + 1
            nogoStim = np.unique(self.trialStim[blockTrials & self.nogoTrials])
            fig = plt.figure(figsize=(8,8))
            fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
            gs = matplotlib.gridspec.GridSpec(2,2)
            for trials,trialType in zip((self.goTrials,self.nogoTrials,self.autoRewarded,self.catchTrials),
                                        ('go','no-go','auto reward','catch')):
                trials = trials & blockTrials
                i = 0 if trialType in ('go','no-go') else 1
                j = 0 if trialType in ('go','auto reward') else 1
                ax = fig.add_subplot(gs[i,j])
                ax.add_patch(matplotlib.patches.Rectangle([-self.quiescentFrames/self.frameRate,0],width=self.quiescentFrames/self.frameRate,height=trials.sum()+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
                ax.add_patch(matplotlib.patches.Rectangle([self.responseWindowTime[0],0],width=np.diff(self.responseWindowTime),height=trials.sum()+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
                for i,st in enumerate(self.stimStartTimes[trials]):
                    lt = self.lickTimes - st
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
                title = trialType + ' trials (n=' + str(trials.sum()) + '), engaged (n=' + str(self.engagedTrials[trials].sum()) + ')'
                if trialType == 'go':
                    title += '\n' + 'hit rate ' + str(round(self.hitRate[blockInd],2))
                elif trialType == 'no-go':
                    title += ('\n' + 'same ' + str(round(self.falseAlarmSameModal[blockInd],2)) + 
                              ', diff go ' + str(round(self.falseAlarmDiffModalGo[blockInd],2)) +
                              ', diff nogo ' + str(round(self.falseAlarmDiffModalNogo[blockInd],2)))
                elif trialType == 'catch':
                    title += '\n' + 'catch rate ' + str(round(self.catchResponseRate[blockInd],2))
                ax.set_title(title)   
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(pdf,format='pdf')
        
        for blockInd,goStim in enumerate(self.blockStimRewarded):
            blockTrials = self.trialBlock == blockInd + 1
            nogoStim = np.unique(self.trialStim[blockTrials & self.nogoTrials])
            fig = plt.figure(figsize=(8,8))
            fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
            gs = matplotlib.gridspec.GridSpec(2,2)
            for trials,trialType in zip((self.goTrials,self.nogoTrials,self.autoRewarded,self.catchTrials),
                                        ('go','no-go','auto reward','catch')):
                trials = trials & blockTrials
                i = 0 if trialType in ('go','no-go') else 1
                j = 0 if trialType in ('go','auto reward') else 1
                ax = fig.add_subplot(gs[i,j])
                ax.add_patch(matplotlib.patches.Rectangle([-self.quiescentFrames/self.frameRate,0],width=self.quiescentFrames/self.frameRate,height=self.trialEndTimes[-1]+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
                ax.add_patch(matplotlib.patches.Rectangle([self.responseWindowTime[0],0],width=np.diff(self.responseWindowTime),height=self.trialEndTimes[-1]+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
                for i,st in enumerate(self.stimStartTimes[trials]):
                    lt = self.lickTimes - st
                    trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
                    ax.vlines(trialLickTimes,st-preTime,st+postTime,colors='k')       
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xlim([-preTime,postTime])
                ax.set_ylim([0,self.trialEndTimes[-1]+1])
                ax.set_xlabel('time from stimulus onset (s)')
                ax.set_ylabel('session time (s)')
                title = trialType + ' trials (n=' + str(trials.sum()) + '), engaged (n=' + str(self.engagedTrials[trials].sum()) + ')'
                if trialType == 'go':
                    title += '\n' + 'hit rate ' + str(round(self.hitRate[blockInd],2))
                elif trialType == 'no-go':
                    title += ('\n' + 'same ' + str(round(self.falseAlarmSameModal[blockInd],2)) + 
                              ', diff go ' + str(round(self.falseAlarmDiffModalGo[blockInd],2)) +
                              ', diff nogo ' + str(round(self.falseAlarmDiffModalNogo[blockInd],2)))
                elif trialType == 'catch':
                    title += '\n' + 'catch rate ' + str(round(self.catchResponseRate[blockInd],2))
                ax.set_title(title)   
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(pdf,format='pdf')
                
        
        # plot lick latency
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        stimLabels = np.unique(self.trialStim)
        notCatch = stimLabels != 'catch'
        clrs = np.zeros((len(stimLabels),3)) + 0.5
        clrs[notCatch] = plt.cm.plasma(np.linspace(0,0.85,notCatch.sum()))[:,:3]
        for stim,clr in zip(stimLabels,clrs):
            trials = (self.trialStim==stim) & self.trialResponse
            rt = self.frameTimes[self.trialResponseFrame[trials].astype(int)] - self.stimStartTimes[trials]
            rtSort = np.sort(rt)
            cumProb = [np.sum(rt<=i)/rt.size for i in rtSort]
            ax.plot(rtSort,cumProb,color=clr,label=stim)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,self.responseWindowTime[1]+0.1])
        ax.set_ylim([0,1.02])
        ax.set_xlabel('response time (s)')
        ax.set_ylabel('cumulative probability')
        ax.legend()
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        
        
        # plot mean running speed for each block of trials
        if self.runningSpeed is not None:
            preFrames,postFrames = [int(t * self.frameRate) for t in (preTime,postTime)]
            for blockInd,goStim in enumerate(self.blockStimRewarded):
                blockTrials = self.trialBlock == blockInd + 1
                nogoStim = np.unique(self.trialStim[blockTrials & self.nogoTrials])
                fig = plt.figure(figsize=(8,8))
                fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim + ', nogo=' + str(nogoStim))
                gs = matplotlib.gridspec.GridSpec(2,2)
                axs = []
                ymax = 1
                for trials,trialType in zip((self.goTrials,self.nogoTrials,self.autoRewarded,self.catchTrials),
                                            ('go','no-go','auto reward','catch')):
                    trials = trials & blockTrials
                    i = 0 if trialType in ('go','no-go') else 1
                    j = 0 if trialType in ('go','auto reward') else 1
                    ax = fig.add_subplot(gs[i,j])
                    ax.add_patch(matplotlib.patches.Rectangle([-self.quiescentFrames/self.frameRate,0],width=self.quiescentFrames/self.frameRate,height=100,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
                    ax.add_patch(matplotlib.patches.Rectangle([self.responseWindowTime[0],0],width=np.diff(self.responseWindowTime),height=100,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
                    speed = np.full((trials.sum(),preFrames+postFrames),np.nan)
                    for i,sf in enumerate(self.stimStartFrame[trials]):
                        if sf >= preFrames and sf+postFrames < self.frameTimes.size:
                            speed[i] = self.runningSpeed[sf-preFrames:sf+postFrames]
                    meanSpeed = np.nanmean(speed,axis=0)
                    ymax = max(ymax,meanSpeed.max())
                    ax.plot(np.arange(-preTime,postTime,1/self.frameRate),meanSpeed)
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False)
                    ax.set_xlim([-preTime,postTime])
                    ax.set_xlabel('time from stimulus onset (s)')
                    ax.set_ylabel('mean running speed (cm/s)')
                    ax.set_title(trialType + ' trials (n=' + str(trials.sum()) + '), engaged (n=' + str(self.engagedTrials[trials].sum()) + ')')
                    axs.append(ax)
                for ax in axs:
                    ax.set_ylim([0,1.05*ymax])
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(pdf,format='pdf')
        
        
        # plot frame intervals
        longFrames = self.frameIntervals > 1.5/self.frameRate
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        bins = np.arange(-0.5/self.frameRate,self.frameIntervals.max()+1/self.frameRate,1/self.frameRate)
        ax.hist(self.frameIntervals,bins=bins,color='k')
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
        for sf,ef in zip(self.trialStartFrame,self.trialEndFrame):
            trialQuiescentViolations.append(np.sum((self.quiescentViolationFrames > sf) & (self.quiescentViolationFrames < ef)))
        
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(2,1,1)
        if self.quiescentViolationFrames.size > 0:
            ax.plot(self.frameTimes[self.quiescentViolationFrames],np.arange(self.quiescentViolationFrames.size)+1,'k')
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
        interTrialIntervals = np.diff(self.frameTimes[self.stimStartFrame])
        
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
        if self.runningSpeed is not None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(self.frameTimes,self.runningSpeed[:self.frameTimes.size],'k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([0,self.frameTimes[-1]])
            ax.set_xlabel('time (s)')
            ax.set_ylabel('running speed (cm/s)')
            plt.tight_layout()
            fig.savefig(pdf,format='pdf')
            
            
        pdf.close()
        plt.close('all')
    # end makeSummaryPdf
# end DynRoutData
    

def sortExps(exps):
    startTimes = [time.strptime(obj.startTime,'%Y%m%d_%H%M%S') for obj in exps]
    return [z[0] for z in sorted(zip(exps,startTimes),key=lambda i: i[1])]
    
    
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
    obj.makeSummaryPdf()
    
   
#
hitRate = []
falseAlarmRate = []
falseAlarmSameModal = []
falseAlarmDiffModalGo = []
falseAlarmDiffModalNogo = []
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
        falseAlarmDiffModalGo.append(obj.falseAlarmDiffModalGo)
        falseAlarmDiffModalNogo.append(obj.falseAlarmDiffModalNogo)
        catchRate.append(obj.catchResponseRate)
        blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate)
falseAlarmRate = np.array(falseAlarmRate)
falseAlarmSameModal = np.array(falseAlarmSameModal)
falseAlarmDiffModalGo = np.array(falseAlarmDiffModalGo)
falseAlarmDiffModalNogo = np.array(falseAlarmDiffModalNogo)
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
for i,(r,lbl) in enumerate(zip((hitRate,falseAlarmSameModal,falseAlarmDiffModalGo,falseAlarmDiffModalNogo,catchRate),
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


