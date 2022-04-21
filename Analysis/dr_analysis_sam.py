# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import os
import time
import h5py
import numpy as np
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
        
        self.subjectName = d['subjectName'][()]
        self.rigName = d['rigName'][()]
        self.taskVersion = d['taskVersion'][()] if 'taskVersion' in d.keys() else None
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
        
        assert(self.nTrials == self.goTrials.sum() + self.nogoTrials.sum() + self.autoRewarded.sum() + self.catchTrials.sum())
        
        self.sameModalNogoTrials = self.nogoTrials & np.array([stim[:-1]==rew[:-1] for stim,rew in zip(self.trialStim,self.rewardedStim)])
        self.diffModalGoTrials = self.nogoTrials & np.in1d(self.trialStim,self.blockStimRewarded)
        self.diffModalNogoTrials = self.nogoTrials & ~self.sameModalNogoTrials & ~self.diffModalGoTrials
        
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
        self.falseAlarmDiffModalGo = []
        self.falseAlarmDiffModalNogo = []
        self.dprimeSameModal = []
        self.dprimeDiffModalGo = []
        for blockInd,rew in enumerate(self.blockStimRewarded):
            blockTrials = (self.trialBlock == blockInd + 1) #& self.engagedTrials 
            self.catchResponseRate.append(self.catchResponseTrials[blockTrials].sum() / self.catchTrials[blockTrials].sum())
            self.hitRate.append(self.hitTrials[blockTrials].sum() / self.goTrials[blockTrials].sum())
            self.hitCount.append(self.hitTrials[blockTrials].sum())
            self.falseAlarmRate.append(self.falseAlarmTrials[blockTrials].sum() / self.nogoTrials[blockTrials].sum())
            sameModal = blockTrials & self.sameModalNogoTrials
            diffModalGo = blockTrials & self.diffModalGoTrials
            diffModalNogo = blockTrials & self.diffModalNogoTrials
            self.falseAlarmSameModal.append(self.falseAlarmTrials[sameModal].sum() / sameModal.sum())
            self.falseAlarmDiffModalGo.append(self.falseAlarmTrials[diffModalGo].sum() / diffModalGo.sum())
            self.falseAlarmDiffModalNogo.append(self.falseAlarmTrials[diffModalNogo].sum() / diffModalNogo.sum())
            self.dprimeSameModal.append(self.calcDprime(self.hitRate[-1],self.falseAlarmSameModal[-1],self.goTrials[blockTrials].sum(),sameModal.sum()))
            self.dprimeDiffModalGo.append(self.calcDprime(self.hitRate[-1],self.falseAlarmDiffModalGo[-1],self.goTrials[blockTrials].sum(),diffModalGo.sum()))
        
    
    def calcDprime(self,hitRate,falseAlarmRate,goTrials,nogoTrials):
        hr = self.adjustResponseRate(hitRate,goTrials)
        far = self.adjustResponseRate(falseAlarmRate,nogoTrials)
        z = [scipy.stats.norm.ppf(r) for r in (hr,far)]
        return z[0]-z[1]


    def adjustResponseRate(self,r,n):
        if r == 0:
            r = 0.5/n
        elif r == 1:
            r = 1 - 0.5/n
        return r
    
    
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
                title = trialType + ' trials (n=' + str(trials.sum()) + ')'#, engaged (n=' + str(self.engagedTrials[trials].sum()) + ')'
                if trialType == 'go':
                    title += '\n' + 'hit rate ' + str(round(self.hitRate[blockInd],2)) + ', # hits ' + str(int(self.hitCount[blockInd]))
                elif trialType == 'no-go':
                    title += ('\n'+ 'false alarm same ' + str(round(self.falseAlarmSameModal[blockInd],2)) + 
                              ', diff go ' + str(round(self.falseAlarmDiffModalGo[blockInd],2)) +
                              ', diff nogo ' + str(round(self.falseAlarmDiffModalNogo[blockInd],2)) +
                              '\n' + 'dprime same ' + str(round(self.dprimeSameModal[blockInd],2)) +
                              ', diff go ' + str(round(self.dprimeDiffModalGo[blockInd],2)))
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
                title = trialType + ' trials (n=' + str(trials.sum()) + ')'#, engaged (n=' + str(self.engagedTrials[trials].sum()) + ')'
                if trialType == 'go':
                    title += '\n' + 'hit rate ' + str(round(self.hitRate[blockInd],2)) + ', # hits ' + str(int(self.hitCount[blockInd]))
                elif trialType == 'no-go':
                    title += ('\n'+ 'false alarm same ' + str(round(self.falseAlarmSameModal[blockInd],2)) + 
                              ', diff go ' + str(round(self.falseAlarmDiffModalGo[blockInd],2)) +
                              ', diff nogo ' + str(round(self.falseAlarmDiffModalNogo[blockInd],2)) +
                              '\n' + 'dprime same ' + str(round(self.dprimeSameModal[blockInd],2)) +
                              ', diff go ' + str(round(self.dprimeDiffModalGo[blockInd],2)))
                elif trialType == 'catch':
                    title += '\n' + 'catch rate ' + str(round(self.catchResponseRate[blockInd],2))
                ax.set_title(title)   
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(pdf,format='pdf')
            
            
        # ori
        if len(obj.gratingOri['vis2']) > 1:
            for blockInd,goStim in enumerate(self.blockStimRewarded):
                blockTrials = (self.trialBlock == blockInd + 1) & ~self.autoRewarded & ~self.catchTrials
                oris = np.unique(self.trialGratingOri)
                r = []
                for ori in oris:
                    trials = blockTrials & (self.trialGratingOri == ori)
                    r.append(self.trialResponse[trials].sum() / trials.sum())
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
            for blockInd,goStim in enumerate(self.blockStimRewarded):
                blockTrials = self.trialBlock == blockInd + 1
                fig = plt.figure()
                fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
                ax = fig.add_subplot(1,1,1)
                for trials,lbl in zip((self.goTrials,self.nogoTrials),('go','nogo')):
                    r = []
                    for c in obj.visContrast:
                        tr = trials & blockTrials & (self.trialVisContrast == c)
                        r.append(self.trialResponse[tr].sum() / tr.sum())
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
        stimLabels = np.unique(self.trialStim)
        notCatch = stimLabels != 'catch'
        clrs = np.zeros((len(stimLabels),3)) + 0.5
        clrs[notCatch] = plt.cm.plasma(np.linspace(0,0.85,notCatch.sum()))[:,:3]
        for stim,clr in zip(stimLabels,clrs):
            trials = (self.trialStim==stim) & self.trialResponse
            rt = self.responseTimes[trials]
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
        runPlotTime = np.arange(-preTime,postTime+1/self.frameRate,1/self.frameRate)
        if self.runningSpeed is not None:
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
                    speed = []
                    for st in self.stimStartTimes[trials]:
                        if st >= preTime and st+postTime <= self.frameTimes[-1]:
                            i = (self.frameTimes >= st-preTime) & (self.frameTimes <= st+postTime)
                            speed.append(np.interp(runPlotTime,self.frameTimes[i]-st,self.runningSpeed[i]))
                    meanSpeed = np.nanmean(speed,axis=0)
                    ymax = max(ymax,meanSpeed.max())
                    ax.plot(runPlotTime,meanSpeed)
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
        for trials,lbl in zip((obj.goTrials,obj.diffModalGoTrials),('goTrials','nogoTrials')):
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
    goProbFirst = []
    goProbLast = []
    goProbPrev = []
    nogoProbFirst = []
    nogoProbLast = []
    nogoProbPrev = []
    goLatFirst = []
    goLatLast = []
    goLatPrev = []
    nogoLatFirst = []
    nogoLatLast = []
    nogoLatPrev = [] 
    for block in blockData:
        if goStim in block['goStim'] and block['blockNum'] > 1:
            nTransitions += 1
            for d in blockData:
                if d['mouseID']==block['mouseID'] and d['blockNum']==block['blockNum']-1:
                    prevBlock = d
                    break
            goProbFirst.append(block['goTrials']['response'][0])
            goProbLast.append(block['goTrials']['response'][-1])
            goProbPrev.append(prevBlock['nogoTrials']['response'][-1])
            nogoProbFirst.append(block['nogoTrials']['response'][0])
            nogoProbLast.append(block['nogoTrials']['response'][-1])
            nogoProbPrev.append(prevBlock['goTrials']['response'][-1])
            goLatFirst.append(block['goTrials']['responseTime'][0])
            goLatLast.append(block['goTrials']['responseTime'][-1])
            goLatPrev.append(prevBlock['nogoTrials']['responseTime'][-1])
            nogoLatFirst.append(block['nogoTrials']['responseTime'][0])
            nogoLatLast.append(block['nogoTrials']['responseTime'][-1])
            nogoLatPrev.append(prevBlock['goTrials']['responseTime'][-1])
    
    title = (blockType+' rewarded blocks\n'
             'mean and 95% ci across transitions\n('+
             str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for prev,first,last,clr,lbl in zip((goProbPrev,nogoProbPrev),(goProbFirst,nogoProbFirst),(goProbLast,nogoProbLast),colors,labels):
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
    for prev,first,last,clr,lbl in zip((goLatPrev,nogoLatPrev),(goLatFirst,nogoLatFirst),(goLatLast,nogoLatLast),colors,labels):
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

    
   
# learning summary plots
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


