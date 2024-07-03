# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:30:37 2022

@author: svc_ccg
"""

import contextlib
import glob
import os
import pathlib
import re
import time
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import scipy.cluster


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask')


class DynRoutData():
    
    def __init__(self):
        self.frameRate = 60
        self.engagedThresh = 10
    
    
    def loadBehavData(self,filePath,h5pyFile=None):

        self.behavDataPath = filePath

        if h5pyFile and isinstance(h5pyFile,h5py.File):
            # allow an already-open h5py File instance to be used,
            # but not closed by context block below
            d = h5pyFile
            context = contextlib.nullcontext()
        else:
            context = h5py.File(filePath,'r')
            d = context.__enter__()

        with context:
        
            # self.subjectName = d['subjectName'][()]
            self.subjectName = re.search('.*_([0-9]{6})_',os.path.basename(self.behavDataPath)).group(1)
            self.rigName = d['rigName'].asstr()[()]
            self.computerName = d['computerName'].asstr()[()] if 'computerName' in d and  d['computerName'].dtype=='O' else None
            self.taskVersion = d['taskVersion'].asstr()[()] if 'taskVersion' in d else None
            self.startTime = d['startTime'].asstr()[()]
            
            self.frameIntervals = d['frameIntervals'][:]
            self.frameTimes = np.concatenate(([0],np.cumsum(self.frameIntervals)))
            self.lastFrame =d['lastFrame'][()] if 'lastFrame' in d else None
            if self.lastFrame is not None and self.lastFrame != self.frameIntervals.size:
                print('\n',self.subjectName,self.startTime,'n frames',self.lastFrame,self.frameIntervals.size,'\n')
            
            self.trialEndFrame = d['trialEndFrame'][:]
            self.trialEndTimes = self.frameTimes[self.trialEndFrame]
            self.nTrials = self.trialEndFrame.size
            self.trialStartFrame = d['trialStartFrame'][:self.nTrials]
            self.trialStartTimes = self.frameTimes[self.trialStartFrame]
            self.stimStartFrame = d['trialStimStartFrame'][:self.nTrials]
            self.stimStartTimes = self.frameTimes[self.stimStartFrame]
            
            self.newBlockAutoRewards = d['newBlockAutoRewards'][()]
            self.newBlockGoTrials = d['newBlockGoTrials'][()]
            self.newBlockNogoTrials = d['newBlockNogoTrials'][()] if 'newBlockNogoTrials' in d else 0
            self.newBlockCatchTrials = d['newBlockCatchTrials'][()] if 'newBlockCatchTrials' in d else 0
            self.autoRewardOnsetFrame = d['autoRewardOnsetFrame'][()]
            
            self.trialRepeat = d['trialRepeat'][:self.nTrials]
            self.incorrectTrialRepeats = d['incorrectTrialRepeats'][()]
            self.incorrectTimeoutFrames = d['incorrectTimeoutFrames'][()]
            
            self.quiescentFrames = d['quiescentFrames'][()]
            self.quiescentViolationFrames = d['quiescentViolationFrames'][:] if 'quiescentViolationFrames' in d.keys() else d['quiescentMoveFrames'][:]    
            
            self.responseWindow = d['responseWindow'][:]
            self.responseWindowTime = np.array(self.responseWindow)/self.frameRate
            
            self.trialStim = d['trialStim'].asstr()[:self.nTrials]
            self.trialBlock = d['trialBlock'][:self.nTrials]
            self.blockTrial = np.concatenate([np.arange(np.sum(self.trialBlock==i)) for i in np.unique(self.trialBlock)])
            self.blockStartTimes = self.trialStartTimes[[np.where(self.trialBlock==i)[0][0] for i in np.unique(self.trialBlock)]]
            self.blockFirstStimTimes = self.stimStartTimes[[np.where(self.trialBlock==i)[0][0] for i in np.unique(self.trialBlock)]]
            self.blockStimRewarded = d['blockStimRewarded'].asstr()[:]
            self.rewardedStim = self.blockStimRewarded[self.trialBlock-1]
            
            self.rewardFrames = d['rewardFrames'][:]
            self.rewardTimes = self.frameTimes[self.rewardFrames]
            self.rewardSize = d['rewardSize'][:]
            self.trialResponse = d['trialResponse'][:self.nTrials]
            self.trialResponseFrame = d['trialResponseFrame'][:self.nTrials]
            self.trialRewarded = d['trialRewarded'][:self.nTrials]
            
            if 'trialAutoRewardScheduled' in d:
                self.autoRewardScheduled = d['trialAutoRewardScheduled'][:self.nTrials]
                self.autoRewarded = d['trialAutoRewarded'][:self.nTrials]
                if len(self.autoRewardScheduled) < self.nTrials:
                    self.autoRewardScheduled = np.zeros(self.nTrials,dtype=bool)
                    self.autoRewardScheduled[self.blockTrial < self.newBlockAutoRewards] = True
                if len(self.autoRewarded) < self.nTrials:
                    self.autoRewarded = self.autoRewardScheduled & np.in1d(self.stimStartFrame+self.autoRewardOnsetFrame,self.rewardFrames)
            else:
                self.autoRewardScheduled = d['trialAutoRewarded'][:self.nTrials]
                self.autoRewarded = self.autoRewardScheduled & np.in1d(self.stimStartFrame+self.autoRewardOnsetFrame,self.rewardFrames)
            self.rewardEarned = self.trialRewarded & (~self.autoRewarded)
            
            
            self.responseTimes = np.full(self.nTrials,np.nan)
            self.responseTimes[self.trialResponse] = self.frameTimes[self.trialResponseFrame[self.trialResponse].astype(int)] - self.stimStartTimes[self.trialResponse]
            
            self.lickFrames = d['lickFrames'][:]
            self.minLickInterval = 0.05
            if len(self.lickFrames) > 0:
                lickTimesDetected = self.frameTimes[self.lickFrames]
                isLick = np.concatenate(([True], np.diff(lickTimesDetected) > self.minLickInterval))
                self.lickTimes = lickTimesDetected[isLick]
            else:
                self.lickTimes = np.array([])
            
            if 'rotaryEncoder' in d and isinstance(d['rotaryEncoder'][()],bytes) and d['rotaryEncoder'].asstr()[()] == 'digital':
                self.runningSpeed = np.concatenate(([np.nan],np.diff(d['rotaryEncoderCount'][:]) * ((2 * np.pi * d['wheelRadius'][()] * self.frameRate) / d['rotaryEncoderCountsPerRev'][()])))
            else:
                self.runningSpeed = None
            
            self.visContrast = d['visStimContrast'][()]
            self.trialVisContrast = d['trialVisStimContrast'][:self.nTrials]
            if 'gratingOri' in d:
                self.gratingOri = {key: d['gratingOri'][key][()] for key in d['gratingOri']}
            else:
                self.gratingOri = {key: d['gratingOri_'+key][()] for key in ('vis1','vis2')}
            self.trialGratingOri = d['trialGratingOri'][:self.nTrials]
            
            self.soundVolume = d['soundVolume'][()]
            self.trialSoundVolume = d['trialSoundVolume'][:self.nTrials]
            
            if (('optoParams' in d and isinstance(d['optoParams'],h5py._hl.group.Group)) or 
                ('optoRegions' in d and len(d['optoRegions']) > 0) or
                ('optoProb' in d and d['optoProb'][()] > 0)
               ):
                self.trialOptoOnsetFrame = d['trialOptoOnsetFrame'][:self.nTrials]
                self.trialOptoDur = d['trialOptoDur'][:self.nTrials]
                self.trialOptoVoltage = d['trialOptoVoltage'][:self.nTrials]
                for param in ('trialGalvoVoltage','trialGalvoX','trialGalvoY'):
                    if param in d:
                        setattr(self,param,d[param][:self.nTrials])
                if 'optoParams' in d and isinstance(d['optoParams'],h5py._hl.group.Group):
                    self.optoParams = {}
                    for key in d['optoParams'].keys():
                        if key == 'label':
                            self.optoParams[key] = d['optoParams'][key].asstr()[()]
                        elif key == 'device':
                            self.optoParams[key] = [val.strip('\'[]').split(',') for val in d['optoParams'][key].asstr()[()]]
                        else:
                            self.optoParams[key] = d['optoParams'][key][()] 
                    self.trialOptoParamsIndex = d['trialOptoParamsIndex'][:self.nTrials]
                    self.trialOptoLabel = d['trialOptoLabel'].asstr()[:self.nTrials]
                    self.trialOptoDevice = [val.strip('\'[]').split(',') for val in d['trialOptoDevice'].asstr()[:self.nTrials]]
                    self.trialOptoDelay = d['trialOptoDelay'][:self.nTrials]
                    self.trialOptoOnRamp = d['trialOptoOnRamp'][:self.nTrials]
                    self.trialOptoOffRamp = d['trialOptoOffRamp'][:self.nTrials]
                    self.trialOptoSinFreq = d['trialOptoSinFreq'][:self.nTrials]
                    self.trialGalvoDwellTime = d['trialGalvoDwellTime'][:self.nTrials]
                elif 'optoRegions' in d and len(d['optoRegions']) > 0:
                    optoRegions = d['optoRegions'].asstr()[()]
                    optoVoltage = d['optoVoltage'][()]
                    galvoVoltage = d['galvoVoltage'][()]
                    self.trialOptoLabel = np.full(self.nTrials,'no opto',dtype=object)
                    for lbl,ov,gv in zip(optoRegions,optoVoltage,galvoVoltage):
                        self.trialOptoLabel[(self.trialOptoVoltage==ov) & np.all(self.trialGalvoVoltage==gv,axis=1)] = lbl
                    self.trialOptoVoltage = np.array([[val] for val in self.trialOptoVoltage])
                    self.trialGalvoVoltage = self.trialGalvoVoltage[:,None,:]
            
        self.catchTrials = self.trialStim == 'catch'
        self.multimodalTrials = np.array(['+' in stim for stim in self.trialStim])
        self.goTrials = (self.trialStim == self.rewardedStim) & (~self.autoRewardScheduled)
        self.nogoTrials = (self.trialStim != self.rewardedStim) & (~self.catchTrials) & (~self.multimodalTrials)
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
        self.dprimeNonrewardedModal = []
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
            self.dprimeNonrewardedModal.append(calcDprime(self.falseAlarmOtherModalGo[-1],self.falseAlarmOtherModalNogo[-1],otherModalGo.sum(),otherModalNogo.sum()))
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


def getPerformanceStats(df,sessions):
    hits = []
    dprimeSame = []
    dprimeOther = []
    for i in sessions:
        if isinstance(df.loc[i,'hits'],str):
            hits.append([int(s) for s in re.findall('[0-9]+',df.loc[i,'hits'])])
            dprimeSame.append([float(s) for s in re.findall('-*[0-9].[0-9]*|nan',df.loc[i,'d\' same modality'])])
            dprimeOther.append([float(s) for s in re.findall('-*[0-9].[0-9]*|nan',df.loc[i,'d\' other modality go stim'])])
        else:
            hits.append(df.loc[i,'hits'])
            dprimeSame.append(df.loc[i,'d\' same modality'])
            dprimeOther.append(df.loc[i,'d\' other modality go stim'])
    return hits,dprimeSame,dprimeOther


def getFirstExperimentSession(df):
    experimentSessions = np.where(['multimodal' in task
                                   or 'contrast'in task
                                   or 'opto' in task
                                   or 'nogo' in task
                                   or 'noAR' in task
                                   or 'rewardOnly' in task
                                   or 'no reward' in task
                                   for task in df['task version']])[0]
    firstExperimentSession = experimentSessions[0] if len(experimentSessions) > 0 else None
    return firstExperimentSession


def getSessionsToPass(mouseId,df,sessions,stage,hitThresh=100,dprimeThresh=1.5):
    sessionsToPass = np.nan
    for sessionInd in sessions:
        if sessionInd > sessions[0]:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if ((stage in (1,2) and all(h[0] >= hitThresh for h in hits) and all(d[0] >= dprimeThresh for d in dprimeSame)) or
                (stage==5 and np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3))):
                sessionsToPass = np.where(sessions==sessionInd)[0][0] + 1
                break
    if np.isnan(sessionsToPass):
        if stage in (1,2) and mouseId in (614910,684071,682893):
            sessionsToPass = len(sessions)
    return sessionsToPass


def getSessionData(mouseId,startTime):
    if not isinstance(startTime,str):
        startTime = startTime.strftime('%Y%m%d_%H%M%S')
    fileName = 'DynamicRouting1_' + str(mouseId) + '_' + startTime + '.hdf5'
    filePath = os.path.join(baseDir,'Data',str(mouseId),fileName)
    obj = DynRoutData()
    obj.loadBehavData(filePath)
    return obj

  
def updateTrainingSummary(mouseIds=None,replaceData=False):
    excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
    sheets = pd.read_excel(excelPath,sheet_name=None)
    writer =  pd.ExcelWriter(excelPath,mode='a',engine='openpyxl',if_sheet_exists='replace',datetime_format='%Y%m%d_%H%M%S')
    allMiceDf = sheets['all mice']
    if mouseIds is None:
        mouseIds = allMiceDf['mouse id']
    for mouseId in mouseIds:
        mouseInd = np.where(allMiceDf['mouse id']==mouseId)[0][0]
        if not replaceData and not allMiceDf.loc[mouseInd,'alive']:
            continue
        mouseId = str(mouseId)
        mouseDir = os.path.join(baseDir,'Data',mouseId)
        if not os.path.isdir(mouseDir):
            continue
        behavFiles = glob.glob(os.path.join(mouseDir,'DynamicRouting*.hdf5'))
        df = sheets[mouseId] if mouseId in sheets else None
        exps = []
        for f in behavFiles:
            startTime = re.search('.*_([0-9]{8}_[0-9]{6})',f).group(1)
            startTime = pd.to_datetime(startTime,format='%Y%m%d_%H%M%S')
            if replaceData or df is None or np.sum(df['start time']==startTime)==0:
                try:
                    obj = DynRoutData()
                    obj.loadBehavData(f)
                    exps.append(obj)
                except Exception as err:
                    print('\nerror loading '+f+'\n')
                    print(repr(err))
        if len(exps) < 1:
            continue
        exps = sortExps(exps)
        for obj in exps:
            try:
                data = {'start time': pd.to_datetime(obj.startTime,format='%Y%m%d_%H%M%S'),
                        'rig name': obj.rigName,
                        'task version': obj.taskVersion,
                        'hits': obj.hitCount,
                        'd\' same modality': np.round(obj.dprimeSameModal,2),
                        'd\' other modality go stim': np.round(obj.dprimeOtherModalGo,2),
                        'quiescent violations': obj.quiescentViolationFrames.size,
                        'pass': 0,
                        'ignore': 0,
                        'hab': 0,
                        'ephys': 0}  
                if df is None:
                    df = pd.DataFrame(data)
                    sessionInd = 0
                else:
                    if 'rig name' not in df.columns:
                        df.insert(1,'rig name','')
                    sessionInd = df['start time'] == data['start time']
                    sessionInd = np.where(sessionInd)[0][0] if sessionInd.sum()>0 else df.shape[0]
                    df.loc[sessionInd] = list(data.values())
                
                if 'stage' in obj.taskVersion and 'templeton' not in obj.taskVersion:
                    regimen = int(allMiceDf.loc[mouseInd,'regimen'])
                    hitThresh = 150 if regimen==1 else 100
                    dprimeThresh = 1.5
                    lowRespThresh = 10
                    task = df.loc[sessionInd,'task version']
                    prevTask = df.loc[sessionInd-1,'task version'] if sessionInd>0 else ''
                    passStage = 0
                    handOff = False
                    if 'stage 0' in task:
                        passStage = 1
                        nextTask = 'stage 1 AMN' if regimen > 4 else 'stage 1'
                    else:
                        if sessionInd > 0:
                            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
                        if 'stage 1' in task:
                            if 'stage 1' in prevTask and all(h[0] < lowRespThresh for h in hits):
                                passStage = -1
                                nextTask = 'stage 0'
                            elif 'stage 1' in prevTask and all(h[0] >= hitThresh for h in hits) and all(d[0] >= dprimeThresh for d in dprimeSame):
                                passStage = 1
                                nextTask = 'stage 2 AMN' if regimen > 4 else 'stage 2'
                            else:
                                nextTask = 'stage 1 AMN' if regimen > 4 else 'stage 1'
                        elif 'stage 2' in task:
                            if 'stage 2' in prevTask and all(h[0] >= hitThresh for h in hits) and all(d[0] >= dprimeThresh for d in dprimeSame):
                                passStage = 1
                                if regimen>6:
                                    nextTask = 'stage 5 ori AMN'
                                elif regimen in (5,6):
                                    nextTask = 'stage variable ori AMN'
                                else:
                                    nextTask = 'stage 3 ori'
                            else:
                                nextTask = 'stage 2 AMN' if regimen > 4 else 'stage 2'
                        elif 'stage 3' in task:
                            remedial = any('stage 4' in s for s in df['task version'])
                            if ('stage 3' in prevTask
                                 and ((regimen==1 and all(all(h >= hitThresh for h in hc) for hc in hits) and all(all(d >= dprimeThresh for d in dp) for dp in dprimeSame))
                                      or (regimen>1 and all(all(h >= hitThresh/2 for h in hc) for hc in hits) and all(all(d >= dprimeThresh for d in dp) for dp in dprimeSame+dprimeOther)))):
                                passStage = 1
                                if regimen==2 and not any('stage 3 tone' in s for s in df['task version']):
                                    nextTask = 'stage 3 tone'
                                elif regimen==3:
                                    nextTask = 'stage 4 ori tone ori'
                                elif regimen==4:
                                    nextTask = 'stage 5 ori tone'
                                else:
                                    nextTask = 'stage 4 tone ori' if remedial and 'tone' in task else 'stage 4 ori tone'
                            else:
                                if remedial:
                                    nextTask = 'stage 3 ori' if 'ori' in task else 'stage 3 tone'
                                elif (regimen==2 and not any('stage 3 tone' in s for s in df['task version'])) or regimen>2:
                                    nextTask = 'stage 3 ori'
                                else:
                                    nextTask = 'stage 3 tone' if 'ori' in task else 'stage 3 ori'
                        elif 'stage 4' in task:
                            if 'stage 4' in prevTask:
                                lowRespOri = (('stage 4 ori' in prevTask and hits[0][0] < lowRespThresh and hits[1][1] < lowRespThresh)
                                              or ('stage 4 tone' in prevTask and hits[0][1] < lowRespThresh and hits[1][0] < lowRespThresh))
                                lowRespTone = (('stage 4 tone' in prevTask and hits[0][0] < lowRespThresh and hits[1][1] < lowRespThresh)
                                               or ('stage 4 ori' in prevTask and hits[0][1] < lowRespThresh and hits[1][0] < lowRespThresh))
                            if 'stage 4' in prevTask and (lowRespOri or lowRespTone):
                                passStage = -1
                                nextTask = 'stage 3 ori' if lowRespOri else 'stage 3 tone'
                            elif 'stage 4' in prevTask and all(all(d >= dprimeThresh for d in dp) for dp in dprimeSame+dprimeOther):
                                passStage = 1
                                nextTask = 'stage 5 ori tone'
                            elif regimen==3:
                                nextTask = 'stage 4 ori tone ori'
                            else:
                                nextTask = 'stage 4 ori tone' if 'stage 4 tone' in task else 'stage 4 tone ori'
                        elif 'stage 5' in task:
                            if 'stage 5' in prevTask and np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                                passStage = 1
                                handOff = True
                            if regimen==8 and 'stage 5' in prevTask and 'repeats' not in prevTask:
                                handOff = True
                            if 'AMN' in task:
                                nextTask = 'stage 5 AMN ori' if 'stage 5 ori' in task else 'stage 5 ori AMN'
                            else:
                                nextTask = 'stage 5 tone ori' if 'stage 5 ori' in task else 'stage 5 ori tone'
                        elif 'stage variable' in task:
                            if not np.any(np.isnan(obj.dprimeOtherModalGo)):
                                passStage = 1
                                if 'AMN' in task:
                                    nextTask = 'stage 5 AMN ori' if 'stage 5 ori' in task else 'stage 5 ori AMN'
                                else:
                                    nextTask = 'stage 5 tone ori' if 'stage 5 ori' in task else 'stage 5 ori tone'
                            else:
                                if 'AMN' in task:
                                    nextTask = 'stage variable AMN ori' if 'stage variable ori' in task else 'stage variable ori AMN'
                                else:
                                    nextTask = 'stage variable tone ori' if 'stage variable ori' in task else 'stage variable ori tone'
                    if 'stage 3' in nextTask and regimen>1:
                        nextTask += ' distract'
                    if regimen>3 and 'stage 2' not in nextTask and nextTask != 'hand off':
                        nextTask += ' moving'
                    if not handOff and allMiceDf.loc[mouseInd,'timeouts'] and 'stage 0' not in nextTask and ((regimen>3 and 'timeouts' in prevTask) or 'stage 5' not in nextTask):
                        nextTask += ' timeouts'
                    if regimen==8 and not handOff and 'stage 5' in nextTask and ('stage 5' not in task or 'repeats' in task):
                        nextTask += ' repeats'
                    if regimen==3 and ('stage 1' in nextTask or 'stage 2' in nextTask):
                        nextTask += ' long'
                    df.loc[sessionInd,'pass'] = passStage
                    
                    if df.shape[0] in (1,sessionInd+1):
                        allMiceDf.loc[mouseInd,'next task version'] = nextTask
            except:
                print('error processing '+mouseId+', '+obj.startTime+'\n')
        
        df.to_excel(writer,sheet_name=obj.subjectName,index=False)
        sheet = writer.sheets[obj.subjectName]
        for col in ('ABCDEFGHIJK'):
            if col in ('H','I','J','K'):
                w = 10
            elif col in ('B','G'):
                w = 15
            elif col=='C':
                w = 40
            else:
                w = 30
            sheet.column_dimensions[col].width = w
       
    allMiceDf.to_excel(writer,sheet_name='all mice',index=False)
    sheet = writer.sheets['all mice']
    for col in ('ABCDEFGHIJKL'):
        if col in ('E','F'):
            w = 20
        elif col=='L':
            w = 30
        else:
            w = 12
        sheet.column_dimensions[col].width = w
    writer.save()
    writer.close()
    
    
def updateTrainingSummaryNSB():
    excelPath = os.path.join(baseDir,'DynamicRoutingTrainingNSB.xlsx')
    sheets = pd.read_excel(excelPath,sheet_name=None)
    writer =  pd.ExcelWriter(excelPath,mode='a',engine='openpyxl',if_sheet_exists='replace',datetime_format='%Y%m%d_%H%M%S')
    allMiceDf = sheets['all mice']

    mouseIds = allMiceDf['mouse id']
    for mouseId in mouseIds:
        mouseInd = np.where(allMiceDf['mouse id']==mouseId)[0][0]
        if not allMiceDf.loc[mouseInd,'alive']:
            continue
        mouseId = str(mouseId)
        mouseDir = allMiceDf.loc[mouseInd,'data path']
        behavFiles = glob.glob(os.path.join(mouseDir,'**','DynamicRouting*.hdf5'))
        behavFiles += glob.glob(os.path.join(baseDir,'Data',mouseId,'DynamicRouting*.hdf5'))
        df = sheets[mouseId] if mouseId in sheets else None
        exps = []
        for f in set(behavFiles):
            startTime = re.search('.*_([0-9]{8}_[0-9]{6})',f).group(1)
            startTime = pd.to_datetime(startTime,format='%Y%m%d_%H%M%S')
            if df is None or np.sum(df['start time']==startTime) < 1:
                try:
                    obj = DynRoutData()
                    obj.loadBehavData(f)
                    exps.append(obj)
                except Exception as err:
                    print('\nerror loading '+f+'\n')
                    print(repr(err))
        if len(exps) < 1:
            continue
        exps = sortExps(exps)
        for obj in exps:
            try:
                data = {'start time': pd.to_datetime(obj.startTime,format='%Y%m%d_%H%M%S'),
                        'rig name': obj.rigName,
                        'computer name': obj.computerName,
                        'task version': obj.taskVersion,
                        'hits': obj.hitCount,
                        'd\' same modality': np.round(obj.dprimeSameModal,2),
                        'd\' other modality go stim': np.round(obj.dprimeOtherModalGo,2),
                        'quiescent violations': obj.quiescentViolationFrames.size,
                        'ignore': 0,
                        'hab': 0,
                        'ephys': 0}  
                if df is None:
                    df = pd.DataFrame(data)
                    sessionInd = 0
                else:
                    sessionInd = df['start time'] == data['start time']
                    sessionInd = np.where(sessionInd)[0][0] if sessionInd.sum()>0 else df.shape[0]
                    df.loc[sessionInd] = list(data.values())
            except:
                print('error processing '+mouseId+', '+obj.startTime+'\n')
        
        df.to_excel(writer,sheet_name=obj.subjectName,index=False)
        sheet = writer.sheets[obj.subjectName]
        for col in ('ABCDEFGHIJK'):
            if col in ('I','J','K'):
                w = 10
            elif col in ('B','C','H'):
                w = 15
            elif col=='D':
                w = 40
            else:
                w = 30
            sheet.column_dimensions[col].width = w
    
    writer.save()
    writer.close()


def fitCurve(func,x,y,initGuess=None,bounds=None):
    return scipy.optimize.curve_fit(func,x,y,p0=initGuess,bounds=bounds)[0]
    

def calcLogisticDistrib(x,a,b,m,s):
    # a: amplitude, b: offset, m: x at 50% max y, s: scale
    return a * (1 / (1 + np.exp(-(x - m) / s))) + b

def inverseLogistic(y,a,b,m,s):
    return m - s * np.log((a / (y - b)) - 1)


def calcWeibullDistrib(x,a,b,j,k):
    # a: amplitude, b: offset, j: shape, k: scale
    return a * (1 - np.exp(-(x / j) ** k)) + b

def inverseWeibull(y,a,b,j,k):
    return j * (-np.log(1 - ((y - b) / a))) ** (1/k)
    

def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    # if plot:
    #     fig = plt.figure(facecolor='w')
    #     ax = fig.add_subplot(1,1,1)
    #     ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
    #     ax.set_xlim((0.5,eigVal.size+0.5))
    #     ax.set_ylim((0,1.02))
    #     ax.set_xlabel('PC')
    #     ax.set_ylabel('Cumulative Fraction of Variance')
    #     for side in ('right','top'):
    #         ax.spines[side].set_visible(False)
    #     ax.tick_params(direction='out',top=False,right=False)
        
    #     fig = plt.figure(facecolor='w')
    #     ax = fig.add_subplot(1,1,1)
    #     im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
    #     ax.set_xlabel('PC')
    #     ax.set_ylabel('Parameter')
    #     ax.set_title('PC Weightings')
    #     for side in ('right','top'):
    #         ax.spines[side].set_visible(False)
    #     ax.tick_params(direction='out',top=False,right=False)
    #     cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    #     cb.ax.tick_params(length=0)
    #     cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec


def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None,labels=None,xmax=None,nreps=1000,title=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustId = None
    else:
        clustId = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    # if plot:
    #     plt.figure(facecolor='w')
    #     ax = plt.subplot(1,1,1)
    #     colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
    #     if colors is not None:
    #         scipy.cluster.hierarchy.set_link_color_palette(list(colors))
    #     if labels=='off':
    #         labels=None
    #         noLabels=True
    #     else:
    #         noLabels=False
    #     scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k',labels=labels,no_labels=noLabels)
    #     scipy.cluster.hierarchy.set_link_color_palette(None)
    #     ax.set_yticks([])
    #     for side in ('right','top','left','bottom'):
    #         ax.spines[side].set_visible(False)
    #     if title is not None:
    #         ax.set_title(title)
    #     plt.tight_layout()
            
    #     plt.figure(facecolor='w')
    #     ax = plt.subplot(1,1,1)
    #     k = np.arange(linkageMat.shape[0])+2
    #     if nreps>0:
    #         randLinkage = np.zeros((nreps,linkageMat.shape[0]))
    #         shuffledData = data.copy()
    #         for i in range(nreps):
    #             for j in range(data.shape[1]):
    #                 shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
    #             _,m = cluster(shuffledData,method=method,metric=metric)
    #             randLinkage[i] = m[::-1,2]
    #         ax.plot(k,np.percentile(randLinkage,2.5,axis=0),'k--')
    #         ax.plot(k,np.percentile(randLinkage,97.5,axis=0),'k--')
    #     ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
    #     if xmax is None:
    #         ax.set_xlim([0,k[-1]+1])
    #     else:
    #         ax.set_xlim([0,xmax])
    #     ax.set_xlabel('Cluster')
    #     ax.set_ylabel('Linkage Distance')
    #     for side in ('right','top'):
    #         ax.spines[side].set_visible(False)
    #     ax.tick_params(direction='out',top=False,right=False)
    #     if title is not None:
    #         ax.set_title(title)
    #     plt.tight_layout()
    return clustId,linkageMat

