# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:30:37 2022

@author: svc_ccg
"""

import glob
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


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"


class DynRoutData():
    
    def __init__(self):
        self.frameRate = 60
        self.engagedThresh = 10
    
    
    def loadBehavData(self,filePath,h5pyFile=None):

        self.behavDataPath = filePath
        
        with h5pyFile or h5py.File(filePath,'r') as d:
        
            # self.subjectName = d['subjectName'][()]
            self.subjectName = re.search('.*_([0-9]{6})_',os.path.basename(self.behavDataPath)).group(1)
            self.rigName = d['rigName'].asstr()[()]
            self.computerName = d['computerName'].asstr()[()] if 'computerName' in d and  d['computerName'].dtype=='O' else None
            self.taskVersion = d['taskVersion'].asstr()[()] if 'taskVersion' in d else None
            self.startTime = d['startTime'].asstr()[()]
            
            self.frameIntervals = d['frameIntervals'][:]
            self.frameTimes = np.concatenate(([0],np.cumsum(self.frameIntervals)))
            
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
            
            self.soundVolume = d['soundVolume'][()]
            self.trialSoundVolume = d['trialSoundVolume'][:self.nTrials]
            
            if ('optoParams' in d and isinstance(d['optoParams'],h5py._hl.group.Group)) or ('optoRegions' in d and len(d['optoRegions']) > 0):
                self.trialOptoOnsetFrame = d['trialOptoOnsetFrame'][:self.nTrials]
                self.trialOptoDur = d['trialOptoDur'][:self.nTrials]
                self.trialOptoVoltage = d['trialOptoVoltage'][:self.nTrials]
                self.trialGalvoVoltage = d['trialGalvoVoltage'][:self.nTrials]
                if 'optoParams' in d:
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
                elif 'optoRegions' in d:
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
                except:
                    print('\nerror loading '+f+'\n')
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
                        'pass': 0,
                        'ignore': 0}  
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
                            if 'stage 5' in prevTask and 'repeats' not in prevTask:
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
        for col in ('ABCDEFGH'):
            if col in ('B','G','H'):
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
        for f in behavFiles:
            startTime = re.search('.*_([0-9]{8}_[0-9]{6})',f).group(1)
            startTime = pd.to_datetime(startTime,format='%Y%m%d_%H%M%S')
            if df is None or np.sum(df['start time']==startTime) < 1:
                try:
                    obj = DynRoutData()
                    obj.loadBehavData(f)
                    exps.append(obj)
                except:
                    print('\nerror loading '+f+'\n')
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
                        'ignore': 0}  
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
        for col in ('ABCDEFGH'):
            if col in ('B','C','H'):
                w = 15
            elif col=='D':
                w = 40
            else:
                w = 30
            sheet.column_dimensions[col].width = w
    
    writer.save()
    writer.close()


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

