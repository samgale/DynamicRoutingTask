# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools
import random
import numpy as np
from psychopy import visual
from TaskControl import TaskControl


class DynamicRouting1(TaskControl):
    
    def __init__(self,rigName,taskVersion=None):
        TaskControl.__init__(self,rigName)
        self.taskVersion = taskVersion
        self.maxFrames = 60 * 3600
        self.maxTrials = None
        self.spacebarRewardsEnabled = False
        
        # block stim is one list per block containing one or more 'vis#' or 'sound#' or a list of these for multimodal stimuli
        # first element rewarded
        # last block continues until end of session
        self.blockStim = [['vis1','vis2']] 
        self.blockStimRewarded = ['vis1']
        self.blockStimProb = 'even sampling' # 'equal', 'even sampling', or list of probabilities for each stimulus in each block adding to one
        self.evenSampleContrastVolume = False # evenly sample contrasts and volumes if blockStimProb is 'even sampling'
        self.blockProbCatch = [0.1] # fraction of trials for each block with no stimulus and no reward
        self.trialsPerBlock = None # None or sequence of trial numbers for each block; use this or framesPerBlock
        self.framesPerBlock = None # None or sequence of frame numbers for each block
        self.newBlockGoTrials = 5 # number of consecutive go trials at the start of each block (otherwise random)
        self.newBlockAutoRewards = 5 # number of autorewarded trials at the start of each block

        self.preStimFramesFixed = 90 # min frames between start of trial and stimulus onset
        self.preStimFramesVariableMean = 60 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 360 # max total preStim frames
        self.quiescentFrames = 90 # frames before stim onset during which licks delay stim onset
        self.responseWindow = [6,60]
        self.postResponseWindowFrames = 180

        self.autoRewardOnsetFrame = 6 # frames after stimulus onset at which autoreward occurs
        self.autoRewardMissTrials = 10 # None or consecutive miss trials after which autoreward delivered on next go trial

        self.rewardProbGo = 1 # probability of reward after response on go trial
        self.rewardProbCatch = 0 # probability of autoreward at end of response window on catch trial
        
        self.rewardSound = None # None or name of sound trigger, 'tone', or 'noise' for sound played with reward delivery
        self.rewardSoundDur = 0.1 # seconds
        self.rewardSoundVolume = 0.1 # 0-1
        self.rewardSoundFreq = 10000 # Hz
        
        self.incorrectTrialRepeats = 0 # maximum number of incorrect trial repeats
        self.incorrectTimeoutFrames = 0 # extended gray screen following incorrect trial
        self.incorrectTimeoutColor = 0 # -1 to 1
        self.incorrectSound = None # None or name of sound trigger, 'tone', or 'noise' for sound played after incorrect trial
        self.incorrectSoundDur = 3 # seconds
        self.incorrectSoundVolume = 0.1 # 0-1
        self.incorrectSoundFreq = [2000,20000] # Hz
        
        # visual stimulus params
        # parameters that can vary across trials are lists
        self.visStimType = 'grating'
        self.visStimFrames = [30] # duration of visual stimulus
        self.visStimContrast = [1]
        self.gratingSize = 50 # degrees
        self.gratingSF = 0.04 # cycles/deg
        self.gratingTF = 0 # cycles/s
        self.gratingOri = {'vis1':[0],'vis2':[90]} # clockwise degrees from vertical
        self.gratingPhase = [0,0.5]
        self.gratingType = 'sqr' # 'sin' or sqr'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos
        
        # auditory stimulus params
        self.saveSoundArray = False
        self.soundType = 'tone' # 'tone', 'linear sweep', 'log sweep', 'noise', 'AM noise', or dict
        self.soundRandomSeed = None
        self.soundDur = [0.5] # seconds
        self.soundVolume = [0.1] # 0-1
        self.toneFreq = {'sound1':6000,'sound2':10000} # Hz
        self.linearSweepFreq = {'sound1':[6000,10000],'sound2':[10000,6000]}
        self.logSweepFreq = {'sound1':[3,2.5],'sound2':[3,3.5]} # log2(kHz)
        self.noiseFiltFreq = {'sound1':[4000,8000],'sound2':[8000,16000]} # Hz
        self.ampModFreq = {'sound1':20,'sound2':40} # Hz
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)

        if rigName == 'NP3':
            self.soundVolume = [1.0]
            self.soundRandomSeed = 0
            self.saveSoundArray = True

    
    def setDefaultParams(self,taskVersion):
        # dynamic routing task versions
        if taskVersion in ('stage 0','stage 0 moving'):
            # auto rewards
            self.blockStim = [['vis1','vis2']]
            self.blockStimRewarded = ['vis1']
            if 'moving' in taskVersion:
                self.gratingTF = 2
            self.maxTrials = 150
            self.newBlockAutoRewards = 150
            self.quiescentFrames = 0
            self.blockProbCatch = [0]

        elif taskVersion in ('stage 1','stage 1 moving','stage 1 timeouts','stage 1 moving timeouts',
                             'stage 1 long','stage 1 moving long','stage 1 timeouts long','stage 1 moving timeouts long'):
            # ori discrim with or without timeouts
            self.blockStim = [['vis1','vis2']]
            self.blockStimRewarded = ['vis1']
            self.incorrectTrialRepeats = 3
            if 'moving' in taskVersion:
                self.gratingTF = 2
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1
            if 'long' in taskVersion:
                self.maxFrames = 75 * 3600

        elif taskVersion in ('stage 2','stage 2 timeouts','stage 2 long','stage 2 timeouts long',
                             'stage 2 AMN','stage 2 AMN timeouts','stage 2 AMN long','stage 2 AMN timeouts long'):
            # tone discrim with or without timeouts
            self.soundType = 'AM noise' if 'AMN' in taskVersion else 'tone'
            self.blockStim = [['sound1','sound2']]
            self.blockStimRewarded = ['sound1']
            self.incorrectTrialRepeats = 3
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1
            if 'long' in taskVersion:
                self.maxFrames = 75 * 3600

        elif taskVersion in ('stage 3 ori','stage 3 ori moving','stage 3 ori timeouts','stage 3 ori moving timeouts',
                             'stage 3 tone','stage 3 tone timeouts','stage 3 AMN','stage 3 AMN timeouts'):
            # ori or tone discrim
            if 'ori' in taskVersion:
                self.blockStim = [['vis1','vis2']]
                self.blockStimRewarded = ['vis1']
            else:
                self.soundType = 'AM noise' if 'AMN' in taskVersion else 'tone'
                self.blockStim = [['sound1','sound2']]
                self.blockStimRewarded = ['sound1']
            if 'moving' in taskVersion:
                self.gratingTF = 2
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('stage 3 ori distract','stage 3 ori distract moving',
                             'stage 3 ori distract timeouts','stage 3 ori distract moving timeouts',
                             'stage 3 tone distract','stage 3 tone distract timeouts',
                             'stage 3 ori AMN distract','stage 3 ori AMN distract moving',
                             'stage 3 ori AMN distract timeouts','stage 3 ori AMN distract moving timeouts',
                             'stage 3 AMN distract','stage 3 AMN distract timeouts'):
            # ori or tone discrim with distractors
            self.blockStim = [['vis1','vis2','sound1','sound2']]
            self.soundType = 'AM noise' if 'AMN' in taskVersion else 'tone'
            if 'ori' in taskVersion:
                self.blockStimRewarded = ['vis1']
            else:
                self.blockStimRewarded = ['sound1']
            if 'moving' in taskVersion:
                self.gratingTF = 2
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('stage 4 ori tone','stage 4 tone ori',
                             'stage 4 ori tone moving','stage 4 tone ori moving',
                             'stage 4 ori tone timeouts','stage 4 tone ori timeouts',
                             'stage 4 ori tone moving timeouts','stage 4 tone ori moving timeouts',
                             'stage 4 ori tone ori','stage 4 ori tone ori timeouts',
                             'stage 4 ori tone ori moving','stage 4 ori tone ori moving timeouts',
                             'stage 4 ori AMN','stage 4 AMN ori',
                             'stage 4 ori AMN moving','stage 4 AMN ori moving',
                             'stage 4 ori AMN timeouts','stage 4 AMN ori timeouts',
                             'stage 4 ori AMN moving timeouts','stage 4 AMN ori moving timeouts',
                             'stage 4 ori AMN ori','stage 4 ori AMN ori timeouts',
                             'stage 4 ori AMN ori moving','stage 4 ori AMN ori moving timeouts'):
            # 2 or 3 blocks of all 4 stimuli, switch rewarded modality
            if 'ori tone ori' in taskVersion or 'ori AMN ori' in taskVersion:
                self.blockStim = [['vis1','vis2','sound1','sound2']] * 3
                self.blockStimRewarded = ['vis1','sound1','vis1']
                self.framesPerBlock = np.array([20] * 3) * 3600
                self.blockProbCatch = [0.1] * 3
            else:
                self.blockStim = [['vis1','vis2','sound1','sound2']] * 2
                if 'ori tone' in taskVersion or 'ori AMN' in taskVersion:
                    self.blockStimRewarded = ['vis1','sound1']
                else:
                    self.blockStimRewarded = ['sound1','vis1']
                self.framesPerBlock = np.array([30,30]) * 3600
                self.blockProbCatch = [0.1,0.1]
            self.soundType = 'AM noise' if 'AMN' in taskVersion else 'tone'
            self.maxFrames = None
            if 'moving' in taskVersion:
                self.gratingTF = 2
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('stage 5 ori tone','stage 5 tone ori',
                             'stage 5 ori tone moving','stage 5 tone ori moving',
                             'stage 5 ori tone timeouts','stage 5 tone ori timeouts',
                             'stage 5 ori tone moving timeouts','stage 5 tone ori moving timeouts',
                             'stage 5 ori AMN','stage 5 AMN ori',
                             'stage 5 ori AMN moving','stage 5 AMN ori moving',
                             'stage 5 ori AMN timeouts','stage 5 AMN ori timeouts',
                             'stage 5 ori AMN moving timeouts','stage 5 AMN ori moving timeouts'):
            # 6 blocks
            self.blockStim = [['vis1','vis2','sound1','sound2']] * 6
            self.soundType = 'AM noise' if 'AMN' in taskVersion else 'tone'
            if 'ori tone' in taskVersion or 'ori AMN' in taskVersion:
                self.blockStimRewarded = ['vis1','sound1'] * 3
            else:
                self.blockStimRewarded = ['sound1','vis1'] * 3
            self.maxFrames = None
            self.framesPerBlock = np.array([10] * 6) * 3600
            self.blockProbCatch = [0.1] * 6
            if 'moving' in taskVersion:
                self.gratingTF = 2
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('contrast volume','volume contrast'):
            self.blockStim = [['vis1','vis2','sound1','sound2']] * 2
            self.soundType = 'tone'
            if taskVersion == 'contrast volume':
                self.blockStimRewarded = ['vis1','sound1']
            else:
                self.blockStimRewarded = ['sound1','vis1']
            self.maxFrames = None
            self.framesPerBlock = np.array([30,30]) * 3600
            self.blockProbCatch = [0.05,0.05]
            self.evenSampleContrastVolume = True
            self.visStimContrast = [0.01,0.02,0.03,0.04,0.05,0.06]
            self.soundVolume = [0.01,0.012,0.014,0.016,0.018,0.02]

        elif taskVersion in ('multimodal ori tone','multimodal tone ori'):
            self.blockStim = [['vis1','vis2','sound1','sound2','vis1+sound1','vis1+sound2','vis2+sound1','vis2+sound2']] * 2
            self.soundType = 'tone'
            if 'ori tone' in taskVersion:
                self.blockStimRewarded = ['vis1','sound1']
            else:
                self.blockStimRewarded = ['sound1','vis1']
            self.maxFrames = None
            self.framesPerBlock = np.array([30,30]) * 3600
            self.blockProbCatch = [0.1,0.1]
            self.visStimContrast = [1]
            self.soundVolume = [0.1]

        elif taskVersion == 'passive':
            self.blockStim = [['vis1','vis2'] + ['sound'+str(i+1) for i in range(10)]]
            self.blockStimProb = 'even sampling'
            self.soundType = {'sound1':'tone','sound2':'tone',
                              'sound3':'linear sweep','sound4':'linear sweep',
                              'sound5':'log sweep','sound6':'log sweep',
                              'sound7':'noise','sound8':'noise',
                              'sound9':'AM noise','sound10':'AM noise'}
            self.toneFreq = {'sound1':6000,'sound2':10000}
            self.linearSweepFreq = {'sound3':[6000,10000],'sound4':[10000,6000]}
            self.logSweepFreq = {'sound5':[3,2.5],'sound6':[3,3.5]}
            self.noiseFiltFreq = {'sound7':[4000,8000],'sound8':[8000,16000]}
            self.ampModFreq = {'sound9':20,'sound10':40}
            self.soundVolume = [1]
            self.newBlockGoTrials = 0
            self.newBlockAutoRewards = 0
            self.autoRewardMissTrials = None
            self.soundRandomSeed = 0
            self.saveSoundArray = True

        # templeton task versions
        elif 'templeton' in taskVersion:
            self.maxFrames = 60 * 3600
            self.visStimFrames = [30]
            self.soundDur = [0.5]
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.soundType = 'tone'
            self.newBlockGoTrials = 5
            self.newBlockAutoRewards = 5
            self.autoRewardMissTrials = 10

            if 'stage 0 vis' in taskVersion:
                self.blockStim = [['vis1','vis2']]
                self.blockStimRewarded = ['vis1']
                self.maxTrials = 150
                self.newBlockAutoRewards = 150
                self.quiescentFrames = 0
                self.blockProbCatch = [0]

            elif 'stage 1 vis' in taskVersion:
                self.blockStim = [['vis1','vis2']]
                self.blockStimRewarded = ['vis1']
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1
                self.incorrectTrialRepeats = 3

            elif 'stage 2 vis' in taskVersion:
                self.blockStim = [['vis1','vis2','sound1','sound2']]
                self.blockStimRewarded = ['vis1']
                self.visStimFrames = [30,45,60]
                self.soundDur = [0.5,0.75,1.0]
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240
                self.postResponseWindowFrames = 120

            elif 'stage 0 aud' in taskVersion:
                self.blockStim = [['sound1','sound2']]
                self.blockStimRewarded = ['sound1']
                self.maxTrials = 150
                self.newBlockAutoRewards = 150
                self.quiescentFrames = 0
                self.blockProbCatch = [0]

            elif 'stage 1 aud' in taskVersion:
                self.blockStim = [['sound1','sound2']]
                self.blockStimRewarded = ['sound1']
                self.incorrectTimeoutFrames = 180
                self.incorrectSound = 'noise'
                self.incorrectTrialRepeats = 3

            elif 'stage 2 aud' in taskVersion:
                self.blockStim = [['sound1','sound2','vis1','vis2']]
                self.blockStimRewarded = ['sound1']
                self.visStimFrames = [30,45,60]
                self.soundDur = [0.5,0.75,1.0]
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240
                self.postResponseWindowFrames = 120

        
        elif 'hearing check' in taskVersion:
            self.blockStim = [['sound1','sound2']]
            self.blockStimRewarded = ['sound1']
            self.soundType = 'tone'
            self.soundDur = [1]
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.maxFrames = None
            self.framesPerBlock = np.array([60]) * 3600

            self.preStimFramesFixed = 30 
            self.preStimFramesVariableMean = 30 
            self.preStimFramesMax = 240
            self.postResponseWindowFrames = 120

            self.newBlockGoTrials = 5
            self.autoRewardMissTrials = 5
            self.newBlockAutoRewards = 5

            self.soundVolume = [0.1,0.2,0.4,0.8]
            # self.soundVolume = [0.1,0.5,1.0]

            if 'test' in taskVersion:
                self.framesPerBlock = np.array([1]) * 3600
                self.newBlockGoTrials = 1
                self.autoRewardMissTrials = 5
                self.newBlockAutoRewards = 1

        else:
            raise ValueError(taskVersion + ' is not a recognized task version')

        if 'rewardProb' in taskVersion:
            self.rewardProbGo = 0.90 # probability of reward after response on go trial
            self.rewardProbCatch = 0.20 # probability of autoreward at end of response window on catch trial

        if 'maxvol' in taskVersion:
            self.soundVolume = [1.0]

        if 'record' in taskVersion:
            self.soundVolume = [1.0]
            self.soundRandomSeed = 0
            self.saveSoundArray = True
    

    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create visual stimulus
        if self.visStimType == 'grating':
            edgeBlurWidth = {'fringeWidth':self.gratingEdgeBlurWidth} if self.gratingEdge=='raisedCos' else None
            visStim = visual.GratingStim(win=self._win,
                                         units='pix',
                                         mask=self.gratingEdge,
                                         maskParams=edgeBlurWidth,
                                         tex=self.gratingType,
                                         pos=(0,0),
                                         size=int(self.gratingSize * self.pixelsPerDeg), 
                                         sf=self.gratingSF / self.pixelsPerDeg)

        # sound for reward or incorrect response
        if self.soundMode == 'internal':
            if self.rewardSound is not None:
                rewardSoundArray = self.makeSoundArray(soundType=self.rewardSound,
                                                       dur=self.rewardSoundDur,
                                                       vol=self.rewardSoundVolume,
                                                       freq=self.rewardSoundFreq)
            if self.incorrectSound is not None:
                incorrectSoundArray = self.makeSoundArray(soundType=self.incorrectSound,
                                                          dur=self.incorrectSoundDur,
                                                          vol=self.incorrectSoundVolume,
                                                          freq=self.incorrectSoundFreq)
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialStim = []
        self.trialVisStimFrames = []
        self.trialVisStimContrast = []
        self.trialGratingOri = []
        self.trialGratingPhase = []
        self.trialSoundType = []
        self.trialSoundDur = []
        self.trialSoundVolume = []
        self.trialSoundFreq = []
        self.trialSoundAM = []
        self.trialSoundArray = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.trialRewarded = []
        self.trialAutoRewarded = []
        self.quiescentViolationFrames = [] # frames where quiescent period was violated
        self.trialRepeat = [False]
        self.trialBlock = []
        blockNumber = 0 # current block
        blockTrials = None # total number of trials to occur in current block
        blockFrames = None # total number of frames to occur in current block
        blockTrialCount = 0 # number of trials completed in current block
        blockFrameCount = 0 # number of frames completed in current block
        blockAutoRewardCount = 0
        missTrialCount = 0
        incorrectRepeatCount = 0
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getInputData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                if len(self.trialStartFrame) < 1:
                    preStimFrames += self.postResponseWindowFrames
                self.trialPreStimFrames.append(preStimFrames) # can grow larger than preStimFrames during quiescent period
                
                if self.trialRepeat[-1]:
                    self.trialStim.append(self.trialStim[-1])
                else:
                    if (blockNumber == 0 or 
                        (blockNumber < len(self.blockStim) and
                        (blockTrialCount == blockTrials or (blockFrames is not None and blockFrameCount >= blockFrames)))):
                        # start new block of trials
                        blockNumber += 1
                        blockTrials = None if self.trialsPerBlock is None else self.trialsPerBlock[blockNumber-1]
                        blockFrames = None if self.framesPerBlock is None else self.framesPerBlock[blockNumber-1]
                        blockTrialCount = 0
                        blockFrameCount = 0
                        blockAutoRewardCount = 0
                        missTrialCount = 0
                        incorrectRepeatCount = 0
                        blockStim = self.blockStim[blockNumber-1]
                        stimProb = None if self.blockStimProb in ('equal','even sampling') else self.blockStimProb[blockNumber-1]
                        stimSample = []
                        probCatch = self.blockProbCatch[blockNumber-1]

                    visStimFrames = 0
                    visStim.contrast = 0
                    visStim.ori = 0
                    visStim.phase = 0
                    soundType = ''
                    soundDur = 0
                    soundVolume = 0
                    soundFreq = [np.nan,np.nan]
                    soundAM = np.nan
                    soundArray = np.array([])
                    if blockTrialCount >= self.newBlockGoTrials and random.random() < probCatch:
                        self.trialStim.append('catch')
                    else:
                        if blockTrialCount < self.newBlockGoTrials:
                            self.trialStim.append(self.blockStimRewarded[blockNumber-1])
                        elif self.blockStimProb == 'even sampling':
                            if len(stimSample) < 1:
                                if self.evenSampleContrastVolume:
                                    for stim in blockStim:
                                        if 'vis' in stim and 'sound' in stim:
                                            stimSample += list(itertools.product([stim],self.visStimContrast,self.soundVolume))
                                        elif 'vis' in stim:
                                            stimSample += list(itertools.product([stim],self.visStimContrast,[0]))
                                        elif 'sound' in stim:
                                            stimSample += list(itertools.product([stim],[0],self.soundVolume))
                                    random.shuffle(stimSample)
                                else:
                                    stimSample = random.sample(blockStim*5,len(blockStim)*5)
                            if self.evenSampleContrastVolume:
                                stim,contrast,soundVolume = stimSample.pop(0)
                                visStim.contrast = contrast
                                self.trialStim.append(stim)
                            else:
                                self.trialStim.append(stimSample.pop(0))
                        else:
                            self.trialStim.append(np.random.choice(blockStim,p=stimProb))
                        stimNames = self.trialStim[-1].split('+')
                        visName = [stim for stim in stimNames if 'vis' in stim]
                        visName = visName[0] if len(visName) > 0 else None
                        soundName = [stim for stim in stimNames if 'sound' in stim]
                        soundName = soundName[0] if len(soundName) > 0 else None
                        if visName is not None:
                            visStimFrames = random.choice(self.visStimFrames)
                            if blockTrialCount < self.newBlockGoTrials:
                                visStim.contrast = max(self.visStimContrast)
                            elif not (self.blockStimProb == 'even sampling' and self.evenSampleContrastVolume):
                                visStim.contrast = random.choice(self.visStimContrast)
                            if self.visStimType == 'grating':
                                visStim.ori = random.choice(self.gratingOri[visName])
                                visStim.phase = random.choice(self.gratingPhase)
                        if soundName is not None:
                            soundType = self.soundType[soundName] if isinstance(self.soundType,dict) else self.soundType
                            if self.soundMode == 'internal':
                                soundDur = random.choice(self.soundDur)
                                if blockTrialCount < self.newBlockGoTrials:
                                    soundVolume = max(self.soundVolume)
                                elif not (self.blockStimProb == 'even sampling' and self.evenSampleContrastVolume):
                                    soundVolume = random.choice(self.soundVolume)
                                if soundType == 'tone':
                                    soundFreq = self.toneFreq[soundName]
                                elif soundType == 'linear sweep':
                                    soundFreq = self.linearSweepFreq[soundName]
                                elif soundType == 'log sweep':
                                    soundFreq = self.logSweepFreq[soundName]
                                elif soundType == 'noise':
                                    soundFreq = self.noiseFiltFreq[soundName]
                                elif soundType == 'AM noise':
                                    soundFreq = (2000,20000)
                                    soundAM = self.ampModFreq[soundName]
                                soundArray = self.makeSoundArray(soundType,soundDur,soundVolume,soundFreq,soundAM,self.soundRandomSeed)
                
                self.trialStartFrame.append(self._sessionFrame)
                self.trialBlock.append(blockNumber)
                self.trialVisStimFrames.append(visStimFrames)
                self.trialVisStimContrast.append(visStim.contrast)
                self.trialGratingOri.append(visStim.ori)
                self.trialGratingPhase.append(visStim.phase)
                self.trialSoundType.append(soundType)
                self.trialSoundDur.append(soundDur)
                self.trialSoundVolume.append(soundVolume)
                if soundType == 'tone':
                    self.trialSoundFreq.append([soundFreq,np.nan])
                else:
                    self.trialSoundFreq.append(soundFreq)
                self.trialSoundAM.append(soundAM)
                if self.saveSoundArray:
                    self.trialSoundArray.append(soundArray)
                
                if self.blockStimRewarded[blockNumber-1] in self.trialStim[-1]:
                    if blockAutoRewardCount < self.newBlockAutoRewards or missTrialCount == self.autoRewardMissTrials:
                        self.trialAutoRewarded.append(True)
                        autoRewardFrame = self.autoRewardOnsetFrame
                        blockAutoRewardCount += 1
                    else:
                        self.trialAutoRewarded.append(False)
                    rewardSize = self.solenoidOpenTime if self.trialAutoRewarded[-1] or random.random() < self.rewardProbGo else 0
                elif self.trialStim[-1] == 'catch' and random.random() < self.rewardProbCatch:
                    self.trialAutoRewarded.append(True)
                    autoRewardFrame = self.responseWindow[1]
                    rewardSize = self.solenoidOpenTime
                else:
                    self.trialAutoRewarded.append(False)
                    rewardSize = 0
                
                hasResponded = False
                rewardDelivered = False
                timeoutFrames = 0

            # extend pre stim gray frames if lick occurs during quiescent period
            if self._lick and self.trialPreStimFrames[-1] - self.quiescentFrames < self._trialFrame < self.trialPreStimFrames[-1]:
                self.quiescentViolationFrames.append(self._sessionFrame)
                self.trialPreStimFrames[-1] += randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
            
            # show/trigger stimulus
            if self._trialFrame == self.trialPreStimFrames[-1]:
                self.trialStimStartFrame.append(self._sessionFrame)
                if soundDur > 0:
                    if self.soundMode == 'internal':
                        self._sound = [soundArray]
            if (visStimFrames > 0
                and self.trialPreStimFrames[-1] <= self._trialFrame < self.trialPreStimFrames[-1] + visStimFrames):
                if self.gratingTF > 0:
                    visStim.phase = visStim.phase + self.gratingTF/self.frameRate
                visStim.draw()
            
            # trigger auto reward
            if self.trialAutoRewarded[-1] and not rewardDelivered and self._trialFrame == self.trialPreStimFrames[-1] + autoRewardFrame:
                self._reward = rewardSize
                if self.rewardSound is not None:
                    self.stopSound()
                    if self.soundMode == 'internal':
                        self._sound = [rewardSoundArray]
                self.trialRewarded.append(True)
                rewardDelivered = True
            
            # check for response within response window
            if (self._lick and not hasResponded 
                and self.trialPreStimFrames[-1] + self.responseWindow[0] <= self._trialFrame < self.trialPreStimFrames[-1] + self.responseWindow[1]):
                self.trialResponse.append(True)
                self.trialResponseFrame.append(self._sessionFrame)
                if self.trialStim[-1] != 'catch':
                    if rewardSize > 0:
                        if not rewardDelivered:
                            self._reward = rewardSize
                            if self.rewardSound is not None:
                                if self.soundMode == 'internal':
                                    self._sound = [rewardSoundArray]
                            self.trialRewarded.append(True)
                            rewardDelivered = True
                    else:
                        timeoutFrames = self.incorrectTimeoutFrames
                        if timeoutFrames > 0:
                            self._win.color = self.incorrectTimeoutColor
                        if self.incorrectSound is not None:
                            self.stopSound()
                            if self.soundMode == 'internal':
                                self._sound = [incorrectSoundArray]
                hasResponded = True  
                
            # end trial after response window plus any post response window frames and timeout
            if timeoutFrames > 0 and self._trialFrame == self.trialPreStimFrames[-1] + self.responseWindow[1] + timeoutFrames:
                self._win.color = self.monBackgroundColor
            
            if self._trialFrame == self.trialPreStimFrames[-1] + self.responseWindow[1] + self.postResponseWindowFrames + timeoutFrames:
                if not hasResponded:
                    self.trialResponse.append(False)
                    self.trialResponseFrame.append(np.nan)

                if rewardDelivered:
                    if self.trialStim[-1] != 'catch':
                        missTrialCount = 0
                else:
                    self.trialRewarded.append(False)
                    if rewardSize > 0 and self.trialStim[-1] != 'catch':
                        missTrialCount += 1
                    
                self.trialEndFrame.append(self._sessionFrame)
                self._trialFrame = -1
                blockTrialCount += 1
                
                if (rewardSize == 0 and self.trialStim[-1] != 'catch' and self.trialResponse[-1]  
                    and incorrectRepeatCount < self.incorrectTrialRepeats):
                    # repeat trial after response to unrewarded stimulus
                    incorrectRepeatCount += 1
                    self.trialRepeat.append(True)
                else:
                    incorrectRepeatCount = 0
                    self.trialRepeat.append(False)

                if len(self.trialStartFrame) == self.maxTrials:
                    self._continueSession = False
            
            blockFrameCount += 1
            if (blockNumber == len(self.blockStim) and 
                (blockTrialCount == blockTrials or (blockFrames is not None and blockFrameCount >= blockFrames))):
                self._continueSession = False

            self.showFrame()



def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))



if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    task = DynamicRouting1(params['rigName'],params['taskVersion'])
    task.start(params['subjectName'])