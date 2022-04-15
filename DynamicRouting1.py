# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
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
        
        # block stim is one list per block containing one or more 'vis#' or 'sound#'; first element rewarded
        # last block continues until end of session
        self.blockStim = [['vis1','vis2']] 
        self.blockStimProb = 'equal' # 'equal', 'sequential', or list of probabilities for each stimulus in each block adding to one
        self.blockProbCatch = [0.1] # fraction of trials for each block with no stimulus and no reward
        self.trialsPerBlock = None # None or sequence of trial numbers for each block; use this or framesPerBlock
        self.framesPerBlock = None # None or sequence of frame numbers for each block
        self.newBlockGoTrials = 3 # number of consecutive go trials at the start of each block (otherwise random)
        self.newBlockAutoRewards = 3 # number of autorewarded trials at the start of each block

        self.preStimFramesFixed = 90 # min frames between start of trial and stimulus onset
        self.preStimFramesVariableMean = 60 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 360 # max total preStim frames
        self.quiescentFrames = 90 # frames before stim onset during which licks delay stim onset
        self.responseWindow = [6,54]
        self.postResponseWindowFrames = 180

        self.autoRewardOnsetFrame = 6 # frames after stimulus onset at which autoreward occurs
        self.autoRewardMissTrials = 10 # None or consecutive miss trials after which autoreward delivered on next go trial
        
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
        self.gratingOri = {'vis1':[0],'vis2':[90]} # clockwise degrees from vertical
        self.gratingPhase = [0,0.5]
        self.gratingType = 'sqr' # 'sin' or sqr'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos
        
        # auditory stimulus params
        self.saveSoundArray = False
        self.soundType = 'tone' # 'tone', 'linear sweep', 'log sweep', 'noise', 'AM noise', or dict
        self.soundDur = [0.5] # seconds
        self.soundVolume = [0.1] # 0-1
        self.toneFreq = {'sound1':6000,'sound2':10000} # Hz
        self.linearSweepFreq = {'sound1':[6000,10000],'sound2':[10000,6000]}
        self.logSweepFreq = {'sound1':[3,2.5],'sound2':[3,3.5]} # log2(kHz)
        self.noiseFiltFreq = {'sound1':[4000,8000],'sound2':[8000,16000]} # Hz
        self.ampModFreq = {'sound1':20,'sound2':40} # Hz
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)

    
    def setDefaultParams(self,taskVersion):
        # dynamic routing task versions
        if taskVersion == 'stage 0':
            # auto rewards
            self.blockStim = [['vis1','vis2']]
            self.maxTrials = 150
            self.newBlockAutoRewards = 150
            self.quiescentFrames = 0
            self.blockProbCatch = [0]

        elif taskVersion in ('stage 1','stage 1 timeouts'):
            # ori discrim with or without timeouts
            self.blockStim = [['vis1','vis2']]
            self.incorrectTrialRepeats = 3
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('stage 2','stage 2 timeouts'):
            # tone discrim with or without timeouts
            self.soundType = 'tone'
            self.blockStim = [['sound1','sound2']]
            self.incorrectTrialRepeats = 3
            if 'timeouts' in taskVersion:
                self.incorrectSound = 'noise'
                self.incorrectTimeoutFrames = 180
                self.incorrectTimeoutColor = -1

        elif taskVersion in ('stage 3 ori','stage 3 tone'):
            # ori or tone discrim
            if 'ori' in taskVersion:
                self.blockStim = [['vis1','vis2']]
            else:
                self.soundType = 'tone'
                self.blockStim = [['sound1','sound2']]

        elif taskVersion in ('stage 4 ori tone','stage 4 tone ori'):
            # 2 blocks of all 4 stimuli, switch rewarded modality
            if 'ori tone' in taskVersion:
                self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']]
            else:
                self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']]
            self.framesPerBlock = np.array([30,30]) * 3600
            self.blockProbCatch = [0.1,0.1]

        elif taskVersion in ('stage 5 ori tone','stage 5 tone ori'):
            # 6 blocks
            if 'ori tone' in taskVersion:
                self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']] * 3
            else:
                self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']] * 3
            self.framesPerBlock = np.array([10] * 6) * 3600
            self.blockProbCatch = [0.1] * 6

        elif taskVersion == 'passive':
            self.blockStim = [['vis1','vis2'] + ['sound'+str(i+1) for i in range(10)]]
            self.blockStimProb = 'sequential'
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
            self.autoRewardMissTrials = 0
            self.saveSoundArray = True

        # templeton task versions
        elif 'templeton ori discrim' in taskVersion: 
            self.blockStim = [['vis1','vis2']]
            self.maxFrames = 60 * 3600
            self.visStimFrames = [30,60,90]
            self.soundDur = [0.5,1,1.5]
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.soundType = 'tone'
            self.spacebarRewardsEnabled = True
            if 'add sound' in taskVersion:
                self.blockStim = [['vis1','vis2','sound1','sound2']]
                self.preStimFramesFixed = 30 
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 10
                if '0' in taskVersion:
                    self.blockStimProb = [[0.4,0.4,0.1,0.1]]
                    #self.maxTrials = 600
                    self.newBlockAutoRewards = 5
                elif '1' in taskVersion:
                    self.postResponseWindowFrames = 120
                    #self.maxTrials = 900
                    self.newBlockAutoRewards = 5
                elif 'switch' in taskVersion:
                    self.visStimFrames = [60]
                    self.soundDur = [1]
                    self.postResponseWindowFrames = 120
                    #self.maxTrials = 900
                    self.newBlockAutoRewards = 25
                    self.newBlockGoTrials = 25
                    self.autoRewardMissTrials = 3
                else:
                    raise ValueError(taskVersion + ' is not a recognized task version')
                    
            else:
                if '0' in taskVersion:
                    self.visStimFrames = [60]
                    self.responseWindow = [6,60]
                    self.quiescentFrames = 60
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 100
                    self.newBlockGoTrials = 100
                    self.autoRewardMissTrials = 2
                elif '1' in taskVersion:
                    self.visStimFrames = [60]
                    self.responseWindow = [6,60]
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 10
                    self.newBlockGoTrials = 10
                    self.autoRewardMissTrials = 5
                    if '1a' in taskVersion:
                        self.autoRewardMissTrials = 2
                elif '2' in taskVersion:
                    self.incorrectTimeoutFrames = 420
                    self.preStimFramesFixed = 30 
                    self.preStimFramesVariableMean = 30 
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 5
                    self.newBlockGoTrials = 5
                    self.autoRewardMissTrials = 10
                elif '3' in taskVersion:
                    self.incorrectTimeoutFrames = 420
                    self.incorrectTrialRepeats = 3
                    self.preStimFramesFixed = 30 
                    self.preStimFramesVariableMean = 30 
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 5
                    self.newBlockGoTrials = 5
                    self.autoRewardMissTrials = 10
                else:
                    raise ValueError(taskVersion + ' is not a recognized task version')
                
            if 'AVTO' in taskVersion:
                self.incorrectTimeoutFrames = 300 
                self.incorrectTimeoutColor = -1
                self.incorrectSound = 'noise'
                self.incorrectSoundDur = 5
        
        elif 'templeton tone discrim' in taskVersion: 
            self.blockStim = [['sound1','sound2']]
            self.soundDur = [0.5,1,1.5]
            self.maxFrames = 60 * 3600
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.soundType = 'tone'
            self.spacebarRewardsEnabled = True
            if 'add vis' in taskVersion:
                self.blockStim = [['sound1','sound2','vis1','vis2']]
                self.preStimFramesFixed = 30 
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 10
                if '0' in taskVersion:
                    self.blockStimProb = [[0.4,0.4,0.1,0.1]]
                    #self.maxTrials = 600
                    self.newBlockAutoRewards = 5
                elif '1' in taskVersion:
                    self.postResponseWindowFrames = 120
                    #self.maxTrials = 900
                    self.newBlockAutoRewards = 5
                elif 'switch' in taskVersion:
                    self.visStimFrames = [60]
                    self.soundDur = [1]
                    self.postResponseWindowFrames = 120
                    #self.maxTrials = 900
                    self.newBlockAutoRewards = 25
                    self.newBlockGoTrials = 25
                    self.autoRewardMissTrials = 3
                else:
                    raise ValueError(taskVersion + ' is not a recognized task version')

            else:
                if '0' in taskVersion:
                    self.soundDur = [1.0]
                    self.responseWindow = [6,60]
                    self.quiescentFrames = 60
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 100
                    self.newBlockGoTrials = 100
                    self.autoRewardMissTrials = 2
                elif '1' in taskVersion:
                    self.soundDur = [1.0]
                    self.responseWindow = [6,60]
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 10
                    self.newBlockGoTrials = 10
                    self.autoRewardMissTrials = 5
                elif '2' in taskVersion:
                    self.incorrectTimeoutFrames = 420
                    self.preStimFramesFixed = 30 
                    self.preStimFramesVariableMean = 30 
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 5
                    self.newBlockGoTrials = 5
                    self.autoRewardMissTrials = 10
                elif '3' in taskVersion:
                    self.incorrectTimeoutFrames = 420
                    self.incorrectTrialRepeats = 3
                    self.preStimFramesFixed = 30 
                    self.preStimFramesVariableMean = 30 
                    #self.maxTrials = 450
                    self.newBlockAutoRewards = 5
                    self.newBlockGoTrials = 5
                    self.autoRewardMissTrials = 10
                else:
                    raise ValueError(taskVersion + ' is not a recognized task version')
                
            if 'AVTO' in taskVersion:
                self.incorrectTimeoutFrames = 300 
                self.incorrectTimeoutColor = -1
                self.incorrectSound = 'noise'
                self.incorrectSoundDur = 5

        elif 'templeton sweep discrim' in taskVersion: 
            self.blockStim = [['sound1','sound2']]
            self.soundDur = [0.25]
            self.maxFrames = 60 * 3600
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.soundType = 'log sweep'
            self.spacebarRewardsEnabled = True

            if '0' in taskVersion:
                self.newBlockAutoRewards = 200
                self.newBlockGoTrials = 100
                self.autoRewardMissTrials = 2
                self.maxFrames = 30 * 3600
            elif '1' in taskVersion:
                self.newBlockAutoRewards = 10
                self.newBlockGoTrials = 10
                self.autoRewardMissTrials = 5
            elif '2' in taskVersion:
                self.incorrectTimeoutFrames = 180 
                self.incorrectTimeoutColor = -1
                # self.incorrectSound = 'noise'
                # self.incorrectSoundDur = 3 
                self.newBlockAutoRewards = 5
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 10
            elif '3' in taskVersion:
                self.incorrectTimeoutFrames = 300
                self.incorrectTimeoutColor = -1
                # self.incorrectSound = 'noise'
                # self.incorrectSoundDur = 5
                self.incorrectTrialRepeats = 3  
                self.newBlockAutoRewards = 5
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 10
            else:
                raise ValueError(taskVersion + ' is not a recognized task version')
            
            if 'AVTO' in taskVersion:
                self.incorrectTimeoutFrames = 300 
                self.incorrectTimeoutColor = -1
                self.incorrectSound = 'noise'
                self.incorrectSoundDur = 5

        elif ('templeton switch vis aud' in taskVersion)|('templeton switch aud vis' in taskVersion):
            self.soundType = 'tone'
            self.visStimFrames = [60]
            self.soundDur = [1]
            self.responseWindow = [6,60]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1,0.1]
            self.spacebarRewardsEnabled = True

            self.maxFrames = None
            self.framesPerBlock = np.array([30,30]) * 3600
            
            self.preStimFramesFixed = 30 
            self.preStimFramesVariableMean = 30 
            self.preStimFramesMax = 240
            self.postResponseWindowFrames = 120

            self.newBlockGoTrials = 10
            self.autoRewardMissTrials = 5
            self.newBlockAutoRewards = 10

            if 'trl rpt' in taskVersion:
                self.incorrectTrialRepeats = 3

            if 'TO' in taskVersion:
                self.incorrectTimeoutFrames = 180 
                self.incorrectTimeoutColor = -1
                self.incorrectSound = 'noise'
                self.incorrectSoundDur = 3

            if 'test' in taskVersion:
                self.framesPerBlock = np.array([2,2]) * 3600
                self.newBlockGoTrials = 2
                self.autoRewardMissTrials = 10
                self.newBlockAutoRewards = 2

            if '4x' in taskVersion:
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 5
                self.newBlockAutoRewards = 5
                self.blockProbCatch = [0.1,0.1,0.1,0.1]
                self.framesPerBlock = np.array([15,15,15,15]) * 3600
                if 'test' in taskVersion:
                    self.newBlockGoTrials = 2
                    self.newBlockAutoRewards = 2
                    self.framesPerBlock = np.array([2,2,2,2]) * 3600
                if 'vis aud' in taskVersion:
                    self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']]*2
                elif 'aud vis' in taskVersion:
                    self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']]*2
            elif '6x' in taskVersion:
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 5
                self.newBlockAutoRewards = 5
                self.blockProbCatch = [0.1,0.1,0.1,0.1,0.1,0.1]
                self.framesPerBlock = np.array([10,10,10,10,10,10]) * 3600
                if 'test' in taskVersion:
                    self.newBlockGoTrials = 2
                    self.newBlockAutoRewards = 2
                    self.framesPerBlock = np.array([1,1,1,1,1,1]) * 3600
                if 'vis aud' in taskVersion:
                    self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']]*3
                elif 'aud vis' in taskVersion:
                    self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']]*3
            elif '12x' in taskVersion:
                self.newBlockGoTrials = 5
                self.autoRewardMissTrials = 5
                self.newBlockAutoRewards = 5
                self.blockProbCatch = [0.1]*12
                self.framesPerBlock = np.array([5]*12) * 3600
                if 'test' in taskVersion:
                    self.newBlockGoTrials = 2
                    self.newBlockAutoRewards = 2
                    self.framesPerBlock = np.array([1]*12) * 3600
                if 'vis aud' in taskVersion:
                    self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']]*6
                elif 'aud vis' in taskVersion:
                    self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']]*6
            else:
                if 'vis aud' in taskVersion:
                    self.blockStim = [['vis1','vis2','sound1','sound2'],['sound1','sound2','vis1','vis2']]
                elif 'aud vis' in taskVersion: 
                    self.blockStim = [['sound1','sound2','vis1','vis2'],['vis1','vis2','sound1','sound2']]

            if '3autos' in taskVersion:
                self.newBlockGoTrials = 3
                self.newBlockAutoRewards = 3
            if '2autos' in taskVersion:
                self.newBlockGoTrials = 2
                self.newBlockAutoRewards = 2
            if '0autos' in taskVersion:
                self.newBlockGoTrials = 0
                self.newBlockAutoRewards = 0
                self.autoRewardMissTrials = None
        
        elif 'hearing check' in taskVersion:
            self.blockStim = [['sound1','sound2']]
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

        if 'maxvol' in taskVersion:
            self.soundVolume = [1.0]
    
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
        self.blockStimRewarded = [] # stimulus that is rewarded each block
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
                        stimProb = None if self.blockStimProb in ('equal','sequential') else self.blockStimProb[blockNumber-1]
                        stimIndex = []
                        probCatch = self.blockProbCatch[blockNumber-1]
                        self.blockStimRewarded.append(blockStim[0])

                    visStimFrames = 0
                    visStim.contrast = 0
                    visStim.ori = 0
                    visStim.phase = 0
                    soundType = ''
                    soundDur = np.nan
                    soundVolume = np.nan
                    soundFreq = [np.nan,np.nan]
                    soundAM = np.nan
                    soundArray = np.array([])
                    if random.random() < probCatch:
                        self.trialStim.append('catch')
                    else:
                        if blockTrialCount < self.newBlockGoTrials:
                            self.trialStim.append(blockStim[0])
                        elif self.blockStimProb == 'sequential':
                            if len(stimIndex) < 1:
                                stimIndex = random.sample(range(len(blockStim)),len(blockStim))
                            self.trialStim.append(blockStim[stimIndex.pop(0)])
                        else:
                            self.trialStim.append(np.random.choice(blockStim,p=stimProb))
                        if 'vis' in self.trialStim[-1]:
                            visStimFrames = random.choice(self.visStimFrames)
                            visStim.contrast = random.choice(self.visStimContrast)
                            if self.visStimType == 'grating':
                                visStim.ori = random.choice(self.gratingOri[self.trialStim[-1]])
                                visStim.phase = random.choice(self.gratingPhase)
                        else:
                            soundType = self.soundType[self.trialStim[-1]] if isinstance(self.soundType,dict) else self.soundType
                            if self.soundMode == 'internal':
                                soundDur = random.choice(self.soundDur)
                                soundVolume = random.choice(self.soundVolume)
                                if soundType == 'tone':
                                    soundFreq = self.toneFreq[self.trialStim[-1]]
                                elif soundType == 'linear sweep':
                                    soundFreq = self.linearSweepFreq[self.trialStim[-1]]
                                elif soundType == 'log sweep':
                                    soundFreq = self.logSweepFreq[self.trialStim[-1]]
                                elif soundType == 'noise':
                                    soundFreq = self.noiseFiltFreq[self.trialStim[-1]]
                                elif soundType == 'AM noise':
                                    soundFreq = (2000,20000)
                                    soundAM = self.ampModFreq[self.trialStim[-1]]
                                soundArray = self.makeSoundArray(soundType,soundDur,soundVolume,soundFreq,soundAM)
                
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
                
                if self.trialStim[-1] == self.blockStimRewarded[-1]:
                    if blockAutoRewardCount < self.newBlockAutoRewards or missTrialCount == self.autoRewardMissTrials:
                        self.trialAutoRewarded.append(True)
                        blockAutoRewardCount += 1
                    else:
                        self.trialAutoRewarded.append(False)
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
                if 'sound' in self.trialStim[-1]:
                    if self.soundMode == 'internal':
                        self._sound = [soundArray]
            if (visStimFrames > 0
                and self.trialPreStimFrames[-1] <= self._trialFrame < self.trialPreStimFrames[-1] + visStimFrames):
                visStim.draw()
            
            # trigger auto reward
            if self.trialAutoRewarded[-1] and not hasResponded and self._trialFrame == self.trialPreStimFrames[-1] + self.autoRewardOnsetFrame:
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
                if rewardSize > 0:
                    if not rewardDelivered:
                        self._reward = rewardSize
                        if self.rewardSound is not None:
                            if self.soundMode == 'internal':
                                self._sound = [rewardSoundArray]
                        self.trialRewarded.append(True)
                        rewardDelivered = True
                elif self.trialStim[-1] != 'catch':
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
                    if rewardSize > 0:
                        missTrialCount += 1

                if rewardDelivered:
                    missTrialCount = 0
                else:
                    self.trialRewarded.append(False)
                    
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
            
            self.showFrame()
            blockFrameCount += 1
            if (blockNumber == len(self.blockStim) and 
                (blockTrialCount == blockTrials or (blockFrames is not None and blockFrameCount >= blockFrames))):
                self._continueSession = False



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