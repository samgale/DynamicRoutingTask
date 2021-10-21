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
        self.blockStim = [['vis1']]
        self.blockStimProb = 'equal' # 'equal' or list of probabilities for each stimulus in each block adding to one
        self.trialsPerBlock = None # None or [min,max] trials per block; use this or framesPerBlock
        self.framesPerBlock = None # None or [min,max] frames per block

        self.blockProbCatch = [0.15] # fraction of trials for each block with no stimulus and no reward

        self.newBlockAutoRewards = 5 # number of autorewarded trials at the start of each block
        self.autoRewardMissTrials = 10 # None or consecutive miss trials after which autoreward delivered on next go trial
        self.autoRewardOnsetFrame = 9 # frames after stimulus onset at which autoreward occurs
        
        self.preStimFramesFixed = 90 # min frames between start of trial and stimulus onset
        self.preStimFramesVariableMean = 60 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 360 # max total preStim frames
        self.quiescentFrames = 90 # frames before stim onset during which licks delay stim onset
        self.responseWindow = [9,54]
        self.postResponseWindowFrames = 180
        
        self.incorrectTrialRepeats = 0 # maximum number of incorrect trial repeats
        self.incorrectTimeoutFrames = 0 # extended gray screen following incorrect trial
        self.incorrectNoiseDur = 0 # duation in secons of noise playback after incorrect trial
        
        # visual stimulus params
        # parameters that can vary across trials are lists
        self.visStimType = 'grating'
        self.visStimFrames = [15] # duration of visual stimulus
        self.visStimContrast = [1]
        self.gratingSize = 50 # degrees
        self.gratingSF = 0.04 # cycles/deg
        self.gratingOri = {'vis1':0,'vis2':90} # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sin' or sqr'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos
        
        # auditory stimulus params
        self.soundType = None # 'tone' or 'noise'
        self.soundVolume = 0.1 # 0-1
        self.soundDur = [0.25] # seconds
        self.toneFreq = {'sound1':6000,'sound2':10000} # Hz
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)

    
    def setDefaultParams(self,taskVersion):
        # dynamic routing task versions
        if taskVersion[:-2] == 'vis detect':
            self.blockStim = [['vis1']]
            if taskVersion[-1] == '0':
                self.maxTrials = 150
                self.newBlockAutoRewards = 150
                self.quiescentFrames = 0
                self.blockProbCatch = [0]
            elif taskVersion[-1] == '1':
                self.blockProbCatch = [0.2]
            elif taskVersion[-1] == '2':
                self.blockProbCatch = [0.5]

        elif taskVersion == 'vis detect switch to sound':
            self.blockStim = [['vis1'],['sound1','vis1']]
            self.soundType = 'tone'
            self.framesPerBlock = [10 * 3600] * 2
            self.newBlockAutoRewards = 10
            self.blockProbCatch = [0.5,0]

        elif taskVersion == 'sound detect with vis':
            self.blockStim = [['sound1','vis1']]
            self.soundType = 'tone'
            self.blockProbCatch = [0]
            
        elif taskVersion == 'sound detect switch to vis':
            self.blockStim = [['sound1','vis1'],['vis1']]
            self.soundType = 'tone'
            self.framesPerBlock = [10 * 3600] * 2
            self.newBlockAutoRewards = 10
            self.blockProbCatch = [0,0.5]
            
        elif taskVersion[:-2] == 'ori discrim':
            if taskVersion[-1] == '0':
                self.blockStim = [['vis1','vis2']]
                self.maxTrials = 150
                self.newBlockAutoRewards = 150
                self.quiescentFrames = 0
                self.blockProbCatch = [0]
            else:
                if taskVersion[-1] == '1':
                    self.blockStim = [['vis1','vis2']]
                elif taskVersion[-1] == '2':
                    self.blockStim = [['vis2','vis1']]
                    self.newBlockAutoRewards = 10
                    self.autoRewardMissTrials = 5
                self.blockProbCatch = [0.15]
                
        elif taskVersion == 'ori discrim switch':
            self.blockStim = [['vis1','vis2'],['vis2','vis1']]
            self.framesPerBlock = [15 *3600] * 2
            self.newBlockAutoRewards = 10
            self.blockProbCatch = [0.15]

        elif taskVersion == 'sound test':
            self.blockStim = [['sound1']]
            self.microphoneCh = 2
            self.soundLibrary = 'psychtoolbox'
            self.soundType = 'noise'
            self.soundHanningDur = 0
            self.blockProbCatch = [0]
            self.newBlockAutoRewards = 0
            self.autoRewardMissTrials = None
            self.maxTrials = 10

        elif taskVersion == 'test':
            self.blockStim = [['vis1','sound1']]
            self.blockStimProb = [[0.2,0.8]]
            self.blockProbCatch = [0]
            self.soundType = 'tone'
            self.newBlockAutoRewards = 0
            self.autoRewardMissTrials = None
            self.incorrectTimeoutFrames = 600

        # templeton task versions
        elif 'templeton ori discrim' in taskVersion: 
            self.blockStim = [['vis1','vis2']]
            self.visStimFrames = [30,60,90]
            self.responseWindow = [9,90]
            self.quiescentFrames = 90
            self.blockProbCatch = [0.1]
            self.spacebarRewardsEnabled = True
            if 'test' in taskVersion:
                self.quiescentFrames = 0
                self.maxTrials = 10
                self.newBlockAutoRewards = 10
            elif 'detect 0' in taskVersion:
                self.blockStim = [['vis1']]
                self.visStimFrames = [90]
                self.quiescentFrames = 0
                self.maxTrials = 200
                self.newBlockAutoRewards = 200
            elif '0' in taskVersion:
                self.visStimFrames = [90]
                self.quiescentFrames = 60
                self.maxTrials = 400
                self.newBlockAutoRewards = 400
            elif '1' in taskVersion:
                self.maxFrames = 70 * 3600
                self.visStimFrames = [90]
                self.maxTrials = 400
                self.newBlockAutoRewards = 10
                self.autoRewardMissTrials = 5
            elif '2' in taskVersion:
                self.maxFrames = 80 * 3600
                self.incorrectTimeoutFrames = 400
                self.preStimFramesFixed = 30 
                self.preStimFramesVariableMean = 30 
                self.maxTrials = 450
                self.newBlockAutoRewards = 5
                self.autoRewardMissTrials = 10
            elif 'add sound 1' in taskVersion:
                self.maxFrames = 80 * 3600
                self.blockStim = [['vis1','vis2','sound1','sound2']]
                self.blockStimProb = [[0.4,0.4,0.1,0.1]]
                self.responseWindow = [9,90]
                self.preStimFramesFixed = 30 
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240 
                self.postResponseWindowFrames = 120
                self.maxTrials = 600
                self.newBlockAutoRewards = 5
            elif 'add sound 2' in taskVersion:
                self.maxFrames = 80 * 3600
                self.blockStim = [['vis1','vis2','sound1','sound2']]
                self.responseWindow = [9,90]
                self.preStimFramesFixed = 30 
                self.preStimFramesVariableMean = 30 
                self.preStimFramesMax = 240 
                self.postResponseWindowFrames = 120
                self.maxTrials = 600
                self.newBlockAutoRewards = 5

        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
    
    
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
        
        # things to keep track of
        self.trialStartFrame = []
        self.trialEndFrame = []
        self.trialPreStimFrames = []
        self.trialStimStartFrame = []
        self.trialStim = []
        self.trialVisStimFrames = []
        self.trialVisStimContrast = []
        self.trialGratingOri = []
        self.trialSoundDur = []
        self.trialToneFreq = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.trialRewarded = []
        self.trialAutoRewarded = []
        self.quiescentViolationFrames = [] # frames where quiescent period was violated
        self.trialRepeat = [False]
        self.trialBlock = []
        self.blockStimRewarded = [] # stimulus that is rewarded each block
        blockNumber = 0 # current block
        blockTrials = 0 # total number of trials to occur in current block
        blockFrames = 0 # total number of frames to occur in current block
        blockTrialCount = 0 # number of trials completed in current block
        blockFrameCount = 0 # number of frames completed in current block
        blockAutoRewardCount = 0
        missTrialCount = 0
        incorrectRepeatCount = 0
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames) # can grow larger than preStimFrames during quiescent period
                
                if self.trialRepeat[-1]:
                    self.trialStim.append(self.trialStim[-1])
                else:
                    if (blockNumber == 0 or 
                        (blockNumber < len(self.blockStim) and (blockTrialCount == blockTrials or blockFrameCount >= blockFrames))):
                        # start new block of trials
                        blockNumber += 1
                        blockTrials = None if self.trialsPerBlock is None else random.randint(*self.trialsPerBlock)
                        blockFrames = None if self.framesPerBlock is None else random.randint(*self.framesPerBlock)
                        blockTrialCount = 0
                        blockFrameCount = 0
                        blockAutoRewardCount = 0
                        blockIndex = (blockNumber-1) % len(self.blockStim)
                        blockStim = self.blockStim[blockIndex]
                        stimProb = None if self.blockStimProb == 'equal' else self.blockStimProb[blockIndex]
                        probCatch = self.blockProbCatch[blockIndex]
                        self.blockStimRewarded.append(blockStim[0])
                    if random.random() < probCatch:
                        self.trialStim.append('catch')
                        visStimFrames = 0
                        visStim.contrast = 0
                        soundDur = 0
                        toneFreq = np.nan
                    else:
                        self.trialStim.append(np.random.choice(blockStim,p=stimProb))
                        if 'vis' in self.trialStim[-1]:
                            visStimFrames = random.choice(self.visStimFrames)
                            visStim.contrast = random.choice(self.visStimContrast)
                            if self.visStimType == 'grating':
                                visStim.ori = self.gratingOri[self.trialStim[-1]]
                            soundDur = 0
                            toneFreq = np.nan
                        else:
                            visStimFrames = 0
                            visStim.contrast = 0
                            soundDur = random.choice(self.soundDur)
                            if self.soundMode == 'internal':
                                if self.soundType == 'tone':
                                    toneFreq = self.toneFreq[self.trialStim[-1]]
                                    soundArray = self.makeSoundArray('tone',soundDur,self.soundVolume,toneFreq)
                                else:
                                    soundArray = self.makeSoundArray('noise',soundDur)
                
                self.trialStartFrame.append(self._sessionFrame)
                self.trialBlock.append(blockNumber)
                self.trialVisStimFrames.append(visStimFrames)
                self.trialVisStimContrast.append(visStim.contrast)
                if self.visStimType == 'grating':
                    self.trialGratingOri.append(visStim.ori)
                self.trialSoundDur.append(soundDur)
                if self.soundType == 'tone':
                    self.trialToneFreq.append(toneFreq)
                
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
                if soundDur > 0:
                    if self.soundMode == 'external':
                        self._sound = self.trialStim[-1]
                    else:
                        self._sound = [soundArray]
            if (visStimFrames > 0
                and self.trialPreStimFrames[-1] <= self._trialFrame < self.trialPreStimFrames[-1] + visStimFrames):
                visStim.draw()
            
            # trigger auto reward
            if self.trialAutoRewarded[-1] and not hasResponded and self._trialFrame == self.trialPreStimFrames[-1] + self.autoRewardOnsetFrame:
                self._reward = rewardSize
                self.trialRewarded.append(True)
                rewardDelivered = True
            
            # check for response within response window
            if (self._lick and not hasResponded 
                and self.trialPreStimFrames[-1] + self.responseWindow[0] <= self._trialFrame < self.trialPreStimFrames[-1] + self.responseWindow[1]):
                self.trialResponse.append(True)
                self.trialResponseFrame.append(self._sessionFrame)
                if rewardSize > 0:
                    if not rewardDelivered:
                        self.trialRewarded.append(True)
                        self._reward = rewardSize
                        rewardDelivered = True
                elif self.trialStim[-1] != 'catch':
                    timeoutFrames = self.incorrectTimeoutFrames
                    if self.incorrectNoiseDur > 0: 
                        self._sound = [self.makeSoundArray('noise',self.incorrectNoiseDur)]
                hasResponded = True
                
            # end trial after response window plus any post response window frames and timeout
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
                
                if (self.trialStim[-1] != 'catch' and not self.trialRewarded[-1]  
                    and incorrectRepeatCount < self.incorrectTrialRepeats):
                    incorrectRepeatCount += 1
                    self.trialRepeat.append(True)
                else:
                    incorrectRepeatCount = 0
                    self.trialRepeat.append(False)

                if len(self.trialStartFrame) == self.maxTrials:
                    self._continueSession = False
            
            self.showFrame()
            blockFrameCount += 1



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