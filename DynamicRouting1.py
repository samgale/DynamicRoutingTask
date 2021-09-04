# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import random
from psychopy import visual    
from TaskControl import TaskControl


class DynamicRouting1(TaskControl):
    
    def __init__(self,rigName):
        TaskControl.__init__(self,rigName)
        
        # block stim is one list per block containing 1 or 2 of 'vis#' or 'sound#'
        # first element rewarded
        self.blockStim = [['vis1','vis2'],['vis2','vis1']]
        self.trialsPerBlock = [1,1] # min and max trials per block
        self.probCatch = 0 # fraction of trials with no stimulus and no reward
        
        self.preStimFramesFixed = 360 # min frames between end of previous trial and stimulus onset
        self.preStimFramesVariableMean = 120 # mean of additional preStim frames drawn from exponential distribution
        self.preStimFramesMax = 600 # max total preStim frames
        
        self.rewardWindow = [9,45]
        
        self.useIncorrectSound = False # play sound when trial is incorrect
        self.incorrectTrialRepeats = 0 # maximum number of incorrect trial repeats
        self.incorrectTimeoutFrames = 0 # extended gray screen following incorrect trial
        
        # visual stimulus params
        # parameters that can vary across trials are lists
        self.gratingFrames = 3 # duration of target stimulus
        self.gratingContrast = [0.25,1]
        self.gratingSize = 25 # degrees
        self.gratingSF = 0.08 # cycles/deg
        self.gratingOri = {'vis1':0,'vis2':90} # clockwise degrees from vertical
        self.gratingType = 'sqr' # 'sqr' or 'sin'
        self.gratingEdge= 'raisedCos' # 'circle' or 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08 # only applies to raisedCos

    
    def setDefaultParams(self,name,taskVersion=None):
        if name == 'training0':
            # stim moves to reward automatically; wheel movement ignored
            self.moveStim = True
            self.normAutoMoveRate = 0.5
            self.maxResponseWaitFrames = 3600
            self.blockProbGoRight = [0.5]
            self.rewardBothDirs = True
            self.postRewardTargetFrames = 60
            
        elif name == 'training1':
            # learn to associate wheel movement with stimulus movement and reward
            # either diretion rewarded
            self.setDefaultParams('training0',taskVersion)
            self.normAutoMoveRate = 0
            
        elif name == 'training2':
            # one side rewarded
            # introduce quiescent period, shorter response window, incorrect repeats, and catch trials
            self.setDefaultParams('training1',taskVersion)
            self.rewardBothDirs = False 
            self.quiescentFrames = 60
            self.maxResponseWaitFrames = 1200 # adjust this 
            self.useIncorrectNoise = True
            self.incorrectTimeoutFrames = 360
            self.incorrectTrialRepeats = 5 # will repeat for unanswered trials
            self.probCatch = 0.1
            
        elif name == 'training3':
            # introduce block structure
            self.setDefaultParams('training2',taskVersion)
            self.trialsPerBlock = [3,8]
            self.blockProbGoRight = [0,1]
            
        else:
            print(str(name)+' is not a recognized set of default parameters')
    
     
    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create visual stimulus
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
        self.trialGratingContrast = []
        self.trialGratingOri = []
        self.trialResponse = []
        self.trialResponseFrame = []
        self.trialRewarded = []
        self.trialRepeat = [False]
        self.trialBlock = []
        self.trialBlockStimRewarded = []
        blockTrials = None # number of trials of current block
        blockTrialCount = None # number of trials completed in current block
        incorrectRepeatCount = 0
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # if starting a new trial
            if self._trialFrame == 0:
                preStimFrames = randomExponential(self.preStimFramesFixed,self.preStimFramesVariableMean,self.preStimFramesMax)
                self.trialPreStimFrames.append(preStimFrames)
                
                if not self.trialRepeat[-1]:
                    if blockTrials is not None and random.random() < self.probCatch:
                        self.trialBlockStimRewarded.append('none')
                        self.trialStim.append('catch')
                        visStim.ontrast = 0
                    else:
                        if blockTrials is None or blockTrialCount == blockTrials:
                            blockTrials = random.randint(*self.trialsPerBlock)
                            blockTrialCount = 1
                            if len(self.trialBlock) < 1:
                                self.trialBlock.append(0)
                            else:
                                self.trialBlock.append(self.trialBlock[-1] + 1)
                            blockStim = self.blockStim[self.trialBlock[-1] % len(self.blockStim)]
                        else:
                            blockTrialCount += 1
                        
                        self.trialBlockStimRewarded.append(blockStim[0])
                        self.trialStim.append(random.choice(blockStim))
                        if 'vis' in self.trialStim[-1]:
                            visStim.contrast = random.choice(self.gratingContrast)
                            visStim.ori = self.gratingOri[self.trialStim[-1]]
                        else:
                            visStim.contrast = 0

                self.trialStartFrame.append(self._sessionFrame)
                self.trialGratingContrast.append(visStim.contrast)
                self.trialGratingOri.append(visStim.ori)
                if blockTrialCount > 1:
                    self.trialBlock.append(self.trialBlock[-1])
                rewardSize = self.solenoidOpenTime if self.trialStim[-1]==self.trialBlockStimRewarded[-1] else 0
                hasResponded = False
            
            # if gray screen period is complete but before response
            if not hasResponded and self._trialFrame >= self.trialPreStimFrames[-1]:
                if self._trialFrame == self.trialPreStimFrames[-1]:
                    self.trialStimStartFrame.append(self._sessionFrame)
                    if 'sound' in self.trialStim[-1]:
                        setattr(self,'_'+self.trialStim[-1],True)
                if self._trialFrame < self.trialPreStimFrames[-1] + self.gratingFrames:
                    if visStim.contrast > 0:
                        visStim.draw()
                
                if (self._lick and 
                    self.trialPreStimFrames[-1] + self.rewardWindow[0] <= self._trialFrame < self.trialPreStimFrames[-1] + self.rewardWindow[1]):
                    
                    self.trialResponse.append(True)
                    self.trialResponseFrame.append(self._sessionFrame)
                    if self.trialStim[-1] != 'catch':
                        if rewardSize > 0:
                            self.trialRewarded.append(True)
                            self._reward = rewardSize
                        else:
                            self.trialRewarded.append(False)
                            if self.useIncorrectSound:
                                pass
                    hasResponded = True
                
            # end trial
            if ((hasResponded and self.trialStim[-1] != 'catch') or
                self._trialFrame == self.trialPreStimFrames[-1] + self.rewardWindow[1]):
                
                if not hasResponded:
                    self.trialResponse.append(False)
                    self.trialResponseFrame.append(self._sessionFrame)
                    self.trialRewarded.append(False)
                elif self.trialStim[-1] == 'catch':
                    self.trialRewarded.append(False)
                    
                self.trialEndFrame.append(self._sessionFrame)
                self._trialFrame = -1
                
                if (self.trialStim[-1] != 'catch' and not self.trialRewarded[-1] and 
                    incorrectRepeatCount < self.incorrectTrialRepeats):
                    
                    incorrectRepeatCount += 1
                    self.trialRepeat.append(True)
                else:
                    incorrectRepeatCount = 0
                    self.trialRepeat.append(False)
            
            self.showFrame()


def randomExponential(fixed,variableMean,maxTotal):
    val = fixed + random.expovariate(1/variableMean) if variableMean > 1 else fixed + variableMean
    return int(min(val,maxTotal))


if __name__ == "__main__":
    pass