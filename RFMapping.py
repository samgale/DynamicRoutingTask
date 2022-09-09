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


class RFMapping(TaskControl):
    
    def __init__(self,rigName,taskVersion=None):
        TaskControl.__init__(self,rigName)
        
        self.maxFrames = None
        self.maxBlocks = 6
        self.stimFrames = 15
        self.interStimFrames = 15
        
        # visual stimulus params
        # parameters that can vary across trials are lists
        self.warp = 'spherical'
        self.visStimType = 'grating'
        self.visStimContrast = 1
        self.visStimSize = 20 # degrees
        self.gratingSF = 0.08 # cycles/deg
        self.gratingTF = 4 # cycles/s
        self.gratingOri = np.arange(0,360,45)
        self.gratingEdge= 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08
        
        # auditory stimulus params
        self.soundType = 'tone'
        self.soundDur = 0.25 # seconds
        self.soundVolume = 1 if rigName=='NP3' else 0.1
        self.toneFreq = np.arange(4000,16001,1000) # Hz
        self.saveSoundArray = True
        self.soundRandomSeed = 0
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)
    
    
    def setDefaultParams(self,taskVersion):
        if True:
            pass
        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
            
            
    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create visual stimulus
        visStimSizePix = self.visStimSize * self.pixelsPerDeg
        visStim = visual.GratingStim(win=self._win,
                                     units='pix',
                                     tex='sin',
                                     mask=self.gratingEdge,
                                     maskParams={'fringeWidth':self.gratingEdgeBlurWidth},
                                     size=visStimSizePix, 
                                     sf=self.gratingSF/self.pixelsPerDeg,
                                     contrast=self.visStimContrast)
            
        # calculate vis stim grid positions
        self.gridX,self.gridY = [np.linspace(-s/2 + visStimSizePix/2, s/2 - visStimSizePix/2, int(np.ceil(s/visStimSizePix))) for s in self.monSizePix]
          
        # make list of stimulus parameters for each trial
        n = len(self.gratingOri)//2 # non-vis trial multiplier
        trialParams =  n * [(np.nan,)*4] # no stim trials
        trialParams += n * list(itertools.product([np.nan],[np.nan],[np.nan],self.toneFreq))
        trialParams += list(itertools.product(self.gridX,self.gridY,self.gratingOri,[np.nan]))
        
        # things to keep track of
        self.stimStartFrame = []
        self.trialVisXY = []
        self.trialGratingOri = []
        self.trialSoundFreq = []
        self.trialSoundArray = []
        block = -1 # index of current block
        blockTrial = 0 # index of current trial in block
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getInputData()
            
            # if starting new trial
            if self._trialFrame == 0:
                
                if block < 0 or blockTrial == len(trialParams):
                    block += 1
                    if self.maxBlocks is not None and block == self.maxBlocks:
                        break # end session
                    else:
                        # start new block of trials
                        blockTrial = 0
                        random.shuffle(trialParams)
                
                x,y,ori,freq = trialParams[blockTrial]
                pos = (x,y)
                
                if not np.isnan(ori):
                    visStim.pos = pos
                    visStim.ori = ori
                    visStim.phase = 0
                
                if np.isnan(freq):
                    soundArray = np.array([])
                else:
                    soundArray = self.makeSoundArray(self.soundType,self.soundDur,self.soundVolume,freq,None,self.soundRandomSeed)
                
                self.stimStartFrame.append(self._sessionFrame)
                self.trialVisXY.append(pos)
                self.trialGratingOri.append(ori)
                self.trialSoundFreq.append(freq)
                if self.saveSoundArray:
                    self.trialSoundArray.append(soundArray)

            # show/trigger stimulus
            if self._trialFrame == 0 and soundArray.size > 0:
                if self.soundMode == 'internal':
                    self._sound = [soundArray]
            if self._trialFrame < self.stimFrames and not np.isnan(ori):
                visStim.phase = visStim.phase + self.gratingTF/self.frameRate
                visStim.draw() 
                
            # end trial after stimFrames and interStimFrames
            if self._trialFrame == self.stimFrames + self.interStimFrames:
                self._trialFrame = -1
                blockTrial += 1
            
            self.showFrame()



if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    task = RFMapping(params['rigName'],params['taskVersion'])
    task.start(params['subjectName'])